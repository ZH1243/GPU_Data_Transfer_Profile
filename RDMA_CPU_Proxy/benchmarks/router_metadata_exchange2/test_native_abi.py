import ctypes
import os
import subprocess
import unittest
from types import SimpleNamespace

from native import Proxy, Stats


class NativeAbiTests(unittest.TestCase):
    def test_stats_c_layout_matches_ctypes(self):
        result = subprocess.run([os.environ['RME2_ABI_BIN']], check=True, text=True, capture_output=True)
        layout = dict(line.split() for line in result.stdout.splitlines())
        self.assertEqual(int(layout.pop('size')), ctypes.sizeof(Stats))
        for field, offset in layout.items():
            self.assertEqual(int(offset), getattr(Stats, field).offset, field)

    def test_direct_output_borrows_pointer_and_checks_capacity(self):
        proxy = Proxy.__new__(Proxy)
        proxy.handle, proxy.tokens = 123, 33
        for address, size, valid in ((4096, 165, True), (0, 165, False),
                                     (4097, 165, False), (4096, 164, False)):
            def enable(handle, payload, capacity):
                self.assertEqual(handle, 123)
                ctypes.cast(payload, ctypes.POINTER(ctypes.c_void_p))[0] = address
                ctypes.cast(capacity, ctypes.POINTER(ctypes.c_uint64))[0] = size
                return 0
            proxy.lib = SimpleNamespace(rme2_enable_direct_x3_x4=enable)
            if valid:
                self.assertEqual(proxy.enable_direct_x3_x4(), address)
            else:
                with self.assertRaisesRegex(RuntimeError, 'invalid native'):
                    proxy.enable_direct_x3_x4()

    def test_staging_capture_retains_tensor_owners_and_uses_separate_api(self):
        calls = []
        proxy = Proxy.__new__(Proxy)
        proxy.handle = 123
        proxy.lib = SimpleNamespace(
            rme2_staging_capture_begin=lambda handle, stream: calls.append(('begin', handle, stream)),
            rme2_staging_capture_end=lambda handle: calls.append(('end', handle)),
        )
        stream = SimpleNamespace(cuda_stream=456)
        plan = SimpleNamespace(launch=lambda s: calls.append(('gather', s.cuda_stream)))
        proxy.capture_staging(plan, stream)
        self.assertEqual(calls, [('begin', 123, 456), ('gather', 456), ('end', 123)])
        self.assertIs(proxy._staging_plan, plan)
        self.assertIs(proxy._staging_stream, stream)


if __name__ == '__main__':
    unittest.main()
