import subprocess
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from benchmark import enqueue_iteration


class LaunchDispatchTests(unittest.TestCase):
    def test_native_never_launches_python_kernels_or_events(self):
        for native, graph_mode, expected in (
            (True, True, ['submit']),
            (False, True, ['begin', 'graph', 'end', 'submit']),
            (False, False, ['begin', 'kernels', 'end', 'submit']),
        ):
            calls = []
            proxy = SimpleNamespace(submit=lambda *args: calls.append('submit'))
            plan = SimpleNamespace(launch=lambda *args: calls.append('kernels'))
            graph = SimpleNamespace(replay=lambda: calls.append('graph')) if graph_mode else None
            begin = SimpleNamespace(record=lambda *args: calls.append('begin'))
            end = SimpleNamespace(record=lambda *args: calls.append('end'))
            enqueue_iteration(proxy, 1, plan, object(), graph, begin, end, native)
            self.assertEqual(calls, expected)

    def test_invalid_modes_fail_before_cuda_import(self):
        program = str(Path(__file__).with_name('benchmark.py'))
        for flags in (['--metadata-launch-barrier'], ['--native-metadata-launch', '--mode', 'eager'],
                      ['--batch-payload-copy', '--per-source-readback']):
            result = subprocess.run([sys.executable, program, *flags], text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('requires', result.stderr)

    def test_two_node_options_fail_before_cuda_import(self):
        program = str(Path(__file__).with_name('benchmark.py'))
        env = dict(os.environ, WORLD_SIZE='16', RANK='8', LOCAL_RANK='0', LOCAL_WORLD_SIZE='8')
        for flag in ('--per-source-readback', '--batch-payload-copy'):
            result = subprocess.run([sys.executable, program, flag], env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('two-node mode requires default all-senders', result.stderr)
        result = subprocess.run([sys.executable, program, '--rdma-device-map', ''],
                                env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('one or eight NIC names', result.stderr)


if __name__ == '__main__':
    unittest.main()
