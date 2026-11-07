import subprocess
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from benchmark import enqueue_iteration, options
from unittest.mock import patch


class LaunchDispatchTests(unittest.TestCase):
    def test_receiver_kernel_option_and_topology(self):
        for flags, enabled in (([], False), (['--receiver-kernel'], True),
                               (['--receiver-kernel', '--skip-local-staging'], True),
                               (['--receiver-kernel', '--local-staging-after-d2h'], True)):
            with patch.object(sys, 'argv', ['benchmark.py', *flags]):
                self.assertEqual(options().receiver_kernel, enabled)
        program = str(Path(__file__).with_name('benchmark.py'))
        env = dict(os.environ, WORLD_SIZE='8', RANK='0', LOCAL_RANK='0', LOCAL_WORLD_SIZE='8')
        result = subprocess.run([sys.executable, program, '--receiver-kernel'],
                                env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires two-node mode', result.stderr)

    def test_staging_readback_is_opt_in(self):
        for flags, before, after in (([], False, False),
                                     (['--d2h-after-local-staging'], True, False),
                                     (['--local-staging-after-d2h'], False, True)):
            with patch.object(sys, 'argv', ['benchmark.py', *flags]):
                args = options()
            self.assertEqual(args.d2h_after_local_staging, before)
            self.assertEqual(args.local_staging_after_d2h, after)
            self.assertFalse(args.skip_local_staging)

    def test_direct_output_option(self):
        for flags, enabled in (([], False), (['--direct-x3-x4-output'], True)):
            with patch.object(sys, 'argv', ['benchmark.py', *flags]):
                self.assertEqual(options().direct_x3_x4_output, enabled)
        program = str(Path(__file__).with_name('benchmark.py'))
        env = dict(os.environ, WORLD_SIZE='8', RANK='0', LOCAL_RANK='0', LOCAL_WORLD_SIZE='8')
        result = subprocess.run([sys.executable, program, '--direct-x3-x4-output'],
                                env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires two-node mode', result.stderr)

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
                      ['--batch-payload-copy', '--per-source-readback'], ['--token-dimension', '0'],
                      ['--d2h-after-local-staging', '--skip-local-staging'],
                      ['--local-staging-after-d2h', '--skip-local-staging'],
                      ['--receiver-kernel', '--d2h-after-local-staging'],
                      ['--local-staging-after-d2h', '--d2h-after-local-staging']):
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
