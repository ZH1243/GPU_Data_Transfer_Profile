"""CPU coverage for deferred scatter metadata layout and option validation.

GPU publication ordering is exercised by test_worker_composition.py; the actual
native event generations and H2D submission are covered by test_cpu_scatter.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rdma_proxy_worker as worker


class PublicationOptionsTests(unittest.TestCase):
    def parse(self, *flags):
        parser = argparse.ArgumentParser()
        worker.add_down_arguments(parser)
        args = parser.parse_args(flags)
        args.concurrent_kernel = True
        args.grouped_gemm_activation = "swiglu"
        return args

    def test_defaults_and_explicit_modes(self):
        common = ("--cpu-scatter-preparation", "--local-gpu-combine", "--down-projection2")
        defaults = self.parse(*common)
        self.assertIsNone(defaults.cpu_scatter_publication)
        worker.validate_down_arguments(defaults)
        for mode in ("deferred", "per-batch"):
            with self.subTest(mode=mode):
                args = self.parse(*common, f"--cpu-scatter-publication={mode}")
                worker.validate_down_arguments(args)
                self.assertEqual(args.cpu_scatter_publication, mode)

    def test_explicit_mode_requires_cpu_preparation(self):
        for mode in ("deferred", "per-batch"):
            args = self.parse("--local-gpu-combine", "--down-projection2",
                              f"--cpu-scatter-publication={mode}")
            with self.assertRaisesRegex(ValueError, "cpu-scatter-preparation"):
                worker.validate_down_arguments(args)

    def test_submission_wait_has_finite_positive_timeout(self):
        common = ("--cpu-scatter-preparation", "--local-gpu-combine", "--down-projection2")
        for timeout in ("0", "-1", "nan", "inf", "1e20"):
            with self.subTest(timeout=timeout):
                args = self.parse(*common, f"--return-token-timeout={timeout}")
                with self.assertRaisesRegex(ValueError, "finite.*positive"):
                    worker.validate_down_arguments(args)
        args = self.parse(*common, "--return-token-timeout=0.125")
        worker.validate_down_arguments(args)


class MetadataSlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        cls.torch = torch

    def test_byte_layout_alignment_and_single_copy_round_trip(self):
        torch = self.torch
        for routes in (1, 3, 4, 257):
            for columns in (1, 2):
                for weighted in (False, True):
                    with self.subTest(routes=routes, columns=columns, weighted=weighted):
                        shape = (routes,) if columns == 1 else (routes, columns)
                        addresses = torch.arange(routes * columns, dtype=torch.int64).reshape(shape)
                        addresses += 0x123456700000000
                        weights = (torch.arange(routes, dtype=torch.float32) / 8 if weighted else None)
                        slab, address_view, weight_view = worker.make_cpu_scatter_metadata_storage(
                            torch, addresses, weights)
                        self.assertEqual(slab.dtype, torch.uint8)
                        self.assertTrue(slab.is_contiguous())
                        self.assertEqual(address_view.dtype, torch.int64)
                        self.assertEqual(address_view.shape, addresses.shape)
                        self.assertEqual(address_view.data_ptr(), slab.data_ptr())
                        torch.testing.assert_close(address_view, addresses, rtol=0, atol=0)
                        address_bytes = addresses.numel() * addresses.element_size()
                        if weighted:
                            weight_offset = weight_view.data_ptr() - slab.data_ptr()
                            self.assertEqual(weight_offset % 16, 0)
                            self.assertGreaterEqual(weight_offset, address_bytes)
                            self.assertLess(weight_offset - address_bytes, 16)
                            self.assertEqual(weight_view.dtype, torch.float32)
                            self.assertEqual(weight_view.shape, weights.shape)
                            self.assertEqual(slab.numel(), weight_offset + routes * 4)
                            self.assertTrue(torch.equal(slab[address_bytes:weight_offset],
                                                        torch.zeros(weight_offset - address_bytes,
                                                                    dtype=torch.uint8)))
                            torch.testing.assert_close(weight_view, weights, rtol=0, atol=0)
                        else:
                            self.assertIsNone(weight_view)
                            self.assertEqual(slab.numel(), address_bytes)
                        # Distinct destination storage models one complete H2D
                        # upload. Consumers retain aliased views across iterations.
                        destination, copied_addresses, copied_weights = worker.make_cpu_scatter_metadata_storage(
                            torch, torch.zeros_like(addresses),
                            torch.zeros_like(weights) if weighted else None)
                        for iteration in range(3):
                            address_view.copy_(addresses + iteration * 4096)
                            if weighted:
                                weight_view.copy_(weights if iteration % 2 else torch.zeros_like(weights))
                            destination.fill_(255)
                            destination.copy_(slab)
                            torch.testing.assert_close(copied_addresses, address_view, rtol=0, atol=0)
                            if weighted:
                                torch.testing.assert_close(copied_weights, weight_view, rtol=0, atol=0)

    def test_invalid_storage_is_rejected_before_native_borrow(self):
        torch = self.torch
        for addresses in (torch.zeros(0, dtype=torch.int64), torch.zeros(3, dtype=torch.int32),
                          torch.zeros(3, 1, dtype=torch.int64), torch.zeros(3, 3, dtype=torch.int64),
                          torch.zeros(6, dtype=torch.int64)[::2]):
            with self.subTest(shape=addresses.shape, dtype=addresses.dtype):
                with self.assertRaisesRegex(ValueError, "contiguous int64"):
                    worker.make_cpu_scatter_metadata_storage(torch, addresses)
        addresses = torch.zeros(3, 2, dtype=torch.int64)
        for weights in (torch.ones(2), torch.ones(3, dtype=torch.float64), torch.ones(6)[::2]):
            with self.subTest(shape=weights.shape, dtype=weights.dtype):
                with self.assertRaisesRegex(ValueError, "contiguous FP32"):
                    worker.make_cpu_scatter_metadata_storage(torch, addresses, weights)


class MetadataDependencyTests(unittest.TestCase):
    def test_native_wait_receives_iteration_current_stream_and_timeout(self):
        calls = []
        runner = worker.DownProjectionRunner.__new__(worker.DownProjectionRunner)
        stream = SimpleNamespace(cuda_stream=0x123456789)
        runner.up = SimpleNamespace(cuda_device_id=7, torch=SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda device: (
                self.assertEqual(device, 7), stream)[1])))
        runner._cpu_scatter_handle = object()
        runner._cpu_scatter_library = SimpleNamespace(rdma_proxy_last_error=lambda: b"publication timed out")
        runner.cpu_scatter_timeout_ms = 125
        runner._cpu_scatter_wait = lambda *args: (calls.append(args), 0)[1]
        for iteration in (0, 1, 11):
            runner.wait_cpu_scatter_ready(iteration)
        self.assertEqual(calls, [(runner._cpu_scatter_handle, iteration, stream.cuda_stream, 125)
                                 for iteration in (0, 1, 11)])
        runner._cpu_scatter_wait = lambda *args: -1
        with self.assertRaisesRegex(RuntimeError, "iteration 12.*publication timed out"):
            runner.wait_cpu_scatter_ready(12)
        runner._cpu_scatter_wait = None
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            runner.wait_cpu_scatter_ready(13)

    def test_down_and_combine_are_ordered_after_metadata_gate(self):
        def make_runner(events):
            event = lambda label: SimpleNamespace(record=lambda: events.append(label))
            runner = worker.DownProjectionRunner.__new__(worker.DownProjectionRunner)
            runner.up = SimpleNamespace(torch=SimpleNamespace(inference_mode=nullcontext))
            runner._log_combine_startup = lambda message: None
            runner.reset_start, runner.mapping_start = event("reset"), event("mapping")
            runner.gemm_start, runner.end = event("gemm"), event("done")
            runner.inputs = SimpleNamespace(zero_up_output=lambda: events.append("zero"))
            runner.combine, runner.cpu_scatter = True, True
            runner.route_weights = object()  # Deferred weighted metadata is retained.
            runner.return_exchange = SimpleNamespace(publish_counts=lambda: events.append("counts"))
            runner.intra_node_combine = SimpleNamespace(mapping_ready=event("combine-ready"),
                                                       launch=lambda: events.append("combine"))
            runner.launch_gemm = lambda: events.append("down")
            return runner

        events = []
        runner = make_runner(events)
        runner.launch(wait_for_metadata=lambda: events.append("metadata-wait"))
        self.assertEqual(events, ["reset", "zero", "mapping", "metadata-wait", "counts",
                                  "gemm", "combine-ready", "down", "combine", "done"])
        events.clear()

        def fail():
            events.append("metadata-failed")
            raise RuntimeError("native submission gate failed")

        with self.assertRaisesRegex(RuntimeError, "submission gate failed"):
            make_runner(events).launch(wait_for_metadata=fail)
        self.assertEqual(events, ["reset", "zero", "mapping", "metadata-failed"])


if __name__ == "__main__":
    unittest.main()
