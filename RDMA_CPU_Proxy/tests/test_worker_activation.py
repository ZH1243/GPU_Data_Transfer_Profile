"""CPU CLI checks and optional numerical Hopper checks for the embedded runner.

Run: python -m unittest discover -s RDMA_CPU_Proxy/tests -p test_worker_activation.py -v
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


WORKER = Path(__file__).resolve().parents[1] / "scripts" / "rdma_proxy_worker.py"
spec = importlib.util.spec_from_file_location("rdma_proxy_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ActivationCliTests(unittest.TestCase):
    def run_cli(self, *options):
        return subprocess.run(
            [sys.executable, str(WORKER), "--library=/unused.so",
             "--cuda-device-map=0,1", "--rdma-device-map=nic0,nic1",
             "--dry-run", *options],
            env={**os.environ, "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2",
                 "RANK": "0", "WORLD_SIZE": "4", "GROUP_RANK": "0"},
            text=True, capture_output=True,
        )

    def test_scheduler_projection_width(self):
        for activation, width in (("none", 128), ("relu", 128), ("swiglu", 256)):
            with self.subTest(activation=activation):
                result = self.run_cli(
                    "--concurrent-kernel", f"--grouped-gemm-activation={activation}",
                    "--grouped-gemm-output-dimension=128",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--expert_gemm_dimension={width}", result.stdout)

    def test_activation_requires_concurrent_kernel(self):
        result = self.run_cli("--grouped-gemm-activation=relu")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --concurrent-kernel", result.stderr)

    def test_down_options_and_rejections(self):
        common = ("--concurrent-kernel", "--grouped-gemm-activation=swiglu")
        valid = ("--down-projection2", "--down-gather-table",
                 "--down-gather-table-expert-waves", "--down-gather-table-wave-barrier",
                 "--down-pingpong", "--down-scatter-table", "--down-multi-buffer-scatter",
                 "--down-n-tile-major-output", "--down-epilogue-store=bulk_rows_reduce")
        result = self.run_cli(*common, *valid)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--down-", result.stdout)
        for options in (("--down-projection2",),
                        (*common, "--down-tile-m=128"),
                        (*common, "--down-projection2", "--down-scatter-table"),
                        (*common, "--down-projection2", "--down-gather-table-wave-barrier"),
                        (*common, "--down-projection2", "--down-unknown")):
            with self.subTest(options=options):
                self.assertNotEqual(self.run_cli(*options).returncode, 0)

    def test_local_combine_cli(self):
        result = self.run_cli("--local-gpu-combine", "--down-projection2",
                              "--concurrent-kernel", "--grouped-gemm-activation=swiglu")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--local-gpu-combine", result.stdout)
        self.assertNotEqual(self.run_cli("--local-gpu-combine").returncode, 0)


class CombineCompactionTests(unittest.TestCase):
    def test_compaction_preserves_input_order_and_reduces_duplicates(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                # Source 1 is empty; source 2 is already compact. Source 0 has
                # gaps and repeated expert contributions in arbitrary order.
                mapping = torch.tensor([[0, 6], [2, 1], [0, 2], [0, 6], [2, 0]],
                                       device=device)
                compact, counts, inverse = worker.compact_combine_mapping(
                    torch, mapping, 3, 8)
                torch.testing.assert_close(compact, torch.tensor(
                    [[0, 1], [2, 1], [0, 0], [0, 1], [2, 0]], device=device))
                torch.testing.assert_close(counts, torch.tensor([2, 0, 2], device=device))
                torch.testing.assert_close(inverse, torch.tensor(
                    [[2, 6, -1, -1, -1, -1, -1, -1], [-1] * 8,
                     [0, 1, -1, -1, -1, -1, -1, -1]], device=device))
                contributions = torch.tensor([1., 2., 3., 4., 5.], device=device)
                output = torch.zeros(24, device=device)
                output.index_add_(0, compact[:, 0] * 8 + compact[:, 1], contributions)
                expected = torch.zeros((3, 8), device=device)
                expected[0, :2] = torch.tensor([3., 5.], device=device)
                expected[2, :2] = torch.tensor([5., 2.], device=device)
                torch.testing.assert_close(output.view(3, 8), expected)
                # Recompute with changed membership: no stale valid rows or inverse map.
                _, counts, inverse = worker.compact_combine_mapping(
                    torch, torch.tensor([[0, 7]], device=device), 3, 8)
                torch.testing.assert_close(counts, torch.tensor([1, 0, 0], device=device))
                expected_inverse = torch.full((3, 8), -1, device=device)
                expected_inverse[0, 0] = 7
                torch.testing.assert_close(inverse, expected_inverse)


class HopperActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("requires a Hopper CUDA device")
        cls.torch = torch

    def test_combine_dynamic_mapping(self):
        torch = self.torch
        # Expert 1 arrives first. Expert 0's first batch consumes only source 1;
        # its second batch consumes source 0. N groups duplicate each descriptor.
        rows = [[1, 0, 2, 3, 2, 3], [0, 0, 0, 0, 0, 2],
                [0, 0, 0, 2, 2, 2]]
        table = torch.tensor([r[:1] + [n] + r[2:] for r in rows for n in (0, 2)],
                             dtype=torch.int32, device="cuda")
        indices = tuple(torch.tensor(x, dtype=torch.int32, device="cuda")
                        for x in ([4, 1, 4, 0], [2, 3, 2, 0]))
        expected = torch.tensor([[1, 2], [1, 3], [0, 4], [0, 1], [0, 4], [1, 2]],
                                device="cuda")
        torch.testing.assert_close(worker.local_combine_mapping(
            torch, table, indices, 6, 2), expected)
        # A new iteration with a different source arrival order changes the map.
        rows = [[0, 0, 0, 2, 0, 0], [1, 0, 2, 3, 2, 3], [0, 0, 2, 2, 0, 2]]
        table.copy_(torch.tensor([r[:1] + [n] + r[2:] for r in rows for n in (0, 2)],
                                 dtype=torch.int32, device="cuda"))
        expected = expected[[2, 3, 0, 1, 4, 5]]
        torch.testing.assert_close(worker.local_combine_mapping(
            torch, table, indices, 6, 2), expected)

    def test_runner_numerical_outputs(self):
        torch = self.torch
        root = Path(os.environ.get(
            "QUACK_ROOT", WORKER.parents[3] / "quack_for_communication_overlap"
        ))
        for activation in (None, "relu", "swiglu"):
            with self.subTest(activation=activation), torch.inference_mode():
                torch.manual_seed(123)
                buffers = worker.TorchDeviceBuffers(
                    torch, torch.cuda.current_device(), node_rank=0, local_gpu_index=0
                )
                xs, indices = [], []
                for source in range(2):
                    # Down width spans two N=128 tiles to test packed strides.
                    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
                    idx = torch.tensor([4, 1, 6, 4, 3, 0, 0], device="cuda", dtype=torch.int32)
                    buffers.tensors[(worker.DEVICE_BUFFER_NVLINK_RECEIVE, -1, 0, source)] = x
                    buffers.tensors[(worker.DEVICE_BUFFER_ROUTER_A_IDX, -1, 0, source)] = idx
                    xs.append(x)
                    indices.append(idx)
                # Ragged expert batches, two input sources, and padded A_idx tails.
                table = torch.tensor(
                    [[0, 0, 0, 3, 0, 3], [1, 0, 3, 5, 3, 5]],
                    device="cuda", dtype=torch.int32,
                )
                buffers.tensors[(worker.DEVICE_BUFFER_GATHER_TABLE, -1, -1, -1)] = table
                buffers.tensors[(worker.DEVICE_BUFFER_GATHER_READY_ROWS, -1, -1, -1)] = (
                    torch.tensor([2], device="cuda", dtype=torch.int32)
                )
                runner = worker.GroupedGemmRunner(
                    torch, buffers, cuda_device_id=torch.cuda.current_device(),
                    world_size=1, num_experts=2, output_dimension=128,
                    activation=activation, tile_m=128, tile_n=128, tile_k=None,
                    cluster_m=1, max_swizzle_size=8, quack_root=root,
                )
                with torch.cuda.stream(torch.cuda.Stream()):
                    runner.launch()
                torch.cuda.synchronize()
                parts = []
                for expert, (start, end) in enumerate(((0, 3), (3, 5))):
                    for x, idx in zip(xs, indices):
                        value = x[idx[start:end].long()].float() @ runner.weights[expert].float()
                        if activation == "relu":
                            value = torch.relu(value)
                        elif activation == "swiglu":
                            gate, up = value.chunk(2, dim=-1)
                            value = torch.nn.functional.silu(gate) * up
                        parts.append(value)
                expected = torch.cat(parts).to(torch.bfloat16)
                torch.testing.assert_close(runner.output[:10], expected, atol=6e-2, rtol=2e-3)

                if activation == "swiglu":
                    # Reuse the actual fused up output, including its unused padding.
                    variants = (
                        (),
                        ("--local-gpu-combine",),
                        ("--local-gpu-combine", "--down-n-tile-major-output",
                         "--down-gather-table", "--down-gather-table-expert-waves",
                         "--down-gather-table-wave-barrier", "--down-pingpong"),
                        ("--down-gather-table",),
                        ("--down-scatter-table", "--down-epilogue-store=bulk_rows_reduce"),
                        ("--down-scatter-table", "--down-multi-buffer-scatter",
                         "--down-scatter-output-buffers=2", "--down-n-tile-major-output",
                         "--down-gather-table", "--down-gather-table-expert-waves",
                         "--down-gather-table-wave-barrier", "--down-pingpong",
                         "--down-epilogue-store=bulk_rows_reduce"),
                    )
                    import argparse
                    parser = argparse.ArgumentParser()
                    worker.add_down_arguments(parser)
                    for flags in variants:
                        with self.subTest(down=flags):
                            options = parser.parse_args(["--down-projection2", *flags])
                            down = worker.DownProjectionRunner(torch, runner, [6, 4], options)
                            reference = torch.cat([
                                runner.output[start:end].float() @ down.weights[e].float()
                                for e, (start, end) in enumerate(((0, 6), (6, 10)))
                            ])
                            if options.local_gpu_combine:
                                # Independent known expert/source packing; two experts
                                # collide at token 4 in each separate source output.
                                # Used input rows [1, 3, 4, 6] map to [0, 1, 2, 3].
                                destinations = torch.tensor(
                                    [2, 0, 3, 10, 8, 11, 2, 1, 10, 9], device="cuda")
                                scattered = torch.zeros((16, 256), device="cuda")
                                scattered.index_add_(0, destinations, reference)
                                reference = scattered
                            elif down.inputs.scatter_table is not None:
                                scattered = torch.zeros_like(reference)
                                scattered.index_add_(0, down.inputs.scatter_table.long(), reference)
                                reference = scattered
                            for _ in range(2):
                                # A second launch catches missing reduction destination resets.
                                down.launch()
                                torch.cuda.synchronize()
                                if options.local_gpu_combine:
                                    torch.testing.assert_close(down.combine_valid_rows,
                                        torch.tensor([4, 4], device="cuda"))
                                    torch.testing.assert_close(down.combine_output_to_input,
                                        torch.tensor([[1, 3, 4, 6, -1, -1, -1, -1]] * 2,
                                                     device="cuda"))
                                    for source, output in enumerate(down.collected_combine_outputs()):
                                        expected = reference[source * 8:source * 8 + 4]
                                        torch.testing.assert_close(output, expected.to(torch.bfloat16),
                                                                   atol=6e-2, rtol=2e-3)
                                        raw = down.inputs.output_buffers()[source].view(-1)
                                        # Both layouts occupy exactly one 4-token prefix.
                                        torch.testing.assert_close(raw[4 * 256:],
                                                                   torch.zeros_like(raw[4 * 256:]))
                                else:
                                    torch.testing.assert_close(
                                        down.inputs.collected_up_output(), reference.to(torch.bfloat16),
                                        atol=6e-2, rtol=2e-3,
                                    )


if __name__ == "__main__":
    unittest.main()
