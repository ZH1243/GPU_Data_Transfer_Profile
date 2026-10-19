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
                    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
                    idx = torch.tensor([4, 1, 6, 0, 3, 0, 0], device="cuda", dtype=torch.int32)
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


if __name__ == "__main__":
    unittest.main()
