"""CPU CLI checks and numerical SM90/SM100 checks for the embedded runner.

Run: python -m unittest discover -s RDMA_CPU_Proxy/tests -p test_worker_activation.py -v
"""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


WORKER = Path(__file__).resolve().parents[1] / "scripts" / "rdma_proxy_worker.py"
spec = importlib.util.spec_from_file_location("rdma_proxy_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class CommunicationOnlyCliTests(unittest.TestCase):
    def test_hopper_launcher_defaults(self):
        # An empty optional site-argument array fails under macOS Bash 3 + nounset.
        env = {key: value for key, value in os.environ.items()
               if key not in ("PEER_HOST", "MASTER_ADDR", "CUDA_DEVICE_MAP", "RDMA_DEVICE_MAP",
                              "RDMA_GID_INDEX", "RDMA_PORT", "RDMA_PROXY_BUILD_DIR")}
        env.update(TORCHRUN_BIN="echo", RDMA_PROXY_GEMM_ARCH="auto")
        for node in (0, 1):
            result = subprocess.run(["bash", str(WORKER.parent / f"run_node{node}_torchrun.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            options = shlex.split(result.stdout)
            self.assertIn("--gemm-arch=auto", options)
            self.assertIn(f"--library={WORKER.parent.parent}/build-hopper/librdma_cpu_proxy.so", options)
            self.assertNotIn("--down-pingpong", options)

    def test_dual_device_map(self):
        for rank in (0, 1):
            result = subprocess.run(
                [sys.executable, str(WORKER), "--library=/unused.so",
                 "--cuda-device-map=0,1", "--rdma-device-map=nic0+nic1,nic2+nic3",
                 "--num_qps_per_peer=20", "--communication-only", "--dry-run"],
                env={**os.environ, "LOCAL_RANK": str(rank), "LOCAL_WORLD_SIZE": "2",
                     "RANK": str(rank), "WORLD_SIZE": "4", "GROUP_RANK": "0"},
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"--rdma_device_name=nic{rank * 2}", result.stdout)
            self.assertIn(f"--rdma_device_name2=nic{rank * 2 + 1}", result.stdout)
        for device_map in ("a+a,b+c", "a+,b+c", "a+b+c,d+e"):
            result = subprocess.run(
                [sys.executable, str(WORKER), "--library=/unused.so",
                 "--cuda-device-map=0,1", f"--rdma-device-map={device_map}", "--dry-run"],
                env={**os.environ, "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2",
                     "RANK": "0", "WORLD_SIZE": "4", "GROUP_RANK": "0"},
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("two distinct devices", result.stderr)

    def test_existing_launch_scripts(self):
        for script_name, node, nodes in (
            ("run_node0_torchrun.sh", 0, 2),
            ("run_node1_torchrun.sh", 1, 2),
            ("run_single_node_torchrun.sh", 0, 1),
        ):
            script = WORKER.parent / script_name
            # Exercise real shell expansion, including architecture/site arrays,
            # while replacing torchrun with an argv recorder (no CUDA/network).
            with tempfile.TemporaryDirectory() as temp:
                recorder = Path(temp) / "torchrun-recorder"
                recorder.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
                recorder.chmod(0o700)
                launched = subprocess.run(
                    ["bash", str(script)], capture_output=True, text=True,
                    env={**os.environ, "TORCHRUN_BIN": str(recorder),
                         "RDMA_PROXY_GEMM_ARCH": "blackwell", "MASTER_ADDR": "127.0.0.1",
                         "PEER_HOST": "127.0.0.2", "CUDA_DEVICE_MAP": "0,1,2,3,4,5,6,7",
                         "RDMA_DEVICE_MAP": "nic0,nic1,nic2,nic3,nic4,nic5,nic6,nic7",
                         "RDMA_GID_INDEX": "-1"},
                )
                self.assertEqual(launched.returncode, 0, launched.stderr)
                tokens = json.loads(launched.stdout)
            options = tokens[tokens.index(str(WORKER)) + 1:]
            for mode_first in (False, True):
                with self.subTest(script=script_name, mode_first=mode_first):
                    mode = ["--communication-only"]
                    argv = mode + options if mode_first else options + mode
                    result = subprocess.run(
                        [sys.executable, str(WORKER), *argv, "--dry-run"],
                        env={**os.environ, "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "8",
                             "RANK": str(node * 8), "WORLD_SIZE": str(nodes * 8), "GROUP_RANK": str(node)},
                        text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("communication-only mode", result.stdout)
                    native = shlex.split(result.stdout.splitlines()[-1])
                    values = dict(arg[2:].split("=", 1) for arg in native
                                  if arg.startswith("--") and "=" in arg)
                    self.assertEqual(values["num_nodes"], str(nodes))
                    for key in ("router_routing_enabled", "router_local_input_staging_enabled",
                                "nvlink_forwarding_enabled", "nvlink_forward_preparation_enabled",
                                "nvlink_forward_submit_epilogue_enabled"):
                        self.assertEqual(values[key], "true")
                    for key in ("nvlink_return_completion_notifications_enabled",
                                "nvlink_return_ready_gpu_publication_enabled", "mock_mode"):
                        self.assertEqual(values[key], "false")
                    self.assertNotIn("--concurrent-kernel", native)
                    self.assertNotIn("--down-projection2", native)

    def test_unknown_down_option_still_rejected(self):
        result = subprocess.run(
            [sys.executable, str(WORKER), "--library=/unused.so",
             "--cuda-device-map=0,1", "--rdma-device-map=nic0,nic1",
             "--communication-only", "--down-typo", "--dry-run"],
            text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported --down-* option", result.stderr)


class ActivationCliTests(unittest.TestCase):
    def test_architecture_and_pingpong_selection(self):
        for capability in ((9, 0), (10, 0), (10, 3)):
            self.assertEqual(worker.resolve_gemm_architecture("auto", capability), capability[0])
        self.assertEqual(worker.resolve_gemm_architecture("blackwell", (10, 3)), 10)
        for requested, capability in (("hopper", (10, 3)), ("blackwell", (9, 0)),
                                      ("auto", (12, 0)), ("auto", (8, 0))):
            with self.assertRaises(ValueError):
                worker.resolve_gemm_architecture(requested, capability)
        self.assertTrue(worker.resolve_down_pingpong(None, 9, True))
        self.assertFalse(worker.resolve_down_pingpong(None, 10, True))
        self.assertFalse(worker.resolve_down_pingpong(None, 9, False))
        for values in ((True, 10, True), (False, 9, True)):
            with self.assertRaises(ValueError):
                worker.resolve_down_pingpong(*values)
        common = ("--concurrent-kernel", "--grouped-gemm-activation=swiglu",
                  "--down-projection2", "--down-gather-table",
                  "--down-gather-table-expert-waves", "--down-gather-table-wave-barrier",
                  "--down-epilogue-store=bulk_rows_reduce", "--down-gather-table-publish-ready")
        for architecture, flag in (("hopper", "--down-pingpong"),
                                   ("blackwell", "--no-down-pingpong")):
            for explicit in ((), (flag,)):
                result = self.run_cli(*common, f"--gemm-arch={architecture}", *explicit,
                                      "--check-projections")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--expert_gemm_device_capacity={9 if architecture == 'hopper' else 10}",
                              result.stdout)
            wrong = "--no-down-pingpong" if architecture == "hopper" else "--down-pingpong"
            self.assertNotEqual(self.run_cli(*common, f"--gemm-arch={architecture}", wrong).returncode, 0)

    def test_cpu_scatter_cli(self):
        self.assertNotEqual(self.run_cli("--cpu-scatter-preparation").returncode, 0)
        result = self.run_cli("--cpu-scatter-preparation", "--concurrent-kernel",
                              "--grouped-gemm-activation=swiglu", "--down-projection2",
                              "--local-gpu-combine")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--cpu-scatter-preparation", result.stdout)

    def test_return_results_cli(self):
        common = ("--concurrent-kernel", "--grouped-gemm-activation=swiglu",
                  "--down-projection2", "--local-gpu-combine", "--down-pingpong",
                  "--down-gather-table", "--down-gather-table-expert-waves",
                  "--down-gather-table-wave-barrier", "--down-gather-table-publish-ready")
        for extra in (("--return-token-results",),
                      ("--return-token-results", "--return-token-results-per-group",
                       "--down-n-tile-major-output", "--check-return-token-results")):
            result = self.run_cli(*common, *extra)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("--return-token", result.stdout)
        for extra in (("--return-token-results-per-group",),
                      ("--return-token-results", "--return-token-results-per-group"),
                      ("--check-return-token-results",),
                      ("--return-token-results", "--return-token-timeout=0"),
                      ("--return-token-results", "--return-token-timeout=nan")):
            self.assertNotEqual(self.run_cli(*common, *extra).returncode, 0)
        self.assertNotEqual(self.run_cli("--return-token-results").returncode, 0)

    def test_intra_node_combine_cli(self):
        common = ("--concurrent-kernel", "--grouped-gemm-activation=swiglu",
                  "--down-projection2", "--local-gpu-combine", "--down-pingpong",
                  "--down-gather-table", "--down-gather-table-expert-waves",
                  "--down-gather-table-wave-barrier", "--down-gather-table-publish-ready",
                  "--return-token-results", "--return-token-results-per-group",
                  "--down-n-tile-major-output", "--intra-node-combine")
        tiled = self.run_cli(*common, "--check-intra-node-combine",
                             "--intra-node-combine-n-tile-major-output",
                             "--intra-node-combine-publish-ready")
        self.assertEqual(tiled.returncode, 0, tiled.stderr)
        self.assertNotIn("--intra-node-combine-n-tile-major-output", tiled.stdout)
        network = self.run_cli(*common, "--intra-node-combine-n-tile-major-output",
                               "--intra-node-combine-publish-ready", "--rdma-return-token-results",
                               "--check-rdma-return-token-results", "--rdma-return-chunk-bytes=37")
        self.assertEqual(network.returncode, 0, network.stderr)
        self.assertNotIn("--rdma-return-token-results", network.stdout)
        final = self.run_cli(*common, "--intra-node-combine-n-tile-major-output",
                            "--intra-node-combine-publish-ready", "--rdma-return-token-results",
                            "--inter-node-combine", "--check-inter-node-combine")
        self.assertEqual(final.returncode, 0, final.stderr)
        self.assertNotIn("--inter-node-combine", final.stdout)
        for extra in (("--inter-node-combine",), ("--check-inter-node-combine",)):
            self.assertNotEqual(self.run_cli(*common, *extra).returncode, 0)
        for extra in (("--rdma-return-token-results",), ("--check-rdma-return-token-results",),
                      ("--rdma-return-token-results", "--intra-node-combine-n-tile-major-output",
                       "--intra-node-combine-publish-ready", "--rdma-return-chunk-bytes=0")):
            self.assertNotEqual(self.run_cli(*common, *extra).returncode, 0)

        result = self.run_cli(*common, "--check-intra-node-combine")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--nvlink_return_completion_notifications_enabled=true", result.stdout)
        self.assertIn("--nvlink_return_ready_gpu_publication_enabled=true", result.stdout)
        for options in (("--intra-node-combine",), ("--check-intra-node-combine",),
                        ("--intra-node-combine-n-tile-major-output",),
                        ("--intra-node-combine-publish-ready",),
                        (*common, "--intra-node-combine-ctas=0")):
            self.assertNotEqual(self.run_cli(*options).returncode, 0)

    def test_single_node_inter_node_combine(self):
        common = ("--concurrent-kernel", "--grouped-gemm-activation=swiglu",
                  "--down-projection2", "--local-gpu-combine", "--down-pingpong",
                  "--down-gather-table", "--down-gather-table-expert-waves",
                  "--down-gather-table-wave-barrier", "--down-gather-table-publish-ready",
                  "--return-token-results", "--return-token-results-per-group",
                  "--down-n-tile-major-output", "--intra-node-combine",
                  "--intra-node-combine-n-tile-major-output", "--inter-node-combine")
        result = self.run_cli(*common, "--check-inter-node-combine", nodes=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--num_nodes=1", result.stdout)
        self.assertNotIn("--inter-node-combine", result.stdout)
        for count in (1, 4, 40):
            tuned = self.run_cli(*common, f"--inter-node-combine-ctas={count}", nodes=1)
            self.assertEqual(tuned.returncode, 0, tuned.stderr)
            self.assertNotIn("--inter-node-combine-ctas", tuned.stdout)
        for count in (0, -1):
            self.assertNotEqual(self.run_cli(*common, f"--inter-node-combine-ctas={count}",
                                            nodes=1).returncode, 0)
        self.assertNotEqual(self.run_cli("--inter-node-combine-ctas=4", nodes=1).returncode, 0)
        # Multiple nodes still require reverse RDMA; one node rejects it.
        multi = self.run_cli(*common)
        self.assertNotEqual(multi.returncode, 0)
        self.assertIn("requires --rdma-return-token-results", multi.stderr)
        self.assertNotEqual(self.run_cli(*common, "--rdma-return-token-results",
                                        "--intra-node-combine-publish-ready", nodes=1).returncode, 0)
        for missing in ("--intra-node-combine", "--intra-node-combine-n-tile-major-output"):
            self.assertNotEqual(self.run_cli(*(x for x in common if x != missing),
                                            nodes=1).returncode, 0)

    def run_cli(self, *options, nodes=2):
        return subprocess.run(
            [sys.executable, str(WORKER), "--library=/unused.so",
             "--cuda-device-map=0,1", "--rdma-device-map=nic0,nic1",
             "--dry-run", *options],
            env={**os.environ, "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2",
                 "RANK": "0", "WORLD_SIZE": str(nodes * 2), "GROUP_RANK": "0"},
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

    def test_down_sm_count_cli(self):
        common = ("--concurrent-kernel", "--grouped-gemm-activation=swiglu",
                  "--down-projection2")
        for extra in (("--down-sm-count", "40"),
                      ("--down-sm-count=40", "--down-cluster-m=2")):
            with self.subTest(extra=extra):
                result = self.run_cli(*common, *extra)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("--down-", result.stdout)
        for count in (0, -1, 3):
            with self.subTest(count=count):
                result = self.run_cli(*common, f"--down-sm-count={count}",
                                      "--down-cluster-m=2")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be positive and divisible by --down-cluster-m", result.stderr)
        result = self.run_cli(*common[:-1], "--down-sm-count=40")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--down-* options require --down-projection2", result.stderr)

    def test_local_combine_cli(self):
        result = self.run_cli("--local-gpu-combine", "--down-projection2",
                              "--concurrent-kernel", "--grouped-gemm-activation=swiglu")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--local-gpu-combine", result.stdout)
        self.assertNotEqual(self.run_cli("--local-gpu-combine").returncode, 0)

    def test_down_publish_ready_prerequisites(self):
        options = ("--concurrent-kernel", "--grouped-gemm-activation=swiglu",
                   "--down-projection2", "--down-gather-table",
                   "--down-gather-table-expert-waves", "--down-gather-table-wave-barrier",
                   "--down-pingpong", "--down-epilogue-store=bulk_rows_reduce",
                   "--down-gather-table-publish-ready")
        result = self.run_cli(*options)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--down-", result.stdout)
        for required in (option for option in options[:-1] if option != "--down-pingpong"):
            with self.subTest(missing=required):
                result = self.run_cli(*(option for option in options if option != required))
                self.assertNotEqual(result.returncode, 0)

    def test_trace_down_ready_prerequisites(self):
        options = ("--concurrent-kernel", "--grouped-gemm-activation=swiglu",
                   "--down-projection2", "--down-gather-table",
                   "--down-gather-table-expert-waves", "--down-gather-table-wave-barrier",
                   "--down-pingpong", "--down-epilogue-store=bulk_rows_reduce",
                   "--down-gather-table-publish-ready", "--trace-down-ready")
        result = self.run_cli(*options)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--trace-down-ready", result.stdout)
        for required in ("--down-projection2", "--down-gather-table-publish-ready"):
            result = self.run_cli(*(option for option in options if option != required))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--trace-down-ready requires", result.stderr)


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
    # Retain the class name for existing remote commands; both architectures run it.
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
            raise unittest.SkipTest("requires a Hopper or Blackwell CUDA device")
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
                        ("--down-sm-count=2",),
                        ("--down-sm-count=2", "--down-gather-table",
                         "--down-gather-table-expert-waves",
                         "--down-gather-table-wave-barrier", "--down-pingpong",
                         "--down-epilogue-store=bulk_rows",
                         "--down-gather-table-publish-ready"),
                        ("--local-gpu-combine",),
                        ("--local-gpu-combine", "--down-n-tile-major-output",
                         "--down-gather-table", "--down-gather-table-expert-waves",
                         "--down-gather-table-wave-barrier", "--down-pingpong"),
                        ("--local-gpu-combine", "--down-n-tile-major-output",
                         "--down-gather-table", "--down-gather-table-expert-waves",
                         "--down-gather-table-wave-barrier", "--down-pingpong",
                         "--down-gather-table-publish-ready"),
                        ("--down-gather-table",),
                        ("--down-scatter-table", "--down-epilogue-store=bulk_rows_reduce"),
                        ("--down-scatter-table", "--down-multi-buffer-scatter",
                         "--down-scatter-output-buffers=2", "--down-n-tile-major-output",
                         "--down-gather-table", "--down-gather-table-expert-waves",
                         "--down-gather-table-wave-barrier", "--down-pingpong",
                         "--down-epilogue-store=bulk_rows_reduce"),
                        ("--down-gather-table", "--down-gather-table-expert-waves",
                         "--down-gather-table-wave-barrier", "--down-pingpong",
                         "--down-epilogue-store=bulk_rows",
                         "--down-gather-table-publish-ready"),
                        ("--local-gpu-combine", "--down-n-tile-major-output",
                         "--down-gather-table", "--down-gather-table-expert-waves",
                         "--down-gather-table-wave-barrier", "--down-pingpong",
                         "--down-gather-table-publish-ready", "--trace-down-ready"),
                    )
                    import argparse
                    parser = argparse.ArgumentParser()
                    worker.add_down_arguments(parser)
                    for flags in variants:
                        if torch.cuda.get_device_capability()[0] == 10:
                            flags = tuple(flag for flag in flags if flag != "--down-pingpong")
                        with self.subTest(down=flags):
                            options = parser.parse_args(["--down-projection2", *flags])
                            host_flag = None

                            def configure_trace(groups):
                                # Supply mapped host storage without a two-node proxy.
                                # Production allocates it through the proxy C ABI.
                                import ctypes
                                nonlocal host_flag
                                self.assertEqual(groups, 2)
                                host_flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
                                driver = ctypes.CDLL("libcuda.so.1")
                                get_pointer = driver.cuMemHostGetDevicePointer_v2
                                get_pointer.argtypes = [
                                    ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint,
                                ]
                                get_pointer.restype = ctypes.c_int
                                address = ctypes.c_uint64()
                                self.assertEqual(get_pointer(
                                    ctypes.byref(address), host_flag.data_ptr(), 0), 0)
                                return address.value

                            down = worker.DownProjectionRunner(
                                torch, runner, [6, 4], options, trace_configure=configure_trace)
                            # Warmup must not consume/overwrite the live gather
                            # table, and real launches must replace its dummy map.
                            saved_up = runner.output.clone()
                            saved_table = runner.work_table.clone()
                            with torch.inference_mode(False):
                                down.warmup()
                            torch.testing.assert_close(runner.work_table, saved_table)
                            for output in down.inputs.output_buffers():
                                torch.testing.assert_close(output, torch.zeros_like(output))
                            if options.local_gpu_combine:
                                torch.testing.assert_close(
                                    down.combine_valid_rows,
                                    torch.tensor([7, 3], dtype=torch.int64, device="cuda"))
                            runner.output.copy_(saved_up)
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
                                if host_flag is not None:
                                    host_flag.zero_()
                                # Match the worker caller: it does not enter inference mode.
                                with torch.inference_mode(False):
                                    down.launch()
                                torch.cuda.synchronize()
                                if host_flag is not None:
                                    self.assertEqual(host_flag.item(), 2)
                                if options.down_gather_table_publish_ready:
                                    # N=256, tile N=128, swizzle=1: both output groups
                                    # must be published on each invocation.
                                    self.assertEqual(down.inputs.ready_groups.item(), 2)
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
