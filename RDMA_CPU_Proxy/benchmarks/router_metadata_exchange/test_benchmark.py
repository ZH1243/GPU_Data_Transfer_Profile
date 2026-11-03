"""Semantic checks and real multi-process loopback transport smoke tests."""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import python_reference as b


def config(**overrides):
    args = dict(rank=0, world=4, hosts=["node0", "node1"], gpus_per_node=2,
                num_tokens=53, num_experts=16, top_k=4, seed=1234,
                token_dimension=4096, element_bytes=2, tokens_per_chunk=7)
    args.update(overrides)
    return SimpleNamespace(**args)


class MetadataTests(unittest.TestCase):
    def test_routing_semantics(self):
        a = config()
        for source in range(a.world):
            data = b.prepare(a, source, 0)
            # Independently reconstruct original R and resolve received buffer
            # indices back to token IDs, checking every remote expert's membership.
            routes = []
            for token in range(a.num_tokens):
                random0 = b.splitmix64((a.seed + source) ^ token)
                stride = 1 + b.splitmix64(random0) % (a.num_experts - 1)
                while math.gcd(stride, a.num_experts) != 1:
                    stride = stride + 1 if stride + 1 < a.num_experts else 1
                routes.append({(random0 % a.num_experts + k * stride) % a.num_experts
                               for k in range(a.top_k)})
            self.assertEqual(sum(len(data[(d, "expert")][3]) for d in range(a.world)),
                             a.num_tokens * a.top_k)
            self.assertNotEqual(data, b.prepare(a, source, 1))
            for dest in range(a.world):
                _, _, offsets, indices = data[(dest, "expert")]
                receiver = config(rank=dest)
                installed = {}
                b.install(receiver, source, "expert", b.encode(data[(dest, "expert")]), installed)
                self.assertEqual(installed[(source, "expert")]["indices"], indices)
                if dest // a.gpus_per_node == source // a.gpus_per_node:
                    continue
                ingress = dest // a.gpus_per_node * a.gpus_per_node + source % a.gpus_per_node
                x3, x4 = data[(ingress, "x3x4")][2:]
                self.assertEqual(x4, sorted(x4, reverse=True))
                if dest % a.gpus_per_node == source % a.gpus_per_node:
                    buffer_tokens = x3
                else:
                    bit = (source - dest) % a.gpus_per_node
                    buffer_tokens = [t for t, mask in zip(x3, x4) if mask & (1 << bit)]
                for local_expert in range(a.num_experts // a.world):
                    expert = dest * (a.num_experts // a.world) + local_expert
                    actual = [buffer_tokens[i] for i in indices[offsets[local_expert]:offsets[local_expert + 1]]]
                    expected = [token for token in buffer_tokens if expert in routes[token]]
                    self.assertEqual(actual, expected)

    def test_x3_install_and_invalid_metadata(self):
        a = config(rank=2)
        data = b.prepare(a, 0, 0)
        payload = b.encode(data[(2, "x3x4")])
        installed = {}
        b.install(a, 0, "x3x4", payload, installed)
        record = installed[(0, "x3x4")]
        self.assertEqual([t for c in record["chunks"] for t in c["source_token_indices"]], record["indices"])
        self.assertEqual(sum(c["length_bytes"] for c in record["chunks"]),
                         len(record["indices"]) * a.token_dimension * a.element_bytes)
        with self.assertRaises(ValueError):
            b.install(a, 0, "x3x4", payload, installed)
        bad = list(data[(2, "expert")])
        bad[3] = [a.num_tokens] * len(bad[3])
        with self.assertRaises(ValueError):
            b.install(a, 0, "expert", b.encode(bad), {})

    def test_empty_expert_lists(self):
        a = config(num_tokens=1, top_k=1)
        data = b.prepare(a, 0, 0)
        for dest in range(a.world):
            a.rank = dest
            b.install(a, 0, "expert", b.encode(data[(dest, "expert")]), {})


class ProcessTests(unittest.TestCase):
    def test_native_timeout_reaches_python(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = dict(os.environ, RANK="0", WORLD_SIZE="2", LOCAL_RANK="0")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("benchmark.py")),
             "--hosts=127.0.0.1,127.0.0.1", "--gpus-per-node=1",
             f"--port-base={port}", "--timeout=0.1"], env=env,
            capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TCP operation timed out", result.stderr)

    def run_job(self, nodes, per_node, preserialized=False, num_tokens=137,
                num_experts=16, top_k=4, wire_format="binary", disable_validation=False, local_shared_memory=False):
        # Distinct node port ranges emulate hosts without loopback aliases.
        hosts = ["127.0.0.1"] * nodes
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            processes = []
            logs = []
            try:
                for rank in range(nodes * per_node):
                    log = open(Path(directory) / f"rank{rank}.log", "w+")
                    logs.append(log)
                    env = dict(os.environ, RANK=str(rank), WORLD_SIZE=str(nodes * per_node),
                               LOCAL_RANK=str(rank % per_node))
                    cmd = [sys.executable, str(Path(__file__).with_name("benchmark.py")),
                           "--hosts=" + ",".join(hosts), f"--gpus-per-node={per_node}",
                           f"--bind-host={hosts[rank // per_node]}", f"--port-base={port}",
                           f"--node-port-stride={per_node}",
                           f"--wire-format={wire_format}",
                           f"--num-tokens={num_tokens}", f"--num-experts={num_experts}", f"--top-k={top_k}",
                           "--datasets=2", "--warmup=1", "--iterations=3", "--timeout=60",
                           f"--output={output}"]
                    if local_shared_memory:
                        cmd.append("--local-shared-memory")
                    if preserialized:
                        cmd.append("--preserialized")
                    if disable_validation:
                        cmd.append("--disable-expert-index-validation")
                    if num_tokens < 1000:
                        cmd.append("--verify")
                    processes.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=log))
                for process, log in zip(processes, logs):
                    code = process.wait(timeout=90)
                    log.seek(0)
                    self.assertEqual(code, 0, log.read())
                report = json.loads(output.read_text())
                self.assertEqual(report["backend"], "cpp")
                self.assertEqual(report["configuration"]["local_shared_memory"], local_shared_memory)
                self.assertEqual(report["expert_index_validation_enabled"], not disable_validation)
                self.assertEqual(report["configuration"]["expert_index_validation_enabled"], not disable_validation)
                self.assertEqual(report["wire_format"], wire_format)
                self.assertEqual(report["configuration"]["wire_format"], wire_format)
                self.assertEqual(report["configuration"]["timing_schema"], 4)
                self.assertEqual(report["configuration"]["receive_storage"],
                                 "direct_reusable_vectors" if wire_format == "binary" else "payload_buffer")
                self.assertEqual(len(report["iterations"]), 3)
                critical = [max(rows, key=lambda r: r["elapsed_ms"]) for rows in report["iterations"]]
                self.assertAlmostEqual(report["critical_rank_expert_index_bounds_ms"]["mean"],
                                       sum(r["expert_index_bounds_ms"] for r in critical) / len(critical), places=7)
                self.assertAlmostEqual(sum(stats["mean"] for stats in
                                           report["critical_rank_receive_process_ms"].values()),
                                       report["critical_rank_stage_ms"]["receive_process_ms"]["mean"], places=7)
                self.assertAlmostEqual(sum(report["critical_rank_stage_ms"][field]["mean"]
                                           for field in b.STAGE_FIELDS),
                                       report["critical_rank_ms"]["mean"], places=7)
                for row in report["iterations"]:
                    self.assertEqual(sum(r["sent_bytes"] for r in row), sum(r["received_bytes"] for r in row))
                    self.assertEqual(sum(r["shm_sent_bytes"] for r in row),
                                     sum(r["shm_received_bytes"] for r in row))
                    if local_shared_memory and per_node > 1:
                        self.assertGreater(sum(r["shm_sent_bytes"] for r in row), 0)
                    else:
                        self.assertEqual(sum(r["shm_sent_bytes"] for r in row), 0)
                    if local_shared_memory and nodes == 1:
                        self.assertEqual(sum(r["sent_bytes"] + r["received_bytes"] for r in row), 0)
                    for result in row:
                        self.assertEqual(result["exchange_ms"], result["tcp_exchange_ms"])
                        self.assertEqual(result["installed_expert_sources"], nodes * per_node)
                        self.assertEqual(result["installed_x3x4_sources"], nodes - 1)
                        self.assertGreater(result["elapsed_ms"], 0)
                        breakdown = result["receive_process_breakdown_ms"]
                        self.assertGreaterEqual(result["expert_index_bounds_ms"], 0)
                        self.assertLessEqual(result["expert_index_bounds_ms"], breakdown["expert_deserialize_ms"])
                        if wire_format == "text" or disable_validation:
                            self.assertEqual(result["expert_index_bounds_ms"], 0)
                        self.assertEqual(len(breakdown), 9)
                        for value in breakdown.values():
                            self.assertGreaterEqual(value, 0)
                        self.assertAlmostEqual(sum(breakdown.values()), result["receive_process_ms"], places=7)
                        if nodes == 1:
                            for field in ("x3x4_deserialize_ms", "x3x4_validate_ms",
                                          "chunk_build_ms", "x3x4_record_install_ms"):
                                self.assertEqual(breakdown[field], 0)
                        elif num_tokens > 1:
                            self.assertGreater(breakdown["chunk_build_ms"], 0)
                        for field in b.STAGE_FIELDS:
                            self.assertGreaterEqual(result[field], 0)
                        self.assertAlmostEqual(sum(result[field] for field in b.STAGE_FIELDS),
                                               result["elapsed_ms"], places=7)
                        self.assertAlmostEqual(result["install_ms"],
                                               result["self_install_ms"] + result["receive_process_ms"], places=7)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                for log in logs:
                    log.close()

    def test_shared_memory_local_only(self):
        self.run_job(1, 4, local_shared_memory=True)

    def test_shared_memory_hybrid(self):
        self.run_job(2, 2, local_shared_memory=True)

    def test_shared_memory_preserialized_validation_disabled(self):
        self.run_job(2, 2, preserialized=True, disable_validation=True, local_shared_memory=True)

    def test_shared_memory_empty_lists(self):
        self.run_job(2, 2, num_tokens=1, top_k=1, local_shared_memory=True)

    def test_shared_memory_single_proxy_per_node(self):
        self.run_job(2, 1, local_shared_memory=True)

    def test_shared_memory_default_workload(self):
        self.run_job(2, 8, num_tokens=20480, num_experts=128, top_k=8, local_shared_memory=True)

    def test_shared_memory_rejects_text(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("benchmark.py")),
             "--hosts=127.0.0.1", "--local-shared-memory", "--wire-format=text"],
            capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--local-shared-memory requires --wire-format=binary", result.stderr)

    def test_four_local_proxies(self):
        self.run_job(1, 4)

    def test_two_nodes_matching_gpu(self):
        self.run_job(2, 2)

    def test_preserialized(self):
        self.run_job(2, 2, preserialized=True)

    def test_text_baseline(self):
        self.run_job(2, 2, wire_format="text")

    def test_disable_expert_validation(self):
        self.run_job(2, 2, disable_validation=True)

    def test_text_preserialized(self):
        self.run_job(2, 2, preserialized=True, wire_format="text")

    def test_binary_empty_lists(self):
        self.run_job(2, 2, num_tokens=1, top_k=1)

    def test_default_workload_sixteen_proxies(self):
        self.run_job(2, 8, num_tokens=20480, num_experts=128, top_k=8)


if __name__ == "__main__":
    unittest.main()
