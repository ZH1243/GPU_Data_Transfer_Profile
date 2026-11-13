"""CPU checks for route identities and independent weighted projection references.

The CUDA composition and native CPU-publisher tests cover device execution.
These tests run without a GPU and never load librdma_cpu_proxy.so.
"""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import projection_validation as validation
from test_worker_composition import HostTensor, descriptor_rows, worker


class OriginalAssignmentsTests(unittest.TestCase):
    def test_original_reference_setup_uses_small_descriptors_and_cleans_files(self):
        selected, weights = [0, 2, 3, 4], [0.25, 0.75, 0.5, 0.5]

        def query(handle, experts, values, capacity, count, tokens, top_k):
            count._obj.value, tokens._obj.value, top_k._obj.value = 4, 2, 2
            if experts is not None:
                self.assertEqual(capacity, 4)
                experts[:] = selected
                values[:] = weights
            return 0

        stages = []

        def gather(stage, value=True):
            stages.append(stage)
            if isinstance(value, dict):
                self.assertEqual(set(value), {"path", "tokens", "top_k"})
                self.assertEqual(Path(value["path"]).stat().st_size, 32)
                return [value, value]
            return [value, value]

        exchange = SimpleNamespace(require=lambda status: self.assertEqual(status, 0),
                                   control=SimpleNamespace(gather=gather), node=0, rank=0, world=2)
        up = SimpleNamespace(source_pairs=[(0, 0), (0, 1), (1, 0), (1, 1)])
        make_temp = tempfile.mkstemp
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(validation.tempfile, "mkstemp",
                                   side_effect=lambda **kwargs: make_temp(dir=directory)):
                validation.configure_original_reference(
                    up, None, SimpleNamespace(rdma_proxy_get_local_router_assignments=query),
                    None, exchange, num_experts=8)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(stages, ["original-route-reference-files", "original-route-reference-read"])
        self.assertEqual(set(up.original_reference), {0, 1})
        self.assertEqual(up.original_reference[0], [(0, 0, 0.25, 0)])
        self.assertEqual(up.original_reference[1], [(0, 0, 0.25, 0)])

    def test_same_gpu_x3_and_forwarded_x4_are_different_rows(self):
        # Two experts/GPU, two GPUs/node, two nodes. Source GPU 0 uses
        # bit 0 for GPU 0 and bit 1 for GPU 1; descending mask order is
        # token 1 (mask 3), token 0 (mask 2), token 2 (mask 1).
        selected = [[2, 4], [0, 3], [1, 5], [6, 7]]
        weights = [[0.25, 0.75], [0.125, 0.875], [0.5, 0.5], [0.0, 1.0]]
        common = dict(source_gpu=0, destination_node=0, num_nodes=2,
                      gpus_per_node=2, num_experts=8)
        same = validation.original_source_routes(selected, weights, destination_gpu=0, **common)
        forwarded = validation.original_source_routes(selected, weights, destination_gpu=1, **common)
        self.assertEqual(same, [(0, 0, 0.125, 1), (1, 2, 0.5, 2)])
        self.assertEqual(forwarded, [(1, 0, 0.875, 1), (0, 1, 0.25, 0)])
        self.assertEqual(validation.original_source_routes(
            [], [], destination_gpu=1, **common), [])

    def test_original_assignments_reject_invalid_values(self):
        common = dict(source_gpu=0, destination_node=0, destination_gpu=0,
                      num_nodes=1, gpus_per_node=1, num_experts=2)
        for selected, weights in [([[0, 0]], [[0.25, 0.75]]), ([[0]], [[-1.0]]),
                                  ([[0]], [[float("nan")]]), ([[2]], [[1.0]])]:
            with self.subTest(selected=selected, weights=weights), self.assertRaises(AssertionError):
                validation.original_source_routes(selected, weights, **common)

    def test_exact_source_list_positions_survive_arrival_changes(self):
        counts = [[5, 2], [3, 4], [0, 0]]
        indices = [[0, 1, 2, 0, 1, 0, 1], [1, 1, 2, 0, 1, 1, 2], []]
        weights = [[0.125 * (i + 1) for i in range(len(index))] for index in indices]
        up = SimpleNamespace(a_idx=[HostTensor(value) for value in indices],
                             x_buffers=[SimpleNamespace(shape=(3, 8)) for _ in indices],
                             route_weights=[HostTensor(value) for value in weights])
        previous = None
        for order in ([0, 1, 2], [2, 1, 0], [0, 1, 2]):
            up.work_table = HostTensor(descriptor_rows(counts, 4, 3, order))
            packed = validation.route_weight_reference(up, None, 14)
            down = SimpleNamespace(route_weights=HostTensor(packed))
            self.assertEqual(validation.route_weight_reference(up, down, 14), packed)
            self.assertEqual(sorted(packed), sorted(sum(weights, [])))
            if previous is not None:
                self.assertNotEqual(packed, previous)
            previous = packed
            bad = list(packed)
            bad[0], bad[-1] = bad[-1], bad[0]
            down.route_weights = HostTensor(bad)
            with self.assertRaisesRegex(AssertionError, "source/list positions"):
                validation.route_weight_reference(up, down, 14)


class WeightedReferenceCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        cls.torch = torch

    def test_production_gpu_mapping_on_cpu_matches_independent_positions(self):
        torch = self.torch
        counts = [[5, 2], [3, 4], [0, 0]]
        for order in ([0, 1, 2], [2, 1, 0]):
            table = torch.tensor(descriptor_rows(counts, 4, 3, order), dtype=torch.int32)
            up = SimpleNamespace(work_table=table,
                                 a_idx=[HostTensor(list(range(sum(value)))) for value in counts],
                                 x_buffers=[SimpleNamespace(shape=(7, 1)) for _ in counts])
            source, position = worker.local_route_positions(torch, table, 3, 14, 3)
            expected = validation.route_positions(up, 14)
            self.assertEqual(list(zip(source.tolist(), position.tolist())),
                             [(src, index) for _, src, index in expected])

    def _composition(self, policy, order):
        torch = self.torch
        torch.manual_seed(47)
        # Multiple sources and experts contribute to original token zero.
        assignments = {0: [(0, 0, 0.125), (0, 1, 0.5), (1, 0, 0.875)],
                       1: [(0, 0, 0.75), (1, 0, 0.25), (1, 2, 0.0)], 2: []}
        if policy != "distinct":
            assignments = {source: [(expert, token, 1.0 if policy == "ones" else 0.0)
                                     for expert, token, _ in routes]
                           for source, routes in assignments.items()}
        x = tuple(torch.randn(3, 8) * 0.2 for _ in assignments)
        wu, wd = torch.randn(2, 8, 12), torch.randn(2, 6, 4)
        idx = tuple(torch.tensor([token for _, token, _ in routes], dtype=torch.int64)
                    for routes in assignments.values())
        weights = tuple(torch.tensor([value for _, _, value in routes], dtype=torch.float32)
                        for routes in assignments.values())
        counts = [[sum(expert == e for expert, _, _ in routes) for e in range(2)]
                  for routes in assignments.values()]
        table = torch.tensor(descriptor_rows(counts, 2, 3, order), dtype=torch.int32)
        up = SimpleNamespace(a_idx=idx, x_buffers=x, route_weights=weights, work_table=table,
                             weights=wu, activation="swiglu", gated=True,
                             original_reference={source: [(e, token, value, token) for e, token, value in routes]
                                                 for source, routes in assignments.items()},
                             original_reference_recompute=True)
        metadata = validation.route_metadata(up, 6)
        output = torch.empty(6, 6)
        for row, (expert, source, token) in enumerate(metadata):
            pre = x[source][token] @ wu[expert]
            output[row] = torch.nn.functional.silu(pre[:6]) * pre[6:]
        up.output = output
        expected = []
        inverse = []
        sizes = []
        for source, original in assignments.items():
            tokens = sorted({token for _, token, _ in original})
            inverse.append(tokens + [-1] * (3 - len(tokens)))
            sizes.append(len(tokens))
            result = torch.zeros(len(tokens), 4)
            # Deliberately avoid gather metadata in this reference.
            for expert, token, value in original:
                pre = x[source][token] @ wu[expert]
                intermediate = torch.nn.functional.silu(pre[:6]) * pre[6:]
                result[tokens.index(token)] += (intermediate @ wd[expert]) * value
            physical = torch.zeros(3, 4)
            if tokens:
                physical.flatten()[:len(tokens) * 4] = result.view(len(tokens), 2, 2).permute(1, 0, 2).reshape(-1)
            expected.append(physical)
        inputs = SimpleNamespace(output_buffers=lambda: tuple(expected), n_tile_major_tile_n=2,
                                 ready_groups=torch.tensor([2]), wave_ready_prefix=torch.tensor([1, 2]))
        down = SimpleNamespace(inputs=inputs, combine=True, weights=wd,
                               combine_valid_rows=torch.tensor(sizes),
                               combine_output_to_input=torch.tensor(inverse),
                               route_weights=torch.tensor(validation.route_weight_reference(up, None, 6)))
        return up, down

    def test_independent_full_composition_and_corruption(self):
        torch = self.torch
        for policy in ("distinct", "ones", "zeros"):
            for order in ([0, 1, 2], [2, 1, 0]):
                with self.subTest(policy=policy, order=order):
                    up, down = self._composition(policy, order)
                    validation.check_projections(torch, up, down, 6, column_chunk=2, row_chunk=2)
                    if policy == "ones":
                        up.route_weights = down.route_weights = None
                        validation.check_projections(torch, up, down, 6, column_chunk=2, row_chunk=2)
                    down.inputs.output_buffers()[0].flatten()[0] += 1
                    with self.assertRaises(AssertionError):
                        validation.check_projections(torch, up, down, 6, column_chunk=2, row_chunk=2)

    def test_original_assignment_check_rejects_shared_mapping_error(self):
        torch = self.torch
        up, down = self._composition("distinct", [0, 1, 2])
        # Corrupt immutable metadata and its packed derivative consistently.
        # Only the original-assignment reference can detect this shared error.
        up.route_weights[0][0] = 0.75
        down.route_weights.copy_(torch.tensor(validation.route_weight_reference(up, None, 6)))
        with self.assertRaisesRegex(AssertionError, "original token/expert/weight"):
            validation.check_projections(torch, up, down, 6)

    def test_weight_before_fp16_conversion_avoids_overflow(self):
        torch = self.torch
        x = torch.tensor([[512.0]], dtype=torch.float16)
        up = SimpleNamespace(a_idx=(torch.tensor([0]),), x_buffers=(x,),
                             work_table=torch.tensor([[0, 0, 0, 1]]),
                             weights=torch.ones(1, 1, 1, dtype=torch.float16),
                             output=x.clone(), activation=None, gated=False,
                             route_weights=(torch.tensor([0.5]),))
        result = torch.tensor([[32768.0]], dtype=torch.float16)
        inputs = SimpleNamespace(output_buffers=lambda: (result,), n_tile_major_tile_n=None,
                                 ready_groups=None, scatter_output_rows=None, scatter_table=None)
        down = SimpleNamespace(inputs=inputs, combine=False,
                               weights=torch.tensor([[[128.0]]], dtype=torch.float16),
                               route_weights=torch.tensor([0.5]))
        validation.check_projections(torch, up, down, 1)


if __name__ == "__main__":
    unittest.main()
