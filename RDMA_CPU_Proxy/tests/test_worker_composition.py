"""Regression for the actual embedded runners, without NICs or another node.

QUACK_ROOT=/path/to/quack python -m unittest discover -s tests \
    -p test_worker_composition.py -v

CUDA cases run on SM90 and SM100/SM103 and otherwise skip. They stream poisoned
tables, use partial multi-source clusters, run fused SwiGLU -> compact additive
down -> concurrent ready-gated combine, and compare values on every iteration.
CPU metadata covers per-batch publication and a deferred single-slab upload
intentionally delayed until after up completes, with an explicit event dependency.
"""

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import projection_validation as validation

spec = importlib.util.spec_from_file_location("composition_worker", SCRIPTS / "rdma_proxy_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def descriptor_rows(counts, cluster_rows, n_groups, source_order=None):
    """Synthetic proxy order: each batch followed by its repeated N groups."""
    offsets = [[0] for _ in counts]
    for source, values in enumerate(counts):
        for value in values:
            offsets[source].append(offsets[source][-1] + value)
    descriptors = []
    source_order = list(range(len(counts))) if source_order is None else source_order
    for expert in range(len(counts[0])):
        consumed = [0] * len(counts)
        while any(consumed[source] < values[expert] for source, values in enumerate(counts)):
            room, by_source = cluster_rows, {}
            for source in source_order:
                values = counts[source]
                begin = offsets[source][expert] + consumed[source]
                take = min(room, values[expert] - consumed[source])
                by_source[source] = [begin, begin + take]
                consumed[source] += take
                room -= take
            ranges = [value for source in range(len(counts)) for value in by_source[source]]
            for group in range(n_groups):
                descriptors.append([expert, group, *ranges])
    # Arrival order is deliberately different from expert/output order.
    return descriptors[-n_groups:] + descriptors[:-n_groups]


class HostTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class MetadataTests(unittest.TestCase):
    def test_partial_multi_source_descriptors_cover_all_routes(self):
        for arch, tile, cluster_rows in ((9, 128, 256), (9, 256, 512),
                                        (10, 128, 128), (10, 256, 256)):
            with self.subTest(arch=arch, tile=tile):
                counts = [[cluster_rows + 3, 9], [5, cluster_rows // 2 + 1],
                          [cluster_rows // 2 - 1, 0]]
                rows = descriptor_rows(counts, cluster_rows, 4)
                indices = [list(range(sum(values))) for values in counts]
                up = SimpleNamespace(a_idx=[HostTensor(index) for index in indices],
                                     x_buffers=[SimpleNamespace(shape=(len(index), 8)) for index in indices],
                                     work_table=HostTensor(rows))
                routes = sum(map(sum, counts))
                actual = validation.route_metadata(up, routes)
                expected = set()
                for source, values in enumerate(counts):
                    offset = 0
                    for expert, count in enumerate(values):
                        expected.update((expert, source, token) for token in range(offset, offset + count))
                        offset += count
                self.assertEqual(set(actual), expected)
                ordered = []
                for expert in range(len(counts[0])):
                    for source, values in enumerate(counts):
                        start = sum(values[:expert])
                        ordered.extend((expert, source, token)
                                       for token in range(start, start + values[expert]))
                self.assertEqual(actual, ordered)
                for row in rows:
                    self.assertLessEqual(sum(end - start for start, end in zip(row[2::2], row[3::2])),
                                         cluster_rows)
                up.work_table = HostTensor(rows[4:])
                with self.assertRaisesRegex(AssertionError, "uncovered"):
                    validation.route_metadata(up, routes)

    def test_projection_checker_detects_corruption(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        torch.manual_seed(9)
        x = torch.randn(5, 8) * 0.02  # Catch zero output even at tiny proxy magnitudes.
        weights = torch.randn(1, 8, 12)
        value = x[[3, 1, 3]] @ weights[0]
        output = torch.nn.functional.silu(value[:, :6]) * value[:, 6:]
        up = SimpleNamespace(a_idx=(torch.tensor([3, 1, 3]),), x_buffers=(x,),
                             work_table=torch.tensor([[0, 0, 0, 3]]), weights=weights,
                             output=output.clone(), activation="swiglu", gated=True)
        validation.check_projections(torch, up, None, 3, column_chunk=2, row_chunk=2)
        down_weights = torch.randn(1, 6, 4)
        expected = torch.zeros(2, 4)
        expected.index_add_(0, torch.tensor([1, 0, 1]), output @ down_weights[0])
        packed = torch.zeros(2, 10)
        packed.flatten()[:8] = expected.view(2, 2, 2).permute(1, 0, 2).reshape(-1)
        inputs = SimpleNamespace(output_buffers=lambda: (packed,), n_tile_major_tile_n=2,
                                 ready_groups=torch.tensor([2]), wave_ready_prefix=torch.tensor([1, 2]))
        down = SimpleNamespace(inputs=inputs, combine=True, weights=down_weights,
                               combine_valid_rows=torch.tensor([2]),
                               combine_output_to_input=torch.tensor([[1, 3, -1, -1, -1]]))
        validation.check_projections(torch, up, down, 3, column_chunk=2, row_chunk=2)
        packed.flatten()[0] += 1
        with self.assertRaises(AssertionError):
            validation.check_projections(torch, up, down, 3, column_chunk=2)
        up.output.zero_()
        with self.assertRaises(AssertionError):
            validation.check_projections(torch, up, None, 3, column_chunk=2)


class CompositionCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
            raise unittest.SkipTest("requires SM90 or SM100/SM103")
        cls.torch = torch
        cls.arch = torch.cuda.get_device_capability()[0]
        cls.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        cls.torch.backends.cuda.matmul.allow_tf32 = cls.previous_tf32

    def test_streamed_up_compact_down_and_concurrent_combine(self):
        for tile_m in ((128, 256) if self.arch == 10 else (128,)):
            for sm_count in (4, 8):
                with self.subTest(tile_m=tile_m, sm_count=sm_count):
                    self._composition(tile_m, sm_count)

    def test_weighted_streamed_composition(self):
        for tile_m in ((128, 256) if self.arch == 10 else (128,)):
            for preparation in ("gpu", "per-batch", "deferred"):
                for policy in ("distinct", "ones", "zeros"):
                    with self.subTest(tile_m=tile_m, preparation=preparation, weights=policy):
                        self._composition(tile_m, 4, policy=policy, preparation=preparation)

    def test_unweighted_deferred_metadata_after_up_completion(self):
        for tile_m in ((128, 256) if self.arch == 10 else (128,)):
            with self.subTest(tile_m=tile_m):
                self._composition(tile_m, 4, preparation="deferred", publication_default=True)

    def _composition(self, tile_m, sm_count, policy=None, preparation="gpu", publication_default=False):
        torch = self.torch
        root = Path(os.environ.get("QUACK_ROOT", SCRIPTS.parents[2] / "quack_for_communication_overlap"))
        sys.path.insert(0, str(root))
        from quack.gemm_config import cta_tile_shape_m
        from quack.cute_dsl_utils import get_max_active_clusters
        from run.final_combine import FinalCombine
        import cuda.bindings.driver as cuda

        torch.manual_seed(134)
        tile_n, cluster_m, hidden, intermediate = 128, 2, 384, 256
        span = cta_tile_shape_m(tile_m, cluster_m, self.arch) * cluster_m
        counts = [[span + 3, 9], [5, span // 2 + 1], [span // 2 - 1, 0], [0, 0]]
        rows = descriptor_rows(counts, span, 2 * intermediate // tile_n)
        capacity = max(map(sum, counts)) + 13
        sources, input_rows = len(counts), span + 17
        original = {}
        buffers = worker.TorchDeviceBuffers(torch, torch.cuda.current_device(), node_rank=0,
                                            local_gpu_index=0)
        for source in range(sources):
            x = torch.randn(input_rows, hidden, dtype=torch.bfloat16, device="cuda") * 0.2
            index = torch.full((capacity,), -12345, dtype=torch.int32, device="cuda")
            # Original source-token/expert assignments are created before any
            # gather table or packed-row order. Experts share source tokens;
            # each (source, original token, expert) assignment is unique.
            original[source] = []
            selected = []
            values = []
            for expert, count in enumerate(counts[source]):
                for token in range(count):
                    value = (1.0 if policy in (None, "ones") else 0.0 if policy == "zeros"
                             else (source + 1) * 0.125 + (token % 7 + 1) * 0.0625 + expert * 0.1875)
                    selected.append(token)
                    values.append(value)
                    original[source].append((expert, token, value, token))
            index[:len(selected)] = torch.tensor(selected, dtype=torch.int32, device="cuda")
            buffers.tensors[(worker.DEVICE_BUFFER_NVLINK_RECEIVE, -1, 0, source)] = x
            buffers.tensors[(worker.DEVICE_BUFFER_ROUTER_A_IDX, -1, 0, source)] = index
            if policy is not None:
                weights = torch.full((capacity,), float("nan"), dtype=torch.float32, device="cuda")
                weights[:len(values)] = torch.tensor(values, dtype=torch.float32, device="cuda")
                buffers.tensors[(worker.DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS, -1, 0, source)] = weights
        table = torch.tensor(rows, dtype=torch.int32, device="cuda")
        ready = torch.full((1,), len(rows), dtype=torch.int32, device="cuda")
        buffers.tensors[(worker.DEVICE_BUFFER_GATHER_TABLE, -1, -1, -1)] = table
        buffers.tensors[(worker.DEVICE_BUFFER_GATHER_READY_ROWS, -1, -1, -1)] = ready
        up = worker.GroupedGemmRunner(
            torch, buffers, cuda_device_id=torch.cuda.current_device(), world_size=1,
            num_experts=2, output_dimension=intermediate, activation="swiglu",
            tile_m=tile_m, tile_n=tile_n, tile_k=None, cluster_m=cluster_m,
            max_swizzle_size=1, quack_root=root)
        up.original_reference = original
        up.original_reference_recompute = True
        parser = argparse.ArgumentParser()
        worker.add_down_arguments(parser)
        options = parser.parse_args([
            "--down-projection2", "--local-gpu-combine", "--down-n-tile-major-output",
            "--down-gather-table", "--down-gather-table-expert-waves",
            "--down-gather-table-wave-barrier", "--down-gather-table-publish-ready",
            "--trace-down-ready", f"--down-tile-m={tile_m}", f"--down-tile-n={tile_n}",
            "--down-cluster-m=2", "--down-max-swizzle-size=1", f"--down-sm-count={sm_count}"])
        options.weighted_contributions = policy is not None
        options.cpu_scatter_preparation = preparation != "gpu"
        options.cpu_scatter_publication = (preparation if preparation != "gpu" and not publication_default
                                           else None)
        host_ready = torch.zeros(1, dtype=torch.int32, pin_memory=True)

        def configure_trace(groups):
            self.assertEqual(groups, hidden // tile_n)
            status, pointer = cuda.cuMemHostGetDevicePointer(host_ready.data_ptr(), 0)
            self.assertEqual(status, cuda.CUresult.CUDA_SUCCESS)
            return int(pointer)

        expert_counts = [sum(values[expert] for values in counts) for expert in range(2)]
        routes = sum(expert_counts)
        down = worker.DownProjectionRunner(torch, up, expert_counts, options,
                                           trace_configure=configure_trace)
        if preparation != "gpu":
            self.assertEqual(down.cpu_scatter_publication, preparation)
        down.warmup()
        # Build initial compact maps before simulated CPU publication starts.
        down.cpu_scatter = False
        up.launch()
        down.launch()
        torch.cuda.synchronize()
        expected_work = sum(math.ceil(count / span) for count in expert_counts) * (hidden // tile_n)
        self.assertEqual(down.inputs.wave_work_ids,
                         min(get_max_active_clusters(2, sm_count=sm_count), expected_work))
        self.assertTrue(all(end - begin <= span
                            for _, begin, end, _ in down.inputs.gather_work_table.cpu().tolist()))
        valid = down.combine_valid_rows.tolist()
        indices = [down.combine_output_to_input[source, :size].clone()
                   for source, size in enumerate(valid)]
        compact = [output.flatten()[:size * hidden]
                   for output, size in zip(down.inputs.output_buffers(), valid)]
        combine = FinalCombine(compact, indices, input_rows, hidden, tile_n=tile_n, num_ctas=2,
                               register_prefetch=True, ready_groups=down.inputs.ready_groups)
        combine()  # Warm the real callable before any waiting kernel is launched.
        torch.cuda.synchronize()
        producer, publication, consumer = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        pinned_table = table.cpu().pin_memory()
        pinned_counts = [torch.tensor([end], dtype=torch.int32, pin_memory=True)
                         for end in range(1, len(rows) + 1)]
        pinned_x = [x.cpu().pin_memory() for x in up.x_buffers]

        pinned_route_weights = None
        pinned_addresses = None
        pinned_metadata = None
        publication_up_done = None
        metadata_ready = torch.cuda.Event()

        def publish():
            # H2D copies use copy engines: a spinning full-SM up GEMM cannot
            # starve a CUDA fill kernel needed to publish its own readiness.
            with torch.inference_mode(), torch.cuda.stream(publication):
                for destination, source in zip(up.x_buffers, pinned_x):
                    destination.copy_(source, non_blocking=True)
                for index, count in enumerate(pinned_counts):
                    if (preparation == "per-batch" and pinned_addresses is not None
                            and pinned_table[index, 1].item() == 0):
                        # Mirrors native CPU scatter preparation: route weights
                        # and addresses become visible before gather readiness.
                        # The actual C++ publisher is covered by test_cpu_scatter_cuda.
                        ranges = pinned_table[index, 2:].tolist()
                        begin = sum(ranges[::2])
                        end = begin + sum(stop - start for start, stop in zip(ranges[::2], ranges[1::2]))
                        if pinned_route_weights is not None:
                            down.route_weights[begin:end].copy_(pinned_route_weights[begin:end], non_blocking=True)
                        down.inputs.scatter_row_addresses[begin:end].copy_(pinned_addresses[begin:end],
                                                                          non_blocking=True)
                    table[index:index + 1].copy_(pinned_table[index:index + 1], non_blocking=True)
                    ready.copy_(count, non_blocking=True)
                if pinned_metadata is not None:
                    # Intentionally defer metadata beyond ALL up computation.
                    # Gathering readiness can no longer imply metadata visibility.
                    publication.wait_event(publication_up_done)
                    down.cpu_scatter_metadata.copy_(pinned_metadata, non_blocking=True)
                    # Recording is submitted before the consumer stream waits;
                    # otherwise a reused event could observe a previous iteration.
                    metadata_ready.record(publication)

        publish()
        publication.synchronize()
        for iteration in range(3):
            changed = descriptor_rows(counts, span, 2 * intermediate // tile_n,
                                      list(range(sources))[::-1] if iteration % 2 else None)
            pinned_table.copy_(torch.tensor(changed, dtype=torch.int32))
            table.copy_(pinned_table)
            if preparation != "gpu":
                # Use the tested GPU address builder only to obtain invariant
                # compact destination addresses; source weights are packed by a
                # separate CPU descriptor walk, never by the implementation.
                down.cpu_scatter = False
                with torch.inference_mode():
                    down._prepare_combine(up.work_table, up.a_idx)
                torch.cuda.synchronize()
                pinned_addresses = down.inputs.scatter_row_addresses.cpu().pin_memory()
                if policy is not None:
                    expected_values = [up.route_weights[source][position].item()
                                       for _, source, position in validation.route_positions(up, routes)]
                    pinned_route_weights = torch.tensor(expected_values, dtype=torch.float32).pin_memory()
                if preparation == "deferred":
                    pinned_metadata = down.cpu_scatter_metadata.cpu().pin_memory()
                    self.assertEqual(down.inputs.scatter_row_addresses.data_ptr(),
                                     down.cpu_scatter_metadata.data_ptr())
                    if pinned_route_weights is not None:
                        offset = down.route_weights.data_ptr() - down.cpu_scatter_metadata.data_ptr()
                        self.assertEqual(offset % 16, 0)
                        pinned_metadata[offset:offset + routes * 4].view(torch.float32).copy_(pinned_route_weights)
                down.cpu_scatter = True
            for source in pinned_x:
                source.mul_(-1.1)  # New values each iteration detect stale reads.
            with torch.inference_mode():
                up.output.fill_(float("nan"))
                table.fill_(-1)
                ready.zero_()
                down.inputs.wave_store_workspace.zero_()
                combine.output.fill_(float("nan"))
                if down.route_weights is not None:
                    down.route_weights.fill_(float("nan"))
                if preparation == "deferred":
                    # Includes invalid addresses and NaN weights. Only the
                    # current-generation slab upload can make down safe to run.
                    down.cpu_scatter_metadata.fill_(255)
            host_ready.zero_()
            torch.cuda.synchronize()
            with torch.cuda.stream(producer):
                up.launch()  # Enqueue before any descriptor/input publication.
                if preparation == "deferred":
                    publication_up_done = torch.cuda.Event()
                    publication_up_done.record(producer)
            publish()
            with torch.cuda.stream(producer):
                if preparation == "deferred":
                    down.launch(wait_for_metadata=lambda: producer.wait_event(metadata_ready))
                else:
                    down.launch()
            with torch.cuda.stream(consumer):
                combine()  # Acquires GPU ready prefixes while down stores continue.
            producer.synchronize()
            publication.synchronize()
            consumer.synchronize()
            self.assertEqual(host_ready.item(), hidden // tile_n)
            self.assertEqual(down.inputs.ready_groups.item(), hidden // tile_n)
            self.assertTrue(torch.isnan(up.output[routes:]).all().item())
            validation.check_projections(torch, up, down, routes, column_chunk=128)
            expected = torch.zeros(input_rows, hidden, device="cuda")
            for selected, output in zip(indices, down.collected_combine_outputs()):
                expected.index_add_(0, selected.long(), output.float())
            torch.testing.assert_close(combine.output, expected.to(torch.bfloat16), rtol=0, atol=0)
            # Full independent end-to-end reference uses original assignments,
            # original source input rows, and both expert matrices. It never
            # reads A_idx, a work/scatter table, or the packed scaling vector.
            independent = torch.zeros(input_rows, hidden, device="cuda")
            for source, assignments in original.items():
                for expert in range(2):
                    selected = [(token, value) for exp, token, value, _ in assignments if exp == expert]
                    if not selected:
                        continue
                    tokens = torch.tensor([token for token, _ in selected], device="cuda")
                    scale = torch.tensor([value for _, value in selected], device="cuda")
                    gate_up = pinned_x[source][tokens.cpu()].to("cuda").float() @ up.weights[expert].float()
                    intermediate_ref = (torch.nn.functional.silu(gate_up[:, :intermediate]) *
                                        gate_up[:, intermediate:]).to(up.output.dtype)
                    contribution = ((intermediate_ref.float() @ down.weights[expert].float()) *
                                    scale[:, None]).to(up.output.dtype).float()
                    independent.index_add_(0, tokens, contribution)
            validation._assert_values(torch, combine.output, independent, "independent original composition",
                                      reduction=True)
            for output, size in zip(down.inputs.output_buffers(), valid):
                torch.testing.assert_close(output.flatten()[size * hidden:],
                                           torch.zeros_like(output.flatten()[size * hidden:]))


if __name__ == "__main__":
    unittest.main()
