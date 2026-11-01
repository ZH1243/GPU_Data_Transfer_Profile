#!/usr/bin/env python3
"""Single-GPU, all-inputs-ready benchmark of the production intra-node kernel."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quack-root', type=Path, default=Path(os.environ.get(
        'QUACK_ROOT', str(Path(__file__).resolve().parents[3] / 'quack_for_communication_overlap'))))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--rows-per-domain', type=int, default=20480,
                        help='output rows per source-node domain, not pre-routing token count')
    parser.add_argument('--domains', type=int, default=2)
    parser.add_argument('--peers', type=int, default=8)
    parser.add_argument('--contributors', type=int, default=4,
                        help='distinct contributing GPUs per row (synthetic fixed fan-in)')
    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--tile-n', type=int, default=128)
    parser.add_argument('--group-tiles', type=int, default=1)
    parser.add_argument('--ctas', type=int, default=8)
    parser.add_argument('--threads', type=int, choices=(256, 512), default=512)
    parser.add_argument('--elements-per-thread', type=int, choices=(8, 16, 32, 64, 128), default=32)
    parser.add_argument('--prefetch-peers', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--readiness-polling', choices=('group', 'vector'), default='group',
                        help='group: optimized acquisition; vector: original kernel baseline')
    parser.add_argument('--domain-traversal', choices=('local', 'all'), default='local',
                        help='local: scan only this token domain; all: previous baseline')
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--no-host-publication', action='store_true',
                        help='omit mapped-host ready-prefix store used by RDMA return')
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    for name in ('rows_per_domain', 'domains', 'hidden', 'tile_n', 'group_tiles', 'ctas', 'iterations'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.warmup < 0 or not 1 <= args.contributors <= args.peers <= 8:
        parser.error('require warmup >= 0 and 1 <= contributors <= peers <= 8')
    if args.hidden % args.tile_n or args.tile_n % args.elements_per_thread:
        parser.error('tile-n must divide hidden and elements-per-thread must divide tile-n')
    if args.domains * args.rows_per_domain * args.hidden > 2**31 - 1:
        parser.error('output exceeds kernel int32 indexing range')
    args.quack_root = args.quack_root.expanduser().resolve()
    if not (args.quack_root / 'run' / 'final_combine.py').is_file():
        parser.error(
            f'QuACK checkout not found at {args.quack_root}: expected run/final_combine.py. '
            'Set --quack-root (or QUACK_ROOT) to your actual quack_for_communication_overlap '
            'checkout. Omit --quack-root to use QUACK_ROOT or the sibling checkout default; '
            '/path/to/... is only a documentation placeholder.')
    return args


def main():
    args = parse_args()
    sys.path.insert(0, str(args.quack_root.resolve()))
    import torch
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from intra_node_combine_kernel import RegisterIntraNodeKernel

    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError('This benchmark targets Hopper (SM90)')
    properties = torch.cuda.get_device_properties(args.device)
    # The production publication algorithm polls other CTAs. Avoid oversubscribing
    # the device with blocks that might wait on blocks which have not been scheduled.
    if args.ctas > properties.multi_processor_count:
        raise ValueError('ctas must not exceed the device SM count')
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dtype = getattr(torch, args.dtype)
    total_rows = args.domains * args.rows_per_domain
    tiles = args.hidden // args.tile_n
    groups = (tiles + args.group_tiles - 1) // args.group_tiles
    domains = tuple((d * args.rows_per_domain, args.rows_per_domain) for d in range(args.domains))
    selected = []
    input_domains = []
    for offset, count in domains:
        peer_rows = [[] for _ in range(args.peers)]
        for row in range(count):
            for peer in rng.sample(range(args.peers), args.contributors):
                peer_rows[peer].append(offset + row)
        active_rows = [rows for rows in peer_rows if rows]
        input_domains.append((offset, count, len(selected), len(active_rows)))
        selected.extend(active_rows)
    counts = tuple(map(len, selected))
    rows = torch.full((len(counts), total_rows), -1, dtype=torch.int32, device='cuda')
    buffers = []
    # Build the independent FP32 reference in the same deterministic peer order.
    expected = torch.zeros((total_rows, args.hidden), dtype=torch.float32, device='cuda')
    for p, selected_rows in enumerate(selected):
        index = torch.tensor(selected_rows, dtype=torch.int64, device='cuda')
        rows[p, index] = torch.arange(len(selected_rows), dtype=torch.int32, device='cuda')
        values = torch.randn((len(selected_rows), args.hidden), dtype=dtype, device='cuda')
        expected[index] += values.float()
        buffers.append(values.view(len(selected_rows), tiles, args.tile_n)
                       .permute(1, 0, 2).contiguous().view(-1))
    del values, index
    output = torch.empty(total_rows * args.hidden, dtype=dtype, device='cuda')
    ready = torch.full((1,), groups, dtype=torch.int32, device='cuda')
    cancel = torch.zeros_like(ready)
    published = torch.zeros_like(ready)
    slots = torch.zeros(groups * args.ctas, dtype=torch.int32, device='cuda')
    host_published = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    addresses = [ready.data_ptr(), cancel.data_ptr(), slots.data_ptr(), published.data_ptr()]
    if not args.no_host_publication:
        status, pointer = cuda.cuMemHostGetDevicePointer(host_published.data_ptr(), 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'cuMemHostGetDevicePointer failed: {status}')
        addresses.append(int(pointer))
    pointers = torch.tensor(addresses, dtype=torch.int64, device='cuda')
    arguments = (tuple(from_dlpack(b, assumed_align=16) for b in buffers),
                 from_dlpack(rows), from_dlpack(output, assumed_align=16), from_dlpack(pointers))
    stream = torch.cuda.Stream()
    kernel = RegisterIntraNodeKernel(counts, args.hidden, args.tile_n, args.ctas,
                                    args.elements_per_thread, args.threads, True, args.tile_n)
    kernel.source_tokens = total_rows
    kernel.output_domains = domains
    kernel.input_domains = tuple(input_domains) if args.domain_traversal == 'local' else None
    kernel.group_tiles = args.group_tiles
    kernel.prefetch_peers = args.prefetch_peers
    kernel.poll_once_per_group = args.readiness_polling == 'group'
    kernel.publish_ready = True
    kernel.publish_rdma_ready = not args.no_host_publication
    # Keep wait_for_inputs=True: the production acquire loads still execute.
    lanes = 32
    while (args.tile_n // args.elements_per_thread) % lanes:
        lanes //= 2
    kernel.lanes_per_group = lanes
    compiled = cute.compile(kernel, *arguments, cuda.CUstream(stream.cuda_stream))
    torch.cuda.synchronize()

    def reset():
        slots.zero_()
        published.zero_()

    def launch():
        compiled(*arguments, cuda.CUstream(stream.cuda_stream))

    # Exercise the callable before capture. The graph contains only the kernel;
    # resets, validation, allocations and compilation are outside event timing.
    with torch.cuda.stream(stream):
        reset()
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        reset()
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        launch()

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    for iteration in range(args.warmup + args.iterations):
        host_published.zero_()  # Previous replay has been joined.
        with torch.cuda.stream(stream):
            reset()
            start.record(stream)
            graph.replay()
            end.record(stream)
        end.synchronize()
        if published.item() != groups:
            raise RuntimeError('incomplete GPU output-ready prefix')
        if not args.no_host_publication and host_published.item() != groups:
            raise RuntimeError('incomplete mapped-host output-ready prefix')
        if iteration >= args.warmup:
            samples.append(start.elapsed_time(end))

    # Check every output element outside timing, without a second full-size copy.
    for offset, count in domains:
        actual = output[offset * args.hidden:(offset + count) * args.hidden].view(tiles, count, args.tile_n)
        for tile in range(tiles):
            reference = expected[offset:offset + count, tile * args.tile_n:(tile + 1) * args.tile_n]
            torch.testing.assert_close(actual[tile], reference.to(dtype), rtol=0, atol=0)
    median = statistics.median(samples)
    output_bytes = output.numel() * output.element_size()
    input_bytes = sum(b.numel() * b.element_size() for b in buffers)
    result = dict(device=properties.name, config={k: str(v) if isinstance(v, Path) else v
                                                 for k, v in vars(args).items()},
                  total_rows=total_rows, input_rows_by_domain_peer=counts, column_groups=groups,
                  latency_ms=dict(min=min(samples), median=median, mean=statistics.mean(samples),
                                  p95=sorted(samples)[max(0, (95 * len(samples) + 99) // 100 - 1)],
                                  max=max(samples)),
                  output_bytes=output_bytes, input_payload_bytes=input_bytes,
                  output_GB_per_s=output_bytes / (median * 1e6),
                  effective_payload_GB_per_s=(input_bytes + output_bytes) / (median * 1e6),
                  samples_ms=samples, correctness='PASSED')
    print(f'Correctness: PASSED; {total_rows} rows, {groups} column groups, {args.ctas} CTAs; '
          f'readiness polling={args.readiness_polling}; domain traversal={args.domain_traversal}')
    print(f'Kernel latency: median={median:.4f} ms, min={min(samples):.4f} ms, '
          f'p95={result["latency_ms"]["p95"]:.4f} ms')
    print(f'Output rate: {result["output_GB_per_s"]:.2f} GB/s; '
          f'effective input+output payload: {result["effective_payload_GB_per_s"]:.2f} GB/s')
    print('Repeated-buffer CUDA-graph replay; payload rates exclude metadata and are not measured HBM traffic.')
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Results: {args.json_output}')


if __name__ == '__main__':
    main()
