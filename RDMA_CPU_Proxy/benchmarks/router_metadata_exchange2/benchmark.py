#!/usr/bin/env python3
"""Single-node GPU routing metadata -> NVLink IPC -> pinned C++ proxy benchmark."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import statistics
import time
import uuid

from native import Proxy
from reference import make_routes, metadata, destination_slice, sample_seed, sample_pattern


def summary(values):
    ordered = sorted(values)
    return dict(min=ordered[0], median=statistics.median(ordered), mean=statistics.mean(ordered),
                p95=ordered[int(0.95 * (len(ordered) - 1))], max=ordered[-1])


def print_breakdown(metrics, per_source_readback=False, metadata_barrier=False):
    labels = (
        ("end_to_end_ms", "End to end (wall)"),
        ("metadata_gpu_ms", "Seven metadata kernels (GPU)"),
        ("metadata_completion_wait_ms", "CPU wait for local metadata completion"),
        ("metadata_barrier_wait_ms", "Shared-memory metadata barrier wait"),
        ("exchange_gpu_ms", "GPU-to-GPU exchange (GPU)"),
        ("sender_completion_wait_ms", "Proxy wait for local GPU completion"),
        ("peer_completion_wait_ms", "Proxy wait for peer completions"),
        ("headers_d2h_ms", "Header D2H + completion (wall)"),
        ("payload_d2h_ms", "Payload checks + D2H + completion (wall)"),
        ("install_ms", "CPU validation + install + reset"),
        ("native_ms", "Native submit through installation (wall)"),
        ("host_outside_native_ms", "Host time outside native interval (wall)"),
    )
    print("\nTiming breakdown in microseconds (us).")
    print("Each iteration uses the rank with the longest end-to-end time;")
    print("all rows use that same rank's sample, not separate per-stage maxima.")
    print(f"{'Measurement':45s} {'median':>10s} {'mean':>10s} {'p95':>10s}")
    for key, label in labels:
        if metadata_barrier and not per_source_readback and key == "sender_completion_wait_ms":
            label = "Proxy wait for local exchange completion"
        if per_source_readback:
            label = {
                "sender_completion_wait_ms": "Local completion observation span (wall)",
                "peer_completion_wait_ms": "All-source readiness observation span",
                "headers_d2h_ms": "Sum of source header completion latencies",
                "payload_d2h_ms": "Sum of source payload completion latencies",
            }.get(key, label)
        values = metrics[key]
        print(f"{label:45s} {values['median'] * 1000:10.3f} "
              f"{values['mean'] * 1000:10.3f} {values['p95'] * 1000:10.3f}")
    print("GPU and CPU intervals overlap: do not sum these rows.")
    if per_source_readback:
        print("Per-source mode: readiness spans overlap readback/installation; D2H rows")
        print("sum eight overlapping source latencies, not elapsed readback wall time.")
    print("Host time outside native includes metadata submission and wait-return overhead;")
    print("it is end_to_end - native, not an isolated Python execution measurement.", flush=True)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path,
                        default=Path(__file__).resolve().parent / "build/librouter_metadata_exchange2.so")
    parser.add_argument("--tokens", type=int, default=20480)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--verify", choices=("none", "boundaries", "all"), default="boundaries",
                        help="exact CPU checks outside timing; boundaries checks first and final invocation")
    parser.add_argument("--route-pattern", choices=("affine", "hot-low", "hot-high", "cycle"),
                        default="affine", help="cycle stresses empty lists and changing lengths")
    parser.add_argument("--skip-index-validation", action="store_true",
                        help="omit timed per-index scans; structural header checks always remain")
    parser.add_argument("--metadata-barrier", action="store_true",
                        help="wait for all GPUs to finish metadata before submitting exchange (timed)")
    parser.add_argument("--per-source-readback", action="store_true",
                        help="start each source's readback when ready, overlapping late sources")
    parser.add_argument("--cpu-core-map", default="",
                        help="eight Linux CPU IDs for native proxy thread affinity; empty leaves unbound")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=Path("results/metadata_exchange2.json"))
    args = parser.parse_args()
    if not 1 <= args.tokens <= 10000000 or args.warmup < 1 or args.iterations < 1:
        parser.error("require tokens=1..10000000, warmup>=1, iterations>=1")
    if not 0 <= args.seed < 2**64 or not 0 < args.timeout < 86400:
        parser.error("require uint64 seed and finite timeout between 0 and 86400 seconds")
    try:
        args.cores = [int(x) for x in args.cpu_core_map.split(",")] if args.cpu_core_map else [-1] * 8
    except ValueError:
        parser.error("cpu-core-map must contain eight integer CPU IDs")
    if len(args.cores) != 8 or any(c < -1 or c >= 1024 for c in args.cores):
        parser.error("cpu-core-map must contain eight CPU IDs in -1..1023")
    return args


def verify(args, iteration, rank, plan, proxy):
    import torch
    pattern = sample_pattern(args.route_pattern, iteration)
    for source in range(8):
        routes = make_routes(args.tokens, sample_seed(args.seed, source, iteration), pattern)
        offsets, values, x3, x4 = metadata(routes, source)
        expected = destination_slice(offsets, values, rank)
        if proxy.snapshot(source) != expected:
            raise AssertionError(f"rank={rank} source={source} iteration={iteration}: installed metadata mismatch")
        if source == rank:
            if plan.r.cpu().tolist() != routes:
                raise AssertionError("GPU route generator mismatch")
            actual = (plan.expert_offsets.cpu().tolist(), plan.expert_token_indices.cpu().tolist(),
                      plan.x3.cpu().tolist(), plan.x4.cpu().tolist())
            if actual != (offsets, values, x3, x4):
                raise AssertionError("single-node CuTe metadata mismatch")
            if plan.node_offsets.cpu().tolist() != [0, args.tokens]:
                raise AssertionError("single-node offsets mismatch")
            counts = plan.buffers[2].cpu().to(dtype=torch.int64)
            prefixes = counts.cumsum(dim=0)
            if (not torch.equal(plan.buffers[3].cpu().to(torch.int64), prefixes - counts)
                    or not torch.equal(plan.buffers[13].cpu().to(torch.int64), prefixes[-1])):
                raise AssertionError("CuTe input scan scratch mismatch")


def main():
    args = options()
    import torch
    import torch.distributed as dist
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from metadata_cute_single import MetadataPlan, RouteGenerator

    rank = int(os.environ.get("RANK", "-1"))
    local = int(os.environ.get("LOCAL_RANK", "-1"))
    if (int(os.environ.get("WORLD_SIZE", "0")) != 8
            or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 8 or rank != local):
        raise RuntimeError("launch with torchrun --standalone --nnodes=1 --nproc_per_node=8")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError("requires eight visible peer-accessible CUDA GPUs on one Linux node")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if torch.cuda.get_device_capability(local)[0] < 9:
        raise RuntimeError("this experiment targets Hopper or newer GPUs")
    for peer in range(8):
        if peer != local and not torch.cuda.can_device_access_peer(local, peer):
            raise RuntimeError(f"GPU {local} cannot access GPU {peer}; no host fallback is allowed")
    # CPU-only collectives for setup, iteration boundaries and final reports.
    # No metadata payload travels through torch.distributed.
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout))
    proxy = None
    clean = False
    try:
        configs = [None] * 8
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        dist.all_gather_object(configs, (socket.gethostname(), config))
        if any(c != configs[0] for c in configs):
            raise RuntimeError("all ranks must be on the same host with identical options")
        identity = ["/rme2_" + uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(identity, src=0)
        def create_proxy():
            return Proxy(args.library.resolve(), rank, local, args.tokens, identity[0],
                         args.timeout, not args.skip_index_validation, args.cores[rank], args.per_source_readback, args.metadata_barrier)
        if rank == 0:
            proxy = create_proxy()
        dist.barrier()  # creator finished initializing shared control
        if rank != 0:
            proxy = create_proxy()
        handles = [None] * 8
        dist.all_gather_object(handles, proxy.export())
        proxy.connect(handles)
        dist.barrier()  # all mappings exist before unlink / GPU writes
        if rank == 0:
            proxy.unlink()

        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            r = torch.empty((args.tokens, 8), dtype=torch.int32, device=device)
            plan = MetadataPlan(r, rank)
            r_argument = from_dlpack(r)
            stream_argument = cuda.CUstream(stream.cuda_stream)
            generator = cute.compile(RouteGenerator(), r_argument, cutlass.Uint64(args.seed), stream_argument)
            hot_low = torch.arange(8, dtype=torch.int32, device=device).expand(args.tokens, 8)
            hot_high = torch.arange(120, 128, dtype=torch.int32, device=device).expand(args.tokens, 8)

        def generate(iteration):
            pattern = sample_pattern(args.route_pattern, iteration)
            with torch.cuda.stream(stream):
                if pattern == "affine":
                    generator(r_argument, cutlass.Uint64(sample_seed(args.seed, rank, iteration)), stream_argument)
                else:
                    r.copy_(hot_low if pattern == "hot-low" else hot_high)

        # Prime every kernel before graph capture and before measured work.
        generate(1)
        plan.launch(stream)
        stream.synchronize()
        graph = None
        if args.mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plan.launch(stream)
            stream.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Force lazy event creation outside measurements.
        begin.record(stream); end.record(stream); end.synchronize()
        properties = torch.cuda.get_device_properties(local)
        if rank == 0:
            print(f"Single node: 8 GPUs, 128 experts, top-8, 16 experts/GPU; {properties.name}", flush=True)
            print(f"mode={args.mode}; tokens/source={args.tokens}; verify={args.verify}; "
                  f"per_source_readback={args.per_source_readback}; metadata_barrier={args.metadata_barrier}; native proxy threads active", flush=True)
        records = []
        total = args.warmup + args.iterations
        for iteration in range(1, total + 1):
            generate(iteration)
            stream.synchronize()  # input generation excluded from wall/GPU timings
            dist.barrier()         # align start; previous iteration also has a consumption barrier
            started = time.perf_counter_ns()
            with torch.cuda.stream(stream):
                begin.record(stream)
                if graph is None:
                    plan.launch(stream)
                else:
                    graph.replay()
                end.record(stream)
                proxy.submit(iteration, plan, stream)
            record = proxy.wait()
            record["end_to_end_ms"] = (time.perf_counter_ns() - started) / 1e6
            record["host_outside_native_ms"] = record["end_to_end_ms"] - record["native_ms"]
            record["metadata_gpu_ms"] = begin.elapsed_time(end)
            record["iteration"] = iteration
            if iteration > args.warmup:
                records.append(record)
            if args.verify == "all" or (args.verify == "boundaries" and iteration in (1, total)):
                verify(args, iteration, rank, plan, proxy)
            # No producer may regenerate R/metadata or overwrite an inbox until
            # ALL consumers finished D2H and any out-of-band verification.
            dist.barrier()
        reports = [None] * 8
        dist.all_gather_object(reports, dict(rank=rank, device_name=properties.name,
                                           cpu_core=args.cores[rank], records=records))
        if rank == 0:
            critical_reports = [max(reports, key=lambda report: report["records"][i]["end_to_end_ms"])
                                for i in range(args.iterations)]
            critical_records = [report["records"][i] for i, report in enumerate(critical_reports)]
            critical = [r["end_to_end_ms"] for r in critical_records]
            metrics = {key: summary([r[key] for report in reports for r in report["records"]])
                       for key in records[0] if key.endswith("_ms")}
            critical_metrics = {key: summary([r[key] for r in critical_records]) for key in metrics}
            report = dict(schema_version=4, backend="cpp-thread-cuda-ipc", topology=dict(
                nodes=1, gpus=8, experts=128, experts_per_gpu=16, top_k=8),
                configuration=config, versions=dict(torch=torch.__version__, cuda=torch.version.cuda),
                critical_rank_end_to_end_ms=summary(critical), all_rank_metrics=metrics,
                critical_rank_metrics=critical_metrics,
                critical_rank_per_iteration=[r["rank"] for r in critical_reports],
                readback_timing_semantics=("per-source D2H latency sums; readiness observation spans overlap readback"
                                          if args.per_source_readback else "batched D2H wall stages after all-source wait"),
                timing_note="Overlapping GPU/CPU spans are not additive. Generation, setup, barriers and exact verification excluded.",
                ranks=reports)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(dict(critical_rank_end_to_end_ms=summary(critical),
                                  output=str(args.output), verification=args.verify), indent=2), flush=True)
            print_breakdown(critical_metrics, args.per_source_readback, args.metadata_barrier)
        dist.barrier()
        proxy.disconnect()
        dist.barrier()  # close every IPC import before any owning GPU allocation is freed
        clean = True
    finally:
        if proxy is not None:
            proxy.close()
        # A failed rank must exit promptly; no new collective in error cleanup.
        if clean:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
