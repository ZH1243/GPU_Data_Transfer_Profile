#!/usr/bin/env python3
"""GPU routing metadata -> intra-node IPC -> local D2H / remote HBM-to-DRAM RDMA."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
import uuid

from native import Proxy
from reference import make_routes, metadata, destination_slice, sample_seed, sample_pattern, metadata_two
from topology import rank_topology, validate_hosts, rdma_device
from local_staging import StagingWorkload


def summary(values):
    ordered = sorted(values)
    return dict(min=ordered[0], median=statistics.median(ordered), mean=statistics.mean(ordered),
                p95=ordered[int(0.95 * (len(ordered) - 1))], max=ordered[-1])


def print_breakdown(metrics, per_source_readback=False, metadata_barrier=False, batch_payload_copy=False,
                    direct_x3_x4_output=False):
    labels = (
        ("end_to_end_ms", "End to end (wall)"),
        ("metadata_launch_barrier_ms", "Native metadata launch barrier wait"),
        ("metadata_gpu_ms", "Seven metadata kernels (GPU)"),
        ("metadata_completion_wait_ms", "CPU wait for local metadata completion"),
        ("metadata_barrier_wait_ms", "Shared-memory metadata barrier wait"),
        ("exchange_gpu_ms", "GPU exchange / x3/x4 packing (GPU)"),
        ("local_staging_gpu_ms", "Local staging gather (GPU)"),
        ("sender_completion_wait_ms", "Proxy wait for local GPU completion"),
        ("peer_completion_wait_ms", "Proxy wait for peer completions"),
        ("staging_before_d2h_wait_ms", "Local staging wait before D2H (wall)"),
        ("headers_d2h_ms", "Header D2H + completion (wall)"),
        ("payload_d2h_ms", "Payload checks + D2H + completion (wall)"),
        ("payload_checks_ms", "  Payload: header checks + addresses"),
        ("payload_submit_ms", "  Payload: cudaMemcpyAsync calls"),
        ("payload_event_record_ms", "  Payload: completion event recording"),
        ("payload_completion_ms", "  Payload: completion wait / detection"),
        ("payload_other_ms", "  Payload: bookkeeping / timer overhead"),
        ("install_ms", "CPU validation + install + reset"),
        ("rdma_post_ms", "RDMA record checks + write submissions"),
        ("rdma_poll_ms", "RDMA CQ polling CPU time"),
        ("rdma_post_recv_ms", "RDMA receive credit reposting"),
        ("rdma_work_complete_ms", "RDMA submission through all completions"),
        ("local_staging_wait_ms", "Residual staging wait after metadata exchange"),
        ("native_ms", "Native submit through all work (wall)"),
        ("host_outside_native_ms", "Host time outside native interval (wall)"),
    )
    print("\nTiming breakdown in microseconds (us).")
    print("Each iteration uses the rank with the longest end-to-end time;")
    print("all rows use that same rank's sample, not separate per-stage maxima.")
    print(f"{'Measurement':45s} {'median':>10s} {'mean':>10s} {'p95':>10s}")
    for key, label in labels:
        if direct_x3_x4_output and key == "exchange_gpu_ms":
            label = "GPU exchange / x3/x4 header (GPU)"
        if batch_payload_copy and key == "payload_submit_ms":
            label = "  Payload: cudaMemcpyBatchAsync call"
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
    print("Local staging runs after GPU exchange; --d2h-after-local-staging gates readback on it.")
    print("--local-staging-after-d2h launches gather after local D2H; remaining RDMA may overlap.")
    print("By default staging may overlap D2H/RDMA. Its residual wait")
    print("is measured after metadata installation/send completion, not added GPU execution time.")
    print("Payload subrows partition the payload total per sample (means add; medians/p95 need not).")
    print("Completion wait is residual CPU-observed latency, not pure DMA time; transfers can")
    print("already run during cudaMemcpyAsync submissions. Extra timestamps add measurement overhead.")
    if per_source_readback:
        print("Per-source mode: readiness spans overlap readback/installation; D2H rows")
        print("sum eight overlapping source latencies, not elapsed readback wall time.")
    print("Host time outside native includes caller submission and wait-return overhead;")
    print("it is end_to_end - native, not an isolated Python execution measurement.", flush=True)


def enqueue_iteration(proxy, iteration, plan, stream, graph, begin, end, native_launch):
    if not native_launch:
        begin.record(stream)
        if graph is None:
            plan.launch(stream)
        else:
            graph.replay()
        end.record(stream)
    # In native mode this only queues a command; the C++ worker launches metadata,
    # exchange and optional staging. Route/activation generation is already complete.
    proxy.submit(iteration, plan, stream)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path,
                        default=Path(__file__).resolve().parent / "build/librouter_metadata_exchange2.so")
    parser.add_argument("--rdma-device-map", default=os.environ.get("RDMA_DEVICE_MAP", ""),
                        help="two-node mode: one or eight verbs NIC names, indexed by local GPU")
    parser.add_argument("--rdma-port", type=int, default=1)
    parser.add_argument("--rdma-gid-index", type=int, default=3,
                        help="RoCE GID index; -1 selects InfiniBand LID routing")
    parser.add_argument("--tokens", type=int, default=20480)
    parser.add_argument("--token-dimension", type=int, default=4096,
                        help="BF16 elements per local staging row (default: 4096)")
    parser.add_argument("--skip-local-staging", action="store_true",
                        help="metadata-only baseline; omit activation buffers and local gather")
    parser.add_argument("--d2h-after-local-staging", action="store_true",
                        help="wait for this GPU's gather before header/payload D2H; also delays outgoing RDMA")
    parser.add_argument("--local-staging-after-d2h", action="store_true",
                        help="launch local gather after all local payload D2H completes; may overlap remaining RDMA")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--verify", choices=("none", "boundaries", "all"), default="boundaries",
                        help="exact CPU checks outside timing; boundaries checks first and final invocation")
    parser.add_argument("--route-pattern", choices=("affine", "hot-low", "hot-high", "cycle"),
                        default="affine", help="cycle stresses empty lists and changing lengths")
    parser.add_argument("--skip-index-validation", action="store_true",
                        help="omit timed expert-index and x3/x4 scans; structural header checks always remain")
    parser.add_argument("--direct-x3-x4-output", action="store_true",
                        help="two-node only: metadata kernel 4 also writes remote x3/x4 directly into the RDMA buffer")
    parser.add_argument("--native-metadata-launch", action="store_true",
                        help="native proxy worker launches captured CuTe graph; requires --mode graph")
    parser.add_argument("--metadata-launch-barrier", action="store_true",
                        help="shared-memory barrier in native workers immediately before metadata launch")
    parser.add_argument("--metadata-barrier", action="store_true",
                        help="wait for all GPUs to finish metadata before submitting exchange (timed)")
    parser.add_argument("--batch-payload-copy", action="store_true",
                        help="one cudaMemcpyBatchAsync for nonempty payloads; incompatible with per-source readback")
    parser.add_argument("--per-source-readback", action="store_true",
                        help="start each source's readback when ready, overlapping late sources")
    parser.add_argument("--cpu-core-map", default="",
                        help="eight Linux CPU IDs for native proxy thread affinity; empty leaves unbound")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=Path("results/metadata_exchange2.json"))
    args = parser.parse_args()
    if args.local_staging_after_d2h and args.d2h_after_local_staging:
        parser.error("--local-staging-after-d2h requires --d2h-after-local-staging to be disabled")
    if args.local_staging_after_d2h and args.skip_local_staging:
        parser.error("--local-staging-after-d2h requires local staging; omit --skip-local-staging")
    if args.d2h_after_local_staging and args.skip_local_staging:
        parser.error("--d2h-after-local-staging requires local staging; omit --skip-local-staging")
    if not 1 <= args.rdma_port <= 255 or not -1 <= args.rdma_gid_index <= 255:
        parser.error("invalid RDMA port/GID index")
    if args.batch_payload_copy and args.per_source_readback:
        parser.error("--batch-payload-copy requires --per-source-readback to be disabled")
    if args.native_metadata_launch and args.mode != "graph":
        parser.error("--native-metadata-launch requires --mode graph")
    if args.metadata_launch_barrier and not args.native_metadata_launch:
        parser.error("--metadata-launch-barrier requires --native-metadata-launch")
    if not 1 <= args.tokens <= 10000000 or args.warmup < 1 or args.iterations < 1:
        parser.error("require tokens=1..10000000, warmup>=1, iterations>=1")
    if args.token_dimension < 1:
        parser.error("local staging requires --token-dimension >= 1")
    if not 0 <= args.seed < 2**64 or not 0 < args.timeout < 86400:
        parser.error("require uint64 seed and finite timeout between 0 and 86400 seconds")
    try:
        args.cores = [int(x) for x in args.cpu_core_map.split(",")] if args.cpu_core_map else [-1] * 8
    except ValueError:
        parser.error("cpu-core-map must contain eight integer CPU IDs")
    if len(args.cores) != 8 or any(c < -1 or c >= 1024 for c in args.cores):
        parser.error("cpu-core-map must contain eight CPU IDs in -1..1023")
    return args


def verify(args, iteration, rank, plan, proxy, nodes=1, stats=None, staging=None):
    import torch
    pattern = sample_pattern(args.route_pattern, iteration)
    expected_bytes = dict(received_index_bytes=0, peer_received_index_bytes=0,
                          rdma_sent_bytes=0, rdma_received_bytes=0,
                          x3_x4_sent_bytes=0, x3_x4_received_bytes=0,
                          header_bytes=1088 if nodes == 2 else 576)
    for source in range(8 * nodes):
        routes = make_routes(args.tokens, sample_seed(args.seed, source, iteration, nodes), pattern)
        if nodes == 2:
            offsets, values, x3, x4, node_offsets = metadata_two(routes, source % 8)
        else:
            offsets, values, x3, x4 = metadata(routes, source)
            node_offsets = [0, args.tokens]
        expected = destination_slice(offsets, values, rank, nodes)
        payload_bytes = len(expected[1]) * 4
        expected_bytes["received_index_bytes"] += payload_bytes
        if source != rank:
            expected_bytes["peer_received_index_bytes"] += payload_bytes
        if nodes == 2 and source // 8 != rank // 8:
            expected_bytes["rdma_received_bytes"] += 64 + payload_bytes
        if proxy.snapshot(source) != expected:
            raise AssertionError(f"rank={rank} source={source} iteration={iteration}: installed metadata mismatch")
        if nodes == 2 and source == (rank ^ 8):
            node = rank // 8
            first, last = node_offsets[node:node + 2]
            if proxy.snapshot_x3_x4() != (x3[first:last], x4[first:last]):
                raise AssertionError(f"rank={rank} source={source} iteration={iteration}: received x3/x4 mismatch")
            expected_bytes["x3_x4_received_bytes"] = (last - first) * 5
            expected_bytes["rdma_received_bytes"] += 64 + (last - first) * 5
        if source == rank:
            if staging is not None:
                first, last = node_offsets[rank // 8:rank // 8 + 2]
                staging.verify(x3[first:last])
            if nodes == 2:
                remote = 1 - rank // 8
                route_bytes = (node_offsets[remote + 1] - node_offsets[remote]) * 5
                expected_bytes["x3_x4_sent_bytes"] = route_bytes
                # Expert sends belong to the relay, aggregating all local sources;
                # their expected volume is accumulated below for every local source.
                expected_bytes["rdma_sent_bytes"] += 64 + route_bytes
            if plan.r.cpu().tolist() != routes:
                raise AssertionError("GPU route generator mismatch")
            actual = (plan.expert_offsets.cpu().tolist(), plan.expert_token_indices.cpu().tolist(),
                      plan.x3[:len(x3)].cpu().tolist(), plan.x4[:len(x4)].cpu().tolist())
            if actual != (offsets, values, x3, x4):
                raise AssertionError("CuTe metadata mismatch")
            if plan.node_offsets.cpu().tolist() != node_offsets:
                raise AssertionError("CuTe node offsets mismatch")
            counts = plan.buffers[2].cpu().to(dtype=torch.int64)
            prefixes = counts.cumsum(dim=0)
            if (not torch.equal(plan.buffers[3].cpu().to(torch.int64), prefixes - counts)
                    or not torch.equal(plan.buffers[13].cpu().to(torch.int64), prefixes[-1])):
                raise AssertionError("CuTe input scan scratch mismatch")
        if nodes == 2 and source // 8 == rank // 8:
            remote_experts = destination_slice(offsets, values, rank ^ 8, nodes)
            expected_bytes["rdma_sent_bytes"] += 64 + len(remote_experts[1]) * 4
    if stats is not None:
        for key, expected in expected_bytes.items():
            if stats[key] != expected:
                raise AssertionError(f"rank={rank} iteration={iteration}: {key}={stats[key]}, expected {expected}")


def main():
    args = options()
    world, rank, local, nodes, node = rank_topology(os.environ)
    if nodes == 2 and (args.per_source_readback or args.batch_payload_copy):
        raise ValueError("two-node mode requires default all-senders readback; disable per-source and batch-copy options")
    if args.direct_x3_x4_output and nodes != 2:
        raise ValueError("--direct-x3-x4-output requires two-node mode")
    nic = rdma_device(args.rdma_device_map, local) if nodes == 2 else None
    import torch
    import torch.distributed as dist
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "get_expert_token_idx_list_gpu"))
    if nodes == 2:
        from metadata_cute import MetadataPlan, RouteGenerator
    else:
        from metadata_cute_single import MetadataPlan, RouteGenerator

    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError("requires eight visible peer-accessible CUDA GPUs on each Linux node")
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
        configs = [None] * world
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        dist.all_gather_object(configs, (socket.gethostname(), config))
        validate_hosts(configs, nodes)
        identity = ["/rme2_" + uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(identity, src=0)
        shm_name = identity[0] + "_node" + str(node)
        def create_proxy():
            return Proxy(args.library.resolve(), local, local, args.tokens, shm_name,
                         args.timeout, not args.skip_index_validation, args.cores[local],
                         args.per_source_readback, args.metadata_barrier, args.batch_payload_copy, nodes, node,
                         d2h_after_local_staging=args.d2h_after_local_staging,
                         local_staging_after_d2h=args.local_staging_after_d2h)
        if local == 0:
            proxy = create_proxy()
        dist.barrier()  # creator finished initializing shared control
        if local != 0:
            proxy = create_proxy()
        if nodes == 2:
            descriptor = proxy.rdma_init(nic, args.rdma_port, args.rdma_gid_index)
            descriptors = [None] * world
            dist.all_gather_object(descriptors, descriptor)
            proxy.rdma_connect(descriptors[(1 - node) * 8 + local])
        handles = [None] * world
        dist.all_gather_object(handles, proxy.export())
        proxy.connect(handles[node * 8:node * 8 + 8])
        dist.barrier()  # all mappings exist before unlink / GPU writes
        if local == 0:
            proxy.unlink()

        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            # The two-node plan validates its initial R during construction.
            r = torch.arange(8, dtype=torch.int32, device=device).expand(args.tokens, 8).clone()
            plan_options = {}
            if args.direct_x3_x4_output:
                plan_options = dict(rdma_x3_x4_payload=proxy.enable_direct_x3_x4(),
                                    remote_output_node=1 - node)
            plan = MetadataPlan(r, local, **plan_options)
            if args.direct_x3_x4_output:
                # Borrowed native allocation must outlive every captured launch.
                plan._rdma_output_owner = proxy
            r_argument = from_dlpack(r)
            stream_argument = cuda.CUstream(stream.cuda_stream)
            generator = cute.compile(RouteGenerator(), r_argument, cutlass.Uint64(args.seed), stream_argument)
            hot_low = torch.arange(8, dtype=torch.int32, device=device).expand(args.tokens, 8)
            hot_high = torch.arange(120, 128, dtype=torch.int32, device=device).expand(args.tokens, 8)
            staging = (None if args.skip_local_staging else
                       StagingWorkload(plan, node, rank, args.token_dimension))

        def generate(iteration, check_staging=False):
            pattern = sample_pattern(args.route_pattern, iteration)
            with torch.cuda.stream(stream):
                if pattern == "affine":
                    generator(r_argument, cutlass.Uint64(sample_seed(args.seed, rank, iteration, nodes)), stream_argument)
                else:
                    r.copy_(hot_low if pattern == "hot-low" else hot_high)
                if staging is not None:
                    staging.prepare(iteration, check_staging)

        # Prime every kernel before graph capture and before measured work.
        generate(1, check_staging=True)
        plan.launch(stream)
        if staging is not None:
            staging.plan.launch(stream)
        stream.synchronize()
        graph = None
        if args.native_metadata_launch:
            proxy.capture_metadata(plan, stream, args.metadata_launch_barrier)
        elif args.mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plan.launch(stream)
            stream.synchronize()
        if staging is not None:
            # Always a separate single-kernel graph, including eager metadata
            # mode, so native can enqueue it immediately after exchange readiness.
            proxy.capture_staging(staging.plan, stream)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Force lazy event creation outside measurements.
        begin.record(stream); end.record(stream); end.synchronize()
        properties = torch.cuda.get_device_properties(local)
        if rank == 0:
            print(f"{nodes} node(s), {world} GPUs, 128 experts, top-8, {128 // world} experts/GPU; {properties.name}", flush=True)
            print(f"mode={args.mode}; tokens/source={args.tokens}; verify={args.verify}; "
                  f"native_metadata_launch={args.native_metadata_launch}; metadata_launch_barrier={args.metadata_launch_barrier}; "
                  f"batch_payload_copy={args.batch_payload_copy}; per_source_readback={args.per_source_readback}; metadata_barrier={args.metadata_barrier}; native proxy threads active", flush=True)
            if nodes == 2:
                print("x3/x4 exchange enabled: same-index remote slice; nine RDMA records per proxy/iteration; "
                      f"direct_x3_x4_output={args.direct_x3_x4_output}", flush=True)
            print(f"local_staging={staging is not None}; d2h_after_local_staging={args.d2h_after_local_staging}; "
                  f"local_staging_after_d2h={args.local_staging_after_d2h}; token_dimension={args.token_dimension}; "
                  f"dtype=BF16; gather={staging.plan.implementation if staging else 'disabled'}", flush=True)
        records = []
        total = args.warmup + args.iterations
        for iteration in range(1, total + 1):
            verify_iteration = args.verify == "all" or (args.verify == "boundaries" and iteration in (1, total))
            generate(iteration, check_staging=verify_iteration)
            stream.synchronize()  # route/activation generation and poisoning excluded
            dist.barrier()         # align start; previous iteration also has a consumption barrier
            started = time.perf_counter_ns()
            with torch.cuda.stream(stream):
                enqueue_iteration(proxy, iteration, plan, stream, graph, begin, end,
                                  args.native_metadata_launch)
            record = proxy.wait()
            record["end_to_end_ms"] = (time.perf_counter_ns() - started) / 1e6
            record["host_outside_native_ms"] = record["end_to_end_ms"] - record["native_ms"]
            if not args.native_metadata_launch:
                record["metadata_gpu_ms"] = begin.elapsed_time(end)
            record["iteration"] = iteration
            if iteration > args.warmup:
                records.append(record)
            if verify_iteration:
                verify(args, iteration, rank, plan, proxy, nodes, record, staging)
            # No producer may regenerate R/metadata or overwrite an inbox until
            # ALL consumers finished D2H and any out-of-band verification.
            dist.barrier()
        reports = [None] * world
        dist.all_gather_object(reports, dict(rank=rank, device_name=properties.name,
                                           node=node, local_rank=local, rdma_device=nic, cpu_core=args.cores[local], records=records))
        if rank == 0:
            critical_reports = [max(reports, key=lambda report: report["records"][i]["end_to_end_ms"])
                                for i in range(args.iterations)]
            critical_records = [report["records"][i] for i, report in enumerate(critical_reports)]
            critical = [r["end_to_end_ms"] for r in critical_records]
            metrics = {key: summary([r[key] for report in reports for r in report["records"]])
                       for key in records[0] if key.endswith("_ms")}
            critical_metrics = {key: summary([r[key] for r in critical_records]) for key in metrics}
            report = dict(schema_version=13, backend=("cpp-thread-cuda-ipc-gpudirect-rdma" if nodes == 2 else "cpp-thread-cuda-ipc"), topology=dict(
                nodes=nodes, gpus=world, gpus_per_node=8, experts=128, experts_per_gpu=128 // world, top_k=8),
                completion_endpoint=("16 source expert tables and remote same-index x3/x4 installed; nine outgoing RDMA sends drained"
                                     if nodes == 2 else "8 source expert tables installed")
                                    + ("; local token staging complete" if staging is not None else ""),
                x3_x4_exchange=nodes == 2,
                x3_x4_payload_producer=("metadata kernel 4 (direct output)" if args.direct_x3_x4_output
                                       else "GPU exchange packing" if nodes == 2 else None),
                local_staging=dict(enabled=staging is not None, dtype="bfloat16", token_dimension=args.token_dimension,
                                   d2h_after_local_staging=args.d2h_after_local_staging,
                                   local_staging_after_d2h=args.local_staging_after_d2h,
                                   implementation=staging.plan.implementation if staging else None,
                                   launch=("separate native CUDA graph after local D2H" if args.local_staging_after_d2h
                                           else "separate native CUDA graph after exchange") if staging else None),
                configuration=config, versions=dict(torch=torch.__version__, cuda=torch.version.cuda),
                critical_rank_end_to_end_ms=summary(critical), all_rank_metrics=metrics,
                critical_rank_metrics=critical_metrics,
                critical_rank_per_iteration=[r["rank"] for r in critical_reports],
                readback_timing_semantics=("per-source D2H latency sums; readiness observation spans overlap readback"
                                          if args.per_source_readback else ("local D2H overlaps RDMA; payload completion includes CPU progress delays"
                                          if nodes == 2 else "batched D2H wall stages after all-source wait")),
                timing_note="Overlapping GPU/CPU spans are not additive. Local staging is included in native/end-to-end completion. By default it may overlap D2H/RDMA; d2h_after_local_staging runs gather first; local_staging_after_d2h runs gather after local D2H and may overlap remaining RDMA. The pre-D2H staging wait is separate from D2H timers. Route/activation generation, setup, outer Gloo barriers and exact verification excluded; enabled native barriers included.",
                ranks=reports)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(dict(critical_rank_end_to_end_ms=summary(critical),
                                  output=str(args.output), verification=args.verify), indent=2), flush=True)
            print_breakdown(critical_metrics, args.per_source_readback, args.metadata_barrier, args.batch_payload_copy,
                            args.direct_x3_x4_output)
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
