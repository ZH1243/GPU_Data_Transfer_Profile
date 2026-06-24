#!/usr/bin/env python3
"""
Benchmark all-to-all GPU P2P traffic using batched CE requests.

Each rank owns one source GPU. In every iteration, GPU i submits outgoing copy
requests to peer GPUs. By default every peer destination has a dedicated
source-device CUDA stream and, when enabled by --destination-copy-specs, one
cudaMemcpyBatchAsync call per iteration. Use --single-stream to submit all
destination batches on one source-device CUDA stream instead.

This is intended for profiling two overlap questions:
  1. Can the host-side metadata handoff for src/dst address lists overlap across
     cudaMemcpyBatchAsync requests submitted to different streams?
  2. Can the resulting GPU Memcpy PtoP transfers overlap across streams?

Example on one 8-GPU node:
  torchrun --standalone --nproc_per_node=8 \
    nvlink_all_to_all_multistream_batch_copy_engine_test.py \
    --destination-copy-specs "1*1024K,2*512K,0,2*512K,2*512K,2*512K,32*8K" \
    --iters 100 --warmup 10 --check

The example has seven destination entries, matching the seven peers in an
8-GPU run. Entry "32*8K" means one batch call to that peer containing 32
independent 8 KiB copies. Entry "0" disables traffic to that peer.
"""

import argparse
import threading
import time
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist

from nvlink_copy_engine_test import (
    MemcpyBatch,
    _get_cuda_memcpy_batch_async,
    all_reduce_max_float,
    all_reduce_sum_float,
    barrier,
    build_contiguous_offsets,
    cuda_can_access_peer,
    cuda_enable_peer_access,
    cuda_memcpy_batch_async,
    cuda_set_device,
    cuda_stream_synchronize,
    fmt_bytes,
    get_rank_info,
    parse_nbytes,
)


DEFAULT_DESTINATION_COPY_SPECS = (
    "1*1024K,2*512K,0,2*512K,2*512K,2*512K,32*8K"
)


def destination_order(src_device: int,
                      world_size: int,
                      rotate_by_source: bool) -> Tuple[int, ...]:
    if not rotate_by_source:
        return tuple(device for device in range(world_size) if device != src_device)
    return tuple((src_device + offset) % world_size for offset in range(1, world_size))


def parse_gpu_ids(spec: str) -> Tuple[int, ...]:
    values = [part.strip() for part in spec.split(",")]
    if any(not value for value in values):
        raise argparse.ArgumentTypeError(
            "GPU IDs must be a comma-separated list like 0 or 0,2,4"
        )
    try:
        gpu_ids = tuple(int(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU IDs must be integers in a comma-separated list like 0 or 0,2,4"
        ) from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative")
    return tuple(dict.fromkeys(gpu_ids))


def parse_destination_copy_specs(spec: str) -> Tuple[Tuple[int, ...], ...]:
    """
    Parse per-peer copy specs.

    Grammar per comma-separated destination:
      0            no traffic to this destination
      SIZE         one source buffer of SIZE
      COUNT*SIZE   COUNT source buffers, each SIZE bytes
    """
    entries = [part.strip() for part in spec.split(",")]
    if any(not entry for entry in entries):
        raise argparse.ArgumentTypeError(
            "--destination-copy-specs must be comma-separated entries like "
            "1*1024K,2*512K,0,32*8K"
        )

    parsed: List[Tuple[int, ...]] = []
    for entry in entries:
        compact = entry.replace(" ", "")
        if compact == "0":
            parsed.append(())
            continue

        if "*" in compact:
            pieces = compact.split("*")
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                raise argparse.ArgumentTypeError(
                    f"invalid destination copy spec {entry!r}; expected COUNT*SIZE"
                )
            try:
                count = int(pieces[0])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid copy count in destination copy spec {entry!r}"
                ) from exc
            size_text = pieces[1]
        else:
            count = 1
            size_text = compact

        if count <= 0:
            raise argparse.ArgumentTypeError(
                f"copy count must be positive in destination copy spec {entry!r}"
            )
        try:
            nbytes = parse_nbytes(size_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid copy size in destination copy spec {entry!r}: {exc}"
            ) from exc
        if nbytes <= 0:
            raise argparse.ArgumentTypeError(
                f"copy size must be positive in destination copy spec {entry!r}"
            )
        parsed.append(tuple(nbytes for _ in range(count)))

    return tuple(parsed)


def build_gapped_offsets(copy_sizes: Sequence[int],
                         gap_size: int) -> Tuple[int, ...]:
    offsets: List[int] = []
    offset = 0
    for index, copy_size in enumerate(copy_sizes):
        offsets.append(offset)
        offset += copy_size
        if index + 1 < len(copy_sizes):
            offset += gap_size
    return tuple(offsets)


def allocation_size(offsets: Sequence[int], copy_sizes: Sequence[int]) -> int:
    if not offsets:
        return 0
    return offsets[-1] + copy_sizes[-1]


def chunk_value(rank: int, destination_index: int, copy_index: int) -> int:
    return (rank * 37 + destination_index * 19 + copy_index * 7 + 11) % 251


def enqueue_iteration(dst_devices: Sequence[int],
                      batches: Dict[int, MemcpyBatch],
                      streams: Dict[int, torch.cuda.Stream]) -> None:
    for dst_device in dst_devices:
        batch = batches.get(dst_device)
        if batch is None:
            continue
        cuda_memcpy_batch_async(batch, streams[dst_device])


def enqueue_iterations_concurrent(num_iterations: int,
                                  src_device: int,
                                  enabled_dst_devices: Sequence[int],
                                  batches: Dict[int, MemcpyBatch],
                                  streams: Dict[int, torch.cuda.Stream]) -> None:
    if num_iterations <= 0:
        return
    if not enabled_dst_devices:
        return

    start_barrier = threading.Barrier(len(enabled_dst_devices) + 1)
    done_barrier = threading.Barrier(len(enabled_dst_devices) + 1)
    worker_errors: List[BaseException] = []
    error_lock = threading.Lock()

    def first_worker_error() -> BaseException | None:
        with error_lock:
            for exc in worker_errors:
                if not isinstance(exc, threading.BrokenBarrierError):
                    return exc
            return worker_errors[0] if worker_errors else None

    def worker(dst_device: int) -> None:
        try:
            cuda_set_device(src_device)
            batch = batches[dst_device]
            stream = streams[dst_device]
            for _ in range(num_iterations):
                start_barrier.wait()
                cuda_memcpy_batch_async(batch, stream)
                done_barrier.wait()
        except BaseException as exc:
            with error_lock:
                worker_errors.append(exc)
            start_barrier.abort()
            done_barrier.abort()

    workers = [
        threading.Thread(
            target=worker,
            args=(dst_device,),
            name=f"batch-submit-cuda-{dst_device}",
        )
        for dst_device in enabled_dst_devices
    ]
    for worker_thread in workers:
        worker_thread.start()

    try:
        for _ in range(num_iterations):
            start_barrier.wait()
            done_barrier.wait()
    except threading.BrokenBarrierError as exc:
        original_error = first_worker_error()
        if original_error is not None:
            raise RuntimeError(
                "concurrent cudaMemcpyBatchAsync submission failed"
            ) from original_error
        raise RuntimeError(
            "concurrent cudaMemcpyBatchAsync submission barrier broke"
        ) from exc
    except BaseException:
        start_barrier.abort()
        done_barrier.abort()
        raise
    finally:
        for worker_thread in workers:
            worker_thread.join()

    original_error = first_worker_error()
    if original_error is not None:
        raise RuntimeError(
            "concurrent cudaMemcpyBatchAsync submission failed"
        ) from original_error


def synchronize_streams(streams: Dict[int, torch.cuda.Stream]) -> None:
    seen_streams = set()
    for stream in streams.values():
        if stream.cuda_stream in seen_streams:
            continue
        seen_streams.add(stream.cuda_stream)
        cuda_stream_synchronize(stream)


def count_unique_streams(streams: Dict[int, torch.cuda.Stream]) -> int:
    return len({stream.cuda_stream for stream in streams.values()})


def check_destinations(destinations: Dict[int, torch.Tensor],
                       dst_devices: Sequence[int],
                       dst_offset_groups: Dict[int, Tuple[int, ...]],
                       copy_size_groups: Dict[int, Tuple[int, ...]],
                       rank: int) -> Tuple[bool, str]:
    for destination_index, dst_device in enumerate(dst_devices):
        dst = destinations.get(dst_device)
        if dst is None:
            continue

        sample_indices: List[int] = []
        expected_values: List[int] = []
        for copy_index, (offset, nbytes) in enumerate(
            zip(dst_offset_groups[dst_device], copy_size_groups[dst_device])
        ):
            for within_chunk in sorted({0, nbytes // 2, nbytes - 1}):
                sample_indices.append(offset + within_chunk)
                expected_values.append(chunk_value(rank, destination_index, copy_index))

        torch.cuda.set_device(dst_device)
        index_tensor = torch.tensor(sample_indices, dtype=torch.int64, device=dst.device)
        expected_tensor = torch.tensor(expected_values, dtype=torch.uint8)
        actual = dst[index_tensor].cpu()
        mismatches = torch.nonzero(actual != expected_tensor, as_tuple=False).flatten()
        if mismatches.numel() == 0:
            continue

        first = int(mismatches[0].item())
        return (
            False,
            f"dst=cuda:{dst_device} sample_offset={sample_indices[first]} "
            f"expected={expected_values[first]} actual={int(actual[first].item())}",
        )

    return True, ""


def format_destination_plan(dst_devices: Sequence[int],
                            copy_size_groups: Dict[int, Tuple[int, ...]]) -> str:
    parts = []
    for dst_device in dst_devices:
        sizes = copy_size_groups[dst_device]
        if not sizes:
            parts.append(f"cuda:{dst_device}=0")
            continue
        unique_sizes = set(sizes)
        if len(unique_sizes) == 1:
            parts.append(f"cuda:{dst_device}={len(sizes)}*{fmt_bytes(sizes[0])}")
        else:
            parts.append(
                f"cuda:{dst_device}="
                + "+".join(fmt_bytes(nbytes) for nbytes in sizes)
            )
    return ",".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--destination-copy-specs", "--destination_copy_specs",
                   type=parse_destination_copy_specs,
                   default=parse_destination_copy_specs(DEFAULT_DESTINATION_COPY_SPECS),
                   metavar="SPECS",
                   help="comma-separated per-destination specs. Each entry is 0, SIZE, "
                        "or COUNT*SIZE. Example/default: "
                        f"{DEFAULT_DESTINATION_COPY_SPECS!r}")
    p.add_argument("--source-buffer-gap-size", "--source_buffer_gap_size",
                   type=parse_nbytes, default=parse_nbytes("64K"),
                   help="gap inserted between source buffers inside each destination "
                        "batch so source addresses are discontinuous. Default: 64K")
    p.add_argument("--iters", type=int, default=100, help="timed iterations")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("--check", action="store_true",
                   help="verify copied bytes on every enabled destination GPU")
    p.add_argument("--sleep-before", type=float, default=0.0,
                   help="seconds to sleep before benchmark, useful for attaching profilers")
    p.add_argument("--concurrent-host-submission", action="store_true",
                   help="submit one cudaMemcpyBatchAsync per enabled destination from "
                        "separate host threads released by a per-iteration barrier. "
                        "Default: submit the batch calls sequentially from the main thread.")
    p.add_argument("--single-stream", action="store_true",
                   help="submit all enabled destination batches on one source-device "
                        "CUDA stream. Default: use one stream per enabled destination.")
    p.add_argument("--rotate-destination-order", action="store_true",
                   help="map spec entries to (src+1)%%world_size, ..., "
                        "(src-1)%%world_size. By default entries map to ascending "
                        "GPU IDs with the source omitted.")
    p.add_argument("--active-source-gpus", nargs="?", const="0", default=None,
                   type=parse_gpu_ids, metavar="IDS",
                   help="restrict copying to comma-separated source GPU IDs. If enabled "
                        "without IDS, only GPU 0 copies. Default: all GPUs copy.")
    args = p.parse_args()

    rank, world_size, local_rank = get_rank_info()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch build.")

    num_gpus = torch.cuda.device_count()
    if world_size > num_gpus:
        raise RuntimeError(f"world_size={world_size}, but only {num_gpus} CUDA devices visible.")
    if world_size < 2:
        raise RuntimeError("This benchmark requires at least two ranks/GPUs.")
    if len(args.destination_copy_specs) != world_size - 1:
        raise ValueError(
            "--destination-copy-specs must specify exactly world_size - 1 entries "
            f"({world_size - 1} for this run), got {len(args.destination_copy_specs)}"
        )
    if args.source_buffer_gap_size < 0:
        raise ValueError("--source-buffer-gap-size must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    src_device = local_rank
    torch.cuda.set_device(src_device)
    if args.active_source_gpus is None:
        active_source_gpus = tuple(range(world_size))
    else:
        active_source_gpus = args.active_source_gpus
        invalid_gpus = [
            gpu_id for gpu_id in active_source_gpus
            if gpu_id >= world_size
        ]
        if invalid_gpus:
            raise ValueError(
                f"--active-source-gpus contains GPU ID(s) outside this run's "
                f"0..{world_size - 1} range: {invalid_gpus}"
            )

    source_is_active = src_device in set(active_source_gpus)
    if source_is_active:
        _get_cuda_memcpy_batch_async()

    dst_devices = (
        destination_order(src_device, world_size, args.rotate_destination_order)
        if source_is_active
        else ()
    )
    copy_size_groups: Dict[int, Tuple[int, ...]] = {
        dst_device: args.destination_copy_specs[destination_index]
        for destination_index, dst_device in enumerate(dst_devices)
    }
    enabled_dst_devices = tuple(
        dst_device for dst_device in dst_devices if copy_size_groups[dst_device]
    )
    if source_is_active and not enabled_dst_devices:
        raise ValueError("--destination-copy-specs disables every destination")

    batches_per_iter = len(enabled_dst_devices)
    copies_per_iter = sum(len(copy_size_groups[dst]) for dst in enabled_dst_devices)
    bytes_per_destination = {
        dst_device: sum(copy_size_groups[dst_device])
        for dst_device in enabled_dst_devices
    }
    bytes_per_iter = sum(bytes_per_destination.values())

    source_backings: Dict[int, torch.Tensor] = {}
    destinations: Dict[int, torch.Tensor] = {}
    src_offset_groups: Dict[int, Tuple[int, ...]] = {}
    dst_offset_groups: Dict[int, Tuple[int, ...]] = {}
    batches: Dict[int, MemcpyBatch] = {}
    streams: Dict[int, torch.cuda.Stream] = {}
    total_source_alloc = 0
    total_destination_alloc = 0

    if source_is_active:
        unsupported_peers = [
            device for device in enabled_dst_devices
            if not cuda_can_access_peer(src_device, device)
        ]
        if unsupported_peers:
            raise RuntimeError(
                f"Source GPU {src_device} cannot access destination peer GPU(s) "
                f"{unsupported_peers}. Check CUDA_VISIBLE_DEVICES and nvidia-smi topo -m."
            )

        for dst_device in enabled_dst_devices:
            cuda_enable_peer_access(src_device, dst_device)

        for destination_index, dst_device in enumerate(dst_devices):
            copy_sizes = copy_size_groups[dst_device]
            if not copy_sizes:
                continue

            src_offsets = build_gapped_offsets(copy_sizes, args.source_buffer_gap_size)
            dst_offsets = build_contiguous_offsets(copy_sizes)
            source_allocation_size = allocation_size(src_offsets, copy_sizes)
            destination_allocation_size = allocation_size(dst_offsets, copy_sizes)
            total_source_alloc += source_allocation_size
            total_destination_alloc += destination_allocation_size

            torch.cuda.set_device(src_device)
            source = torch.empty(
                source_allocation_size,
                dtype=torch.uint8,
                device=f"cuda:{src_device}",
            )
            source.fill_((rank + 211) % 251)
            for copy_index, (offset, nbytes) in enumerate(zip(src_offsets, copy_sizes)):
                source[offset:offset + nbytes].fill_(
                    chunk_value(rank, destination_index, copy_index)
                )
            source_backings[dst_device] = source
            src_offset_groups[dst_device] = src_offsets

            torch.cuda.set_device(dst_device)
            destination = torch.empty(
                destination_allocation_size,
                dtype=torch.uint8,
                device=f"cuda:{dst_device}",
            )
            destination.fill_((rank + 113) % 251)
            destinations[dst_device] = destination
            dst_offset_groups[dst_device] = dst_offsets

        torch.cuda.synchronize(src_device)
        for dst_device in enabled_dst_devices:
            torch.cuda.synchronize(dst_device)

        for dst_device in enabled_dst_devices:
            source = source_backings[dst_device]
            destination = destinations[dst_device]
            src_ptrs = tuple(
                source.data_ptr() + offset for offset in src_offset_groups[dst_device]
            )
            dst_ptrs = tuple(
                destination.data_ptr() + offset for offset in dst_offset_groups[dst_device]
            )
            batches[dst_device] = MemcpyBatch(
                dst_ptrs, src_ptrs, copy_size_groups[dst_device]
            )

        torch.cuda.set_device(src_device)
        if args.single_stream:
            shared_stream = torch.cuda.Stream(device=src_device)
            streams = {
                dst_device: shared_stream
                for dst_device in enabled_dst_devices
            }
        else:
            streams = {
                dst_device: torch.cuda.Stream(device=src_device)
                for dst_device in enabled_dst_devices
            }

    barrier()
    if args.sleep_before > 0:
        if rank == 0:
            print(f"Sleeping for {args.sleep_before:.1f}s before benchmark...", flush=True)
        time.sleep(args.sleep_before)
    barrier()

    cuda_set_device(src_device)
    if source_is_active:
        if args.concurrent_host_submission:
            enqueue_iterations_concurrent(
                args.warmup, src_device, enabled_dst_devices, batches, streams
            )
        else:
            for _ in range(args.warmup):
                enqueue_iteration(dst_devices, batches, streams)
        synchronize_streams(streams)
    barrier()

    cuda_set_device(src_device)
    if source_is_active:
        t0 = time.perf_counter()
        if args.concurrent_host_submission:
            enqueue_iterations_concurrent(
                args.iters, src_device, enabled_dst_devices, batches, streams
            )
        else:
            for _ in range(args.iters):
                enqueue_iteration(dst_devices, batches, streams)
        synchronize_streams(streams)
        t1 = time.perf_counter()
        elapsed = t1 - t0
    else:
        elapsed = 0.0
    barrier()

    local_gib = (bytes_per_iter * args.iters) / (1024**3)
    local_egress_bw = local_gib / elapsed if elapsed > 0 else 0.0

    ref_device = torch.device(f"cuda:{local_rank}")
    max_elapsed = all_reduce_max_float(elapsed, ref_device)
    sum_gib = all_reduce_sum_float(local_gib, ref_device)
    aggregate_bw = sum_gib / max_elapsed if max_elapsed > 0 else 0.0

    ok = True
    failure_detail = ""
    if args.check and source_is_active:
        ok, failure_detail = check_destinations(
            destinations, dst_devices, dst_offset_groups, copy_size_groups, rank
        )

    torch.cuda.set_device(src_device)
    for output_rank in range(world_size):
        barrier()
        if rank == output_rank:
            plan_text = (
                format_destination_plan(dst_devices, copy_size_groups)
                if source_is_active
                else "inactive"
            )
            print(
                f"rank={rank:02d} source=cuda:{src_device} "
                f"active={source_is_active} active_sources={list(active_source_gpus)} "
                f"destination_plan={plan_text} streams={count_unique_streams(streams)} "
                f"stream_mode={'single' if args.single_stream else 'per-destination'} "
                f"host_submission="
                f"{'concurrent' if args.concurrent_host_submission else 'sequential'} "
                f"batches/iter={batches_per_iter} copies/iter={copies_per_iter} "
                f"bytes/iter={fmt_bytes(bytes_per_iter)} "
                f"source_alloc={fmt_bytes(total_source_alloc)} "
                f"destination_alloc={fmt_bytes(total_destination_alloc)} "
                f"source_gap={fmt_bytes(args.source_buffer_gap_size)} "
                f"iters={args.iters} elapsed={elapsed:.6f}s "
                f"egress_bw={local_egress_bw:.2f} GiB/s "
                f"check={'OK' if ok else f'FAIL:{failure_detail}'}",
                flush=True,
            )
        barrier()

    if rank == 0:
        aggregate_copies_per_iter = sum(
            len(spec) for spec in args.destination_copy_specs
        ) * len(active_source_gpus)
        aggregate_batches_per_iter = sum(
            1 for spec in args.destination_copy_specs if spec
        ) * len(active_source_gpus)
        print(
            f"\nAggregate multistream all-to-all traffic over {world_size} ranks "
            f"with {len(active_source_gpus)} active source rank(s): "
            f"batches/iteration={aggregate_batches_per_iter}, "
            f"copies/iteration={aggregate_copies_per_iter}, "
            f"moved={sum_gib:.2f} GiB, slowest_elapsed={max_elapsed:.6f}s, "
            f"aggregate_bw={aggregate_bw:.2f} GiB/s",
            flush=True,
        )
        print(
            "\nProfiler expectation: each active rank submits one "
            "cudaMemcpyBatchAsync per enabled destination per iteration. "
            f"Host submission is "
            f"{'barrier-released from one thread per enabled destination' if args.concurrent_host_submission else 'sequential from the main thread'}, "
            f"and destination batches use "
            f"{'one shared source-device CUDA stream' if args.single_stream else 'one dedicated source-device CUDA stream per enabled destination'}.",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
