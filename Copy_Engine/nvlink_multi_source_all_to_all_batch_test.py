#!/usr/bin/env python3
"""
Benchmark all-to-all NVLink/P2P copies with multiple source buffers per GPU.

Each rank owns one source GPU. On that GPU it creates N logical source buffers,
named A, B, C, ... for display. The logical buffers are separated by an explicit
gap, while every source buffer is split into world_size - 1 contiguous
sub-buffers, one sub-buffer for each peer GPU:

  A = [A1, A2, ..., A(world_size-1)]
  B = [B1, B2, ..., B(world_size-1)]
  ...

For source GPU i, sub-buffer index k is sent to GPU (i + k + 1) % world_size.
In each iteration the script submits one destination-group at a time:

  batch 0: A1, B1, C1, ... -> GPU (i + 1)
  batch 1: A2, B2, C2, ... -> GPU (i + 2)
  ...

With --copy-mode batch, each destination group is one cudaMemcpyBatchAsync, so
each rank submits world_size - 1 batched calls per iteration. With
--copy-mode separate, the same grouped order is used, but each entry is a
cudaMemcpyPeerAsync.

Example on one 8-GPU node:
  torchrun --standalone --nproc_per_node=8 nvlink_multi_source_all_to_all_batch_test.py \
    --num_source_buffers 4 \
    --source-buffer-sizes 64K,128K,256K,512K \
    --copy-mode batch \
    --iters 100 \
    --warmup 10 \
    --check
"""

import argparse
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
    cuda_memcpy_peer_async,
    cuda_set_device,
    cuda_stream_synchronize,
    fmt_bytes,
    get_rank_info,
    parse_copy_sizes,
    parse_nbytes,
)


def source_buffer_name(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"S{index}"


def destination_order(src_device: int, world_size: int) -> Tuple[int, ...]:
    return tuple((src_device + offset) % world_size for offset in range(1, world_size))


def chunk_value(rank: int, source_index: int, destination_index: int) -> int:
    return (rank * 37 + source_index * 19 + destination_index * 7) % 251


def enqueue_iteration(copy_mode: str,
                      src_device: int,
                      dst_devices: Sequence[int],
                      src_ptr_groups: Dict[int, Tuple[int, ...]],
                      dst_ptr_groups: Dict[int, Tuple[int, ...]],
                      sub_buffer_sizes: Sequence[int],
                      batches: Dict[int, MemcpyBatch],
                      stream: torch.cuda.Stream) -> None:
    for dst_device in dst_devices:
        if copy_mode == "batch":
            cuda_memcpy_batch_async(batches[dst_device], stream)
            continue

        for src_ptr, dst_ptr, nbytes in zip(
            src_ptr_groups[dst_device],
            dst_ptr_groups[dst_device],
            sub_buffer_sizes,
        ):
            cuda_memcpy_peer_async(
                dst_ptr, dst_device, src_ptr, src_device, nbytes, stream
            )


def check_destinations(destinations: Dict[int, torch.Tensor],
                       dst_offsets: Sequence[int],
                       dst_devices: Sequence[int],
                       sub_buffer_sizes: Sequence[int],
                       rank: int) -> Tuple[bool, str]:
    for destination_index, dst_device in enumerate(dst_devices):
        dst = destinations[dst_device]
        sample_indices: List[int] = []
        expected_values: List[int] = []

        for source_index, (offset, nbytes) in enumerate(
            zip(dst_offsets, sub_buffer_sizes)
        ):
            for within_chunk in sorted({0, nbytes // 2, nbytes - 1}):
                sample_indices.append(offset + within_chunk)
                expected_values.append(chunk_value(rank, source_index, destination_index))

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


def format_source_sizes(sub_buffer_sizes: Sequence[int]) -> str:
    parts = [
        f"{source_buffer_name(index)}i={fmt_bytes(nbytes)}"
        for index, nbytes in enumerate(sub_buffer_sizes)
    ]
    return ",".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-source-buffers", "--num_source_buffers",
                   dest="num_source_buffers", type=int, default=4,
                   help="number of independent source buffers per GPU. Default: 4")
    p.add_argument("--nbytes", "--copy-size", dest="nbytes",
                   type=parse_nbytes, default=parse_nbytes("1M"),
                   help="bytes per source sub-buffer when --source-buffer-sizes "
                        "is not set. Default: 1M")
    p.add_argument("--source-buffer-sizes", "--source_buffer_sizes",
                   type=parse_copy_sizes,
                   help="comma-separated per-source sub-buffer sizes, for example "
                        "64K,128K,256K,512K for Ai,Bi,Ci,Di")
    p.add_argument("--source-buffer-gap-size", "--source_buffer_gap_size",
                   type=parse_nbytes, default=parse_nbytes("64K"),
                   help="gap between logical source buffers A/B/C/... to guarantee "
                        "they are not contiguous. Default: 64K")
    p.add_argument("--copy-mode", choices=["separate", "batch"], default="batch",
                   help="separate: one cudaMemcpyPeerAsync per source buffer per "
                        "destination; batch: one cudaMemcpyBatchAsync per destination. "
                        "Default: batch")
    p.add_argument("--iters", type=int, default=100, help="timed iterations")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("--check", action="store_true",
                   help="verify copied bytes on every destination GPU")
    p.add_argument("--sleep-before", type=float, default=0.0,
                   help="seconds to sleep before benchmark, useful for attaching profilers")
    args = p.parse_args()

    rank, world_size, local_rank = get_rank_info()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch build.")

    num_gpus = torch.cuda.device_count()
    if world_size > num_gpus:
        raise RuntimeError(f"world_size={world_size}, but only {num_gpus} CUDA devices visible.")
    if world_size < 2:
        raise RuntimeError("This benchmark requires at least two ranks/GPUs.")
    if args.num_source_buffers <= 0:
        raise ValueError("--num-source-buffers must be positive")
    if args.nbytes <= 0:
        raise ValueError("--nbytes must be positive")
    if args.source_buffer_gap_size < 0:
        raise ValueError("--source-buffer-gap-size must be non-negative")
    if args.source_buffer_sizes is not None:
        if len(args.source_buffer_sizes) != args.num_source_buffers:
            raise ValueError(
                "--source-buffer-sizes must specify exactly --num-source-buffers values"
            )
        if any(nbytes <= 0 for nbytes in args.source_buffer_sizes):
            raise ValueError("--source-buffer-sizes values must be positive")
        sub_buffer_sizes = args.source_buffer_sizes
    else:
        sub_buffer_sizes = tuple(args.nbytes for _ in range(args.num_source_buffers))
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    if args.copy_mode == "batch":
        _get_cuda_memcpy_batch_async()

    src_device = local_rank
    dst_devices = destination_order(src_device, world_size)
    num_destinations = len(dst_devices)
    entries_per_batch = args.num_source_buffers
    batches_per_iter = num_destinations
    copies_per_iter = batches_per_iter * entries_per_batch
    bytes_per_destination = sum(sub_buffer_sizes)
    bytes_per_iter = bytes_per_destination * num_destinations

    unsupported_peers = [
        device for device in dst_devices
        if not cuda_can_access_peer(src_device, device)
    ]
    if unsupported_peers:
        raise RuntimeError(
            f"Source GPU {src_device} cannot access destination peer GPU(s) "
            f"{unsupported_peers}. Check CUDA_VISIBLE_DEVICES and nvidia-smi topo -m."
        )

    for dst_device in dst_devices:
        cuda_enable_peer_access(src_device, dst_device)

    source_buffer_extents = tuple(nbytes * num_destinations for nbytes in sub_buffer_sizes)
    source_buffer_offsets: List[int] = []
    source_backing_size = 0
    for source_index, extent in enumerate(source_buffer_extents):
        source_buffer_offsets.append(source_backing_size)
        source_backing_size += extent
        if source_index + 1 < args.num_source_buffers:
            source_backing_size += args.source_buffer_gap_size

    torch.cuda.set_device(src_device)
    source_backing = torch.empty(
        source_backing_size,
        dtype=torch.uint8,
        device=f"cuda:{src_device}",
    )
    source_backing.fill_((rank + 211) % 251)

    source_buffers: List[torch.Tensor] = []
    for source_index, (source_offset, sub_buffer_size, source_extent) in enumerate(
        zip(source_buffer_offsets, sub_buffer_sizes, source_buffer_extents)
    ):
        src = source_backing[source_offset:source_offset + source_extent]
        for destination_index in range(num_destinations):
            offset = destination_index * sub_buffer_size
            src[offset:offset + sub_buffer_size].fill_(
                chunk_value(rank, source_index, destination_index)
            )
        source_buffers.append(src)

    dst_offsets = build_contiguous_offsets(sub_buffer_sizes)
    destinations: Dict[int, torch.Tensor] = {}
    for dst_device in dst_devices:
        torch.cuda.set_device(dst_device)
        dst = torch.empty(
            bytes_per_destination,
            dtype=torch.uint8,
            device=f"cuda:{dst_device}",
        )
        dst.fill_((rank + 113) % 251)
        destinations[dst_device] = dst

    torch.cuda.synchronize(src_device)
    for dst_device in dst_devices:
        torch.cuda.synchronize(dst_device)

    src_ptr_groups: Dict[int, Tuple[int, ...]] = {}
    dst_ptr_groups: Dict[int, Tuple[int, ...]] = {}
    batches: Dict[int, MemcpyBatch] = {}
    for destination_index, dst_device in enumerate(dst_devices):
        src_ptrs = tuple(
            source_buffers[source_index].data_ptr()
            + destination_index * sub_buffer_sizes[source_index]
            for source_index in range(args.num_source_buffers)
        )
        dst_ptrs = tuple(
            destinations[dst_device].data_ptr() + dst_offsets[source_index]
            for source_index in range(args.num_source_buffers)
        )
        src_ptr_groups[dst_device] = src_ptrs
        dst_ptr_groups[dst_device] = dst_ptrs
        batches[dst_device] = MemcpyBatch(dst_ptrs, src_ptrs, sub_buffer_sizes)

    torch.cuda.set_device(src_device)
    stream = torch.cuda.Stream(device=src_device)

    barrier()
    if args.sleep_before > 0:
        if rank == 0:
            print(f"Sleeping for {args.sleep_before:.1f}s before benchmark...", flush=True)
        time.sleep(args.sleep_before)
    barrier()

    cuda_set_device(src_device)
    for _ in range(args.warmup):
        enqueue_iteration(
            args.copy_mode, src_device, dst_devices, src_ptr_groups,
            dst_ptr_groups, sub_buffer_sizes, batches, stream
        )
    cuda_stream_synchronize(stream)
    barrier()

    cuda_set_device(src_device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        enqueue_iteration(
            args.copy_mode, src_device, dst_devices, src_ptr_groups,
            dst_ptr_groups, sub_buffer_sizes, batches, stream
        )
    cuda_stream_synchronize(stream)
    t1 = time.perf_counter()
    barrier()

    elapsed = t1 - t0
    local_gib = (bytes_per_iter * args.iters) / (1024**3)
    local_egress_bw = local_gib / elapsed

    torch.cuda.set_device(local_rank)
    ref_device = torch.device(f"cuda:{local_rank}")
    max_elapsed = all_reduce_max_float(elapsed, ref_device)
    sum_gib = all_reduce_sum_float(local_gib, ref_device)
    aggregate_bw = sum_gib / max_elapsed

    ok = True
    failure_detail = ""
    if args.check:
        ok, failure_detail = check_destinations(
            destinations, dst_offsets, dst_devices, sub_buffer_sizes, rank
        )

    torch.cuda.set_device(src_device)
    for output_rank in range(world_size):
        barrier()
        if rank == output_rank:
            print(
                f"rank={rank:02d} source=cuda:{src_device} destinations={list(dst_devices)} "
                f"mode={args.copy_mode} source_buffers={args.num_source_buffers} "
                f"batches/iter={batches_per_iter} entries/batch={entries_per_batch} "
                f"copies/iter={copies_per_iter} bytes/destination={fmt_bytes(bytes_per_destination)} "
                f"bytes/iter={fmt_bytes(bytes_per_iter)} "
                f"source_alloc={fmt_bytes(source_backing_size)} "
                f"source_gap={fmt_bytes(args.source_buffer_gap_size)} "
                f"sub_buffer_sizes={format_source_sizes(sub_buffer_sizes)} "
                f"iters={args.iters} elapsed={elapsed:.6f}s "
                f"egress_bw={local_egress_bw:.2f} GiB/s "
                f"check={'OK' if ok else f'FAIL:{failure_detail}'}",
                flush=True,
            )
        barrier()

    if rank == 0:
        submission_description = (
            f"{batches_per_iter} cudaMemcpyBatchAsync calls, "
            f"each with {entries_per_batch} entries"
            if args.copy_mode == "batch"
            else f"{copies_per_iter} cudaMemcpyPeerAsync calls grouped by destination"
        )
        print(
            f"\nAggregate multi-source all-to-all traffic over {world_size} ranks: "
            f"copies/iteration={world_size * copies_per_iter}, "
            f"moved={sum_gib:.2f} GiB, slowest_elapsed={max_elapsed:.6f}s, "
            f"aggregate_bw={aggregate_bw:.2f} GiB/s",
            flush=True,
        )
        print(
            f"\nProfiler expectation: each rank submits {submission_description} per iteration.",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
