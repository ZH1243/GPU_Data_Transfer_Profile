#!/usr/bin/env python3
"""
Profile how cudaMemcpyBatchAsync maps same-pair copies with different layouts.

Every copy in an iteration goes from the same source GPU to the same destination
GPU. The selected layout changes only whether adjacent batch entries have
contiguous source and/or destination addresses:

  contiguous:        contiguous sources, contiguous destinations
  src-discontinuous: gaps between sources, contiguous destinations
  dst-discontinuous: contiguous sources, gaps between destinations
  both-discontinuous: gaps between sources and destinations

Examples on one 8-GPU node:
  torchrun --standalone --nproc_per_node=8 nvlink_batch_address_layout_test.py \
    --copy-size 1M --copies-per-iter 8 --layout contiguous --copy-mode batch --check
  torchrun --standalone --nproc_per_node=8 nvlink_batch_address_layout_test.py \
    --copy-size 1M --copies-per-iter 8 --layout src-discontinuous \
    --gap-size 64K --copy-mode batch --check
  torchrun --standalone --nproc_per_node=8 nvlink_batch_address_layout_test.py \
    --copy-size 1M --copies-per-iter 8 --layout dst-discontinuous \
    --gap-size 64K --copy-mode batch --check
  torchrun --standalone --nproc_per_node=8 nvlink_batch_address_layout_test.py \
    --copy-size 1M --copies-per-iter 8 --layout both-discontinuous \
    --gap-size 64K --copy-mode batch --check

nsys profile \
    -s none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,cudnn,cublas \
    -o nvlink_batch_8*125000_continuous \
    --force-overwrite=true \
    torchrun --standalone --nproc_per_node=8 nvlink_p2p_batch_addresslayout.py \
    --copy-size 125000 \
    --copies-per-iter 8 \
    --layout contiguous \
    --gap-size 3M \
    --copy-mode batch \
    --iters 100 \
    --warmup 10 \
    --mode ring \
    --check

The gaps are inside one allocation on each GPU. This guarantees non-contiguous
addresses without depending on allocator placement. Each source chunk has a
different value, and destination gaps use a sentinel checked by --check.
"""

import argparse
import time
from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist

from nvlink_copy_engine_test import (
    MemcpyBatch,
    _get_cuda_memcpy_batch_async,
    all_reduce_max_float,
    all_reduce_sum_float,
    barrier,
    choose_pair,
    cuda_can_access_peer,
    cuda_enable_peer_access,
    cuda_memcpy_batch_async,
    cuda_memcpy_peer_async,
    cuda_set_device,
    cuda_stream_synchronize,
    fmt_bytes,
    get_rank_info,
    parse_nbytes,
)


LAYOUTS = (
    "contiguous",
    "src-discontinuous",
    "dst-discontinuous",
    "both-discontinuous",
)
SOURCE_GAP_SENTINEL = 252
DESTINATION_GAP_SENTINEL = 253


def build_layout(layout: str,
                 copies_per_iter: int,
                 nbytes: int,
                 gap_size: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
    src_discontinuous = layout in ("src-discontinuous", "both-discontinuous")
    dst_discontinuous = layout in ("dst-discontinuous", "both-discontinuous")

    src_stride = nbytes + gap_size if src_discontinuous else nbytes
    dst_stride = nbytes + gap_size if dst_discontinuous else nbytes
    src_offsets = tuple(index * src_stride for index in range(copies_per_iter))
    dst_offsets = tuple(index * dst_stride for index in range(copies_per_iter))
    return src_offsets, dst_offsets, src_stride, dst_stride


def allocation_size(offsets: Sequence[int], nbytes: int) -> int:
    return offsets[-1] + nbytes


def chunk_value(rank: int, chunk_index: int) -> int:
    return (rank * 17 + chunk_index) % 251


def enqueue_iteration(copy_mode: str,
                      dst_ptrs: Sequence[int],
                      dst_device: int,
                      src_ptrs: Sequence[int],
                      src_device: int,
                      nbytes: int,
                      batch: MemcpyBatch,
                      stream: torch.cuda.Stream) -> None:
    if copy_mode == "batch":
        cuda_memcpy_batch_async(batch, stream)
        return

    for dst_ptr, src_ptr in zip(dst_ptrs, src_ptrs):
        cuda_memcpy_peer_async(
            dst_ptr, dst_device, src_ptr, src_device, nbytes, stream
        )


def check_destination(dst: torch.Tensor,
                      dst_offsets: Sequence[int],
                      dst_stride: int,
                      nbytes: int,
                      rank: int) -> Tuple[bool, str]:
    sample_indices: List[int] = []
    expected_values: List[int] = []

    for chunk_index, offset in enumerate(dst_offsets):
        for within_chunk in sorted({0, nbytes // 2, nbytes - 1}):
            sample_indices.append(offset + within_chunk)
            expected_values.append(chunk_value(rank, chunk_index))

    if dst_stride > nbytes:
        gap_size = dst_stride - nbytes
        for offset in dst_offsets[:-1]:
            for within_gap in sorted({0, gap_size // 2, gap_size - 1}):
                sample_indices.append(offset + nbytes + within_gap)
                expected_values.append(DESTINATION_GAP_SENTINEL)

    index_tensor = torch.tensor(sample_indices, dtype=torch.int64, device=dst.device)
    expected_tensor = torch.tensor(expected_values, dtype=torch.uint8)
    actual = dst[index_tensor].cpu()
    mismatches = torch.nonzero(actual != expected_tensor, as_tuple=False).flatten()
    if mismatches.numel() == 0:
        return True, ""

    first = int(mismatches[0].item())
    return (
        False,
        f"sample_offset={sample_indices[first]} expected={expected_values[first]} "
        f"actual={int(actual[first].item())}",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nbytes", "--copy-size", dest="nbytes",
                   type=parse_nbytes, default=parse_nbytes("1M"),
                   help="bytes per copy. Default: 1M")
    p.add_argument("--copies-per-iter", type=int, default=8,
                   help="number of copies in each iteration. Default: 8")
    p.add_argument("--layout", choices=LAYOUTS, default="contiguous",
                   help="source/destination address layout. Default: contiguous")
    p.add_argument("--gap-size", type=parse_nbytes, default=parse_nbytes("64K"),
                   help="gap between discontinuous copy regions. Default: 64K")
    p.add_argument("--copy-mode", choices=["separate", "batch"], default="batch",
                   help="separate: one cudaMemcpyPeerAsync per copy; "
                        "batch: one cudaMemcpyBatchAsync per iteration. Default: batch")
    p.add_argument("--iters", type=int, default=100, help="timed iterations")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("--mode", choices=["ring", "reverse-ring", "pair"], default="ring",
                   help="GPU-pair pattern. Default: ring")
    p.add_argument("--check", action="store_true",
                   help="verify each destination chunk and destination gaps")
    p.add_argument("--sleep-before", type=float, default=0.0,
                   help="seconds to sleep before benchmark, useful for attaching profilers")
    args = p.parse_args()

    rank, world_size, local_rank = get_rank_info()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch build.")
    if world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"world_size={world_size}, but only {torch.cuda.device_count()} CUDA devices visible."
        )
    if args.nbytes <= 0:
        raise ValueError("--nbytes must be positive")
    if args.copies_per_iter <= 0:
        raise ValueError("--copies-per-iter must be positive")
    if args.gap_size <= 0 and args.layout != "contiguous":
        raise ValueError("--gap-size must be positive for a discontinuous layout")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    if args.copy_mode == "batch":
        _get_cuda_memcpy_batch_async()

    src_device, dst_device = choose_pair(local_rank, world_size, args.mode)
    if not cuda_can_access_peer(dst_device, src_device):
        raise RuntimeError(
            f"Destination GPU {dst_device} cannot access source peer GPU {src_device}. "
            "Check CUDA_VISIBLE_DEVICES and nvidia-smi topo -m."
        )
    cuda_enable_peer_access(dst_device, src_device)

    src_offsets, dst_offsets, src_stride, dst_stride = build_layout(
        args.layout, args.copies_per_iter, args.nbytes, args.gap_size
    )
    src_allocation_size = allocation_size(src_offsets, args.nbytes)
    dst_allocation_size = allocation_size(dst_offsets, args.nbytes)
    bytes_per_iter = args.nbytes * args.copies_per_iter

    torch.cuda.set_device(src_device)
    src = torch.empty(src_allocation_size, dtype=torch.uint8, device=f"cuda:{src_device}")
    src.fill_(SOURCE_GAP_SENTINEL)
    for chunk_index, offset in enumerate(src_offsets):
        src[offset:offset + args.nbytes].fill_(chunk_value(rank, chunk_index))

    torch.cuda.set_device(dst_device)
    dst = torch.empty(dst_allocation_size, dtype=torch.uint8, device=f"cuda:{dst_device}")
    dst.fill_(DESTINATION_GAP_SENTINEL)

    torch.cuda.synchronize(src_device)
    torch.cuda.synchronize(dst_device)

    src_ptrs = tuple(src.data_ptr() + offset for offset in src_offsets)
    dst_ptrs = tuple(dst.data_ptr() + offset for offset in dst_offsets)
    batch = MemcpyBatch(dst_ptrs, src_ptrs, args.nbytes)

    torch.cuda.set_device(dst_device)
    stream = torch.cuda.Stream(device=dst_device)

    barrier()
    if args.sleep_before > 0:
        if rank == 0:
            print(f"Sleeping for {args.sleep_before:.1f}s before benchmark...", flush=True)
        time.sleep(args.sleep_before)
    barrier()

    cuda_set_device(dst_device)
    for _ in range(args.warmup):
        enqueue_iteration(
            args.copy_mode, dst_ptrs, dst_device, src_ptrs, src_device,
            args.nbytes, batch, stream
        )
    cuda_stream_synchronize(stream)
    barrier()

    cuda_set_device(dst_device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        enqueue_iteration(
            args.copy_mode, dst_ptrs, dst_device, src_ptrs, src_device,
            args.nbytes, batch, stream
        )
    cuda_stream_synchronize(stream)
    t1 = time.perf_counter()
    barrier()

    elapsed = t1 - t0
    local_gib = (bytes_per_iter * args.iters) / (1024**3)
    local_bw = local_gib / elapsed

    ref_device = torch.device(f"cuda:{local_rank}")
    max_elapsed = all_reduce_max_float(elapsed, ref_device)
    sum_gib = all_reduce_sum_float(local_gib, ref_device)
    aggregate_bw = sum_gib / max_elapsed

    ok = True
    failure_detail = ""
    if args.check:
        torch.cuda.set_device(dst_device)
        ok, failure_detail = check_destination(
            dst, dst_offsets, dst_stride, args.nbytes, rank
        )

    torch.cuda.set_device(local_rank)
    for output_rank in range(world_size):
        barrier()
        if rank == output_rank:
            print(
                f"rank={rank:02d} copy=cuda:{src_device}->cuda:{dst_device} "
                f"layout={args.layout} mode={args.copy_mode} "
                f"copies/iter={args.copies_per_iter} bytes/copy={fmt_bytes(args.nbytes)} "
                f"src_stride={fmt_bytes(src_stride)} dst_stride={fmt_bytes(dst_stride)} "
                f"src_alloc={fmt_bytes(src_allocation_size)} "
                f"dst_alloc={fmt_bytes(dst_allocation_size)} "
                f"elapsed={elapsed:.6f}s local_bw={local_bw:.2f} GiB/s "
                f"check={'OK' if ok else f'FAIL:{failure_detail}'}",
                flush=True,
            )
        barrier()

    if rank == 0:
        print(
            f"\nAggregate over {world_size} ranks: layout={args.layout}, "
            f"moved={sum_gib:.2f} GiB, slowest_elapsed={max_elapsed:.6f}s, "
            f"aggregate_bw={aggregate_bw:.2f} GiB/s",
            flush=True,
        )
        print(
            "\nProfiler question: does each cudaMemcpyBatchAsync map to one coalesced "
            "Memcpy PtoP activity or multiple activities for this layout?",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
