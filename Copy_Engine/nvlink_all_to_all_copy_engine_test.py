#!/usr/bin/env python3
"""
Benchmark all-to-all GPU P2P traffic using CUDA copy engines.

Each rank owns one source GPU. In every iteration, GPU i copies the same source
buffer to one private destination buffer on every other participating GPU.
Across all ranks, this creates concurrent all-to-all traffic.

Compare separate and batched API launches on one 8-GPU node:
  torchrun --standalone --nproc_per_node=8 nvlink_all_to_all_copy_engine_test.py \
    --nbytes 16M --copy-mode separate --iters 100 --check
  torchrun --standalone --nproc_per_node=8 nvlink_all_to_all_copy_engine_test.py \
    --nbytes 16M --copy-mode batch --iters 100 --check

Per rank and iteration:
  - separate: world_size - 1 cudaMemcpyPeerAsync calls
  - batch:    one cudaMemcpyBatchAsync containing world_size - 1 copies

The batch copies all read the same source buffer and write different destination
buffers, so they are independent and do not rely on ordering within the batch.
"""

import argparse
import time
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist

from nvlink_copy_engine_test import (
    MemcpyBatch,
    _get_cuda_memcpy_batch_async,
    all_reduce_max_float,
    all_reduce_sum_float,
    barrier,
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


def enqueue_iteration(copy_mode: str,
                      src_ptr: int,
                      src_device: int,
                      dst_ptrs: Sequence[int],
                      dst_devices: Sequence[int],
                      nbytes: int,
                      batch: MemcpyBatch,
                      stream: torch.cuda.Stream) -> None:
    if copy_mode == "batch":
        cuda_memcpy_batch_async(batch, stream)
        return

    for dst_ptr, dst_device in zip(dst_ptrs, dst_devices):
        cuda_memcpy_peer_async(
            dst_ptr, dst_device, src_ptr, src_device, nbytes, stream
        )


def destination_order(src_device: int,
                      world_size: int,
                      rotate_by_source: bool) -> tuple[int, ...]:
    if not rotate_by_source:
        return tuple(device for device in range(world_size) if device != src_device)
    return tuple((src_device + offset) % world_size for offset in range(1, world_size))


def check_destinations(destinations: Dict[int, torch.Tensor],
                       nbytes: int,
                       expected: int) -> List[int]:
    failed: List[int] = []
    sample_indexes = torch.tensor(
        sorted({0, nbytes // 2, nbytes - 1}), dtype=torch.int64
    )
    for device, dst in destinations.items():
        torch.cuda.set_device(device)
        samples = dst[sample_indexes.to(device=f"cuda:{device}")].cpu()
        if not bool(torch.all(samples == expected).item()):
            failed.append(device)
    return failed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nbytes", "--copy-size", dest="nbytes",
                   type=parse_nbytes, default=parse_nbytes("16M"),
                   help="bytes copied from GPU i to each other GPU per iteration. Default: 16M")
    p.add_argument("--copy-mode", choices=["separate", "batch"], default="separate",
                   help="separate: one cudaMemcpyPeerAsync call per destination; "
                        "batch: one cudaMemcpyBatchAsync call for all destinations. "
                        "Default: separate")
    p.add_argument("--iters", type=int, default=100, help="timed iterations")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("--check", action="store_true",
                   help="verify copied bytes on every destination GPU")
    p.add_argument("--sleep-before", type=float, default=0.0,
                   help="seconds to sleep before benchmark, useful for attaching profilers")
    p.add_argument("--rotate-destination-order", action="store_true",
                   help="send from GPU i to (i+1)%%world_size, ..., (i-1)%%world_size. "
                        "By default destinations are ascending GPU IDs with the source omitted.")
    args = p.parse_args()

    rank, world_size, local_rank = get_rank_info()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch build.")

    num_gpus = torch.cuda.device_count()
    if world_size > num_gpus:
        raise RuntimeError(f"world_size={world_size}, but only {num_gpus} CUDA devices visible.")
    if world_size < 2:
        raise RuntimeError("This benchmark requires at least two ranks/GPUs.")
    if args.nbytes <= 0:
        raise ValueError("--nbytes must be positive")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    if args.copy_mode == "batch":
        _get_cuda_memcpy_batch_async()

    src_device = local_rank
    dst_devices = destination_order(
        src_device, world_size, args.rotate_destination_order
    )
    copies_per_iter = len(dst_devices)
    bytes_per_iter = args.nbytes * copies_per_iter

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

    torch.cuda.set_device(src_device)
    src = torch.empty(args.nbytes, dtype=torch.uint8, device=f"cuda:{src_device}")
    expected = (rank + 17) % 251
    src.fill_(expected)

    destinations: Dict[int, torch.Tensor] = {}
    for dst_device in dst_devices:
        torch.cuda.set_device(dst_device)
        dst = torch.empty(args.nbytes, dtype=torch.uint8, device=f"cuda:{dst_device}")
        dst.fill_((expected + 1) % 251)
        destinations[dst_device] = dst

    torch.cuda.synchronize(src_device)
    for dst_device in dst_devices:
        torch.cuda.synchronize(dst_device)

    src_ptr = src.data_ptr()
    src_ptrs = (src_ptr,) * copies_per_iter
    dst_ptrs = tuple(destinations[device].data_ptr() for device in dst_devices)
    batch = MemcpyBatch(dst_ptrs, src_ptrs, args.nbytes)

    # One source-device stream owns all outgoing copies from this rank.
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
            args.copy_mode, src_ptr, src_device, dst_ptrs, dst_devices,
            args.nbytes, batch, stream
        )
    cuda_stream_synchronize(stream)
    barrier()

    cuda_set_device(src_device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        enqueue_iteration(
            args.copy_mode, src_ptr, src_device, dst_ptrs, dst_devices,
            args.nbytes, batch, stream
        )
    cuda_stream_synchronize(stream)
    t1 = time.perf_counter()
    barrier()

    elapsed = t1 - t0
    local_gib = (bytes_per_iter * args.iters) / (1024**3)
    local_egress_bw = local_gib / elapsed

    ref_device = torch.device(f"cuda:{local_rank}")
    max_elapsed = all_reduce_max_float(elapsed, ref_device)
    sum_gib = all_reduce_sum_float(local_gib, ref_device)
    aggregate_bw = sum_gib / max_elapsed

    failed_destinations: List[int] = []
    if args.check:
        failed_destinations = check_destinations(destinations, args.nbytes, expected)
    ok = not failed_destinations

    # Keep NCCL barriers associated with this rank's local/source GPU.
    torch.cuda.set_device(src_device)
    for output_rank in range(world_size):
        barrier()
        if rank == output_rank:
            print(
                f"rank={rank:02d} source=cuda:{src_device} destinations={list(dst_devices)} "
                f"mode={args.copy_mode} copies/iter={copies_per_iter} "
                f"bytes/copy={fmt_bytes(args.nbytes)} bytes/iter={fmt_bytes(bytes_per_iter)} "
                f"iters={args.iters} elapsed={elapsed:.6f}s "
                f"egress_bw={local_egress_bw:.2f} GiB/s "
                f"check={'OK' if ok else f'FAIL:{failed_destinations}'}",
                flush=True,
            )
        barrier()

    if rank == 0:
        submission_description = (
            "one cudaMemcpyBatchAsync"
            if args.copy_mode == "batch"
            else f"{copies_per_iter} cudaMemcpyPeerAsync calls"
        )
        print(
            f"\nAggregate all-to-all traffic over {world_size} ranks: "
            f"copies/iteration={world_size * copies_per_iter}, "
            f"moved={sum_gib:.2f} GiB, slowest_elapsed={max_elapsed:.6f}s, "
            f"aggregate_bw={aggregate_bw:.2f} GiB/s",
            flush=True,
        )
        print(
            f"\nProfiler expectation: each rank submits "
            f"{submission_description} per iteration.",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
