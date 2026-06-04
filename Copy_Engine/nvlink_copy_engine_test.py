#!/usr/bin/env python3
"""
nvlink_copy_engine_test.py

Benchmark / sanity-test GPU-to-GPU P2P copies using CUDA Runtime
cudaMemcpyPeerAsync from Python.

Intent:
  - Move data between NVIDIA GPUs over P2P/NVLink using the CUDA copy engine.
  - Avoid launching an SM copy kernel for the transfer itself.
  - Make the transfer easy to verify with Nsight Systems:
        look for cudaMemcpyPeerAsync / GPU Memcpy PtoP / [CUDA memcpy PtoP],
        not a CUDA kernel doing the copy.

Typical launch on one 8-GPU node:
  torchrun --standalone --nproc_per_node=8 nvlink_copy_engine_test.py --nbytes 1G --iters 100

Notes:
  - This requires CUDA-capable PyTorch.
  - P2P access must be supported between the selected GPU pairs.
  - NVLink use depends on your system topology. Check with:
        nvidia-smi topo -m
"""

import argparse
import ctypes
import os
import sys
import time
from typing import Tuple

import torch
import torch.distributed as dist


# ----------------------------
# CUDA runtime binding via ctypes
# ----------------------------

def _load_libcudart():
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    last_err = None
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            last_err = exc
    raise RuntimeError(f"Could not load CUDA runtime library. Last error: {last_err}")


_cudart = _load_libcudart()

_cudaError_t = ctypes.c_int
_cudaStream_t = ctypes.c_void_p

_cudart.cudaGetErrorString.argtypes = [_cudaError_t]
_cudart.cudaGetErrorString.restype = ctypes.c_char_p

_cudart.cudaSetDevice.argtypes = [ctypes.c_int]
_cudart.cudaSetDevice.restype = _cudaError_t

_cudart.cudaDeviceCanAccessPeer.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
_cudart.cudaDeviceCanAccessPeer.restype = _cudaError_t

_cudart.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
_cudart.cudaDeviceEnablePeerAccess.restype = _cudaError_t

_cudart.cudaMemcpyPeerAsync.argtypes = [
    ctypes.c_void_p,  # dst
    ctypes.c_int,     # dstDevice
    ctypes.c_void_p,  # src
    ctypes.c_int,     # srcDevice
    ctypes.c_size_t,  # count
    _cudaStream_t,    # stream
]
_cudart.cudaMemcpyPeerAsync.restype = _cudaError_t

_cudart.cudaStreamSynchronize.argtypes = [_cudaStream_t]
_cudart.cudaStreamSynchronize.restype = _cudaError_t


CUDA_SUCCESS = 0
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704


def cuda_check(code: int, what: str) -> None:
    if code != CUDA_SUCCESS:
        msg = _cudart.cudaGetErrorString(code)
        msg = msg.decode("utf-8") if msg else f"CUDA error {code}"
        raise RuntimeError(f"{what} failed: {msg} ({code})")


def cuda_set_device(dev: int) -> None:
    cuda_check(_cudart.cudaSetDevice(dev), f"cudaSetDevice({dev})")


def cuda_can_access_peer(dev: int, peer: int) -> bool:
    can = ctypes.c_int(0)
    cuda_check(
        _cudart.cudaDeviceCanAccessPeer(ctypes.byref(can), dev, peer),
        f"cudaDeviceCanAccessPeer({dev}, {peer})",
    )
    return bool(can.value)


def cuda_enable_peer_access(dev: int, peer: int) -> None:
    """
    Enable dev -> peer access from the perspective of dev.
    If already enabled, treat it as success.
    """
    cuda_set_device(dev)
    code = _cudart.cudaDeviceEnablePeerAccess(peer, 0)
    if code == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
        return
    cuda_check(code, f"cudaDeviceEnablePeerAccess(dev={dev}, peer={peer})")


def cuda_memcpy_peer_async(dst_tensor: torch.Tensor,
                           dst_device: int,
                           src_tensor: torch.Tensor,
                           src_device: int,
                           nbytes: int,
                           stream: torch.cuda.Stream) -> None:
    """
    Enqueue cudaMemcpyPeerAsync(dst@dst_device <- src@src_device) on stream.
    """
    cuda_set_device(dst_device)
    code = _cudart.cudaMemcpyPeerAsync(
        ctypes.c_void_p(dst_tensor.data_ptr()),
        ctypes.c_int(dst_device),
        ctypes.c_void_p(src_tensor.data_ptr()),
        ctypes.c_int(src_device),
        ctypes.c_size_t(nbytes),
        _cudaStream_t(stream.cuda_stream),
    )
    cuda_check(code, "cudaMemcpyPeerAsync")


def cuda_stream_synchronize(stream: torch.cuda.Stream) -> None:
    code = _cudart.cudaStreamSynchronize(_cudaStream_t(stream.cuda_stream))
    cuda_check(code, "cudaStreamSynchronize")


# ----------------------------
# Helpers
# ----------------------------

def parse_nbytes(s: str) -> int:
    s = s.strip().upper()
    scale = 1
    for suffix, mult in [
        ("KIB", 1024),
        ("MIB", 1024**2),
        ("GIB", 1024**3),
        ("KB", 1000),
        ("MB", 1000**2),
        ("GB", 1000**3),
        ("K", 1024),
        ("M", 1024**2),
        ("G", 1024**3),
    ]:
        if s.endswith(suffix):
            scale = mult
            s = s[: -len(suffix)]
            break
    return int(float(s) * scale)


def fmt_bytes(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def get_rank_info() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        # Allow single-process debugging.
        return 0, 1, int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_max_float(x: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    t = torch.tensor([x], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def all_reduce_sum_float(x: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    t = torch.tensor([x], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def choose_pair(local_rank: int, world_size: int, mode: str) -> Tuple[int, int]:
    """
    Returns (src_device, dst_device) for this process.
    """
    if mode == "ring":
        return local_rank, (local_rank + 1) % world_size
    if mode == "reverse-ring":
        return local_rank, (local_rank - 1) % world_size
    if mode == "pair":
        if world_size % 2 != 0:
            raise ValueError("--mode pair requires an even number of ranks/GPUs")
        if local_rank % 2 == 0:
            return local_rank, local_rank + 1
        return local_rank, local_rank - 1
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nbytes", type=parse_nbytes, default=parse_nbytes("1G"),
                   help="bytes per copy, supports K/M/G, KB/MB/GB, KiB/MiB/GiB. Default: 1G")
    p.add_argument("--iters", type=int, default=100, help="timed iterations")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("--mode", choices=["ring", "reverse-ring", "pair"], default="ring",
                   help="copy pattern. ring: i->i+1, reverse-ring: i->i-1, pair: even<->odd")
    p.add_argument("--check", action="store_true",
                   help="verify a few copied bytes after the timed loop")
    p.add_argument("--sleep-before", type=float, default=0.0,
                   help="seconds to sleep before benchmark, useful for attaching profilers")
    args = p.parse_args()

    rank, world_size, local_rank = get_rank_info()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch build.")

    num_gpus = torch.cuda.device_count()
    if world_size > num_gpus:
        raise RuntimeError(f"world_size={world_size}, but only {num_gpus} CUDA devices visible.")

    if args.nbytes <= 0:
        raise ValueError("--nbytes must be positive")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")

    src_dev, dst_dev = choose_pair(local_rank, world_size, args.mode)

    # Check and enable peer access in both directions. The transfer needs src/dst peer capability;
    # enabling both directions is convenient for bidirectional/pair experiments.
    can_src_to_dst = cuda_can_access_peer(src_dev, dst_dev)
    can_dst_to_src = cuda_can_access_peer(dst_dev, src_dev)

    if not can_src_to_dst:
        raise RuntimeError(
            f"GPU {src_dev} cannot access peer GPU {dst_dev}. "
            "Check CUDA_VISIBLE_DEVICES and nvidia-smi topo -m."
        )

    cuda_enable_peer_access(src_dev, dst_dev)
    if can_dst_to_src:
        cuda_enable_peer_access(dst_dev, src_dev)

    # Allocate one byte tensor on src and one on dst. uint8 makes nbytes exact.
    torch.cuda.set_device(src_dev)
    src = torch.empty(args.nbytes, dtype=torch.uint8, device=f"cuda:{src_dev}")
    src.fill_((rank + 17) % 251)

    torch.cuda.set_device(dst_dev)
    dst = torch.empty(args.nbytes, dtype=torch.uint8, device=f"cuda:{dst_dev}")

    # Use a non-default stream on the destination device to make profiler timelines easy to read.
    torch.cuda.set_device(dst_dev)
    stream = torch.cuda.Stream(device=dst_dev)

    # Warm up.
    barrier()
    if args.sleep_before > 0:
        if rank == 0:
            print(f"Sleeping for {args.sleep_before:.1f}s before benchmark...", flush=True)
        time.sleep(args.sleep_before)
    barrier()

    for _ in range(args.warmup):
        cuda_memcpy_peer_async(dst, dst_dev, src, src_dev, args.nbytes, stream)
    cuda_stream_synchronize(stream)
    barrier()

    # Time with wall-clock around a batch of async copies + stream synchronize.
    # This avoids inserting timing kernels/events into the copy stream.
    t0 = time.perf_counter()
    for _ in range(args.iters):
        cuda_memcpy_peer_async(dst, dst_dev, src, src_dev, args.nbytes, stream)
    cuda_stream_synchronize(stream)
    t1 = time.perf_counter()
    barrier()

    elapsed = t1 - t0
    local_gib = (args.nbytes * args.iters) / (1024**3)
    local_bw = local_gib / elapsed

    # For concurrent torchrun results:
    # - max elapsed is the time until the slowest rank finished
    # - sum bytes / max elapsed approximates aggregate node bandwidth
    ref_device = torch.device(f"cuda:{local_rank}")
    max_elapsed = all_reduce_max_float(elapsed, ref_device)
    sum_gib = all_reduce_sum_float(local_gib, ref_device)
    agg_bw = sum_gib / max_elapsed

    # Optional correctness check. Pull back only a tiny slice.
    ok = True
    if args.check:
        torch.cuda.set_device(dst_dev)
        sample = dst[: min(4096, args.nbytes)].cpu()
        expected = (rank + 17) % 251
        ok = bool(torch.all(sample == expected).item())

    for r in range(world_size):
        barrier()
        if rank == r:
            print(
                f"rank={rank:02d} local_rank={local_rank:02d} "
                f"copy cuda:{src_dev} -> cuda:{dst_dev} "
                f"bytes/copy={fmt_bytes(args.nbytes)} iters={args.iters} "
                f"elapsed={elapsed:.6f}s local_bw={local_bw:.2f} GiB/s "
                f"p2p={int(can_src_to_dst)} check={'OK' if ok else 'FAIL'}",
                flush=True,
            )
        barrier()

    if rank == 0:
        print(
            f"\nAggregate over {world_size} ranks: moved={sum_gib:.2f} GiB, "
            f"slowest_elapsed={max_elapsed:.6f}s, aggregate_bw={agg_bw:.2f} GiB/s",
            flush=True,
        )
        print(
            "\nProfiler expectation: Nsight Systems should show cudaMemcpyPeerAsync / GPU Memcpy PtoP "
            "activity for the transfer, not an SM copy kernel.",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
