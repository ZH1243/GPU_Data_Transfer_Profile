#!/usr/bin/env python3
"""
nvlink_copy_engine_test.py

Benchmark / sanity-test GPU-to-GPU P2P copies using CUDA Runtime
cudaMemcpyPeerAsync or cudaMemcpyBatchAsync from Python.

Intent:
  - Move data between NVIDIA GPUs over P2P/NVLink using the CUDA copy engine.
  - Avoid launching an SM copy kernel for the transfer itself.
  - Make the transfer easy to verify with Nsight Systems:
        look for cudaMemcpyPeerAsync / GPU Memcpy PtoP / [CUDA memcpy PtoP],
        not a CUDA kernel doing the copy.

Typical launch on one 8-GPU node:
  torchrun --standalone --nproc_per_node=8 nvlink_copy_engine_test.py --nbytes 1G --iters 100

Compare separate and batched API launches:
  torchrun --standalone --nproc_per_node=8 nvlink_copy_engine_test.py \
    --copy-size 16M --copies-per-iter 8 --copy-mode separate
  torchrun --standalone --nproc_per_node=8 nvlink_copy_engine_test.py \
    --copy-size 16M --copies-per-iter 8 --copy-mode batch

Use nsys to profile (a sample command)
   nsys profile \
    -s none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,cudnn,cublas \
    -o nvlink_p2p_copy_100m \
    --gpu-metrics-devices=0 \
    --gpu-metrics-set=gh100 \
    --gpu-metrics-frequency=10000 \
    --force-overwrite=true \
    torchrun --standalone --nproc_per_node=8 nvlink_copy_engine_test.py \
    --nbytes 100M \
    --iters 100 \
    --warmup 10 \
    --mode ring \
    --check

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
from typing import Sequence, Tuple, Union

import torch
import torch.distributed as dist


# ----------------------------
# CUDA runtime binding via ctypes
# ----------------------------

def _load_libcudart():
    candidates = [
        "libcudart.so",
        "libcudart.so.13",
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


class _CudaMemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


class _CudaMemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", _CudaMemLocation),
        ("dstLocHint", _CudaMemLocation),
        ("flags", ctypes.c_uint),
    ]


_cudart.cudaGetErrorString.argtypes = [_cudaError_t]
_cudart.cudaGetErrorString.restype = ctypes.c_char_p

_cudart.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
_cudart.cudaRuntimeGetVersion.restype = _cudaError_t

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
CUDA_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
CUDA_MEMCPY_BATCH_MIN_RUNTIME_VERSION = 12080
CUDA_13_RUNTIME_VERSION = 13000

_cuda_memcpy_batch_async_fn = None
_cuda_memcpy_batch_runtime_version = None


def cuda_check(code: int, what: str) -> None:
    if code != CUDA_SUCCESS:
        msg = _cudart.cudaGetErrorString(code)
        msg = msg.decode("utf-8") if msg else f"CUDA error {code}"
        raise RuntimeError(f"{what} failed: {msg} ({code})")


def cuda_set_device(dev: int) -> None:
    cuda_check(_cudart.cudaSetDevice(dev), f"cudaSetDevice({dev})")


def cuda_runtime_version() -> int:
    version = ctypes.c_int(0)
    cuda_check(_cudart.cudaRuntimeGetVersion(ctypes.byref(version)), "cudaRuntimeGetVersion")
    return version.value


def _get_cuda_memcpy_batch_async():
    """
    Bind cudaMemcpyBatchAsync lazily so separate mode still works with CUDA < 12.8.

    CUDA 12.8/12.9 include a failIdx argument. CUDA 13.x removed it from the C API.
    """
    global _cuda_memcpy_batch_async_fn, _cuda_memcpy_batch_runtime_version

    if _cuda_memcpy_batch_async_fn is not None:
        return _cuda_memcpy_batch_async_fn, _cuda_memcpy_batch_runtime_version

    runtime_version = cuda_runtime_version()
    if runtime_version < CUDA_MEMCPY_BATCH_MIN_RUNTIME_VERSION:
        raise RuntimeError(
            "cudaMemcpyBatchAsync requires CUDA Runtime 12.8 or newer; "
            f"loaded runtime version is {runtime_version}."
        )

    try:
        fn = _cudart.cudaMemcpyBatchAsync
    except AttributeError as exc:
        raise RuntimeError(
            "The loaded CUDA Runtime does not export cudaMemcpyBatchAsync. "
            "Use --copy-mode separate or load CUDA Runtime 12.8 or newer."
        ) from exc

    common_argtypes = [
        ctypes.POINTER(ctypes.c_void_p),       # dsts
        ctypes.POINTER(ctypes.c_void_p),       # srcs
        ctypes.POINTER(ctypes.c_size_t),       # sizes
        ctypes.c_size_t,                       # count
        ctypes.POINTER(_CudaMemcpyAttributes), # attrs
        ctypes.POINTER(ctypes.c_size_t),       # attrsIdxs
        ctypes.c_size_t,                       # numAttrs
    ]
    if runtime_version < CUDA_13_RUNTIME_VERSION:
        fn.argtypes = common_argtypes + [
            ctypes.POINTER(ctypes.c_size_t),   # failIdx
            _cudaStream_t,
        ]
    else:
        fn.argtypes = common_argtypes + [_cudaStream_t]
    fn.restype = _cudaError_t

    _cuda_memcpy_batch_async_fn = fn
    _cuda_memcpy_batch_runtime_version = runtime_version
    return fn, runtime_version


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


def cuda_memcpy_peer_async(dst_ptr: int,
                           dst_device: int,
                           src_ptr: int,
                           src_device: int,
                           nbytes: int,
                           stream: torch.cuda.Stream) -> None:
    """
    Enqueue cudaMemcpyPeerAsync(dst@dst_device <- src@src_device) on stream.
    """
    code = _cudart.cudaMemcpyPeerAsync(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_int(dst_device),
        ctypes.c_void_p(src_ptr),
        ctypes.c_int(src_device),
        ctypes.c_size_t(nbytes),
        _cudaStream_t(stream.cuda_stream),
    )
    cuda_check(code, "cudaMemcpyPeerAsync")


class MemcpyBatch:
    """Own the host-side argument arrays used by repeated cudaMemcpyBatchAsync calls."""

    def __init__(self,
                 dst_ptrs: Sequence[int],
                 src_ptrs: Sequence[int],
                 nbytes: Union[int, Sequence[int]]) -> None:
        if len(dst_ptrs) != len(src_ptrs):
            raise ValueError("source and destination pointer counts must match")
        if not dst_ptrs:
            raise ValueError("a memcpy batch must contain at least one copy")

        self.count = len(dst_ptrs)
        if isinstance(nbytes, int):
            sizes = [nbytes] * self.count
        else:
            sizes = list(nbytes)
            if len(sizes) != self.count:
                raise ValueError("copy-size count must match pointer count")

        self.dsts = (ctypes.c_void_p * self.count)(*dst_ptrs)
        self.srcs = (ctypes.c_void_p * self.count)(*src_ptrs)
        self.sizes = (ctypes.c_size_t * self.count)(*sizes)

        # All copies use stable device pointers and the same stream-ordering attribute.
        self.attrs = (_CudaMemcpyAttributes * 1)()
        self.attrs[0].srcAccessOrder = CUDA_MEMCPY_SRC_ACCESS_ORDER_STREAM
        self.attrs_idxs = (ctypes.c_size_t * 1)(0)


def cuda_memcpy_batch_async(batch: MemcpyBatch,
                            stream: torch.cuda.Stream) -> None:
    """
    Enqueue one cudaMemcpyBatchAsync containing all independent copies in batch.
    """
    fn, runtime_version = _get_cuda_memcpy_batch_async()

    args = [
        batch.dsts,
        batch.srcs,
        batch.sizes,
        ctypes.c_size_t(batch.count),
        batch.attrs,
        batch.attrs_idxs,
        ctypes.c_size_t(1),
    ]
    if runtime_version < CUDA_13_RUNTIME_VERSION:
        fail_idx = ctypes.c_size_t(-1)
        code = fn(*args, ctypes.byref(fail_idx), _cudaStream_t(stream.cuda_stream))
        failure_detail = f", fail_idx={fail_idx.value}" if code != CUDA_SUCCESS else ""
    else:
        code = fn(*args, _cudaStream_t(stream.cuda_stream))
        failure_detail = ""
    cuda_check(code, f"cudaMemcpyBatchAsync{failure_detail}")


def enqueue_iteration(copy_mode: str,
                      dst_ptrs: Sequence[int],
                      dst_device: int,
                      src_ptrs: Sequence[int],
                      src_device: int,
                      copy_sizes: Sequence[int],
                      batch: MemcpyBatch,
                      stream: torch.cuda.Stream) -> None:
    if copy_mode == "batch":
        cuda_memcpy_batch_async(batch, stream)
        return

    for dst_ptr, src_ptr, copy_size in zip(dst_ptrs, src_ptrs, copy_sizes):
        cuda_memcpy_peer_async(
            dst_ptr, dst_device, src_ptr, src_device, copy_size, stream
        )


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


def parse_copy_sizes(spec: str) -> Tuple[int, ...]:
    values = [part.strip() for part in spec.split(",")]
    if any(not value for value in values):
        raise argparse.ArgumentTypeError(
            "--copy-sizes must be a comma-separated list like 64K,128K,1M"
        )
    try:
        return tuple(parse_nbytes(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_contiguous_offsets(copy_sizes: Sequence[int]) -> Tuple[int, ...]:
    offsets = []
    offset = 0
    for copy_size in copy_sizes:
        offsets.append(offset)
        offset += copy_size
    return tuple(offsets)


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
    p.add_argument("--nbytes", "--copy-size", dest="nbytes",
                   type=parse_nbytes, default=parse_nbytes("1G"),
                   help="bytes per copy, supports K/M/G, KB/MB/GB, KiB/MiB/GiB. Default: 1G")
    p.add_argument("--copies-per-iter", type=int, default=1,
                   help="number of independent memory copies in each iteration. Default: 1")
    p.add_argument("--non-uniform-copy-size", action="store_true",
                   help="use per-copy sizes from --copy-sizes instead of --copy-size")
    p.add_argument("--copy-sizes", type=parse_copy_sizes,
                   help="comma-separated bytes for each copy in one iteration, "
                        "for example 64K,128K,1M; requires --non-uniform-copy-size")
    p.add_argument("--copy-mode", choices=["separate", "batch"], default="separate",
                   help="separate: one cudaMemcpyPeerAsync call per copy; "
                        "batch: one cudaMemcpyBatchAsync call per iteration. Default: separate")
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
    if args.copies_per_iter <= 0:
        raise ValueError("--copies-per-iter must be positive")
    if args.non_uniform_copy_size:
        if args.copy_sizes is None:
            raise ValueError("--non-uniform-copy-size requires --copy-sizes")
        if len(args.copy_sizes) != args.copies_per_iter:
            raise ValueError(
                "--copy-sizes must specify exactly --copies-per-iter values"
            )
        if any(copy_size <= 0 for copy_size in args.copy_sizes):
            raise ValueError("--copy-sizes values must be positive")
        copy_sizes = args.copy_sizes
    else:
        if args.copy_sizes is not None:
            raise ValueError("--copy-sizes requires --non-uniform-copy-size")
        copy_sizes = tuple(args.nbytes for _ in range(args.copies_per_iter))
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    if args.copy_mode == "batch":
        # Fail before allocating large GPU buffers when the runtime cannot run this mode.
        _get_cuda_memcpy_batch_async()

    src_dev, dst_dev = choose_pair(local_rank, world_size, args.mode)

    # Copies are submitted on a destination-device stream, so dst must be able to access src.
    # Enable the reverse direction too when available for bidirectional/pair experiments.
    can_src_to_dst = cuda_can_access_peer(src_dev, dst_dev)
    can_dst_to_src = cuda_can_access_peer(dst_dev, src_dev)

    if not can_dst_to_src:
        raise RuntimeError(
            f"Destination GPU {dst_dev} cannot access source peer GPU {src_dev}. "
            "Check CUDA_VISIBLE_DEVICES and nvidia-smi topo -m."
        )

    cuda_enable_peer_access(dst_dev, src_dev)
    if can_src_to_dst:
        cuda_enable_peer_access(src_dev, dst_dev)

    # Allocate one contiguous source/destination region, split into independent copies.
    # uint8 makes both the per-copy size and pointer offsets exact.
    total_nbytes = sum(copy_sizes)
    copy_offsets = build_contiguous_offsets(copy_sizes)
    torch.cuda.set_device(src_dev)
    src = torch.empty(total_nbytes, dtype=torch.uint8, device=f"cuda:{src_dev}")
    src.fill_((rank + 17) % 251)

    torch.cuda.set_device(dst_dev)
    dst = torch.empty(total_nbytes, dtype=torch.uint8, device=f"cuda:{dst_dev}")
    dst.fill_((rank + 18) % 251)

    # The copy stream is on dst_dev, so finish initialization on both devices explicitly.
    torch.cuda.synchronize(src_dev)
    torch.cuda.synchronize(dst_dev)

    src_ptrs = tuple(src.data_ptr() + offset for offset in copy_offsets)
    dst_ptrs = tuple(dst.data_ptr() + offset for offset in copy_offsets)
    batch = MemcpyBatch(dst_ptrs, src_ptrs, copy_sizes)

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

    cuda_set_device(dst_dev)
    for _ in range(args.warmup):
        enqueue_iteration(
            args.copy_mode, dst_ptrs, dst_dev, src_ptrs, src_dev,
            copy_sizes, batch, stream
        )
    cuda_stream_synchronize(stream)
    barrier()

    # Time with wall-clock around a batch of async copies + stream synchronize.
    # This avoids inserting timing kernels/events into the copy stream.
    cuda_set_device(dst_dev)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        enqueue_iteration(
            args.copy_mode, dst_ptrs, dst_dev, src_ptrs, src_dev,
            copy_sizes, batch, stream
        )
    cuda_stream_synchronize(stream)
    t1 = time.perf_counter()
    barrier()

    elapsed = t1 - t0
    local_gib = (total_nbytes * args.iters) / (1024**3)
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
        copy_starts = torch.tensor(
            copy_offsets, dtype=torch.int64, device=f"cuda:{dst_dev}"
        )
        sample = dst[copy_starts].cpu()
        expected = (rank + 17) % 251
        ok = bool(torch.all(sample == expected).item())

    if args.non_uniform_copy_size:
        copy_size_detail = (
            f"bytes/iter={fmt_bytes(total_nbytes)} "
            f"copy_sizes={','.join(fmt_bytes(size) for size in copy_sizes)} "
        )
    else:
        copy_size_detail = (
            f"bytes/copy={fmt_bytes(args.nbytes)} bytes/iter={fmt_bytes(total_nbytes)} "
        )

    for r in range(world_size):
        barrier()
        if rank == r:
            print(
                f"rank={rank:02d} local_rank={local_rank:02d} "
                f"copy cuda:{src_dev} -> cuda:{dst_dev} "
                f"mode={args.copy_mode} copies/iter={args.copies_per_iter} "
                f"{copy_size_detail}"
                f"iters={args.iters} "
                f"elapsed={elapsed:.6f}s local_bw={local_bw:.2f} GiB/s "
                f"p2p={int(can_dst_to_src)} check={'OK' if ok else 'FAIL'}",
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
            f"\nProfiler expectation: Nsight Systems should show "
            f"{'cudaMemcpyBatchAsync' if args.copy_mode == 'batch' else 'cudaMemcpyPeerAsync'} "
            "/ GPU Memcpy PtoP activity for the transfer, not an SM copy kernel.",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
