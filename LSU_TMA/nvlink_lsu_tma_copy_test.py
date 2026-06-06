#!/usr/bin/env python3
"""
nvlink_lsu_tma_copy_test.py

Benchmark / sanity-test GPU-to-GPU P2P copies over NVLink using SM work
instead of CUDA copy-engine APIs.

Methods:
  - lsu: ordinary SM global load/store kernel.
  - tma: Hopper bulk asynchronous copy, staged global -> shared -> global.

Typical launch on one 8-GPU Hopper node:
  torchrun --standalone --nproc_per_node=8 LSU_TMA/nvlink_lsu_tma_copy_test.py \
    --copy-size 1G --iters 100 --method lsu --executor dst --num-sms 0 --check
  torchrun --standalone --nproc_per_node=8 LSU_TMA/nvlink_lsu_tma_copy_test.py \
    --copy-size 1G --iters 100 --method tma --executor src --num-sms 0 --tma-tile-bytes 64K --check
  torchrun --standalone --nproc_per_node=8 LSU_TMA/nvlink_lsu_tma_copy_test.py \
    --copy-size 1G --iters 100 --method tma --executor src --num-sms 0 \
    --tma-tile-bytes 64K --tma-inter-tile-bytes 8K --check

Nsight Systems expectation:
  - lsu should show an SM kernel doing global loads/stores.
  - tma should show an SM kernel that issues Hopper TMA bulk async copies.

A sample command using the nsys profiling:
   nsys profile \
    -s none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,cudnn,cublas \
    -o tma_persistent_sm*3_intra_64k_inter_8k_1m*100 \
    --gpu-metrics-devices=0 \
    --gpu-metrics-set=gh100 \
    --gpu-metrics-frequency=10000 \
    --force-overwrite=true \
    torchrun --standalone --nproc_per_node=8 nvlink_lsu_tma_copy_test.py \
    --copy-size 1M \
    --iters 100 \
    --method tma \
    --executor src \
    --num-sms 3 \
    --tma-tile-bytes 64K \
    --tma-inter-tile-bytes 8K \
    --persistent-kernel \
    --check

Notes:
  - This requires CUDA-capable PyTorch and a CUDA toolkit available to
    torch.utils.cpp_extension.load_inline.
  - TMA mode requires Hopper or newer, CUDA 12-era headers with <cuda/ptx>,
    16-byte aligned pointers, and transfer/tile sizes that are multiples of 16.
  - --num-sms controls the number of CTAs launched per kernel. CTAs are not
    pinned to specific SMs, but when the CTA count is <= the physical SM count,
    it is the usual way to bound how many SMs can participate on the executor
    GPU selected by --executor.
  - --persistent-kernel launches one warmup kernel and one timed kernel, with
    each kernel looping over the requested number of copy iterations on device.
  - In TMA source-executor mode, --tma-tile-bytes controls the local
    global->shared staging tile, while --tma-inter-tile-bytes controls the
    shared->peer-global store chunk.
"""

import argparse
import ctypes
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline


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
    cuda_set_device(dev)
    code = _cudart.cudaDeviceEnablePeerAccess(peer, 0)
    if code == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
        return
    cuda_check(code, f"cudaDeviceEnablePeerAccess(dev={dev}, peer={peer})")


def cuda_stream_synchronize(stream: torch.cuda.Stream) -> None:
    code = _cudart.cudaStreamSynchronize(_cudaStream_t(stream.cuda_stream))
    cuda_check(code, "cudaStreamSynchronize")


# ----------------------------
# CUDA extension
# ----------------------------

CUDA_SRC = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <utility>

namespace {

using block_barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

__device__ inline bool elected_thread() {
  return threadIdx.x == 0;
}

__global__ void lsu_copy_kernel(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    size_t nbytes,
    int64_t iteration_count) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  bool aligned16 = ((dst_addr | src_addr | nbytes) & 0xfu) == 0;

  for (int64_t iter = 0; iter < iteration_count; ++iter) {
    if (aligned16) {
      const uint4* src16 = reinterpret_cast<const uint4*>(src);
      uint4* dst16 = reinterpret_cast<uint4*>(dst);
      size_t nvec = nbytes / sizeof(uint4);
      for (size_t i = tid; i < nvec; i += stride) {
        dst16[i] = src16[i];
      }
    } else {
      for (size_t i = tid; i < nbytes; i += stride) {
        dst[i] = src[i];
      }
    }
  }
}

__global__ void tma_copy_kernel(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    size_t nbytes,
    size_t intra_tile_bytes,
    size_t inter_tile_bytes,
    int64_t iteration_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(16) unsigned char smem[];

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  for (int64_t iter = 0; iter < iteration_count; ++iter) {
    size_t block_stride = static_cast<size_t>(gridDim.x) * intra_tile_bytes;
    for (size_t offset = static_cast<size_t>(blockIdx.x) * intra_tile_bytes;
         offset < nbytes;
         offset += block_stride) {
      size_t remaining = nbytes - offset;
      size_t bytes = remaining < intra_tile_bytes ? remaining : intra_tile_bytes;

      if (elected_thread()) {
        cuda::memcpy_async(
            smem,
            src + offset,
            cuda::aligned_size_t<16>(bytes),
            bar);
      }

      block_barrier::arrival_token token = bar.arrive();
      bar.wait(std::move(token));

      ptx::fence_proxy_async(ptx::space_shared);
      __syncthreads();

      if (elected_thread()) {
        for (size_t inner = 0; inner < bytes; inner += inter_tile_bytes) {
          size_t store_remaining = bytes - inner;
          size_t store_bytes = store_remaining < inter_tile_bytes
              ? store_remaining
              : inter_tile_bytes;
          ptx::cp_async_bulk(
              ptx::space_global,
              ptx::space_shared,
              dst + offset + inner,
              smem + inner,
              store_bytes);
          ptx::cp_async_bulk_commit_group();
          ptx::cp_async_bulk_wait_group(ptx::n32_t<0>());
        }
      }
      __syncthreads();
    }
  }
#else
  (void)dst;
  (void)src;
  (void)nbytes;
  (void)intra_tile_bytes;
  (void)inter_tile_bytes;
  (void)iteration_count;
#endif
}

void validate_common(
    const torch::Tensor& dst,
    const torch::Tensor& src,
    int64_t nbytes,
    int64_t num_ctas,
    int64_t threads,
    int64_t iteration_count) {
  TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  TORCH_CHECK(dst.scalar_type() == torch::kUInt8, "dst must have dtype uint8");
  TORCH_CHECK(src.scalar_type() == torch::kUInt8, "src must have dtype uint8");
  TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
  TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
  TORCH_CHECK(nbytes > 0, "nbytes must be positive");
  TORCH_CHECK(num_ctas > 0, "num_ctas must be positive");
  TORCH_CHECK(threads > 0, "threads must be positive");
  TORCH_CHECK(threads <= 1024, "threads must be <= 1024");
  TORCH_CHECK(iteration_count > 0, "iteration_count must be positive");
  TORCH_CHECK(dst.numel() >= nbytes, "dst tensor is smaller than nbytes");
  TORCH_CHECK(src.numel() >= nbytes, "src tensor is smaller than nbytes");
}

}  // namespace

void launch_lsu_copy(
    torch::Tensor dst,
    torch::Tensor src,
    int64_t nbytes,
    int64_t num_ctas,
    int64_t threads,
    int64_t iteration_count,
    int64_t executor_device) {
  validate_common(dst, src, nbytes, num_ctas, threads, iteration_count);
  c10::Device exec_device(
      c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(executor_device));
  c10::cuda::CUDAGuard device_guard(exec_device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(executor_device);
  lsu_copy_kernel<<<static_cast<unsigned int>(num_ctas),
                    static_cast<unsigned int>(threads),
                    0,
                    stream>>>(
      static_cast<unsigned char*>(dst.data_ptr()),
      static_cast<const unsigned char*>(src.data_ptr()),
      static_cast<size_t>(nbytes),
      iteration_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_tma_copy(
    torch::Tensor dst,
    torch::Tensor src,
    int64_t nbytes,
    int64_t num_ctas,
    int64_t threads,
    int64_t intra_tile_bytes,
    int64_t inter_tile_bytes,
    int64_t iteration_count,
    int64_t executor_device) {
  validate_common(dst, src, nbytes, num_ctas, threads, iteration_count);
  TORCH_CHECK((nbytes % 16) == 0, "TMA nbytes must be a multiple of 16");
  TORCH_CHECK(
      (intra_tile_bytes % 16) == 0,
      "TMA intra_tile_bytes must be a multiple of 16");
  TORCH_CHECK(
      (inter_tile_bytes % 16) == 0,
      "TMA inter_tile_bytes must be a multiple of 16");
  TORCH_CHECK(intra_tile_bytes > 0, "TMA intra_tile_bytes must be positive");
  TORCH_CHECK(inter_tile_bytes > 0, "TMA inter_tile_bytes must be positive");
  TORCH_CHECK(
      intra_tile_bytes >= inter_tile_bytes,
      "TMA intra_tile_bytes must be >= inter_tile_bytes");

  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst.data_ptr());
  uintptr_t src_addr = reinterpret_cast<uintptr_t>(src.data_ptr());
  TORCH_CHECK((dst_addr % 16) == 0, "TMA dst pointer must be 16-byte aligned");
  TORCH_CHECK((src_addr % 16) == 0, "TMA src pointer must be 16-byte aligned");

  c10::Device exec_device(
      c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(executor_device));
  c10::cuda::CUDAGuard device_guard(exec_device);

  int major = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, executor_device));
  TORCH_CHECK(major >= 9, "TMA mode requires compute capability 9.0 or newer");

  int max_smem = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_smem,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      executor_device));
  TORCH_CHECK(
      intra_tile_bytes <= max_smem,
      "TMA intra_tile_bytes exceeds device opt-in dynamic shared memory limit");

  C10_CUDA_CHECK(cudaFuncSetAttribute(
      tma_copy_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(intra_tile_bytes)));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(executor_device);
  tma_copy_kernel<<<static_cast<unsigned int>(num_ctas),
                    static_cast<unsigned int>(threads),
                    static_cast<size_t>(intra_tile_bytes),
                    stream>>>(
      static_cast<unsigned char*>(dst.data_ptr()),
      static_cast<const unsigned char*>(src.data_ptr()),
      static_cast<size_t>(nbytes),
      static_cast<size_t>(intra_tile_bytes),
      static_cast<size_t>(inter_tile_bytes),
      iteration_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_copy_ext = None


def get_copy_extension():
    global _copy_ext
    if _copy_ext is not None:
        return _copy_ext

    # The script targets Hopper. This also lets TMA mode compile on systems
    # where PyTorch cannot infer the arch before the first CUDA context exists.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

    _copy_ext = load_inline(
        name="nvlink_lsu_tma_copy_ext",
        cpp_sources=r"""
#include <torch/extension.h>

void launch_lsu_copy(
    torch::Tensor dst,
    torch::Tensor src,
    int64_t nbytes,
    int64_t num_ctas,
    int64_t threads,
    int64_t iteration_count,
    int64_t executor_device);

void launch_tma_copy(
    torch::Tensor dst,
    torch::Tensor src,
    int64_t nbytes,
    int64_t num_ctas,
    int64_t threads,
    int64_t intra_tile_bytes,
    int64_t inter_tile_bytes,
    int64_t iteration_count,
    int64_t executor_device);
""",
        cuda_sources=CUDA_SRC,
        functions=["launch_lsu_copy", "launch_tma_copy"],
        extra_cflags=["-std=c++17"],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr"],
        verbose=False,
    )
    return _copy_ext


def enqueue_iteration(method: str,
                      dst: torch.Tensor,
                      src: torch.Tensor,
                      total_nbytes: int,
                      num_ctas: int,
                      threads_per_cta: int,
                      tma_intra_tile_bytes: int,
                      tma_inter_tile_bytes: int,
                      iteration_count: int,
                      executor_device: int,
                      stream: torch.cuda.Stream) -> None:
    ext = get_copy_extension()
    with torch.cuda.stream(stream):
        if method == "lsu":
            ext.launch_lsu_copy(
                dst, src, total_nbytes, num_ctas, threads_per_cta,
                iteration_count, executor_device
            )
        elif method == "tma":
            ext.launch_tma_copy(
                dst, src, total_nbytes, num_ctas, threads_per_cta,
                tma_intra_tile_bytes, tma_inter_tile_bytes,
                iteration_count, executor_device
            )
        else:
            raise ValueError(f"Unknown method: {method}")


def enqueue_iterations(method: str,
                       dst: torch.Tensor,
                       src: torch.Tensor,
                       total_nbytes: int,
                       num_ctas: int,
                       threads_per_cta: int,
                       tma_intra_tile_bytes: int,
                       tma_inter_tile_bytes: int,
                       executor_device: int,
                       stream: torch.cuda.Stream,
                       iteration_count: int,
                       persistent_kernel: bool) -> None:
    if iteration_count <= 0:
        return
    if persistent_kernel:
        enqueue_iteration(
            method, dst, src, total_nbytes, num_ctas, threads_per_cta,
            tma_intra_tile_bytes, tma_inter_tile_bytes, iteration_count,
            executor_device, stream
        )
        return

    for _ in range(iteration_count):
        enqueue_iteration(
            method, dst, src, total_nbytes, num_ctas, threads_per_cta,
            tma_intra_tile_bytes, tma_inter_tile_bytes, 1, executor_device,
            stream
        )


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


def resolve_num_ctas(executor_device: int, requested_num_sms: int) -> Tuple[int, int]:
    sm_count = torch.cuda.get_device_properties(executor_device).multi_processor_count
    if requested_num_sms == 0:
        return sm_count, sm_count
    if requested_num_sms < 0:
        raise ValueError("--num-sms must be non-negative; use 0 for all SMs")
    if requested_num_sms > sm_count:
        raise ValueError(
            f"--num-sms={requested_num_sms} exceeds cuda:{executor_device} SM count {sm_count}"
        )
    return requested_num_sms, sm_count


def check_tma_constraints(total_nbytes: int,
                          tma_intra_tile_bytes: int,
                          tma_inter_tile_bytes: int,
                          executor: str,
                          inter_tile_was_set: bool) -> None:
    if total_nbytes % 16 != 0:
        raise ValueError("--method tma requires total bytes per iteration to be a multiple of 16")
    if tma_intra_tile_bytes <= 0:
        raise ValueError("--tma-tile-bytes must be positive")
    if tma_intra_tile_bytes % 16 != 0:
        raise ValueError("--tma-tile-bytes must be a multiple of 16")
    if tma_inter_tile_bytes <= 0:
        raise ValueError("--tma-inter-tile-bytes must be positive")
    if tma_inter_tile_bytes % 16 != 0:
        raise ValueError("--tma-inter-tile-bytes must be a multiple of 16")

    if inter_tile_was_set:
        if executor != "src":
            raise ValueError("--tma-inter-tile-bytes is only supported with --executor src")
        if tma_intra_tile_bytes <= tma_inter_tile_bytes:
            raise ValueError(
                "--tma-tile-bytes must be larger than --tma-inter-tile-bytes"
            )
        if tma_intra_tile_bytes % tma_inter_tile_bytes != 0:
            raise ValueError(
                "--tma-tile-bytes must be a multiple of --tma-inter-tile-bytes"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nbytes", "--copy-size", dest="nbytes",
                   type=parse_nbytes, default=parse_nbytes("1G"),
                   help="bytes per logical copy. Default: 1G")
    p.add_argument("--copies-per-iter", type=int, default=1,
                   help="number of contiguous logical copies per iteration. Default: 1")
    p.add_argument("--method", choices=["lsu", "tma"], default="lsu",
                   help="SM transfer method. lsu: normal global load/store kernel; "
                        "tma: Hopper TMA global->shared->global staging. Default: lsu")
    p.add_argument("--executor", choices=["src", "dst"], default="dst",
                   help="GPU whose SMs execute the copy kernel. "
                        "src: source-side push; dst: destination-side pull. Default: dst")
    p.add_argument("--num-sms", type=int, default=0,
                   help="number of CTAs to launch, used as an SM participation cap. "
                        "0 means all SMs on the executor GPU. Default: 0")
    p.add_argument("--threads-per-cta", type=int, default=256,
                   help="threads per CTA. Default: 256")
    p.add_argument("--tma-tile-bytes", "--tma-intra-tile-bytes",
                   dest="tma_intra_tile_bytes",
                   type=parse_nbytes, default=parse_nbytes("64K"),
                   help="intra-GPU global->shared staging tile size for --method tma. "
                        "Default: 64K")
    p.add_argument("--tma-inter-tile-bytes", type=parse_nbytes, default=None,
                   help="source-executor TMA shared->peer-global store chunk size. "
                        "Only supported with --method tma --executor src. "
                        "If omitted, uses --tma-tile-bytes.")
    p.add_argument("--persistent-kernel", action="store_true",
                   help="launch one kernel per benchmark phase and loop over copy iterations "
                        "inside the kernel. Warmup, if nonzero, is a separate warmup kernel.")
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
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.threads_per_cta <= 0 or args.threads_per_cta > 1024:
        raise ValueError("--threads-per-cta must be in [1, 1024]")

    src_dev, dst_dev = choose_pair(local_rank, world_size, args.mode)
    total_nbytes = args.nbytes * args.copies_per_iter
    tma_inter_tile_was_set = args.tma_inter_tile_bytes is not None
    tma_inter_tile_bytes = (
        args.tma_inter_tile_bytes
        if args.tma_inter_tile_bytes is not None
        else args.tma_intra_tile_bytes
    )
    if tma_inter_tile_was_set and args.method != "tma":
        raise ValueError("--tma-inter-tile-bytes is only supported with --method tma")
    if args.method == "tma":
        check_tma_constraints(
            total_nbytes,
            args.tma_intra_tile_bytes,
            tma_inter_tile_bytes,
            args.executor,
            tma_inter_tile_was_set,
        )

    executor_dev = src_dev if args.executor == "src" else dst_dev
    peer_dev = dst_dev if args.executor == "src" else src_dev

    # The kernel runs on executor_dev and must be able to access the peer tensor.
    # src executor: local load + remote store, a source-side push.
    # dst executor: remote load + local store, a destination-side pull.
    can_src_to_dst = cuda_can_access_peer(src_dev, dst_dev)
    can_dst_to_src = cuda_can_access_peer(dst_dev, src_dev)
    can_executor_to_peer = can_src_to_dst if args.executor == "src" else can_dst_to_src
    if not can_executor_to_peer:
        raise RuntimeError(
            f"Executor GPU {executor_dev} cannot access peer GPU {peer_dev}. "
            "Check CUDA_VISIBLE_DEVICES and nvidia-smi topo -m."
        )

    cuda_enable_peer_access(executor_dev, peer_dev)
    if args.executor == "src" and can_dst_to_src:
        cuda_enable_peer_access(dst_dev, src_dev)
    elif args.executor == "dst" and can_src_to_dst:
        cuda_enable_peer_access(src_dev, dst_dev)

    num_ctas, physical_sms = resolve_num_ctas(executor_dev, args.num_sms)

    torch.cuda.set_device(src_dev)
    src = torch.empty(total_nbytes, dtype=torch.uint8, device=f"cuda:{src_dev}")
    src.fill_((rank + 17) % 251)

    torch.cuda.set_device(dst_dev)
    dst = torch.empty(total_nbytes, dtype=torch.uint8, device=f"cuda:{dst_dev}")
    dst.fill_((rank + 18) % 251)

    torch.cuda.synchronize(src_dev)
    torch.cuda.synchronize(dst_dev)

    torch.cuda.set_device(executor_dev)
    stream = torch.cuda.Stream(device=executor_dev)

    # Compile before timed warmup so extension build time is never included.
    get_copy_extension()

    barrier()
    if args.sleep_before > 0:
        if rank == 0:
            print(f"Sleeping for {args.sleep_before:.1f}s before benchmark...", flush=True)
        time.sleep(args.sleep_before)
    barrier()

    cuda_set_device(executor_dev)
    enqueue_iterations(
        args.method, dst, src, total_nbytes, num_ctas, args.threads_per_cta,
        args.tma_intra_tile_bytes, tma_inter_tile_bytes, executor_dev, stream, args.warmup,
        args.persistent_kernel
    )
    cuda_stream_synchronize(stream)
    barrier()

    cuda_set_device(executor_dev)
    t0 = time.perf_counter()
    enqueue_iterations(
        args.method, dst, src, total_nbytes, num_ctas, args.threads_per_cta,
        args.tma_intra_tile_bytes, tma_inter_tile_bytes, executor_dev, stream, args.iters,
        args.persistent_kernel
    )
    cuda_stream_synchronize(stream)
    t1 = time.perf_counter()
    barrier()

    elapsed = t1 - t0
    local_gib = (total_nbytes * args.iters) / (1024**3)
    local_bw = local_gib / elapsed

    ref_device = torch.device(f"cuda:{local_rank}")
    max_elapsed = all_reduce_max_float(elapsed, ref_device)
    sum_gib = all_reduce_sum_float(local_gib, ref_device)
    agg_bw = sum_gib / max_elapsed

    ok = True
    if args.check:
        torch.cuda.set_device(dst_dev)
        copy_starts = torch.arange(
            args.copies_per_iter, dtype=torch.int64, device=f"cuda:{dst_dev}"
        ) * args.nbytes
        sample = dst[copy_starts].cpu()
        expected = (rank + 17) % 251
        ok = bool(torch.all(sample == expected).item())

    for r in range(world_size):
        barrier()
        if rank == r:
            tma_detail = (
                f" tma_intra_tile={fmt_bytes(args.tma_intra_tile_bytes)} "
                f"tma_inter_tile={fmt_bytes(tma_inter_tile_bytes)}"
                if args.method == "tma"
                else ""
            )
            print(
                f"rank={rank:02d} local_rank={local_rank:02d} "
                f"copy cuda:{src_dev} -> cuda:{dst_dev} "
                f"method={args.method} executor={args.executor}:cuda:{executor_dev} "
                f"persistent={int(args.persistent_kernel)} "
                f"ctas={num_ctas} physical_sms={physical_sms} "
                f"threads/cta={args.threads_per_cta}{tma_detail} "
                f"copies/iter={args.copies_per_iter} "
                f"bytes/copy={fmt_bytes(args.nbytes)} bytes/iter={fmt_bytes(total_nbytes)} "
                f"iters={args.iters} "
                f"elapsed={elapsed:.6f}s local_bw={local_bw:.2f} GiB/s "
                f"p2p={int(can_executor_to_peer)} check={'OK' if ok else 'FAIL'}",
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
            "\nProfiler expectation: Nsight Systems should show a CUDA kernel doing "
            f"{'ordinary SM global load/store copies' if args.method == 'lsu' else 'Hopper TMA bulk async copies'}, "
            f"executing on the {args.executor} GPU, not "
            "cudaMemcpyPeerAsync/cudaMemcpyBatchAsync copy-engine activity.",
            flush=True,
        )
        if args.persistent_kernel:
            print(
                "Persistent mode expectation: one timed transfer kernel contains "
                f"{args.iters} device-side copy iterations.",
                flush=True,
            )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
