#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace {

using block_barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr int kCopyLsu = 0;
constexpr int kCopyTma = 1;

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ uint8_t random_payload_byte(
    uint64_t seed,
    int64_t element,
    int64_t byte_offset) {
  uint64_t x = seed;
  x ^= static_cast<uint64_t>(element) * 0xd2b74407b1ce6e93ULL;
  x ^= static_cast<uint64_t>(byte_offset) * 0xca5a826395121157ULL;
  return static_cast<uint8_t>(splitmix64(x) & 0xffu);
}

__device__ __forceinline__ uint8_t random_7bit_key(
    uint64_t seed,
    int64_t element,
    uint32_t probability_threshold) {
  uint8_t value = 0;
  for (int bit = 0; bit < 7; ++bit) {
    uint64_t x = seed ^ 0xa0761d6478bd642fULL;
    x ^= static_cast<uint64_t>(element * 7 + bit) * 0xe7037ed1a0b428dbULL;
    uint32_t sample = static_cast<uint32_t>(splitmix64(x) >> 32);
    if (sample < probability_threshold || probability_threshold == 0xffffffffu) {
      value |= static_cast<uint8_t>(1u << bit);
    }
  }
  return value;
}

__global__ void init_payload_keys_kernel(
    uint8_t* __restrict__ payload,
    uint8_t* __restrict__ keys,
    int64_t num_elements,
    int64_t payload_size,
    uint32_t probability_threshold,
    uint64_t seed) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  int64_t payload_bytes = num_elements * payload_size;
  for (int64_t i = tid; i < payload_bytes; i += stride) {
    int64_t element = i / payload_size;
    int64_t byte_offset = i - element * payload_size;
    payload[i] = random_payload_byte(seed, element, byte_offset);
  }

  for (int64_t element = tid; element < num_elements; element += stride) {
    keys[element] = random_7bit_key(seed, element, probability_threshold);
  }
}

__device__ void copy_payload_tma(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    int64_t payload_size,
    int64_t tma_tile_bytes,
    block_barrier& bar,
    uint8_t* __restrict__ smem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  for (int64_t tile = 0; tile < payload_size; tile += tma_tile_bytes) {
    int64_t remaining = payload_size - tile;
    int64_t bytes = remaining < tma_tile_bytes ? remaining : tma_tile_bytes;
    int64_t tma_bytes = bytes & ~static_cast<int64_t>(0xf);

    if (tma_bytes > 0) {
      if (threadIdx.x == 0) {
        cuda::memcpy_async(
            smem,
            src + tile,
            cuda::aligned_size_t<16>(static_cast<size_t>(tma_bytes)),
            bar);
      }
      block_barrier::arrival_token token = bar.arrive();
      bar.wait(std::move(token));

      ptx::fence_proxy_async(ptx::space_shared);
      __syncthreads();

      if (threadIdx.x == 0) {
        ptx::cp_async_bulk(
            ptx::space_global,
            ptx::space_shared,
            dst + tile,
            smem,
            static_cast<size_t>(tma_bytes));
        ptx::cp_async_bulk_commit_group();
        ptx::cp_async_bulk_wait_group(ptx::n32_t<0>());
      }
      __syncthreads();
    }

    for (int64_t i = tma_bytes + threadIdx.x; i < bytes; i += blockDim.x) {
      dst[tile + i] = src[tile + i];
    }
    __syncthreads();
  }
#else
  for (int64_t i = threadIdx.x; i < payload_size; i += blockDim.x) {
    dst[i] = src[i];
  }
#endif
}

__global__ void reorder_payloads_lsu_kernel(
    const uint8_t* __restrict__ payload,
    uint8_t* __restrict__ reordered_payload,
    const int32_t* __restrict__ reordered_indices,
    int64_t num_elements,
    int64_t payload_size) {
  int64_t elems_per_cta =
      (num_elements + static_cast<int64_t>(gridDim.x) - 1) /
      static_cast<int64_t>(gridDim.x);
  int64_t src_begin = static_cast<int64_t>(blockIdx.x) * elems_per_cta;
  int64_t src_end = src_begin + elems_per_cta;
  src_end = src_end < num_elements ? src_end : num_elements;
  if (src_begin >= src_end) {
    return;
  }

  uintptr_t src_addr = reinterpret_cast<uintptr_t>(payload);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(reordered_payload);
  bool aligned16 =
      ((src_addr | dst_addr | static_cast<uintptr_t>(payload_size)) & 0xfu) == 0;

  if (aligned16) {
    const uint4* src16 = reinterpret_cast<const uint4*>(payload);
    uint4* dst16 = reinterpret_cast<uint4*>(reordered_payload);
    int64_t vecs_per_payload = payload_size / static_cast<int64_t>(sizeof(uint4));
    int64_t total_vecs = (src_end - src_begin) * vecs_per_payload;
    for (int64_t work = threadIdx.x; work < total_vecs; work += blockDim.x) {
      int64_t local_src = work / vecs_per_payload;
      int64_t vec_offset = work - local_src * vecs_per_payload;
      int64_t src = src_begin + local_src;
      int64_t dst = static_cast<int64_t>(reordered_indices[src]);
      dst16[dst * vecs_per_payload + vec_offset] =
          src16[src * vecs_per_payload + vec_offset];
    }
  } else {
    int64_t total_bytes = (src_end - src_begin) * payload_size;
    for (int64_t work = threadIdx.x; work < total_bytes; work += blockDim.x) {
      int64_t local_src = work / payload_size;
      int64_t byte_offset = work - local_src * payload_size;
      int64_t src = src_begin + local_src;
      int64_t dst = static_cast<int64_t>(reordered_indices[src]);
      reordered_payload[dst * payload_size + byte_offset] =
          payload[src * payload_size + byte_offset];
    }
  }
}

__global__ void reorder_payloads_tma_kernel(
    const uint8_t* __restrict__ payload,
    uint8_t* __restrict__ reordered_payload,
    const int32_t* __restrict__ reordered_indices,
    int64_t num_elements,
    int64_t payload_size,
    int64_t tma_tile_bytes) {
  extern __shared__ __align__(16) uint8_t smem[];

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  int64_t elems_per_cta =
      (num_elements + static_cast<int64_t>(gridDim.x) - 1) /
      static_cast<int64_t>(gridDim.x);
  int64_t src_begin = static_cast<int64_t>(blockIdx.x) * elems_per_cta;
  int64_t src_end = src_begin + elems_per_cta;
  src_end = src_end < num_elements ? src_end : num_elements;

  for (int64_t src = src_begin; src < src_end; ++src) {
    int64_t dst = static_cast<int64_t>(reordered_indices[src]);
    copy_payload_tma(
        reordered_payload + dst * payload_size,
        payload + src * payload_size,
        payload_size,
        tma_tile_bytes,
        bar,
        smem);
  }
}

void validate_u8_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name,
    int64_t min_numel) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, name, " must have dtype uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() >= min_numel, name, " is smaller than required");
}

void validate_i32_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name,
    int64_t min_numel) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must have dtype int32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() >= min_numel, name, " is smaller than required");
}

void validate_common(
    int64_t num_elements,
    int64_t payload_size,
    int64_t num_ctas,
    int64_t threads,
    int64_t device) {
  TORCH_CHECK(num_elements > 0, "num_elements must be positive");
  TORCH_CHECK(payload_size > 0, "payload_size must be positive");
  TORCH_CHECK(num_elements <= std::numeric_limits<int32_t>::max(),
              "num_elements must fit int32");
  TORCH_CHECK(num_ctas > 0, "num_ctas must be positive");
  TORCH_CHECK(threads > 0 && threads <= 1024, "threads must be in 1..1024");
  TORCH_CHECK(device >= 0, "device must be non-negative");
}

uint32_t probability_to_threshold(double p) {
  TORCH_CHECK(p >= 0.0 && p <= 1.0, "bit_probability must be in [0, 1]");
  double scaled = p * static_cast<double>(std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(scaled);
}

}  // namespace

void launch_kernel_a_cpu_hybrid(
    torch::Tensor payload,
    torch::Tensor keys,
    int64_t num_elements,
    int64_t payload_size,
    double bit_probability,
    uint64_t seed,
    int64_t num_ctas,
    int64_t threads,
    int64_t device) {
  validate_common(num_elements, payload_size, num_ctas, threads, device);
  validate_u8_cuda_tensor(payload, "payload", num_elements * payload_size);
  validate_u8_cuda_tensor(keys, "keys", num_elements);

  c10::Device exec_device(
      c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device));
  c10::cuda::CUDAGuard device_guard(exec_device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

  init_payload_keys_kernel<<<
      static_cast<unsigned int>(num_ctas),
      static_cast<unsigned int>(threads),
      0,
      stream>>>(
      static_cast<uint8_t*>(payload.data_ptr()),
      static_cast<uint8_t*>(keys.data_ptr()),
      num_elements,
      payload_size,
      probability_to_threshold(bit_probability),
      seed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_kernel_b_cpu_hybrid(
    torch::Tensor payload,
    torch::Tensor reordered_payload,
    torch::Tensor reordered_indices,
    int64_t num_elements,
    int64_t payload_size,
    int64_t num_ctas,
    int64_t threads,
    int64_t payload_copy_method,
    int64_t tma_tile_bytes,
    int64_t device) {
  validate_common(num_elements, payload_size, num_ctas, threads, device);
  validate_u8_cuda_tensor(payload, "payload", num_elements * payload_size);
  validate_u8_cuda_tensor(
      reordered_payload, "reordered_payload", num_elements * payload_size);
  validate_i32_cuda_tensor(reordered_indices, "reordered_indices", num_elements);

  c10::Device exec_device(
      c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device));
  c10::cuda::CUDAGuard device_guard(exec_device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

  if (payload_copy_method == kCopyLsu) {
    reorder_payloads_lsu_kernel<<<
        static_cast<unsigned int>(num_ctas),
        static_cast<unsigned int>(threads),
        0,
        stream>>>(
        static_cast<const uint8_t*>(payload.data_ptr()),
        static_cast<uint8_t*>(reordered_payload.data_ptr()),
        static_cast<const int32_t*>(reordered_indices.data_ptr()),
        num_elements,
        payload_size);
  } else if (payload_copy_method == kCopyTma) {
    TORCH_CHECK(tma_tile_bytes > 0, "tma_tile_bytes must be positive");
    TORCH_CHECK((tma_tile_bytes % 16) == 0,
                "tma_tile_bytes must be a multiple of 16");
    TORCH_CHECK((payload_size % 16) == 0,
                "TMA payload copy requires payload_size to be a multiple of 16");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16) == 0,
                "TMA payload pointer must be 16-byte aligned");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(reordered_payload.data_ptr()) % 16) == 0,
                "TMA reordered_payload pointer must be 16-byte aligned");

    int major = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device));
    TORCH_CHECK(major >= 9, "TMA payload copy requires Hopper / compute capability 9.0+");

    int max_smem = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    TORCH_CHECK(tma_tile_bytes <= max_smem,
                "tma_tile_bytes exceeds dynamic shared memory opt-in limit");

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        reorder_payloads_tma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(tma_tile_bytes)));

    reorder_payloads_tma_kernel<<<
        static_cast<unsigned int>(num_ctas),
        static_cast<unsigned int>(threads),
        static_cast<size_t>(tma_tile_bytes),
        stream>>>(
        static_cast<const uint8_t*>(payload.data_ptr()),
        static_cast<uint8_t*>(reordered_payload.data_ptr()),
        static_cast<const int32_t*>(reordered_indices.data_ptr()),
        num_elements,
        payload_size,
        tma_tile_bytes);
  } else {
    TORCH_CHECK(false, "payload_copy_method must be 0 (lsu) or 1 (tma)");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
