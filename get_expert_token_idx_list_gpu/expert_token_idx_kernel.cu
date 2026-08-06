#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWarps = kBlockThreads / 32;
constexpr int kMaxExperts = 1024;

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ int device_gcd(int a, int b) {
  while (b != 0) {
    int r = a % b;
    a = b;
    b = r;
  }
  return a;
}

// Each row is an affine permutation prefix.  A stride coprime to K guarantees
// that the top-k expert IDs in a row are distinct without a rejection loop.
__global__ void generate_r_kernel(
    int32_t* __restrict__ r,
    int num_tokens,
    int top_k,
    int num_experts,
    uint64_t seed) {
  int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= num_tokens) {
    return;
  }

  uint64_t random0 = splitmix64(seed ^ static_cast<uint64_t>(token));
  uint64_t random1 = splitmix64(random0);
  int start = static_cast<int>(random0 % static_cast<uint64_t>(num_experts));
  int stride = 1;
  if (num_experts > 1) {
    stride = 1 + static_cast<int>(
        random1 % static_cast<uint64_t>(num_experts - 1));
    while (device_gcd(stride, num_experts) != 1) {
      ++stride;
      if (stride == num_experts) {
        stride = 1;
      }
    }
  }

  int64_t row = static_cast<int64_t>(token) * top_k;
  int expert = start;
  for (int route = 0; route < top_k; ++route) {
    r[row + route] = expert;
    expert += stride;
    if (expert >= num_experts) {
      expert -= num_experts;
    }
  }
}

// Pass 1: independently count each expert route and each distinct destination
// GPU in every 256-token chunk.  The latter is a token count, not a route
// count: two experts on the same GPU still produce one logical token position.
__global__ void count_chunks_kernel(
    const int32_t* __restrict__ r,
    int num_tokens,
    int top_k,
    int num_experts,
    int experts_per_gpu,
    int num_gpus,
    int32_t* __restrict__ chunk_counts) {
  extern __shared__ int32_t counts[];
  int num_bins = num_experts + num_gpus;
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    counts[bin] = 0;
  }
  __syncthreads();

  int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token < num_tokens) {
    int64_t row = static_cast<int64_t>(token) * top_k;
    for (int route = 0; route < top_k; ++route) {
      int expert = r[row + route];
      atomicAdd(&counts[expert], 1);

      int gpu = expert / experts_per_gpu;
      bool first_route_to_gpu = true;
      for (int previous = 0; previous < route; ++previous) {
        if (r[row + previous] / experts_per_gpu == gpu) {
          first_route_to_gpu = false;
          break;
        }
      }
      if (first_route_to_gpu) {
        atomicAdd(&counts[num_experts + gpu], 1);
      }
    }
  }
  __syncthreads();

  int32_t* output = chunk_counts +
      static_cast<int64_t>(blockIdx.x) * num_bins;
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    output[bin] = counts[bin];
  }
}

// Pass 2: scan chunks independently for every bin.  This small kernel also
// scans the expert totals to form the compact ragged-array offsets.
__global__ void scan_chunk_counts_kernel(
    const int32_t* __restrict__ chunk_counts,
    int32_t* __restrict__ chunk_prefixes,
    int32_t* __restrict__ expert_offsets,
    int num_chunks,
    int num_experts,
    int num_gpus) {
  int num_bins = num_experts + num_gpus;
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    int32_t running = 0;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
      int64_t index = static_cast<int64_t>(chunk) * num_bins + bin;
      chunk_prefixes[index] = running;
      running += chunk_counts[index];
    }
    if (bin < num_experts) {
      expert_offsets[bin + 1] = running;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t running = 0;
    expert_offsets[0] = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      int32_t count = expert_offsets[expert + 1];
      running += count;
      expert_offsets[expert + 1] = running;
    }
  }
}

template <int ITEMS_PER_THREAD>
__global__ __launch_bounds__(kBlockThreads) void scatter_chunks_kernel(
    const int32_t* __restrict__ r,
    int num_tokens,
    int num_experts,
    int experts_per_gpu,
    int num_gpus,
    int radix_end_bit,
    const int32_t* __restrict__ chunk_counts,
    const int32_t* __restrict__ chunk_prefixes,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ expert_token_indices) {
  using BlockSort = cub::BlockRadixSort<
      int32_t, kBlockThreads, ITEMS_PER_THREAD, int32_t>;
  __shared__ typename BlockSort::TempStorage sort_storage;

  // Dynamic shared memory: per-expert offsets inside this chunk followed by
  // destination-GPU counts for each warp.
  extern __shared__ int32_t dynamic_shared[];
  int32_t* local_expert_offsets = dynamic_shared;
  int32_t* warp_gpu_counts = local_expert_offsets + num_experts + 1;

  int token = blockIdx.x * blockDim.x + threadIdx.x;
  bool valid_token = token < num_tokens;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int32_t experts[ITEMS_PER_THREAD];
  int32_t local_indices[ITEMS_PER_THREAD];

#pragma unroll
  for (int route = 0; route < ITEMS_PER_THREAD; ++route) {
    experts[route] = valid_token
        ? r[static_cast<int64_t>(token) * ITEMS_PER_THREAD + route]
        : num_experts;  // Sentinel sorts after every real expert.
    local_indices[route] = 0;
  }

  // Count the tokens for every destination GPU in each warp.  Looping GPUs
  // (normally only 8 on an 8-GPU node) makes the rank stable by token number,
  // even when two rows list their routes in different orders.
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    bool routed_to_gpu = false;
#pragma unroll
    for (int route = 0; route < ITEMS_PER_THREAD; ++route) {
      routed_to_gpu |= valid_token &&
          (experts[route] / experts_per_gpu == gpu);
    }
    unsigned mask = __ballot_sync(0xffffffffu, routed_to_gpu);
    if (lane == 0) {
      warp_gpu_counts[gpu * kWarps + warp] = __popc(mask);
    }
  }
  __syncthreads();

  int num_bins = num_experts + num_gpus;
  int64_t chunk_row = static_cast<int64_t>(blockIdx.x) * num_bins;
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    bool routed_to_gpu = false;
#pragma unroll
    for (int route = 0; route < ITEMS_PER_THREAD; ++route) {
      routed_to_gpu |= valid_token &&
          (experts[route] / experts_per_gpu == gpu);
    }
    unsigned mask = __ballot_sync(0xffffffffu, routed_to_gpu);
    if (routed_to_gpu) {
      int32_t index = chunk_prefixes[chunk_row + num_experts + gpu];
      for (int previous_warp = 0; previous_warp < warp; ++previous_warp) {
        index += warp_gpu_counts[gpu * kWarps + previous_warp];
      }
      index += __popc(mask & ((1u << lane) - 1u));
#pragma unroll
      for (int route = 0; route < ITEMS_PER_THREAD; ++route) {
        if (experts[route] / experts_per_gpu == gpu) {
          local_indices[route] = index;
        }
      }
    }
  }

  // The stable block radix sort groups routes by expert while retaining token
  // order within every expert.  Since input is blocked, its original order is
  // (token, route), exactly the order of R.
  BlockSort(sort_storage).Sort(
      experts, local_indices, 0, radix_end_bit);
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t running = 0;
    local_expert_offsets[0] = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      running += chunk_counts[chunk_row + expert];
      local_expert_offsets[expert + 1] = running;
    }
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int expert = experts[item];
    if (expert < num_experts) {
      int sorted_position = threadIdx.x * ITEMS_PER_THREAD + item;
      int in_chunk_rank = sorted_position - local_expert_offsets[expert];
      int32_t destination = expert_offsets[expert] +
          chunk_prefixes[chunk_row + expert] + in_chunk_rank;
      expert_token_indices[destination] = local_indices[item];
    }
  }
}

// Correct stable fallback for uncommon top-k values.  One CTA per expert
// scans tokens in order.  It trades bandwidth for generality and keeps the
// optimized path focused on the usual MoE top-k values.
__global__ void scatter_by_expert_fallback_kernel(
    const int32_t* __restrict__ r,
    int num_tokens,
    int top_k,
    int experts_per_gpu,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ expert_token_indices) {
  __shared__ int32_t gpu_warp_counts[kWarps];
  __shared__ int32_t expert_warp_counts[kWarps];
  __shared__ int32_t gpu_base;
  __shared__ int32_t expert_base;

  int expert = blockIdx.x;
  int owner_gpu = expert / experts_per_gpu;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    gpu_base = 0;
    expert_base = 0;
  }
  __syncthreads();

  for (int tile = 0; tile < num_tokens; tile += blockDim.x) {
    int token = tile + threadIdx.x;
    bool routed_to_gpu = false;
    bool routed_to_expert = false;
    if (token < num_tokens) {
      int64_t row = static_cast<int64_t>(token) * top_k;
      for (int route = 0; route < top_k; ++route) {
        int route_expert = r[row + route];
        routed_to_gpu |= route_expert / experts_per_gpu == owner_gpu;
        routed_to_expert |= route_expert == expert;
      }
    }

    unsigned gpu_mask = __ballot_sync(0xffffffffu, routed_to_gpu);
    unsigned expert_mask = __ballot_sync(0xffffffffu, routed_to_expert);
    if (lane == 0) {
      gpu_warp_counts[warp] = __popc(gpu_mask);
      expert_warp_counts[warp] = __popc(expert_mask);
    }
    __syncthreads();

    int32_t gpu_rank = gpu_base;
    int32_t expert_rank = expert_base;
    for (int previous_warp = 0; previous_warp < warp; ++previous_warp) {
      gpu_rank += gpu_warp_counts[previous_warp];
      expert_rank += expert_warp_counts[previous_warp];
    }
    gpu_rank += __popc(gpu_mask & ((1u << lane) - 1u));
    expert_rank += __popc(expert_mask & ((1u << lane) - 1u));

    if (routed_to_expert) {
      expert_token_indices[expert_offsets[expert] + expert_rank] = gpu_rank;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int32_t gpu_tile_count = 0;
      int32_t expert_tile_count = 0;
      for (int w = 0; w < kWarps; ++w) {
        gpu_tile_count += gpu_warp_counts[w];
        expert_tile_count += expert_warp_counts[w];
      }
      gpu_base += gpu_tile_count;
      expert_base += expert_tile_count;
    }
    __syncthreads();
  }
}

void check_cuda_int32(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt32,
      name,
      " must have dtype torch.int32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(
      tensor.get_device() == reference.get_device(),
      name,
      " must be on the same CUDA device as R");
}

int radix_end_bit_for(int sentinel) {
  int bits = 0;
  do {
    ++bits;
    sentinel >>= 1;
  } while (sentinel != 0);
  return bits;
}

template <int TOP_K>
void launch_scatter_specialized(
    const int32_t* r,
    int num_tokens,
    int num_experts,
    int experts_per_gpu,
    int num_gpus,
    int num_chunks,
    int radix_end_bit,
    const int32_t* chunk_counts,
    const int32_t* chunk_prefixes,
    const int32_t* expert_offsets,
    int32_t* expert_token_indices,
    size_t dynamic_smem,
    cudaStream_t stream) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      scatter_chunks_kernel<TOP_K>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_smem)));
  scatter_chunks_kernel<TOP_K><<<
      num_chunks, kBlockThreads, dynamic_smem, stream>>>(
      r,
      num_tokens,
      num_experts,
      experts_per_gpu,
      num_gpus,
      radix_end_bit,
      chunk_counts,
      chunk_prefixes,
      expert_offsets,
      expert_token_indices);
}

}  // namespace

void generate_r_cuda(
    torch::Tensor r,
    int64_t num_experts_arg,
    uint64_t seed) {
  check_cuda_int32(r, "R");
  TORCH_CHECK(r.dim() == 2, "R must have shape [T, top_k]");
  TORCH_CHECK(r.size(0) > 0, "T must be positive");
  TORCH_CHECK(r.size(1) > 0, "top_k must be positive");
  TORCH_CHECK(
      num_experts_arg > 0 && num_experts_arg <= kMaxExperts,
      "num_experts must be in [1, ", kMaxExperts, "]");
  TORCH_CHECK(
      r.size(1) <= num_experts_arg,
      "top_k cannot exceed num_experts");
  TORCH_CHECK(
      r.size(0) <= std::numeric_limits<int32_t>::max(),
      "T must fit in int32");

  int device = r.get_device();
  c10::cuda::CUDAGuard guard(
      c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  int num_tokens = static_cast<int>(r.size(0));
  int top_k = static_cast<int>(r.size(1));
  int num_experts = static_cast<int>(num_experts_arg);
  int blocks = (num_tokens + kBlockThreads - 1) / kBlockThreads;
  generate_r_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      r.data_ptr<int32_t>(), num_tokens, top_k, num_experts, seed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void get_expert_token_idx_cuda(
    torch::Tensor r,
    int64_t num_experts_arg,
    int64_t experts_per_gpu_arg,
    torch::Tensor expert_token_indices,
    torch::Tensor expert_offsets,
    torch::Tensor chunk_counts,
    torch::Tensor chunk_prefixes) {
  check_cuda_int32(r, "R");
  check_cuda_int32(expert_token_indices, "expert_token_indices");
  check_cuda_int32(expert_offsets, "expert_offsets");
  check_cuda_int32(chunk_counts, "chunk_counts");
  check_cuda_int32(chunk_prefixes, "chunk_prefixes");
  TORCH_CHECK(r.dim() == 2, "R must have shape [T, top_k]");
  TORCH_CHECK(r.size(0) > 0 && r.size(1) > 0, "T and top_k must be positive");
  TORCH_CHECK(
      num_experts_arg > 0 && num_experts_arg <= kMaxExperts,
      "num_experts must be in [1, ", kMaxExperts, "]");
  TORCH_CHECK(
      experts_per_gpu_arg > 0 && num_experts_arg % experts_per_gpu_arg == 0,
      "experts_per_gpu must be positive and divide num_experts");
  TORCH_CHECK(r.size(1) <= num_experts_arg, "top_k cannot exceed num_experts");
  TORCH_CHECK(
      r.numel() <= std::numeric_limits<int32_t>::max(),
      "T * top_k must fit in int32");

  check_same_device(expert_token_indices, r, "expert_token_indices");
  check_same_device(expert_offsets, r, "expert_offsets");
  check_same_device(chunk_counts, r, "chunk_counts");
  check_same_device(chunk_prefixes, r, "chunk_prefixes");

  int num_tokens = static_cast<int>(r.size(0));
  int top_k = static_cast<int>(r.size(1));
  int num_experts = static_cast<int>(num_experts_arg);
  int experts_per_gpu = static_cast<int>(experts_per_gpu_arg);
  int num_gpus = num_experts / experts_per_gpu;
  int num_chunks = (num_tokens + kBlockThreads - 1) / kBlockThreads;
  int num_bins = num_experts + num_gpus;
  int64_t scratch_elements = static_cast<int64_t>(num_chunks) * num_bins;

  TORCH_CHECK(
      expert_token_indices.numel() >= r.numel(),
      "expert_token_indices needs at least T * top_k entries");
  TORCH_CHECK(
      expert_offsets.numel() >= num_experts + 1,
      "expert_offsets needs num_experts + 1 entries");
  TORCH_CHECK(
      chunk_counts.numel() >= scratch_elements,
      "chunk_counts is too small; expected at least ", scratch_elements);
  TORCH_CHECK(
      chunk_prefixes.numel() >= scratch_elements,
      "chunk_prefixes is too small; expected at least ", scratch_elements);

  int device = r.get_device();
  c10::cuda::CUDAGuard guard(
      c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

  size_t count_smem = static_cast<size_t>(num_bins) * sizeof(int32_t);
  count_chunks_kernel<<<num_chunks, kBlockThreads, count_smem, stream>>>(
      r.data_ptr<int32_t>(),
      num_tokens,
      top_k,
      num_experts,
      experts_per_gpu,
      num_gpus,
      chunk_counts.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  scan_chunk_counts_kernel<<<1, kBlockThreads, 0, stream>>>(
      chunk_counts.data_ptr<int32_t>(),
      chunk_prefixes.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      num_chunks,
      num_experts,
      num_gpus);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t scatter_smem = static_cast<size_t>(
      num_experts + 1 + num_gpus * kWarps) * sizeof(int32_t);
  int radix_end_bit = radix_end_bit_for(num_experts);
  const int32_t* input = r.data_ptr<int32_t>();
  const int32_t* counts = chunk_counts.data_ptr<int32_t>();
  const int32_t* prefixes = chunk_prefixes.data_ptr<int32_t>();
  const int32_t* offsets = expert_offsets.data_ptr<int32_t>();
  int32_t* output = expert_token_indices.data_ptr<int32_t>();

#define LAUNCH_TOP_K(TOP_K_VALUE)                                           \
  launch_scatter_specialized<TOP_K_VALUE>(                                 \
      input, num_tokens, num_experts, experts_per_gpu, num_gpus,            \
      num_chunks, radix_end_bit, counts, prefixes, offsets, output,         \
      scatter_smem, stream)

  switch (top_k) {
    case 1: LAUNCH_TOP_K(1); break;
    case 2: LAUNCH_TOP_K(2); break;
    case 4: LAUNCH_TOP_K(4); break;
    case 8: LAUNCH_TOP_K(8); break;
    case 16: LAUNCH_TOP_K(16); break;
    default:
      scatter_by_expert_fallback_kernel<<<
          num_experts, kBlockThreads, 0, stream>>>(
          input,
          num_tokens,
          top_k,
          experts_per_gpu,
          offsets,
          output);
      break;
  }
#undef LAUNCH_TOP_K
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
