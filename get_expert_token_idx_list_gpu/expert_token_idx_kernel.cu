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
constexpr int kGpusPerNode = 8;
constexpr int kNumMasks = 1 << kGpusPerNode;

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

// Node-mask mode, pass 1: count expert routes and stable counting-sort bins.
// There is one bin for every (destination node, 8-bit destination-GPU mask).
__global__ void count_node_input_chunks_kernel(
    const int32_t* __restrict__ r,
    int num_tokens,
    int top_k,
    int num_experts,
    int experts_per_gpu,
    int experts_per_node,
    int num_nodes,
    int32_t* __restrict__ chunk_counts) {
  extern __shared__ int32_t counts[];
  int num_node_mask_bins = num_nodes * kNumMasks;
  int num_bins = num_experts + num_node_mask_bins;
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    counts[bin] = 0;
  }
  __syncthreads();

  int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token < num_tokens) {
    int64_t row = static_cast<int64_t>(token) * top_k;
    for (int route = 0; route < top_k; ++route) {
      atomicAdd(&counts[r[row + route]], 1);
    }

    for (int route = 0; route < top_k; ++route) {
      int expert = r[row + route];
      int node = expert / experts_per_node;
      bool first_route_to_node = true;
      for (int previous = 0; previous < route; ++previous) {
        if (r[row + previous] / experts_per_node == node) {
          first_route_to_node = false;
          break;
        }
      }
      if (!first_route_to_node) {
        continue;
      }

      int mask = 0;
      for (int other = route; other < top_k; ++other) {
        int other_expert = r[row + other];
        if (other_expert / experts_per_node == node) {
          int local_gpu = (other_expert / experts_per_gpu) % kGpusPerNode;
          mask |= 1 << local_gpu;
        }
      }
      atomicAdd(
          &counts[num_experts + node * kNumMasks + mask], 1);
    }
  }
  __syncthreads();

  int32_t* output = chunk_counts +
      static_cast<int64_t>(blockIdx.x) * num_bins;
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    output[bin] = counts[bin];
  }
}

// Node-mask mode, pass 2: scan input chunks, form expert offsets, and form
// absolute output offsets for every node/mask bin in descending-mask order.
__global__ void scan_node_input_counts_kernel(
    const int32_t* __restrict__ chunk_counts,
    int32_t* __restrict__ chunk_prefixes,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ node_offsets,
    int32_t* __restrict__ node_mask_offsets,
    int num_chunks,
    int num_experts,
    int num_nodes) {
  int num_node_mask_bins = num_nodes * kNumMasks;
  int num_bins = num_experts + num_node_mask_bins;
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    int32_t running = 0;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
      int64_t index = static_cast<int64_t>(chunk) * num_bins + bin;
      chunk_prefixes[index] = running;
      running += chunk_counts[index];
    }
    if (bin < num_experts) {
      expert_offsets[bin + 1] = running;
    } else {
      node_mask_offsets[bin - num_experts] = running;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t expert_running = 0;
    expert_offsets[0] = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      int32_t count = expert_offsets[expert + 1];
      expert_running += count;
      expert_offsets[expert + 1] = expert_running;
    }

    int32_t token_running = 0;
    node_offsets[0] = 0;
    for (int node = 0; node < num_nodes; ++node) {
      int base = node * kNumMasks;
      for (int mask = kNumMasks - 1; mask >= 1; --mask) {
        int32_t count = node_mask_offsets[base + mask];
        node_mask_offsets[base + mask] = token_running;
        token_running += count;
      }
      node_mask_offsets[base] = token_running;
      node_offsets[node + 1] = token_running;
    }
  }
}

// Stable counting-sort scatter for common top-k values.  The radix sort is
// block-local; global stability comes from the per-input-chunk prefixes.
template <int ITEMS_PER_THREAD>
__global__ __launch_bounds__(kBlockThreads) void build_node_token_indices_kernel(
    const int32_t* __restrict__ r,
    int num_tokens,
    int num_experts,
    int experts_per_gpu,
    int experts_per_node,
    int num_nodes,
    int radix_end_bit,
    const int32_t* __restrict__ chunk_prefixes,
    const int32_t* __restrict__ node_mask_offsets,
    int32_t* __restrict__ node_token_indices) {
  using BlockSort = cub::BlockRadixSort<
      int32_t, kBlockThreads, ITEMS_PER_THREAD, int32_t>;
  union SharedStorage {
    typename BlockSort::TempStorage sort;
    int32_t sorted_keys[kBlockThreads * ITEMS_PER_THREAD];
  };
  __shared__ SharedStorage storage;

  int token = blockIdx.x * blockDim.x + threadIdx.x;
  bool valid_token = token < num_tokens;
  int sentinel = num_nodes * kNumMasks;
  int32_t keys[ITEMS_PER_THREAD];
  int32_t values[ITEMS_PER_THREAD];

#pragma unroll
  for (int route = 0; route < ITEMS_PER_THREAD; ++route) {
    keys[route] = sentinel;
    values[route] = token;
    if (!valid_token) {
      continue;
    }

    int64_t row = static_cast<int64_t>(token) * ITEMS_PER_THREAD;
    int expert = r[row + route];
    int node = expert / experts_per_node;
    bool first_route_to_node = true;
#pragma unroll
    for (int previous = 0; previous < route; ++previous) {
      first_route_to_node &= r[row + previous] / experts_per_node != node;
    }
    if (!first_route_to_node) {
      continue;
    }

    int mask = 0;
#pragma unroll
    for (int other = 0; other < ITEMS_PER_THREAD; ++other) {
      int other_expert = r[row + other];
      if (other_expert / experts_per_node == node) {
        int local_gpu = (other_expert / experts_per_gpu) % kGpusPerNode;
        mask |= 1 << local_gpu;
      }
    }
    // Ascending encoded keys mean node-major, descending-mask order.
    keys[route] = node * kNumMasks + (kNumMasks - 1 - mask);
  }

  BlockSort(storage.sort).Sort(keys, values, 0, radix_end_bit);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int position = threadIdx.x * ITEMS_PER_THREAD + item;
    storage.sorted_keys[position] = keys[item];
  }
  __syncthreads();

  int num_bins = num_experts + num_nodes * kNumMasks;
  int64_t chunk_row = static_cast<int64_t>(blockIdx.x) * num_bins;
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int key = keys[item];
    if (key == sentinel) {
      continue;
    }

    int low = 0;
    int high = kBlockThreads * ITEMS_PER_THREAD;
    while (low < high) {
      int middle = (low + high) >> 1;
      if (storage.sorted_keys[middle] < key) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    int position = threadIdx.x * ITEMS_PER_THREAD + item;
    int in_chunk_rank = position - low;
    int node = key / kNumMasks;
    int mask = kNumMasks - 1 - key % kNumMasks;
    int group = node * kNumMasks + mask;
    int32_t destination = node_mask_offsets[group] +
        chunk_prefixes[chunk_row + num_experts + group] + in_chunk_rank;
    node_token_indices[destination] = values[item];
  }
}

// General-top-k x3 fallback: each CTA owns one nonzero node/mask bin and scans
// tokens in order.  Common MoE top-k values take the radix-sort path above.
__global__ void build_node_token_indices_fallback_kernel(
    const int32_t* __restrict__ r,
    int num_tokens,
    int top_k,
    int experts_per_gpu,
    int experts_per_node,
    const int32_t* __restrict__ node_mask_offsets,
    int32_t* __restrict__ node_token_indices) {
  __shared__ int32_t warp_counts[kWarps];
  __shared__ int32_t group_base;
  int node = blockIdx.x / (kNumMasks - 1);
  int mask = kNumMasks - 1 - blockIdx.x % (kNumMasks - 1);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    group_base = 0;
  }
  __syncthreads();

  for (int tile = 0; tile < num_tokens; tile += blockDim.x) {
    int token = tile + threadIdx.x;
    int token_mask = 0;
    if (token < num_tokens) {
      int64_t row = static_cast<int64_t>(token) * top_k;
      for (int route = 0; route < top_k; ++route) {
        int expert = r[row + route];
        if (expert / experts_per_node == node) {
          int local_gpu = (expert / experts_per_gpu) % kGpusPerNode;
          token_mask |= 1 << local_gpu;
        }
      }
    }
    bool selected = token_mask == mask;
    unsigned selected_mask = __ballot_sync(0xffffffffu, selected);
    if (lane == 0) {
      warp_counts[warp] = __popc(selected_mask);
    }
    __syncthreads();

    int32_t rank = group_base;
    for (int previous_warp = 0; previous_warp < warp; ++previous_warp) {
      rank += warp_counts[previous_warp];
    }
    rank += __popc(selected_mask & ((1u << lane) - 1u));
    if (selected) {
      int group = node * kNumMasks + mask;
      node_token_indices[node_mask_offsets[group] + rank] = token;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int w = 0; w < kWarps; ++w) {
        group_base += warp_counts[w];
      }
    }
    __syncthreads();
  }
}

// Count experts and destination GPUs in chunks of the reordered per-node x3
// lists.  These prefixes allow all x3 chunks to be scattered in parallel.
__global__ void count_reordered_node_chunks_kernel(
    const int32_t* __restrict__ r,
    int top_k,
    int experts_per_gpu,
    int experts_per_node,
    int num_chunks_per_node,
    const int32_t* __restrict__ node_token_indices,
    const int32_t* __restrict__ node_offsets,
    int32_t* __restrict__ chunk_counts) {
  extern __shared__ int32_t counts[];
  int num_bins = experts_per_node + kGpusPerNode;
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    counts[bin] = 0;
  }
  __syncthreads();

  int node = blockIdx.y;
  int begin = node_offsets[node];
  int end = node_offsets[node + 1];
  int position = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (position < end) {
    int token = node_token_indices[position];
    int mask = 0;
    int64_t row = static_cast<int64_t>(token) * top_k;
    for (int route = 0; route < top_k; ++route) {
      int expert = r[row + route];
      if (expert / experts_per_node == node) {
        int local_expert = expert - node * experts_per_node;
        atomicAdd(&counts[local_expert], 1);
        int local_gpu = local_expert / experts_per_gpu;
        mask |= 1 << local_gpu;
      }
    }
    for (int gpu = 0; gpu < kGpusPerNode; ++gpu) {
      if (mask & (1 << gpu)) {
        atomicAdd(&counts[experts_per_node + gpu], 1);
      }
    }
  }
  __syncthreads();

  int64_t chunk_index =
      static_cast<int64_t>(node) * num_chunks_per_node + blockIdx.x;
  int32_t* output = chunk_counts + chunk_index * num_bins;
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    output[bin] = counts[bin];
  }
}

__global__ void scan_reordered_node_counts_kernel(
    const int32_t* __restrict__ chunk_counts,
    int32_t* __restrict__ chunk_prefixes,
    int num_chunks_per_node,
    int experts_per_node,
    int num_nodes) {
  int num_bins = experts_per_node + kGpusPerNode;
  int total_bins = num_nodes * num_bins;
  for (int global_bin = threadIdx.x;
       global_bin < total_bins;
       global_bin += blockDim.x) {
    int node = global_bin / num_bins;
    int bin = global_bin % num_bins;
    int32_t running = 0;
    for (int chunk = 0; chunk < num_chunks_per_node; ++chunk) {
      int64_t index =
          (static_cast<int64_t>(node) * num_chunks_per_node + chunk) *
              num_bins + bin;
      chunk_prefixes[index] = running;
      running += chunk_counts[index];
    }
  }
}

template <int ITEMS_PER_THREAD>
__global__ __launch_bounds__(kBlockThreads) void scatter_reordered_chunks_kernel(
    const int32_t* __restrict__ r,
    int top_k,
    int experts_per_gpu,
    int experts_per_node,
    int num_chunks_per_node,
    int radix_end_bit,
    const int32_t* __restrict__ node_token_indices,
    const int32_t* __restrict__ node_offsets,
    const int32_t* __restrict__ chunk_counts,
    const int32_t* __restrict__ chunk_prefixes,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ expert_token_indices) {
  using BlockSort = cub::BlockRadixSort<
      int32_t, kBlockThreads, ITEMS_PER_THREAD, int32_t>;
  __shared__ typename BlockSort::TempStorage sort_storage;
  extern __shared__ int32_t dynamic_shared[];
  int32_t* local_expert_offsets = dynamic_shared;
  int32_t* warp_gpu_counts = local_expert_offsets + experts_per_node + 1;

  int node = blockIdx.y;
  int begin = node_offsets[node];
  int end = node_offsets[node + 1];
  int position = begin + blockIdx.x * blockDim.x + threadIdx.x;
  bool valid_token = position < end;
  int token = valid_token ? node_token_indices[position] : 0;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int32_t experts[ITEMS_PER_THREAD];
  int32_t local_indices[ITEMS_PER_THREAD];

#pragma unroll
  for (int route = 0; route < ITEMS_PER_THREAD; ++route) {
    int expert = valid_token
        ? r[static_cast<int64_t>(token) * ITEMS_PER_THREAD + route]
        : -1;
    experts[route] = expert >= node * experts_per_node &&
            expert < (node + 1) * experts_per_node
        ? expert - node * experts_per_node
        : experts_per_node;
    local_indices[route] = 0;
  }

  int num_bins = experts_per_node + kGpusPerNode;
  int64_t chunk_row =
      (static_cast<int64_t>(node) * num_chunks_per_node + blockIdx.x) *
      num_bins;
  for (int gpu = 0; gpu < kGpusPerNode; ++gpu) {
    bool routed_to_gpu = false;
#pragma unroll
    for (int route = 0; route < ITEMS_PER_THREAD; ++route) {
      routed_to_gpu |= experts[route] < experts_per_node &&
          experts[route] / experts_per_gpu == gpu;
    }
    unsigned mask = __ballot_sync(0xffffffffu, routed_to_gpu);
    if (lane == 0) {
      warp_gpu_counts[gpu * kWarps + warp] = __popc(mask);
    }
  }
  __syncthreads();

  for (int gpu = 0; gpu < kGpusPerNode; ++gpu) {
    bool routed_to_gpu = false;
#pragma unroll
    for (int route = 0; route < ITEMS_PER_THREAD; ++route) {
      routed_to_gpu |= experts[route] < experts_per_node &&
          experts[route] / experts_per_gpu == gpu;
    }
    unsigned mask = __ballot_sync(0xffffffffu, routed_to_gpu);
    if (routed_to_gpu) {
      int32_t index = chunk_prefixes[chunk_row + experts_per_node + gpu];
      for (int previous_warp = 0; previous_warp < warp; ++previous_warp) {
        index += warp_gpu_counts[gpu * kWarps + previous_warp];
      }
      index += __popc(mask & ((1u << lane) - 1u));
#pragma unroll
      for (int route = 0; route < ITEMS_PER_THREAD; ++route) {
        if (experts[route] < experts_per_node &&
            experts[route] / experts_per_gpu == gpu) {
          local_indices[route] = index;
        }
      }
    }
  }

  BlockSort(sort_storage).Sort(
      experts, local_indices, 0, radix_end_bit);
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t running = 0;
    local_expert_offsets[0] = 0;
    for (int expert = 0; expert < experts_per_node; ++expert) {
      running += chunk_counts[chunk_row + expert];
      local_expert_offsets[expert + 1] = running;
    }
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int local_expert = experts[item];
    if (local_expert < experts_per_node) {
      int sorted_position = threadIdx.x * ITEMS_PER_THREAD + item;
      int in_chunk_rank =
          sorted_position - local_expert_offsets[local_expert];
      int global_expert = node * experts_per_node + local_expert;
      int32_t destination = expert_offsets[global_expert] +
          chunk_prefixes[chunk_row + local_expert] + in_chunk_rank;
      expert_token_indices[destination] = local_indices[item];
    }
  }
}

// General-top-k expert-list fallback.  One CTA per expert scans that expert's
// already reordered node segment, preserving x3 order exactly.
__global__ void scatter_reordered_by_expert_fallback_kernel(
    const int32_t* __restrict__ r,
    int top_k,
    int experts_per_gpu,
    int experts_per_node,
    const int32_t* __restrict__ node_token_indices,
    const int32_t* __restrict__ node_offsets,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ expert_token_indices) {
  __shared__ int32_t gpu_warp_counts[kWarps];
  __shared__ int32_t expert_warp_counts[kWarps];
  __shared__ int32_t gpu_base;
  __shared__ int32_t expert_base;

  int expert = blockIdx.x;
  int node = expert / experts_per_node;
  int owner_gpu = expert / experts_per_gpu;
  int begin = node_offsets[node];
  int end = node_offsets[node + 1];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    gpu_base = 0;
    expert_base = 0;
  }
  __syncthreads();

  for (int tile = begin; tile < end; tile += blockDim.x) {
    int position = tile + threadIdx.x;
    bool routed_to_gpu = false;
    bool routed_to_expert = false;
    if (position < end) {
      int token = node_token_indices[position];
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
      for (int w = 0; w < kWarps; ++w) {
        gpu_base += gpu_warp_counts[w];
        expert_base += expert_warp_counts[w];
      }
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

template <int TOP_K>
void launch_build_node_token_indices_specialized(
    const int32_t* r,
    int num_tokens,
    int num_experts,
    int experts_per_gpu,
    int experts_per_node,
    int num_nodes,
    int num_chunks,
    int radix_end_bit,
    const int32_t* chunk_prefixes,
    const int32_t* node_mask_offsets,
    int32_t* node_token_indices,
    cudaStream_t stream) {
  build_node_token_indices_kernel<TOP_K><<<
      num_chunks, kBlockThreads, 0, stream>>>(
      r,
      num_tokens,
      num_experts,
      experts_per_gpu,
      experts_per_node,
      num_nodes,
      radix_end_bit,
      chunk_prefixes,
      node_mask_offsets,
      node_token_indices);
}

template <int TOP_K>
void launch_scatter_reordered_specialized(
    const int32_t* r,
    int top_k,
    int experts_per_gpu,
    int experts_per_node,
    int num_chunks_per_node,
    int num_nodes,
    int radix_end_bit,
    const int32_t* node_token_indices,
    const int32_t* node_offsets,
    const int32_t* chunk_counts,
    const int32_t* chunk_prefixes,
    const int32_t* expert_offsets,
    int32_t* expert_token_indices,
    size_t dynamic_smem,
    cudaStream_t stream) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      scatter_reordered_chunks_kernel<TOP_K>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_smem)));
  dim3 grid(num_chunks_per_node, num_nodes);
  scatter_reordered_chunks_kernel<TOP_K><<<
      grid, kBlockThreads, dynamic_smem, stream>>>(
      r,
      top_k,
      experts_per_gpu,
      experts_per_node,
      num_chunks_per_node,
      radix_end_bit,
      node_token_indices,
      node_offsets,
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

void get_expert_token_idx_node_mask_cuda(
    torch::Tensor r,
    int64_t num_experts_arg,
    int64_t experts_per_gpu_arg,
    torch::Tensor expert_token_indices,
    torch::Tensor expert_offsets,
    torch::Tensor node_token_indices,
    torch::Tensor node_offsets,
    torch::Tensor input_chunk_counts,
    torch::Tensor input_chunk_prefixes,
    torch::Tensor node_mask_offsets,
    torch::Tensor reordered_chunk_counts,
    torch::Tensor reordered_chunk_prefixes) {
  check_cuda_int32(r, "R");
  check_cuda_int32(expert_token_indices, "expert_token_indices");
  check_cuda_int32(expert_offsets, "expert_offsets");
  check_cuda_int32(node_token_indices, "node_token_indices");
  check_cuda_int32(node_offsets, "node_offsets");
  check_cuda_int32(input_chunk_counts, "input_chunk_counts");
  check_cuda_int32(input_chunk_prefixes, "input_chunk_prefixes");
  check_cuda_int32(node_mask_offsets, "node_mask_offsets");
  check_cuda_int32(reordered_chunk_counts, "reordered_chunk_counts");
  check_cuda_int32(reordered_chunk_prefixes, "reordered_chunk_prefixes");
  TORCH_CHECK(r.dim() == 2, "R must have shape [T, top_k]");
  TORCH_CHECK(r.size(0) > 0 && r.size(1) > 0, "T and top_k must be positive");
  TORCH_CHECK(
      num_experts_arg > 0 && num_experts_arg <= kMaxExperts,
      "num_experts must be in [1, ", kMaxExperts, "]");
  TORCH_CHECK(
      experts_per_gpu_arg > 0 && num_experts_arg % experts_per_gpu_arg == 0,
      "experts_per_gpu must be positive and divide num_experts");
  TORCH_CHECK(
      num_experts_arg % (experts_per_gpu_arg * kGpusPerNode) == 0,
      "node-mask mode requires exactly ", kGpusPerNode,
      " GPUs per node and a whole number of nodes");
  TORCH_CHECK(r.size(1) <= num_experts_arg, "top_k cannot exceed num_experts");
  TORCH_CHECK(
      r.numel() <= std::numeric_limits<int32_t>::max(),
      "T * top_k must fit in int32");

  check_same_device(expert_token_indices, r, "expert_token_indices");
  check_same_device(expert_offsets, r, "expert_offsets");
  check_same_device(node_token_indices, r, "node_token_indices");
  check_same_device(node_offsets, r, "node_offsets");
  check_same_device(input_chunk_counts, r, "input_chunk_counts");
  check_same_device(input_chunk_prefixes, r, "input_chunk_prefixes");
  check_same_device(node_mask_offsets, r, "node_mask_offsets");
  check_same_device(reordered_chunk_counts, r, "reordered_chunk_counts");
  check_same_device(reordered_chunk_prefixes, r, "reordered_chunk_prefixes");

  int num_tokens = static_cast<int>(r.size(0));
  int top_k = static_cast<int>(r.size(1));
  int num_experts = static_cast<int>(num_experts_arg);
  int experts_per_gpu = static_cast<int>(experts_per_gpu_arg);
  int experts_per_node = experts_per_gpu * kGpusPerNode;
  int num_nodes = num_experts / experts_per_node;
  int num_input_chunks = (num_tokens + kBlockThreads - 1) / kBlockThreads;
  int num_chunks_per_node = num_input_chunks;
  int input_num_bins = num_experts + num_nodes * kNumMasks;
  int reordered_num_bins = experts_per_node + kGpusPerNode;
  int64_t input_scratch_elements =
      static_cast<int64_t>(num_input_chunks) * input_num_bins;
  int64_t reordered_scratch_elements =
      static_cast<int64_t>(num_nodes) * num_chunks_per_node *
      reordered_num_bins;
  int64_t max_node_tokens = static_cast<int64_t>(num_tokens) *
      std::min(top_k, num_nodes);

  TORCH_CHECK(
      expert_token_indices.numel() >= r.numel(),
      "expert_token_indices needs at least T * top_k entries");
  TORCH_CHECK(
      expert_offsets.numel() >= num_experts + 1,
      "expert_offsets needs num_experts + 1 entries");
  TORCH_CHECK(
      node_token_indices.numel() >= max_node_tokens,
      "node_token_indices needs at least T * min(top_k, num_nodes) entries");
  TORCH_CHECK(
      node_offsets.numel() >= num_nodes + 1,
      "node_offsets needs num_nodes + 1 entries");
  TORCH_CHECK(
      input_chunk_counts.numel() >= input_scratch_elements,
      "input_chunk_counts is too small; expected at least ",
      input_scratch_elements);
  TORCH_CHECK(
      input_chunk_prefixes.numel() >= input_scratch_elements,
      "input_chunk_prefixes is too small; expected at least ",
      input_scratch_elements);
  TORCH_CHECK(
      node_mask_offsets.numel() >= num_nodes * kNumMasks,
      "node_mask_offsets needs num_nodes * 256 entries");
  TORCH_CHECK(
      reordered_chunk_counts.numel() >= reordered_scratch_elements,
      "reordered_chunk_counts is too small; expected at least ",
      reordered_scratch_elements);
  TORCH_CHECK(
      reordered_chunk_prefixes.numel() >= reordered_scratch_elements,
      "reordered_chunk_prefixes is too small; expected at least ",
      reordered_scratch_elements);

  int device = r.get_device();
  c10::cuda::CUDAGuard guard(
      c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

  size_t input_count_smem =
      static_cast<size_t>(input_num_bins) * sizeof(int32_t);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      count_node_input_chunks_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(input_count_smem)));
  count_node_input_chunks_kernel<<<
      num_input_chunks, kBlockThreads, input_count_smem, stream>>>(
      r.data_ptr<int32_t>(),
      num_tokens,
      top_k,
      num_experts,
      experts_per_gpu,
      experts_per_node,
      num_nodes,
      input_chunk_counts.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  scan_node_input_counts_kernel<<<1, kBlockThreads, 0, stream>>>(
      input_chunk_counts.data_ptr<int32_t>(),
      input_chunk_prefixes.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      node_offsets.data_ptr<int32_t>(),
      node_mask_offsets.data_ptr<int32_t>(),
      num_input_chunks,
      num_experts,
      num_nodes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int node_radix_end_bit = radix_end_bit_for(num_nodes * kNumMasks);
  const int32_t* input = r.data_ptr<int32_t>();
  const int32_t* input_prefixes = input_chunk_prefixes.data_ptr<int32_t>();
  const int32_t* mask_offsets = node_mask_offsets.data_ptr<int32_t>();
  int32_t* x3 = node_token_indices.data_ptr<int32_t>();

#define LAUNCH_NODE_BUILD(TOP_K_VALUE)                                      \
  launch_build_node_token_indices_specialized<TOP_K_VALUE>(                 \
      input, num_tokens, num_experts, experts_per_gpu, experts_per_node,     \
      num_nodes, num_input_chunks, node_radix_end_bit, input_prefixes,       \
      mask_offsets, x3, stream)

  bool specialized_top_k = true;
  switch (top_k) {
    case 1: LAUNCH_NODE_BUILD(1); break;
    case 2: LAUNCH_NODE_BUILD(2); break;
    case 4: LAUNCH_NODE_BUILD(4); break;
    case 8: LAUNCH_NODE_BUILD(8); break;
    case 16: LAUNCH_NODE_BUILD(16); break;
    default:
      specialized_top_k = false;
      build_node_token_indices_fallback_kernel<<<
          num_nodes * (kNumMasks - 1), kBlockThreads, 0, stream>>>(
          input,
          num_tokens,
          top_k,
          experts_per_gpu,
          experts_per_node,
          mask_offsets,
          x3);
      break;
  }
#undef LAUNCH_NODE_BUILD
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (!specialized_top_k) {
    scatter_reordered_by_expert_fallback_kernel<<<
        num_experts, kBlockThreads, 0, stream>>>(
        input,
        top_k,
        experts_per_gpu,
        experts_per_node,
        x3,
        node_offsets.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(),
        expert_token_indices.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  size_t reordered_count_smem =
      static_cast<size_t>(reordered_num_bins) * sizeof(int32_t);
  dim3 reordered_grid(num_chunks_per_node, num_nodes);
  count_reordered_node_chunks_kernel<<<
      reordered_grid, kBlockThreads, reordered_count_smem, stream>>>(
      input,
      top_k,
      experts_per_gpu,
      experts_per_node,
      num_chunks_per_node,
      x3,
      node_offsets.data_ptr<int32_t>(),
      reordered_chunk_counts.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  scan_reordered_node_counts_kernel<<<1, kBlockThreads, 0, stream>>>(
      reordered_chunk_counts.data_ptr<int32_t>(),
      reordered_chunk_prefixes.data_ptr<int32_t>(),
      num_chunks_per_node,
      experts_per_node,
      num_nodes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int expert_radix_end_bit = radix_end_bit_for(experts_per_node);
  size_t scatter_smem = static_cast<size_t>(
      experts_per_node + 1 + kGpusPerNode * kWarps) * sizeof(int32_t);
  const int32_t* reordered_counts =
      reordered_chunk_counts.data_ptr<int32_t>();
  const int32_t* reordered_prefixes =
      reordered_chunk_prefixes.data_ptr<int32_t>();
  const int32_t* output_offsets = expert_offsets.data_ptr<int32_t>();
  int32_t* output = expert_token_indices.data_ptr<int32_t>();

#define LAUNCH_NODE_SCATTER(TOP_K_VALUE)                                    \
  launch_scatter_reordered_specialized<TOP_K_VALUE>(                        \
      input, top_k, experts_per_gpu, experts_per_node,                      \
      num_chunks_per_node, num_nodes, expert_radix_end_bit, x3,             \
      node_offsets.data_ptr<int32_t>(), reordered_counts,                    \
      reordered_prefixes, output_offsets, output, scatter_smem, stream)

  switch (top_k) {
    case 1: LAUNCH_NODE_SCATTER(1); break;
    case 2: LAUNCH_NODE_SCATTER(2); break;
    case 4: LAUNCH_NODE_SCATTER(4); break;
    case 8: LAUNCH_NODE_SCATTER(8); break;
    case 16: LAUNCH_NODE_SCATTER(16); break;
  }
#undef LAUNCH_NODE_SCATTER
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
