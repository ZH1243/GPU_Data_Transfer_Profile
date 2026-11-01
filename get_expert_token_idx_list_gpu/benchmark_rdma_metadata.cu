// Single-GPU reproduction of RDMA_CPU_Proxy's routing metadata initialization.
#include "expert_token_idx_standalone.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> struct Buffer {
  T* data = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(n) {
    check(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
  }
  ~Buffer() { if (data) cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  std::vector<T> host() const {
    std::vector<T> result(size);
    check(cudaMemcpy(result.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
  }
};

struct Event {
  cudaEvent_t event{};
  Event() { check(cudaEventCreate(&event)); }
  ~Event() { cudaEventDestroy(event); }
};

// Payload setup only: the gather is a bitwise copy, so use deterministic
// 16-bit patterns to validate every BF16-sized element without a host source.
__host__ __device__ static uint16_t payload_value(size_t row, size_t column) {
  uint32_t value = static_cast<uint32_t>(row) * 0x9e3779b9U
                 ^ static_cast<uint32_t>(column) * 0x85ebca6bU;
  value ^= value >> 16;
  return static_cast<uint16_t>(value);
}

__global__ void initialize_payload(uint16_t* data, size_t elements, int dimension) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    data[i] = payload_value(i / dimension, i % dimension);
}

int main(int argc, char** argv) try {
  int device = 0, local_gpu = 0, tokens = 20480, warmup = 20, iters = 200;
  int node_rank = 0, token_dimension = 4096;
  bool profile = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "Usage: benchmark_rdma_metadata [--device 0] [--local-gpu-id 0]\n"
                   "  [--tokens 20480] [--warmup 20] [--iters 200] [--profile]\n"
                   "  [--node-rank 0] [--token-dimension 4096]\n"
                   "Fixed topology: 2 nodes, 8 GPUs/node, 128 experts, top-k 8, seed 1234.\n"
                   "Payload uses 2 bytes/element, matching BF16 in the RDMA test.\n"
                   "--profile captures six metadata kernels + local gather per invocation.\n";
      return 0;
    }
    if (arg == "--profile") { profile = true; continue; }
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
    size_t consumed = 0;
    std::string value = argv[++i];
    int n = std::stoi(value, &consumed);
    if (consumed != value.size()) throw std::runtime_error("invalid integer: " + value);
    if (arg == "--device") device = n;
    else if (arg == "--local-gpu-id") local_gpu = n;
    else if (arg == "--tokens") tokens = n;
    else if (arg == "--warmup") warmup = n;
    else if (arg == "--iters") iters = n;
    else if (arg == "--node-rank") node_rank = n;
    else if (arg == "--token-dimension") token_dimension = n;
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (device < 0 || local_gpu < 0 || local_gpu >= 8 || tokens <= 0 ||
      tokens > 10000000 || warmup < 1 || iters < 1 || node_rank < 0 || node_rank > 1 ||
      token_dimension < 1 || token_dimension > 1048576)
    throw std::runtime_error("invalid arguments (tokens: 1..10000000; warmup/iters >= 1; "
                             "node-rank: 0..1; token-dimension: 1..1048576)");
  check(cudaSetDevice(device));
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, device));
  int runtime = 0, driver = 0;
  check(cudaRuntimeGetVersion(&runtime));
  check(cudaDriverGetVersion(&driver));
  std::cout << "GPU: " << prop.name << ", SMs=" << prop.multiProcessorCount
            << ", CUDA runtime=" << runtime << ", driver=" << driver << '\n'
            << "tokens=" << tokens << ", top_k=8, experts=128, nodes=2, GPUs/node=8"
            << ", local_gpu_id=" << local_gpu << ", seed=1234\n";
  constexpr int top_k = 8, experts = 128, experts_per_gpu = 8, gpus = 8;
  const size_t chunks = (static_cast<size_t>(tokens) + 255) / 256;
  Buffer<int32_t> r(static_cast<size_t>(tokens) * top_k), values(r.size), offsets(129);
  Buffer<int32_t> x3(static_cast<size_t>(tokens) * 2), node_offsets(3);
  Buffer<uint8_t> x4(x3.size);
  Buffer<int32_t> counts(chunks * 640), prefixes(counts.size), mask_offsets(512);
  Buffer<int32_t> reordered_counts(2 * chunks * 72), reordered_prefixes(reordered_counts.size);
  // nullptr is the default stream, matching RouterRouting::initialize_cuda().
  auto launch_metadata = [&] {
    get_expert_token_idx_node_mask_cuda_raw(
        r.data, tokens, top_k, experts, experts_per_gpu, gpus, local_gpu,
        values.data, offsets.data, x3.data, x4.data, node_offsets.data,
        counts.data, prefixes.data, mask_offsets.data,
        reordered_counts.data, reordered_prefixes.data, nullptr);
  };
  generate_r_cuda_raw(r.data, tokens, top_k, experts, 1234, nullptr);
  check(cudaDeviceSynchronize());
  Event begin, end;
  auto sample = [&](auto&& operation) {
    check(cudaEventRecord(begin.event, nullptr));
    operation();
    check(cudaEventRecord(end.event, nullptr));
    check(cudaEventSynchronize(end.event));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, begin.event, end.event));
    return static_cast<double>(ms) * 1000.0;
  };
  const double first_us = sample(launch_metadata);

  // Independent CPU reference: descending mask, stable token order, then
  // expert lists indexing either the full node sequence or compact GPU input.
  const auto routes = r.host();
  std::array<std::vector<std::pair<int, int>>, 2> entries;
  for (int t = 0; t < tokens; ++t) {
    int masks[2] = {};
    for (int k = 0; k < top_k; ++k) {
      int expert = routes[static_cast<size_t>(t) * top_k + k];
      if (expert < 0 || expert >= experts) throw std::runtime_error("invalid generated route");
      masks[expert / 64] |= 1 << ((local_gpu - (expert / 8) % 8 + 8) % 8);
    }
    for (int n = 0; n < 2; ++n)
      if (masks[n]) entries[n].emplace_back(masks[n], t);
  }
  std::vector<int32_t> expected_x3, expected_offsets{0}, expected_values;
  std::vector<uint8_t> expected_x4;
  std::vector<int32_t> expected_nodes{0};
  std::array<std::vector<int32_t>, experts> lists;
  for (int n = 0; n < 2; ++n) {
    std::stable_sort(entries[n].begin(), entries[n].end(),
                     [](auto a, auto b) { return a.first > b.first; });
    int positions[8] = {};
    for (size_t p = 0; p < entries[n].size(); ++p) {
      auto [mask, token] = entries[n][p];
      expected_x3.push_back(token);
      expected_x4.push_back(static_cast<uint8_t>(mask));
      for (int k = 0; k < top_k; ++k) {
        int expert = routes[static_cast<size_t>(token) * top_k + k];
        if (expert / 64 != n) continue;
        int gpu = (expert / 8) % 8;
        lists[expert].push_back(gpu == local_gpu ? static_cast<int>(p) : positions[gpu]);
      }
      for (int gpu = 0; gpu < 8; ++gpu)
        if (mask & (1 << ((local_gpu - gpu + 8) % 8))) ++positions[gpu];
    }
    expected_nodes.push_back(static_cast<int32_t>(expected_x3.size()));
  }
  for (const auto& list : lists) {
    expected_values.insert(expected_values.end(), list.begin(), list.end());
    expected_offsets.push_back(static_cast<int32_t>(expected_values.size()));
  }
  auto validate = [&] {
    const auto actual_x3 = x3.host();
    const auto actual_x4 = x4.host();
    if (offsets.host() != expected_offsets || values.host() != expected_values ||
        node_offsets.host() != expected_nodes ||
        !std::equal(expected_x3.begin(), expected_x3.end(), actual_x3.begin()) ||
        !std::equal(expected_x4.begin(), expected_x4.end(), actual_x4.begin()))
      throw std::runtime_error("CPU reference mismatch");
  };
  validate();
  std::cout << "CPU reference: PASS (x3, x4, node offsets, expert offsets/indices)\n";

  // Like the proxy, use the selected node's x3 slice. R is generated once in
  // this C++ benchmark, so its offset/length can be read once outside timing.
  // Each measured call still regenerates x3 on the GPU before gathering it.
  const auto host_nodes = node_offsets.host();
  const int local_begin = host_nodes[node_rank];
  const int local_rows = host_nodes[node_rank + 1] - local_begin;
  const size_t row_bytes = static_cast<size_t>(token_dimension) * sizeof(uint16_t);
  Buffer<uint16_t> payload(static_cast<size_t>(tokens) * token_dimension);
  Buffer<uint16_t> staging(payload.size);
  initialize_payload<<<std::min<size_t>((payload.size + 255) / 256, 4096), 256>>>(
      payload.data, payload.size, token_dimension);
  check(cudaGetLastError());
  check(cudaMemset(staging.data, 0xa5, staging.size * sizeof(uint16_t)));
  check(cudaDeviceSynchronize());
  auto launch_gather = [&] {
    gather_token_rows_cuda_raw(staging.data, payload.data, x3.data + local_begin,
                              local_rows, row_bytes, nullptr);
  };
  auto launch = [&] {
    launch_metadata();
    launch_gather();  // Seventh kernel, same stream; no intervening host read/sync.
  };
  auto validate_gather = [&] {
    const auto actual = staging.host();
    for (size_t i = 0; i < actual.size(); ++i) {
      const size_t row = i / token_dimension;
      uint16_t expected = 0xa5a5;
      if (row < static_cast<size_t>(local_rows)) {
        const auto source = expected_x3[static_cast<size_t>(local_begin) + row];
        expected = payload_value(static_cast<size_t>(source), i % token_dimension);
      }
      if (actual[i] != expected)
        throw std::runtime_error("local gather mismatch at element " + std::to_string(i));
    }
  };
  std::cout << "Local staging: node_rank=" << node_rank << ", rows=" << local_rows
            << ", token_dimension=" << token_dimension << ", row_bytes=" << row_bytes
            << ", copied_bytes=" << static_cast<size_t>(local_rows) * row_bytes << '\n';
  if (local_rows == 0)
    std::cout << "Empty local-node slice: gather wrapper is a no-op (no seventh kernel).\n";
  const double first_full_us = sample(launch);
  validate();
  validate_gather();
  std::cout << "Local gather CPU reference: PASS (all valid elements and untouched tail)\n";
  for (int i = 0; i < warmup; ++i) launch();
  check(cudaDeviceSynchronize());
  if (profile) {
    check(cudaProfilerStart());
    for (int i = 0; i < iters; ++i) launch();
    check(cudaDeviceSynchronize());
    check(cudaProfilerStop());
    std::cout << "Captured " << iters << " metadata + local gather calls; inspect Nsight Systems for\n"
                 "individual durations and first-kernel-start to last-kernel-end spans.\n";
  } else {
    std::cout << std::fixed << std::setprecision(3)
              << "First metadata call (not process startup): " << first_us << " us\n"
              << "First combined call (metadata already used): " << first_full_us << " us\n";
    auto benchmark = [&](const char* label, auto&& operation) {
      for (int i = 0; i < warmup; ++i) operation();
      check(cudaDeviceSynchronize());
      std::vector<double> samples;
      samples.reserve(iters);
      for (int i = 0; i < iters; ++i) samples.push_back(sample(operation));
      std::sort(samples.begin(), samples.end());
      auto percentile = [&](double q) { return samples[static_cast<size_t>(q * (samples.size() - 1))]; };
      std::cout << label << ", " << iters << " samples:\n"
                << "  min=" << samples.front() << " median=" << percentile(0.5)
                << " mean=" << std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()
                << " p95=" << percentile(0.95) << " max=" << samples.back() << " us\n";
    };
    benchmark("Warm metadata + local gather CUDA-event span (7 kernels)", launch);
    benchmark("Warm metadata-only CUDA-event span (6 kernels)", launch_metadata);
    benchmark("Warm gather-only CUDA-event span", launch_gather);
    std::cout << "Separate timing runs: do not infer exact gather duration by subtracting medians.\n"
                 "Includes submission gaps and boundary event overhead; excludes R generation,\n"
                 "payload setup, allocation, CPU validation, and D2H. Use nsys for kernel durations.\n";
  }
  validate();
  validate_gather();
  return 0;
} catch (const std::exception& e) {
  std::cerr << "ERROR: " << e.what() << '\n';
  return 1;
}
