#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kNumKeyValues = 128;
constexpr int64_t kSingleThreadThreshold = 1 << 16;
constexpr int64_t kTargetGrainElements = 1 << 15;

void validate_cpu_keys(
    const torch::Tensor& tensor,
    const char* name,
    int64_t min_numel) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, name, " must have dtype uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() >= min_numel, name, " is smaller than required");
}

void validate_cpu_i32(
    const torch::Tensor& tensor,
    const char* name,
    int64_t min_numel) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must have dtype int32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() >= min_numel, name, " is smaller than required");
}

int64_t choose_chunk_count(int64_t requested_threads, int64_t num_elements) {
  int64_t pool_threads = static_cast<int64_t>(at::get_num_threads());
  int64_t thread_hint = requested_threads > 0 ? requested_threads : pool_threads;
  thread_hint = std::max<int64_t>(1, thread_hint);

  int64_t grain_chunks =
      (num_elements + kTargetGrainElements - 1) / kTargetGrainElements;
  int64_t target_chunks = std::max<int64_t>(thread_hint * 4, grain_chunks);
  return std::max<int64_t>(1, std::min(target_chunks, num_elements));
}

void build_reordered_indices_single_thread(
    const uint8_t* keys,
    int32_t* reordered_indices,
    int64_t num_elements) {
  std::array<int64_t, kNumKeyValues> histogram{};
  for (int64_t src = 0; src < num_elements; ++src) {
    ++histogram[static_cast<int>(keys[src] & 0x7fu)];
  }

  std::array<int64_t, kNumKeyValues> offsets{};
  int64_t running_offset = 0;
  for (int key = kNumKeyValues - 1; key >= 0; --key) {
    offsets[key] = running_offset;
    running_offset += histogram[key];
  }

  for (int64_t src = 0; src < num_elements; ++src) {
    int key = static_cast<int>(keys[src] & 0x7fu);
    reordered_indices[src] = static_cast<int32_t>(offsets[key]++);
  }
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
    int64_t device);

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
    int64_t device);

void build_reordered_indices_cpu(
    torch::Tensor host_keys,
    torch::Tensor host_reordered_indices,
    int64_t num_elements,
    int64_t cpu_threads) {
  TORCH_CHECK(num_elements > 0, "num_elements must be positive");
  TORCH_CHECK(num_elements <= std::numeric_limits<int32_t>::max(),
              "num_elements must fit int32");
  validate_cpu_keys(host_keys, "host_keys", num_elements);
  validate_cpu_i32(host_reordered_indices, "host_reordered_indices", num_elements);

  const auto* keys = static_cast<const uint8_t*>(host_keys.data_ptr());
  auto* reordered_indices =
      static_cast<int32_t*>(host_reordered_indices.data_ptr());

  if (cpu_threads == 1 || num_elements <= kSingleThreadThreshold) {
    build_reordered_indices_single_thread(keys, reordered_indices, num_elements);
    return;
  }

  int64_t chunk_count = choose_chunk_count(cpu_threads, num_elements);
  std::vector<std::array<int64_t, kNumKeyValues>> local_histograms(chunk_count);
  for (auto& hist : local_histograms) {
    hist.fill(0);
  }

  at::parallel_for(0, chunk_count, 1, [&](int64_t chunk_begin, int64_t chunk_end) {
    for (int64_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
      int64_t begin = num_elements * chunk / chunk_count;
      int64_t end = num_elements * (chunk + 1) / chunk_count;
      auto& hist = local_histograms[chunk];
      for (int64_t src = begin; src < end; ++src) {
        ++hist[static_cast<int>(keys[src] & 0x7fu)];
      }
    }
  });

  std::array<int64_t, kNumKeyValues> key_starts{};
  int64_t running_offset = 0;
  for (int key = kNumKeyValues - 1; key >= 0; --key) {
    key_starts[key] = running_offset;
    for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
      running_offset += local_histograms[chunk][key];
    }
  }
  TORCH_CHECK(running_offset == num_elements,
              "internal error while building key offsets");

  std::vector<std::array<int64_t, kNumKeyValues>> local_offsets(chunk_count);
  for (int key = 0; key < kNumKeyValues; ++key) {
    int64_t offset = key_starts[key];
    for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
      local_offsets[chunk][key] = offset;
      offset += local_histograms[chunk][key];
    }
  }

  at::parallel_for(0, chunk_count, 1, [&](int64_t chunk_begin, int64_t chunk_end) {
    for (int64_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
      int64_t begin = num_elements * chunk / chunk_count;
      int64_t end = num_elements * (chunk + 1) / chunk_count;
      auto offsets = local_offsets[chunk];
      for (int64_t src = begin; src < end; ++src) {
        int key = static_cast<int>(keys[src] & 0x7fu);
        reordered_indices[src] = static_cast<int32_t>(offsets[key]++);
      }
    }
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_kernel_a", &launch_kernel_a_cpu_hybrid,
        "initialize payload and 7-bit key buffers");
  m.def("build_reordered_indices_cpu", &build_reordered_indices_cpu,
        "build src-to-dst payload reorder indices on CPU using 7-bit keys");
  m.def("launch_kernel_b", &launch_kernel_b_cpu_hybrid,
        "copy payloads from original to reordered buffer using src-to-dst indices");
}
