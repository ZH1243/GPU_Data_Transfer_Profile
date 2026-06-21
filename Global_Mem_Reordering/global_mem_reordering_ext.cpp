#include <torch/extension.h>

#include <cstdint>

void launch_kernel_a(
    torch::Tensor payload,
    torch::Tensor keys,
    int64_t num_elements,
    int64_t payload_size,
    int64_t key_size,
    double bit_probability,
    uint64_t seed,
    int64_t num_ctas,
    int64_t threads,
    int64_t device);

void launch_kernel_b(
    torch::Tensor payload,
    torch::Tensor keys,
    torch::Tensor sorted_payload,
    torch::Tensor sorted_keys,
    torch::Tensor source_indices,
    torch::Tensor local_histograms,
    torch::Tensor key_offsets,
    torch::Tensor write_offsets,
    int64_t num_elements,
    int64_t payload_size,
    int64_t key_size,
    int64_t num_ctas,
    int64_t threads,
    int64_t payload_copy_method,
    int64_t tma_tile_bytes,
    int64_t device);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_kernel_a", &launch_kernel_a, "initialize payload and key buffers");
  m.def("launch_kernel_b", &launch_kernel_b, "sort keys descending and reorder payloads");
}
