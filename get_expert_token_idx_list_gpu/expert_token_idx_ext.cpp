#include <torch/extension.h>

#include <cstdint>

void generate_r_cuda(
    torch::Tensor r,
    int64_t num_experts,
    uint64_t seed);

void get_expert_token_idx_cuda(
    torch::Tensor r,
    int64_t num_experts,
    int64_t experts_per_gpu,
    torch::Tensor expert_token_indices,
    torch::Tensor expert_offsets,
    torch::Tensor chunk_counts,
    torch::Tensor chunk_prefixes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "generate_r_",
      &generate_r_cuda,
      "Fill a CUDA int32 routing matrix with distinct random experts per row");
  m.def(
      "get_expert_token_idx",
      &get_expert_token_idx_cuda,
      "Build stable per-expert GPU-local token-index lists");
}
