#pragma once

#include <cstdint>
#include <cstddef>

// Torch-free launch API for embedding the demo kernels in C++ applications.
// All pointers refer to CUDA device memory; stream is a cudaStream_t cast to
// void*, or nullptr for the default stream.
void generate_r_cuda_raw(
    int32_t* r,
    int num_tokens,
    int top_k,
    int num_experts,
    uint64_t seed,
    void* stream);

void get_expert_token_idx_node_mask_x3_cuda_raw(
    const int32_t* r,
    int num_tokens,
    int top_k,
    int num_experts,
    int experts_per_gpu,
    int gpus_per_node,
    int local_gpu_id,
    int32_t* expert_offsets,
    int32_t* node_token_indices,
    uint8_t* node_token_masks,
    int32_t* node_offsets,
    int32_t* input_chunk_counts,
    int32_t* input_chunk_prefixes,
    int32_t* node_mask_offsets,
    void* stream);

// Full node-mask path. In addition to x3/x4, this builds packed per-expert
// token positions in the x3 sequence filtered for each expert's owner GPU.
void get_expert_token_idx_node_mask_cuda_raw(
    const int32_t* r,
    int num_tokens,
    int top_k,
    int num_experts,
    int experts_per_gpu,
    int gpus_per_node,
    int local_gpu_id,
    int32_t* expert_token_indices,
    int32_t* expert_offsets,
    int32_t* node_token_indices,
    uint8_t* node_token_masks,
    int32_t* node_offsets,
    int32_t* input_chunk_counts,
    int32_t* input_chunk_prefixes,
    int32_t* node_mask_offsets,
    int32_t* reordered_chunk_counts,
    int32_t* reordered_chunk_prefixes,
    void* stream);

// Gather complete token rows into the contiguous node-level x3 order. All
// pointers refer to device memory. One CTA owns one destination row so a
// sufficiently large token list can occupy the full GPU.
void gather_token_rows_cuda_raw(
    void* destination,
    const void* source,
    const int32_t* source_token_indices,
    int num_rows,
    std::size_t row_bytes,
    void* stream);
