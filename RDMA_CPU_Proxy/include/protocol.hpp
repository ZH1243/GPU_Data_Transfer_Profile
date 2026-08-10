#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rdma_proxy {

struct QPInfo {
    uint32_t qp_num{0};
    uint16_t lid{0};
    uint8_t gid[16]{};
    uint32_t psn{0};
};

struct MemoryRegionInfo {
    uint64_t addr{0};
    uint32_t lkey{0};
    uint32_t rkey{0};
    std::size_t length{0};
};

struct PeerConnectionInfo {
    int node_rank{-1};
    int gpu_index{-1};
    std::vector<QPInfo> qps;
    MemoryRegionInfo recv_buffer;
};

struct ChunkDescriptor {
    std::size_t chunk_index{0};
    std::size_t start_token{0};
    std::size_t num_tokens{0};
    std::size_t src_offset_bytes{0};
    std::size_t dst_offset_bytes{0};
    std::size_t length_bytes{0};
    int qp_index{-1};
    uint32_t imm_data{0};
    std::vector<std::size_t> source_token_indices;
};

struct RouterX3Metadata {
    int source_node_rank{-1};
    int destination_node_rank{-1};
    int local_gpu_index{-1};
    int num_nodes{0};
    int num_gpus_per_node{0};
    int num_experts{0};
    int top_k{0};
    std::size_t num_tokens{0};
    std::size_t token_dimension{0};
    std::size_t element_bytes{0};
    std::size_t tokens_per_chunk{0};
    std::vector<std::size_t> token_indices;
    std::vector<uint8_t> token_masks;
};

uint32_t encode_immediate(std::size_t chunk_index);
std::size_t decode_immediate(uint32_t imm_data);
uint32_t encode_marker_immediate();
bool is_marker_immediate(uint32_t imm_data);

std::vector<ChunkDescriptor> compute_chunks(
    std::size_t num_tokens,
    std::size_t token_dimension,
    std::size_t dtype_size,
    std::size_t tokens_per_chunk,
    int num_qps_per_peer,
    bool discontinuous_token_payload = false);

std::vector<ChunkDescriptor> compute_chunks_from_token_indices(
    const std::vector<std::size_t>& source_token_indices,
    std::size_t token_dimension,
    std::size_t dtype_size,
    std::size_t tokens_per_chunk,
    int num_qps_per_peer);

std::string serialize_router_x3_metadata(const RouterX3Metadata& metadata);
RouterX3Metadata deserialize_router_x3_metadata(
    const std::string& payload,
    std::size_t maximum_num_tokens);

std::vector<uint8_t> normalize_router_x4_for_nvlink(
    const std::vector<uint8_t>& token_masks,
    int num_gpus_per_node);

std::string serialize_peer_info(const PeerConnectionInfo& info);
PeerConnectionInfo deserialize_peer_info(const std::string& payload);

}  // namespace rdma_proxy
