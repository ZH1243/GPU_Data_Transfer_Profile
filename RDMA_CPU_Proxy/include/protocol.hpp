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
    int qp_index{0};
    uint32_t imm_data{0};
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
    int num_qps_per_peer);

std::string serialize_peer_info(const PeerConnectionInfo& info);
PeerConnectionInfo deserialize_peer_info(const std::string& payload);

}  // namespace rdma_proxy
