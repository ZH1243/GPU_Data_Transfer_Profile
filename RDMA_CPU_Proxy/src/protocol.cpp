#include "protocol.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rdma_proxy {

uint32_t encode_immediate(std::size_t chunk_index) {
    if (chunk_index >= 0xffffffffULL) {
        throw std::runtime_error("chunk index exceeds immediate data capacity");
    }
    return static_cast<uint32_t>(chunk_index);
}

std::size_t decode_immediate(uint32_t imm_data) {
    return static_cast<std::size_t>(imm_data);
}

uint32_t encode_marker_immediate() {
    return 0xffffffffU;
}

bool is_marker_immediate(uint32_t imm_data) {
    return imm_data == encode_marker_immediate();
}

std::vector<ChunkDescriptor> compute_chunks(
    std::size_t num_tokens,
    std::size_t token_dimension,
    std::size_t dtype_size,
    std::size_t tokens_per_chunk,
    int num_qps_per_peer,
    bool discontinuous_token_payload) {
    if (tokens_per_chunk == 0) throw std::runtime_error("tokens_per_chunk must be > 0");
    if (num_qps_per_peer <= 0) throw std::runtime_error("num_qps_per_peer must be > 0");

    const std::size_t token_bytes = token_dimension * dtype_size;
    const std::size_t num_chunks = (num_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
    std::vector<std::size_t> token_order;
    if (discontinuous_token_payload) {
        token_order.resize(num_tokens);
        std::iota(token_order.begin(), token_order.end(), std::size_t{0});
        std::mt19937_64 rng(1);
        std::shuffle(token_order.begin(), token_order.end(), rng);
    }
    std::vector<ChunkDescriptor> chunks;
    chunks.reserve(num_chunks);

    for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
        const std::size_t start = chunk * tokens_per_chunk;
        const std::size_t count = std::min(tokens_per_chunk, num_tokens - start);
        ChunkDescriptor desc;
        desc.chunk_index = chunk;
        desc.start_token = start;
        desc.num_tokens = count;
        desc.src_offset_bytes = start * token_bytes;
        desc.dst_offset_bytes = start * token_bytes;
        desc.length_bytes = count * token_bytes;
        desc.qp_index = -1;
        desc.imm_data = encode_immediate(chunk);
        if (discontinuous_token_payload) {
            desc.source_token_indices.assign(
                token_order.begin() + static_cast<std::ptrdiff_t>(start),
                token_order.begin() + static_cast<std::ptrdiff_t>(start + count));
        }
        chunks.push_back(desc);
    }
    return chunks;
}

std::vector<ChunkDescriptor> compute_chunks_from_token_indices(
    const std::vector<std::size_t>& source_token_indices,
    std::size_t token_dimension,
    std::size_t dtype_size,
    std::size_t tokens_per_chunk,
    int num_qps_per_peer) {
    if (tokens_per_chunk == 0) throw std::runtime_error("tokens_per_chunk must be > 0");
    if (num_qps_per_peer <= 0) throw std::runtime_error("num_qps_per_peer must be > 0");

    const std::size_t token_bytes = token_dimension * dtype_size;
    const std::size_t num_chunks =
        (source_token_indices.size() + tokens_per_chunk - 1) / tokens_per_chunk;
    std::vector<ChunkDescriptor> chunks;
    chunks.reserve(num_chunks);
    for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
        const std::size_t start = chunk * tokens_per_chunk;
        const std::size_t count =
            std::min(tokens_per_chunk, source_token_indices.size() - start);
        ChunkDescriptor desc;
        desc.chunk_index = chunk;
        desc.start_token = start;
        desc.num_tokens = count;
        desc.src_offset_bytes = 0;
        desc.dst_offset_bytes = start * token_bytes;
        desc.length_bytes = count * token_bytes;
        desc.qp_index = -1;
        desc.imm_data = encode_immediate(chunk);
        desc.source_token_indices.assign(
            source_token_indices.begin() + static_cast<std::ptrdiff_t>(start),
            source_token_indices.begin() + static_cast<std::ptrdiff_t>(start + count));
        chunks.push_back(std::move(desc));
    }
    return chunks;
}

std::string serialize_router_x3_metadata(const RouterX3Metadata& metadata) {
    if (metadata.token_masks.size() != metadata.token_indices.size()) {
        throw std::runtime_error("router x3/x4 metadata lengths do not match");
    }
    std::ostringstream out;
    out << "router_x3_x4_v2 "
        << metadata.source_node_rank << ' '
        << metadata.destination_node_rank << ' '
        << metadata.local_gpu_index << ' '
        << metadata.num_nodes << ' '
        << metadata.num_gpus_per_node << ' '
        << metadata.num_experts << ' '
        << metadata.top_k << ' '
        << metadata.num_tokens << ' '
        << metadata.token_dimension << ' '
        << metadata.element_bytes << ' '
        << metadata.tokens_per_chunk << ' '
        << metadata.token_indices.size();
    for (const auto token : metadata.token_indices) out << ' ' << token;
    for (const auto mask : metadata.token_masks) {
        out << ' ' << static_cast<unsigned>(mask);
    }
    return out.str();
}

RouterX3Metadata deserialize_router_x3_metadata(
    const std::string& payload,
    std::size_t maximum_num_tokens) {
    std::istringstream in(payload);
    std::string version;
    RouterX3Metadata metadata;
    std::size_t count = 0;
    in >> version
       >> metadata.source_node_rank
       >> metadata.destination_node_rank
       >> metadata.local_gpu_index
       >> metadata.num_nodes
       >> metadata.num_gpus_per_node
       >> metadata.num_experts
       >> metadata.top_k
       >> metadata.num_tokens
       >> metadata.token_dimension
       >> metadata.element_bytes
       >> metadata.tokens_per_chunk
       >> count;
    if (!in || version != "router_x3_x4_v2") {
        throw std::runtime_error("failed to parse router x3/x4 metadata header");
    }
    if (metadata.num_tokens > maximum_num_tokens || count > metadata.num_tokens) {
        throw std::runtime_error("router x3/x4 metadata token count exceeds configured capacity");
    }
    if (metadata.num_gpus_per_node < 2 || metadata.num_gpus_per_node > 8) {
        throw std::runtime_error("router x3/x4 metadata GPU count is out of range");
    }
    metadata.token_indices.resize(count);
    std::vector<bool> seen(metadata.num_tokens, false);
    for (auto& token : metadata.token_indices) {
        in >> token;
        if (!in || token >= metadata.num_tokens) {
            throw std::runtime_error("router x3 metadata contains an invalid token index");
        }
        if (seen[token]) {
            throw std::runtime_error("router x3 metadata contains a duplicate token index");
        }
        seen[token] = true;
    }
    metadata.token_masks.resize(count);
    const unsigned meaningful_mask = metadata.num_gpus_per_node == 8 ?
        0xffU : ((1U << metadata.num_gpus_per_node) - 1U);
    for (auto& mask : metadata.token_masks) {
        unsigned value = 0;
        in >> value;
        if (!in || value == 0 || value > 0xffU || (value & ~meaningful_mask) != 0) {
            throw std::runtime_error("router x4 metadata contains an invalid GPU mask");
        }
        mask = static_cast<uint8_t>(value);
    }
    std::string trailing;
    if (in >> trailing) {
        throw std::runtime_error("router x3/x4 metadata contains trailing fields");
    }
    return metadata;
}

std::string serialize_router_expert_metadata(
    const RouterExpertMetadata& metadata) {
    if (metadata.experts_per_gpu <= 0 ||
        metadata.expert_offsets.size() !=
            static_cast<std::size_t>(metadata.experts_per_gpu + 1)) {
        throw std::runtime_error("router expert metadata offset count is invalid");
    }
    if (metadata.expert_offsets.front() != 0 ||
        metadata.expert_offsets.back() < 0 ||
        static_cast<std::size_t>(metadata.expert_offsets.back()) !=
            metadata.expert_token_indices.size()) {
        throw std::runtime_error("router expert metadata offset endpoints are invalid");
    }
    for (std::size_t i = 1; i < metadata.expert_offsets.size(); ++i) {
        if (metadata.expert_offsets[i] < metadata.expert_offsets[i - 1]) {
            throw std::runtime_error("router expert metadata offsets are not monotonic");
        }
    }

    std::ostringstream out;
    out << "router_expert_metadata_v1 "
        << metadata.source_node_rank << ' '
        << metadata.source_gpu_index << ' '
        << metadata.destination_node_rank << ' '
        << metadata.destination_gpu_index << ' '
        << metadata.num_nodes << ' '
        << metadata.num_gpus_per_node << ' '
        << metadata.num_experts << ' '
        << metadata.experts_per_gpu << ' '
        << metadata.first_global_expert << ' '
        << metadata.num_tokens << ' '
        << metadata.expert_token_indices.size();
    for (const auto offset : metadata.expert_offsets) out << ' ' << offset;
    for (const auto index : metadata.expert_token_indices) out << ' ' << index;
    return out.str();
}

RouterExpertMetadata deserialize_router_expert_metadata(
    const std::string& payload,
    std::size_t maximum_num_tokens,
    std::size_t maximum_index_count) {
    std::istringstream in(payload);
    std::string version;
    RouterExpertMetadata metadata;
    std::size_t index_count = 0;
    in >> version
       >> metadata.source_node_rank
       >> metadata.source_gpu_index
       >> metadata.destination_node_rank
       >> metadata.destination_gpu_index
       >> metadata.num_nodes
       >> metadata.num_gpus_per_node
       >> metadata.num_experts
       >> metadata.experts_per_gpu
       >> metadata.first_global_expert
       >> metadata.num_tokens
       >> index_count;
    if (!in || version != "router_expert_metadata_v1") {
        throw std::runtime_error("failed to parse router expert metadata header");
    }
    if (metadata.num_nodes <= 0 || metadata.num_gpus_per_node <= 0 ||
        metadata.num_experts <= 0 || metadata.experts_per_gpu <= 0) {
        throw std::runtime_error("router expert metadata dimensions are invalid");
    }
    if (metadata.num_tokens > maximum_num_tokens ||
        index_count > maximum_index_count) {
        throw std::runtime_error("router expert metadata exceeds configured capacity");
    }
    metadata.expert_offsets.resize(
        static_cast<std::size_t>(metadata.experts_per_gpu) + 1);
    for (auto& offset : metadata.expert_offsets) {
        in >> offset;
        if (!in || offset < 0 ||
            static_cast<std::size_t>(offset) > index_count) {
            throw std::runtime_error("router expert metadata contains an invalid offset");
        }
    }
    if (metadata.expert_offsets.front() != 0 ||
        static_cast<std::size_t>(metadata.expert_offsets.back()) != index_count) {
        throw std::runtime_error("router expert metadata offset endpoints are invalid");
    }
    for (std::size_t i = 1; i < metadata.expert_offsets.size(); ++i) {
        if (metadata.expert_offsets[i] < metadata.expert_offsets[i - 1]) {
            throw std::runtime_error("router expert metadata offsets are not monotonic");
        }
    }
    metadata.expert_token_indices.resize(index_count);
    for (auto& index : metadata.expert_token_indices) {
        in >> index;
        if (!in || index < 0 ||
            static_cast<std::size_t>(index) >= metadata.num_tokens) {
            throw std::runtime_error("router expert metadata contains an invalid token index");
        }
    }
    std::string trailing;
    if (in >> trailing) {
        throw std::runtime_error("router expert metadata contains trailing fields");
    }
    return metadata;
}

std::vector<uint8_t> normalize_router_x4_for_nvlink(
    const std::vector<uint8_t>& token_masks,
    int num_gpus_per_node) {
    if (num_gpus_per_node < 2 || num_gpus_per_node > 8) {
        throw std::runtime_error("router x4 normalization requires num_gpus_per_node in [2, 8]");
    }
    const unsigned meaningful_mask = num_gpus_per_node == 8 ?
        0xffU : ((1U << num_gpus_per_node) - 1U);
    const unsigned shift = static_cast<unsigned>(8 - num_gpus_per_node);
    std::vector<uint8_t> normalized;
    normalized.reserve(token_masks.size());
    for (const auto raw_mask : token_masks) {
        const unsigned value = raw_mask;
        if (value == 0 || (value & ~meaningful_mask) != 0) {
            throw std::runtime_error("router x4 contains an invalid GPU mask");
        }
        normalized.push_back(static_cast<uint8_t>(value << shift));
    }
    return normalized;
}

std::string serialize_peer_info(const PeerConnectionInfo& info) {
    std::ostringstream out;
    out << info.node_rank << ' ' << info.gpu_index << ' ' << info.recv_buffer.addr << ' '
        << info.recv_buffer.lkey << ' ' << info.recv_buffer.rkey << ' ' << info.recv_buffer.length << ' '
        << info.qps.size();
    for (const auto& qp : info.qps) {
        out << ' ' << qp.qp_num << ' ' << qp.lid << ' ' << qp.psn;
        for (uint8_t byte : qp.gid) {
            out << ' ' << static_cast<unsigned>(byte);
        }
    }
    return out.str();
}

PeerConnectionInfo deserialize_peer_info(const std::string& payload) {
    std::istringstream in(payload);
    PeerConnectionInfo info;
    std::size_t qp_count = 0;
    in >> info.node_rank >> info.gpu_index >> info.recv_buffer.addr >> info.recv_buffer.lkey
       >> info.recv_buffer.rkey >> info.recv_buffer.length >> qp_count;
    if (!in) throw std::runtime_error("failed to parse peer metadata header");
    info.qps.resize(qp_count);
    for (auto& qp : info.qps) {
        in >> qp.qp_num >> qp.lid >> qp.psn;
        for (auto& byte : qp.gid) {
            unsigned v = 0;
            in >> v;
            byte = static_cast<uint8_t>(v);
        }
        if (!in) throw std::runtime_error("failed to parse QP metadata");
    }
    return info;
}

}  // namespace rdma_proxy
