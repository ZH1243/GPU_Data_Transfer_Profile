#include "protocol.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>

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
    int num_qps_per_peer) {
    if (tokens_per_chunk == 0) throw std::runtime_error("tokens_per_chunk must be > 0");
    if (num_qps_per_peer <= 0) throw std::runtime_error("num_qps_per_peer must be > 0");

    const std::size_t token_bytes = token_dimension * dtype_size;
    const std::size_t num_chunks = (num_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
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
        desc.qp_index = static_cast<int>(chunk % static_cast<std::size_t>(num_qps_per_peer));
        desc.imm_data = encode_immediate(chunk);
        chunks.push_back(desc);
    }
    return chunks;
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
