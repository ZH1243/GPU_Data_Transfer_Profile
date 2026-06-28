#pragma once

#include "config.hpp"
#include "protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rdma_proxy {

struct GpuBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};
    bool is_mock_host_memory{false};
};

struct PeerGpuBuffers {
    int peer_rank{-1};
    GpuBuffer send;
    GpuBuffer recv;
};

class CudaBuffers {
public:
    explicit CudaBuffers(ProxyConfig config);
    ~CudaBuffers();

    CudaBuffers(const CudaBuffers&) = delete;
    CudaBuffers& operator=(const CudaBuffers&) = delete;

    void initialize();
    void copy_tokens_to_send_buffer(int peer_rank, const void* src_device_or_host, std::size_t bytes);
    void fill_test_pattern(int peer_rank, int source_rank, int destination_rank, uint64_t iteration);
    bool validate_recv_pattern(
        int peer_rank,
        int source_rank,
        int destination_rank,
        uint64_t iteration,
        std::string* error) const;

    const std::vector<PeerGpuBuffers>& peer_buffers() const { return buffers_; }
    std::vector<PeerGpuBuffers>& peer_buffers() { return buffers_; }
    PeerGpuBuffers& buffers_for_peer(int peer_rank);
    const PeerGpuBuffers& buffers_for_peer(int peer_rank) const;

    std::size_t token_buffer_bytes() const;

private:
    void allocate_buffer(GpuBuffer& buffer, std::size_t bytes);
    void free_buffer(GpuBuffer& buffer);

    ProxyConfig config_;
    std::vector<PeerGpuBuffers> buffers_;
};

void launch_copy_tokens(void* dst, const void* src, std::size_t bytes, bool mock_mode);

}  // namespace rdma_proxy
