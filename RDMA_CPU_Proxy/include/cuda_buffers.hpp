#pragma once

#include "config.hpp"
#include "protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
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

// CPU-side progress for the per-expert token-index lists attached to one
// source-specific NVLink receive buffer. A head is the number of leading
// entries in that expert's list whose token rows have arrived.
struct RouterExpertTokenHeadState {
    bool iteration_initialized{false};
    uint64_t iteration{0};
    std::size_t received_token_frontier{0};
    std::vector<std::size_t> expert_token_heads;
};

struct NvlinkReceiveBuffer {
    int source_node_rank{-1};
    int source_gpu_index{-1};
    GpuBuffer recv;
    RouterExpertMetadata expert_metadata;
    bool expert_metadata_ready{false};
    RouterExpertTokenHeadState expert_token_head_state;
};

struct CudaForwardCopy {
    void* dst{nullptr};
    const void* src{nullptr};
    std::size_t bytes{0};
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
    void fill_router_test_pattern(int source_rank, uint64_t iteration);
    bool validate_recv_pattern(
        int peer_rank,
        int source_rank,
        int destination_rank,
        uint64_t iteration,
        std::string* error) const;
    bool validate_recv_pattern(
        int peer_rank,
        int source_rank,
        int destination_rank,
        uint64_t iteration,
        const std::vector<ChunkDescriptor>& chunks,
        std::string* error) const;

    const std::vector<PeerGpuBuffers>& peer_buffers() const { return buffers_; }
    std::vector<PeerGpuBuffers>& peer_buffers() { return buffers_; }
    const std::vector<NvlinkReceiveBuffer>& nvlink_receive_buffers() const { return nvlink_recv_buffers_; }
    PeerGpuBuffers& buffers_for_peer(int peer_rank);
    const PeerGpuBuffers& buffers_for_peer(int peer_rank) const;
    const NvlinkReceiveBuffer& nvlink_receive_buffer_for_source(
        int source_node_rank,
        int source_gpu_index) const;
    void install_expert_metadata(RouterExpertMetadata metadata);
    RouterExpertTokenHeadState expert_token_head_state_for_source(
        int source_node_rank,
        int source_gpu_index) const;
    void update_expert_token_heads(
        int source_node_rank,
        int source_gpu_index,
        uint64_t iteration,
        std::size_t start_token,
        std::size_t num_tokens);

    std::size_t token_buffer_bytes() const;
    std::size_t nvlink_receive_buffer_bytes() const;

private:
    void allocate_buffer(GpuBuffer& buffer, std::size_t bytes);
    void free_buffer(GpuBuffer& buffer);

    ProxyConfig config_;
    GpuBuffer router_send_buffer_;
    std::vector<PeerGpuBuffers> buffers_;
    std::vector<NvlinkReceiveBuffer> nvlink_recv_buffers_;
    mutable std::mutex expert_token_heads_mutex_;
};

void launch_copy_tokens(void* dst, const void* src, std::size_t bytes, bool mock_mode);
void* create_cuda_stream(int cuda_device_id, bool nonblocking, bool mock_mode);
void destroy_cuda_stream(void* stream, bool mock_mode);
void synchronize_cuda_stream(void* stream, bool mock_mode);
void enable_cuda_peer_access(int cuda_device_id, int peer_cuda_device_id, bool mock_mode);
void launch_cuda_forward_copy_batch_async(
    const CudaForwardCopy& copy,
    void* stream,
    bool use_batch_api,
    bool mock_mode);
void launch_cuda_forward_copy_batch_async(
    const std::vector<CudaForwardCopy>& copies,
    void* stream,
    bool use_batch_api,
    bool mock_mode);
std::string export_cuda_ipc_memory_handle(void* ptr, bool mock_mode);
void* open_cuda_ipc_memory_handle(const std::string& handle_hex, uint64_t mock_addr, bool mock_mode);
void close_cuda_ipc_memory_handle(void* ptr, bool mock_mode);

}  // namespace rdma_proxy
