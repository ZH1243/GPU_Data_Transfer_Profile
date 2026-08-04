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

struct HostStagingBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};
    bool is_cuda_pinned{false};
};

struct PeerGpuBuffers {
    int peer_rank{-1};
    GpuBuffer send;
    GpuBuffer recv;
};

struct NvlinkReceiveBuffer {
    int source_gpu_index{-1};
    GpuBuffer recv;
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
    const NvlinkReceiveBuffer& nvlink_receive_buffer_for_source(int source_gpu_index) const;
    void copy_nvlink_notification_test_payloads_to_gpu();

    std::size_t token_buffer_bytes() const;
    std::size_t nvlink_receive_buffer_bytes() const;

private:
    void allocate_buffer(GpuBuffer& buffer, std::size_t bytes);
    void free_buffer(GpuBuffer& buffer);
    void allocate_host_staging_buffer(HostStagingBuffer& buffer, std::size_t bytes, uint8_t value);
    void free_host_staging_buffer(HostStagingBuffer& buffer);

    ProxyConfig config_;
    std::vector<PeerGpuBuffers> buffers_;
    std::vector<NvlinkReceiveBuffer> nvlink_recv_buffers_;
    GpuBuffer nvlink_notification_test_buffer_16kib_;
    GpuBuffer nvlink_notification_test_buffer_8b_;
    HostStagingBuffer nvlink_notification_test_payload_16kib_;
    HostStagingBuffer nvlink_notification_test_payload_8b_;
    void* nvlink_notification_copy_stream_{nullptr};
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
