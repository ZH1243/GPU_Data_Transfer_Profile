#pragma once

#include "config.hpp"
#include "protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace rdma_proxy {

struct GpuBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};
    bool is_mock_host_memory{false};
    // Borrowed buffers are allocated and retained by the embedding runtime.
    // CudaBuffers may read/write them but must never release their storage.
    bool is_externally_owned{false};
};

enum class DeviceBufferKind : int32_t {
    kRdmaSend = 1,
    kRdmaReceive = 2,
    kNvlinkReceive = 3,
    kRouterAIdx = 4,
    kGatherTable = 5,
    kGatherReadyRows = 6,
};

enum class DeviceBufferElementType : int32_t {
    kBF16 = 1,
    kFP16 = 2,
    kFP32 = 3,
    kInt32 = 4,
};

struct DeviceBufferAllocationRequest {
    DeviceBufferKind kind{DeviceBufferKind::kRdmaSend};
    DeviceBufferElementType element_type{DeviceBufferElementType::kBF16};
    int peer_rank{-1};
    int source_node_rank{-1};
    int source_gpu_index{-1};
    std::size_t bytes{0};
    std::vector<std::size_t> dimensions;
};

using ExternalDeviceBufferAllocator =
    std::function<void*(const DeviceBufferAllocationRequest&)>;

struct CpuPinnedBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};
    bool is_mock_host_memory{false};
};

struct RouterNotificationPublicationBuffers {
    // QuACK multi-buffer gather-table ABI. Each row is a contiguous int32 row:
    //   (local_expert_id, cid_n_base, start_0, end_0, ..., start_B-1, end_B-1)
    // where buffer slot j is the j-th entry of nvlink_receive_buffers().
    CpuPinnedBuffer host_table;
    CpuPinnedBuffer host_ready_rows;
    GpuBuffer device_table;
    GpuBuffer device_ready_rows;
    std::size_t table_row_bytes{0};
    std::size_t table_rows{0};
    std::size_t table_width{0};
    std::size_t num_input_buffers{0};
    std::size_t a_idx_capacity{0};
    std::size_t n_groups_per_m_cluster{0};
    int32_t group_size{0};
    // CPU-side diagnostics counting successfully enqueued/copied table ranges
    // and their corresponding ready-row publications.
    uint64_t table_flush_count{0};
    uint64_t ready_rows_publication_count{0};
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
    std::vector<std::size_t> expert_token_tails;
};

struct RouterComputationSchedulerState {
    bool initialized{false};
    bool iteration_initialized{false};
    uint64_t iteration{0};
    // M-clusters whose input ranges have been assigned.
    std::size_t published_batches{0};
    // QuACK gather-table rows made available. One M-cluster contributes one
    // row for every N group.
    std::size_t published_rows{0};
    std::vector<std::size_t> expert_total_tokens;
    std::vector<std::size_t> expert_num_ready_tokens;
    std::vector<std::size_t> expert_num_notified_batches;
    std::vector<std::size_t> expert_total_batches;
};

struct NvlinkReceiveBuffer {
    int source_node_rank{-1};
    int source_gpu_index{-1};
    GpuBuffer recv;
    // QuACK A_idx_j. All source buffers use the same padded device allocation
    // size; expert_token_index_count records the meaningful prefix.
    GpuBuffer expert_token_indices_device;
    std::size_t expert_token_index_count{0};
    RouterExpertMetadata expert_metadata;
    bool expert_metadata_ready{false};
    RouterExpertTokenHeadState expert_token_head_state;
};

struct CudaForwardCopy {
    void* dst{nullptr};
    const void* src{nullptr};
    std::size_t bytes{0};
};

struct CudaIpcMemoryHandle {
    std::string handle_hex;
    std::size_t offset{0};
};

class CudaBuffers {
public:
    explicit CudaBuffers(ProxyConfig config);
    ~CudaBuffers();

    CudaBuffers(const CudaBuffers&) = delete;
    CudaBuffers& operator=(const CudaBuffers&) = delete;

    void set_external_device_buffer_allocator(ExternalDeviceBufferAllocator allocator);
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
    RouterComputationSchedulerState router_computation_scheduler_state() const;
    void begin_router_notification_iteration(uint64_t iteration);
    void synchronize_router_notification_publication();
    void record_router_computation_end(uint64_t iteration, uintptr_t cuda_stream);
    float router_computation_elapsed_ms(uint64_t iteration);
    std::size_t router_computation_num_tokens() const;
    void update_expert_token_heads(
        int source_node_rank,
        int source_gpu_index,
        uint64_t iteration,
        std::size_t start_token,
        std::size_t num_tokens);
    void process_router_notification_completion(
        int source_node_rank,
        int source_gpu_index,
        uint64_t iteration,
        std::size_t start_token,
        std::size_t num_tokens);
    void flush_router_notification_publication();
    const RouterNotificationPublicationBuffers& router_notification_publication_buffers() const {
        return router_notification_publication_buffers_;
    }

    std::size_t token_buffer_bytes() const;
    std::size_t nvlink_receive_buffer_bytes() const;

private:
    void allocate_buffer(
        GpuBuffer& buffer,
        const DeviceBufferAllocationRequest& request);
    void free_buffer(GpuBuffer& buffer);
    void allocate_pinned_buffer(CpuPinnedBuffer& buffer, std::size_t bytes);
    void free_pinned_buffer(CpuPinnedBuffer& buffer);
    void initialize_router_computation_scheduler_locked();
    bool is_router_computation_input(const NvlinkReceiveBuffer& buffer) const;
    void reset_router_computation_iteration_locked(uint64_t iteration);
    void update_expert_token_heads_locked(
        NvlinkReceiveBuffer& buffer,
        uint64_t iteration,
        std::size_t start_token,
        std::size_t num_tokens,
        bool flush_ready_entries_per_entry);
    std::size_t schedule_ready_computation_batches_for_expert_locked(
        std::size_t expert,
        bool flush_per_entry);
    std::size_t schedule_ready_computation_batches_locked();
    void flush_router_notification_publication_range(
        std::size_t table_offset,
        std::size_t table_bytes);

    ProxyConfig config_;
    ExternalDeviceBufferAllocator external_device_buffer_allocator_;
    GpuBuffer router_send_buffer_;
    std::vector<PeerGpuBuffers> buffers_;
    std::vector<NvlinkReceiveBuffer> nvlink_recv_buffers_;
    RouterNotificationPublicationBuffers router_notification_publication_buffers_;
    // cudaMemcpyAsync flag publications need immutable pinned source values
    // until their stream operations complete. Slot i permanently contains i.
    CpuPinnedBuffer router_notification_ready_values_;
    void* router_notification_publication_stream_{nullptr};
    void* router_computation_start_event_{nullptr};
    void* router_computation_end_event_{nullptr};
    mutable std::mutex router_computation_timing_mutex_;
    bool router_computation_timing_initialized_{false};
    bool router_computation_start_recorded_{false};
    bool router_computation_end_recorded_{false};
    uint64_t router_computation_timing_iteration_{0};
    mutable std::mutex expert_token_heads_mutex_;
    RouterComputationSchedulerState router_computation_scheduler_state_;
    std::vector<std::size_t> router_computation_buffer_indices_;
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
CudaIpcMemoryHandle export_cuda_ipc_memory_handle(void* ptr, bool mock_mode);
void* open_cuda_ipc_memory_handle(const std::string& handle_hex, uint64_t mock_addr, bool mock_mode);
void close_cuda_ipc_memory_handle(void* ptr, bool mock_mode);

}  // namespace rdma_proxy
