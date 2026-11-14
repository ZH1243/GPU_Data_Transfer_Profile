#pragma once

#include "config.hpp"
#include "protocol.hpp"
#include "rdma_proxy_c_api.h"

#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rdma_proxy {

struct GpuBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};
    bool is_mock_host_memory{false};
    // Borrowed buffers are allocated and retained by the embedding runtime.
    // CudaBuffers may read/write them but must never release their storage.
    bool is_externally_owned{false};
    // Aliases share storage with another GpuBuffer descriptor and never own
    // that storage, regardless of whether the original allocation is internal
    // or supplied by an embedding runtime.
    bool is_alias{false};
};

enum class DeviceBufferKind : int32_t {
    kRdmaSend = 1,
    kRdmaReceive = 2,
    kNvlinkReceive = 3,
    kRouterAIdx = 4,
    kGatherTable = 5,
    kGatherReadyRows = 6,
    kRouterRouteWeights = 7,
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
    // Cumulative CPU-side diagnostics for table/ready publications, packed
    // metadata rows, and successfully enqueued/copied metadata transfers.
    uint64_t table_flush_count{0};
    uint64_t ready_rows_publication_count{0};
    std::atomic<uint64_t> cpu_scatter_packed_rows{0};
    // Ready-publication count observed when the last deferred packing began.
    std::atomic<uint64_t> cpu_scatter_pack_ready_publication_count{0};
    std::atomic<uint64_t> cpu_scatter_copy_count{0};
    std::atomic<uint64_t> cpu_scatter_copy_bytes{0};
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
    // Explicit scheduler passes, including passes that find no complete cluster.
    uint64_t scheduling_passes{0};
    std::vector<std::size_t> expert_total_tokens;
    std::vector<std::size_t> expert_num_ready_tokens;
    std::vector<std::size_t> expert_num_notified_batches;
    std::vector<std::size_t> expert_total_batches;
};

// Computed on request, outside the notification hot path. N-group repetitions
// count as one M cluster. Publication counts are cumulative across iterations.
struct RouterNotificationSchedulingDiagnostics {
    uint64_t scheduling_passes{0};
    std::size_t m_clusters{0};
    std::size_t expert_runs{0};
    std::size_t max_consecutive_m_clusters{0};
    uint64_t table_flush_count{0};
    uint64_t ready_rows_publication_count{0};
};

struct NvlinkReceiveBuffer {
    int source_node_rank{-1};
    int source_gpu_index{-1};
    GpuBuffer recv;
    // QuACK A_idx_j. All source buffers use the same padded device allocation
    // size; expert_token_index_count records the meaningful prefix.
    GpuBuffer expert_token_indices_device;
    GpuBuffer expert_route_weights_device;
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

// Final host arguments, filled by the preparation thread and retained until
// stream completion. No conversion or allocation is needed at submission.
struct CudaPreparedForwardBatch {
    std::vector<void*> dsts;
    std::vector<void*> srcs;
    std::vector<std::size_t> sizes;
#if RDMA_PROXY_HAVE_CUDA
    cudaMemcpyAttributes attributes{};
#endif
    std::size_t attribute_index{0};
    std::size_t attribute_count{1};

    void reserve(std::size_t capacity);
    void clear();
    void append(void* dst, const void* src, std::size_t bytes);
};
void launch_cuda_prepared_forward_batch_async(
    CudaPreparedForwardBatch& batch, void* stream, bool mock_mode);

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
    void stage_local_router_input(
        const std::vector<std::size_t>& source_token_indices,
        const int32_t* device_source_token_indices);
    void install_expert_metadata(RouterExpertMetadata metadata);
    RouterExpertTokenHeadState expert_token_head_state_for_source(
        int source_node_rank,
        int source_gpu_index) const;
    RouterComputationSchedulerState router_computation_scheduler_state() const;
    RouterNotificationSchedulingDiagnostics router_notification_scheduling_diagnostics() const;
    void begin_router_notification_iteration(uint64_t iteration);
    void synchronize_router_notification_publication();
    void record_router_computation_end(uint64_t iteration, uintptr_t cuda_stream);
    float router_computation_elapsed_ms(uint64_t iteration);
    std::size_t router_computation_num_tokens() const;
    void configure_cpu_scatter(const RdmaProxyCpuScatterConfig& config);
    void configure_cpu_route_weights(uintptr_t packed_weights, std::size_t routes);
    void configure_cpu_scatter_deferred(std::size_t bytes);
    void configure_cpu_scatter_threaded(std::size_t bytes);
    void stop_cpu_scatter_worker();
    void wait_cpu_scatter_ready(uint64_t iteration, uintptr_t cuda_stream, uint64_t timeout_ms);
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
    // The receiver must finish transfer visibility handling before ingestion.
    // Ingestion advances contiguous source Heads only: no descriptors, scatter
    // metadata or nonzero ready prefixes are constructed/published here.
    // As with the immediate API, an implicit iteration start can publish zero.
    void ingest_router_notification_completion(
        int source_node_rank,
        int source_gpu_index,
        uint64_t iteration,
        std::size_t start_token,
        std::size_t num_tokens);
    // Schedule one complete collection, grouped in increasing expert order.
    // The caller controls completion boundaries and must not expose pending
    // collections to an immediate/early-local scheduling pass.
    void schedule_router_notification_completions(uint64_t iteration);
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
    void alias_buffer(
        GpuBuffer& buffer,
        const GpuBuffer& source,
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
        bool flush_ready_entries_per_entry = false);
    void ingest_router_notification_completion_locked(
        int source_node_rank, int source_gpu_index, uint64_t iteration,
        std::size_t start_token, std::size_t num_tokens,
        bool flush_ready_entries_per_entry = false);
    std::size_t schedule_ready_computation_batches_for_expert_locked(
        std::size_t expert,
        bool flush_per_entry);
    std::size_t schedule_ready_computation_batches_locked(bool flush_per_entry);
    void schedule_router_notification_completions_locked();
    void flush_router_notification_publication_range(
        std::size_t table_offset,
        std::size_t table_bytes);

    ProxyConfig config_;
    friend struct CpuScatterTestAccess;
    void configure_cpu_scatter_slab_locked(std::size_t bytes);
    std::pair<std::size_t, std::size_t> pack_cpu_scatter_batch(const int32_t* ranges);
    void prepare_cpu_scatter_batch(const int32_t* ranges);
    void upload_cpu_scatter_slab();
    void commit_cpu_scatter_prefix_locked();
    void cpu_scatter_worker_loop() noexcept;
    void wait_threaded_cpu_scatter_ready(uint64_t iteration, uintptr_t cuda_stream, uint64_t timeout_ms);
    void publish_deferred_cpu_scatter_locked();
    CpuPinnedBuffer cpu_scatter_staging_;
    CpuPinnedBuffer cpu_route_weights_staging_;
    CpuPinnedBuffer cpu_scatter_deferred_staging_;
    std::size_t cpu_scatter_routes_{0};
    std::size_t cpu_scatter_weight_offset_{0};
    std::size_t cpu_scatter_staged_rows_{0};
    bool cpu_scatter_submitted_{false};
    void* cpu_scatter_ready_event_{nullptr};
    std::condition_variable cpu_scatter_ready_cv_;
    // Notification/reset always acquire scheduler then worker state. The worker
    // never acquires scheduler state and releases this lock for ALL packing.
    bool cpu_scatter_threaded_{false};
    std::thread cpu_scatter_worker_thread_;
    std::mutex cpu_scatter_worker_mutex_;
    std::condition_variable cpu_scatter_worker_cv_;
    // Final consumers sleep separately from per-prefix work/progress waiters.
    std::condition_variable cpu_scatter_worker_ready_cv_;
    bool cpu_scatter_worker_busy_{false};
    bool cpu_scatter_worker_stop_{false};
    bool cpu_scatter_worker_epoch_valid_{false};
    bool cpu_scatter_worker_cancelled_{false};
    bool cpu_scatter_worker_submitted_{false};
    uint64_t cpu_scatter_worker_epoch_{0};
    std::size_t cpu_scatter_worker_committed_rows_{0};
    std::size_t cpu_scatter_worker_consumed_rows_{0};
    std::size_t cpu_scatter_worker_staged_rows_{0};
    uint64_t cpu_scatter_worker_ready_publication_count_{0};
    std::exception_ptr cpu_scatter_worker_error_;
    // Only the friend test fixture installs this, before any epoch is active.
    std::function<void()> cpu_scatter_pack_test_hook_;
    uintptr_t cpu_route_weights_device_{0};
    uintptr_t cpu_scatter_device_{0};
    std::size_t cpu_scatter_columns_{0};
    // Per-source A_idx entry -> final address (and optional N-tile stride).
    std::vector<std::vector<int64_t>> cpu_scatter_lookup_;
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
void select_cuda_device_for_thread(int cuda_device_id, bool mock_mode);
// Query once during initialization; true means TO_OWNER requires an explicit flush.
bool gpudirect_rdma_writes_need_owner_flush(int cuda_device_id, bool mock_mode);
void flush_gpudirect_rdma_writes(int cuda_device_id, bool mock_mode);
void destroy_cuda_stream(void* stream, bool mock_mode);
void synchronize_cuda_stream(void* stream, bool mock_mode);
// False only for cudaErrorNotReady; all other CUDA failures throw.
bool query_cuda_stream(void* stream, bool mock_mode);
// A recorded stream prefix. Share ownership between CPU consumers, but never
// re-record an event while a consumer still owns the previous recording.
class CudaForwardEvent {
public:
    explicit CudaForwardEvent(bool mock_mode);
    ~CudaForwardEvent();
    CudaForwardEvent(const CudaForwardEvent&) = delete;
    CudaForwardEvent& operator=(const CudaForwardEvent&) = delete;
    void record(void* stream);
    bool ready() const;
    void synchronize() const;

private:
    friend class CudaForwardEventPool;
    void reset_for_acquisition();
    bool recyclable() const;
    void* event_{nullptr};
    bool mock_mode_{false};
    bool recorded_{false};
    mutable bool completed_{false};
    mutable bool failed_{false};
    mutable std::mutex mutex_;
};

// Fixed capacity. Leases keep storage alive and return a slot only after the
// final reference disappears and its recording has successfully completed.
class CudaForwardEventPool {
public:
    CudaForwardEventPool(std::size_t capacity, int cuda_device_id, bool mock_mode);
    std::vector<std::shared_ptr<CudaForwardEvent>> acquire_batch(
        std::size_t count, std::chrono::milliseconds timeout);
    void stop();
private:
    struct State;
    std::shared_ptr<State> state_;
};
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
