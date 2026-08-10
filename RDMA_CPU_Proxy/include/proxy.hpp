#pragma once

#include "config.hpp"
#include "cuda_buffers.hpp"
#include "protocol.hpp"
#include "qp_worker.hpp"
#include "rdma_connection.hpp"
#include "rdma_context.hpp"
#include "router_routing.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rdma_proxy {

class Proxy {
public:
    explicit Proxy(ProxyConfig config);
    ~Proxy();

    Proxy(const Proxy&) = delete;
    Proxy& operator=(const Proxy&) = delete;

    void initialize();
    void run();
    void run_once();
    void shutdown();

private:
    struct PeerState {
        int peer_rank{-1};
        int remote_gpu_index{-1};
        MemoryRegionInfo local_send_mr;
        MemoryRegionInfo local_recv_mr;
        MemoryRegionInfo remote_recv_mr;
        std::vector<ChunkDescriptor> receive_chunks;
        std::vector<uint8_t> router_x4;
        std::vector<uint8_t> forwarding_routing_table;
        std::vector<std::unique_ptr<RdmaQueuePair>> qps;
        std::vector<std::unique_ptr<QPWorker>> workers;
    };

    struct QPCompletionBaseline {
        uint64_t sends{0};
        uint64_t recvs{0};
        uint64_t send_markers{0};
        uint64_t recv_markers{0};
        uint64_t post_errors{0};
        uint64_t cq_errors{0};
        uint64_t unexpected_imms{0};
        std::vector<uint64_t> immediate_counts;
    };

    struct ForwardDestinationBufferState {
        int source_node_rank{-1};
        void* ptr{nullptr};
        std::size_t bytes{0};
        bool imported_cuda_ipc{false};
    };

    struct ForwardDestinationState {
        int gpu_index{-1};
        int cuda_device_id{-1};
        std::vector<ForwardDestinationBufferState> source_buffers;
    };

    struct ForwardingIterationStats {
        std::size_t batch_count{0};
        std::size_t bandwidth_sample_count{0};
        std::size_t total_bytes{0};
        double total_seconds{0.0};
        double sum_batch_bandwidth_gbytes_per_sec{0.0};
        double sum_batch_bandwidth_gbits_per_sec{0.0};
    };

    struct ForwardingOutOfOrderPeerState {
        std::unique_ptr<std::atomic<int8_t>[]> chunk_status;
        std::unique_ptr<std::atomic<uint64_t>[]> arrivals_by_chunk;
        std::size_t total_chunks{0};
        std::size_t chunks_per_iteration{0};
        uint64_t applied_batch_sequence{0};
    };

    struct LocalIterationSyncHeader;
    struct LocalIterationSyncSlot;
    struct NvlinkForwardNotification;
    struct NvlinkForwardNotificationHeader;
    struct NvlinkForwardNotificationQueue;
    struct NvlinkForwardNotificationDispatchState;
    struct ForwardNotificationDestinationState {
        int gpu_index{-1};
        NvlinkForwardNotificationHeader* header{nullptr};
        std::size_t bytes{0};
        int fd{-1};
    };

    PeerConnectionInfo make_local_peer_info(const PeerState& peer) const;
    void setup_peer(PeerGpuBuffers& buffers);
    RouterX3Metadata exchange_router_receive_metadata(
        const PeerAddress& peer_addr) const;
    void synchronize_peer_ready(const PeerAddress& peer_addr, const PeerState& peer) const;
    void initialize_local_iteration_sync();
    void release_local_iteration_sync();
    std::string local_iteration_sync_shm_name() const;
    LocalIterationSyncSlot* local_iteration_sync_slot(int gpu_index) const;
    void initialize_nvlink_forward_notifications();
    void release_nvlink_forward_notifications();
    std::string nvlink_forward_notification_shm_name(int gpu_index) const;
    NvlinkForwardNotificationQueue* nvlink_forward_notification_queue_for_source(
        NvlinkForwardNotificationHeader* header,
        int source_gpu) const;
    NvlinkForwardNotification* nvlink_forward_notification_entry(
        NvlinkForwardNotificationHeader* header,
        NvlinkForwardNotificationQueue* queue,
        uint64_t position) const;
    void prepare_forwarding_notification_destinations();
    void initialize_nvlink_forward_notification_dispatch();
    void enqueue_forward_completion_notifications(
        std::vector<NvlinkForwardNotification>&& notifications);
    void publish_forward_completion_notification(const NvlinkForwardNotification& notification);
    void mark_forward_notification_senders_done();
    bool nvlink_forward_notification_queues_complete() const;
    void nvlink_forward_notification_dispatch_loop();
    void nvlink_forward_notification_loop();
    void drain_nvlink_forward_notification_queue(NvlinkForwardNotificationQueue* queue);
    std::string format_nvlink_forward_notification_log(
        const NvlinkForwardNotification& notification,
        uint64_t dequeue_timestamp_ns) const;
    void enqueue_nvlink_forward_notification_log(
        const NvlinkForwardNotification& notification,
        uint64_t dequeue_timestamp_ns);
    void flush_nvlink_forward_notification_log_queue();
    std::size_t synchronize_local_nvlink_forward_batch_start(
        uint64_t iteration,
        std::size_t batch_index_in_iteration,
        uint64_t batch_sequence,
        std::size_t available_batch_chunks) const;
    void run_iteration(uint64_t iteration);
    void synchronize_iteration_start(uint64_t iteration) const;
    void synchronize_iteration(uint64_t iteration) const;
    void synchronize_local_iteration_phase(const std::string& phase, uint64_t iteration) const;
    void fill_iteration_send_buffers(uint64_t iteration);
    std::vector<std::size_t> sequential_peer_order() const;
    std::vector<ChunkDescriptor> make_chunks(int peer_rank = -1) const;
    std::vector<QPCompletionBaseline> capture_baselines(
        const PeerState& peer,
        const std::vector<ChunkDescriptor>& chunks) const;
    std::shared_ptr<DynamicChunkDistributor> enqueue_chunks(
        PeerState& peer,
        const PeerGpuBuffers& buffers,
        const std::vector<ChunkDescriptor>& chunks);
    void wait_for_iteration(
        const PeerState& peer,
        const std::vector<QPCompletionBaseline>& baselines,
        const std::shared_ptr<DynamicChunkDistributor>& distributor) const;
    void wait_for_outgoing_transfer(
        const PeerState& peer,
        const std::vector<QPCompletionBaseline>& baselines,
        const std::shared_ptr<DynamicChunkDistributor>& distributor) const;
    void start_forwarding_thread();
    void stop_forwarding_thread();
    void prepare_forwarding_routing_tables();
    void prepare_forwarding_destinations();
    void publish_local_nvlink_receive_buffers() const;
    ForwardDestinationState load_forward_destination(int dst_gpu) const;
    const ForwardDestinationBufferState& forward_destination_buffer(
        const ForwardDestinationState& destination,
        int source_node_rank) const;
    std::string nvlink_exchange_file(int gpu_index) const;
    void forwarding_loop();
    std::vector<std::size_t> nvlink_forward_peer_order() const;
    std::size_t forwarding_tokens_for_peer(const PeerState& peer) const;
    bool forwarding_batch_available(
        const PeerState& peer,
        const std::vector<ChunkDescriptor>& chunks,
        std::size_t batch_start_token,
        std::size_t batch_tokens,
        uint64_t required_count) const;
    bool forwarding_chunk_available(
        const PeerState& peer,
        const ChunkDescriptor& chunk,
        uint64_t required_count) const;
    void issue_forwarding_batch(
        const PeerState& peer,
        const PeerGpuBuffers& buffers,
        uint64_t iteration,
        std::size_t batch_index_in_iteration,
        std::size_t batch_start_token,
        std::size_t batch_tokens);
    void forwarding_ready_loop();
    void record_out_of_order_forwarding_arrival(std::size_t peer_index, std::size_t chunk_index);
    void wait_for_forwarding_iteration(uint64_t iteration);
    void set_forwarding_error(const std::string& error);
    void check_forwarding_error() const;
    std::size_t verify_immediates(
        const PeerState& peer,
        const std::vector<ChunkDescriptor>& chunks,
        const std::vector<QPCompletionBaseline>& baselines,
        const IterationAssignment& assignment,
        uint64_t iteration) const;
    std::size_t validate_received_data(uint64_t iteration) const;
    void report_rdma_bandwidth_summary() const;
    void report_iteration(
        uint64_t iteration,
        std::chrono::steady_clock::time_point start,
        double seconds,
        std::size_t total_bytes,
        const std::vector<std::vector<QPCompletionBaseline>>& baselines,
        const std::vector<IterationAssignment>& assignments,
        std::size_t verification_errors,
        std::size_t validation_errors);

    ProxyConfig config_;
    CudaBuffers cuda_buffers_;
    RouterRouting router_routing_;
    RdmaContext rdma_context_;
    ConnectionManager connection_manager_;
    std::vector<PeerState> peers_;
    std::atomic<bool> forwarding_stop_{false};
    std::thread forwarding_thread_;
    std::thread forwarding_ready_thread_;
    void* forwarding_stream_{nullptr};
    mutable std::mutex forwarding_mutex_;
    std::vector<std::size_t> forwarding_next_batch_by_peer_;
    std::vector<std::size_t> forwarding_next_chunk_by_peer_;
    // Router-driven forwarding compacts each destination independently. These
    // cursors are owned by the forwarding thread and reset at iteration
    // boundaries; keeping the source cursor as well makes ordered forwarding
    // an explicit invariant instead of an assumption of the copy layout.
    std::vector<uint64_t> forwarding_compaction_iteration_by_peer_;
    std::vector<std::size_t> forwarding_compaction_next_source_token_by_peer_;
    std::vector<std::vector<std::size_t>> forwarding_compaction_next_destination_token_by_peer_;
    std::vector<uint64_t> forwarding_out_of_order_next_batch_by_peer_;
    std::unique_ptr<std::atomic<std::size_t>[]> forwarding_ready_batches_by_peer_;
    std::unique_ptr<std::atomic<std::size_t>[]> forwarding_ready_chunks_by_peer_;
    std::unique_ptr<std::atomic<std::size_t>[]> forwarding_out_of_order_current_start_by_peer_;
    std::unique_ptr<std::atomic<std::size_t>[]> forwarding_out_of_order_current_end_by_peer_;
    std::unique_ptr<std::atomic<std::size_t>[]> forwarding_out_of_order_current_length_by_peer_;
    std::unique_ptr<std::atomic<std::size_t>[]> forwarding_out_of_order_current_tail_ready_by_peer_;
    std::unique_ptr<std::atomic<uint64_t>[]> forwarding_out_of_order_current_version_by_peer_;
    std::unique_ptr<std::atomic<uint64_t>[]> forwarding_out_of_order_ready_next_batch_by_peer_;
    std::unique_ptr<std::atomic<std::size_t>[]> forwarding_out_of_order_command_start_by_peer_;
    std::unique_ptr<std::atomic<std::size_t>[]> forwarding_out_of_order_command_length_by_peer_;
    std::unique_ptr<std::atomic<uint64_t>[]> forwarding_out_of_order_command_sequence_by_peer_;
    std::unique_ptr<std::atomic<uint64_t>[]> forwarding_out_of_order_command_ack_by_peer_;
    std::unique_ptr<std::atomic<std::size_t>[]> forwarding_out_of_order_issued_chunks_by_peer_;
    std::size_t forwarding_ready_peer_count_{0};
    std::vector<ForwardingOutOfOrderPeerState> forwarding_out_of_order_peer_states_;
    std::vector<ForwardDestinationState> forwarding_destinations_;
    std::vector<ForwardNotificationDestinationState> forwarding_notification_destinations_;
    std::vector<ForwardingIterationStats> forwarding_iteration_stats_;
    std::string forwarding_error_;
    std::atomic<uint64_t> forwarding_batch_available_calls_{0};
    std::atomic<uint64_t> forwarding_batch_available_total_ns_{0};
    std::atomic<uint64_t> forwarding_batches_issued_{0};
    std::thread nvlink_forward_notification_thread_;
    std::thread nvlink_forward_notification_dispatch_thread_;
    NvlinkForwardNotificationHeader* nvlink_forward_notification_header_{nullptr};
    std::size_t nvlink_forward_notification_size_{0};
    int nvlink_forward_notification_fd_{-1};
    std::string nvlink_forward_notification_name_;
    std::unique_ptr<NvlinkForwardNotificationDispatchState> nvlink_forward_notification_dispatch_;
    std::atomic<bool> nvlink_forward_notification_dispatch_stop_{false};
    std::atomic<bool> nvlink_forward_notification_receiver_stop_{false};
    std::atomic<uint64_t> nvlink_forward_notifications_enqueued_{0};
    std::atomic<uint64_t> nvlink_forward_notifications_sent_{0};
    std::atomic<uint64_t> nvlink_forward_notifications_received_{0};
    std::atomic<uint64_t> nvlink_forward_notifications_dropped_{0};
    std::mutex nvlink_forward_notification_log_mutex_;
    std::deque<std::string> nvlink_forward_notification_log_queue_;
    std::vector<double> rdma_iteration_bandwidth_gbps_;
    LocalIterationSyncHeader* local_iteration_sync_header_{nullptr};
    std::size_t local_iteration_sync_size_{0};
    int local_iteration_sync_fd_{-1};
    std::string local_iteration_sync_name_;
    bool initialized_{false};
};

}  // namespace rdma_proxy
