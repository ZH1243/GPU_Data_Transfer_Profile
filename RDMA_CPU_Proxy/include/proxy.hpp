#pragma once

#include "config.hpp"
#include "cuda_buffers.hpp"
#include "protocol.hpp"
#include "qp_worker.hpp"
#include "rdma_connection.hpp"
#include "rdma_context.hpp"

#include <atomic>
#include <chrono>
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

    struct ForwardDestinationState {
        int gpu_index{-1};
        int cuda_device_id{-1};
        void* ptr{nullptr};
        std::size_t bytes{0};
        bool imported_cuda_ipc{false};
    };

    struct ForwardingIterationStats {
        std::size_t batch_count{0};
        std::size_t bandwidth_sample_count{0};
        std::size_t total_bytes{0};
        double total_seconds{0.0};
        double sum_batch_bandwidth_gbytes_per_sec{0.0};
        double sum_batch_bandwidth_gbits_per_sec{0.0};
    };

    struct LocalIterationSyncHeader;
    struct LocalIterationSyncSlot;

    PeerConnectionInfo make_local_peer_info(const PeerState& peer) const;
    void setup_peer(PeerGpuBuffers& buffers);
    void synchronize_peer_ready(const PeerAddress& peer_addr, const PeerState& peer) const;
    void initialize_local_iteration_sync();
    void release_local_iteration_sync();
    std::string local_iteration_sync_shm_name() const;
    LocalIterationSyncSlot* local_iteration_sync_slot(int gpu_index) const;
    void run_iteration(uint64_t iteration);
    void synchronize_iteration_start(uint64_t iteration) const;
    void synchronize_iteration(uint64_t iteration) const;
    void synchronize_local_iteration_phase(const std::string& phase, uint64_t iteration) const;
    void fill_iteration_send_buffers(uint64_t iteration);
    std::vector<std::size_t> sequential_peer_order() const;
    std::vector<ChunkDescriptor> make_chunks() const;
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
    std::string nvlink_exchange_file(int gpu_index) const;
    void forwarding_loop();
    std::vector<std::size_t> nvlink_forward_peer_order() const;
    bool forwarding_batch_available(
        const PeerState& peer,
        const std::vector<ChunkDescriptor>& chunks,
        std::size_t batch_start_token,
        std::size_t batch_tokens,
        uint64_t required_count) const;
    void issue_forwarding_batch(
        const PeerState& peer,
        const PeerGpuBuffers& buffers,
        uint64_t iteration,
        std::size_t batch_index_in_iteration,
        std::size_t batch_start_token);
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
        std::size_t bytes_per_peer,
        const std::vector<std::vector<QPCompletionBaseline>>& baselines,
        const std::vector<IterationAssignment>& assignments,
        std::size_t verification_errors,
        std::size_t validation_errors);

    ProxyConfig config_;
    CudaBuffers cuda_buffers_;
    RdmaContext rdma_context_;
    ConnectionManager connection_manager_;
    std::vector<PeerState> peers_;
    std::atomic<bool> forwarding_stop_{false};
    std::thread forwarding_thread_;
    void* forwarding_stream_{nullptr};
    mutable std::mutex forwarding_mutex_;
    std::vector<std::size_t> forwarding_next_batch_by_peer_;
    std::vector<std::vector<uint8_t>> forwarding_routing_tables_by_peer_;
    std::vector<ForwardDestinationState> forwarding_destinations_;
    std::vector<ForwardingIterationStats> forwarding_iteration_stats_;
    std::string forwarding_error_;
    std::atomic<uint64_t> forwarding_batch_available_calls_{0};
    std::atomic<uint64_t> forwarding_batch_available_total_ns_{0};
    std::atomic<uint64_t> forwarding_batches_issued_{0};
    std::vector<double> rdma_iteration_bandwidth_gbps_;
    LocalIterationSyncHeader* local_iteration_sync_header_{nullptr};
    std::size_t local_iteration_sync_size_{0};
    int local_iteration_sync_fd_{-1};
    std::string local_iteration_sync_name_;
    bool initialized_{false};
};

}  // namespace rdma_proxy
