#pragma once

#include "config.hpp"
#include "cuda_buffers.hpp"
#include "protocol.hpp"
#include "qp_worker.hpp"
#include "rdma_connection.hpp"
#include "rdma_context.hpp"

#include <chrono>
#include <memory>
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

    PeerConnectionInfo make_local_peer_info(const PeerState& peer) const;
    void setup_peer(PeerGpuBuffers& buffers);
    void synchronize_peer_ready(const PeerAddress& peer_addr, const PeerState& peer) const;
    void run_iteration(uint64_t iteration);
    void synchronize_iteration_start(uint64_t iteration) const;
    void synchronize_iteration(uint64_t iteration) const;
    void fill_iteration_send_buffers(uint64_t iteration);
    std::vector<ChunkDescriptor> make_chunks() const;
    std::vector<QPCompletionBaseline> capture_baselines(
        const PeerState& peer,
        const std::vector<ChunkDescriptor>& chunks) const;
    void enqueue_chunks(PeerState& peer, const PeerGpuBuffers& buffers, const std::vector<ChunkDescriptor>& chunks);
    void wait_for_iteration(
        const PeerState& peer,
        const std::vector<ChunkDescriptor>& chunks,
        const std::vector<QPCompletionBaseline>& baselines) const;
    std::size_t verify_immediates(
        const PeerState& peer,
        const std::vector<ChunkDescriptor>& chunks,
        const std::vector<QPCompletionBaseline>& baselines,
        uint64_t iteration) const;
    std::size_t validate_received_data(uint64_t iteration) const;
    void report_iteration(
        uint64_t iteration,
        std::chrono::steady_clock::time_point start,
        double seconds,
        std::size_t bytes_per_peer,
        const std::vector<ChunkDescriptor>& chunks,
        const std::vector<std::vector<QPCompletionBaseline>>& baselines,
        std::size_t verification_errors,
        std::size_t validation_errors) const;

    ProxyConfig config_;
    CudaBuffers cuda_buffers_;
    RdmaContext rdma_context_;
    ConnectionManager connection_manager_;
    std::vector<PeerState> peers_;
    bool initialized_{false};
};

}  // namespace rdma_proxy
