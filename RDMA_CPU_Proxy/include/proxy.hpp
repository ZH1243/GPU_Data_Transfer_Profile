#pragma once

#include "config.hpp"
#include "cuda_buffers.hpp"
#include "protocol.hpp"
#include "qp_worker.hpp"
#include "rdma_connection.hpp"
#include "rdma_context.hpp"

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
    void run_once();
    void shutdown();

private:
    struct PeerState {
        int peer_rank{-1};
        MemoryRegionInfo local_send_mr;
        MemoryRegionInfo local_recv_mr;
        MemoryRegionInfo remote_recv_mr;
        std::vector<std::unique_ptr<RdmaQueuePair>> qps;
        std::vector<std::unique_ptr<QPWorker>> workers;
    };

    PeerConnectionInfo make_local_peer_info(const PeerState& peer) const;
    void setup_peer(PeerGpuBuffers& buffers);
    void enqueue_chunks(PeerState& peer, const PeerGpuBuffers& buffers);

    ProxyConfig config_;
    CudaBuffers cuda_buffers_;
    RdmaContext rdma_context_;
    ConnectionManager connection_manager_;
    std::vector<PeerState> peers_;
    bool initialized_{false};
};

}  // namespace rdma_proxy
