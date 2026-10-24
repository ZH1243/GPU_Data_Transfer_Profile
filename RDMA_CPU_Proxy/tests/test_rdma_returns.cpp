#include "proxy.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace rdma_proxy {
struct RdmaReturnsTestAccess {
    static void require(bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    }
    static void exercise(bool empty) {
        ProxyConfig config;
        config.mock_mode = true;
        config.num_nodes = 2;
        config.num_gpus_per_node = 8;
        config.num_iterations = 2;
        config.router_routing_enabled = true;
        config.router_local_input_staging_enabled = true;
        config.router_num_experts = 128;
        config.router_top_k = empty ? 1 : 8;
        config.num_tokens = empty ? 0 : 64;
        config.token_dimension = 40;
        config.dtype = DataType::kFP16;
        config.peers = {PeerAddress{1, "mock", 12345}};
        // The reverse path must bypass the forward per-token scatter layout.
        config.rdma_chunk_per_token_sge_enabled = true;
        Proxy proxy(config);
        proxy.rdma_context_.initialize();
        proxy.router_routing_.initialize();
        proxy.initialized_ = true;
        Proxy::PeerState peer;
        peer.peer_rank = 1;
        peer.router_x4 = proxy.router_routing_.token_masks_for_node(1);
        const auto rows = peer.router_x4.size();
        for (int q = 0; q < 8; ++q) {
            auto qp = std::make_unique<RdmaQueuePair>(proxy.rdma_context_, config, 1, q);
            qp->connect(qp->local_info());
            auto worker = std::make_unique<QPWorker>(*qp, 16, 2);
            worker->configure_expected_chunks(1);
            worker->post_initial_receives(128);
            worker->start();
            peer.qps.push_back(std::move(qp));
            peer.workers.push_back(std::move(worker));
        }
        proxy.peers_.push_back(std::move(peer));
        const auto bytes = rows * 40 * 2;
        std::vector<unsigned char> source(std::max(std::size_t{2}, bytes));
        std::vector<unsigned char> dest(source.size(), 0);
        RdmaProxyNetworkReturnBuffer buffer{1, reinterpret_cast<uintptr_t>(source.data()),
            reinterpret_cast<uintptr_t>(dest.data()), rows, rows};
        const auto address = proxy.configure_rdma_returns(&buffer, 1, 40, 8, 2, 2, 37, 3000);
        auto* prefix = reinterpret_cast<int32_t*>(address);
        std::vector<unsigned char> forward_source(80, 19), forward_dest(80);
        for (uint64_t iteration = 0; iteration < 2; ++iteration) {
            std::fill(dest.begin(), dest.end(), 0);
            for (std::size_t i = 0; i < bytes; ++i) source[i] = 1 + (i + iteration) % 254;
            proxy.prepare_rdma_returns(iteration);
            require(__atomic_load_n(prefix, __ATOMIC_ACQUIRE) == 0, "prefix not reset");
            auto forward = std::make_shared<DynamicChunkDistributor>(
                compute_chunks(1, 40, 2, 1, 8), 1, 8,
                reinterpret_cast<uintptr_t>(forward_source.data()), 0,
                reinterpret_cast<uintptr_t>(forward_dest.data()), 0);
            for (auto& w : proxy.peers_[0].workers) {
                SendTask task;
                task.distributor = forward;
                w->enqueue(task);
            }
            proxy.service_rdma_returns();
            require(proxy.rdma_returns_->peers[0].next_group == 0, "released unready work");
            __atomic_store_n(prefix, 4, __ATOMIC_RELEASE);
            bool rejected = false;
            try { proxy.service_rdma_returns(); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "accepted an out-of-range ready prefix");
            __atomic_store_n(prefix, 1, __ATOMIC_RELEASE);
            proxy.service_rdma_returns();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            for (;;) {
                bool complete = true;
                for (const auto& w : proxy.peers_[0].workers) {
                    require(!w->cq_errors() && !w->post_errors(), "worker error");
                    complete &= w->return_recv_markers() == iteration * 3 + 1;
                }
                if (complete) break;
                require(std::chrono::steady_clock::now() < deadline, "group timeout");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const auto first_bytes = rows * 16 * 2;
            require(std::equal(source.begin(), source.begin() + first_bytes, dest.begin()), "first group differs");
            require(std::all_of(dest.begin() + first_bytes, dest.end(), [](auto b) { return b == 0; }), "unready group transferred");
            // Skip a polled prefix and include a partial final group (5 tiles / 2).
            __atomic_store_n(prefix, 3, __ATOMIC_RELEASE);
            proxy.finish_rdma_returns();
            require(std::equal(source.begin(), source.begin() + bytes, dest.begin()), "full payload differs");
            uint64_t sends = 0, recvs = 0;
            for (const auto& w : proxy.peers_[0].workers) {
                sends += w->send_completions();
                recvs += w->recv_completions();
                require(w->send_marker_completions() == iteration + 1 && w->recv_marker_completions() == iteration + 1,
                        "return markers polluted forward counts");
            }
            require(sends == iteration + 1 && recvs == iteration + 1, "return data polluted forward counts");
            require(forward_dest == forward_source, "forward payload differs");
        }
        // A stale marker must fail the return protocol, not count as a forward arrival.
        auto& qp = proxy.peers_[0].qps[0];
        auto& worker = proxy.peers_[0].workers[0];
        qp->post_send_with_immediate(kReturnWrIdBit | (1ULL << 63),
                                    kReturnImmediateBit | kReturnMarkerBit | 5);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!worker->cq_errors() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        require(worker->cq_errors() == 1, "stale return marker was accepted");
        proxy.shutdown();
    }
};
}
int main() {
    rdma_proxy::RdmaReturnsTestAccess::exercise(false);
    rdma_proxy::RdmaReturnsTestAccess::exercise(true);
    std::cout << "test_rdma_returns passed\n";
}
