#include "proxy.hpp"

#include "logging.hpp"

#include <algorithm>
#include <stdexcept>

namespace rdma_proxy {

Proxy::Proxy(ProxyConfig config)
    : config_(std::move(config)),
      cuda_buffers_(config_),
      rdma_context_(config_),
      connection_manager_(config_) {}

Proxy::~Proxy() {
    shutdown();
}

void Proxy::initialize() {
    if (initialized_) return;
    RDMA_PROXY_LOG_INFO("initializing proxy: ", config_summary(config_));

    cuda_buffers_.initialize();
    rdma_context_.initialize();

    for (auto& peer_buffers : cuda_buffers_.peer_buffers()) {
        setup_peer(peer_buffers);
    }
    initialized_ = true;
}

void Proxy::run_once() {
    if (!initialized_) throw std::runtime_error("proxy is not initialized");
    for (std::size_t i = 0; i < peers_.size(); ++i) {
        enqueue_chunks(peers_[i], cuda_buffers_.peer_buffers()[i]);
    }
}

void Proxy::shutdown() {
    for (auto& peer : peers_) {
        for (auto& worker : peer.workers) {
            if (worker) worker->stop();
        }
        rdma_context_.deregister_memory(peer.local_send_mr);
        rdma_context_.deregister_memory(peer.local_recv_mr);
    }
    peers_.clear();
    initialized_ = false;
}

PeerConnectionInfo Proxy::make_local_peer_info(const PeerState& peer) const {
    PeerConnectionInfo info;
    info.node_rank = config_.node_rank;
    info.gpu_index = config_.local_gpu_index;
    info.recv_buffer = peer.local_recv_mr;
    info.qps.reserve(peer.qps.size());
    for (const auto& qp : peer.qps) {
        info.qps.push_back(qp->local_info());
    }
    return info;
}

void Proxy::setup_peer(PeerGpuBuffers& buffers) {
    PeerState peer;
    peer.peer_rank = buffers.peer_rank;
    peer.local_send_mr = rdma_context_.register_memory(
        buffers.send.ptr, buffers.send.bytes, "send_buffer_peer_" + std::to_string(buffers.peer_rank));
    peer.local_recv_mr = rdma_context_.register_memory(
        buffers.recv.ptr, buffers.recv.bytes, "recv_buffer_peer_" + std::to_string(buffers.peer_rank));

    for (int q = 0; q < config_.num_qps_per_peer; ++q) {
        peer.qps.emplace_back(new RdmaQueuePair(rdma_context_, config_, buffers.peer_rank, q));
    }

    const auto local_info = make_local_peer_info(peer);
    const auto& peer_addr = *std::find_if(config_.peers.begin(), config_.peers.end(), [&](const PeerAddress& p) {
        return p.node_rank == buffers.peer_rank;
    });
    const auto remote_info = connection_manager_.exchange_peer_info(peer_addr, local_info);
    if (remote_info.qps.size() != peer.qps.size()) {
        throw std::runtime_error("remote QP count does not match local QP count");
    }
    peer.remote_recv_mr = remote_info.recv_buffer;

    for (std::size_t q = 0; q < peer.qps.size(); ++q) {
        peer.qps[q]->connect(remote_info.qps[q]);
        auto worker = std::make_unique<QPWorker>(*peer.qps[q], config_.completion_poll_batch_size);
        worker->post_initial_receives(config_.recv_queue_depth);
        worker->start();
        peer.workers.emplace_back(std::move(worker));
    }

    RDMA_PROXY_LOG_INFO("peer ", buffers.peer_rank, " initialized with ", peer.qps.size(), " RC QPs");
    peers_.push_back(std::move(peer));
}

void Proxy::enqueue_chunks(PeerState& peer, const PeerGpuBuffers& buffers) {
    const auto chunks = compute_chunks(
        config_.num_tokens,
        config_.token_dimension,
        dtype_size(config_.dtype),
        config_.tokens_per_chunk,
        config_.num_qps_per_peer);

    uint64_t wr_id = 1;
    for (const auto& chunk : chunks) {
        SendTask task;
        task.wr_id = (static_cast<uint64_t>(peer.peer_rank) << 48) |
                     (static_cast<uint64_t>(chunk.qp_index) << 32) |
                     static_cast<uint64_t>(chunk.chunk_index);
        if (task.wr_id == 0) task.wr_id = wr_id++;
        task.chunk = chunk;
        task.local_base = reinterpret_cast<uintptr_t>(buffers.send.ptr);
        task.local_lkey = peer.local_send_mr.lkey;
        task.remote_base = static_cast<uintptr_t>(peer.remote_recv_mr.addr);
        task.remote_rkey = peer.remote_recv_mr.rkey;
        peer.workers.at(static_cast<std::size_t>(chunk.qp_index))->enqueue(task);
    }
    RDMA_PROXY_LOG_INFO("enqueued ", chunks.size(), " chunks for peer ", peer.peer_rank);
}

}  // namespace rdma_proxy
