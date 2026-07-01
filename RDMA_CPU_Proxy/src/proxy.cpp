#include "proxy.hpp"

#include "logging.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

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
    run_iteration(0);
    synchronize_iteration(0);
}

void Proxy::run() {
    if (!initialized_) throw std::runtime_error("proxy is not initialized");
    for (uint64_t iteration = 0;
         config_.num_iterations == 0 || iteration < static_cast<uint64_t>(config_.num_iterations);
         ++iteration) {
        run_iteration(iteration);
        synchronize_iteration(iteration);
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
    peer.remote_gpu_index = remote_info.gpu_index;

    for (std::size_t q = 0; q < peer.qps.size(); ++q) {
        peer.qps[q]->connect(remote_info.qps[q]);
        auto worker = std::make_unique<QPWorker>(*peer.qps[q], config_.completion_poll_batch_size);
        worker->configure_expected_chunks(make_chunks().size());
        worker->post_initial_receives(config_.recv_queue_depth);
        worker->start();
        peer.workers.emplace_back(std::move(worker));
    }

    synchronize_peer_ready(peer_addr, peer);
    RDMA_PROXY_LOG_INFO("peer ", buffers.peer_rank, " initialized with ", peer.qps.size(), " RC QPs");
    peers_.push_back(std::move(peer));
}

void Proxy::synchronize_peer_ready(const PeerAddress& peer_addr, const PeerState& peer) const {
    std::ostringstream local;
    local << "peer_ready"
          << " rank=" << config_.node_rank
          << " gpu=" << config_.local_gpu_index
          << " peer_rank=" << peer.peer_rank
          << " remote_gpu=" << peer.remote_gpu_index;
    const auto expected_rank = "rank=" + std::to_string(peer.peer_rank);
    const auto expected_gpu = "gpu=" + std::to_string(peer.remote_gpu_index);

    RDMA_PROXY_LOG_INFO("local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " waiting for peer ready remote_rank=", peer.peer_rank,
                        " remote_gpu=", peer.remote_gpu_index);
    const auto remote = connection_manager_.exchange_control_message(
        peer_addr, local.str(), config_.completion_timeout_ms);
    if (remote.find(expected_rank) == std::string::npos ||
        remote.find(expected_gpu) == std::string::npos) {
        throw std::runtime_error("peer ready synchronization mismatch: " + remote);
    }
    RDMA_PROXY_LOG_INFO("local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " peer ready remote_rank=", peer.peer_rank,
                        " remote_gpu=", peer.remote_gpu_index);
}

void Proxy::run_iteration(uint64_t iteration) {
    const auto chunks = make_chunks();
    const auto bytes_per_peer = cuda_buffers_.token_buffer_bytes();

    fill_iteration_send_buffers(iteration);

    std::vector<std::vector<QPCompletionBaseline>> baselines;
    baselines.reserve(peers_.size());
    for (const auto& peer : peers_) {
        baselines.push_back(capture_baselines(peer, chunks));
    }
    synchronize_iteration_start(iteration);

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < peers_.size(); ++i) {
        enqueue_chunks(peers_[i], cuda_buffers_.peer_buffers()[i], chunks);
    }
    for (std::size_t i = 0; i < peers_.size(); ++i) {
        wait_for_iteration(peers_[i], chunks, baselines[i]);
    }
    const auto end = std::chrono::steady_clock::now();
    const auto seconds = std::chrono::duration<double>(end - start).count();

    std::size_t verification_errors = 0;
    for (std::size_t i = 0; i < peers_.size(); ++i) {
        verification_errors += verify_immediates(peers_[i], chunks, baselines[i], iteration);
    }
    const auto validation_errors = validate_received_data(iteration);

    report_iteration(iteration, seconds, bytes_per_peer, chunks, baselines, verification_errors, validation_errors);
    if (verification_errors != 0) {
        RDMA_PROXY_LOG_WARN("iteration=", iteration,
                            " observed ", verification_errors,
                            " missing/extra data immediate completion count(s); payload validation decides pass/fail");
    }
    if (validation_errors != 0) {
        throw std::runtime_error("iteration " + std::to_string(iteration) +
                                 " completed with " + std::to_string(validation_errors) +
                                 " payload validation errors");
    }
}

void Proxy::fill_iteration_send_buffers(uint64_t iteration) {
    if (!config_.fill_test_data) return;
    for (const auto& buffers : cuda_buffers_.peer_buffers()) {
        cuda_buffers_.fill_test_pattern(buffers.peer_rank, config_.node_rank, buffers.peer_rank, iteration);
    }
}

void Proxy::synchronize_iteration_start(uint64_t iteration) const {
    if (peers_.empty()) return;

    std::ostringstream local;
    local << "iteration_start"
          << " rank=" << config_.node_rank
          << " gpu=" << config_.local_gpu_index
          << " iteration=" << iteration;
    const auto expected_phase = "iteration_start";
    const auto expected_iteration = "iteration=" + std::to_string(iteration);

    RDMA_PROXY_LOG_INFO("iteration=", iteration, " waiting for ", peers_.size(), " peer start barrier(s)");
    for (const auto& peer : peers_) {
        const auto it = std::find_if(config_.peers.begin(), config_.peers.end(), [&](const PeerAddress& p) {
            return p.node_rank == peer.peer_rank;
        });
        if (it == config_.peers.end()) {
            throw std::runtime_error("cannot synchronize unknown peer rank " + std::to_string(peer.peer_rank));
        }

        const auto remote = connection_manager_.exchange_control_message(
            *it, local.str(), config_.completion_timeout_ms);
        if (remote.find(expected_phase) == std::string::npos ||
            remote.find(expected_iteration) == std::string::npos) {
            throw std::runtime_error("peer start synchronization mismatch: " + remote);
        }
        RDMA_PROXY_LOG_DEBUG("iteration=", iteration, " start synchronized with peer=", peer.peer_rank);
    }
    RDMA_PROXY_LOG_INFO("iteration=", iteration, " peer start barrier complete");
}

void Proxy::synchronize_iteration(uint64_t iteration) const {
    if (peers_.empty()) return;

    std::ostringstream local;
    local << "iteration_done"
          << " rank=" << config_.node_rank
          << " gpu=" << config_.local_gpu_index
          << " iteration=" << iteration;
    const auto expected_iteration = "iteration=" + std::to_string(iteration);

    RDMA_PROXY_LOG_INFO("iteration=", iteration, " waiting for ", peers_.size(), " peer synchronization barrier(s)");
    for (const auto& peer : peers_) {
        const auto it = std::find_if(config_.peers.begin(), config_.peers.end(), [&](const PeerAddress& p) {
            return p.node_rank == peer.peer_rank;
        });
        if (it == config_.peers.end()) {
            throw std::runtime_error("cannot synchronize unknown peer rank " + std::to_string(peer.peer_rank));
        }

        const auto remote = connection_manager_.exchange_control_message(
            *it, local.str(), config_.completion_timeout_ms);
        if (remote.find(expected_iteration) == std::string::npos) {
            throw std::runtime_error("peer synchronization iteration mismatch: " + remote);
        }
        RDMA_PROXY_LOG_DEBUG("iteration=", iteration, " synchronized with peer=", peer.peer_rank);
    }
    RDMA_PROXY_LOG_INFO("iteration=", iteration, " peer synchronization complete");
}

std::vector<ChunkDescriptor> Proxy::make_chunks() const {
    return compute_chunks(
        config_.num_tokens,
        config_.token_dimension,
        dtype_size(config_.dtype),
        config_.tokens_per_chunk,
        config_.num_qps_per_peer);
}

std::vector<Proxy::QPCompletionBaseline> Proxy::capture_baselines(
    const PeerState& peer,
    const std::vector<ChunkDescriptor>& chunks) const {
    std::vector<QPCompletionBaseline> baselines(peer.workers.size());
    for (std::size_t q = 0; q < peer.workers.size(); ++q) {
        const auto& worker = peer.workers[q];
        auto& baseline = baselines[q];
        baseline.sends = worker->send_completions();
        baseline.recvs = worker->recv_completions();
        baseline.send_markers = worker->send_marker_completions();
        baseline.recv_markers = worker->recv_marker_completions();
        baseline.post_errors = worker->post_errors();
        baseline.cq_errors = worker->cq_errors();
        baseline.unexpected_imms = worker->unexpected_immediate_completions();
        baseline.immediate_counts.resize(chunks.size());
        for (const auto& chunk : chunks) {
            baseline.immediate_counts[chunk.chunk_index] = worker->received_immediate_count(chunk.chunk_index);
        }
    }
    return baselines;
}

void Proxy::enqueue_chunks(
    PeerState& peer,
    const PeerGpuBuffers& buffers,
    const std::vector<ChunkDescriptor>& chunks) {

    uint64_t wr_id = 1;
    std::vector<std::size_t> posted_by_qp(peer.workers.size(), 0);
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
        const auto q = static_cast<std::size_t>(chunk.qp_index);
        const auto ordinal_on_qp = ++posted_by_qp.at(q);
        task.signaled = config_.data_signal_interval > 0 &&
                        ordinal_on_qp % static_cast<std::size_t>(config_.data_signal_interval) == 0;
        peer.workers.at(q)->enqueue(task);
    }
    for (std::size_t q = 0; q < peer.workers.size(); ++q) {
        SendTask marker;
        marker.marker = true;
        marker.wr_id = (static_cast<uint64_t>(peer.peer_rank) << 48) |
                       (static_cast<uint64_t>(q) << 32) |
                       0xffffffffULL;
        marker.signaled = true;
        peer.workers[q]->enqueue(marker);
    }
    RDMA_PROXY_LOG_INFO("enqueued ", chunks.size(), " chunks for peer ", peer.peer_rank);
}

void Proxy::wait_for_iteration(
    const PeerState& peer,
    const std::vector<ChunkDescriptor>& chunks,
    const std::vector<QPCompletionBaseline>& baselines) const {
    std::vector<std::size_t> expected_data_by_qp(peer.workers.size(), 0);
    std::vector<std::size_t> expected_send_completions_by_qp(peer.workers.size(), 0);
    for (const auto& chunk : chunks) {
        const auto q = static_cast<std::size_t>(chunk.qp_index);
        const auto ordinal_on_qp = ++expected_data_by_qp.at(q);
        if (config_.data_signal_interval > 0 &&
            ordinal_on_qp % static_cast<std::size_t>(config_.data_signal_interval) == 0) {
            ++expected_send_completions_by_qp.at(q);
        }
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.completion_timeout_ms);
    while (true) {
        bool complete = true;
        for (std::size_t q = 0; q < peer.workers.size(); ++q) {
            const auto& worker = peer.workers[q];
            if (worker->post_errors() != baselines[q].post_errors ||
                worker->cq_errors() != baselines[q].cq_errors ||
                worker->unexpected_immediate_completions() != baselines[q].unexpected_imms) {
                const auto error = worker->last_error();
                throw std::runtime_error("QP error local_rank=" + std::to_string(config_.node_rank) +
                                         " local_gpu=" + std::to_string(config_.local_gpu_index) +
                                         " remote_rank=" + std::to_string(peer.peer_rank) +
                                         " remote_gpu=" + std::to_string(peer.remote_gpu_index) +
                                         " qp=" + std::to_string(q) +
                                         (error.empty() ? "" : " last_error=" + error));
            }
            const auto expected_sends = static_cast<uint64_t>(expected_send_completions_by_qp[q]);
            if (worker->send_completions() < baselines[q].sends + expected_sends ||
                worker->send_marker_completions() < baselines[q].send_markers + 1 ||
                worker->recv_marker_completions() < baselines[q].recv_markers + 1) {
                complete = false;
            }
        }
        if (complete) return;
        if (std::chrono::steady_clock::now() >= deadline) {
            std::ostringstream out;
            out << "timed out waiting for completions local_rank=" << config_.node_rank
                << " local_gpu=" << config_.local_gpu_index
                << " remote_rank=" << peer.peer_rank
                << " remote_gpu=" << peer.remote_gpu_index;
            for (std::size_t q = 0; q < peer.workers.size(); ++q) {
                out << " qp" << q
                    << " send=" << (peer.workers[q]->send_completions() - baselines[q].sends)
                    << "/" << expected_send_completions_by_qp[q]
                    << " recv=" << (peer.workers[q]->recv_completions() - baselines[q].recvs)
                    << "/" << expected_data_by_qp[q]
                    << " send_marker=" << (peer.workers[q]->send_marker_completions() - baselines[q].send_markers)
                    << "/1"
                    << " recv_marker=" << (peer.workers[q]->recv_marker_completions() - baselines[q].recv_markers)
                    << "/1";
            }
            throw std::runtime_error(out.str());
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

std::size_t Proxy::verify_immediates(
    const PeerState& peer,
    const std::vector<ChunkDescriptor>& chunks,
    const std::vector<QPCompletionBaseline>& baselines,
    uint64_t iteration) const {
    std::size_t errors = 0;
    for (const auto& chunk : chunks) {
        const auto q = static_cast<std::size_t>(chunk.qp_index);
        const auto observed = peer.workers[q]->received_immediate_count(chunk.chunk_index);
        const auto expected = baselines[q].immediate_counts[chunk.chunk_index] + 1;
        if (observed != expected) {
            ++errors;
            RDMA_PROXY_LOG_WARN("iteration=", iteration,
                                " peer=", peer.peer_rank,
                                " qp=", chunk.qp_index,
                                " chunk=", chunk.chunk_index,
                                " immediate_count=", observed,
                                " expected=", expected);
        }
    }
    return errors;
}

std::size_t Proxy::validate_received_data(uint64_t iteration) const {
    if (!config_.validate_data) return 0;
    std::size_t errors = 0;
    for (const auto& buffers : cuda_buffers_.peer_buffers()) {
        std::string error;
        const int expected_source = config_.mock_mode ? config_.node_rank : buffers.peer_rank;
        const int expected_destination = config_.mock_mode ? buffers.peer_rank : config_.node_rank;
        if (!cuda_buffers_.validate_recv_pattern(
                buffers.peer_rank, expected_source, expected_destination, iteration, &error)) {
            ++errors;
            RDMA_PROXY_LOG_ERROR("receive validation failed: ", error);
        }
    }
    return errors;
}

void Proxy::report_iteration(
    uint64_t iteration,
    double seconds,
    std::size_t bytes_per_peer,
    const std::vector<ChunkDescriptor>& chunks,
    const std::vector<std::vector<QPCompletionBaseline>>& baselines,
    std::size_t verification_errors,
    std::size_t validation_errors) const {
    const auto total_bytes = bytes_per_peer * peers_.size();
    const double gbps = seconds > 0.0 ? (static_cast<double>(total_bytes) * 8.0 / seconds / 1.0e9) : 0.0;
    const double latency_us = seconds * 1.0e6;

    RDMA_PROXY_LOG_INFO("iteration=", iteration,
                        " local_rank=", config_.node_rank,
                        " local_gpu=", config_.local_gpu_index,
                        " total_bytes=", total_bytes,
                        " elapsed_us=", static_cast<uint64_t>(latency_us),
                        " bandwidth_gbps=", std::fixed, std::setprecision(3), gbps,
                        " immediate_mismatches=", verification_errors,
                        " validation_errors=", validation_errors);

    std::vector<std::size_t> bytes_by_qp(static_cast<std::size_t>(config_.num_qps_per_peer), 0);
    std::vector<std::size_t> expected_send_completions_by_qp(
        static_cast<std::size_t>(config_.num_qps_per_peer), 0);
    std::vector<std::size_t> chunks_by_qp(static_cast<std::size_t>(config_.num_qps_per_peer), 0);
    for (const auto& chunk : chunks) {
        const auto q = static_cast<std::size_t>(chunk.qp_index);
        bytes_by_qp.at(q) += chunk.length_bytes;
        const auto ordinal_on_qp = ++chunks_by_qp.at(q);
        if (config_.data_signal_interval > 0 &&
            ordinal_on_qp % static_cast<std::size_t>(config_.data_signal_interval) == 0) {
            ++expected_send_completions_by_qp.at(q);
        }
    }

    for (std::size_t peer_index = 0; peer_index < peers_.size(); ++peer_index) {
        const auto& peer = peers_[peer_index];
        for (std::size_t q = 0; q < peer.workers.size(); ++q) {
            const auto& worker = peer.workers[q];
            const auto send_delta = worker->send_completions() - baselines[peer_index][q].sends;
            const auto recv_delta = worker->recv_completions() - baselines[peer_index][q].recvs;
            const auto send_marker_delta =
                worker->send_marker_completions() - baselines[peer_index][q].send_markers;
            const auto recv_marker_delta =
                worker->recv_marker_completions() - baselines[peer_index][q].recv_markers;
            const auto post_error_delta = worker->post_errors() - baselines[peer_index][q].post_errors;
            const auto cq_error_delta = worker->cq_errors() - baselines[peer_index][q].cq_errors;
            const auto unexpected_delta =
                worker->unexpected_immediate_completions() - baselines[peer_index][q].unexpected_imms;
            const auto errors = post_error_delta + cq_error_delta + unexpected_delta;
            const double qp_gbps = seconds > 0.0 ?
                (static_cast<double>(bytes_by_qp[q]) * 8.0 / seconds / 1.0e9) : 0.0;
            RDMA_PROXY_LOG_INFO("qp_report iteration=", iteration,
                                " local_rank=", config_.node_rank,
                                " local_gpu=", config_.local_gpu_index,
                                " peer=", peer.peer_rank,
                                " remote_rank=", peer.peer_rank,
                                " remote_gpu=", peer.remote_gpu_index,
                                " qp=", q,
                                " bytes=", bytes_by_qp[q],
                                " elapsed_us=", static_cast<uint64_t>(latency_us),
                                " bandwidth_gbps=", std::fixed, std::setprecision(3), qp_gbps,
                                " send_completions=", send_delta,
                                " expected_data_send_completions=", expected_send_completions_by_qp[q],
                                " recv_immediate_completions=", recv_delta,
                                " send_marker_completions=", send_marker_delta,
                                " recv_marker_completions=", recv_marker_delta,
                                " post_errors=", post_error_delta,
                                " cq_errors=", cq_error_delta,
                                " unexpected_immediates=", unexpected_delta,
                                " errors=", errors);
        }
    }
}

}  // namespace rdma_proxy
