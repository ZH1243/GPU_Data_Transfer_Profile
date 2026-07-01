#include "qp_worker.hpp"

#include "logging.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace rdma_proxy {
namespace {

constexpr uint64_t kMarkerWrIdBit = 1ULL << 63;

bool is_marker_wr_id(uint64_t wr_id) {
    return (wr_id & kMarkerWrIdBit) != 0;
}

}  // namespace

std::size_t IterationAssignment::assigned_chunks() const {
    std::size_t total = 0;
    for (const auto count : chunks_by_qp) total += count;
    return total;
}

void SendQueue::push(SendTask task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) throw std::runtime_error("cannot push to closed send queue");
        queue_.push(task);
    }
    cv_.notify_one();
}

bool SendQueue::pop(SendTask& task, const std::atomic<bool>& stop) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return stop.load() || closed_ || !queue_.empty(); });
    if (queue_.empty()) return false;
    task = queue_.front();
    queue_.pop();
    return true;
}

void SendQueue::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    cv_.notify_all();
}

DynamicChunkDistributor::DynamicChunkDistributor(
    std::vector<ChunkDescriptor> chunks,
    int peer_rank,
    std::size_t qp_count,
    uintptr_t local_base,
    uint32_t local_lkey,
    uintptr_t remote_base,
    uint32_t remote_rkey)
    : chunks_(std::move(chunks)),
      peer_rank_(peer_rank),
      local_base_(local_base),
      local_lkey_(local_lkey),
      remote_base_(remote_base),
      remote_rkey_(remote_rkey) {
    assignment_.qp_by_chunk.assign(chunks_.size(), -1);
    assignment_.chunks_by_qp.assign(qp_count, 0);
    assignment_.bytes_by_qp.assign(qp_count, 0);
    assignment_.expected_send_completions_by_qp.assign(qp_count, 0);
}

bool DynamicChunkDistributor::next(int qp_index, SendTask& task) {
    if (qp_index < 0) throw std::runtime_error("invalid QP index");
    const auto q = static_cast<std::size_t>(qp_index);

    std::lock_guard<std::mutex> lock(mutex_);
    if (q >= assignment_.chunks_by_qp.size()) {
        throw std::runtime_error("QP index exceeds dynamic distributor size");
    }
    if (next_chunk_ >= chunks_.size()) return false;

    auto chunk = chunks_[next_chunk_++];
    chunk.qp_index = qp_index;
    if (chunk.chunk_index >= assignment_.qp_by_chunk.size()) {
        throw std::runtime_error("chunk index exceeds dynamic assignment size");
    }

    assignment_.qp_by_chunk[chunk.chunk_index] = qp_index;
    ++assignment_.chunks_by_qp[q];
    assignment_.bytes_by_qp[q] += chunk.length_bytes;
    ++assignment_.expected_send_completions_by_qp[q];

    task = SendTask{};
    task.wr_id = (static_cast<uint64_t>(peer_rank_) << 48) |
                 (static_cast<uint64_t>(q) << 32) |
                 static_cast<uint64_t>(chunk.chunk_index);
    if (task.wr_id == 0) task.wr_id = 1;
    task.chunk = chunk;
    task.local_base = local_base_;
    task.local_lkey = local_lkey_;
    task.remote_base = remote_base_;
    task.remote_rkey = remote_rkey_;
    task.signaled = true;
    return true;
}

uint64_t DynamicChunkDistributor::marker_wr_id(int qp_index) const {
    if (qp_index < 0) throw std::runtime_error("invalid QP index");
    return (static_cast<uint64_t>(peer_rank_) << 48) |
           (static_cast<uint64_t>(static_cast<std::size_t>(qp_index)) << 32) |
           0xffffffffULL;
}

IterationAssignment DynamicChunkDistributor::assignment() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return assignment_;
}

std::size_t DynamicChunkDistributor::chunk_count() const {
    return chunks_.size();
}

QPWorker::QPWorker(RdmaQueuePair& qp, int poll_batch_size, int max_in_flight_chunks)
    : qp_(qp),
      poll_batch_size_(poll_batch_size),
      max_in_flight_chunks_(std::max(1, max_in_flight_chunks)) {}

QPWorker::~QPWorker() {
    stop();
}

void QPWorker::start() {
    stop_.store(false);
    send_thread_ = std::thread(&QPWorker::send_loop, this);
    cq_thread_ = std::thread(&QPWorker::cq_loop, this);
}

void QPWorker::stop() {
    stop_.store(true);
    send_queue_.close();
    completion_cv_.notify_all();
    if (send_thread_.joinable()) send_thread_.join();
    if (cq_thread_.joinable()) cq_thread_.join();
}

void QPWorker::enqueue(SendTask task) {
    send_queue_.push(task);
}

void QPWorker::post_initial_receives(int recv_queue_depth) {
    for (int i = 0; i < recv_queue_depth; ++i) {
        qp_.post_receive(next_recv_wr_id_.fetch_add(1));
    }
}

void QPWorker::configure_expected_chunks(std::size_t num_chunks) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    immediate_counts_.assign(num_chunks, 0);
}

uint64_t QPWorker::received_immediate_count(std::size_t chunk_index) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    if (chunk_index >= immediate_counts_.size()) return 0;
    return immediate_counts_[chunk_index];
}

std::chrono::steady_clock::time_point QPWorker::latest_send_marker_time() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return latest_send_marker_time_;
}

std::chrono::steady_clock::time_point QPWorker::latest_recv_marker_time() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return latest_recv_marker_time_;
}

std::string QPWorker::last_error() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return last_error_;
}

void QPWorker::send_loop() {
    while (!stop_.load()) {
        SendTask task;
        if (!send_queue_.pop(task, stop_)) {
            continue;
        }
        try {
            if (task.distributor) {
                run_dynamic_iteration(task.distributor);
            } else {
                post_task(task);
            }
        } catch (const std::exception& e) {
            post_errors_.fetch_add(1);
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                last_error_ = e.what();
            }
            RDMA_PROXY_LOG_ERROR("send worker peer=", qp_.peer_rank(), " qp=", qp_.qp_index(), " failed: ", e.what());
            stop_.store(true);
            completion_cv_.notify_all();
        }
    }
}

void QPWorker::post_task(const SendTask& task) {
    if (task.marker) {
        qp_.post_send_with_immediate(task.wr_id | kMarkerWrIdBit, encode_marker_immediate());
    } else {
        qp_.post_write_with_immediate(
            task.wr_id,
            task.local_base + task.chunk.src_offset_bytes,
            task.local_lkey,
            task.remote_base + task.chunk.dst_offset_bytes,
            task.remote_rkey,
            task.chunk.length_bytes,
            task.chunk.imm_data,
            task.signaled);
    }
}

void QPWorker::run_dynamic_iteration(const std::shared_ptr<DynamicChunkDistributor>& distributor) {
    const auto max_in_flight = static_cast<std::size_t>(max_in_flight_chunks_);
    std::size_t in_flight = 0;
    uint64_t observed_send_completions = send_completions_.load();
    bool distributor_drained = false;

    auto retire_completed = [&] {
        const auto completed = send_completions_.load();
        if (completed <= observed_send_completions) return;

        const auto delta = static_cast<std::size_t>(completed - observed_send_completions);
        in_flight = delta >= in_flight ? 0 : in_flight - delta;
        observed_send_completions = completed;
    };

    while (!stop_.load()) {
        retire_completed();

        while (!stop_.load() && !distributor_drained && in_flight < max_in_flight) {
            SendTask task;
            if (!distributor->next(qp_.qp_index(), task)) {
                distributor_drained = true;
                break;
            }
            post_task(task);
            ++in_flight;
        }

        if (distributor_drained && in_flight == 0) break;
        if (!wait_for_send_completion(observed_send_completions)) return;
    }

    SendTask marker;
    marker.marker = true;
    marker.wr_id = distributor->marker_wr_id(qp_.qp_index());
    marker.signaled = true;
    post_task(marker);
}

bool QPWorker::wait_for_send_completion(uint64_t baseline) {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, [&] {
        return stop_.load() ||
               send_completions_.load() > baseline ||
               post_errors_.load() != 0 ||
               cq_errors_.load() != 0;
    });
    return !stop_.load() && send_completions_.load() > baseline;
}

void QPWorker::cq_loop() {
    std::vector<Completion> completions;
    completions.reserve(static_cast<std::size_t>(poll_batch_size_));

    while (!stop_.load()) {
        try {
            completions.clear();
            const int n = qp_.poll(completions, poll_batch_size_);
            if (n == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            for (const auto& c : completions) {
                if (c.kind == CompletionKind::kSend) {
                    if (is_marker_wr_id(c.wr_id)) {
                        send_marker_completions_.fetch_add(1);
                        {
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            latest_send_marker_time_ = std::chrono::steady_clock::now();
                        }
                    } else {
                        send_completions_.fetch_add(1);
                        completion_cv_.notify_all();
                    }
                } else {
                    if (is_marker_immediate(c.imm_data)) {
                        recv_marker_completions_.fetch_add(1);
                        {
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            latest_recv_marker_time_ = std::chrono::steady_clock::now();
                        }
                    } else {
                        recv_completions_.fetch_add(1);
                        const auto chunk_index = decode_immediate(c.imm_data);
                        {
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            if (chunk_index < immediate_counts_.size()) {
                                ++immediate_counts_[chunk_index];
                            } else {
                                unexpected_immediate_completions_.fetch_add(1);
                            }
                        }
                        RDMA_PROXY_LOG_DEBUG("recv imm completion peer=", qp_.peer_rank(),
                                             " qp=", qp_.qp_index(),
                                             " chunk=", chunk_index);
                    }
                    qp_.post_receive(next_recv_wr_id_.fetch_add(1));
                }
            }
        } catch (const std::exception& e) {
            cq_errors_.fetch_add(1);
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                last_error_ = e.what();
            }
            RDMA_PROXY_LOG_ERROR("CQ worker peer=", qp_.peer_rank(), " qp=", qp_.qp_index(), " failed: ", e.what());
            stop_.store(true);
            completion_cv_.notify_all();
        }
    }
}

}  // namespace rdma_proxy
