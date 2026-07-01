#include "qp_worker.hpp"

#include "logging.hpp"

#include <chrono>
#include <stdexcept>

namespace rdma_proxy {
namespace {

constexpr uint64_t kMarkerWrIdBit = 1ULL << 63;

bool is_marker_wr_id(uint64_t wr_id) {
    return (wr_id & kMarkerWrIdBit) != 0;
}

}  // namespace

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

QPWorker::QPWorker(RdmaQueuePair& qp, int poll_batch_size)
    : qp_(qp), poll_batch_size_(poll_batch_size) {}

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
        } catch (const std::exception& e) {
            post_errors_.fetch_add(1);
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                last_error_ = e.what();
            }
            RDMA_PROXY_LOG_ERROR("send worker peer=", qp_.peer_rank(), " qp=", qp_.qp_index(), " failed: ", e.what());
            stop_.store(true);
        }
    }
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
        }
    }
}

}  // namespace rdma_proxy
