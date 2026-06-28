#include "qp_worker.hpp"

#include "logging.hpp"

#include <chrono>
#include <stdexcept>

namespace rdma_proxy {

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

void QPWorker::send_loop() {
    while (!stop_.load()) {
        SendTask task;
        if (!send_queue_.pop(task, stop_)) {
            continue;
        }
        try {
            qp_.post_write_with_immediate(
                task.wr_id,
                task.local_base + task.chunk.src_offset_bytes,
                task.local_lkey,
                task.remote_base + task.chunk.dst_offset_bytes,
                task.remote_rkey,
                task.chunk.length_bytes,
                task.chunk.imm_data);
        } catch (const std::exception& e) {
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
                    send_completions_.fetch_add(1);
                } else {
                    recv_completions_.fetch_add(1);
                    RDMA_PROXY_LOG_DEBUG("recv imm completion peer=", qp_.peer_rank(),
                                         " qp=", qp_.qp_index(),
                                         " chunk=", decode_immediate(c.imm_data));
                    qp_.post_receive(next_recv_wr_id_.fetch_add(1));
                }
            }
        } catch (const std::exception& e) {
            RDMA_PROXY_LOG_ERROR("CQ worker peer=", qp_.peer_rank(), " qp=", qp_.qp_index(), " failed: ", e.what());
            stop_.store(true);
        }
    }
}

}  // namespace rdma_proxy
