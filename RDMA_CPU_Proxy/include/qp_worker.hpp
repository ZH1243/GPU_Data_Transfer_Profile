#pragma once

#include "protocol.hpp"
#include "rdma_connection.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace rdma_proxy {

struct SendTask {
    uint64_t wr_id{0};
    ChunkDescriptor chunk;
    uintptr_t local_base{0};
    uint32_t local_lkey{0};
    uintptr_t remote_base{0};
    uint32_t remote_rkey{0};
    bool signaled{true};
    bool marker{false};
};

class SendQueue {
public:
    void push(SendTask task);
    bool pop(SendTask& task, const std::atomic<bool>& stop);
    void close();

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<SendTask> queue_;
    bool closed_{false};
};

class QPWorker {
public:
    QPWorker(RdmaQueuePair& qp, int poll_batch_size);
    ~QPWorker();

    QPWorker(const QPWorker&) = delete;
    QPWorker& operator=(const QPWorker&) = delete;

    void start();
    void stop();
    void enqueue(SendTask task);
    void post_initial_receives(int recv_queue_depth);
    void configure_expected_chunks(std::size_t num_chunks);

    uint64_t send_completions() const { return send_completions_.load(); }
    uint64_t recv_completions() const { return recv_completions_.load(); }
    uint64_t send_marker_completions() const { return send_marker_completions_.load(); }
    uint64_t recv_marker_completions() const { return recv_marker_completions_.load(); }
    uint64_t post_errors() const { return post_errors_.load(); }
    uint64_t cq_errors() const { return cq_errors_.load(); }
    uint64_t unexpected_immediate_completions() const { return unexpected_immediate_completions_.load(); }
    uint64_t received_immediate_count(std::size_t chunk_index) const;
    std::chrono::steady_clock::time_point latest_send_marker_time() const;
    std::chrono::steady_clock::time_point latest_recv_marker_time() const;
    std::string last_error() const;

private:
    void send_loop();
    void cq_loop();

    RdmaQueuePair& qp_;
    int poll_batch_size_{16};
    SendQueue send_queue_;
    std::atomic<bool> stop_{false};
    std::thread send_thread_;
    std::thread cq_thread_;
    std::atomic<uint64_t> next_recv_wr_id_{1};
    std::atomic<uint64_t> send_completions_{0};
    std::atomic<uint64_t> recv_completions_{0};
    std::atomic<uint64_t> send_marker_completions_{0};
    std::atomic<uint64_t> recv_marker_completions_{0};
    std::atomic<uint64_t> post_errors_{0};
    std::atomic<uint64_t> cq_errors_{0};
    std::atomic<uint64_t> unexpected_immediate_completions_{0};
    mutable std::mutex stats_mutex_;
    std::vector<uint64_t> immediate_counts_;
    std::chrono::steady_clock::time_point latest_send_marker_time_{};
    std::chrono::steady_clock::time_point latest_recv_marker_time_{};
    std::string last_error_;
};

}  // namespace rdma_proxy
