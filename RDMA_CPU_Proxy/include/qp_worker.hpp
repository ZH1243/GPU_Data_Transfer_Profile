#pragma once

#include "protocol.hpp"
#include "rdma_connection.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>

namespace rdma_proxy {

struct SendTask {
    uint64_t wr_id{0};
    ChunkDescriptor chunk;
    uintptr_t local_base{0};
    uint32_t local_lkey{0};
    uintptr_t remote_base{0};
    uint32_t remote_rkey{0};
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

    uint64_t send_completions() const { return send_completions_.load(); }
    uint64_t recv_completions() const { return recv_completions_.load(); }

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
};

}  // namespace rdma_proxy
