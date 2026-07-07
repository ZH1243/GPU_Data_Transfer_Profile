#pragma once

#include "protocol.hpp"
#include "rdma_connection.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace rdma_proxy {

class DynamicChunkDistributor;

struct IterationAssignment {
    std::vector<int> qp_by_chunk;
    std::vector<std::size_t> chunks_by_qp;
    std::vector<std::size_t> bytes_by_qp;
    std::vector<std::size_t> expected_send_completions_by_qp;

    std::size_t assigned_chunks() const;
};

struct SendTask {
    uint64_t wr_id{0};
    ChunkDescriptor chunk;
    uintptr_t local_base{0};
    uint32_t local_lkey{0};
    uintptr_t remote_base{0};
    uint32_t remote_rkey{0};
    bool signaled{true};
    bool marker{false};
    std::shared_ptr<DynamicChunkDistributor> distributor;
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

class DynamicChunkDistributor {
public:
    DynamicChunkDistributor(
        std::vector<ChunkDescriptor> chunks,
        int peer_rank,
        std::size_t qp_count,
        uintptr_t local_base,
        uint32_t local_lkey,
        uintptr_t remote_base,
        uint32_t remote_rkey);

    bool next(int qp_index, SendTask& task);
    void make_marker(int qp_index, SendTask& task) const;
    IterationAssignment assignment() const;
    std::size_t chunk_count() const;

private:
    mutable std::mutex mutex_;
    std::vector<ChunkDescriptor> chunks_;
    int peer_rank_{-1};
    uintptr_t local_base_{0};
    uint32_t local_lkey_{0};
    uintptr_t remote_base_{0};
    uint32_t remote_rkey_{0};
    std::size_t next_chunk_{0};
    IterationAssignment assignment_;
};

class QPWorker {
public:
    QPWorker(RdmaQueuePair& qp, int poll_batch_size, int max_in_flight_chunks);
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
    std::vector<uint64_t> received_immediate_counts() const;
    std::chrono::steady_clock::time_point latest_send_marker_time() const;
    std::chrono::steady_clock::time_point latest_recv_marker_time() const;
    std::string last_error() const;

private:
    void send_loop();
    void cq_loop();
    void post_task(const SendTask& task);
    void run_dynamic_iteration(const std::shared_ptr<DynamicChunkDistributor>& distributor);
    bool wait_for_send_completion(uint64_t baseline);

    RdmaQueuePair& qp_;
    int poll_batch_size_{16};
    int max_in_flight_chunks_{1};
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
    std::condition_variable completion_cv_;
    std::mutex completion_mutex_;
    mutable std::mutex stats_mutex_;
    std::vector<uint64_t> immediate_counts_;
    std::chrono::steady_clock::time_point latest_send_marker_time_{};
    std::chrono::steady_clock::time_point latest_recv_marker_time_{};
    std::string last_error_;
};

}  // namespace rdma_proxy
