#pragma once

#include "config.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rdma_proxy {

class CudaBuffers;

enum class ForwardTaskType : uint32_t {
    kCompute = 1,
    kExit = 2,
};

// One immutable, self-contained output-tile command. Raw pointers are GPU
// virtual addresses whose lifetime is owned by CudaBuffers. Stable buffer IDs
// and generation are retained for diagnostics and stale-task rejection.
struct alignas(128) ForwardComputeTask {
    uint32_t type{0};
    uint32_t dtype{0};
    uint64_t generation{0};
    uint32_t queue_id{0};
    uint32_t receive_buffer_id{0};
    uint32_t output_buffer_id{0};
    uint32_t token_row_offset{0};
    uint32_t valid_token_rows{0};
    uint32_t output_column_offset{0};
    uint32_t valid_output_columns{0};
    uint32_t tile_m{0};
    uint32_t tile_n{0};
    uint32_t matrix_m{0};
    uint32_t matrix_n{0};
    uint32_t matrix_k{0};
    uint32_t flags{0};
    uint64_t a_base{0};
    uint64_t b_base{0};
    uint64_t d_base{0};
    uint64_t enqueue_timestamp_ns{0};
    uint64_t reserved[3]{0, 0, 0};
};

static_assert(sizeof(ForwardComputeTask) == 128, "ForwardComputeTask must occupy one 128-byte cache line");

// Protocol reference: UCCL-EP ep/include/fifo_device.hpp and its d2h_queue
// wrappers. This is an independent reverse-direction adaptation: one CPU
// producer publishes work for multiple GPU consumers. As in UCCL-EP, the hot
// poll target lives in memory local to the consumer: published_head and the
// task ring are in device memory, while consumed_sequence is mapped pinned host
// memory. The host stages a task and copies it to the device ring before it
// advances the device-resident published head. The GPU writes the
// host-resident reuse sequence only after the task has completed.
struct alignas(128) ForwardQueueSignal {
    uint64_t sequence{0};
    uint8_t padding[120]{};
};

struct ForwardDeviceQueueStats {
    uint64_t poll_iterations{0};
    uint64_t tasks_claimed{0};
    uint64_t tasks_completed{0};
    uint64_t exit_tasks_consumed{0};
    uint64_t stale_tasks{0};
    uint64_t invalid_tasks{0};
};

// Plain pointers in this view are device addresses. published_head, tasks,
// dequeue_position, and stats use device memory. consumed is a device mapping
// of pinned host memory so the CPU can observe slot reuse without a D2H copy.
struct ForwardDeviceQueueView {
    uint64_t* published_head{nullptr};
    ForwardQueueSignal* consumed{nullptr};
    ForwardComputeTask* tasks{nullptr};
    uint64_t* dequeue_position{nullptr};
    ForwardDeviceQueueStats* stats{nullptr};
    uint32_t capacity{0};
    uint32_t queue_id{0};
    uint64_t generation{0};
};

struct ForwardComputationStats {
    uint64_t generated_tasks{0};
    uint64_t queue_full_stalls{0};
    uint64_t poll_iterations{0};
    uint64_t tasks_claimed{0};
    uint64_t tasks_completed{0};
    uint64_t exit_tasks_consumed{0};
    uint64_t stale_tasks{0};
    uint64_t invalid_tasks{0};
    double iteration_seconds{0.0};
    double enqueue_seconds{0.0};
    std::vector<uint64_t> generated_by_queue;
    std::vector<uint64_t> completed_by_queue;
    std::vector<uint32_t> ctas_per_queue;
    std::vector<uint32_t> physical_sm_ids;
};

struct ForwardReadyRegion {
    uint64_t generation{0};
    uint32_t receive_buffer_id{0};
    uint32_t output_buffer_id{0};
    uint32_t token_row_offset{0};
    uint32_t valid_token_rows{0};
    uint32_t matrix_n{0};
    uint32_t matrix_k{0};
    uint32_t tile_m{0};
    uint32_t tile_n{0};
    uint32_t dtype{0};
    uint64_t a_base{0};
    uint64_t b_base{0};
    uint64_t d_base{0};
};

std::vector<ForwardComputeTask> partition_forward_ready_region(const ForwardReadyRegion& region);
std::vector<uint32_t> partition_ctas_across_queues(uint32_t num_ctas, uint32_t num_queues);

class ForwardComputation {
public:
    ForwardComputation(ProxyConfig config, CudaBuffers& buffers);
    ~ForwardComputation();

    ForwardComputation(const ForwardComputation&) = delete;
    ForwardComputation& operator=(const ForwardComputation&) = delete;

    void initialize();
    void begin_iteration(uint64_t iteration);
    std::size_t enqueue_ready_region(
        uint64_t iteration,
        int source_gpu,
        std::size_t peer_slot,
        std::size_t start_token,
        std::size_t num_tokens,
        std::size_t byte_offset);
    void finish_iteration(uint64_t iteration);
    void abort_iteration() noexcept;
    void shutdown() noexcept;

    bool wait_until_completed(uint64_t count, std::chrono::milliseconds timeout) const;
    ForwardComputationStats stats() const;
    bool iteration_active() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rdma_proxy
