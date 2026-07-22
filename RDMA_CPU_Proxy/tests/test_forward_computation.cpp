#include "cuda_buffers.hpp"
#include "forward_computation.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

rdma_proxy::ProxyConfig make_config() {
    rdma_proxy::ProxyConfig config;
    config.node_rank = 0;
    config.num_nodes = 2;
    config.local_gpu_index = 0;
    config.num_gpus_per_node = 3;
    config.num_tokens = 32;
    config.token_dimension = 16;
    config.tokens_per_chunk = 16;
    config.num_qps_per_peer = 1;
    config.dtype = rdma_proxy::DataType::kBF16;
    config.mock_mode = true;
    config.nvlink_forwarding_enabled = true;
    config.nvlink_forward_completion_notifications_enabled = true;
    config.nvlink_forward_computation_enabled = true;
    config.nvlink_forward_computation_output_dim = 32;
    config.nvlink_forward_computation_tile_m = 16;
    config.nvlink_forward_computation_tile_n = 16;
    config.nvlink_forward_computation_num_queues = 2;
    config.nvlink_forward_computation_queue_depth = 2;
    config.completion_timeout_ms = 5000;
    config.peers.push_back(rdma_proxy::PeerAddress{1, "mock", 1});
    return config;
}

void fill_input(rdma_proxy::CudaBuffers& buffers, int source_gpu, const rdma_proxy::ProxyConfig& config) {
    const auto& receive = buffers.nvlink_receive_buffer_for_source(source_gpu);
    auto* values = static_cast<uint16_t*>(receive.recv.ptr);
    for (std::size_t row = 0; row < config.num_tokens; ++row) {
        for (std::size_t k = 0; k < config.token_dimension; ++k) {
            const int bucket = static_cast<int>(
                (row * 3 + k * 5 + static_cast<std::size_t>(source_gpu)) % 17) - 8;
            const float value = static_cast<float>(bucket) / 16.0F;
            values[row * config.token_dimension + k] = float_to_bf16(value);
        }
    }
}

void validate_output(
    const rdma_proxy::CudaBuffers& buffers,
    int source_gpu,
    const rdma_proxy::ProxyConfig& config) {
    const auto* a = static_cast<const uint16_t*>(
        buffers.nvlink_receive_buffer_for_source(source_gpu).recv.ptr);
    const auto* b = static_cast<const uint16_t*>(buffers.nvlink_computation_weight_buffer().ptr);
    const auto* d = static_cast<const uint16_t*>(
        buffers.nvlink_computation_output_buffer_for_source(source_gpu).output.ptr);
    for (std::size_t row = 0; row < config.num_tokens; ++row) {
        for (std::size_t column = 0; column < config.nvlink_forward_computation_output_dim; ++column) {
            float expected = 0.0F;
            for (std::size_t k = 0; k < config.token_dimension; ++k) {
                expected += bf16_to_float(a[row * config.token_dimension + k]) *
                            bf16_to_float(b[k * config.nvlink_forward_computation_output_dim + column]);
            }
            const float actual = bf16_to_float(
                d[row * config.nvlink_forward_computation_output_dim + column]);
            const float rounded_expected = bf16_to_float(float_to_bf16(expected));
            if (std::fabs(actual - rounded_expected) > 1.0e-3F) {
                std::cerr << "matrix mismatch source=" << source_gpu
                          << " row=" << row << " column=" << column
                          << " expected=" << rounded_expected << " actual=" << actual << '\n';
                std::abort();
            }
        }
    }
}

}  // namespace

int main() {
    {
        rdma_proxy::ForwardReadyRegion example;
        example.generation = 1;
        example.receive_buffer_id = 1;
        example.output_buffer_id = 1;
        example.valid_token_rows = 256;
        example.matrix_n = 6400;
        example.matrix_k = 4096;
        example.tile_m = 128;
        example.tile_n = 128;
        example.dtype = static_cast<uint32_t>(rdma_proxy::DataType::kBF16);
        example.a_base = 0x1000;
        example.b_base = 0x2000;
        example.d_base = 0x3000;
        const auto tasks = rdma_proxy::partition_forward_ready_region(example);
        assert(tasks.size() == 100);
        assert(tasks.front().valid_token_rows == 128);
        assert(tasks.back().output_column_offset == 6272);
    }

    {
        const auto assignment = rdma_proxy::partition_ctas_across_queues(5, 2);
        assert(assignment.size() == 2);
        assert(assignment[0] == 3);
        assert(assignment[1] == 2);
    }

    {
        assert(rdma_proxy::forward_queue_available_slots(0, 0, 8192) == 8192);
        assert(rdma_proxy::forward_queue_available_slots(100, 0, 8192) == 8092);
        assert(rdma_proxy::forward_queue_available_slots(100, 75, 8192) == 8167);
        assert(rdma_proxy::forward_queue_available_slots(8192, 0, 8192) == 0);
        bool rejected_invalid_tail = false;
        try {
            (void)rdma_proxy::forward_queue_available_slots(10, 11, 8192);
        } catch (const std::runtime_error&) {
            rejected_invalid_tail = true;
        }
        assert(rejected_invalid_tail);
    }

    {
        const auto balanced = rdma_proxy::allocate_forward_tasks_for_wave(
            {200, 200, 200, 200, 200, 200, 200, 200}, 1000);
        assert(balanced == std::vector<std::size_t>({125, 125, 125, 125, 125, 125, 125, 125}));

        const auto capped = rdma_proxy::allocate_forward_tasks_for_wave(
            {50, 200, 200, 200, 200, 200, 200, 200}, 1000);
        assert(capped[0] == 50);
        std::size_t capped_total = 0;
        for (std::size_t q = 0; q < capped.size(); ++q) {
            capped_total += capped[q];
            if (q != 0) assert(capped[q] == 135 || capped[q] == 136);
        }
        assert(capped_total == 1000);

        const auto capacity_limited = rdma_proxy::allocate_forward_tasks_for_wave(
            {100, 100, 100, 100, 100, 100, 100, 100}, 1000);
        assert(capacity_limited == std::vector<std::size_t>({100, 100, 100, 100, 100, 100, 100, 100}));
    }

    auto config = make_config();
    rdma_proxy::CudaBuffers buffers(config);
    buffers.initialize();
    fill_input(buffers, 1, config);
    fill_input(buffers, 2, config);

    rdma_proxy::ForwardComputation computation(config, buffers);
    computation.initialize();
    computation.begin_iteration(0);

    const auto row_bytes = config.token_dimension * rdma_proxy::dtype_size(config.dtype);
    const auto first_tasks = computation.enqueue_ready_region(0, 1, 0, 0, 15, 0);
    assert(first_tasks == 2);
    // This proves the first notification is consumed before the final ready
    // notification for the same receive buffer is published.
    assert(computation.wait_until_completed(first_tasks, std::chrono::milliseconds(2000)));

    assert(computation.enqueue_ready_region(0, 1, 0, 15, 17, 15 * row_bytes) == 4);
    assert(computation.enqueue_ready_region(0, 2, 0, 0, 32, 0) == 4);
    computation.finish_iteration(0);
    const auto first_stats = computation.stats();
    assert(first_stats.generated_tasks == 10);
    assert(first_stats.publication_waves == 0);
    assert(first_stats.descriptor_batch_calls == 0);
    assert(first_stats.head_batch_calls == 0);
    assert(first_stats.tasks_claimed == 10);
    assert(first_stats.tasks_completed == 10);
    assert(first_stats.exit_tasks_consumed == 4);
    assert(first_stats.stale_tasks == 0);
    assert(first_stats.invalid_tasks == 0);
    validate_output(buffers, 1, config);
    validate_output(buffers, 2, config);

    // Reuse all rings and buffers with a new generation. Queue depth two forces
    // wraparound repeatedly while four mock CTAs consume two queues.
    computation.begin_iteration(1);
    assert(computation.enqueue_ready_region(1, 1, 0, 0, 32, 0) == 4);
    computation.finish_iteration(1);
    const auto second_stats = computation.stats();
    assert(second_stats.generated_tasks == 4);
    assert(second_stats.tasks_completed == 4);
    assert(second_stats.exit_tasks_consumed == 4);
    assert(second_stats.stale_tasks == 0);
    validate_output(buffers, 1, config);
    const auto* untouched = static_cast<const uint16_t*>(
        buffers.nvlink_computation_output_buffer_for_source(2).output.ptr);
    for (std::size_t i = 0; i < config.num_tokens * config.nvlink_forward_computation_output_dim; ++i) {
        assert(untouched[i] == 0);
    }

    computation.shutdown();

    // Opt-in wave batching redistributes each notification according to the
    // currently reusable slots. Eight tasks cannot fit in the combined four
    // queue slots, so publication requires at least two descriptor/head waves.
    auto wave_config = make_config();
    wave_config.nvlink_forward_computation_output_dim = 64;
    wave_config.nvlink_forward_computation_wave_batching_enabled = true;
    rdma_proxy::CudaBuffers wave_buffers(wave_config);
    wave_buffers.initialize();
    fill_input(wave_buffers, 1, wave_config);
    rdma_proxy::ForwardComputation wave_computation(wave_config, wave_buffers);
    wave_computation.initialize();
    wave_computation.begin_iteration(10);
    assert(wave_computation.enqueue_ready_region(10, 1, 0, 0, 32, 0) == 8);
    const auto active_wave_stats = wave_computation.stats();
    assert(active_wave_stats.publication_waves >= 2);
    assert(active_wave_stats.descriptor_batch_calls == active_wave_stats.publication_waves);
    assert(active_wave_stats.head_batch_calls == active_wave_stats.publication_waves);
    wave_computation.finish_iteration(10);
    const auto wave_stats = wave_computation.stats();
    assert(wave_stats.generated_tasks == 8);
    assert(wave_stats.tasks_completed == 8);
    assert(wave_stats.publication_waves >= 2);
    assert(wave_stats.descriptor_batch_calls == wave_stats.publication_waves);
    assert(wave_stats.head_batch_calls == wave_stats.publication_waves);
    validate_output(wave_buffers, 1, wave_config);
    wave_computation.shutdown();

    // Load-only mode preserves task lifecycle and completion accounting but
    // intentionally leaves the cleared output tensor untouched.
    auto load_only_config = config;
    load_only_config.nvlink_forward_computation_load_only_enabled = true;
    rdma_proxy::ForwardComputation load_only_computation(load_only_config, buffers);
    load_only_computation.initialize();
    load_only_computation.begin_iteration(2);
    assert(load_only_computation.enqueue_ready_region(2, 1, 0, 0, 32, 0) == 4);
    load_only_computation.finish_iteration(2);
    const auto load_only_stats = load_only_computation.stats();
    assert(load_only_stats.generated_tasks == 4);
    assert(load_only_stats.tasks_claimed == 4);
    assert(load_only_stats.tasks_completed == 4);
    assert(load_only_stats.exit_tasks_consumed == 4);
    const auto* load_only_output = static_cast<const uint16_t*>(
        buffers.nvlink_computation_output_buffer_for_source(1).output.ptr);
    for (std::size_t i = 0; i < config.num_tokens * config.nvlink_forward_computation_output_dim; ++i) {
        assert(load_only_output[i] == 0);
    }
    load_only_computation.shutdown();

    // Dequeue-only mode exercises just the persistent queue lifecycle. It
    // claims and completes every task without reading A/B or writing D.
    auto dequeue_only_config = config;
    dequeue_only_config.nvlink_forward_computation_dequeue_only_enabled = true;
    rdma_proxy::ForwardComputation dequeue_only_computation(dequeue_only_config, buffers);
    dequeue_only_computation.initialize();
    dequeue_only_computation.begin_iteration(3);
    assert(dequeue_only_computation.enqueue_ready_region(3, 1, 0, 0, 32, 0) == 4);
    dequeue_only_computation.finish_iteration(3);
    const auto dequeue_only_stats = dequeue_only_computation.stats();
    assert(dequeue_only_stats.generated_tasks == 4);
    assert(dequeue_only_stats.tasks_claimed == 4);
    assert(dequeue_only_stats.tasks_completed == 4);
    assert(dequeue_only_stats.exit_tasks_consumed == 4);
    const auto* dequeue_only_output = static_cast<const uint16_t*>(
        buffers.nvlink_computation_output_buffer_for_source(1).output.ptr);
    for (std::size_t i = 0; i < config.num_tokens * config.nvlink_forward_computation_output_dim; ++i) {
        assert(dequeue_only_output[i] == 0);
    }
    dequeue_only_computation.shutdown();

    std::cout << "forward computation host validation passed\n";
    return 0;
}
