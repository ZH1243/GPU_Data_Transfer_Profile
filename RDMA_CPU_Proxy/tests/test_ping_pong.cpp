#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>

int main() {
    using namespace rdma_proxy;
    Logger::instance().set_level(LogLevel::kError);

    auto make_config = [] {
        ProxyConfig config;
        config.node_rank = 0;
        config.num_nodes = 2;
        config.local_gpu_index = 0;
        config.num_gpus_per_node = 1;
        config.num_tokens = 65;
        config.token_dimension = 8;
        config.tokens_per_chunk = 7;
        config.num_qps_per_peer = 3;
        config.completion_poll_batch_size = 8;
        config.data_signal_interval = 0;
        config.max_in_flight_chunks_per_qp = 3;
        config.send_queue_depth = 64;
        config.recv_queue_depth = 64;
        config.cq_depth = 128;
        config.num_iterations = 2;
        config.completion_timeout_ms = 30000;
        config.dtype = DataType::kFP16;
        config.mock_mode = true;
        config.fill_test_data = true;
        config.validate_data = true;
        config.peers.push_back(PeerAddress{1, "mock-peer", 18515});
        return config;
    };

    // Keep the legacy case, then cover both handoff positions and dynamic tails.
    for (int mode = 0; mode < 6; ++mode) {
        std::cerr << "router forwarding test mode=" << mode << "\n";
        auto config0 = make_config();
        config0.num_gpus_per_node = 2;
        config0.num_tokens = 65;
        config0.tokens_per_chunk = 4;
        config0.num_qps_per_peer = 1;
        config0.num_iterations = 2;
        config0.router_routing_enabled = true;
        config0.router_num_experts = 4;
        config0.router_top_k = 1;
        config0.nvlink_forwarding_enabled = true;
        config0.nvlink_forward_threshold_tokens = 8;
        config0.nvlink_forward_chunk_tokens = 1;
        config0.nvlink_forward_synchronize_batches = true;
        config0.nvlink_forward_local_batch_sync_enabled = true;
        config0.nvlink_forward_completion_notifications_enabled = true;
        config0.router_local_input_staging_enabled = true;
        config0.nvlink_forward_ping_pong_enabled = mode != 0;
        config0.nvlink_forward_ping_pong_handoff_copy = mode == 2 ? 8 : 1;
        if (mode != 0) {
            config0.nvlink_forward_notification_queue_depth = 1;
            config0.nvlink_forward_synchronize_iteration = false;
        }
        if (mode == 3) {
            config0.nvlink_forward_min_threshold_chunks = 2;
            config0.nvlink_forward_max_threshold_chunks = 3;
        }
        if (mode == 4) {
            // top_k=1 and single-token batches exercise direct-only handoffs.
            config0.nvlink_forward_threshold_tokens = 1;
        }
        if (mode == 5) {
            config0.num_tokens = 1;
            config0.nvlink_forward_threshold_tokens = 1;
        }
        config0.fill_test_data = false;
        config0.validate_data = false;
        const auto sync_nonce = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        config0.local_iteration_sync_run_id =
            "test_measured_run_router_unequal_batch_sync_" + sync_nonce;
        config0.nvlink_forward_exchange_dir =
            "/tmp/rdma_cpu_proxy_test_router_unequal_batch_sync_" + sync_nonce;
        config0.local_gpu_index = 0;
        config0.cuda_device_id = 0;

        auto config1 = config0;
        config1.local_gpu_index = 1;
        config1.cuda_device_id = 1;

        struct RouterBatchShape {
            uint64_t seed{0};
            std::size_t remote_tokens{0};
            std::size_t remote_batches{0};
            std::size_t local_tokens{0};
            std::size_t local_batches{0};
            std::size_t remote_forwarded_to_other_gpu{0};
            std::size_t local_forwarded_to_other_gpu{0};
        };
        const auto routing_shape = [](ProxyConfig config, uint64_t seed) {
            config.router_seed = seed;
            RouterRouting routing(config);
            routing.initialize();
            const auto& remote_tokens = routing.token_indices_for_node(1);
            const auto& remote_masks = routing.token_masks_for_node(1);
            const auto& local_tokens = routing.token_indices_for_node(0);
            const auto& local_masks = routing.token_masks_for_node(0);
            const int other_gpu = (config.local_gpu_index + 1) % config.num_gpus_per_node;
            int bit = config.local_gpu_index - other_gpu;
            if (bit < 0) bit += config.num_gpus_per_node;
            const auto other_gpu_mask = static_cast<uint8_t>(1U << bit);
            const auto remote_forwarded = static_cast<std::size_t>(std::count_if(
                remote_masks.begin(), remote_masks.end(),
                [&](uint8_t mask) { return (mask & other_gpu_mask) != 0; }));
            const auto local_forwarded = static_cast<std::size_t>(std::count_if(
                local_masks.begin(), local_masks.end(),
                [&](uint8_t mask) { return (mask & other_gpu_mask) != 0; }));
            return RouterBatchShape{
                seed,
                remote_tokens.size(),
                (remote_tokens.size() + config.nvlink_forward_threshold_tokens - 1) /
                    config.nvlink_forward_threshold_tokens,
                local_tokens.size(),
                (local_tokens.size() + config.nvlink_forward_threshold_tokens - 1) /
                    config.nvlink_forward_threshold_tokens,
                remote_forwarded,
                local_forwarded};
        };

        RouterBatchShape shape0;
        RouterBatchShape shape1;
        bool found_unequal_batch_counts = false;
        for (uint64_t seed0 = 1; seed0 < 128 && !found_unequal_batch_counts; ++seed0) {
            const auto candidate0 = routing_shape(config0, seed0);
            if (mode == 5 ? candidate0.remote_tokens != 0 :
                (candidate0.remote_forwarded_to_other_gpu == 0 ||
                 candidate0.local_forwarded_to_other_gpu == 0)) {
                continue;
            }
            for (uint64_t seed1 = 128; seed1 < 512; ++seed1) {
                const auto candidate1 = routing_shape(config1, seed1);
                if (mode == 5 ? candidate1.remote_tokens != 0 :
                    (candidate1.remote_forwarded_to_other_gpu != 0 &&
                     candidate1.local_forwarded_to_other_gpu != 0 &&
                     candidate0.remote_batches != candidate1.remote_batches &&
                     candidate0.local_batches != candidate1.local_batches)) {
                    shape0 = candidate0;
                    shape1 = candidate1;
                    found_unequal_batch_counts = true;
                    break;
                }
            }
        }
        if (!found_unequal_batch_counts) {
            std::cerr << "could not construct router batch counts for local sync test mode=" << mode << "\n";
            return 1;
        }
        config0.router_seed = shape0.seed;
        config1.router_seed = shape1.seed;

        validate_config(config0);
        validate_config(config1);

        auto verify_drained = [&](const Proxy& proxy, int gpu) {
            if (mode == 0) return;
            const auto& own = gpu == 0 ? shape0 : shape1;
            const auto& other = gpu == 0 ? shape1 : shape0;
            const auto check = [&](int node, int source, std::size_t expected) {
                const auto state = proxy.router_expert_token_head_state_for_source(node, source);
                if (state.received_token_frontier != expected ||
                    (expected != 0 && state.iteration != config0.num_iterations - 1)) {
                    throw std::runtime_error("ping-pong run returned before all source notifications drained");
                }
            };
            check(1, gpu, own.remote_tokens);
            check(1, 1 - gpu, other.remote_forwarded_to_other_gpu);
            check(0, gpu, own.local_tokens);
            check(0, 1 - gpu, other.local_forwarded_to_other_gpu);
        };

        std::exception_ptr error0;
        std::exception_ptr error1;
        std::thread gpu0([&] {
            try {
                Proxy proxy(config0);
                proxy.initialize();
                proxy.run();
                verify_drained(proxy, 0);
                proxy.shutdown();
            } catch (...) {
                error0 = std::current_exception();
            }
        });
        std::thread gpu1([&] {
            try {
                Proxy proxy(config1);
                proxy.initialize();
                proxy.run();
                verify_drained(proxy, 1);
                proxy.shutdown();
            } catch (...) {
                error1 = std::current_exception();
            }
        });
        gpu0.join();
        gpu1.join();
        if (error0) std::rethrow_exception(error0);
        if (error1) std::rethrow_exception(error1);
    }

    std::cout << "test_ping_pong passed\n";
    return 0;
}
