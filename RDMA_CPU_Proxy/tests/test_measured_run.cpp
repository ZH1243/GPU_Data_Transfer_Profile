#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>
#include <vector>

namespace {

uint8_t router_test_pattern_byte(
    const rdma_proxy::ProxyConfig& config,
    uint64_t iteration,
    std::size_t offset) {
    uint64_t x = static_cast<uint64_t>(offset);
    x ^= (iteration + 1) * 0x9e3779b97f4a7c15ULL;
    x ^= (static_cast<uint64_t>(config.node_rank + 1) << 48);
    // Router test data uses destination rank -1, whose encoded +1 field is 0.
    x ^= (static_cast<uint64_t>(config.local_gpu_index + 1) << 16);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return static_cast<uint8_t>(x & 0xffU);
}

std::vector<uint8_t> expected_compact_router_destination(
    const rdma_proxy::ProxyConfig& config,
    const rdma_proxy::RouterRouting& routing,
    int peer_rank,
    int destination_gpu,
    uint64_t iteration) {
    const auto& source_tokens = routing.token_indices_for_node(peer_rank);
    const auto& masks = routing.token_masks_for_node(peer_rank);
    const auto token_bytes =
        config.token_dimension * rdma_proxy::dtype_size(config.dtype);
    int bit = config.local_gpu_index - destination_gpu;
    if (bit < 0) bit += config.num_gpus_per_node;
    const auto destination_mask = static_cast<uint8_t>(1U << bit);

    std::vector<uint8_t> expected;
    for (std::size_t position = 0; position < source_tokens.size(); ++position) {
        if ((masks[position] & destination_mask) == 0) continue;
        const auto source_byte_offset = source_tokens[position] * token_bytes;
        for (std::size_t byte = 0; byte < token_bytes; ++byte) {
            expected.push_back(router_test_pattern_byte(
                config, iteration, source_byte_offset + byte));
        }
    }
    return expected;
}

}  // namespace

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
        config.completion_timeout_ms = 5000;
        config.dtype = DataType::kFP16;
        config.mock_mode = true;
        config.fill_test_data = true;
        config.validate_data = true;
        config.peers.push_back(PeerAddress{1, "mock-peer", 18515});
        return config;
    };

    {
        auto config = make_config();
        validate_config(config);

        Proxy proxy(config);
        proxy.initialize();
        proxy.run();
        proxy.shutdown();
    }

    {
        auto config = make_config();
        config.rdma_chunk_per_token_sge_enabled = true;
        validate_config(config);

        Proxy proxy(config);
        proxy.initialize();
        proxy.run();
        proxy.shutdown();
    }

    {
        auto config = make_config();
        config.rdma_chunk_per_token_sge_enabled = true;
        config.rdma_discontinuous_token_payload_enabled = true;
        validate_config(config);

        Proxy proxy(config);
        proxy.initialize();
        proxy.run();
        proxy.shutdown();
    }

    {
        auto config = make_config();
        config.num_gpus_per_node = 2;
        config.router_routing_enabled = true;
        config.router_num_experts = 8;
        config.router_top_k = 2;
        config.router_seed = 9876;
        validate_config(config);

        Proxy proxy(config);
        proxy.initialize();
        proxy.run();
        proxy.shutdown();
    }

    {
        auto config = make_config();
        config.num_gpus_per_node = 3;
        config.num_tokens = 97;
        config.tokens_per_chunk = 7;
        config.num_iterations = 2;
        config.router_routing_enabled = true;
        config.router_num_experts = 12;
        config.router_top_k = 4;
        config.router_seed = 9876;
        config.nvlink_forwarding_enabled = true;
        config.nvlink_forward_threshold_tokens = 11;
        config.nvlink_forward_chunk_tokens = 1;

        const auto destination_bytes =
            config.num_tokens * config.token_dimension * dtype_size(config.dtype);
        std::vector<uint8_t> destination_gpu1(destination_bytes, 0);
        std::vector<uint8_t> destination_gpu2(destination_bytes, 0);
        config.nvlink_forward_destinations = {
            NvlinkForwardDestination{
                1,
                1,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(destination_gpu1.data())),
                destination_gpu1.size()},
            NvlinkForwardDestination{
                2,
                2,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(destination_gpu2.data())),
                destination_gpu2.size()},
        };
        validate_config(config);

        RouterRouting expected_routing(config);
        expected_routing.initialize();
        const auto expected_gpu1 = expected_compact_router_destination(
            config, expected_routing, 1, 1, config.num_iterations - 1);
        const auto expected_gpu2 = expected_compact_router_destination(
            config, expected_routing, 1, 2, config.num_iterations - 1);
        if (expected_gpu1.empty() || expected_gpu2.empty()) {
            std::cerr << "router compaction test did not route tokens to both destinations\n";
            return 1;
        }

        Proxy proxy(config);
        proxy.initialize();
        for (int source_node = 0; source_node < config.num_nodes; ++source_node) {
            for (int source_gpu = 0; source_gpu < config.num_gpus_per_node; ++source_gpu) {
                const auto& metadata = proxy.router_expert_metadata_for_source(
                    source_node, source_gpu);
                if (metadata.source_node_rank != source_node ||
                    metadata.source_gpu_index != source_gpu ||
                    metadata.destination_node_rank != config.node_rank ||
                    metadata.destination_gpu_index != config.local_gpu_index ||
                    metadata.experts_per_gpu != 2 ||
                    metadata.expert_offsets.size() != 3) {
                    std::cerr << "router expert all-to-all metadata is incomplete\n";
                    return 1;
                }
            }
        }
        proxy.run();
        proxy.shutdown();

        auto check_compact_destination = [](
            const std::vector<uint8_t>& actual,
            const std::vector<uint8_t>& expected,
            const char* name) {
            if (!std::equal(expected.begin(), expected.end(), actual.begin())) {
                std::cerr << name << " router destination is not compact or has incorrect token order\n";
                return false;
            }
            if (std::any_of(actual.begin() + static_cast<std::ptrdiff_t>(expected.size()),
                            actual.end(),
                            [](uint8_t byte) { return byte != 0; })) {
                std::cerr << name << " router destination has data after its compact token span\n";
                return false;
            }
            return true;
        };
        if (!check_compact_destination(destination_gpu1, expected_gpu1, "GPU 1") ||
            !check_compact_destination(destination_gpu2, expected_gpu2, "GPU 2")) {
            return 1;
        }
    }

    {
        auto config = make_config();
        config.num_nodes = 4;
        config.num_iterations = 1;
        config.sequential_peer_transfers = true;
        config.local_forwarding_rdma_overlap_enabled = true;
        config.peers = {
            PeerAddress{1, "mock-peer-b", 18515},
            PeerAddress{2, "mock-peer-c", 18515},
            PeerAddress{3, "mock-peer-d", 18515},
        };
        validate_config(config);

        Proxy proxy(config);
        proxy.initialize();
        proxy.run();
        proxy.shutdown();
    }

    {
        auto config = make_config();
        config.num_gpus_per_node = 2;
        config.num_iterations = 1;
        config.nvlink_forwarding_enabled = true;
        config.nvlink_forward_threshold_tokens = config.num_tokens;
        config.nvlink_forward_chunk_tokens = config.num_tokens;
        config.nvlink_routing_probability = 1.0;
        const auto destination_bytes = config.num_tokens * config.token_dimension * dtype_size(config.dtype);
        std::vector<uint8_t> destination(destination_bytes, 0);
        config.nvlink_forward_destinations.push_back(
            NvlinkForwardDestination{
                1,
                1,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(destination.data())),
                destination.size()});
        validate_config(config);

        Proxy proxy(config);
        proxy.initialize();
        proxy.run();
        proxy.shutdown();

        const bool copied = std::any_of(destination.begin(), destination.end(), [](uint8_t v) { return v != 0; });
        if (!copied) {
            std::cerr << "NVLink mock forwarding destination remained empty\n";
            return 1;
        }
    }

    {
        auto config = make_config();
        config.num_gpus_per_node = 2;
        config.num_iterations = 1;
        config.nvlink_forwarding_enabled = true;
        config.nvlink_forward_use_round_robin = true;
        config.nvlink_forward_threshold_tokens = config.num_tokens;
        config.nvlink_forward_chunk_tokens = config.num_tokens;
        const auto destination_bytes = config.num_tokens * config.token_dimension * dtype_size(config.dtype);
        std::vector<uint8_t> destination(destination_bytes, 0);
        config.nvlink_forward_destinations.push_back(
            NvlinkForwardDestination{
                1,
                1,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(destination.data())),
                destination.size()});
        validate_config(config);

        Proxy proxy(config);
        proxy.initialize();
        proxy.run();
        proxy.shutdown();

        const bool copied = std::any_of(destination.begin(), destination.end(), [](uint8_t v) { return v != 0; });
        if (!copied) {
            std::cerr << "NVLink mock round-robin forwarding destination remained empty\n";
            return 1;
        }
    }

    {
        auto config0 = make_config();
        config0.num_tokens = 64;
        config0.tokens_per_chunk = 8;
        config0.num_gpus_per_node = 2;
        config0.num_iterations = 1;
        config0.nvlink_forwarding_enabled = true;
        config0.nvlink_forward_use_round_robin = false;
        config0.nvlink_forward_threshold_tokens = 32;
        config0.nvlink_forward_chunk_tokens = 32;
        config0.nvlink_forward_synchronize_batches = true;
        config0.nvlink_forward_completion_notifications_enabled = true;
        config0.nvlink_forward_notification_queue_depth = 8;
        config0.nvlink_forward_notification_log_enabled = true;
        config0.nvlink_forward_notification_log_dir = "/tmp/rdma_cpu_proxy_test_notification_logs";
        config0.nvlink_forward_local_batch_sync_enabled = true;
        config0.nvlink_routing_probability = 1.0;
        config0.local_iteration_sync_run_id = "test_measured_run_nvlink_batch_sync";
        config0.local_gpu_index = 0;
        config0.cuda_device_id = 0;

        auto config1 = config0;
        config1.local_gpu_index = 1;
        config1.cuda_device_id = 1;

        const auto destination_bytes = config0.num_tokens * config0.token_dimension * dtype_size(config0.dtype);
        std::vector<uint8_t> destination0(destination_bytes, 0);
        std::vector<uint8_t> destination1(destination_bytes, 0);
        config0.nvlink_forward_destinations.push_back(
            NvlinkForwardDestination{
                1,
                1,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(destination0.data())),
                destination0.size()});
        config1.nvlink_forward_destinations.push_back(
            NvlinkForwardDestination{
                0,
                0,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(destination1.data())),
                destination1.size()});

        validate_config(config0);
        validate_config(config1);

        std::exception_ptr error0;
        std::exception_ptr error1;
        std::thread gpu0([&] {
            try {
                Proxy proxy(config0);
                proxy.initialize();
                proxy.run();
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
                proxy.shutdown();
            } catch (...) {
                error1 = std::current_exception();
            }
        });
        gpu0.join();
        gpu1.join();
        if (error0) std::rethrow_exception(error0);
        if (error1) std::rethrow_exception(error1);

        const bool copied0 = std::any_of(destination0.begin(), destination0.end(), [](uint8_t v) { return v != 0; });
        const bool copied1 = std::any_of(destination1.begin(), destination1.end(), [](uint8_t v) { return v != 0; });
        if (!copied0 || !copied1) {
            std::cerr << "NVLink mock batch-synchronized forwarding destination remained empty\n";
            return 1;
        }
        const std::filesystem::path log0 =
            std::filesystem::path(config0.nvlink_forward_notification_log_dir) /
            "nvlink_forward_notifications_rank_0_gpu_0.log";
        const std::filesystem::path log1 =
            std::filesystem::path(config0.nvlink_forward_notification_log_dir) /
            "nvlink_forward_notifications_rank_0_gpu_1.log";
        if (!std::filesystem::exists(log0) || !std::filesystem::exists(log1) ||
            std::filesystem::file_size(log0) == 0 || std::filesystem::file_size(log1) == 0) {
            std::cerr << "NVLink notification log files were not written\n";
            return 1;
        }
        std::ifstream notification_log(log0);
        const std::string notification_text(
            (std::istreambuf_iterator<char>(notification_log)),
            std::istreambuf_iterator<char>());
        if (notification_text.find("nvlink_forward_notification") == std::string::npos ||
            notification_text.find("num_tokens=32") == std::string::npos) {
            std::cerr << "NVLink notification log did not contain the expected completion records\n";
            return 1;
        }
    }

    {
        auto config0 = make_config();
        config0.num_tokens = 64;
        config0.tokens_per_chunk = 8;
        config0.num_gpus_per_node = 2;
        config0.num_iterations = 1;
        config0.nvlink_forwarding_enabled = true;
        config0.nvlink_forward_min_threshold_chunks = 1;
        config0.nvlink_forward_max_threshold_chunks = 3;
        config0.nvlink_forward_out_of_order_chunks_enabled = true;
        config0.nvlink_forward_chunk_tokens = 8;
        config0.nvlink_forward_synchronize_batches = true;
        config0.nvlink_forward_local_batch_sync_enabled = true;
        config0.nvlink_routing_probability = 1.0;
        config0.local_iteration_sync_run_id = "test_measured_run_nvlink_dynamic_batch_sync";
        config0.local_gpu_index = 0;
        config0.cuda_device_id = 0;

        auto config1 = config0;
        config1.local_gpu_index = 1;
        config1.cuda_device_id = 1;

        const auto destination_bytes = config0.num_tokens * config0.token_dimension * dtype_size(config0.dtype);
        std::vector<uint8_t> destination0(destination_bytes, 0);
        std::vector<uint8_t> destination1(destination_bytes, 0);
        config0.nvlink_forward_destinations.push_back(
            NvlinkForwardDestination{
                1,
                1,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(destination0.data())),
                destination0.size()});
        config1.nvlink_forward_destinations.push_back(
            NvlinkForwardDestination{
                0,
                0,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(destination1.data())),
                destination1.size()});

        validate_config(config0);
        validate_config(config1);

        std::exception_ptr error0;
        std::exception_ptr error1;
        std::thread gpu0([&] {
            try {
                Proxy proxy(config0);
                proxy.initialize();
                proxy.run();
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
                proxy.shutdown();
            } catch (...) {
                error1 = std::current_exception();
            }
        });
        gpu0.join();
        gpu1.join();
        if (error0) std::rethrow_exception(error0);
        if (error1) std::rethrow_exception(error1);

        const bool copied0 = std::any_of(destination0.begin(), destination0.end(), [](uint8_t v) { return v != 0; });
        const bool copied1 = std::any_of(destination1.begin(), destination1.end(), [](uint8_t v) { return v != 0; });
        if (!copied0 || !copied1) {
            std::cerr << "NVLink mock dynamic batch-synchronized forwarding destination remained empty\n";
            return 1;
        }
    }

    {
        auto config0 = make_config();
        config0.node_rank = 0;
        config0.num_nodes = 1;
        config0.local_gpu_index = 0;
        config0.cuda_device_id = 0;
        config0.num_gpus_per_node = 2;
        config0.num_iterations = 2;
        config0.local_iteration_sync_enabled = true;
        config0.local_iteration_sync_dir = "/tmp/rdma_cpu_proxy_test_local_iteration_sync";
        config0.local_iteration_sync_run_id = "test_measured_run";
        config0.peers.clear();

        auto config1 = config0;
        config1.local_gpu_index = 1;
        config1.cuda_device_id = 1;

        validate_config(config0);
        validate_config(config1);

        std::exception_ptr error0;
        std::exception_ptr error1;
        std::thread gpu0([&] {
            try {
                Proxy proxy(config0);
                proxy.initialize();
                proxy.run();
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

    std::cout << "test_measured_run passed\n";
    return 0;
}
