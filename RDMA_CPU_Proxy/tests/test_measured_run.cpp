#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

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
        config.num_nodes = 4;
        config.num_iterations = 1;
        config.sequential_peer_transfers = true;
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
        config0.nvlink_forward_use_round_robin = true;
        config0.nvlink_forward_threshold_tokens = 32;
        config0.nvlink_forward_chunk_tokens = 32;
        config0.nvlink_forward_synchronize_batches = true;
        config0.nvlink_forward_local_batch_sync_enabled = true;
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
