#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
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

    std::cout << "test_measured_run passed\n";
    return 0;
}
