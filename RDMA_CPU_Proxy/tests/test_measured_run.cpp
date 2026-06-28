#include "logging.hpp"
#include "proxy.hpp"

#include <iostream>

int main() {
    using namespace rdma_proxy;

    Logger::instance().set_level(LogLevel::kError);

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
    validate_config(config);

    Proxy proxy(config);
    proxy.initialize();
    proxy.run();
    proxy.shutdown();

    std::cout << "test_measured_run passed\n";
    return 0;
}
