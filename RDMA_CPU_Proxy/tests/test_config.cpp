#include "config.hpp"

#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    assert(argc == 2);
    const auto config = rdma_proxy::load_config_file(argv[1]);

    assert(config.node_rank == 0);
    assert(config.num_nodes == 4);
    assert(config.local_gpu_index == 0);
    assert(config.num_gpus_per_node == 8);
    assert(config.num_tokens == 5000);
    assert(config.token_dimension == 128);
    assert(config.tokens_per_chunk == 32);
    assert(config.num_qps_per_peer == 10);
    assert(config.dtype == rdma_proxy::DataType::kBF16);
    assert(config.mock_mode);
    assert(config.peers.size() == 3);
    assert(config.peers[0].node_rank == 1);
    assert(config.peers[0].host == "node-b.example.com");

    std::cout << rdma_proxy::config_summary(config) << '\n';
    std::cout << "test_config passed\n";
    return 0;
}
