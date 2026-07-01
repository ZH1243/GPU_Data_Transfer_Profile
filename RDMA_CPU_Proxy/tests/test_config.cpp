#include "config.hpp"

#include <cassert>
#include <fstream>
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
    assert(config.data_signal_interval == 16);
    assert(config.max_in_flight_chunks_per_qp == 4);
    assert(config.num_iterations == 1);
    assert(config.completion_timeout_ms == 30000);
    assert(config.dtype == rdma_proxy::DataType::kBF16);
    assert(config.mock_mode);
    assert(config.fill_test_data);
    assert(config.validate_data);
    assert(config.cpu_affinity == "auto");
    assert(config.peers.size() == 3);
    assert(config.peers[0].node_rank == 1);
    assert(config.peers[0].host == "node-b.example.com");

    const char* multi_peer_cli_args[] = {
        "test_config",
        "--config",
        argv[1],
        "--listen_port=18520",
    };
    const auto multi_peer_cli_config = rdma_proxy::load_config(4, const_cast<char**>(multi_peer_cli_args));
    assert(multi_peer_cli_config.listen_port == 18520);
    for (const auto& peer : multi_peer_cli_config.peers) {
        assert(peer.port == 18520);
    }

    const char* single_peer_config = "single_peer_cli_config.json";
    {
        std::ofstream out(single_peer_config);
        out << R"json({
  "node_rank": 0,
  "num_nodes": 2,
  "local_gpu_index": 0,
  "num_gpus_per_node": 8,
  "num_tokens": 5000,
  "token_dimension": 128,
  "tokens_per_chunk": 32,
  "num_qps_per_peer": 10,
  "rdma_device_name": "mlx5_0",
  "rdma_port": 1,
  "gid_index": 3,
  "cuda_device_id": 0,
  "listen_port": 18515,
  "connection_manager_port": 18515,
  "completion_poll_batch_size": 16,
  "data_signal_interval": 0,
  "max_in_flight_chunks_per_qp": 2,
  "send_queue_depth": 256,
  "recv_queue_depth": 256,
  "cq_depth": 512,
  "dtype": "bf16",
  "mock_mode": true,
  "log_level": "info",
  "peers": [
    {
      "node_rank": 1,
      "host": "node-b.example.com",
      "port": 18515
    }
  ]
})json";
    }

    const char* cli_args[] = {
        "test_config",
        "--config",
        single_peer_config,
        "--listen_port=18516",
        "--peer_host=node-b-gpu1.example.com",
    };
    const auto cli_config = rdma_proxy::load_config(5, const_cast<char**>(cli_args));
    assert(cli_config.listen_port == 18516);
    assert(cli_config.peers.size() == 1);
    assert(cli_config.peers[0].port == 18516);
    assert(cli_config.peers[0].host == "node-b-gpu1.example.com");

    const char* peer_port_args[] = {
        "test_config",
        "--config",
        argv[1],
        "--peer_port=18521",
        "--num_iterations=3",
        "--completion_timeout_ms=1000",
        "--validate_data=false",
        "--cpu_affinity=0-95,192-287",
    };
    const auto peer_port_config = rdma_proxy::load_config(8, const_cast<char**>(peer_port_args));
    for (const auto& peer : peer_port_config.peers) {
        assert(peer.port == 18521);
    }
    assert(peer_port_config.num_iterations == 3);
    assert(peer_port_config.completion_timeout_ms == 1000);
    assert(!peer_port_config.validate_data);
    assert(peer_port_config.cpu_affinity == "0-95,192-287");

    const char* signal_interval_args[] = {
        "test_config",
        "--config",
        argv[1],
        "--data_signal_interval=0",
        "--max_in_flight_chunks_per_qp=8",
    };
    const auto signal_interval_config = rdma_proxy::load_config(5, const_cast<char**>(signal_interval_args));
    assert(signal_interval_config.data_signal_interval == 0);
    assert(signal_interval_config.max_in_flight_chunks_per_qp == 8);

    std::cout << rdma_proxy::config_summary(config) << '\n';
    std::cout << "test_config passed\n";
    return 0;
}
