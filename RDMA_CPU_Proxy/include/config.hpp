#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rdma_proxy {

enum class DataType {
    kBF16,
    kFP16,
    kFP32,
};

struct PeerAddress {
    int node_rank{-1};
    std::string host;
    uint16_t port{0};
};

struct ProxyConfig {
    int node_rank{0};
    int num_nodes{1};
    int local_gpu_index{0};
    int num_gpus_per_node{8};
    std::size_t num_tokens{0};
    std::size_t token_dimension{0};
    std::size_t tokens_per_chunk{1};
    int num_qps_per_peer{1};

    std::string rdma_device_name;
    uint8_t rdma_port{1};
    int gid_index{-1};
    int cuda_device_id{0};
    uint16_t listen_port{18515};
    uint16_t connection_manager_port{18515};

    int completion_poll_batch_size{16};
    int data_signal_interval{1};
    int send_queue_depth{128};
    int recv_queue_depth{128};
    int cq_depth{256};
    std::size_t num_iterations{1};
    uint64_t completion_timeout_ms{30000};

    DataType dtype{DataType::kBF16};
    bool mock_mode{false};
    bool fill_test_data{true};
    bool validate_data{true};
    std::string cpu_affinity;
    std::string log_level{"info"};
    std::vector<PeerAddress> peers;
};

std::size_t dtype_size(DataType dtype);
std::string to_string(DataType dtype);
DataType dtype_from_string(const std::string& value);

ProxyConfig load_config_file(const std::string& path);
ProxyConfig load_config(int argc, char** argv);
void validate_config(const ProxyConfig& config);
std::string config_summary(const ProxyConfig& config);

}  // namespace rdma_proxy
