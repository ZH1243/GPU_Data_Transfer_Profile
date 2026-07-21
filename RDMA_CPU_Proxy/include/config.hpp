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

struct NvlinkForwardDestination {
    int gpu_index{-1};
    int cuda_device_id{-1};
    uint64_t buffer_addr{0};
    std::size_t buffer_bytes{0};
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
    int max_in_flight_chunks_per_qp{1};
    bool rdma_chunk_per_token_sge_enabled{false};
    bool rdma_discontinuous_token_payload_enabled{false};
    int send_queue_depth{128};
    int recv_queue_depth{128};
    int cq_depth{256};
    std::size_t num_iterations{1};
    uint64_t completion_timeout_ms{30000};

    DataType dtype{DataType::kBF16};
    bool mock_mode{false};
    bool fill_test_data{true};
    bool validate_data{true};
    bool sequential_peer_transfers{false};
    bool nvlink_forwarding_enabled{false};
    std::size_t nvlink_forward_threshold_tokens{0};
    std::size_t nvlink_forward_threshold_chunks{0};
    std::size_t nvlink_forward_min_threshold_chunks{0};
    std::size_t nvlink_forward_max_threshold_chunks{0};
    bool nvlink_forward_out_of_order_chunks_enabled{false};
    std::size_t nvlink_forward_chunk_tokens{0};
    bool nvlink_forward_use_batch_api{true};
    bool nvlink_forward_stream_nonblocking{true};
    bool nvlink_forward_synchronize_batches{false};
    bool nvlink_forward_completion_notifications_enabled{false};
    std::size_t nvlink_forward_notification_queue_depth{1024};
    bool nvlink_forward_notification_log_enabled{false};
    std::string nvlink_forward_notification_log_dir{"/tmp/rdma_cpu_proxy_nvlink_notifications"};
    bool nvlink_forward_computation_enabled{false};
    std::size_t nvlink_forward_computation_output_dim{0};
    std::size_t nvlink_forward_computation_tile_m{128};
    std::size_t nvlink_forward_computation_tile_n{128};
    std::size_t nvlink_forward_computation_num_queues{1};
    std::size_t nvlink_forward_computation_queue_depth{1024};
    bool nvlink_forward_computation_load_only_enabled{false};
    bool nvlink_forward_computation_dequeue_only_enabled{false};
    bool nvlink_forward_computation_log_enabled{false};
    bool nvlink_forward_local_batch_sync_enabled{false};
    bool nvlink_forward_synchronize_iteration{true};
    bool nvlink_forward_log_batches{false};
    bool log_qp_reports{false};
    bool log_marker_wait_reports{false};
    bool nvlink_forward_use_round_robin{false};
    double nvlink_routing_probability{0.5};
    uint64_t nvlink_routing_seed{1};
    std::string nvlink_forward_exchange_dir{"/tmp/rdma_cpu_proxy_nvlink"};
    bool local_iteration_sync_enabled{false};
    std::string local_iteration_sync_dir{"/tmp/rdma_cpu_proxy_local_iteration_sync"};
    std::string local_iteration_sync_run_id;
    std::string rdma_bandwidth_summary_dir{"/tmp/rdma_cpu_proxy_results"};
    std::string cpu_affinity;
    std::string log_level{"info"};
    std::vector<PeerAddress> peers;
    std::vector<NvlinkForwardDestination> nvlink_forward_destinations;
};

std::size_t dtype_size(DataType dtype);
std::string to_string(DataType dtype);
DataType dtype_from_string(const std::string& value);
bool nvlink_forward_dynamic_threshold_enabled(const ProxyConfig& config);
std::size_t effective_nvlink_forward_threshold_tokens(const ProxyConfig& config);

ProxyConfig load_config_file(const std::string& path);
ProxyConfig load_config(int argc, char** argv);
void validate_config(const ProxyConfig& config);
std::string config_summary(const ProxyConfig& config);
std::string config_help();

}  // namespace rdma_proxy
