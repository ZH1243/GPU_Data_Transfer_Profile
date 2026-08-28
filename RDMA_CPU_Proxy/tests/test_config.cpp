#include "config.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>

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
    assert(!config.rdma_chunk_per_token_sge_enabled);
    assert(!config.rdma_discontinuous_token_payload_enabled);
    assert(!config.router_routing_enabled);
    assert(config.router_num_experts == 256);
    assert(config.router_top_k == 8);
    assert(config.router_seed == 1234);
    assert(config.router_metadata_port_base == 19515);
    assert(config.num_iterations == 1);
    assert(config.completion_timeout_ms == 30000);
    assert(config.dtype == rdma_proxy::DataType::kBF16);
    assert(config.mock_mode);
    assert(config.fill_test_data);
    assert(config.validate_data);
    assert(!config.sequential_peer_transfers);
    assert(!config.local_forwarding_rdma_overlap_enabled);
    assert(!config.nvlink_forwarding_enabled);
    assert(config.nvlink_forward_threshold_tokens == 700);
    assert(config.nvlink_forward_threshold_chunks == 0);
    assert(config.nvlink_forward_min_threshold_chunks == 0);
    assert(config.nvlink_forward_max_threshold_chunks == 0);
    assert(!config.nvlink_forward_out_of_order_chunks_enabled);
    assert(!rdma_proxy::nvlink_forward_dynamic_threshold_enabled(config));
    assert(rdma_proxy::effective_nvlink_forward_threshold_tokens(config) == 700);
    assert(config.nvlink_forward_chunk_tokens == 100);
    assert(config.nvlink_forward_use_batch_api);
    assert(config.nvlink_forward_stream_nonblocking);
    assert(!config.nvlink_forward_synchronize_batches);
    assert(!config.nvlink_forward_completion_notifications_enabled);
    assert(!config.nvlink_forward_notification_flush_only_enabled);
    assert(!config.nvlink_forward_notification_flush_per_entry_enabled);
    assert(!config.router_local_input_staging_enabled);
    assert(config.nvlink_forward_notification_flag_update_mode ==
           rdma_proxy::NvlinkForwardNotificationFlagUpdateMode::kStreamWrite);
    assert(rdma_proxy::to_string(
               rdma_proxy::NvlinkForwardNotificationFlagUpdateMode::kMemcpy) ==
           "memcpy");
    bool rejected_invalid_flag_update_mode = false;
    try {
        (void)rdma_proxy::nvlink_forward_notification_flag_update_mode_from_string(
            "invalid");
    } catch (const std::runtime_error&) {
        rejected_invalid_flag_update_mode = true;
    }
    assert(rejected_invalid_flag_update_mode);
    assert(config.expert_gemm_m_tile == 128);
    assert(config.expert_gemm_n_tile == 256);
    assert(config.expert_gemm_dimension == 8192);
    assert(config.expert_gemm_cluster_m == 1);
    assert(config.expert_gemm_max_swizzle_size == 8);
    assert(config.nvlink_forward_notification_queue_depth == 1024);
    assert(!config.nvlink_forward_notification_log_enabled);
    assert(config.nvlink_forward_notification_log_dir == "/tmp/rdma_cpu_proxy_nvlink_notifications");
    assert(!config.nvlink_forward_local_batch_sync_enabled);
    assert(config.nvlink_forward_synchronize_iteration);
    assert(!config.nvlink_forward_log_batches);
    assert(!config.log_qp_reports);
    assert(!config.log_marker_wait_reports);
    assert(!config.nvlink_forward_use_round_robin);
    assert(config.nvlink_routing_probability == 0.5);
    assert(config.nvlink_routing_seed == 1);
    assert(config.nvlink_forward_exchange_dir == "/tmp/rdma_cpu_proxy_nvlink");
    assert(!config.local_iteration_sync_enabled);
    assert(config.local_iteration_sync_dir == "/tmp/rdma_cpu_proxy_local_iteration_sync");
    assert(config.local_iteration_sync_run_id.empty());
    assert(config.rdma_bandwidth_summary_dir == "/tmp/rdma_cpu_proxy_results");
    assert(config.nvlink_forward_destinations.empty());
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
        "--sequential_peer_transfers=true",
        "--local_forwarding_rdma_overlap_enabled=true",
        "--log_qp_reports=true",
        "--log_marker_wait_reports=true",
        "--nvlink_forward_local_batch_sync_enabled=false",
        "--local_iteration_sync_enabled=true",
        "--local_iteration_sync_dir=/tmp/rdma_cpu_proxy_test_local_sync",
        "--local_iteration_sync_run_id=test_cli",
        "--rdma_bandwidth_summary_dir=/tmp/rdma_cpu_proxy_test_results",
        "--nvlink_forward_threshold_chunks=5",
        "--nvlink_forward_min_threshold_chunks=2",
        "--nvlink_forward_max_threshold_chunks=6",
        "--nvlink_forward_out_of_order_chunks_enabled=true",
        "--expert_gemm_m_tile=64",
        "--expert_gemm_n_tile=128",
        "--expert_gemm_dimension=4096",
        "--expert_gemm_cluster_m=2",
        "--expert_gemm_max_swizzle_size=4",
        "--nvlink_forward_notification_flush_per_entry_enabled=false",
        "--nvlink_forward_notification_flag_update_mode=memcpy",
        "--cpu_affinity=0-95,192-287",
    };
    const auto peer_port_config = rdma_proxy::load_config(28, const_cast<char**>(peer_port_args));
    for (const auto& peer : peer_port_config.peers) {
        assert(peer.port == 18521);
    }
    assert(peer_port_config.num_iterations == 3);
    assert(peer_port_config.completion_timeout_ms == 1000);
    assert(!peer_port_config.validate_data);
    assert(peer_port_config.sequential_peer_transfers);
    assert(peer_port_config.local_forwarding_rdma_overlap_enabled);
    assert(peer_port_config.log_qp_reports);
    assert(peer_port_config.log_marker_wait_reports);
    assert(peer_port_config.local_iteration_sync_enabled);
    assert(peer_port_config.local_iteration_sync_dir == "/tmp/rdma_cpu_proxy_test_local_sync");
    assert(peer_port_config.local_iteration_sync_run_id == "test_cli");
    assert(peer_port_config.rdma_bandwidth_summary_dir == "/tmp/rdma_cpu_proxy_test_results");
    assert(peer_port_config.nvlink_forward_threshold_chunks == 5);
    assert(peer_port_config.nvlink_forward_min_threshold_chunks == 2);
    assert(peer_port_config.nvlink_forward_max_threshold_chunks == 6);
    assert(peer_port_config.nvlink_forward_out_of_order_chunks_enabled);
    assert(peer_port_config.expert_gemm_m_tile == 64);
    assert(peer_port_config.expert_gemm_n_tile == 128);
    assert(peer_port_config.expert_gemm_dimension == 4096);
    assert(peer_port_config.expert_gemm_cluster_m == 2);
    assert(peer_port_config.expert_gemm_max_swizzle_size == 4);
    assert(!peer_port_config.nvlink_forward_notification_flush_per_entry_enabled);
    assert(peer_port_config.nvlink_forward_notification_flag_update_mode ==
           rdma_proxy::NvlinkForwardNotificationFlagUpdateMode::kMemcpy);
    assert(peer_port_config.cpu_affinity == "0-95,192-287");

    auto invalid_overlap_config = peer_port_config;
    invalid_overlap_config.sequential_peer_transfers = false;
    bool rejected_overlap_without_sequential_transfers = false;
    try {
        rdma_proxy::validate_config(invalid_overlap_config);
    } catch (const std::runtime_error&) {
        rejected_overlap_without_sequential_transfers = true;
    }
    assert(rejected_overlap_without_sequential_transfers);

    const char* signal_interval_args[] = {
        "test_config",
        "--config",
        argv[1],
        "--data_signal_interval=0",
        "--max_in_flight_chunks_per_qp=8",
        "--rdma_chunk_per_token_sge_enabled=true",
        "--rdma_discontinuous_token_payload_enabled=true",
    };
    const auto signal_interval_config = rdma_proxy::load_config(7, const_cast<char**>(signal_interval_args));
    assert(signal_interval_config.data_signal_interval == 0);
    assert(signal_interval_config.max_in_flight_chunks_per_qp == 8);
    assert(signal_interval_config.rdma_chunk_per_token_sge_enabled);
    assert(signal_interval_config.rdma_discontinuous_token_payload_enabled);

    auto invalid_discontinuous_config = signal_interval_config;
    invalid_discontinuous_config.rdma_chunk_per_token_sge_enabled = false;
    bool rejected_discontinuous_without_sge = false;
    try {
        rdma_proxy::validate_config(invalid_discontinuous_config);
    } catch (const std::runtime_error&) {
        rejected_discontinuous_without_sge = true;
    }
    assert(rejected_discontinuous_without_sge);

    auto router_config = signal_interval_config;
    router_config.router_routing_enabled = true;
    router_config.router_num_experts = 256;
    router_config.router_top_k = 8;
    router_config.rdma_chunk_per_token_sge_enabled = false;
    router_config.rdma_discontinuous_token_payload_enabled = false;
    rdma_proxy::validate_config(router_config);
    assert(rdma_proxy::effective_rdma_chunk_per_token_sge_enabled(router_config));
    assert(rdma_proxy::effective_rdma_discontinuous_token_payload_enabled(router_config));

    auto router_nvlink_config = router_config;
    router_nvlink_config.nvlink_forwarding_enabled = true;
    router_nvlink_config.nvlink_forward_threshold_tokens = 700;
    router_nvlink_config.nvlink_forward_chunk_tokens = 100;
    rdma_proxy::validate_config(router_nvlink_config);

    auto valid_flush_only_config = router_nvlink_config;
    valid_flush_only_config.nvlink_forward_synchronize_batches = true;
    valid_flush_only_config.nvlink_forward_completion_notifications_enabled = true;
    valid_flush_only_config.nvlink_forward_notification_flush_only_enabled = true;
    rdma_proxy::validate_config(valid_flush_only_config);

    auto valid_flush_per_entry_config = valid_flush_only_config;
    valid_flush_per_entry_config.nvlink_forward_notification_flush_only_enabled = false;
    valid_flush_per_entry_config.nvlink_forward_notification_flush_per_entry_enabled = true;
    rdma_proxy::validate_config(valid_flush_per_entry_config);

    auto valid_local_staging_config = valid_flush_only_config;
    valid_local_staging_config.nvlink_forward_notification_flush_only_enabled = false;
    valid_local_staging_config.fill_test_data = false;
    valid_local_staging_config.router_local_input_staging_enabled = true;
    rdma_proxy::validate_config(valid_local_staging_config);

    auto invalid_local_staging_fill_config = valid_local_staging_config;
    invalid_local_staging_fill_config.fill_test_data = true;
    bool rejected_local_staging_fill = false;
    try {
        rdma_proxy::validate_config(invalid_local_staging_fill_config);
    } catch (const std::runtime_error&) {
        rejected_local_staging_fill = true;
    }
    assert(rejected_local_staging_fill);

    auto invalid_flush_per_entry_without_notifications = router_nvlink_config;
    invalid_flush_per_entry_without_notifications
        .nvlink_forward_notification_flush_per_entry_enabled = true;
    bool rejected_flush_per_entry_without_notifications = false;
    try {
        rdma_proxy::validate_config(invalid_flush_per_entry_without_notifications);
    } catch (const std::runtime_error&) {
        rejected_flush_per_entry_without_notifications = true;
    }
    assert(rejected_flush_per_entry_without_notifications);

    auto invalid_combined_flush_modes = valid_flush_only_config;
    invalid_combined_flush_modes.nvlink_forward_notification_flush_per_entry_enabled = true;
    bool rejected_combined_flush_modes = false;
    try {
        rdma_proxy::validate_config(invalid_combined_flush_modes);
    } catch (const std::runtime_error&) {
        rejected_combined_flush_modes = true;
    }
    assert(rejected_combined_flush_modes);

    auto invalid_flush_only_without_notifications = router_nvlink_config;
    invalid_flush_only_without_notifications.nvlink_forward_notification_flush_only_enabled = true;
    bool rejected_flush_only_without_notifications = false;
    try {
        rdma_proxy::validate_config(invalid_flush_only_without_notifications);
    } catch (const std::runtime_error&) {
        rejected_flush_only_without_notifications = true;
    }
    assert(rejected_flush_only_without_notifications);

    auto invalid_router_round_robin_config = router_nvlink_config;
    invalid_router_round_robin_config.nvlink_forward_use_round_robin = true;
    bool rejected_router_round_robin = false;
    try {
        rdma_proxy::validate_config(invalid_router_round_robin_config);
    } catch (const std::runtime_error&) {
        rejected_router_round_robin = true;
    }
    assert(rejected_router_round_robin);

    auto invalid_router_out_of_order_config = router_nvlink_config;
    invalid_router_out_of_order_config.nvlink_forward_threshold_tokens = 0;
    invalid_router_out_of_order_config.nvlink_forward_min_threshold_chunks = 1;
    invalid_router_out_of_order_config.nvlink_forward_max_threshold_chunks = 4;
    invalid_router_out_of_order_config.nvlink_forward_out_of_order_chunks_enabled = true;
    bool rejected_router_out_of_order = false;
    try {
        rdma_proxy::validate_config(invalid_router_out_of_order_config);
    } catch (const std::runtime_error&) {
        rejected_router_out_of_order = true;
    }
    assert(rejected_router_out_of_order);

    auto invalid_router_experts_config = router_config;
    invalid_router_experts_config.router_num_experts = 255;
    bool rejected_router_experts = false;
    try {
        rdma_proxy::validate_config(invalid_router_experts_config);
    } catch (const std::runtime_error&) {
        rejected_router_experts = true;
    }
    assert(rejected_router_experts);

    const char* nvlink_config_path = "nvlink_forward_config.json";
    {
        std::ofstream out(nvlink_config_path);
        out << R"json({
  "node_rank": 0,
  "num_nodes": 2,
  "local_gpu_index": 0,
  "num_gpus_per_node": 4,
  "num_tokens": 300,
  "token_dimension": 16,
  "tokens_per_chunk": 25,
  "num_qps_per_peer": 2,
  "rdma_device_name": "mlx5_0",
  "rdma_port": 1,
  "gid_index": 3,
  "cuda_device_id": 0,
  "completion_poll_batch_size": 16,
  "data_signal_interval": 0,
  "max_in_flight_chunks_per_qp": 2,
  "send_queue_depth": 256,
  "recv_queue_depth": 256,
  "cq_depth": 512,
  "dtype": "fp16",
  "mock_mode": true,
  "nvlink_forwarding_enabled": true,
  "nvlink_forward_threshold_chunks": 12,
  "nvlink_forward_min_threshold_chunks": 0,
  "nvlink_forward_max_threshold_chunks": 0,
  "nvlink_forward_out_of_order_chunks_enabled": false,
  "nvlink_forward_chunk_tokens": 100,
  "nvlink_forward_use_batch_api": true,
  "nvlink_forward_stream_nonblocking": true,
  "nvlink_forward_synchronize_batches": false,
  "nvlink_forward_completion_notifications_enabled": false,
  "nvlink_forward_notification_flush_only_enabled": false,
  "nvlink_forward_notification_flush_per_entry_enabled": false,
  "nvlink_forward_notification_flag_update_mode": "memcpy",
  "nvlink_forward_notification_queue_depth": 256,
  "nvlink_forward_notification_log_enabled": false,
  "nvlink_forward_notification_log_dir": "/tmp/rdma_cpu_proxy_test_notifications",
  "nvlink_forward_local_batch_sync_enabled": false,
  "nvlink_forward_synchronize_iteration": true,
  "nvlink_forward_log_batches": true,
  "log_qp_reports": true,
  "log_marker_wait_reports": true,
  "nvlink_forward_use_round_robin": true,
  "nvlink_routing_probability": 0.25,
  "nvlink_routing_seed": 1234,
  "nvlink_forward_destinations": [
    {"gpu_index": 1, "cuda_device_id": 1, "buffer_addr": "0x100000", "buffer_bytes": 9600},
    {"gpu_index": 2, "cuda_device_id": 2, "buffer_addr": "0x200000", "buffer_bytes": 9600},
    {"gpu_index": 3, "cuda_device_id": 3, "buffer_addr": "0x300000", "buffer_bytes": 9600}
  ],
  "peers": [
    {
      "node_rank": 1,
      "host": "node-b.example.com",
      "port": 18515
    }
  ]
})json";
    }
    const auto nvlink_config = rdma_proxy::load_config_file(nvlink_config_path);
    assert(nvlink_config.nvlink_forwarding_enabled);
    assert(!nvlink_config.nvlink_forward_local_batch_sync_enabled);
    assert(nvlink_config.nvlink_forward_log_batches);
    assert(nvlink_config.nvlink_forward_threshold_tokens == 0);
    assert(nvlink_config.nvlink_forward_threshold_chunks == 12);
    assert(nvlink_config.nvlink_forward_min_threshold_chunks == 0);
    assert(nvlink_config.nvlink_forward_max_threshold_chunks == 0);
    assert(!nvlink_config.nvlink_forward_out_of_order_chunks_enabled);
    assert(!rdma_proxy::nvlink_forward_dynamic_threshold_enabled(nvlink_config));
    assert(rdma_proxy::effective_nvlink_forward_threshold_tokens(nvlink_config) == 300);
    assert(!nvlink_config.nvlink_forward_completion_notifications_enabled);
    assert(!nvlink_config.nvlink_forward_notification_flush_only_enabled);
    assert(!nvlink_config.nvlink_forward_notification_flush_per_entry_enabled);
    assert(nvlink_config.nvlink_forward_notification_flag_update_mode ==
           rdma_proxy::NvlinkForwardNotificationFlagUpdateMode::kMemcpy);
    assert(nvlink_config.nvlink_forward_notification_queue_depth == 256);
    assert(!nvlink_config.nvlink_forward_notification_log_enabled);
    assert(nvlink_config.nvlink_forward_notification_log_dir == "/tmp/rdma_cpu_proxy_test_notifications");
    assert(nvlink_config.log_qp_reports);
    assert(nvlink_config.log_marker_wait_reports);
    assert(nvlink_config.nvlink_forward_use_round_robin);
    assert(nvlink_config.nvlink_routing_probability == 0.25);
    assert(nvlink_config.nvlink_routing_seed == 1234);
    assert(nvlink_config.nvlink_forward_destinations.size() == 3);
    assert(nvlink_config.nvlink_forward_destinations[0].buffer_addr == 0x100000ULL);

    auto invalid_threshold_config = nvlink_config;
    invalid_threshold_config.nvlink_forward_threshold_tokens = 301;
    bool rejected_mismatched_threshold = false;
    try {
        rdma_proxy::validate_config(invalid_threshold_config);
    } catch (const std::runtime_error&) {
        rejected_mismatched_threshold = true;
    }
    assert(rejected_mismatched_threshold);

    auto invalid_batch_sync_config = nvlink_config;
    invalid_batch_sync_config.nvlink_forward_local_batch_sync_enabled = true;
    invalid_batch_sync_config.nvlink_forward_synchronize_batches = false;
    bool rejected_batch_sync_without_stream_sync = false;
    try {
        rdma_proxy::validate_config(invalid_batch_sync_config);
    } catch (const std::runtime_error&) {
        rejected_batch_sync_without_stream_sync = true;
    }
    assert(rejected_batch_sync_without_stream_sync);

    auto valid_batch_sync_config = nvlink_config;
    valid_batch_sync_config.nvlink_forward_local_batch_sync_enabled = true;
    valid_batch_sync_config.nvlink_forward_synchronize_batches = true;
    rdma_proxy::validate_config(valid_batch_sync_config);

    auto invalid_notification_config = nvlink_config;
    invalid_notification_config.nvlink_forward_completion_notifications_enabled = true;
    invalid_notification_config.nvlink_forward_synchronize_batches = false;
    bool rejected_notifications_without_stream_sync = false;
    try {
        rdma_proxy::validate_config(invalid_notification_config);
    } catch (const std::runtime_error&) {
        rejected_notifications_without_stream_sync = true;
    }
    assert(rejected_notifications_without_stream_sync);

    auto valid_notification_config = nvlink_config;
    valid_notification_config.nvlink_forward_completion_notifications_enabled = true;
    valid_notification_config.nvlink_forward_synchronize_batches = true;
    valid_notification_config.nvlink_forward_notification_queue_depth = 8;
    rdma_proxy::validate_config(valid_notification_config);

    auto invalid_expert_gemm_config = valid_notification_config;
    invalid_expert_gemm_config.router_routing_enabled = true;
    invalid_expert_gemm_config.router_num_experts = 256;
    invalid_expert_gemm_config.router_top_k = 8;
    invalid_expert_gemm_config.expert_gemm_dimension = 8193;
    bool rejected_nondivisible_n_swizzle_groups = false;
    try {
        rdma_proxy::validate_config(invalid_expert_gemm_config);
    } catch (const std::runtime_error&) {
        rejected_nondivisible_n_swizzle_groups = true;
    }
    assert(rejected_nondivisible_n_swizzle_groups);

    auto valid_partial_n_tile_config = invalid_expert_gemm_config;
    valid_partial_n_tile_config.expert_gemm_max_swizzle_size = 3;
    valid_partial_n_tile_config.nvlink_forward_use_round_robin = false;
    rdma_proxy::validate_config(valid_partial_n_tile_config);

    auto invalid_flush_only_without_router = valid_notification_config;
    invalid_flush_only_without_router.nvlink_forward_notification_flush_only_enabled = true;
    bool rejected_flush_only_without_router = false;
    try {
        rdma_proxy::validate_config(invalid_flush_only_without_router);
    } catch (const std::runtime_error&) {
        rejected_flush_only_without_router = true;
    }
    assert(rejected_flush_only_without_router);

    auto invalid_flush_per_entry_without_router = valid_notification_config;
    invalid_flush_per_entry_without_router
        .nvlink_forward_notification_flush_per_entry_enabled = true;
    bool rejected_flush_per_entry_without_router = false;
    try {
        rdma_proxy::validate_config(invalid_flush_per_entry_without_router);
    } catch (const std::runtime_error&) {
        rejected_flush_per_entry_without_router = true;
    }
    assert(rejected_flush_per_entry_without_router);

    auto invalid_notification_log_config = nvlink_config;
    invalid_notification_log_config.nvlink_forward_notification_log_enabled = true;
    invalid_notification_log_config.nvlink_forward_completion_notifications_enabled = false;
    bool rejected_notification_log_without_notifications = false;
    try {
        rdma_proxy::validate_config(invalid_notification_log_config);
    } catch (const std::runtime_error&) {
        rejected_notification_log_without_notifications = true;
    }
    assert(rejected_notification_log_without_notifications);

    auto valid_notification_log_config = valid_notification_config;
    valid_notification_log_config.nvlink_forward_notification_log_enabled = true;
    valid_notification_log_config.nvlink_forward_notification_log_dir =
        "/tmp/rdma_cpu_proxy_test_notifications_valid";
    rdma_proxy::validate_config(valid_notification_log_config);

    auto dynamic_config = nvlink_config;
    dynamic_config.nvlink_forward_use_round_robin = false;
    dynamic_config.nvlink_forward_min_threshold_chunks = 2;
    dynamic_config.nvlink_forward_max_threshold_chunks = 4;
    assert(rdma_proxy::nvlink_forward_dynamic_threshold_enabled(dynamic_config));
    rdma_proxy::validate_config(dynamic_config);

    auto dynamic_out_of_order_config = dynamic_config;
    dynamic_out_of_order_config.nvlink_forward_out_of_order_chunks_enabled = true;
    rdma_proxy::validate_config(dynamic_out_of_order_config);

    auto invalid_out_of_order_infinite_config = dynamic_out_of_order_config;
    invalid_out_of_order_infinite_config.num_iterations = 0;
    bool rejected_out_of_order_infinite = false;
    try {
        rdma_proxy::validate_config(invalid_out_of_order_infinite_config);
    } catch (const std::runtime_error&) {
        rejected_out_of_order_infinite = true;
    }
    assert(rejected_out_of_order_infinite);

    auto invalid_dynamic_range_config = dynamic_config;
    invalid_dynamic_range_config.nvlink_forward_min_threshold_chunks = 5;
    invalid_dynamic_range_config.nvlink_forward_max_threshold_chunks = 4;
    bool rejected_dynamic_range = false;
    try {
        rdma_proxy::validate_config(invalid_dynamic_range_config);
    } catch (const std::runtime_error&) {
        rejected_dynamic_range = true;
    }
    assert(rejected_dynamic_range);

    auto invalid_dynamic_round_robin_config = dynamic_config;
    invalid_dynamic_round_robin_config.nvlink_forward_use_round_robin = true;
    bool rejected_dynamic_round_robin = false;
    try {
        rdma_proxy::validate_config(invalid_dynamic_round_robin_config);
    } catch (const std::runtime_error&) {
        rejected_dynamic_round_robin = true;
    }
    assert(rejected_dynamic_round_robin);

    std::cout << rdma_proxy::config_summary(config) << '\n';
    std::cout << "test_config passed\n";
    return 0;
}
