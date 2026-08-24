#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${RDMA_PROXY_BUILD_DIR:-${project_dir}/build-hopper}"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"
master_addr="${MASTER_ADDR:-28.49.38.169}"
master_port="${MASTER_PORT:-29500}"

"${torchrun_bin}" \
    --nnodes=2 \
    --nproc_per_node=7 \
    --node_rank=0 \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    --max_restarts=0 \
    "${script_dir}/rdma_proxy_worker.py" \
    --library="${build_dir}/librdma_cpu_proxy.so" \
    --cuda-device-map=0,1,2,3,5,6,7 \
    --rdma-device-map=mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5 \
    --listen-port-base=18515 \
    --config "${project_dir}/config/node0_config.json" \
    --node_rank=0 \
    --num_nodes=2 \
    --token_dimension=4096 \
    --num_iterations=5 \
    --tokens_per_chunk=32 \
    --num_qps_per_peer=8 \
    --num_tokens=10240 \
    --cpu_affinity=auto \
    --fill_test_data=false \
    --validate_data=false \
    --completion_poll_batch_size=64 \
    --max_in_flight_chunks_per_qp=4 \
    --rdma_chunk_per_token_sge_enabled=false \
    --rdma_discontinuous_token_payload_enabled=false \
    --router_routing_enabled=true \
    --router_computation_kernel_enabled=true \
    --num_experts=112 \
    --top_k=16 \
    --router_seed=1234 \
    --nvlink_forward_notification_flush_per_entry_enabled=false \
    --sequential_peer_transfers=true \
    --nvlink_forwarding_enabled=true \
    --nvlink_forward_use_round_robin=false \
    --nvlink_forward_use_batch_api=true \
    --nvlink_forward_synchronize_batches=true \
    --nvlink_forward_completion_notifications_enabled=true \
    --nvlink_forward_notification_log_enabled=false \
    --nvlink_forward_notification_flush_only_enabled=false \
    --nvlink_forward_local_batch_sync_enabled=false \
    --nvlink_forward_log_batches=false \
    --local_iteration_sync_enabled=true \
    --log_qp_reports=false \
    --mock_mode=false
