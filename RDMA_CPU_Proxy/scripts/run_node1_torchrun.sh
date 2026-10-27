#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${RDMA_PROXY_BUILD_DIR:-${project_dir}/build-hopper}"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"
master_addr="${MASTER_ADDR:-28.59.80.49}"
master_port="${MASTER_PORT:-29500}"

"${torchrun_bin}" \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    --max_restarts=0 \
    "${script_dir}/rdma_proxy_worker.py" \
    --library="${build_dir}/librdma_cpu_proxy.so" \
    --cuda-device-map=0,1,2,3,4,5,6,7 \
    --rdma-device-map=mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5 \
    --listen-port-base=18515 \
    --config "${project_dir}/config/node1_config.json" \
    --node_rank=1 \
    --num_nodes=2 \
    --token_dimension=4096 \
    --num_iterations=5 \
    --tokens_per_chunk=32 \
    --num_qps_per_peer=8 \
    --num_tokens=20480 \
    --cpu_affinity=auto \
    --fill_test_data=false \
    --validate_data=false \
    --completion_poll_batch_size=64 \
    --max_in_flight_chunks_per_qp=8 \
    --rdma_chunk_per_token_sge_enabled=false \
    --rdma_discontinuous_token_payload_enabled=false \
    --router_routing_enabled=true \
    --router_computation_kernel_enabled=true \
    --num_experts=128 \
    --top_k=8 \
    --router_seed=1234 \
    --local_forwarding_rdma_overlap_enabled=true \
    --nvlink_forward_notification_flush_per_entry_enabled=false \
    --sequential_peer_transfers=true \
    --nvlink_forwarding_enabled=true \
    --nvlink_forward_use_round_robin=false \
    --nvlink_forward_use_batch_api=true \
    --nvlink_forward_synchronize_batches=true \
    --nvlink_forward_completion_notifications_enabled=true \
    --nvlink_forward_preparation_enabled=true \
    --nvlink_forward_prepared_batch_chunks=64 \
    --nvlink_forward_submit_epilogue_enabled=true \
    --nvlink_forward_prepared_atomic_ring_enabled=true \
    --nvlink_forward_completion_mode=stream_sync \
    --nvlink_forward_prepared_queue_depth=16 \
    --nvlink_forward_ping_pong2_enabled=false \
    --nvlink_forward_local_staging_prepared_enabled=true \
    --nvlink_forward_prepared_nvtx_enabled=false \
    --nvlink_forward_notification_log_enabled=false \
    --nvlink_forward_notification_flush_only_enabled=false \
    --nvlink_forward_local_batch_sync_enabled=false \
    --nvlink_forward_log_batches=false \
    --local_iteration_sync_enabled=true \
    --log_qp_reports=false \
    --mock_mode=false \
    --router_local_input_staging_enabled=true \
    --concurrent-kernel \
    --grouped-gemm-m-tile=128 \
    --grouped-gemm-n-tile=128 \
    --grouped-gemm-output-dimension=512 \
    --grouped-gemm-cluster-m=2 \
    --grouped-gemm-max-swizzle-size=8 \
    --grouped-gemm-activation=swiglu \
    --down-projection2 --down-sm-count=70 --local-gpu-combine \
    --down-n-tile-major-output --down-tile-m=128 --down-tile-n=128 \
    --down-pingpong --down-gather-table --down-gather-table-expert-waves \
    --down-max-swizzle-size=1 --down-gather-table-wave-barrier \
    --down-epilogue-store bulk_rows_reduce \
    --down-gather-table-publish-ready \
    --return-token-results --return-token-results-per-group \
    --check-return-token-results \
    --nvlink_return_completion_notifications_enabled=true \
    --nvlink_return_ready_gpu_publication_enabled=true \
    --intra-node-combine \
    --intra-node-combine-ctas=8 \
    --intra-node-combine-n-tile-major-output \
    --check-intra-node-combine \
    --intra-node-combine-publish-ready \
    --rdma-return-token-results \
    --check-rdma-return-token-results \
    --inter-node-combine \
    --inter-node-combine-ctas=78 \
    --check-inter-node-combine \
    "$@"
