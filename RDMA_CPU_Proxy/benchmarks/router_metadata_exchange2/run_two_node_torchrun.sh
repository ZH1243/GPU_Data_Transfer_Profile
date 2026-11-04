#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
node_rank="${1:?usage: bash run_two_node_torchrun.sh NODE_RANK [benchmark options]}"
shift
if [[ "${node_rank}" != 0 && "${node_rank}" != 1 ]]; then
  echo "NODE_RANK must be 0 or 1" >&2
  exit 2
fi
exec "${TORCHRUN_BIN:-torchrun}" --nnodes=2 --nproc_per_node=8 --node_rank="${node_rank}" \
  --master_addr="${MASTER_ADDR:-28.59.80.49}" --master_port="${MASTER_PORT:-29520}" --max_restarts=0 \
  "${script_dir}/benchmark.py" \
  --library "${RME2_LIBRARY:-${script_dir}/build/librouter_metadata_exchange2.so}" \
  --rdma-device-map "${RDMA_DEVICE_MAP:-mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5}" \
  --output "${RESULT_JSON:-${script_dir}/results/metadata_exchange2_two_node.json}" "$@"
