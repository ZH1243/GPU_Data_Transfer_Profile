#!/usr/bin/env bash
set -euo pipefail
node_rank="$1"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
node0_host="${NODE0_HOST:-28.59.80.49}"
node1_host="${NODE1_HOST:-29.162.224.226}"
processes="${NPROC_PER_NODE:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${TORCHRUN_BIN:-torchrun}" \
    --nnodes=2 --nproc_per_node="${processes}" --node_rank="${node_rank}" \
    --master_addr="${MASTER_ADDR:-${node0_host}}" --master_port="${MASTER_PORT:-29510}" \
    --max_restarts=0 "${script_dir}/benchmark.py" \
    --hosts="${node0_host},${node1_host}" --gpus-per-node="${processes}" \
    --port-base="${METADATA_PORT_BASE:-20515}" \
    --num-tokens=20480 --num-experts=128 --top-k=8 \
    --datasets=4 --warmup=5 --iterations=50 \
    --output="${RESULT_JSON:-${script_dir}/results/metadata_exchange.json}" "$@"
