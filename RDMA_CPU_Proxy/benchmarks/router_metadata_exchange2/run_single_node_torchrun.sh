#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${TORCHRUN_BIN:-torchrun}" --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 \
  "${script_dir}/benchmark.py" \
  --library "${RME2_LIBRARY:-${script_dir}/build/librouter_metadata_exchange2.so}" \
  --output "${RESULT_JSON:-${script_dir}/results/metadata_exchange2.json}" "$@"
