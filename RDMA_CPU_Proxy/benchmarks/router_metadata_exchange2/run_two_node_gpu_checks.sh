#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
node_rank="${1:?pass node rank 0 or 1; run on both nodes with identical options}"
shift
modes=(eager graph)
for arg in "$@"; do
  if [[ "${arg}" == "--native-metadata-launch" ]]; then modes=(graph); fi
done
for mode in "${modes[@]}"; do
  for tokens in 1 33 257 8193 16385; do
    bash "${script_dir}/run_two_node_torchrun.sh" "${node_rank}" --mode "${mode}" --tokens "${tokens}" \
      --warmup 1 --iterations 6 --route-pattern cycle --verify all \
      --seed 18446744073709551614 \
      --output "${script_dir}/results/check_two_${mode}_${tokens}.json" "$@"
  done
done
