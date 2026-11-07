#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
node_rank="${1:?pass node rank 0 or 1; run on both nodes with identical options}"
shift
modes=(eager graph)
serialized=false
skip_staging=false
direct_output=false
receiver=false
receiver_overlap=false
for arg in "$@"; do
  if [[ "${arg}" == "--d2h-after-local-staging" || "${arg}" == "--local-staging-after-d2h" ]]; then serialized=true; fi
  if [[ "${arg}" == "--skip-local-staging" ]]; then skip_staging=true; fi
  if [[ "${arg}" == "--native-metadata-launch" ]]; then modes=(graph); fi
  if [[ "${arg}" == "--direct-x3-x4-output" ]]; then direct_output=true; fi
  if [[ "${arg}" == "--receiver-kernel" ]]; then receiver=true; fi
  if [[ "${arg}" == "--receiver-overlap-exchange" ]]; then receiver_overlap=true; fi
done
for mode in "${modes[@]}"; do
  for tokens in 1 33 257 8193 16385; do
    bash "${script_dir}/run_two_node_torchrun.sh" "${node_rank}" --mode "${mode}" --tokens "${tokens}" \
      --warmup 1 --iterations 6 --route-pattern cycle --verify all \
      --seed 18446744073709551614 \
      --output "${script_dir}/results/check_two_${mode}_${tokens}.json" "$@"
  done
  # Byte-copy fallback and partial vectorized rows; cycle also checks empty
  # local gathers, changing lengths, untouched tails and fresh input payloads.
  for width in 7 8; do
    bash "${script_dir}/run_two_node_torchrun.sh" "${node_rank}" --mode "${mode}" --tokens 33 \
      --token-dimension "${width}" --warmup 1 --iterations 6 --route-pattern cycle --verify all \
      --output "${script_dir}/results/check_two_${mode}_staging_d${width}.json" "$@"
  done
  if ! "${serialized}"; then
    bash "${script_dir}/run_two_node_torchrun.sh" "${node_rank}" --mode "${mode}" --tokens 33 \
      --skip-local-staging --warmup 1 --iterations 3 --route-pattern cycle --verify all \
      --output "${script_dir}/results/check_two_${mode}_no_staging.json" "$@"
  fi

  if ! "${skip_staging}" && ! "${serialized}"; then
    if ! "${receiver}"; then
      bash "${script_dir}/run_two_node_torchrun.sh" "${node_rank}" --mode "${mode}" --tokens 33 \
        --d2h-after-local-staging --warmup 1 --iterations 6 --route-pattern cycle --verify all \
        --output "${script_dir}/results/check_two_${mode}_serialized_staging.json" "$@"
    fi
    bash "${script_dir}/run_two_node_torchrun.sh" "${node_rank}" --mode "${mode}" --tokens 33 \
      --local-staging-after-d2h --warmup 1 --iterations 6 --route-pattern cycle --verify all \
      --output "${script_dir}/results/check_two_${mode}_staging_after_d2h.json" "$@"
  fi

  if ! "${direct_output}"; then
    # Exercise producer-owned remote output, including empty/full slices and
    # masks whose start address changes each iteration. Exact verification
    # checks both canonical metadata and received x3/x4 records.
    bash "${script_dir}/run_two_node_torchrun.sh" "${node_rank}" --mode "${mode}" --tokens 257 \
      --direct-x3-x4-output --warmup 1 --iterations 6 --route-pattern cycle --verify all \
      --output "${script_dir}/results/check_two_${mode}_direct_output.json" "$@"
  fi
  if "${receiver}" && ! "${receiver_overlap}"; then
    # Receiver baseline runs also exercise the alternate GPU event dependency.
    bash "${script_dir}/run_two_node_torchrun.sh" "${node_rank}" --mode "${mode}" --tokens 257 \
      --receiver-overlap-exchange --warmup 1 --iterations 6 --route-pattern cycle --verify all \
      --output "${script_dir}/results/check_two_${mode}_receiver_overlap.json" "$@"
  fi
done
