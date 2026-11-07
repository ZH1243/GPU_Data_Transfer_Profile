#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Warp/chunk/scan carry boundaries; cycle includes empty receive lists and
# growth/shrinkage using the same allocations. Full content checks every time.
modes=(eager graph)
serialized=false
skip_staging=false
for arg in "$@"; do
  if [[ "${arg}" == "--d2h-after-local-staging" || "${arg}" == "--local-staging-after-d2h" ]]; then serialized=true; fi
  if [[ "${arg}" == "--skip-local-staging" ]]; then skip_staging=true; fi
  if [[ "${arg}" == "--native-metadata-launch" ]]; then modes=(graph); fi
done
for mode in "${modes[@]}"; do
  for tokens in 1 33 257 8193 16385; do
    bash "${script_dir}/run_single_node_torchrun.sh" --mode "${mode}" --tokens "${tokens}" \
      --warmup 1 --iterations 3 --route-pattern cycle --verify all \
      --seed 18446744073709551614 \
      --output "${script_dir}/results/check_${mode}_${tokens}.json" "$@"
  done
  for width in 7 8; do
    bash "${script_dir}/run_single_node_torchrun.sh" --mode "${mode}" --tokens 33 \
      --token-dimension "${width}" --warmup 1 --iterations 3 --route-pattern cycle --verify all \
      --output "${script_dir}/results/check_${mode}_staging_d${width}.json" "$@"
  done
  if ! "${serialized}"; then
    bash "${script_dir}/run_single_node_torchrun.sh" --mode "${mode}" --tokens 33 \
      --skip-local-staging --warmup 1 --iterations 3 --route-pattern cycle --verify all \
      --output "${script_dir}/results/check_${mode}_no_staging.json" "$@"
  fi

  if ! "${skip_staging}" && ! "${serialized}"; then
    bash "${script_dir}/run_single_node_torchrun.sh" --mode "${mode}" --tokens 33 \
      --d2h-after-local-staging --warmup 1 --iterations 6 --route-pattern cycle --verify all \
      --output "${script_dir}/results/check_${mode}_serialized_staging.json" "$@"
    bash "${script_dir}/run_single_node_torchrun.sh" --mode "${mode}" --tokens 33 \
      --local-staging-after-d2h --warmup 1 --iterations 6 --route-pattern cycle --verify all \
      --output "${script_dir}/results/check_${mode}_staging_after_d2h.json" "$@"
  fi

done
