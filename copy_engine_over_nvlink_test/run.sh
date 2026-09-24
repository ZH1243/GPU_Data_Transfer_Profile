#!/usr/bin/env bash
set -eo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
executable=${NVLINK_PROXY_BIN:-"$script_dir/build/nvlink_all_to_all"}
devices=0,1,2,3,4,5,6,7
args=()
while (($#)); do
    case "$1" in
        --devices)
            if (($# < 2)); then echo 'Missing --devices value' >&2; exit 2; fi
            devices=$2; shift 2 ;;
        --rank|--shared-file)
            echo "$1 is assigned by run.sh" >&2; exit 2 ;;
        --help)
            echo 'Usage: ./run.sh [--devices 0,1,...] [proxy options]'
            echo 'Launches one executable per GPU; assigns --rank and --shared-file.'
            echo 'NVLINK_PROXY_BIN overrides the executable; NVLINK_SHM_DIR defaults to /dev/shm.'
            "$executable" --help
            exit 0 ;;
        *) args+=("$1"); shift ;;
    esac
done
if [[ ! $devices =~ ^[0-9]+(,[0-9]+)+$ ]]; then
    echo '--devices must contain at least two comma-separated GPU ordinals' >&2; exit 2
fi
IFS=, read -r -a gpu_ids <<< "$devices"
if ((${#gpu_ids[@]} > 32)); then echo 'At most 32 GPUs are supported' >&2; exit 2; fi
if [[ ! -x $executable ]]; then echo "Executable not found: $executable; build the project first" >&2; exit 2; fi
run_dir=$(mktemp -d "${NVLINK_SHM_DIR:-/dev/shm}/nvlink-a2a.XXXXXXXX")
pids=()
cleanup() {
    local pid
    for pid in "${pids[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    rm -f -- "$run_dir/shared" "$run_dir/shared.initial"
    rmdir -- "$run_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((rank=0; rank<${#gpu_ids[@]}; ++rank)); do
    "$executable" --rank "$rank" --shared-file "$run_dir/shared" \
        --devices "$devices" "${args[@]}" &
    pids+=("$!")
done
# Poll every child, so a failed higher rank is noticed even if rank 0 is waiting.
while ((${#pids[@]})); do
    remaining=()
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            remaining+=("$pid")
        else
            if wait "$pid"; then :; else
                status=$?
                echo "Proxy PID $pid failed (exit $status); stopping remaining proxies" >&2
                exit "$status"
            fi
        fi
    done
    pids=("${remaining[@]}")
    if ((${#pids[@]})); then sleep 0.1; fi
done
