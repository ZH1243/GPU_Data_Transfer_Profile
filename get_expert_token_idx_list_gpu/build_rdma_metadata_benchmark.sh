#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${RDMA_METADATA_BUILD_DIR:-${script_dir}/build}"
mkdir -p "${build_dir}"
"${NVCC:-nvcc}" -O3 -std=c++17 -lineinfo -arch="${CUDA_ARCH:-sm_90}" \
    -DEXPERT_TOKEN_IDX_STANDALONE=1 \
    "${script_dir}/benchmark_rdma_metadata.cu" \
    "${script_dir}/expert_token_idx_kernel.cu" \
    -o "${build_dir}/benchmark_rdma_metadata"
echo "Built ${build_dir}/benchmark_rdma_metadata"
