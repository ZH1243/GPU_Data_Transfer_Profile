#!/usr/bin/env bash
# Sourced by the two-node launchers. Keep the original Hopper deployment as
# the default; Blackwell deployments must supply their actual network map.
gemm_arch="${RDMA_PROXY_GEMM_ARCH:-auto}"
case "${gemm_arch}" in
    auto|hopper) default_build=build-hopper ;;
    blackwell)
        default_build=build-blackwell
        : "${MASTER_ADDR:?Set MASTER_ADDR to node 0's rendezvous address}"
        : "${PEER_HOST:?Set PEER_HOST to the other node's proxy address}"
        : "${CUDA_DEVICE_MAP:?Set CUDA_DEVICE_MAP to the local CUDA device indices}"
        : "${RDMA_DEVICE_MAP:?Set RDMA_DEVICE_MAP to the matching local NIC entries}"
        : "${RDMA_GID_INDEX:?Set RDMA_GID_INDEX for this fabric (-1 for no global addressing)}"
        ;;
    *) echo "RDMA_PROXY_GEMM_ARCH must be auto, hopper, or blackwell" >&2; exit 2 ;;
esac
build_dir="${RDMA_PROXY_BUILD_DIR:-${project_dir}/${default_build}}"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"
master_addr="${MASTER_ADDR:-28.59.80.49}"
master_port="${MASTER_PORT:-29500}"
cuda_device_map="${CUDA_DEVICE_MAP:-0,1,2,3,4,5,6,7}"
rdma_device_map="${RDMA_DEVICE_MAP:-mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5}"
IFS=',' read -r -a cuda_devices <<< "${cuda_device_map}"
nproc_per_node="${NPROC_PER_NODE:-${#cuda_devices[@]}}"
site_args=("--gemm-arch=${gemm_arch}")
if [[ -n "${PEER_HOST:-}" ]]; then site_args+=("--peer_host=${PEER_HOST}"); fi
if [[ -n "${RDMA_GID_INDEX:-}" ]]; then site_args+=("--gid_index=${RDMA_GID_INDEX}"); fi
if [[ -n "${RDMA_PORT:-}" ]]; then site_args+=("--rdma_port=${RDMA_PORT}"); fi
