#!/usr/bin/env python3
"""Load one RDMA CPU proxy into each torchrun worker process."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import shlex
import sys


EXPECTED_ABI_VERSION = 1


def environment_integer(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is not set; launch this worker with torchrun")
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from error


def comma_separated_values(name: str, value: str, expected_count: int) -> list[str]:
    values = [item.strip() for item in value.split(",")]
    if any(not item for item in values):
        raise RuntimeError(f"{name} contains an empty entry: {value!r}")
    if len(values) != expected_count:
        raise RuntimeError(
            f"{name} has {len(values)} entries, but torchrun launched "
            f"LOCAL_WORLD_SIZE={expected_count} workers"
        )
    return values


def build_proxy_argv(
    args: argparse.Namespace,
    proxy_args: list[str],
) -> tuple[list[str], dict[str, int | str]]:
    local_rank = environment_integer("LOCAL_RANK")
    local_world_size = environment_integer("LOCAL_WORLD_SIZE")
    rank = environment_integer("RANK")
    world_size = environment_integer("WORLD_SIZE")

    if args.launcher_local_rank is not None and args.launcher_local_rank != local_rank:
        raise RuntimeError(
            f"--local-rank={args.launcher_local_rank} disagrees with LOCAL_RANK={local_rank}"
        )
    if local_world_size <= 0 or world_size <= 0:
        raise RuntimeError("LOCAL_WORLD_SIZE and WORLD_SIZE must be positive")
    if world_size % local_world_size != 0:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} is not divisible by LOCAL_WORLD_SIZE={local_world_size}"
        )
    if local_rank < 0 or local_rank >= local_world_size:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside [0, {local_world_size})"
        )

    num_nodes = world_size // local_world_size
    group_rank_value = os.environ.get("GROUP_RANK")
    node_rank = int(group_rank_value) if group_rank_value is not None else rank // local_world_size
    if node_rank < 0 or node_rank >= num_nodes:
        raise RuntimeError(f"derived node rank {node_rank} is outside [0, {num_nodes})")

    cuda_devices = comma_separated_values(
        "--cuda-device-map", args.cuda_device_map, local_world_size
    )
    rdma_devices = comma_separated_values(
        "--rdma-device-map", args.rdma_device_map, local_world_size
    )
    try:
        cuda_device_id = int(cuda_devices[local_rank])
    except ValueError as error:
        raise RuntimeError(
            f"CUDA device map entry {cuda_devices[local_rank]!r} is not an integer"
        ) from error

    listen_port = args.listen_port_base + local_rank
    if listen_port < 1 or listen_port > 65535:
        raise RuntimeError(f"derived proxy listen port {listen_port} is invalid")

    # Generated values are appended so torchrun's topology and the per-rank
    # device/NIC maps take precedence over defaults in the JSON configuration.
    generated_args = [
        f"--node_rank={node_rank}",
        f"--num_nodes={num_nodes}",
        f"--num_gpus_per_node={local_world_size}",
        f"--local_gpu_index={local_rank}",
        f"--cuda_device_id={cuda_device_id}",
        f"--rdma_device_name={rdma_devices[local_rank]}",
        f"--listen_port={listen_port}",
    ]
    metadata: dict[str, int | str] = {
        "rank": rank,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "cuda_device_id": cuda_device_id,
        "rdma_device_name": rdma_devices[local_rank],
        "listen_port": listen_port,
    }
    return ["rdma_cpu_proxy", *proxy_args, *generated_args], metadata


def load_proxy_library(path: Path) -> ctypes.CDLL:
    if not path.is_file():
        raise RuntimeError(
            f"RDMA proxy shared library does not exist: {path}; "
            "build the rdma_cpu_proxy_shared target first"
        )

    library = ctypes.CDLL(str(path))
    library.rdma_proxy_abi_version.argtypes = []
    library.rdma_proxy_abi_version.restype = ctypes.c_int
    library.rdma_proxy_run_argv.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.rdma_proxy_run_argv.restype = ctypes.c_int
    library.rdma_proxy_last_error.argtypes = []
    library.rdma_proxy_last_error.restype = ctypes.c_char_p

    abi_version = library.rdma_proxy_abi_version()
    if abi_version != EXPECTED_ABI_VERSION:
        raise RuntimeError(
            f"RDMA proxy ABI mismatch: worker expects {EXPECTED_ABI_VERSION}, "
            f"library provides {abi_version}"
        )
    return library


def run_proxy(library: ctypes.CDLL, proxy_argv: list[str]) -> int:
    encoded_argv = [os.fsencode(argument) for argument in proxy_argv]
    c_argv = (ctypes.c_char_p * len(encoded_argv))(*encoded_argv)
    status = library.rdma_proxy_run_argv(len(encoded_argv), c_argv)
    if status != 0:
        error_pointer = library.rdma_proxy_last_error()
        error = (
            error_pointer.decode("utf-8", errors="replace")
            if error_pointer
            else "unknown proxy error"
        )
        raise RuntimeError(error)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one librdma_cpu_proxy.so instance in a torchrun worker",
        allow_abbrev=False,
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cuda-device-map", required=True)
    parser.add_argument("--rdma-device-map", required=True)
    parser.add_argument("--listen-port-base", type=int, default=18515)
    parser.add_argument("--local-rank", "--local_rank", dest="launcher_local_rank", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the derived C++ argv without loading the shared library",
    )
    args, proxy_args = parser.parse_known_args()

    proxy_argv, metadata = build_proxy_argv(args, proxy_args)
    print(
        "starting embedded RDMA CPU proxy "
        f"rank={metadata['rank']} node_rank={metadata['node_rank']} "
        f"local_gpu={metadata['local_rank']} cuda_device={metadata['cuda_device_id']} "
        f"rdma_device={metadata['rdma_device_name']} "
        f"listen_port={metadata['listen_port']}",
        flush=True,
    )
    if args.dry_run:
        print(shlex.join(proxy_argv), flush=True)
        return 0

    library = load_proxy_library(args.library.resolve())
    return run_proxy(library, proxy_argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"rdma_proxy_worker failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)

