#!/usr/bin/env python3
"""Launch one CPU-only C++ metadata proxy thread per torchrun worker."""
import argparse
import ctypes
import os
from pathlib import Path
import sys
import threading


INT_FIELDS = ("rank", "world", "gpus_per_node", "port_base", "node_port_stride",
              "num_tokens", "num_experts", "top_k", "token_dimension", "element_bytes",
              "tokens_per_chunk", "datasets", "warmup", "iterations", "preserialized", "verify", "binary", "disable_expert_index_validation", "local_shared_memory", "receive_drain", "receive_drain_max_steps", "same_index_aggregation", "rdma", "rdma_port", "rdma_gid_index")


class NativeConfig(ctypes.Structure):
    _fields_ = ([(name, ctypes.c_int32) for name in INT_FIELDS] +
                [("seed", ctypes.c_uint64), ("timeout", ctypes.c_double),
                 ("hosts", ctypes.c_char_p), ("bind_host", ctypes.c_char_p),
                 ("output", ctypes.c_char_p), ("rdma_devices", ctypes.c_char_p)])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def default_library():
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return Path(os.environ.get("ROUTER_METADATA_BENCH_LIBRARY",
                str(Path(__file__).resolve().parent / "build" / ("librouter_metadata_bench" + suffix))))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hosts", required=True, help="comma-separated TCP node addresses, in node-rank order")
    p.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("LOCAL_WORLD_SIZE", "8")))
    p.add_argument("--port-base", type=int, default=20515)
    p.add_argument("--node-port-stride", type=int, default=0,
                   help="port offset per node; use GPUs-per-node to emulate nodes on one host")
    p.add_argument("--bind-host", default="0.0.0.0")
    p.add_argument("--num-tokens", type=int, default=20480)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--token-dimension", type=int, default=4096)
    p.add_argument("--element-bytes", type=int, default=2)
    p.add_argument("--tokens-per-chunk", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--datasets", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--preserialized", action="store_true")
    p.add_argument("--disable-expert-index-validation", action="store_true",
                   help="binary mode: skip expert index bounds and sorted-order scans; retain structural checks")
    p.add_argument("--same-index-aggregation", action="store_true",
                   help="binary mode: aggregate remote metadata through matching local ranks; enables local shared memory")
    p.add_argument("--receive-drain-max-steps", type=int, default=16,
                   help="maximum receive steps per ready socket when draining (1..1024)")
    p.add_argument("--receive-drain", action="store_true",
                   help="advance readable sockets until blocked, complete, or the receive-step budget is reached")
    p.add_argument("--local-shared-memory", action="store_true",
                   help="binary mode: exchange same-node expert metadata through shared host memory")
    p.add_argument("--wire-format", choices=("binary", "text"), default="binary",
                   help="binary sends native array bytes; text retains the comparison baseline")
    p.add_argument("--output", type=Path, help="rank-0 JSON report path")
    p.add_argument("--library", type=Path, default=default_library(),
                   help="native shared library (build with CMake before launching)")
    p.add_argument("--verify", action="store_true",
                   help="check received content against regenerated source metadata outside timing")
    p.add_argument("--transport", choices=("tcp", "rdma"), default="tcp",
                   help="RDMA enables binary same-index aggregation and shared memory")
    p.add_argument("--rdma-device-map", default=os.environ.get("RDMA_DEVICE_MAP", ""),
                   help="one verbs device, or a comma-separated device per local GPU")
    p.add_argument("--rdma-port", type=int, default=1)
    p.add_argument("--rdma-gid-index", type=int, default=3)
    a = p.parse_args()
    a.rdma = a.transport == "rdma"
    a.same_index_aggregation = a.same_index_aggregation or a.rdma
    if a.rdma:
        devices = a.rdma_device_map.split(",")
        require(all(devices) and len(devices) in (1, a.gpus_per_node), "RDMA requires a device map with one device or one per local GPU")
        require(1 <= a.rdma_port <= 255 and -1 <= a.rdma_gid_index <= 255, "invalid RDMA port/GID index")
    require(1 <= a.receive_drain_max_steps <= 1024, "--receive-drain-max-steps must be in 1..1024")
    a.binary = a.wire_format == "binary"
    require(not a.same_index_aggregation or a.binary, "--same-index-aggregation requires --wire-format=binary")
    a.local_shared_memory = a.local_shared_memory or a.same_index_aggregation
    require(not a.disable_expert_index_validation or a.binary,
            "--disable-expert-index-validation requires --wire-format=binary; text uses the unchanged production deserializer")
    require(not a.local_shared_memory or a.binary,
            "--local-shared-memory requires --wire-format=binary")
    a.hosts = a.hosts.split(",")
    a.rank = int(os.environ["RANK"])
    a.world = int(os.environ["WORLD_SIZE"])
    require(all(a.hosts), "empty node host")
    require(1 <= a.gpus_per_node <= 8 and a.world == len(a.hosts) * a.gpus_per_node,
            "world size must equal nodes * GPUs per node (1..8)")
    require(0 <= a.rank < a.world and int(os.environ["LOCAL_RANK"]) == a.rank % a.gpus_per_node,
            "invalid rank mapping")
    for name in ("num_tokens", "num_experts", "top_k", "token_dimension", "element_bytes",
                 "tokens_per_chunk", "datasets", "iterations", "timeout"):
        require(getattr(a, name) > 0, name + " must be positive")
    require(a.warmup >= 0 and a.top_k <= a.num_experts and a.num_experts % a.world == 0,
            "invalid warmup/top-k/expert partition")
    require(a.node_port_stride >= 0 and 0 < a.port_base and
            a.port_base + (len(a.hosts) - 1) * a.node_port_stride + a.gpus_per_node <= 65536,
            "invalid port range")
    return a


def main():
    a = arguments()
    require(a.library.is_file(),
            f"Native library not found: {a.library}. Build first: "
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j")
    library = ctypes.CDLL(str(a.library.resolve()))
    library.router_metadata_bench_abi_version.restype = ctypes.c_uint32
    require(library.router_metadata_bench_abi_version() == 7, "native benchmark ABI mismatch; rebuild the shared library")
    library.router_metadata_bench_run.argtypes = [ctypes.POINTER(NativeConfig),
                                                 ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
    library.router_metadata_bench_run.restype = ctypes.c_int
    for name in INT_FIELDS:
        require(-(1 << 31) <= int(getattr(a, name)) < (1 << 31), name + " exceeds int32")
    require(0 <= a.seed < (1 << 64), "seed exceeds uint64")
    config = NativeConfig(**{name: int(getattr(a, name)) for name in INT_FIELDS},
                          seed=a.seed, timeout=a.timeout, hosts=",".join(a.hosts).encode(),
                          rdma_devices=a.rdma_device_map.encode(), bind_host=a.bind_host.encode(), output=os.fsencode(a.output) if a.output else b"")
    error = ctypes.create_string_buffer(4096)
    outcome = []

    def run_native():
        try:
            # CDLL releases the GIL. All benchmark work and timing occur in C++.
            status = library.router_metadata_bench_run(ctypes.byref(config), error, len(error))
            outcome.append(None if status == 0 else RuntimeError(error.value.decode(errors="replace")))
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=run_native, name="router-metadata-proxy", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(timeout=0.5)
    require(bool(outcome), "native proxy thread returned no status")
    if outcome[0] is not None:
        raise outcome[0]


if __name__ == "__main__":
    main()
