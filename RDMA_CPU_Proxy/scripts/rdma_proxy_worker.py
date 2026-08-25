#!/usr/bin/env python3
"""Load one RDMA CPU proxy into each torchrun worker process."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import queue
import shlex
import sys
import threading


EXPECTED_ABI_VERSION = 1
CONCURRENT_KERNEL_ELEMENTS = 1 << 20

DEVICE_BUFFER_RDMA_SEND = 1
DEVICE_BUFFER_RDMA_RECEIVE = 2
DEVICE_BUFFER_NVLINK_RECEIVE = 3
DEVICE_BUFFER_ROUTER_A_IDX = 4
DEVICE_BUFFER_GATHER_TABLE = 5
DEVICE_BUFFER_GATHER_READY_ROWS = 6

DEVICE_BUFFER_BF16 = 1
DEVICE_BUFFER_FP16 = 2
DEVICE_BUFFER_FP32 = 3
DEVICE_BUFFER_INT32 = 4


class DeviceBufferRequest(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("kind", ctypes.c_int32),
        ("element_type", ctypes.c_int32),
        ("peer_rank", ctypes.c_int32),
        ("source_node_rank", ctypes.c_int32),
        ("source_gpu_index", ctypes.c_int32),
        ("dimension_count", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
        ("dimensions", ctypes.c_uint64 * 2),
    ]


DeviceBufferAllocatorCallback = ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.POINTER(DeviceBufferRequest),
)


class TorchDeviceBuffers:
    """Own proxy-visible CUDA tensors and service C++ allocation requests."""

    _KIND_NAMES = {
        DEVICE_BUFFER_RDMA_SEND: "rdma_send",
        DEVICE_BUFFER_RDMA_RECEIVE: "rdma_receive",
        DEVICE_BUFFER_NVLINK_RECEIVE: "nvlink_receive",
        DEVICE_BUFFER_ROUTER_A_IDX: "router_a_idx",
        DEVICE_BUFFER_GATHER_TABLE: "gather_table",
        DEVICE_BUFFER_GATHER_READY_ROWS: "gather_ready_rows",
    }

    def __init__(self, torch_module, cuda_device_id: int) -> None:
        self.torch = torch_module
        self.cuda_device_id = cuda_device_id
        self.tensors: dict[tuple[int, int, int, int], object] = {}
        self.allocation_error: BaseException | None = None
        # ctypes callbacks must themselves remain strongly referenced for as
        # long as C++ can invoke them.
        self.callback = DeviceBufferAllocatorCallback(self._allocate)

    def _allocate(
        self,
        _context: int | None,
        request_pointer: ctypes.POINTER(DeviceBufferRequest),
    ) -> int:
        try:
            if not request_pointer:
                raise RuntimeError("C++ requested a device buffer without a descriptor")
            request = request_pointer.contents
            if request.struct_size != ctypes.sizeof(DeviceBufferRequest):
                raise RuntimeError(
                    "device-buffer request structure size mismatch: "
                    f"C++={request.struct_size}, Python={ctypes.sizeof(DeviceBufferRequest)}"
                )
            if request.version != 1:
                raise RuntimeError(
                    f"unsupported device-buffer request version {request.version}"
                )
            if request.dimension_count < 1 or request.dimension_count > 2:
                raise RuntimeError(
                    f"invalid device-buffer dimension count {request.dimension_count}"
                )

            dtype_by_element_type = {
                DEVICE_BUFFER_BF16: self.torch.bfloat16,
                DEVICE_BUFFER_FP16: self.torch.float16,
                DEVICE_BUFFER_FP32: self.torch.float32,
                DEVICE_BUFFER_INT32: self.torch.int32,
            }
            try:
                dtype = dtype_by_element_type[request.element_type]
                kind_name = self._KIND_NAMES[request.kind]
            except KeyError as error:
                raise RuntimeError(
                    "C++ requested an unknown device-buffer kind or element type: "
                    f"kind={request.kind}, element_type={request.element_type}"
                ) from error

            shape = tuple(
                int(request.dimensions[index])
                for index in range(request.dimension_count)
            )
            tensor = self.torch.empty(
                shape,
                dtype=dtype,
                device=f"cuda:{self.cuda_device_id}",
            )
            tensor_bytes = tensor.numel() * tensor.element_size()
            if tensor_bytes != request.bytes:
                raise RuntimeError(
                    f"{kind_name} tensor shape {shape} occupies {tensor_bytes} bytes, "
                    f"but C++ requested {request.bytes}"
                )

            # The launch scripts disable C++ test-pattern filling. Give the
            # shared RDMA send tensor finite floating-point payloads now so it
            # is also suitable as the future communication/GEMM input source.
            if request.kind == DEVICE_BUFFER_RDMA_SEND:
                tensor.normal_(mean=0.0, std=0.02)

            key = (
                int(request.kind),
                int(request.peer_rank),
                int(request.source_node_rank),
                int(request.source_gpu_index),
            )
            if key in self.tensors:
                raise RuntimeError(f"duplicate C++ device-buffer request for key {key}")
            self.tensors[key] = tensor
            return int(tensor.data_ptr())
        except BaseException as error:
            self.allocation_error = error
            return 0

    def finish_initialization(self) -> None:
        if self.allocation_error is not None:
            raise RuntimeError(
                f"Python device-buffer allocation failed: {self.allocation_error}"
            ) from self.allocation_error
        # In particular, finish the random send-buffer initialization before
        # the first C++ iteration is allowed to post RDMA work.
        self.torch.cuda.synchronize(self.cuda_device_id)
        total_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.tensors.values()
        )
        counts: dict[str, int] = {}
        for kind, _, _, _ in self.tensors:
            name = self._KIND_NAMES[kind]
            counts[name] = counts.get(name, 0) + 1
        print(
            "Python owns proxy CUDA buffers: "
            f"count={len(self.tensors)} bytes={total_bytes} by_kind={counts}",
            flush=True,
        )

    def tensors_of_kind(self, kind: int) -> list[object]:
        """Return tensors in the same allocation/source order used by C++."""
        return [
            tensor
            for (tensor_kind, _, _, _), tensor in self.tensors.items()
            if tensor_kind == kind
        ]

    def only_tensor_of_kind(self, kind: int):
        tensors = self.tensors_of_kind(kind)
        if len(tensors) != 1:
            raise RuntimeError(
                f"expected one {self._KIND_NAMES[kind]} tensor, got {len(tensors)}"
            )
        return tensors[0]


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
    library.rdma_proxy_create.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.rdma_proxy_create.restype = ctypes.c_void_p
    library.rdma_proxy_set_device_buffer_allocator.argtypes = [
        ctypes.c_void_p,
        DeviceBufferAllocatorCallback,
        ctypes.c_void_p,
    ]
    library.rdma_proxy_set_device_buffer_allocator.restype = ctypes.c_int
    library.rdma_proxy_initialize.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_initialize.restype = ctypes.c_int
    library.rdma_proxy_run.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_run.restype = ctypes.c_int
    library.rdma_proxy_get_num_iterations.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rdma_proxy_get_num_iterations.restype = ctypes.c_int
    library.rdma_proxy_run_iteration.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    library.rdma_proxy_run_iteration.restype = ctypes.c_int
    library.rdma_proxy_finish.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_finish.restype = ctypes.c_int
    library.rdma_proxy_shutdown.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_shutdown.restype = ctypes.c_int
    library.rdma_proxy_destroy.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_destroy.restype = None
    library.rdma_proxy_last_error.argtypes = []
    library.rdma_proxy_last_error.restype = ctypes.c_char_p

    abi_version = library.rdma_proxy_abi_version()
    if abi_version != EXPECTED_ABI_VERSION:
        raise RuntimeError(
            f"RDMA proxy ABI mismatch: worker expects {EXPECTED_ABI_VERSION}, "
            f"library provides {abi_version}"
        )
    return library


def encode_proxy_argv(proxy_argv: list[str]) -> ctypes.Array:
    encoded_argv = [os.fsencode(argument) for argument in proxy_argv]
    return (ctypes.c_char_p * len(encoded_argv))(*encoded_argv)


def proxy_last_error(library: ctypes.CDLL) -> str:
    error_pointer = library.rdma_proxy_last_error()
    return (
        error_pointer.decode("utf-8", errors="replace")
        if error_pointer
        else "unknown proxy error"
    )


def require_proxy_success(library: ctypes.CDLL, status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed: {proxy_last_error(library)}")


def import_torch_for_proxy(cuda_device_id: int):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("the Python-owned buffer path requires PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("the Python-owned buffer path requires an available CUDA device")
    torch.cuda.set_device(cuda_device_id)
    return torch


def install_python_device_buffers(
    library: ctypes.CDLL,
    handle: int,
    torch_module,
    cuda_device_id: int,
) -> TorchDeviceBuffers:
    buffers = TorchDeviceBuffers(torch_module, cuda_device_id)
    require_proxy_success(
        library,
        library.rdma_proxy_set_device_buffer_allocator(
            handle, buffers.callback, None
        ),
        "installing Python device-buffer allocator",
    )
    return buffers


def run_proxy(
    library: ctypes.CDLL,
    proxy_argv: list[str],
    cuda_device_id: int,
) -> int:
    c_argv = encode_proxy_argv(proxy_argv)
    torch = import_torch_for_proxy(cuda_device_id)
    handle = library.rdma_proxy_create(len(proxy_argv), c_argv)
    if not handle:
        raise RuntimeError(f"proxy creation failed: {proxy_last_error(library)}")

    initialized = False
    device_buffers: TorchDeviceBuffers | None = None
    try:
        device_buffers = install_python_device_buffers(
            library, handle, torch, cuda_device_id
        )
        initialize_status = library.rdma_proxy_initialize(handle)
        if initialize_status != 0 and device_buffers.allocation_error is not None:
            raise RuntimeError(
                "proxy initialization failed after Python rejected a buffer request: "
                f"{device_buffers.allocation_error}; C++: {proxy_last_error(library)}"
            ) from device_buffers.allocation_error
        require_proxy_success(library, initialize_status, "proxy initialization")
        initialized = True
        device_buffers.finish_initialization()
        require_proxy_success(library, library.rdma_proxy_run(handle), "proxy run")
        return 0
    finally:
        if initialized:
            shutdown_status = library.rdma_proxy_shutdown(handle)
            if shutdown_status != 0:
                print(
                    f"proxy shutdown failed: {proxy_last_error(library)}",
                    file=sys.stderr,
                    flush=True,
                )
        library.rdma_proxy_destroy(handle)


def run_proxy_with_concurrent_kernel(
    library: ctypes.CDLL,
    proxy_argv: list[str],
    cuda_device_id: int,
) -> int:
    """Let Python launch one independent CUDA kernel with each proxy iteration."""

    c_argv = encode_proxy_argv(proxy_argv)
    torch = import_torch_for_proxy(cuda_device_id)
    handle = library.rdma_proxy_create(len(proxy_argv), c_argv)
    if not handle:
        raise RuntimeError(f"proxy creation failed: {proxy_last_error(library)}")

    initialized = False
    coordinator: threading.Thread | None = None
    commands: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    device_buffers: TorchDeviceBuffers | None = None
    try:
        device_buffers = install_python_device_buffers(
            library, handle, torch, cuda_device_id
        )
        initialize_status = library.rdma_proxy_initialize(handle)
        if initialize_status != 0 and device_buffers.allocation_error is not None:
            raise RuntimeError(
                "proxy initialization failed after Python rejected a buffer request: "
                f"{device_buffers.allocation_error}; C++: {proxy_last_error(library)}"
            ) from device_buffers.allocation_error
        require_proxy_success(library, initialize_status, "proxy initialization")
        initialized = True
        device_buffers.finish_initialization()

        num_iterations = ctypes.c_uint64()
        require_proxy_success(
            library,
            library.rdma_proxy_get_num_iterations(handle, ctypes.byref(num_iterations)),
            "querying num_iterations",
        )
        if num_iterations.value == 0:
            raise RuntimeError(
                "--concurrent-kernel requires a finite, nonzero --num_iterations"
            )

        kernel_stream = torch.cuda.Stream(device=cuda_device_id)
        kernel_finished = torch.cuda.Event(enable_timing=False, blocking=False)
        with torch.cuda.stream(kernel_stream):
            kernel_tensor = torch.zeros(
                CONCURRENT_KERNEL_ELEMENTS,
                dtype=torch.float32,
                device=f"cuda:{cuda_device_id}",
            )
        kernel_stream.synchronize()

        def proxy_iteration_coordinator() -> None:
            while True:
                command = commands.get()
                if command is None:
                    return
                iteration, start_barrier = command
                try:
                    # The Python main thread waits on the same barrier immediately
                    # before enqueueing its CUDA kernel.
                    start_barrier.wait()
                    status = library.rdma_proxy_run_iteration(handle, iteration)
                    error = None if status == 0 else proxy_last_error(library)
                    results.put((status, error))
                except BaseException as error:
                    results.put((-1, f"proxy coordinator failed: {error}"))
                    return

        coordinator = threading.Thread(
            target=proxy_iteration_coordinator,
            name="rdma-proxy-iteration-coordinator",
        )
        coordinator.start()

        print(
            "concurrent-kernel mode enabled: Python will launch one independent "
            f"torch.add kernel over {CONCURRENT_KERNEL_ELEMENTS} float32 elements "
            "with each proxy iteration",
            flush=True,
        )

        for iteration in range(num_iterations.value):
            start_barrier = threading.Barrier(2)
            commands.put((iteration, start_barrier))
            start_barrier.wait()

            kernel_error: BaseException | None = None
            try:
                # add_ is an asynchronous pointwise CUDA kernel. Recording and
                # waiting on this stream-local event does not synchronize the
                # proxy's independent CUDA streams.
                with torch.cuda.stream(kernel_stream):
                    kernel_tensor.add_(1.0)
                    kernel_finished.record(kernel_stream)
                kernel_finished.synchronize()
            except BaseException as error:
                kernel_error = error

            proxy_status, proxy_error = results.get()
            if kernel_error is not None:
                raise RuntimeError(
                    f"concurrent CUDA kernel failed at iteration {iteration}: {kernel_error}"
                ) from kernel_error
            if proxy_status != 0:
                raise RuntimeError(
                    f"proxy iteration {iteration} failed: {proxy_error}"
                )

        commands.put(None)
        coordinator.join()
        coordinator = None
        require_proxy_success(library, library.rdma_proxy_finish(handle), "proxy finish")
        return 0
    finally:
        if coordinator is not None:
            commands.put(None)
            coordinator.join()
        if initialized:
            shutdown_status = library.rdma_proxy_shutdown(handle)
            if shutdown_status != 0:
                print(
                    f"proxy shutdown failed: {proxy_last_error(library)}",
                    file=sys.stderr,
                    flush=True,
                )
        library.rdma_proxy_destroy(handle)


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
        "--concurrent-kernel",
        action="store_true",
        help="launch an independent PyTorch CUDA kernel with every proxy iteration",
    )
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
    if args.concurrent_kernel:
        return run_proxy_with_concurrent_kernel(
            library, proxy_argv, int(metadata["cuda_device_id"])
        )
    return run_proxy(library, proxy_argv, int(metadata["cuda_device_id"]))


def worker_entrypoint() -> int:
    """Preserve a readable local diagnostic before TorchElastic records it."""
    try:
        return main()
    except Exception as error:
        print(f"rdma_proxy_worker failed: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    try:
        from torch.distributed.elastic.multiprocessing.errors import record
    except ImportError:
        # A direct dry-run can still explain a missing PyTorch installation.
        recorded_worker_entrypoint = worker_entrypoint
    else:
        recorded_worker_entrypoint = record(worker_entrypoint)
    raise SystemExit(recorded_worker_entrypoint())
