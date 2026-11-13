#!/usr/bin/env python3
"""Load one RDMA CPU proxy into each torchrun worker process."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import queue
import re
import shlex
import sys
import threading


EXPECTED_ABI_VERSION = 1

# Keep CLI parsing (including --dry-run) independent of CUDA/QuACK imports.
ORDINARY_ACTIVATIONS = ("silu", "silu-tanh", "relu", "relu_sq", "gelu_tanh_approx", "tanh")
GATED_ACTIVATIONS = (
    "swiglu", "swiglu-tanh", "swiglu_oai", "swiglu_oai-tanh", "reglu", "geglu", "glu",
)


def resolve_gemm_architecture(requested, capability):
    """Use the device's major architecture, including B300's SM103 as SM100."""
    major, minor = capability
    if major not in (9, 10):
        raise ValueError(f"proxy GEMMs require Hopper SM90 or Blackwell SM100, got SM{major}{minor}")
    expected = {"auto": major, "hopper": 9, "blackwell": 10}[requested]
    if expected != major:
        raise ValueError(f"--gemm-arch={requested} disagrees with device SM{major}{minor}")
    return major


def resolve_down_pingpong(requested, device_capacity, wave_barrier):
    if requested is None:
        return device_capacity == 9 and wave_barrier
    if device_capacity == 10 and requested:
        raise ValueError("Blackwell requires --no-down-pingpong (or omit the pingpong option)")
    if device_capacity == 9 and wave_barrier and not requested:
        raise ValueError("Hopper down wave barriers require --down-pingpong")
    return requested


def projection_dimension(output_dimension: int, activation: str | None) -> int:
    """The table schedules projection columns, before gated activation halves them."""
    return output_dimension * (2 if activation in GATED_ACTIVATIONS else 1)

DEVICE_BUFFER_RDMA_SEND = 1
DEVICE_BUFFER_RDMA_RECEIVE = 2
DEVICE_BUFFER_NVLINK_RECEIVE = 3
DEVICE_BUFFER_ROUTER_A_IDX = 4
DEVICE_BUFFER_GATHER_TABLE = 5
DEVICE_BUFFER_GATHER_READY_ROWS = 6
DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS = 7

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
        DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS: "router_route_weights",
    }

    def __init__(
        self,
        torch_module,
        cuda_device_id: int,
        *,
        node_rank: int,
        local_gpu_index: int,
    ) -> None:
        self.torch = torch_module
        self.cuda_device_id = cuda_device_id
        self.node_rank = node_rank
        self.local_gpu_index = local_gpu_index
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
            key = (
                int(request.kind),
                int(request.peer_rank),
                int(request.source_node_rank),
                int(request.source_gpu_index),
            )
            if (
                request.kind == DEVICE_BUFFER_NVLINK_RECEIVE
                and request.source_node_rank != self.node_rank
                and request.source_gpu_index == self.local_gpu_index
            ):
                rdma_key = (
                    DEVICE_BUFFER_RDMA_RECEIVE,
                    int(request.source_node_rank),
                    -1,
                    -1,
                )
                try:
                    tensor = self.tensors[rdma_key]
                except KeyError as error:
                    raise RuntimeError(
                        "direct-RDMA input alias was requested before its peer "
                        f"receive tensor: source_node={request.source_node_rank}"
                    ) from error
                tensor_bytes = tensor.numel() * tensor.element_size()
                if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor_bytes != request.bytes:
                    raise RuntimeError(
                        "direct-RDMA input alias descriptor does not match its RDMA "
                        f"receive tensor: shape={shape}, bytes={request.bytes}"
                    )
                if key in self.tensors:
                    raise RuntimeError(f"duplicate C++ device-buffer request for key {key}")
                self.tensors[key] = tensor
                return int(tensor.data_ptr())

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
        unique_tensors = {
            int(tensor.data_ptr()): tensor for tensor in self.tensors.values()
        }
        total_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in unique_tensors.values()
        )
        counts: dict[str, int] = {}
        for kind, _, _, _ in self.tensors:
            name = self._KIND_NAMES[kind]
            counts[name] = counts.get(name, 0) + 1
        print(
            "Python owns proxy CUDA buffers: "
            f"logical_count={len(self.tensors)} "
            f"physical_count={len(unique_tensors)} bytes={total_bytes} "
            f"by_kind={counts}",
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


class GroupedGemmRunner:
    """Launch QuACK's readiness-gated multi-buffer GroupedGEMM."""

    def __init__(
        self,
        torch_module,
        device_buffers: TorchDeviceBuffers,
        *,
        cuda_device_id: int,
        world_size: int,
        num_experts: int,
        output_dimension: int,
        activation: str | None,
        tile_m: int,
        tile_n: int,
        tile_k: int | None,
        cluster_m: int,
        max_swizzle_size: int,
        quack_root: Path,
    ) -> None:
        self.torch = torch_module
        self.cuda_device_id = cuda_device_id
        if not device_buffers.tensors_of_kind(DEVICE_BUFFER_GATHER_READY_ROWS):
            raise RuntimeError(
                "--concurrent-kernel requires the notification-driven router scheduler; "
                "set --nvlink_forward_completion_notifications_enabled=true"
            )
        capability = torch_module.cuda.get_device_capability(cuda_device_id)
        self.device_capacity = resolve_gemm_architecture("auto", capability)
        if num_experts <= 0 or num_experts % world_size != 0:
            raise RuntimeError(
                f"--num-experts={num_experts} must be divisible by WORLD_SIZE={world_size}"
            )
        quack_root = quack_root.resolve()
        if not (quack_root / "quack" / "gemm.py").is_file():
            raise RuntimeError(f"QuACK repository not found at {quack_root}")
        quack_root_text = str(quack_root)
        if quack_root_text not in sys.path:
            sys.path.insert(0, quack_root_text)
        try:
            from quack.gemm import gemm as quack_gemm
        except Exception as error:
            raise RuntimeError(f"failed to import QuACK from {quack_root}: {error}") from error
        self.quack_gemm = quack_gemm
        self.activation = activation
        self.gated = activation in GATED_ACTIVATIONS
        self.gemm_dimension = projection_dimension(output_dimension, activation)
        if activation is not None:
            from quack.gemm_config import GemmConfig
            from quack.gemm_interface import gemm_act

            self.quack_gemm_act = gemm_act
            self.activation_config = GemmConfig(
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                cluster_m=cluster_m, cluster_n=1, cluster_k=1,
                pingpong=False, is_dynamic_persistent=False,
                max_swizzle_size=max_swizzle_size,
                device_capacity=self.device_capacity, use_tma_gather=False,
            )

        a_idx_entries = [
            (key, tensor)
            for key, tensor in device_buffers.tensors.items()
            if key[0] == DEVICE_BUFFER_ROUTER_A_IDX
        ]
        if not a_idx_entries:
            raise RuntimeError(
                "C++ exposed no active A_idx tensors; forwarded-input gather mode is not initialized"
            )
        x_buffers = []
        a_idx = []
        source_pairs = []
        for (_, _, source_node, source_gpu), index_tensor in a_idx_entries:
            x_key = (
                DEVICE_BUFFER_NVLINK_RECEIVE,
                -1,
                source_node,
                source_gpu,
            )
            try:
                x_tensor = device_buffers.tensors[x_key]
            except KeyError as error:
                raise RuntimeError(
                    "missing NVLink tensor matching active A_idx source "
                    f"node={source_node}, gpu={source_gpu}"
                ) from error
            x_buffers.append(x_tensor)
            a_idx.append(index_tensor)
            source_pairs.append((source_node, source_gpu))

        first_x = x_buffers[0]
        if first_x.dtype not in (torch_module.bfloat16, torch_module.float16):
            raise RuntimeError(
                f"QuACK runner supports bf16/fp16 inputs, got {first_x.dtype}"
            )
        if any(tensor.shape != first_x.shape or tensor.dtype != first_x.dtype
               for tensor in x_buffers):
            raise RuntimeError("active NVLink input tensors do not share shape and dtype")
        first_a_idx = a_idx[0]
        if any(tensor.shape != first_a_idx.shape or tensor.dtype != torch_module.int32
               for tensor in a_idx):
            raise RuntimeError("active A_idx tensors do not share int32 shape")

        work_table = device_buffers.only_tensor_of_kind(
            DEVICE_BUFFER_GATHER_TABLE
        )
        ready_rows = device_buffers.only_tensor_of_kind(
            DEVICE_BUFFER_GATHER_READY_ROWS
        )
        expected_width = 2 + 2 * len(x_buffers)
        if work_table.ndim != 2 or work_table.shape[1] != expected_width:
            raise RuntimeError(
                f"gather table shape {tuple(work_table.shape)} does not match "
                f"{len(x_buffers)} active inputs (expected width {expected_width})"
            )
        if work_table.shape[0] == 0:
            raise RuntimeError("QuACK gather table has zero rows")

        clusters_n = math.ceil(self.gemm_dimension / tile_n)
        group_size = min(max_swizzle_size, clusters_n)
        if group_size == 0 or clusters_n % group_size != 0:
            raise RuntimeError(
                "GroupedGEMM N clusters must be divisible by the effective swizzle group"
            )

        local_experts = num_experts // world_size
        hidden_dimension = int(first_x.shape[1])
        # Gated epilogues use QuACK's concat-B layout: contiguous [gate; up]
        # rows in storage, exposed logically as [E, K, 2H].
        weight_shape = (
            (local_experts, self.gemm_dimension, hidden_dimension)
            if self.gated else (local_experts, hidden_dimension, self.gemm_dimension)
        )
        self.weights = torch_module.randn(
            weight_shape,
            dtype=first_x.dtype,
            device=first_x.device,
        )
        if self.gated:
            self.weights = self.weights.transpose(1, 2)
        self.weights.mul_(1.0 / math.sqrt(hidden_dimension))
        self.output = torch_module.empty(
            (len(x_buffers) * int(first_a_idx.numel()), output_dimension),
            dtype=first_x.dtype,
            device=first_x.device,
        )
        self.x_buffers = tuple(x_buffers)
        self.source_pairs = tuple(source_pairs)
        self.a_idx = tuple(a_idx)
        weight_entries = device_buffers.tensors_of_kind(DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS)
        self.route_weights = None
        if weight_entries:
            self.route_weights = tuple(device_buffers.tensors[
                (DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS, -1, node, gpu)
            ] for node, gpu in source_pairs)
            if any(weight.shape != index.shape or weight.dtype != torch_module.float32
                   for weight, index in zip(self.route_weights, self.a_idx)):
                raise RuntimeError("routing weights must be FP32 and aligned with each source A_idx")
        # Immutable lookup storage; only the routed-row vector changes per iteration.
        self.route_weights_stacked = (torch_module.stack(self.route_weights)
                                      if self.route_weights is not None else None)
        torch_module.cuda.synchronize(cuda_device_id)
        self.work_table = work_table
        self.ready_rows = ready_rows
        self.transposed_weights = self.weights.transpose(1, 2)
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.cluster_m = cluster_m
        self.max_swizzle_size = max_swizzle_size
        from quack.gemm_config import cta_tile_shape_m
        self.cta_tile_m = cta_tile_shape_m(tile_m, cluster_m, self.device_capacity)
        print(
            "GroupedGEMM configured: "
            f"device_capacity={self.device_capacity} mma_tile_m={tile_m} "
            f"cta_tile_m={self.cta_tile_m} cluster_rows={self.cta_tile_m * cluster_m} "
            f"activation={activation or 'none'} projection_N={self.gemm_dimension} "
            f"active_sources={source_pairs} X={tuple(first_x.shape)} "
            f"A_idx={tuple(first_a_idx.shape)} W={tuple(self.weights.shape)} "
            f"output={tuple(self.output.shape)} table={tuple(work_table.shape)}",
            flush=True,
        )

    def launch(self) -> None:
        if self.activation is not None:
            self.quack_gemm_act(
                self.x_buffers,
                self.weights,
                activation=self.activation,
                postact_out=self.output,
                A_idx=self.a_idx,
                gather_work_table=self.work_table,
                gather_work_table_ready=self.ready_rows,
                multi_buffer_gather=True,
                store_preact=False,
                dynamic_scheduler=False,
                tuned=False,
                config=self.activation_config,
                concat_layout=("B",) if self.gated else None,
            )
            return
        self.quack_gemm(
            self.x_buffers,
            self.transposed_weights,
            self.output,
            C=None,
            tile_count_semaphore=None,
            tile_M=self.tile_m,
            tile_N=self.tile_n,
            tile_K=self.tile_k,
            cluster_M=self.cluster_m,
            cluster_N=1,
            cluster_K=1,
            pingpong=False,
            persistent=True,
            is_dynamic_persistent=False,
            max_swizzle_size=self.max_swizzle_size,
            cu_seqlens_m=None,
            A_idx=self.a_idx,
            gather_work_table=self.work_table,
            gather_work_table_ready=self.ready_rows,
            multi_buffer_gather=True,
            use_tma_gather=False,
        )


def add_down_arguments(parser):
    parser.add_argument("--cpu-scatter-preparation", action="store_true",
                        help="prepare down scatter addresses on the CPU; requires local GPU combine")
    parser.add_argument("--cpu-scatter-publication", choices=("deferred", "threaded", "per-batch"), default=None,
                        help="CPU metadata policy: threaded packs on a dedicated CPU thread with one final upload; "
                             "deferred (default) packs and uploads after all up gather entries; "
                             "per-batch uploads each M batch before gather readiness")
    parser.add_argument("--inter-node-combine", action="store_true",
                        help="combine node sums into original token order after local combine and any RDMA returns")
    parser.add_argument("--inter-node-combine-ctas", type=int, default=None,
                        help="positive CTA count for inter-node combine; defaults to GPU SM count")
    parser.add_argument("--check-inter-node-combine", action=argparse.BooleanOptionalAction, default=False,
                        help="check final token-order output against an FP32 reference")
    parser.add_argument("--rdma-return-token-results", action="store_true",
                        help="return completed intra-node combine column groups over existing RDMA QPs")
    parser.add_argument("--check-rdma-return-token-results", action=argparse.BooleanOptionalAction, default=False,
                        help="compare returned payload SHA256 with the remote source after each iteration")
    parser.add_argument("--rdma-return-chunk-bytes", type=int, default=1048576)
    parser.add_argument("--intra-node-combine", action="store_true",
                        help="combine returned and retained down outputs while down GEMM runs")
    parser.add_argument("--intra-node-combine-n-tile-major-output", action="store_true",
                        help="store each source output as [N tiles, input rows, tile N]")
    parser.add_argument("--intra-node-combine-publish-ready", action="store_true",
                        help="publish completed intra-node output column-group prefix in HBM")
    parser.add_argument("--check-intra-node-combine", action=argparse.BooleanOptionalAction, default=False,
                        help="compare intra-node output with an FP32 reference each iteration")
    parser.add_argument("--intra-node-combine-ctas", type=int, default=4)
    parser.add_argument("--intra-node-combine-elements-per-thread", type=int,
                        choices=(8, 16, 32, 64, 128), default=32)
    parser.add_argument("--intra-node-combine-threads-per-cta", type=int,
                        choices=(256, 512), default=512)
    parser.add_argument("--intra-node-combine-prefetch-peers", type=int,
                        choices=(1, 2, 4), default=4)
    parser.add_argument("--return-token-results", action="store_true",
                        help="return local-combine outputs to their same-node NVLink senders")
    parser.add_argument("--return-token-results-per-group", action="store_true",
                        help="return each ready column group while down GEMM continues")
    parser.add_argument("--check-return-token-results", action=argparse.BooleanOptionalAction, default=False,
                        help="compare received valid bytes with producing GPU outputs each iteration")
    parser.add_argument("--return-token-timeout", type=float, default=120.0,
                        help="same-node return setup/completion and CPU metadata submission timeout in seconds")
    parser.add_argument("--down-projection2", action="store_true")
    parser.add_argument(
        "--down-sm-count", type=int, default=None,
        help="cap down GEMM persistent CTAs; positive multiple of --down-cluster-m. "
             "Defaults to hardware capacity; does not reserve physical SMs",
    )
    parser.add_argument("--trace-down-ready", action="store_true",
                        help="poll down-output readiness in the C++ notification receiver")
    parser.add_argument("--local-gpu-combine", action="store_true",
                        help="sum down outputs into separate per-source token buffers, without routing weights")
    for name, default in (("tile-m", 128), ("tile-n", 128), ("cluster-m", 1),
                          ("max-swizzle-size", 1), ("scatter-output-buffers", 2)):
        parser.add_argument(f"--down-{name}", type=int, default=default)
    parser.add_argument("--down-pingpong", action=argparse.BooleanOptionalAction, default=None,
                        help="default: enabled for Hopper waves, disabled for Blackwell")
    for name in ("bulk-smem-padding", "bulk-smem-cols", "bulk-smem-buffers", "bulk-store-warps"):
        parser.add_argument(f"--down-{name}", type=int, default=None,
                            help="explicit SM100 bulk-store compilation setting; omitted keeps QuACK defaults")
    for name in ("scatter-table", "n-tile-major-output",
                 "multi-buffer-scatter", "gather-table", "gather-table-expert-waves",
                 "gather-table-wave-barrier"):
        parser.add_argument(f"--down-{name}", action="store_true")
    parser.add_argument("--down-epilogue-store",
                        choices=("tma", "bulk_rows", "bulk_rows_reduce"), default="tma")
    parser.add_argument(
        "--down-gather-table-publish-ready", action="store_true",
        help="publish completed down-output column groups to a GPU flag; "
             "requires down expert waves and the wave barrier",
    )


def validate_down_arguments(args):
    if args.cpu_scatter_preparation and not args.local_gpu_combine:
        raise ValueError("--cpu-scatter-preparation requires --local-gpu-combine")
    if args.cpu_scatter_publication is not None and not args.cpu_scatter_preparation:
        raise ValueError("--cpu-scatter-publication requires --cpu-scatter-preparation")
    if args.cpu_scatter_preparation and (
        not math.isfinite(args.return_token_timeout) or args.return_token_timeout <= 0
        or args.return_token_timeout > ((1 << 63) - 1) // 1000
    ):
        raise ValueError("--return-token-timeout must be finite, positive, and fit signed 64-bit milliseconds")
    if args.inter_node_combine_ctas is not None:
        if not args.inter_node_combine or args.inter_node_combine_ctas <= 0:
            raise ValueError("--inter-node-combine-ctas requires --inter-node-combine and a positive count")
    if args.check_inter_node_combine and not args.inter_node_combine:
        raise ValueError("--check-inter-node-combine requires --inter-node-combine")
    if args.inter_node_combine and not (args.intra_node_combine
                                        and args.intra_node_combine_n_tile_major_output):
        raise ValueError("--inter-node-combine requires --intra-node-combine "
                         "and --intra-node-combine-n-tile-major-output")
    if args.check_rdma_return_token_results and not args.rdma_return_token_results:
        raise ValueError("--check-rdma-return-token-results requires --rdma-return-token-results")
    if args.rdma_return_token_results:
        if not (args.intra_node_combine and args.intra_node_combine_n_tile_major_output
                and args.intra_node_combine_publish_ready):
            raise ValueError("--rdma-return-token-results requires --intra-node-combine, "
                             "--intra-node-combine-n-tile-major-output and --intra-node-combine-publish-ready")
        if not 0 < args.rdma_return_chunk_bytes <= 2**32 - 1:
            raise ValueError("--rdma-return-chunk-bytes must be in [1, 2**32-1]")
    if args.intra_node_combine_publish_ready and not args.intra_node_combine:
        raise ValueError("--intra-node-combine-publish-ready requires --intra-node-combine")
    if args.intra_node_combine_n_tile_major_output and not args.intra_node_combine:
        raise ValueError("--intra-node-combine-n-tile-major-output requires --intra-node-combine")
    if args.check_intra_node_combine and not args.intra_node_combine:
        raise ValueError("--check-intra-node-combine requires --intra-node-combine")
    if args.intra_node_combine:
        if not (args.local_gpu_combine and args.return_token_results_per_group):
            raise ValueError("--intra-node-combine requires --local-gpu-combine and "
                             "--return-token-results-per-group with their prerequisites")
        if args.intra_node_combine_ctas <= 0:
            raise ValueError("--intra-node-combine-ctas must be positive")
        if args.down_sm_count is None:
            args.down_sm_count = 40

    if args.return_token_results_per_group and (
        not args.return_token_results or not args.down_n_tile_major_output
    ):
        raise ValueError("--return-token-results-per-group requires --return-token-results "
                         "and --down-n-tile-major-output")
    if args.check_return_token_results and not args.return_token_results:
        raise ValueError("--check-return-token-results requires --return-token-results")
    if args.return_token_results:
        if not args.local_gpu_combine or not args.down_gather_table_publish_ready:
            raise ValueError("--return-token-results requires --local-gpu-combine "
                             "and --down-gather-table-publish-ready")
        if not math.isfinite(args.return_token_timeout) or args.return_token_timeout <= 0:
            raise ValueError("--return-token-timeout must be finite and positive")
    if args.trace_down_ready and (
        not args.down_projection2 or not args.down_gather_table_publish_ready
    ):
        raise ValueError("--trace-down-ready requires --down-projection2 and "
                         "--down-gather-table-publish-ready")
    if args.local_gpu_combine and not args.down_projection2:
        raise ValueError("--local-gpu-combine requires --down-projection2")
    if not args.down_projection2:
        return
    if args.local_gpu_combine:
        # Combine owns the destination count and requires additive scatter.
        args.down_scatter_table = True
        args.down_multi_buffer_scatter = True
        args.down_epilogue_store = "bulk_rows_reduce"
    if not args.concurrent_kernel or args.grouped_gemm_activation in (None, "none"):
        raise ValueError("--down-projection2 requires --concurrent-kernel and a fused activation")
    for name in ("tile_m", "tile_n", "cluster_m", "max_swizzle_size"):
        if getattr(args, "down_" + name) <= 0:
            raise ValueError(f"--down-{name.replace('_', '-')} must be positive")
    if args.down_sm_count is not None and (
        args.down_sm_count <= 0 or args.down_sm_count % args.down_cluster_m
    ):
        raise ValueError("--down-sm-count must be positive and divisible by --down-cluster-m")
    if args.down_scatter_table and args.down_epilogue_store == "tma":
        raise ValueError("down scatter requires a bulk-row epilogue")
    if (args.down_multi_buffer_scatter or args.down_n_tile_major_output) and not args.down_scatter_table:
        raise ValueError("down multi-buffer/tiled output requires --down-scatter-table")
    if (not args.local_gpu_combine and args.down_multi_buffer_scatter
            and args.down_scatter_output_buffers < 2):
        raise ValueError("--down-scatter-output-buffers must be at least 2")
    if args.down_gather_table_expert_waves and not args.down_gather_table:
        raise ValueError("down expert waves require --down-gather-table")
    if args.down_gather_table_publish_ready and (
        not args.down_gather_table_expert_waves or not args.down_gather_table_wave_barrier
    ):
        raise ValueError("--down-gather-table-publish-ready requires "
                         "--down-gather-table-expert-waves and --down-gather-table-wave-barrier")
    if args.down_gather_table_wave_barrier and (
        not args.down_gather_table_expert_waves
        or args.down_epilogue_store == "tma"
    ):
        raise ValueError("down wave barrier requires expert waves and a bulk-row epilogue")
    requested_arch = getattr(args, "gemm_arch", "auto")
    if requested_arch != "auto":
        resolve_down_pingpong(args.down_pingpong, 9 if requested_arch == "hopper" else 10,
                              args.down_gather_table_wave_barrier)


def local_route_positions(torch, work_table, num_sources, routes, n_groups, trace=None):
    """Recover each packed row's exact (source, A_idx list position) on device.

    Multi-buffer QuACK places a table row at sum(start_j), concatenating its
    source ranges. N groups repeat those ranges. Publication order is arbitrary;
    neither expert order nor source order alone describes the packed output.
    Called on the up/down stream after up has consumed the complete table.
    """
    def mark(message):
        if trace is not None:
            trace("mapping: " + message)

    mark("table conversion begin")
    ranges = work_table[::n_groups, 2:].long().reshape(-1, num_sources, 2)
    starts, ends = ranges[:, :, 0], ranges[:, :, 1]
    mark("table conversion submitted; lengths begin")
    lengths = ends - starts
    mark("lengths submitted; sum begin")
    start_sums = starts.sum(dim=1, keepdim=True)
    mark("sum submitted; cumsum begin")
    cumulative_lengths = lengths.cumsum(dim=1)
    mark("cumsum submitted; output ends begin")
    output_ends = start_sums + cumulative_lengths
    del start_sums, cumulative_lengths
    # Empty source segments must never win a search, even at batch boundaries.
    mark("output ends submitted; keys begin")
    keys = torch.where(lengths > 0, output_ends, routes + 1).flatten()
    mark("keys submitted; argsort begin")
    order = keys.argsort()
    mark("argsort submitted; arange begin")
    rows = torch.arange(routes, device=work_table.device, dtype=torch.int64)
    mark("arange submitted; sorted keys gather begin")
    sorted_keys = keys[order]
    mark("sorted keys submitted; searchsorted begin")
    positions = torch.searchsorted(sorted_keys, rows, right=True)
    mark("searchsorted submitted; segment gather begin")
    segments = order[positions]
    del sorted_keys, positions
    mark("segment gather submitted; source and index arithmetic begin")
    source = segments % num_sources
    output_starts = (output_ends - lengths).flatten()
    index = starts.flatten()[segments] + rows - output_starts[segments]
    mark("index arithmetic submitted")
    return source, index


def local_combine_mapping(torch, work_table, a_idx, routes, n_groups, trace=None):
    """Map routed rows to input rows; list positions also distinguish expert weights."""
    source, index = local_route_positions(torch, work_table, len(a_idx), routes, n_groups, trace)
    indices = torch.stack(a_idx)
    token_indices = indices[source, index].long()
    result = torch.stack((source, token_indices), dim=1)
    return result


def compact_combine_mapping(torch, mapping, num_sources, capacity):
    """Compact distinct used rows per source in ascending input-row order.

    Returns route destinations, device-side valid counts, and a capacity-sized
    output-to-input map (-1 beyond each valid prefix). Duplicate expert routes
    share a destination. Only meaningful routes participate, never A_idx padding.
    All operations stay on the calling CUDA stream, with fixed-size allocations.
    """
    flat = mapping[:, 0] * capacity + mapping[:, 1]
    hits = torch.zeros(num_sources * capacity, dtype=torch.int64, device=mapping.device)
    hits.scatter_add_(0, flat, torch.ones_like(flat))
    used = hits.reshape(num_sources, capacity) > 0
    ranks = used.long().cumsum(dim=1) - 1
    compact = torch.stack((mapping[:, 0], ranks.flatten()[flat]), dim=1)
    rows = torch.arange(capacity, device=mapping.device, dtype=torch.int64)
    output_to_input = torch.where(used, rows, capacity).sort(dim=1).values
    output_to_input = torch.where(output_to_input < capacity, output_to_input, -1)
    return compact, used.sum(dim=1), output_to_input


def make_cpu_scatter_metadata_storage(torch, scatter_row_addresses, route_weights=None):
    """One owned byte allocation, with contiguous typed views for one native H2D.

    Preserve QuACK's tensor shapes and align FP32 route weights to 16 bytes.
    The native deferred publisher validates this exact layout before borrowing it.
    """
    if (scatter_row_addresses.dtype != torch.int64 or not scatter_row_addresses.is_contiguous()
            or scatter_row_addresses.ndim not in (1, 2)
            or (scatter_row_addresses.ndim == 2 and scatter_row_addresses.shape[1] != 2)
            or scatter_row_addresses.shape[0] <= 0):
        raise ValueError("CPU scatter storage requires contiguous int64 routes or routes-by-2 addresses")
    scatter_bytes = scatter_row_addresses.numel() * scatter_row_addresses.element_size()
    weight_offset = (scatter_bytes + 15) // 16 * 16
    if route_weights is not None and (
        route_weights.dtype != torch.float32 or route_weights.shape != (scatter_row_addresses.shape[0],)
        or route_weights.device != scatter_row_addresses.device or not route_weights.is_contiguous()
    ):
        raise ValueError("CPU scatter weights require contiguous FP32 storage matching address rows and device")
    total_bytes = (scatter_bytes if route_weights is None else
                   weight_offset + route_weights.numel() * route_weights.element_size())
    storage = torch.zeros(total_bytes, dtype=torch.uint8, device=scatter_row_addresses.device)
    addresses = storage[:scatter_bytes].view(torch.int64).reshape(scatter_row_addresses.shape)
    addresses.copy_(scatter_row_addresses)
    weights = None
    if route_weights is not None:
        weights = storage[weight_offset:].view(torch.float32)
        weights.copy_(route_weights)
    return storage, addresses, weights


class DownProjectionRunner:
    """Compose QuACK's existing plain grouped-GEMM runner with the up output."""

    def __init__(self, torch, up, counts, options, trace_configure=None):
        # GroupedGemmRunner has already inserted the selected QuACK checkout.
        from run import hopper_pregather_gemm as pregather

        routes = sum(counts)
        if len(counts) != up.weights.shape[0] or routes <= 0 or routes > up.output.shape[0]:
            raise ValueError("expert counts do not describe the up-projection output")
        # Reuse QuACK's defaults for internal options outside the exposed subset.
        parser = argparse.ArgumentParser()
        pregather.add_kernel_arguments(parser)
        args = parser.parse_args([])
        for name, value in vars(options).items():
            if name.startswith("down_") and name != "down_projection2":
                setattr(args, name[5:], value)
        args.device_capacity = up.device_capacity
        args.pingpong = resolve_down_pingpong(
            args.pingpong, up.device_capacity, args.gather_table_wave_barrier)
        hidden = int(up.output.shape[1])
        output_dim = int(up.x_buffers[0].shape[1])
        vars(args).update(tokens=routes, routes=routes, experts=len(counts),
                          hidden=hidden, output_dim=output_dim, activation=None,
                          down_projection=False, iterations=1, timing_samples=1, warmup=0)
        self.combine = options.local_gpu_combine
        self.cpu_scatter = options.cpu_scatter_preparation
        self.cpu_scatter_publication = options.cpu_scatter_publication or "deferred"
        self.cpu_scatter_metadata = None
        self._cpu_scatter_wait = None
        self.cpu_scatter_timeout_ms = (max(1, math.ceil(options.return_token_timeout * 1000))
                                       if self.cpu_scatter else 0)
        self.cpu_scatter_counts = None
        self.up = up
        self.routes = routes
        self.route_weights = None
        if getattr(options, "weighted_contributions", False):
            if up.route_weights is None:
                raise RuntimeError("weighted contributions require native per-source routing weights")
            # Own the packed runtime argument through every launch and native shutdown.
            self.route_weights = torch.zeros(routes, dtype=torch.float32, device=up.output.device)
        elif getattr(up, "route_weights", None) is not None:
            raise RuntimeError("native routing weights disagree with --no-weighted-contributions")
        clusters_n = math.ceil(up.gemm_dimension / up.tile_n)
        self.combine_n_groups = clusters_n // min(up.max_swizzle_size, clusters_n)
        if self.combine:
            args.scatter_table = True
            args.multi_buffer_scatter = True
            args.epilogue_store = "bulk_rows_reduce"
            args.scatter_output_buffers = len(up.x_buffers)
        validation_args = argparse.Namespace(**vars(args))
        if self.combine:
            # The benchmark partitions `routes` among outputs. Our externally
            # allocated source buffers can be mostly empty and have independent
            # capacities; build_scatter_row_addresses validates those below.
            validation_args.multi_buffer_scatter = False
        pregather.validate_args(validation_args, prepacked=True)
        self.weights = torch.randn(
            (len(counts), hidden, output_dim), dtype=up.output.dtype, device=up.output.device
        )
        self.weights.mul_(1 / math.sqrt(hidden))
        # A_idx arrays are padded, but only this expert-contiguous prefix is written.
        external = {}
        if self.combine:
            from quack.gemm import build_scatter_row_addresses

            rows = int(up.x_buffers[0].shape[0])
            outputs = tuple(torch.empty(
                (math.ceil(output_dim / args.tile_n), rows * args.tile_n)
                if args.n_tile_major_output else (rows, output_dim),
                dtype=up.output.dtype, device=up.output.device,
            ) for _ in up.x_buffers)
            # Validate every possible destination once, outside iteration timing.
            flat = torch.arange(len(outputs) * rows, device=up.output.device)
            all_mapping = torch.stack((flat // rows, flat % rows), dim=1)
            self.combine_addresses = build_scatter_row_addresses(
                outputs, all_mapping, reduce=True,
                tile_n=args.tile_n if args.n_tile_major_output else None,
            )
            self.combine_rows = rows
            self.combine_valid_rows = torch.zeros(
                len(outputs), dtype=torch.int64, device=up.output.device)
            self.combine_output_to_input = torch.full(
                (len(outputs), rows), -1, dtype=torch.int64, device=up.output.device)
            external = dict(output_buffers=outputs, scatter_mapping=torch.zeros(
                (routes, 2), dtype=torch.int64, device=up.output.device))
        if self.route_weights is not None:
            external["route_weights"] = self.route_weights
        self.inputs = pregather.prepare_grouped_inputs(
            args, up.output[:routes], self.weights, counts, **external
        )
        if self.cpu_scatter and self.cpu_scatter_publication in ("deferred", "threaded"):
            (self.cpu_scatter_metadata, self.inputs.scatter_row_addresses,
             self.route_weights) = make_cpu_scatter_metadata_storage(
                torch, self.inputs.scatter_row_addresses, self.route_weights)
            self.inputs.route_weights = self.route_weights
        if options.trace_down_ready or options.return_token_results:
            if trace_configure is None:
                raise ValueError("down readiness requires the proxy trace allocator")
            self.trace_groups = int(self.inputs.wave_ready_prefix[-1].item())
            address = trace_configure(self.trace_groups)
            self.inputs.wave_ready_host_address = torch.tensor(
                [address], dtype=torch.int64, device=up.output.device)
        self.launch_gemm = pregather.make_launch(args, self.inputs)
        self.combine_tile_n = args.tile_n if self.combine and args.n_tile_major_output else None
        self.return_exchange = None
        self.intra_node_combine = None
        self._trace_first_combine_launch = self.combine
        self.flops = 2 * routes * hidden * output_dim
        self.reset_start = torch.cuda.Event(enable_timing=True)
        self.mapping_start = torch.cuda.Event(enable_timing=True)
        self.gemm_start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(up.cuda_device_id)
        print(f"DownGEMM configured: device_capacity={up.device_capacity} "
              f"tile=({args.tile_m},{args.tile_n}) cluster_m={args.cluster_m} "
              f"pingpong={args.pingpong} sm_count={args.sm_count} "
              f"wave_work_ids={self.inputs.wave_work_ids} "
              f"bulk_store_config={pregather.bulk_store_config_for_args(args)} "
              f"routes={routes} H={hidden} K={output_dim} "
              f"counts={counts} output_shapes="
              f"{[tuple(x.shape) for x in self.inputs.output_buffers()]}", flush=True)
        if self.combine:
            print(f"Local GPU combine: sources={up.source_pairs} "
                  f"rows_per_source={self.combine_rows} epilogue=bulk_rows_reduce "
                  "positions=compact prefixes in input-row order; "
                  f"routing_weights={'FP32' if self.route_weights is not None else 'disabled'}",
                  flush=True)

    def warmup(self):
        """Initialize the down CUDA path before any readiness-gated up launch."""
        torch = self.up.torch
        print(f"[gpu={self.up.cuda_device_id}] DownGEMM warming up before concurrent iterations", flush=True)
        with torch.inference_mode():
            self.up.output.zero_()
            self.inputs.zero_up_output()
            if self.combine:
                # Exercise the full mapping path without reading the unpublished
                # live table. One synthetic batch covers all routes; remaining
                # rows are empty. Preserve production shapes, strides and dtypes.
                table = torch.empty_like(self.up.work_table)
                table.zero_()
                indices = []
                remaining = self.routes
                for source, real_idx in enumerate(self.up.a_idx):
                    count = min(remaining, real_idx.numel())
                    remaining -= count
                    table[:, 2 + 2 * source:4 + 2 * source] = count
                    table[:self.combine_n_groups, 2 + 2 * source] = 0
                    idx = torch.arange(real_idx.numel(), device=real_idx.device,
                                       dtype=real_idx.dtype) % self.combine_rows
                    indices.append(idx)
                if remaining:
                    raise RuntimeError("combine warmup routes exceed A_idx capacity")
                self._prepare_combine(table, tuple(indices))
                print(f"[gpu={self.up.cuda_device_id}] Combine mapping warmup submitted",
                      flush=True)
            # Warm the selected down kernel using the prepared scatter addresses.
            self.launch_gemm()
            for event in (self.reset_start, self.mapping_start, self.gemm_start, self.end):
                event.record()
        torch.cuda.synchronize(self.up.cuda_device_id)
        print(f"[gpu={self.up.cuda_device_id}] DownGEMM warmup complete", flush=True)

    def _log_combine_startup(self, message):
        if self._trace_first_combine_launch:
            print(f"[gpu={self.up.cuda_device_id}] combine startup: {message}", flush=True)

    def configure_cpu_scatter(self, library, handle):
        """Register borrowed scatter storage after warmup, before iteration zero."""
        class Config(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("struct_size", "num_sources", "routes", "capacity", "width",
                         "tile_n", "element_bytes")] + [
                ("scatter_addresses", ctypes.c_size_t),
                ("source_pairs", ctypes.POINTER(ctypes.c_int32)),
                ("output_bases", ctypes.POINTER(ctypes.c_uint64)),
                ("counts", ctypes.POINTER(ctypes.c_int64)),
                ("output_to_input", ctypes.POINTER(ctypes.c_int64))]

        try:
            configure = library.rdma_proxy_configure_cpu_scatter
        except AttributeError as error:
            raise RuntimeError("--cpu-scatter-preparation requires rebuilding librdma_cpu_proxy") from error
        configure.argtypes = [ctypes.c_void_p, ctypes.POINTER(Config)]
        configure.restype = ctypes.c_int
        torch = self.up.torch
        sources = len(self.up.source_pairs)
        pairs = (ctypes.c_int32 * (2 * sources))(
            *(value for pair in self.up.source_pairs for value in pair))
        bases = (ctypes.c_uint64 * sources)(
            *(buffer.data_ptr() for buffer in self.inputs.output_buffers()))
        counts = torch.empty(sources, dtype=torch.int64)
        inverse = torch.empty((sources, self.combine_rows), dtype=torch.int64)
        config = Config(ctypes.sizeof(Config), sources, self.routes, self.combine_rows,
                        int(self.up.x_buffers[0].shape[1]), self.combine_tile_n or 0,
                        self.up.output.element_size(), self.inputs.scatter_row_addresses.data_ptr(),
                        pairs, bases,
                        ctypes.cast(counts.data_ptr(), ctypes.POINTER(ctypes.c_int64)),
                        ctypes.cast(inverse.data_ptr(), ctypes.POINTER(ctypes.c_int64)))
        require_proxy_success(library, configure(handle, ctypes.byref(config)),
                              "configuring CPU scatter preparation")
        if self.route_weights is not None:
            try:
                configure_weights = library.rdma_proxy_configure_cpu_route_weights
            except AttributeError as error:
                raise RuntimeError("weighted CPU preparation requires rebuilding librdma_cpu_proxy") from error
            configure_weights.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
            configure_weights.restype = ctypes.c_int
            require_proxy_success(library, configure_weights(
                handle, self.route_weights.data_ptr(), self.routes),
                "configuring CPU packed routing weights")
        if self.cpu_scatter_publication in ("deferred", "threaded"):
            try:
                configure_publication = getattr(
                    library, f"rdma_proxy_configure_cpu_scatter_{self.cpu_scatter_publication}")
                wait_ready = library.rdma_proxy_wait_cpu_scatter_ready
            except AttributeError as error:
                raise RuntimeError(
                    f"{self.cpu_scatter_publication} CPU scatter requires rebuilding librdma_cpu_proxy") from error
            configure_publication.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            configure_publication.restype = ctypes.c_int
            wait_ready.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint64]
            wait_ready.restype = ctypes.c_int
            require_proxy_success(library, configure_publication(handle, self.cpu_scatter_metadata.numel()),
                                  f"configuring {self.cpu_scatter_publication} CPU scatter publication")
            self._cpu_scatter_wait = wait_ready
            self._cpu_scatter_library, self._cpu_scatter_handle = library, handle
        with torch.inference_mode():
            self.combine_valid_rows.copy_(counts)
            self.combine_output_to_input.copy_(inverse)
        self.cpu_scatter_counts = counts
        torch.cuda.synchronize(self.up.cuda_device_id)
        if self.cpu_scatter_publication == "threaded":
            policy = ("dedicated CPU scatter thread; "
                      f"one {self.cpu_scatter_metadata.numel()}-byte metadata H2D after final gather publication")
        elif self.cpu_scatter_publication == "deferred":
            policy = (f"CPU packing then one {self.cpu_scatter_metadata.numel()}-byte "
                      "metadata H2D after final gather publication")
        else:
            policy = "scatter batches published before gather readiness"
        print(f"CPU scatter preparation enabled: compact metadata fixed at setup; {policy}", flush=True)

    def wait_cpu_scatter_ready(self, iteration):
        """Join the current native upload after its event record has been submitted.

        Waiting on a CUDA event before the producer records this iteration would
        observe an old/unrecorded event. The native API first waits for host-side
        submission, then queues the event dependency without synchronizing the GPU.
        """
        if self._cpu_scatter_wait is None:
            raise RuntimeError("CPU scatter publication wait is not configured")
        stream = self.up.torch.cuda.current_stream(self.up.cuda_device_id)
        require_proxy_success(self._cpu_scatter_library, self._cpu_scatter_wait(
            self._cpu_scatter_handle, iteration, int(stream.cuda_stream), self.cpu_scatter_timeout_ms),
            f"waiting for CPU scatter metadata for iteration {iteration}")

    def _prepare_combine(self, work_table, a_idx):
        torch = self.up.torch
        source, index = local_route_positions(
            torch, work_table, len(a_idx), self.routes, self.combine_n_groups,
            trace=self._log_combine_startup if self._trace_first_combine_launch else None)
        mapping = torch.stack((source, torch.stack(a_idx)[source, index].long()), dim=1)
        if self.route_weights is not None:
            self.route_weights.copy_(self.up.route_weights_stacked[source, index])
        self._log_combine_startup("route mapping submitted; compaction begin")
        mapping, valid_rows, output_to_input = compact_combine_mapping(
            self.up.torch, mapping, len(self.up.x_buffers), self.combine_rows,
        )
        self._log_combine_startup("compaction submitted; scatter address updates begin")
        self.combine_valid_rows.copy_(valid_rows)
        self.combine_output_to_input.copy_(output_to_input)
        flat = mapping[:, 0] * self.combine_rows + mapping[:, 1]
        self.inputs.scatter_mapping.copy_(mapping)
        self.inputs.scatter_table.copy_(flat)
        addresses = self.combine_addresses[flat]
        if self.combine_tile_n is not None:
            # Tile zero's addresses are unchanged. Pack subsequent N tiles
            # using the distinct valid-token count, not allocation capacity.
            addresses[:, 1] = (valid_rows[mapping[:, 0]] * self.combine_tile_n
                               * self.up.output.element_size())
        self.inputs.scatter_row_addresses.copy_(addresses)

    def launch(self, wait_for_metadata=None):
        # QuACK prepares output and metadata tensors under inference_mode.
        # Keep their resets and in-place updates in the same mode.
        with self.up.torch.inference_mode():
            self._log_combine_startup("reset begin")
            self.reset_start.record()
            self.inputs.zero_up_output()
            self.mapping_start.record()
            if wait_for_metadata is not None:
                # Include an exposed upload/wakeup gap in mapping_ms. CPU packing
                # that overlaps dispatch/up remains outside that GPU interval.
                wait_for_metadata()
            if self.combine and not self.cpu_scatter:
                self._log_combine_startup("route mapping begin")
                self._prepare_combine(self.up.work_table, self.up.a_idx)
            elif self.route_weights is not None and not self.cpu_scatter:
                source, index = local_route_positions(
                    self.up.torch, self.up.work_table, len(self.up.a_idx),
                    self.routes, self.combine_n_groups)
                self.route_weights.copy_(self.up.route_weights_stacked[source, index])
            if self.return_exchange is not None:
                self.return_exchange.publish_counts()
            self.gemm_start.record()
            self._log_combine_startup("scatter updates submitted")
            if self.intra_node_combine is not None:
                self._log_combine_startup("intra-node mapping-ready event record begin")
                self.intra_node_combine.mapping_ready.record()
                self._log_combine_startup("intra-node mapping-ready event record returned")
            self._log_combine_startup("down GEMM launch begin")
            self.launch_gemm()
            self._log_combine_startup("down GEMM launch returned")
            if self.intra_node_combine is not None:
                self.intra_node_combine.launch()
            self.end.record()
            self._trace_first_combine_launch = False

    def collected_combine_outputs(self):
        """Return one row-major valid tensor per source for inspection/testing.

        Synchronizes this runner's last launch. The timed path uses device-side
        counts and packed buffers directly, without this host count readback.
        QuACK's generic collected_up_output assumes capacity-based tile strides
        and must not be used for these dynamically packed combine outputs.
        """
        if not self.combine:
            raise ValueError("combine outputs require --local-gpu-combine")
        self.end.synchronize()
        counts = self.combine_valid_rows.tolist()
        width = int(self.up.x_buffers[0].shape[1])
        result = []
        for buffer, count in zip(self.inputs.output_buffers(), counts):
            if self.combine_tile_n is None:
                result.append(buffer[:count])
            else:
                tile = self.combine_tile_n
                n_tiles = math.ceil(width / tile)
                result.append(buffer.view(-1)[:n_tiles * count * tile]
                              .view(n_tiles, count, tile).permute(1, 0, 2)
                              .reshape(count, n_tiles * tile)[:, :width])
        return tuple(result)


def get_expert_counts(library, handle):
    try:
        query = library.rdma_proxy_get_computation_expert_counts
    except AttributeError as error:
        raise RuntimeError("--down-projection2 requires rebuilding librdma_cpu_proxy") from error
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
                      ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int
    size = ctypes.c_uint64()
    require_proxy_success(library, query(handle, None, 0, ctypes.byref(size)), "querying expert count")
    counts = (ctypes.c_uint64 * size.value)()
    require_proxy_success(library, query(handle, counts, size.value, ctypes.byref(size)),
                          "querying per-expert routed rows")
    return list(counts)


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
    # Commas separate GPUs; '+' separates the two devices owned by one GPU.
    device_pairs = [[name.strip() for name in entry.split("+")] for entry in rdma_devices]
    for pair in device_pairs:
        if len(pair) not in (1, 2) or any(not name.strip() for name in pair) or len(set(pair)) != len(pair):
            raise RuntimeError("--rdma-device-map requires one device or two distinct devices joined by '+' per GPU")
    local_devices = device_pairs[local_rank]
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
        f"--rdma_device_name={local_devices[0]}",
        f"--rdma_device_name2={local_devices[1] if len(local_devices) == 2 else ''}",
        f"--listen_port={listen_port}",
        "--router_weighted_contributions_enabled="
        + str(getattr(args, "weighted_contributions", False)).lower(),
    ]
    if args.num_experts is not None:
        generated_args.append(f"--num_experts={args.num_experts}")
    if args.concurrent_kernel:
        generated_args.extend(
            [
                "--router_computation_forwarded_inputs_only=true",
                "--router_local_input_staging_enabled=true",
                f"--expert_gemm_m_tile={args.expert_gemm_m_tile}",
                f"--expert_gemm_n_tile={args.expert_gemm_n_tile}",
                f"--expert_gemm_dimension={projection_dimension(args.expert_gemm_dimension, args.grouped_gemm_activation)}",
                f"--expert_gemm_cluster_m={args.expert_gemm_cluster_m}",
                f"--expert_gemm_max_swizzle_size={args.expert_gemm_max_swizzle_size}",
            ]
        )
        architecture = getattr(args, "gemm_arch", "auto")
        if architecture != "auto":
            generated_args.append(
                f"--expert_gemm_device_capacity={9 if architecture == 'hopper' else 10}")
    if args.inter_node_combine and num_nodes > 1 and not args.rdma_return_token_results:
        raise ValueError("--inter-node-combine requires --rdma-return-token-results with multiple nodes")
    if args.rdma_return_token_results and num_nodes < 2:
        raise ValueError("--rdma-return-token-results requires at least two nodes")
    if args.intra_node_combine:
        generated_args.extend([
            "--nvlink_return_completion_notifications_enabled=true",
            "--nvlink_return_ready_gpu_publication_enabled=true",
        ])
    metadata: dict[str, int | str] = {
        "rank": rank,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "cuda_device_id": cuda_device_id,
        "rdma_device_name": rdma_devices[local_rank],
        "listen_port": listen_port,
        "world_size": world_size,
        "num_nodes": num_nodes,
    }
    return ["rdma_cpu_proxy", *proxy_args, *generated_args], metadata


def check_cuda_runtime_compatibility(torch_cuda_version, library_path, *, mappings=None):
    """Reject mixed runtimes before creating a proxy or starting peer exchanges.

    Match the major ABI, not the patch version. Linux maps repeat a library for
    each segment; the unique path count also preserves QuACK's IPC contract.
    """
    if mappings is None:
        maps_path = Path("/proc/self/maps")
        if not maps_path.exists():
            return  # The real embedded CUDA/RDMA path runs on Linux.
        mappings = maps_path.read_text()
    runtimes = {}
    for line in mappings.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        mapped_path = fields[5]
        path = mapped_path.removesuffix(" (deleted)")
        match = re.fullmatch(r"libcudart\.so(?:\.(\d+)(?:\..*)?)?", Path(path).name)
        if match:
            runtimes[mapped_path] = int(match[1]) if match[1] else None
    expected_major = int(torch_cuda_version.split(".")[0])
    if len(runtimes) != 1 or next(iter(runtimes.values()), None) != expected_major:
        raise RuntimeError(
            f"CUDA runtime mismatch before proxy initialization: PyTorch was built for "
            f"CUDA {torch_cuda_version}, but after loading {library_path} the process has "
            f"these CUDA runtimes: {sorted(runtimes)}. Expected exactly one libcudart.so."
            f"{expected_major}. Rebuild the proxy in a fresh build directory using the "
            f"CUDA {torch_cuda_version} toolkit: set both CMAKE_CUDA_COMPILER to its "
            "bin/nvcc and CUDAToolkit_ROOT to that toolkit root, then point "
            "RDMA_PROXY_BUILD_DIR at the new build. Do not bypass QuACK's runtime check."
        )


def load_proxy_library(path: Path, *, torch_cuda_version=None) -> ctypes.CDLL:
    if not path.is_file():
        raise RuntimeError(
            f"RDMA proxy shared library does not exist: {path}; "
            "build the rdma_cpu_proxy_shared target first"
        )

    library = ctypes.CDLL(str(path))
    if torch_cuda_version is not None:
        check_cuda_runtime_compatibility(torch_cuda_version, path)
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
    library.rdma_proxy_prepare_iteration.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]
    library.rdma_proxy_prepare_iteration.restype = ctypes.c_int
    library.rdma_proxy_record_computation_end.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    library.rdma_proxy_record_computation_end.restype = ctypes.c_int
    library.rdma_proxy_get_computation_elapsed_ms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.rdma_proxy_get_computation_elapsed_ms.restype = ctypes.c_int
    library.rdma_proxy_get_computation_num_tokens.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rdma_proxy_get_computation_num_tokens.restype = ctypes.c_int
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
    *,
    node_rank: int,
    local_gpu_index: int,
) -> TorchDeviceBuffers:
    buffers = TorchDeviceBuffers(
        torch_module,
        cuda_device_id,
        node_rank=node_rank,
        local_gpu_index=local_gpu_index,
    )
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
    *,
    node_rank: int,
    local_gpu_index: int,
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
            library,
            handle,
            torch,
            cuda_device_id,
            node_rank=node_rank,
            local_gpu_index=local_gpu_index,
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


class DownReadyTrace:
    """The C++ proxy owns the mapped flag and polls it on its receiver thread."""

    def __init__(self, library, handle):
        self.library, self.handle = library, handle
        try:
            self.configure_fn = library.rdma_proxy_configure_down_ready_trace
            self.read_fn = library.rdma_proxy_read_down_ready_trace
        except AttributeError as error:
            raise RuntimeError("--trace-down-ready requires rebuilding librdma_cpu_proxy") from error
        self.configure_fn.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                      ctypes.POINTER(ctypes.c_size_t)]
        self.configure_fn.restype = ctypes.c_int
        self.read_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
        ]
        self.read_fn.restype = ctypes.c_int

    def configure(self, groups):
        self.groups = groups
        address = ctypes.c_size_t()
        require_proxy_success(self.library, self.configure_fn(
            self.handle, groups, ctypes.byref(address)), "configuring down-ready trace")
        return address.value

    def read(self, iteration):
        groups = (ctypes.c_int32 * self.groups)()
        times = (ctypes.c_double * self.groups)()
        count = ctypes.c_uint64()
        require_proxy_success(self.library, self.read_fn(
            self.handle, iteration, groups, times, self.groups, ctypes.byref(count)),
            "reading down-ready trace")
        return [(groups[i], times[i]) for i in range(count.value)]


def run_proxy_with_concurrent_kernel(
    library: ctypes.CDLL,
    proxy_argv: list[str],
    cuda_device_id: int,
    *,
    node_rank: int,
    local_gpu_index: int,
    world_size: int,
    num_experts: int,
    expert_gemm_dimension: int,
    activation: str | None,
    expert_gemm_m_tile: int,
    expert_gemm_n_tile: int,
    expert_gemm_k_tile: int | None,
    expert_gemm_cluster_m: int,
    expert_gemm_max_swizzle_size: int,
    quack_root: Path,
    down_options: argparse.Namespace,
) -> int:
    """Launch readiness-gated QuACK GroupedGEMM with each proxy iteration."""

    torch = import_torch_for_proxy(cuda_device_id)
    capacity = resolve_gemm_architecture(
        getattr(down_options, "gemm_arch", "auto"),
        torch.cuda.get_device_capability(cuda_device_id))
    if down_options.down_projection2:
        resolve_down_pingpong(down_options.down_pingpong, capacity,
                              down_options.down_gather_table_wave_barrier)
    # Resolve before C++ allocates tables: SM100 MMA M is not per-CTA M.
    proxy_argv = [*proxy_argv, f"--expert_gemm_device_capacity={capacity}"]
    c_argv = encode_proxy_argv(proxy_argv)
    handle = library.rdma_proxy_create(len(proxy_argv), c_argv)
    if not handle:
        raise RuntimeError(f"proxy creation failed: {proxy_last_error(library)}")

    initialized = False
    coordinator: threading.Thread | None = None
    commands: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    device_buffers: TorchDeviceBuffers | None = None
    trace = None
    return_exchange = None
    success = False
    kernel_stream = None
    down = None
    try:
        device_buffers = install_python_device_buffers(
            library,
            handle,
            torch,
            cuda_device_id,
            node_rank=node_rank,
            local_gpu_index=local_gpu_index,
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

        grouped_gemm = GroupedGemmRunner(
            torch,
            device_buffers,
            cuda_device_id=cuda_device_id,
            world_size=world_size,
            num_experts=num_experts,
            output_dimension=expert_gemm_dimension,
            activation=activation,
            tile_m=expert_gemm_m_tile,
            tile_n=expert_gemm_n_tile,
            tile_k=expert_gemm_k_tile,
            cluster_m=expert_gemm_cluster_m,
            max_swizzle_size=expert_gemm_max_swizzle_size,
            quack_root=quack_root,
        )
        computation_tokens = ctypes.c_uint64()
        require_proxy_success(
            library,
            library.rdma_proxy_get_computation_num_tokens(
                handle, ctypes.byref(computation_tokens)
            ),
            "querying GroupedGEMM routed-token count",
        )
        if computation_tokens.value == 0:
            raise RuntimeError("GroupedGEMM has zero meaningful routed tokens")
        hidden_dimension = int(grouped_gemm.x_buffers[0].shape[1])
        kernel_stream = torch.cuda.Stream(device=cuda_device_id)
        down = None
        if down_options.down_projection2:
            counts = get_expert_counts(library, handle)
            if sum(counts) != computation_tokens.value:
                raise RuntimeError("per-expert counts disagree with total routed rows")
            if down_options.trace_down_ready or down_options.return_token_results:
                trace = DownReadyTrace(library, handle)
            down = DownProjectionRunner(
                torch, grouped_gemm, counts, down_options,
                trace_configure=trace.configure if trace is not None else None)
            with torch.cuda.stream(kernel_stream):
                down.warmup()
            if down.cpu_scatter:
                down.configure_cpu_scatter(library, handle)
            if down_options.return_token_results:
                from rdma_return_exchange import ReturnExchange
                return_exchange = ReturnExchange(
                    torch, library, handle, down, down_options,
                    node_rank, local_gpu_index, cuda_device_id, num_iterations.value)
                if down_options.intra_node_combine:
                    from intra_node_combine import IntraNodeCombine
                    if down_options.trace_down_ready:
                        print(f"[gpu={cuda_device_id}] Intra-node combine setup/compilation begin",
                              flush=True)
                    down.intra_node_combine = IntraNodeCombine(return_exchange, down_options)
                    if down_options.inter_node_combine:
                        from inter_node_combine import InterNodeCombine
                        if down_options.trace_down_ready:
                            print(f"[gpu={cuda_device_id}] Inter-node combine setup/compilation begin",
                                  flush=True)
                        down.intra_node_combine.inter_node_combine = InterNodeCombine(
                            down.intra_node_combine, down_options)
            if (down_options.check_projections and down.route_weights is not None
                    and return_exchange is not None):
                from projection_validation import configure_original_reference
                configure_original_reference(
                    grouped_gemm, down, library, handle, return_exchange,
                    num_experts=num_experts)
        useful_flops = (
            2
            * int(computation_tokens.value)
            * hidden_dimension
            * grouped_gemm.gemm_dimension
        )
        iteration_throughputs: list[tuple[int, float, float]] = []
        down_timings = []
        down_traces = []
        kernel_finished = torch.cuda.Event(enable_timing=False, blocking=False)

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
            "concurrent-kernel mode enabled: Python will launch one readiness-gated "
            "QuACK multi-buffer GroupedGEMM with each proxy iteration",
            flush=True,
        )

        for iteration in range(num_iterations.value):
            def log_startup(message):
                if iteration == 0 and down is not None and down.combine:
                    print(f"[gpu={cuda_device_id}] iteration=0: {message}", flush=True)

            if down is not None and down.intra_node_combine is not None:
                down.intra_node_combine.prepare_iteration()
            log_startup("proxy preparation begin")
            require_proxy_success(
                library,
                library.rdma_proxy_prepare_iteration(handle, iteration),
                f"preparing proxy iteration {iteration}",
            )
            start_barrier = threading.Barrier(2)
            commands.put((iteration, start_barrier))
            start_barrier.wait()
            log_startup("proxy released; up GEMM launch begin")

            kernel_error: BaseException | None = None
            try:
                with torch.cuda.stream(kernel_stream):
                    grouped_gemm.launch()
                log_startup("up GEMM launch returned")
                require_proxy_success(
                    library,
                    library.rdma_proxy_record_computation_end(
                        handle,
                        iteration,
                        int(kernel_stream.cuda_stream),
                    ),
                    f"recording GroupedGEMM end for iteration {iteration}",
                )
                with torch.cuda.stream(kernel_stream):
                    if down is not None:
                        needs_metadata_wait = (down.cpu_scatter and
                                               down.cpu_scatter_publication in ("deferred", "threaded"))
                        down.launch(wait_for_metadata=(lambda: down.wait_cpu_scatter_ready(iteration))
                                    if needs_metadata_wait else None)
                    if return_exchange is not None:
                        return_exchange.arm(iteration)
                    kernel_finished.record(kernel_stream)
            except BaseException as error:
                kernel_error = error
                # Report before waiting for proxy progress, which may be stalled.
                print(f"concurrent CUDA launch failed at iteration {iteration}: {error}",
                      file=sys.stderr, flush=True)

            log_startup("waiting for proxy result")
            proxy_status, proxy_error = results.get()
            log_startup("proxy result received")
            if kernel_error is not None:
                raise RuntimeError(
                    f"concurrent CUDA kernel failed at iteration {iteration}: {kernel_error}"
                ) from kernel_error
            if proxy_status != 0:
                raise RuntimeError(
                    f"proxy iteration {iteration} failed: {proxy_error}"
                )
            kernel_finished.synchronize()
            log_startup("GPU iteration complete")
            if trace is not None:
                down_traces.append((iteration, trace.read(iteration)))
            if return_exchange is not None:
                return_exchange.finish_iteration(iteration)
                if down.intra_node_combine is not None:
                    down.intra_node_combine.finish()
                    if down.intra_node_combine.inter_node_combine is not None:
                        down.intra_node_combine.inter_node_combine.run()
                    # No rank may overwrite returned buffers while a peer still combines.
                    return_exchange.control.gather(f"intra-node-done-{iteration}")
            if down_options.check_projections:
                from projection_validation import check_projections
                check_projections(torch, grouped_gemm, down, int(computation_tokens.value))
                if return_exchange is not None:
                    return_exchange.control.gather(f"projections-checked-{iteration}")
                print(f"Projection numerical checks passed: iteration={iteration}", flush=True)
            elapsed_ms = ctypes.c_float()
            require_proxy_success(
                library,
                library.rdma_proxy_get_computation_elapsed_ms(
                    handle, iteration, ctypes.byref(elapsed_ms)
                ),
                f"querying GroupedGEMM timing for iteration {iteration}",
            )
            if elapsed_ms.value <= 0.0:
                raise RuntimeError(
                    f"GroupedGEMM iteration {iteration} reported invalid "
                    f"elapsed time {elapsed_ms.value} ms"
                )
            effective_tflops = useful_flops / (elapsed_ms.value * 1.0e9)
            iteration_throughputs.append(
                (iteration, float(elapsed_ms.value), effective_tflops)
            )
            if down is not None:
                reset_ms = down.reset_start.elapsed_time(down.mapping_start)
                mapping_ms = down.mapping_start.elapsed_time(down.gemm_start)
                down_ms = down.gemm_start.elapsed_time(down.end)
                # Sum measured stages; excludes host launch gaps between stages.
                tail_ms = down.reset_start.elapsed_time(down.end)
                down_timings.append((iteration, reset_ms, mapping_ms, down_ms,
                                     elapsed_ms.value + tail_ms))

        commands.put(None)
        coordinator.join()
        coordinator = None
        require_proxy_success(library, library.rdma_proxy_finish(handle), "proxy finish")
        print(
            "GroupedGEMM effective throughput by iteration "
            f"(useful_routes={computation_tokens.value}, K={hidden_dimension}, "
            f"projection_N={grouped_gemm.gemm_dimension}, "
            f"output_N={expert_gemm_dimension}, activation={activation or 'none'}; "
            "GEMM FLOPs only, timing includes fused activation):",
            flush=True,
        )
        for iteration, elapsed_value, effective_tflops in iteration_throughputs:
            print(
                f"  iteration {iteration}: elapsed_ms={elapsed_value:.4f} "
                f"effective_throughput={effective_tflops:.2f} TFLOP/s",
                flush=True,
            )
        for iteration, reset_ms, mapping_ms, down_ms, combined_ms in down_timings:
            print(f"  down iteration {iteration}: reset_ms={reset_ms:.4f} "
                  f"mapping_ms={mapping_ms:.4f} "
                  f"gemm_ms={down_ms:.4f} "
                  f"effective_throughput={down.flops / (max(down_ms, 1e-9) * 1e9):.2f} TFLOP/s "
                  f"up_plus_down_ms={combined_ms:.4f}", flush=True)
        if down_options.trace_down_ready:
            print("CPU down-ready trace: detection times since iteration preparation "
                  "(includes up GEMM and host delays; not GPU timestamps). "
                  "Ranges are completed group prefixes; jumps share one observation.",
                  flush=True)
            width = min(down_options.down_max_swizzle_size,
                        math.ceil(hidden_dimension / down_options.down_tile_n)) * down_options.down_tile_n
            for iteration, observations in down_traces:
                previous = 0
                for ready, elapsed_us in observations:
                    print(f"  down-ready iteration {iteration}: groups=[{previous},{ready}) "
                          f"columns=[{previous * width},{min(ready * width, hidden_dimension)}) "
                          f"detected_us={elapsed_us:.3f}", flush=True)
                    previous = ready
        success = True
        return 0
    finally:
        if down is not None and down.intra_node_combine is not None:
            down.intra_node_combine.abort()
        # Keep borrowed inputs, scatter destinations, and any mapped host flag
        # alive until the last possible kernel write, including launch failures.
        if kernel_stream is not None:
            try:
                kernel_stream.synchronize()
            except Exception as error:
                print(f"compute stream cleanup failed: {error}", file=sys.stderr, flush=True)
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
        if return_exchange is not None:
            return_exchange.close(success and shutdown_status == 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one librdma_cpu_proxy.so instance in a torchrun worker",
        allow_abbrev=False,
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cuda-device-map", required=True)
    parser.add_argument("--rdma-device-map", required=True,
                        help="comma-separated GPU entries; use deviceA+deviceB for two devices per GPU")
    parser.add_argument("--listen-port-base", type=int, default=18515)
    parser.add_argument("--local-rank", "--local_rank", dest="launcher_local_rank", type=int)
    parser.add_argument(
        "--concurrent-kernel",
        action="store_true",
        help="launch QuACK multi-buffer GroupedGEMM with every proxy iteration",
    )
    parser.add_argument(
        "--communication-only",
        action="store_true",
        help="override GEMM, return, and combine options; run only proxy dispatch/forwarding",
    )
    parser.add_argument("--num-experts", "--num_experts", type=int)
    parser.add_argument("--gemm-arch", choices=("auto", "hopper", "blackwell"), default="auto",
                        help="resolve GEMM/table geometry before native initialization; explicit values are checked against the GPU")
    parser.add_argument("--check-projections", action=argparse.BooleanOptionalAction, default=False,
                        help="check up/down numerical values each iteration, outside GEMM timing")
    parser.add_argument("--weighted-contributions", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="multiply down FP32 accumulators by fixed normalized routing weights")
    parser.add_argument(
        "--grouped-gemm-activation",
        choices=("none", *ORDINARY_ACTIVATIONS, *GATED_ACTIVATIONS),
        default="none",
        help="fused up-projection activation; requires --concurrent-kernel",
    )
    parser.add_argument("--grouped-gemm-m-tile", dest="expert_gemm_m_tile", type=int, default=128)
    parser.add_argument("--grouped-gemm-n-tile", dest="expert_gemm_n_tile", type=int, default=256)
    parser.add_argument(
        "--grouped-gemm-k-tile", dest="expert_gemm_k_tile", type=int, default=None
    )
    parser.add_argument(
        "--grouped-gemm-output-dimension",
        dest="expert_gemm_dimension",
        type=int,
        default=8192,
        help="post-activation width; gated activations use twice this GEMM width",
    )
    parser.add_argument(
        "--grouped-gemm-cluster-m", dest="expert_gemm_cluster_m", type=int, default=1
    )
    parser.add_argument(
        "--grouped-gemm-max-swizzle-size",
        dest="expert_gemm_max_swizzle_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--quack-root",
        type=Path,
        default=Path(os.environ.get("QUACK_ROOT", Path(__file__).resolve().parents[3] / "quack_for_communication_overlap")),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the derived C++ argv without loading the shared library",
    )
    add_down_arguments(parser)
    args, proxy_args = parser.parse_known_args()
    if any(arg.startswith(("--down-", "--no-down-")) for arg in proxy_args):
        parser.error("unsupported --down-* option")
    if args.communication_only:
        # Apply the mode after parsing so it wins regardless of argument order.
        # Reset every option owned by the down/return/combine pipeline together.
        down_parser = argparse.ArgumentParser(add_help=False)
        add_down_arguments(down_parser)
        vars(args).update(vars(down_parser.parse_args([])))
        args.concurrent_kernel = False
        args.check_projections = False
        args.weighted_contributions = False
        args.grouped_gemm_activation = "none"
        for name in vars(args):
            if name.startswith("expert_gemm_"):
                setattr(args, name, parser.get_default(name))
    try:
        validate_down_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.communication_only and not args.down_projection2 and any(
        arg.startswith(("--down-", "--no-down-")) for arg in sys.argv[1:]
    ):
        parser.error("--down-* options require --down-projection2")
    if args.grouped_gemm_activation != "none" and not args.concurrent_kernel:
        parser.error("--grouped-gemm-activation requires --concurrent-kernel")
    if args.check_projections and not args.concurrent_kernel:
        parser.error("--check-projections requires --concurrent-kernel")
    if args.weighted_contributions and not (args.down_projection2 and args.down_gather_table):
        parser.error("--weighted-contributions requires --down-projection2 and --down-gather-table")
    if args.expert_gemm_dimension <= 0:
        parser.error("--grouped-gemm-output-dimension must be positive")
    args.grouped_gemm_activation = (
        None if args.grouped_gemm_activation == "none" else args.grouped_gemm_activation
    )

    proxy_argv, metadata = build_proxy_argv(args, proxy_args)
    if args.communication_only:
        # Last CLI values override both the launch script and JSON settings.
        proxy_argv.extend([
            "--nvlink_return_completion_notifications_enabled=false",
            "--nvlink_return_ready_gpu_publication_enabled=false",
        ])
        print("communication-only mode: GEMM, result returns, and combine disabled", flush=True)
    print(
        "starting embedded RDMA CPU proxy "
        f"rank={metadata['rank']} node_rank={metadata['node_rank']} "
        f"local_gpu={metadata['local_rank']} cuda_device={metadata['cuda_device_id']} "
        f"rdma_device={metadata['rdma_device_name']} "
        f"listen_port={metadata['listen_port']}",
        flush=True,
    )
    if args.dry_run:
        if args.concurrent_kernel and args.gemm_arch == "auto":
            print("GEMM architecture: auto (resolved from CUDA before native initialization; "
                  "use --gemm-arch=blackwell or hopper to inspect explicit scheduler geometry)", flush=True)
        print(shlex.join(proxy_argv), flush=True)
        return 0

    # Load PyTorch's runtime first so a matching native SONAME reuses it. Check
    # immediately after dlopen, before native initialization can contact peers.
    torch = import_torch_for_proxy(int(metadata["cuda_device_id"]))
    library = load_proxy_library(args.library.resolve(), torch_cuda_version=torch.version.cuda)
    if args.concurrent_kernel:
        if args.num_experts is None:
            raise RuntimeError("--concurrent-kernel requires --num-experts")
        return run_proxy_with_concurrent_kernel(
            library,
            proxy_argv,
            int(metadata["cuda_device_id"]),
            node_rank=int(metadata["node_rank"]),
            local_gpu_index=int(metadata["local_rank"]),
            world_size=int(metadata["world_size"]),
            num_experts=args.num_experts,
            expert_gemm_dimension=args.expert_gemm_dimension,
            activation=args.grouped_gemm_activation,
            expert_gemm_m_tile=args.expert_gemm_m_tile,
            expert_gemm_n_tile=args.expert_gemm_n_tile,
            expert_gemm_k_tile=args.expert_gemm_k_tile,
            expert_gemm_cluster_m=args.expert_gemm_cluster_m,
            expert_gemm_max_swizzle_size=args.expert_gemm_max_swizzle_size,
            quack_root=args.quack_root,
            down_options=args,
        )
    return run_proxy(
        library,
        proxy_argv,
        int(metadata["cuda_device_id"]),
        node_rank=int(metadata["node_rank"]),
        local_gpu_index=int(metadata["local_rank"]),
    )


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
