#!/usr/bin/env python3
"""Generate R on the GPU, build expert token-index lists, check, and benchmark.

Example (H100):
  python run.py --tokens 16384 --top-k 8 --experts 256 \
      --experts-per-gpu 32 --check --warmup 20 --iters 200
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.cpp_extension import load


BLOCK_TOKENS = 256
_extension = None


def load_extension():
    global _extension
    if _extension is not None:
        return _extension
    here = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(here / ".torch_extensions"))
    _extension = load(
        name="expert_token_idx_ext",
        sources=[
            str(here / "expert_token_idx_ext.cpp"),
            str(here / "expert_token_idx_kernel.cu"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )
    return _extension


def allocate_outputs(
    r: torch.Tensor, num_experts: int, experts_per_gpu: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = r.shape[0]
    num_gpus = num_experts // experts_per_gpu
    num_chunks = (num_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    num_bins = num_experts + num_gpus
    values = torch.empty(r.numel(), dtype=torch.int32, device=r.device)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=r.device)
    counts = torch.empty(
        (num_chunks, num_bins), dtype=torch.int32, device=r.device
    )
    prefixes = torch.empty_like(counts)
    return values, offsets, counts, prefixes


def allocate_node_mask_outputs(
    r: torch.Tensor,
    num_experts: int,
    experts_per_gpu: int,
    gpus_per_node: int,
):
    num_tokens, top_k = r.shape
    experts_per_node = experts_per_gpu * gpus_per_node
    num_nodes = num_experts // experts_per_node
    num_node_masks = 1 << gpus_per_node
    num_input_chunks = (num_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    input_num_bins = num_experts + num_nodes * num_node_masks
    reordered_num_bins = experts_per_node + gpus_per_node

    values = torch.empty(r.numel(), dtype=torch.int32, device=r.device)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=r.device)
    max_node_tokens = num_tokens * min(top_k, num_nodes)
    node_token_indices = torch.empty(
        max_node_tokens, dtype=torch.int32, device=r.device
    )
    node_token_masks = torch.empty(
        max_node_tokens, dtype=torch.uint8, device=r.device
    )
    node_offsets = torch.empty(num_nodes + 1, dtype=torch.int32, device=r.device)
    input_counts = torch.empty(
        (num_input_chunks, input_num_bins), dtype=torch.int32, device=r.device
    )
    input_prefixes = torch.empty_like(input_counts)
    node_mask_offsets = torch.empty(
        (num_nodes, num_node_masks), dtype=torch.int32, device=r.device
    )
    reordered_counts = torch.empty(
        (num_nodes, num_input_chunks, reordered_num_bins),
        dtype=torch.int32,
        device=r.device,
    )
    reordered_prefixes = torch.empty_like(reordered_counts)
    return (
        values,
        offsets,
        node_token_indices,
        node_token_masks,
        node_offsets,
        input_counts,
        input_prefixes,
        node_mask_offsets,
        reordered_counts,
        reordered_prefixes,
    )


def cpu_reference(
    r: torch.Tensor, num_experts: int, experts_per_gpu: int
) -> List[List[int]]:
    routes = r.cpu().tolist()
    num_gpus = num_experts // experts_per_gpu
    gpu_positions = [0] * num_gpus
    result: List[List[int]] = [[] for _ in range(num_experts)]
    for row in routes:
        routed_gpus = {expert // experts_per_gpu for expert in row}
        for expert in row:
            result[expert].append(gpu_positions[expert // experts_per_gpu])
        for gpu in routed_gpus:
            gpu_positions[gpu] += 1
    return result


def check_result(
    r: torch.Tensor,
    values: torch.Tensor,
    offsets: torch.Tensor,
    num_experts: int,
    experts_per_gpu: int,
) -> None:
    expected = cpu_reference(r, num_experts, experts_per_gpu)
    host_values = values.cpu()
    host_offsets = offsets.cpu().tolist()
    if host_offsets[0] != 0 or host_offsets[-1] != r.numel():
        raise AssertionError(
            f"invalid offsets endpoints: {host_offsets[0]}, {host_offsets[-1]}"
        )
    for expert, expected_list in enumerate(expected):
        begin, end = host_offsets[expert], host_offsets[expert + 1]
        actual = host_values[begin:end].tolist()
        if actual != expected_list:
            raise AssertionError(
                f"expert {expert} mismatch: expected {expected_list[:16]}, "
                f"got {actual[:16]} (lengths {len(expected_list)} vs {len(actual)})"
            )


def cpu_node_mask_reference(
    r: torch.Tensor,
    num_experts: int,
    experts_per_gpu: int,
    gpus_per_node: int,
    local_gpu_id: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    routes = r.cpu().tolist()
    experts_per_node = experts_per_gpu * gpus_per_node
    num_nodes = num_experts // experts_per_node
    node_entries: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]

    for token, row in enumerate(routes):
        masks = {}
        for expert in row:
            node = expert // experts_per_node
            local_gpu = (expert // experts_per_gpu) % gpus_per_node
            bit = (local_gpu_id - local_gpu) % gpus_per_node
            masks[node] = masks.get(node, 0) | (1 << bit)
        for node, mask in masks.items():
            node_entries[node].append((mask, token))

    # Python's sort is stable, matching the CUDA counting sort's token-order
    # tie break for equal masks.
    for entries in node_entries:
        entries.sort(key=lambda item: -item[0])

    expert_lists: List[List[int]] = [[] for _ in range(num_experts)]
    for node, entries in enumerate(node_entries):
        gpu_positions = [0] * gpus_per_node
        for node_position, (mask, token) in enumerate(entries):
            for expert in routes[token]:
                if expert // experts_per_node == node:
                    local_gpu = (expert // experts_per_gpu) % gpus_per_node
                    expert_lists[expert].append(
                        node_position
                        if local_gpu == local_gpu_id
                        else gpu_positions[local_gpu]
                    )
            for gpu in range(gpus_per_node):
                bit = (local_gpu_id - gpu) % gpus_per_node
                if mask & (1 << bit):
                    gpu_positions[gpu] += 1
    return (
        expert_lists,
        [[token for _, token in entries] for entries in node_entries],
        [[mask for mask, _ in entries] for entries in node_entries],
    )


def check_node_mask_result(
    r: torch.Tensor,
    values: torch.Tensor,
    offsets: torch.Tensor,
    node_token_indices: torch.Tensor,
    node_token_masks: torch.Tensor,
    node_offsets: torch.Tensor,
    num_experts: int,
    experts_per_gpu: int,
    gpus_per_node: int,
    local_gpu_id: int,
) -> None:
    expected_experts, expected_nodes, expected_masks = cpu_node_mask_reference(
        r, num_experts, experts_per_gpu, gpus_per_node, local_gpu_id
    )
    host_values = values.cpu()
    host_offsets = offsets.cpu().tolist()
    host_node_values = node_token_indices.cpu()
    host_node_masks = node_token_masks.cpu()
    host_node_offsets = node_offsets.cpu().tolist()
    if host_offsets[0] != 0 or host_offsets[-1] != r.numel():
        raise AssertionError(
            f"invalid expert offsets endpoints: {host_offsets[0]}, "
            f"{host_offsets[-1]}"
        )
    if host_node_offsets[0] != 0:
        raise AssertionError(f"invalid first node offset: {host_node_offsets[0]}")
    for node, expected in enumerate(expected_nodes):
        begin, end = host_node_offsets[node], host_node_offsets[node + 1]
        actual = host_node_values[begin:end].tolist()
        if actual != expected:
            raise AssertionError(
                f"node {node} x3 mismatch: expected {expected[:16]}, "
                f"got {actual[:16]} (lengths {len(expected)} vs {len(actual)})"
            )
        expected_node_masks = expected_masks[node]
        actual_masks = host_node_masks[begin:end].tolist()
        if actual_masks != expected_node_masks:
            raise AssertionError(
                f"node {node} x4 mismatch: expected {expected_node_masks[:16]}, "
                f"got {actual_masks[:16]} "
                f"(lengths {len(expected_node_masks)} vs {len(actual_masks)})"
            )
    for expert, expected in enumerate(expected_experts):
        begin, end = host_offsets[expert], host_offsets[expert + 1]
        actual = host_values[begin:end].tolist()
        if actual != expected:
            raise AssertionError(
                f"expert {expert} mismatch: expected {expected[:16]}, "
                f"got {actual[:16]} (lengths {len(expected)} vs {len(actual)})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16_384)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--experts-per-gpu", type=int, default=32)
    parser.add_argument(
        "--node-mask-sort",
        action="store_true",
        help="reorder each node's sender tokens by descending GPU mask",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        choices=range(2, 9),
        default=8,
        metavar="N",
        help="GPUs per node for --node-mask-sort (2-8; default: 8)",
    )
    parser.add_argument(
        "--local-gpu-id",
        type=int,
        default=0,
        help="node-local GPU running the algorithm (default: 0)",
    )
    parser.add_argument(
        "--x3-x4-to-cpu",
        action="store_true",
        help="copy x3/x4 and node offsets to pinned CPU buffers",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires an NVIDIA GPU")
    if args.tokens <= 0 or args.top_k <= 0 or args.experts <= 0:
        raise ValueError("tokens, top-k, and experts must be positive")
    if args.experts > 1024:
        raise ValueError("this kernel supports at most 1024 experts")
    if args.top_k > args.experts:
        raise ValueError("top-k cannot exceed experts")
    if args.tokens * args.top_k > 2**31 - 1:
        raise ValueError("tokens * top-k must fit in int32")
    if args.experts_per_gpu <= 0 or args.experts % args.experts_per_gpu:
        raise ValueError("experts-per-gpu must be positive and divide experts")
    if not 0 <= args.local_gpu_id < args.gpus_per_node:
        raise ValueError("local-gpu-id must be in [0, gpus-per-node - 1]")
    if args.x3_x4_to_cpu and not args.node_mask_sort:
        raise ValueError("x3-x4-to-cpu requires --node-mask-sort")
    if args.node_mask_sort and args.experts % (
        args.experts_per_gpu * args.gpus_per_node
    ):
        raise ValueError(
            "node-mask mode requires experts to be divisible by "
            "experts-per-gpu * gpus-per-node"
        )
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if not 0 <= args.seed <= 2**64 - 1:
        raise ValueError("seed must fit in uint64")

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if props.major < 9:
        raise RuntimeError(
            f"this benchmark targets Hopper (sm_90+), got sm_{props.major}{props.minor}"
        )
    extension = load_extension()

    # R is allocated directly in HBM and initialized by generate_r_.
    r = torch.empty(
        (args.tokens, args.top_k), dtype=torch.int32, device=device
    )
    extension.generate_r_(r, args.experts, args.seed)
    node_token_indices = None
    node_offsets = None
    if args.node_mask_sort:
        (
            values,
            offsets,
            node_token_indices,
            node_token_masks,
            node_offsets,
            input_counts,
            input_prefixes,
            node_mask_offsets,
            reordered_counts,
            reordered_prefixes,
        ) = allocate_node_mask_outputs(
            r, args.experts, args.experts_per_gpu, args.gpus_per_node
        )

        def launch() -> None:
            extension.get_expert_token_idx_node_mask(
                r,
                args.experts,
                args.experts_per_gpu,
                args.gpus_per_node,
                args.local_gpu_id,
                False,
                values,
                offsets,
                node_token_indices,
                node_token_masks,
                node_offsets,
                input_counts,
                input_prefixes,
                node_mask_offsets,
                reordered_counts,
                reordered_prefixes,
            )

        def launch_x3() -> None:
            extension.get_expert_token_idx_node_mask(
                r,
                args.experts,
                args.experts_per_gpu,
                args.gpus_per_node,
                args.local_gpu_id,
                True,
                values,
                offsets,
                node_token_indices,
                node_token_masks,
                node_offsets,
                input_counts,
                input_prefixes,
                node_mask_offsets,
                reordered_counts,
                reordered_prefixes,
            )

    else:
        values, offsets, counts, prefixes = allocate_outputs(
            r, args.experts, args.experts_per_gpu
        )

        def launch() -> None:
            extension.get_expert_token_idx(
                r,
                args.experts,
                args.experts_per_gpu,
                values,
                offsets,
                counts,
                prefixes,
            )

    launch()
    torch.cuda.synchronize()
    num_node_tokens = (
        node_offsets[-1].item() if args.node_mask_sort else None
    )
    host_node_token_indices = None
    host_node_token_masks = None
    host_node_offsets = None
    if args.x3_x4_to_cpu:
        num_nodes = args.experts // (
            args.experts_per_gpu * args.gpus_per_node
        )
        host_node_token_indices = torch.empty(
            num_node_tokens, dtype=torch.int32, device="cpu", pin_memory=True
        )
        host_node_token_masks = torch.empty(
            num_node_tokens, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        host_node_offsets = torch.empty(
            num_nodes + 1, dtype=torch.int32, device="cpu", pin_memory=True
        )

    if args.node_mask_sort:
        def launch_x3_output() -> None:
            launch_x3()
            if args.x3_x4_to_cpu:
                host_node_token_indices.copy_(
                    node_token_indices[:num_node_tokens], non_blocking=True
                )
                host_node_token_masks.copy_(
                    node_token_masks[:num_node_tokens], non_blocking=True
                )
                host_node_offsets.copy_(node_offsets, non_blocking=True)

    if args.check:
        if args.node_mask_sort:
            check_node_mask_result(
                r,
                values,
                offsets,
                node_token_indices,
                node_token_masks,
                node_offsets,
                args.experts,
                args.experts_per_gpu,
                args.gpus_per_node,
                args.local_gpu_id,
            )
        else:
            check_result(
                r, values, offsets, args.experts, args.experts_per_gpu
            )

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        launch()
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    latency_us = elapsed_ms * 1000.0 / args.iters

    x3_latency_us = None
    if args.node_mask_sort:
        for _ in range(args.warmup):
            launch_x3_output()
        torch.cuda.synchronize()

        x3_start = torch.cuda.Event(enable_timing=True)
        x3_end = torch.cuda.Event(enable_timing=True)
        x3_start.record()
        for _ in range(args.iters):
            launch_x3_output()
        x3_end.record()
        x3_end.synchronize()
        x3_elapsed_ms = x3_start.elapsed_time(x3_end)
        x3_latency_us = x3_elapsed_ms * 1000.0 / args.iters
        if args.x3_x4_to_cpu and args.check:
            check_node_mask_result(
                r,
                values,
                offsets,
                host_node_token_indices,
                host_node_token_masks,
                host_node_offsets,
                args.experts,
                args.experts_per_gpu,
                args.gpus_per_node,
                args.local_gpu_id,
            )

    mode_details = (
        f"mode=node-mask-sort GPUs_per_node={args.gpus_per_node} "
        f"local_gpu_id={args.local_gpu_id}"
        if args.node_mask_sort
        else "mode=gpu"
    )
    if args.node_mask_sort:
        output_location = "cpu" if args.x3_x4_to_cpu else "gpu"
        mode_details += f" x3_tokens={num_node_tokens}"
        mode_details += f" x3_x4_output={output_location}"
        mode_details += f" node_offsets_output={output_location}"
        mode_details += f" x3_latency={x3_latency_us:.3f} us"
    print(
        f"device={props.name} sm={props.major}.{props.minor} "
        f"T={args.tokens} topK={args.top_k} K={args.experts} "
        f"n={args.experts_per_gpu} GPUs={args.experts // args.experts_per_gpu} "
        f"{mode_details} "
        f"latency={latency_us:.3f} us "
        f"check={'OK' if args.check else 'not requested'}"
    )


if __name__ == "__main__":
    main()
