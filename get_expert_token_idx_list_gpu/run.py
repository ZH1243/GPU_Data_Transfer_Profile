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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16_384)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--experts-per-gpu", type=int, default=32)
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
    if args.check:
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

    print(
        f"device={props.name} sm={props.major}.{props.minor} "
        f"T={args.tokens} topK={args.top_k} K={args.experts} "
        f"n={args.experts_per_gpu} GPUs={args.experts // args.experts_per_gpu} "
        f"latency={latency_us:.3f} us "
        f"check={'OK' if args.check else 'not requested'}"
    )


if __name__ == "__main__":
    main()
