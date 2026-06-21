#!/usr/bin/env python3
"""
Benchmark global-memory key sorting plus physical payload reordering.

Kernel A initializes two global-memory buffers on each torchrun rank:
  - payload: num_elements records, each payload_size bytes
  - keys:    num_elements records, each key_size bytes

The first byte of every key record is treated as an unsigned 8-bit key.
Kernel B then:
  1. counting-sorts the keys from largest to smallest
  2. physically reorders payload records into the same sorted order

The payload reorder path can use cooperative ordinary global load/store copies
(--payload-copy-method lsu) or a Hopper TMA-staged path
(--payload-copy-method tma). The --num-sms parameter controls the number of
CTAs launched for both steps of kernel B; with one CTA per requested SM, this is
the usual way to bound the participating SMs for this benchmark.

Example:
  torchrun --standalone --nproc_per_node=8 \
    Global_Mem_Reordering/global_mem_reordering_test.py \
    --num-elements 10000 --payload-size 8K --key-size 1 \
    --bit-probability 0.4 --num-sms 0 --iters 100 --warmup 10 --check
"""

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load


KEY_VALUES = 256
COPY_METHOD_TO_ID = {"lsu": 0, "tma": 1}


def parse_nbytes(text: str) -> int:
    value = text.strip().upper()
    scale = 1
    for suffix, multiplier in [
        ("KIB", 1024),
        ("MIB", 1024**2),
        ("GIB", 1024**3),
        ("KB", 1000),
        ("MB", 1000**2),
        ("GB", 1000**3),
        ("K", 1024),
        ("M", 1024**2),
        ("G", 1024**3),
    ]:
        if value.endswith(suffix):
            scale = multiplier
            value = value[: -len(suffix)]
            break
    try:
        result = int(float(value) * scale)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte size: {text!r}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("byte size must be positive")
    return result


def fmt_bytes(nbytes: int) -> str:
    for suffix, scale in [("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)]:
        if nbytes >= scale:
            return f"{nbytes / scale:.2f} {suffix}"
    return f"{nbytes} B"


def get_rank_info() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return rank, world_size, local_rank
    return 0, 1, 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


_extension = None


def get_extension():
    global _extension
    if _extension is not None:
        return _extension

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    here = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(here / ".torch_extensions"))
    _extension = load(
        name="global_mem_reordering_ext",
        sources=[
            str(here / "global_mem_reordering_ext.cpp"),
            str(here / "global_mem_reordering_kernels.cu"),
        ],
        extra_cflags=["-std=c++17"],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr"],
        verbose=False,
    )
    return _extension


def launch_kernel_a(ext,
                    payload: torch.Tensor,
                    keys: torch.Tensor,
                    args: argparse.Namespace,
                    init_ctas: int,
                    device: int) -> None:
    ext.launch_kernel_a(
        payload,
        keys,
        args.num_elements,
        args.payload_size,
        args.key_size,
        args.bit_probability,
        args.seed,
        init_ctas,
        args.threads_per_cta,
        device,
    )


def launch_kernel_b(ext,
                    payload: torch.Tensor,
                    keys: torch.Tensor,
                    sorted_payload: torch.Tensor,
                    sorted_keys: torch.Tensor,
                    source_indices: torch.Tensor,
                    local_histograms: torch.Tensor,
                    key_offsets: torch.Tensor,
                    write_offsets: torch.Tensor,
                    args: argparse.Namespace,
                    num_b_ctas: int,
                    device: int) -> None:
    ext.launch_kernel_b(
        payload,
        keys,
        sorted_payload,
        sorted_keys,
        source_indices,
        local_histograms,
        key_offsets,
        write_offsets,
        args.num_elements,
        args.payload_size,
        args.key_size,
        num_b_ctas,
        args.threads_per_cta,
        COPY_METHOD_TO_ID[args.payload_copy_method],
        args.tma_tile_bytes,
        device,
    )


def check_outputs(payload: torch.Tensor,
                  keys: torch.Tensor,
                  sorted_payload: torch.Tensor,
                  sorted_keys: torch.Tensor,
                  source_indices: torch.Tensor,
                  num_elements: int,
                  payload_size: int,
                  key_size: int) -> Tuple[bool, str]:
    key_values = sorted_keys.view(num_elements, key_size)[:, 0].cpu()
    if torch.any(key_values[:-1] < key_values[1:]):
        bad = int(torch.nonzero(key_values[:-1] < key_values[1:], as_tuple=False)[0].item())
        return False, (
            f"keys not descending at sorted positions {bad}/{bad + 1}: "
            f"{int(key_values[bad])} < {int(key_values[bad + 1])}"
        )

    original_counts = torch.bincount(
        keys.view(num_elements, key_size)[:, 0].cpu().to(torch.int64),
        minlength=KEY_VALUES,
    )
    sorted_counts = torch.bincount(key_values.to(torch.int64), minlength=KEY_VALUES)
    if not torch.equal(original_counts, sorted_counts):
        return False, "sorted key histogram does not match original key histogram"

    samples = min(64, num_elements)
    if samples == 1:
        positions = torch.zeros(1, dtype=torch.int64, device=payload.device)
    else:
        positions = (
            torch.arange(samples, dtype=torch.int64, device=payload.device)
            * (num_elements - 1)
            // (samples - 1)
        )
    src_indices = source_indices[positions].to(torch.int64)

    byte_offsets = sorted({0, payload_size // 2, payload_size - 1})
    payload_2d = payload.view(num_elements, payload_size)
    sorted_payload_2d = sorted_payload.view(num_elements, payload_size)
    for offset in byte_offsets:
        expected = payload_2d[src_indices, offset]
        actual = sorted_payload_2d[positions, offset]
        mismatch = torch.nonzero(actual != expected, as_tuple=False)
        if mismatch.numel() > 0:
            sample = int(mismatch[0].item())
            return False, (
                f"payload mismatch at sorted element {int(positions[sample])}, "
                f"payload byte {offset}"
            )
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-elements", "--num_elements", type=int, default=10_000)
    parser.add_argument("--payload-size", "--payload_size", type=parse_nbytes,
                        default=parse_nbytes("8K"))
    parser.add_argument("--key-size", "--key_size", type=parse_nbytes,
                        default=1)
    parser.add_argument("--bit-probability", "--bit_probability", type=float,
                        default=0.4,
                        help="probability that each key bit is initialized to 1")
    parser.add_argument("--num-sms", "--num_sms", type=int, default=0,
                        help="CTAs/SMs used by kernel B; 0 means all visible SMs")
    parser.add_argument("--threads-per-cta", "--threads_per_cta", type=int,
                        default=256)
    parser.add_argument("--payload-copy-method", "--payload_copy_method",
                        choices=("tma", "lsu"), default="tma")
    parser.add_argument("--tma-tile-bytes", "--tma_tile_bytes", type=parse_nbytes,
                        default=parse_nbytes("8K"))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sleep-before", "--sleep_before", type=float, default=0.0)
    args = parser.parse_args()

    if args.num_elements <= 0:
        raise ValueError("--num-elements must be positive")
    if args.num_elements > 2**31 - 1:
        raise ValueError("--num-elements must fit int32")
    if args.key_size <= 0:
        raise ValueError("--key-size must be positive")
    if not (0.0 <= args.bit_probability <= 1.0):
        raise ValueError("--bit-probability must be in [0, 1]")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.threads_per_cta <= 0 or args.threads_per_cta > 1024:
        raise ValueError("--threads-per-cta must be in 1..1024")

    rank, world_size, local_rank = get_rank_info()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch build")

    num_devices = torch.cuda.device_count()
    if local_rank >= num_devices:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {num_devices} CUDA devices are visible"
        )

    device = local_rank
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if args.num_sms == 0:
        num_b_ctas = props.multi_processor_count
    else:
        num_b_ctas = args.num_sms
    if num_b_ctas <= 0 or num_b_ctas > props.multi_processor_count:
        raise ValueError(
            f"--num-sms must be in 1..{props.multi_processor_count}, or 0 for all SMs"
        )

    if args.payload_copy_method == "tma" and props.major < 9:
        raise RuntimeError("--payload-copy-method tma requires Hopper / compute capability 9.0+")
    if args.payload_copy_method == "tma" and args.payload_size % 16 != 0:
        raise ValueError("--payload-copy-method tma requires --payload-size to be a multiple of 16")
    if args.payload_copy_method == "tma" and args.tma_tile_bytes % 16 != 0:
        raise ValueError("--payload-copy-method tma requires --tma-tile-bytes to be a multiple of 16")

    total_payload_bytes = args.num_elements * args.payload_size
    total_key_bytes = args.num_elements * args.key_size

    ext = get_extension()

    payload = torch.empty(total_payload_bytes, dtype=torch.uint8, device=device)
    keys = torch.empty(total_key_bytes, dtype=torch.uint8, device=device)
    sorted_payload = torch.empty_like(payload)
    sorted_keys = torch.empty_like(keys)
    source_indices = torch.empty(args.num_elements, dtype=torch.int32, device=device)
    local_histograms = torch.empty(num_b_ctas * KEY_VALUES, dtype=torch.int32, device=device)
    key_offsets = torch.empty(KEY_VALUES, dtype=torch.int32, device=device)
    write_offsets = torch.empty(KEY_VALUES, dtype=torch.int32, device=device)

    init_ctas = max(props.multi_processor_count * 4, num_b_ctas)
    launch_kernel_a(ext, payload, keys, args, init_ctas, device)
    torch.cuda.synchronize(device)

    barrier()
    if args.sleep_before > 0:
        if rank == 0:
            print(f"Sleeping for {args.sleep_before:.1f}s before benchmark...", flush=True)
        time.sleep(args.sleep_before)
    barrier()

    for _ in range(args.warmup):
        launch_kernel_b(
            ext, payload, keys, sorted_payload, sorted_keys, source_indices,
            local_histograms, key_offsets, write_offsets, args, num_b_ctas, device
        )
    torch.cuda.synchronize(device)
    barrier()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        launch_kernel_b(
            ext, payload, keys, sorted_payload, sorted_keys, source_indices,
            local_histograms, key_offsets, write_offsets, args, num_b_ctas, device
        )
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    barrier()

    elapsed = t1 - t0
    bytes_reordered = total_payload_bytes * args.iters
    gib_reordered = bytes_reordered / (1024**3)
    local_bw = gib_reordered / elapsed if elapsed > 0 else 0.0

    ok = True
    failure = ""
    if args.check:
        ok, failure = check_outputs(
            payload, keys, sorted_payload, sorted_keys, source_indices,
            args.num_elements, args.payload_size, args.key_size
        )

    for output_rank in range(world_size):
        barrier()
        if rank == output_rank:
            print(
                f"rank={rank:02d} device=cuda:{device} "
                f"num_elements={args.num_elements} "
                f"payload_size={fmt_bytes(args.payload_size)} "
                f"key_size={fmt_bytes(args.key_size)} "
                f"bit_probability={args.bit_probability:.3f} "
                f"kernel_b_ctas={num_b_ctas} threads_per_cta={args.threads_per_cta} "
                f"payload_copy_method={args.payload_copy_method} "
                f"tma_tile={fmt_bytes(args.tma_tile_bytes)} "
                f"payload_buffer={fmt_bytes(total_payload_bytes)} "
                f"key_buffer={fmt_bytes(total_key_bytes)} "
                f"iters={args.iters} elapsed={elapsed:.6f}s "
                f"payload_reorder_bw={local_bw:.2f} GiB/s "
                f"check={'OK' if ok else 'FAIL:' + failure}",
                flush=True,
            )
        barrier()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
