#!/usr/bin/env python3
"""
Benchmark GPU initialization, CPU 7-bit key sorting, and GPU payload reordering.

Kernel A initializes payload and one-byte key buffers in GPU global memory. Only
the lower seven bits of every key byte are populated and used for sorting.

The CPU hybrid path then:
  1. copies the key buffer from GPU to pinned CPU memory
  2. builds a src-to-dst reordered payload index on CPU with a parallel
     counting sort over the 128 possible key values
  3. copies that index buffer back to GPU
  4. launches kernel B to physically copy payload records into a new GPU buffer

Example:
  torchrun --standalone --nproc_per_node=8 \
    Global_Mem_Reordering_CPU_Hybrid/global_mem_reordering_cpu_hybrid_test.py \
    --num-elements 10000 --payload-size 8K --bit-probability 0.4 \
    --num-sms 0 --payload-copy-method lsu --iters 100 --warmup 10 --check
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load


KEY_VALUES = 128
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
        name="global_mem_reordering_cpu_hybrid_ext",
        sources=[
            str(here / "global_mem_reordering_cpu_hybrid_ext.cpp"),
            str(here / "global_mem_reordering_cpu_hybrid_kernels.cu"),
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
        args.bit_probability,
        args.seed,
        init_ctas,
        args.threads_per_cta,
        device,
    )


def launch_kernel_b(ext,
                    payload: torch.Tensor,
                    reordered_payload: torch.Tensor,
                    reordered_indices: torch.Tensor,
                    args: argparse.Namespace,
                    num_b_ctas: int,
                    device: int) -> None:
    ext.launch_kernel_b(
        payload,
        reordered_payload,
        reordered_indices,
        args.num_elements,
        args.payload_size,
        num_b_ctas,
        args.threads_per_cta,
        COPY_METHOD_TO_ID[args.payload_copy_method],
        args.tma_tile_bytes,
        device,
    )


def run_hybrid_once(ext,
                    payload: torch.Tensor,
                    keys: torch.Tensor,
                    reordered_payload: torch.Tensor,
                    host_keys: torch.Tensor,
                    host_reordered_indices: torch.Tensor,
                    reordered_indices: torch.Tensor,
                    args: argparse.Namespace,
                    num_b_ctas: int,
                    device: int) -> Dict[str, float]:
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    host_keys.copy_(keys, non_blocking=True)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    timings["key_d2h"] = t1 - t0

    ext.build_reordered_indices_cpu(
        host_keys,
        host_reordered_indices,
        args.num_elements,
        args.cpu_threads,
    )
    t2 = time.perf_counter()
    timings["cpu_sort"] = t2 - t1

    reordered_indices.copy_(host_reordered_indices, non_blocking=True)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    timings["index_h2d"] = t3 - t2

    launch_kernel_b(
        ext, payload, reordered_payload, reordered_indices, args, num_b_ctas, device
    )
    torch.cuda.synchronize(device)
    t4 = time.perf_counter()
    timings["kernel_b"] = t4 - t3
    timings["end_to_end"] = t4 - t0
    return timings


def check_outputs(payload: torch.Tensor,
                  keys: torch.Tensor,
                  reordered_payload: torch.Tensor,
                  host_keys: torch.Tensor,
                  host_reordered_indices: torch.Tensor,
                  reordered_indices: torch.Tensor,
                  num_elements: int,
                  payload_size: int) -> Tuple[bool, str]:
    if torch.any(host_keys & 0x80):
        bad = int(torch.nonzero(host_keys & 0x80, as_tuple=False)[0].item())
        return False, f"key {bad} has ignored high bit set"

    dst_cpu = host_reordered_indices.to(torch.int64)
    sorted_dst, _ = torch.sort(dst_cpu)
    expected = torch.arange(num_elements, dtype=torch.int64)
    if not torch.equal(sorted_dst, expected):
        return False, "reordered payload indices are not a permutation"

    ordered_keys = torch.empty(num_elements, dtype=torch.uint8)
    ordered_keys[dst_cpu] = host_keys & 0x7f
    if torch.any(ordered_keys[:-1] < ordered_keys[1:]):
        bad = int(torch.nonzero(ordered_keys[:-1] < ordered_keys[1:], as_tuple=False)[0].item())
        return False, (
            f"keys not descending at reordered positions {bad}/{bad + 1}: "
            f"{int(ordered_keys[bad])} < {int(ordered_keys[bad + 1])}"
        )

    samples = min(64, num_elements)
    if samples == 1:
        sample_src_cpu = torch.zeros(1, dtype=torch.int64)
    else:
        sample_src_cpu = (
            torch.arange(samples, dtype=torch.int64)
            * (num_elements - 1)
            // (samples - 1)
        )
    sample_src_gpu = sample_src_cpu.to(device=payload.device)
    sample_dst_gpu = reordered_indices[sample_src_gpu].to(torch.int64)

    byte_offsets = sorted({0, payload_size // 2, payload_size - 1})
    payload_2d = payload.view(num_elements, payload_size)
    reordered_2d = reordered_payload.view(num_elements, payload_size)
    for offset in byte_offsets:
        expected_bytes = payload_2d[sample_src_gpu, offset]
        actual_bytes = reordered_2d[sample_dst_gpu, offset]
        mismatch = torch.nonzero(actual_bytes != expected_bytes, as_tuple=False)
        if mismatch.numel() > 0:
            sample = int(mismatch[0].item())
            return False, (
                f"payload mismatch for source element {int(sample_src_cpu[sample])}, "
                f"payload byte {offset}"
            )
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-elements", "--num_elements", type=int, default=10_000)
    parser.add_argument("--payload-size", "--payload_size", type=parse_nbytes,
                        default=parse_nbytes("8K"))
    parser.add_argument("--bit-probability", "--bit_probability", type=float,
                        default=0.4,
                        help="probability that each of the lower 7 key bits is set to 1")
    parser.add_argument("--num-sms", "--num_sms", type=int, default=0,
                        help="CTAs/SMs used by kernel B; 0 means all visible SMs")
    parser.add_argument("--threads-per-cta", "--threads_per_cta", type=int,
                        default=256)
    parser.add_argument("--payload-copy-method", "--payload_copy_method",
                        choices=("tma", "lsu"), default="lsu")
    parser.add_argument("--tma-tile-bytes", "--tma_tile_bytes", type=parse_nbytes,
                        default=parse_nbytes("8K"))
    parser.add_argument("--cpu-threads", "--cpu_threads", type=int, default=0,
                        help="CPU sort worker threads; 0 keeps PyTorch's current thread count")
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
    if not (0.0 <= args.bit_probability <= 1.0):
        raise ValueError("--bit-probability must be in [0, 1]")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.threads_per_cta <= 0 or args.threads_per_cta > 1024:
        raise ValueError("--threads-per-cta must be in 1..1024")
    if args.cpu_threads < 0:
        raise ValueError("--cpu-threads must be non-negative")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

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
    total_key_bytes = args.num_elements
    total_index_bytes = args.num_elements * 4

    ext = get_extension()

    payload = torch.empty(total_payload_bytes, dtype=torch.uint8, device=device)
    keys = torch.empty(total_key_bytes, dtype=torch.uint8, device=device)
    reordered_payload = torch.empty_like(payload)
    reordered_indices = torch.empty(args.num_elements, dtype=torch.int32, device=device)

    host_keys = torch.empty(args.num_elements, dtype=torch.uint8, pin_memory=True)
    host_reordered_indices = torch.empty(args.num_elements, dtype=torch.int32, pin_memory=True)

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
        run_hybrid_once(
            ext, payload, keys, reordered_payload, host_keys,
            host_reordered_indices, reordered_indices, args, num_b_ctas, device
        )
    barrier()

    totals = {
        "key_d2h": 0.0,
        "cpu_sort": 0.0,
        "index_h2d": 0.0,
        "kernel_b": 0.0,
        "end_to_end": 0.0,
    }
    for _ in range(args.iters):
        timings = run_hybrid_once(
            ext, payload, keys, reordered_payload, host_keys,
            host_reordered_indices, reordered_indices, args, num_b_ctas, device
        )
        for name, value in timings.items():
            totals[name] += value
    barrier()

    avg = {name: value / args.iters for name, value in totals.items()}
    payload_gib = total_payload_bytes / (1024**3)
    kernel_b_bw = payload_gib / avg["kernel_b"] if avg["kernel_b"] > 0 else 0.0
    end_to_end_bw = payload_gib / avg["end_to_end"] if avg["end_to_end"] > 0 else 0.0

    ok = True
    failure = ""
    if args.check:
        ok, failure = check_outputs(
            payload, keys, reordered_payload, host_keys, host_reordered_indices,
            reordered_indices, args.num_elements, args.payload_size
        )

    for output_rank in range(world_size):
        barrier()
        if rank == output_rank:
            print(
                f"rank={rank:02d} device=cuda:{device} "
                f"num_elements={args.num_elements} "
                f"payload_size={fmt_bytes(args.payload_size)} "
                f"key_size=1 B key_bits=7 "
                f"bit_probability={args.bit_probability:.3f} "
                f"kernel_b_ctas={num_b_ctas} threads_per_cta={args.threads_per_cta} "
                f"payload_copy_method={args.payload_copy_method} "
                f"tma_tile={fmt_bytes(args.tma_tile_bytes)} "
                f"cpu_threads={torch.get_num_threads()} "
                f"payload_buffer={fmt_bytes(total_payload_bytes)} "
                f"key_buffer={fmt_bytes(total_key_bytes)} "
                f"index_buffer={fmt_bytes(total_index_bytes)} "
                f"iters={args.iters} "
                f"avg_key_d2h={avg['key_d2h'] * 1e3:.3f}ms "
                f"avg_cpu_sort={avg['cpu_sort'] * 1e3:.3f}ms "
                f"avg_index_h2d={avg['index_h2d'] * 1e3:.3f}ms "
                f"avg_kernel_b={avg['kernel_b'] * 1e3:.3f}ms "
                f"avg_end_to_end={avg['end_to_end'] * 1e3:.3f}ms "
                f"kernel_b_payload_bw={kernel_b_bw:.2f} GiB/s "
                f"end_to_end_payload_bw={end_to_end_bw:.2f} GiB/s "
                f"check={'OK' if ok else 'FAIL:' + failure}",
                flush=True,
            )
        barrier()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
