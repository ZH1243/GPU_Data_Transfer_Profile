#!/usr/bin/env python3
"""Single-GPU CuTe DSL benchmark for the RDMA two-node routing metadata workload."""

import argparse
import math
import statistics
import time


def make_routes(tokens, seed=1234, pattern="affine"):
    """CPU input setup, bit-identical to generate_r_kernel for 128 experts.

    Kept outside timing and independent of nvcc/PyTorch extensions. Additional
    patterns exercise sparse nodes, hot experts, and non-affine routing.
    """
    import random

    mask64 = (1 << 64) - 1

    def splitmix64(value):
        value = (value + 0x9E3779B97F4A7C15) & mask64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask64
        return value ^ (value >> 31)

    rng = random.Random(seed)
    result = []
    for token in range(tokens):
        if pattern == "affine":
            first = splitmix64(seed ^ token)
            second = splitmix64(first)
            start = first % 128
            stride = 1 + second % 127
            while math.gcd(stride, 128) != 1:
                stride += 1
                if stride == 128:
                    stride = 1
            row = [(start + route * stride) % 128 for route in range(8)]
        elif pattern == "node0":
            row = list(range(8))
        elif pattern == "node1":
            row = list(range(120, 128))
        elif pattern == "spread":
            row = [((token + route * 2) % 16) * 8 + token % 8 for route in range(8)]
        elif pattern == "random":
            row = rng.sample(range(128), 8)
        else:
            raise ValueError(f"unknown pattern: {pattern}")
        result.append(row)
    return result


def run_check_suite(torch, MetadataPlan, device):
    cases = [(n, 0, "affine") for n in (1, 31, 32, 33, 255, 256, 257)]
    cases += [(257, gpu, pattern) for gpu in (0, 7)
              for pattern in ("node0", "node1", "spread", "random")]
    # More than 32 chunks, with both a chunk tail and a warp-scan tail.
    cases.append((8449, 3, "random"))
    for tokens, local_gpu, pattern in cases:
        r = torch.tensor(make_routes(tokens, pattern=pattern), dtype=torch.int32, device=device)
        plan = MetadataPlan(r, local_gpu)
        for b in plan.buffers:
            b.fill_(127)  # Catch missing initialization or reads of scratch padding.
        plan.launch()
        torch.cuda.synchronize(device)
        plan.check()
        # Same plan/addresses, different routes: test scratch reset and graph replay.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.launch()
        r.copy_(torch.tensor(make_routes(tokens, seed=4321, pattern="random"),
                             dtype=torch.int32, device=device))
        graph.replay()
        torch.cuda.synchronize(device)
        plan.check()
        print(f"PASS T={tokens}, local_gpu={local_gpu}, pattern={pattern}; changed-R graph replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--local-gpu-id", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20480)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    parser.add_argument("--profile", action="store_true",
                        help="capture only warm launches via cudaProfilerStart/Stop")
    parser.add_argument("--compare-cuda", action="store_true",
                        help="also build the existing CUDA extension and compare outputs on identical R")
    parser.add_argument("--check-suite", action="store_true",
                        help="add GPU edge-case tests (compiles multiple specializations)")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 10_000_000 or args.warmup < 1 or args.iters < 1:
        parser.error("tokens must be 1..10000000; warmup/iters must be positive")
    if args.device < 0 or not 0 <= args.local_gpu_id < 8 or not 0 <= args.seed < 2**64:
        parser.error("require device >= 0, local-gpu-id in 0..7, seed in 0..2^64-1")
    if args.profile and args.mode == "both":
        parser.error("choose --mode eager or --mode graph when using --profile")

    import torch
    from metadata_cute import MetadataPlan

    if not torch.cuda.is_available():
        raise RuntimeError("Run this benchmark on a CUDA GPU, not the local Mac")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}, SMs={properties.multi_processor_count}, "
          f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
    print(f"tokens={args.tokens}, top_k=8, experts=128, nodes=2, GPUs/node=8, "
          f"local_gpu_id={args.local_gpu_id}, seed={args.seed}")
    r = torch.tensor(make_routes(args.tokens, args.seed), dtype=torch.int32, device=device)
    start = time.perf_counter()
    plan = MetadataPlan(r, args.local_gpu_id)
    torch.cuda.synchronize(device)
    print(f"Validation/allocation/JIT setup: {time.perf_counter() - start:.3f} s (excluded)")
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def sample(launch):
        begin.record()
        launch()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) * 1000.0

    first = sample(plan.launch)
    plan.check()
    print(f"CPU reference: PASS; first execution after JIT: {first:.3f} us")

    if args.compare_cuda:
        from run import allocate_node_mask_outputs, load_extension

        extension = load_extension()
        generated = torch.empty_like(r)
        extension.generate_r_(generated, 128, args.seed)
        if not torch.equal(generated, r):
            raise AssertionError("CPU synthetic generator differs from generate_r_kernel")
        outputs = allocate_node_mask_outputs(r, 128, 8, 8)
        extension.get_expert_token_idx_node_mask(r, 128, 8, 8, args.local_gpu_id,
                                                False, *outputs)
        expected = (plan.expert_token_indices, plan.expert_offsets,
                    plan.x3, plan.x4, plan.node_offsets)
        valid = int(plan.node_offsets[-1].item())
        for i, (actual, reference) in enumerate(zip(expected, outputs[:5])):
            if i in (2, 3):
                actual, reference = actual[:valid], reference[:valid]
            if not torch.equal(actual, reference):
                raise AssertionError(f"CUDA C++ comparison failed for output {i}")
        print("CUDA C++ generator and all five metadata outputs: PASS")

    for _ in range(args.warmup):
        plan.launch()
    torch.cuda.synchronize(device)
    graph = None
    if args.mode in ("graph", "both"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.launch()
        graph.replay()
        torch.cuda.synchronize(device)
        plan.check()

    launchers = []
    if args.mode in ("eager", "both"):
        launchers.append(("eager", plan.launch))
    if graph is not None:
        launchers.append(("graph", graph.replay))
    for mode, launch in launchers:
        for _ in range(args.warmup):
            launch()
        torch.cuda.synchronize(device)
        if args.profile:
            torch.cuda.profiler.start()
            for _ in range(args.iters):
                launch()
            torch.cuda.synchronize(device)
            torch.cuda.profiler.stop()
            print(f"Captured {args.iters} {mode} six-kernel calls; see nsys for durations/gaps")
        else:
            times = sorted(sample(launch) for _ in range(args.iters))
            print(f"{mode}: min={times[0]:.3f} median={statistics.median(times):.3f} "
                  f"mean={statistics.mean(times):.3f} p95={times[int(.95*(len(times)-1))]:.3f} "
                  f"max={times[-1]:.3f} us ({args.iters} samples)")
        plan.check()
    print("Timings exclude JIT, allocation, input generation, and CPU checks. "
          "Event spans include submission gaps and event-boundary overhead. "
          "Graph replay still executes six separate kernels.")
    if args.check_suite:
        run_check_suite(torch, MetadataPlan, device)


if __name__ == "__main__":
    main()
