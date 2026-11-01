#!/usr/bin/env python3
"""Single-GPU CuTe DSL benchmark for the RDMA two-node routing metadata workload."""

import argparse
import math
import statistics
import time


def iteration_seed(base, iteration):
    return (base + iteration) & ((1 << 64) - 1)


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
    def check_scan(plan):
        # Check scratch too: later scatters do not consume expert-bin prefixes,
        # so checking only final metadata would miss errors in those columns.
        counts = plan.buffers[2].cpu().to(torch.int64)
        inclusive = counts.cumsum(dim=0)
        if not torch.equal(plan.buffers[3].cpu().to(torch.int64), inclusive - counts):
            raise AssertionError("parallel input chunk-prefix scan mismatch")
        if not torch.equal(plan.input_totals.cpu().to(torch.int64), inclusive[-1]):
            raise AssertionError("parallel input bin-total mismatch")

    cases = [(n, 0, "affine") for n in (1, 31, 32, 33, 255, 256, 257)]
    cases += [(257, gpu, pattern) for gpu in (0, 7)
              for pattern in ("node0", "node1", "spread", "random")]
    # Scan boundaries: 31, 32, 33, 34, and 65 chunks. Include partial chunks
    # and multiple carry steps; the main default benchmark covers 80 chunks.
    cases += [(n, 3, "random") for n in (7936, 8192, 8193, 8449, 16385)]
    for tokens, local_gpu, pattern in cases:
        r = torch.tensor(make_routes(tokens, pattern=pattern), dtype=torch.int32, device=device)
        plan = MetadataPlan(r, local_gpu)
        for b in plan.buffers:
            b.fill_(127)  # Catch missing initialization or reads of scratch padding.
        plan.launch()
        torch.cuda.synchronize(device)
        plan.check()
        check_scan(plan)
        # Same plan/addresses, different routes: test scratch reset and graph replay.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.launch()
        r.copy_(torch.tensor(make_routes(tokens, seed=4321, pattern="random"),
                             dtype=torch.int32, device=device))
        graph.replay()
        torch.cuda.synchronize(device)
        plan.check()
        check_scan(plan)
        print(f"PASS T={tokens}, local_gpu={local_gpu}, pattern={pattern}; changed-R graph replay")


def check_staging(torch, staging, check_tail=False):
    metadata = staging.metadata
    offsets = metadata.node_offsets.cpu().tolist()
    begin, end = offsets[staging.node_rank:staging.node_rank + 2]
    indices = metadata.x3[begin:end].long()
    expected = staging.source.index_select(0, indices).view(torch.uint8)
    actual = staging.destination[:end - begin].view(torch.uint8)
    if not torch.equal(actual, expected):
        raise AssertionError("CuTe local staging payload mismatch")
    if check_tail and not torch.all(staging.destination[end - begin:].view(torch.uint8) == 0xA5):
        raise AssertionError("CuTe local staging modified the unused tail")


def run_staging_check_suite(torch, device):
    from metadata_cute import MetadataPlan, LocalStagingPlan

    for tokens, width in ((1, 1), (33, 7), (257, 4096)):
        r = torch.tensor(make_routes(tokens), dtype=torch.int32, device=device)
        metadata = MetadataPlan(r, 7)
        # Arbitrary bits exercise bit-preserving copies, including BF16 NaNs.
        source = torch.randint(0, 256, (tokens, width * 2),
                               dtype=torch.uint8, device=device).view(torch.bfloat16)
        for node in (0, 1):
            destination = torch.empty_like(source)
            staging = LocalStagingPlan(metadata, source, destination, node)
            destination.view(torch.uint8).fill_(0xA5)
            metadata.launch()
            staging.launch()
            torch.cuda.synchronize(device)
            check_staging(torch, staging, check_tail=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                metadata.launch()
                staging.launch()
            # Full/empty slices and changed lengths on the same captured graph.
            for pattern in ("node0", "node1", "random"):
                r.copy_(torch.tensor(make_routes(tokens, pattern=pattern),
                                     dtype=torch.int32, device=device))
                source.view(torch.uint8).bitwise_xor_(0x5A)
                destination.view(torch.uint8).fill_(0xA5)
                graph.replay()
                torch.cuda.synchronize(device)
                metadata.check()
                check_staging(torch, staging, check_tail=True)
            print(f"PASS staging T={tokens}, D={width}, node={node}; changed-R/payload graph replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--local-gpu-id", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20480)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    parser.add_argument("--workload", choices=("metadata", "staging", "combined"),
                        default="metadata", help="time metadata, local gather, or both")
    parser.add_argument("--node-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--token-dimension", type=int, default=4096)
    parser.add_argument("--profile", action="store_true",
                        help="capture warm R generation + metadata calls via cudaProfilerStart/Stop")
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
    if args.token_dimension < 1:
        parser.error("token-dimension must be positive")

    import torch
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from metadata_cute import MetadataPlan, RouteGenerator, LocalStagingPlan

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
    staging = None
    if args.workload != "metadata":
        payload = torch.randint(0, 256, (args.tokens, args.token_dimension * 2),
                                dtype=torch.uint8, device=device).view(torch.bfloat16)
        staging = LocalStagingPlan(plan, payload, torch.empty_like(payload), args.node_rank)

    def launch_workload():
        if args.workload != "staging":
            plan.launch()
        if staging is not None:
            staging.launch()

    r_argument = from_dlpack(r)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    generator = cute.compile(RouteGenerator(), r_argument, cutlass.Uint64(args.seed), stream)

    def update_r(seed):
        generator(r_argument, cutlass.Uint64(seed), stream)

    def prepare_input(seed):
        update_r(seed)
        if args.workload == "staging":
            plan.launch()  # Fresh x3/offsets, outside gather-only timing/capture graph.

    def check_outputs():
        plan.check()
        if staging is not None:
            check_staging(torch, staging)

    # Check the GPU generator against the independent CPU implementation,
    # including uint64 seed wraparound, before relying on it for measurements.
    for seed in (args.seed, iteration_seed(args.seed, 1), (1 << 64) - 1, 0):
        update_r(seed)
        expected_r = torch.tensor(make_routes(args.tokens, seed), dtype=torch.int32)
        if not torch.equal(r.cpu(), expected_r):
            raise AssertionError(f"CuTe route generator mismatch for seed={seed}")
    torch.cuda.synchronize(device)
    print(f"Validation/allocation/JIT setup: {time.perf_counter() - start:.3f} s (excluded)")
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def sample(launch, seed):
        # Same stream orders generator -> start event -> metadata -> end event.
        # R keeps its address/shape; generation is excluded from elapsed time.
        prepare_input(seed)
        begin.record()
        launch()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) * 1000.0

    first = sample(launch_workload, args.seed)
    check_outputs()
    print(f"CPU reference: PASS; workload={args.workload}; first execution after JIT: {first:.3f} us")

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

    for i in range(args.warmup):
        prepare_input(iteration_seed(args.seed, i + 1))
        launch_workload()
    torch.cuda.synchronize(device)
    graph = None
    if args.mode in ("graph", "both"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_workload()
        prepare_input(iteration_seed(args.seed, args.warmup + 1))
        graph.replay()
        torch.cuda.synchronize(device)
        check_outputs()

    launchers = []
    if args.mode in ("eager", "both"):
        launchers.append(("eager", launch_workload))
    if graph is not None:
        launchers.append(("graph", graph.replay))
    for mode, launch in launchers:
        # Both modes see identical seed sequences for a paired comparison.
        for i in range(args.warmup):
            prepare_input(iteration_seed(args.seed, i + 1))
            launch()
        torch.cuda.synchronize(device)
        if args.profile:
            torch.cuda.profiler.start()
            for i in range(args.iters):
                prepare_input(iteration_seed(args.seed, args.warmup + 1 + i))
                launch()
            torch.cuda.synchronize(device)
            torch.cuda.profiler.stop()
            print(f"Captured {args.iters} {mode} {args.workload} calls; exclude generate_routes "
                  "and, for staging-only, the preceding seven metadata kernels from timing")
        else:
            times = sorted(sample(launch, iteration_seed(args.seed, args.warmup + 1 + i))
                           for i in range(args.iters))
            print(f"{mode} {args.workload}: min={times[0]:.3f} median={statistics.median(times):.3f} "
                  f"mean={statistics.mean(times):.3f} p95={times[int(.95*(len(times)-1))]:.3f} "
                  f"max={times[-1]:.3f} us ({args.iters} samples)")
        check_outputs()
    print("Fresh GPU-generated R before every warmup/measured invocation; fixed shape/address. "
          "Eager and graph modes use the same seed sequence. "
          "Timings exclude JIT, allocation, input generation, and CPU checks. "
          "Event spans include submission gaps and event-boundary overhead. "
          "Metadata/combined/staging graphs contain seven/eight/one kernels respectively.")
    if args.check_suite:
        run_check_suite(torch, MetadataPlan, device)
        run_staging_check_suite(torch, device)


if __name__ == "__main__":
    main()
