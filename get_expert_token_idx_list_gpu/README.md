# GPU expert token-index lists

This directory builds stable, compact per-expert lists directly from an
`int32` CUDA routing matrix `R[T, topK]`.  Experts are assigned contiguously:
expert `e` belongs to GPU `e // n`.

The result is a ragged array represented by:

- `expert_token_indices`: `T * topK` packed `int32` values.
- `expert_offsets`: `num_experts + 1` offsets.  Expert `e` owns
  `expert_token_indices[expert_offsets[e]:expert_offsets[e + 1]]`.

Each value is the token's position in the ordered sequence of tokens routed to
that expert's GPU.  A token routed to several experts on one GPU consumes only
one position on that GPU.  Lists preserve input token order.

## Run

### Standalone reproduction of the RDMA two-node metadata workload

This benchmark needs **one CUDA GPU**, a CUDA toolkit, and a host C++ compiler.
It does not need PyTorch, RDMA, a NIC, or a second node. Run it on the remote
Hopper machine, not the local Mac:

```bash
bash get_expert_token_idx_list_gpu/build_rdma_metadata_benchmark.sh
get_expert_token_idx_list_gpu/build/benchmark_rdma_metadata
```

Defaults match `RDMA_CPU_Proxy/scripts/run_node{0,1}_torchrun.sh`: 20,480 tokens,
128 experts, top-k 8, two logical nodes, eight logical GPUs per node, eight
experts per GPU, and seed 1234. `--device 0` selects the physical CUDA device;
`--local-gpu-id 0` independently selects the logical source GPU (0 through 7).
The 4,096-element token payload and 32-token RDMA chunks do not enter these
metadata kernels; their internal chunk size is 256 tokens.

The executable links the existing CUDA source in standalone mode and calls
the **same raw six-kernel API on the default stream** as `RouterRouting`.
It generates R once, allocates scratch once, and validates all meaningful
outputs against an independent CPU reference before and after benchmarking.
No production kernel or launch wrapper is modified. Build flags are `-O3`
and `-lineinfo` with `sm_90`; use the same optimization/architecture settings
in the proxy build when comparing results. `NVCC`, `CUDA_ARCH`, and
`RDMA_METADATA_BUILD_DIR` can override the compiler, target, and output path.

The default run reports a first-call measurement separately, then min,
median, mean, p95, and max over 200 samples after 20 warmup calls. Each sample
uses CUDA events around one complete six-kernel call and synchronizes after
the ending event. It excludes R generation, allocations, D2H copies, and
validation. The event span includes submission gaps and boundary event
overhead, so it approximates, but is not identical to, Nsight's interval
from the first kernel's start to the last kernel's end. The first call is
not a full cold-process measurement: CUDA initialization and R generation
have already run. Warm repetitions reuse cached data and do not reproduce
the full proxy's CPU/GPU contention or initialization environment.

For individual kernel durations and exact timeline gaps, capture the
uninstrumented six-kernel chain with Nsight Systems:

```bash
nsys profile --trace=cuda --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o rdma_metadata \
  get_expert_token_idx_list_gpu/build/benchmark_rdma_metadata \
  --profile --warmup 20 --iters 20
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum rdma_metadata.nsys-rep
```

The capture excludes generation, validation, and warmup. There are no events
between kernels or calls in profile mode; a final synchronization drains
the captured work. Each call contains the following six kernels, in order:

1. `count_node_input_chunks_kernel`
2. `scan_node_input_counts_kernel`
3. `build_node_token_indices_kernel<8>`
4. `count_reordered_node_chunks_kernel`
5. `scan_reordered_node_counts_kernel`
6. `scatter_reordered_chunks_kernel<8>`

Use `cuda_gpu_kern_sum` for per-kernel averages. In the GUI, for each group
of six, measure `last.end - first.start`; subtract the sum of the six kernel
durations to obtain the five internal gaps. Do not include gaps between
calls. `--profile --iters 1` isolates one warm chain. Profiling can perturb
short timings; compare with an unprofiled event run. This benchmark measures
local metadata generation, not RDMA/NVLink, metadata exchange, or GEMM.

### CuTe DSL metadata implementation (single GPU)

`metadata_cute.py` implements the metadata pipeline in Python CuTe DSL. It
does not modify the CUDA C++ kernels or integrate with the RDMA proxy yet.
Use the remote environment that runs the existing CuTe GEMM/combine kernels:
CUDA-enabled PyTorch, `cuda-python` (`cuda.bindings.driver`), and NVIDIA
`nvidia-cutlass-dsl`. The implementation uses `cute.arch.atomic_or`,
`atomic_add`, `popc`, shuffle operations, and `cutlass.utils.SmemAllocator`;
these APIs are documented in the
[CUTLASS 4.8 CuTe arch reference](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/cute_dsl_api/cute_arch.html).
No QuACK checkout is required. GPU JIT compilation and execution must be
validated on the remote machine; local syntax/host tests are not GPU validation.

```bash
python get_expert_token_idx_list_gpu/benchmark_metadata_cute.py --device 0
```

Defaults match the standalone benchmark: `T=20480`, 128 experts, top-k eight,
two logical nodes, eight logical GPUs/node, seed 1234. `--local-gpu-id` selects
mask ordering independently of `--device`. This first implementation specializes
to that topology; `--tokens`, `--seed`, and `--local-gpu-id` are configurable.
Inputs must contain eight **distinct** expert IDs in `[0,128)` per token.
Before every warmup and measured invocation, a CuTe GPU generator overwrites
the same R allocation with fresh routes. Its runtime seed advances by one
(with uint64 wraparound), without recompilation or recapture. It reproduces
the existing CUDA generator's affine-permutation algorithm; a CPU generator
checks it during setup, including seed-wrap cases. Eager and graph modes use
identical seed sequences for comparison. The initial call uses `--seed`;
warmups use `seed+1..seed+warmup`, and measured calls start at
`seed+warmup+1`. This preserves shape and device address while changing data.

The six kernels are:

| CuTe kernel | Work |
|---|---|
| `count_input` | Cache two masks/token; count experts; build per-mask warp bitmaps |
| `scan_input` | Chunk prefixes, expert offsets, descending mask/node offsets |
| `build_node_indices` | Stable x3/x4 scatter using bitmap population-count ranks |
| `count_reordered` | Build expert/GPU bitmaps and counts in reordered x3 order |
| `scan_reordered` | Parallel prefix over chunks, one warp per expert/GPU bin |
| `scatter_experts` | Stable expert lists with direct/compact input indexing |

This is a six-kernel implementation, not a fused kernel or a line-by-line
translation of CUB sorting. Bitmap ranks remove the local radix sorts at the
cost of extra scratch traffic. The first scan still uses one CTA, but its
expert and mask offset scans are parallel block scans. The second scan uses
18 CTAs. There is no assumption that CuTe alone makes the pipeline faster;
measure on H20 before drawing performance conclusions.

By default the benchmark reports **eager** and **graph** timings separately:

- Eager: a compiled CuTe host entry launches all six kernels. CPU submission
  overhead can differ from the standalone C++ benchmark.
- Graph: replay the same six kernels through one CUDA Graph launch. Capture
  and JIT costs are excluded. This isolates repeated execution with reduced
  host submission overhead; graph replay does not fuse kernels.

Both modes warm up and validate every meaningful output against the existing
CPU reference after their final invocation; the first invocation is also
checked. On the same CUDA stream, each timed sample enqueues input generation,
then the start event, metadata processing, and the end event. Generation is
excluded from elapsed time but can affect cache state, as a preceding router
would. The metadata-only graph reads fresh R on every replay. CUDA-event spans
include internal gaps and boundary event
overhead. Compare kernel durations in nsys to distinguish implementation cost
from host submission cost; eager-versus-graph timing is not a direct
language-performance comparison. First execution is reported **after JIT**,
so it is not equivalent to the standalone executable's first-use setup.

For additional checks (extra specializations can take time to compile):

```bash
python get_expert_token_idx_list_gpu/benchmark_metadata_cute.py \
  --device 0 --check-suite --compare-cuda
```

`--compare-cuda` additionally needs nvcc and the PyTorch extension build
dependencies used by `run.py`. It compares the synthetic generator and all
five metadata outputs with the existing CUDA implementation on the same R.
`--check-suite` checks warp/chunk tails, empty destination nodes, hot experts,
different mask rotations, non-affine random routing, and graph replay after
changing R in place. These checks are outside all timed/captured regions.

```bash
nsys profile --trace=cuda --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o metadata_cute_eager \
  python get_expert_token_idx_list_gpu/benchmark_metadata_cute.py \
  --device 0 --profile --mode eager --iters 20
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum metadata_cute_eager.nsys-rep
```

Choose `--mode graph` and a different output name to profile graph replay.
The CuTe profiling capture includes one `generate_routes` kernel before each
six-kernel metadata chain. Exclude that generator and the gap before the first
metadata kernel when computing metadata-only durations. The event benchmark
already excludes generation. Compilation, correctness checks, and warmup are
outside the capture.
For library use, construct `MetadataPlan(r, local_gpu_id)`, then call
`plan.launch(stream)` on an ordered stream. Outputs are `plan.x3`, `plan.x4`,
`plan.node_offsets`, `plan.expert_offsets`, and `plan.expert_token_indices`.
Allocate/compile once; changing R in place is supported if its contract is
preserved. One plan's scratch/output buffers cannot be used concurrently.
Caller-provided streams must wait for any producer that updates R.

Host-only contract tests (no CUDA/PyTorch dependency):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s get_expert_token_idx_list_gpu -p 'test_metadata_cute.py'
```

### Original Python/CUDA C++ driver

PyTorch with CUDA and a CUDA toolkit are required.  The extension is compiled
for Hopper (`sm_90`) by default.

```bash
python run.py --tokens 16384 --top-k 8 --experts 256 \
  --experts-per-gpu 32 --check --warmup 20 --iters 200
```

## Node mask ordering

Pass `--node-mask-sort` to enable the node-aware mode:

```bash
python run.py --tokens 16384 --top-k 8 --experts 256 \
  --experts-per-gpu 16 --node-mask-sort --gpus-per-node 4 \
  --local-gpu-id 0 --check
```

The node size is configured with `--gpus-per-node N`, where `N` can be any
integer from 2 through 8 and defaults to 8. With 16 experts per GPU and
`--gpus-per-node 4`, experts 0--63 belong to node 0, experts 64--127 to node 1,
and so on. For each token and destination node, an N-bit mask records the GPUs
in that node selected by the token (`bit g` represents node-local GPU `g`).
Tokens sent to each node are stably sorted by the mask's unsigned numeric value
in descending order; equal masks retain input-token order. The number of mask
bins and all scratch-buffer sizes are derived from `N`.

`--local-gpu-id x` selects the node-local GPU on which the algorithm runs and
controls bit significance. Local GPU `(x + 1) % N` maps to the most significant
meaningful bit, `(x + 2) % N` maps to the next bit, and so on; local GPU `x`
maps to the least significant bit. Equivalently, local GPU `g` maps to bit
`(x - g) % N`. The default local GPU ID is 0, and valid values are 0 through
`N - 1`.

The additional output is another packed ragged array:

- `node_token_indices` is `x3`, containing original token IDs in sender-buffer
  order.
- `node_token_masks` is `x4`, a `torch.uint8` array aligned one-to-one with
  `x3`. Each value is the unsigned GPU mask used to order that token. A
  receiver can inspect it to determine which node-local GPUs should receive
  the corresponding arriving token.
- `node_offsets` has `num_nodes + 1` entries.  Node `i` owns
  the same slice in both `node_token_indices` and `node_token_masks`:
  `[node_offsets[i]:node_offsets[i + 1]]`.

Only the low `N` bits of each `x4` byte are meaningful, where
`N = --gpus-per-node`; the upper `8 - N` bits are zero. The bit-to-GPU mapping
is controlled by `--local-gpu-id` as described above.

The temporary `x1` and `x2` arrays are never materialized.  The CUDA code uses
per-chunk `(node, mask)` histograms and a stable counting-sort scatter to
write aligned `x3` and `x4` entries directly. It then computes the per-GPU
filtered ranks in `x3`
order and rebuilds `expert_token_indices`. For destinations whose node-local
GPU index differs from `local_gpu_id`, each value matches the token's compact
arrival position after NVLink forwarding. For the destination whose local GPU
index equals `local_gpu_id`, each value is instead the token's position in the
complete destination-node `x3` slice, allowing that GPU to consume its RDMA
receive buffer directly without a compaction copy. Output and scratch
allocation remain outside the measured interval.

The Python driver allocates `R` in GPU HBM, invokes the CUDA input generator,
runs one correctness check against a CPU implementation, and reports average
CUDA-event latency.  Output and scratch buffers are allocated once and reused,
so allocation time is excluded.

In node-mask mode, `latency` is the end-to-end algorithm latency and
`x3_latency` measures only the first three phases needed to materialize the
aligned `node_token_indices`/`node_token_masks` (`x3`/`x4`) outputs:
input-chunk counting, prefix/offset scanning, and the stable node/mask scatter.
Both averages use `--iters`; the full path and the standalone x3/x4 path each
use `--warmup` before their respective measurements.

Pass `--x3-x4-to-cpu` with `--node-mask-sort` to copy the valid x3 and x4
slices into the pinned CPU tensors `host_node_token_indices` and
`host_node_token_masks`. The `num_nodes + 1` offsets are also copied to the
pinned CPU tensor `host_node_offsets`, allowing node `i` to use the slice
`[host_node_offsets[i]:host_node_offsets[i + 1]]` in both packed arrays. In
this mode, the copies are enqueued directly after the x3/x4 kernels, and the
ending CUDA event is recorded after all three copies. Consequently,
`x3_latency` includes x3/x4 generation and all D2H transfers; event
synchronization guarantees the CPU buffers are ready. The terminal reports
`x3_x4_output=cpu node_offsets_output=cpu` instead of the corresponding `gpu`
values.

## Kernel structure

The fast path uses three stream-ordered kernels:

1. Count expert routes and unique destination-GPU tokens per 256-token chunk.
2. Scan the small chunk-count table to produce global prefixes and expert
   offsets.
3. Compute GPU-local token positions and use a stable block radix sort to pack
   expert lists in input-token order.

`topK` values 1, 2, 4, 8, and 16 use the block-radix fast path.  Other values
use a stable general fallback.  The implementation supports at most 1024
experts, and `n` must divide `K`.
