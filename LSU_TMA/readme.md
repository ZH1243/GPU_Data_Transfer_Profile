# LSU/TMA NVLink Copy Tests

This folder contains two PyTorch/CUDA benchmarks for measuring GPU-to-GPU P2P copies over NVLink using SM work instead of CUDA copy-engine APIs. Both scripts build an inline CUDA extension with `torch.utils.cpp_extension.load_inline`, allocate peer tensors on visible CUDA devices, enable CUDA peer access, run warmup iterations, time the requested copy loop, and print per-rank plus aggregate bandwidth.

Both benchmarks support two SM copy methods:

- `lsu`: a normal CUDA kernel that copies with global loads and stores from SMs.
- `tma`: a Hopper TMA path that stages data from global memory to shared memory, then stores from shared memory to peer global memory with bulk async copy instructions.

TMA mode requires Hopper or newer GPUs, CUDA 12-era headers with `<cuda/ptx>`, 16-byte aligned pointers, and copy/tile sizes that are multiples of 16.

## Files

### `nvlink_lsu_tma_copy_test.py`

This is the one-source-to-one-destination benchmark. Each rank selects one source GPU and one destination GPU according to `--mode`, then copies `--copy-size * --copies-per-iter` bytes per timed iteration.

The copy kernel can execute on either side:

- `--executor src`: source-side push. The source GPU reads local source data and writes to the peer destination.
- `--executor dst`: destination-side pull. The destination GPU reads from the peer source and writes local destination data.

The output line reports the selected `cuda:src -> cuda:dst` pair, copy method, executor GPU, CTA count, bytes per iteration, elapsed time, local bandwidth, peer-access status, and optional check result.

Example:

```bash
torchrun --standalone --nproc_per_node=8 LSU_TMA/nvlink_lsu_tma_copy_test.py \
  --copy-size 1G \
  --iters 100 \
  --method lsu \
  --executor dst \
  --num-sms 0 \
  --check
```

### `nvlink_lsu_tma_round_robin_copy_test.py`

This is the one-source-to-many-destinations benchmark. Each rank owns source GPU `i`, splits its source byte range into `--round-robin-bytes` chunks, and sends consecutive chunks to all other GPUs in a rotating order.

For example, on 8 GPUs with `--round-robin-bytes 8K`, rank/GPU `i` sends chunk 0 to `i+1`, chunk 1 to `i+2`, and so on until it wraps around. Chunks are written at the same byte offsets in each destination buffer.

Only source-side execution is implemented in this script, so `--executor` must be `src`.

Example:

```bash
torchrun --standalone --nproc_per_node=8 LSU_TMA/nvlink_lsu_tma_round_robin_copy_test.py \
  --copy-size 1M \
  --round-robin-bytes 8K \
  --iters 100 \
  --method tma \
  --executor src \
  --num-sms 3 \
  --tma-tile-bytes 64K \
  --tma-inter-tile-bytes 8K \
  --persistent-kernel \
  --check
```

## Arguments

Size arguments are parsed by the scripts' `parse_nbytes()` helper. Plain numbers are bytes. Supported suffixes include `K`, `M`, `G`, `KiB`, `MiB`, `GiB`, `KB`, `MB`, and `GB`. `K/M/G` use powers of 1024.

### Shared Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--nbytes`, `--copy-size` | `1G` | Bytes per logical copy or logical source range. |
| `--copies-per-iter` | `1` | Number of contiguous logical copies/source ranges included in each benchmark iteration. Total bytes per iteration are `--copy-size * --copies-per-iter`. |
| `--method {lsu,tma}` | `lsu` | Selects the SM transfer method. `lsu` uses ordinary global load/store instructions. `tma` uses Hopper TMA global-to-shared staging and shared-to-global peer stores. |
| `--executor {src,dst}` | `dst` for single-destination, `src` for round-robin | Selects which GPU's SMs execute the copy kernel. In the round-robin script, only `src` is implemented. |
| `--num-sms` | `0` | Number of CTAs to launch, used as an SM participation cap. `0` means use all physical SMs on the executor GPU. CTAs are not pinned to exact SM IDs. |
| `--threads-per-cta` | `256` | Threads per CTA. Must be in `[1, 1024]`. |
| `--tma-tile-bytes`, `--tma-intra-tile-bytes` | `64K` | TMA global-to-shared staging tile size. Must be positive, a multiple of 16, and no larger than the device's opt-in dynamic shared memory limit. |
| `--tma-inter-tile-bytes` | same as `--tma-tile-bytes` | TMA shared-to-peer-global store chunk size for source-side TMA. Only valid with `--method tma --executor src`. If explicitly set, it must be positive, a multiple of 16, smaller than `--tma-tile-bytes`, and divide `--tma-tile-bytes`. |
| `--persistent-kernel` | off | Runs one warmup kernel and one timed kernel, with each kernel looping over the requested iterations on device. Without this flag, each iteration launches a separate kernel. |
| `--iters` | `100` | Number of timed iterations. Must be positive. |
| `--warmup` | `10` | Number of warmup iterations before timing. Must be non-negative. |
| `--mode {ring,reverse-ring,pair}` | `ring` | Selects peer pattern. Exact meaning differs between the two scripts; see below. |
| `--check` | off | Verifies sampled destination bytes after the timed loop. |
| `--sleep-before` | `0.0` | Sleeps for this many seconds before the benchmark, useful for attaching profilers. |

### Round-Robin-Only Argument

| Argument | Default | Description |
| --- | --- | --- |
| `--round-robin-bytes`, `--round-robin-size` | `8K` | Chunk size used to rotate destination GPUs. Must be positive. In TMA mode it must also be a multiple of 16. |

## Copy Patterns

In `nvlink_lsu_tma_copy_test.py`, `--mode` chooses one destination for each rank:

- `ring`: rank/GPU `i` copies to `(i + 1) % world_size`.
- `reverse-ring`: rank/GPU `i` copies to `(i - 1) % world_size`.
- `pair`: even local ranks copy to the next GPU, odd local ranks copy to the previous GPU. This requires an even number of ranks/GPUs.

In `nvlink_lsu_tma_round_robin_copy_test.py`, `--mode` chooses the destination order:

- `ring`: chunks rotate through `i+1`, `i+2`, ... until all other GPUs are covered.
- `reverse-ring`: chunks rotate through `i-1`, `i-2`, ... until all other GPUs are covered.
- `pair`: accepted by argparse but rejected at runtime because pair mode is not meaningful for all-peer round-robin traffic.

## Key Differences Between the Tests

| Topic | `nvlink_lsu_tma_copy_test.py` | `nvlink_lsu_tma_round_robin_copy_test.py` |
| --- | --- | --- |
| Traffic shape | One source GPU to one destination GPU per rank. | One source GPU to all other GPUs per rank, rotating by chunk. |
| Destination selection | Single destination from `--mode`. | Ordered destination list from `--mode`; excludes the source GPU. |
| Executor support | Supports `--executor src` and `--executor dst`. | Only supports `--executor src`. |
| Additional argument | None. | Adds `--round-robin-bytes` / `--round-robin-size`. |
| Kernel inputs | Direct `src` and `dst` tensors. | Source tensor plus a CUDA tensor of destination pointers. |
| Validation with `--check` | Samples the first byte of each logical copy in the single destination. | Samples offsets from chunks assigned to each destination GPU. |
| `--mode pair` | Supported when world size is even. | Rejected at runtime. |
| Profiler expectation | A kernel copies one peer buffer pair; no CUDA copy-engine APIs should dominate. | A source-side kernel rotates peer stores among destination buffers; no CUDA copy-engine APIs should dominate. |

## Profiling Notes

The scripts are designed so Nsight Systems should show CUDA kernels doing the transfer work, not `cudaMemcpyPeerAsync` or `cudaMemcpyBatchAsync` copy-engine activity. With `--persistent-kernel`, the timed phase should appear as one transfer kernel containing `--iters` device-side copy iterations.

A typical Nsight Systems wrapper is:

```bash
nsys profile \
  -s none \
  --cpuctxsw=none \
  --trace=cuda,nvtx,cudnn,cublas \
  --gpu-metrics-devices=0 \
  --gpu-metrics-set=gh100 \
  --gpu-metrics-frequency=10000 \
  --force-overwrite=true \
  torchrun --standalone --nproc_per_node=8 LSU_TMA/nvlink_lsu_tma_copy_test.py \
    --copy-size 1M \
    --iters 100 \
    --method tma \
    --executor src \
    --num-sms 3 \
    --tma-tile-bytes 64K \
    --tma-inter-tile-bytes 8K \
    --persistent-kernel \
    --check
```
