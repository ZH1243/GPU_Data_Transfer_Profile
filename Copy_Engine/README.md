# Copy Engine NVLink Benchmarks

This folder contains Python benchmarks for studying GPU-to-GPU peer copies over
NVLink/P2P using CUDA copy-engine APIs from PyTorch. The scripts use `ctypes` to
call CUDA Runtime functions directly, so the profiled transfer should appear in
Nsight Systems as `cudaMemcpyPeerAsync`, `cudaMemcpyBatchAsync`, or GPU
`Memcpy PtoP` activity, not as an SM copy kernel.

All scripts are intended to be launched with one process per GPU, for example:

```bash
torchrun --standalone --nproc_per_node=8 Copy_Engine/nvlink_copy_engine_test.py --check
```

## Requirements

- NVIDIA GPUs visible to PyTorch.
- CUDA-capable PyTorch with `torch.distributed`/NCCL for multi-process runs.
- P2P access between the selected GPU pairs. Check topology with:

```bash
nvidia-smi topo -m
```

`--copy-mode batch` requires CUDA Runtime 12.8 or newer because it uses
`cudaMemcpyBatchAsync`. `--copy-mode separate` only uses `cudaMemcpyPeerAsync`.

Size arguments accept raw bytes or suffixes: `K`, `M`, `G`, `KB`, `MB`, `GB`,
`KiB`, `MiB`, and `GiB`. `K/M/G` are interpreted as binary units.

## File Overview

### `nvlink_copy_engine_test.py`

Base pairwise copy-engine benchmark.

Each rank chooses one source GPU and one destination GPU according to `--mode`.
In each iteration it copies one or more contiguous chunks from the source
allocation to the destination allocation. This is the simplest script for
checking P2P bandwidth, verifying copy-engine activity in Nsight Systems, and
comparing repeated `cudaMemcpyPeerAsync` calls with one `cudaMemcpyBatchAsync`.

Example:

```bash
torchrun --standalone --nproc_per_node=8 Copy_Engine/nvlink_copy_engine_test.py \
  --copy-size 16M \
  --copies-per-iter 8 \
  --copy-mode batch \
  --iters 100 \
  --warmup 10 \
  --mode ring \
  --check
```

### `nvlink_all_to_all_copy_engine_test.py`

All-to-all copy-engine benchmark.

Each rank owns one source GPU. In every iteration, that source buffer is copied
to one private destination buffer on every other participating GPU. Across all
ranks, this creates concurrent all-to-all traffic. Use this script when you want
to stress aggregate node traffic instead of a single pair pattern.

Per rank and iteration:

- `separate`: submits `world_size - 1` `cudaMemcpyPeerAsync` calls.
- `batch`: submits one `cudaMemcpyBatchAsync` containing `world_size - 1`
  independent copies.

Example:

```bash
torchrun --standalone --nproc_per_node=8 Copy_Engine/nvlink_all_to_all_copy_engine_test.py \
  --copy-size 16M \
  --copy-mode separate \
  --iters 100 \
  --warmup 10 \
  --check
```

### `nvlink_batch_address_layout_test.py`

Batch address-layout experiment.

Every copy in an iteration goes from the same source GPU to the same destination
GPU, but the source and destination addresses can be contiguous or separated by
gaps inside a single allocation. This script is useful for answering a profiler
question: does one `cudaMemcpyBatchAsync` appear as one coalesced P2P memcpy
activity or as multiple activities for a given address layout?

Available layouts:

- `contiguous`: contiguous sources and contiguous destinations.
- `src-discontinuous`: gaps between sources, contiguous destinations.
- `dst-discontinuous`: contiguous sources, gaps between destinations.
- `both-discontinuous`: gaps between both sources and destinations.

Example:

```bash
torchrun --standalone --nproc_per_node=8 Copy_Engine/nvlink_batch_address_layout_test.py \
  --copy-size 1M \
  --copies-per-iter 8 \
  --layout both-discontinuous \
  --gap-size 64K \
  --copy-mode batch \
  --iters 100 \
  --warmup 10 \
  --mode ring \
  --check
```

## Differences Between Scripts

| Script | Traffic pattern | Main question | Copies per rank per iteration |
| --- | --- | --- | --- |
| `nvlink_copy_engine_test.py` | One source GPU to one destination GPU | Baseline P2P copy-engine bandwidth and API comparison | `--copies-per-iter` |
| `nvlink_all_to_all_copy_engine_test.py` | One source GPU to every other GPU | Aggregate all-to-all traffic and launch mode comparison | `world_size - 1` |
| `nvlink_batch_address_layout_test.py` | One source GPU to one destination GPU with configurable address layout | Whether batch copies are coalesced/split by address contiguity | `--copies-per-iter` |

## Common Concepts

### Copy Modes

- `separate`: enqueue one `cudaMemcpyPeerAsync` for each copy.
- `batch`: enqueue one `cudaMemcpyBatchAsync` containing all copies for that
  iteration.

### Pair Modes

Used by `nvlink_copy_engine_test.py` and `nvlink_batch_address_layout_test.py`.

- `ring`: rank/GPU `i` copies to GPU `(i + 1) % world_size`.
- `reverse-ring`: rank/GPU `i` copies to GPU `(i - 1) % world_size`.
- `pair`: even ranks copy to the next GPU and odd ranks copy to the previous
  GPU. This requires an even number of ranks/GPUs.

### Output Metrics

Each script prints per-rank timing/bandwidth and an aggregate summary:

- `elapsed`: wall-clock time for timed iterations plus stream synchronization.
- `local_bw` or `egress_bw`: per-rank bandwidth.
- `aggregate_bw`: sum of all moved bytes divided by the slowest rank elapsed
  time.
- `check`: optional lightweight correctness status when `--check` is enabled.

## Available Arguments

### `nvlink_copy_engine_test.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--nbytes`, `--copy-size` | `1G` | Bytes per copy. Supports byte suffixes listed above. |
| `--copies-per-iter` | `1` | Number of independent copies submitted in each iteration. |
| `--copy-mode` | `separate` | `separate` for one `cudaMemcpyPeerAsync` per copy, or `batch` for one `cudaMemcpyBatchAsync` per iteration. |
| `--iters` | `100` | Number of timed iterations. |
| `--warmup` | `10` | Number of warmup iterations before timing. |
| `--mode` | `ring` | GPU-pair pattern: `ring`, `reverse-ring`, or `pair`. |
| `--check` | disabled | Verify a few copied bytes after the timed loop. |
| `--sleep-before` | `0.0` | Seconds to sleep before benchmark, useful for attaching profilers. |

### `nvlink_all_to_all_copy_engine_test.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--nbytes`, `--copy-size` | `16M` | Bytes copied from this rank's source GPU to each other GPU per iteration. |
| `--copy-mode` | `separate` | `separate` for one `cudaMemcpyPeerAsync` per destination, or `batch` for one `cudaMemcpyBatchAsync` for all destinations. |
| `--iters` | `100` | Number of timed iterations. |
| `--warmup` | `10` | Number of warmup iterations before timing. |
| `--check` | disabled | Verify copied bytes on every destination GPU. |
| `--sleep-before` | `0.0` | Seconds to sleep before benchmark, useful for attaching profilers. |

### `nvlink_batch_address_layout_test.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--nbytes`, `--copy-size` | `1M` | Bytes per copy. Supports byte suffixes listed above. |
| `--copies-per-iter` | `8` | Number of copies in each iteration. |
| `--layout` | `contiguous` | Address layout: `contiguous`, `src-discontinuous`, `dst-discontinuous`, or `both-discontinuous`. |
| `--gap-size` | `64K` | Gap between discontinuous copy regions. Used for discontinuous layouts. |
| `--copy-mode` | `batch` | `separate` for one `cudaMemcpyPeerAsync` per copy, or `batch` for one `cudaMemcpyBatchAsync` per iteration. |
| `--iters` | `100` | Number of timed iterations. |
| `--warmup` | `10` | Number of warmup iterations before timing. |
| `--mode` | `ring` | GPU-pair pattern: `ring`, `reverse-ring`, or `pair`. |
| `--check` | disabled | Verify each destination chunk and destination gaps. |
| `--sleep-before` | `0.0` | Seconds to sleep before benchmark, useful for attaching profilers. |

## Nsight Systems Example

```bash
nsys profile \
  -s none \
  --cpuctxsw=none \
  --trace=cuda,nvtx,cudnn,cublas \
  --force-overwrite=true \
  -o nvlink_copy_engine_profile \
  torchrun --standalone --nproc_per_node=8 Copy_Engine/nvlink_copy_engine_test.py \
    --copy-size 100M \
    --iters 100 \
    --warmup 10 \
    --mode ring \
    --check
```
