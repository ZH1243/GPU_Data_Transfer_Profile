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

By default, each rank visits destination GPUs in ascending ID order, omitting its
own source GPU. Use `--rotate-destination-order` to make source GPU `i` visit
`(i + 1) % world_size`, ..., `(i - 1) % world_size` instead.

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

### `nvlink_multi_source_all_to_all_batch_test.py`

Multi-source all-to-all batched copy benchmark.

Each rank owns one source GPU with multiple logical source buffers, such as
A/B/C/D. Each logical source buffer is split into `world_size - 1` contiguous
sub-buffers, one per peer GPU. For GPU `i`, the first sub-buffer from every
source buffer goes to GPU `(i + 1) % world_size`, the second goes to GPU
`(i + 2) % world_size`, and so on.

The logical source buffers are separated by an explicit gap, but the sub-buffers
inside each logical source buffer are contiguous. By default, sizes are selected
per source buffer: A1/A2/... share one size, B1/B2/... share another size, and
so on. With `--destination-buffer-sizes`, sizes are selected per destination
index instead: A1/B1/C1/D1 share one size, A2/B2/C2/D2 share the next size, and
so on.

In batch mode, each rank submits one `cudaMemcpyBatchAsync` per destination GPU
per iteration. For example, with 8 GPUs and `--num-source-buffers 4`, each rank
submits 7 batch calls per iteration, and each batch contains 4 entries:
A_k/B_k/C_k/D_k for one destination.

Example:

```bash
torchrun --standalone --nproc_per_node=8 Copy_Engine/nvlink_multi_source_all_to_all_batch_test.py \
  --num_source_buffers 4 \
  --source-buffer-sizes 64K,128K,256K,512K \
  --copy-mode batch \
  --iters 100 \
  --warmup 10 \
  --check
```

Destination-indexed size example, where A1/B1/C1/D1 are 64K,
A2/B2/C2/D2 are 128K, and so on:

```bash
torchrun --standalone --nproc_per_node=8 Copy_Engine/nvlink_multi_source_all_to_all_batch_test.py \
  --num_source_buffers 4 \
  --destination-buffer-sizes 64K,128K,256K,512K,1M,2M,4M \
  --copy-mode batch \
  --iters 100 \
  --warmup 10 \
  --check
```

## Differences Between Scripts

| Script | Traffic pattern | Main question | Copies per rank per iteration |
| --- | --- | --- | --- |
| `nvlink_copy_engine_test.py` | One source GPU to one destination GPU | Baseline P2P copy-engine bandwidth and API comparison | `--copies-per-iter` |
| `nvlink_all_to_all_copy_engine_test.py` | One source GPU to every other GPU | Aggregate all-to-all traffic and launch mode comparison | `world_size - 1` |
| `nvlink_batch_address_layout_test.py` | One source GPU to one destination GPU with configurable address layout | Whether batch copies are coalesced/split by address contiguity | `--copies-per-iter` |
| `nvlink_multi_source_all_to_all_batch_test.py` | Multiple logical source buffers on one GPU to every other GPU, grouped by destination | How per-destination batched calls behave when source buffers are non-contiguous but each source's sub-buffers are contiguous | `(world_size - 1) * --num-source-buffers` |

## Common Concepts

### Copy Modes

- `separate`: enqueue one `cudaMemcpyPeerAsync` for each copy.
- `batch`: enqueue one `cudaMemcpyBatchAsync` containing all copies for that
  iteration or destination group, depending on the script.

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
| `--non-uniform-copy-size` | disabled | Use per-copy sizes from `--copy-sizes` instead of one uniform `--copy-size`. |
| `--copy-sizes` | unset | Comma-separated byte sizes for the copies in one iteration, for example `64K,128K,1M`. Must contain exactly `--copies-per-iter` values when non-uniform mode is enabled. |
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
| `--rotate-destination-order` | disabled | Send from GPU `i` to `(i + 1) % world_size`, ..., `(i - 1) % world_size` instead of ascending destination GPU IDs with the source omitted. |

### `nvlink_batch_address_layout_test.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--nbytes`, `--copy-size` | `1M` | Bytes per copy. Supports byte suffixes listed above. |
| `--copies-per-iter` | `8` | Number of copies in each iteration. |
| `--non-uniform-copy-size` | disabled | Use per-copy sizes from `--copy-sizes` instead of one uniform `--copy-size`. |
| `--copy-sizes` | unset | Comma-separated byte sizes for the copies in one iteration, for example `64K,128K,1M`. Must contain exactly `--copies-per-iter` values when non-uniform mode is enabled. |
| `--layout` | `contiguous` | Address layout: `contiguous`, `src-discontinuous`, `dst-discontinuous`, or `both-discontinuous`. |
| `--gap-size` | `64K` | Gap between discontinuous copy regions. Used for discontinuous layouts. |
| `--copy-mode` | `batch` | `separate` for one `cudaMemcpyPeerAsync` per copy, or `batch` for one `cudaMemcpyBatchAsync` per iteration. |
| `--iters` | `100` | Number of timed iterations. |
| `--warmup` | `10` | Number of warmup iterations before timing. |
| `--mode` | `ring` | GPU-pair pattern: `ring`, `reverse-ring`, or `pair`. |
| `--check` | disabled | Verify each destination chunk and destination gaps. |
| `--sleep-before` | `0.0` | Seconds to sleep before benchmark, useful for attaching profilers. |

### `nvlink_multi_source_all_to_all_batch_test.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--num-source-buffers`, `--num_source_buffers` | `4` | Number of logical source buffers per GPU, such as A/B/C/D. |
| `--nbytes`, `--copy-size` | `1M` | Bytes per source sub-buffer when `--source-buffer-sizes` is not set. For example, this is the size of each A_i/B_i/C_i/D_i chunk. |
| `--source-buffer-sizes`, `--source_buffer_sizes` | unset | Comma-separated per-source sub-buffer sizes, for example `64K,128K,256K,512K` for A_i/B_i/C_i/D_i. Must contain exactly `--num-source-buffers` values. |
| `--destination-buffer-sizes`, `--destination_buffer_sizes` | unset | Enables destination-indexed sizing. Comma-separated sizes for A1/B1/..., A2/B2/..., etc. Must contain exactly `world_size - 1` values and cannot be combined with `--source-buffer-sizes`. |
| `--source-buffer-gap-size`, `--source_buffer_gap_size` | `64K` | Gap between logical source buffers A/B/C/... so the logical source buffers are not contiguous. Sub-buffers within each logical source buffer remain contiguous. |
| `--copy-mode` | `batch` | `separate` for one `cudaMemcpyPeerAsync` per source buffer per destination, or `batch` for one `cudaMemcpyBatchAsync` per destination. |
| `--iters` | `100` | Number of timed iterations. |
| `--warmup` | `10` | Number of warmup iterations before timing. |
| `--check` | disabled | Verify copied bytes on every destination GPU. |
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
