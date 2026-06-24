# Global Memory Reordering CPU Hybrid Test

This folder contains a PyTorch/CUDA benchmark for the same logical workflow as
`Global_Mem_Reordering/`, but with the key ordering step moved to the CPU.

1. Kernel A initializes a payload buffer and a one-byte key buffer in GPU global
   memory.
2. The key buffer is copied from GPU to pinned CPU memory.
3. The CPU builds a reordered payload index.
4. The reordered index is copied back to GPU memory.
5. Kernel B copies payload records from the original GPU buffer into a new GPU
   buffer according to that index.

Each key is stored as one byte, but only the lower seven bits are generated and
used. Sorting is descending by `key & 0x7f`, matching the original test's
largest-key-first behavior.

## Files

| File | Description |
| --- | --- |
| `global_mem_reordering_cpu_hybrid_test.py` | Python driver. Builds the extension, allocates GPU and pinned CPU buffers, times the hybrid stages, and optionally checks correctness. |
| `global_mem_reordering_cpu_hybrid_ext.cpp` | PyTorch extension binding plus the CPU parallel counting-sort helper. |
| `global_mem_reordering_cpu_hybrid_kernels.cu` | CUDA implementation of kernel A and kernel B. |

## CPU Sorting Method

The CPU does not use a comparison sort. Since the effective key space has only
128 values, the extension uses a parallel counting-sort style pass:

- split the source records across CPU worker threads
- build one local 128-bin histogram per worker
- compute descending global starts for keys `127..0`
- give each worker a per-key output range
- scan the worker's source records and write `reordered_indices[src] = dst`

For small key buffers, the extension uses a single-thread counting-sort path to
avoid thread-pool scheduling overhead. Larger buffers use PyTorch's intra-op
thread pool, so repeated benchmark iterations reuse CPU worker threads instead
of spawning new `std::thread` workers each iteration.

The produced index is a source-to-destination mapping. For example, if
`reordered_indices[:3] == [132, 200, 1]`, then original payloads `0`, `1`, and
`2` are copied to reordered payload slots `132`, `200`, and `1`.

## Kernel B Copy Methods

`--payload-copy-method lsu` uses ordinary global load/store instructions. Each
CTA owns a contiguous source-record range, and threads in the CTA walk a
flattened payload-byte/vector space, so different threads can copy different
payload chunks independently.

`--payload-copy-method tma` uses a Hopper TMA-staged copy path. Each CTA owns a
contiguous source-record range and copies records into their reordered
destinations using global-to-shared `cuda::memcpy_async` plus shared-to-global
bulk async stores.

`--num-sms` controls the number of CTAs launched for kernel B. Launching one CTA
per requested SM is the usual way to bound participating SMs for this kind of
benchmark, although CUDA does not expose an exact "CTA X must run on SM X"
contract here.

## Basic Usage

Default-sized run on 8 GPUs:

```bash
torchrun --standalone --nproc_per_node=8 \
  Global_Mem_Reordering_CPU_Hybrid/global_mem_reordering_cpu_hybrid_test.py \
  --num-elements 10000 \
  --payload-size 8K \
  --bit-probability 0.4 \
  --num-sms 0 \
  --payload-copy-method lsu \
  --iters 100 \
  --warmup 10 \
  --check
```

Use Hopper TMA for kernel B:

```bash
torchrun --standalone --nproc_per_node=8 \
  Global_Mem_Reordering_CPU_Hybrid/global_mem_reordering_cpu_hybrid_test.py \
  --num-elements 10000 \
  --payload-size 8K \
  --payload-copy-method tma \
  --tma-tile-bytes 8K \
  --num-sms 10 \
  --iters 100 \
  --warmup 10 \
  --check
```

Pin the CPU sort to a specific worker count:

```bash
torchrun --standalone --nproc_per_node=8 \
  Global_Mem_Reordering_CPU_Hybrid/global_mem_reordering_cpu_hybrid_test.py \
  --cpu-threads 64 \
  --payload-copy-method lsu \
  --check
```

## Arguments

Size arguments accept plain bytes or suffixes such as `K`, `M`, `G`, `KB`,
`MB`, `GB`, `KiB`, `MiB`, and `GiB`. `K/M/G` are interpreted as binary units.

| Argument | Default | Description |
| --- | --- | --- |
| `--num-elements` | `10000` | Number of logical records in the payload and key buffers. |
| `--payload-size` | `8K` | Bytes per payload record. |
| `--bit-probability` | `0.4` | Probability that each of the lower seven generated key bits is set to 1. |
| `--num-sms` | `0` | Number of CTAs launched for kernel B. `0` means use all SMs visible on the local GPU. |
| `--threads-per-cta` | `256` | Threads per CTA. Must be in `[1, 1024]`. |
| `--payload-copy-method {lsu,tma}` | `lsu` | Kernel-B payload copy implementation. |
| `--tma-tile-bytes` | `8K` | Dynamic shared-memory tile size for TMA mode. Must be a multiple of 16 and within the device opt-in shared-memory limit. |
| `--cpu-threads` | `0` | CPU sort worker threads. `0` keeps PyTorch's current intra-op thread count. Values greater than zero call `torch.set_num_threads()`. |
| `--iters` | `100` | Timed hybrid iterations. |
| `--warmup` | `10` | Warmup hybrid iterations before timing. |
| `--seed` | `1234` | Seed for deterministic device-side pseudo-random initialization. |
| `--check` | disabled | Verifies key-bit constraints, index permutation, descending key order, and sampled payload bytes. |
| `--sleep-before` | `0.0` | Seconds to sleep before warmup/timing, useful for attaching profilers. |

## Output

Each rank prints one line similar to:

```text
rank=00 device=cuda:0 num_elements=10000 payload_size=8.00 KiB key_size=1 B key_bits=7 bit_probability=0.400 kernel_b_ctas=10 threads_per_cta=256 payload_copy_method=lsu tma_tile=8.00 KiB cpu_threads=auto payload_buffer=78.12 MiB key_buffer=9.77 KiB index_buffer=39.06 KiB iters=100 avg_key_d2h=...ms avg_cpu_sort=...ms avg_index_h2d=...ms avg_kernel_b=...ms avg_end_to_end=...ms kernel_b_payload_bw=... GiB/s end_to_end_payload_bw=... GiB/s check=OK
```

The bandwidth numerators use only the physically reordered payload bytes. The
end-to-end bandwidth includes key download, CPU sorting, index upload, and
kernel-B payload copy time in the denominator.

## Requirements And Caveats

- Requires CUDA-capable PyTorch and a CUDA toolkit available to
  `torch.utils.cpp_extension.load`.
- Pinned CPU tensors are used for the GPU-to-CPU key copy and CPU-to-GPU index
  copy.
- TMA mode requires Hopper or newer GPUs, CUDA headers with `<cuda/ptx>`, and
  compute capability 9.0+.
- TMA mode requires `--payload-size` and `--tma-tile-bytes` to be multiples of
  16. Use `--payload-copy-method lsu` for non-16-byte-aligned experiments.
- The extension build cache is placed under
  `Global_Mem_Reordering_CPU_Hybrid/.torch_extensions` by default.

## Profiling Notes

Nsight Systems should show the explicit GPU/CPU handoff in addition to kernel B:

```bash
nsys profile \
  -s none \
  --cpuctxsw=none \
  --trace=cuda,nvtx,cudnn,cublas \
  --gpu-metrics-devices=0 \
  --gpu-metrics-set=gh100 \
  --gpu-metrics-frequency=10000 \
  --force-overwrite=true \
  -o global_mem_reordering_cpu_hybrid_lsu_sm10 \
  torchrun --standalone --nproc_per_node=8 \
    Global_Mem_Reordering_CPU_Hybrid/global_mem_reordering_cpu_hybrid_test.py \
    --num-elements 10000 \
    --payload-size 8K \
    --num-sms 10 \
    --payload-copy-method lsu \
    --iters 100 \
    --warmup 10 \
    --check
```
