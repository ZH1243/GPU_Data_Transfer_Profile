# Global Memory Reordering Test

This folder contains a PyTorch/CUDA benchmark for studying a two-stage global
memory workflow on Hopper GPUs:

1. Kernel A initializes a payload buffer and a key buffer in GPU global memory.
2. Kernel B sorts 8-bit keys from largest to smallest, then physically reorders
   payload records into the sorted key order.

The test is intended to be launched with one `torchrun` process per GPU on the
node. Each rank operates on its local CUDA device independently.

## Files

| File | Description |
| --- | --- |
| `global_mem_reordering_test.py` | Python driver. Parses arguments, builds the CUDA extension, allocates tensors, launches warmup/timed iterations, and optionally checks correctness. |
| `global_mem_reordering_ext.cpp` | PyTorch extension binding for kernel A and kernel B. |
| `global_mem_reordering_kernels.cu` | CUDA implementation of buffer initialization, 8-bit counting sort, and payload reordering. |

## What The Test Does

Kernel A initializes two global-memory buffers:

- `payload`: `num_elements` records, each `payload_size` bytes.
- `keys`: `num_elements` records, each `key_size` bytes.

Payload bytes are pseudo-randomized on device. Each key byte is initialized bit
by bit, with probability `--bit-probability` of setting each bit to 1. Kernel B
uses the first byte of each key record as an unsigned 8-bit sort key.

Kernel B runs three CUDA kernels internally:

- build per-CTA histograms over the 256 possible key values
- build descending key offsets from those histograms
- write sorted keys and physically reordered payloads

Because keys are 8-bit values, this uses counting sort instead of a comparison
sort. The output order is descending by key value.

## Basic Usage

Default-sized run on 8 GPUs:

```bash
torchrun --standalone --nproc_per_node=8 \
  Global_Mem_Reordering/global_mem_reordering_test.py \
  --num-elements 10000 \
  --payload-size 8K \
  --key-size 1 \
  --bit-probability 0.4 \
  --num-sms 0 \
  --iters 100 \
  --warmup 10 \
  --check
```

Use a fixed SM/CTA budget for kernel B:

```bash
torchrun --standalone --nproc_per_node=8 \
  Global_Mem_Reordering/global_mem_reordering_test.py \
  --num-elements 10000 \
  --payload-size 8K \
  --key-size 1 \
  --num-sms 10 \
  --payload-copy-method tma \
  --iters 100 \
  --warmup 10 \
  --check
```

Use ordinary SM global load/store copies instead of the Hopper TMA path:

```bash
torchrun --standalone --nproc_per_node=8 \
  Global_Mem_Reordering/global_mem_reordering_test.py \
  --payload-copy-method lsu \
  --num-sms 10 \
  --iters 100 \
  --warmup 10 \
  --check
```

## Arguments

Size arguments accept plain bytes or suffixes such as `K`, `M`, `G`, `KB`,
`MB`, `GB`, `KiB`, `MiB`, and `GiB`. `K/M/G` are interpreted as binary units.

| Argument | Default | Description |
| --- | --- | --- |
| `--num-elements` | `10000` | Number of logical records in the payload and key buffers. |
| `--payload-size` | `8K` | Bytes per payload record. |
| `--key-size` | `1` | Bytes per key record. Kernel B sorts by the first byte of each key record. |
| `--bit-probability` | `0.4` | Probability that each generated key bit is set to 1. |
| `--num-sms` | `0` | Number of CTAs launched for kernel B. `0` means use all SMs visible on the local GPU. CTAs are not pinned to exact SM IDs. |
| `--threads-per-cta` | `256` | Threads per CTA. Must be in `[1, 1024]`. |
| `--payload-copy-method {tma,lsu}` | `tma` | Payload reorder method. `tma` uses Hopper global-to-shared staging plus shared-to-global bulk async stores; `lsu` uses ordinary global load/store copies. |
| `--tma-tile-bytes` | `8K` | Dynamic shared-memory tile size used by the TMA payload path. Must be a multiple of 16 and within the device opt-in shared-memory limit. |
| `--iters` | `100` | Timed kernel-B iterations. |
| `--warmup` | `10` | Warmup kernel-B iterations before timing. |
| `--seed` | `1234` | Seed for deterministic device-side pseudo-random initialization. |
| `--check` | disabled | Verifies descending key order, key histogram preservation, and sampled payload bytes. |
| `--sleep-before` | `0.0` | Seconds to sleep before warmup/timing, useful for attaching profilers. |

## Requirements And Caveats

- Requires CUDA-capable PyTorch and a CUDA toolkit available to
  `torch.utils.cpp_extension.load`.
- The default `--payload-copy-method tma` requires Hopper or newer GPUs,
  CUDA headers with `<cuda/ptx>`, and compute capability 9.0+.
- TMA mode requires `--payload-size` and `--tma-tile-bytes` to be multiples of
  16. Use `--payload-copy-method lsu` for non-16-byte-aligned experiments.
- The Python driver allocates the global tensors with PyTorch, then kernel A
  initializes them on device. This keeps tensor lifetime and ownership simple
  while preserving device-side initialization.
- The extension build cache is placed under
  `Global_Mem_Reordering/.torch_extensions` by default.

## Output

Each rank prints one line similar to:

```text
rank=00 device=cuda:0 num_elements=10000 payload_size=8.00 KiB key_size=1 B bit_probability=0.400 kernel_b_ctas=10 threads_per_cta=256 payload_copy_method=tma tma_tile=8.00 KiB payload_buffer=78.12 MiB key_buffer=9.77 KiB iters=100 elapsed=...s payload_reorder_bw=... GiB/s check=OK
```

The reported `payload_reorder_bw` is based on payload bytes physically written
into the sorted output buffer during the timed kernel-B iterations. It does not
include key/histogram traffic in the numerator.

## Profiling Notes

Nsight Systems should show CUDA kernels, not CUDA copy-engine API activity. A
typical wrapper is:

```bash
nsys profile \
  -s none \
  --cpuctxsw=none \
  --trace=cuda,nvtx,cudnn,cublas \
  --gpu-metrics-devices=0 \
  --gpu-metrics-set=gh100 \
  --gpu-metrics-frequency=10000 \
  --force-overwrite=true \
  -o global_mem_reordering_tma_sm10 \
  torchrun --standalone --nproc_per_node=8 \
    Global_Mem_Reordering/global_mem_reordering_test.py \
    --num-elements 10000 \
    --payload-size 8K \
    --key-size 1 \
    --num-sms 10 \
    --payload-copy-method tma \
    --iters 100 \
    --warmup 10 \
    --check
```
