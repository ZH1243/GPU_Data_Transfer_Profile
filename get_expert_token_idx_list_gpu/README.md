# GPU expert token-index lists

This directory builds stable, compact per-expert lists directly from an
`int32` CUDA routing matrix `R[T, topK]`.  Experts are assigned contiguously:
expert `e` belongs to GPU `e // n`.

The result is a ragged array represented by:

- `expert_token_indices`: `T * topK` packed `int32` values.
- `expert_offsets`: `K + 1` offsets.  Expert `e` owns
  `expert_token_indices[expert_offsets[e]:expert_offsets[e + 1]]`.

Each value is the token's position in the ordered sequence of tokens routed to
that expert's GPU.  A token routed to several experts on one GPU consumes only
one position on that GPU.  Lists preserve input token order.

## Run

PyTorch with CUDA and a CUDA toolkit are required.  The extension is compiled
for Hopper (`sm_90`) by default.

```bash
python run.py --tokens 16384 --top-k 8 --experts 256 \
  --experts-per-gpu 32 --check --warmup 20 --iters 200
```

The Python driver allocates `R` in GPU HBM, invokes the CUDA input generator,
runs one correctness check against a CPU implementation, and reports average
CUDA-event latency.  Output and scratch buffers are allocated once and reused,
so allocation time is excluded.

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
