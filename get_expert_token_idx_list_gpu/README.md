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
  --experts-per-gpu 16 --node-mask-sort --gpus-per-node 4 --check
```

The node size is configured with `--gpus-per-node N`, where `N` can be any
integer from 2 through 8 and defaults to 8. With 16 experts per GPU and
`--gpus-per-node 4`, experts 0--63 belong to node 0, experts 64--127 to node 1,
and so on. For each token and destination node, an N-bit mask records the GPUs
in that node selected by the token (`bit g` represents node-local GPU `g`).
Tokens sent to each node are stably sorted by the mask's unsigned numeric value
in descending order; equal masks retain input-token order. The number of mask
bins and all scratch-buffer sizes are derived from `N`.

The additional output is another packed ragged array:

- `node_token_indices` is `x3`, containing original token IDs in sender-buffer
  order.
- `node_offsets` has `num_nodes + 1` entries.  Node `i` owns
  `node_token_indices[node_offsets[i]:node_offsets[i + 1]]`.

The temporary `x1` and `x2` arrays are never materialized.  The CUDA code uses
per-chunk `(node, mask)` histograms and a stable counting-sort scatter to
write `x3` directly.  It then computes the per-GPU filtered ranks in `x3`
order and rebuilds `expert_token_indices`, so each value matches the token's
new arrival position at its destination GPU.  Output and scratch allocation
remain outside the measured interval.

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
