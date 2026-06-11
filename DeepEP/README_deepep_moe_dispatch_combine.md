# Standalone DeepEP MoE Dispatch/Combine Test

`test_deepep_moe_dispatch_combine.py` runs only this path:

```text
random local tokens
-> randomized top-k routing metadata
-> DeepEP fused dispatch
-> Megatron-style grouped expert MLP
-> DeepEP fused combine
-> final local output
```

The test does not import Megatron-LM and does not run a model, router,
optimizer, training loop, or data pipeline. The DeepEP wrapper logic and expert
MLP structure were copied/adapted from Megatron-LM into the standalone file.

## One Node

```bash
torchrun \
  --nproc_per_node=8 \
  ./DeepEP/test_deepep_moe_dispatch_combine.py \
  --num-local-tokens 4096 \
  --token-hidden 7168 \
  --num-of-experts 256 \
  --topk 8 \
  --ep 8 \
  --moe-ffn-hidden-size 2048
```

## Multiple Nodes

Run the same command on each node with the usual `torchrun` rendezvous
arguments. Example for 2 nodes with 8 GPUs each and one EP group of 16 ranks:

```bash
torchrun \
  --nnodes=2 \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=8 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ./DeepEP/test_deepep_moe_dispatch_combine.py \
  --num-local-tokens 4096 \
  --token-hidden 7168 \
  --num-of-experts 256 \
  --topk 8 \
  --ep 16 \
  --moe-ffn-hidden-size 2048
```

If `--ep` is smaller than `WORLD_SIZE`, the script creates independent
contiguous EP groups of size `--ep`.

## Useful Arguments

- `--num-local-tokens`: local token count per rank.
- `--token-hidden`: hidden size of each token.
- `--num-of-experts`: number of experts in each EP group.
- `--topk`: number of experts selected per token.
- `--ep`: expert-parallel group size.
- `--moe-ffn-hidden-size`: expert FFN hidden size.
- `--dtype {bf16,fp16,fp32}`: token and expert parameter dtype.
- `--activation {swiglu,gelu,silu,relu}`: expert activation, default `swiglu`.
- `--expert-backend {auto,te,torch}`: use Transformer Engine grouped linear,
  torch fallback, or auto-select.
- `--warmup-iters`, `--benchmark-iters`: iteration counts.
- `--deepep-num-sms`: DeepEP SM count passed to `Buffer.set_num_sms`.
- `--no-print-timing`: suppress timing lines.
- `--no-check-correctness`: skip lightweight output sanity checks.
- `--no-async-finish`: disable DeepEP async-finish mode.
- `--no-allocate-on-comm-stream`: disable communication-stream allocation.
- `--rerandomize-routing-each-iter`: regenerate random top-k metadata per iter.

## Assumptions And Limitations

- Run with `torchrun`; each process must map to exactly one CUDA GPU.
- CUDA, NCCL distributed initialization, and the `deep_ep` Python package are
  required.
- The Megatron-LM DeepEP flex dispatcher requires `EP > 1`; this test follows
  that assumption.
- `--ep` must divide `WORLD_SIZE`.
- `--num-of-experts` must divide `--ep` so each rank owns the same number of
  local experts.
- Tensor parallelism for experts is fixed to 1.
- Transformer Engine grouped linear is used when available. The `torch`
  fallback keeps the test runnable for dispatch/combine validation, but it is
  not the TE grouped GEMM kernel path.
