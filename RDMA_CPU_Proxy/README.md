# RDMA CPU Proxy for GPU Token Buffers

This project is a C++17/CUDA scaffold for a CPU-side RDMA proxy that initiates GPU-to-GPU token-buffer transfers with RDMA Reliable Connection (RC). It is designed for NVIDIA Hopper GPUs plus NVIDIA/Mellanox NICs, with GPUDirect RDMA used when the platform and verbs provider support CUDA device-pointer memory registration.

The code is intentionally split between real hardware paths and explicit mock mode. Mock mode is useful for building and testing chunk/config logic on machines without CUDA or RDMA, but it is not presented as a successful RDMA transfer.

## Architecture

Launch one CPU proxy process per local GPU. A proxy owns exactly one `local_gpu_index`, allocates GPU send/receive token buffers for every peer node, registers those buffers with the NIC, connects RC QPs to matching proxies on other nodes, and posts RDMA Write with Immediate work requests.

Communication is restricted by same local GPU index:

- Node A GPU 0 communicates only with GPU 0 on other nodes.
- Node A GPU 1 communicates only with GPU 1 on other nodes.
- With `num_nodes = 4`, one GPU-0 proxy on Node A connects to Node B GPU 0, Node C GPU 0, and Node D GPU 0.

For each peer node, the proxy creates `num_qps_per_peer` RC QPs. For each QP it starts:

- one send/WQ thread that posts RDMA Write with Immediate WQEs
- one CQ polling thread that polls send completions and receive/immediate completions

The CQ worker reposts receive WRs after receive/immediate completions so the responder keeps enough receive WQEs available.
The measured run waits until every chunk has produced both a send completion and a receive-with-immediate completion on the local QPs before reporting an iteration complete.

## GPU Buffer Layout

Each token buffer is contiguous row-major memory with shape:

```text
(num_tokens, token_dimension)
```

The default element type is BF16, represented as two bytes per element. For each peer, the proxy allocates:

- one GPU send token buffer
- one GPU receive token buffer

Buffer size is:

```text
num_tokens * token_dimension * sizeof(dtype)
```

The CUDA path uses `cudaMalloc` and host-to-device copies for generated test payloads. In mock mode, host memory is allocated instead. When `fill_test_data` is enabled, every iteration fills each peer send buffer with deterministic byte data keyed by source rank, destination rank, GPU index, iteration, and byte offset. When `validate_data` is enabled, the receive buffer is copied back and checked after all expected immediate completions arrive.

## RDMA and GPUDirect RDMA

The real path uses libibverbs/rdma-core APIs:

- open RDMA device/context
- allocate protection domain
- register GPU buffers with `ibv_reg_mr`
- create CQ and RC QPs
- transition QPs through INIT, RTR, and RTS
- post receive WRs
- post `IBV_WR_RDMA_WRITE_WITH_IMM`
- poll completions

Registering CUDA device memory with `ibv_reg_mr` depends on the host/NIC/driver stack. If registration fails, the proxy reports an explicit error mentioning GPUDirect RDMA requirements. It does not silently fall back to host staging.

## Chunking and QP Distribution

The send buffer contains `num_tokens` rows. Tokens are grouped into contiguous chunks of `tokens_per_chunk`, except the final chunk may be smaller.

For each chunk:

```text
src_addr = local_send_buffer_base + chunk_start_token * token_dimension * sizeof(dtype)
dst_addr = remote_recv_buffer_base + chunk_start_token * token_dimension * sizeof(dtype)
length   = num_tokens_in_chunk * token_dimension * sizeof(dtype)
imm_data = chunk_index
```

Chunks are distributed round-robin across QPs for each peer:

```text
chunk 0  -> QP 0
chunk 1  -> QP 1
...
chunk 9  -> QP 9
chunk 10 -> QP 0
```

The immediate value currently encodes the chunk index as a 32-bit unsigned value. Extend `protocol.hpp` if you need flags, stream IDs, or sequence numbers.

## Measured Iterations

The executable runs `num_iterations` measured iterations. Each iteration:

- fills per-peer send buffers with deterministic test data when `fill_test_data=true`
- enqueues every chunk across the peer QPs
- waits for all local send completions and all local receive-with-immediate completions
- verifies that every expected chunk immediate was observed on the expected QP
- validates received bytes when `validate_data=true`
- reports elapsed time, aggregate bandwidth, completion counters, and error counters per QP

Set `num_iterations` to `0` for an indefinite loop. `completion_timeout_ms` bounds how long an iteration waits for completions before failing with per-QP progress details.

## Why Receive WRs Are Required

RDMA Write without Immediate does not consume a receive WQE on the responder. RDMA Write with Immediate does produce a receive-side completion (`IBV_WC_RECV_RDMA_WITH_IMM`), and that completion consumes one posted receive WR. Therefore each receiver must keep receive WRs posted on every RC QP that may receive Write-with-Immediate operations.

This project posts zero-SGE receive WRs because the payload lands directly in the remote GPU receive buffer named by the RDMA write remote address. The receive WR exists to carry the immediate completion event.

## Build

Mock-capable build on a development machine:

```bash
cmake -S RDMA_CPU_Proxy -B RDMA_CPU_Proxy/build
cmake --build RDMA_CPU_Proxy/build
ctest --test-dir RDMA_CPU_Proxy/build --output-on-failure
```

Require CUDA and libibverbs on an RDMA-capable node:

```bash
cmake -S RDMA_CPU_Proxy -B RDMA_CPU_Proxy/build \
  -DRDMA_PROXY_REQUIRE_CUDA=ON \
  -DRDMA_PROXY_REQUIRE_VERBS=ON
cmake --build RDMA_CPU_Proxy/build
```

If CUDA or libibverbs is found automatically, the relevant real path is compiled in. Otherwise the binary can still be built for mock-mode tests.

## Example Launch

Edit `config/example_config.json` for each node and GPU process. Every proxy should use the same `local_gpu_index` value as the GPU it owns, and peers should point to the same-index proxy on other nodes.

Node A, GPU 0:

```bash
./RDMA_CPU_Proxy/build/rdma_cpu_proxy \
  --config RDMA_CPU_Proxy/config/example_config.json \
  --node_rank=0 \
  --local_gpu_index=0 \
  --cuda_device_id=0 \
  --listen_port=18515 \
  --mock_mode=false
```

Node B, GPU 0 would use `--node_rank=1`, `--cuda_device_id=0`, and a peer list containing Node A/C/D GPU-0 proxy addresses.

For 8 GPUs per node, run 8 processes per node, usually with different ports per GPU:

```bash
for gpu in 0 1 2 3 4 5 6 7; do
  ./RDMA_CPU_Proxy/build/rdma_cpu_proxy \
    --config RDMA_CPU_Proxy/config/example_config.json \
    --local_gpu_index=${gpu} \
    --cuda_device_id=${gpu} \
    --listen_port=$((18515 + gpu)) &
done
wait
```

## Configuration

Required parameters are represented in `config/example_config.json`:

- `node_rank`, `num_nodes`
- `local_gpu_index`, `num_gpus_per_node`
- `num_tokens`, `token_dimension`, `tokens_per_chunk`
- `num_qps_per_peer`
- `rdma_device_name`, `rdma_port`, `gid_index`
- `cuda_device_id`
- `listen_port`, `connection_manager_port`
- `peers`
- `completion_poll_batch_size`
- `send_queue_depth`, `recv_queue_depth`, `cq_depth`
- `num_iterations`, `completion_timeout_ms`
- `dtype`
- `mock_mode`
- `fill_test_data`, `validate_data`

Command-line overrides use `--key=value`. `--listen_port=value` also updates every `peers[].port`, matching the common launch convention where GPU `k` uses the same metadata port on every node. You can also use `--peer_port=value` to update every peer port explicitly. `--peer_host=value` is supported only when the config has exactly one peer.

## Current Limitations and TODOs

- Metadata exchange is a minimal TCP exchange intended to make QP bring-up concrete. Production deployments usually replace this with an existing control plane.
- The server side accepts one metadata connection at a time. Multi-peer simultaneous bring-up may need a threaded or event-driven connection manager.
- QP attributes such as MTU, SL, traffic class, timeout, retry count, and GID handling should be tuned per fabric.
- CUDA copy staging currently uses a generic byte-copy kernel. Integrate the real token producer path if tokens are already in the registered send buffer.
- Mock mode copies payload bytes locally and self-completes WQEs for development visibility; it does not simulate fabric ordering or remote process state.
- Error recovery is fail-fast. Production use should add QP teardown/reconnect and health reporting.
