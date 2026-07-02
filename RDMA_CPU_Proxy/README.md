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
Before the first measured send, peers also exchange a TCP ready barrier after QPs are connected and receive WQEs are posted.

When `nvlink_forwarding_enabled=true`, each proxy also starts one forwarding thread for its local GPU. The QP CQ workers still own CQ polling and immediate decoding; the forwarding thread monitors their per-chunk immediate counters, processes peer-node receive buffers in deterministic descending ring order, and issues intra-node GPU-to-GPU copies after a complete forwarding batch has arrived.

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

## Intra-Node NVLink Forwarding

NVLink forwarding is disabled by default and does not change RDMA-only behavior. When enabled, each proxy allocates one inbound NVLink receive buffer on its local GPU for every other local source GPU. With 8 GPUs per node, each proxy allocates 7 such buffers. Each buffer is sized for all remote-node streams, so with 4 nodes it has 3 peer-node slots.

The proxy publishes these local inbound buffers through `nvlink_forward_exchange_dir` using CUDA IPC handles. Other local GPU proxies import the specific buffer assigned to their source GPU and forward from their per-peer RDMA receive buffer to that imported destination buffer using the CUDA copy engine. Each forwarding copy is submitted as a one-entry `cudaMemcpyBatchAsync` call by default, and all copies for one forwarding batch are enqueued sequentially into the same CUDA stream.

The current implementation supports the common full-fanout layout only:

```text
nvlink_forward_threshold_tokens == nvlink_forward_chunk_tokens * (num_gpus_per_node - 1)
num_tokens must be a multiple of nvlink_forward_threshold_tokens
```

For source GPU `g`, chunk `i` in a forwarding batch goes to:

```text
dst_gpu = (g + 1 + i) % num_gpus_per_node
```

The source token offset is copied into the destination buffer slot for the current peer-node RDMA receive buffer. For example, with GPU 0, threshold 700, chunk size 100, and 8 local GPUs, tokens `[0,100)` go to GPU 1, `[100,200)` to GPU 2, and so on through GPU 7. GPU 1 starts at GPU 2 and wraps around to GPU 0.

`nvlink_forward_destinations` remains as a manual-address override for experiments. When it is empty, the normal path is to allocate local buffers and exchange CUDA IPC metadata automatically through `nvlink_forward_exchange_dir`. Use a unique exchange directory per run to avoid stale metadata files from an earlier launch.

The forwarding thread assumes the sequential peer-transfer order for this path and uses descending ring order. On `node_rank=0` with four nodes, receive buffers are processed as peer rank 3, then 2, then 1.

Set `nvlink_forward_log_batches=true` to emit per-batch/per-copy forwarding traces. Those trace logs include `iteration`, `src_gpu`, `dst_gpu`, `peer_rank`, `batch`, `chunk`, `token_offset`, `token_count`, byte count, and source/destination addresses. The option is disabled by default because large runs can produce many forwarding batches.

When `nvlink_forward_synchronize_batches=true`, each forwarding batch is timed from before its copy operations are enqueued until after `cudaStreamSynchronize()` returns. If `nvlink_forward_log_batches=true`, each synchronized batch also logs `elapsed_us`, `bandwidth_GBps`, and `bandwidth_gbps`. At iteration completion, the proxy reports the arithmetic mean of synchronized batch bandwidths as `average_batch_bandwidth_GBps` / `average_batch_bandwidth_gbps` and the aggregate `total_forwarded_bytes / total_synchronized_seconds` as `aggregate_synchronized_bandwidth_GBps` / `aggregate_synchronized_bandwidth_gbps`.

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

## Chunking and Dynamic QP Distribution

The send buffer contains `num_tokens` rows. Tokens are grouped into contiguous chunks of `tokens_per_chunk`, except the final chunk may be smaller.

For each chunk:

```text
src_addr = local_send_buffer_base + chunk_start_token * token_dimension * sizeof(dtype)
dst_addr = remote_recv_buffer_base + chunk_start_token * token_dimension * sizeof(dtype)
length   = num_tokens_in_chunk * token_dimension * sizeof(dtype)
imm_data = chunk_index
```

Chunks are not statically bound to QPs when the chunk list is built. For each peer and iteration, the proxy creates one shared dynamic chunk distributor. Each QP send worker asks the distributor for chunks while its local in-flight count is below `max_in_flight_chunks_per_qp`. The worker posts signaled RDMA Write-with-Immediate WRs, its CQ worker observes send CQEs, and each send CQE frees one in-flight slot so the send worker can pull more chunks. Faster QPs therefore pull more chunks during that iteration.

```text
QP 0 in_flight < max window -> asks distributor -> receives next unassigned chunk
QP 3 observes a send CQE     -> frees a slot    -> receives next unassigned chunk
```

After the distributor has no chunks left, each QP worker posts one final zero-payload `SEND_WITH_IMM` marker. The iteration waits for one local send marker completion and one receive-side marker completion per QP.

The immediate value still encodes the chunk index as a 32-bit unsigned value. Extend `protocol.hpp` if you need flags, stream IDs, or sequence numbers.

Dynamic distribution needs send CQEs to manage the per-QP in-flight window, so dynamically assigned data WRs are signaled. The `data_signal_interval` setting is still parsed for configuration compatibility, but the dynamic path does not use reduced data-signaling cadence. The final per-QP `SEND_WITH_IMM` marker is also signaled, so each iteration has a local send-side completion that proves earlier WRs on that QP have drained.

## Measured Iterations

The executable runs `num_iterations` measured iterations. Each iteration:

- fills per-peer send buffers with deterministic test data when `fill_test_data=true`
- captures completion counter baselines, then exchanges a TCP iteration-start barrier so no peer can post measured RDMA writes before the receiver has captured its baseline
- starts one dynamic chunk distributor per peer and one distributor task per QP worker
- waits for one local send-side drain marker and one receive-side drain marker per QP
- reports whether every expected data immediate was observed across the peer QPs
- validates received bytes when `validate_data=true`
- reports elapsed time, aggregate bandwidth, completion counters, and error counters per QP
- exchanges a TCP iteration-done barrier with every peer before starting the next iteration or tearing down RDMA resources

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
  --cpu_affinity=auto \
  --listen_port=18515 \
  --mock_mode=false
```

Node B, GPU 0 would use `--node_rank=1`, `--cuda_device_id=0`, and a peer list containing Node A/C/D GPU-0 proxy addresses.

`--cpu_affinity=auto` runs `nvidia-smi topo -m`, reads the `CPU Affinity` column for `cuda_device_id`, and binds the proxy process before CUDA/RDMA initialization and before QP worker threads are created. Worker threads inherit that CPU mask. You can also pass an explicit Linux CPU list such as `--cpu_affinity=0-95,192-287`, or disable binding with `--cpu_affinity=none`.

For the topology shown in the prompt:

- GPUs 0-3 bind to NUMA 0 CPUs: `0-95,192-287`
- GPUs 4-7 bind to NUMA 1 CPUs: `96-191,288-383`

For 8 GPUs per node, run 8 processes per node, usually with different ports per GPU:

```bash
for gpu in 0 1 2 3 4 5 6 7; do
  ./RDMA_CPU_Proxy/build/rdma_cpu_proxy \
    --config RDMA_CPU_Proxy/config/example_config.json \
    --local_gpu_index=${gpu} \
    --cuda_device_id=${gpu} \
    --cpu_affinity=auto \
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
- `data_signal_interval`
- `max_in_flight_chunks_per_qp`
- `send_queue_depth`, `recv_queue_depth`, `cq_depth`
- `num_iterations`, `completion_timeout_ms`
- `dtype`
- `mock_mode`
- `fill_test_data`, `validate_data`
- `nvlink_forwarding_enabled`
- `nvlink_forward_threshold_tokens`
- `nvlink_forward_chunk_tokens`
- `nvlink_forward_use_batch_api`
- `nvlink_forward_stream_nonblocking`
- `nvlink_forward_synchronize_batches`
- `nvlink_forward_synchronize_iteration`
- `nvlink_forward_log_batches`
- `nvlink_forward_exchange_dir`
- `nvlink_forward_destinations`
- `cpu_affinity`

Command-line overrides use `--key=value`. `--listen_port=value` also updates every `peers[].port`, matching the common launch convention where GPU `k` uses the same metadata port on every node. You can also use `--peer_port=value` to update every peer port explicitly. `--peer_host=value` is supported only when the config has exactly one peer.

## Current Limitations and TODOs

- Metadata exchange is a minimal TCP exchange intended to make QP bring-up concrete. Production deployments usually replace this with an existing control plane.
- The server side accepts one metadata connection at a time. Multi-peer simultaneous bring-up may need a threaded or event-driven connection manager.
- QP attributes such as MTU, SL, traffic class, timeout, retry count, and GID handling should be tuned per fabric.
- CUDA copy staging currently uses a generic byte-copy kernel. Integrate the real token producer path if tokens are already in the registered send buffer.
- Mock mode copies payload bytes locally and self-completes WQEs for development visibility; it does not simulate fabric ordering or remote process state.
- Error recovery is fail-fast. Production use should add QP teardown/reconnect and health reporting.
