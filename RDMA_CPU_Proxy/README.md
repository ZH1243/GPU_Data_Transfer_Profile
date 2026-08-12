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

When `nvlink_forwarding_enabled=true`, each proxy also starts NVLink forwarding threads for its local GPU. The QP CQ workers still own CQ polling and immediate decoding; the forwarding-ready thread monitors their per-chunk immediate counters, while the forwarding thread processes peer-node receive buffers in deterministic descending ring order and issues intra-node GPU-to-GPU copies after a forwarding batch has arrived.

## GPU Buffer Layout

Each token buffer is contiguous row-major memory with shape:

```text
(num_tokens, token_dimension)
```

The default element type is BF16, represented as two bytes per element. In the legacy mode, for each peer the proxy allocates:

- one GPU send token buffer
- one GPU receive token buffer

Buffer size is:

```text
num_tokens * token_dimension * sizeof(dtype)
```

The CUDA path uses `cudaMalloc` and host-to-device copies for generated test payloads. In mock mode, host memory is allocated instead. When `fill_test_data` is enabled, every iteration fills each peer send buffer with deterministic byte data keyed by source rank, destination rank, GPU index, iteration, and byte offset. When `validate_data` is enabled, the receive buffer is copied back and checked after all expected immediate completions arrive.

## Router-Driven RDMA and NVLink Mode

Set `router_routing_enabled=true` to select router-driven RDMA. Configure the global expert count and top-k with `num_experts` and `top_k`; `experts_per_gpu` is derived as:

```text
num_experts / (num_nodes * num_gpus_per_node)
```

This mode requires `num_experts` to divide evenly across all GPUs, `num_gpus_per_node` to be in `[2, 8]`, and `num_experts <= 1024`. `router_seed` controls the test routing-table generator. Different proxies may use different seeds or actual router contents; only the tensor dimensions and chunk configuration must match.

At initialization the proxy:

- allocates one `[num_tokens, top_k]` int32 routing table `R` in HBM;
- launches the same random router-table and full node-mask-sort kernels used by `get_expert_token_idx_list_gpu` with `local_gpu_index` as `local-gpu-id`, producing x3/x4 plus `expert_token_indices` and `expert_offsets`;
- leaves `x3` and `x4` in HBM, copies the node offsets and both aligned arrays through CUDA into pinned CPU memory, and retains their per-node slices;
- exchanges each destination-specific `x3`/`x4` pair and its tensor/chunk metadata with that peer over the existing TCP control channel;
- directly exchanges destination-specific expert metadata with every other GPU proxy, including proxies on the same node;
- ignores `x3[local_node]` for outgoing RDMA and builds a different chunk list for every remote peer node.

Router mode allocates one shared RDMA send buffer on the local GPU instead of one send buffer per peer. Every `x3[peer_node]` list is divided into `tokens_per_chunk` entries. For example, `[3, 1, 4, 9, 10, 45]` with `tokens_per_chunk=3` produces chunks `[3, 1, 4]` and `[9, 10, 45]`. Each chunk uses one SGE per token to gather those discontinuous rows from the shared send buffer and writes them contiguously to the peer receive buffer. Router mode enables the effective behavior of both `rdma_chunk_per_token_sge_enabled` and `rdma_discontinuous_token_payload_enabled`; those two legacy test flags do not also need to be set.

Because every sender may have different router contents, outgoing completion and byte accounting use the local `x3[peer_node]`, while incoming completion, validation, and optional NVLink forwarding use the actual aligned `x3`/`x4` received over TCP. The receiver therefore observes the exact sender-side ordering without assuming identical routing tables.

When both `router_routing_enabled=true` and `nvlink_forwarding_enabled=true`, the remote `x4` slice becomes the receive buffer's NVLink forwarding table. Its low-`N` bit encoding is normalized to the legacy forwarding-table column layout, so the copy path remains the same and visits destination GPUs in this order for every batch:

```text
(local_gpu_index + 1) % N, (local_gpu_index + 2) % N, ...,
(local_gpu_index - 1) % N
```

Router-driven forwarding supports fixed and dynamic chunk thresholds, but always forwards batches in source-token order. `nvlink_forward_out_of_order_chunks_enabled=true` is rejected for this combined mode. A fixed threshold may produce a smaller final batch because a peer's `x3` length is data-dependent. Router mode does not support `nvlink_forward_use_round_robin=true` or `nvlink_forward_local_batch_sync_enabled=true`; local proxies can have different `x3`/`x4` lengths, so a shared batch-count barrier is not well-defined.

In this combined router/NVLink mode, every destination GPU allocates one separate inbound buffer for every global `(source_node_rank, source_gpu_index)` pair, including currently unused pairs. The number of allocations per destination GPU is therefore `num_nodes * num_gpus_per_node`; a 4-node, 8-GPU deployment allocates 32 buffers. The forwarding path uses the `(num_nodes - 1) * (num_gpus_per_node - 1)` remote-node/other-local-GPU pairs (21 in that deployment). Each allocation is `token_buffer_bytes`, because one global source GPU can contribute at most `num_tokens` token rows. Within an iteration, every source/destination pair maintains an independent destination cursor: tokens selected by consecutive forwarding batches are appended with no gaps. The cursor resets to token zero at the next iteration. Because the source node is part of the allocation identity, exact router buffers do not add the legacy peer-node slot offset.

Each source-specific inbound buffer also owns a `RouterExpertMetadata` table for the experts living on the destination GPU. Its `expert_offsets` has `experts_per_gpu + 1` entries and is normalized to zero; local expert `e` owns `expert_token_indices[expert_offsets[e]:expert_offsets[e + 1]]`. Those indices address rows in that particular source-specific compact buffer. `first_global_expert` identifies the global ID of local expert zero. Code using a `Proxy` can retrieve the table with `router_expert_metadata_for_source(source_node_rank, source_gpu_index)`.

When NVLink forwarding completion notifications are also enabled, every source-specific inbound buffer keeps a CPU-side Head and Tail for each expert list. A dequeued notification identifies the buffer by `(peer_rank, source_gpu)` and supplies the newly completed compact token range. The proxy advances each Head past all expert-list indices below the new received-token frontier and adds the Head delta to that local expert's `num_ready_tokens`. The Tail is the number of leading list entries already assigned to GPU work: `[0, Tail)` is published, `[Tail, Head)` is ready but unpublished, and `[Head, end)` has not arrived. Notification ranges are required to be contiguous. Heads, Tails, ready counts, and per-expert notified-batch counts reset on a new iteration. Code can obtain thread-safe progress snapshots from `router_expert_token_head_state_for_source(...)` and `router_computation_scheduler_state()`.

The grouped-GEMM scheduler uses `expert_gemm_m_tile` tokens per computation batch. After the all-to-all expert metadata exchange, it sums every source list length to obtain the total token count for each local expert and allocates exactly `sum_e ceil(expert_tokens[e] / expert_gemm_m_tile)` map rows. This lets the last row for an expert use its exact remaining-token threshold rather than waiting for a full M tile. `expert_gemm_dimension` must be divisible by `expert_gemm_n_tile`; each map row represents `expert_gemm_dimension / expert_gemm_n_tile` ready work IDs.

Each fixed-size row has this GPU-facing `uint32_t` ABI:

```text
global_expert_id, num_used_receive_buffers,
receive_buffers[num_nodes * num_gpus_per_node] {
    receive_buffer_id, tail, length
}
```

`receive_buffer_id` is `source_node_rank * num_gpus_per_node + source_gpu_index`. `tail` is the selected expert list's value before consumption, `length` is the number of tokens taken from that list, and unused receive-buffer slots are zero. The row size is therefore `8 + 12 * num_nodes * num_gpus_per_node` bytes. `RouterNotificationPublicationBuffers` exposes the host/device pointers together with `map_entry_bytes`, `num_map_entries`, and `works_per_batch`.

New rows are appended in notification processing order. The proxy copies only the contiguous range of rows appended for that dequeued notification, then publishes the new `uint32_t` flag on the same nonblocking CUDA stream. The flag equals `published_rows * works_per_batch`, so work IDs below the flag are ready. It is published as zero when a new iteration resets the scheduler, even if that iteration's first completion does not produce a row. Publication uses `cuStreamWriteValue32` with its default system-wide memory barrier; observing a new nonzero flag therefore implies the associated map rows are visible in HBM.

Set `nvlink_forward_notification_flush_per_entry_enabled=true` to use fine-grained publication. In this mode, immediately after advancing one expert's Head and `num_ready_tokens`, the proxy emits every newly ready batch for that expert. Each emitted row updates its selected-list Tails and ready-work flag, then independently enqueues one row-sized H2D copy followed by one flag publication. These operations remain asynchronous, so CPU notification processing continues with the next row and expert without synchronizing the CUDA stream. When the option is false, all rows produced by one dequeued notification are still coalesced into one contiguous map copy and one flag publication. Per-entry mode requires router completion notifications and cannot be combined with flush-only mode.

Set `nvlink_forward_notification_flush_only_enabled=true` to bypass all Head, Tail, and scheduler updates. Every dequeued router completion notification copies only the first map row and then publishes the current host flag. This option requires both `router_routing_enabled=true` and `nvlink_forward_completion_notifications_enabled=true`.

The expert tables use a direct TCP all-to-all. Global proxy rank is `node_rank * num_gpus_per_node + local_gpu_index`; every rank sends one destination-specific table to every other rank and installs its own table locally. Each proxy runs one receiver that accepts the other `X - 1` messages in any arrival order. Connections use `router_metadata_port_base + local_gpu_index`. Remote node hosts are taken from `peers`, while same-node exchanges use IPv6 loopback. Configure the same unused `router_metadata_port_base` on every proxy in a run.

Example overrides:

```bash
./RDMA_CPU_Proxy/build/rdma_cpu_proxy \
  --config RDMA_CPU_Proxy/config/example_config.json \
  --router_routing_enabled=true \
  --num_experts=256 \
  --top_k=8 \
  --router_seed=1234 \
  --router_metadata_port_base=19515 \
  --nvlink_forwarding_enabled=true \
  --nvlink_forward_threshold_tokens=700 \
  --mock_mode=false
```

## Intra-Node NVLink Forwarding

NVLink forwarding is disabled by default and does not change RDMA-only behavior. In non-router mode, each proxy allocates one inbound NVLink receive buffer on its local GPU for every other local source GPU. With 8 GPUs per node, each proxy allocates 7 such buffers. Each buffer is sized for all remote-node streams, so with 4 nodes it has 3 peer-node slots. The combined router/NVLink mode instead uses the global-source allocation layout described above.

The proxy publishes these local inbound buffers through `nvlink_forward_exchange_dir` using CUDA IPC handles. Other local GPU proxies import the specific buffer assigned to their source GPU and forward from their per-peer RDMA receive buffer to that imported destination buffer using the CUDA copy engine. Each destination GPU receives at most one `cudaMemcpyBatchAsync` call per forwarding batch, and all destination calls for one batch are enqueued sequentially into the same CUDA stream.

By default forwarding uses per-buffer routing tables. Set `nvlink_forward_use_round_robin=true` to use the older fixed round-robin layout instead.

Each peer-node RDMA receive buffer has a CPU-side routing table with one `uint8_t` row per token. The eight bits represent routing columns 0 through 7. For a node with `num_gpus_per_node=N`, only columns `0..N-2` are active forwarding columns; columns `N-1..7` are generated and included in sorting, but ignored when issuing NVLink copies. Active column `j` maps to local GPU:

```text
(local_gpu_index + 1 + j) % num_gpus_per_node
```

Column 7 is stored as the least significant bit. In an 8-GPU layout, columns 0 through 6 cover the seven peer GPUs and column 7 is ignored. In a 4-GPU layout, columns 0 through 2 cover the three peer GPUs and columns 3 through 7 are ignored. Routing rows are generated randomly on the CPU using `nvlink_routing_probability`, sorted in descending unsigned-byte order, and then interpreted as matching the token order in the RDMA receive buffer.

For a forwarding batch, each destination scans the routing rows for that batch, gathers the matching token source addresses, and copies them into a continuous destination span with one batched copy call. In combined router/NVLink mode, the next ordered batch starts immediately after the preceding batch's routed tokens for that destination, so the destination remains continuous across batch boundaries. Because tokens may route to multiple destinations, the total forwarded byte count can exceed the batch's source byte count.

The legacy randomly generated routing-table implementation requires:

```text
num_gpus_per_node <= 8
num_tokens must be a multiple of the effective NVLink forwarding threshold
```

The exact-multiple requirement does not apply to router-driven `x4` tables; their final forwarding batch may be smaller than the configured threshold.

Set `nvlink_forward_threshold_tokens` to choose the forwarding batch size directly in tokens. Alternatively, set `nvlink_forward_threshold_chunks` to choose the batch size in RDMA chunks; the effective token count becomes:

```text
nvlink_forward_threshold_chunks * tokens_per_chunk
```

For example, with `nvlink_forward_threshold_chunks=10` and `tokens_per_chunk=32`, each forwarding batch contains 320 tokens. If both `nvlink_forward_threshold_tokens` and `nvlink_forward_threshold_chunks` are set, they must describe the same effective token count.

Set `nvlink_forward_min_threshold_chunks` and `nvlink_forward_max_threshold_chunks` to enable dynamic chunk-threshold forwarding instead of fixed token-threshold forwarding. In this mode, the forwarding-ready thread tracks the contiguous RDMA chunk frontier for each peer buffer: if chunks through `N` are ready but chunk `N+1` is missing, later completed chunks do not move the frontier until the gap arrives. The forwarding thread compares that contiguous ready frontier with its own forwarded frontier, caps the next candidate batch at `nvlink_forward_max_threshold_chunks`, and waits until at least `nvlink_forward_min_threshold_chunks` are available. A final iteration tail smaller than the minimum is allowed once all remaining chunks for that iteration are ready.

When dynamic chunk-threshold forwarding is combined with `nvlink_forward_local_batch_sync_enabled=true`, each GPU proxy writes its candidate chunk count into the same-node shared-memory barrier. After all local GPU proxies have arrived, they use the minimum posted candidate count as the common next batch size, so all local proxies issue the same dynamic forwarding span. Dynamic chunk thresholds are not supported with `nvlink_forward_use_round_robin=true`.

Set `nvlink_forward_out_of_order_chunks_enabled=true` to remove the contiguous-frontier requirement from dynamic chunk forwarding in legacy, non-router mode. In this mode, the forwarding-ready thread keeps per-chunk ready and forwarded state, finds the largest contiguous ready-but-not-forwarded chunk run, and publishes that run's start/end/length to the forwarding thread. The forwarding thread publishes each selected forwarded range back to the ready thread as a batch sequence, issues the GPU-to-GPU copy, and waits for the ready thread to acknowledge and publish the next batch sequence before selecting another range. This allows chunks 12 and 13 to forward even if chunk 11 is still missing. The selected batch size is still capped by `nvlink_forward_max_threshold_chunks`, still needs `nvlink_forward_min_threshold_chunks` unless it is a fully ready iteration tail, and still uses the same local GPU-proxy minimum-size sync when local batch sync is enabled. This mode requires finite `num_iterations` and is rejected when router-driven forwarding is enabled.

In round-robin mode, the previous full-fanout rule is restored:

```text
effective forwarding threshold == nvlink_forward_chunk_tokens * (num_gpus_per_node - 1)
```

For source GPU `g`, chunk `i` in a forwarding batch goes to `(g + 1 + i) % num_gpus_per_node`.

`nvlink_forward_destinations` remains as a manual-address override for experiments. When it is empty, the normal path is to allocate local buffers and exchange CUDA IPC metadata automatically through `nvlink_forward_exchange_dir`. Use a unique exchange directory per run to avoid stale metadata files from an earlier launch.

The forwarding thread assumes the sequential peer-transfer order for this path and uses descending ring order. On `node_rank=0` with four nodes, receive buffers are processed as peer rank 3, then 2, then 1.

Set `nvlink_forward_log_batches=true` to emit per-batch/per-copy forwarding traces. Those trace logs include `iteration`, `src_gpu`, `dst_gpu`, `peer_rank`, `batch`, `route_column`, `batch_start_token`, routed token count, byte count, and first source/destination addresses. The option is disabled by default because large runs can produce many forwarding batches.

When `nvlink_forward_synchronize_batches=true`, each forwarding batch is timed from before its copy operations are enqueued until after `cudaStreamSynchronize()` returns. If `nvlink_forward_log_batches=true`, each synchronized batch also logs `elapsed_us`, `bandwidth_GBps`, and `bandwidth_gbps`. At iteration completion, zero-byte batches are counted as synchronized batches but excluded from bandwidth samples. The proxy reports the arithmetic mean of non-empty synchronized batch bandwidths as `average_batch_bandwidth_GBps` / `average_batch_bandwidth_gbps` and the aggregate `total_forwarded_bytes / non_empty_synchronized_seconds` as `aggregate_synchronized_bandwidth_GBps` / `aggregate_synchronized_bandwidth_gbps`.

Set `nvlink_forward_completion_notifications_enabled=true` to let a source GPU proxy notify destination GPU proxies after synchronized forwarding batches complete. This feature requires both `nvlink_forwarding_enabled=true` and `nvlink_forward_synchronize_batches=true`. Each receiver GPU proxy creates one POSIX shared-memory segment with one single-producer/single-consumer queue for every other local source GPU. With 8 GPUs per node, the GPU-0 proxy owns 7 inbound notification queues, one each for source GPUs 1 through 7. After the sender's `cudaStreamSynchronize()` returns for a forwarding batch, the forwarding thread enqueues notification records into an in-process handoff queue and can continue to later forwarding batches. A sender-side notification dispatch thread polls that handoff queue and pushes notifications into the receiver queues for the GPUs that received forwarded bytes. A receiver-side polling thread dequeues notifications from the queue tail.

Each notification includes the source GPU, destination GPU, remote peer rank/slot, iteration, batch index, destination `start_token`, `num_tokens`, byte offset, and byte length. The `start_token` and `num_tokens` identify the continuous ready span inside the receiver's NVLink receive buffer for that source GPU and peer-node slot. Use `nvlink_forward_notification_queue_depth` to size each per-source queue.

Set `nvlink_forward_notification_log_enabled=true` to record receiver-side notification dequeue events to files. This mode requires `nvlink_forward_completion_notifications_enabled=true`. When the receiver notification thread dequeues an entry, it formats the notification into a single text line with `dequeue_timestamp_ns` and appends that line to an in-process log queue. During shutdown, each GPU proxy drains its log queue and writes one file under `nvlink_forward_notification_log_dir`, named `nvlink_forward_notifications_rank_<rank>_gpu_<gpu>.log`.

Set `nvlink_forward_local_batch_sync_enabled=true` to add a same-node GPU-proxy barrier before every synchronized NVLink forwarding batch. Once a proxy has observed that the next batch is ready for forwarding, it marks the batch-start sequence in shared memory and waits until every local GPU proxy has reached the same sequence before issuing that batch. Because each forwarding thread reaches the next batch-start barrier only after the previous batch's `cudaStreamSynchronize()` has returned, this makes the proxies start each batch together while still allowing them to move directly from a completed batch to checking readiness for the next one. The first batch uses the same barrier, with the previous batch treated as already complete. This option requires `nvlink_forward_synchronize_batches=true` and uses the same local shared-memory run identity as `local_iteration_sync_run_id`.

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

By default, each data chunk is posted as one `IBV_WR_RDMA_WRITE_WITH_IMM` with one SGE covering the full contiguous chunk. When `rdma_chunk_per_token_sge_enabled=true`, the proxy still posts one write-with-immediate per chunk, but builds one local SGE per token in that chunk. For example, `tokens_per_chunk=32` creates 32 SGEs for full chunks and fewer SGEs for the final partial chunk. The remote write target remains the same contiguous destination span.

When both `rdma_chunk_per_token_sge_enabled=true` and `rdma_discontinuous_token_payload_enabled=true`, the proxy shuffles token indices with a fixed seed and assigns that randomized order to chunks. A chunk can therefore gather non-contiguous source tokens, such as `[3, 1, 8]`, into one write-with-immediate. The gathered payload is packed into the chunk's contiguous remote destination span. This mode is intended for RDMA-only testing and is rejected when NVLink forwarding is enabled.

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
- reports elapsed time and aggregate bandwidth; when `log_qp_reports=true`, also reports completion counters and error counters per QP
- exchanges a TCP iteration-done barrier with every peer before starting the next iteration or tearing down RDMA resources

At the end of a finite run, each GPU proxy writes its RDMA bandwidth summary to `rdma_bandwidth_summary_dir` as `rdma_bandwidth_summary_rank_<rank>_gpu_<gpu>.txt`. The file contains average, min, max, median, and variance for the per-iteration `bandwidth_gbps` samples.

Set `local_iteration_sync_enabled=true` to add a same-node GPU-proxy barrier at the iteration boundaries. With this enabled, the proxy first completes the existing same-GPU-index cross-node barrier, then marks its `iteration_start` / `iteration_done` slot in a POSIX shared-memory segment and polls the slots for every local GPU index on the node. The same local barrier is also applied after the existing cross-node iteration-done barrier. Use the same `local_iteration_sync_run_id` for every proxy in one launch, and prefer a unique value per launch.

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
- `nvlink_forward_threshold_chunks`
- `nvlink_forward_min_threshold_chunks`
- `nvlink_forward_max_threshold_chunks`
- `nvlink_forward_out_of_order_chunks_enabled`
- `nvlink_forward_chunk_tokens`
- `nvlink_forward_use_batch_api`
- `nvlink_forward_stream_nonblocking`
- `nvlink_forward_synchronize_batches`
- `nvlink_forward_completion_notifications_enabled`
- `nvlink_forward_notification_flush_only_enabled`
- `nvlink_forward_notification_flush_per_entry_enabled`
- `expert_gemm_m_tile`
- `expert_gemm_n_tile`
- `expert_gemm_dimension`
- `nvlink_forward_notification_queue_depth`
- `nvlink_forward_notification_log_enabled`
- `nvlink_forward_notification_log_dir`
- `nvlink_forward_local_batch_sync_enabled`
- `nvlink_forward_synchronize_iteration`
- `nvlink_forward_log_batches`
- `local_iteration_sync_enabled`
- `local_iteration_sync_dir`
- `local_iteration_sync_run_id`
- `rdma_bandwidth_summary_dir`
- `log_qp_reports`
- `log_marker_wait_reports`
- `nvlink_forward_use_round_robin`
- `nvlink_routing_probability`
- `nvlink_routing_seed`
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
