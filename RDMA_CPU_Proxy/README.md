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

## Optional RDMA QP calibration

For a **two-node, RDMA-only** run, enable `rdma_qp_calibration_enabled=true`
on every proxy on both nodes to select a QP set before real iteration zero.
It is disabled by default; leaving it disabled preserves the existing launch,
transfer, and reporting path. This option currently rejects NVLink forwarding
and deployments with more or fewer than two nodes. Router-selected RDMA payloads
are supported, as are contiguous payloads with router routing disabled.

After Python finishes initializing its GPU tensors, the first iteration
preparation performs calibration using the existing buffers, memory
registrations, chunk layout, SGE pattern, QP count, and in-flight limit. Each
round runs all local GPU pairs concurrently, including previously passing
pairs. Both endpoints exchange their median outgoing payload bandwidth over
TCP. A pair passes only when **both directions** reach the threshold; the
threshold is the aggregate across the pair's QPs, not per-QP bandwidth and not
the sum of transmit and receive bandwidth. `340` means **340 Gb/s (42.5 GB/s)**.

Only pairs that have never passed quiesce their workers and recreate their complete QP/CQ and
worker sets at both endpoints. Buffers, registered memory, and router metadata
are retained. Calibration succeeds as soon as every pair has passed at least
once and retained its QPs; pairs do not need to pass in the same round, and
there is no additional confirmation round. Once a pair passes, its QP set is
permanently retained during calibration, even if its bandwidth falls below the
threshold in later rounds. It continues sampling under concurrent load. Logs
show `pair_pass` for the current round and `qp_set_retained` for this persistent
retention decision. Retained pairs cannot exhaust the QP-generation budget;
the calibration timeout still applies while waiting for other pairs to pass.
Calibration samples do
not count toward `num_iterations` or the normal bandwidth summary; normal
iteration completion baselines exclude calibration traffic. The real run uses
the accepted QPs without reconnecting them.

| Setting | Default | Meaning |
| --- | --- | --- |
| `rdma_qp_calibration_enabled` | `false` | Enable startup calibration on both nodes |
| `min_bandwidth_gbps_needed` | `340` | Required median outgoing payload Gb/s at each endpoint |
| `rdma_qp_calibration_warmup_iterations` | `2` | Unmeasured samples at the start of each round |
| `rdma_qp_calibration_sample_iterations` | `5` | Measured samples per round |
| `rdma_qp_calibration_max_attempts` | `10` | Maximum QP generations per pair, including the initial set |
| `rdma_qp_calibration_timeout_ms` | `300000` | Calibration time budget, starting at first iteration preparation |

These options work in JSON and as `--key=value` CLI overrides. Calibration
settings must match across all workers. Local calibration synchronization is
automatically enabled even if `local_iteration_sync_enabled=false`; use the
same `local_iteration_sync_run_id` for all local workers and a fresh ID for
separate launches. The existing `completion_timeout_ms` still bounds individual
control exchanges, local barriers, and completion waits. Exhaustion or timeout
fails the run; it does not silently start with a below-threshold QP set. Fatal
peer failures are propagated through control-exchange/barrier timeouts.

On attempt exhaustion, each currently failing pair appends a
`qp_calibration_history` block to its error (including the Python `RuntimeError`).
It identifies both endpoints, the threshold, and payload size, then lists every
round's QP generation, local minimum/maximum/median, remote median, pass decision,
and QP retention status. `pair_min_median_gbps` is the smaller of the two endpoint
medians, so it can be compared directly with the threshold across generations.
Pairs that passed the last round report that another pair exhausted its attempts.
History is available even with informational logging disabled.

Rebuild `rdma_cpu_proxy_shared` on both remote nodes. Append the following to
**each of your existing RDMA-only torchrun commands**:

```bash
--rdma_qp_calibration_enabled=true \
--min_bandwidth_gbps_needed=340 \
--rdma_qp_calibration_warmup_iterations=2 \
--rdma_qp_calibration_sample_iterations=5 \
--rdma_qp_calibration_max_attempts=10 \
--rdma_qp_calibration_timeout_ms=300000
```

The checked-in torchrun scripts enable forwarding by default. For those scripts,
also supply the RDMA-only overrides:

```bash
--nvlink_forwarding_enabled=false \
--router_local_input_staging_enabled=false \
--nvlink_forward_local_batch_sync_enabled=false \
--nvlink_forward_completion_notifications_enabled=false \
--nvlink_forward_preparation_enabled=false \
--nvlink_forward_submit_epilogue_enabled=false \
--nvlink_forward_prepared_atomic_ring_enabled=false \
--local_forwarding_rdma_overlap_enabled=false
```

Logs beginning with `qp_calibration` report round, QP generation, local/remote
median bandwidth, threshold, pair decision, and QP retention status. QP creation
logs include the QPNs of each replacement set. The score uses outgoing payload
bytes divided by the elapsed sample time through observation of both local-send
and remote-receive end markers, including software scheduling and CQ polling
latency. It is not Nsight's active-PCIe bandwidth metric. Initialization,
barriers, payload filling, and optional payload validation are outside the
sample's timed interval. Empty routed traffic cannot be calibrated.

Recreating QPs may improve path distribution but cannot guarantee bandwidth if
the limit is CPU scheduling, insufficient outstanding work, or shared-fabric
congestion. Calibration traffic is visible in Nsight before real iteration zero.
Mock-mode tests exercise the lifecycle and coordination only; their bandwidth
values do not represent RDMA hardware performance.

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

Router-driven forwarding supports fixed and dynamic chunk thresholds, but always forwards batches in source-token order. `nvlink_forward_out_of_order_chunks_enabled=true` is rejected for this combined mode. A fixed threshold may produce a smaller final batch because a peer's `x3` length is data-dependent. Router mode does not support `nvlink_forward_use_round_robin=true`. Local batch synchronization is supported with unequal `x3`/`x4` lengths: a proxy retires from the remaining batch rounds after it finishes the current iteration, and the still-active proxies continue synchronizing with one another.

In this combined router/NVLink mode, every destination GPU keeps one logical inbound descriptor for every global `(source_node_rank, source_gpu_index)` pair. A 4-node, 8-GPU deployment therefore has 32 logical descriptors. The `(num_nodes - 1) * (num_gpus_per_node - 1)` remote-node/other-local-GPU pairs (21 in that deployment) own compact NVLink receive buffers. The additional `num_nodes - 1` remote-node/same-local-GPU descriptors (3 in that deployment) alias the proxy's existing peer RDMA receive buffers instead of allocating or copying another token buffer. With `router_local_input_staging_enabled=false`, `router_computation_forwarded_inputs_only=true` exposes those 24 remote inputs to GroupedGEMM. With local staging enabled, the existing eight local-node descriptors are populated too, so all 32 inputs become active without another token-buffer allocation. Each standalone allocation is `token_buffer_bytes`, because one global source GPU can contribute at most `num_tokens` token rows. Within an iteration, every compact source/destination pair maintains an independent destination cursor: tokens selected by consecutive forwarding batches are appended with no gaps. The cursor resets to token zero at the next iteration. Because the source node is part of the allocation identity, exact router buffers do not add the legacy peer-node slot offset.

`router_local_input_staging_enabled=true` gathers the local-node `x3` rows from the shared router send tensor into the already allocated `(local_node, local_gpu)` input before the first iteration. That source GPU consumes the full staged x3 sequence directly; the proxy drains the staged source in batches capped by `nvlink_forward_max_threshold_chunks`, compacts x4-selected rows into the other local GPUs, and sends the same completion notifications as the remote path. With local batch synchronization enabled, these batches use a dedicated local-input-staging phase; a proxy with a shorter local `x3` retires from the remaining staging rounds without retiring from the later remote-forwarding phase. Remote-node forwarding is held until the local drain finishes for that proxy and iteration. The staging copy is intentionally performed once and reused, so this mode requires `fill_test_data=false`, automatic CUDA-IPC destinations, and an embedded send tensor and routing table that remain unchanged across iterations.

Each source-specific inbound buffer also owns a `RouterExpertMetadata` table for the experts living on the destination GPU. Its `expert_offsets` has `experts_per_gpu + 1` entries and is normalized to zero; local expert `e` owns `expert_token_indices[expert_offsets[e]:expert_offsets[e + 1]]`. Those indices address rows in that particular source-specific compact buffer. `first_global_expert` identifies the global ID of local expert zero. Code using a `Proxy` can retrieve the table with `router_expert_metadata_for_source(source_node_rank, source_gpu_index)`.

When NVLink forwarding completion notifications are also enabled, every source-specific inbound buffer keeps a CPU-side Head and Tail for each expert list. A dequeued notification identifies the buffer by `(peer_rank, source_gpu)` and supplies the newly completed compact token range. The proxy advances each Head past all expert-list indices below the new received-token frontier and adds the Head delta to that local expert's `num_ready_tokens`. The Tail is the number of leading list entries already assigned to GPU work: `[0, Tail)` is published, `[Tail, Head)` is ready but unpublished, and `[Head, end)` has not arrived. Notification ranges are required to be contiguous. Heads, Tails, ready counts, and per-expert notified-batch counts reset on a new iteration. Code can obtain thread-safe progress snapshots from `router_expert_token_head_state_for_source(...)` and `router_computation_scheduler_state()`.

The grouped-GEMM scheduler now publishes the same multi-buffer gather-table ABI used by QuACK's `hopper_stream_gather_table_gemm.py`. After the all-to-all expert metadata exchange, each source-specific `expert_token_indices` vector is copied to GPU HBM as one QuACK `A_idx_j` array. For compact NVLink inputs, an index is the token's compact rank among tokens routed to that destination GPU. For a remote same-local-GPU direct input, an index is instead the token's position in the complete destination-node `x3` RDMA receive sequence. All `A_idx_j` allocations have the same padded length, while `expert_token_index_count` records each meaningful prefix. Receive buffer slot `j`, `A_idx_j`, and pair `(start_j, end_j)` in every table row use the same stable active-input order.

One M-cluster holds `expert_gemm_m_tile * expert_gemm_cluster_m` tokens. Let `clusters_n = ceil(expert_gemm_dimension / expert_gemm_n_tile)`, `group_size = min(expert_gemm_max_swizzle_size, clusters_n)`, and `n_groups = clusters_n / group_size`; `clusters_n` must be divisible by `group_size`. The proxy allocates `sum_e ceil(expert_tokens[e] / M_cluster_capacity) * n_groups` rows. A partial final M-cluster is emitted when all remaining tokens for that expert are ready.

Each fixed-size row has QuACK's contiguous `int32_t` ABI:

```text
local_expert_id, cid_n_base,
start_0, end_0, start_1, end_1, ..., start_B-1, end_B-1
```

Here `B = num_nodes * num_gpus_per_node`. Pair `(start_j, end_j)` is an absolute half-open range in `A_idx_j`, including that expert's `expert_offsets[e]`; an unused buffer has `start_j == end_j`. Each ready M-cluster contributes `n_groups` rows with identical buffer ranges and `cid_n_base = 0, group_size, ...`. The row width is `2 + 2 * B` integers. `RouterNotificationPublicationBuffers` exposes the host/device table and ready-row pointers plus its shape, group, and `A_idx` metadata.

The NVLink completion receiver thread advances Heads, prepares newly ready rows in pinned host memory, copies only that contiguous row range to the GPU table, and then publishes an `int32_t` ready-row count on the same nonblocking CUDA stream. As in QuACK, row `i` is consumable when `ready_rows >= i + 1`. The proxy publishes zero before starting each iteration. Flag publication uses either `cuStreamWriteValue32` or a four-byte `cudaMemcpyAsync`, selected by `nvlink_forward_notification_flag_update_mode`, so observing a larger ready-row count implies all rows in that newly visible prefix are already in HBM.

Set `nvlink_forward_notification_flush_per_entry_enabled=true` to use fine-grained publication. In this mode, each emitted table row independently enqueues one row-sized H2D copy followed by one ready-row publication. These operations remain asynchronous, so CPU notification processing continues without synchronizing the CUDA stream. When the option is false, all rows produced by one dequeued notification are coalesced into one contiguous table copy and one ready-row publication. Per-entry mode requires router completion notifications and cannot be combined with flush-only mode.

Set `nvlink_forward_notification_flush_only_enabled=true` to bypass all Head, Tail, and scheduler updates. Every dequeued router completion notification copies only the first table row and then publishes the current host ready-row value. This diagnostic option requires both `router_routing_enabled=true` and `nvlink_forward_completion_notifications_enabled=true`.

Set `nvlink_forward_notification_flag_update_mode` to select how the CPU proxy publishes the 32-bit ready-row flag to GPU HBM. `stream-write` (the default, preserving the original behavior) uses `cuStreamWriteValue32`; `memcpy` uses a four-byte `cudaMemcpyAsync` from an immutable CUDA-pinned staging value. In both modes, table-row copies and flag publication use the same nonblocking CUDA stream, so the flag is ordered after the rows it makes visible.

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

When dynamic chunk-threshold forwarding is combined with `nvlink_forward_local_batch_sync_enabled=true`, each active GPU proxy writes its candidate chunk count into the same-node shared-memory barrier. After every proxy is either ready or retired for the current iteration, the active proxies use the minimum posted candidate count as the common next batch size. Dynamic chunk thresholds are not supported with `nvlink_forward_use_round_robin=true`.

Set `nvlink_forward_out_of_order_chunks_enabled=true` to remove the contiguous-frontier requirement from dynamic chunk forwarding in legacy, non-router mode. In this mode, the forwarding-ready thread keeps per-chunk ready and forwarded state, finds the largest contiguous ready-but-not-forwarded chunk run, and publishes that run's start/end/length to the forwarding thread. The forwarding thread publishes each selected forwarded range back to the ready thread as a batch sequence, issues the GPU-to-GPU copy, and waits for the ready thread to acknowledge and publish the next batch sequence before selecting another range. This allows chunks 12 and 13 to forward even if chunk 11 is still missing. The selected batch size is still capped by `nvlink_forward_max_threshold_chunks`, still needs `nvlink_forward_min_threshold_chunks` unless it is a fully ready iteration tail, and still uses the same local GPU-proxy minimum-size sync when local batch sync is enabled. This mode requires finite `num_iterations` and is rejected when router-driven forwarding is enabled.

In round-robin mode, the previous full-fanout rule is restored:

```text
effective forwarding threshold == nvlink_forward_chunk_tokens * (num_gpus_per_node - 1)
```

For source GPU `g`, chunk `i` in a forwarding batch goes to `(g + 1 + i) % num_gpus_per_node`.

`nvlink_forward_destinations` remains as a manual-address override for experiments. When it is empty, the normal path is to allocate local buffers and exchange CUDA IPC metadata automatically through `nvlink_forward_exchange_dir`. Use a unique exchange directory per run to avoid stale metadata files from an earlier launch.

The forwarding thread assumes the sequential peer-transfer order for this path and uses descending ring order. On `node_rank=0` with four nodes, receive buffers are processed as peer rank 3, then 2, then 1.

Set `local_forwarding_rdma_overlap_enabled=true` to prime the first outgoing
peer in ascending sequential-transfer order before the proxy drains its local
router-staging source. This allows the first remote RDMA send/receive stage to
overlap local NVLink forwarding. The option requires
`sequential_peer_transfers=true`. It does not allow remote NVLink forwarding to
overtake the local source, and later outgoing peers remain sequential. The
default is false, preserving the local-first behavior.

Set `nvlink_forward_log_batches=true` to emit per-batch/per-copy forwarding traces. Those trace logs include `iteration`, `src_gpu`, `dst_gpu`, `peer_rank`, `batch`, `route_column`, `batch_start_token`, routed token count, byte count, and first source/destination addresses. The option is disabled by default because large runs can produce many forwarding batches.

When `nvlink_forward_synchronize_batches=true`, each forwarding batch is timed from before its copy operations are enqueued until after `cudaStreamSynchronize()` returns. If `nvlink_forward_log_batches=true`, each synchronized batch also logs `elapsed_us`, `bandwidth_GBps`, and `bandwidth_gbps`. At iteration completion, zero-byte batches are counted as synchronized batches but excluded from bandwidth samples. The proxy reports the arithmetic mean of non-empty synchronized batch bandwidths as `average_batch_bandwidth_GBps` / `average_batch_bandwidth_gbps` and the aggregate `total_forwarded_bytes / non_empty_synchronized_seconds` as `aggregate_synchronized_bandwidth_GBps` / `aggregate_synchronized_bandwidth_gbps`.

Set `nvlink_forward_completion_notifications_enabled=true` to let a source GPU proxy notify destination GPU proxies after synchronized forwarding batches complete. This feature requires both `nvlink_forwarding_enabled=true` and `nvlink_forward_synchronize_batches=true`. Each receiver GPU proxy creates one POSIX shared-memory segment with one single-producer/single-consumer queue for every local source GPU, including itself. With 8 GPUs per node, each proxy therefore owns 8 inbound queues. With ping-pong forwarding disabled, after the sender's `cudaStreamSynchronize()` returns for a forwarding batch, the forwarding thread enqueues notification records into an in-process handoff queue and can continue to later forwarding batches. A sender-side notification dispatch thread polls that handoff queue and pushes notifications into the receiver queues for the GPUs that received forwarded bytes. In router mode it also pushes a self-notification for the newly complete direct-RDMA `x3` range without issuing an NVLink copy. A receiver-side polling thread dequeues all notifications through the same path. Before publishing direct-input work, the sender calls `cuFlushGPUDirectRDMAWrites(...TO_OWNER)` when native device ordering does not cover owner-GPU consumers, so the already-running GroupedGEMM can safely consume the RDMA-written rows. CUDA builds wrap that call in an NVTX range named `RDMA proxy: cuFlushGPUDirectRDMAWrites(TO_OWNER), cuda_device=N`; collect with `--trace=nvtx` and inspect the forwarding CPU thread's NVTX row.

Each notification includes the source GPU, destination GPU, remote peer rank/slot, iteration, batch index, destination `start_token`, `num_tokens`, byte offset, and byte length. The `start_token` and `num_tokens` identify the continuous ready span inside the receiver's NVLink receive buffer for that source GPU and peer-node slot. Use `nvlink_forward_notification_queue_depth` to size each per-source queue.

Set `nvlink_forward_notification_log_enabled=true` to record receiver-side notification dequeue events to files. This mode requires `nvlink_forward_completion_notifications_enabled=true`. When the receiver notification thread dequeues an entry, it formats the notification into a single text line with `dequeue_timestamp_ns` and appends that line to an in-process log queue. During shutdown, each GPU proxy drains its log queue and writes one file under `nvlink_forward_notification_log_dir`, named `nvlink_forward_notifications_rank_<rank>_gpu_<gpu>.log`.

Set `nvlink_forward_local_batch_sync_enabled=true` to add a same-node GPU-proxy barrier before every synchronized NVLink forwarding batch. The barrier is keyed by phase, iteration, and batch round. Router local-input staging and remote receive-buffer forwarding use separate phases, so completing the staging phase does not prevent a proxy from participating in remote forwarding. Once a proxy has observed that its next batch is ready, it publishes that round in shared memory and waits until every local proxy is either ready for the same round or has completed that phase for the iteration. A completed proxy retires from later rounds in that phase, allowing router-driven proxies with shorter `x3`/`x4` arrays to stop participating while the remaining proxies continue together. With ping-pong forwarding disabled, each forwarding path reaches the next batch-start barrier only after the previous batch's `cudaStreamSynchronize()` has returned, so active proxies still start each round together. This option requires `nvlink_forward_synchronize_batches=true` and uses the same local shared-memory run identity as `local_iteration_sync_run_id`.

### Opt-in ping-pong forwarding

Set `nvlink_forward_ping_pong_enabled=true` to alternate remote RDMA forwarding
batches between two CPU threads, A and B, on the existing forwarding stream.
The default is `false`, which keeps the original batch synchronization and
notification behavior. Local router staging continues to use its original path.

The mode requires router routing, ordered chunks, the batch copy API, local batch
synchronization, and completion notifications. Keep
`nvlink_forward_synchronize_batches=true`; in this mode remote batches use
batch-end events for their completion waits instead of `cudaStreamSynchronize`.
Round-robin forwarding and out-of-order chunks are not supported by this mode.
Enable the same mode on every local proxy; mismatched notification protocols are
rejected during initialization.

`nvlink_forward_ping_pong_handoff_copy` is the one-based index of the nonempty
`cudaMemcpyBatchAsync` call that triggers the next forwarding thread (default
`1`). A remote batch submits one call per nonempty destination, with multiple
copy runs inside each call. The trigger event is recorded at that stream position,
but its handle is released to the other thread only after all copies, the batch-end
event, the complete notification batch, and compaction/submission cursors are committed.
The submitting thread then waits for batch-end completion outside the forwarding-stream
mutex before waiting for the other thread to return the turn.
The next thread waits for that event and for its next RDMA range to be ready,
then participates in the next local batch barrier, including the existing
minimum-available-chunk selection. It can submit while the preceding batch is
still copying; execution remains ordered by the shared stream.

If a batch has fewer calls than the configured index, its batch-end event is the
handoff event. A batch requiring no NVLink copies also records an end marker and
passes the turn. Each iteration starts with A. Proxies with no remaining remote
work retire from the phase, so uneven and empty workloads do not require a
nonexistent successor event.

Each remote batch enqueues one dispatch entry containing its batch-end event and
an ordered notification vector: direct same-GPU RDMA input first, then nonempty
destinations in copy-submission order. The dispatcher waits once for the batch-end
event, then publishes the entire vector before popping the next batch. There are
no per-destination notification events. Before starting forwarding threads, the
proxy queries `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING` once for its
CUDA device and caches whether a `TO_OWNER` flush is required. Native `OWNER` or
`ALL_DEVICES` ordering skips the flush entirely; otherwise direct inputs retain
the GPUDirect RDMA visibility flush after all destination copy submissions and
before recording the batch-end event and enqueueing notifications. The same
cached decision gates the non-ping-pong path's existing end-of-batch flush.
Initialization logs the ordering value and `owner_flush_required`; a failed
query fails initialization rather than silently skipping the flush. Mock mode
requires no CUDA query or flush. Local-staging batches use the same vector
dispatch path with no event because their copies are already synchronized.
Batch-end gating reduces event overhead but delays early destinations and direct
inputs until all copies in the batch complete, which can affect GEMM overlap.
Forwarding events are preallocated during initialization in a fixed pool of
`2 * num_gpus_per_node` events per proxy (14 for the seven-GPU torchrun scripts).
A producer reserves two slots before submitting a batch: one handoff marker and
one batch-end marker. The submitting thread and dispatcher share the batch-end
event. Short or zero-copy batches share that event with the next thread too, and
return the unused handoff reservation.
Dispatcher backlog can retain older events, so exhaustion waits for capacity
with `completion_timeout_ms`; shutdown or a forwarding error wakes pool waiters.
The pool never grows during forwarding. Recorded events return only after
successful completion and release of every consumer reference; failed or
unconfirmed recordings remain quarantined until shutdown.

The dispatcher, next producer, and submitting producer use `cudaEventSynchronize`
at their batch-end, handoff, and batch-end wait locations, respectively.
Shared-event completion is cached, so concurrent consumers do not repeat the CUDA wait. There is
no per-copy wait in the submitting producer. Events retain `cudaEventDisableTiming`
without `cudaEventBlockingSync`, so CUDA may busy-wait internally. A CUDA event
synchronization has no timeout or cancellation; unlike the former query loop,
`completion_timeout_ms` cannot interrupt it, and a stuck GPU can delay shutdown.

Event handles remain inside the source process. In this mode
`nvlink_forward_notification_queue_depth` also bounds the pending in-process
notification queue in units of batches; shared-memory receiver queues still count
individual notifications. Iteration completion waits for submitted batches to
finish and for all local sources' iteration notifications to be consumed before
returning to the embedding caller.

For the checked-in launchers, append the local-sync override explicitly because
the scripts default it to false:

```bash
# Node 0
RDMA_CPU_Proxy/scripts/run_node0_torchrun.sh --concurrent-kernel \
  --nvlink_forward_local_batch_sync_enabled=true \
  --nvlink_forward_ping_pong_enabled=true \
  --nvlink_forward_ping_pong_handoff_copy=1

# Node 1
RDMA_CPU_Proxy/scripts/run_node1_torchrun.sh --concurrent-kernel \
  --nvlink_forward_local_batch_sync_enabled=true \
  --nvlink_forward_ping_pong_enabled=true \
  --nvlink_forward_ping_pong_handoff_copy=1
```

The overlap can hide the barrier only when enough GPU copy work remains after
the handoff becomes usable. It coordinates CPU submissions; actual batch start
times can differ between GPUs. Profile the copy timeline and GroupedGEMM
throughput on Hopper when tuning the handoff index. With
`nvlink_forward_log_batches=true`, completion logs identify the A/B lane.
Per-batch host elapsed times can overlap in this mode, so their summed durations
are not an iteration wall-clock bandwidth measurement.

`test_ping_pong` exercises legacy and ping-pong mock paths, unequal workloads,
dynamic tails, direct-only batches, empty remote phases, and queue backpressure.
`test_forward_events` checks event lifetime guards locally and additionally checks
asynchronous stream-prefix completion in a CUDA build with an available GPU.
Run it on a remote CUDA node with
`ctest --test-dir RDMA_CPU_Proxy/build-hopper -R test_forward_events --output-on-failure`.
This event test does not require an RDMA NIC.

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

## Shared Library and torchrun

The build also produces a shared library for embedding the proxy in another
process:

```text
RDMA_CPU_Proxy/build-hopper/librdma_cpu_proxy.so
```

Build both the legacy executable and the shared library with:

```bash
cmake -S RDMA_CPU_Proxy -B RDMA_CPU_Proxy/build-hopper \
  -DRDMA_PROXY_REQUIRE_CUDA=ON \
  -DRDMA_PROXY_REQUIRE_VERBS=ON
cmake --build RDMA_CPU_Proxy/build-hopper \
  --target rdma_cpu_proxy rdma_cpu_proxy_shared
```

The executable remains available at
`RDMA_CPU_Proxy/build-hopper/rdma_cpu_proxy`. The shared library exports the
stable C interface declared in `include/rdma_proxy_c_api.h`. Its opaque handle
API supports separate create, initialize, run, shutdown, and destroy calls;
`rdma_proxy_run_argv` provides a complete blocking lifecycle for embedders that
want the proxy to retain its normal internal allocation behavior.

The supplied torchrun worker loads the library with `ctypes.CDLL`. It derives
the proxy topology from `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `RANK`, `WORLD_SIZE`,
and `GROUP_RANK`, then appends the per-worker GPU, NIC, and TCP-port overrides
to the normal proxy command-line arguments. It does not initialize a Torch
distributed process group; torchrun is used only to launch and monitor one
Python/proxy process per GPU.

The worker installs `rdma_proxy_set_device_buffer_allocator` before proxy
initialization. When C++ needs an RDMA send/receive buffer, source-specific
NVLink receive buffer, padded router `A_idx`, gather table, or ready-row flag,
it sends Python a typed shape/size request. Python allocates an ordinary CUDA
`torch.Tensor`, returns its `data_ptr()`, and retains the tensor through proxy
shutdown. C++ borrows and registers those pointers but never calls `cudaFree`
on them. Router-dependent shapes are therefore still computed by the existing
C++ metadata exchange rather than duplicated in Python. This ownership change
applies only to the supplied Python worker; the standalone executable and
`rdma_proxy_run_argv` continue to allocate their buffers internally. The worker
requires CUDA-enabled PyTorch in both normal and concurrent-kernel modes.

For the checked-in two-node, seven-GPU launch, start node 0 and node 1 with:

```bash
RDMA_CPU_Proxy/scripts/run_node0_torchrun.sh
RDMA_CPU_Proxy/scripts/run_node1_torchrun.sh
```

The scripts preserve the original mapping of logical GPUs `0..6` to physical
CUDA devices `0,1,2,3,5,6,7`, including the corresponding RDMA devices. They
also pass the same proxy workload options as `run_node0.sh` and `run_node1.sh`.
The rendezvous defaults to node 0 at `28.49.38.169:29500`; override it when
needed:

```bash
MASTER_ADDR=<node-0-address> MASTER_PORT=<unused-port> \
  RDMA_CPU_Proxy/scripts/run_node0_torchrun.sh
```

Set `RDMA_PROXY_BUILD_DIR` if the shared library is in a build directory other
than `RDMA_CPU_Proxy/build-hopper`, or `TORCHRUN_BIN` to select a different
torchrun executable. Use a fixed node count and `--max_restarts=0`; the proxy's
peer connections and synchronization barriers assume fixed membership.

### Python-controlled concurrent kernel

Pass `--concurrent-kernel` to opt into a Python-controlled iteration loop:

```bash
RDMA_CPU_Proxy/scripts/run_node0_torchrun.sh --concurrent-kernel
RDMA_CPU_Proxy/scripts/run_node1_torchrun.sh --concurrent-kernel
```

Without this option, the worker calls the handle-based blocking `rdma_proxy_run`
entry point. With the option enabled, Python instead owns the iteration loop,
leaving the initialized C++ QP/CQ/forwarding threads alive. At each
iteration a persistent Python coordinator thread and the Python main thread are
released from the same barrier. Before releasing them, the main thread calls
`rdma_proxy_prepare_iteration`, which resets the proxy-owned gather ready flag
to zero and waits until that reset is GPU-visible. The coordinator then calls
`rdma_proxy_run_iteration`, while the main thread launches QuACK's persistent
multi-buffer GroupedGEMM on a dedicated PyTorch stream. Its inputs are the
Python-owned NVLink receive tensors, padded `A_idx` tensors, dynamically
published gather table, and ready-row flag.

Concurrent mode enables both `router_computation_forwarded_inputs_only` and
`router_local_input_staging_enabled`. For the supplied 2-node x 7-GPU launch,
GroupedGEMM therefore consumes all 14 matching `(X, A_idx)` pairs per GPU: seven
remote-node inputs and seven staged/forwarded local-node inputs. Python allocates and randomly
initializes local expert weights once, then allocates an output tensor sized for
the padded route capacity. The default QuACK repository is the sibling
`quack_for_communication_overlap` checkout; override it with `--quack-root`.
The first iteration includes QuACK compilation overhead.

Each GPU process also reports per-iteration effective throughput at the end of
the run. C++ records a CUDA start event on its publication stream immediately
before the first non-empty gather-table copy. Python asks C++ to record the end
event on the QuACK stream directly after the GroupedGEMM launch, so that event
executes when the kernel finishes. The reported useful work is
`2 * active_routed_tokens * K * N`; padded `A_idx` capacity is not counted.
Because the interval begins at the first table flush, it includes readiness
stalling and table streaming after that point, not just GEMM math.

Concurrent-kernel mode requires a Hopper SM90 GPU, CUDA-enabled PyTorch, QuACK,
`--num-experts`, and a finite nonzero `num_iterations`. Tile/output parameters
default to the proxy scheduler values and can be overridden with the
`--grouped-gemm-*` worker options; the worker forwards matching scheduler values
to C++. The shell launchers forward additional arguments to the worker.

### Fused up-projection activation

Add `--grouped-gemm-activation=swiglu` (or an ordinary activation such as `relu`)
alongside `--concurrent-kernel` on both nodes. For example, append these options
to the existing forwarding configuration:

```bash
--concurrent-kernel --grouped-gemm-activation=swiglu \
  --grouped-gemm-output-dimension=8192
```

The default activation is `none`, preserving the plain GEMM launch. Supported
ordinary activations are `silu`, `silu-tanh`, `relu`, `relu_sq`,
`gelu_tanh_approx`, and `tanh`. Gated activations are `swiglu`, `swiglu-tanh`,
`swiglu_oai`, `swiglu_oai-tanh`, `reglu`, `geglu`, and `glu`.

The output-dimension option always specifies the **post-activation width** H.
Ordinary activations use weights shaped `[E, K, H]`. Gated activations use
logical weights `[E, K, 2H]`, containing gate and up projections, and produce
`[R, H]` output. The worker forwards the actual projection width (2H for gated
activations) as `expert_gemm_dimension` to C++, so the published table covers
both projections. Thus SwiGLU with H=8192 schedules a GEMM width of 16384.
Python's explicit tile/cluster/swizzle options continue to override the JSON
scheduler settings in concurrent mode.

Activation uses QuACK's existing `gemm_act` fused epilogue with an explicit
Hopper configuration, the same multi-buffer gather table and ready-row flag,
and no saved pre-activation tensor. No separate activation kernel, down
projection, or token-combine stage is launched. No QuACK or C++ implementation
changes are needed. Timing includes the fused epilogue; effective TFLOP/s counts
only GEMM arithmetic, including both projections for gated activations.

Run the worker's CLI checks and numerical Hopper tests with:

```bash
python -m unittest discover -s RDMA_CPU_Proxy/tests -p test_worker_activation.py -v
```

Run from the parent of `RDMA_CPU_Proxy`. Set `QUACK_ROOT` for a non-sibling QuACK
checkout. Numerical tests compare plain GEMM, ReLU, and SwiGLU against FP32
references with multiple input buffers and ragged expert batches. They require
Hopper and CUDA PyTorch and are skipped otherwise. The full two-node run is
still required to validate RDMA/activation overlap.

### Configurable down projection

Add `--down-projection2` to concurrent mode with a fused activation to run a
plain grouped down GEMM after the up GEMM on the same CUDA stream. The down
input is only the meaningful expert-contiguous prefix of the activated output;
padding in the up allocation is excluded. Down weights have logical shape
`[local_experts, activated_width, token_dimension]` and are initialized once.
The up and down GEMMs execute sequentially; down wave barriers do not provide
up/down overlap.

Rebuild `librdma_cpu_proxy` on both Hopper nodes: this feature adds the
`rdma_proxy_get_computation_expert_counts` C ABI query. No QuACK changes are
required, but the selected checkout must contain `run/hopper_pregather_gemm.py`:
the worker reuses its table/scatter preparation and kernel launch helpers.

Append the following options to both existing node commands:

```bash
--concurrent-kernel --grouped-gemm-activation=swiglu \
--grouped-gemm-output-dimension=8192 --down-projection2 \
--down-tile-m=128 --down-tile-n=128 --down-pingpong \
--down-scatter-table --down-n-tile-major-output \
--down-multi-buffer-scatter --down-scatter-output-buffers=4 \
--down-gather-table --down-gather-table-expert-waves \
--down-max-swizzle-size=1 --down-gather-table-wave-barrier \
--down-epilogue-store=bulk_rows_reduce
```

Down defaults are tile M/N=128/128, cluster M=1, swizzle=1, pingpong disabled,
and `tma` stores. `--down-cluster-m` also permits explicit cluster sizing.
Add `--down-sm-count=40` to cap the down GEMM's persistent CTA count. The value
must be positive and divisible by `--down-cluster-m`; omitting it keeps QuACK's
default hardware capacity. This limits launch parallelism without reserving
specific physical SMs and does not affect the up GEMM. The same cap is used
when preparing down expert-wave metadata. It requires a QuACK checkout whose
`run/hopper_pregather_gemm.py` supports `--sm-count`. Only the Python worker
needs updating for this option; no C++ library rebuild is required.

Without the down gather table, the kernel uses per-expert sequence lengths.
With it, a separate static table and identity indices describe the materialized
activated input. Expert waves require that table; the wave barrier additionally
requires pingpong and a bulk-row epilogue. QuACK validates device-dependent wave
capacity and tile/layout constraints. No down options are forwarded to C++.

Add `--down-gather-table-publish-ready` to publish completed down-output column
groups to `DownProjectionRunner.inputs.ready_groups`, a GPU `int32[1]` flag.
It requires `--down-gather-table-expert-waves` and
`--down-gather-table-wave-barrier`, including their gather-table, pingpong, and
bulk-row epilogue prerequisites. For example, append the flag to the down
projection command above. The worker passes it to QuACK's existing preparation
and launch helpers, which reset the readiness flag on every down invocation.
A GPU consumer must acquire-load the flag before reading the completed column
prefix; each group spans `effective_swizzle_size * down_tile_n` columns.
This publishes output readiness only; host tracing is separately enabled below.
Up and down GEMMs still run sequentially.
Use a QuACK checkout containing `--gather-table-publish-ready`; this worker
change does not require a C++ library rebuild.

To observe down-output readiness on the CPU, also append `--trace-down-ready`.
Rebuild `rdma_cpu_proxy_shared` on both Hopper nodes for this option. It requires
`--down-projection2`, `--down-gather-table-publish-ready`, and the existing
notification-driven concurrent mode. The proxy allocates a mapped pinned host
counter and passes its device address to QuACK's `wave_ready_host_address`.
QuACK publishes the completed prefix with system-scope release stores; the
existing NVLink notification receiver polls it with acquire loads. No additional
polling thread is created. Counter checks are interleaved with notification
processing, and continue after gather-table publication finishes.

Each iteration resets and arms the counter before GPU launches. After GPU
completion, Python waits for the receiver's final observation before preparing
the next iteration. Logs report completed group/column ranges and CPU detection
microseconds since the counter was armed during iteration preparation. These
times include up computation and host launch/compilation delays; they are not
GPU completion timestamps. A jump is recorded as one observation, and the first
iteration includes cold compilation. Polling uses `completion_timeout_ms`,
including the time spent in compilation and up computation; increase it if
needed for cold runs. Shutdown cancels polling. This feature only observes
readiness; it does not return token results or perform additional communication.

From `RDMA_CPU_Proxy`, rebuild on each remote node:

```bash
cmake --build build-hopper --target rdma_cpu_proxy_shared -j4
```

Append the following to both existing node commands to enable a SwiGLU up
projection, local down-output combine, and CPU readiness tracing:

```bash
--concurrent-kernel --grouped-gemm-activation=swiglu \
--down-projection2 --local-gpu-combine \
--down-n-tile-major-output --down-tile-m=128 --down-tile-n=128 \
--down-pingpong --down-gather-table --down-gather-table-expert-waves \
--down-max-swizzle-size=1 --down-gather-table-wave-barrier \
--down-gather-table-publish-ready --trace-down-ready
```

`--local-gpu-combine` selects the additive bulk-row epilogue required by this
configuration. It combines into local per-source buffers without routing
weights. Add `--return-token-results` below to return these outputs to their
same-node NVLink senders.

Scatter matches the standalone benchmark: a random permutation distributes
routed rows across separately allocated local outputs. It does not map results
back to original tokens, apply routing weights, or return results over RDMA.
Although `bulk_rows_reduce` uses additive stores, permutation destinations do
not collide. Destination buffers are zeroed before every down invocation to
prevent accumulation across iterations. N-tile-major buffers have shape
`[ceil(token_dimension / down_tile_n), rows_in_buffer * down_tile_n]`; they are
not row-major tensors and require layout conversion for ordinary consumers.

Existing up-GEMM timings remain up-only. Additional reports show down output
reset time, down GEMM time/TFLOP/s, and `up_plus_down_ms`, the sum of measured
stages (including reset, excluding gaps between stages). Down FLOPs are
`2 * meaningful_routes * activated_width * token_dimension`. Cold compilation
can inflate initial event timings; use later iterations for performance analysis.

### Local GPU combine without routing weights

Append `--local-gpu-combine` together with `--down-projection2` and a fused
up activation to the existing concurrent commands on both nodes, for example:

```bash
--concurrent-kernel --grouped-gemm-activation=swiglu \
--down-projection2 --local-gpu-combine
```

This option selects multi-buffer scatter and `bulk_rows_reduce`, replacing the
benchmark permutation. It owns those settings, including the destination-buffer
count: `--down-scatter-output-buffers` does not control combine outputs. The
existing down tile, pingpong, gather-table, expert-wave, wave-barrier, and
N-tile-major options remain available with their usual constraints.

Each computing GPU allocates one separate output buffer per active source
`(node, GPU)` in the up runner's source order: 16 buffers for the two-node,
eight-GPU configuration. Each has the same logical token-row capacity and
feature width as its corresponding input. Inputs are never overwritten.
Outputs are row-major by default; `--down-n-tile-major-output` uses the existing
QuACK tiled layout. `DownProjectionRunner.inputs.output_buffers()` exposes the
buffers; `down.up.source_pairs` identifies their sources.

Each output stores only a compact prefix of logically valid results. Distinct
input rows used by any local expert are sorted in ascending input-row order;
output row k sums all local expert contributions for the k-th such input row.
For example, used input rows [2, 5, 9] produce output rows [0, 1, 2], regardless
of expert or arrival order. Multiple experts using row 5 still share output row 1.
Already compact NVLink inputs retain their row positions. Local-staging and
direct-RDMA inputs lose any gaps between locally used rows.

Allocation capacity remains equal to input capacity. All destinations are zeroed
each iteration, including the unused tail. The GPU int64 tensor
`down.combine_valid_rows[source]` gives the valid prefix length; the GPU int64
tensor `down.combine_output_to_input[source, k]` gives its input-buffer row
(-1 beyond the valid prefix). These tensors update on the down stream each
iteration; consumers must wait for that stream before reading them. N-tile-major
stores use `valid_rows[source] * tile_n * element_size` as the N-tile byte stride.
Thus all valid results occupy one contiguous allocation prefix in both layouts:
`n * feature_width` elements for row-major and
`ceil(feature_width / tile_n) * n * tile_n` for tiled output. Tiled output remains
N-tile-major, with feature padding if the width is not divisible by tile_n;
view its prefix as `[num_n_tiles, n, tile_n]`, not using the allocation's original
capacity-based shape. An empty source has a zero-length prefix and no stores.
`down.collected_combine_outputs()` synchronizes the last launch and returns one
row-major valid tensor per source for inspection. Do not use QuACK's generic
`inputs.collected_up_output()` for combine: it assumes capacity-based strides.
Input-row indices are not necessarily original source-token indices. No
cross-GPU reduction or routing weights are applied. Return transfer is opt-in
with `--return-token-results`.

The up scheduler's packed row order can change with notification arrival order.
After up completes on its CUDA stream, the worker reconstructs each output row's
source and token from that iteration's gather table and `A_idx` arrays. Repeated
N groups are removed; source segments are ordered by their packed output offsets.
Only those meaningful routes determine compaction; padded A_idx entries are
excluded. A per-source used-row mask and prefix sum assign compact destinations.
The resulting scatter addresses are rebuilt on the GPU every iteration before
down launches on the same stream. Address lookup for all valid destination rows
is validated once during initialization. No host readback of the live table is
needed. This adds GPU mapping work between up and down, reported separately as
`mapping_ms`; `up_plus_down_ms` includes reset, mapping, and down time in addition
to the existing up timing.

### Intra-node combine concurrent with down GEMM

Append `--intra-node-combine` to the prepared, per-group return configuration
on every rank. It uses a register-prefetch CuTe kernel adapted from QuACK's
`run/final_combine.py`, with striped 128-bit loads and FP32 accumulation in
computing-GPU order. QuACK itself does not need source changes or reinstalling.
Rebuild `rdma_cpu_proxy_shared` on both Hopper nodes for the new routing-mask
query, and deploy the updated Python scripts together.

For the two-node scripts, retain the prepared-forwarding options and append:

```bash
--concurrent-kernel --down-projection2 --local-gpu-combine \
--down-n-tile-major-output --down-tile-m=128 --down-tile-n=128 \
--down-pingpong --down-gather-table --down-gather-table-expert-waves \
--down-max-swizzle-size=1 --down-gather-table-wave-barrier \
--down-gather-table-publish-ready \
--return-token-results --return-token-results-per-group \
--intra-node-combine --down-sm-count=40 \
--check-intra-node-combine
```

The new flag enables both native return-completion notifications and HBM ready
publication. Their prepared submit/epilogue prerequisites still apply.
`--check-intra-node-combine` is optional and compares every iteration with an
independent FP32 PyTorch reference after all returns finish; omit it for timing.
The single-node script already enables the required return path, so append
`--intra-node-combine` (and optionally the check).

Defaults are four persistent CTAs, 512 threads/CTA, 32 elements/thread, and four
prefetched peers. Tune with `--intra-node-combine-ctas`,
`--intra-node-combine-threads-per-cta`,
`--intra-node-combine-elements-per-thread`, and
`--intra-node-combine-prefetch-peers`. The kernel requires BF16/FP16 tiled inputs
with feature width divisible by down tile N and tile N divisible by elements/thread.
Column groups may contain multiple N tiles; readiness uses the down swizzle group
width. `--down-sm-count` defaults to 40 when intra-node combine is enabled. The
worker checks that the down CTA budget leaves room for the combine CTAs (one
whole cluster reserved per combine CTA). These are launch budgets, not SM affinity.

The combine stream waits for an event recorded after down-output reset and
mapping, before down GEMM. It then polls the native HBM ready prefix with
system-acquire loads while down GEMM runs on its own stream. A group is consumed
only after all peer returns and the two retained outputs are ready. The existing
seven peer copy batches remain unchanged. Iteration completion waits for combine
and a same-node barrier before any rank can reuse incoming buffers. A separate
cancellation flag releases polling during exceptional teardown.

The C ABI query `rdma_proxy_get_intra_node_source_masks` exports immutable routing
masks in full staging/RDMA input-row order, with bit g identifying computing GPU g.
It uses actual received remote routing metadata, not an assumption of equal seeds.
For each computing GPU, ascending selected rows map to its compact contribution
rows, including the compact retained self output. Mapping and compilation happen
once during setup; this benchmark's routing and source order are fixed across
iterations. There is no runtime exchange of row maps or per-group transpose.

`down.intra_node_combine.outputs[source_node]` exposes a row-major tensor of shape
`[populated_staging_or_RDMA_rows, token_dimension]` by default. Append
`--intra-node-combine-n-tile-major-output` (requires `--intra-node-combine`) to
write each source directly as a contiguous tensor of shape
`[token_dimension / down_tile_n, populated_input_rows, down_tile_n]`.
The output tile width equals `--down-tile-n`. No transpose kernel is launched.
`--check-intra-node-combine` supports either layout.

For two nodes, one tensor is the intra-node sum for local staging and one is the
intra-node sum for the local RDMA input. They are views into one output allocation,
with each source packed in its own contiguous span, and are valid after combine
completion. In tiled mode `down.intra_node_combine.output` is the flat backing
allocation; use the per-source views rather than a global N-tile-major reshape.
For a source with R populated rows, a group starting at N tile t and containing
q tiles occupies byte offset `t * R * down_tile_n * element_bytes` and byte count
`q * R * down_tile_n * element_bytes` relative to that source view. Thus even a
multi-tile group is contiguous independently for each RDMA peer. Empty sources
have zero-length spans. Row order within each tile remains the full input order.

The layout option changes storage only. Add
`--intra-node-combine-publish-ready` to publish output completion as described
below. The input HBM ready prefix must not be used as an output-completion signal.
The kernel reduces 14 received plus two retained contributions in one launch;
single-node mode reduces seven received plus one retained contribution.
This adds no routing weights, reverse RDMA, or final cross-node reduction.
Existing GEMM event timings exclude combine completion and its additional barrier,
although concurrent execution can affect down-GEMM performance.

Run the standalone numerical/readiness test on Hopper from this repository:

```bash
RDMA_PROXY_TEST_CUDA=1 QUACK_ROOT=/path/to/quack_for_communication_overlap \
python -m unittest discover -s RDMA_CPU_Proxy/tests -p test_intra_node_combine.py -v
```

### Intra-node combine output-ready prefix

Add `--intra-node-combine-publish-ready` alongside `--intra-node-combine` on every
rank. It supports both output layouts; use
`--intra-node-combine-n-tile-major-output` for contiguous per-source column-group
spans suitable for a later RDMA return.

`down.intra_node_combine.ready_groups` is a stable, one-element CUDA `int32`
tensor in local GPU HBM (`None` when disabled). Its `.data_ptr()` exposes the
address for a future consumer. Value N means output column groups `[0,N)` are
complete for **all source outputs**, including staging and every RDMA peer.
Group width is `min(down_max_swizzle_size, num_N_tiles) * down_tile_n`; a final
group may be shorter. This flag is separate from the native input-ready prefix.
No reverse RDMA transfers or CPU notifications are added by this option.

With publication enabled, the combine kernel assigns work separately for each
column group. All CTAs, including CTAs with no work in that group, execute a
block barrier and release-store their completion slot in HBM. CTA 0 acquire-loads
all slots and system-release-stores the completed prefix. CTA 0 is the only
publisher and processes groups in order, so the prefix cannot regress or skip
unfinished groups. Other CTAs can proceed to later groups without waiting for
CTA 0. Ordinary output stores precede publication; consumers must use an acquire
load/polling protocol (system scope for cross-device use), not an ordinary cached
load. Empty outputs have no payload and complete vacuously. Cancellation releases
polling without marking unfinished groups complete.

The worker resets the prefix and all CTA slots to zero before proxy iteration
preparation, and synchronizes this reset before allowing new work. The addresses
remain stable across iterations. Embedders must finish all output consumers
before the next iteration/reset or teardown. The worker already waits for combine
completion and its same-node barrier; integrating reverse RDMA must also join
those transfers before buffer reuse. Disabled mode allocates no output-ready flag
or slots and keeps its original work distribution.

`--check-intra-node-combine` also checks that the final prefix equals
`down.intra_node_combine.num_groups`. The opt-in Hopper test consumes each group
as soon as its prefix appears, before releasing the next input group, and compares
it to an FP32 reference. It covers row-major/tiled outputs, iteration reuse,
partial final groups, idle CTAs, and cancellation. Only Python/CuTe files change
for this feature; no additional C++ rebuild is needed.

### Return intra-node combine results over RDMA

Rebuild the shared library on both Hopper nodes. Add these options to the existing
multi-node down-GEMM / NVLink-return command on **every rank**:

```bash
--intra-node-combine \
--intra-node-combine-n-tile-major-output \
--intra-node-combine-publish-ready \
--rdma-return-token-results
```

`--rdma-return-token-results` is opt-in and requires at least two nodes and the
three combine options above (with their existing down-GEMM/NVLink prerequisites).
It keeps the HBM output-ready prefix and additionally system-release-publishes
that prefix to native-owned, CUDA-mapped pinned CPU memory. The coordinator
acquire-polls this host flag during forward waits and while draining returns.
Each newly completed group releases a precomputed chunk distributor to the
existing RDMA send workers. No extra send or CQ threads are created. A worker
finishes its current forward distributor before processing queued returns, so
returns can overlap down-GEMM but do not preempt that QP's forward transfer.

At setup, each remote-source combine output is registered directly as an RDMA
source; no packing/copy kernel is added. A separate GPU receive buffer is
allocated and registered for each remote node, and peers exchange its address,
rkey, and layout. Each group is contiguous in that source's N-tile-major span.
`--rdma-return-chunk-bytes` (default 1048576) splits that span into work for the
existing QPs; it does not change the column-group size. Empty domains still
exchange completion markers. Return immediates/counters are separate from
forward arrivals and never trigger forward NVLink copies.

After the proxy iteration completes, received tensors are accessible through
`down.intra_node_combine.rdma_return.received[peer_node]`, with shape
`[token_dimension / down_tile_n, outgoing_x3_rows_for_peer, down_tile_n]`.
Their rows follow this GPU's original outgoing x3 order to that node. In the
2-node case there is one such receive tensor per GPU; the locally retained
staging combine output remains in `down.intra_node_combine.outputs[local_node]`.
This step transports node sums back; it does **not** perform the final reduction
across nodes or scatter them into original source-token order.

Every QP sends a generation-tagged group-end marker. Before returning from an
iteration the coordinator waits for every outgoing and incoming return marker
and performs the required GPUDirect receive-visibility flush. The next iteration
resets the host prefix only after prior transfers and GPU work have joined.
Registered storage remains alive until native shutdown.

Add `--check-rdma-return-token-results` on every rank for a post-iteration SHA256
comparison of each received tensor against the peer's registered source. It
copies bytes to CPU and exchanges hashes over the control channel; use it for
correctness runs, not performance measurements. Combine it with
`--check-intra-node-combine` to validate both numerical reduction and transport.
The standalone opt-in Hopper combine test also exercises mapped-host publication.

### Return local-combine results to NVLink senders

Rebuild the shared library on both Hopper nodes:

```bash
cmake --build build-hopper --target rdma_cpu_proxy_shared -j4
```

Append the following to both existing node commands, retaining your prepared
forwarding options:

```bash
--concurrent-kernel --grouped-gemm-activation=swiglu \
--down-projection2 --local-gpu-combine \
--down-n-tile-major-output --down-tile-m=128 --down-tile-n=128 \
--down-pingpong --down-gather-table --down-gather-table-expert-waves \
--down-max-swizzle-size=1 --down-gather-table-wave-barrier \
--down-gather-table-publish-ready \
--return-token-results --return-token-results-per-group
```

Omit `--return-token-results-per-group` for whole-buffer returns after down
completion. Whole-buffer mode permits row-major output too. Both modes require
local combine and down ready-prefix publication (including its existing
prerequisites). `--trace-down-ready` is optional: return mode enables the mapped
host counter and receiver polling automatically, without requiring trace logs.
The QuACK checkout must include `run/token_exchange.py` and the CUDA IPC export
helper in `run/hopper_stream_gather_table_gemm.py`; no QuACK source changes or
QuACK CPU-proxy rebuild are needed.

On node A GPU i, each source `(source_node, GPU j)` with `j != i` is returned to
node A GPU j. In the two-node/eight-GPU case this means 14 outputs, grouped into
seven `cudaMemcpyBatchAsync` calls, each containing two independent segments.
The local-staging and direct-RDMA outputs remain on GPU i. Each destination
allocates 14 separate persistent receive buffers keyed by
`(computing_peer_gpu, original_source_node)`. The allocation capacity matches
the combine outputs; only valid prefixes are copied. At 10,240 rows, width
4,096 and BF16 this adds about 1.09 GiB of receive storage per GPU.

Source and destination use identical physical layouts. For tiled output,
each valid prefix is `[ceil(width / tile_n), valid_rows, tile_n]`, independent
of the capacity-based allocation shape. Each group's byte offset and size
use that source's valid-row count; the two segments in a peer batch may differ
in length. Empty segments and all-empty batches are skipped. Receive tails
are unspecified and must not be consumed.

The existing notification receiver submits returns on a dedicated nonblocking
CUDA stream. Per-group mode submits every newly ready group exactly once, even
when a poll observes a prefix jump. After submitting a group's batches to all
destination GPUs, it calls `cudaStreamSynchronize` on the return stream before
submitting the next ready group. This blocks the notification receiver until
that group's copies complete; down GEMM can continue on its separate compute
stream.
Whole-buffer mode additionally queries an event recorded after down GEMM;
the iteration is armed only after that event is recorded, preventing reuse of
the previous iteration's event state. Compact valid counts are copied to pinned
host storage on the compute stream after mapping and before down GEMM. A
positive system-release ready prefix therefore also establishes count readiness.

In whole-buffer mode, the receiver continues polling until the return stream
completes; per-group mode waits synchronously for each group, including the last.
Python waits for return completion and a same-node rank barrier
before starting the next iteration. The barrier also exchanges valid counts.
`down.return_exchange.received[(peer, source_node)]` exposes the receive tensor;
`received_valid_rows[(peer, source_node)]` gives its current valid count after
the barrier. Exported allocations and imported mappings remain alive until all
workers stop and all imports close. Non-identity CUDA device maps are supported;
all workers must use consistent CUDA visibility and distinct devices.

Return transfers and the rank barrier are excluded from the existing GEMM event
timings, although overlapping transfers can affect down-GEMM performance. The
small count readback is included in mapping time. This feature performs only the
reverse intra-node NVLink hop: remote-origin results still need a later reverse
RDMA transfer, and final token reduction/routing weights are not implemented.

To notify each destination GPU's forwarding thread after every returned column
group, append this option on every local rank (on both nodes for a two-node run):

```bash
--nvlink_return_completion_notifications_enabled=true
```

This option defaults to `false`; when disabled, no return notification queues,
messages, or incoming-prefix waits are added. It requires
`--return-token-results --return-token-results-per-group` and the existing
prepared submit/epilogue mode (`nvlink_forward_submit_epilogue_enabled=true`,
with preparation enabled and both ping-pong modes disabled). Both the atomic
ring and mutex prepared queue are supported. All local ranks must use the same
setting and column-group layout. `nvlink_forward_notification_queue_depth`
(default 1024) must hold at least one full iteration's column groups. This
keeps reverse notifications from blocking the dispatcher behind a full queue
while it still has forward notifications to send.

After the receiver successfully synchronizes a group's return stream, it hands
one group-completion envelope to the existing dispatcher. The dispatcher sends
`(iteration, column_group)` to every local GPU, including itself and peers whose
valid-row counts are zero. The self notification covers the retained outputs;
it issues no copy. The down-GEMM ready prefix was observed before submitting
that group's returns, so retained outputs are already ready at this point.
Each destination owns a separate shared-memory return queue for each producing
GPU, including self. These queues
are separate from the up-projection notification queues. Source identity is
implicit in the queue; one notification covers that group's copies for all
original-source-node buffers returned to this destination.

The forwarding submitter drains the return queues, including while waiting
for new forward work. It advances a per-peer contiguous prefix and publishes
their minimum to an atomic CPU-DRAM flag. A prefix of `N` means groups `[0,N)`
have arrived from every other local GPU and this GPU's retained outputs are
ready for those groups. In the two-node/eight-GPU case this covers 14 received
buffers plus two retained outputs, with the same seven peer copy batches.
It does not imply final reduction or reverse RDMA. No further GPU work is
launched by the consumer. Empty contributions still advance the prefix.

Rebuild the shared library and upgrade all local workers together: notification
protocol version 4 includes self readiness and rejects older workers. Self uses
the same ordered queue and retirement checks as peers. Its notification waits
for outgoing return copies to complete, conservatively delaying self readiness.

The prefix resets to zero during iteration preparation. Stale, duplicate, and
out-of-order messages are rejected; an early next-iteration message remains
queued until local preparation. C++ embedders can read
`Proxy::return_ready_prefix(iteration)`; the C ABI exposes
`rdma_proxy_get_return_ready_prefix(handle, iteration, &prefix)`. These are
nonblocking, iteration-qualified snapshots of CPU state, not GPU-readable
flags. When enabled, `read_down_ready_trace` also waits for all incoming
notifications to be consumed and all outgoing notifications to be dispatched
and consumed by their destinations.
The Python worker already calls this before its existing same-node barrier,
so normal iteration reuse and shutdown occur after notifications drain.
Exceptional shutdown cancels unfinished work rather than claiming completion.

To additionally publish the aggregate return-ready prefix into this GPU's HBM,
enable the following alongside return notifications:

```bash
--nvlink_return_completion_notifications_enabled=true \
--nvlink_return_ready_gpu_publication_enabled=true
```

GPU publication defaults to `false`. When disabled, the CPU-only return path
retains its existing behavior and allocates no publication flags or stream.
When enabled, the proxy allocates a pinned host `int32` and a separate HBM
`int32`. Each increase in the all-peer prefix updates the pinned host value
and enqueues `cuStreamWriteValue32(..., CU_STREAM_WRITE_VALUE_DEFAULT)` on a
dedicated nonblocking stream. This uses the same default memory-barrier mode
as the up-GEMM ready-row publication. There is no per-update stream
synchronization and no memcpy fallback. The value is passed directly to the
driver, so subsequent host-flag changes cannot overwrite an in-flight memcpy
source. Both the CPU and GPU values retain the `[0,N)` prefix meaning.

Iteration preparation publishes zero and synchronizes the publication stream
before allowing new notifications to advance the flag. Reset and publication
are serialized under the progress lock, including when next-iteration messages
arrive early. Iteration completion also waits for the final flag publication.
Shutdown stops the producers, drains the publication stream, and frees both
allocations. The stream has no dependency on the compute or return-copy streams.

After `configure_down_ready_trace`, embedders can obtain the stable addresses
with `Proxy::return_ready_publication_addresses()` (host, device), or the C ABI:

```cpp
uintptr_t host_flag = 0, device_flag = 0;
int status = rdma_proxy_get_return_ready_publication(handle, &host_flag, &device_flag);
```

The addresses are borrowed until shutdown and must not be written by consumers.
Use acquire/atomic host reads or the existing CPU prefix accessor. The host
snapshot describes an enqueued publication; the HBM update can lag it. GPU
consumers need a suitable acquire polling protocol, and must finish before
the next iteration preparation or shutdown. This option only publishes the
flag; it does not launch a GPU consumer, reduction, or reverse RDMA operation.

The local mock tests cover publication, reset, and notification integration.
To check actual pinned allocation and HBM writes on Hopper, build the
`test_return_ready_publication` target and run
`ctest --test-dir build-hopper -R '^test_return_ready_publication_cuda$' --output-on-failure`.

For remote validation, add `--check-return-token-results` on both nodes. Every
iteration compares each received valid physical prefix byte-for-byte with the
producing GPU's combine output, then waits for every rank to finish checking
before reuse. Run both whole-buffer and per-group modes; these checks validate
transport, while the existing Hopper numerical tests validate combine arithmetic.
Validation allocates a temporary comparison buffer and is outside GEMM timing.

`--return-token-timeout=120` controls same-node IPC setup/completion barriers in
seconds. The native ready/copy poll retains `completion_timeout_ms`; use a larger
value such as `--completion_timeout_ms=120000` if first-iteration compilation
exceeds the current limit. The control plane uses node-local `/dev/shm`, scoped
by torchrun identity and node rank. It is removed on successful shutdown;
after an aborted run use a fresh torchrun rendezvous identity/port (or remove
the stale `quack-rdma-return-node*` file after all workers have exited).

Local CPU checks:

```bash
ctest --test-dir build -R 'test_return_copies|test_down_ready_trace' --output-on-failure
python -m unittest discover -s tests -p test_worker_activation.py -v
```

Mock tests cover compact byte ranges, two segments per peer, empty/unequal
sources, partial final groups, prefix jumps, arm ordering, capacity rejection,
and repeated iterations. CUDA IPC, GPU publication visibility, and actual
copy/GEMM overlap require validation on Hopper.

No C++ ABI or QuACK changes are needed beyond the existing down-projection
requirements. Run `test_worker_activation.py` on Hopper for numerical coverage
of colliding expert contributions, distinct sources, both output layouts,
iteration reset, and changing gather-table publication order. CPU CLI tests
alone do not validate CUDA reduction correctness.

The Python test command above now also checks down CLI validation and, on Hopper,
compares down outputs against FP32 references for sequence-length scheduling,
table scheduling, scatter/reduction, and multi-buffer N-tile-major expert waves
with a wave barrier. Repeated launches check output reset. These single-GPU
checks do not replace the two-node integration run.

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
- `nvlink_forward_preparation_enabled`
- `nvlink_forward_ping_pong2_enabled` (default `false`)
- `nvlink_forward_submit_epilogue_enabled` (default `false`)
- `nvlink_forward_prepared_atomic_ring_enabled` (default `false`; requires submit/epilogue mode)
- `nvlink_forward_prepared_batch_chunks`
- `nvlink_forward_prepared_queue_depth`
- `nvlink_forward_prepared_nvtx_enabled`
- `nvlink_forward_out_of_order_chunks_enabled`
- `nvlink_forward_chunk_tokens`
- `nvlink_forward_use_batch_api`
- `nvlink_forward_stream_nonblocking`
- `nvlink_forward_synchronize_batches`
- `nvlink_forward_completion_notifications_enabled`
- `nvlink_forward_notification_flush_only_enabled`
- `nvlink_forward_notification_flush_per_entry_enabled`
- `nvlink_forward_notification_flag_update_mode` (`stream-write` or `memcpy`)
- `expert_gemm_m_tile`
- `expert_gemm_n_tile`
- `expert_gemm_dimension`
- `expert_gemm_cluster_m`
- `expert_gemm_max_swizzle_size`
- `nvlink_forward_notification_queue_depth`
- `nvlink_forward_notification_log_enabled`
- `nvlink_forward_notification_log_dir`
- `nvlink_forward_local_batch_sync_enabled`
- `nvlink_forward_ping_pong_enabled`
- `nvlink_forward_ping_pong_handoff_copy`
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

## Opt-in NVLink preparation pipeline

`nvlink_forward_preparation_enabled` defaults to `false`; disabling it retains
legacy readiness counters, batching, barrier size negotiation, and submission.
Enable it on **all local proxies** with an explicit fixed chunk count:

```bash
# Example n=32; choose/tune this value on the Hopper nodes.
RDMA_CPU_Proxy/scripts/run_node0_torchrun.sh \
  --local_forwarding_rdma_overlap_enabled=true \
  --nvlink_forward_local_batch_sync_enabled=true \
  --nvlink_forward_preparation_enabled=true \
  --nvlink_forward_prepared_batch_chunks=32 \
  --nvlink_forward_prepared_queue_depth=2
```

Use the same options with `run_node1_torchrun.sh` on node 1. Both CLI overrides
and JSON configuration support these settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `nvlink_forward_preparation_enabled` | `false` | Enable CPU preparation on the readiness thread. |
| `nvlink_forward_prepared_batch_chunks` | `0` | Required explicit positive `n` when enabled; no automatically chosen batch size. |
| `nvlink_forward_prepared_queue_depth` | `2` | Reusable slots per proxy, including the slot being submitted; minimum 2. |
| `nvlink_forward_prepared_nvtx_enabled` | `false` | Enable detailed prepared forwarding/preparation NVTX ranges in CUDA builds. |

The mode requires router routing, ordered chunks, batch API submission, and
synchronization of all destination copies after each batch. Round-robin and ping-pong must be off.
These conditions are validated at startup.

In this mode the legacy token/chunk/min/max thresholds are ignored. Each remote
peer's iteration is divided into exactly `n` received chunks per batch, except
for its shorter final batch; the last chunk may itself contain fewer tokens.
Local staging also uses `n` chunks plus a tail, but remains on the iteration
coordinator thread. With `nvlink_forward_local_batch_sync_enabled=true`, both
phases use a one-stage local batch-start barrier without negotiating chunk
counts or waiting for a second acknowledgment round. Each proxy publishes its
arrival and waits for every active local proxy to arrive or retire. Independent
arrival state for each phase prevents remote forwarding from overwriting local
staging readiness. Finished proxies retire from their phase; active proxies
submit their own batch unchanged even when tails differ. The non-prepared path
retains its two-stage batch-size negotiation barrier.

Rebuild all local proxies together when upgrading: the local synchronization
shared-memory layout is versioned and has changed for this barrier.

Set `nvlink_forward_local_batch_sync_enabled=false` on all local proxies to
skip the batch-start handshake in both phases. Each proxy then submits its
prepared batches independently as they become ready. Per-batch CUDA stream
synchronization, completion notifications, and the wait for that proxy's local
staging to finish remain in place. This does not disable
`local_iteration_sync_enabled` or iteration-end notification draining.

For a forwarding-only benchmark, set
`nvlink_forward_completion_notifications_enabled=false` on all local proxies.
Prepared forwarding and local input staging still copy the same data, but skip
notification construction, enqueueing, dispatch/receiver threads, and draining.
This mode requires `local_iteration_sync_enabled=true` so all local sources
finish copying before destination buffers can be reused by the next iteration.
Per-batch CUDA synchronization remains required; local batch synchronization
can be either enabled or disabled. Notification logging and notification flush
options must be disabled. Expert readiness and gather-table publication do not
advance, so `--concurrent-kernel` cannot be used in this mode.

The readiness thread scans immutable routing metadata, compacts destination
positions, and builds the final CUDA pointer/size/attribute arrays directly in
bounded reusable heap slots. It alone advances remote preparation cursors.
The forwarding thread waits for local staging, enters the start barrier if enabled,
submits once per nonempty destination, synchronizes the stream, performs any
required GPUDirect visibility flush, and, when notifications are enabled, copies them into
separate bounded reusable dispatch storage before releasing the slot. Batches
with no outgoing copies still publish the direct same-GPU input notification
when notifications are enabled.

Remote preparation can overlap local staging and earlier remote copies; remote
submission cannot precede local staging completion. Iteration completion tracks
completed chunks, not prepared chunks, and, when notifications are enabled,
drains notification publication and receiver processing before iteration reuse.
Queue waits respond to stop/error
signals; failed partial submissions retain argument storage through stream
cleanup. Existing cross-process barrier timeouts still apply if another proxy
fails or never participates.

The shared-library C ABI and Python worker require no changes. CPU-only tests
exercise mock copies, compaction, unequal tails, slot reuse, notification drain,
and multiple iterations/peers with notifications and batch barriers on and off.
CUDA/GPUDirect behavior and performance require
building and profiling on the remote Hopper systems. Compare identical fixed
batch sizes when measuring preparation time, queue starvation, barrier time,
submission gaps, and stream wait time.

### Profiling the prepared forwarding inter-batch gap

Set `--nvlink_forward_prepared_nvtx_enabled=true` on both torchrun commands to
emit the detailed static-label NVTX ranges in CUDA builds. The same boolean
key is supported in JSON configuration. It defaults to `false`: none of the
new `nvlink_prepared/*` or `nvlink_prepare/*` ranges call NVTX, and forwarding,
queueing, and synchronization behavior remains unchanged. Existing NVTX
markings (such as the GPUDirect flush range) retain their original behavior.
Rebuild `rdma_cpu_proxy_shared` in the Hopper build directory on both nodes,
then profile with CUDA and NVTX tracing enabled
(`nsys profile --trace=cuda,nvtx ...`).
Inspect the forwarding CPU thread in each torchrun worker process. These are
CPU wall-time ranges, not GPU execution durations.

Between the end of `nvlink_prepared/stream_sync` and the next
`nvlink_prepared/submit_copies`, the ranges follow this order (all labels below
have the `nvlink_prepared/` prefix):

| Range | Work measured |
| --- | --- |
| `rdma_owner_flush` | Optional GPUDirect owner visibility flush, including device selection and the existing nested flush range. Absent when native ordering suffices. |
| `completion_bookkeeping` | Compute elapsed batch time. |
| `notification_handoff` | Enqueue completed notifications for the dispatcher; nested `notification_lock`, `notification_queue_full_wait`, and `notification_publish_and_wake` separate contention, backpressure, and publication/wakeup. |
| `completion_stats` | Update completed chunks and bandwidth statistics; nested `stats_lock` measures mutex acquisition. |
| `batch_counter_and_log` | Increment batch count and optionally log. |
| `release_slot` | Return the completed argument slot; nested `release_slot_lock` measures queue mutex acquisition. |
| `wake_producer` | Notify the preparation thread that queue capacity is available. |
| `acquire_next_batch` | Acquire the next published slot; nested `queue_lock`, `queue_empty_wait`, and `local_staging_wait` separate mutex contention, missing prepared work, and unfinished local staging. |
| `iteration_bookkeeping` | Maintain the iteration and barrier round. |
| `local_batch_barrier` | Publish arrival and check/wait for other local proxies. Even the last arrival must execute the barrier's stores and peer scan. |
| `stream_lock` | Acquire the forwarding stream mutex before submission. |

With the same NVTX flag enabled, prepared barriers also emit:

- `nvlink_prepared/barrier_publish_arrival`: the arrival stores and the following marker.
- `nvlink_prepared/barrier_arrival_published`: an instant marker immediately after
  the stores, labeled with node, local GPU, phase (`local` staging or `remote`
  forwarding), zero-based iteration, and one-based barrier round.
- `nvlink_prepared/barrier_exit`: one instant marker on successful completion,
  with the same identity plus `polls` (including the successful scan) and
  `error_lock_acquire_ns` (accumulated error-check mutex acquisition time).

Match identities across proxies on the same node. An arrival marker proves
publication has already happened, but a thread can be descheduled between the
stores and the marker. These diagnostics also cover local-staging barriers,
which run on the iteration coordinator rather than the remote forwarding thread.

The enabled path reads the CPU steady clock before and immediately after each
error-check mutex acquisition, without NVTX calls inside the polling loop.
Acquisition time includes uncontended locking, timestamp overhead, and any
CPU descheduling in that interval; it is not a pure measure of lock contention.
Correlate with CPU scheduling to distinguish those causes. Label formatting
and marker emission also add overhead to the enclosing barrier range. Disabled
or non-CUDA builds skip the added timestamps, formatting, and markers and use
the original error-check call. Cancellation/error exits do not emit a successful
`barrier_exit` marker.

`submit_copies` covers the whole destination loop, with no extra NVTX calls
between consecutive `cudaMemcpyBatchAsync` submissions. Empty destinations
are skipped, so a batch with no outgoing copies has no CUDA copy call.

The preparation thread additionally emits `nvlink_prepare/build_batch`,
`publish_batch`, `queue_full_wait`, and `no_ready_batch_backoff` (each with the
`nvlink_prepare/` prefix). The last range wraps the existing 10 microsecond
sleep when no batch progresses; OS scheduling can extend that sleep. Correlate
it with a forwarding-thread `queue_empty_wait` before attributing a gap to it.
Queue depth 2 includes the in-use slot, so it allows at most one additional
prepared batch while that slot is retained.

Condition-variable wait ranges include wakeup scheduling and mutex
reacquisition. Short gaps between ranges include loop control, unlocks, and
instrumentation itself; CPU descheduling can extend any range. NVTX adds
measurement overhead, so compare instrumented timings with the earlier trace.
The presence of an approximately 10 microsecond gap alone does not identify
its cause, even on the slowest proxy.

### Opt-in submit/epilogue prepared forwarding

Set `--nvlink_forward_submit_epilogue_enabled=true` on both nodes to keep one
remote-copy submitter X and one epilogue worker Y. The default is `false`, which
preserves the existing forwarding paths. This mode requires preparation and its
prerequisites, and rejects either ping-pong mode being enabled at the same time.
Both CLI and JSON configuration accept the flag; the torchrun scripts pass it
through to the embedded shared library.

For example, after rebuilding the shared library on both Hopper nodes:

```bash
RDMA_CPU_Proxy/scripts/run_node0_torchrun.sh \
  --num_experts=128 --top_k=128 \
  --local_forwarding_rdma_overlap_enabled=true \
  --nvlink_forward_local_batch_sync_enabled=true \
  --nvlink_forward_completion_notifications_enabled=false \
  --nvlink_forward_preparation_enabled=true \
  --nvlink_forward_prepared_batch_chunks=64 \
  --nvlink_forward_prepared_queue_depth=4 \
  --nvlink_forward_prepared_nvtx_enabled=true \
  --nvlink_forward_ping_pong_enabled=false \
  --nvlink_forward_ping_pong2_enabled=false \
  --nvlink_forward_submit_epilogue_enabled=true
```

Run `run_node1_torchrun.sh` with the same options on node 1.

Preparation publishes immutable copy arguments and notification records into one
shared ring. X waits for local staging, enters the ordinary prepared batch-start
barrier with the other local X threads, submits the copies on the forwarding
stream, and calls `cudaStreamSynchronize`. Only successful synchronization permits
X to publish an atomic completed-prefix count with release ordering. X then
continues to the next batch without waiting for Y.

Y acquire-loads that count and handles completed batches in order: optional
GPUDirect owner flush, notification enqueue, statistics/progress updates, and
slot release. Notification enqueue copies records into the existing dispatch
queue before release; actual delivery remains the dispatcher's responsibility.
Without notifications, Y still updates progress and releases every slot. Failed
submission or synchronization cancels the pipeline without publishing success.
Iteration completion includes Y's retirement and the existing notification drain.
Local staging continues to run on the coordinator thread.

Queue depth is the **total number of shared slots**, with a minimum of two.
Depth two allows X and Y to overlap; depth three or more also leaves room for
preparation while both workers retain a slot. Depth four matches the total slot
capacity of ping-pong2 with depth two per lane. A slow epilogue fills the bounded
ring and eventually backpressures X through preparation.

Y busy-polls completion with `cpu_relax`, so the completion signal from X needs
no condition-variable wakeup. Both workers observe cancellation. Automatic CPU
affinity only selects an allowed CPU set; it does not dedicate separate physical
cores. Profile CPU contention when evaluating this mode.

With prepared NVTX enabled, X retains `nvlink_prepared/submit_copies` and
`nvlink_prepared/stream_sync`; Y owns the existing notification, statistics, and
release ranges. Additional markers are `nvlink_submit_epilogue/copy_done_publish`
and `nvlink_submit_epilogue/copy_done_wait`. Batch copy timing ends before the
completion signal and excludes Y's epilogue. Performance improvement requires
measurement on the target GPUs; mock tests verify ordering and lifecycle only.

### Optional stream-query completion for submit/epilogue mode

Add `--nvlink_forward_completion_mode=stream_query` on both nodes to poll
`cudaStreamQuery` in X after submitting each remote batch. The default is
`stream_sync`, preserving the existing `cudaStreamSynchronize` path. Query mode
requires `nvlink_forward_submit_epilogue_enabled=true` and works with either ring
implementation. The option is supported by CLI and JSON configuration.

Only `cudaSuccess` authorizes publication to Y. `cudaErrorNotReady` continues
polling with `cpu_relax`; other CUDA errors propagate through forwarding error
handling. Cancellation is checked every poll, and `completion_timeout_ms` is
checked every 1024 unsuccessful polls. Cancellation, timeout, or CUDA failure
does not publish that batch as complete. Shutdown still synchronizes the stream
before releasing slots and device buffers. The stream mutex remains held across
submission and polling. Local staging and iteration/shutdown synchronization are
unchanged.

The wait appears as `nvlink_prepared/stream_query` instead of
`nvlink_prepared/stream_sync`. There are no per-poll NVTX markers, but Nsight CUDA
tracing can record each query and affect timing. Polling consumes CPU and is not
guaranteed to reduce latency: compare throughput and unprofiled host timings on
the Hopper nodes. Mock copies complete synchronously, so mock tests cannot
measure GPU polling latency or exercise device-side pending/error conditions.

### Optional atomic prepared ring for submit/epilogue mode

Add `--nvlink_forward_prepared_atomic_ring_enabled=true` to the submit/epilogue
commands on both nodes. It requires `nvlink_forward_submit_epilogue_enabled=true`
and is accepted in CLI and JSON configuration. Its default is `false`, preserving
the mutex/condition-variable ring and all other existing forwarding paths.

In this mode, preparation release-publishes a ready-batch counter after filling
all descriptor fields. X acquire-loads that counter before reading the next slot;
it continues to release-publish copy completion only after stream synchronization.
Y acquire-loads completion, performs the epilogue, then release-publishes retirement.
Preparation acquire-loads retirement before reusing a slot. These counters remain
monotonic across iterations; iteration completion still waits for all epilogues
and notification draining. The ring need not hold an entire iteration: depths
of two or more retain bounded backpressure and safe slot reuse.

There are no prepared-ring mutex acquisitions or condition-variable waits in
preparation, X acquisition, or Y retirement when this flag is enabled. Empty/full
ring waits, iteration activation, and local-staging waits poll with `cpu_relax`
and check cancellation. The full-ring wait retains the completion timeout; the
existing no-RDMA-ready-batch backoff remains. Stream, statistics/error, and
notification queue mutexes retain their existing behavior.

`nvlink_prepared/acquire_next_batch` remains visible, but its `queue_lock` child
and Y's `release_slot_lock`/`wake_producer` ranges are absent. `queue_empty_wait`,
`local_staging_wait`, and `nvlink_prepare/queue_full_wait` now represent polling
when needed. Polling trades sleeping/wakeup overhead for CPU usage; measure on
the Hopper nodes with appropriate CPU placement.

### Opt-in ping-pong2 prepared forwarding

Add `--nvlink_forward_ping_pong2_enabled=true` to **both** torchrun commands,
keeping the preparation options, for example:

```bash
RDMA_CPU_Proxy/scripts/run_node0_torchrun.sh \
  --local_forwarding_rdma_overlap_enabled=true \
  --nvlink_forward_local_batch_sync_enabled=true \
  --nvlink_forward_completion_notifications_enabled=false \
  --nvlink_forward_preparation_enabled=true \
  --nvlink_forward_prepared_batch_chunks=64 \
  --nvlink_forward_prepared_queue_depth=2 \
  --nvlink_forward_prepared_nvtx_enabled=true \
  --nvlink_forward_ping_pong2_enabled=true
```

Use `run_node1_torchrun.sh` with the same options on node 1. The new flag is
also accepted in JSON. Its default is `false`, which keeps the existing
single-consumer prepared path and existing event-based ping-pong behavior.
Ping-pong2 requires preparation and its existing prerequisites; it cannot be
combined with `nvlink_forward_ping_pong_enabled=true`. It supports completion
notifications and local batch synchronization independently enabled or disabled,
subject to the existing preparation validation rules.

One preparation thread assigns remote batches alternately to independent A/B
queues. `nvlink_forward_prepared_queue_depth` is **per lane** in this mode,
including each lane's retained batch: depth 2 allocates four slots total.
Local-input staging remains on the coordinator thread. Every iteration starts
on A, even after an odd number of batches in the preceding iteration.

Each forwarding lane acquires its next prepared batch, waits for local staging,
and then **busy-polls an atomic submission sequence**. Only the owner enters
its local batch-start barrier and submits copies on the shared CUDA stream.
A synchronizes with A on other local proxies; B synchronizes with B, using
separate arrival slots and lane-specific rounds. A retired remote phase also
satisfies subsequent rounds, so unequal tails and zero-batch proxies are allowed.
After successful `cudaStreamSynchronize`, the owner releases the stream mutex
and signals the next sequence before notification handoff, statistics, slot
release, or producer wakeup. The other lane can execute while that cleanup runs.
A failed synchronization does not pass ownership. Failure/stop flags and periodic
timeout checks terminate handoff polling; successful barrier polls avoid the
statistics/error mutex.

Notifications use two bounded lane queues (each sized by
`nvlink_forward_notification_queue_depth`) and the existing single dispatcher.
The dispatcher publishes one complete batch envelope at a time in submission
sequence order, even if a later lane enqueues first. Empty envelopes still advance
the sequence. Local-staging notifications retain their existing queue and precede
remote publication for that iteration. Any required direct-input GPUDirect owner
flush runs after handoff but before the corresponding envelope is enqueued.
The receiver and gather-table update path are unchanged. Iteration retirement
waits for both lanes' copy, notification handoff, accounting, and slot release;
notification draining still completes before iteration reuse.

With `nvlink_forward_prepared_nvtx_enabled=true`, the existing prepared ranges
appear on both forwarding threads, with additional ranges:

- `nvlink_ping_pong2/handoff_wait_A` and `handoff_wait_B`: busy-polling for ownership.
- `nvlink_ping_pong2/handoff_publish`: publishing ownership after releasing the stream mutex.

Remote barrier markers include `lane=A` or `lane=B`; `round` is one-based within
that lane and iteration. `error_lock_acquire_ns=0` in ping-pong2 reflects the
atomic error-check path, rather than a measured lock acquisition. Batch timing
ends at stream synchronization, excluding the overlapped notification/flush work.
The remaining copy-to-copy gap includes handoff observation, barrier work, stream
lock acquisition, and submission overhead. Queue starvation, CPU descheduling,
or slow cleanup can still add latency. Existing CPU affinity chooses an allowed
CPU set; it does not dedicate separate physical cores to the two spinning lanes.

Rebuild the shared library on both Hopper nodes and upgrade all local workers
together: the local synchronization shared-memory layout is now version 7.
CPU mock tests cover both modes, unequal tails, empty inputs/copies, multiple
peers and iterations, bounded queues, notification reordering, and shutdown/error
paths. CUDA synchronization, GPUDirect visibility, and latency improvements must
be validated on the remote Hopper systems.

## Current Limitations and TODOs

- Metadata exchange is a minimal TCP exchange intended to make QP bring-up concrete. Production deployments usually replace this with an existing control plane.
- The server side accepts one metadata connection at a time. Multi-peer simultaneous bring-up may need a threaded or event-driven connection manager.
- QP attributes such as MTU, SL, traffic class, timeout, retry count, and GID handling should be tuned per fabric.
- CUDA copy staging currently uses a generic byte-copy kernel. Integrate the real token producer path if tokens are already in the registered send buffer.
- Mock mode copies payload bytes locally and self-completes WQEs for development visibility; it does not simulate fabric ordering or remote process state.
- Error recovery is fail-fast. Production use should add QP teardown/reconnect and health reporting.

### Shared prepared pipeline for local staging (opt-in)

Add `--nvlink_forward_local_staging_prepared_enabled=true` to both torchrun
commands to forward locally staged tokens through the same preparation,
submission, and epilogue threads as remote RDMA input. The default is `false`,
which retains local forwarding on the calling/coordinator thread.

This mode requires `router_local_input_staging_enabled=true`,
`nvlink_forward_preparation_enabled=true`, and
`nvlink_forward_submit_epilogue_enabled=true`, together with their existing
prerequisites. Both ping-pong modes must remain disabled. It supports the mutex
queue or atomic ring, `stream_sync` or `stream_query`, and notifications and
local batch synchronization independently enabled or disabled.

The preparation thread publishes all local batches before remote batches into
one ring. Local data is already ready after staging completes, so no CQ checks
are performed for this source. Batch size and queue depth apply to both sources;
queue depth is the total shared capacity, not a separate capacity per source.
The submission thread preserves separate local/remote batch-barrier phases.
The epilogue publishes local notifications and local-phase completion before
allowing remote submissions. Empty local sources publish a control descriptor
so other local proxies can finish their barriers; it is excluded from data-batch
statistics. Iteration completion includes local epilogues and notification drain.

With `local_forwarding_rdma_overlap_enabled=true`, RDMA can progress while the
shared pipeline forwards local input. When false, the coordinator waits for
local forwarding before enqueueing RDMA transfers. No additional threads are
created. Local source gathering still happens during iteration preparation;
this mode moves forwarding of the staged buffer, not the gathering operation.

For the prepared atomic-ring configuration, keep the existing options and add:

```bash
--nvlink_forward_local_staging_prepared_enabled=true
```

Mock coverage:

```bash
ctest --test-dir RDMA_CPU_Proxy/build -R test_unified_local --output-on-failure
```

Real CUDA/RDMA performance must be measured on the target GPU nodes.

### Inter-node combine in original token order

After rebuilding `rdma_cpu_proxy_shared` on both Hopper nodes, append these to
both existing prepared-forwarding / down-GEMM commands:

```bash
--intra-node-combine-n-tile-major-output \
--rdma-return-token-results \
--inter-node-combine
```

`--inter-node-combine` is opt-in and requires `--intra-node-combine` and
`--intra-node-combine-n-tile-major-output`. With multiple nodes it additionally
requires `--rdma-return-token-results` and all its existing prerequisites. It runs after native reverse-RDMA completion
(including receive visibility) and local intra-node combine completion. A single
CuTe kernel gathers the local staging node sum and each remote returned node sum,
accumulates in FP32 in ascending node order, and writes a contiguous
`[num_tokens, token_dimension]` tensor in original source-token order. There is
no separate scatter kernel, no atomics, and no group-readiness polling. Tokens
without contributions are written zero. Routing weights are not applied; input
node sums have already been rounded to their BF16/FP16 storage precision.

For a single-node run, retain the existing down-GEMM, NVLink-return, and
intra-node-combine prerequisites, and append only:

```bash
--intra-node-combine-n-tile-major-output \
--inter-node-combine
```

Do not enable RDMA returns in this case. The kernel reads only the local staging
node sum after intra-node combine finishes, converting its tiled layout and x3
row order directly to original token order. Output-ready publication is optional
for this single-node path because the worker joins the producer stream first.

Read the result through
`down.intra_node_combine.inter_node_combine.output`. It has the same dtype as the
node sums. The worker joins the kernel before any next-iteration buffer reuse;
its CUDA-event time is available as `.elapsed_ms` and is excluded from existing
up/down GEMM timing. Disabled mode allocates no final output or inverse map.
For 204800 tokens and width 4096, the additional BF16 output occupies 1.56 GiB.

Use `--inter-node-combine-ctas=40` to select a positive CTA count independently
of intra-node combine. Omitting it uses one CTA per GPU SM. Each CTA has 256
threads and processes a strided share of the output; this is a launch-grid
size, not SM affinity. It works for both single-node and multi-node runs.

The new C ABI `rdma_proxy_get_outgoing_token_indices` exposes this source GPU's
per-destination x3 lists, including the local domain, and full input token count.
The inverse row maps are constructed once during setup because benchmark routing
is fixed. Each node's input has its own row count and order; equal row indices
across node buffers are not assumed to identify the same token.

Add `--check-inter-node-combine` for an independent FP32 scatter/add reference
check each iteration. It uses bounded output-row blocks and casts to the output
dtype before comparison. Combine with the existing intra-node and RDMA-return
checks to validate the earlier stages as well. Checks add overhead.

Run the standalone kernel tests on Hopper:

```bash
RDMA_PROXY_TEST_CUDA=1 QUACK_ROOT=/path/to/quack_for_communication_overlap \
python -m unittest discover -s RDMA_CPU_Proxy/tests -p test_inter_node_combine.py -v
```

The tests cover unequal and shuffled x3 domains, missing contributions, empty
inputs, BF16/FP16, multiple tile widths, and changed payloads across iterations.
