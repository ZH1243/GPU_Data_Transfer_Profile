# Persistent NVLink Forwarding Computation

## Existing flow and integration points

`Proxy::initialize()` allocates the per-peer RDMA send/receive buffers, registers them, creates QPs, and then starts the NVLink forwarding subsystem. CQ workers decode write-with-immediate completions. The forwarding-ready thread turns the per-chunk immediate counters into contiguous or out-of-order ready ranges. The forwarding thread copies those ranges from the per-peer RDMA receive buffer to a destination GPU's per-source NVLink receive buffer.

When synchronized forwarding notifications are enabled, `issue_forwarding_batch()` waits for its forwarding stream, pushes notification records into an in-process dispatch queue, and continues. A dispatch thread is the producer for one POSIX shared-memory SPSC queue per source/destination GPU pair. The destination proxy's notification thread polls and parses those queues. Each record identifies the iteration, peer slot, source/destination GPUs, ready token range, and byte range.

The new mode hooks into that receiver polling path. A data notification is converted immediately into output-tile tasks; it is not held until `wait_for_forwarding_iteration()` completes. `run_iteration()` launches the persistent kernel before the iteration-start barrier and shuts it down only after every source's ordered iteration-done record has been parsed.

The complete iteration sequence is:

1. Reset the computation queues/output buffers and launch one persistent CTA per SM.
2. Enter the existing cross-node iteration-start barrier and start RDMA work.
3. Forward and synchronize each ready NVLink range; dispatch its notification.
4. On the destination, partition every notification and enqueue compute tasks immediately.
5. After a source finishes all forwarding for the iteration, append one ordered iteration-done record to every destination stream.
6. After the destination parses all source done records, enqueue the exact number of exit tasks needed by each queue and wait for the kernel stream to terminate.
7. Publish `receiver_completed_generation` acknowledgements to every source.
8. Each source waits for all destination acknowledgements before it can finish the iteration and reuse the receive-buffer generation.

With the feature disabled, none of the computation buffers, device task/publication queues, mapped reuse signals, done records, acknowledgement waits, streams, or kernels are created; the prior forwarding behavior is unchanged.

## Modified files

- `include/config.hpp`, `src/config.cpp`, `src/main.cpp`: options, validation, summaries, and `--help`.
- `include/cuda_buffers.hpp`, `src/cuda_buffers.cpp`: one output buffer per NVLink receive buffer and one deterministic BF16/FP16 weight matrix.
- `include/forward_computation.hpp`, `src/forward_computation.cpp`: task format, reverse-direction queue, task partitioning, mock executor, lifecycle, backpressure, and counters.
- `src/cuda_kernels.cu`: persistent multi-consumer kernel, system-scope queue operations, asynchronously double-buffered operand staging, and tensor-core GEMM.
- `include/proxy.hpp`, `src/proxy.cpp`: notification translation, ordered iteration-done records, buffer-lifetime acknowledgements, startup, and shutdown.
- `tests/test_forward_computation.cpp`: Mac-compatible host/mock correctness, incremental notifications, multiple buffers/queues, queue pressure/wraparound, uneven CTA assignment, exits, and iteration reuse.
- `tests/validate_forward_computation.cu`: Hopper validation against cuBLAS, including the requested 256x4096x6400 case.

## CPU-to-GPU queue

The design is adapted from UCCL-EP's `ep/include/fifo_device.hpp` and
`d2h_queue_*` ownership scheme, with producer and consumer directions reversed.
It is implemented locally to avoid depending on UCCL headers, verbs types, or
build flags.

Each queue follows the same consumer-local polling principle as UCCL-EP, with
the producer and consumer directions reversed:

- a pinned host staging ring and a matching device-memory `ForwardComputeTask`
  ring;
- pinned host publication staging and a device-memory published-head counter;
- 128-byte reuse signals in `cudaHostAllocMapped` memory, written only by the
  GPU and polled only by the CPU.

The persistent GPU CTAs poll only the device-memory published head. The
emergency abort flag is also device memory. Consequently, an empty queue does
not generate repeated mapped-host reads over PCIe. A device-memory
`dequeue_position` is shared by the CTAs assigned to that queue. Only GPU
consumers perform CAS on that position, so the protocol does not require
CPU/GPU atomic RMW on mapped PCIe memory.

For absolute position `p` and slot `p % capacity`:

- The CPU may reuse the slot only when `consumed_sequence == p`.
- It writes the complete descriptor into pinned staging memory.
- A nonblocking publication stream copies one or more contiguous descriptors
  into the device task ring, then advances the device-resident published head
  with one 8-byte copy. Ring wrap is split into contiguous batches.
- A CTA polls the published head in GPU memory. When `p < published_head`, it
  claims `p` using `atomicCAS` on the device dequeue counter. Exactly one CTA
  wins.
- The winning CTA computes the tile. Only after D is visible does it write `consumed_sequence = p + capacity` with `st.release.sys.global`.
- The CPU observes reuse with an acquire load.

This prevents overwrite both before claim and while a claimed task is still reading A/B or writing D. Positions are 64-bit monotonic values; modulo is used only to select storage. Queues are reset only after the prior persistent kernel terminates, so practical wraparound is bounded by 2^64 operations within one iteration.

The two H2D copies use the same CUDA stream, so the published head cannot
advance before its descriptors reach device memory. Ready-region tasks are
grouped per queue and copied in contiguous batches, amortizing CUDA submission
overhead.
Write-combined host memory is deliberately not used for the CPU-polled reuse
signals; normal cached mapped memory keeps the host acquire operations
well-defined without architecture-specific write-combining flush behavior.

When a queue is full, the notification thread applies bounded backpressure and reports a timeout using `completion_timeout_ms`. It never drops a compute task. Queue-full stall counters are reported at iteration completion. An emergency device-resident abort signal lets every CTA leave its polling loop during partial shutdown; normal shutdown uses exit tasks.

## Task descriptor

Each descriptor is immutable and self-contained:

- compute/exit type and BF16/FP16 type;
- iteration generation and queue ID;
- stable receive/output buffer IDs;
- token-row offset and valid row count;
- output-column offset and valid column count;
- tile M/N plus matrix M/N/K metadata;
- A, B, and D GPU virtual addresses;
- enqueue timestamp and diagnostic flags.

`byte_offset` from the notification is converted to a global row within the source GPU's receive buffer and cross-checked against `peer_slot * num_tokens + start_token`. Partial row/column tails are supported. For divisible dimensions, task count is exactly `(M_ready / tile_m) * (N / tile_n)`; the requested 256x4096x6400, 128x128 example creates 100 tasks.

## CTA assignment and kernel

The launch has `gridDim.x == multiProcessorCount`. More than half of the SM's shared-memory capacity is reserved dynamically per block, and the CUDA occupancy API is checked at launch to require exactly one resident CTA per SM. CUDA still chooses which block runs on which physical SM; no block-index-to-SM-index relationship is assumed. The kernel records `%smid` for diagnostics only.

Logical CTA `blockIdx.x` polls queue `blockIdx.x % num_queues`. This is deterministic and gives every queue either `floor(SMs/Q)` or `ceil(SMs/Q)` CTAs. Queue counts that do not divide the SM count are supported with a warning and are covered by the host assignment test. `num_queues >= SM_count` is rejected.

Each 256-thread CTA uses eight warps. For a task, warps distribute the 16x16 output fragments. Full fragments use FP16/BF16 WMMA tensor-core operations with FP32 accumulation. Global A/B fragments are staged into per-warp shared-memory double buffers with `__pipeline_memcpy_async`; staging of K tile `i+1` overlaps tensor-core work for K tile `i`. Results are converted to the configured 16-bit output type. Partial 16x16 edge fragments use a scalar correctness fallback. Different CTAs naturally overlap queue polling, operand loading, tensor-core work, and stores across independent output tiles.

### Load-only mode

Set `nvlink_forward_computation_load_only_enabled=true` to execute each normal
compute task as an operand-staging task. Each assigned warp walks the same
output fragments and K tiles as GEMM and transfers both A (forwarded token rows)
and B (weights) from global memory into its double-buffered shared-memory
stages. Full fragments retain the asynchronous pipeline; partial token/output
tails use guarded global-to-shared loads. Shared values are consumed only enough
to prevent compiler dead-code elimination. The kernel performs no WMMA or
scalar multiply-accumulate operations and writes nothing to D.

Queue claim/completion counters, exit tasks, reuse acknowledgements, and
iteration lifetime protection remain active. Output buffers are still allocated
and cleared to keep the existing descriptor ABI and buffer lookup unchanged,
so they remain zero after a load-only iteration. This sub-mode is rejected
unless `nvlink_forward_computation_enabled=true`.

## Buffer and weight lifetime

Every per-source NVLink receive buffer has a matching output buffer with row capacity `num_tokens * remote_peer_count` and feature width `nvlink_forward_computation_output_dim`. One row-major KxN weight buffer is allocated in GPU global memory and initialized once with deterministic small pseudo-random values before any persistent kernel launch. A and B use BF16 or FP16; WMMA accumulates in FP32; D is converted back to the input type.

Outputs are cleared before each iteration. A/B/D remain allocated until the proxy and computation stream are shut down. Ordered done records plus `receiver_completed_generation` acknowledgements prevent the next source generation from overwriting A before all tasks from the prior generation and all persistent CTAs have completed.

## Configuration

The mode requires:

```text
nvlink_forwarding_enabled=true
nvlink_forward_synchronize_batches=true
nvlink_forward_completion_notifications_enabled=true
nvlink_forward_computation_enabled=true
```

New options are:

```text
nvlink_forward_computation_output_dim
nvlink_forward_computation_tile_m
nvlink_forward_computation_tile_n
nvlink_forward_computation_num_queues
nvlink_forward_computation_queue_depth
nvlink_forward_computation_load_only_enabled
nvlink_forward_computation_log_enabled
```

`token_dimension` is K. K, N, tile M, and tile N must be multiples of 16. Only BF16 and FP16 are accepted. Buffer/task allocation overflow and zero queue values are rejected. Runtime setup requires Hopper compute capability 9.0+, mapped host memory, fewer queues than SMs, and enough opt-in shared memory to enforce one resident CTA per SM.

## Build and run

On a CUDA 12/Hopper node:

```bash
cmake -S RDMA_CPU_Proxy -B RDMA_CPU_Proxy/build-hopper \
  -DRDMA_PROXY_REQUIRE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build RDMA_CPU_Proxy/build-hopper -j
```

Run the proxy by adding these overrides to the existing per-GPU command:

```bash
RDMA_CPU_Proxy/build-hopper/rdma_cpu_proxy \
  --config RDMA_CPU_Proxy/config/example_config.json \
  --nvlink_forwarding_enabled=true \
  --nvlink_forward_synchronize_batches=true \
  --nvlink_forward_completion_notifications_enabled=true \
  --nvlink_forward_computation_enabled=true \
  --nvlink_forward_computation_output_dim=6400 \
  --nvlink_forward_computation_tile_m=128 \
  --nvlink_forward_computation_tile_n=128 \
  --nvlink_forward_computation_num_queues=8
```

Run quick and full GPU validation:

```bash
RDMA_CPU_Proxy/build-hopper/validate_forward_computation
RDMA_CPU_Proxy/build-hopper/validate_forward_computation --full
```

The full mode compares all 256x6400 output values for K=4096 against `cublasGemmEx`. Both quick and full validation then run a second load-only iteration, check that every task completes, and verify that every output element remains zero. The normal iteration also waits until all tasks from the first half-notification complete before publishing the second, demonstrating incremental computation. Iteration logs report generated/claimed/completed/exit counts, full-queue stalls, enqueue and total time, per-queue load when verbose logging is enabled, and how many tasks were already complete when the final notification was parsed.

## Known limitations

- TMA is not used yet. Hopper's TMA tensor-map setup is awkward for dynamically supplied per-task base addresses and partial tails; the current kernel uses correct system-scope queues plus asynchronous `cp.async`-class pipeline intrinsics and WMMA. A future specialization can prebuild tensor maps for stable buffer IDs.
- Output conversion/store is synchronous. Loading K tile `i+1` overlaps tensor-core work for K tile `i`, while independent CTAs overlap stores with other tasks.
- One CTA owns an entire configured output tile. Very large tiles increase per-task latency; 128x128 is the intended starting point.
- The device-resident publication protocol and mapped reuse-signal visibility
  must be validated on the deployment platform/IOMMU. Initialization rejects
  non-Hopper GPUs; cluster validation is required because the development Mac
  has no CUDA device.
- The mapped notification ABI version was increased from 2 to 3 for the receiver completion acknowledgement. Every local proxy in a run must use the same build.
- Every local proxy must use the same computation enablement and dimensional configuration. Mixed enabled/disabled peers are rejected indirectly by the bounded generation-ack timeout rather than through a separate configuration negotiation protocol.
