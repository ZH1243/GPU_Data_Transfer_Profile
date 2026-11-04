# GPU router metadata exchange, single node

This experiment runs **eight torchrun Python processes on one Linux Hopper node**.
Each process owns one GPU and loads `librouter_metadata_exchange2.so`. That library
owns a C++ proxy thread. Python generates fresh routes and launches seven CuTe DSL
metadata kernels; a CUDA C++ kernel writes destination-specific metadata directly
into CUDA-IPC-mapped peer GPU buffers. Each receiver proxy reads the results into
persistent pinned memory and installs pointer/count views of that memory.

This is an isolated **benchmark-specific proxy**, not the production `Proxy` class
or `librdma_cpu_proxy.so`. The production class currently generates/exchanges
metadata internally during initialization, owns vector-backed records, and rejects
reinstallation. This experiment implements the proposed replacement data path and
production routing/index conventions without changing those APIs. Host tests compare
the new reference and installed views against the actual production `RouterRouting`
CPU implementation. It does not measure token payload dispatch, GEMMs, RDMA,
GPU `A_idx`/scheduler construction, local `x3/x4` CPU installation, or combine.
The completion endpoint is **all eight source expert tables installed at the CPU proxy**.

## Build and run on the remote machine

Requirements: Linux, a CUDA toolkit with `nvcc`, CMake >= 3.18, a C++17 compiler,
PyTorch with CUDA and Gloo, `cuda.bindings`, and the same CUTLASS CuTe DSL environment
used by `get_expert_token_idx_list_gpu/benchmark_metadata_cute.py`. No libibverbs,
QuACK checkout, PyTorch C++ extension build, or NCCL metadata collective is needed.
All eight GPUs must be visible to every process and support peer access. Use a
single-node NVLink/NVSwitch machine; peer access alone does not prove the physical
route is NVLink (check `nvidia-smi topo -m`). No host-transport fallback is implemented.

From this directory on the Hopper node:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
ctest --test-dir build --output-on-failure

# Initial full-content check, including empty lists and length changes:
bash ./run_single_node_torchrun.sh \
  --tokens 257 --warmup 1 --iterations 6 --route-pattern cycle --verify all

# Default workload: 20,480 tokens/source, 5 warmups, 50 measured iterations.
bash ./run_single_node_torchrun.sh

# Paired modes; both use the same seeds. Graph captures only the seven metadata kernels.
RESULT_JSON=results/eager.json bash ./run_single_node_torchrun.sh --mode eager
RESULT_JSON=results/graph.json bash ./run_single_node_torchrun.sh --mode graph

# Extended remote GPU checks, including scan carry and partial-chunk boundaries:
bash ./run_gpu_checks.sh
```

The launch script always uses `--standalone --nnodes=1 --nproc_per_node=8` and disables
torchrun restarts. `CUDA_VISIBLE_DEVICES` may reorder the eight physical GPUs; logical
rank maps to the corresponding visible CUDA ordinal. Do not restrict each worker
to a different single visible device. `TORCHRUN_BIN`, `RME2_LIBRARY`, and `RESULT_JSON`
override the launcher binary, shared library, and output path. Output paths are
overwritten on subsequent runs. `--timeout` defaults to 120 seconds, including the
Gloo collective timeout; increase it for slow first-time JIT or exhaustive checking.

`--cpu-core-map=...` accepts eight CPU IDs and pins only the native proxy threads.
Choose cores near each GPU's NUMA node on the actual machine. By default they are
unbound. The proxy actively polls CUDA completion and shared CPU counters, so reserve
CPU capacity for all eight processes and their native threads.

## Routing contract and CuTe adaptation

The topology is truly **one node, eight GPUs, 128 experts, top-8**, hence **16 experts
per GPU**. It does not preserve two logical nodes while omitting remote traffic.
Token count is configurable; GPU count, expert count, and top-k are fixed for this
first experiment.

`metadata_cute_single.py` adapts the existing two-node seven-kernel implementation:

1. `count_input`: expert counts and stable node-mask bitmaps.
2. `scan_input_chunks`: chunk prefixes for 384 expert/mask bins.
3. `build_input_offsets`: expert, mask, and node offsets.
4. `build_node_indices`: stable descending-mask `x3/x4` order.
5. `count_reordered`: 136 expert/GPU bins in reordered order.
6. `scan_reordered`: prefixes over reordered chunks.
7. `scatter_experts`: expert index lists with the direct-input exception.

Routes are distinct expert IDs in `[0,128)`. For source `s` and destination `d`,
the mask bit is `(s-d) mod 8`. The source's own expert lists index the full reordered
node input; other destinations' lists index their compacted input. List order and
duplicates across different expert lists are preserved. Destination offsets are
normalized to zero. `x3/x4` are computed and verified but not exchanged: there is
no inter-node `x3/x4` path in this experiment.

The default GPU generator matches the existing SplitMix64/start/odd-stride generator.
Iteration `i` (starting at 1), source `s`, uses `(seed + s + 8*(i-1)) mod 2^64`.
Every warmup and measured invocation receives fresh routes. `--route-pattern cycle`
alternates affine, all experts 0..7, and all experts 120..127, exercising zero-length
receives and growth/shrinkage in the same allocations. Hot-pattern inputs are filled
outside timing, instead of running the random generator.

## Copy and synchronization protocol

Setup allocates dedicated GPU inboxes with `cudaMalloc`, exports CUDA IPC handles,
and opens peer mappings. Handles and configuration travel through Gloo only during
setup. Dedicated allocations avoid exposing PyTorch allocator slabs. Every source
owns a disjoint slot in each destination inbox; no remote allocation or append atomic
is necessary. A slot's header contains the source slice's absolute begin followed by
17 normalized offsets. Each slot has worst-case `T*8` int32 payload capacity, but only
the valid prefix is transferred. At default T, each receiver has approximately 5 MiB
of GPU inbox space and 5 MiB of pinned storage, excluding CuTe scratch.

For every iteration in the default batched-readback mode:

1. Generate R, finish generation, and align ranks with a CPU collective outside timing.
2. Start the Python wall timer. Launch the seven metadata kernels (eager or graph).
3. Queue the exchange kernel on the **same CUDA stream**. One CTA per destination
   reads source offsets and directly writes the destination header and index slice.
   Every writer executes a system fence. The self CTA writes only a header.
4. The native thread polls its source exchange completion event, then release-publishes
   the iteration number in a cache-line-separated POSIX shared-memory CPU counter.
   Receivers acquire-poll all eight source counters. This is a conservative all-senders
   completion protocol; no GPU spins on remote flags and no Python collective occurs
   inside the timed exchange. CPU polling/publication and peer waiting remain timed.
5. The receiver copies all headers (576 bytes) to pinned storage, waits for that copy,
   validates counts, and submits exact-length D2H copies into each source's final
   pinned record. Self payload is read directly from the CuTe output, bypassing the
   local inbox payload. Empty payloads submit no copy.
6. After D2H completion, install borrowed views and reset per-expert heads/tails.
   There is no pinned-to-vector copy, serialization, or per-iteration allocation in
   the native data path. Optional full index bounds/sorted-order checks are timed.
7. Return to Python and stop the wall timer. Optional exact verification runs outside
   timing. A consumption barrier then prevents ANY producer from reusing its output
   or inbox while another proxy still reads it.

The header copy is needed because host-submitted D2H copies require host-known sizes.
Payload capacity is conservative; transfer size is exact. The design avoids sender
D2H, sender packing buffers, self peer payload copies, and receiver CPU payload copies.
It is not zero-copy: peer payloads still move GPU-to-GPU and then GPU-to-CPU.

### Native worker metadata launch and pre-launch barrier

`--native-metadata-launch` moves metadata graph replay and exchange submission onto
the native proxy worker. It requires `--mode graph` (the default). Add
`--metadata-launch-barrier` to synchronize those workers through shared CPU memory
immediately before they record the start event and launch the metadata graph.
The launch barrier requires native launch; invalid option combinations fail early.
Both flags are off by default, preserving the existing Python launch path.

Update the Python files and **rebuild the library (ABI 4)**:

```bash
cmake --build build -j
# Correctness with native launch, pre-launch alignment, fresh inputs and empty lists:
bash ./run_gpu_checks.sh --native-metadata-launch --metadata-launch-barrier

RESULT_JSON=results/native_launch_barrier.json bash ./run_single_node_torchrun.sh \
  --native-metadata-launch --metadata-launch-barrier \
  --skip-index-validation --warmup 20 --iterations 200
```

For comparison, run native launch without `--metadata-launch-barrier`, and the
original mode without either flag. To measure residual skew after metadata finishes,
also add the independent `--metadata-barrier` described below. All combinations
support `--per-source-readback`. The GPU check script selects only graph mode when
native launch is requested; run it with the additional options to check combinations.

Python still generates R, compiles CuTe and warms its kernels during setup. The C API
begins stream capture, Python queues the seven compiled kernels once, and C++ ends
capture, instantiates and uploads its own CUDA graph executable. This does not depend
on PyTorch exposing raw graph handles. Capture/JIT/upload are outside timing. The
Python wrapper retains the plan and stream, keeping captured tensor addresses alive;
the native proxy destroys its executable only after worker shutdown and stream
completion. The original graph definition is destroyed after instantiation.

Each iteration completes R generation outside timing, then Python queues a native
command and waits. The worker selects its GPU, optionally waits for the pre-launch
barrier, replays the metadata graph and submits exchange on the same stream. Python
does not replay metadata or record metadata timing events in this path. The native
worker's CUDA events supply `metadata_gpu_ms`. There is no Python scheduling step
between barrier release and graph submission. CPU scheduling and CUDA launch latency
can still produce GPU start skew.

`metadata_launch_barrier_ms` is the timed worker arrival/publication/wait interval
before metadata launch (zero when disabled). It is included in `native_ms` and
`end_to_end_ms`, together with command handoff and worker scheduling. Consequently,
waiting can move into this barrier; compare end-to-end latency rather than just peer
wait. The metadata-launch and post-metadata barriers use separate cache-line-aligned,
epoch-tagged arrival arrays and the existing abort/timeout protocol.

### Optional barrier after metadata processing

`--metadata-barrier` (default off) synchronizes after **GPU metadata completion**,
before exchange submission. In the default launch path it runs inside the native
submission call made by each Python caller. With `--native-metadata-launch`, it runs
on the proxy worker after graph submission. An event queued on the metadata stream must
complete before the caller release-publishes its iteration in a separate shared-memory
arrival slot. The C++ callers acquire-poll all eight arrivals, checking timeout/abort,
then enqueue their exchange kernels. Slots carry monotonically increasing epochs;
the existing consumption barrier prevents reuse while peers still consume metadata.

Update Python and rebuild the library together (native **ABI 4**):

```bash
cmake --build build -j
bash ./run_single_node_torchrun.sh --metadata-barrier \
  --tokens 257 --warmup 1 --iterations 6 --route-pattern cycle --verify all

RESULT_JSON=results/metadata_barrier.json bash ./run_single_node_torchrun.sh \
  --metadata-barrier --skip-index-validation --warmup 20 --iterations 200
```

The extra waits remain inside `native_ms` and `end_to_end_ms`, with separate rows:

- `metadata_completion_wait_ms`: recording and polling the metadata completion event
  in the C++ caller. It measures remaining metadata execution and CPU detection delay,
  not the full seven-kernel duration (which remains `metadata_gpu_ms`).
- `metadata_barrier_wait_ms`: publishing this GPU's metadata-ready arrival and waiting
  for all other GPUs' arrivals. This exposes readiness skew after metadata processing.

Both fields are zero when disabled. In default batched readback, the subsequent
`sender_completion_wait_ms` now waits for exchange completion only, since metadata
has completed; `peer_completion_wait_ms` still waits for peer exchange publications.
Neither CPU barrier release nor CPU completion detection guarantees simultaneous GPU
starts. Compare end-to-end time as well as the waits: waiting may move from the peer
exchange stage into the metadata barrier. The flag can also combine with
`--per-source-readback`, whose overlapping timing semantics below still apply.

### Optional per-source readback

Pass `--per-source-readback` to remove the all-sources gate before readback.
The default remains unchanged. **Rebuild the shared library and update Python
together**: the native interface is now ABI 4; Python rejects older libraries.

```bash
cmake --build build -j
# Verify changed lengths, empty payloads, and both eager/graph execution:
bash ./run_gpu_checks.sh --per-source-readback

# Performance comparison with the same validation/warmup/iteration settings:
RESULT_JSON=results/per_source.json bash ./run_single_node_torchrun.sh \
  --per-source-readback --skip-index-validation --warmup 20 --iterations 200
RESULT_JSON=results/batched.json bash ./run_single_node_torchrun.sh \
  --skip-index-validation --warmup 20 --iterations 200
```

The native worker runs a nonblocking sweep over source states:
`waiting for publication -> header in flight -> payload in flight -> installed`.
It also polls and publishes its own GPU completion while progressing other sources;
it does not first block on its own GPU. As soon as a source's iteration counter is
observed, its header is copied. Header completion permits structural validation and
exact-length payload submission. Payload completion permits installing that source's
pinned-memory view. Each source has a reusable nonblocking CUDA stream and event,
so a delayed source cannot impose a host-side wait on other sources. Physical copy
engines may still serialize transfers. Self payload retains the direct CuTe-buffer
path; empty payloads skip the copy but record completion normally.

The worker returns only after all sources are installed. The existing consumption
barrier, iteration counters, timeout/abort checks, and collective IPC teardown remain
in force. The completion publication still means a source finished writing **all**
destinations; this feature does not introduce finer per-destination GPU signals.

Per-source mode changes the interpretation of these existing timing fields (the
terminal labels and JSON `readback_timing_semantics` identify the distinction):

- `sender_completion_wait_ms`: elapsed worker time until local completion is observed,
  including any overlapping peer readback work, rather than an isolated wait.
- `peer_completion_wait_ms`: elapsed worker time until all source publications are
  observed, overlapping earlier readbacks/installations, rather than blocked time.
- `headers_d2h_ms` and `payload_d2h_ms`: sums of eight per-source submission-to-observed-
  completion latencies, including CPU scheduling/query delays. These intervals overlap
  and can exceed native/end-to-end wall time. Payload latency includes header checking.
- `install_ms`: summed CPU validation/installation/reset time across all eight sources.

Use `end_to_end_ms` and `native_ms` for performance comparison; do not subtract the
overlapping per-source stages from native time to derive an overhead residual. The
extra small header copies, event queries and streams can offset overlap benefits;
speedup must be measured on the remote machine.

Pinned views are owned by the native proxy and valid until its next submission or
destruction. Only the native worker installs them. The Python API exposes snapshots
solely for out-of-band checking. There is only one in-flight iteration. Setup unlinks
the shared-memory name after all mappings exist; normal teardown closes all CUDA IPC
imports, performs a barrier, and only then frees the owning GPU arenas. Native waits
check a timeout and shared abort counter. Torchrun terminates the job if a rank fails;
there is no attempt to recover a CUDA context after an asynchronous device error.

## Measurements and validation

JSON contains every measured per-rank sample, distributions, configuration, versions,
and the distribution of **maximum per-rank end-to-end duration** for each iteration.
This maximum uses aligned starts and local durations, not a cross-device timestamp
subtraction. CUDA event clocks are never compared across GPUs.

The terminal also prints median/mean/p95 timings in **microseconds**, selecting the
rank with the longest end-to-end duration on each iteration and using that same
rank's measurements for every row. These distributions are stored as
`critical_rank_metrics`, with selected ranks in `critical_rank_per_iteration`.
`all_rank_metrics` retains distributions over all rank/iteration samples. JSON
timing fields remain in **milliseconds** (schema version 5).

| Field | Meaning |
|---|---|
| `end_to_end_ms` | Python wall time from metadata submission through native installation/wait return |
| `metadata_gpu_ms` | CUDA event span around the seven kernels, including submission gaps |
| `exchange_gpu_ms` | CUDA event span around the direct-write kernel |
| `sender_completion_wait_ms` | Native worker wait for local stream completion, possibly including unfinished metadata |
| `peer_completion_wait_ms` | Wait after local publication for all peer publications |
| `headers_d2h_ms` | Wall time for header submission and completion |
| `payload_d2h_ms` | Header checks, exact payload submissions, and completion |
| `install_ms` | CPU validation, view installation, and progress reset |
| `native_ms` | Native submit through worker installation, before return to Python |
| `host_outside_native_ms` | End-to-end minus native duration: host submission before the native timer and wakeup/return overhead after it; not isolated Python CPU time |
| `received_index_bytes` | All D2H index payload bytes, including self |
| `peer_received_index_bytes` | Received peer index bytes, excluding self and headers |
| `header_bytes` | Header D2H bytes (576 per rank per iteration) |

These timings overlap and **must not be added together**. Reported bytes are logical
payload bytes, not hardware link transaction counters. The all-senders completion
wait, header round trip, and eight small payload submissions may dominate latency;
the experiment makes these costs visible. Optional per-source progress is described
above; GPU-only readiness is not implemented here. The table describes default-mode
timings; per-source mode uses the explicitly documented overlapping measurements.

Full timed index validation is enabled by default. `--skip-index-validation` omits
index scans but retains structural header bounds/order checks. `--verify boundaries`
(default) checks the first warmup and final measured result against the independent
CPU reference, including every received source table, GPU generator, all meaningful
CuTe outputs, and input scan scratch. Use `--verify all` for every invocation or
`--verify none` only after remote correctness has been established.

Local host-only checks (including on macOS):

```bash
cmake -S . -B /tmp/rme2-host-build -DRME2_HOST_TESTS_ONLY=ON
cmake --build /tmp/rme2-host-build -j
ctest --test-dir /tmp/rme2-host-build --output-on-failure
```

These test the actual production CPU routing compatibility, generator equivalence,
index conventions, zero-length lists, borrowed-view installation, progress reset,
seed wraparound, and malformed metadata. They do **not** compile or execute the
CUDA/CuTe kernels or validate NVLink performance. The native CUDA build and
`run_gpu_checks.sh` must be run on the remote eight-GPU machine.
