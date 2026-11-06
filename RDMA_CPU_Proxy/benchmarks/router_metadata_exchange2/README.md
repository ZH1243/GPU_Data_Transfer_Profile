# GPU router metadata exchange, one or two nodes

The default single-node mode runs **eight torchrun Python processes on one Linux Hopper node**.
Each process owns one GPU and loads `librouter_metadata_exchange2.so`. That library
owns a C++ proxy thread. Python generates fresh routes and launches seven CuTe DSL
metadata kernels; a CUDA C++ kernel writes destination-specific metadata directly
into CUDA-IPC-mapped peer GPU buffers. Each receiver proxy reads the results into
persistent pinned memory and installs pointer/count views of that memory.
After the GPU exchange kernel, a CuTe gather rebuilds the local token staging
buffer each iteration, overlapping CPU metadata readback/communication when possible.

This is an isolated **benchmark-specific proxy**, not the production `Proxy` class
or `librdma_cpu_proxy.so`. The production class currently generates/exchanges
metadata internally during initialization, owns vector-backed records, and rejects
reinstallation. This experiment implements the proposed replacement data path and
production routing/index conventions without changing those APIs. Host tests compare
the new reference and installed views against the actual production `RouterRouting`
CPU implementation. It does not measure inter-GPU token payload dispatch, GEMMs,
GPU `A_idx`/scheduler construction, local `x3/x4` CPU installation, or combine.
In single-node mode the completion endpoint is **all eight source expert tables
installed at the CPU proxy and local token staging complete**. Two-node mode also
exchanges remote `x3/x4`, as below.

## Local token staging (default)

Both one- and two-node modes now include the existing CuTe `LocalStagingPlan`
from `get_expert_token_idx_list_gpu/metadata_cute.py`. Each process owns reusable
source/destination BF16 tensors `[tokens, token_dimension]`; `--token-dimension`
defaults to 4096. At T=20,480 these add 160 MiB each (320 MiB total per GPU),
outside the registered metadata arenas. The vectorized 128-bit gather is used
for aligned rows; odd widths such as 7 select its existing byte-copy fallback.

The gather reads the current local-node slice directly on the GPU:

```text
begin = node_offsets[node]
rows  = node_offsets[node + 1] - begin
staging[row, :] = input_tokens[x3[begin + row], :],  0 <= row < rows
```

No row count is copied to the host before the gather. The slice can grow, shrink
or become empty between iterations; the unused destination tail is untouched.
Synthetic activation bit patterns change with rank, row, column and iteration.
Activation refresh, allocations, compilation, graph capture/upload and verification
are outside timing, just like route generation. The gather itself is timed.

The stream/progress ordering is:

```text
GPU work stream: seven metadata kernels -> exchange -> exchange-ready event
                                                        -> local gather -> staging-done event
CPU worker:                    wait for exchange-ready -> publish/wait local producers
                                                        -> header D2H -> local payload D2H + RDMA
                                                        -> install metadata / drain sends
                                                        -> wait for remaining staging tail -> return
```

Exchange readiness is recorded **before** the gather, so CPU publication,
readback on separate CUDA streams, and RDMA do not wait for staging completion.
The gather uses separate activation buffers and reads this source's local `x3`;
it needs no incoming remote metadata. There is no device-wide synchronization
inside timing. Hardware overlap is permitted, not guaranteed: gather traffic may
contend with D2H/RDMA for GPU memory bandwidth. The consumption barrier protects
both metadata and staging inputs/outputs from premature reuse.

The gather is captured as a **separate single-kernel CUDA graph**, launched by
native code after exchange in every mode. `--mode eager/graph` continues to
select how the seven metadata kernels are submitted; `--native-metadata-launch`
moves their graph launch to the native worker. The staging graph is never included
inside the metadata graph or inside the exchange GPU timer.

`local_staging_gpu_ms` measures the gather's CUDA event span.
`local_staging_wait_ms` measures only the residual CPU wait after metadata work
finishes (including completion-query overhead). `native_ms` and `end_to_end_ms`
now require **both metadata completion and staging completion**. Do not add the
gather GPU duration to the D2H/RDMA spans. JSON schema 10 records the selected
staging workload and expanded completion endpoint.

Use `--skip-local-staging` on all ranks for the metadata-only baseline; this omits
activation allocations/refresh and the gather, and reports both staging timings
as zero. Existing commands enable staging automatically. Update Python files and
rebuild the native library on both nodes for **ABI 9** before running.

## Two-node exchange (ABI 9)

The benchmark also supports **two Linux Hopper nodes, eight GPUs each**, keeping
128 experts and top-8, hence **eight experts per GPU**. It reuses the existing
seven-kernel `get_expert_token_idx_list_gpu/metadata_cute.py` implementation.
Both nodes generate fresh, independently seeded routes every iteration. Source
identity is global rank; mask ordering and the direct-input exception use the
original source's local GPU index. Seeds are
`(seed + global_rank + 16*(iteration-1)) mod 2^64`.

Two-node mode exchanges **expert tables plus the same-index remote source's
`x3/x4` slice every iteration**, enabled by default with no additional flag.
This matches the production test's `x3/x4` source/destination convention while
using GPU-to-host RDMA instead of its initialization-time TCP exchange. Update
Python and rebuild the library on **both nodes**: ABI 9 / wire format 2 is not
compatible with older benchmark binaries. Existing benchmark commands still work,
but their end-to-end timings now include `x3/x4` packing, transfer and installation
and, by default, local token staging completion.

Build on **both remote machines**, from this benchmark directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DRME2_REQUIRE_RDMA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Use the same Python files and rebuilt shared library on both nodes. Requirements
in addition to the single-node CUDA/CuTe environment: libibverbs/rdma-core headers
and library, a working GPU-memory registration path (the `ibv_reg_mr` CUDA pointer
path, typically `nvidia-peermem`), a compatible NIC/GPU PCIe topology, and enough
locked-memory allowance for the registered arenas. There is no CPU staging or TCP
payload fallback and no DMA-BUF registration backend in this implementation.
`RME2_REQUIRE_RDMA=ON` fails configuration without verbs; otherwise a CUDA build
can still run single-node mode and rejects two-node use without verbs support.

For an initial full-content check, run the corresponding command on each node:

```bash
# Node 0
bash ./run_node0_torchrun.sh --tokens 257 --warmup 1 --iterations 6 \
  --route-pattern cycle --verify all
# Node 1
bash ./run_node1_torchrun.sh --tokens 257 --warmup 1 --iterations 6 \
  --route-pattern cycle --verify all
```

The scripts default to rendezvous `28.59.80.49:29520`. Set `MASTER_ADDR` and
`MASTER_PORT` consistently if needed. The default NIC map matches the production
scripts: `mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5`.
Override `RDMA_DEVICE_MAP` or `--rdma-device-map` with eight names indexed by local
CUDA ordinal, or one NIC name for all proxies. `--rdma-port` defaults to 1 and
`--rdma-gid-index` to 3; use -1 for InfiniBand LID routing. The RC connection uses
MTU 4096 and the production path's RoCE hop-limit/retry settings. Check these
against the target fabric. CPU core maps and NIC maps may differ between nodes;
workload and protocol options must match. Keep all eight GPUs visible to every
local worker, as in single-node mode. Only global rank 0 writes the result JSON.

After correctness checks, omit the small-workload arguments for the default
20,480-token run. Eager, graph, `--native-metadata-launch`, and the two optional
node-local metadata barriers work in two-node mode. For example, pass on both nodes:

```bash
--native-metadata-launch --metadata-launch-barrier \
  --skip-index-validation --warmup 20 --iterations 200
```

This first two-node path uses the all-local-senders readiness gate. It explicitly
rejects `--per-source-readback` and `--batch-payload-copy`; those optimizations
remain available in single-node mode.

### Two-node data path and lifetime

1. Each receiver/relay owns a dedicated CUDA IPC arena and a persistent pinned
   host arena. Both are registered once with verbs. The HBM arena has two banks
   of eight source slots: one bank per destination node. Each record contains a
   64-byte header followed by capacity for `T*8` int32 indices. An additional
   `x3/x4` record holds a 64-byte header and capacity for `T` int32 token IDs plus
   `T` uint8 masks. A separate 1,088-byte directory duplicates all 17 headers for
   one small D2H transfer. At T=20,480, each proxy reserves approximately 10.1 MiB
   each of HBM and pinned host records.
2. After the seven metadata kernels, 16 CTAs (one per global destination) write
   both destination-node slices into the appropriate local relay GPU. All payloads,
   including relay-self slices, enter the registered arena. Records preserve original
   source identity, destination, full 64-bit iteration, and nine normalized offsets.
   The same-index remote CTA also packs its own source's remote-node `x3/x4`
   slice into its registered local arena; it does not collect other sources'
   `x3/x4`. For source node `n`, the slice is
   `[node_offsets[1-n], node_offsets[2-n])`. The packed record contains its header,
   then the valid int32 `x3` entries, then the valid uint8 `x4` masks. No additional
   packing kernel is launched. Every writer system-fences its writes before its
   exchange-completion event. The local staging gather follows that event on the
   same GPU stream and can overlap the subsequent CPU/D2H/RDMA work.
3. CPU proxies publish producer event completion through **node-local** shared
   counters. Each relay waits for all eight local producers. GPU registration enables
   `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`; NIC reads are posted only after producer
   completion and directory readback. The relay's own event alone is insufficient.
4. Each relay validates the directory, queues eight exact-size local D2H payload
   copies, and posts nine signaled RDMA Writes With Immediate on one same-index
   RC connection: eight expert records and one `x3/x4` record. Each RDMA source is
   the relay's HBM record and each destination
   is the remote proxy's final pinned record. There is no sender CPU payload copy,
   extra GPU packing pass, or receiver CPU payload copy. Empty lists still transfer
   a header and produce a receive completion.
5. The worker polls CUDA and RDMA completions together. Local D2H completion
   permits installation of the eight local-source views. Each successful RDMA
   receive completion permits identity/epoch/length validation and installation
   of that original remote source's view. Immediate tags 0..7 identify expert
   sources; tag 8 identifies `x3/x4`. Its receive installs borrowed pointers to the
   final pinned token IDs/masks and their count. Nine receive WQEs are preposted;
   credits are replenished during timing. CQ polling uses a fixed-size stack buffer.
6. `wait()` returns after **all 16 source expert tables and the remote same-index
   `x3/x4` slice are installed, all nine outgoing sends complete, and local token
   staging finishes** (unless `--skip-local-staging`).
   Empty `x3/x4` slices still send a header and require a completion.
   The outer all-16-rank consumption barrier protects
   record reuse, including verification. There is one iteration in flight. Setup,
   route/activation generation, graph compilation, verification, and outer barriers remain outside
   end-to-end timing. Optional native barriers synchronize only the eight workers
   on each node and remain timed.

The global rank mapping is `node * 8 + local_gpu`; torchrun rank groups determine
node identity, so the two machines may report the same container hostname.
Gloo exchanges IPC handles and
QP/MR descriptors during setup, then provides iteration boundaries and reporting;
no expert or `x3/x4` payload uses Gloo. IPC mappings and shared-memory counters stay
local to each node. Normal shutdown stops workers, closes every IPC import,
collectively waits, destroys QPs/deregisters MRs, and frees their arenas. A local
worker failure publishes node-local abort; cross-node failures surface through
CQ errors, bounded native/Gloo waits, and torchrun termination. There is no recovery
of an in-flight failed iteration.

The endpoint is **expert tables and remote `x3/x4` installed at benchmark CPU
proxies, plus local staging completion**. The received `x3/x4` view is not wired
into production token dispatch. Local `x3/x4` CPU installation, GPU `A_idx`,
scheduler reconstruction, inter-GPU token movement, GEMM, and combine remain excluded.
This does not change the production proxy or its initialization-only metadata API.

### Two-node timing and remote validation

JSON schema 10 reports `rdma_post_ms` (record checks and write submissions),
`rdma_poll_ms` (time inside CQ polling), `rdma_post_recv_ms` (credit reposting), and
`rdma_work_complete_ms` (span from starting submissions until all send/receive
completions are processed). `rdma_sent_bytes` and `rdma_received_bytes` include
all nine record headers and both expert and `x3/x4` payloads.
`x3_x4_sent_bytes` and `x3_x4_received_bytes` separately count only the `x3/x4`
payload (five bytes per routed token, excluding its header). The outgoing and
incoming lengths may differ. `received_index_bytes` includes all 16 installed
expert-index payloads, excluding `x3/x4`;
`peer_received_index_bytes` excludes only this process's original source record.
`header_bytes` is the 1,088-byte directory D2H. RDMA and `x3_x4_*` byte fields are zero in single-node
mode. Per-rank reports include node/local rank and selected NIC; critical-rank
statistics select among all 16 ranks.

Local payload copies overlap RDMA work. `payload_completion_ms` includes CPU
observation delay while submitting/progressing RDMA; it is not pure DMA time.
`rdma_work_complete_ms` overlaps D2H and includes receiver validation/installation
and progress scheduling. Do not add overlapping stage times. Compare the overall
`end_to_end_ms` distribution. Source payload volume remains `T*8` indices across
all 16 destinations, plus up to `5*T` bytes of outgoing `x3/x4` per source
(100 KiB at T=20,480). `exchange_gpu_ms` includes packing this record;
`install_ms` includes its receiver validation/view installation. These costs are
also included in `end_to_end_ms`. `x3_x4_exchange` in JSON identifies the expanded
two-node workload; older expert-only measurements have a different endpoint.

Run extended correctness checks on both nodes (same optional flags):

```bash
# Node 0 / node 1 respectively
bash ./run_two_node_gpu_checks.sh 0
bash ./run_two_node_gpu_checks.sh 1
# Repeat with --native-metadata-launch --metadata-launch-barrier --metadata-barrier
```

These cover eager/graph execution, partial chunks and scan carry, seed wraparound,
empty local/remote receives, changing lengths, every installed source table, and
the exact remote `x3/x4` token IDs/masks. `cycle` alternates affine/hot-low/hot-high,
so each direction exercises empty, full and changing `x3/x4` lengths in reused
allocations. Exact verification also checks all reported byte counters and every
staged activation bit against the independent CPU-reference local `x3` slice,
plus untouched-tail sentinels. The GPU scripts also exercise staging dimensions
7 and 8 (fallback and partial vectorized rows) and `--skip-local-staging`.
The host suite compares both topology references with production CPU routing and
checks record identity, length bounds, borrowed installation, and launch topology.
Host checks cannot establish CUDA compilation, peer-write/NIC visibility, or RDMA
correctness/performance; those checks require the two remote Hopper nodes.

## Single-node mode

The sections below describe the preserved single-node path and its options.

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

# Paired metadata modes; staging always uses a separate native graph.
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
normalized to zero. In single-node mode `x3/x4` are computed and verified but not
exchanged. Two-node mode exchanges the remote-node slice as described above.

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

1. Generate R, refresh input activations, finish generation, and align ranks with
   a CPU collective outside timing.
2. Start the Python wall timer. Launch the seven metadata kernels (eager or graph).
3. Queue the exchange kernel on the **same CUDA stream**. One CTA per destination
   reads source offsets and directly writes the destination header and index slice.
   Every writer executes a system fence. The self CTA writes only a header.
   Record exchange completion, then queue the local staging graph on this stream.
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
7. Wait for any remaining local staging work, return to Python and stop the wall
   timer. Optional exact verification runs outside
   timing. A consumption barrier then prevents ANY producer from reusing its output
   or inbox while another proxy still reads it.

The header copy is needed because host-submitted D2H copies require host-known sizes.
Payload capacity is conservative; transfer size is exact. The design avoids sender
D2H, sender packing buffers, self peer payload copies, and receiver CPU payload copies.
It is not zero-copy: peer payloads still move GPU-to-GPU and then GPU-to-CPU.

### Optional batch payload copy (CUDA 12.9)

`--batch-payload-copy` replaces the receiver's nonempty payload copy loop with one
`cudaMemcpyBatchAsync`. It is off by default and requires `--per-source-readback`
to be disabled. It works with Python or native metadata launch and either barrier.
Update Python and rebuild the shared library together (**ABI 9**), using the remote
CUDA 12.9 toolkit:

```bash
cmake --build build -j
bash ./run_gpu_checks.sh --batch-payload-copy --native-metadata-launch --metadata-launch-barrier

RESULT_JSON=results/batch_payload.json bash ./run_single_node_torchrun.sh \
  --native-metadata-launch --metadata-launch-barrier \
  --batch-payload-copy --skip-index-validation --warmup 20 --iterations 200
```

Repeat without `--batch-payload-copy` for a matched baseline. Build-time guards keep
older CUDA builds usable for the default path, but explicitly reject enabling this
feature without CUDA 12.8+ headers/runtime. There is no silent fallback if the batch
API fails. Ensure CMake selected the CUDA 12.9 toolkit, not an older cached installation.

After the unchanged header D2H and structural checks, persistent fixed-size arrays
hold each nonempty copy's source, destination and exact byte length. Self metadata
still reads directly from CuTe output. The batch uses stream-ordered source access,
with one attribute group covering all entries. Payloads are independent and their
pinned destinations are disjoint, so no intra-batch ordering is required. The same
single completion event gates CPU installation and the iteration reuse barrier.
Empty destinations submit no copy; an entirely empty payload set skips the batch
call but still records completion. No payload packing or extra payload copy occurs.

`payload_submit_ms` now measures the single batch API call, labelled accordingly in
terminal output. Descriptor preparation is included in `payload_checks_ms`; tiny
attribute setup remains in `payload_other_ms`. The total and all other subrows retain
their meanings. JSON configuration records the selected mode. Errors identify the
source corresponding to CUDA's failed batch index when available.

The API reduces host submissions; it does not guarantee one contiguous DMA transfer
or improved end-to-end latency. See the [CUDA 12.9 batch-copy API documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__MEMORY.html).

### Native worker metadata launch and pre-launch barrier

`--native-metadata-launch` moves metadata graph replay and exchange submission onto
the native proxy worker. It requires `--mode graph` (the default). Add
`--metadata-launch-barrier` to synchronize those workers through shared CPU memory
immediately before they record the start event and launch the metadata graph.
The launch barrier requires native launch; invalid option combinations fail early.
Both flags are off by default, preserving the existing Python launch path.

Update the Python files and **rebuild the library (ABI 9)**:

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

Update Python and rebuild the library together (native **ABI 9**):

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
together**: the native interface is now ABI 9; Python rejects older libraries.

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
timing fields remain in **milliseconds** (schema version 10).

| Field | Meaning |
|---|---|
| `end_to_end_ms` | Python wall time from metadata submission through installation, send completion, staging completion and wait return |
| `metadata_gpu_ms` | CUDA event span around the seven kernels, including submission gaps |
| `exchange_gpu_ms` | CUDA event span around the direct-write kernel |
| `local_staging_gpu_ms` | CUDA event span around local token-row gather; zero when skipped |
| `local_staging_wait_ms` | Residual CPU wait for gather completion after metadata exchange/installation; includes query overhead |
| `sender_completion_wait_ms` | Native worker wait for local stream completion, possibly including unfinished metadata |
| `peer_completion_wait_ms` | Wait after local publication for all peer publications |
| `headers_d2h_ms` | Wall time for header submission and completion |
| `payload_d2h_ms` | Header checks, exact payload submissions, and completion |
| `payload_checks_ms` | Sum of CPU header checks and source/destination address and length preparation |
| `payload_submit_ms` | Sum of wall durations of nonempty `cudaMemcpyAsync` calls, or the single batch call when enabled, including CUDA API overhead |
| `payload_event_record_ms` | CPU wall time recording payload completion events |
| `payload_completion_ms` | Batched mode: polling/yielding until the completion event is observed; per-source mode: summed intervals from event recording to observed completion, including other source work between polls |
| `payload_other_ms` | Parent payload time minus the four measured subintervals: bookkeeping, loop and timestamp-boundary overhead |
| `install_ms` | CPU validation, view installation, and progress reset |
| `native_ms` | Native submit through metadata installation, outgoing sends and staging completion, before return to Python |
| `host_outside_native_ms` | End-to-end minus native duration: host submission before the native timer and wakeup/return overhead after it; not isolated Python CPU time |
| `received_index_bytes` | All received expert-index payload bytes, including self; excludes x3/x4 |
| `peer_received_index_bytes` | Received peer index bytes, excluding self and headers |
| `header_bytes` | Header D2H bytes (576 single-node; 1,088 two-node, per rank per iteration) |
| `x3_x4_sent_bytes`, `x3_x4_received_bytes` | Two-node x3/x4 payload bytes, excluding headers; zero single-node |

The five `payload_*` subrows partition `payload_d2h_ms` per sample. Their **means**
add to the parent mean, up to rounding; medians and p95 values need not add. Do not
add the subrows to the parent again. Timing is always enabled, so rebuild the native
library and update Python together to obtain this breakdown. Extra clock reads can
increase measured latency slightly, particularly for very small copies/checks.
Copies may execute while subsequent CUDA API calls are being submitted; the final
completion wait measures only residual CPU-observed latency, not total DMA time.
Per-source mode retains overlapping latency sums, not additive wall stages.

The other timings overlap and **must not be added together**. Reported bytes are logical
payload bytes, not hardware link transaction counters. The all-senders completion
wait, header round trip, and eight small payload submissions may dominate latency;
the experiment makes these costs visible. Optional per-source progress is described
above; GPU-only readiness is not implemented here. The table describes default-mode
timings; per-source mode uses the explicitly documented overlapping measurements.

Full timed index validation is enabled by default. `--skip-index-validation` omits
expert-index and x3/x4 token/mask scans but retains structural header bounds/order,
record identity, epoch and length checks. `--verify boundaries`
(default) checks the first warmup and final measured result against the independent
CPU reference, including every received source table, GPU generator, all meaningful
CuTe outputs, input scan scratch, received remote x3/x4 in two-node mode, and byte
counters. With staging enabled, it also checks every valid gathered row and the
untouched tail. Input activations change each iteration even for fixed hot routes.
Use `--verify all` for every invocation or
`--verify none` only after remote correctness has been established.

Local host-only checks (including on macOS):

```bash
cmake -S . -B /tmp/rme2-host-build -DRME2_HOST_TESTS_ONLY=ON
cmake --build /tmp/rme2-host-build -j
ctest --test-dir /tmp/rme2-host-build --output-on-failure
```

These test the actual production CPU routing compatibility, generator equivalence,
index conventions, zero-length lists, borrowed-view installation, progress reset,
seed wraparound, malformed metadata and C/ctypes stats layout. They do **not** compile or execute the
CUDA/CuTe kernels or validate NVLink performance. The native CUDA build and
`run_gpu_checks.sh` must be run on the remote eight-GPU machine.
