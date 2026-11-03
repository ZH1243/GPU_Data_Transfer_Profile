# CPU router metadata exchange benchmark

This benchmark embeds a **C++ shared library** in each torchrun Python worker.
`benchmark.py` parses options and rank environment variables, loads the library
with `ctypes.CDLL`, and starts one proxy thread. C++ performs connection setup,
input preparation, synchronization, serialization, exchange, validation, CPU
installation, all timing, and report generation. Python performs no metadata
processing or communication and the GIL is released during the native call.

The experiment remains CPU-only: no CUDA, RDMA verbs, GPU allocations, actual
token transfers, or GPU scheduler initialization. It reuses the main project's
`RouterRouting` CPU generator, `protocol.cpp` serializers/deserializers, and
`compute_chunks_from_token_indices()`. The standalone library does not initialize
the full GPU/RDMA `Proxy` class. Binary array transfer is now the default. `--wire-format=text` retains the original
text baseline. Self-table copying and full validation remain enabled. Changes
are confined to this benchmark; the production proxy protocol is unchanged.

## Build on both remote machines

From this directory, before starting torchrun:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/librouter_metadata_bench.so` on Linux (`.dylib` on macOS).
The build needs a C++17 compiler, CMake, and POSIX sockets/threads; it requires
neither CUDA nor libibverbs and does not modify the main proxy build. Build once
before launching workers, not separately in every torchrun process. Rebuild on
both nodes after native source changes.

Python 3.9+ and torchrun are needed for launching; there is no Torch distributed
process group. Override the shared-library location with `--library=/path/to/lib`
or `ROUTER_METADATA_BENCH_LIBRARY`. The C ABI is declared in `native_proxy.h`;
exceptions return a nonzero status and an error string to Python. Native sockets
use deadlines and close via RAII on success or failure.

## Run on two nodes

From this directory, start the corresponding command on each machine:

```bash
# Node 0
bash ./run_node0_torchrun.sh

# Node 1
bash ./run_node1_torchrun.sh
```

Defaults match the current proxy scripts: 8 processes per node, 20,480 tokens
per process, 128 experts, top-8, hidden dimension 4,096, two-byte elements, and
32 tokens per receive chunk. Each process represents one GPU proxy but does
not access a GPU. There are 4 precomputed routing datasets, 5 warmup iterations,
and 50 measured iterations.

The launchers default to node addresses `28.59.80.49` and `29.162.224.226`, taken
from the existing node configurations. Override addresses on **both nodes** if
needed:

```bash
NODE0_HOST=10.0.0.10 NODE1_HOST=10.0.0.11 \
  bash ./run_node0_torchrun.sh --iterations=100 --datasets=8

NODE0_HOST=10.0.0.10 NODE1_HOST=10.0.0.11 \
  bash ./run_node1_torchrun.sh --iterations=100 --datasets=8
```

Use identical workload options on both nodes. Supported launcher environment
variables:

| Variable | Default | Purpose |
|---|---|---|
| `NODE0_HOST`, `NODE1_HOST` | Addresses above | TCP metadata node addresses |
| `MASTER_ADDR` | `NODE0_HOST` | Torchrun rendezvous address |
| `MASTER_PORT` | `29510` | Torchrun rendezvous port |
| `NPROC_PER_NODE` | `8` | Processes representing GPUs on each node |
| `METADATA_PORT_BASE` | `20515` | Listener port = base + local rank |
| `TORCHRUN_BIN` | `torchrun` | Launcher executable |
| `ROUTER_METADATA_BENCH_LIBRARY` | `build/librouter_metadata_bench.so` | Native library path |
| `RESULT_JSON` | `results/metadata_exchange.json` in this directory | Rank-0 report |

Only node 0 writes the report; the same path is overwritten on later runs.
Use a different `RESULT_JSON` for each comparison. Choose unused rendezvous and
metadata ports for concurrent jobs. TCP follows the OS route to these addresses,
not the RDMA device mapping from the main proxy test. For meaningful measurements,
use the intended network interface, an otherwise quiet machine, and consistent
CPU/NUMA placement. This benchmark does not pin processes automatically.

## Optional within-node shared memory

Rebuild the library and update `benchmark.py` on both nodes (C ABI version 4).
The default remains all-TCP. To replace local expert-metadata TCP connections
with shared host memory, pass `--local-shared-memory` on **both nodes**:

```bash
# Node 0, from this benchmark directory
RESULT_JSON=results/shared_memory.json bash ./run_node0_torchrun.sh \
  --local-shared-memory --disable-expert-index-validation

# Node 1
bash ./run_node1_torchrun.sh \
  --local-shared-memory --disable-expert-index-validation
```

This option requires binary mode. Remote expert metadata and x3/x4 still use
TCP, self metadata still uses the existing local copy, and control collectives
still use TCP outside timing. For eight proxies per node, each proxy replaces
seven local expert channels, retaining eight remote expert channels and one
remote x3/x4 channel. Use matching options on every rank; topology-affecting
option mismatches can fail during connection setup before configuration checking.

One POSIX shared-memory region is created per logical node. A unique identity
is distributed through the control mesh; no manual run ID is required. Each
source owns a cache-line-aligned arena containing packed destination slices,
counts, and a release/acquire sequence counter. Capacity is bounded by
`num_tokens * top_k` indices per source across all local destinations, rather
than reserving that bound per destination. Default shared storage is about
5 MiB per node. Local workers must share the same POSIX shared-memory namespace
and permissions; Linux containers need sufficient `/dev/shm` capacity.

Allocation, mapping, and producer-arena prefaulting occur before timing. Each
iteration copies current dataset arrays into shared memory **inside timing**,
then publishes them. Receivers copy their slices into reusable CPU records and
apply the same validation policy as TCP reception. The existing global
completion collective prevents overwrite until every reader has finished.
The region is unlinked after all workers map it and unmapped on teardown;
exception cleanup also unlinks a region owned by that worker. An uncatchable
termination during initial setup can leave a named object, but a new run uses
a different identity.

While local arrivals are pending, the proxy scans sequence counters and polls
remote sockets without blocking. After local reception completes, normal
blocking socket polling resumes. This intentionally consumes CPU while waiting
for local producers; keep CPU/NUMA placement consistent in comparisons.

Reports include `local_shared_memory` in configuration. Per-iteration
`shm_sent_bytes` / `shm_received_bytes` count offsets and indices copied through
shared memory (excluding publication fields). Existing `sent_bytes` /
`received_bytes` count only TCP metadata wire bytes. A single-node shared-memory
run therefore has zero TCP metadata bytes, although control traffic remains.

For compatibility, `tcp_exchange_ms` retains its field name but, with this
option, measures **the combined TCP/shared-memory exchange stage**, including
shared payload copies and readiness polling, excluding receive processing.
Each per-rank record also exposes the same value as `exchange_ms`; these are
aliases and must not be added together. The five existing stages still sum to
`elapsed_ms`. Compare `critical_rank_ms` for end-to-end benefit rather than
interpreting the legacy stage name as isolated TCP time.

For correctness, add `--verify` on both nodes to compare every received table
against source-specific expected data outside timing. Local subprocess tests
cover repeated datasets, local-only and hybrid transports, empty lists,
pre-serialized mode, disabled validation, one proxy per node, and 16 proxies
with the default workload. These are CPU correctness tests, not Hopper
performance measurements.

## Protocol and input fidelity

By default, every process establishes a full-duplex persistent expert-metadata connection
with every other process. Matching local ranks on different nodes additionally
establish a separate x3/x4 connection. Same-node expert messages use IPv4
loopback; remote messages use `--hosts`. Control synchronization has separate
connections to rank 0. All sockets use `TCP_NODELAY`.

Input preparation follows the router's SplitMix64/start/stride generation,
stable descending node-mask ordering, normalized expert offsets, and compact
destination-buffer indexing. Seeds vary by source and dataset:
`seed + source_rank + dataset_index * world_size`. Inputs are generated before
measurement and cycled across iterations. This models changing routing without
including route generation time or retaining an unbounded number of datasets.

Each source sends only the eight destination experts' metadata (with the default
topology) to each recipient. x3/x4 are sent only to matching local ranks on remote
nodes. Self expert metadata is copied locally. For 16 processes, each iteration
contains 240 directed expert messages and 16 directed x3/x4 messages.

Binary mode sends the existing contiguous array storage directly with `sendmsg()`
scatter/gather I/O. A message consists of three segments: the framing/protocol
header, the first array, and the second array. No decimal conversion or packed
array copy occurs on the sender. Partial sends advance across segment boundaries.
Input datasets stay alive until all sends finish.

The 16-byte network-order envelope contains payload length and iteration number.
The binary header contains a magic/version word, message kind, identities,
dimensions, and array counts, all as big-endian uint64 fields. Expert messages
have 13 header words; x3/x4 messages have 14. Arrays follow immediately:

| Message | First array | Second array |
|---|---|---|
| Expert metadata | Little-endian `int32` offsets | Little-endian `int32` indices |
| x3/x4 | Little-endian `uint64` x3 indices | `uint8` x4 masks |

The x3 width matches the existing CPU `vector<size_t>` representation. Binary
mode explicitly requires little-endian hosts with 64-bit `size_t` (the target
Hopper hosts satisfy this). It does not transmit raw C++ structs or pointers.
Header/count/length checks run before allocations and array access. The receiver reads the fixed-size header first, checks its counts and envelope
length, and receives each array directly into source-specific typed vectors.
Their capacities are reserved once before warmup and retained across iterations.
There is no full binary payload buffer and no payload-to-vector `memcpy`.
Ordinary TCP still copies from kernel socket storage into application memory;
this is not kernel-bypass or zero-copy networking.

Expert index capacity is conservatively reserved at `num_tokens * top_k` entries
per source, offsets at `experts_per_gpu + 1`, and x3/x4 at `num_tokens` each.
This trades memory capacity for avoiding per-iteration reallocation. Vector
lengths still change to match each message, and growth within reserved capacity
can zero-initialize newly exposed elements. That work remains measured.
Incomplete or invalid messages never reach computation/verification; validation
runs only after both arrays finish arriving, before the exchange completes.

Text mode uses production `router_expert_metadata_v1` / `router_x3_x4_v2` bodies
with the same iteration envelope. All ranks must choose the same wire format;
configuration exchange checks this before measurements. Neither benchmark mode
is a wire-compatible endpoint for a running production proxy.

A single nonblocking C++ `poll()` loop progresses sending and receiving on both
metadata paths. They can overlap on the network; native header preparation/serialization and
decoding execute serially on each proxy thread. Socket addresses use IPv4. Received expert lists are validated
and attached to source-specific CPU records, including zeroed head/tail state.
Received x3/x4 are validated, retained, and used to construct receive-chunk
descriptions with the same fields as `compute_chunks_from_token_indices()`.
No GPU `A_idx` allocation, gather-table publication, or H2D copy occurs.

## Timing and output

Each iteration has this sequence:

1. Barrier on separate control sockets, outside timing.
2. Start a local monotonic timer.
3. Prepare binary headers, or serialize outgoing text in text mode.
   `--preserialized` moves this preparation outside timing.
4. Copy the self expert table; frame and exchange peer messages.
5. Decode, validate, and install all received CPU metadata.
6. Stop after all local sends and all local installations complete.
7. Gather timing records and release all ranks, outside timing.

Connection setup, CPU routing generation, warmup, result aggregation, logging,
and disposal of the installed records after the iteration are excluded.
Serialization, framing, poll descriptor setup, receive buffering, parsing, validation,
self installation, and receive-chunk construction are included. Binary metadata records and typed storage are reused in place. Text-mode records
are rebuilt; receive-chunk descriptors are still rebuilt in both modes.

The report identifies `backend: "cpp"` and `wire_format: "binary"` or `"text"` (in configuration),
and contains all measured per-rank/per-iteration records and:

- `critical_rank_ms`: statistics of the **maximum rank-local elapsed time** for
  each iteration (minimum, mean, median, nearest-rank p95/p99, maximum).
- `critical_rank_stage_ms`: stage statistics from the same slowest-total-time
  rank selected in each iteration, printed to the console as well as saved.
  These are not independently selected stage maxima from different ranks.
- `rank_summaries`: per-rank elapsed and stage statistics.
- `sent_bytes` / `received_bytes`: wire bytes, including the 16-byte benchmark
  envelopes but excluding handshakes and control traffic.
- Installed expert and x3/x4 source counts for each rank.

Every per-rank record contains five non-overlapping local wall-time stages:

| Field | Included work |
|---|---|
| `serialize_ms` | Binary protocol-header construction or outgoing text serialization; cached-header/payload selection in pre-serialized mode |
| `self_install_ms` | Local expert-table copy and construction of its CPU record |
| `communication_prepare_ms` | Frame buffers, receive state, poll descriptor construction, deadline setup |
| `tcp_exchange_ms` | Exchange-loop elapsed time excluding receive processing: socket waits, sends/receives, buffering, envelope checks, poll bookkeeping, and final completeness check |
| `receive_process_ms` | Sum of incoming binary decoding or text parsing, validation, CPU installation, and receive-chunk construction |

The five stages sum to `elapsed_ms` for each sample. For compatibility,
`install_ms` remains the sum of `self_install_ms` and `receive_process_ms`;
do not add it again. `tcp_exchange_ms` is computed as the exchange-loop interval
minus the accumulated receive processing intervals; it includes C++/OS
overhead and waiting for remote CPU work, not just network transfer latency.
Instrumentation overhead is included in the measured stages.

Stage means for the critical-rank records sum to the mean total. Their medians
and percentiles generally do not sum, because each statistic can select different
iterations. Network transfer can progress while another rank processes metadata;
the stage split describes local wall time, not globally exclusive activities.

### Receive-processing breakdown

The console and JSON also report `critical_rank_receive_process_ms`, using the
same slowest-total-time rank in each iteration. Per-rank summaries and individual
records contain `receive_process_breakdown_ms`. These nine substages sum to
`receive_process_ms` for each sample:

| Field | Included work |
|---|---|
| `expert_deserialize_ms` | Binary header decoding, reserved-vector resizing, and array validation; or production text parsing/allocation/validation |
| `expert_validate_ms` | Additional source/destination/topology checks and per-expert index ordering checks |
| `expert_record_install_ms` | Text mode moves vectors into the source map; binary mode already receives into that record, so this boundary has only instrumentation overhead |
| `expert_progress_init_ms` | Text mode allocates head/tail vectors and inserts a record; binary mode zeroes existing counters |
| `x3x4_deserialize_ms` | Binary header decoding, reserved-vector resizing, and array validation; or production text parsing/allocation/validation |
| `x3x4_validate_ms` | Additional source/destination/topology and tensor/chunk-configuration checks |
| `chunk_build_ms` | `compute_chunks_from_token_indices()`: descriptor allocation/filling and copies of per-chunk source-token-index lists |
| `x3x4_record_install_ms` | Stores/replaces chunk descriptors; text mode also moves parsed x3/x4 vectors into a new source record |
| `overhead_ms` | Remaining time inside the parent receive-processing interval, including function entry/exit, temporary cleanup, and timing bookkeeping |

The `*_deserialize_ms` names are retained in both modes. In binary mode they
measure fixed-header decoding, resizing within reserved vector capacity, and validation;
there is no decimal parsing. In text mode, validation and allocation are
interleaved inside the production deserializers and remain timed as a unit. The explicit `*_validate_ms` fields cover only the additional checks after
deserialization. Timers do not change the validation policy or wire format.
Self installation is excluded from this breakdown and remains `self_install_ms`.
Single-node runs have zero x3/x4 and chunk-build times because no such messages
are exchanged. Rebuild the shared library on both nodes before rerunning; report
configuration `timing_schema: 4` identifies this instrumentation version.

The maximum rank-local duration needs no synchronized clocks. It is not an exact
global makespan: barrier-release and OS scheduling skew can influence it. A send
finishing only means the local socket accepted its bytes; the corresponding
receiver's measured interval includes installation. The post-exchange collective
ensures all ranks have finished before another iteration starts.

To exclude sender serialization, run both nodes with:

```bash
bash ./run_node0_torchrun.sh --preserialized
bash ./run_node1_torchrun.sh --preserialized
```

In this mode binary headers or text payloads are precomputed; framing, socket
I/O setup, and all receiver work remain timed. Binary arrays are already in DRAM
in either mode, so `--preserialized` saves only small-header preparation for binary. `serialize_ms` is then only the negligible cached-payload selection cost.
Do not interpret total time minus parsing time as pure network latency: work
overlaps across ranks and socket progress includes C++ and OS overhead.

## Scope of comparisons

This measures a proposed repeated-exchange path. Unlike current initialization,
TCP connections persist and x3/x4 is not separated from expert exchange by RDMA
setup. Expert peers also progress concurrently rather than following the current
sequential outgoing send loop. The production metadata generation/serialization
and chunk-building code is reused, but this standalone transport and CPU record
installation are benchmark implementations, not the full production proxy.

All metadata arrays are native C++ vectors. Self installation copies the local
expert vectors and initializes progress counters. Binary arrays are received into reusable source records and validated there;
text-mode vectors are moved into records after parsing. Binary framing references the original arrays and copies only the small header;
text mode still copies framed payloads and converts integers to/from text.
Reference-only self installation remains a possible future optimization.
GPU routing, D2H/H2D copies, and GPU scheduler initialization remain excluded.
Compare C++ reports with prior Python reports as separate implementations; the
Python selector implementation has been replaced by native `poll()`.

Use `--verify` on both nodes to regenerate expected source-specific metadata in
C++ before timing and check every received expert/x3/x4 table after timing.
This is for correctness checks, not performance runs: it consumes extra memory
and CPU and changes cache state, although verification itself is outside timing.

## Local correctness tests

```bash
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s . -p 'test_benchmark.py' -v
```

Build the library first. Native codec tests check binary/text equivalence,
malformed payload rejection, and partial scatter/gather sends. Subprocess tests
cover both wire formats and empty arrays. Additional tests check expert membership after resolving compact indices, empty expert
lists, malformed metadata rejection, chunk construction, and real subprocess
exchanges through the Python wrapper and C++ library. The small native runs
use `--verify`; an additional test checks native timeout propagation to Python.
Transport smoke tests use four same-node ranks and two logical nodes
with two ranks each, including pre-serialized mode. Both logical nodes use
`127.0.0.1`, with `--node-port-stride=2` giving them distinct port ranges.
The default stride is zero for real separate hosts.
An additional test runs all 16 proxies with 20,480 tokens, 128 experts, and top-8.
They need permission to bind loopback sockets. They do not measure remote-network
performance and do not require torchrun.

For a single-node torchrun smoke test:

```bash
torchrun --standalone --nproc_per_node=4 benchmark.py \
  --hosts=127.0.0.1 --num-tokens=137 --num-experts=16 --top-k=4 \
  --datasets=2 --warmup=1 --iterations=3 --output=/tmp/router_metadata_smoke.json
```

Run `python3 benchmark.py --help` for all benchmark arguments.

`python_reference.py` is an independent metadata oracle for unit tests only;
it is not imported by the benchmark worker.

For a comparison, run both nodes once with the default binary mode and once with
`--wire-format=text`, using different output paths. For example, on node 0:

```bash
RESULT_JSON=results/binary.json bash ./run_node0_torchrun.sh --wire-format=binary
RESULT_JSON=results/text.json bash ./run_node0_torchrun.sh --wire-format=text
```

Run node 1 with the matching mode for each run. Upgrade/rebuild the library on
both nodes: the Python/native C ABI is now version 4. Reports identify the wire
format; timing field names remain stable for comparisons.

For direct binary reception, time inside socket `recv()` calls belongs to
`tcp_exchange_ms`. Header decoding/vector-length preparation and completed-array
validation belong to the corresponding `*_deserialize_ms` field. The latter now
sums work before and after the array transfers; it excludes waiting for bytes.
`configuration.receive_storage` is `direct_reusable_vectors` for binary and
`payload_buffer` for text. The binary wire format is unchanged; the current wrapper requires C ABI
version 4. Rebuild both nodes for comparable timing/schema reports. Native tests check
that receive-array addresses stay stable across changing routing datasets.

### Isolated expert index-bounds scan

In binary mode, `critical_rank_expert_index_bounds_ms` reports the time spent
inside the existing `std::all_of` scan checking `0 <= index < num_tokens` for
received expert indices. It uses the same slowest-total-time rank as the other
critical-rank summaries. `expert_index_bounds_ms` is also present in each rank
summary and per-iteration record.

This is a **nested subcomponent of `expert_deserialize_ms`**, not an additional
stage: do not add it to the receive-stage or total-stage sums. It excludes offset
checks, header decoding, vector resizing, per-expert sorted-order checks, and
error-message handling. No validation is skipped or duplicated. The measurement
contains timer overhead; very short or empty scans may have zero measured duration.

For the mean on the selected critical ranks, divide
`critical_rank_expert_index_bounds_ms.mean` by
`critical_rank_receive_process_ms.expert_deserialize_ms.mean` to estimate the
fraction attributable to this scan. Text mode reports zero for the isolated
field because its checks remain interleaved with parsing in the unchanged
production deserializer; zero there means **not separately measured**.
Rebuild on both nodes. `timing_schema: 4` identifies the new nested timing.

### Disable the two expert-index scans

For binary-mode measurements, append this option on **both nodes**:

```bash
# Node 0
bash ./run_node0_torchrun.sh --disable-expert-index-validation
# Node 1
bash ./run_node1_torchrun.sh --disable-expert-index-validation
```

This skips only the expert index-bounds scan (`0 <= index < num_tokens`) and
per-expert sorted-order scans. Header, identity, topology, byte-length, capacity,
and expert-offset checks remain enabled, as do x3/x4 checks. Validation is on
by default. The option is binary-only: text mode keeps using the unchanged
production deserializer and rejects this option rather than partially applying it.

Console and JSON reports include `expert_index_validation_enabled: false`.
`expert_index_bounds_ms` becomes exactly zero. `expert_deserialize_ms` still
includes header/length handling and offset validation; `expert_validate_ms`
still includes identity/configuration checks, so neither parent stage should
be expected to become zero. If `--verify` is also enabled, exact-content checks
still run outside the timed interval.

Rebuild on both nodes before using the updated wrapper (C ABI version 4):

```bash
cmake --build build -j
```

This option changes only the benchmark. Production `RDMA_CPU_Proxy` validation
and metadata exchange are unchanged.
