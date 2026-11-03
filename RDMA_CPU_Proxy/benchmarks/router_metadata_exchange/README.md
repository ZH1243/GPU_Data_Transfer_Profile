# CPU router metadata exchange benchmark

This standalone Python benchmark models the metadata path in `RDMA_CPU_Proxy`.
It uses CPU memory and TCP sockets only: no CUDA, RDMA verbs, GPU allocations,
token payload transfers, or GPU scheduler initialization. The benchmark itself
uses only Python's standard library (Python 3.9+). PyTorch is needed only for the
`torchrun` launcher; no Torch distributed process group is initialized.

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
| `RESULT_JSON` | `results/metadata_exchange.json` in this directory | Rank-0 report |

Only node 0 writes the report; the same path is overwritten on later runs.
Use a different `RESULT_JSON` for each comparison. Choose unused rendezvous and
metadata ports for concurrent jobs. TCP follows the OS route to these addresses,
not the RDMA device mapping from the main proxy test. For meaningful measurements,
use the intended network interface, an otherwise quiet machine, and consistent
CPU/NUMA placement. This benchmark does not pin processes automatically.

## Protocol and input fidelity

Every process establishes a full-duplex persistent expert-metadata connection
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

Message bodies preserve the C++ decimal-text formats `router_expert_metadata_v1`
and `router_x3_x4_v2`. A benchmark-specific network-order envelope contains two
uint64 values: payload length and iteration number. The iteration number detects
stale messages. These persistent benchmark channels are **not wire-compatible
endpoints for a running production proxy**.

A single nonblocking selector loop progresses sending and receiving on both
metadata paths. They can overlap on the network; Python serialization and
parsing execute serially in each process. Received expert lists are validated
and attached to source-specific CPU records, including zeroed head/tail state.
Received x3/x4 are validated, retained, and used to construct receive-chunk
descriptions with the same fields as `compute_chunks_from_token_indices()`.
No GPU `A_idx` allocation, gather-table publication, or H2D copy occurs.

## Timing and output

Each iteration has this sequence:

1. Barrier on separate control sockets, outside timing.
2. Start a local monotonic timer.
3. Serialize all outgoing metadata (default mode).
4. Copy the self expert table; frame and exchange peer messages.
5. Decode, validate, and install all received CPU metadata.
6. Stop after all local sends and all local installations complete.
7. Gather timing records and release all ranks, outside timing.

Connection setup, CPU routing generation, warmup, result aggregation, logging,
and disposal of the installed records after the iteration are excluded.
Serialization, framing, selector setup, receive buffering, parsing, validation,
self installation, and receive-chunk construction are included. The receive
tables are rebuilt on each iteration rather than updated in place.

The report contains all measured per-rank/per-iteration records and:

- `critical_rank_ms`: statistics of the **maximum rank-local elapsed time** for
  each iteration (minimum, mean, median, nearest-rank p95/p99, maximum).
- `rank_summaries`: per-rank elapsed, serialization, and decode/validation/
  installation statistics. `install_ms` also includes the local self-table copy.
- `sent_bytes` / `received_bytes`: wire bytes, including the 16-byte benchmark
  envelopes but excluding handshakes and control traffic.
- Installed expert and x3/x4 source counts for each rank.

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

In this mode payloads are precomputed; framing and all receiver work remain
timed. `serialize_ms` is then only the negligible cached-payload selection cost.
Do not interpret total time minus parsing time as pure network latency: work
overlaps across ranks and socket progress includes Python and OS overhead.

## Scope of comparisons

This measures a proposed repeated-exchange path. Unlike current initialization,
TCP connections persist and x3/x4 is not separated from expert exchange by RDMA
setup. Expert peers also progress concurrently rather than following the current
sequential outgoing send loop. Python object storage, copying, text conversion,
and validation differ from C++ vectors and streams. Absolute numbers therefore
are not predictions of C++ performance or full per-iteration router overhead.
Use this as a controlled CPU/TCP experiment and confirm optimizations in C++.

## Local correctness tests

```bash
python3 -m unittest discover -s . -p 'test_benchmark.py' -v
```

Tests check expert membership after resolving compact indices, empty expert
lists, malformed metadata rejection, chunk construction, and real subprocess
exchanges. Transport smoke tests use four same-node ranks and two logical nodes
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
