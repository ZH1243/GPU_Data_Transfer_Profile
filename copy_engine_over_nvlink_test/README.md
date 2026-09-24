# Multiprocess NVLink all-to-all copy benchmark

Linux C++17 project for a single node. The `run.sh` shell launcher starts one instance of
`nvlink_all_to_all` per selected GPU. The executable runs exactly one proxy and
never creates other processes. Each proxy owns one CUDA context,
one nonblocking stream, one source allocation, and one receive allocation per
sender. CUDA IPC exposes receive allocations directly to peer processes. No MPI,
NCCL, Python, or custom GPU copy kernels are used.

## Build and run on the remote node

Requires Linux, CMake >= 3.20, GCC/Clang with C++17, CUDA Toolkit >= 12.8 and a
compatible driver. The host C++ compiler links cudart; nvcc compilation is not
needed. GPUs must support UVA, CUDA IPC, and all-pairs peer access, with default
compute mode. Intended for Hopper GPUs with a fully connected NVLink/NVSwitch fabric.

```bash
cd copy_engine_over_nvlink_test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build -j
ctest --test-dir build --output-on-failure
./run.sh \
  --devices 0,1,2,3,4,5,6,7 \
  --sizes 16MB,12MB,8MB,8MB,6MB,4MB,2MB \
  --batches 5 --iterations 100
```

For a small initial correctness run:

```bash
./run.sh --devices 0,1 --sizes 1MiB --batches 3 --iterations 2
```

`--devices` specifies CUDA-visible ordinals in rank order; `CUDA_VISIBLE_DEVICES`
may remap them. All children inherit the same visible GPU list. The list length
sets n (2..32); provide exactly n-1 sizes. Eight GPUs and the illustrated sizes
are defaults; batches default to 5 and iterations to 10. Positive integer sizes
accept bytes or B/KB/MB/GB (decimal) and KiB/MiB/GiB (binary). Thus the example
uses **80 MB** (5 × 16 MB) for the source on each GPU, plus 280 MB total receive
storage (360 MB of payload allocations per GPU). Each GPU still sends 280 MB
per iteration because destinations reuse prefixes of each source window.

Optional `--cpus 0,1,2,3,4,5,6,7` pins one proxy to each listed logical CPU. Choose
available CPUs near their GPUs using your node's topology and job CPU allocation;
reserve a distinct CPU per proxy for tight spin barriers. `--timeout-seconds 120`
sets the timeout of each barrier. A proxy failure aborts the job; the launcher
reaps children and terminates remaining proxies. SIGINT/SIGTERM to the launcher
also terminates them. The script creates a private, unique directory in `/dev/shm`
for a file mapped with `MAP_SHARED` by all proxies. Rank 0 initializes the file
and atomically publishes it before peers attach. The script removes the shared
file and directory after all proxies exit. Concurrent runs use separate directories.
`NVLINK_SHM_DIR` overrides `/dev/shm` (use a local memory-backed filesystem);
`NVLINK_PROXY_BIN` overrides the default `build/nvlink_all_to_all` executable.
SIGKILL to the launcher cannot run its cleanup trap; in that case, terminate its
remaining proxies and remove its `/dev/shm/nvlink-a2a.*` directory manually.
The timeout covers barrier waits, not a driver call hung inside CUDA.

## One executable invocation = one proxy

The script launches commands equivalent to these for two GPUs:

```bash
run_dir=$(mktemp -d /dev/shm/nvlink-a2a.XXXXXXXX)
./build/nvlink_all_to_all --rank 0 --shared-file "$run_dir/shared" \
  --devices 0,1 --sizes 1MiB --batches 3 --iterations 2 &
p0=$!
./build/nvlink_all_to_all --rank 1 --shared-file "$run_dir/shared" \
  --devices 0,1 --sizes 1MiB --batches 3 --iterations 2 &
p1=$!
wait "$p0" "$p1"
rm -f -- "$run_dir/shared" "$run_dir/shared.initial"
rmdir -- "$run_dir"
```

Prefer `run.sh` for automatic failure monitoring and signal cleanup. For manual
launches, use a fresh directory for each run, identical benchmark options and
GPU visibility for all ranks, one unique rank in `[0,n)`, and the same absolute
`--shared-file` path. Rank r controls `devices[r]`. Nonzero ranks may start first;
they wait up to `--timeout-seconds` for rank 0 to publish shared memory. Running
the executable without `--rank` and `--shared-file` reports a usage error.

## Exact copy and synchronization schedule

Let `S[k]` be the size for relative destination k, with k=1..n-1,
`T=sum(S)`, `M=max(S)`, and B be batches per iteration. Rank i sends to `(i+k)%n`.

- Source allocation: `B*M` bytes.
- Source offset in zero-based batch b: `b*M`, identical for every destination.
- Peer k reads the first `S[k]` bytes of the window `[b*M, (b+1)*M)`.
- Destination allocation on rank j dedicated to sender i:
  `B*S[(j-i+n)%n]` bytes.
- Destination offset: `b*S[k]`.
- Sent bytes per iteration: `B*T`, independent of the smaller source allocation.

For the example, batch b sends 16, 12, 8, 8, 6, 4 and 2 MB prefixes of the
same 16 MB window. Batch indices run from 0 to B-1 in every iteration. This
models source reuse across destinations; it does not reproduce RDMA routing masks.

Every iteration starts with a shared-memory barrier. Every batch submits n-1
**separate** `cudaMemcpyBatchAsync` calls, each containing exactly one source,
one destination, and one size, in increasing k order on the same stream.
`cudaStreamSynchronize` follows the last submission. A shared-memory barrier
then waits for every sender's stream to finish, including after the last batch.
The next iteration reuses the allocations and resets offsets to zero.

The barrier uses cache-line-separated, lock-free 64-bit arrival counters with
release stores/acquire loads in coherent shared DRAM. Each process waits until
all counters are **at least** its current epoch, avoiding a deadlock if a fast
peer enters the next barrier before another peer finishes polling. CPU spinning
reduces wake-up overhead but cannot guarantee identical start timestamps under
OS scheduling. There is no CPU or GPU barrier between individual peer copies.

Initialization, IPC setup and verification occur outside the timed loop. Receive
buffers are zeroed; each source window is initialized once with a nonzero pattern
dependent on sender and batch. All destinations receive prefixes of that pattern. By default every received byte is checked after
the final iteration using a bounded host staging buffer. `--no-verify` skips this
check. Iterations deliberately reuse identical source data; final verification
checks routing and offsets, not whether every preceding iteration executed.
All imported IPC mappings close before any owner frees its allocation.

## Copy engines, NVLink and measurement

The code sets `cudaMemcpySrcAccessOrderStream` and
`cudaMemcpyFlagPreferOverlapWithCompute`. NVIDIA documents the latter as a
**hint**, so this API cannot strictly force copy-engine use. The program checks
all-pairs P2P capability and fails if a pair lacks it; P2P capability alone does
not establish that a path uses NVLink rather than PCIe. Check the actual node:

```bash
nvidia-smi topo -m
nsys profile --trace=cuda,osrt --wait=all --force-overwrite=true \
  -o nvlink_all_to_all ./run.sh \
  --devices 0,1,2,3,4,5,6,7 \
  --sizes 16MB,12MB,8MB,8MB,6MB,4MB,2MB --batches 5 --iterations 10
```

Inspect all child processes, their CUDA API calls, and GPU P2P memcpy/engine
activity in Nsight Systems. Confirm NVLink/NVSwitch connectivity in the topology.
Profiling adds overhead; use an unprofiled run for timing.

Output includes commented configuration/verification lines and CSV rows per rank
and iteration. `submit_and_sync_ms` sums host elapsed times from first submission
through stream synchronization for each batch, excluding barrier waits.
`iteration_with_barriers_ms` includes batch barriers and bookkeeping, excluding
the initial iteration barrier. `sent_GBps` is outgoing payload bytes divided by
that rank's iteration wall time, in decimal GB/s; it is not bidirectional or
aggregate fabric bandwidth. No warm-up iterations are hidden, so initial runtime
costs can affect iteration 0. There are no CUDA timing events or hot-loop prints.

API references:
[CUDA 12.8 memory API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY.html),
[CUDA IPC guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interprocess-communication).

## Host-only checks on a Mac or Linux machine without CUDA

```bash
cmake -S . -B build-host -DBUILD_GPU_BENCHMARK=OFF
cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
```

These check size parsing/overflow, asymmetric directed layouts for 2..32 ranks,
10,000 barriers across four actual processes, barrier abort/timeout behavior,
and independently opened shared-memory mappings with delayed rank-0 startup.
They do not validate CUDA IPC, the GPU API calls, engine selection or NVLink traffic.

The shell launcher's rank assignment, argument forwarding, failure handling and
signal cleanup can also be tested without CUDA using Python 3:

```bash
python3 tests/test_launcher.py
```
