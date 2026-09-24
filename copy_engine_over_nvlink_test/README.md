# Multiprocess NVLink all-to-all copy benchmark

Linux C++17 project for a single node. One launcher forks one proxy process per
selected GPU **before any CUDA initialization**. Each proxy owns one CUDA context,
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
./build/nvlink_all_to_all \
  --devices 0,1,2,3,4,5,6,7 \
  --sizes 16MB,12MB,8MB,8MB,6MB,4MB,2MB \
  --batches 5 --iterations 100
```

For a small initial correctness run:

```bash
./build/nvlink_all_to_all --devices 0,1 --sizes 1MiB --batches 3 --iterations 2
```

`--devices` specifies CUDA-visible ordinals in rank order; `CUDA_VISIBLE_DEVICES`
may remap them. All children inherit the same visible GPU list. The list length
sets n (2..32); provide exactly n-1 sizes. Eight GPUs and the illustrated sizes
are defaults; batches default to 5 and iterations to 10. Positive integer sizes
accept bytes or B/KB/MB/GB (decimal) and KiB/MiB/GiB (binary). Thus the example
uses **280 MB**, not 330 MB, for the source on each GPU, plus 280 MB total receive
storage (560 MB of payload allocations per GPU).

Optional `--cpus 0,1,2,3,4,5,6,7` pins one proxy to each listed logical CPU. Choose
available CPUs near their GPUs using your node's topology and job CPU allocation;
reserve a distinct CPU per proxy for tight spin barriers. `--timeout-seconds 120`
sets the timeout of each barrier. A proxy failure aborts the job; the launcher
reaps children and terminates remaining proxies. SIGINT/SIGTERM to the launcher
also terminates them. Shared memory is anonymous and leaves no named SHM files.
The timeout covers barrier waits, not a driver call hung inside CUDA.

## Exact copy and synchronization schedule

Let `S[k]` be the size for relative destination k, with k=1..n-1,
`T=sum(S)`, and B be batches per iteration. Rank i sends to `(i+k)%n`.

- Source allocation: B*T bytes.
- Source offset in zero-based batch b: `b*T + sum(S[1..k-1])`.
- Destination allocation on rank j dedicated to sender i:
  `B*S[(j-i+n)%n]` bytes.
- Destination offset: `b*S[k]`.

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
buffers are zeroed; source segments get a nonzero pattern dependent on sender,
batch and relative destination. By default every received byte is checked after
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
  -o nvlink_all_to_all ./build/nvlink_all_to_all \
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
10,000 barriers across four actual processes, and barrier abort/timeout behavior.
They do not validate CUDA IPC, the GPU API calls, engine selection or NVLink traffic.
