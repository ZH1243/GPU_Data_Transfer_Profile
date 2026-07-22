# NVLink iteration-done notification benchmark

This CPU-only project isolates the following `RDMA_CPU_Proxy` sequence:

```cpp
enqueue_forward_iteration_done_notifications(iteration);
wait_for_nvlink_forward_computation_notifications(iteration);
```

It intentionally contains no RDMA, CUDA, or GPU work.

## Preserved execution structure

Each simulated GPU is a separate process, as in the original deployment. Each
process contains:

- the main thread that enqueues iteration-done records and waits for records
  from every other proxy;
- a persistent forwarding thread;
- a persistent forwarding-ready thread;
- one notification dispatch thread that consumes the mutex-protected local
  deque and publishes records;
- one notification receiver thread that polls the process's inbound queues.

The forwarding and forwarding-ready threads are deliberately idle busy-pollers.
They preserve the relevant CPU scheduling pressure without doing RDMA or GPU
work. Use `--aux-threads=false` only to measure the notification machinery
without those two threads.

Every destination owns a POSIX shared-memory segment with one SPSC ring per
other source process. Notification records, queue layout, FIFO dispatch,
release/acquire publication, generation counters, shutdown ordering, and
`cpu_relax()` polling follow the original proxy design.

A process-shared barrier immediately before the timed region aligns the main
threads. Consequently, the reported number measures notification machinery and
CPU scheduling rather than forwarding-completion skew between proxies.

## Build and run

```bash
cmake -S NVLink_Iteration_Done_Benchmark \
      -B NVLink_Iteration_Done_Benchmark/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build NVLink_Iteration_Done_Benchmark/build -j

NVLink_Iteration_Done_Benchmark/build/iteration_done_benchmark \
  --proxies=8 \
  --warmup=2000 \
  --iterations=100000 \
  --queue-depth=1024
```

Options:

```text
--proxies=N             Simulated local GPU proxy processes (default: 8)
--warmup=N              Untimed warm-up iterations (default: 2000)
--iterations=N          Measured iterations (default: 100000)
--queue-depth=N         Capacity of every source/destination SPSC ring (default: 1024)
--timeout-ms=N          Per-iteration timeout (default: 10000)
--aux-threads=BOOL      Keep forwarding and forwarding-ready busy threads (default: true)
--cpu-affinity=LIST     Linux CPU mask inherited by every child thread, e.g. 0-31,64-95
--proxy-cpus=LIST       Assign one distinct Linux CPU per proxy, e.g. 0,2,4,6
--proxy-cpu-sets=SETS   Assign one distinct CPU set per proxy, e.g. '0-3;4-7'
--help                  Show command-line help
```

`--cpu-affinity` deliberately binds the whole proxy process before its threads
are created, matching `RDMA_CPU_Proxy` rather than pinning individual threads.
Leave it as `none` to use the launcher environment's inherited affinity.

To bind different proxies to different CPUs, provide exactly one CPU ID per
proxy:

```bash
NVLink_Iteration_Done_Benchmark/build/iteration_done_benchmark \
  --proxies=4 \
  --proxy-cpus=0,2,4,6 \
  --warmup=2000 \
  --iterations=100000
```

Here proxy 0 uses CPU 0, proxy 1 uses CPU 2, and so on. Every thread inside a
proxy inherits that proxy's single-CPU mask. Since the default model has five
active threads per proxy, this intentionally creates intra-proxy CPU
contention. Use `--aux-threads=false` if the experiment should include only the
main, dispatch, and receiver threads. `--proxy-cpus` and `--cpu-affinity` are
mutually exclusive.

To give each proxy multiple CPUs, use a quoted semicolon-separated list of CPU
sets:

```bash
NVLink_Iteration_Done_Benchmark/build/iteration_done_benchmark \
  --proxies=4 \
  --proxy-cpu-sets='0-4;5-9;10-14;15-19' \
  --warmup=2000 \
  --iterations=100000
```

This maps proxy 0 to CPUs 0-4, proxy 1 to CPUs 5-9, and so on. Each proxy must
have exactly one non-empty CPU set, and sets may not overlap. The shell quotes
are required because an unquoted semicolon terminates a shell command. All
threads belonging to a proxy inherit its assigned set and may migrate only
within that set. `--proxy-cpu-sets` is mutually exclusive with `--proxy-cpus`
and `--cpu-affinity`.

The output reports aggregate and per-proxy distributions for:

- `total`: both selected calls together;
- `enqueue`: construction plus insertion of all done records into the local
  dispatch deque;
- `wait`: waiting for done-generation records from every other proxy.

For low-noise measurements, use a Release build, disable hot-path logging, run
on an otherwise idle machine, and select an affinity mask with enough physical
cores for all busy threads. With auxiliary threads enabled, there are five
active threads per proxy process.

## Scope

This benchmark does not model normal data notifications or computation-task
queue backpressure. In the full proxy, normal records ordered ahead of the done
record can add receiver-side latency. The benchmark is specifically the
best-case iteration-done control path requested here.
