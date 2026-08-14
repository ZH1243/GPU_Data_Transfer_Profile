# CPU notification of a persistent Hopper kernel

This project launches exactly as many persistent CTAs as the selected GPU has
SMs. Each CTA keeps its own `j` in shared memory, initialized to its CTA index.
CTA `i` repeatedly polls one 32-bit value in device global memory, advances
`j += S` when the observed value is greater than `j`, and writes the first
`j > w` to `A[i]` before exiting.

The host owns a pinned 4-byte mirror initialized to zero. Every `t`
microseconds it adds `b` to that mirror, then publishes the new immediate value
to the device allocation using the CUDA Driver API's
`cuStreamWriteValue32`. (`cuStreamWriteValue32` accepts an immediate value, not
a host source pointer, so the pinned allocation is the requested CPU-side
state/mirror rather than a DMA source.) The result array is copied into a
second pinned host allocation and printed after the kernel returns.

## Build

CUDA 12.x, CMake 3.24 or newer, and a Hopper-class GPU are expected.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The default CUDA architecture is `90`. If desired, select architecture `90a`
explicitly when using a toolkit that supports it:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90a
```

## Run

```bash
./build/bin/cpu_notify_gpu_persistent_kernel \
  --period-us 1000 --increment 3 --threshold 1000
```

Short parameter names matching the problem statement are also accepted:

```bash
./build/bin/cpu_notify_gpu_persistent_kernel --t 1000 --b 3 --w 1000
```

Other options are `--device N` and `--threads-per-cta N`; run with `--help` for
the complete list. `S` is always detected from the selected GPU because the
program must launch one CTA per physical SM.

## Exact-threshold edge case

The stated rules contain one deadlocking parameter combination. If repeated
additions of `b` make the CPU value land exactly on `w`, the CPU must stop, but
the CTA whose sequence reaches `j == w` needs to observe a value strictly
greater than `j` before it can advance and terminate with `j > w`. The program
therefore rejects configurations where `ceil(w / b) * b == w`. Choose a `b`
that overshoots `w` (the defaults finish at 1002), or change one of the protocol
comparisons if exact landing must be supported.

## Residency

The kernel only needs four shared bytes for `j`, but the host opts into and
launches with enough dynamic shared memory for CUDA's occupancy calculation to
permit only one active CTA per SM. With `S` CTAs total, all `S` CTAs can be
resident concurrently and no SM can host two of them.
