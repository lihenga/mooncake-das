# HIP final L4 H2D benchmark

This standalone benchmark exercises the actual HIP accelerator class used for
Mooncake L4's final host-to-device copy. Both prefetched and on-demand DFS reads
can reach this stage after their host source is ready. It excludes DFS reads,
sessions, RPCs, prefetch scheduling, and SGLang/HiCache transfers.

## Build and run

From the repository root, with HIP and a C++20 compiler:

```bash
H2D_ARCH=gfx936 bash mooncake-store/benchmarks/hip_h2d/run.sh build
H2D_REPEATS=120 H2D_DEVICE=0 bash mooncake-store/benchmarks/hip_h2d/run.sh chain
H2D_DEVICE=1 bash mooncake-store/benchmarks/hip_h2d/run.sh chain
# Isolate the copy kernels using preallocated mapped descriptors:
bash mooncake-store/benchmarks/hip_h2d/run.sh scatter
bash mooncake-store/benchmarks/hip_h2d/run.sh grouped
```

The runner compiles only the real accelerator implementation, registry/runtime
support, and kernel code object. It does not build or install a Mooncake wheel.
`H2D_BUILD_DIR` selects an isolated output directory. `MC_COPY_KERNEL_PATH` is
set by the runner, so the installed Mooncake package is untouched. To compare an
older revision, build **the same bench.cpp** against that revision's accelerator
sources and code object. The baseline need not contain the new grouped symbol.

`check` covers copy correctness and canaries, range/thread-group boundaries,
warm descriptor reuse, multiple pending submissions, concurrent submit/sync,
oversized one-shot descriptor buffers, DMA on special streams, and allocation,
launch, and pre-completion synchronization failures. GNU linker wrappers inject
these failures without changing production behavior. This is a manual HIP
regression executable, not a claim about the full repository's unit tests.

## Measurements (2026-11-21)

Baseline: `0ef6c5f75e7bcdaed30fb448f0744ec670df4540`,
`v0.3.14_dev_l4`. Candidate: the production sources in this PR. The installed
0.3.14 package reported revision `7bef4e`; its two relevant source files had no
diff from the baseline, but the benchmark builds its own accelerator/code object.

- Two Hygon BW devices, `gfx936:sramecc+:xnack-`, 80 CUs, wave64, 8 MiB L2.
- DTK dcc 25.10/clang17; reported HIP 6.3.26113; no fixed GPU clocks.
- Built in the supplied build container, run in the supplied target container.
- GPU0 and GPU1 were tested separately, with host memory and submission bound to
  NUMA0; both devices map to NUMA0. No other GPU workload was observed.
- Five warmups then 120 samples for each shape. On each GPU the chain order was
  baseline, candidate, candidate, baseline. Both confirmation runs are included.
- Synthetic independent scatter ranges: source/destination gaps and destination
  permutation; at least three rotating slots and a working set beyond device L2.
- Payload bytes only; decimal **GB/s = payload bytes / microseconds / 1000**.
  Values are rounded in the tables; CSVs contain the original run summaries.

### Effective bandwidth of the final H2D call

`chain` wall time includes preparing ranges, descriptor allocation/reuse,
submission, completion observation, and descriptor retirement. It is a warmed
microbenchmark, not a production request or model latency measurement. GPU0,
first confirmation run:

| Independent ranges | Baseline us | Candidate us | Baseline GB/s | Candidate GB/s |
| --- | ---: | ---: | ---: | ---: |
| 32 x 656 B | 212.551 | 23.740 | 0.099 | 0.884 |
| 128 x 16 KiB | 294.591 | 67.510 | 7.119 | 31.064 |
| 256 x 41,984 B | 505.242 | 258.980 | 21.273 | 41.501 |
| 1024 x 41,984 B | 1200.384 | 921.033 | 35.815 | 46.678 |
| 256 x 64 KiB | 630.582 | 381.111 | 26.606 | 44.022 |
| 256 x 41,991 B (tail) | 561.042 | 261.951 | 19.160 | 41.037 |
| 256 x 41,984 B (8 B offset) | 587.092 | 281.411 | 18.307 | 38.193 |
| 1 x 16 MiB (DMA) | 396.081 | 391.381 | 42.358 | 42.867 |

The large effective-bandwidth gains include removing repeated pinned descriptor
allocation/mapping/free. They must not be presented as pure kernel speedups.

### Copy-kernel bandwidth

`scatter`/`grouped` use preallocated descriptors and device events around the
same module-launch API; there is no per-batch pinned allocation. This separates
kernel organization gains from descriptor-cache gains. GPU0:

| Independent ranges | Scatter us | Grouped us | Scatter GB/s | Grouped GB/s |
| --- | ---: | ---: | ---: | ---: |
| 128 x 16 KiB | 60.960 | 54.400 | 34.402 | 38.551 |
| 256 x 41,984 B | 242.240 | 227.840 | 44.369 | 47.173 |
| 1024 x 41,984 B | 924.960 | 883.682 | 46.479 | 48.651 |
| 256 x 64 KiB | 367.520 | 349.121 | 45.650 | 48.056 |

`chain` device-event time is **not pure kernel time**: its start event precedes
host descriptor preparation/submission, so baseline event time includes gaps.
Use wall-based `effective_GBs` for chain comparisons and `event_GBs` only for
isolated kernel comparisons. The deliberately forced single-range scatter is
not the actual dispatch; production correctly uses DMA for that shape.

## Validation and limits

Both GPUs passed **76 manual regression checks each**, including the injected
allocation, launch, and synchronization failures. Every benchmark row has
`correct=1`. The changed-file pre-commit checks passed (using a local Windows
Bash entry-point adapter for the repository's unchanged formatting hook).
The same 76 checks also passed with the baseline code object, validating the
optional grouped-symbol fallback.

The grouped strategy is limited to validated gfx936 devices and uniform
16--64 KiB batches with at least 128 ranges and 8 B aligned endpoints. Older
code objects and other shapes/devices retain scatter/DMA behavior. Descriptor
storage is reused only after successful completion; a failed sync retains it.

No full wheel build, full CTest/CI, real DFS/RDMA request replay, multi-GPU
simultaneous bandwidth, or model TTFT/throughput was validated. First-use and
capacity-growth allocations still pay pinned allocation costs. A device fault
also requires the caller to keep payload buffers alive; retaining descriptors
alone cannot repair arbitrary caller error recovery.

Run summaries and check output: [results/gfx936](results/gfx936). Each CSV keeps
P50/P95 and both timing definitions rather than just the best observed result.
