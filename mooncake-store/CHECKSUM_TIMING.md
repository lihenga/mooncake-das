# Checksum write staging and timing

## Scope and acceptance gates

Branch: `wp/checksum-timing`; diagnostic base:
`8cac42ee4e47e17fb2e65ca2b2d28e292c5c2125`. The initial worktree was clean.

Ordinary BatchPut with checksum enabled and a BUCKET DFS backend computes CRC
from the owned CPU slices after normal DFS staging. It finishes CRC before
queueing DFS I/O, publishing any replica, or returning to the caller. Both the
foreground memory End and background DFS End (including retries) carry the same
optional checksum; zero is a present checksum. No RPC or metadata format changes.

The CRC64-ECMA polynomial, initial value, byte order and slice accumulation order
are unchanged. CRC now covers the bytes produced by the normal D2H, so corruption
introduced by that copy itself is outside this validation boundary.

Single Put, synchronous DFS, BatchUpsert, preferred same-node writes, and reads
keep their previous checksum calculation paths. When staging cannot be reused,
pending operations compute CRC from their original slices before finalization.
Null nonempty slices are rejected before memory transfers. CRC errors become
terminal errors and the existing batch finalizer revokes their allocations.
Staging failure or exceptions release owned buffers and fail the DFS transfer;
no background task is queued. Successful CRC values are never recomputed or
replaced with an absent value during fallback.

Existing replica policy is preserved: BUCKET writes require memory and DFS
allocation/staging success, so a foreground DFS failure revokes both replicas.
Once staging and CRC succeed, a later background DFS I/O failure can revoke only
DFS while retaining the readable memory replica and its checksum.

## Reading the diagnostics

Set only `MOONCAKE_STORE_CHECKSUM=1`. `MC_STORE_CHECKSUM_TIMING` remains enabled by
default; `MC_STORE_CHECKSUM_TIMING=0` disables diagnostics. An active process emits
approximately one cumulative summary every ten seconds, without per-key success
logs or a reporting thread. Counters include failed attempts. Stages with no calls
and no active work are omitted.

| Stage | Meaning after this change |
| --- | --- |
| `write_checksum_reused` | Per-object foreground CRC on owned DFS CPU slices. `crc_us` measures CRC updates; `d2h_us`, `gpu_bytes`, and `staging_us` are zero. |
| `write_checksum` | Original calculation, including checksum-specific D2H when GPU data needs fallback or a write uses an unchanged path. |
| `dfs_stage` | Normal staging per DFS batch. `wall_us` includes allocation, host copies and D2H; `d2h_us` isolates normal GPU copies; `staging_us` covers buffer acquisition and host allocation/copy. `bytes` is logical input bytes, `gpu_bytes` is attempted D2H bytes. The new part counters apply to the asynchronous path. |
| `dfs_submit` | Selecting DFS work and descriptors, excluding staging, CRC, enqueue and I/O. |
| `dfs_enqueue` | Async queue handoff after staging and CRC, including shutdown checks. |
| `dfs_write_io` | Synchronous or background filesystem BatchWrite. |
| `batch_put` | Entire foreground BatchPut, including allocation, transfer, normal DFS staging, CRC and memory finalization; excludes waiting for background DFS I/O. |

All times are cumulative microseconds. Compare two steady-state summaries from
the same PID using `delta(wall_us) / delta(calls)` for average stage latency and
`delta(crc_us) / delta(calls)` for CRC alone. Checksum calls count objects;
`batch_put` and `dfs_stage` calls count batches. Match batch size, object bytes,
replica topology and workload before comparing runs. Do not sum `batch_put` with
its children, or `dfs_stage.wall_us` with its D2H/staging parts. Read verification
stages retain their existing nested timing semantics.

For a workload consisting of optimized GPU BatchPut writes, expect reused CRC
calls/bytes to grow, zero reused D2H, normal D2H bytes in `dfs_stage`, and no growth
of checksum-specific GPU bytes in `write_checksum`. If that last counter grows,
check fallback failures and other write APIs before concluding reuse failed.
Compare new `batch_put` average with the previous 43.2 ms at identical load;
removing the previous 20.8 ms checksum D2H is an expectation, not a measured speedup.
CRC cost should remain comparable after normalizing bytes. Reading/prefetch costs
are not optimized by this change.

## Validation ledger (2026-10-12)

- VERIFIED: Three CRC test bodies compiled with the unchanged production
  `crc_checksum.cpp` using a temporary Zig C++17 compiler and a small assert
  runner for TEST/EXPECT_EQ. Standard vector `123456789`, empty data, streaming,
  nonaligned data, empty/multiple slices and different chunk boundaries agree.
  This is not a full GoogleTest/Store build. Temporary tools are outside the repo.
- VERIFIED: Scoped pre-commit generic checks, codespell and CMake formatting.
  Diff whitespace checks passed. The unavailable C++ hook is explicitly skipped
  when running the remaining hooks; the full hook run is not reported as passed.
- INFERRED: The foreground checksum gate precedes async enqueue and all End
  calls; existing Master End semantics preserve the same value in either order
  and on retry. Terminal CRC errors reach the existing ALL revoke path.
- BLOCKED: Full Store compilation and integration-test execution on this
  Windows host (no WSL Linux installation or Linux Store dependencies). The old
  GCC 5.3 cannot compile the existing constexpr CRC table. C++ formatting hook
  cannot launch `/bin/bash`; invoking the same script with Git Bash also fails
  because clang-format 20.1.8 crashes at startup (exit `0xC0000005`). A clean
  temporary formatter installation reproduces that crash.
- UNTESTED: Real GPU D2H, measured latency/throughput, enqueue failure injection,
  and live inference workloads. No online service was started or stopped.

Added integration coverage counts foreground D2H using a CPU-backed test device,
blocks DFS to check checksum availability at BatchPut return, overwrites caller
buffers, injects staging failure/exception and fallback CRC failure, checks missing
backend, rejected bucket allocation and null-slice handling, checks memory/DFS End
in both orders with retries and a zero checksum, and preserves checksum after
background I/O failure.

Next action on the Linux build host (test servers are local to the test process):

```sh
cmake --build build --target crc_checksum_test dfs_immutable_bucket_client_test
ctest --test-dir build -R '^(crc_checksum_test|dfs_immutable_bucket_client_test|dfs_bucket_write_checksum_test)$' --output-on-failure
pre-commit run --files mooncake-common/tests/crc_checksum_test.cpp mooncake-store/include/checksum_timing.h mooncake-store/include/client_service.h mooncake-store/src/client_service.cpp mooncake-store/tests/CMakeLists.txt mooncake-store/tests/dfs_immutable_bucket_client_test.cpp mooncake-store/CHECKSUM_TIMING.md
```

CTest runs the bucket suite in separate processes with checksum explicitly off
and on. Then deploy manually and compare the cumulative diagnostic deltas above.
