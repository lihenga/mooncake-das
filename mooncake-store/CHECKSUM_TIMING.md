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

## Decode timeout diagnostics (2026-10-12 follow-up)

The two supplied prefill logs and their matching decode logs were inspected
read-only on 10.211.11.24. No inference service was started or stopped.

| Run directory suffix | Unique D timeout rooms | Matching P records | P records with forward time | Queue + forward over 300 s |
| --- | ---: | ---: | ---: | ---: |
| `20261012_161532_204-pd-multi-turn-qa_100` | 122 | 122 | 41 | 36 of 41 |
| `20261012_140157_204-pd-multi-turn-qa_100` | 126 | 126 | 43 | 39 of 43 |

D reports `timed out after 300.0s in KVPoll.WaitingForInput`. Correlation uses
`bootstrap_room`, since P and D request IDs differ. In the first run, room
5492594264339385507 timed out on D at 16:27:42; its P completion at 16:27:50
reports queue 277736.10 ms and forward 29284.29 ms. In the second run, room
5742558300886468828 timed out on D at 14:14:37; P completes at 14:14:45 with
queue 278299.68 ms and forward 29064.52 ms. Both examples have cached tokens 0.
This verifies that queueing consumed most of these requests' deadline; it does
not establish why the queue grew or that checksum caused the timeout.

P CQ warnings have `outstanding_age_ms=0` throughout both supplied logs. Their
large `poll_gap_ms` alone does not establish a stuck RDMA operation. Do not use
those warnings to conclude that the Decode transfer itself blocked for 300 s.

Additional stages use the existing default-on `MC_STORE_CHECKSUM_TIMING` switch:

| Stage | Scope |
| --- | --- |
| `batch_query` | Master replica-list RPC and result construction for a batch. |
| `session_prefetch` | Entire session prefetch batch, including planning, query, DFS read and final readiness checks. `lock_wait_us` covers its initial and final session mutex acquisition only. |
| `session_range_get` | Entire foreground session range-read batch. |
| `session_memory_get` | Foreground memory replica reads and their completion wait. |
| `dfs_prefetch_batch_get` / `dfs_foreground_batch_get` | DFS Client::BatchGet, including I/O and checksum verification, excluding subsequent result processing and GPU scatter. `bytes` is allocated aligned staging arena size. |
| `dfs_scatter` | Foreground GPU scatter submission and synchronization, excluding scatter planning. |

Existing `prefetch_checksum` / `foreground_checksum` measure each object's
verification, and `dfs_prefetch_read_io` / `dfs_foreground_read_io` isolate
filesystem read calls. Compare the new batch stages with these children to
separate whole-batch delay from per-object CRC. A per-object checksum average
cannot be interpreted as the latency of a complete prefetch batch.

Summaries remain cumulative, approximately every 10 s when instrumented calls
finish. New fields are `objects` (completed scopes' input object count),
`lock_wait_us`, `slow_calls` (wall time >=100 ms) and `over_1s_calls` (>=1 s).
`objects` includes duplicate input keys in the outer session scopes. For each
PID and stage, subtract consecutive summaries before comparing averages or
slow-call rates. Parent stages include children: never sum them together.
`wall_us - cpu_us` includes I/O, locks and thread descheduling; it is not a
measurement of I/O alone. Early failures are also timed.

Completed batch scopes taking >=500 ms additionally emit `checksum_slow` with
stage, objects, bytes, wall/CPU time and available CRC/D2H/lock parts. A per-stage,
per-process atomic deadline limits these warnings to one every 10 s. No object
keys are logged, and per-object CRC never emits slow warnings. The fast path
adds batch timing and aggregate counters, without per-key clocks, polling,
extra transfers, RPCs or a reporter thread. Existing CRC timing clocks remain.
Warnings describe completed operations; an entirely stuck process cannot
produce a new summary without another instrumented operation finishing.

On the next run, correlate D timeout `bootstrap_room` with P `ReqTimeStats`
queue/forward times first. Near that timestamp, inspect `checksum_slow` and
summary deltas from each P PID: query delays point toward metadata access;
prefetch delays with high checksum CPU time point toward CRC; DFS I/O or scatter
delays identify other storage stages. Large `session_prefetch.lock_wait_us`
identifies contention at the two measured locks. Mooncake logs do not measure
SGLang scheduler queueing or cross-rank readiness waits directly. Compare
checksum off/on runs at identical load to test the checksum hypothesis.

Follow-up validation:

- VERIFIED: The actual diagnostic header and production `environ.cpp` compiled
  with Linux g++ C++17 and GoogleTest/glog in a temporary directory. Two enabled
  tests and one separate disabled-process test passed: idempotent completion,
  active/byte/object counters, slow thresholds, lock wait accounting, per-stage
  warning rate limiting/expiry, no per-key warning and the disable switch.
- VERIFIED: The repository's changed-line C++ formatter script passed using
  Linux clang-format 20.1.8 on the touched sources. Scoped pre-commit generic,
  codespell and CMake checks passed on Windows with its unavailable C++ hook
  explicitly skipped; the equivalent C++ script was checked on Linux.
- INFERRED: Added RAII scopes leave storage, checksum and readiness behavior
  unchanged. Diagnostics have bounded batch overhead and rate-limited output.
- BLOCKED: Full Store build/integration execution on the Windows workspace;
  the remote check compiled only the independent diagnostics test and helpers.
- UNTESTED: Live workload overhead, full GPU/DFS integration and a causal
  explanation of queue growth. Deployment and inference retesting remain manual.

Linux build commands for the registered tests:

```sh
cmake --build build --target checksum_timing_test
ctest --test-dir build -R '^checksum_timing(_disabled)?_test$' --output-on-failure
```
