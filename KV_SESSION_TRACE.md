# KV session trace runbook and status

## Scope

Diagnostic-only branches `wp/kv-session-trace`:

- Mooncake base: `081e457c1603853f9580e22db03646d0531dc6ed`.
- SGLang base: `66bc2d4fddd83364a6d0d4d7edcda9d4f5aaa835`.
- Reuse the session evidence from `47f25773`, with default-on logging and bounded
  per-operation samples. Do not change lease, retry, scheduling or cleanup behavior.

## Acceptance and status

- VERIFIED: both branches created from the requested bases; initial trees clean.
- VERIFIED: existing refresh skips missing sessions; Python shared-reference reuse
  does not validate the underlying session. This is a hypothesis for repeated
  failures, not proof of which request path caused the production incident.
- Required: correlate request, rank, worker, load batch, C++ operation and key.
- Required: distinguish query, lock, queue, GPU-event wait and transfer durations;
  record lease deadlines, refresh reasons and cleanup residuals.
- Required: disabled diagnostics issue no extra RPC and preserve existing results.
- VERIFIED: seven CPU-only Python correlation tests passed (nested/native
  contexts, exceptions, worker isolation, disabled path, complete membership,
  hash vectors and unchanged return values).
- BLOCKED: full Mooncake compilation and SGLang integration tests on this host:
  Windows has no Linux build dependencies or PyTorch; SSH to 10.211.11.24
  returned `Connection reset` during key exchange.
- UNTESTED: multi-rank/GPU behavior, emitted native logs and benchmark overhead.
- User requested default-on diagnostics and explicitly stopped further tests
  and checks to conserve tokens. No more validation is being run.
- UNTESTED: the final default-on revision. Earlier Python checks apply only to
  the earlier revision; full native compilation was not completed.
- Next: install both diagnostic branches on the test host and rerun the workload.
  The optional verification commands below are provided for the operator only.

## Run with diagnostics enabled by default

Both repositories use the branch `wp/kv-session-trace`. Install both changes
inside the actual SGLang container; replacing only Python files is insufficient
because the trace bridge is a new native binding. No lease/timeout/worker-count
changes are part of these branches. The original always-on diagnostics commit
is not a prerequisite: its relevant instrumentation is incorporated here.

**No environment-variable or launch-command changes are required.** Diagnostics
are enabled by default in both branches. Use the existing INFO-level Python and
Mooncake log collection and retain all Prefill ranks. The client-side Mooncake
library emits these logs; updating the Master alone is insufficient. Record the
Master configuration and P/D launch commands as in the previous run.

An optional `MOONCAKE_KV_SESSION_TRACE=0` override disables tracing after a
process restart for a separate overhead comparison; it is not needed for this
rerun. Installing these branches on both sides is the normal enablement path.

Check the installed native binding without setting up a connection:

```bash
python - <<'PY'
import inspect
from mooncake.store import MooncakeDistributedStore
from sglang.srt.mem_cache.unified_cache import kv_session_trace
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
assert hasattr(MooncakeDistributedStore, "set_session_trace_context")
assert kv_session_trace.ENABLED
print("trace module:", inspect.getfile(kv_session_trace))
print("linker module:", inspect.getfile(mooncake_direct_linker))
PY
```

First run a small cache-hit request. Require Python `KV_TRACE` records with
`native_bridge:true` and C++ `KV_SESSION_BEGIN schema=2` records whose `trace_id`
matches a Python span. Also require `load.complete` and `operation=end` records.
If the bridge is false or the C++ trace ID is `none` within a traced load, fix the
installation/log collection before spending another full benchmark run. Preserve
installed source/wheel versions; log markers do not prove the entire package's
commit. Run the same workload/configuration with tracing disabled when measuring
the instrumentation's overhead.

## Correlate a request and all shared holders

Python records are JSON after `KV_TRACE`. Each span has `trace_id`,
`parent_trace`, `rid` (or `rids` for a shared load batch), host, PID, rank, thread,
linker identity, wall and monotonic timestamps. A native operation has `op_id`,
`parent_id`, client pointer and the inherited Python `trace_id`. Native pipeline
workers retain that trace and emit `plan_op_id` linking back to the plan.

1. Start with the request's `rid`; follow `parent_trace` to its child spans.
   For batched reads, find `load.dispatch`/`load_layer_wise.begin` and its `rids`.
2. Follow `trace_id` into Mooncake, then `op_id`/`parent_id` into the native calls.
3. `prepare.acquired_refs` emits complete `[key_hash, refcount_after_acquire]`
   membership, in chunks of 128 with `offset` and `total`. Shared keys are also
   shown in `prepare.shared_refs`; that is explanatory, not a second acquisition.
4. Search a failing `key_hash` in other requests' membership records on the
   **same client/process** to find competing or long-lived holders.
   `prefetch.claim` links the synthetic `session_rid` to the real request ID;
   it transfers ownership and does not acquire another reference.
5. Follow `release.before_refs` and `release.remaining_refs`. An end with positive
   Python refcount deliberately retains the shared session. `release.partition`
   shows how many keys were actually passed to native end.

Hashes are FNV-1a/64 over the tagged UTF-8 key, matching both languages. Raw keys,
prompts and KV payloads are not logged. Start capture before the workload so an
earlier acquisition/claim is not missing from reference reconstruction.

## Account for the lease budget

The native lease starts at **query initiation**, not response receipt or Python
session preparation completion. A successful refresh establishes a new deadline;
a failed/skipped refresh does not. `master_batch_query` records the server TTL,
`lease_base=query_start`, deadline and remaining time when the response arrives.

| Interval | Evidence |
| --- | --- |
| Master query/renewal round trip | `master_batch_query`, `master_rpc_us_sum/max`, `remaining_at_response_us` |
| Python session-lock wait | `prepare.lock.wait_ms`, `refresh.lock.wait_ms` |
| C++ session-lock wait | `session_lock_wait_us_sum/max` |
| Prefetch worker queue | `prefetch.dequeue.queue_ms` |
| Preparation and DFS prefetch | `prepare_load`, `prefetch.io`, `operation=dfs_transfer` |
| DFS cache lookup/allocation/read/H2D | `dfs_cache_hit_us`, `dfs_alloc_us`, `dfs_batch_get_us_sum/max`, `dfs_scatter_plan_us`, `dfs_scatter_submit_us`, `dfs_scatter_sync_us` |
| Ready prefetch waiting for admission | `prefetch.finish` to `refresh.decision`, plus `ready_wait_ms` |
| Rank admission agreement | `admission.ready_collective`, `admission.revalidation_collective`, `admission.claim_collective` spans |
| Queued async load | `load.enqueue`, `load.dispatch`, `load.dequeue.queue_ms` |
| Waiting for the existing GPU event | `load.gpu_event_ready.wait_ms` |
| Python layout/plan creation | `read_plan.layouts_ready`, `read_plan.created` |
| Native plan and range metadata | `plan_setup_us_sum/max`, `range_metadata_us_sum/max`, `metadata_build_us_sum/max` |
| Memory transfer | `min_remaining_at_submit_us`, `transfer_us_sum/max` |
| Transfer completed but waiting for session lock | `memory_expired_waiting_completion_lock`, completion `session_lock_wait_us_sum/max` |
| Selected diagnostic output overhead | `begin_log_us_sum/max`, `event_emit_us` (excludes the END line itself) |

Native events are buffered until the operation exits. Use `observed_mono_us` /
`observed_wall_us` for the event time, **not the glog line's flush timestamp**.
`deadline_wall_us` is an estimate derived from monotonic remaining time; use
monotonic intervals for duration calculations. Do not compare monotonic epochs
across hosts or assume Python/C++ clocks share an epoch; correlate their wall
timestamps, subject to clock synchronization.

`age_ms` and `start_to_first_read_ms` start at session insertion, after query;
they exclude the initial query RTT. `since_refresh_ms` starts at the last
successful refresh insertion (or creation when `generation_refresh_ok=0`).
`last_refresh_trace` identifies which refresh succeeded. For the full lease
budget, start from that query's initiation and deadline, then lay out sequential
intervals through the failed read. Spans are inclusive and some I/O overlaps:
do not add parent durations to their children, or sum parallel worker durations.

## Interpret failure and cleanup

- `refresh.decision` records private/shared ownership, known/unknown Python
  lease age, skip threshold and ready wait. `refresh.skipped` means the fresh
  private-session optimization ran, not that Master renewal succeeded.
- `refresh_session_missing` means no query was issued for that key. Other
  reasons distinguish Master error/result-count mismatch, fresh lease already
  expired, expiry before commit, missing/changed session during query, and
  incompatible cached buffer/replica.
- `refresh.result` shows all returned code counts. `load_back.revalidated`
  distinguishes local success from another rank's failure; `load_back.fallback`
  names the next path. Recursive normal-path retries retain the request ID.
- `range_lease_expired`: expired before range admission; the session is erased.
- `memory_transfer_expired_after_io`: the transfer finished after the deadline.
- `memory_expired_waiting_completion_lock`: transfer finished before the
  deadline, but the subsequent session-lock acquisition/check was too late.
- `current_deadline_changed=1`: the current key-indexed session has a different
  deadline than the completed transfer's snapshot. This is evidence to inspect
  concurrent refresh/start; instrumentation does not change the erase behavior.
- `prefetch_lease_expired_before_io` / `prefetch_lease_expired_after_io` distinguish
  the two prefetch expiry checks. `site=file:line` identifies the diagnostic
  branch in the compiled source. `KV_READ_PLAN_FAILURE.site` identifies the
  reporting/throwing branch; it is separate from the producer of `-707`.
- Native `end` audits `requested_residual_sessions`,
  `requested_residual_buffers` and `requested_residual_access_records` while
  holding the same mutex used for deletion. Zero means none of the requested
  keys remained in those maps at that point. Global `sessions_after` may include
  other requests. Detached/shared buffer handles may still be held by in-flight
  operations: map absence is not proof that all memory has already been freed.
- `range_gc_orphan_buffer` identifies subsequent cleanup of a buffer whose
  session was previously removed through another path.

Count queries from `master_batch_query` and existence checks from
`master_batch_exists`: sum `logical_master_calls` on END records only. Do not
also add the parent start/refresh `master_query_calls`, which describes the same
calls. These are logical client calls, not transport retries or shard RPCs.
Shared load batches cannot be counted independently for each member request.
Per-key `starts`, `query_count`, `refresh_attempts`, and `refresh_ok` cover retained
history across generations; `generation_refresh_ok` is generation-local.

## Volume and evidence limits

The native sampler keeps four examples **per reason per operation**, with full
reason counts and `omitted_events`. There is no per-range success dump. Native
history is capped at 65,536 keys per client, with `history_evictions` and
`history_known` exposing lost history. A history miss is not evidence of no
earlier refresh. Python acquisition/release membership is complete so a sampled
failing key can still be traced to all captured holders.

Tracing issues no new existence/query/refresh/read RPC and adds no retry,
collective or GPU synchronization. It still adds CPU, hashing, locking and log
I/O overhead; this has not been benchmarked. Keep full-rank logs, avoid combining
this run with other verbose per-key logging, and inspect the smoke test's volume.
This instrumentation does not establish P-side failure as the cause of every
Decode timeout and does not change either side's failure-handling policy.

## Local validation commands

From the SGLang repository:

```bash
python test/manual/test_kv_session_trace.py
python -m compileall -q python/sglang/srt/mem_cache/unified_cache/kv_session_trace.py \
  python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py \
  python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py \
  python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py
```

On a fully provisioned test host, also run the existing direct-linker ReadPlan
and prefetch capability tests, and compile the Mooncake Store library and Python
binding with the existing project build procedure before installing the wheel.
