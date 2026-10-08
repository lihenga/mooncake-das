# DFS write diagnostics

Diagnostics are enabled by default on this branch. Set
`MC_STORE_DFS_DIAGNOSTICS=0` before starting a process to disable them. No
additional GPU queries, synchronization, profiling events, or kernels are
introduced by diagnostics. The retained batch completion events are part of the
D2H concurrency optimization, not GPU profiling.

Save both prefill and decode SGLang logs, the master log, and the launch scripts.
Check for `STORE_DIAGNOSTICS_CONFIG version=2` in the deployed client's log.
The startup records identify the endpoint/client and relevant environment
settings; they report configured values, not proof that a transfer used a path.

## Correlation and timing

- `STORE_TRACE_BEGIN` / `STORE_TRACE_END`: correlate by **pid + trace**, not the
  glog thread-id column. `parent` links the Python-facing native wrapper, Client
  BatchPut, local memcpy worker, and background DFS write. `owner` identifies
  the client or worker pool within that process. Key fingerprints identify the
  first/last key without dumping keys or reading payloads.
- `real_batch_put` covers nested-slice construction, the native write, metric
  reporting, and result conversion. It starts after Python argument conversion;
  it does not measure time waiting for Python/SGLang to invoke the API.
- `batch_put` covers configuration, object construction, checksum, allocation
  RPC, external staging, transfer submission, DFS staging, replica wait, D2H
  synchronization, queue submission, finalization RPC, and result collection.
  D2H fields include arena size/budget, pinned/pageable status, copy counts and
  bytes, completion events and fallbacks. Map-lock and device-lock waits are
  separate phases. DMA completion and source-buffer lifetime stay synchronous
  with BatchPut return.
- `dfs_background_write` covers queue wait, DFS I/O, completion RPCs, and arena
  release. A successful BatchPut can precede the background write; it is not
  evidence of completed DFS persistence. Check the background result separately.
- `get_session_start` / `get_session_ranges` cover metadata/session locking,
  plan construction, memory/local-disk/DFS reads, scatter submission and sync,
  and access recording. Existing detailed read logs remain available.
- `local_memcpy` identifies the actual synchronous GPU-copy worker path,
  including queueing, pointer queries, context selection, and copy time. Fast
  tasks are silent; execution >=100 ms or failures produce summaries. Slow
  queue-only summaries are limited to one per worker per second.
- `batch_query` / `batch_exists` normally remain silent; calls >=100 ms produce
  summaries. This avoids flooding logs with normal metadata traffic.
- `STORE_GPU_ERROR` includes the failing HIP/CUDA-compatible API's returned
  status and message, without querying/clearing an extra GPU error state.

Phase durations use a monotonic CPU clock and include scheduling delay. Nested
parent/child times overlap: **do not sum them**. `d2h_sync_total_us` is CPU time
waiting for completion, not DMA duration. `stream_device_wait_us` excludes the
map lock; use the separately recorded map-lock phase as well.

## Calls that do not return

One process-wide watchdog scans active calls every two seconds. A call older
than five seconds generates `STORE_TRACE_STALL`, at most once every five seconds
per call, including its current phase, phase age, total age, and executor thread.
The watchdog neither shares GPU streams nor waits on master RPCs. It can report
stuck RPCs, locks, GPU copies, or background I/O even without an END record.
`STORE_TRACE_HEARTBEAT` reports process-wide active/started/completed counts every
ten seconds, including intervals with no active Store call.

For example: long `master_start` points to allocation/RPC; long
`dfs_stream_map_lock` or `dfs_stream_device_lock` identifies stream acquisition;
long `synchronous_gpu_copy` identifies the memcpy worker; long `dfs_queue`,
`dfs_io`, or `dfs_completion_rpc` distinguishes backend queueing, writing, and
publication. Increasing D-node KV timeouts does not resolve these bottlenecks.

If SGLang forward remains slow while native calls are short and heartbeat
`active=0`, time is outside the instrumented Store APIs. This narrows the search
to SGLang scheduling/compute or indirect GPU contention, but does not by itself
prove which GPU kernel, producer stream, or runtime synchronization is responsible.
The API has no SGLang request-id/bootstrap-room parameter, so those are correlated
through SGLang timestamps rather than invented request identifiers. No claim is
made that one run can resolve an arbitrary issue outside this repository.

Normal writes log at batch boundaries, not per slice/key. Log output and CPU
bookkeeping still have overhead; its impact on PD throughput must be measured in
the actual deployment, not inferred from the lightweight diagnostic tests.
