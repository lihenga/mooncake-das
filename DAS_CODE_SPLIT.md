# DAS extension boundaries

The current community comparison point is official v0.3.14,
`e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c` (PR #4880). The merge also
includes `v0.3.14_dev` through `b6fb061b`. The earlier split comparison point
was the PR #3732 backport `b01664ce699fbd432e6721b4514ff6afac848451`; use
that point only when comparing the pre-upgrade split.

This branch keeps existing classes and public APIs. Independent DAS definitions
live in separate translation units or support headers. Existing community
functions retain the calls that connect those definitions to the main flow.

## Where to change DAS behavior

| Area | Implementation / support |
| --- | --- |
| DFS synchronous writes, async ownership and write outcome reporting | `mooncake-store/src/l4/client_service_l4.cpp`, `include/l4/client_write_internal.h`, `include/l4/async_dfs_write_context.h` |
| DFS initialization, recovery, bucket eviction and backfill | `mooncake-store/src/l4/master_service_l4.cpp` |
| Session refresh, prefetch, chunked reads and H2D scatter | `mooncake-store/src/l4/real_client_l4.cpp`, `include/l4/session_read_types.h`, `include/l4/dfs_h2d_stream_pool.h` |
| Bucket descriptors, epoch validation, batch I/O and admission | `mooncake-store/src/l4/distributed_storage_backend_l4.cpp`, `include/l4/dfs_descriptor_internal.h` |
| Client RPC extensions | `mooncake-store/src/l4/master_client_l4.cpp`, `include/l4/master_client_rpc_internal.h` |
| Master RPC extensions and epoch stamping | `mooncake-store/src/l4/rpc_service_l4.cpp`, `include/l4/rpc_service_internal.h`, `include/l4/rpc_types.h` |
| Master HTTP handlers and JSON support | `mooncake-store/src/l4/master_admin_service_l4.cpp`, `include/l4/admin_http_internal.h` |
| Metrics and client accounting | `mooncake-store/src/l4/master_metric_manager_l4.cpp`, `src/l4/segment_metrics.cpp`, `include/l4/client_metrics.h` |
| Unified buffer allocation, pinned pool and FileStorage lifecycle | `mooncake-store/src/l4/client_buffer_l4.cpp`, `src/l4/pinned_buffer_pool_l4.cpp`, `src/l4/file_storage_l4.cpp` |
| POSIX / optional 3FS extensions | `mooncake-store/src/l4/posix_fs_adapter_l4.cpp`, `src/l4/hf3fs_adapter_l4.cpp` |
| DFS environment parsing | `mooncake-store/src/l4/file_storage_config.cpp`, bucket parsing in `distributed_storage_backend_l4.cpp` |
| HA service publication | `mooncake-store/src/l4/etcd_service_ready.cpp`, `include/l4/etcd_service_ready.h`, `src/l4/etcd_helper_das.cpp` |
| Accelerator fallbacks and HIP scatter ownership | `mooncake-store/src/l4/device_copy.cpp`, `include/l4/hip_pinned_runtime.h`, `include/l4/hip_scatter_context.h` |
| Common RPC lifetime / retained pools | `mooncake-common/include/das/rpc_drain_guard.h`, `das/shared_rpc_pool.h` |
| DAS defaults and SHCA conversions | `mooncake-common/include/das/defaults.h`, `das/rdma_compat.h` |
| Hygon transport selection and kernel loading | `mooncake-transfer-engine/src/das/transfer_engine_hygon.cpp`, `src/transport/hip_transport/hip_transport_hygon.cpp` |
| Hygon RPC stubs and allocator | `mooncake-transfer-engine/include/das/hip_rpc_compat.h`, `das/hygon_allocator.h` |
| ReadPlan / ReadLayout type bindings | `mooncake-integration/store/das/read_bindings.cpp` |
| Optional Store method bindings | `mooncake-integration/store/das/store_bindings.h` |
| Python wheel/driver compatibility | `mooncake-wheel/mooncake/_variant_compat.py` |
| Store extension source membership | `mooncake-store/src/l4/sources.cmake` |

Paths abbreviated as `include/` or `src/` in the Store rows are relative to
`mooncake-store/`. Independently added modules such as `read_plan.*`,
`read_layout.*`, `immutable_bucket_allocator.*`, `dfs_buffer_usage.h` and
`runtime_log_config.*` remain in their existing locations.

## Deliberate differences in community files

These differences are part of the integration contract and are not duplicate
implementations:

- New member declarations, state, ownership and synchronization fields. Moving
  implementations does not remove the need for a complete class definition.
- Public virtual methods and serialization fields, including compatible DFS
  epochs. Their names, order and visibility must remain consistent across peers.
- Calls at lifecycle boundaries: startup, admission, finalize/revoke, teardown,
  recovery and metadata mutation. Keep locks and detached owners in the original
  scope unless a separate change proves a new ownership contract.
- Conditions inside shared Put/Get/Upsert and metadata operations. Those methods
  serve community replica types as well as DFS; do not copy them into a second
  DAS implementation just to hide the diff.
- Metric constructors and registration, RPC/HTTP endpoint registration and
  source lists. These connect the independent implementations to existing
  targets and APIs.
- Device ABI widths, accelerator selection and HIP compilation boundaries.
  Generated sources still require the appropriate native declarations.
- Small layout/accounting accessors and template member operations that depend
  directly on private class state. Avoid artificial callbacks or class-body
  includes to hide these changes.
- Native-language build, CI, packaging, version, Go, documentation and test
  changes. They remain native-language changes rather than becoming C++ data.
- Copyright and license notices, and intentional historical deletions.

## Rules for future updates

1. Implement new DAS-only functions in the appropriate extension unit. Keep
   member functions in their original class and namespace.
2. Put shared complete private types in an internal header. Keep templates
   visible to all translation units that instantiate them. Do not copy a
   template implementation into both `.cpp` files.
3. Preserve the identity and lifetime of static state. In particular, retained
   RPC pools, read trace IDs and the HIP runtime mutex must not gain per-file
   copies.
4. Add Store sources to the correct shared/master/client set. Keep 3FS sources
   under `USE_3FS`; do not introduce client/GPU dependencies into master-only
   objects. The HIP transport's existing source glob includes its Hygon unit.
5. Register Python read types before methods using them. Preserve GIL scopes,
   ownership guards, Python names and legacy setup overloads.
6. Keep refactors separate from policy changes. A moved function should retain
   its body, error handling, validation and timing of side effects.
7. Review residual diffs against the fixed community point and review behavior
   against the pre-refactor branch. A smaller line count alone is not proof of
   reduced merge risk or equivalent behavior.
8. Keep code already accepted upstream in its community form. For the fixed
   v0.3.14 target, the classic/TENT SHCA conversions and RDMA depth helpers
   stay inline; only additional device-transport adaptations use
   `das/rdma_compat.h`. See `DAS_UPSTREAM_0_3_14.md` before integrating that tag.

## Build and focused validation

Run the SHCA build in a Linux Hygon/SHCA environment:

```bash
GOPROXY=https://goproxy.cn,https://goproxy.io,direct bash scripts/variant_build.sh shca
```

When unit tests are enabled, select the tests covering the changed area:
DFS bucket allocator/backend/client, POSIX and sync reads, client buffers,
FileStorage config, master/client metrics, session bindings, reconnect and
RDMA capability tests. Use focused gtest filters for the large master and
pybind suites when validating only the DFS extension path.

The Python e2e scripts `test_read_layout.py` and `test_store_compatibility.py`
can use an isolated TCP master via `MOONCAKE_MASTER`. Do not install a validation
wheel into a running service's Python environment; use a temporary venv or
isolated target directory.

A successful SHCA build does not validate CUDA, 3FS, other wheel variants,
cross-node RDMA, real DFS deployment or model-serving performance.

## v0.3.14 integration boundaries

- Keep upstream namespace accessors and ObjectState as the only metadata owner.
  DAS backfill state belongs to ObjectState; never restore the old shard tables.
- ReadLayout types/methods are registered only in `store/das/read_bindings.cpp`
  and `store/das/store_bindings.h`. L3's inline registrations must not be added
  again in `store_py.cpp`. Python variant checking similarly has one owner in
  `_variant_compat.py`, called from the canonical `python/mooncake/__init__.py`.
- RPC registrations remain upstream's pooled registrations plus the seven DAS
  endpoints; adding a second registration crashes Master startup.
- `storage_backend_interface.cpp` owns backend-independent shared definitions;
  master-only targets must not link a client/GPU implementation for these.
- Synchronous eviction/invalidation test entry points belong to
  `tests/master_service/master_service_test_peer.*`, outside the public Master API.
- The variant script incorporates L3's multi-protocol/HyLink support while
  retaining `USE_HYGON` for DAS scatter kernels and fake legacy HIP RPC for
  DTK 26.04. New HyLink uses its separate community transport.
- After updating packaging, parse TOML and check source/target membership:
  conflict-free merges can duplicate `[tool.mooncake]`, ReadLayout sources,
  test targets or Python registrations.


## Lifecycle and regression checks for the next update

- Start DFS recovery/eviction/backfill workers once after Master initialization,
  and join them during destruction. Graceful-unmount callbacks only unmount;
  never place worker startup in a callback while resolving constructor hunks.
  The native backfill and grace-period unmount tests cover this boundary.
- `Client::WriteDfsReplicas` now lives with the DFS-specific write pipeline in
  `src/l4/client_service_l4.cpp`. Community Put/Upsert keep their call sites.
  Per-key DFS outcome counters count successful bytes and failed keys even when
  a successful MEMORY replica makes the public write return success.
- Asynchronous write counters settle exactly once through the context's
  completion gate. A deadline reporter and a late worker cannot count the same
  write twice; they retain buffer ownership until physical I/O completes.
- Rebuild every tested executable against the current shared library before a
  wide run. A stale Master test executable may still reference removed public
  testing wrappers even though the current TestPeer API builds successfully.
- Use explicit native test contracts: aligned shard reservations have no extra
  padding page; DFS-only writes are rejected before publication; MEMORY stays
  readable when DFS I/O fails; batch lease errors stay batch-stage errors;
  default Master pools fail connection attempts promptly, with opt-in retries.

- DFS backfill selection uses `include/l4/dfs_backfill_index.h`, synchronized
  at namespace write-release. It snapshots at most 256 candidate publications
  per maintenance tick and never scans healthy objects. Keep the generation
  when resolving a snapshot so a deleted/recreated key cannot inherit old work.

- Resolve `FileStorage`'s backend before initializing the unified DFS arena:
  filesystem DFS can be inferred from the distributed backend even when the
  caller did not pre-populate `enable_dfs`. Arena initialization belongs after
  that inference and before client staging/prefetch receives its allocator.
- Reconcile the environment-to-Master snapshot fields together with their
  client heartbeat consumers. Keeping setters without constructor parsing can
  silently replace configured heartbeat/timeout budgets with defaults.
- Keep DFS hit accounting beside the community first-readable-replica choice.
  A DFS descriptor later in a memory-first list is not a DFS hit.
- Both Put and Upsert pass DFS admission/fallback state through every physical
  allocation path. Reuse the same fallback error policy; never disable a key's
  validity/liveness checks to accommodate an obsolete fixture.

- A process-wide RPC pool can retain a closed socket when a server restarts at
  the same address. `MasterClient::Connect` retries only its read-only
  `ServiceReady` transport failure once; timeouts and business RPCs are never
  replayed by this bootstrap recovery.
