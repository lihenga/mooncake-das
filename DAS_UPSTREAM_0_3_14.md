# Preparing the v0.3.14 integration

The fixed official target is
[`v0.3.14`](https://github.com/kvcache-ai/Mooncake/tree/v0.3.14), commit
`e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c`. This document records preparation
for a later integration. The current DAS branch has not merged that tag.
Do not substitute a moving community `main` when reproducing this comparison.

## Preparations in this branch

- Keep classic/TENT SHCA LID conversions and RDMA depth helpers in their
  community form. Those implementations are present in the target already;
  replacing them with DAS calls creates unnecessary merge conflicts. The
  additional IBGDA device adaptations still use `das/rdma_compat.h`.
- Maintain Store extension sources in `mooncake-store/src/l4/sources.cmake`,
  included before source lists are consumed. Preserve the shared/master/client
  division and `USE_3FS` condition. Add binding sources through `target_sources`
  separately from the community `pybind11_add_module` source list.
- Keep independent implementations at the boundaries in `DAS_CODE_SPLIT.md`.
  A source move does not protect the moved body from future community changes.

## What the merge preview establishes

Using Git's default `merge-tree --write-tree` against the fixed tag, the
pre-refactor branch `d0703f4cec0dbead1f324337ab41e1ca80643ed7` produces 98
conflicting paths and 410 conflict blocks. After the split and these
preparations, the preview produces 97 conflicting paths and 370 conflict
blocks. Counts describe textual conflict markers, including build, test and
documentation files; they do not measure semantic compatibility.

The preview only writes Git objects. It does not merge into the branch or
alter its working tree/index. A tree containing conflict markers is not a
usable integration result. In particular, the automatically merged extension
files still need an interface review against the new community classes.

## Migration points that need deliberate resolution

| Area | Community change and required DAS action |
| --- | --- |
| Master metadata | The target replaces direct `metadata_shards_` access with tenant namespaces, `MetadataAccessorRW`/`MetadataAccessorRO` and namespace cursors. Port `RunBucketDfsEvictionInternal`, `RunShardDfsEviction` and related invalidation/recovery paths in `src/l4/master_service_l4.cpp` to those contracts. Do not restore the old shard tables beside the new namespace store. |
| Eviction and replica ownership | Account for `ObjectPart`, per-object in-flight work, the Evictor and tenant quota policy. Preserve the new route/accessor lock ordering while carrying over DFS bucket reservations, invalidations, epochs and backfill ownership. Review the shared Put/Get/Finalize hooks together with the moved DFS definitions. |
| Restore and readiness | The target adds a restore lifecycle and worker admission. Connect DAS service-ready publication and recovered DFS replicas to that lifecycle; verify readiness is published after usable state and that failed/partial restore does not admit work early. Review graceful drain and leadership release as one lifecycle. |
| DummyClient RPC template | The target adds `RpcDrainGuard::ScopedCall` to `invoke_rpc` and `invoke_batch_rpc` and changes destructor draining. `invoke_rpc` now lives in `include/l4/dummy_client_rpc_internal.h`; transfer the new guard into that definition, keep one template body, and retain the target's matching state/destructor changes. Choosing only the local deletion in `dummy_client.cpp` would lose the upstream guard. |
| HTTP shared helpers | Community updates to `master_admin_service.cpp` include helpers/types moved into `include/l4/admin_http_internal.h`. Reconcile those changes in the shared header and retain new endpoints. Do not reintroduce a second anonymous-namespace copy beside `admin_detail` or drop new reflected fields. |
| RPC and serialization | Compare changed method signatures, tenant/key identity, replica descriptors, reflection fields and version negotiation before resolving the Store headers. Carry the DAS epoch/session fields into the new definitions once. A successful textual merge does not establish compatibility between old/new clients and masters. |
| Python and wheel layout | The target uses `PYBIND11_MODULE(_store, m)` and changes the package layout. Preserve the target's module/output names and facade, then call `RegisterReadTypes` before `RegisterStoreExtensions`. Retain capability reporting, GIL scopes, legacy setup overloads and the variant check. Test an installed wheel in a fresh environment. |
| Build source lists | The target removes/renames Store files and changes packaging/dependencies. Start from its source lists, retain the single DAS source include, and update shared internal header paths. Check master-only and client targets as well as the full wheel; do not compile obsolete community units merely to satisfy old DAS includes. |

Store paths in this table are relative to `mooncake-store/`. The two
conflicting paths introduced by moving shared definitions, `dummy_client.cpp`
and `master_admin_service.cpp`, are intentional review points: the upstream
changes must follow the definitions to their new headers.

## Suggested integration order

1. Keep the validated DAS split as its own commit. On a dedicated integration
   branch, merge the fixed official tag so Git records the actual community
   parent. Do not use blanket `ours`/`theirs` resolution.
2. Resolve the community metadata/namespace/object contracts, then adapt the
   DAS Master implementation to them. Resolve client/RPC/serialization and
   restore/readiness against the same contracts.
3. Reconcile the moved templates/helpers and Python layout, then the source
   lists, dependencies, SHCA build and wheel packaging. Review newly added
   community methods even where Git reported no conflict.
4. Build the SHCA variant and rerun the focused comparisons below before
   treating the integration as usable. Retain failures and exclusions in the
   validation record instead of weakening assertions or changing policies to
   obtain a green run.

## Validation needed for the actual integration

- Master tenant isolation, namespace traversal, bucket eviction/invalidation,
  recovery, backfill and restore admission/readiness.
- Async DFS buffer ownership and teardown, chunk limits, partial failures,
  End/Revoke ordering, checksums and stale master epochs.
- Session/ReadPlan/ReadLayout reads, prefetch consumption and metric semantics.
- DummyClient in-flight RPC drain and shutdown, external buffers and reconnect.
- Full SHCA build plus installed-wheel API/compatibility tests. Run hardware,
  DFS deployment and other variants separately when those environments exist.

The split's paired baseline has existing failures in POSIX admission timing,
sync-client revocation expectations and bucket allocator/backend recovery.
Record the exact failed cases before integration so new regressions can be
distinguished from those failures. A clean preview or build alone cannot
validate this migration.
