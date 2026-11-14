# B300 preparation branch

Branch: `v0.3.14_dev_l4_b300`, based on `v0.3.14_dev_l4` at
`925c31eefef6e09c06c83be05afbdb485767c9dd`.

## Build

The existing DCU command remains supported:

```bash
GOPROXY=https://goproxy.cn,https://goproxy.io,direct \
bash scripts/variant_build.sh --shca --hip --etcd
```

The script detects a missing optional `hip_hsa_ext.h` and enables the existing
fake fabric-RPC implementation. HIP IPC, SHCA and the Store H2D scatter kernel
remain enabled; unsupported HIP fabric RPC is not tested or claimed available.
If the SDK provides that header, the real implementation remains selected.

For a future NVIDIA build environment:

```bash
bash scripts/variant_build.sh --cuda --etcd
```

CUDA is mutually exclusive with HIP/HyLink. GPU transports enable
multi-protocol routing. SHCA describes a NIC ABI, not generic IB support; add
`--shca` only for an actual SHCA NIC. CUDA configuration has been dry-run
checked here, not compiled or executed on NVIDIA hardware. The existing CUDA
memcpy/stream/pinned-host fallback is unchanged; the Hygon Store scatter kernel
has not been rewritten as an untested CUDA kernel.

## Bounded bucket descriptor cache

`MOONCAKE_DFS_BUCKET_FD_CACHE_LIMIT` bounds the combined regular/direct bucket
lookup maps (default 1024 handles per backend). The cache uses capacity eviction,
not LRU. In-flight readers keep their own shared handle, so eviction does not
close a handle still used by a read. Temporary/in-flight descriptors and other
process FDs are outside this lookup-map limit.

This removes unbounded growth with the number of previously accessed live
buckets. It does not implement distributed Master metadata, incremental bucket
snapshots, a 10PB namespace, or a 10PB recovery guarantee.

## Ranged write sessions

RealClient's memory-only ranged put path now checks the effective replica config
after environment policy is attached. A forced DFS replica is rejected before
reservation, since that API cannot write/finalize DFS. Normal full-object writes
and the size-only internal API's default semantics are preserved. Rebuild all
components together when using this branch's C++ interface change.

## Recovery contract

`MOONCAKE_DFS_BUCKET_RECOVERY_ENABLED=false` deletes scanned legacy bucket files;
it does not merely ignore them. With recovery enabled, sealed bucket metadata
can restore objects, while active unsealed tails are still disposable cache
state. This is not zero-loss database persistence. Use isolated directories and
make the application's accepted cache-loss/rebuild behavior explicit.

Tests corrected in this branch explicitly enable recovery, preserve tenant
identity in copied read views, wait for active read leases before Upsert, and
retry a transient pooled-socket EOF only during fixture initialization. These
changes do not relax production lease or tenant checks.

## Validation limits

The stage includes selected C++ suites, real POSIX DFS recovery into CPU and all
eight DCUs, shared-key/session cancellation tests, real Qwen3 cache restoration,
single-host separate-process SHCA/IB transfers, and 10K/100K-object scale probes.
Detailed commands and evidence live in
`/stortest/zhy_spcae/fixbug/b300-stage-20261114/evidence`.

No NVIDIA execution, target RoCE, two-node inference, real Kimi model or 10PB
capacity/performance validation was performed. Build/import success does not
establish any of those properties.
