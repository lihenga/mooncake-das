# B300 preparation branch

Branch: `v0.3.14_dev_l4_b300`, based on `v0.3.14_dev_l4` at
`f855c7e7cb26b2654a265b75d97d96bcecd0a445`.

## B300 build and runtime

The target is `GPU-10-130-10-8`: eight NVIDIA B300 GPUs, CUDA 13, and mlx5
RDMA NICs. `/jsyw` is currently an XFS volume; it is not evidence of a shared
DFS/3FS or storage-side RoCE path.

```bash
bash scripts/variant_build.sh --cuda --etcd
```

Use the ordinary RDMA build, without `--shca`, `--hip` or `--hylink`.
SHCA is a NIC ABI, not another name for IB. The preinstalled
`mooncake-transfer-engine-cuda13==0.3.13.post1` is not this L4 branch and lacks
its native DFS session-prefetch and ReadPlan APIs. Build the client, master and
Python extension together before enabling those features in SGLang.

CUDA 13 builds now submit Store H2D batches of at least 32 independent ranges
through `cudaMemcpyBatchAsync`. Smaller batches and older CUDA toolkits retain
the existing per-range copy. This is a native runtime batch, not the Hygon
mapped-host scatter kernel; `h2d_kernel_batches` must not count it as a kernel.
It needs ordinary pinned staging rather than a GPU-readable mapped-host arena.

The existing HIP/SHCA paths remain available, but the B300 branch does not need
DCU runtime compatibility workarounds. CUDA and HIP/HyLink remain mutually
exclusive. Shared session, lease, prefetch ownership and completion semantics
apply to both platforms.

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

## B300 validation and limits

All remote validation files are under
`/jsyw/zhy_spcae/mooncake-b300-audit-20261116`. The isolated validation container
mounts this directory at `/audit`; compiler outputs, temporary files and caches
stay there. No packages are replaced in the user's `sglang-b300` container.

The accelerator implementation was compiled against the actual CUDA SDK and
its regression passed on all eight B300 GPUs. It covers empty/small batches,
32/65 noncontiguous ranges, unaligned addresses, tails and untouched gaps.
Nine isolated H2D microbenchmarks compare runtime batching with the original
per-range implementation: 32/128/512 ranges of 256 B, 4 KiB and 64 KiB. Median
completion latency improved by approximately 2.3--5.1x in that experiment.
This is not a model-throughput or a complete DFS/RDMA performance result.

The Store extension, Transfer Engine extension and Master also compile and
link in the isolated CUDA environment. That verification uses local dependency
prefixes, with etcd and Rust bindings disabled; it is not a wheel-packaging or
etcd-HA validation.

The new native extensions and the SGLang branch pass real POSIX DFS restore
checks on CPU and all eight B300 GPUs: six combinations per device covering
layer/page reads, prefetch and native ReadPlan, plus missing-key rejection.
The checks use synthetic KV/state data and TCP transport. They do not use a
real Kimi model. A separate same-host, separate-process check using `mlx5_1`
and RDMA protocol also restored data correctly into CPU and all eight CUDA
destinations. It is not an inter-node bandwidth or zero-staging proof.

The RDMA validation container exposes `/dev/infiniband` and has unlimited
memlock. Its default security policy denied NUMA queries, so topology fallback
was exercised; allow the required NUMA syscalls before evaluating placement-
sensitive performance. The existing user container was not recreated for this.

Historical DCU/POSIX validation is recorded separately in the earlier stage
artifacts. It does not establish NVIDIA GDR, target RoCE, Kimi L4 restoration,
DCP support, cross-node inference or capacity at scale. The single-host CUDA
checks here do not establish all of those properties either.
