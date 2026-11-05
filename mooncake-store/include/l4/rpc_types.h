// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.h"
#include "service_metrics.h"
#include "storage/distributed/dfs_io_timeout.h"

namespace mooncake {

struct DfsMissingBucketReport {
    int64_t bucket_id = -1;
};
YLT_REFL(DfsMissingBucketReport, bucket_id);

struct DfsBucketSnapshotResponse {
    UUID epoch{};
    uint64_t version = 0;
    bool unchanged = false;
    uint32_t heartbeat_interval_seconds = 300;
    uint64_t batch_timeout_fixed_ms = kDefaultDfsBatchTimeoutFixedMs;
    std::vector<int64_t> bucket_ids;
};
YLT_REFL(DfsBucketSnapshotResponse, epoch, version, unchanged,
         heartbeat_interval_seconds, batch_timeout_fixed_ms, bucket_ids);

YLT_REFL(ServiceL4Metrics, read_operations, read_bytes, read_duration_us,
         read_duration_buckets, restore_ranges, restore_bytes);

// Cumulative, per-client service-level I/O metrics. Reports are sent from a
// background thread so they do not add RPCs to foreground data operations or
// the heartbeat path. The master deduplicates reports by client UUID and
// sequence number.
struct ServiceMetricsSnapshot {
    std::string service_endpoint;
    uint64_t sequence = 0;
    uint64_t dual_write_both_success = 0;
    uint64_t dual_write_memory_only = 0;
    uint64_t dual_write_dfs_only = 0;
    uint64_t dual_write_neither_success = 0;
    uint64_t dual_write_pending = 0;
    uint64_t read_failures_memory = 0;
    uint64_t read_failures_dfs = 0;
    ServiceL4Metrics l4;
};
YLT_REFL(ServiceMetricsSnapshot, service_endpoint, sequence,
         dual_write_both_success, dual_write_memory_only, dual_write_dfs_only,
         dual_write_neither_success, dual_write_pending, read_failures_memory,
         read_failures_dfs, l4);

}  // namespace mooncake
