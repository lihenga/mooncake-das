// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>
#include "hybrid_metric.h"
#include "service_metrics.h"

namespace mooncake {

const std::vector<double> kStorageLatencySecondsBucket = {
    kL4ReadLatencySecondsBuckets.begin(), kL4ReadLatencySecondsBuckets.end()};

struct DirectStorageMetric {
    std::array<std::string, 2> access_labels = {"source", "result"};
    std::array<std::string, 1> cache_labels = {"result"};
    std::array<std::string, 1> eviction_labels = {"reason"};
    std::array<std::string, 3> io_labels = {"operation", "source", "result"};
    std::array<std::string, 2> l4_read_labels = {"mode", "result"};
    std::array<std::string, 1> l4_restore_labels = {"mode"};
    explicit DirectStorageMetric(std::map<std::string, std::string> labels = {})
        : access_total("mooncake_direct_access_total",
                       "Direct logical object accesses by selected source",
                       labels, access_labels),
          access_bytes_total(
              "mooncake_direct_access_bytes_total",
              "Direct logical object access payload bytes by selected source",
              labels, access_labels),
          session_cache_access_total(
              "mooncake_direct_session_cache_access_total",
              "Get-session object cache lookups", labels, cache_labels),
          session_cache_evictions_total(
              "mooncake_direct_session_cache_evictions_total",
              "Get-session object cache evictions", labels, eviction_labels),
          io_operations_total("mooncake_direct_io_operations_total",
                              "Direct storage I/O operations", labels,
                              io_labels),
          io_bytes_total("mooncake_direct_io_bytes_total",
                         "Direct storage I/O payload bytes", labels, io_labels),
          io_duration_seconds(
              "mooncake_direct_io_duration_seconds",
              "Direct storage I/O wall-clock duration in seconds",
              kStorageLatencySecondsBucket, labels, io_labels),
          prefetched_tokens_total("mooncake_prefetched_tokens_total",
                                  "Total prompt tokens successfully prefetched "
                                  "by the direct linker",
                                  labels),
          l4_read_operations_total(
              "mooncake_l4_read_operations_total",
              "DFS BatchGet calls by prefetch or on-demand mode", labels,
              l4_read_labels),
          l4_read_bytes_total(
              "mooncake_l4_read_bytes_total",
              "Successfully read DFS bytes by prefetch or on-demand mode",
              labels, l4_read_labels),
          l4_read_duration_seconds(
              "mooncake_l4_read_duration_seconds",
              "DFS BatchGet duration in seconds by prefetch or on-demand mode",
              kStorageLatencySecondsBucket, labels, l4_read_labels),
          l4_restore_ranges_total("mooncake_l4_restore_ranges_total",
                                  "Successfully restored DFS ranges by "
                                  "prefetched or on-demand mode",
                                  labels, l4_restore_labels),
          l4_restore_bytes_total("mooncake_l4_restore_bytes_total",
                                 "Successfully restored DFS payload bytes by "
                                 "prefetched or on-demand mode",
                                 labels, l4_restore_labels) {}

    ylt::metric::hybrid_counter_2t access_total;
    ylt::metric::hybrid_counter_2t access_bytes_total;
    ylt::metric::hybrid_counter_1t session_cache_access_total;
    ylt::metric::hybrid_counter_1t session_cache_evictions_total;
    ylt::metric::hybrid_counter_3t io_operations_total;
    ylt::metric::hybrid_counter_3t io_bytes_total;
    ylt::metric::hybrid_histogram_3d io_duration_seconds;
    ylt::metric::counter_t prefetched_tokens_total;
    ylt::metric::hybrid_counter_2t l4_read_operations_total;
    ylt::metric::hybrid_counter_2t l4_read_bytes_total;
    ylt::metric::hybrid_histogram_2d l4_read_duration_seconds;
    ylt::metric::hybrid_counter_1t l4_restore_ranges_total;
    ylt::metric::hybrid_counter_1t l4_restore_bytes_total;

    void ObserveL4Read(bool prefetch, bool success, uint64_t bytes,
                       double duration_seconds) {
        const std::array<std::string, 2> label = {
            prefetch ? "prefetch" : "on_demand",
            success ? "success" : "failure"};
        l4_read_operations_total.inc(label);
        l4_read_bytes_total.inc(label, bytes);
        l4_read_duration_seconds.observe(label, duration_seconds);
    }

    void ObserveL4Restore(bool prefetched, uint64_t ranges, uint64_t bytes) {
        const std::array<std::string, 1> label = {prefetched ? "prefetched"
                                                             : "on_demand"};
        l4_restore_ranges_total.inc(label, ranges);
        l4_restore_bytes_total.inc(label, bytes);
    }
    void ObserveAccess(const std::string& source, bool success,
                       uint64_t bytes) {
        const std::array<std::string, 2> label = {
            source, success ? "success" : "failure"};
        access_total.inc(label);
        access_bytes_total.inc(label, bytes);
    }

    void ObservePrefetchedTokens(uint64_t tokens) {
        prefetched_tokens_total.inc(tokens);
    }

    void ObserveSessionCache(bool hit) {
        session_cache_access_total.inc({hit ? "hit" : "miss"});
    }

    void ObserveSessionCacheEviction() {
        session_cache_evictions_total.inc({"session_missing"});
    }

    void ObserveIo(const std::string& operation, const std::string& source,
                   bool success, uint64_t bytes, double duration_seconds) {
        const std::array<std::string, 3> label = {
            operation, source, success ? "success" : "failure"};
        io_operations_total.inc(label);
        io_bytes_total.inc(label, bytes);
        io_duration_seconds.observe(label, duration_seconds);
    }

    void serialize(std::string& str) {
        prefetched_tokens_total.serialize(str);
        l4_read_operations_total.serialize(str);
        l4_read_bytes_total.serialize(str);
        l4_read_duration_seconds.serialize(str);
        l4_restore_ranges_total.serialize(str);
        l4_restore_bytes_total.serialize(str);
        access_total.serialize(str);
        access_bytes_total.serialize(str);
        session_cache_access_total.serialize(str);
        session_cache_evictions_total.serialize(str);
        io_operations_total.serialize(str);
        io_bytes_total.serialize(str);
        io_duration_seconds.serialize(str);
    }
};

// SSD latency bucket: microseconds, tuned for SSD/network storage
// Range: 50us (high-end NVMe) to 30s (3fs/nfs large object batch writes)
inline const std::vector<double> kSsdLatencyBucket = {
    50,       100,     200,                      // <200us (high-end NVMe)
    500,      1000,    2000,    5000,    10000,  // 500us - 10ms
    20000,    50000,   100000,  200000,          // 10ms - 200ms
    500000,   1000000, 2000000, 5000000,         // 500ms - 5s
    10000000, 30000000                           // 10s - 30s (3fs/nfs)
};

struct UnifiedBufferMetric {
    UnifiedBufferMetric(std::map<std::string, std::string> labels = {})
        : capacity_bytes("mooncake_dfs_unified_buffer_capacity_bytes",
                         "Unified DFS buffer capacity in bytes", labels),
          read_bytes("mooncake_dfs_unified_buffer_read_bytes",
                     "Unified DFS buffer read allocation bytes", labels),
          prefetch_bytes("mooncake_dfs_unified_buffer_prefetch_bytes",
                         "Unified DFS buffer prefetch allocation bytes",
                         labels),
          write_bytes("mooncake_dfs_unified_buffer_write_bytes",
                      "Unified DFS buffer write allocation bytes", labels),
          total_bytes("mooncake_dfs_unified_buffer_total_bytes",
                      "Unified DFS buffer total allocation bytes", labels),
          peak_total_bytes("mooncake_dfs_unified_buffer_peak_total_bytes",
                           "Unified DFS buffer peak allocation bytes", labels),
          write_backlog("mooncake_dfs_unified_buffer_write_backlog",
                        "In-flight asynchronous DFS writes", labels),
          read_allocation_failures(
              "mooncake_dfs_unified_buffer_read_allocation_failures_total",
              "Unified DFS buffer read allocation failures", labels),
          prefetch_allocation_failures(
              "mooncake_dfs_unified_buffer_prefetch_allocation_failures_total",
              "Unified DFS buffer prefetch allocation failures", labels),
          write_allocation_failures(
              "mooncake_dfs_unified_buffer_write_allocation_failures_total",
              "Unified DFS buffer write allocation failures", labels) {}

    ylt::metric::gauge_t capacity_bytes;
    ylt::metric::gauge_t read_bytes;
    ylt::metric::gauge_t prefetch_bytes;
    ylt::metric::gauge_t write_bytes;
    ylt::metric::gauge_t total_bytes;
    ylt::metric::gauge_t peak_total_bytes;
    ylt::metric::gauge_t write_backlog;
    ylt::metric::counter_t read_allocation_failures;
    ylt::metric::counter_t prefetch_allocation_failures;
    ylt::metric::counter_t write_allocation_failures;

    void serialize(std::string& str) {
        capacity_bytes.serialize(str);
        read_bytes.serialize(str);
        prefetch_bytes.serialize(str);
        write_bytes.serialize(str);
        total_bytes.serialize(str);
        peak_total_bytes.serialize(str);
        write_backlog.serialize(str);
        read_allocation_failures.serialize(str);
        prefetch_allocation_failures.serialize(str);
        write_allocation_failures.serialize(str);
    }
};

}  // namespace mooncake
