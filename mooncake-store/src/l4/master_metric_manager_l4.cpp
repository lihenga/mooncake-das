// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "master_metric_manager.h"

#include <glog/logging.h>
#include <algorithm>
#include <iomanip>  // For std::fixed, std::setprecision
#include <limits>   // For std::numeric_limits
#include <sstream>  // For string building during serialization
#include <vector>   // Required by histogram serialization
#include <cmath>
#include <limits>

#include "segment.h"
#include "utils.h"
#include "rpc_types.h"

namespace mooncake {

tl::expected<void, ErrorCode> MasterMetricManager::ReportServiceMetrics(
    const UUID& client_id, const ServiceMetricsSnapshot& snapshot) {
    if (snapshot.service_endpoint.empty() || snapshot.sequence == 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    const std::array<uint64_t, 4> dual_write{
        snapshot.dual_write_both_success, snapshot.dual_write_memory_only,
        snapshot.dual_write_dfs_only, snapshot.dual_write_neither_success};
    const std::array<uint64_t, 2> read_failures{snapshot.read_failures_memory,
                                                snapshot.read_failures_dfs};
    const std::array<const char*, 4> outcomes{"both_success", "memory_only",
                                              "dfs_only", "neither_success"};
    const std::array<const char*, 2> replica_types{"memory", "dfs"};

    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    auto client_it = service_metric_clients_.find(client_id);
    const ServiceMetricClientState* previous =
        client_it == service_metric_clients_.end() ? nullptr
                                                   : &client_it->second;
    if (previous && previous->endpoint != snapshot.service_endpoint) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (previous && snapshot.sequence == previous->sequence) {
        if (dual_write != previous->dual_write ||
            read_failures != previous->read_failures ||
            snapshot.dual_write_pending != previous->pending ||
            snapshot.l4 != previous->l4) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
    }
    if (previous && snapshot.sequence < previous->sequence) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    std::array<uint64_t, 4> dual_delta{};
    std::array<uint64_t, 2> read_delta{};
    for (size_t i = 0; i < dual_write.size(); ++i) {
        const uint64_t old_value = previous ? previous->dual_write[i] : 0;
        if (dual_write[i] < old_value ||
            dual_write[i] - old_value >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        dual_delta[i] = dual_write[i] - old_value;
    }
    for (size_t i = 0; i < read_failures.size(); ++i) {
        const uint64_t old_value = previous ? previous->read_failures[i] : 0;
        if (read_failures[i] < old_value ||
            read_failures[i] - old_value >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        read_delta[i] = read_failures[i] - old_value;
    }

    ServiceL4Metrics l4_delta;
    const ServiceL4Metrics empty_l4;
    const auto& old_l4 = previous ? previous->l4 : empty_l4;
    auto counter_delta = [](const auto& current, const auto& old, auto& delta) {
        for (size_t i = 0; i < current.size(); ++i) {
            if (current[i] < old[i] ||
                current[i] - old[i] >
                    static_cast<uint64_t>(
                        std::numeric_limits<int64_t>::max())) {
                return false;
            }
            delta[i] = current[i] - old[i];
        }
        return true;
    };
    if (!counter_delta(snapshot.l4.read_operations, old_l4.read_operations,
                       l4_delta.read_operations) ||
        !counter_delta(snapshot.l4.read_bytes, old_l4.read_bytes,
                       l4_delta.read_bytes) ||
        !counter_delta(snapshot.l4.read_duration_us, old_l4.read_duration_us,
                       l4_delta.read_duration_us) ||
        !counter_delta(snapshot.l4.restore_ranges, old_l4.restore_ranges,
                       l4_delta.restore_ranges) ||
        !counter_delta(snapshot.l4.restore_bytes, old_l4.restore_bytes,
                       l4_delta.restore_bytes)) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    for (size_t i = 0; i < snapshot.l4.read_operations.size(); ++i) {
        if (!counter_delta(snapshot.l4.read_duration_buckets[i],
                           old_l4.read_duration_buckets[i],
                           l4_delta.read_duration_buckets[i])) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        uint64_t observations = 0;
        for (const auto count : snapshot.l4.read_duration_buckets[i]) {
            if (count > snapshot.l4.read_operations[i] - observations) {
                return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
            }
            observations += count;
        }
        if (observations != snapshot.l4.read_operations[i] ||
            (l4_delta.read_operations[i] == 0 &&
             (l4_delta.read_bytes[i] != 0 ||
              l4_delta.read_duration_us[i] != 0))) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
    }

    if (snapshot.dual_write_pending >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    auto& endpoint_pending =
        service_pending_by_endpoint_[snapshot.service_endpoint];
    const int64_t old_client_pending =
        previous && previous->pending_active
            ? static_cast<int64_t>(previous->pending)
            : 0;
    const int64_t new_client_pending =
        static_cast<int64_t>(snapshot.dual_write_pending);
    if (endpoint_pending < old_client_pending ||
        new_client_pending > std::numeric_limits<int64_t>::max() -
                                 (endpoint_pending - old_client_pending)) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    const int64_t new_endpoint_pending =
        endpoint_pending - old_client_pending + new_client_pending;

    // Initialize every outcome series on the first accepted report for an
    // endpoint so zero-count outcomes remain visible in Prometheus output.
    for (const char* outcome : outcomes) {
        service_dual_write_total_.inc({snapshot.service_endpoint, outcome}, 0);
    }
    for (const char* replica_type : replica_types) {
        service_read_failures_total_.inc(
            {snapshot.service_endpoint, replica_type}, 0);
    }
    for (size_t i = 0; i < dual_delta.size(); ++i) {
        if (dual_delta[i] != 0) {
            service_dual_write_total_.inc(
                {snapshot.service_endpoint, outcomes[i]},
                static_cast<int64_t>(dual_delta[i]));
        }
    }
    for (size_t i = 0; i < read_delta.size(); ++i) {
        if (read_delta[i] != 0) {
            service_read_failures_total_.inc(
                {snapshot.service_endpoint, replica_types[i]},
                static_cast<int64_t>(read_delta[i]));
        }
    }
    for (size_t i = 0; i < l4_delta.read_operations.size(); ++i) {
        const std::array<std::string, 3> labels = {
            snapshot.service_endpoint, i < 2 ? "prefetch" : "on_demand",
            i % 2 == 0 ? "success" : "failure"};
        service_l4_read_operations_total_.inc(labels,
                                              l4_delta.read_operations[i]);
        service_l4_read_bytes_total_.inc(labels, l4_delta.read_bytes[i]);
        if (l4_delta.read_operations[i] != 0) {
            service_l4_read_duration_seconds_.merge(
                labels, l4_delta.read_duration_buckets[i],
                static_cast<double>(l4_delta.read_duration_us[i]) / 1e6);
        }
    }
    for (size_t i = 0; i < l4_delta.restore_bytes.size(); ++i) {
        const std::array<std::string, 2> labels = {
            snapshot.service_endpoint, i == 0 ? "prefetched" : "on_demand"};
        service_l4_restore_ranges_total_.inc(labels,
                                             l4_delta.restore_ranges[i]);
        service_l4_restore_bytes_total_.inc(labels, l4_delta.restore_bytes[i]);
    }
    if (!previous || !previous->pending_active) {
        service_reporting_clients_.inc({snapshot.service_endpoint}, 1);
    }
    endpoint_pending = new_endpoint_pending;
    service_dual_write_pending_.update({snapshot.service_endpoint},
                                       new_endpoint_pending);

    ServiceMetricClientState state;
    state.endpoint = snapshot.service_endpoint;
    state.sequence = snapshot.sequence;
    state.dual_write = dual_write;
    state.pending = snapshot.dual_write_pending;
    state.read_failures = read_failures;
    state.l4 = snapshot.l4;
    state.last_report = std::chrono::steady_clock::now();
    service_metric_clients_[client_id] = std::move(state);
    return {};
}

void MasterMetricManager::ExpireServiceMetricsPending(
    std::chrono::steady_clock::time_point now) {
    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    for (auto& [client_id, state] : service_metric_clients_) {
        if (!state.pending_active ||
            now - state.last_report < std::chrono::seconds(30)) {
            continue;
        }
        auto& pending = service_pending_by_endpoint_.at(state.endpoint);
        pending -= static_cast<int64_t>(state.pending);
        service_dual_write_pending_.update({state.endpoint}, pending);
        state.pending_active = false;
        service_reporting_clients_.dec({state.endpoint}, 1);
    }
}

void MasterMetricManager::set_dfs_storage_usage(uint64_t capacity_bytes,
                                                uint64_t used_bytes,
                                                bool unlimited) {
    const auto clamp = [](uint64_t value) {
        return static_cast<int64_t>(std::min<uint64_t>(
            value, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
    };
    dfs_capacity_bytes_.update(clamp(capacity_bytes));
    dfs_used_bytes_.update(clamp(used_bytes));
    dfs_capacity_unlimited_metric_.update(unlimited ? 1 : 0);
}

void MasterMetricManager::inc_dfs_replica(const std::string& status,
                                          int64_t val) {
    dfs_replicas_.inc({status}, val);
}

void MasterMetricManager::dec_dfs_replica(const std::string& status,
                                          int64_t val) {
    dfs_replicas_.dec({status}, val);
}

void MasterMetricManager::transition_dfs_replica(const std::string& from,
                                                 const std::string& to) {
    dfs_replicas_.dec({from});
    dfs_replicas_.inc({to});
}

void MasterMetricManager::inc_dfs_replica_skipped_total(
    const std::string& reason) {
    dfs_replica_skipped_total_.inc({reason});
}

void MasterMetricManager::set_dfs_bucket_pool_state(
    uint64_t ready_count, uint64_t create_inflight,
    uint64_t ready_reserved_bytes) {
    const auto clamp = [](uint64_t value) {
        return static_cast<int64_t>(std::min<uint64_t>(
            value, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
    };
    dfs_bucket_ready_count_.update(clamp(ready_count));
    dfs_bucket_create_inflight_.update(clamp(create_inflight));
    dfs_bucket_ready_reserved_bytes_.update(clamp(ready_reserved_bytes));
}

void MasterMetricManager::observe_dfs_bucket_create_latency_us(
    int64_t latency_us) {
    dfs_bucket_create_latency_us_.observe(latency_us);
}

void MasterMetricManager::inc_dfs_bucket_create_total(
    const std::string& result) {
    dfs_bucket_create_total_.inc({result});
}

void MasterMetricManager::inc_dfs_bucket_pool_exhausted_total() {
    dfs_bucket_pool_exhausted_total_.inc();
}

void MasterMetricManager::observe_dfs_bucket_pool_wait_latency_us(
    int64_t latency_us) {
    dfs_bucket_pool_wait_latency_us_.observe(latency_us);
}

void MasterMetricManager::observe_dfs_bucket_rollover_latency_us(
    int64_t latency_us) {
    dfs_bucket_rollover_latency_us_.observe(latency_us);
}

void MasterMetricManager::inc_dfs_cache_hit_nums(int64_t val) {
    dfs_cache_hit_nums_.inc(val);
}

void MasterMetricManager::inc_dfs_cache_hit_bytes(int64_t val) {
    dfs_cache_hit_bytes_.inc(val);
}

void MasterMetricManager::inc_dfs_bucket_eviction_attempts(int64_t val) {
    dfs_bucket_eviction_attempts_.inc(val);
}

void MasterMetricManager::inc_dfs_bucket_evictions(int64_t val) {
    dfs_bucket_evictions_.inc(val);
}

void MasterMetricManager::inc_dfs_bucket_low_utilization_eviction_attempts(
    int64_t val) {
    dfs_bucket_low_utilization_eviction_attempts_.inc(val);
}

void MasterMetricManager::inc_dfs_bucket_low_utilization_evictions(
    int64_t val) {
    dfs_bucket_low_utilization_evictions_.inc(val);
}

void MasterMetricManager::inc_dfs_bucket_evicted_replicas(int64_t val) {
    dfs_bucket_evicted_replicas_.inc(val);
}

void MasterMetricManager::inc_dfs_bucket_evicted_bytes(int64_t val) {
    dfs_bucket_evicted_bytes_.inc(val);
}

}  // namespace mooncake
