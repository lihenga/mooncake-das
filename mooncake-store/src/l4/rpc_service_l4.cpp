// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "rpc_service.h"
#include "l4/rpc_service_internal.h"

#include <cstddef>
#include <functional>
#include <string_view>
#include <type_traits>

#include <ylt/coro_rpc/coro_rpc_server.hpp>
#include <ylt/util/tl/expected.hpp>

#include "master_metric_manager.h"
#include "master_service.h"
#include "rpc_helper.h"
#include "types.h"
#include "common/scoped_vlog_timer.h"
#include "version.h"

namespace mooncake {

using namespace rpc_detail;

void WrappedMasterService::RefreshDfsMetrics() const {
    master_service_.RefreshDfsMetrics();
}

std::vector<tl::expected<bool, ErrorCode>>
WrappedMasterService::BatchExistKeyByReplicaType(
    const std::vector<std::string>& keys, const std::string& tenant_id,
    ReplicaType replica_type) {
    ScopedVLogTimer timer(1, "BatchExistKeyByReplicaType");
    const size_t total_keys = keys.size();
    timer.LogRequest("keys_count=", total_keys,
                     ", replica_type=", static_cast<int>(replica_type));
    MasterMetricManager::instance().inc_batch_exist_key_requests(total_keys);

    auto result = WithRequestTenantBatch(
        master_service_.IsTenantQuotaEnabled() ? std::string_view(tenant_id)
                                               : TenantId::kDefaultValue,
        keys.size(), [&](const TenantId& resolved_tenant_id) {
            return master_service_.BatchExistKeyByReplicaType(
                keys, resolved_tenant_id, replica_type);
        });

    size_t failure_count = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        if (!result[i].has_value()) {
            failure_count++;
            LOG(ERROR) << "BatchExistKeyByReplicaType failed for key[" << i
                       << "] '" << keys[i]
                       << "': " << toString(result[i].error());
        }
    }

    if (failure_count == total_keys) {
        MasterMetricManager::instance().inc_batch_exist_key_failures(
            failure_count);
    } else if (failure_count != 0) {
        MasterMetricManager::instance().inc_batch_exist_key_partial_success(
            failure_count);
    }

    timer.LogResponse("total=", result.size(),
                      ", success=", result.size() - failure_count,
                      ", failures=", failure_count);
    return result;
}

tl::expected<CopyStartResponse, ErrorCode>
WrappedMasterService::DfsBackfillStart(const UUID& client_id,
                                       const std::string& key,
                                       const std::string& tenant_id,
                                       const std::string& source,
                                       const UUID& task_id) {
    auto result = execute_rpc(
        "DfsBackfillStart",
        [&] {
            return WithWriteTenant(
                tenant_id, master_service_.IsTenantQuotaEnabled(),
                [&](const TenantId& tenant) {
                    return master_service_.DfsBackfillStart(
                        client_id, key, tenant, source, task_id);
                });
        },
        [&](auto& timer) { timer.LogRequest("key=", key, ", task=", task_id); },
        [] { MasterMetricManager::instance().inc_copy_start_requests(); },
        [] { MasterMetricManager::instance().inc_copy_start_failures(); });
    if (result) {
        StampDfsEpoch(result->targets, master_service_.GetDfsMasterEpoch());
    }
    return result;
}

tl::expected<void, ErrorCode> WrappedMasterService::DfsBackfillFinish(
    const UUID& client_id, const std::string& key, const std::string& tenant_id,
    const UUID& task_id, bool success) {
    return execute_rpc(
        "DfsBackfillFinish",
        [&] {
            return WithRequestTenant(
                master_service_.IsTenantQuotaEnabled()
                    ? std::string_view(tenant_id)
                    : TenantId::kDefaultValue,
                [&](const TenantId& tenant) {
                    return success ? master_service_.CopyEnd(client_id, key,
                                                             tenant, task_id)
                                   : master_service_.CopyRevoke(
                                         client_id, key, tenant, task_id);
                });
        },
        [&](auto& timer) { timer.LogRequest("key=", key, ", task=", task_id); },
        [] { MasterMetricManager::instance().inc_copy_end_requests(); },
        [] { MasterMetricManager::instance().inc_copy_end_failures(); });
}

std::vector<tl::expected<void, ErrorCode>>
WrappedMasterService::BatchInvalidateDfsBuckets(
    const UUID& client_id, const std::vector<DfsMissingBucketReport>& reports,
    const std::string& tenant_id) {
    ScopedVLogTimer timer(1, "BatchInvalidateDfsBuckets");
    timer.LogRequest("client_id=", client_id,
                     ", report_count=", reports.size());
    auto results = WithRequestTenantBatch(
        master_service_.IsTenantQuotaEnabled() ? std::string_view(tenant_id)
                                               : TenantId::kDefaultValue,
        reports.size(), [&](const TenantId& resolved_tenant_id) {
            return master_service_.BatchInvalidateDfsBuckets(
                client_id, reports, resolved_tenant_id);
        });
    size_t failure_count = 0;
    for (const auto& result : results) {
        if (!result.has_value()) ++failure_count;
    }
    timer.LogResponse("total=", results.size(),
                      ", success=", results.size() - failure_count,
                      ", failures=", failure_count);
    return results;
}

tl::expected<void, ErrorCode> WrappedMasterService::ReportServiceMetrics(
    const UUID& client_id, const ServiceMetricsSnapshot& snapshot) {
    return MasterMetricManager::instance().ReportServiceMetrics(client_id,
                                                                snapshot);
}

tl::expected<int64_t, ErrorCode> WrappedMasterService::SetDfsMaxBucketCount(
    int64_t new_max_bucket_count) {
    return master_service_.SetDfsMaxBucketCount(new_max_bucket_count);
}

tl::expected<uint32_t, ErrorCode> WrappedMasterService::SetDfsHeartbeatInterval(
    uint32_t new_interval_seconds) {
    return master_service_.SetDfsHeartbeatInterval(new_interval_seconds);
}

uint64_t WrappedMasterService::SetDfsBatchTimeoutFixedMs(
    uint64_t new_timeout_ms) {
    return master_service_.SetDfsBatchTimeoutFixedMs(new_timeout_ms);
}

tl::expected<uint8_t, ErrorCode> WrappedMasterService::SetDfsFilterMode(
    uint8_t new_filter_mode) {
    return master_service_.SetDfsFilterMode(new_filter_mode);
}

uint8_t WrappedMasterService::GetDfsFilterMode() const {
    return master_service_.GetDfsFilterMode();
}

tl::expected<UUID, ErrorCode> WrappedMasterService::GetDfsMasterEpoch() {
    return master_service_.GetDfsMasterEpoch();
}

tl::expected<DfsBucketSnapshotResponse, ErrorCode>
WrappedMasterService::GetDfsBucketSnapshot(const UUID& client_id,
                                           const UUID& last_epoch,
                                           uint64_t last_version) {
    ScopedVLogTimer timer(1, "GetDfsBucketSnapshot");
    return master_service_.GetDfsBucketSnapshot(client_id, last_epoch,
                                                last_version);
}

}  // namespace mooncake
