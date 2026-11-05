// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "master_client.h"
#include "l4/master_client_rpc_internal.h"

#include <async_simple/coro/FutureAwaiter.h>
#include <async_simple/coro/Lazy.h>
#include <async_simple/coro/SyncAwait.h>

#include <csignal>
#include <string>
#include <vector>
#include <ylt/coro_rpc/impl/coro_rpc_client.hpp>
#include <ylt/util/tl/expected.hpp>

#include "mutex.h"
#include "rpc_service.h"
#include "types.h"
#include "utils/scoped_vlog_timer.h"
#include "master_metric_manager.h"
#include "version.h"

namespace mooncake {

void MasterClient::EnableHaConnectionPolicy() {
    MutexLocker lock(&connect_mutex_);
    if (!client_addr_param_.empty()) {
        ha_control_client_accessor_.GetOrCreateClientPool(client_addr_param_);
    }
    ha_connection_policy_enabled_.store(true, std::memory_order_release);
}

std::vector<tl::expected<bool, ErrorCode>>
MasterClient::BatchExistKeyByReplicaType(
    const std::vector<std::string>& object_keys, ReplicaType replica_type) {
    ScopedVLogTimer timer(1, "MasterClient::BatchExistKeyByReplicaType");
    timer.LogRequest("keys_count=", object_keys.size(),
                     ", replica_type=", static_cast<int>(replica_type));

    auto result =
        invoke_batch_rpc<&WrappedMasterService::BatchExistKeyByReplicaType,
                         bool>(object_keys.size(), object_keys,
                               tenant_id_.value(), replica_type);
    timer.LogResponse("result=", result.size(), " keys");
    return result;
}

tl::expected<void, ErrorCode> MasterClient::ReportServiceMetrics(
    const ServiceMetricsSnapshot& snapshot) {
    return invoke_rpc<&WrappedMasterService::ReportServiceMetrics, void>(
        client_id_, snapshot);
}

std::function<tl::expected<UUID, ErrorCode>()>
MasterClient::MakeDfsMasterEpochProvider() {
    return [accessor = client_accessor_]() -> tl::expected<UUID, ErrorCode> {
        const auto pool = accessor->GetClientPool();
        if (!pool) return tl::make_unexpected(ErrorCode::RPC_FAIL);
        return async_simple::coro::syncAwait(
            [&]() -> async_simple::coro::Lazy<tl::expected<UUID, ErrorCode>> {
                auto response = co_await pool->send_request(
                    [&](coro_rpc::coro_rpc_client& client)
                        -> async_simple::coro::Lazy<
                            tl::expected<UUID, ErrorCode>> {
                        auto result = co_await client.call_for<
                            &WrappedMasterService::GetDfsMasterEpoch>(
                            std::chrono::seconds(2));
                        if (!result) {
                            co_return tl::make_unexpected(
                                result.error().code == coro_rpc::errc::timed_out
                                    ? ErrorCode::RPC_TIMEOUT
                                    : ErrorCode::RPC_FAIL);
                        }
                        if (!*result) {
                            co_return tl::make_unexpected(result->error());
                        }
                        co_return result->value();
                    });
                if (!response) {
                    co_return tl::make_unexpected(ErrorCode::RPC_FAIL);
                }
                co_return std::move(*response);
            }());
    };
}

tl::expected<DfsBucketSnapshotResponse, ErrorCode>
MasterClient::GetDfsBucketSnapshot(const UUID& client_id,
                                   const UUID& last_epoch,
                                   uint64_t last_version) {
    return invoke_rpc<&WrappedMasterService::GetDfsBucketSnapshot,
                      DfsBucketSnapshotResponse>(client_id, last_epoch,
                                                 last_version);
}

tl::expected<CopyStartResponse, ErrorCode> MasterClient::DfsBackfillStart(
    const std::string& key, const std::string& tenant_id,
    const std::string& source, const UUID& task_id) {
    return invoke_rpc<&WrappedMasterService::DfsBackfillStart,
                      CopyStartResponse>(client_id_, key, tenant_id, source,
                                         task_id);
}

tl::expected<void, ErrorCode> MasterClient::DfsBackfillFinish(
    const std::string& key, const std::string& tenant_id, const UUID& task_id,
    bool success) {
    return invoke_rpc<&WrappedMasterService::DfsBackfillFinish, void>(
        client_id_, key, tenant_id, task_id, success);
}

std::vector<tl::expected<void, ErrorCode>>
MasterClient::BatchInvalidateDfsBuckets(
    const std::vector<DfsMissingBucketReport>& reports) {
    ScopedVLogTimer timer(1, "MasterClient::BatchInvalidateDfsBuckets");
    timer.LogRequest("report_count=", reports.size());
    auto results =
        invoke_batch_rpc<&WrappedMasterService::BatchInvalidateDfsBuckets,
                         void>(reports.size(), client_id_, reports,
                               tenant_id_.value());
    timer.LogResponse("result=", results.size(), " operations");
    return results;
}

}  // namespace mooncake
