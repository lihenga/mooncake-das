// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "master_client.h"

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
#include "common/scoped_vlog_timer.h"
#include "master_metric_manager.h"
#include "version.h"

namespace mooncake {

template <auto Method>
struct RpcNameTraits;

template <>
struct RpcNameTraits<&WrappedMasterService::ExistKey> {
    static constexpr const char* value = "ExistKey";
};

template <>
struct RpcNameTraits<&WrappedMasterService::ProbeKey> {
    static constexpr const char* value = "ProbeKey";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchExistKey> {
    static constexpr const char* value = "BatchExistKey";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchProbeKey> {
    static constexpr const char* value = "BatchProbeKey";
};

template <>
struct RpcNameTraits<&WrappedMasterService::GetReplicaList> {
    static constexpr const char* value = "GetReplicaList";
};

template <>
struct RpcNameTraits<&WrappedMasterService::GetReplicaListForAdmin> {
    static constexpr const char* value = "GetReplicaListForAdmin";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchGetReplicaListForAdmin> {
    static constexpr const char* value = "BatchGetReplicaListForAdmin";
};

template <>
struct RpcNameTraits<&WrappedMasterService::CalcCacheStats> {
    static constexpr const char* value = "CalcCacheStats";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchQueryIp> {
    static constexpr const char* value = "BatchQueryIp";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchReplicaClear> {
    static constexpr const char* value = "BatchReplicaClear";
};

template <>
struct RpcNameTraits<&WrappedMasterService::GetReplicaListByRegex> {
    static constexpr const char* value = "GetReplicaListByRegex";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchGetReplicaList> {
    static constexpr const char* value = "BatchGetReplicaList";
};

template <>
struct RpcNameTraits<&WrappedMasterService::PutStart> {
    static constexpr const char* value = "PutStart";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchPutStart> {
    static constexpr const char* value = "BatchPutStart";
};

template <>
struct RpcNameTraits<&WrappedMasterService::PutEnd> {
    static constexpr const char* value = "PutEnd";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchPutEnd> {
    static constexpr const char* value = "BatchPutEnd";
};

template <>
struct RpcNameTraits<&WrappedMasterService::PutRevoke> {
    static constexpr const char* value = "PutRevoke";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchPutRevoke> {
    static constexpr const char* value = "BatchPutRevoke";
};

template <>
struct RpcNameTraits<&WrappedMasterService::UpsertStart> {
    static constexpr const char* value = "UpsertStart";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchUpsertStart> {
    static constexpr const char* value = "BatchUpsertStart";
};

template <>
struct RpcNameTraits<&WrappedMasterService::UpsertEnd> {
    static constexpr const char* value = "UpsertEnd";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchUpsertEnd> {
    static constexpr const char* value = "BatchUpsertEnd";
};

template <>
struct RpcNameTraits<&WrappedMasterService::UpsertRevoke> {
    static constexpr const char* value = "UpsertRevoke";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchUpsertRevoke> {
    static constexpr const char* value = "BatchUpsertRevoke";
};

template <>
struct RpcNameTraits<&WrappedMasterService::Remove> {
    static constexpr const char* value = "Remove";
};

template <>
struct RpcNameTraits<&WrappedMasterService::RemoveByRegex> {
    static constexpr const char* value = "RemoveByRegex";
};

template <>
struct RpcNameTraits<&WrappedMasterService::RemoveAll> {
    static constexpr const char* value = "RemoveAll";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchRemove> {
    static constexpr const char* value = "BatchRemove";
};

template <>
struct RpcNameTraits<&WrappedMasterService::MountSegment> {
    static constexpr const char* value = "MountSegment";
};

template <>
struct RpcNameTraits<&WrappedMasterService::MountNoFSegment> {
    static constexpr const char* value = "MountNoFSegment";
};

template <>
struct RpcNameTraits<&WrappedMasterService::ReMountSegment> {
    static constexpr const char* value = "ReMountSegment";
};

template <>
struct RpcNameTraits<&WrappedMasterService::ReMountNoFSegment> {
    static constexpr const char* value = "ReMountNoFSegment";
};

template <>
struct RpcNameTraits<&WrappedMasterService::UnmountSegment> {
    static constexpr const char* value = "UnmountSegment";
};

template <>
struct RpcNameTraits<&WrappedMasterService::GracefulUnmountSegment> {
    static constexpr const char* value = "GracefulUnmountSegment";
};

template <>
struct RpcNameTraits<&WrappedMasterService::UnmountNoFSegment> {
    static constexpr const char* value = "UnmountNoFSegment";
};

template <>
struct RpcNameTraits<&WrappedMasterService::GetAllNoFSegments> {
    static constexpr const char* value = "GetAllNoFSegments";
};

template <>
struct RpcNameTraits<&WrappedMasterService::GetNoFSegmentsByName> {
    static constexpr const char* value = "GetNoFSegmentsByName";
};

template <>
struct RpcNameTraits<&WrappedMasterService::Ping> {
    static constexpr const char* value = "Ping";
};

template <>
struct RpcNameTraits<&WrappedMasterService::GetFsdir> {
    static constexpr const char* value = "GetFsdir";
};

template <>
struct RpcNameTraits<&WrappedMasterService::QuerySegmentStatusById> {
    static constexpr const char* value = "QuerySegmentStatusById";
};

template <>
struct RpcNameTraits<&WrappedMasterService::GetStorageConfig> {
    static constexpr const char* value = "GetStorageConfig";
};

template <>
struct RpcNameTraits<&WrappedMasterService::ServiceReady> {
    static constexpr const char* value = "ServiceReady";
};

template <>
struct RpcNameTraits<&WrappedMasterService::MountLocalDiskSegment> {
    static constexpr const char* value = "MountLocalDiskSegment";
};

template <>
struct RpcNameTraits<&WrappedMasterService::UnmountLocalDiskSegment> {
    static constexpr const char* value = "UnmountLocalDiskSegment";
};

template <>
struct RpcNameTraits<&WrappedMasterService::OffloadObjectHeartbeat> {
    static constexpr const char* value = "OffloadObjectHeartbeat";
};

template <>
struct RpcNameTraits<&WrappedMasterService::ReportSsdCapacity> {
    static constexpr const char* value = "ReportSsdCapacity";
};

template <>
struct RpcNameTraits<&WrappedMasterService::NotifyOffloadSuccess> {
    static constexpr const char* value = "NotifyOffloadSuccess";
};

template <>
struct RpcNameTraits<&WrappedMasterService::PromotionObjectHeartbeat> {
    static constexpr const char* value = "PromotionObjectHeartbeat";
};

template <>
struct RpcNameTraits<&WrappedMasterService::PromotionAllocStart> {
    static constexpr const char* value = "PromotionAllocStart";
};

template <>
struct RpcNameTraits<&WrappedMasterService::RegisterPrefetchTask> {
    static constexpr const char* value = "RegisterPrefetchTask";
};

template <>
struct RpcNameTraits<&WrappedMasterService::NotifyPromotionSuccess> {
    static constexpr const char* value = "NotifyPromotionSuccess";
};

template <>
struct RpcNameTraits<&WrappedMasterService::NotifyPromotionFailure> {
    static constexpr const char* value = "NotifyPromotionFailure";
};

template <>
struct RpcNameTraits<&WrappedMasterService::CopyStart> {
    static constexpr const char* value = "CopyStart";
};
template <>
struct RpcNameTraits<&WrappedMasterService::DynamicReplicaCopyStart> {
    static constexpr const char* value = "DynamicReplicaCopyStart";
};

template <>
struct RpcNameTraits<&WrappedMasterService::CopyEnd> {
    static constexpr const char* value = "CopyEnd";
};
template <>
struct RpcNameTraits<&WrappedMasterService::DynamicReplicaCopyEnd> {
    static constexpr const char* value = "DynamicReplicaCopyEnd";
};

template <>
struct RpcNameTraits<&WrappedMasterService::CopyRevoke> {
    static constexpr const char* value = "CopyRevoke";
};
template <>
struct RpcNameTraits<&WrappedMasterService::DynamicReplicaCopyRevoke> {
    static constexpr const char* value = "DynamicReplicaCopyRevoke";
};

template <>
struct RpcNameTraits<&WrappedMasterService::MoveStart> {
    static constexpr const char* value = "MoveStart";
};

template <>
struct RpcNameTraits<&WrappedMasterService::MoveEnd> {
    static constexpr const char* value = "MoveEnd";
};

template <>
struct RpcNameTraits<&WrappedMasterService::MoveRevoke> {
    static constexpr const char* value = "MoveRevoke";
};

template <>
struct RpcNameTraits<&WrappedMasterService::CreateCopyTask> {
    static constexpr const char* value = "CreateCopyTask";
};

template <>
struct RpcNameTraits<&WrappedMasterService::CreateMoveTask> {
    static constexpr const char* value = "CreateMoveTask";
};

template <>
struct RpcNameTraits<&WrappedMasterService::QueryTask> {
    static constexpr const char* value = "QueryTask";
};

template <>
struct RpcNameTraits<&WrappedMasterService::FetchTasks> {
    static constexpr const char* value = "FetchTasks";
};

template <>
struct RpcNameTraits<&WrappedMasterService::MarkTaskToComplete> {
    static constexpr const char* value = "MarkTaskToComplete";
};

template <>
struct RpcNameTraits<&WrappedMasterService::EvictDiskReplica> {
    static constexpr const char* value = "EvictDiskReplica";
};

template <>
struct RpcNameTraits<&WrappedMasterService::EvictDiskReplicaIfCurrent> {
    static constexpr const char* value = "EvictDiskReplicaIfCurrent";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchEvictDiskReplica> {
    static constexpr const char* value = "BatchEvictDiskReplica";
};

template <>
struct RpcNameTraits<&WrappedMasterService::PollRemoveAll> {
    static constexpr const char* value = "PollRemoveAll";
};

#define DEFINE_WEIGHT_RPC_NAME(method)                    \
    template <>                                           \
    struct RpcNameTraits<&WrappedMasterService::method> { \
        static constexpr const char* value = #method;     \
    }

DEFINE_WEIGHT_RPC_NAME(BeginWeightImport);
DEFINE_WEIGHT_RPC_NAME(CommitWeightImport);
DEFINE_WEIGHT_RPC_NAME(AbortWeightImport);
DEFINE_WEIGHT_RPC_NAME(GetWeightRevision);
DEFINE_WEIGHT_RPC_NAME(ListWeightRevisions);
DEFINE_WEIGHT_RPC_NAME(AcquireWeightRevisionLease);
DEFINE_WEIGHT_RPC_NAME(RenewWeightRevisionLease);
DEFINE_WEIGHT_RPC_NAME(ReleaseWeightRevisionLease);
DEFINE_WEIGHT_RPC_NAME(StartWeightResidencyOperation);
DEFINE_WEIGHT_RPC_NAME(QueryWeightOperation);
DEFINE_WEIGHT_RPC_NAME(DeleteWeightRevision);

#undef DEFINE_WEIGHT_RPC_NAME

template <>
struct RpcNameTraits<&WrappedMasterService::BatchExistKeyByReplicaType> {
    static constexpr const char* value = "BatchExistKeyByReplicaType";
};

template <>
struct RpcNameTraits<&WrappedMasterService::ReportServiceMetrics> {
    static constexpr const char* value = "ReportServiceMetrics";
};

template <>
struct RpcNameTraits<&WrappedMasterService::DfsBackfillStart> {
    static constexpr const char* value = "DfsBackfillStart";
};

template <>
struct RpcNameTraits<&WrappedMasterService::DfsBackfillFinish> {
    static constexpr const char* value = "DfsBackfillFinish";
};

template <>
struct RpcNameTraits<&WrappedMasterService::BatchInvalidateDfsBuckets> {
    static constexpr const char* value = "BatchInvalidateDfsBuckets";
};

template <>
struct RpcNameTraits<&WrappedMasterService::GetDfsBucketSnapshot> {
    static constexpr const char* value = "GetDfsBucketSnapshot";
};
template <auto ServiceMethod, typename ReturnType, typename... Args>
tl::expected<ReturnType, ErrorCode> MasterClient::invoke_rpc_with_client_pool(
    const std::shared_ptr<RpcClientPool::ClientPool>& client_pool,
    Args&&... args) {
    RpcDrainGuard::ScopedCall inflight(rpc_drain_);
    if (!inflight.ok()) {
        return tl::make_unexpected(ErrorCode::RPC_FAIL);
    }
    // Increment RPC counter
    if (metrics_) {
        metrics_->rpc_count.inc({RpcNameTraits<ServiceMethod>::value});
    }

    auto start_time = std::chrono::steady_clock::now();
    return async_simple::coro::syncAwait(
        [&]() -> async_simple::coro::Lazy<tl::expected<ReturnType, ErrorCode>> {
            auto ret = co_await client_pool->send_request(
                [&](coro_io::client_reuse_hint,
                    coro_rpc::coro_rpc_client& client) {
                    return client.send_request<ServiceMethod>(
                        std::forward<Args>(args)...);
                });
            if (!ret.has_value()) {
                LOG(ERROR) << "Client not available";
                co_return tl::make_unexpected(ErrorCode::RPC_FAIL);
            }
            auto result = co_await std::move(ret.value());
            if (!result) {
                if (result.error().code == coro_rpc::errc::timed_out) {
                    LOG(ERROR) << "RPC call timed out: " << result.error().msg;
                    co_return tl::make_unexpected(ErrorCode::RPC_TIMEOUT);
                }
                LOG(ERROR) << "RPC call failed: " << result.error().msg;
                co_return tl::make_unexpected(ErrorCode::RPC_FAIL);
            }
            if (metrics_) {
                auto end_time = std::chrono::steady_clock::now();
                auto latency =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time);
                metrics_->rpc_latency.observe(
                    {RpcNameTraits<ServiceMethod>::value}, latency.count());
            }
            co_return result->result();
        }());
}

template <auto ServiceMethod, typename ReturnType, typename... Args>
tl::expected<ReturnType, ErrorCode> MasterClient::invoke_rpc_with_pool(
    RpcClientPool& client_accessor, Args&&... args) {
    return invoke_rpc_with_client_pool<ServiceMethod, ReturnType>(
        client_accessor.GetClientPool(), std::forward<Args>(args)...);
}

template <auto ServiceMethod, typename ReturnType, typename... Args>
tl::expected<ReturnType, ErrorCode> MasterClient::invoke_rpc(Args&&... args) {
    return invoke_rpc_with_pool<ServiceMethod, ReturnType>(
        *client_accessor_, std::forward<Args>(args)...);
}

template <auto ServiceMethod, typename ResultType, typename... Args>
std::vector<tl::expected<ResultType, ErrorCode>> MasterClient::invoke_batch_rpc(
    size_t input_size, Args&&... args) {
    RpcDrainGuard::ScopedCall inflight(rpc_drain_);
    if (!inflight.ok()) {
        return std::vector<tl::expected<ResultType, ErrorCode>>(
            input_size, tl::make_unexpected(ErrorCode::RPC_FAIL));
    }
    auto pool = client_accessor_->GetClientPool();

    // Increment RPC counter
    if (metrics_) {
        metrics_->rpc_count.inc({RpcNameTraits<ServiceMethod>::value});
    }

    auto start_time = std::chrono::steady_clock::now();
    return async_simple::coro::syncAwait(
        [&]() -> async_simple::coro::Lazy<
                  std::vector<tl::expected<ResultType, ErrorCode>>> {
            auto ret = co_await pool->send_request(
                [&](coro_io::client_reuse_hint,
                    coro_rpc::coro_rpc_client& client) {
                    return client.send_request<ServiceMethod>(
                        std::forward<Args>(args)...);
                });
            if (!ret.has_value()) {
                LOG(ERROR) << "Client not available";
                co_return std::vector<tl::expected<ResultType, ErrorCode>>(
                    input_size, tl::make_unexpected(ErrorCode::RPC_FAIL));
            }
            auto result = co_await std::move(ret.value());
            if (!result) {
                const ErrorCode err_code =
                    (result.error().code == coro_rpc::errc::timed_out)
                        ? ErrorCode::RPC_TIMEOUT
                        : ErrorCode::RPC_FAIL;
                LOG(ERROR) << "Batch RPC call failed: " << result.error().msg;
                std::vector<tl::expected<ResultType, ErrorCode>> error_results;
                error_results.reserve(input_size);
                for (size_t i = 0; i < input_size; ++i) {
                    error_results.emplace_back(tl::make_unexpected(err_code));
                }
                co_return error_results;
            }
            if (metrics_) {
                auto end_time = std::chrono::steady_clock::now();
                auto latency =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time);
                metrics_->rpc_latency.observe(
                    {RpcNameTraits<ServiceMethod>::value}, latency.count());
            }
            co_return result->result();
        }());
}

}  // namespace mooncake
