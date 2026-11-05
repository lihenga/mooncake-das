// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "master_metric_manager.h"
#include "storage/distributed/dfs_log_limiter.h"
#include "types.h"

namespace mooncake::master_policy_detail {

inline bool IsDfsReplicaFallbackError(ErrorCode error) {
    return error == ErrorCode::FILE_OPEN_FAIL ||
           error == ErrorCode::FILE_WRITE_FAIL ||
           error == ErrorCode::NO_AVAILABLE_HANDLE ||
           error == ErrorCode::DFS_SERVICE_UNAVAILABLE;
}

inline std::optional<ErrorCode> InitialDfsReplicaFallbackReason(
    const ReplicateConfig& config, bool dfs_available) {
    if (config.dfs_replica_num > 0 && !dfs_available) {
        return ErrorCode::DFS_SERVICE_UNAVAILABLE;
    }
    return std::nullopt;
}

inline void RecordDfsReplicaFallback(std::string_view key, ErrorCode reason) {
    MasterMetricManager::instance().inc_dfs_replica_skipped_total(
        toString(reason));
    static DfsLogLimiter fallback_log_limiter;
    if (auto suppressed = fallback_log_limiter.ShouldLog()) {
        LOG(WARNING) << "key=" << key
                     << ", action=dfs_replica_skipped_memory_fallback"
                     << ", error=" << toString(reason)
                     << ", suppressed=" << *suppressed;
    }
}

}  // namespace mooncake::master_policy_detail
