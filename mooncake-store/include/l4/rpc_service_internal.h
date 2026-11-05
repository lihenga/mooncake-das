// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rpc_service.h"

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
#include "utils/scoped_vlog_timer.h"
#include "version.h"

namespace mooncake::rpc_detail {

inline void StampDfsEpoch(std::vector<Replica::Descriptor>& replicas,
                          const UUID& epoch) {
    for (auto& replica : replicas) {
        if (replica.is_dfs_replica()) {
            replica.get_dfs_descriptor().master_epoch = epoch;
        }
    }
}

inline void StampDfsEpoch(
    std::unordered_map<std::string, std::vector<Replica::Descriptor>>&
        responses,
    const UUID& epoch) {
    for (auto& [key, replicas] : responses) StampDfsEpoch(replicas, epoch);
}

inline void StampDfsEpoch(GetReplicaListResponse& response, const UUID& epoch) {
    StampDfsEpoch(response.replicas, epoch);
}

template <typename T>
inline void StampDfsEpoch(tl::expected<T, ErrorCode>& result,
                          const UUID& epoch) {
    if (result) StampDfsEpoch(*result, epoch);
}

template <typename T>
inline void StampDfsEpoch(std::vector<tl::expected<T, ErrorCode>>& results,
                          const UUID& epoch) {
    for (auto& result : results) StampDfsEpoch(result, epoch);
}

inline tl::expected<TenantId, ErrorCode> ResolveRequestTenantId(
    std::string_view raw) {
    TenantId tenant_id{std::string(raw)};
    if (!tenant_id.IsValid()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    return tenant_id;
}

inline tl::expected<TenantId, ErrorCode> ResolveTenantIdForWrite(
    std::string_view raw, bool enable_multi_tenants) {
    if (!enable_multi_tenants) {
        return TenantId::Default();
    }
    if (raw.empty()) {
        return tl::make_unexpected(ErrorCode::TENANT_NOT_REGISTERED);
    }
    TenantId tenant_id{std::string(raw)};
    if (!tenant_id.IsValid()) {
        return tl::make_unexpected(ErrorCode::TENANT_NOT_REGISTERED);
    }
    return tenant_id;
}

template <typename Fn>
inline auto WithRequestTenant(std::string_view raw, Fn&& fn) {
    using Result = std::invoke_result_t<Fn, const TenantId&>;
    auto tenant_id = ResolveRequestTenantId(raw);
    if (!tenant_id) {
        return Result(tl::make_unexpected(tenant_id.error()));
    }
    return std::invoke(std::forward<Fn>(fn), tenant_id.value());
}

template <typename Fn>
inline auto WithRequestTenantBatch(std::string_view raw, size_t result_count,
                                   Fn&& fn) {
    using Results = std::invoke_result_t<Fn, const TenantId&>;
    auto tenant_id = ResolveRequestTenantId(raw);
    if (!tenant_id) {
        using Result = typename Results::value_type;
        return Results(result_count,
                       Result(tl::make_unexpected(tenant_id.error())));
    }
    return std::invoke(std::forward<Fn>(fn), tenant_id.value());
}

template <typename Fn>
inline auto WithWriteTenant(std::string_view raw, bool enable_multi_tenants,
                            Fn&& fn) {
    using Result = std::invoke_result_t<Fn, const TenantId&>;
    auto tenant_id = ResolveTenantIdForWrite(raw, enable_multi_tenants);
    if (!tenant_id) {
        return Result(tl::make_unexpected(tenant_id.error()));
    }
    return std::invoke(std::forward<Fn>(fn), tenant_id.value());
}

}  // namespace mooncake::rpc_detail
