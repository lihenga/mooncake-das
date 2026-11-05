// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dummy_client.h"
#include "real_client.h"
#include <async_simple/coro/SyncAwait.h>
#include <type_traits>

namespace mooncake {

template <auto ServiceMethod>
constexpr bool can_invoke_when_disconnected() {
    using Method = std::remove_reference_t<decltype(ServiceMethod)>;
    return std::is_same_v<
               Method, std::remove_reference_t<decltype(&RealClient::ping)>> ||
           std::is_same_v<Method,
                          std::remove_reference_t<
                              decltype(&RealClient::service_ready_internal)>> ||
           std::is_same_v<Method,
                          std::remove_reference_t<
                              decltype(&RealClient::is_shm_mapped_internal)>> ||
           std::is_same_v<Method,
                          std::remove_reference_t<
                              decltype(&RealClient::ascend_shm_internal)>> ||
           std::is_same_v<Method,
                          std::remove_reference_t<
                              decltype(&RealClient::ascend_ipc_shm_internal)>>;
}

template <auto ServiceMethod, typename ReturnType, typename... Args>
tl::expected<ReturnType, ErrorCode> DummyClient::invoke_rpc(Args&&... args) {
    RpcDrainGuard::ScopedCall inflight(rpc_drain_);
    if (!inflight.ok()) return tl::make_unexpected(ErrorCode::RPC_FAIL);
    auto pool = client_accessor_.GetClientPool();

    if constexpr (!can_invoke_when_disconnected<ServiceMethod>()) {
        if (!connected_.load()) {
            LOG(ERROR) << "Dummy Client not connected";
            return tl::make_unexpected(ErrorCode::RPC_FAIL);
        }
    }

    return async_simple::coro::syncAwait(
        [&]() -> async_simple::coro::Lazy<tl::expected<ReturnType, ErrorCode>> {
            auto ret = co_await pool->send_request(
                [&](coro_io::client_reuse_hint,
                    coro_rpc::coro_rpc_client& client) {
                    return client.send_request<ServiceMethod>(
                        std::forward<Args>(args)...);
                });
            if (!ret.has_value()) {
                LOG(ERROR) << "Dummy Client not available";
                co_return tl::make_unexpected(ErrorCode::RPC_FAIL);
            }
            auto result = co_await std::move(ret.value());
            if (!result) {
                LOG(ERROR) << "RPC call failed: " << result.error().msg;
                co_return tl::make_unexpected(ErrorCode::RPC_FAIL);
            }
            co_return result->result();
        }());
}

}  // namespace mooncake
