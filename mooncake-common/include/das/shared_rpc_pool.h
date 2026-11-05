// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <ylt/coro_io/client_pool.hpp>
#include <ylt/coro_rpc/coro_rpc_client.hpp>

namespace mooncake {

namespace detail {

using MasterClientPool = coro_io::client_pool<coro_rpc::coro_rpc_client>;

inline std::shared_ptr<MasterClientPool> SharedPoolRegistryImpl(
    std::string_view address, MasterClientPool::pool_config config,
    coro_io::io_context_pool& io_context_pool) {
    static std::mutex registry_mutex;
    static std::unordered_map<std::string, std::shared_ptr<MasterClientPool>>
        registry;
    std::lock_guard<std::mutex> lock(registry_mutex);

    std::string key(address);
    auto append = [&key](auto value) {
        key.push_back('|');
        key.append(std::to_string(value));
    };
    append(config.max_connection);
    append(config.connect_retry_count);
    append(config.reconnect_wait_time.count());
    append(config.client_config.connect_timeout_duration.count());
    append(config.client_config.request_timeout_duration.count());
    append(static_cast<size_t>(config.client_config.socket_config.index()));

    auto& pool = registry[key];
    if (!pool) {
        pool = MasterClientPool::create(std::string(address), std::move(config),
                                        io_context_pool);
    }
    return pool;
}

}  // namespace detail

}  // namespace mooncake
