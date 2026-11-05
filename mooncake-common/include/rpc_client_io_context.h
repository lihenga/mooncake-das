#pragma once

#include <atomic>
#include "das/shared_rpc_pool.h"
#include "das/rpc_drain_guard.h"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <ylt/coro_io/client_pool.hpp>
#include <ylt/coro_io/io_context_pool.hpp>
#include <ylt/coro_rpc/coro_rpc_client.hpp>

namespace mooncake {

std::shared_ptr<coro_io::io_context_pool> CreateRpcClientIoContextPool(
    uint32_t thread_count);

template <typename PoolTag>
coro_io::io_context_pool& GetRpcClientIoContextPool(uint32_t thread_count) {
    static const auto io_pool = CreateRpcClientIoContextPool(thread_count);
    return *io_pool;
}

/**
 * A replaceable client pool. Pools are retained process-wide because ylt's
 * background reconnect coroutines may outlive the active holder.
 */
class RpcClientPool {
   public:
    using ClientPool = coro_io::client_pool<coro_rpc::coro_rpc_client>;
    using PoolConfig = ClientPool::pool_config;

    explicit RpcClientPool(coro_io::io_context_pool& io_context_pool,
                           PoolConfig config = {})
        : io_context_pool_(io_context_pool), config_(std::move(config)) {
        config_.host_alive_detect_duration = std::chrono::seconds(0);
    }

    std::shared_ptr<ClientPool> GetOrCreateClientPool(
        std::string_view address) {
        std::lock_guard<std::shared_mutex> lock(mutex_);
        if (!client_pool_ || address_ != address) {
            client_pool_ = detail::SharedPoolRegistryImpl(
                address, config_, io_context_pool_);
            address_ = address;
        }
        return client_pool_;
    }

    std::shared_ptr<ClientPool> GetClientPool() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return client_pool_;
    }

   private:
    mutable std::shared_mutex mutex_;
    coro_io::io_context_pool& io_context_pool_;
    PoolConfig config_;
    std::string address_;
    std::shared_ptr<ClientPool> client_pool_;
};

}  // namespace mooncake
