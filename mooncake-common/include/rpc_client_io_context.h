#pragma once

#include "das/shared_rpc_pool.h"
#include "das/rpc_drain_guard.h"
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
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

class Environ;

// RPC client I/O thread counts. MC_RPC_CLIENT_IO_THREADS sets the shared
// value; MC_STORE_RPC_CLIENT_IO_THREADS and MC_TE_RPC_CLIENT_IO_THREADS
// override it per component. Invalid or non-positive values fall back.
struct RpcClientIoThreadsConfig {
    uint32_t common{0};
    uint32_t store{0};
    uint32_t transfer_engine{0};

    static RpcClientIoThreadsConfig FromEnvironment(const Environ& env);
    // Resolved from the process environment once, on first use.
    static const RpcClientIoThreadsConfig& Process();
};

std::shared_ptr<coro_io::io_context_pool> CreateRpcClientIoContextPool(
    uint32_t thread_count);

template <typename PoolTag>
coro_io::io_context_pool& GetRpcClientIoContextPool(uint32_t thread_count) {
    static const auto io_pool = CreateRpcClientIoContextPool(thread_count);
    return *io_pool;
}

namespace detail {

// ClientRequester's offload pool collection has the same lifetime hazard: its
// pools host background reconnect coroutines, so they must outlive the owning
// RealClient. Collections are few in practice (one per RealClient), so they
// are parked here for the process lifetime instead of being freed at teardown.
inline void KeepClientPoolsAlive(
    std::shared_ptr<coro_io::client_pools<coro_rpc::coro_rpc_client>> pools) {
    static std::mutex keep_mutex;
    static std::vector<
        std::shared_ptr<coro_io::client_pools<coro_rpc::coro_rpc_client>>>
        keep;
    std::lock_guard<std::mutex> lock(keep_mutex);
    keep.push_back(std::move(pools));
}

}  // namespace detail

/**
 * A replaceable client pool for callers that communicate with one target at a
 * time. Pools live in the process-wide registry above; an address switch only
 * re-points this holder, it never frees a pool.
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
            client_pool_ = SharedPoolRegistry(address);
            address_ = address;
        }
        return client_pool_;
    }

    std::shared_ptr<ClientPool> GetClientPool() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return client_pool_;
    }

   private:
    std::shared_ptr<ClientPool> SharedPoolRegistry(std::string_view address) {
        return detail::SharedPoolRegistryImpl(address, config_,
                                              io_context_pool_);
    }

    mutable std::shared_mutex mutex_;
    coro_io::io_context_pool& io_context_pool_;
    PoolConfig config_;
    std::string address_;
    std::shared_ptr<ClientPool> client_pool_;
};

}  // namespace mooncake
