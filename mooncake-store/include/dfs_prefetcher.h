#pragma once

// DFS replica prefetcher.
//
// sglang calls batch_is_exist() ~3s before it fetches the same keys via
// batch_get_session_start()/batch_get_into_multi_buffer_ranges(). DFS-resident
// objects are otherwise read synchronously from the distributed filesystem on
// every get. This component uses the exist->get window to asynchronously pull
// DFS-only objects into client DRAM so the later get scatters from a local
// buffer instead of hitting DFS.
//
// Only keys whose best replica (SelectBestReplica policy) is a DFS replica
// are prefetched; keys with memory/NOF/local-disk replicas are left alone.
//
// Trigger: RealClient::batchIsExist_internal() -> NotifyExistTrue().
// Consume: RealClient::process_session_disk_dfs_reads() -> TryConsume().

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "client_buffer.h"
#include "client_service.h"
#include "replica.h"
#include "replica_selection.h"
#include "thread_pool.h"
#include "types.h"

namespace mooncake {

struct DfsPrefetchMetric;

struct DfsPrefetchConfig {
    // Total bytes reserved by pending/ready prefetch buffers. New prefetches
    // are skipped (never queued) once exceeded, so the shared
    // ClientBufferAllocator arena keeps headroom for on-demand reads.
    uint64_t max_bytes = 4ull << 30;  // MC_STORE_DFS_PREFETCH_MAX_BYTES
    // Time an unconsumed entry stays consumable before GC frees its buffer.
    uint64_t ttl_ms = 10000;  // MC_STORE_DFS_PREFETCH_TTL_MS
    // Driver threads issuing synchronous Client::BatchGet calls.
    uint32_t io_threads = 8;  // MC_STORE_DFS_PREFETCH_IO_THREADS
    // How long a consumer waits for an in-flight prefetch before falling
    // back to a direct DFS read.
    uint64_t wait_timeout_ms = 2000;  // MC_STORE_DFS_PREFETCH_WAIT_TIMEOUT_MS
    // Minimum interval before a failed key may be prefetched again.
    uint64_t retry_backoff_ms = 500;  // MC_STORE_DFS_PREFETCH_RETRY_BACKOFF_MS
    // Max keys per BatchQuery / BatchGet call; larger inputs are chunked.
    size_t max_batch_keys = 256;  // MC_STORE_DFS_PREFETCH_MAX_BATCH_KEYS

    static DfsPrefetchConfig FromEnv();
};

// Cached read of MC_STORE_ENABLE_DFS_PREFETCH (default off).
bool dfs_prefetch_enabled();

enum class PrefetchState { READING, READY, FAILED, CONSUMED };

struct PrefetchEntry {
    std::unique_ptr<BufferHandle> buffer;
    // Capacity charged against DfsPrefetchConfig::max_bytes while the entry
    // is tracked. Reset to 0 when the charge is released, so inflight
    // accounting never double-counts a release.
    uint64_t total_size = 0;
    Replica::Descriptor replica;
    std::chrono::steady_clock::time_point lease_timeout;
    std::atomic<PrefetchState> state{PrefetchState::READING};
    ErrorCode last_error = ErrorCode::OK;
    // Set when the read completes (READY or FAILED); consumers wait on it.
    std::shared_ptr<std::promise<void>> done;
    std::shared_future<void> done_future;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point failed_at;
};

class DfsPrefetcher {
   public:
    using BatchQueryFn =
        std::function<std::vector<tl::expected<QueryResult, ErrorCode>>(
            const std::vector<std::string> &)>;
    using BatchGetFn = std::function<std::vector<tl::expected<void, ErrorCode>>(
        const std::vector<std::string> &, const std::vector<QueryResult> &,
        std::unordered_map<std::string, std::vector<Slice>> &)>;
    using LocalEndpointsFn =
        std::function<std::unordered_set<std::string>()>;
    using AllocateFn = std::function<std::optional<BufferHandle>(size_t)>;

    DfsPrefetcher(DfsPrefetchConfig config, BatchQueryFn query_fn,
                  BatchGetFn get_fn, LocalEndpointsFn endpoints_fn,
                  AllocateFn alloc_fn, DfsPrefetchMetric *metric = nullptr);
    ~DfsPrefetcher();

    DfsPrefetcher(const DfsPrefetcher &) = delete;
    DfsPrefetcher &operator=(const DfsPrefetcher &) = delete;

    // Non-blocking: enqueue keys reported as existing by batchIsExist.
    void NotifyExistTrue(std::vector<std::string> keys);

    // Consume a prefetched object for `key`.
    // - out_found=false: no entry exists (caller takes the normal read path).
    // - out_found=true, returns non-null: READY entry consumed; ownership of
    //   the buffer transfers to the caller.
    // - out_found=true, returns nullptr: read failed or wait timed out
    //   (caller falls back to the normal read path).
    std::shared_ptr<BufferHandle> TryConsume(const std::string &key,
                                             bool &out_found);

    void Shutdown();

   private:
    void CoordinatorLoop();
    void ProcessKeys(const std::vector<std::string> &keys);
    void RunIoBatch(std::vector<std::pair<std::string,
                                          std::shared_ptr<PrefetchEntry>>>
                        batch);
    void GcExpired();
    std::unordered_map<std::string,
                       std::shared_ptr<PrefetchEntry>>::iterator
    EraseEntryLocked(
        std::unordered_map<std::string,
                           std::shared_ptr<PrefetchEntry>>::iterator it);
    void TrackInflight(int64_t delta);

    const DfsPrefetchConfig config_;
    BatchQueryFn query_fn_;
    BatchGetFn get_fn_;
    LocalEndpointsFn endpoints_fn_;
    AllocateFn alloc_fn_;
    DfsPrefetchMetric *metric_;  // not owned; may be nullptr

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::string> pending_keys_;
    bool stop_ = false;
    std::thread coordinator_thread_;

    std::mutex entries_mutex_;
    std::unordered_map<std::string, std::shared_ptr<PrefetchEntry>> entries_;
    std::atomic<int64_t> inflight_bytes_{0};

    std::unique_ptr<ThreadPool> io_pool_;
};

}  // namespace mooncake
