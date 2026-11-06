#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include "pinned_buffer_pool.h"

namespace mooncake {

class DfsWriteStagingPool {
   public:
    // Owns a fixed number of arenas. A slot remains unavailable until the
    // asynchronous DFS task releases its context.
    struct Slot {
        PinnedBufferPool::Buffer backing;
    };

    DfsWriteStagingPool(size_t arena_size, size_t arena_count);
    ~DfsWriteStagingPool();

    DfsWriteStagingPool(const DfsWriteStagingPool&) = delete;
    DfsWriteStagingPool& operator=(const DfsWriteStagingPool&) = delete;

    std::shared_ptr<Slot> Acquire(
        std::chrono::milliseconds timeout = std::chrono::seconds(10),
        bool* timed_out = nullptr);
    void Release(std::shared_ptr<Slot> slot);
    void Shutdown();

    size_t arena_size() const { return arena_size_; }
    size_t capacity() const;
    size_t available() const;

    // Sizing counters. `chunks_requested` counts every chunk submitted for a
    // slot, `slot_wait_timeouts` the ones that gave up; `in_use_peak` is the
    // high-water mark of simultaneously held slots, and
    // `pending_wait_peak` the most chunks ever waiting for a slot at once.
    // `max_chunks_per_batch` is the largest chunk count a single batch asked
    // for; arena_count must be >= that to avoid queueing.
    struct Stats {
        std::atomic<uint64_t> batches{0};
        std::atomic<uint64_t> chunks_requested{0};
        std::atomic<uint64_t> slot_wait_timeouts{0};
        // Chunk payload larger than arena_size_: such a chunk never fits.
        std::atomic<uint64_t> oversized_chunks{0};
        std::atomic<uint64_t> max_chunks_per_batch{0};
        std::atomic<uint64_t> max_chunk_bytes{0};
        std::atomic<uint64_t> in_use_peak{0};
        std::atomic<uint64_t> pending_wait_peak{0};
        std::atomic<uint64_t> total_slot_wait_us{0};
        std::atomic<int64_t> last_summary_ns{0};
    };
    const Stats& stats() const { return stats_; }
    // Records one scheduled batch: `chunks` is how many slots it needs and
    // `max_chunk_bytes` the largest chunk payload it produced. Logs a summary
    // at most every 30 s.
    void ObserveBatch(size_t chunks, size_t max_chunk_bytes);
    // Logs the current counters. Peak values are kept so the log is not a
    // sampling artifact.
    void LogSummary(const char* reason);

   private:
    const size_t arena_size_;
    std::vector<std::shared_ptr<Slot>> slots_;
    std::vector<std::shared_ptr<Slot>> free_slots_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool shutting_down_{false};
    // Chunks currently blocked in Acquire, guarded by mutex_.
    uint64_t waiting_ = 0;
    mutable Stats stats_;
};

}  // namespace mooncake
