#include "dfs_write_staging_pool.h"

#include <stdexcept>
#include <utility>

#include <glog/logging.h>

namespace mooncake {

DfsWriteStagingPool::DfsWriteStagingPool(size_t arena_size,
                                         size_t arena_count)
    : arena_size_(arena_size) {
    if (arena_size == 0 || arena_count == 0) {
        throw std::invalid_argument("DFS write staging arena is disabled");
    }

    slots_.reserve(arena_count);
    free_slots_.reserve(arena_count);
    for (size_t i = 0; i < arena_count; ++i) {
        auto backing = PinnedBufferPool::AllocatePinned(arena_size);
        if (!backing.data) {
            backing = PinnedBufferPool::Buffer::Pageable(arena_size);
        }
        if (!backing.data) {
            throw std::runtime_error(
                "Failed to allocate DFS write staging arena");
        }
        auto slot = std::make_shared<Slot>();
        slot->backing = std::move(backing);
        slots_.push_back(slot);
        free_slots_.push_back(std::move(slot));
    }
}

DfsWriteStagingPool::~DfsWriteStagingPool() {
    Shutdown();
    if (stats_.batches.load(std::memory_order_relaxed) > 0) {
        LogSummary("final");
    }
}

std::shared_ptr<DfsWriteStagingPool::Slot> DfsWriteStagingPool::Acquire(
    std::chrono::milliseconds timeout, bool* timed_out) {
    if (timed_out) *timed_out = false;
    const auto wait_start = std::chrono::steady_clock::now();
    stats_.chunks_requested.fetch_add(1, std::memory_order_relaxed);

    std::unique_lock<std::mutex> lock(mutex_);
    const auto ready = [this] { return shutting_down_ || !free_slots_.empty(); };
    bool acquired = ready();
    if (!acquired) {
        // How many chunks are queued behind the busy slots: this is the number
        // that says whether arena_count is the binding constraint.
        ++waiting_;
        uint64_t peak =
            stats_.pending_wait_peak.load(std::memory_order_relaxed);
        while (waiting_ > peak &&
               !stats_.pending_wait_peak.compare_exchange_weak(
                   peak, waiting_, std::memory_order_relaxed)) {
        }
        acquired = timeout == std::chrono::milliseconds::max()
                       ? (cv_.wait(lock, ready), true)
                       : cv_.wait_for(lock, timeout, ready);
        --waiting_;
    }
    const uint64_t wait_us =
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_start)
                .count());
    stats_.total_slot_wait_us.fetch_add(wait_us, std::memory_order_relaxed);
    if (!acquired) {
        if (timed_out) *timed_out = true;
        stats_.slot_wait_timeouts.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (shutting_down_) return nullptr;
    auto slot = std::move(free_slots_.back());
    free_slots_.pop_back();
    const uint64_t in_use = slots_.size() - free_slots_.size();
    uint64_t in_use_peak = stats_.in_use_peak.load(std::memory_order_relaxed);
    while (in_use > in_use_peak &&
           !stats_.in_use_peak.compare_exchange_weak(
               in_use_peak, in_use, std::memory_order_relaxed)) {
    }
    return slot;
}

void DfsWriteStagingPool::Release(std::shared_ptr<Slot> slot) {
    if (!slot) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutting_down_) return;
    free_slots_.push_back(std::move(slot));
    cv_.notify_one();
}

void DfsWriteStagingPool::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutting_down_) return;
    shutting_down_ = true;
    cv_.notify_all();
}

size_t DfsWriteStagingPool::capacity() const {
    return arena_size_ * slots_.size();
}

size_t DfsWriteStagingPool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_slots_.size();
}

void DfsWriteStagingPool::ObserveBatch(size_t chunks,
                                       size_t max_chunk_payload_bytes) {
    stats_.batches.fetch_add(1, std::memory_order_relaxed);
    if (max_chunk_payload_bytes > arena_size_) {
        // One batch packed more bytes into a chunk than a slot can hold; the
        // packer caps chunks at arena_size_ today, so this flags a mis-sized
        // MC_STORE_DFS_WRITE_CHUNK_SIZE_BYTES.
        stats_.oversized_chunks.fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t chunk_peak =
        stats_.max_chunks_per_batch.load(std::memory_order_relaxed);
    while (chunks > chunk_peak &&
           !stats_.max_chunks_per_batch.compare_exchange_weak(
               chunk_peak, chunks, std::memory_order_relaxed)) {
    }
    uint64_t bytes_peak =
        stats_.max_chunk_bytes.load(std::memory_order_relaxed);
    while (max_chunk_payload_bytes > bytes_peak &&
           !stats_.max_chunk_bytes.compare_exchange_weak(
               bytes_peak, max_chunk_payload_bytes,
               std::memory_order_relaxed)) {
    }

    constexpr int64_t kSummaryIntervalNs = 30LL * 1000 * 1000 * 1000;
    const int64_t now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    int64_t last_ns = stats_.last_summary_ns.load(std::memory_order_relaxed);
    if (now_ns - last_ns >= kSummaryIntervalNs &&
        stats_.last_summary_ns.compare_exchange_strong(
            last_ns, now_ns, std::memory_order_relaxed)) {
        LogSummary("summary");
    }
}

void DfsWriteStagingPool::LogSummary(const char* reason) {
    const uint64_t chunks =
        stats_.chunks_requested.load(std::memory_order_relaxed);
    LOG(INFO) << "DFS put staging pool " << reason
              << ": arena_count=" << slots_.size()
              << ", chunk_size_bytes=" << arena_size_
              << ", pool_bytes=" << capacity()
              << ", batches=" << stats_.batches.load(std::memory_order_relaxed)
              << ", chunks_requested=" << chunks
              << ", max_chunks_per_batch="
              << stats_.max_chunks_per_batch.load(std::memory_order_relaxed)
              << ", in_use_peak="
              << stats_.in_use_peak.load(std::memory_order_relaxed)
              << ", pending_wait_peak="
              << stats_.pending_wait_peak.load(std::memory_order_relaxed)
              << ", slot_wait_timeouts="
              << stats_.slot_wait_timeouts.load(std::memory_order_relaxed)
              << ", avg_slot_wait_us="
              << (chunks == 0 ? 0
                              : stats_.total_slot_wait_us.load(
                                    std::memory_order_relaxed) /
                                    chunks)
              << ", oversized_chunks="
              << stats_.oversized_chunks.load(std::memory_order_relaxed)
              << ", max_chunk_bytes="
              << stats_.max_chunk_bytes.load(std::memory_order_relaxed);
}

}  // namespace mooncake
