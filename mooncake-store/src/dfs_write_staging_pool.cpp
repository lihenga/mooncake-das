#include "dfs_write_staging_pool.h"

#include <glog/logging.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mooncake {

namespace {
constexpr size_t kDfsWriteStagingAlignment = 4096;
}

DfsWriteStagingPool::DfsWriteStagingPool(size_t arena_size,
                                         const std::string& protocol)
    : arena_size_(arena_size), backing_is_pinned_(false) {
    if (arena_size == 0) {
        throw std::invalid_argument("DFS write staging arena is disabled");
    }

    auto backing = PinnedBufferPool::AllocatePinned(arena_size);
    if (!backing.data) {
        LOG(WARNING) << "Pinned DFS write staging allocation failed, size="
                     << arena_size << "; falling back to pageable backing";
        backing = PinnedBufferPool::Buffer::Pageable(arena_size);
    }
    if (!backing.data) {
        throw std::runtime_error("Failed to allocate DFS write staging arena");
    }
    backing_is_pinned_ = backing.pinned_host.addr != nullptr;
    backing_ = std::make_shared<PinnedBufferPool::Buffer>(std::move(backing));
    allocator_ = ClientBufferAllocator::create(
        backing_->data, backing_->capacity, protocol, backing_->device_data,
        backing_);
}

DfsWriteStagingPool::Slot::~Slot() {
    allocation.reset();
    if (auto pool = owner.lock()) pool->NotifyReleased();
}

char* DfsWriteStagingPool::Slot::data() const {
    return allocation ? static_cast<char*>(allocation->ptr()) : nullptr;
}

size_t DfsWriteStagingPool::Slot::capacity() const {
    return allocation ? allocation->size() : 0;
}

DfsWriteStagingPool::~DfsWriteStagingPool() { Shutdown(); }

std::shared_ptr<DfsWriteStagingPool::Slot> DfsWriteStagingPool::Acquire(
    size_t size, std::chrono::milliseconds timeout, bool* timed_out) {
    if (timed_out) *timed_out = false;
    if (size == 0 || size > arena_size_) return nullptr;

    const auto deadline = timeout == std::chrono::milliseconds::max()
                              ? std::chrono::steady_clock::time_point::max()
                              : std::chrono::steady_clock::now() + timeout;
    for (;;) {
        uint64_t generation;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (shutting_down_) return nullptr;
            generation = release_generation_;
        }

        auto allocation =
            allocator_->allocate_aligned(size, kDfsWriteStagingAlignment);
        if (allocation) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (shutting_down_) {
                allocation.reset();
                return nullptr;
            }
            auto slot = std::make_shared<Slot>();
            slot->allocation.emplace(std::move(*allocation));
            slot->owner = shared_from_this();
            return slot;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        if (shutting_down_) return nullptr;
        if (release_generation_ != generation) continue;
        if (timeout == std::chrono::milliseconds::zero()) {
            if (timed_out) *timed_out = true;
            return nullptr;
        }
        const auto released = [&] {
            return shutting_down_ || release_generation_ != generation;
        };
        if (deadline == std::chrono::steady_clock::time_point::max()) {
            cv_.wait(lock, released);
        } else if (!cv_.wait_until(lock, deadline, released)) {
            if (timed_out) *timed_out = true;
            return nullptr;
        }
    }
}

void DfsWriteStagingPool::Release(std::shared_ptr<Slot> slot) {
    slot.reset();
}

void DfsWriteStagingPool::NotifyReleased() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++release_generation_;
    }
    // Waiters may request different sizes. Wake all of them so a smaller
    // request is not stranded behind a larger request selected by notify_one.
    cv_.notify_all();
}

void DfsWriteStagingPool::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutting_down_) return;
    shutting_down_ = true;
    cv_.notify_all();
}

size_t DfsWriteStagingPool::available() const {
    auto report = allocator_->storageReport();
    return report.totalFreeSpace;
}

}  // namespace mooncake
