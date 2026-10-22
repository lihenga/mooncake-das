#pragma once

#include <condition_variable>
#include <cstddef>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <cstdint>

#include "client_buffer.h"
#include "pinned_buffer_pool.h"

namespace mooncake {

class DfsWriteStagingPool
    : public std::enable_shared_from_this<DfsWriteStagingPool> {
   public:
    // Owns one fixed-capacity arena. Each slot represents one sub-allocation
    // and remains alive until the asynchronous DFS task releases its context.
    struct Slot {
        std::optional<BufferHandle> allocation;
        std::weak_ptr<DfsWriteStagingPool> owner;

        ~Slot();
        char* data() const;
        size_t capacity() const;
    };

    DfsWriteStagingPool(size_t arena_size, const std::string& protocol);
    ~DfsWriteStagingPool();

    DfsWriteStagingPool(const DfsWriteStagingPool&) = delete;
    DfsWriteStagingPool& operator=(const DfsWriteStagingPool&) = delete;

    std::shared_ptr<Slot> Acquire(
        size_t size,
        std::chrono::milliseconds timeout = std::chrono::seconds(10),
        bool* timed_out = nullptr);
    void Release(std::shared_ptr<Slot> slot);
    void Shutdown();

    size_t arena_size() const { return arena_size_; }
    size_t capacity() const { return arena_size_; }
    size_t available() const;
    bool backing_is_pinned() const { return backing_is_pinned_; }

   private:
    void NotifyReleased();

    const size_t arena_size_;
    bool backing_is_pinned_ = false;
    std::shared_ptr<PinnedBufferPool::Buffer> backing_;
    std::shared_ptr<ClientBufferAllocator> allocator_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    uint64_t release_generation_{0};
    bool shutting_down_{false};
};

}  // namespace mooncake
