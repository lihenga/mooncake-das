#pragma once

#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "device/accelerator_registry.h"
#include "pinned_host_buffer.h"

namespace mooncake {

/**
 * PinnedBufferPool: Thread-safe pool of reusable pinned host memory buffers.
 *
 * Platform-specific pinned host allocation is delegated to AcceleratorDevice.
 *
 * Acquire() falls back to pageable memory if pinned allocation fails, while
 * AcquirePinned() is suitable for asynchronous DMA and never falls back.
 *
 * Cached buffers are grouped by allocation size and bounded by total bytes.
 * This permits a large batch arena to be reused without allowing cached pinned
 * memory to grow without limit.
 */
class PinnedBufferPool {
   public:
    static constexpr size_t kDefaultMaxCachedBytes = 256ULL * 1024 * 1024;

    struct Buffer {
        PinnedHostBuffer pinned_host;
        std::unique_ptr<char[]> pageable_host;
        char* data = nullptr;
        size_t capacity = 0;
        bool mapped = false;
        void* device_data = nullptr;

        Buffer() = default;
        explicit Buffer(PinnedHostBuffer pinned_host)
            : pinned_host(std::move(pinned_host)),
              data(static_cast<char*>(this->pinned_host.addr)),
              capacity(this->pinned_host.size),
              mapped(this->pinned_host.device_addr != nullptr),
              device_data(this->pinned_host.device_addr) {}

        static Buffer Pageable(size_t size) {
            Buffer buf;
            buf.pageable_host = std::make_unique<char[]>(size);
            buf.data = buf.pageable_host.get();
            buf.capacity = size;
            return buf;
        }

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept
            : pinned_host(std::move(other.pinned_host)),
              pageable_host(std::move(other.pageable_host)),
              data(other.data),
              capacity(other.capacity),
              mapped(other.mapped),
              device_data(other.device_data) {
            other.data = nullptr;
            other.capacity = 0;
            other.mapped = false;
            other.device_data = nullptr;
        }
        Buffer& operator=(Buffer&& other) noexcept {
            if (this != &other) {
                pinned_host = std::move(other.pinned_host);
                pageable_host = std::move(other.pageable_host);
                data = other.data;
                capacity = other.capacity;
                mapped = other.mapped;
                device_data = other.device_data;
                other.data = nullptr;
                other.capacity = 0;
                other.mapped = false;
                other.device_data = nullptr;
            }
            return *this;
        }
    };

    explicit PinnedBufferPool(size_t max_cached_bytes = kDefaultMaxCachedBytes)
        : max_cached_bytes_(max_cached_bytes) {}

    ~PinnedBufferPool() { Clear(); }

    Buffer Acquire(size_t size);

    // Acquire only pinned storage; unlike Acquire(), this never falls back to
    // pageable memory and is intended for asynchronous DMA sources.
    Buffer AcquirePinned(size_t size, bool* from_cache = nullptr,
                         bool mapped = false);

    // Never falls back to pageable memory. When mapped is true, allocation
    // succeeds only if the accelerator also provides a device-visible alias.
    static Buffer AllocatePinned(size_t size, bool mapped = false);

    void Release(Buffer buf);

    void Clear();

    size_t cached_bytes() const;

    static constexpr size_t kSmallAllocationLimit = 1ULL * 1024 * 1024;
    static constexpr size_t kSmallAllocationMinimum = 4ULL * 1024;
    static constexpr size_t kLargeAllocationAlignment = 2ULL * 1024 * 1024;

    static size_t SizeClass(size_t size);

    Buffer TakeCached(size_t capacity, bool pinned_only, bool mapped = false);

    static Buffer AllocPinnedOnly(size_t capacity, bool mapped = false);

    static Buffer AllocWithPageableFallback(size_t capacity);

   private:
    static Buffer AllocNew(size_t size);

    static void FreeBuffer(Buffer& buf);

    const size_t max_cached_bytes_;
    mutable std::mutex mutex_;
    size_t cached_bytes_ = 0;
    std::map<size_t, std::vector<Buffer>> pool_;
};

}  // namespace mooncake
