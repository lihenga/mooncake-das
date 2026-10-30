#include "client_buffer.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>
#include <sys/mman.h>  // For shm_open, mmap, munmap
#include <sys/stat.h>  // For S_IRUSR, S_IWUSR
#include <fcntl.h>     // For O_CREAT, O_RDWR
#include <unistd.h>    // For ftruncate, close, shm_unlink

#include "utils.h"

#if defined(USE_SUNRISE)
#include "sunrise_allocator.h"
#endif

namespace mooncake {

std::shared_ptr<ClientBufferAllocator> ClientBufferAllocator::create(
    size_t size, const std::string& protocol, bool use_hugepage,
    bool use_spdk_dma) {
    return std::shared_ptr<ClientBufferAllocator>(
        new ClientBufferAllocator(size, protocol, use_hugepage, use_spdk_dma));
}

std::shared_ptr<ClientBufferAllocator> ClientBufferAllocator::create(
    void* addr, size_t size, const std::string& protocol, void* device_addr,
    std::shared_ptr<void> backing_owner,
    std::shared_ptr<DfsBufferUsageTracker> usage_tracker,
    DfsBufferUsageKind usage_kind) {
    auto allocator = std::shared_ptr<ClientBufferAllocator>(
        new ClientBufferAllocator(addr, size, protocol, device_addr));
    allocator->backing_owner_ = std::move(backing_owner);
    allocator->SetUsageTracker(std::move(usage_tracker), usage_kind);
    return allocator;
}

ClientBufferAllocator::ClientBufferAllocator(size_t size,
                                             const std::string& protocol,
                                             bool use_hugepage,
                                             bool use_spdk_dma)
    : buffer_size_(size),
      use_hugepage_(use_hugepage),
      protocol(protocol),
      use_spdk_dma_(use_spdk_dma) {
    if (size == 0) {
        buffer_ = nullptr;
        allocator_ = nullptr;
        return;
    }
    // Align to 64 bytes(cache line size) for better cache performance
    constexpr size_t alignment = 64;
    if (use_hugepage_) {
        buffer_ = allocate_buffer_mmap_memory(size, alignment);
    } else {
#if defined(USE_SUNRISE)
        if (protocol == "sunrise_link") {
            buffer_ = sunrise_allocate_memory(size, alignment, false);
        } else
#endif
        {
            buffer_ = allocate_buffer_allocator_memory(
                size, protocol, alignment, use_spdk_dma_);
        }
    }
    if (!buffer_) {
        throw std::bad_alloc();
    }

    allocator_ = mooncake::offset_allocator::OffsetAllocator::create(
        reinterpret_cast<uint64_t>(buffer_), size);
}

ClientBufferAllocator::ClientBufferAllocator(void* addr, size_t size,
                                             const std::string& protocol,
                                             void* device_addr)
    : buffer_(addr),
      buffer_size_(size),
      device_buffer_(device_addr),
      protocol(protocol),
      is_external_memory_(true) {
    allocator_ = mooncake::offset_allocator::OffsetAllocator::create(
        reinterpret_cast<uint64_t>(buffer_), size);
}

ClientBufferAllocator::~ClientBufferAllocator() {
    // Free the aligned allocated memory or unmap shared memory
    if (!is_external_memory_ && buffer_) {
        if (use_hugepage_) {
            free_buffer_mmap_memory(buffer_, buffer_size_);
        } else {
            free_memory(protocol, buffer_, use_spdk_dma_);
        }
    }
}

offset_allocator::OffsetAllocStorageReport
ClientBufferAllocator::storageReport() const {
    if (allocator_ == nullptr) {
        return {0, 0};
    }
    return allocator_->storageReport();
}

void ClientBufferAllocator::SetUsageTracker(
    std::shared_ptr<DfsBufferUsageTracker> tracker,
    DfsBufferUsageKind kind) {
    usage_tracker_ = std::move(tracker);
    usage_kind_ = kind;
}

void ClientBufferAllocator::NotifyAllocationReleased() {
    {
        std::lock_guard<std::mutex> lock(allocation_wait_mutex_);
        ++allocation_generation_;
    }
    allocation_wait_cv_.notify_all();
}

size_t ClientBufferAllocator::allocation_size_for_request(size_t size) const {
    if (!allocator_) return 0;
    return allocator_->normalizedAllocationSize(size);
}

std::optional<BufferHandle> ClientBufferAllocator::allocate(size_t size) {
    if (allocator_ == nullptr) {
        return std::nullopt;
    }
    auto handle = allocator_->allocate(size);
    if (!handle) {
        return std::nullopt;
    }

    return std::make_optional<BufferHandle>(
        shared_from_this(), std::move(*handle), usage_tracker_, usage_kind_);
}

std::optional<BufferHandle> ClientBufferAllocator::allocate_aligned_wait(
    size_t size, size_t alignment, DfsBufferUsageKind kind,
    std::chrono::milliseconds timeout, bool* timed_out) {
    if (timed_out) *timed_out = false;
    const bool finite_timeout = timeout != std::chrono::milliseconds::max();
    const auto deadline = finite_timeout
                              ? std::chrono::steady_clock::now() + timeout
                              : std::chrono::steady_clock::time_point::max();
    for (;;) {
        std::unique_lock<std::mutex> lock(allocation_wait_mutex_);
        const auto generation = allocation_generation_;
        lock.unlock();
        if (auto allocation = allocate_aligned(size, alignment, kind)) {
            return allocation;
        }
        lock.lock();
        if (timeout == std::chrono::milliseconds::zero()) {
            if (timed_out) *timed_out = true;
            break;
        }
        const auto ready = [this, generation] {
            return allocation_generation_ != generation;
        };
        if (timeout == std::chrono::milliseconds::max()) {
            allocation_wait_cv_.wait(lock, ready);
        } else if (allocation_wait_cv_.wait_until(lock, deadline, ready) ==
                   false) {
            if (timed_out) *timed_out = true;
            break;
        }
    }
    return std::nullopt;
}

std::optional<BufferHandle> ClientBufferAllocator::allocate_aligned(
    size_t size, size_t alignment) {
    return allocate_aligned(size, alignment, usage_kind_);
}

std::optional<BufferHandle> ClientBufferAllocator::allocate_aligned(
    size_t size, size_t alignment, DfsBufferUsageKind kind) {
    if (size == 0 || alignment == 0) {
        return std::nullopt;
    }

    const auto allocate_for_kind = [this, kind](size_t allocation_size)
        -> std::optional<BufferHandle> {
        if (allocator_ == nullptr) return std::nullopt;
        auto handle = allocator_->allocate(allocation_size);
        if (!handle) return std::nullopt;
        return BufferHandle(shared_from_this(), std::move(*handle),
                            usage_tracker_, kind);
    };

    auto allocation = allocate_for_kind(size);
    if (!allocation) return std::nullopt;

    const auto address = reinterpret_cast<uintptr_t>(allocation->ptr());
    if (address % alignment == 0) {
        return allocation;
    }

    allocation.reset();
    if (size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return std::nullopt;
    }
    allocation = allocate_for_kind(size + alignment - 1);
    if (!allocation) return std::nullopt;

    auto owner = std::make_shared<BufferHandle>(std::move(allocation.value()));
    const auto owner_allocation_size = owner->allocation_size();
    auto* host_base = static_cast<char*>(owner->ptr());
    const size_t remainder = reinterpret_cast<uintptr_t>(host_base) % alignment;
    const size_t offset = remainder == 0 ? 0 : alignment - remainder;
    void* device_ptr = owner->device_ptr();
    if (device_ptr) {
        device_ptr = static_cast<char*>(device_ptr) + offset;
    }
    auto allocator = shared_from_this();
    return BufferHandle(
        host_base + offset, size,
        [owner = std::move(owner), allocator]() mutable {
            owner.reset();
            allocator->NotifyAllocationReleased();
        },
        device_ptr, owner_allocation_size);
}

BufferHandle::BufferHandle(
    std::shared_ptr<ClientBufferAllocator> allocator,
    mooncake::offset_allocator::OffsetAllocationHandle handle,
    std::shared_ptr<DfsBufferUsageTracker> usage_tracker,
    DfsBufferUsageKind usage_kind)
    : allocator_(std::move(allocator)),
      handle_(std::move(handle)),
      usage_(usage_tracker
                 ? std::make_unique<DfsBufferAllocationUsage>(
                       std::move(usage_tracker), usage_kind,
                       handle_.allocationSize())
                 : nullptr) {}

BufferHandle::BufferHandle(BufferHandle&& other) noexcept
    : allocator_(std::move(other.allocator_)),
      handle_(std::move(other.handle_)),
      usage_(std::move(other.usage_)),
      view_ptr_(other.view_ptr_),
      view_size_(other.view_size_),
      view_allocation_size_(other.view_allocation_size_),
      release_fn_(std::move(other.release_fn_)),
      view_device_ptr_(other.view_device_ptr_) {
    other.view_ptr_ = nullptr;
    other.view_size_ = 0;
    other.view_allocation_size_ = 0;
    other.view_device_ptr_ = nullptr;
}

BufferHandle& BufferHandle::operator=(BufferHandle&& other) noexcept {
    if (this == &other) return *this;
    auto old_allocator = allocator_;
    if (release_fn_) {
        release_fn_();
        release_fn_ = nullptr;
    }
    usage_.reset();
    handle_ = std::move(other.handle_);
    usage_ = std::move(other.usage_);
    if (old_allocator) old_allocator->NotifyAllocationReleased();
    allocator_ = std::move(other.allocator_);
    view_ptr_ = other.view_ptr_;
    view_size_ = other.view_size_;
    view_allocation_size_ = other.view_allocation_size_;
    release_fn_ = std::move(other.release_fn_);
    view_device_ptr_ = other.view_device_ptr_;
    other.view_ptr_ = nullptr;
    other.view_size_ = 0;
    other.view_allocation_size_ = 0;
    other.view_device_ptr_ = nullptr;
    return *this;
}

BufferHandle::BufferHandle(void* ptr, size_t size,
                           std::function<void()> release_fn,
                           void* device_ptr, size_t allocation_size)
    : view_ptr_(ptr),
      view_size_(size),
      view_allocation_size_(allocation_size == 0 ? size : allocation_size),
      release_fn_(std::move(release_fn)),
      view_device_ptr_(device_ptr) {}

BufferHandle::~BufferHandle() {
    if (release_fn_) {
        release_fn_();
        release_fn_ = nullptr;
    }
    if (allocator_) {
        usage_.reset();
        handle_ = offset_allocator::OffsetAllocationHandle();
        allocator_->NotifyAllocationReleased();
    }
    // Otherwise OffsetAllocationHandle destructor handles deallocation
}

void* BufferHandle::ptr() const {
    return view_ptr_ ? view_ptr_ : handle_.ptr();
}

size_t BufferHandle::size() const {
    return view_ptr_ ? view_size_ : handle_.size();
}

size_t BufferHandle::allocation_size() const {
    return view_ptr_ ? view_allocation_size_ : handle_.allocationSize();
}

void* BufferHandle::device_ptr() const {
    if (view_ptr_) return view_device_ptr_;
    if (!allocator_ || !allocator_->getDeviceBase() || !handle_.ptr()) {
        return nullptr;
    }
    const auto offset = static_cast<char*>(handle_.ptr()) -
                        static_cast<char*>(allocator_->getBase());
    return static_cast<char*>(allocator_->getDeviceBase()) + offset;
}

// Utility functions for buffer and slice management
std::vector<Slice> split_into_slices(BufferHandle& handle) {
    auto base = static_cast<uint8_t*>(handle.ptr());
    auto length = handle.size();
    return split_into_slices(base, length);
}

std::vector<Slice> split_into_slices(void* buffer, size_t length) {
    std::vector<Slice> slices;
    auto base = static_cast<uint8_t*>(buffer);
    size_t offset = 0;

    while (offset < length) {
        size_t chunk_size = std::min(length - offset, kMaxSliceSize);
        slices.push_back({base + offset, chunk_size});
        offset += chunk_size;
    }
    return slices;
}

uint64_t calculate_total_size(const Replica::Descriptor& replica) {
    uint64_t total_length = 0;
    if (replica.is_disk_replica()) {
        auto& disk_descriptor = replica.get_disk_descriptor();
        total_length = disk_descriptor.object_size;
    } else if (replica.is_local_disk_replica()) {
        total_length = replica.get_local_disk_descriptor().object_size;
    } else if (replica.is_dfs_replica()) {
        total_length = replica.get_dfs_descriptor().object_size;
    } else if (replica.is_nof_replica()) {
        total_length = replica.get_nof_descriptor().buffer_descriptor.size_;
    } else {
        total_length = replica.get_memory_descriptor().buffer_descriptor.size_;
    }
    return total_length;
}

int allocateSlices(std::vector<Slice>& slices,
                   const Replica::Descriptor& replica, void* buffer_ptr) {
    if (replica.is_disk_replica()) {
        // For disk-based replica, split into slices based on file size
        uint64_t offset = 0;
        uint64_t total_length = replica.get_disk_descriptor().object_size;
        while (offset < total_length) {
            auto chunk_size = std::min(total_length - offset, kMaxSliceSize);
            void* chunk_ptr = static_cast<char*>(buffer_ptr) + offset;
            slices.emplace_back(Slice{chunk_ptr, chunk_size});
            offset += chunk_size;
        }
    } else if (replica.is_local_disk_replica()) {
        slices.emplace_back(
            Slice{buffer_ptr, replica.get_local_disk_descriptor().object_size});
    } else if (replica.is_dfs_replica()) {
        // Use aligned_size (value + padding) so the slice length is a multiple
        // of the DFS block alignment, letting O_DIRECT go through the zero-bounce
        // preadv path. The trailing padding is read but not scattered.
        const auto& desc = replica.get_dfs_descriptor();
        const uint64_t slice_size =
            desc.aligned_size > 0 ? desc.aligned_size : desc.object_size;
        slices.emplace_back(Slice{buffer_ptr, slice_size});
    } else if (replica.is_nof_replica()) {
        auto& handle = replica.get_nof_descriptor().buffer_descriptor;
        void* chunk_ptr = buffer_ptr;
        slices.emplace_back(Slice{chunk_ptr, handle.size_});
    } else {
        // For memory-based replica, split into slices based on buffer
        // descriptors
        auto& handle = replica.get_memory_descriptor().buffer_descriptor;
        void* chunk_ptr = buffer_ptr;
        slices.emplace_back(Slice{chunk_ptr, handle.size_});
    }
    return 0;
}

}  // namespace mooncake
