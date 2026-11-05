// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

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

offset_allocator::OffsetAllocStorageReport
ClientBufferAllocator::storageReport() const {
    if (allocator_ == nullptr) {
        return {0, 0};
    }
    return allocator_->storageReport();
}

void ClientBufferAllocator::SetUsageTracker(
    std::shared_ptr<DfsBufferUsageTracker> tracker, DfsBufferUsageKind kind) {
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

void ClientBufferAllocator::ShutdownAllocationWaiters() {
    {
        std::lock_guard<std::mutex> lock(allocation_wait_mutex_);
        allocation_wait_shutdown_ = true;
        ++allocation_generation_;
    }
    allocation_wait_cv_.notify_all();
}

size_t ClientBufferAllocator::allocation_size_for_request(size_t size) const {
    if (!allocator_) return 0;
    return allocator_->normalizedAllocationSize(size);
}

size_t ClientBufferAllocator::allocation_size_for_aligned_request(
    size_t size, size_t alignment) const {
    if (size == 0 || alignment == 0 ||
        size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return 0;
    }
    return allocation_size_for_request(size + alignment - 1);
}

std::optional<BufferHandle> ClientBufferAllocator::allocate_aligned_wait(
    size_t size, size_t alignment, DfsBufferUsageKind kind,
    std::chrono::milliseconds timeout, bool* timed_out,
    bool* request_exceeds_limit_out) {
    if (timed_out) *timed_out = false;
    if (request_exceeds_limit_out) *request_exceeds_limit_out = false;
    const bool finite_timeout = timeout != std::chrono::milliseconds::max();
    const auto deadline = finite_timeout
                              ? std::chrono::steady_clock::now() + timeout
                              : std::chrono::steady_clock::time_point::max();
    for (;;) {
        std::unique_lock<std::mutex> lock(allocation_wait_mutex_);
        if (allocation_wait_shutdown_) return std::nullopt;
        const auto generation = allocation_generation_;
        lock.unlock();
        std::optional<BufferHandle> allocation;
        // Only the attempt that uses the caller's real size can prove the
        // request itself is too large. The padded attempt asks for
        // size + alignment - 1, so exceeding a hard limit there does not mean
        // the real request could never fit once space is freed.
        bool request_exceeds_limit = false;
        const auto try_allocate =
            [&](size_t request_size, bool require_aligned,
                bool* exceeds_limit) -> std::optional<BufferHandle> {
            if (request_size == 0 || !allocator_) return std::nullopt;
            const auto estimated_size =
                allocator_->normalizedAllocationSize(request_size);
            if (estimated_size == 0) {
                *exceeds_limit = true;
                return std::nullopt;
            }
            if (usage_tracker_) {
                const auto admission =
                    usage_tracker_->TryReserve(kind, estimated_size);
                if (admission != DfsBufferAdmissionFailure::kNone) {
                    if (admission ==
                        DfsBufferAdmissionFailure::kRequestExceedsLimit) {
                        *exceeds_limit = true;
                    }
                    return std::nullopt;
                }
            }
            auto handle = allocator_->allocate(request_size);
            if (!handle) {
                if (usage_tracker_) {
                    usage_tracker_->CancelReservation(kind, estimated_size);
                }
                return std::nullopt;
            }
            if (require_aligned &&
                reinterpret_cast<uintptr_t>(handle->ptr()) % alignment != 0) {
                handle.reset();
                if (usage_tracker_) {
                    usage_tracker_->CancelReservation(kind, estimated_size);
                }
                return std::nullopt;
            }
            DCHECK_EQ(estimated_size, handle->allocationSize());
            return BufferHandle(shared_from_this(), std::move(*handle),
                                usage_tracker_, kind, estimated_size);
        };
        if (size != 0 && alignment != 0) {
            allocation = try_allocate(size, true, &request_exceeds_limit);
        }
        if (!allocation && size != 0 && alignment != 0 &&
            size <= std::numeric_limits<size_t>::max() - (alignment - 1)) {
            // A padded miss only means the aligned address could not be
            // satisfied. Its outcome must not be reported as the caller's
            // request exceeding the limit, so the limit flag it writes is
            // deliberately discarded.
            bool padded_limit_ignored = false;
            auto padded_allocation = try_allocate(size + alignment - 1, false,
                                                  &padded_limit_ignored);
            if (padded_allocation) {
                auto owner = std::make_shared<BufferHandle>(
                    std::move(*padded_allocation));
                const auto owner_allocation_size = owner->allocation_size();
                auto* host_base = static_cast<char*>(owner->ptr());
                const size_t remainder =
                    reinterpret_cast<uintptr_t>(host_base) % alignment;
                const size_t offset =
                    remainder == 0 ? 0 : alignment - remainder;
                void* device_ptr = owner->device_ptr();
                if (device_ptr) {
                    device_ptr = static_cast<char*>(device_ptr) + offset;
                }
                auto allocator = shared_from_this();
                allocation.emplace(
                    host_base + offset, size,
                    [owner = std::move(owner), allocator]() mutable {
                        owner.reset();
                        allocator->NotifyAllocationReleased();
                    },
                    device_ptr, owner_allocation_size);
            }
        }
        if (allocation) {
            return allocation;
        }
        if (request_exceeds_limit) {
            if (request_exceeds_limit_out) *request_exceeds_limit_out = true;
            return std::nullopt;
        }
        lock.lock();
        if (allocation_wait_shutdown_) return std::nullopt;
        // Check the wait budget before deciding whether to retry or keep
        // waiting. Doing it first means a busy generation counter can no
        // longer let a bounded wait outlive its deadline, and the retry path
        // below does not need its own copy of these checks.
        if (timeout == std::chrono::milliseconds::zero()) {
            if (timed_out) *timed_out = true;
            break;
        }
        if (finite_timeout && std::chrono::steady_clock::now() >= deadline) {
            if (timed_out) *timed_out = true;
            break;
        }
        if (allocation_generation_ != generation) continue;
        const auto ready = [this, generation] {
            return allocation_wait_shutdown_ ||
                   allocation_generation_ != generation;
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

    const auto allocate_for_kind =
        [this, kind](size_t allocation_size) -> std::optional<BufferHandle> {
        if (allocator_ == nullptr) return std::nullopt;
        const auto estimated_size =
            allocator_->normalizedAllocationSize(allocation_size);
        if (estimated_size == 0) return std::nullopt;
        if (usage_tracker_) {
            const auto admission =
                usage_tracker_->TryReserve(kind, estimated_size);
            if (admission != DfsBufferAdmissionFailure::kNone) {
                return std::nullopt;
            }
        }
        auto handle = allocator_->allocate(allocation_size);
        if (!handle) {
            if (usage_tracker_) {
                usage_tracker_->CancelReservation(kind, estimated_size);
            }
            return std::nullopt;
        }
        DCHECK_EQ(estimated_size, handle->allocationSize());
        return BufferHandle(shared_from_this(), std::move(*handle),
                            usage_tracker_, kind, estimated_size);
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

}  // namespace mooncake
