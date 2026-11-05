// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "pinned_buffer_pool.h"
#include <limits>

namespace mooncake {

PinnedBufferPool::Buffer PinnedBufferPool::Acquire(size_t size) {
    const size_t capacity = SizeClass(size);
    if (capacity == 0) return {};
    {
        std::lock_guard<std::mutex> lk(mutex_);
        Buffer buf = TakeCached(capacity, false);
        if (buf.data) return buf;
    }
    return AllocWithPageableFallback(capacity);
}

PinnedBufferPool::Buffer PinnedBufferPool::AcquirePinned(size_t size,
                                                         bool* from_cache,
                                                         bool mapped) {
    if (from_cache) *from_cache = false;
    const size_t capacity = SizeClass(size);
    if (capacity == 0) return {};
    {
        std::lock_guard<std::mutex> lk(mutex_);
        Buffer buf = TakeCached(capacity, true, mapped);
        if (buf.data) {
            if (from_cache) *from_cache = true;
            return buf;
        }
    }
    return AllocPinnedOnly(capacity, mapped);
}

PinnedBufferPool::Buffer PinnedBufferPool::AllocatePinned(size_t size,
                                                          bool mapped) {
    auto accelerators = device::GetAcceleratorRegistry().RuntimeAccelerators();
    for (auto* accelerator : accelerators.Devices()) {
        auto host = mapped ? accelerator->AllocateMappedPinnedHost(size)
                           : accelerator->AllocatePinnedHost(size);
        if (host.addr && (!mapped || host.device_addr)) {
            return Buffer(std::move(host));
        }
        host.reset();
    }
    return {};
}

void PinnedBufferPool::Release(Buffer buf) {
    if (!buf.data || buf.capacity == 0) return;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (buf.capacity <= max_cached_bytes_ - cached_bytes_) {
            cached_bytes_ += buf.capacity;
            pool_[buf.capacity].push_back(std::move(buf));
            return;
        }
    }

    // Releasing pinned host memory may block in the accelerator runtime.
    // Do not hold the pool mutex while invoking the buffer deleter, so one
    // slow hipHostFree cannot block unrelated Acquire/Release operations.
    FreeBuffer(buf);
}

void PinnedBufferPool::Clear() {
    std::map<size_t, std::vector<Buffer>> buffers_to_free;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        buffers_to_free.swap(pool_);
        cached_bytes_ = 0;
    }

    // As in Release(), destroy pinned allocations without holding the pool
    // mutex because the platform deleter may perform a blocking operation.
    for (auto& [_, buffers] : buffers_to_free) {
        for (auto& buf : buffers) {
            FreeBuffer(buf);
        }
    }
}

size_t PinnedBufferPool::cached_bytes() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return cached_bytes_;
}

size_t PinnedBufferPool::SizeClass(size_t size) {
    if (size == 0) return 0;
    if (size <= kSmallAllocationLimit) {
        size_t capacity = kSmallAllocationMinimum;
        while (capacity < size) capacity *= 2;
        return capacity;
    }
    if (size >
        std::numeric_limits<size_t>::max() - (kLargeAllocationAlignment - 1)) {
        return 0;
    }
    return ((size + kLargeAllocationAlignment - 1) /
            kLargeAllocationAlignment) *
           kLargeAllocationAlignment;
}

PinnedBufferPool::Buffer PinnedBufferPool::TakeCached(size_t capacity,
                                                      bool pinned_only,
                                                      bool mapped) {
    for (auto it = pool_.lower_bound(capacity); it != pool_.end(); ++it) {
        auto& buffers = it->second;
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (pinned_only && !buffers[i].pinned_host.addr) continue;
            if (mapped && (!buffers[i].mapped || !buffers[i].device_data))
                continue;
            Buffer buf = std::move(buffers[i]);
            if (i != buffers.size() - 1) {
                buffers[i] = std::move(buffers.back());
            }
            buffers.pop_back();
            cached_bytes_ -= buf.capacity;
            if (buffers.empty()) pool_.erase(it);
            return buf;
        }
    }
    return {};
}

PinnedBufferPool::Buffer PinnedBufferPool::AllocPinnedOnly(size_t capacity,
                                                           bool mapped) {
    const auto& registry = device::GetAcceleratorRegistry();
    auto runtime_accelerator = registry.RuntimeAccelerators();
    for (auto* accelerator : runtime_accelerator.Devices()) {
        auto host = mapped ? accelerator->AllocateMappedPinnedHost(capacity)
                           : accelerator->AllocatePinnedHost(capacity);
        if (host.addr) {
            Buffer buffer(std::move(host));
            buffer.mapped = mapped && buffer.device_data != nullptr;
            if (mapped && !buffer.mapped) {
                FreeBuffer(buffer);
                continue;
            }
            return buffer;
        }
    }
    return {};
}

PinnedBufferPool::Buffer PinnedBufferPool::AllocWithPageableFallback(
    size_t capacity) {
    Buffer buf = AllocPinnedOnly(capacity);
    if (buf.data) return buf;
    return Buffer::Pageable(capacity);
}

PinnedBufferPool::Buffer PinnedBufferPool::AllocNew(size_t size) {
    auto pinned = AllocatePinned(size);
    if (pinned.data) return pinned;
    return Buffer::Pageable(size);
}

void PinnedBufferPool::FreeBuffer(Buffer& buf) {
    buf.pinned_host.reset();
    buf.pageable_host.reset();
    buf = {};
}

}  // namespace mooncake
