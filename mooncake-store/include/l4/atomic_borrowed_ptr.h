// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>

namespace mooncake {

// Atomically publishes a borrowed allocator pointer. Its unique_ptr owner is
// kept stable for the lifetime of MasterService.
template <typename T>
class AtomicBorrowedPtr {
   public:
    T* get() const { return ptr_.load(std::memory_order_acquire); }
    T* operator->() const { return get(); }
    operator T*() const { return get(); }
    void store(T* ptr) { ptr_.store(ptr, std::memory_order_release); }

   private:
    std::atomic<T*> ptr_{nullptr};
};

}  // namespace mooncake
