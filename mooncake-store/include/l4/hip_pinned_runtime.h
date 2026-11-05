// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <mutex>
#include <shared_mutex>

namespace mooncake::device::das {

inline std::shared_mutex &HipRuntimeStateMutex() {
    // Pinned-buffer deleters may run during static teardown after this device
    // singleton starts destructing, so the lock must remain valid until exit.
    static auto *mutex = new std::shared_mutex();
    return *mutex;
}

inline hipError_t AllocateHipPinnedHost(void **addr, size_t size,
                                        unsigned int flags) {
    std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
    return hipHostMalloc(addr, size, flags);
}

inline hipError_t GetHipHostDevicePointer(void **device_addr, void *host_addr) {
    std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
    return hipHostGetDevicePointer(device_addr, host_addr, 0);
}

inline void FreeHipPinnedHost(void *addr) {
    std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
    (void)hipHostFree(addr);
}

inline void FreeHipPinnedHostBuffer(void *addr) { FreeHipPinnedHost(addr); }

}  // namespace mooncake::device::das
#endif
