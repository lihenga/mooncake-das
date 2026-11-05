// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef USE_HYGON
#include <hip/hip_runtime.h>
#include <sys/types.h>
#include <iostream>

namespace mooncake::das {

struct DeviceGuard {
    int old_device;
    DeviceGuard(int new_device) {
        hipGetDevice(&old_device);
        hipSetDevice(new_device);
    }
    ~DeviceGuard() { hipSetDevice(old_device); }
};

constexpr int kAlignment = 2ULL * 1024 * 1024;  // 2MB

inline void *AllocateFabricMemory(ssize_t size, int device,
                                  hipStream_t stream) {
    (void)stream;
    if (size <= 0) {
        return nullptr;
    }
    DeviceGuard guard(device);

    void *ptr = nullptr;
    size = (size + kAlignment - 1) & ~(kAlignment - 1);
    hipError_t result = hipMalloc(&ptr, size);
    if (result != hipSuccess) {
        std::cerr << "hipMalloc failed: " << result << "\n";
        return nullptr;
    }
    return ptr;
}

inline void FreeFabricMemory(void *ptr, ssize_t ssize, int device,
                             hipStream_t stream) {
    (void)ssize;
    (void)stream;
    if (ptr) {
        DeviceGuard guard(device);
        hipError_t result = hipFree(ptr);
        if (result != hipSuccess) {
            std::cerr << "hipFree failed: " << result << "\n";
        }
    }
}

}  // namespace mooncake::das
#endif
