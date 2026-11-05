// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "device/runtime_accelerator.h"
#include <cstring>

namespace mooncake::device {

bool AcceleratorDevice::CopyFromHostAsync(void* dst, const void* src,
                                          size_t size, void* stream) const {
    (void)stream;
    return Copy(dst, src, size, CopyDirection::kHostToDevice);
}

bool AcceleratorDevice::CopyFromHostBatchAsync(
    std::span<const HostCopyRange> ranges, void* stream,
    bool* used_batch_kernel) const {
    if (used_batch_kernel) *used_batch_kernel = false;
    bool success = true;
    for (const auto& range : ranges) {
        // Keep attempting the remaining ranges. A backend failure for one
        // range must not turn the logical batch into an early-aborted
        // submission with an ambiguous partial result.
        success = CopyFromHostAsync(range.dst, range.src, range.size, stream) &&
                  success;
    }
    return success;
}

bool AcceleratorDevice::CreateStream(void** stream) const {
    (void)stream;
    return false;
}

bool AcceleratorDevice::SynchronizeStream(void* stream) const {
    (void)stream;
    return true;
}

void AcceleratorDevice::DestroyStream(void* stream) const { (void)stream; }

PinnedHostBuffer AcceleratorDevice::AllocateMappedPinnedHost(
    size_t size) const {
    return AllocatePinnedHost(size);
}

bool RuntimeAccelerator::CopyFromHostBatchAsync(
    std::span<const HostCopyRange> ranges, void* stream) const {
    if (ranges.empty()) return true;
    PointerInfo pointer_info;
    auto* accelerator = FindDeviceForPointer(ranges.front().dst, &pointer_info);
    if (!accelerator) {
        for (const auto& range : ranges) {
            std::memcpy(range.dst, range.src, range.size);
        }
        return true;
    }
    accelerator->SetContext(pointer_info.device_id);
    return accelerator->CopyFromHostBatchAsync(ranges, stream);
}

bool RuntimeAccelerator::CopyFromHostAsync(void* dst, const void* src,
                                           size_t size, void* stream) const {
    PointerInfo pointer_info;
    auto* accelerator = FindDeviceForPointer(dst, &pointer_info);
    if (!accelerator) {
        std::memcpy(dst, src, size);
        return true;
    }
    accelerator->SetContext(pointer_info.device_id);
    return accelerator->CopyFromHostAsync(dst, src, size, stream);
}

}  // namespace mooncake::device
