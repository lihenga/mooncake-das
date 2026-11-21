#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "device/accelerator_device.h"

namespace mooncake {
namespace device {

class RuntimeAccelerator {
   public:
    RuntimeAccelerator() = default;
    explicit RuntimeAccelerator(std::vector<const AcceleratorDevice*> devices);

    std::span<const AcceleratorDevice* const> Devices() const;

    const AcceleratorDevice* FindDeviceForPointer(
        const void* ptr, PointerInfo* out_info = nullptr) const;

    bool CopyToHost(void* dst, const void* src, size_t size) const;

    bool CopyFromHost(void* dst, const void* src, size_t size) const;

    bool CopyFromHostAsync(void* dst, const void* src, size_t size,
                           void* stream) const;

    bool CopyFromHostBatchAsync(std::span<const HostCopyRange> ranges,
                                void* stream) const;

    // Warms the mapped-host H2D path of the device `device_id` (or of every
    // available device when device_id < 0) for the given streams. No-op when
    // no accelerator backs that device.
    bool WarmMappedHostCopyPath(int32_t device_id,
                                const std::vector<void*>& streams,
                                void* host_buffer, void* host_device_alias,
                                size_t buffer_size) const;

    // Copy between two pointers that may each be host or device resident;
    // falls back to memcpy when neither side is on a known accelerator.
    bool CopyAuto(void* dst, const void* src, size_t size) const;

   private:
    std::vector<const AcceleratorDevice*> devices_;
};

}  // namespace device
}  // namespace mooncake
