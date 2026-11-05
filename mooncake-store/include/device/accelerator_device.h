#pragma once

#include <atomic>
#include "l4/host_copy_range.h"
#include <cstddef>
#include <cstdint>
#include <span>

#include "pinned_host_buffer.h"

namespace mooncake {
namespace device {

enum class AcceleratorVendor {
    kNvidia,
    kMusa,
    kMaca,
    kHygon,
    kCorex,
    kHip,
    kAscend,
    kSunrise,
};

enum class MemoryKind {
    kHost,
    kDevice,
    kUnknown,
};

enum class CopyDirection {
    kHostToHost,
    kHostToDevice,
    kDeviceToHost,
    kDeviceToDevice,
    kAuto,
};

struct PointerInfo {
    MemoryKind kind = MemoryKind::kUnknown;
    int32_t device_id = -1;
};

class AcceleratorDevice {
   public:
    virtual ~AcceleratorDevice() = default;

    virtual AcceleratorVendor Vendor() const = 0;
    virtual bool Available(bool ensure = false) const = 0;
    virtual PointerInfo QueryPointer(const void* ptr) const = 0;
    virtual int32_t CurrentDeviceId() const = 0;
    virtual void SetContext(int32_t device_id) const = 0;
    virtual bool Copy(void* dst, const void* src, size_t size,
                      CopyDirection direction) const = 0;
    // Asynchronous host-to-device copy. Backends without stream support use
    // the synchronous implementation as a correctness-preserving fallback.
    virtual bool CopyFromHostAsync(void* dst, const void* src, size_t size,
                                   void* stream) const;
    // Submit a set of independent host-to-device copies on one stream. The
    // default implementation deliberately keeps the old semantics; backends
    // with a native scatter-copy primitive or a suitable batch kernel can
    // override it to reduce host submission overhead.
    virtual bool CopyFromHostBatchAsync(
        std::span<const HostCopyRange> ranges, void* stream,
        bool* used_batch_kernel = nullptr) const;
    virtual bool CreateStream(void** stream) const;
    virtual bool SynchronizeStream(void* stream) const;
    virtual void DestroyStream(void* stream) const;
    virtual PinnedHostBuffer AllocatePinnedHost(size_t size) const = 0;

    // Allocate host memory with a device-visible mapping when supported. This
    // is separate from ordinary pinned memory because mapped host access is
    // only needed by the DFS scatter-kernel path.
    virtual PinnedHostBuffer AllocateMappedPinnedHost(size_t size) const;
};

class ProbeCachedAcceleratorDevice : public AcceleratorDevice {
   public:
    bool Available(bool ensure = false) const override;

   protected:
    virtual bool ProbeAvailable() const = 0;

   private:
    mutable std::atomic<uint8_t> available_state_{0};
};

}  // namespace device
}  // namespace mooncake
