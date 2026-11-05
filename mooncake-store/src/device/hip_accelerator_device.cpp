#include "device/accelerator_registry.h"
#include "l4/hip_scatter_context.h"
#include "l4/hip_pinned_runtime.h"
#include "pinned_host_buffer.h"

#include "cuda_alike.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(USE_HIP)

namespace mooncake {
namespace device {
namespace {

using namespace das;

class HipAcceleratorDevice final : public ProbeCachedAcceleratorDevice {
   public:
    AcceleratorVendor Vendor() const override {
        return AcceleratorVendor::kHip;
    }

    bool ProbeAvailable() const override {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    PointerInfo QueryPointer(const void* ptr) const override {
        hipPointerAttribute_t attr{};
        if (hipPointerGetAttributes(&attr, ptr) == hipSuccess &&
            attr.type == hipMemoryTypeDevice) {
            return PointerInfo{.kind = MemoryKind::kDevice,
                               .device_id = attr.device};
        }
        hipGetLastError();
        return PointerInfo{.kind = MemoryKind::kHost, .device_id = -1};
    }

    int32_t CurrentDeviceId() const override {
        int device_id = -1;
        return hipGetDevice(&device_id) == hipSuccess ? device_id : -1;
    }

    void SetContext(int32_t device_id) const override {
        if (device_id < 0) return;
        std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
        hipSetDevice(device_id);
    }

    bool Copy(void* dst, const void* src, size_t size,
              CopyDirection direction) const override {
        hipMemcpyKind kind = hipMemcpyDefault;
        switch (direction) {
            case CopyDirection::kHostToDevice:
                kind = hipMemcpyHostToDevice;
                break;
            case CopyDirection::kDeviceToHost:
                kind = hipMemcpyDeviceToHost;
                break;
            case CopyDirection::kDeviceToDevice:
                kind = hipMemcpyDeviceToDevice;
                break;
            case CopyDirection::kHostToHost:
            case CopyDirection::kAuto:
                kind = hipMemcpyDefault;
                break;
        }
        return hipMemcpy(dst, src, size, kind) == hipSuccess;
    }

    bool CopyFromHostAsync(void* dst, const void* src, size_t size,
                           void* stream) const override {
        return hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice,
                              static_cast<hipStream_t>(stream)) == hipSuccess;
    }

#if defined(USE_HYGON)
    bool CopyFromHostBatchAsync(
        std::span<const HostCopyRange> ranges, void* stream,
        bool* used_batch_kernel = nullptr) const override {
        return scatter_context_.CopyFromHostBatchAsync(*this, ranges, stream,
                                                       used_batch_kernel);
    }
#endif

    bool CreateStream(void** stream) const override {
        hipStream_t hip_stream = nullptr;
        if (hipStreamCreate(&hip_stream) != hipSuccess) {
            hipGetLastError();
            return false;
        }
        *stream = static_cast<void*>(hip_stream);
        return true;
    }

    bool SynchronizeStream(void* stream) const override {
        const auto hip_stream = static_cast<hipStream_t>(stream);
        bool success = false;
        {
            std::shared_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
            success = hipStreamSynchronize(hip_stream) == hipSuccess;
        }
#if defined(USE_HYGON)
        scatter_context_.ReleaseStreamDescriptors(stream);
#endif
        return success;
    }

    void DestroyStream(void* stream) const override {
        hipStreamDestroy(static_cast<hipStream_t>(stream));
    }

    PinnedHostBuffer AllocatePinnedHost(size_t size) const override {
        void* addr = nullptr;
        if (AllocateHipPinnedHost(&addr, size, 0) != hipSuccess) {
            hipGetLastError();
            return PinnedHostBuffer();
        }
        return PinnedHostBuffer(addr, size, FreeHipPinnedHostBuffer);
    }

#if defined(USE_HYGON)
    PinnedHostBuffer AllocateMappedPinnedHost(size_t size) const override {
        void *addr = nullptr;
        if (AllocateHipPinnedHost(&addr, size, hipHostMallocMapped) !=
            hipSuccess) {
            hipGetLastError();
            return PinnedHostBuffer();
        }

        void *device_addr = nullptr;
        if (GetHipHostDevicePointer(&device_addr, addr) != hipSuccess) {
            hipGetLastError();
            FreeHipPinnedHost(addr);
            return PinnedHostBuffer();
        }
        return PinnedHostBuffer(addr, size, FreeHipPinnedHostBuffer,
                                device_addr);
    }
#endif

    ~HipAcceleratorDevice() override = default;

   private:
#if defined(USE_HYGON)
    mutable das::HipScatterContext scatter_context_;
#endif
};

const AcceleratorDevice& HipDeviceInstance() {
    static HipAcceleratorDevice device;
    return device;
}

const AcceleratorDeviceRegistrar registered_hip_device(HipDeviceInstance());

}  // namespace
}  // namespace device
}  // namespace mooncake

#endif
