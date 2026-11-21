#include "device/accelerator_registry.h"
#include "pinned_host_buffer.h"

#include "cuda_alike.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <list>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(USE_HIP)

#include <dlfcn.h>
#include <filesystem>

namespace mooncake {
namespace device {
namespace {

std::shared_mutex &HipRuntimeStateMutex() {
    // Pinned-buffer deleters may run during static teardown after this device
    // singleton starts destructing, so the lock must remain valid until exit.
    static auto *mutex = new std::shared_mutex();
    return *mutex;
}

hipError_t AllocateHipPinnedHost(void **addr, size_t size, unsigned int flags) {
    std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
    return hipHostMalloc(addr, size, flags);
}

hipError_t GetHipHostDevicePointer(void **device_addr, void *host_addr) {
    std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
    return hipHostGetDevicePointer(device_addr, host_addr, 0);
}

void FreeHipPinnedHost(void *addr) {
    std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
    (void)hipHostFree(addr);
}

void FreeHipPinnedHostBuffer(void *addr) { FreeHipPinnedHost(addr); }

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

    bool CopyFromHostBatchAsync(
        std::span<const HostCopyRange> ranges, void *stream,
        bool *used_batch_kernel = nullptr) const override {
        if (used_batch_kernel) *used_batch_kernel = false;
        constexpr size_t kKernelMinRanges = 32;
        const auto hip_stream = static_cast<hipStream_t>(stream);
        if (ranges.size() < kKernelMinRanges || !stream ||
            hip_stream == hipStreamLegacy || hip_stream == hipStreamPerThread) {
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }

        bool grouped_shape = ranges.size() >= 128;
        for (const auto &range : ranges) {
            grouped_shape = grouped_shape && range.size >= 16 * 1024 &&
                            range.size <= 64 * 1024 &&
                            range.size == ranges.front().size &&
                            ((reinterpret_cast<uintptr_t>(range.dst) |
                              reinterpret_cast<uintptr_t>(range.src_device)) &
                             7) == 0;
            // The scatter kernel reads the source through its device-visible
            // mapped-host alias. If any source is ordinary pinned memory, keep
            // the whole batch on the DMA path so submission order is preserved.
            if (!range.dst || !range.src || range.size == 0 ||
                !range.src_device) {
                return AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                 stream);
            }
        }

        hipFunction_t function = nullptr;
        bool grouped_kernel = false;
        if (!LoadCopyFunction(CurrentDeviceId(), &function, grouped_shape,
                              &grouped_kernel)) {
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }

        struct DeviceRange {
            uint64_t dst_addr;
            uint64_t src_addr;
            uint64_t size;
        };
        if (ranges.size() >
                std::numeric_limits<size_t>::max() / sizeof(DeviceRange) ||
            ranges.size() > std::numeric_limits<unsigned int>::max()) {
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }

        // Keep the descriptors in mapped pinned host memory. This avoids a
        // second H2D descriptor copy and remains valid until the stream has
        // consumed the kernel. Only completed descriptors may be reused.
        const size_t descriptor_bytes = ranges.size() * sizeof(DeviceRange);
        const int descriptor_device_id = CurrentDeviceId();
        DescriptorBuffer descriptor;
        try {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto &state = descriptors_[stream];
            std::swap(descriptor, state.idle);
            idle_descriptor_bytes_ -= descriptor.capacity;
        } catch (const std::bad_alloc &) {
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }
        if (descriptor.capacity < descriptor_bytes ||
            descriptor.device_id != descriptor_device_id) {
            if (descriptor.host) FreeHipPinnedHost(descriptor.host);
            descriptor = {};
            if (AllocateHipPinnedHost(&descriptor.host, descriptor_bytes,
                                      hipHostMallocMapped) != hipSuccess) {
                hipGetLastError();
                return AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                 stream);
            }
            descriptor.capacity = descriptor_bytes;
            descriptor.device_id = descriptor_device_id;
            if (GetHipHostDevicePointer(&descriptor.device, descriptor.host) !=
                hipSuccess) {
                hipGetLastError();
                FreeHipPinnedHost(descriptor.host);
                return AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                 stream);
            }
        }
        void *host_descriptors = descriptor.host;
        auto *descriptors = static_cast<DeviceRange *>(host_descriptors);
        for (size_t i = 0; i < ranges.size(); ++i) {
            descriptors[i] =
                DeviceRange{reinterpret_cast<uint64_t>(ranges[i].dst),
                            reinterpret_cast<uint64_t>(ranges[i].src_device),
                            ranges[i].size};
        }

        void *device_descriptors = descriptor.device;

        uint64_t range_count = ranges.size();
        void *args[] = {&device_descriptors, &range_count};
        constexpr unsigned int kThreads = 256;
        const auto blocks = static_cast<unsigned int>(
            grouped_kernel ? std::min<size_t>((ranges.size() + 3) / 4, 32)
                           : ranges.size());
        hipError_t launch_result;
        try {
            // Publish under the same lock used to capture the sync boundary.
            // Allocate tracking before launch; sync can splice it without
            // allocating, including when a failed sync must retain ownership.
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto &pending = descriptors_.at(stream).pending;
            pending.push_back(descriptor);
            launch_result =
                hipModuleLaunchKernel(function, blocks, 1, 1, kThreads, 1, 1, 0,
                                      hip_stream, args, nullptr);
            if (launch_result != hipSuccess) pending.pop_back();
        } catch (const std::bad_alloc &) {
            FreeHipPinnedHost(host_descriptors);
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }
        if (launch_result != hipSuccess) {
            hipGetLastError();
            FreeHipPinnedHost(host_descriptors);
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }

        if (used_batch_kernel) *used_batch_kernel = true;
        return true;
    }

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
        std::list<DescriptorBuffer> descriptors;
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto it = descriptors_.find(stream);
            if (it != descriptors_.end()) {
                descriptors.splice(descriptors.end(), it->second.pending);
            }
        }
        bool success = false;
        {
            std::shared_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
            success = hipStreamSynchronize(hip_stream) == hipSuccess;
        }
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto it = descriptors_.find(stream);
            if (it != descriptors_.end() && !success) {
                // An error is not proof that a device has stopped reading.
                it->second.pending.splice(it->second.pending.end(),
                                          descriptors);
                return false;
            }
            // Cache only successful completions on explicit streams. Keep a
            // single idle block per stream with a process-wide idle byte cap.
            if (success && it != descriptors_.end() && stream &&
                hip_stream != hipStreamPerThread &&
                hip_stream != hipStreamLegacy) {
                auto &idle = it->second.idle;
                for (auto &descriptor : descriptors) {
                    if (descriptor.capacity <= kMaxCachedDescriptorBytes &&
                        descriptor.capacity > idle.capacity &&
                        idle_descriptor_bytes_ - idle.capacity +
                                descriptor.capacity <=
                            kMaxIdleDescriptorBytes) {
                        idle_descriptor_bytes_ = idle_descriptor_bytes_ -
                                                 idle.capacity +
                                                 descriptor.capacity;
                        std::swap(idle, descriptor);
                    }
                }
            }
        }
        for (const auto &descriptor : descriptors) {
            if (descriptor.host) FreeHipPinnedHost(descriptor.host);
        }
        return success;
    }

    void DestroyStream(void* stream) const override {
        // Destroy also drains submissions when a caller has not synchronized.
        if (!SynchronizeStream(stream)) return;
        hipStreamDestroy(static_cast<hipStream_t>(stream));
        DescriptorBuffer idle;
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto it = descriptors_.find(stream);
            if (it != descriptors_.end()) {
                idle = it->second.idle;
                idle_descriptor_bytes_ -= idle.capacity;
                descriptors_.erase(it);
            }
        }
        if (idle.host) FreeHipPinnedHost(idle.host);
    }

    PinnedHostBuffer AllocatePinnedHost(size_t size) const override {
        void* addr = nullptr;
        if (AllocateHipPinnedHost(&addr, size, 0) != hipSuccess) {
            hipGetLastError();
            return PinnedHostBuffer();
        }
        return PinnedHostBuffer(addr, size, FreeHipPinnedHostBuffer);
    }

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

    ~HipAcceleratorDevice() override {
        bool has_pending = false;
        {
            std::lock_guard<std::mutex> descriptor_lock(descriptor_mutex_);
            for (const auto &[_, state] : descriptors_) {
                has_pending |= !state.pending.empty();
                if (state.idle.host) FreeHipPinnedHost(state.idle.host);
            }
            // Normal stream-pool shutdown drains pending work. If it failed,
            // retain its raw allocations and modules until process exit.
            descriptors_.clear();
        }
        if (has_pending) return;
        std::lock_guard<std::mutex> lock(module_mutex_);
        for (const auto &[device_id, functions] : copy_functions_) {
            SetContext(device_id);
            hipModuleUnload(functions.module);
        }
    }

   private:
    static std::string CopyKernelPath() {
        const char *directory = std::getenv("MC_COPY_KERNEL_PATH");
        if (directory && directory[0] != '\0') {
            return std::string(directory) + "/mc_copy_kernel.co";
        }

        // Wheels and native installs keep the kernel beside the Store library.
        // auditwheel may move that library into a sibling .libs directory.
        static int library_anchor;
        Dl_info library_info{};
        if (dladdr(&library_anchor, &library_info) && library_info.dli_fname) {
            const auto library_dir =
                std::filesystem::path(library_info.dli_fname).parent_path();
            for (const auto &path : {library_dir / "mc_copy_kernel.co",
                                     library_dir.parent_path() / "mooncake" /
                                         "mc_copy_kernel.co"}) {
                if (std::ifstream(path).good()) return path.string();
            }
        }
        return "/usr/local/lib/python3.10/dist-packages/mooncake/"
               "mc_copy_kernel.co";
    }

    bool LoadCopyFunction(int device_id, hipFunction_t *function,
                          bool request_grouped, bool *used_grouped) const {
        if (device_id < 0) return false;
        std::lock_guard<std::mutex> lock(module_mutex_);
        auto it = copy_functions_.find(device_id);
        if (it != copy_functions_.end()) {
            *used_grouped = request_grouped && it->second.grouped;
            *function = *used_grouped ? it->second.grouped : it->second.base;
            return true;
        }

        const std::string path = CopyKernelPath();
        std::ifstream kernel_file(path);
        if (!kernel_file.good()) return false;

        hipModule_t module = nullptr;
        if (hipModuleLoad(&module, path.c_str()) != hipSuccess) {
            hipGetLastError();
            return false;
        }
        hipFunction_t loaded_function = nullptr;
        if (hipModuleGetFunction(&loaded_function, module,
                                 "MCStoreH2DScatterKernel") != hipSuccess) {
            hipGetLastError();
            hipModuleUnload(module);
            return false;
        }
        hipFunction_t grouped_function = nullptr;
        hipDeviceProp_t properties{};
        const bool tuned_device =
            hipGetDeviceProperties(&properties, device_id) == hipSuccess &&
            std::strncmp(properties.gcnArchName, "gfx936", 6) == 0;
        if (tuned_device &&
            hipModuleGetFunction(&grouped_function, module,
                                 "MCStoreH2DGroupedKernel") != hipSuccess) {
            // Older code objects remain compatible with the baseline entry.
            hipGetLastError();
            grouped_function = nullptr;
        }
        try {
            copy_functions_.emplace(
                device_id,
                CopyFunctions{module, loaded_function, grouped_function});
        } catch (const std::bad_alloc &) {
            hipModuleUnload(module);
            return false;
        }
        *used_grouped = request_grouped && grouped_function;
        *function = *used_grouped ? grouped_function : loaded_function;
        return true;
    }

    struct DescriptorBuffer {
        void *host = nullptr;
        void *device = nullptr;
        size_t capacity = 0;
        int device_id = -1;
    };
    struct DescriptorState {
        std::list<DescriptorBuffer> pending;
        DescriptorBuffer idle;
    };
    // Retain one completed allocation per stream, at most 64 KiB each.
    static constexpr size_t kMaxCachedDescriptorBytes = 64 * 1024;
    static constexpr size_t kMaxIdleDescriptorBytes = 1024 * 1024;
    mutable size_t idle_descriptor_bytes_ = 0;

    mutable std::mutex module_mutex_;
    mutable std::mutex descriptor_mutex_;
    struct CopyFunctions {
        hipModule_t module = nullptr;
        hipFunction_t base = nullptr;
        hipFunction_t grouped = nullptr;
    };
    mutable std::unordered_map<int, CopyFunctions> copy_functions_;
    mutable std::unordered_map<void *, DescriptorState> descriptors_;
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
