// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Modified by Hygon Information Technology Co., Ltd., 2026.

#include "device/accelerator_registry.h"
#include "pinned_host_buffer.h"

#include "cuda_alike.h"

#include <vector>

#ifdef MOONCAKE_CUDA_H2D_SCATTER
#include <algorithm>
#include <limits>
#include <list>
#include <mutex>
#include <new>
#include <unordered_map>

#include "cuda_h2d_scatter.h"
#endif

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_MACA) || \
    defined(USE_COREX) || defined(USE_SUPA) ||                     \
    (defined(USE_HYGON) && !defined(USE_HIP))

namespace mooncake {
namespace device {

void EnsureCudaLikeAcceleratorDeviceLinked() {}

namespace {

void FreeCudaLikePinnedHostBuffer(void* addr) { cudaFreeHost(addr); }

class CudaLikeAcceleratorDevice final : public ProbeCachedAcceleratorDevice {
   public:
    explicit CudaLikeAcceleratorDevice(AcceleratorVendor vendor)
        : vendor_(vendor) {}

    AcceleratorVendor Vendor() const override { return vendor_; }

    bool ProbeAvailable() const override {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    PointerInfo QueryPointer(const void* ptr) const override {
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, ptr) == cudaSuccess &&
            attr.type == cudaMemoryTypeDevice) {
            return PointerInfo{.kind = MemoryKind::kDevice,
                               .device_id = attr.device};
        }
        cudaGetLastError();
        return PointerInfo{.kind = MemoryKind::kHost, .device_id = -1};
    }

    int32_t CurrentDeviceId() const override {
        int device_id = -1;
        return cudaGetDevice(&device_id) == cudaSuccess ? device_id : -1;
    }

    void SetContext(int32_t device_id) const override {
        if (device_id >= 0) cudaSetDevice(device_id);
    }

    bool Copy(void* dst, const void* src, size_t size,
              CopyDirection direction) const override {
        cudaMemcpyKind kind = cudaMemcpyDefault;
        switch (direction) {
            case CopyDirection::kHostToDevice:
                kind = cudaMemcpyHostToDevice;
                break;
            case CopyDirection::kDeviceToHost:
                kind = cudaMemcpyDeviceToHost;
                break;
            case CopyDirection::kDeviceToDevice:
                kind = cudaMemcpyDeviceToDevice;
                break;
            case CopyDirection::kHostToHost:
            case CopyDirection::kAuto:
                kind = cudaMemcpyDefault;
                break;
        }
        return cudaMemcpy(dst, src, size, kind) == cudaSuccess;
    }

    bool CopyFromHostAsync(void* dst, const void* src, size_t size,
                           void* stream) const override {
        return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(stream)) ==
               cudaSuccess;
    }

#if defined(USE_CUDA) && CUDART_VERSION >= 13000
    bool CopyFromHostBatchAsync(
        std::span<const HostCopyRange> ranges, void* stream,
        bool* used_batch_kernel = nullptr) const override {
        if (used_batch_kernel) *used_batch_kernel = false;
        if (ranges.size() < 32) {
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }

#ifdef MOONCAKE_CUDA_H2D_SCATTER
        try {
            if (CanUseScatter(ranges, static_cast<cudaStream_t>(stream))) {
                return CopyMappedBatch(ranges, stream, used_batch_kernel);
            }
        } catch (const std::bad_alloc&) {
            return false;
        }
#endif

        std::vector<void*> destinations;
        std::vector<const void*> sources;
        std::vector<size_t> sizes;
        destinations.reserve(ranges.size());
        sources.reserve(ranges.size());
        sizes.reserve(ranges.size());
        for (const auto& range : ranges) {
            destinations.push_back(range.dst);
            sources.push_back(range.src);
            sizes.push_back(range.size);
        }

        cudaMemcpyAttributes attributes{};
        attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attributes_index = 0;
        // A failed batch may already have submitted copies. The caller must
        // synchronize before releasing buffers; do not resubmit it as DMA.
        return cudaMemcpyBatchAsync(
                   destinations.data(), sources.data(), sizes.data(),
                   ranges.size(), &attributes, &attributes_index, 1,
                   static_cast<cudaStream_t>(stream)) == cudaSuccess;
    }
#endif

    bool CreateStream(void** stream) const override {
        cudaStream_t cuda_stream = nullptr;
        if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        *stream = static_cast<void*>(cuda_stream);
        return true;
    }

    bool SynchronizeStream(void* stream) const override {
#ifdef MOONCAKE_CUDA_H2D_SCATTER
        // Capture ownership before synchronization. A concurrent later submit
        // cannot have its descriptor recycled by this completion boundary.
        std::list<Descriptor> completed;
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto it = descriptors_.find(stream);
            if (it != descriptors_.end()) {
                completed.splice(completed.end(), it->second.pending);
            }
        }
        const bool success =
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) ==
            cudaSuccess;
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto it = descriptors_.find(stream);
            if (!success && it != descriptors_.end()) {
                it->second.pending.splice(it->second.pending.end(), completed);
                return false;
            }
            if (success && it != descriptors_.end()) {
                auto& idle = it->second.idle;
                for (auto& buffer : completed) {
                    if (buffer.capacity <= kMaxCachedDescriptorBytes &&
                        buffer.capacity > idle.capacity &&
                        idle_bytes_ - idle.capacity + buffer.capacity <=
                            kMaxIdleDescriptorBytes) {
                        idle_bytes_ =
                            idle_bytes_ - idle.capacity + buffer.capacity;
                        std::swap(idle, buffer);
                    }
                }
            }
        }
        for (const auto& buffer : completed) {
            if (buffer.host) cudaFreeHost(buffer.host);
        }
        return success;
#else
        return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) ==
               cudaSuccess;
#endif
    }

    void DestroyStream(void* stream) const override {
#ifdef MOONCAKE_CUDA_H2D_SCATTER
        if (!SynchronizeStream(stream)) return;
        if (cudaStreamDestroy(static_cast<cudaStream_t>(stream)) != cudaSuccess)
            return;
        std::lock_guard<std::mutex> lock(descriptor_mutex_);
        auto it = descriptors_.find(stream);
        if (it != descriptors_.end()) {
            idle_bytes_ -= it->second.idle.capacity;
            if (it->second.idle.host) cudaFreeHost(it->second.idle.host);
            descriptors_.erase(it);
        }
#else
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
#endif
    }

    PinnedHostBuffer AllocatePinnedHost(size_t size) const override {
        void* addr = nullptr;
        if (cudaMallocHost(&addr, size) != cudaSuccess) {
            cudaGetLastError();
            return PinnedHostBuffer();
        }
        return PinnedHostBuffer(addr, size, FreeCudaLikePinnedHostBuffer);
    }

#ifdef MOONCAKE_CUDA_H2D_SCATTER
    PinnedHostBuffer AllocateMappedPinnedHost(size_t size) const override {
        void* host = nullptr;
        if (cudaHostAlloc(&host, size, cudaHostAllocMapped) != cudaSuccess) {
            cudaGetLastError();
            return PinnedHostBuffer();
        }
        void* alias = nullptr;
        if (cudaHostGetDevicePointer(&alias, host, 0) != cudaSuccess) {
            cudaGetLastError();
            cudaFreeHost(host);
            return PinnedHostBuffer();
        }
        return PinnedHostBuffer(host, size, FreeCudaLikePinnedHostBuffer,
                                alias);
    }

    ~CudaLikeAcceleratorDevice() override {
        for (const auto& [_, state] : descriptors_) {
            if (state.idle.host) cudaFreeHost(state.idle.host);
            // Failed synchronization does not prove device reads completed.
            // Pending raw allocations must survive until process exit.
        }
    }
#endif

   private:
#ifdef MOONCAKE_CUDA_H2D_SCATTER
    struct Descriptor {
        CudaH2DRange* host = nullptr;
        CudaH2DRange* device = nullptr;
        size_t capacity = 0;
        int device_id = -1;
    };
    struct DescriptorState {
        std::list<Descriptor> pending;
        Descriptor idle;
    };
    static constexpr size_t kMaxCachedDescriptorBytes = 64 * 1024;
    static constexpr size_t kMaxIdleDescriptorBytes = 1024 * 1024;
    mutable std::mutex descriptor_mutex_;
    mutable std::unordered_map<void*, DescriptorState> descriptors_;
    mutable std::unordered_map<int, bool> scatter_devices_;
    mutable size_t idle_bytes_ = 0;

    bool CanUseScatter(std::span<const HostCopyRange> ranges,
                       cudaStream_t stream) const {
        if (!stream || stream == cudaStreamLegacy ||
            stream == cudaStreamPerThread ||
            ranges.size() > std::numeric_limits<unsigned int>::max() ||
            ranges.size() >
                std::numeric_limits<size_t>::max() / sizeof(CudaH2DRange))
            return false;
        for (const auto& range : ranges) {
            // Large copies and byte-only misalignment regressed in the B300
            // synthetic matrix. Leave them on the native batch path.
            if (!range.dst || !range.src || !range.src_device ||
                range.size < 4096 || range.size > 65536 ||
                ((reinterpret_cast<uintptr_t>(range.dst) ^
                  reinterpret_cast<uintptr_t>(range.src_device)) &
                 3))
                return false;
        }
        cudaStreamCaptureStatus capture;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone)
            return false;
        const int device_id = CurrentDeviceId();
        if (device_id < 0) return false;
        std::lock_guard<std::mutex> lock(descriptor_mutex_);
        auto it = scatter_devices_.find(device_id);
        if (it == scatter_devices_.end()) {
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, device_id) != cudaSuccess)
                return false;
            // Enable only the architecture measured by this change.
            it = scatter_devices_
                     .emplace(device_id, properties.major == 10 &&
                                             properties.minor == 3 &&
                                             properties.canMapHostMemory)
                     .first;
        }
        return it->second;
    }

    bool CopyMappedBatch(std::span<const HostCopyRange> ranges, void* stream,
                         bool* used_batch_kernel) const {
        Descriptor buffer;
        try {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto& state = descriptors_[stream];
            std::swap(buffer, state.idle);
            idle_bytes_ -= buffer.capacity;
        } catch (const std::bad_alloc&) {
            return false;
        }
        const size_t bytes = ranges.size() * sizeof(CudaH2DRange);
        const int device_id = CurrentDeviceId();
        if (buffer.capacity < bytes || buffer.device_id != device_id) {
            if (buffer.host) cudaFreeHost(buffer.host);
            buffer = {};
            if (cudaHostAlloc(reinterpret_cast<void**>(&buffer.host), bytes,
                              cudaHostAllocMapped) != cudaSuccess)
                return false;
            buffer.capacity = bytes;
            buffer.device_id = device_id;
            if (cudaHostGetDevicePointer(
                    reinterpret_cast<void**>(&buffer.device), buffer.host, 0) !=
                cudaSuccess) {
                cudaFreeHost(buffer.host);
                return false;
            }
        }
        for (size_t i = 0; i < ranges.size(); ++i) {
            buffer.host[i] = {reinterpret_cast<uint64_t>(ranges[i].dst),
                              reinterpret_cast<uint64_t>(ranges[i].src_device),
                              ranges[i].size};
        }
        cudaError_t result;
        try {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            descriptors_.at(stream).pending.push_back(buffer);
            // Publish and launch under the synchronization boundary lock.
            // Keep ownership on errors until a successful synchronization.
            result = LaunchCudaH2DScatter(
                buffer.device, static_cast<unsigned int>(ranges.size()),
                static_cast<cudaStream_t>(stream));
        } catch (const std::bad_alloc&) {
            cudaFreeHost(buffer.host);
            return false;
        }
        if (result != cudaSuccess) return false;
        if (used_batch_kernel) *used_batch_kernel = true;
        return true;
    }
#endif
    AcceleratorVendor vendor_;
};

#define REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(name, vendor) \
    const CudaLikeAcceleratorDevice name##_device(vendor);  \
    const AcceleratorDeviceRegistrar name##_registrar(name##_device)

#if defined(USE_CUDA)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(nvidia, AcceleratorVendor::kNvidia);
#endif

#if defined(USE_MUSA)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(musa, AcceleratorVendor::kMusa);
#endif

#if defined(USE_MACA)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(maca, AcceleratorVendor::kMaca);
#endif

#if defined(USE_HYGON)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(hygon, AcceleratorVendor::kHygon);
#endif

#if defined(USE_COREX)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(corex, AcceleratorVendor::kCorex);
#endif

#if defined(USE_SUPA)
REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE(supa, AcceleratorVendor::kSupa);
#endif

#undef REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE

}  // namespace
}  // namespace device
}  // namespace mooncake

#endif
