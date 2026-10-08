// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Modified by Hygon Information Technology Co., Ltd., 2026.

#include "device/accelerator_registry.h"
#include "pinned_host_buffer.h"

#include "cuda_alike.h"
#include <glog/logging.h>

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_MACA) || \
    defined(USE_COREX) || (defined(USE_HYGON) && !defined(USE_HIP))

namespace mooncake {
namespace device {

void EnsureCudaLikeAcceleratorDeviceLinked() {}

namespace {

void FreeCudaLikePinnedHostBuffer(void* addr) { cudaFreeHost(addr); }

bool CheckCudaLikeResult(cudaError_t result, const char* api) {
    if (result == cudaSuccess) return true;
    LOG(ERROR) << "STORE_GPU_ERROR backend=cuda_like api=" << api
               << " error=" << static_cast<int>(result)
               << " message=" << cudaGetErrorString(result);
    return false;
}

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
        if (device_id >= 0)
            CheckCudaLikeResult(cudaSetDevice(device_id), "cudaSetDevice");
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
        return CheckCudaLikeResult(cudaMemcpy(dst, src, size, kind),
                                   "cudaMemcpy");
    }

    bool CopyFromHostAsync(void* dst, const void* src, size_t size,
                           void* stream) const override {
        return CheckCudaLikeResult(
            cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                            static_cast<cudaStream_t>(stream)),
            "cudaMemcpyAsync_H2D");
    }

    bool CopyToHostAsync(void* dst, const void* src, size_t size,
                         void* stream) const override {
        return CheckCudaLikeResult(
            cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                            static_cast<cudaStream_t>(stream)),
            "cudaMemcpyAsync_D2H");
    }

    bool CreateStream(void** stream) const override {
        cudaStream_t cuda_stream = nullptr;
        if (!CheckCudaLikeResult(cudaStreamCreate(&cuda_stream),
                                 "cudaStreamCreate")) {
            cudaGetLastError();
            return false;
        }
        *stream = static_cast<void*>(cuda_stream);
        return true;
    }

    bool SynchronizeStream(void* stream) const override {
        return CheckCudaLikeResult(
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
            "cudaStreamSynchronize");
    }

    void DestroyStream(void* stream) const override {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }

    bool CreateEvent(void** event) const override {
        cudaEvent_t cuda_event = nullptr;
        if (!CheckCudaLikeResult(
                cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming),
                "cudaEventCreateWithFlags")) {
            cudaGetLastError();
            return false;
        }
        *event = static_cast<void*>(cuda_event);
        return true;
    }

    bool RecordEvent(void* event, void* stream) const override {
        return CheckCudaLikeResult(
            cudaEventRecord(static_cast<cudaEvent_t>(event),
                            static_cast<cudaStream_t>(stream)),
            "cudaEventRecord");
    }

    bool SynchronizeEvent(void* event) const override {
        return CheckCudaLikeResult(
            cudaEventSynchronize(static_cast<cudaEvent_t>(event)),
            "cudaEventSynchronize");
    }

    void DestroyEvent(void* event) const override {
        cudaEventDestroy(static_cast<cudaEvent_t>(event));
    }

    PinnedHostBuffer AllocatePinnedHost(size_t size) const override {
        void* addr = nullptr;
        if (!CheckCudaLikeResult(cudaMallocHost(&addr, size),
                                 "cudaMallocHost")) {
            cudaGetLastError();
            return PinnedHostBuffer();
        }
        return PinnedHostBuffer(addr, size, FreeCudaLikePinnedHostBuffer);
    }

   private:
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

#undef REGISTER_CUDA_LIKE_ACCELERATOR_DEVICE

}  // namespace
}  // namespace device
}  // namespace mooncake

#endif
