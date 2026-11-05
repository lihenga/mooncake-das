// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(USE_HIP) && defined(USE_HYGON)
#include "device/accelerator_device.h"
#include "l4/hip_pinned_runtime.h"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mooncake::device::das {

class HipScatterContext {
   public:
    bool CopyFromHostBatchAsync(const AcceleratorDevice &owner,
                                std::span<const HostCopyRange> ranges,
                                void *stream,
                                bool *used_batch_kernel = nullptr) const {
        if (used_batch_kernel) *used_batch_kernel = false;
        constexpr size_t kKernelMinRanges = 32;
        if (ranges.size() < kKernelMinRanges) {
            return owner.AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                   stream);
        }

        for (const auto &range : ranges) {
            // The scatter kernel reads the source through its device-visible
            // mapped-host alias. If any source is ordinary pinned memory, keep
            // the whole batch on the DMA path so submission order is preserved.
            if (!range.dst || !range.src || range.size == 0 ||
                !range.src_device) {
                return owner.AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                       stream);
            }
        }

        hipFunction_t function = nullptr;
        if (!LoadCopyFunction(owner.CurrentDeviceId(), &function)) {
            return owner.AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                   stream);
        }

        struct DeviceRange {
            uint64_t dst_addr;
            uint64_t src_addr;
            uint64_t size;
        };
        if (ranges.size() >
                std::numeric_limits<size_t>::max() / sizeof(DeviceRange) ||
            ranges.size() > std::numeric_limits<unsigned int>::max()) {
            return owner.AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                   stream);
        }

        // Keep the descriptors in mapped pinned host memory. This avoids a
        // second H2D descriptor copy and remains valid until the stream has
        // consumed the kernel, at which point SynchronizeStream releases it.
        const size_t descriptor_bytes = ranges.size() * sizeof(DeviceRange);
        void *host_descriptors = nullptr;
        if (AllocateHipPinnedHost(&host_descriptors, descriptor_bytes,
                                  hipHostMallocMapped) != hipSuccess) {
            hipGetLastError();
            return owner.AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                   stream);
        }
        auto *descriptors = static_cast<DeviceRange *>(host_descriptors);
        for (size_t i = 0; i < ranges.size(); ++i) {
            descriptors[i] =
                DeviceRange{reinterpret_cast<uint64_t>(ranges[i].dst),
                            reinterpret_cast<uint64_t>(ranges[i].src_device),
                            ranges[i].size};
        }

        void *device_descriptors = nullptr;
        if (GetHipHostDevicePointer(&device_descriptors, host_descriptors) !=
            hipSuccess) {
            hipGetLastError();
            FreeHipPinnedHost(host_descriptors);
            return owner.AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                   stream);
        }

        uint64_t range_count = ranges.size();
        void *args[] = {&device_descriptors, &range_count};
        constexpr unsigned int kThreads = 256;
        const auto hip_stream = static_cast<hipStream_t>(stream);
        const hipError_t launch_result = hipModuleLaunchKernel(
            function, static_cast<unsigned int>(ranges.size()), 1, 1, kThreads,
            1, 1, 0, hip_stream, args, nullptr);
        if (launch_result != hipSuccess) {
            hipGetLastError();
            FreeHipPinnedHost(host_descriptors);
            return owner.AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                   stream);
        }

        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            pending_descriptors_[static_cast<void *>(hip_stream)].push_back(
                host_descriptors);
        }
        if (used_batch_kernel) *used_batch_kernel = true;
        return true;
    }

    void ReleaseStreamDescriptors(void *stream) const {
        // Synchronization establishes that the kernel no longer dereferences
        // the mapped descriptor array. Free it even when the stream reports a
        // device-side error; the stream has completed in either case.
        std::vector<void *> descriptors;
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto it = pending_descriptors_.find(stream);
            if (it != pending_descriptors_.end()) {
                descriptors = std::move(it->second);
                pending_descriptors_.erase(it);
            }
        }
        for (void *descriptor : descriptors) {
            FreeHipPinnedHost(descriptor);
        }
    }

    ~HipScatterContext() {
        std::vector<void *> descriptors;
        {
            std::lock_guard<std::mutex> descriptor_lock(descriptor_mutex_);
            for (auto &[_, pending] : pending_descriptors_) {
                descriptors.insert(descriptors.end(), pending.begin(),
                                   pending.end());
            }
            pending_descriptors_.clear();
        }
        for (void *descriptor : descriptors) {
            FreeHipPinnedHost(descriptor);
        }

        std::lock_guard<std::mutex> lock(module_mutex_);
        for (const auto &[_, module] : copy_modules_) {
            hipModuleUnload(module);
        }
    }

   private:
    static std::string CopyKernelPath() {
        const char *directory = std::getenv("MC_COPY_KERNEL_PATH");
        if (directory && directory[0] != '\0') {
            return std::string(directory) + "/mc_copy_kernel.co";
        }
        return "/usr/local/lib/python3.10/dist-packages/mooncake/"
               "mc_copy_kernel.co";
    }

    bool LoadCopyFunction(int device_id, hipFunction_t *function) const {
        if (device_id < 0) return false;
        std::lock_guard<std::mutex> lock(module_mutex_);
        auto it = copy_functions_.find(device_id);
        if (it != copy_functions_.end()) {
            *function = it->second;
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
        copy_modules_.emplace(device_id, module);
        copy_functions_.emplace(device_id, loaded_function);
        *function = loaded_function;
        return true;
    }

    mutable std::mutex module_mutex_;
    mutable std::mutex descriptor_mutex_;
    mutable std::unordered_map<int, hipModule_t> copy_modules_;
    mutable std::unordered_map<int, hipFunction_t> copy_functions_;
    mutable std::unordered_map<void *, std::vector<void *>>
        pending_descriptors_;
};

}  // namespace mooncake::device::das
#endif
