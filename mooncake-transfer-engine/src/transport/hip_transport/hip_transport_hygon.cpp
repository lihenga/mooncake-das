// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "transport/hip_transport/hip_transport.h"
#include "das/hip_error.h"
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace mooncake {

#ifdef USE_HYGON
static std::string getCopyKernelPath() {
    static std::string cached_path;
    static std::once_flag flag;

    std::call_once(flag, []() {
        const char* env_path = getenv("MC_COPY_KERNEL_PATH");
        if (env_path && strlen(env_path) > 0) {
            cached_path = std::string(env_path) + "/mc_copy_kernel.co";
            std::ifstream f(cached_path);
            if (f.good()) {
                LOG(INFO) << "Found mc_copy_kernel.co via environment: "
                          << cached_path;
                return;
            }
        }

        std::string default_path =
            "/usr/local/lib/python3.10/dist-packages/mooncake/"
            "mc_copy_kernel.co";
        std::ifstream f(default_path);
        if (f.good()) {
            cached_path = default_path;
            LOG(INFO)
                << "HipTransport: Found mc_copy_kernel.co in default path: "
                << cached_path;
            return;
        }

        LOG(ERROR) << "HipTransport: mc_copy_kernel.co not found. "
                   << "Please set MC_COPY_KERNEL_PATH environment variable "
                   << "or ensure the file exists at "
                      "/usr/local/lib/python3.10/dist-packages/mooncake/";
    });

    return cached_path;
}

void HipTransport::loadCopyModule() {
    if (module_loaded_) return;

    int num_devices = 0;
    if (!checkHip(hipGetDeviceCount(&num_devices),
                  "HipTransport: hipGetDeviceCount failed")) {
        return;
    }

    std::string kernel_path = getCopyKernelPath();
    if (kernel_path.empty()) {
        return;
    }

    int original_device = 0;
    if (!checkHip(hipGetDevice(&original_device),
                  "HipTransport: hipGetDevice failed")) {
        return;
    }

    for (int device_id = 0; device_id < num_devices; ++device_id) {
        if (!checkHip(hipSetDevice(device_id),
                      "HipTransport: hipSetDevice failed")) {
            return;
        }

        hipModule_t copy_module;
        hipFunction_t copy_func;
        hipError_t err = hipModuleLoad(&copy_module, kernel_path.c_str());
        if (err != hipSuccess) {
            LOG(ERROR) << "HipTransport: Failed to load copy_kernel.co: "
                       << hipGetErrorString(err);
            return;
        }

        err = hipModuleGetFunction(&copy_func, copy_module, "MCCopyKernel");
        if (err != hipSuccess) {
            LOG(ERROR) << "HipTransport: Failed to get MCCopyKernel function: "
                       << hipGetErrorString(err);
            checkHip(hipModuleUnload(copy_module),
                     "HipTransport: Failed to unload copy module");
            return;
        }

        device_copy_modules_[device_id] = copy_module;
        device_copy_funcs_[device_id] = copy_func;
    }

    module_loaded_ = true;
    LOG(INFO) << "HipTransport: Custom MCCopyKernel.co loaded on "
              << device_copy_funcs_.size() << " devices.";
    checkHip(hipSetDevice(original_device),
             "HipTransport: hipSetDevice failed to restore original device");
    return;
}
#endif

}  // namespace mooncake
