// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <hip/hip_runtime.h>
#include <glog/logging.h>

namespace mooncake {

inline bool checkHip(hipError_t result, const char* message) {
    if (result != hipSuccess) {
        LOG(ERROR) << message << " (Error code: " << result << " - "
                   << hipGetErrorString(result) << ")";
        return false;
    }
    return true;
}

}  // namespace mooncake
