// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstddef>

namespace mooncake::device {

struct HostCopyRange {
    void* dst = nullptr;
    const void* src = nullptr;
    size_t size = 0;
    // Optional device-visible alias for mapped host memory.
    const void* src_device = nullptr;
};

}  // namespace mooncake::device
