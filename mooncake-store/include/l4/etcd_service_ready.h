// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ha/ha_types.h"
#include <string>
#include <string_view>

namespace mooncake::ha::backends::etcd {

constexpr std::string_view kServiceWarmingPrefix =
    "__mooncake_service_warming__:";

inline std::string MakeServiceWarmingValue(const OwnerToken& owner_token) {
    return std::string(kServiceWarmingPrefix) + owner_token;
}

inline bool IsServiceWarmingValue(const std::string& value) {
    return value.rfind(kServiceWarmingPrefix, 0) == 0;
}
}  // namespace mooncake::ha::backends::etcd
