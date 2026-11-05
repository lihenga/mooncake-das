// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstddef>
namespace mooncake::das {
constexpr size_t kSliceSize = 1048576;
constexpr bool kEnableDestDeviceAffinity = true;
constexpr int kPciRelaxedOrderingMode = 0;
}  // namespace mooncake::das
