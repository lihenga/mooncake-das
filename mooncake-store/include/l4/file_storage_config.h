// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace mooncake {
struct FileStorageConfig;
namespace l4 {
void ReadDfsBufferEnvironment(FileStorageConfig& config);
}
}  // namespace mooncake
