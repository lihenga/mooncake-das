// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace mooncake {

// Gates the verbose per-batch timing logs on the DFS read path (alloc/read
// split, scatter range/byte totals, backend BatchRead/ReadFully). Off by
// default so the hot path pays nothing; set MC_STORE_DFS_READ_TRACE=1 (or
// true/on) to enable. The runtime API can update the current value.
bool dfs_read_trace_enabled();

// Helper: Get integer from environment variable, fallback to default
template <typename T>
T GetEnvOr(const char* name, T default_value) {
    const char* env_val = std::getenv(name);
    if (!env_val || std::string(env_val).empty()) {
        return default_value;
    }
    try {
        long long value = std::stoll(env_val);
        // Check range for unsigned types
        if constexpr (std::is_same_v<T, uint32_t>) {
            if (value < 0 || value > UINT32_MAX) throw std::out_of_range("");
        }
        return static_cast<T>(value);
    } catch (...) {
        return default_value;
    }
}

std::string GetEnvStringOr(const char* name, const std::string& default_value);

}  // namespace mooncake
