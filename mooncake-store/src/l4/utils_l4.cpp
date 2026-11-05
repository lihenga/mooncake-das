// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "utils.h"
#include "runtime_log_config.h"
#include <cstdlib>

namespace mooncake {

bool dfs_read_trace_enabled() { return GetDfsReadTraceEnabled(); }

std::string GetEnvStringOr(const char *name, const std::string &default_value) {
    const char *env_val = std::getenv(name);
    return env_val ? std::string(env_val) : default_value;
}

}  // namespace mooncake
