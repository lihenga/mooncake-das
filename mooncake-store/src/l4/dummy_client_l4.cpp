// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "l4/dummy_client_rpc_internal.h"

namespace mooncake {

std::vector<int> DummyClient::probe_dfs_health() {
    auto result = invoke_rpc<&RealClient::probe_dfs_health, std::vector<int>>();
    if (!result) {
        const int error = toInt(result.error());
        return {error, error, 0};
    }
    return std::move(result.value());
}

int DummyClient::probe_dfs_read() {
    auto result = invoke_rpc<&RealClient::probe_dfs_read_rpc, int>();
    return result ? result.value() : toInt(result.error());
}

int DummyClient::probe_dfs_write() {
    auto result = invoke_rpc<&RealClient::probe_dfs_write_rpc, int>();
    return result ? result.value() : toInt(result.error());
}

int DummyClient::set_dfs_access_policy(bool allow_read, bool allow_write) {
    auto result = invoke_rpc<&RealClient::set_dfs_access_policy, int>(
        allow_read, allow_write);
    return result ? result.value() : toInt(result.error());
}

int DummyClient::set_service_endpoint(const std::string& endpoint) {
    auto result = invoke_rpc<&RealClient::set_service_endpoint, int>(endpoint);
    return result ? result.value() : toInt(result.error());
}

}  // namespace mooncake
