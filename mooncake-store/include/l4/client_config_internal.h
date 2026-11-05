// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "client_service.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "environ.h"
#include "environment_variables.h"
#include <algorithm>
#include <filesystem>

namespace mooncake::client_config_detail {

inline UUID DfsWriteEpoch(const std::vector<Replica::Descriptor>& replicas) {
    for (const auto& replica : replicas) {
        if (replica.is_dfs_replica()) {
            return replica.get_dfs_descriptor().MasterEpoch();
        }
    }
    return {};
}

inline size_t ReadDfsWriteChunkSize() {
    const auto configured = Environ::Read(
        FileStorageEnvironmentVariables::MC_STORE_DFS_WRITE_CHUNK_SIZE_BYTES);
    if (!configured.has_value() || *configured <= 0) {
        return 1ULL * 1024 * 1024 * 1024;
    }
    return static_cast<size_t>(*configured);
}

inline std::chrono::milliseconds ReadDfsWriteArenaWait() {
    constexpr int64_t kDefaultWaitMs = 10000;
    const auto configured = Environ::Read(
        FileStorageEnvironmentVariables::MC_STORE_DFS_WRITE_ARENA_WAIT_MS);
    return std::chrono::milliseconds(
        std::max<int64_t>(configured.value_or(kDefaultWaitMs), 0));
}

inline static std::string NormalizeDfsPath(const std::string& path) {
    if (path.empty()) return {};
    std::error_code ec;
    auto normalized = std::filesystem::absolute(path, ec);
    if (ec) return path;
    return normalized.lexically_normal().string();
}

inline static void WarnOnClientDfsPathMismatch(
    const std::string& master_fsdir) {
    const auto storage_config = FileStorageConfig::FromEnvironment();
    if (storage_config.storage_backend_type !=
        StorageBackendType::kDistributed) {
        return;
    }
    const auto client_fsdir = DistributedStorageConfig::FromEnvironment().fsdir;
    if (client_fsdir.empty() || master_fsdir.empty()) return;

    const auto normalized_client = NormalizeDfsPath(client_fsdir);
    const auto normalized_master = NormalizeDfsPath(master_fsdir);
    if (normalized_client == normalized_master) return;

    LOG(WARNING) << "DFS fsdir differs from master: client='"
                 << normalized_client << "', master='" << normalized_master
                 << ". Verify that both paths refer to the same DFS storage.";
}

}  // namespace mooncake::client_config_detail
