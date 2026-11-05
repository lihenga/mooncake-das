// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "client_service.h"

namespace mooncake {

struct Client::AsyncDfsWriteContext {
    struct ServiceDualWriteCompletion {
        bool eligible = false;
        bool memory_succeeded = false;
        size_t requested_dfs_replicas = 0;
    };

    const std::chrono::steady_clock::time_point started_at =
        std::chrono::steady_clock::now();
    std::vector<std::string> keys;
    std::vector<DistributedFSDescriptor> descriptors;
    std::shared_ptr<ClientBufferAllocator> unified_buffer_allocator;
    std::optional<BufferHandle> staging_buffer;
    std::vector<std::vector<Slice>> slices;
    std::vector<ServiceDualWriteCompletion> service_dual_writes;
    std::atomic<bool> service_metrics_settled{false};
    std::atomic<bool> io_completed{false};
    // Ready before enqueue; every End retry uses the foreground checksum.
    std::vector<std::optional<uint64_t>> object_checksums;
    std::shared_ptr<DistributedStorageBackend> backend;
    bool is_upsert = false;
    std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::time_point::max();

    void ReleaseStagingBuffer();
    ~AsyncDfsWriteContext();
};

}  // namespace mooncake
