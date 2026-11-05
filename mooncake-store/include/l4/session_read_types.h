// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include "client_service.h"
#include <memory>
#include <string>
#include <vector>
#include "client_buffer.h"
#include "rpc_types.h"

namespace mooncake {

struct SessionRangeReadRequest {
    std::string key;
    size_t original_idx;
    Replica::Descriptor replica;
    QueryResult query_result;
    std::vector<void *> buffers;
    std::vector<size_t> sizes;
    std::vector<size_t> src_offsets;
    std::chrono::steady_clock::time_point lease_deadline;
};

using NonMemReadEntry = SessionRangeReadRequest;

// Full-object pinned buffer populated only by waiting-queue prefetch.
struct PrefetchedSessionBuffer {
    std::shared_ptr<BufferHandle> buffer_handle;
    uint64_t total_size;  // object size in bytes
};

struct SessionRangeReadPlan {
    std::vector<SessionRangeReadRequest> memory_requests;
    std::vector<SessionRangeReadRequest> dfs_requests;
};

struct SessionRangeReadContext {
    bool record_access{false};
    uint64_t trace_id{0};
    size_t cache_evicted_count{0};
    std::vector<std::string> access_sources;
    std::chrono::steady_clock::time_point timing_start;
    std::chrono::steady_clock::time_point cache_gc_done;
    std::chrono::steady_clock::time_point memory_done;
    std::chrono::steady_clock::time_point access_done;
};

}  // namespace mooncake
