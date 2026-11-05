// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "storage/distributed/dfs_global_allocator.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/fs_adapter.h"
#include "storage/distributed/posix_fs_adapter.h"
#include "utils.h"
#ifdef USE_3FS
#include "storage/distributed/hf3fs_adapter.h"
#endif

namespace mooncake {

std::vector<BatchAllocateResult> DfsGlobalAllocator::BatchAllocate(
    const std::vector<BatchAllocateRequest>& requests) {
    std::vector<BatchAllocateResult> results;
    results.reserve(requests.size());
    for (const auto& request : requests) {
        results.push_back(
            BatchAllocateResult{request.key, {}, false, ErrorCode::OK});
    }
    if (requests.empty()) return results;

    // Shard mode hashes each key to its own shard, so there is no contiguous
    // region to reserve; allocate one by one but keep the batch all-or-nothing
    // so callers see the same semantics as the bucket allocator.
    for (size_t i = 0; i < requests.size(); ++i) {
        auto descriptor = Allocate(requests[i].key, requests[i].size);
        if (descriptor) {
            results[i].descriptor = std::move(descriptor.value());
            results[i].success = true;
            results[i].error = ErrorCode::OK;
            continue;
        }

        const ErrorCode error = descriptor.error();
        for (size_t j = 0; j < i; ++j) {
            if (!results[j].success) continue;
            Free(results[j].descriptor.offset,
                 results[j].descriptor.aligned_size,
                 results[j].descriptor.shard_idx, results[j].key);
            results[j].success = false;
            results[j].descriptor = DistributedFSDescriptor{};
        }
        for (auto& result : results) {
            result.success = false;
            result.error = error;
            result.descriptor = DistributedFSDescriptor{};
        }
        return results;
    }
    return results;
}

uint64_t DfsGlobalAllocator::GetTotalCapacity() const {
    uint64_t total = 0;
    for (const auto& shard : shards_) {
        if (shard) total += shard->capacity;
    }
    return total;
}

uint64_t DfsGlobalAllocator::GetUsedBytes() const {
    uint64_t used = 0;
    for (const auto& shard : shards_) {
        if (!shard) continue;
        std::shared_lock<std::shared_mutex> lock(shard->handle_mutex);
        const auto report = shard->allocator->storageReport();
        const uint64_t free_space =
            std::min<uint64_t>(report.totalFreeSpace, shard->capacity);
        used += shard->capacity - free_space;
    }
    return used;
}

}  // namespace mooncake
