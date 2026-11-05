// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/bucket_entry_layout.h"
#include <limits>

namespace mooncake::dfs_descriptor_detail {

inline bool IsDfsDescriptorRangeValid(const DistributedFSDescriptor& desc,
                                      const DistributedStorageConfig& config) {
    if (config.alignment == 0 || desc.object_size == 0 ||
        desc.aligned_size < desc.object_size ||
        desc.offset % config.alignment != 0 ||
        desc.aligned_size % config.alignment != 0) {
        return false;
    }
    if (desc.offset > config.shard_capacity ||
        desc.aligned_size > config.shard_capacity - desc.offset) {
        return false;
    }

    constexpr uint64_t kMaxFileOffset =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    return desc.offset <= kMaxFileOffset &&
           desc.aligned_size <= kMaxFileOffset - desc.offset;
}

inline bool IsBucketDescriptorRangeValid(
    const DistributedFSDescriptor& desc, const std::string& key,
    const DistributedStorageConfig& config) {
    if (desc.object_size == 0 || desc.shard_idx < 0) return false;
    if (key.empty()) return false;

    auto layout = RebuildBucketEntryLayout(desc.offset, desc.object_size,
                                           config.alignment);
    if (!layout) return false;
    if (layout->offset != desc.offset) return false;
    if (layout->aligned_size != desc.aligned_size) return false;
    if (layout->end() > config.bucket_capacity) return false;

    constexpr uint64_t kMaxFileOffset =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    return layout->offset <= kMaxFileOffset &&
           layout->aligned_size <= kMaxFileOffset - layout->offset;
}

}  // namespace mooncake::dfs_descriptor_detail
