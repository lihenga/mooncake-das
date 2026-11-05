// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "metadata/object_route.h"

namespace mooncake::l4 {

// Only incomplete DFS copies belong here; healthy objects need no periodic
// route scan. The route observer removes entries on teardown or replacement.
class DfsBackfillIndex {
   public:
    void Sync(const route::WriteGuard& guard) {
        auto& owner = guard.owner_state();
        if (guard.has_object() && guard.state().dfs_backfill &&
            guard.metadata().HasReplica([](const Replica& replica) {
                return replica.is_dfs_replica() && replica.is_completed();
            })) {
            guard.state().dfs_backfill.reset();
        }
        const bool wanted =
            guard.has_object() && bool(guard.state().dfs_backfill);
        if (!wanted && !owner.dfs_backfill_indexed) return;
        std::lock_guard lock(mutex_);
        if (wanted) {
            candidates_.insert_or_assign(guard.key(), guard.generation());
        } else {
            candidates_.erase(guard.key());
        }
        owner.dfs_backfill_indexed = wanted;
    }

    // One maintenance worker calls this. Resume by key, never by an iterator
    // that a concurrent writer can invalidate, and cap both memory and work.
    std::vector<route::ObjectRef> NextBatch(size_t limit) {
        std::vector<route::ObjectRef> result;
        std::lock_guard lock(mutex_);
        auto it = candidates_.upper_bound(after_);
        if (it == candidates_.end()) it = candidates_.begin();
        const size_t count = std::min(limit, candidates_.size());
        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            result.push_back({it->first, it->second});
            if (++it == candidates_.end()) it = candidates_.begin();
        }
        if (!result.empty()) after_ = result.back().key;
        return result;
    }

   private:
    std::mutex mutex_;
    std::map<std::string, route::Generation, std::less<>> candidates_;
    std::string after_;
};

}  // namespace mooncake::l4
