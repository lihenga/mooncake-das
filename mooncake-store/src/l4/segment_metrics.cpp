// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "segment.h"

#include "master_metric_manager.h"
#include "common/zstd_util.h"

#include <functional>
#include <unordered_set>

namespace mooncake {

SegmentManager::~SegmentManager() { ReleaseActiveClientMetrics(); }

void SegmentManager::RefreshActiveClientMetric(const UUID& client_id) {
    const auto client = client_segments_.find(client_id);
    const bool has_memory =
        client != client_segments_.end() &&
        std::any_of(client->second.begin(), client->second.end(),
                    [this](const UUID& segment_id) {
                        const auto segment = mounted_segments_.find(segment_id);
                        return segment != mounted_segments_.end() &&
                               segment->second.segment.size > 0;
                    });
    if (has_memory) {
        if (metric_memory_clients_.insert(client_id).second) {
            MasterMetricManager::instance().inc_active_clients();
        }
    } else if (metric_memory_clients_.erase(client_id) > 0) {
        MasterMetricManager::instance().dec_active_clients();
    }
}

void SegmentManager::ReleaseActiveClientMetrics() {
    if (!metric_memory_clients_.empty()) {
        MasterMetricManager::instance().dec_active_clients(
            metric_memory_clients_.size());
    }
    metric_memory_clients_.clear();
}

}  // namespace mooncake
