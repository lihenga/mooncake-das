// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ha/leadership/leader_coordinator.h"

namespace mooncake::ha {

inline tl::expected<std::optional<MasterView>, ErrorCode>
ReadCurrentViewOrWaitForReady(LeaderCoordinator& coordinator,
                              std::chrono::milliseconds timeout) {
    auto current_view = coordinator.ReadCurrentView();
    if (!current_view || current_view->has_value()) {
        return current_view;
    }
    auto ready_view = coordinator.WaitForViewChange(std::nullopt, timeout);
    if (!ready_view) {
        return tl::make_unexpected(ready_view.error());
    }
    return ready_view->current_view;
}

}  // namespace mooncake::ha
