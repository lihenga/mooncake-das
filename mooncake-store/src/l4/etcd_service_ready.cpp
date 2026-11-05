// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "ha/leadership/backends/etcd/etcd_leader_coordinator.h"
#include "l4/etcd_service_ready.h"

namespace mooncake::ha::backends::etcd {

ErrorCode EtcdLeaderCoordinator::PublishServiceReady(
    const LeadershipSession& session) {
    auto err = EnsureConnected();
    if (err != ErrorCode::OK) {
        return err;
    }
    const auto owner_token = session.owner_token;
    const auto warming_value = MakeServiceWarmingValue(owner_token);
    auto lease_id = ParseLeaseId(session.owner_token);
    if (!lease_id) {
        return lease_id.error();
    }
    return EtcdHelper::TxnCompareAndPutWithLease(
        {{.key = master_view_key_,
          .kind = EtcdHelper::TxnCompareKind::kValueEquals,
          .expected_value = warming_value}},
        {{.key = master_view_key_, .value = session.view.leader_address}},
        lease_id.value());
}

}  // namespace mooncake::ha::backends::etcd
