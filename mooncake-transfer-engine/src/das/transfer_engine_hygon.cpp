// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "transfer_engine_impl.h"
#include <cstdlib>
#include <glog/logging.h>

#ifdef USE_HYGON
namespace mooncake {

int TransferEngineImpl::InstallHygonDefaultTransport() {
    const char* force_hip = getenv("MC_FORCE_HIP");
    const char* force_mnnvl = getenv("MC_FORCE_MNNVL");
    if (force_hip || force_mnnvl) {
        Transport* hip_transport =
            multi_transports_->installTransport("hip", nullptr);
        if (!hip_transport) {
            LOG(ERROR) << "Failed to install HIP transport";
            return -1;
        }
        LOG(INFO) << "Using HIP transport"
                     "(MC_FORCE_HIP or MC_FORCE_MNNVL set)";
    } else if ((local_topology_->getHcaList().size() > 0 &&
                !getenv("MC_FORCE_TCP")) ||
               getenv("MC_FORCE_HCA")) {
        Transport* t =
            multi_transports_->installTransport("rdma", local_topology_);
        if (!t) {
            LOG(ERROR) << "Failed to install RDMA transport";
            return -1;
        }
        LOG(INFO) << "Using RDMA transport (RoCE/iWARP)";
    } else {
        Transport* tcp_transport =
            multi_transports_->installTransport("tcp", nullptr);
        if (!tcp_transport) {
            LOG(ERROR) << "Failed to install TCP transport";
            return -1;
        }
    }

    return 0;
}

}  // namespace mooncake
#endif
