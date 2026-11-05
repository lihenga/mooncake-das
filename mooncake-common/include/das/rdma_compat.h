// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <infiniband/verbs.h>
#ifdef USE_SHCA
#include <infiniband/shca_17b_types.h>
#endif

namespace mooncake::das {

template <typename Lid>
inline uint32_t DecodeRdmaLid(Lid lid) {
#ifdef USE_SHCA
    return u17_to_32(lid);
#else
    return lid;
#endif
}

inline void SetPeerLid(ibv_ah_attr& attributes, uint32_t peer_lid) {
#ifdef USE_SHCA
    attributes.dlid = u32_to_17(peer_lid);
#else
    attributes.dlid = static_cast<uint16_t>(peer_lid);
#endif
}

// The classic engine caps the NIC's read depth at its historical default.
constexpr int kIdealRdAtomicDepth = 16;
inline int clampRdAtomicDepth(int device_max) {
    if (device_max <= 0) return 1;
    return device_max < kIdealRdAtomicDepth ? device_max : kIdealRdAtomicDepth;
}

// TENT also preserves an explicitly requested zero and applies the NIC limit.
inline uint8_t ClampRdAtomicRequest(uint8_t requested, int device_max) {
    if (requested == 0) return 0;
    if (device_max <= 0) return 1;
    if (static_cast<int>(requested) > device_max)
        return static_cast<uint8_t>(device_max);
    return requested;
}

}  // namespace mooncake::das
