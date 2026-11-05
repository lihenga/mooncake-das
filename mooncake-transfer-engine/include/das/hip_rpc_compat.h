// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <hip/hip_runtime.h>
#include <glog/logging.h>

#ifdef USE_HYGON
#if USE_FAKE_HIP_RPC
// hipFabricHandle_t is already provided by hip_runtime.h on DTK; only stub the
// Hygon RPC entry points that may be unavailable without hylink.
struct hipFabricHandle_t {
    unsigned char reserved[64];
};

static hipError_t hipHsaExtRpcMemoryCreate(void* addr, size_t length,
                                           hipFabricHandle_t* handle) {
    (void)addr;
    (void)length;
    (void)handle;
    LOG(ERROR) << "HipTransport: hipHsaExtRpcMemoryCreate is unavailable in "
                  "the current HIP driver";
    return hipErrorNotSupported;
}

static hipError_t hipHsaExtRpcMemoryAttachDevice(
    const hipFabricHandle_t* handle, size_t num_devices, int* devices,
    void** shm_addr) {
    (void)handle;
    (void)num_devices;
    (void)devices;
    (void)shm_addr;
    LOG(ERROR) << "HipTransport: hipHsaExtRpcMemoryAttachDevice is "
                  "unavailable in the current HIP driver";
    return hipErrorNotSupported;
}

static hipError_t hipHsaExtRpcMemoryDetach(void* shm_addr) {
    (void)shm_addr;
    LOG(ERROR) << "HipTransport: hipHsaExtRpcMemoryDetach is unavailable in "
                  "the current HIP driver";
    return hipErrorNotSupported;
}
#else
#include "hip/hip_hsa_ext.h"
#endif
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#endif
