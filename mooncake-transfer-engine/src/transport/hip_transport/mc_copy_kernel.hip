// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include <stdint.h>

struct MCStoreH2DCopyRange {
    uint64_t dst_addr;
    uint64_t src_addr;
    uint64_t size;
};

__device__ __forceinline__ void CopyH2DRange(MCStoreH2DCopyRange range,
                                           uint64_t tid, uint64_t step) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(range.dst_addr);
    const uint8_t* src = reinterpret_cast<const uint8_t*>(range.src_addr);
    // Keep naturally word-aligned copies on their direct vector path.
    // Otherwise peel a shared prefix without reading outside either range.
    if (((range.dst_addr | range.src_addr) & 3) != 0) {
        const uint64_t mismatch = range.dst_addr ^ range.src_addr;
        const uint64_t alignment = (mismatch & 15) == 0 ? 16
                                   : (mismatch & 7) == 0 ? 8
                                   : (mismatch & 3) == 0 ? 4
                                   : (mismatch & 1) == 0 ? 2
                                                         : 1;
        const uint64_t prefix = min(range.size, (-range.src_addr) & (alignment - 1));
        for (uint64_t i = tid; i < prefix; i += step) dst[i] = src[i];
        dst += prefix;
        src += prefix;
        range.dst_addr += prefix;
        range.src_addr += prefix;
        range.size -= prefix;
    }
    uint64_t copied_bytes = 0;

    // A short tail must not force the entire aligned body into byte copies.
    if (((range.dst_addr | range.src_addr) & 15) == 0) {
        const uint64_t count = range.size / 16;
        uint4* dst_v = reinterpret_cast<uint4*>(dst);
        const uint4* src_v = reinterpret_cast<const uint4*>(src);
        for (uint64_t i = tid; i < count; i += step) dst_v[i] = src_v[i];
        copied_bytes = count * 16;
    } else if (((range.dst_addr | range.src_addr) & 7) == 0) {
        const uint64_t count = range.size / 8;
        uint64_t* dst_v = reinterpret_cast<uint64_t*>(dst);
        const uint64_t* src_v = reinterpret_cast<const uint64_t*>(src);
        for (uint64_t i = tid; i < count; i += step) dst_v[i] = src_v[i];
        copied_bytes = count * 8;
    } else if (((range.dst_addr | range.src_addr) & 3) == 0) {
        const uint64_t count = range.size / 4;
        uint32_t* dst_v = reinterpret_cast<uint32_t*>(dst);
        const uint32_t* src_v = reinterpret_cast<const uint32_t*>(src);
        for (uint64_t i = tid; i < count; i += step) dst_v[i] = src_v[i];
        copied_bytes = count * 4;
    } else if (((range.dst_addr | range.src_addr) & 1) == 0) {
        const uint64_t count = range.size / 2;
        uint16_t* dst_v = reinterpret_cast<uint16_t*>(dst);
        const uint16_t* src_v = reinterpret_cast<const uint16_t*>(src);
        for (uint64_t i = tid; i < count; i += step) dst_v[i] = src_v[i];
        copied_bytes = count * 2;
    }
    for (uint64_t i = copied_bytes + tid; i < range.size; i += step) {
        dst[i] = src[i];
    }
}

extern "C" __global__ __launch_bounds__(256) void MCStoreH2DScatterKernel(
    const MCStoreH2DCopyRange* ranges, uint64_t range_count) {
    const uint64_t range_index = static_cast<uint64_t>(blockIdx.x);
    if (range_index < range_count) {
        CopyH2DRange(ranges[range_index], threadIdx.x, blockDim.x);
    }
}

// Four independent 64-thread groups share a block. Each group owns a range.
extern "C" __global__ __launch_bounds__(256) void MCStoreH2DGroupedKernel(
    const MCStoreH2DCopyRange* ranges, uint64_t range_count) {
    constexpr uint64_t kGroupThreads = 64;
    const uint64_t lane = threadIdx.x % kGroupThreads;
    const uint64_t group = threadIdx.x / kGroupThreads;
    const uint64_t groups_per_block = blockDim.x / kGroupThreads;
    const uint64_t worker = blockIdx.x * groups_per_block + group;
    const uint64_t workers = gridDim.x * groups_per_block;
    for (uint64_t range_index = worker; range_index < range_count;
         range_index += workers) {
        CopyH2DRange(ranges[range_index], lane, kGroupThreads);
    }
}
