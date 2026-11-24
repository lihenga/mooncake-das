// SPDX-License-Identifier: Apache-2.0
#include "cuda_h2d_scatter.h"

namespace mooncake::device {
namespace {

// Adapted from the HIP L4 copy in PR #154. Each block owns one range;
// mapped-host reads avoid a second staging or descriptor H2D transfer.
__global__ __launch_bounds__(256) void CudaH2DScatter(
    const CudaH2DRange* ranges) {
    const auto range = ranges[blockIdx.x];
    auto* dst = reinterpret_cast<uint8_t*>(range.dst);
    auto* src = reinterpret_cast<const uint8_t*>(range.src);
    const uint64_t mismatch = range.dst ^ range.src;
    const uint64_t alignment = (mismatch & 15) == 0  ? 16
                               : (mismatch & 7) == 0 ? 8
                               : (mismatch & 3) == 0 ? 4
                               : (mismatch & 1) == 0 ? 2
                                                     : 1;
    const uint64_t prefix = min(range.size, (-range.src) & (alignment - 1));
    for (uint64_t i = threadIdx.x; i < prefix; i += blockDim.x) dst[i] = src[i];
    dst += prefix;
    src += prefix;
    const uint64_t size = range.size - prefix;
    uint64_t copied = 0;
    if (alignment == 16) {
        for (uint64_t i = threadIdx.x; i < size / 16; i += blockDim.x) {
            reinterpret_cast<uint4*>(dst)[i] =
                reinterpret_cast<const uint4*>(src)[i];
        }
        copied = size / 16 * 16;
    } else if (alignment == 8) {
        for (uint64_t i = threadIdx.x; i < size / 8; i += blockDim.x) {
            reinterpret_cast<uint64_t*>(dst)[i] =
                reinterpret_cast<const uint64_t*>(src)[i];
        }
        copied = size / 8 * 8;
    } else if (alignment == 4) {
        for (uint64_t i = threadIdx.x; i < size / 4; i += blockDim.x) {
            reinterpret_cast<uint32_t*>(dst)[i] =
                reinterpret_cast<const uint32_t*>(src)[i];
        }
        copied = size / 4 * 4;
    } else if (alignment == 2) {
        for (uint64_t i = threadIdx.x; i < size / 2; i += blockDim.x) {
            reinterpret_cast<uint16_t*>(dst)[i] =
                reinterpret_cast<const uint16_t*>(src)[i];
        }
        copied = size / 2 * 2;
    }
    for (uint64_t i = copied + threadIdx.x; i < size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace

cudaError_t LaunchCudaH2DScatter(const CudaH2DRange* ranges, unsigned int count,
                                 cudaStream_t stream) {
    CudaH2DScatter<<<count, 256, 0, stream>>>(ranges);
    return cudaGetLastError();
}

}  // namespace mooncake::device
