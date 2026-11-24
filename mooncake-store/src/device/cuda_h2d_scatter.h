// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace mooncake::device {

struct CudaH2DRange {
    uint64_t dst;
    uint64_t src;
    uint64_t size;
};

cudaError_t LaunchCudaH2DScatter(const CudaH2DRange* ranges, unsigned int count,
                                 cudaStream_t stream);

}  // namespace mooncake::device
