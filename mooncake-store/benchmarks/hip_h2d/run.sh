#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
build=${H2D_BUILD_DIR:-"$root/build/hip_h2d"}
mkdir -p "$build"
if [[ ${1:-build} == build ]]; then
  common=(-O3 -std=c++20 -DUSE_HIP -I"$root/mooncake-store/include" -I"$root/mooncake-transfer-engine/include")
  if [[ -n ${H2D_ARCH:-} ]]; then common+=(--offload-arch="$H2D_ARCH"); fi
  support=("$root/mooncake-store/src/device/"{hip_accelerator_device,accelerator_device,accelerator_registry,runtime_accelerator}.cpp)
  hipcc --genco -fPIC -x hip "$root/mooncake-transfer-engine/src/transport/hip_transport/mc_copy_kernel.hip" -o "$build/mc_copy_kernel.co"
  hipcc "${common[@]}" "${BASH_SOURCE[0]%/*}/bench.cpp" "${support[@]}" -ldl -pthread -o "$build/bench"
  hipcc "${common[@]}" "${BASH_SOURCE[0]%/*}/check.cpp" "${support[@]}" -ldl -pthread \
    -Wl,--wrap=hipHostMalloc,--wrap=hipHostFree,--wrap=hipStreamSynchronize,--wrap=hipModuleLaunchKernel -o "$build/check"
else
  export MC_COPY_KERNEL_PATH="$build"
  "$build/check" "${H2D_DEVICE:-0}"
  "$build/bench" "${H2D_REPEATS:-100}" "${H2D_DEVICE:-0}" "${1:-chain}"
fi
