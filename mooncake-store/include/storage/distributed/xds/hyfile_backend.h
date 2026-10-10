#pragma once

#include <memory>
#include <string>

#include <ylt/util/tl/expected.hpp>

#include "storage/distributed/accelerator_file_io.h"

#ifdef MOONCAKE_ENABLE_XDS_TEST_HOOKS
#include "storage/distributed/xds/hyfile_abi.h"
#endif

namespace mooncake {

// Loads the vendor hipFile ABI from library_path and returns a production
// AcceleratorFileIo implementation. The implementation is intentionally
// created through this factory so ordinary Store builds do not acquire a
// link-time dependency on libhyfile.so.
tl::expected<std::shared_ptr<AcceleratorFileIo>, XdsError>
CreateHyFileAcceleratorFileIo(const std::string& library_path);

#ifdef MOONCAKE_ENABLE_XDS_TEST_HOOKS
namespace xds::testing {

// Test-only HIP indirection. Production uses the real HIP runtime entrypoints.
struct HipRuntimeApi {
    int (*get_device_count)(int*) = nullptr;
    int (*set_device)(int) = nullptr;
    int (*device_synchronize)() = nullptr;
    int success = 0;
};

tl::expected<std::shared_ptr<AcceleratorFileIo>, XdsError>
CreateHyFileAcceleratorFileIoForTesting(
    const std::string& runtime_identity, const hyfile_abi::Api& hyfile_api,
    const HipRuntimeApi& hip_api);

void ResetHyFileProcessRuntimeForTesting();

}  // namespace xds::testing
#endif

}  // namespace mooncake
