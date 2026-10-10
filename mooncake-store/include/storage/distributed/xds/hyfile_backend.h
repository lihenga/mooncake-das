#pragma once

#include <memory>
#include <string>

#include <ylt/util/tl/expected.hpp>

#include "storage/distributed/accelerator_file_io.h"

namespace mooncake {

// Loads the vendor hipFile ABI from library_path and returns a production
// AcceleratorFileIo implementation. The implementation is intentionally
// created through this factory so ordinary Store builds do not acquire a
// link-time dependency on libhyfile.so.
tl::expected<std::shared_ptr<AcceleratorFileIo>, XdsError>
CreateHyFileAcceleratorFileIo(const std::string& library_path);

}  // namespace mooncake
