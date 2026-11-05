#pragma once
#include "common/result.h"
#include "common/byte_size.h"
#include "common/client_buffer_allocation.h"
#include "common/network.h"
#include "common/timestamp.h"
#include "l4/dfs_utils.h"
namespace mooncake {
std::string ResolvePathFromKey(const std::string& key,
                               const std::string& root_dir,
                               const std::string& fsdir);

}  // namespace mooncake
