#include "utils.h"

#include "runtime_log_config.h"

namespace mooncake {

bool dfs_read_trace_enabled() { return GetDfsReadTraceEnabled(); }

}  // namespace mooncake
