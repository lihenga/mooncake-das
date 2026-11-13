#include <gtest/gtest.h>

#include "runtime_log_config.h"
#include "utils.h"

namespace mooncake::test {

TEST(DfsReadTraceLinkTest, FollowsRuntimeSetting) {
    const bool previous = GetDfsReadTraceEnabled();
    SetDfsReadTraceEnabled(false);
    EXPECT_FALSE(dfs_read_trace_enabled());
    SetDfsReadTraceEnabled(true);
    EXPECT_TRUE(dfs_read_trace_enabled());
    SetDfsReadTraceEnabled(previous);
}

}  // namespace mooncake::test
