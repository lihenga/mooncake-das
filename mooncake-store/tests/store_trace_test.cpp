#include "store_trace.h"

#include <gtest/gtest.h>

#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mooncake::test {
namespace {

class TraceSink : public google::LogSink {
   public:
    TraceSink() { google::AddLogSink(this); }
    ~TraceSink() override { google::RemoveLogSink(this); }
    void send(google::LogSeverity, const char*, const char*, int,
              const std::tm*, const char* message, size_t length) override {
        std::lock_guard<std::mutex> lock(mutex_);
        messages_.emplace_back(message, length);
    }
    std::string Text() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string result;
        for (const auto& message : messages_) result += message + '\n';
        return result;
    }

   private:
    std::mutex mutex_;
    std::vector<std::string> messages_;
};

TEST(StoreTraceTest, NestedCallsAndBackgroundKeepCorrelation) {
    TraceSink sink;
    uint64_t parent = 0;
    {
        StoreTrace outer("wrapper", &sink, 3);
        parent = outer.id();
        outer.Phase("native_write");
        {
            StoreTrace inner("batch_put", &sink, 3);
            EXPECT_NE(inner.id(), parent);
            inner.Phase("master_start");
            inner.Field("bytes", 1234);
            inner.Result("success");
        }
        EXPECT_EQ(StoreTrace::CurrentId(), parent);
        std::thread worker([&] {
            StoreTrace background("dfs_background_write", &sink, 3, parent);
            background.Phase("dfs_io");
            background.Result("success");
        });
        worker.join();
        outer.Result("success");
    }
    EXPECT_EQ(StoreTrace::CurrentId(), 0);
    const auto log = sink.Text();
    EXPECT_NE(log.find("parent=" + std::to_string(parent)), std::string::npos);
    EXPECT_NE(log.find("bytes=1234"), std::string::npos);
    EXPECT_NE(log.find("master_start_us="), std::string::npos);
    EXPECT_NE(log.find("dfs_io_us="), std::string::npos);
}

TEST(StoreTraceTest, StuckCallsAreReportedWithoutEndAndRateLimited) {
    TraceSink sink;
    {
        StoreTrace trace("stuck", &sink, 1);
        trace.Phase("dfs_stream_map_lock");
        auto now = StoreTrace::Clock::now() + std::chrono::seconds(6);
        StoreTrace::ReportStalls(now);
        StoreTrace::ReportStalls(now);
        const auto log = sink.Text();
        auto first = log.find("STORE_TRACE_STALL");
        ASSERT_NE(first, std::string::npos);
        EXPECT_EQ(log.find("STORE_TRACE_STALL", first + 1), std::string::npos);
        EXPECT_NE(log.find("phase=dfs_stream_map_lock"), std::string::npos);
        EXPECT_EQ(log.find("STORE_TRACE_END"), std::string::npos);
    }
    auto previous = sink.Text();
    StoreTrace::ReportStalls(StoreTrace::Clock::now() +
                             std::chrono::seconds(7));
    EXPECT_EQ(sink.Text(), previous);
}

TEST(StoreTraceTest, ExceptionRestoresThreadStateAndQuietCallsDoNotFloodLogs) {
    TraceSink sink;
    try {
        StoreTrace trace("exception", &sink, 1);
        trace.Phase("copy");
        throw std::runtime_error("test");
    } catch (const std::runtime_error&) {
    }
    EXPECT_EQ(StoreTrace::CurrentId(), 0);
    EXPECT_NE(sink.Text().find("result=exception"), std::string::npos);
    auto before = sink.Text();
    for (int i = 0; i < 20; ++i) {
        StoreTrace quiet("metadata", &sink, 1, 0, true, false);
        quiet.Phase("query");
        quiet.Result("success");
    }
    EXPECT_EQ(sink.Text(), before);
}

TEST(StoreTraceTest, CodesAndRepeatedPhasesAreAggregated) {
    TraceSink sink;
    {
        StoreTrace trace("read", &sink, 3);
        trace.Phase("pointer_query");
        trace.Phase("gpu_copy");
        trace.Phase("pointer_query");
        trace.Phase("gpu_copy");
        trace.Codes({0, 1234, -1});
    }
    const auto log = sink.Text();
    EXPECT_NE(log.find("failures=1 first_error=-1"), std::string::npos);
    auto first = log.find("pointer_query_us=");
    ASSERT_NE(first, std::string::npos);
    EXPECT_EQ(log.find("pointer_query_us=", first + 1), std::string::npos);
}

TEST(StoreTraceTest, UnboundBackgroundAndConcurrentSnapshotsAreSafe) {
    TraceSink sink;
    StoreTrace foreground("foreground", &sink, 1);
    auto background = std::make_shared<StoreTrace>("background", &sink, 1,
                                                   foreground.id(), false);
    EXPECT_EQ(StoreTrace::CurrentId(), foreground.id());
    std::thread worker([&] {
        for (int i = 0; i < 100; ++i) background->Phase("write");
        background->Result("success");
    });
    for (int i = 0; i < 100; ++i) StoreTrace::ReportStalls();
    worker.join();
    background.reset();
    EXPECT_EQ(StoreTrace::CurrentId(), foreground.id());
    foreground.Result("success");
}

TEST(StoreTraceTest, DisabledModeDoesNotRegisterOrLog) {
    if (StoreTrace::Enabled())
        GTEST_SKIP() << "Run with MC_STORE_DFS_DIAGNOSTICS=0";
    TraceSink sink;
    {
        StoreTrace trace("disabled", &sink, 1);
        trace.Phase("copy");
        trace.Field("bytes", 1234);
        EXPECT_EQ(trace.id(), 0);
        EXPECT_EQ(StoreTrace::CurrentId(), 0);
        StoreTrace::ReportStalls(StoreTrace::Clock::now() +
                                 std::chrono::seconds(20));
    }
    EXPECT_TRUE(sink.Text().empty());
}

}  // namespace
}  // namespace mooncake::test
