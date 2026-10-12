#include "checksum_timing.h"

#include <gtest/gtest.h>
#include <string>

namespace mooncake {

class ChecksumTimingTestPeer {
   public:
    static auto& Counters(ChecksumStage stage) {
        return ChecksumTiming::Get().stages[static_cast<size_t>(stage)];
    }
    static void Elapse(ChecksumTiming& timing, uint64_t ns) {
        timing.started_ = ChecksumTiming::Now() - ns;
        ChecksumTiming::Get().next_report =
            ChecksumTiming::Now() + 60000000000ULL;
    }
};

class SlowLogSink : public google::LogSink {
   public:
    void send(google::LogSeverity, const char*, const char*, int, const tm*,
              const char* message, size_t length) override {
        if (std::string(message, length).find("checksum_slow ") !=
            std::string::npos)
            ++calls;
    }
    int calls = 0;
};

TEST(ChecksumTimingTest, BatchCountersAndRateLimit) {
    if (!ChecksumTiming::Enabled()) GTEST_SKIP();
    SlowLogSink sink;
    google::AddLogSink(&sink);
    auto& c = ChecksumTimingTestPeer::Counters(ChecksumStage::SESSION_PREFETCH);
    const auto calls = c.calls.load();
    for (int i = 0; i < 2; ++i) {
        ChecksumTiming timing(ChecksumStage::SESSION_PREFETCH, 4096, 3);
        ChecksumTimingTestPeer::Elapse(timing, 1100000000ULL);
        timing.AddLockWait(ChecksumTiming::Now() - 1000000);
        timing.Finish();
        timing.Finish();
    }
    EXPECT_EQ(c.calls.load() - calls, 2);
    EXPECT_EQ(c.objects.load(), 6);
    EXPECT_EQ(c.bytes.load(), 8192);
    EXPECT_EQ(c.active.load(), 0);
    EXPECT_EQ(c.active_bytes.load(), 0);
    EXPECT_EQ(c.slow_calls.load(), 2);
    EXPECT_EQ(c.over_1s_calls.load(), 2);
    EXPECT_GE(c.lock_wait_ns.load(), 2000000);
    EXPECT_EQ(sink.calls, 1);
    // Each stage has its own limiter, which can expire without sleeping.
    c.next_slow_log_ns = 0;
    {
        ChecksumTiming timing(ChecksumStage::SESSION_PREFETCH, 0, 1);
        ChecksumTimingTestPeer::Elapse(timing, 600000000ULL);
    }
    {
        ChecksumTiming timing(ChecksumStage::BATCH_QUERY, 0, 1);
        ChecksumTimingTestPeer::Elapse(timing, 600000000ULL);
    }
    EXPECT_EQ(sink.calls, 3);
    google::RemoveLogSink(&sink);
}

TEST(ChecksumTimingTest, PerKeyAndShortBatchDoNotLog) {
    if (!ChecksumTiming::Enabled()) GTEST_SKIP();
    SlowLogSink sink;
    google::AddLogSink(&sink);
    {
        ChecksumTiming timing(ChecksumStage::PREFETCH_CHECKSUM, 16);
        ChecksumTimingTestPeer::Elapse(timing, 1100000000ULL);
    }
    {
        ChecksumTiming timing(ChecksumStage::SESSION_RANGE_GET, 0, 1);
        ChecksumTimingTestPeer::Elapse(timing, 200000000ULL);
    }
    EXPECT_EQ(sink.calls, 0);
    EXPECT_EQ(ChecksumTimingTestPeer::Counters(ChecksumStage::SESSION_RANGE_GET)
                  .slow_calls.load(),
              1);
    EXPECT_EQ(ChecksumTimingTestPeer::Counters(ChecksumStage::SESSION_RANGE_GET)
                  .over_1s_calls.load(),
              0);
    google::RemoveLogSink(&sink);
}

TEST(ChecksumTimingTest, DisabledDoesNotRecord) {
    if (ChecksumTiming::Enabled()) GTEST_SKIP();
    {
        ChecksumTiming timing(ChecksumStage::DFS_SCATTER, 8192, 2);
        EXPECT_EQ(timing.StartPart(), 0);
        timing.AddLockWait(0);
        timing.Finish();
    }
    const auto& c =
        ChecksumTimingTestPeer::Counters(ChecksumStage::DFS_SCATTER);
    EXPECT_EQ(c.calls.load(), 0);
    EXPECT_EQ(c.objects.load(), 0);
    EXPECT_EQ(c.active.load(), 0);
}

}  // namespace mooncake
