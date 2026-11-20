#include "read_plan.h"
#include "real_client.h"

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <functional>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>

namespace mooncake {
namespace {
using Keys = std::vector<std::string>;
using Rows = std::vector<std::vector<size_t>>;

class PipelineSetting {
   public:
    explicit PipelineSetting(bool enabled) {
        if (const char *value = std::getenv("MOONCAKE_READ_PLAN_PIPELINE"))
            previous = value;
        setenv("MOONCAKE_READ_PLAN_PIPELINE", enabled ? "1" : "0", 1);
    }
    ~PipelineSetting() {
        if (previous)
            setenv("MOONCAKE_READ_PLAN_PIPELINE", previous->c_str(), 1);
        else
            unsetenv("MOONCAKE_READ_PLAN_PIPELINE");
    }

   private:
    std::optional<std::string> previous;
};

class LeaseClient : public RealClient {
   public:
    struct Call {
        Keys keys;
        Rows sizes, offsets;
    };
    std::function<std::vector<int>(const Call &, size_t)> read;
    std::vector<Call> calls;
    std::vector<Keys> starts, ends;
    std::atomic<int> active_reads{0}, overlapping_starts{0};
    std::vector<int> start_result;
    std::mutex mutex;

    std::vector<int> batch_get_session_start(const Keys &keys) override {
        if (active_reads) ++overlapping_starts;
        std::lock_guard lock(mutex);
        starts.push_back(keys);
        return start_result.empty() ? std::vector<int>(keys.size(), 0)
                                    : start_result;
    }
    GetSessionStartResult batch_get_session_start_with_sources(
        const Keys &keys) override {
        return {batch_get_session_start(keys),
                std::vector<std::string>(keys.size(), "dfs"), false};
    }
    int batch_get_session_end(const Keys &keys) override {
        std::lock_guard lock(mutex);
        ends.push_back(keys);
        return 0;
    }
    std::vector<int> batch_get_into_multi_buffer_ranges(
        const Keys &keys, const std::vector<std::vector<void *>> &,
        const Rows &sizes, const Rows &offsets) override {
        Call call{keys, sizes, offsets};
        size_t index;
        {
            std::lock_guard lock(mutex);
            index = calls.size();
            calls.push_back(call);
        }
        ++active_reads;
        auto result = read ? read(call, index) : success(call);
        --active_reads;
        return result;
    }
    static std::vector<int> success(const Call &call) {
        std::vector<int> result;
        for (const auto &sizes : call.sizes)
            result.push_back(std::accumulate(sizes.begin(), sizes.end(), 0));
        return result;
    }
};

std::vector<ReadLayout> layouts(int groups,
                                const Keys &keys = {"a", "b", "a"}) {
    std::vector<std::vector<ReadComponent>> components;
    for (int group = 0; group < groups; ++group)
        components.push_back({ReadComponent{size_t(0x10000 + group * 64), 4, 4,
                                            size_t(group * 4)}});
    std::vector<size_t> rows(keys.size());
    std::iota(rows.begin(), rows.end(), 0);
    return {{keys, rows, true, components}};
}

TEST(ReadPlanLeaseTest, RetriesCurrentAndLegacyExpiryCodes) {
    PipelineSetting pipeline(false);
    for (int code : {toInt(ErrorCode::SESSION_RANGE_LEASE_EXPIRED),
                     toInt(ErrorCode::SESSION_MEMORY_TRANSFER_LEASE_EXPIRED),
                     toInt(ErrorCode::LEASE_EXPIRED)}) {
        auto client = std::make_shared<LeaseClient>();
        client->read = [code](const auto &call, size_t index) {
            return index == 1 ? std::vector<int>{code, 4, -600}
                              : LeaseClient::success(call);
        };
        ReadPlan plan(client, layouts(3), 3, false, false, true);
        EXPECT_NO_THROW(plan.run());
        ASSERT_EQ(client->starts, std::vector<Keys>{{"a"}});
        ASSERT_EQ(client->calls.size(), 4u);
        EXPECT_EQ(client->calls[2].keys, (Keys{"a", "a"}));
        EXPECT_EQ(client->calls[2].offsets, (Rows{{4}, {4}}));
        EXPECT_EQ(plan.stats(), (std::vector<uint64_t>{3, 9, 36}));
        EXPECT_TRUE(client->ends.empty());
    }
}

TEST(ReadPlanLeaseTest, RetriesMissingSessionsWithBoundedRenewals) {
    auto client = std::make_shared<LeaseClient>();
    client->read = [](const auto &call, size_t index) {
        return index < 2 ? std::vector<int>{-600, -600, -600}
                         : LeaseClient::success(call);
    };
    ReadPlan plan(client, layouts(1), 1, false, false, true);
    EXPECT_NO_THROW(plan.run());
    EXPECT_EQ(client->starts, (std::vector<Keys>{{"a", "b"}, {"a", "b"}}));
}

TEST(ReadPlanLeaseTest, StopsAfterFourRenewalsAndDoesNotPublishReadiness) {
    auto client = std::make_shared<LeaseClient>();
    client->read = [](const auto &call, size_t) {
        std::vector<int> result;
        bool expired = false;
        for (const auto &key : call.keys) {
            result.push_back(key == "b" ? 4 : expired ? -600 : -70706);
            expired |= key == "a";
        }
        return result;
    };
    ReadPlan plan(client, layouts(1), 1, false, false, true);
    try {
        plan.run();
        FAIL() << "expired ranges must fail after bounded retries";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("rc=-70706"),
                  std::string::npos);
    }
    EXPECT_EQ(client->starts.size(), 4u);
    EXPECT_EQ(client->calls.size(), 5u);
    EXPECT_THROW(plan.wait(0), std::runtime_error);
}

TEST(ReadPlanLeaseTest, DoesNotRetryOtherErrorsShortReadsOrMalformedResults) {
    for (const auto &result : std::vector<std::vector<int>>{
             {-70706, -800, -600}, {3, 4, 4}, {4, 4}, {}}) {
        auto client = std::make_shared<LeaseClient>();
        client->read = [result](const auto &, size_t) { return result; };
        ReadPlan plan(client, layouts(1), 1, false, false, true);
        EXPECT_THROW(plan.run(), std::runtime_error);
        EXPECT_TRUE(client->starts.empty());
        EXPECT_EQ(client->calls.size(), 1u);
    }
}

TEST(ReadPlanLeaseTest, RejectsMalformedRetryResults) {
    auto client = std::make_shared<LeaseClient>();
    client->read = [](const auto &, size_t index) {
        return index == 0 ? std::vector<int>{-70706, 4, -600}
                          : std::vector<int>{4};
    };
    ReadPlan plan(client, layouts(1), 1, false, false, true);
    EXPECT_THROW(plan.run(), std::runtime_error);
    EXPECT_THROW(plan.wait(0), std::runtime_error);
}

TEST(ReadPlanLeaseTest, FailedRenewalPreservesSessionOwnership) {
    for (bool borrowed : {false, true}) {
        auto client = std::make_shared<LeaseClient>();
        client->read = [client](const auto &, size_t) {
            client->start_result = {-704};
            return std::vector<int>{-70706};
        };
        ReadPlan plan(client, layouts(1, {"a"}), 1, false, false, borrowed);
        EXPECT_THROW(plan.run(), std::runtime_error);
        EXPECT_EQ(client->ends.size(), borrowed ? 0u : 1u);
        client->read = nullptr;
    }
}

TEST(ReadPlanLeaseTest, RenewsInPageWiseAndRangeReuseModes) {
    PipelineSetting pipeline(false);
    for (bool page_wise : {false, true}) {
        auto client = std::make_shared<LeaseClient>();
        client->read = [](const auto &call, size_t index) {
            return index == 0 ? std::vector<int>(call.keys.size(), -70706)
                              : LeaseClient::success(call);
        };
        ReadPlan plan(client, layouts(2), 2, !page_wise, page_wise, true);
        EXPECT_NO_THROW(plan.run());
        EXPECT_EQ(client->starts.size(), 1u);
        EXPECT_EQ(plan.stats()[2], 24u);
        EXPECT_NO_THROW(plan.wait(1));
    }
}

TEST(ReadPlanLeaseTest, HealthyReadsDoNotRenewSessions) {
    auto client = std::make_shared<LeaseClient>();
    ReadPlan plan(client, layouts(2), 2, false, false, true);
    EXPECT_NO_THROW(plan.run());
    EXPECT_TRUE(client->starts.empty());
}

TEST(ReadPlanLeaseTest, PipelineDrainsOldReadsBeforeRenewal) {
    PipelineSetting pipeline(true);
    auto client = std::make_shared<LeaseClient>();
    std::barrier initial_reads(2);
    client->read = [&](const auto &call, size_t index) {
        if (index < 2) {
            initial_reads.arrive_and_wait();
            if (call.offsets[0][0] == 4)
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            return std::vector<int>{-70706, 4, -600};
        }
        return LeaseClient::success(call);
    };
    ReadPlan plan(client, layouts(2), 2, false, false, true);
    EXPECT_NO_THROW(plan.run());
    EXPECT_EQ(client->overlapping_starts, 0);
    EXPECT_EQ(client->starts.size(), 2u);
    EXPECT_NO_THROW(plan.wait(1));
}
}  // namespace
}  // namespace mooncake
