#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "dfs_write_staging_pool.h"

namespace mooncake::test {

TEST(DfsWriteStagingPoolTest, AllocatesActualChunkSizes) {
    auto pool = std::make_shared<DfsWriteStagingPool>(16, "");
    auto first = pool->Acquire(4, std::chrono::milliseconds(1));
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->capacity(), 4u);

    auto second = pool->Acquire(8, std::chrono::milliseconds(1));
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->capacity(), 8u);

    auto too_large = pool->Acquire(5, std::chrono::milliseconds(1));
    EXPECT_EQ(too_large, nullptr);
}

TEST(DfsWriteStagingPoolTest, WaitsForReleasedBytesWithoutBusyPolling) {
    auto pool = std::make_shared<DfsWriteStagingPool>(16, "");
    auto first = pool->Acquire(16, std::chrono::milliseconds(1));
    ASSERT_NE(first, nullptr);

    std::atomic<bool> acquired{false};
    std::thread waiter([&] {
        auto second = pool->Acquire(8, std::chrono::seconds(1));
        acquired.store(second != nullptr, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_FALSE(acquired.load(std::memory_order_acquire));
    first.reset();
    waiter.join();
    EXPECT_TRUE(acquired.load(std::memory_order_acquire));
}

TEST(DfsWriteStagingPoolTest, ReturnedAllocationCanBeReused) {
    auto pool = std::make_shared<DfsWriteStagingPool>(16, "");
    auto first = pool->Acquire(8, std::chrono::milliseconds(1));
    ASSERT_NE(first, nullptr);
    char* first_address = first->data();
    first.reset();

    auto second = pool->Acquire(8, std::chrono::milliseconds(1));
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->data(), first_address);
}

TEST(DfsWriteStagingPoolTest, ShutdownWakesWaitingAcquire) {
    auto pool = std::make_shared<DfsWriteStagingPool>(16, "");
    auto first = pool->Acquire(16, std::chrono::milliseconds(1));
    ASSERT_NE(first, nullptr);

    std::atomic<bool> finished{false};
    std::thread waiter([&] {
        bool timed_out = false;
        auto second = pool->Acquire(1, std::chrono::seconds(10), &timed_out);
        EXPECT_EQ(second, nullptr);
        EXPECT_FALSE(timed_out);
        finished.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    pool->Shutdown();
    waiter.join();
    EXPECT_TRUE(finished.load(std::memory_order_acquire));
}

}  // namespace mooncake::test
