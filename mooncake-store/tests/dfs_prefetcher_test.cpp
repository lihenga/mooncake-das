// Unit tests for DfsPrefetcher: replica filtering, dedup/backoff, capacity
// accounting, TryConsume semantics, and TTL GC. Dependencies are injected as
// std::function fakes; buffers are allocated from a real
// ClientBufferAllocator.

#include "dfs_prefetcher.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

namespace mooncake {
namespace {

using namespace std::chrono_literals;

constexpr uint64_t kObjSize = 4096;

Replica::Descriptor MakeDfs(uint64_t size = kObjSize) {
    Replica::Descriptor d;
    d.id = 0;
    d.descriptor_variant = DistributedFSDescriptor{"/dfs/key", 0, size, size,
                                                   0};
    d.status = ReplicaStatus::COMPLETE;
    return d;
}

Replica::Descriptor MakeMemory(const std::string &endpoint,
                               uint64_t size = kObjSize) {
    Replica::Descriptor d;
    d.id = 0;
    MemoryDescriptor mem;
    mem.buffer_descriptor.size_ = size;
    mem.buffer_descriptor.buffer_address_ = 0x1000;
    mem.buffer_descriptor.protocol_ = "tcp";
    mem.buffer_descriptor.transport_endpoint_ = endpoint;
    d.descriptor_variant = mem;
    d.status = ReplicaStatus::COMPLETE;
    return d;
}

Replica::Descriptor MakeLocalDisk(const std::string &endpoint,
                                  uint64_t size = kObjSize) {
    Replica::Descriptor d;
    d.id = 0;
    LocalDiskDescriptor local_disk;
    local_disk.object_size = size;
    local_disk.transport_endpoint = endpoint;
    d.descriptor_variant = local_disk;
    d.status = ReplicaStatus::COMPLETE;
    return d;
}

QueryResult MakeQuery(std::vector<Replica::Descriptor> replicas) {
    return QueryResult(std::move(replicas),
                       std::chrono::steady_clock::now() + 60s);
}

class DfsPrefetcherTest : public ::testing::Test {
   protected:
    void SetUp() override {
        allocator_ = ClientBufferAllocator::create(64ull << 20, "tcp");
        ASSERT_NE(allocator_, nullptr);
        config_.max_bytes = 8 * kObjSize;
        config_.ttl_ms = 60000;
        config_.io_threads = 2;
        config_.wait_timeout_ms = 2000;
        config_.retry_backoff_ms = 300;
        config_.max_batch_keys = 16;
    }

    DfsPrefetcher::BatchQueryFn QueryReturning(
        std::vector<Replica::Descriptor> replicas) {
        return [replicas = std::move(replicas)](
                   const std::vector<std::string> &keys) mutable {
            std::vector<tl::expected<QueryResult, ErrorCode>> out;
            out.reserve(keys.size());
            for (size_t i = 0; i < keys.size(); ++i) {
                out.emplace_back(MakeQuery(replicas));
            }
            return out;
        };
    }

    // BatchGet that fills every destination slice with 'P'.
    DfsPrefetcher::BatchGetFn GetOk() {
        return [](const std::vector<std::string> &keys,
                  const std::vector<QueryResult> &,
                  std::unordered_map<std::string, std::vector<Slice>> &slices) {
            std::vector<tl::expected<void, ErrorCode>> out;
            out.reserve(keys.size());
            for (const auto &key : keys) {
                for (auto &slice : slices[key]) {
                    std::memset(slice.ptr, 'P', slice.size);
                }
                out.emplace_back();
            }
            return out;
        };
    }

    DfsPrefetcher::LocalEndpointsFn NoLocal() {
        return [] { return std::unordered_set<std::string>{}; };
    }

    DfsPrefetcher::AllocateFn Alloc() {
        return [this](size_t size) { return allocator_->allocate(size); };
    }

    // Poll `cond` up to `timeout`, sleeping briefly between checks. Returns
    // the last value of cond().
    template <typename F>
    static bool WaitFor(F &&cond, std::chrono::milliseconds timeout =
                                      std::chrono::milliseconds(3000)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (cond()) {
                return true;
            }
            std::this_thread::sleep_for(2ms);
        }
        return cond();
    }

    DfsPrefetchConfig config_;
    std::shared_ptr<ClientBufferAllocator> allocator_;
};

// --- Filtering -------------------------------------------------------------

TEST_F(DfsPrefetcherTest, OnlyDfsBestReplicaIsPrefetched) {
    DfsPrefetcher p(config_,
                    QueryReturning({MakeMemory("remote-node"), MakeDfs()}),
                    GetOk(), NoLocal(), Alloc());
    p.NotifyExistTrue({"k1"});
    bool found = false;
    auto handle = p.TryConsume("k1", found);
    ASSERT_TRUE(found);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(handle->size(), kObjSize);
    EXPECT_EQ(static_cast<char *>(handle->ptr())[0], 'P');
}

TEST_F(DfsPrefetcherTest, MemoryOnlyKeyIsNotPrefetched) {
    std::atomic<int> query_calls{0};
    auto query = [&query_calls](const std::vector<std::string> &keys) {
        ++query_calls;
        std::vector<tl::expected<QueryResult, ErrorCode>> out;
        for (size_t i = 0; i < keys.size(); ++i) {
            out.emplace_back(MakeQuery({MakeMemory("remote-node")}));
        }
        return out;
    };
    DfsPrefetcher p(config_, query, GetOk(), NoLocal(), Alloc());
    p.NotifyExistTrue({"k1"});
    // Wait until the coordinator has definitely processed the notification
    // (it issues a BatchQuery even though the key ends up filtered out).
    ASSERT_TRUE(WaitFor([&query_calls] { return query_calls.load() > 0; }));
    bool found = false;
    EXPECT_EQ(p.TryConsume("k1", found), nullptr);
    EXPECT_FALSE(found);
}

TEST_F(DfsPrefetcherTest, LocalDiskPreferredOverDfs) {
    std::atomic<int> query_calls{0};
    auto query = [&query_calls](const std::vector<std::string> &keys) {
        ++query_calls;
        std::vector<tl::expected<QueryResult, ErrorCode>> out;
        for (size_t i = 0; i < keys.size(); ++i) {
            out.emplace_back(MakeQuery({MakeLocalDisk("peer"), MakeDfs()}));
        }
        return out;
    };
    DfsPrefetcher p(config_, query, GetOk(), NoLocal(), Alloc());
    p.NotifyExistTrue({"k1"});
    ASSERT_TRUE(WaitFor([&query_calls] { return query_calls.load() > 0; }));
    bool found = false;
    EXPECT_EQ(p.TryConsume("k1", found), nullptr);
    EXPECT_FALSE(found);
}

// --- Dedup / backoff -------------------------------------------------------

TEST_F(DfsPrefetcherTest, DuplicateNotifyIsDeduplicated) {
    std::atomic<int> get_calls{0};
    auto get = [&get_calls](const std::vector<std::string> &keys,
                            const std::vector<QueryResult> &,
                            std::unordered_map<std::string, std::vector<Slice>>
                                &slices) {
        ++get_calls;
        std::vector<tl::expected<void, ErrorCode>> out;
        for (const auto &key : keys) {
            for (auto &slice : slices[key]) {
                std::memset(slice.ptr, 'P', slice.size);
            }
            out.emplace_back();
        }
        return out;
    };
    DfsPrefetcher p(config_, QueryReturning({MakeDfs()}), get, NoLocal(),
                    Alloc());
    p.NotifyExistTrue({"k1"});
    p.NotifyExistTrue({"k1"});
    p.NotifyExistTrue({"k1"});

    bool found = false;
    auto handle = p.TryConsume("k1", found);
    ASSERT_TRUE(found);
    ASSERT_NE(handle, nullptr);
    // Only one IO batch should have been issued for the key.
    EXPECT_EQ(get_calls.load(), 1);
}

TEST_F(DfsPrefetcherTest, FailedKeyRespectsBackoffThenRetries) {
    std::atomic<int> get_calls{0};
    auto flaky_get = [&get_calls](
                         const std::vector<std::string> &keys,
                         const std::vector<QueryResult> &,
                         std::unordered_map<std::string, std::vector<Slice>>
                             &slices) {
        const int call = ++get_calls;
        if (call == 1) {
            return std::vector<tl::expected<void, ErrorCode>>(
                keys.size(), tl::unexpected(ErrorCode::INTERNAL_ERROR));
        }
        std::vector<tl::expected<void, ErrorCode>> out;
        for (const auto &key : keys) {
            for (auto &slice : slices[key]) {
                std::memset(slice.ptr, 'Q', slice.size);
            }
            out.emplace_back();
        }
        return out;
    };
    DfsPrefetcher p(config_, QueryReturning({MakeDfs()}), flaky_get, NoLocal(),
                    Alloc());

    p.NotifyExistTrue({"k1"});
    // Wait for the first (failing) read to complete: TryConsume finds the
    // FAILED entry (found=true) and returns nullptr.
    ASSERT_TRUE(WaitFor([&p] {
        bool found = false;
        p.TryConsume("k1", found);
        return found;
    }));
    EXPECT_EQ(get_calls.load(), 1);

    // Immediate retry is suppressed by backoff.
    p.NotifyExistTrue({"k1"});
    std::this_thread::sleep_for(100ms);
    EXPECT_EQ(get_calls.load(), 1);

    // After the backoff window the retry goes through and succeeds.
    std::this_thread::sleep_for(350ms);
    p.NotifyExistTrue({"k1"});
    bool found = false;
    auto handle = p.TryConsume("k1", found);
    ASSERT_TRUE(found);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(static_cast<char *>(handle->ptr())[0], 'Q');
    EXPECT_EQ(get_calls.load(), 2);
}

// --- Capacity --------------------------------------------------------------

TEST_F(DfsPrefetcherTest, CapacityLimitSkipsNewPrefetches) {
    DfsPrefetcher p(config_, QueryReturning({MakeDfs()}), GetOk(), NoLocal(),
                    Alloc());
    // max_bytes = 8 objects; ask for 16.
    std::vector<std::string> keys;
    for (int i = 0; i < 16; ++i) {
        keys.push_back("k" + std::to_string(i));
    }
    p.NotifyExistTrue(keys);

    int hits = 0;
    int misses = 0;
    for (const auto &key : keys) {
        bool found = false;
        auto handle = p.TryConsume(key, found);
        if (found && handle) {
            ++hits;
        } else if (!found) {
            ++misses;
        }
    }
    EXPECT_LE(hits, 8);
    EXPECT_GT(misses, 0);
    EXPECT_EQ(hits + misses, 16);
}

// --- TryConsume semantics --------------------------------------------------

TEST_F(DfsPrefetcherTest, ConsumeWaitsForInflightRead) {
    std::atomic<bool> release_read{false};
    std::atomic<bool> read_started{false};
    auto slow_get = [&release_read, &read_started](
                        const std::vector<std::string> &keys,
                        const std::vector<QueryResult> &,
                        std::unordered_map<std::string, std::vector<Slice>>
                            &slices) {
        read_started.store(true, std::memory_order_release);
        // Block until the test releases us.
        while (!release_read.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(2ms);
        }
        std::vector<tl::expected<void, ErrorCode>> out;
        for (const auto &key : keys) {
            for (auto &slice : slices[key]) {
                std::memset(slice.ptr, 'R', slice.size);
            }
            out.emplace_back();
        }
        return out;
    };
    DfsPrefetcher p(config_, QueryReturning({MakeDfs()}), slow_get, NoLocal(),
                    Alloc());
    p.NotifyExistTrue({"k1"});

    // Ensure the read is actually in-flight before consuming.
    ASSERT_TRUE(WaitFor([&read_started] {
        return read_started.load(std::memory_order_acquire);
    }));

    const auto start = std::chrono::steady_clock::now();
    std::thread releaser([&release_read] {
        std::this_thread::sleep_for(200ms);
        release_read.store(true, std::memory_order_release);
    });
    bool found = false;
    auto handle = p.TryConsume("k1", found);
    releaser.join();
    const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    ASSERT_TRUE(found);
    ASSERT_NE(handle, nullptr);
    EXPECT_GE(waited.count(), 150);  // actually waited for the read
    EXPECT_EQ(static_cast<char *>(handle->ptr())[0], 'R');
}

TEST_F(DfsPrefetcherTest, ConsumeWaitTimeoutFallsBack) {
    config_.wait_timeout_ms = 100;
    std::atomic<bool> release_read{false};
    std::atomic<bool> read_started{false};
    auto blocking_get = [&release_read, &read_started](
                            const std::vector<std::string> &keys,
                            const std::vector<QueryResult> &,
                            std::unordered_map<std::string,
                                               std::vector<Slice>> &) {
        read_started.store(true, std::memory_order_release);
        while (!release_read.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(2ms);
        }
        return std::vector<tl::expected<void, ErrorCode>>(keys.size());
    };
    DfsPrefetcher p(config_, QueryReturning({MakeDfs()}), blocking_get,
                    NoLocal(), Alloc());
    p.NotifyExistTrue({"k1"});
    ASSERT_TRUE(WaitFor([&read_started] {
        return read_started.load(std::memory_order_acquire);
    }));

    const auto start = std::chrono::steady_clock::now();
    bool found = false;
    auto handle = p.TryConsume("k1", found);
    const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    EXPECT_TRUE(found);
    EXPECT_EQ(handle, nullptr);
    EXPECT_GE(waited.count(), 90);
    EXPECT_LT(waited.count(), 2000);

    // Unblock the read so Shutdown() can drain the pool.
    release_read.store(true, std::memory_order_release);
}

// --- TTL GC ----------------------------------------------------------------

TEST_F(DfsPrefetcherTest, ExpiredEntryIsGarbageCollected) {
    config_.ttl_ms = 150;
    std::atomic<int> get_calls{0};
    auto counting_get = [&get_calls](
                            const std::vector<std::string> &keys,
                            const std::vector<QueryResult> &,
                            std::unordered_map<std::string, std::vector<Slice>>
                                &slices) {
        ++get_calls;
        std::vector<tl::expected<void, ErrorCode>> out;
        for (const auto &key : keys) {
            for (auto &slice : slices[key]) {
                std::memset(slice.ptr, 'P', slice.size);
            }
            out.emplace_back();
        }
        return out;
    };
    DfsPrefetcher p(config_, QueryReturning({MakeDfs()}), counting_get,
                    NoLocal(), Alloc());
    p.NotifyExistTrue({"k1"});

    // Wait for the read to complete, then do NOT consume.
    ASSERT_TRUE(WaitFor([&get_calls] { return get_calls.load() > 0; }));

    // GC runs in the coordinator loop (~50ms cadence); the READY entry is
    // erased once it is older than ttl_ms (150ms). 500ms covers read latency
    // + TTL + GC cadence with margin.
    std::this_thread::sleep_for(500ms);
    bool found = false;
    EXPECT_EQ(p.TryConsume("k1", found), nullptr);
    EXPECT_FALSE(found);
}

TEST_F(DfsPrefetcherTest, ShutdownReleasesEverything) {
    std::atomic<int> get_calls{0};
    auto counting_get = [&get_calls](
                            const std::vector<std::string> &keys,
                            const std::vector<QueryResult> &,
                            std::unordered_map<std::string, std::vector<Slice>>
                                &slices) {
        ++get_calls;
        return std::vector<tl::expected<void, ErrorCode>>(keys.size());
    };
    DfsPrefetcher p(config_, QueryReturning({MakeDfs()}), counting_get,
                    NoLocal(), Alloc());
    p.NotifyExistTrue({"k1", "k2", "k3"});
    // Wait until the coordinator allocated buffers and submitted the reads,
    // so Shutdown actually exercises the in-flight cleanup path.
    ASSERT_TRUE(WaitFor([&get_calls] { return get_calls.load() > 0; }));
    p.Shutdown();
    // After shutdown the arena must be fully free again.
    auto alloc = allocator_->allocate(60ull << 20);
    EXPECT_TRUE(alloc.has_value());
}

}  // namespace
}  // namespace mooncake
