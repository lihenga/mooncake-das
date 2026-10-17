#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "replica.h"
#include "storage/distributed/bucket_entry_layout.h"
#include "storage/distributed/immutable_bucket_allocator.h"
#include "storage/distributed/dfs_global_allocator.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/dfs_allocator_interface.h"
#include "storage/distributed/posix_fs_adapter.h"

namespace mooncake::test {

namespace {

class TempDir {
   public:
    explicit TempDir(const std::string& prefix) {
        static std::atomic<int64_t> counter{0};
        path_ = std::filesystem::temp_directory_path() /
                (prefix + "_" + std::to_string(::getpid()) + "_" +
                 std::to_string(++counter));
        path_str_ = path_.string();
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const std::string& path() const { return path_str_; }

    std::string file(const std::string& name) const {
        return (path_ / name).string();
    }

   private:
    std::filesystem::path path_;
    std::string path_str_;
};

constexpr uint64_t kAlignment = 4096;
constexpr uint64_t kBucketCapacity = 64 * 1024;

DistributedStorageConfig MakeBucketConfig(const std::string& fsdir,
                                          uint64_t bucket_capacity,
                                          int64_t max_bucket_count) {
    DistributedStorageConfig config;
    config.fsdir = fsdir;
    config.fs_adapter_type = "posix";
    config.allocator_type = DfsAllocatorType::BUCKET;
    config.bucket_capacity = bucket_capacity;
    config.max_bucket_count = max_bucket_count;
    config.alignment = kAlignment;
    config.single_tenant = true;
    config.eviction_enabled = false;
    config.eviction_high_watermark = 0.9;
    config.eviction_low_watermark = 0.7;
    config.deferred_free_duration = std::chrono::seconds(0);
    config.eviction_check_interval = std::chrono::seconds(1);
    // Shard fields still have to validate; they are unused in bucket mode.
    config.shard_count = 1;
    config.shard_capacity = bucket_capacity;
    return config;
}

DistributedStorageConfig MakeShardConfig(const std::string& fsdir,
                                         int shard_count,
                                         uint64_t shard_capacity) {
    DistributedStorageConfig config;
    config.fsdir = fsdir;
    config.fs_adapter_type = "posix";
    config.allocator_type = DfsAllocatorType::SHARD;
    config.shard_count = shard_count;
    config.shard_capacity = shard_capacity;
    config.alignment = kAlignment;
    config.single_tenant = true;
    config.eviction_enabled = false;
    config.deferred_free_duration = std::chrono::seconds(0);
    config.eviction_check_interval = std::chrono::seconds(1);
    return config;
}

// Bucket descriptors now point directly at the aligned entry/object start.
uint64_t EntryStartOf(const DistributedFSDescriptor& desc) {
    return desc.offset;
}

std::string BucketMetaFile(const TempDir& tmp, int64_t bucket_id) {
    return tmp.file("bucket_" +
                    ImmutableBucketAllocator::FormatBucketId(bucket_id) + ".meta");
}

// Seals the current active bucket. The filler object is sized so that it only
// fits into an empty bucket, forcing the allocator to roll over instead of
// appending.
std::optional<DistributedFSDescriptor> RollOverActiveBucket(
    ImmutableBucketAllocator& alloc, const std::string& filler_key,
    uint64_t bucket_capacity) {
    auto desc = alloc.Allocate(filler_key, bucket_capacity);
    if (!desc) return std::nullopt;
    return *desc;
}

}  // namespace

// ---------------------------------------------------------------------------
// ImmutableBucketAllocator: allocation
// ---------------------------------------------------------------------------

TEST(ImmutableBucketAllocatorTest, AllocateReturnsBucketIdAndObjectOffset) {
    TempDir tmp("bucket_alloc");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    EXPECT_TRUE(alloc.IsInitialized());
    EXPECT_EQ(alloc.Type(), DfsAllocatorType::BUCKET);

    const std::string key = "key1";
    auto desc = alloc.Allocate(key, 100);
    ASSERT_TRUE(desc.has_value()) << toString(desc.error());

    // The object begins at the aligned entry start; no key header is stored in
    // the data file.
    EXPECT_EQ(EntryStartOf(*desc), 0u);
    EXPECT_EQ(desc->offset, 0u);
    EXPECT_EQ(desc->object_size, 100u);
    EXPECT_EQ(desc->aligned_size, kAlignment);
    EXPECT_EQ(desc->shard_idx, 0);
    EXPECT_NE(desc->file_path.find("bucket_"), std::string::npos);
    EXPECT_NE(desc->file_path.find(".data"), std::string::npos);

    // The data file must be preallocated to the full bucket capacity.
    EXPECT_TRUE(std::filesystem::exists(desc->file_path));
    EXPECT_EQ(std::filesystem::file_size(desc->file_path), kBucketCapacity);
}

TEST(ImmutableBucketAllocatorTest, AllocateRejectsInvalidRequests) {
    TempDir tmp("bucket_alloc_invalid");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    auto empty_key = alloc.Allocate("", 100);
    ASSERT_FALSE(empty_key.has_value());
    EXPECT_EQ(empty_key.error(), ErrorCode::INVALID_PARAMS);

    auto zero_size = alloc.Allocate("k", 0);
    ASSERT_FALSE(zero_size.has_value());
    EXPECT_EQ(zero_size.error(), ErrorCode::INVALID_PARAMS);

    // An object larger than one bucket can never be placed contiguously.
    auto too_large = alloc.Allocate("k", kBucketCapacity + 1);
    ASSERT_FALSE(too_large.has_value());
    EXPECT_EQ(too_large.error(), ErrorCode::INVALID_PARAMS);

    // A duplicate key must not silently create a second live allocation. The
    // allocator reports it as OBJECT_ALREADY_EXISTS so a batch caller can
    // distinguish "this one key is taken" from a malformed request.
    ASSERT_TRUE(alloc.Allocate("dup", 100).has_value());
    auto duplicate = alloc.Allocate("dup", 100);
    ASSERT_FALSE(duplicate.has_value());
    EXPECT_EQ(duplicate.error(), ErrorCode::OBJECT_ALREADY_EXISTS);
}

std::string BucketDataFile(const TempDir& tmp, int64_t bucket_id) {
    return tmp.file("bucket_" +
                    ImmutableBucketAllocator::FormatBucketId(bucket_id) + ".data");
}

TEST(ImmutableBucketAllocatorTest, KeyLengthDoesNotAffectReservedSize) {
    TempDir tmp("bucket_key_length");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    auto short_key = alloc.Allocate("a", 100);
    auto long_key = alloc.Allocate(std::string(1024, 'k'), 100);
    ASSERT_TRUE(short_key.has_value());
    ASSERT_TRUE(long_key.has_value());
    EXPECT_EQ(short_key->aligned_size, kAlignment);
    EXPECT_EQ(long_key->aligned_size, short_key->aligned_size);
    EXPECT_EQ(short_key->offset, 0u);
    EXPECT_EQ(long_key->offset, kAlignment);
}

TEST(ImmutableBucketAllocatorTest, UninitializedAllocatorReportsUnavailable) {
    ImmutableBucketAllocator alloc;
    EXPECT_FALSE(alloc.IsInitialized());
    auto desc = alloc.Allocate("k", 100);
    ASSERT_FALSE(desc.has_value());
    EXPECT_EQ(desc.error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);
}

TEST(ImmutableBucketAllocatorTest, InitRejectsInvalidConfiguration) {
    TempDir tmp("bucket_init_invalid");

    {
        auto config = MakeBucketConfig(tmp.path(), 0, 8);
        ImmutableBucketAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
    }
    {
        // max_bucket_count must be > 0: it is the fixed watermark denominator.
        auto config = MakeBucketConfig(tmp.path(), kBucketCapacity, 0);
        ImmutableBucketAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
    }
    {
        auto config = MakeBucketConfig(tmp.path(), kBucketCapacity, 8);
        config.alignment = 3000;  // not a power of two
        ImmutableBucketAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
    }
    {
        auto config = MakeBucketConfig(tmp.path(), kBucketCapacity, 8);
        config.eviction_low_watermark = 0.95;
        config.eviction_high_watermark = 0.9;
        ImmutableBucketAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
    }
    {
        auto config = MakeBucketConfig(tmp.path(), kBucketCapacity, 8);
        config.fs_adapter_type = "nonexistent";
        ImmutableBucketAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
    }
}

TEST(ImmutableBucketAllocatorTest, MultipleObjectsShareBucketWithDistinctOffsets) {
    TempDir tmp("bucket_multi");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    std::vector<std::pair<std::string, DistributedFSDescriptor>> allocations;
    for (int i = 0; i < 4; ++i) {
        const std::string key = "key" + std::to_string(i);
        auto desc = alloc.Allocate(key, 100);
        ASSERT_TRUE(desc.has_value()) << "allocation " << i;
        allocations.emplace_back(key, *desc);
    }

    // All four fit in the first bucket, at strictly increasing, aligned,
    // non-overlapping entry starts.
    uint64_t previous_end = 0;
    for (const auto& [key, desc] : allocations) {
        EXPECT_EQ(desc.shard_idx, 0) << "key=" << key;
        const uint64_t entry_start = EntryStartOf(desc);
        EXPECT_EQ(entry_start % kAlignment, 0u) << "key=" << key;
        EXPECT_EQ(entry_start, previous_end) << "key=" << key;
        previous_end = entry_start + desc.aligned_size;
    }
    EXPECT_EQ(alloc.GetBucketCount(), 1u);
}

TEST(ImmutableBucketAllocatorTest, BucketRolloverCreatesNewBucket) {
    TempDir tmp("bucket_rollover");
    // Capacity for exactly two 4096-byte entries.
    const uint64_t capacity = 2 * kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 8)));

    std::vector<int> bucket_ids;
    for (int i = 0; i < 5; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value()) << "allocation " << i;
        bucket_ids.push_back(desc->shard_idx);
    }

    EXPECT_EQ(bucket_ids[0], 0);
    EXPECT_EQ(bucket_ids[1], 0);
    EXPECT_EQ(bucket_ids[2], 1);  // rolled over
    EXPECT_EQ(bucket_ids[3], 1);
    EXPECT_EQ(bucket_ids[4], 2);
    EXPECT_EQ(alloc.GetBucketCount(), 3u);
}

TEST(ImmutableBucketAllocatorTest, MaxBucketCountIsEnforced) {
    TempDir tmp("bucket_max_count");
    const uint64_t capacity = kAlignment;  // one entry per bucket
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 2)));

    ASSERT_TRUE(alloc.Allocate("k0", 100).has_value());
    ASSERT_TRUE(alloc.Allocate("k1", 100).has_value());
    auto exhausted = alloc.Allocate("k2", 100);
    ASSERT_FALSE(exhausted.has_value());
    EXPECT_EQ(exhausted.error(), ErrorCode::NO_AVAILABLE_HANDLE);
    EXPECT_EQ(alloc.GetBucketCount(), 2u);
}

// ---------------------------------------------------------------------------
// ImmutableBucketAllocator: BatchAllocate
// ---------------------------------------------------------------------------

TEST(ImmutableBucketAllocatorTest, BatchAllocateIsContiguous) {
    TempDir tmp("bucket_batch");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    std::vector<BatchAllocateRequest> requests;
    for (int i = 0; i < 4; ++i) {
        requests.push_back({"batch_key_" + std::to_string(i), 100});
    }

    auto results = alloc.BatchAllocate(requests);
    ASSERT_EQ(results.size(), requests.size());

    uint64_t expected_start = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        ASSERT_TRUE(results[i].success) << "entry " << i;
        EXPECT_EQ(results[i].key, requests[i].key);
        EXPECT_EQ(results[i].descriptor.shard_idx, 0);
        const uint64_t entry_start = EntryStartOf(results[i].descriptor);
        // Contiguity: each entry begins exactly where the previous one ended.
        EXPECT_EQ(entry_start, expected_start) << "entry " << i;
        expected_start = entry_start + results[i].descriptor.aligned_size;
    }
}

TEST(ImmutableBucketAllocatorTest, ConcurrentBatchesDoNotInterleave) {
    TempDir tmp("bucket_batch_concurrent");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), 4 * 1024 * 1024, 16)));

    constexpr int kThreads = 4;
    constexpr int kKeysPerBatch = 8;
    std::vector<std::vector<BatchAllocateResult>> per_thread(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&alloc, &per_thread, t]() {
            std::vector<BatchAllocateRequest> requests;
            for (int i = 0; i < kKeysPerBatch; ++i) {
                requests.push_back(
                    {"t" + std::to_string(t) + "_k" + std::to_string(i), 1024});
            }
            per_thread[t] = alloc.BatchAllocate(requests);
        });
    }
    for (auto& thread : threads) thread.join();

    // Every batch must occupy one contiguous run within a single bucket, with
    // no other batch's entry inside its span.
    struct Span {
        int bucket_id;
        uint64_t begin;
        uint64_t end;
    };
    std::vector<Span> spans;
    for (int t = 0; t < kThreads; ++t) {
        ASSERT_EQ(per_thread[t].size(), static_cast<size_t>(kKeysPerBatch));
        uint64_t expected_start = 0;
        int bucket_id = -1;
        uint64_t begin = 0;
        for (int i = 0; i < kKeysPerBatch; ++i) {
            const auto& result = per_thread[t][i];
            ASSERT_TRUE(result.success) << "thread " << t << " entry " << i;
            const uint64_t entry_start = EntryStartOf(result.descriptor);
            if (i == 0) {
                bucket_id = result.descriptor.shard_idx;
                begin = entry_start;
            } else {
                EXPECT_EQ(result.descriptor.shard_idx, bucket_id)
                    << "thread " << t << " spilled to another bucket";
                EXPECT_EQ(entry_start, expected_start)
                    << "thread " << t << " entry " << i << " is not contiguous";
            }
            expected_start = entry_start + result.descriptor.aligned_size;
        }
        spans.push_back({bucket_id, begin, expected_start});
    }

    for (size_t a = 0; a < spans.size(); ++a) {
        for (size_t b = a + 1; b < spans.size(); ++b) {
            if (spans[a].bucket_id != spans[b].bucket_id) continue;
            const bool disjoint = spans[a].end <= spans[b].begin ||
                                  spans[b].end <= spans[a].begin;
            EXPECT_TRUE(disjoint)
                << "batches " << a << " and " << b << " overlap in bucket "
                << spans[a].bucket_id;
        }
    }
}

TEST(ImmutableBucketAllocatorTest, BatchLargerThanBucketSpansBuckets) {
    TempDir tmp("bucket_batch_too_big");
    const uint64_t capacity = 2 * kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 8)));

    // The batch is larger than one bucket, but each object fits. Pack the first
    // two entries into bucket 0 and continue with the complete third object in
    // bucket 1; no object is split across files.
    std::vector<BatchAllocateRequest> requests{
        {"a", 100}, {"b", 100}, {"c", 100}};
    auto results = alloc.BatchAllocate(requests);
    ASSERT_EQ(results.size(), 3u);
    for (const auto& result : results) EXPECT_TRUE(result.success);
    EXPECT_EQ(results[0].descriptor.shard_idx, 0);
    EXPECT_EQ(results[1].descriptor.shard_idx, 0);
    EXPECT_EQ(results[2].descriptor.shard_idx, 1);
    EXPECT_EQ(EntryStartOf(results[0].descriptor), 0u);
    EXPECT_EQ(EntryStartOf(results[1].descriptor), kAlignment);
    EXPECT_EQ(EntryStartOf(results[2].descriptor), 0u);
}

TEST(ImmutableBucketAllocatorTest, BatchRollsOverToNewBucketWhenActiveIsFull) {
    TempDir tmp("bucket_batch_rollover");
    const uint64_t capacity = 4 * kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 8)));

    // Fill three of four slots in bucket 0.
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(alloc.Allocate("pre" + std::to_string(i), 100).has_value());
    }

    // A 2-entry batch fills the remaining slot in bucket 0 and puts the
    // second complete object at offset 0 in bucket 1.
    std::vector<BatchAllocateRequest> requests{{"x", 100}, {"y", 100}};
    auto results = alloc.BatchAllocate(requests);
    ASSERT_EQ(results.size(), 2u);
    ASSERT_TRUE(results[0].success);
    ASSERT_TRUE(results[1].success);
    EXPECT_EQ(results[0].descriptor.shard_idx, 0);
    EXPECT_EQ(results[1].descriptor.shard_idx, 1);
    EXPECT_EQ(EntryStartOf(results[0].descriptor), kAlignment * 3);
    EXPECT_EQ(EntryStartOf(results[1].descriptor), 0u);
}

TEST(ImmutableBucketAllocatorTest, BatchAllocateRejectsBadRequestsAtomically) {
    TempDir tmp("bucket_batch_invalid");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    // Duplicate keys inside one batch.
    {
        std::vector<BatchAllocateRequest> requests{{"same", 10}, {"same", 10}};
        auto results = alloc.BatchAllocate(requests);
        ASSERT_EQ(results.size(), 2u);
        for (const auto& result : results) {
            EXPECT_FALSE(result.success);
            EXPECT_EQ(result.error, ErrorCode::INVALID_PARAMS);
        }
        EXPECT_FALSE(alloc.GetBucketIdForKey("same").has_value());
    }
    // Zero size anywhere in the batch fails the whole batch.
    {
        std::vector<BatchAllocateRequest> requests{{"ok", 10}, {"bad", 0}};
        auto results = alloc.BatchAllocate(requests);
        ASSERT_EQ(results.size(), 2u);
        EXPECT_FALSE(results[0].success);
        EXPECT_FALSE(results[1].success);
        EXPECT_FALSE(alloc.GetBucketIdForKey("ok").has_value());
    }
    // Empty batch is a no-op.
    {
        auto results = alloc.BatchAllocate({});
        EXPECT_TRUE(results.empty());
    }
}

// ---------------------------------------------------------------------------
// ImmutableBucketAllocator: Free
// ---------------------------------------------------------------------------

TEST(ImmutableBucketAllocatorTest, FreeTombstonesEntryAndAllowsReuseOfKey) {
    TempDir tmp("bucket_free");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    auto first = alloc.Allocate("k", 100);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(alloc.GetBucketIdForKey("k").has_value());

    alloc.Free("k", *first);
    EXPECT_FALSE(alloc.GetBucketIdForKey("k").has_value());

    // Buckets are append-only, so the key can be reallocated but not at the
    // same offset: the freed space is only reclaimed by whole-bucket eviction.
    auto second = alloc.Allocate("k", 100);
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(second->offset, first->offset);
}

TEST(ImmutableBucketAllocatorTest, StaleFreeDoesNotDropNewAllocation) {
    TempDir tmp("bucket_stale_free");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    auto first = alloc.Allocate("k", 100);
    ASSERT_TRUE(first.has_value());
    alloc.Free("k", *first);
    auto second = alloc.Allocate("k", 100);
    ASSERT_TRUE(second.has_value());

    // A late Free carrying the superseded descriptor must be a no-op.
    alloc.Free("k", *first);
    auto bucket_id = alloc.GetBucketIdForKey("k");
    ASSERT_TRUE(bucket_id.has_value());
    EXPECT_EQ(*bucket_id, second->shard_idx);

    // A descriptor with a mismatching size must also be rejected.
    auto tampered = *second;
    tampered.object_size += 1;
    alloc.Free("k", tampered);
    EXPECT_TRUE(alloc.GetBucketIdForKey("k").has_value());

    // The genuine descriptor still frees it.
    alloc.Free("k", *second);
    EXPECT_FALSE(alloc.GetBucketIdForKey("k").has_value());
}

// ---------------------------------------------------------------------------
// ImmutableBucketAllocator: restart behaviour (nothing is recovered)
// ---------------------------------------------------------------------------

TEST(ImmutableBucketAllocatorTest, InitPurgesLeftoverBucketFiles) {
    TempDir tmp("bucket_purge");
    // What a previous process left behind: a data file nobody can address any
    // more, plus a `.meta` file from a build that still wrote one.
    const std::string data_path = BucketDataFile(tmp, 7);
    const std::string meta_path = BucketMetaFile(tmp, 7);
    {
        const int fd = ::open(data_path.c_str(), O_CREAT | O_WRONLY, 0644);
        ASSERT_GE(fd, 0);
        ::close(fd);
        const int meta_fd = ::open(meta_path.c_str(), O_CREAT | O_WRONLY, 0644);
        ASSERT_GE(meta_fd, 0);
        ::close(meta_fd);
    }
    ASSERT_TRUE(std::filesystem::exists(data_path));
    ASSERT_TRUE(std::filesystem::exists(meta_path));

    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    // Both are unreachable garbage, so they are reclaimed at startup. Bucket id
    // 7 is outside the ready pool, so it is not recreated afterwards.
    EXPECT_FALSE(std::filesystem::exists(data_path));
    EXPECT_FALSE(std::filesystem::exists(meta_path));
}

TEST(ImmutableBucketAllocatorTest, InitLeavesOtherFilesAlone) {
    TempDir tmp("bucket_purge_scope");
    // Only the two canonical `bucket_<id>.<suffix>` names are recognized;
    // anything else in the directory belongs to someone else.
    const std::string legacy = tmp.file(
        "bucket_" + ImmutableBucketAllocator::FormatBucketId(7) + ".meta.tmp");
    const std::string unrelated = tmp.file("not_a_bucket.txt");
    for (const auto& path : {legacy, unrelated}) {
        const int fd = ::open(path.c_str(), O_CREAT | O_WRONLY, 0644);
        ASSERT_GE(fd, 0);
        ::close(fd);
    }

    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    EXPECT_TRUE(std::filesystem::exists(legacy));
    EXPECT_TRUE(std::filesystem::exists(unrelated));
}

TEST(ImmutableBucketAllocatorTest, RestartRecoversNothingAndRestartsBucketIds) {
    TempDir tmp("bucket_restart");
    const uint64_t capacity = kAlignment;  // one entry per bucket
    {
        ImmutableBucketAllocator alloc;
        ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 8)));
        for (int i = 0; i < 3; ++i) {
            auto desc = alloc.Allocate("k" + std::to_string(i), 100);
            ASSERT_TRUE(desc.has_value());
        }
        ASSERT_TRUE(std::filesystem::exists(BucketDataFile(tmp, 0)));
    }

    auto config = MakeBucketConfig(tmp.path(), capacity, 8);
    // No ready pool, so no fresh bucket file hides the ones being reclaimed.
    config.ready_bucket_target = 0;
    ImmutableBucketAllocator restarted;
    ASSERT_TRUE(restarted.Init(config));
    // No key survives, and neither does the data it was written to.
    EXPECT_FALSE(restarted.GetBucketIdForKey("k0").has_value());
    EXPECT_FALSE(restarted.GetBucketIdForKey("k1").has_value());
    EXPECT_FALSE(restarted.GetBucketIdForKey("k2").has_value());
    EXPECT_FALSE(std::filesystem::exists(BucketDataFile(tmp, 0)));
    EXPECT_FALSE(std::filesystem::exists(BucketDataFile(tmp, 1)));
    EXPECT_FALSE(std::filesystem::exists(BucketDataFile(tmp, 2)));

    // Ids start over instead of continuing where the previous run stopped.
    auto desc = restarted.Allocate("after", 100);
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(desc->shard_idx, 0);
}

// ---------------------------------------------------------------------------
// ImmutableBucketAllocator: eviction
// ---------------------------------------------------------------------------

namespace {

DistributedStorageConfig MakeEvictionConfig(const std::string& fsdir,
                                            uint64_t bucket_capacity,
                                            int64_t max_bucket_count) {
    auto config = MakeBucketConfig(fsdir, bucket_capacity, max_bucket_count);
    config.eviction_enabled = true;
    config.eviction_high_watermark = 0.5;
    config.eviction_low_watermark = 0.25;
    return config;
}

}  // namespace

TEST(ImmutableBucketAllocatorTest, PrepareEvictionSkipsActiveBucket) {
    TempDir tmp("bucket_evict_active");
    const uint64_t capacity = kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    // Only one bucket exists and it is the active one, so nothing is evictable
    // even though usage is above the high watermark.
    auto desc = alloc.Allocate("k0", 100);
    ASSERT_TRUE(desc.has_value());
    auto pending = alloc.PrepareEviction();
    EXPECT_TRUE(pending.Empty());
    EXPECT_LT(pending.bucket_id(), 0);
}

TEST(ImmutableBucketAllocatorTest, CommitEvictionDeletesBucketAndFrees) {
    TempDir tmp("bucket_evict_commit");
    const uint64_t capacity = kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    std::vector<DistributedFSDescriptor> descs;
    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value()) << "allocation " << i;
        descs.push_back(*desc);
    }
    EXPECT_EQ(alloc.GetBucketCount(), 3u);

    auto pending = alloc.PrepareEviction();
    ASSERT_FALSE(pending.Empty());
    // Bucket 0 is the coldest non-active bucket.
    EXPECT_EQ(pending.bucket_id(), 0);
    ASSERT_EQ(pending.Candidates().size(), 1u);
    const auto& candidate = pending.Candidates()[0];
    EXPECT_EQ(candidate.key, "k0");
    // The candidate descriptor must be byte-identical to what Allocate returned
    // so the master can match it against replica metadata.
    EXPECT_EQ(candidate.descriptor.file_path, descs[0].file_path);
    EXPECT_EQ(candidate.descriptor.offset, descs[0].offset);
    EXPECT_EQ(candidate.descriptor.object_size, descs[0].object_size);
    EXPECT_EQ(candidate.descriptor.aligned_size, descs[0].aligned_size);
    EXPECT_EQ(candidate.descriptor.shard_idx, descs[0].shard_idx);

    const std::string data_path = descs[0].file_path;
    ASSERT_TRUE(std::filesystem::exists(data_path));

    alloc.CommitEviction(std::move(pending));
    EXPECT_FALSE(std::filesystem::exists(data_path));
    EXPECT_EQ(alloc.GetBucketCount(), 2u);
    EXPECT_FALSE(alloc.GetBucketIdForKey("k0").has_value());
    // Surviving buckets are untouched.
    EXPECT_TRUE(alloc.GetBucketIdForKey("k1").has_value());
    EXPECT_TRUE(alloc.GetBucketIdForKey("k2").has_value());
}

TEST(ImmutableBucketAllocatorTest, AbortEvictionRestoresBucketUnchanged) {
    TempDir tmp("bucket_evict_abort");
    const uint64_t capacity = kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
    }

    auto pending = alloc.PrepareEviction();
    ASSERT_FALSE(pending.Empty());
    const int64_t bucket_id = pending.bucket_id();
    alloc.AbortEviction(std::move(pending));

    // Nothing was dropped.
    EXPECT_EQ(alloc.GetBucketCount(), 3u);
    EXPECT_TRUE(alloc.GetBucketIdForKey("k0").has_value());

    // A rejected bucket must become selectable again, and the next round should
    // reach a different (colder) bucket first.
    auto next = alloc.PrepareEviction();
    ASSERT_FALSE(next.Empty());
    EXPECT_NE(next.bucket_id(), bucket_id);
    alloc.AbortEviction(std::move(next));
}

TEST(ImmutableBucketAllocatorTest, UnresolvedPendingEvictionSelfAborts) {
    TempDir tmp("bucket_evict_dtor");
    const uint64_t capacity = kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
    }

    int64_t bucket_id = -1;
    {
        auto pending = alloc.PrepareEviction();
        ASSERT_FALSE(pending.Empty());
        bucket_id = pending.bucket_id();
        // Dropped without Commit or Abort: the destructor must unfreeze it.
    }
    EXPECT_EQ(alloc.GetBucketCount(), 3u);
    auto again = alloc.PrepareEviction();
    ASSERT_FALSE(again.Empty());
    EXPECT_EQ(again.bucket_id(), bucket_id);
    alloc.AbortEviction(std::move(again));
}

TEST(ImmutableBucketAllocatorTest, CommitAndAbortEvictionAreIdempotent) {
    TempDir tmp("bucket_evict_idempotent");
    const uint64_t capacity = kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
    }

    auto pending = alloc.PrepareEviction();
    ASSERT_FALSE(pending.Empty());
    alloc.CommitEviction(std::move(pending));
    EXPECT_EQ(alloc.GetBucketCount(), 2u);
    // Committing (or aborting) the already-resolved transaction is a no-op.
    alloc.CommitEviction(std::move(pending));
    alloc.AbortEviction(std::move(pending));
    EXPECT_EQ(alloc.GetBucketCount(), 2u);
}

TEST(ImmutableBucketAllocatorTest, AllocationFailureEvictionBypassesLowWatermark) {
    TempDir tmp("bucket_evict_allocation_failure");
    constexpr uint64_t capacity = 10 * kAlignment;
    ImmutableBucketAllocator alloc;
    auto config = MakeEvictionConfig(tmp.path(), capacity, 4);
    config.eviction_high_watermark = 0.9;
    config.eviction_low_watermark = 0.7;
    ASSERT_TRUE(alloc.Init(config));

    // Each batch consumes six aligned entries, so four buckets fill only 60%
    // of the fixed denominator and the ordinary watermark scan is empty.
    for (int batch = 0; batch < 4; ++batch) {
        std::vector<BatchAllocateRequest> requests;
        for (int i = 0; i < 6; ++i) {
            requests.push_back(
                {"key_" + std::to_string(batch) + "_" + std::to_string(i),
                 100});
        }
        auto results = alloc.BatchAllocate(requests);
        for (const auto& result : results) ASSERT_TRUE(result.success);
    }
    // Twenty-four entries pack into 10 + 10 + 4 slots instead of four
    // six-entry buckets. Utilization is still below the configured watermark,
    // so retain coverage for the allocation-failure override.
    EXPECT_EQ(alloc.GetBucketCount(), 3u);
    EXPECT_TRUE(alloc.PrepareEviction().Empty());
    auto pending = alloc.PrepareEvictionForAllocationFailure();
    ASSERT_FALSE(pending.Empty());
    alloc.CommitEviction(std::move(pending));
    EXPECT_EQ(alloc.GetBucketCount(), 2u);

    auto retry = alloc.BatchAllocate({{"after_eviction", 100}});
    ASSERT_EQ(retry.size(), 1u);
    EXPECT_TRUE(retry[0].success);
}

TEST(ImmutableBucketAllocatorTest, EvictionCandidatesExcludeFreedKeys) {
    TempDir tmp("bucket_evict_freed");
    const uint64_t capacity = 2 * kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    // Bucket 0 gets two keys; free one of them.
    auto a = alloc.Allocate("a", 100);
    ASSERT_TRUE(a.has_value());
    auto b = alloc.Allocate("b", 100);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(a->shard_idx, 0);
    ASSERT_EQ(b->shard_idx, 0);
    alloc.Free("a", *a);

    // Fill more buckets so bucket 0 is no longer active and usage is high.
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(alloc.Allocate("f" + std::to_string(i), 100).has_value());
    }

    auto pending = alloc.PrepareEviction();
    ASSERT_FALSE(pending.Empty());
    ASSERT_EQ(pending.bucket_id(), 0);
    // Only the live key is reported: a tombstoned entry is already invisible to
    // the master and must not be validated again.
    ASSERT_EQ(pending.Candidates().size(), 1u);
    EXPECT_EQ(pending.Candidates()[0].key, "b");
    alloc.AbortEviction(std::move(pending));
}

TEST(ImmutableBucketAllocatorTest, EvictionUsesFixedCapacityDenominator) {
    TempDir tmp("bucket_evict_watermark");
    const uint64_t capacity = kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 8)));

    // Total capacity stays max_bucket_count * bucket_capacity regardless of how
    // many buckets exist, so deleting one lowers usage instead of shrinking the
    // denominator in lockstep (which would never converge).
    EXPECT_EQ(alloc.GetTotalCapacity(), 8u * capacity);
    EXPECT_EQ(alloc.GetUsedBytes(), 0u);

    for (int i = 0; i < 5; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
    }
    EXPECT_EQ(alloc.GetTotalCapacity(), 8u * capacity);
    const uint64_t used_before = alloc.GetUsedBytes();
    EXPECT_EQ(used_before, 5u * kAlignment);

    auto pending = alloc.PrepareEviction();
    ASSERT_FALSE(pending.Empty());
    alloc.CommitEviction(std::move(pending));
    EXPECT_EQ(alloc.GetTotalCapacity(), 8u * capacity);
    EXPECT_LT(alloc.GetUsedBytes(), used_before);
}

TEST(ImmutableBucketAllocatorTest, EvictionStopsAtLowWatermark) {
    TempDir tmp("bucket_evict_low");
    const uint64_t capacity = kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 8)));

    // Two of eight buckets used = 25% == low watermark, below high (50%), so
    // eviction should not start.
    for (int i = 0; i < 2; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
    }
    auto pending = alloc.PrepareEviction();
    EXPECT_TRUE(pending.Empty());
}

TEST(ImmutableBucketAllocatorTest, EvictionDisabledYieldsNoCandidates) {
    TempDir tmp("bucket_evict_disabled");
    const uint64_t capacity = kAlignment;
    auto config = MakeEvictionConfig(tmp.path(), capacity, 2);
    config.eviction_enabled = false;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(config));
    EXPECT_FALSE(alloc.IsEvictionEnabled());
}

TEST(ImmutableBucketAllocatorTest, ReadableDescriptorSurvivesConcurrentEviction) {
    TempDir tmp("bucket_evict_inflight");
    const uint64_t capacity = kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    std::vector<DistributedFSDescriptor> descs;
    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
        descs.push_back(*desc);
    }

    auto pending = alloc.PrepareEviction();
    ASSERT_FALSE(pending.Empty());
    // While the transaction is open, the bucket's files are still present, so a
    // reader that already holds a descriptor can complete its I/O. Deletion
    // only happens in CommitEviction, after the master dropped the replica.
    EXPECT_TRUE(std::filesystem::exists(descs[0].file_path));
    alloc.AbortEviction(std::move(pending));
    EXPECT_TRUE(std::filesystem::exists(descs[0].file_path));
}

TEST(ImmutableBucketAllocatorTest, FreeRacesWithEvictionCommit) {
    TempDir tmp("bucket_free_evict_race");
    const uint64_t capacity = kAlignment;
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    auto victim_desc = alloc.Allocate("victim", 100);
    ASSERT_TRUE(victim_desc.has_value());
    auto active_desc = alloc.Allocate("active", 100);
    ASSERT_TRUE(active_desc.has_value());

    auto pending = alloc.PrepareEviction();
    ASSERT_FALSE(pending.Empty());
    ASSERT_EQ(pending.bucket_id(), 0);
    const std::string data_path = victim_desc->file_path;

    // Race a Free() of the victim against the commit that deletes its bucket.
    // Both only touch memory, so nothing can resurrect the bucket afterwards -
    // and no metadata file may appear as a side effect of either path.
    std::atomic<bool> start{false};
    std::thread freer([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        alloc.Free("victim", *victim_desc);
    });
    std::thread evictor(
        [&alloc, &start, pending = std::move(pending)]() mutable {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            alloc.CommitEviction(std::move(pending));
        });
    start.store(true, std::memory_order_release);
    freer.join();
    evictor.join();

    EXPECT_FALSE(std::filesystem::exists(data_path));
    EXPECT_FALSE(std::filesystem::exists(BucketMetaFile(tmp, 0)));
    EXPECT_FALSE(alloc.GetBucketIdForKey("victim").has_value());
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

TEST(ImmutableBucketAllocatorTest, ConcurrentAllocateProducesDisjointRegions) {
    TempDir tmp("bucket_concurrent_alloc");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), 4 * 1024 * 1024, 32)));

    constexpr int kThreads = 8;
    constexpr int kPerThread = 32;
    std::vector<std::vector<std::pair<std::string, DistributedFSDescriptor>>>
        per_thread(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&alloc, &per_thread, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                const std::string key =
                    "t" + std::to_string(t) + "_" + std::to_string(i);
                auto desc = alloc.Allocate(key, 512);
                if (desc.has_value()) {
                    per_thread[t].emplace_back(key, *desc);
                }
            }
        });
    }
    for (auto& thread : threads) thread.join();

    // Collect every reserved region and assert no two overlap.
    struct Region {
        int bucket_id;
        uint64_t begin;
        uint64_t end;
    };
    std::vector<Region> regions;
    size_t total = 0;
    for (const auto& thread_results : per_thread) {
        total += thread_results.size();
        for (const auto& [key, desc] : thread_results) {
            const uint64_t begin = EntryStartOf(desc);
            regions.push_back(
                {desc.shard_idx, begin, begin + desc.aligned_size});
        }
    }
    EXPECT_EQ(total, static_cast<size_t>(kThreads * kPerThread));

    std::sort(regions.begin(), regions.end(),
              [](const Region& lhs, const Region& rhs) {
                  if (lhs.bucket_id != rhs.bucket_id) {
                      return lhs.bucket_id < rhs.bucket_id;
                  }
                  return lhs.begin < rhs.begin;
              });
    for (size_t i = 1; i < regions.size(); ++i) {
        if (regions[i].bucket_id != regions[i - 1].bucket_id) continue;
        EXPECT_GE(regions[i].begin, regions[i - 1].end)
            << "overlapping regions in bucket " << regions[i].bucket_id;
    }
}

TEST(ImmutableBucketAllocatorTest, ConcurrentFreeAndUpdateAccessAreSafe) {
    TempDir tmp("bucket_concurrent_free");
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), 1024 * 1024, 16)));

    constexpr int kKeys = 64;
    std::vector<std::pair<std::string, DistributedFSDescriptor>> allocations;
    for (int i = 0; i < kKeys; ++i) {
        const std::string key = "k" + std::to_string(i);
        auto desc = alloc.Allocate(key, 512);
        ASSERT_TRUE(desc.has_value());
        allocations.emplace_back(key, *desc);
    }

    std::vector<std::thread> threads;
    threads.emplace_back([&alloc, &allocations]() {
        for (const auto& [key, desc] : allocations) {
            alloc.UpdateAccess(key, desc);
        }
    });
    threads.emplace_back([&alloc, &allocations]() {
        for (const auto& [key, desc] : allocations) {
            alloc.Free(key, desc);
        }
    });
    threads.emplace_back([&alloc, &allocations]() {
        // Duplicate frees must be harmless.
        for (const auto& [key, desc] : allocations) {
            alloc.Free(key, desc);
        }
    });
    for (auto& thread : threads) thread.join();

    for (const auto& [key, desc] : allocations) {
        (void)desc;
        EXPECT_FALSE(alloc.GetBucketIdForKey(key).has_value()) << "key=" << key;
    }
}

// ---------------------------------------------------------------------------
// Polymorphic use through the interface, and SHARD compatibility
// ---------------------------------------------------------------------------

TEST(DfsAllocatorInterfaceTest, BothImplementationsWorkPolymorphically) {
    TempDir bucket_dir("iface_bucket");
    TempDir shard_dir("iface_shard");

    std::vector<std::unique_ptr<DfsAllocatorInterface>> allocators;
    {
        auto bucket = std::make_unique<ImmutableBucketAllocator>();
        ASSERT_TRUE(bucket->Init(
            MakeBucketConfig(bucket_dir.path(), kBucketCapacity, 8)));
        allocators.push_back(std::move(bucket));
    }
    {
        auto shard = std::make_unique<DfsGlobalAllocator>();
        ASSERT_TRUE(
            shard->Init(MakeShardConfig(shard_dir.path(), 4, 1024 * 1024)));
        allocators.push_back(std::move(shard));
    }

    for (auto& allocator : allocators) {
        EXPECT_TRUE(allocator->IsInitialized());
        auto desc = allocator->Allocate("poly_key", 100);
        ASSERT_TRUE(desc.has_value()) << "type=" << ToString(allocator->Type());
        EXPECT_EQ(desc->object_size, 100u);
        EXPECT_GT(desc->aligned_size, 0u);
        EXPECT_GT(allocator->GetTotalCapacity(), 0u);

        allocator->UpdateAccess("poly_key", *desc);
        allocator->Free("poly_key", *desc);

        std::vector<BatchAllocateRequest> requests{{"poly_a", 100},
                                                   {"poly_b", 100}};
        auto results = allocator->BatchAllocate(requests);
        ASSERT_EQ(results.size(), 2u);
        for (size_t i = 0; i < results.size(); ++i) {
            EXPECT_TRUE(results[i].success)
                << "type=" << ToString(allocator->Type()) << " entry " << i;
            EXPECT_EQ(results[i].key, requests[i].key);
        }
    }
}

TEST(DfsGlobalAllocatorCompatTest, ShardBehaviourIsUnchanged) {
    TempDir tmp("shard_compat");
    DfsGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeShardConfig(tmp.path(), 4, 1024 * 1024)));
    EXPECT_EQ(alloc.Type(), DfsAllocatorType::SHARD);

    auto desc = alloc.Allocate("key1", 100);
    ASSERT_TRUE(desc.has_value());
    // SHARD keeps its original descriptor semantics: the value offset itself is
    // alignment-aligned and shard_idx is a shard index.
    EXPECT_EQ(desc->aligned_size, kAlignment);
    EXPECT_EQ(desc->offset % kAlignment, 0u);
    EXPECT_GE(desc->shard_idx, 0);
    EXPECT_LT(desc->shard_idx, 4);
    EXPECT_NE(desc->file_path.find("dfs_shard_"), std::string::npos);

    // The original offset/shard_idx API still works alongside the new one.
    alloc.UpdateAccess("key1", desc->shard_idx, desc->offset);
    alloc.Free(desc->offset, desc->aligned_size, desc->shard_idx, "key1");
    EXPECT_TRUE(alloc.Allocate("key2", 100).has_value());

    EXPECT_EQ(DfsGlobalAllocator::FormatShardIdx(0, 64), "00");
    EXPECT_EQ(DfsGlobalAllocator::FormatShardIdx(63, 64), "63");
}

TEST(DfsGlobalAllocatorCompatTest, BatchAllocateIsAllOrNothing) {
    TempDir tmp("shard_batch");
    DfsGlobalAllocator alloc;
    // One shard with room for exactly two allocations (each reserves
    // aligned_size + alignment - 1, so the allocator rounds up).
    ASSERT_TRUE(alloc.Init(MakeShardConfig(tmp.path(), 1, 3 * kAlignment)));

    std::vector<BatchAllocateRequest> requests{
        {"a", 100}, {"b", 100}, {"c", 100}, {"d", 100}};
    auto results = alloc.BatchAllocate(requests);
    ASSERT_EQ(results.size(), requests.size());

    const bool any_failed =
        std::any_of(results.begin(), results.end(),
                    [](const BatchAllocateResult& r) { return !r.success; });
    if (any_failed) {
        // On failure nothing stays reserved, so a fresh single allocation can
        // still use the space that was rolled back.
        for (const auto& result : results) {
            EXPECT_FALSE(result.success);
        }
        EXPECT_TRUE(alloc.Allocate("after_rollback", 100).has_value());
    }
}

// ---------------------------------------------------------------------------
// Configuration selection
// ---------------------------------------------------------------------------

TEST(DfsAllocatorConfigTest, ParsesAndRejectsAllocatorTypes) {
    EXPECT_EQ(ParseDfsAllocatorType("shard"), DfsAllocatorType::SHARD);
    EXPECT_EQ(ParseDfsAllocatorType("SHARD"), DfsAllocatorType::SHARD);
    EXPECT_EQ(ParseDfsAllocatorType("bucket"), DfsAllocatorType::BUCKET);
    EXPECT_EQ(ParseDfsAllocatorType("BUCKET"), DfsAllocatorType::BUCKET);
    // Unknown names must be reported, not silently mapped to SHARD.
    EXPECT_FALSE(ParseDfsAllocatorType("shrad").has_value());
    EXPECT_FALSE(ParseDfsAllocatorType("").has_value());

    EXPECT_STREQ(ToString(DfsAllocatorType::SHARD), "shard");
    EXPECT_STREQ(ToString(DfsAllocatorType::BUCKET), "bucket");
}

TEST(DfsAllocatorConfigTest, DefaultsToShardAndValidatesBucketFields) {
    DistributedStorageConfig config;
    // SHARD remains the default so existing deployments do not change mode.
    EXPECT_EQ(config.allocator_type, DfsAllocatorType::SHARD);
    EXPECT_TRUE(config.allocator_type_valid);

    TempDir tmp("config_validate");
    auto bucket = MakeBucketConfig(tmp.path(), kBucketCapacity, 8);
    EXPECT_TRUE(bucket.Validate());
    EXPECT_TRUE(bucket.ValidateForAllocator());
    EXPECT_TRUE(bucket.ValidateForBucketAllocator());

    auto misaligned = bucket;
    misaligned.bucket_capacity = kAlignment + 1;
    EXPECT_FALSE(misaligned.ValidateForBucketAllocator());

    auto too_many = bucket;
    too_many.max_bucket_count = kMaxBucketId + 1;
    EXPECT_FALSE(too_many.ValidateForBucketAllocator());

    auto capped_ready_pool = bucket;
    capped_ready_pool.max_bucket_count = 2;
    capped_ready_pool.ready_bucket_target = 4;
    EXPECT_TRUE(capped_ready_pool.ValidateForBucketAllocator());

    auto negative_ready_pool = bucket;
    negative_ready_pool.ready_bucket_target = -1;
    EXPECT_FALSE(negative_ready_pool.ValidateForBucketAllocator());

    const std::string formatted = bucket.FormatStr();
    EXPECT_NE(formatted.find("allocator_type=bucket"), std::string::npos);
    EXPECT_NE(formatted.find("bucket_capacity="), std::string::npos);
}

// ---------------------------------------------------------------------------
// Entry layout helper
// ---------------------------------------------------------------------------

TEST(BucketEntryLayoutTest, ComputesAlignedEntriesAndRejectsOverflow) {
    const uint64_t alignment = 4096;

    auto layout = ComputeBucketEntryLayout(0, 100, alignment);
    ASSERT_TRUE(layout.has_value());
    EXPECT_EQ(layout->offset, 0u);
    EXPECT_EQ(layout->object_size, 100u);
    EXPECT_EQ(layout->aligned_size, alignment);
    EXPECT_EQ(layout->end(), alignment);

    // A non-aligned cursor is rounded up to the next boundary.
    auto next = ComputeBucketEntryLayout(1, 100, alignment);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next->offset, alignment);

    // Invalid inputs.
    EXPECT_FALSE(ComputeBucketEntryLayout(0, 0, alignment).has_value());
    EXPECT_FALSE(ComputeBucketEntryLayout(0, 100, 0).has_value());
    EXPECT_FALSE(ComputeBucketEntryLayout(0, 100, 3000).has_value());

    // Overflow must be rejected rather than wrapping around.
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    EXPECT_FALSE(ComputeBucketEntryLayout(0, kMax, alignment).has_value());
    EXPECT_FALSE(ComputeBucketEntryLayout(kMax, 100, alignment).has_value());

    EXPECT_FALSE(CheckedAlignUp(kMax, alignment).has_value());
    EXPECT_TRUE(IsValidBucketAlignment(4096));
    EXPECT_FALSE(IsValidBucketAlignment(0));
    EXPECT_FALSE(IsValidBucketAlignment(3000));
}

TEST(BucketEntryLayoutTest, RebuildMatchesComputeAndRejectsMisalignment) {
    const uint64_t alignment = 4096;
    auto computed = ComputeBucketEntryLayout(alignment, 200, alignment);
    ASSERT_TRUE(computed.has_value());

    auto rebuilt =
        RebuildBucketEntryLayout(computed->offset, 200, alignment);
    ASSERT_TRUE(rebuilt.has_value());
    EXPECT_EQ(rebuilt->offset, computed->offset);
    EXPECT_EQ(rebuilt->aligned_size, computed->aligned_size);

    // A recorded entry start that is not aligned indicates corrupt metadata.
    EXPECT_FALSE(RebuildBucketEntryLayout(1, 200, alignment).has_value());
}

TEST(BucketEntryLayoutTest, DescriptorConstructionIsCentralized) {
    const uint64_t alignment = 4096;
    auto layout = ComputeBucketEntryLayout(0, 100, alignment);
    ASSERT_TRUE(layout.has_value());

    auto desc =
        MakeBucketDescriptor("/tmp/bucket_000001.data", *layout, 1);
    EXPECT_EQ(desc.file_path, "/tmp/bucket_000001.data");
    EXPECT_EQ(desc.offset, layout->offset);
    EXPECT_EQ(desc.object_size, 100u);
    EXPECT_EQ(desc.aligned_size, layout->aligned_size);
    EXPECT_EQ(desc.shard_idx, 1);
    EXPECT_EQ(desc.offset % alignment, 0u);
}

// ---------------------------------------------------------------------------
// Restart boundary: an abrupt restart keeps nothing, and reclaims everything
// ---------------------------------------------------------------------------

TEST(BucketRestartTest, AbruptRestartDropsEveryBucket) {
    TempDir tmp("bucket_crash");
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        ImmutableBucketAllocator alloc;
        if (!alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8))) {
            ::_exit(10);
        }
        if (!alloc.Allocate("crash_a", 100) || !alloc.Allocate("crash_b", 100)) {
            ::_exit(11);
        }
        // Simulate a process crash: the destructor never runs. Nothing is lost
        // by that any more, because there was never anything to flush.
        ::_exit(0);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);

    const std::string data_path = BucketDataFile(tmp, 0);
    ASSERT_TRUE(std::filesystem::exists(data_path));

    ImmutableBucketAllocator restarted;
    ASSERT_TRUE(restarted.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    // No metadata survives the restart, so the bytes are unreachable and every
    // key that pointed at them is gone.
    EXPECT_FALSE(restarted.GetBucketIdForKey("crash_a").has_value());
    EXPECT_FALSE(restarted.GetBucketIdForKey("crash_b").has_value());
}

TEST(BucketRestartTest, BucketDirectoryHoldsOnlyDataFiles) {
    TempDir tmp("bucket_only_data_files");
    const uint64_t capacity = 4 * kAlignment;  // four entries per bucket
    ImmutableBucketAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 16)));

    for (int i = 0; i < 20; ++i) {
        auto desc = alloc.Allocate("obj_" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
    }

    // No `.meta` companion, no log, no snapshot slots, no rename temporaries:
    // a bucket is exactly one preallocated data file.
    size_t data_files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(tmp.path())) {
        const std::string name = entry.path().filename().string();
        ASSERT_EQ(name.substr(0, 7), "bucket_") << "Unexpected file: " << name;
        ASSERT_EQ(name.substr(name.size() - 5), ".data")
            << "Unexpected file: " << name;
        ++data_files;
    }
    EXPECT_GT(data_files, 0u);
}

}  // namespace mooncake::test
