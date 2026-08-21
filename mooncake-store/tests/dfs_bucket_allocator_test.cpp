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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ylt/struct_pb.hpp>

#include "crc32c.h"
#include "replica.h"
#include "storage/distributed/bucket_entry_layout.h"
#include "storage/distributed/bucket_global_allocator.h"
#include "storage/distributed/dfs_global_allocator.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/global_allocator_interface.h"
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

// Recomputes the entry start from a descriptor, mirroring what the backend does.
uint64_t EntryStartOf(const DistributedFSDescriptor& desc,
                      const std::string& key) {
    return desc.offset - BucketEntryLayout::kHeaderSize - key.size();
}

}  // namespace

// ---------------------------------------------------------------------------
// BucketGlobalAllocator: allocation
// ---------------------------------------------------------------------------

TEST(BucketGlobalAllocatorTest, AllocateReturnsBucketIdAndValueOffset) {
    TempDir tmp("bucket_alloc");
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    EXPECT_TRUE(alloc.IsInitialized());
    EXPECT_EQ(alloc.Type(), DfsAllocatorType::BUCKET);

    const std::string key = "key1";
    auto desc = alloc.Allocate(key, 100);
    ASSERT_TRUE(desc.has_value()) << toString(desc.error());

    // First entry starts at offset 0; the value sits right after the header
    // plus the key bytes, which is deliberately NOT alignment-aligned.
    EXPECT_EQ(EntryStartOf(*desc, key), 0u);
    EXPECT_EQ(desc->offset, BucketEntryLayout::kHeaderSize + key.size());
    EXPECT_EQ(desc->object_size, 100u);
    EXPECT_EQ(desc->aligned_size, kAlignment);
    EXPECT_EQ(desc->shard_idx, 0);
    EXPECT_NE(desc->file_path.find("bucket_"), std::string::npos);
    EXPECT_NE(desc->file_path.find(".data"), std::string::npos);

    // The data file must be preallocated to the full bucket capacity.
    EXPECT_TRUE(std::filesystem::exists(desc->file_path));
    EXPECT_EQ(std::filesystem::file_size(desc->file_path), kBucketCapacity);
}

TEST(BucketGlobalAllocatorTest, AllocateRejectsInvalidRequests) {
    TempDir tmp("bucket_alloc_invalid");
    BucketGlobalAllocator alloc;
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

TEST(BucketGlobalAllocatorTest, UninitializedAllocatorReportsUnavailable) {
    BucketGlobalAllocator alloc;
    EXPECT_FALSE(alloc.IsInitialized());
    auto desc = alloc.Allocate("k", 100);
    ASSERT_FALSE(desc.has_value());
    EXPECT_EQ(desc.error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);
}

TEST(BucketGlobalAllocatorTest, InitRejectsInvalidConfiguration) {
    TempDir tmp("bucket_init_invalid");

    {
        auto config = MakeBucketConfig(tmp.path(), 0, 8);
        BucketGlobalAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
    }
    {
        // max_bucket_count must be > 0: it is the fixed watermark denominator.
        auto config = MakeBucketConfig(tmp.path(), kBucketCapacity, 0);
        BucketGlobalAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
    }
    {
        auto config = MakeBucketConfig(tmp.path(), kBucketCapacity, 8);
        config.alignment = 3000;  // not a power of two
        BucketGlobalAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
    }
    {
        auto config = MakeBucketConfig(tmp.path(), kBucketCapacity, 8);
        config.eviction_low_watermark = 0.95;
        config.eviction_high_watermark = 0.9;
        BucketGlobalAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::INVALID_PARAMS);
    }
    {
        auto config = MakeBucketConfig(tmp.path(), kBucketCapacity, 8);
        config.fs_adapter_type = "nonexistent";
        BucketGlobalAllocator alloc;
        auto result = alloc.Init(config);
        ASSERT_FALSE(result.has_value());
    }
}

TEST(BucketGlobalAllocatorTest, MultipleObjectsShareBucketWithDistinctOffsets) {
    TempDir tmp("bucket_multi");
    BucketGlobalAllocator alloc;
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
        const uint64_t entry_start = EntryStartOf(desc, key);
        EXPECT_EQ(entry_start % kAlignment, 0u) << "key=" << key;
        EXPECT_EQ(entry_start, previous_end) << "key=" << key;
        previous_end = entry_start + desc.aligned_size;
    }
    EXPECT_EQ(alloc.GetBucketCount(), 1u);
}

TEST(BucketGlobalAllocatorTest, BucketRolloverCreatesNewBucket) {
    TempDir tmp("bucket_rollover");
    // Capacity for exactly two 4096-byte entries.
    const uint64_t capacity = 2 * kAlignment;
    BucketGlobalAllocator alloc;
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

TEST(BucketGlobalAllocatorTest, MaxBucketCountIsEnforced) {
    TempDir tmp("bucket_max_count");
    const uint64_t capacity = kAlignment;  // one entry per bucket
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 2)));

    ASSERT_TRUE(alloc.Allocate("k0", 100).has_value());
    ASSERT_TRUE(alloc.Allocate("k1", 100).has_value());
    auto exhausted = alloc.Allocate("k2", 100);
    ASSERT_FALSE(exhausted.has_value());
    EXPECT_EQ(exhausted.error(), ErrorCode::NO_AVAILABLE_HANDLE);
    EXPECT_EQ(alloc.GetBucketCount(), 2u);
}

// ---------------------------------------------------------------------------
// BucketGlobalAllocator: BatchAllocate
// ---------------------------------------------------------------------------

TEST(BucketGlobalAllocatorTest, BatchAllocateIsContiguous) {
    TempDir tmp("bucket_batch");
    BucketGlobalAllocator alloc;
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
        const uint64_t entry_start =
            EntryStartOf(results[i].descriptor, requests[i].key);
        // Contiguity: each entry begins exactly where the previous one ended.
        EXPECT_EQ(entry_start, expected_start) << "entry " << i;
        expected_start = entry_start + results[i].descriptor.aligned_size;
    }
}

TEST(BucketGlobalAllocatorTest, ConcurrentBatchesDoNotInterleave) {
    TempDir tmp("bucket_batch_concurrent");
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(
        alloc.Init(MakeBucketConfig(tmp.path(), 4 * 1024 * 1024, 16)));

    constexpr int kThreads = 4;
    constexpr int kKeysPerBatch = 8;
    std::vector<std::vector<BatchAllocateResult>> per_thread(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&alloc, &per_thread, t]() {
            std::vector<BatchAllocateRequest> requests;
            for (int i = 0; i < kKeysPerBatch; ++i) {
                requests.push_back({"t" + std::to_string(t) + "_k" +
                                        std::to_string(i),
                                    1024});
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
            const uint64_t entry_start =
                EntryStartOf(result.descriptor, result.key);
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
            EXPECT_TRUE(disjoint) << "batches " << a << " and " << b
                                  << " overlap in bucket "
                                  << spans[a].bucket_id;
        }
    }
}

TEST(BucketGlobalAllocatorTest, BatchLargerThanBucketIsRejected) {
    TempDir tmp("bucket_batch_too_big");
    const uint64_t capacity = 2 * kAlignment;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 8)));

    // Three 4096-byte entries cannot fit in a two-entry bucket, and splitting
    // the batch would defeat the contiguity guarantee, so this must fail.
    std::vector<BatchAllocateRequest> requests{
        {"a", 100}, {"b", 100}, {"c", 100}};
    auto results = alloc.BatchAllocate(requests);
    ASSERT_EQ(results.size(), 3u);
    for (const auto& result : results) {
        EXPECT_FALSE(result.success);
        EXPECT_EQ(result.error, ErrorCode::INVALID_PARAMS);
    }
    // Nothing may have been reserved.
    EXPECT_FALSE(alloc.GetBucketIdForKey("a").has_value());
    EXPECT_FALSE(alloc.GetBucketIdForKey("b").has_value());
}

TEST(BucketGlobalAllocatorTest, BatchRollsOverToNewBucketWhenActiveIsFull) {
    TempDir tmp("bucket_batch_rollover");
    const uint64_t capacity = 4 * kAlignment;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 8)));

    // Fill three of four slots in bucket 0.
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(alloc.Allocate("pre" + std::to_string(i), 100).has_value());
    }

    // A 2-entry batch no longer fits in bucket 0, so it must land wholly in a
    // new bucket rather than being split.
    std::vector<BatchAllocateRequest> requests{{"x", 100}, {"y", 100}};
    auto results = alloc.BatchAllocate(requests);
    ASSERT_EQ(results.size(), 2u);
    ASSERT_TRUE(results[0].success);
    ASSERT_TRUE(results[1].success);
    EXPECT_EQ(results[0].descriptor.shard_idx, 1);
    EXPECT_EQ(results[1].descriptor.shard_idx, 1);
    EXPECT_EQ(EntryStartOf(results[0].descriptor, "x"), 0u);
    EXPECT_EQ(EntryStartOf(results[1].descriptor, "y"), kAlignment);
}

TEST(BucketGlobalAllocatorTest, BatchAllocateRejectsBadRequestsAtomically) {
    TempDir tmp("bucket_batch_invalid");
    BucketGlobalAllocator alloc;
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
// BucketGlobalAllocator: Free / MarkCommitted
// ---------------------------------------------------------------------------

TEST(BucketGlobalAllocatorTest, FreeTombstonesEntryAndAllowsReuseOfKey) {
    TempDir tmp("bucket_free");
    BucketGlobalAllocator alloc;
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

TEST(BucketGlobalAllocatorTest, StaleFreeDoesNotDropNewAllocation) {
    TempDir tmp("bucket_stale_free");
    BucketGlobalAllocator alloc;
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

TEST(BucketGlobalAllocatorTest, MarkCommittedValidatesDescriptorIdentity) {
    TempDir tmp("bucket_commit");
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    auto desc = alloc.Allocate("k", 100);
    ASSERT_TRUE(desc.has_value());

    auto wrong = *desc;
    wrong.offset += kAlignment;
    EXPECT_FALSE(alloc.MarkCommitted("k", wrong));
    EXPECT_FALSE(alloc.MarkCommitted("other_key", *desc));

    EXPECT_TRUE(alloc.MarkCommitted("k", *desc));
    // Idempotent: a duplicate PutEnd for the same generation still succeeds.
    EXPECT_TRUE(alloc.MarkCommitted("k", *desc));
}

// ---------------------------------------------------------------------------
// BucketGlobalAllocator: metadata persistence and recovery
// ---------------------------------------------------------------------------

TEST(BucketGlobalAllocatorTest, MetadataIsPersistedOnAllocate) {
    TempDir tmp("bucket_persist");
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    auto desc = alloc.Allocate("persisted", 100);
    ASSERT_TRUE(desc.has_value());

    const std::string meta_base =
        tmp.file("bucket_" + BucketGlobalAllocator::FormatBucketId(0) + ".meta");
    const bool slot0 = std::filesystem::exists(meta_base + ".0");
    const bool slot1 = std::filesystem::exists(meta_base + ".1");
    ASSERT_NE(slot0, slot1);
    const std::string meta_path = meta_base + (slot0 ? ".0" : ".1");
    EXPECT_GT(std::filesystem::file_size(meta_path), 0u);
    // Stable slots replace the old temp-plus-rename publication protocol.
    EXPECT_FALSE(std::filesystem::exists(meta_base));
    EXPECT_FALSE(std::filesystem::exists(meta_base + ".tmp"));
}

TEST(BucketGlobalAllocatorTest, RecoveryRestoresOnlyCommittedEntries) {
    TempDir tmp("bucket_recover");
    DistributedFSDescriptor committed_desc;
    DistributedFSDescriptor pending_desc;
    {
        BucketGlobalAllocator alloc;
        ASSERT_TRUE(
            alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
        auto committed = alloc.Allocate("committed", 100);
        ASSERT_TRUE(committed.has_value());
        committed_desc = *committed;
        ASSERT_TRUE(alloc.MarkCommitted("committed", committed_desc));

        auto pending = alloc.Allocate("pending", 100);
        ASSERT_TRUE(pending.has_value());
        pending_desc = *pending;
        // Deliberately not committed: it simulates a crash after PutStart but
        // before the DFS data write was confirmed.
    }

    BucketGlobalAllocator recovered;
    ASSERT_TRUE(
        recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    auto replicas = recovered.TakeRecoveredReplicas();
    ASSERT_EQ(replicas.size(), 1u);
    EXPECT_EQ(replicas[0].key, "committed");
    EXPECT_EQ(replicas[0].descriptor.offset, committed_desc.offset);
    EXPECT_EQ(replicas[0].descriptor.object_size, committed_desc.object_size);
    EXPECT_EQ(replicas[0].descriptor.aligned_size,
              committed_desc.aligned_size);
    EXPECT_EQ(replicas[0].descriptor.shard_idx, committed_desc.shard_idx);
    EXPECT_EQ(replicas[0].descriptor.file_path, committed_desc.file_path);

    EXPECT_TRUE(recovered.GetBucketIdForKey("committed").has_value());
    // The uncommitted entry must not come back as readable.
    EXPECT_FALSE(recovered.GetBucketIdForKey("pending").has_value());

    // Its space stays reserved, so a new allocation lands after it rather than
    // overwriting the region.
    auto fresh = recovered.Allocate("fresh", 100);
    ASSERT_TRUE(fresh.has_value());
    EXPECT_GE(EntryStartOf(*fresh, "fresh"),
              EntryStartOf(pending_desc, "pending") +
                  pending_desc.aligned_size);
}

TEST(BucketGlobalAllocatorTest, RecoveryDoesNotRevivedFreedKeys) {
    TempDir tmp("bucket_recover_tombstone");
    {
        BucketGlobalAllocator alloc;
        ASSERT_TRUE(
            alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
        auto desc = alloc.Allocate("gone", 100);
        ASSERT_TRUE(desc.has_value());
        ASSERT_TRUE(alloc.MarkCommitted("gone", *desc));
        alloc.Free("gone", *desc);
        // Free() defers its metadata write so it never fsyncs under a caller's
        // lock; make the tombstone durable before simulating the restart.
        EXPECT_GE(alloc.FlushDirtyMetadata(), 1u);
    }

    BucketGlobalAllocator recovered;
    ASSERT_TRUE(
        recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    EXPECT_TRUE(recovered.TakeRecoveredReplicas().empty());
    EXPECT_FALSE(recovered.GetBucketIdForKey("gone").has_value());
}

TEST(BucketGlobalAllocatorTest, DestructorFlushesDeferredTombstones) {
    TempDir tmp("bucket_dtor_flush");
    {
        BucketGlobalAllocator alloc;
        ASSERT_TRUE(
            alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
        auto desc = alloc.Allocate("gone", 100);
        ASSERT_TRUE(desc.has_value());
        ASSERT_TRUE(alloc.MarkCommitted("gone", *desc));
        alloc.Free("gone", *desc);
        // No explicit flush: a clean shutdown must persist it anyway.
    }

    BucketGlobalAllocator recovered;
    ASSERT_TRUE(
        recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    EXPECT_TRUE(recovered.TakeRecoveredReplicas().empty());
    EXPECT_FALSE(recovered.GetBucketIdForKey("gone").has_value());
}

TEST(BucketGlobalAllocatorTest, FlushDirtyMetadataIsIdempotent) {
    TempDir tmp("bucket_flush_idempotent");
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));

    // Nothing dirty yet: Allocate persists synchronously.
    EXPECT_EQ(alloc.FlushDirtyMetadata(), 0u);

    auto desc = alloc.Allocate("k", 100);
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(alloc.FlushDirtyMetadata(), 0u);

    alloc.Free("k", *desc);
    EXPECT_GE(alloc.FlushDirtyMetadata(), 1u);
    // A second flush has nothing left to write.
    EXPECT_EQ(alloc.FlushDirtyMetadata(), 0u);
}

TEST(BucketGlobalAllocatorTest, RecoveryRejectsCorruptAndTruncatedMetadata) {
    // Corrupt checksum: the bucket must be quarantined, not partially trusted.
    {
        TempDir tmp("bucket_recover_corrupt");
        std::string meta_path;
        {
            BucketGlobalAllocator alloc;
            ASSERT_TRUE(
                alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
            auto desc = alloc.Allocate("k", 100);
            ASSERT_TRUE(desc.has_value());
            ASSERT_TRUE(alloc.MarkCommitted("k", *desc));
            const std::string base = tmp.file(
                "bucket_" + BucketGlobalAllocator::FormatBucketId(0) + ".meta");
            meta_path = base + ".0";
        }
        // Flip bytes in the middle of the payload.
        {
            const int fd = ::open(meta_path.c_str(), O_RDWR);
            ASSERT_GE(fd, 0);
            const auto size = std::filesystem::file_size(meta_path);
            std::vector<char> buffer(size);
            ASSERT_EQ(::pread(fd, buffer.data(), size, 0),
                      static_cast<ssize_t>(size));
            for (size_t i = size / 2; i < size; ++i) {
                buffer[i] = static_cast<char>(~buffer[i]);
            }
            ASSERT_EQ(::pwrite(fd, buffer.data(), size, 0),
                      static_cast<ssize_t>(size));
            ::close(fd);
        }
        // Corrupt the other slot as well; this test verifies quarantine when no
        // valid snapshot remains.
        {
            const std::string other = meta_path.substr(0, meta_path.size() - 1) + "1";
            const int fd = ::open(other.c_str(), O_RDWR);
            ASSERT_GE(fd, 0);
            ASSERT_EQ(::ftruncate(fd, 3), 0);
            ::close(fd);
        }

        BucketGlobalAllocator recovered;
        ASSERT_TRUE(
            recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
        EXPECT_TRUE(recovered.TakeRecoveredReplicas().empty());
        EXPECT_FALSE(recovered.GetBucketIdForKey("k").has_value());
    }

    // Truncated metadata file.
    {
        TempDir tmp("bucket_recover_truncated");
        std::string meta_path;
        {
            BucketGlobalAllocator alloc;
            ASSERT_TRUE(
                alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
            auto desc = alloc.Allocate("k", 100);
            ASSERT_TRUE(desc.has_value());
            ASSERT_TRUE(alloc.MarkCommitted("k", *desc));
            const std::string base = tmp.file(
                "bucket_" + BucketGlobalAllocator::FormatBucketId(0) + ".meta");
            meta_path = base + (std::filesystem::exists(base + ".0") ? ".0" : ".1");
        }
        const std::string other =
            meta_path.substr(0, meta_path.size() - 1) +
            (meta_path.back() == '0' ? "1" : "0");
        ASSERT_EQ(::truncate(meta_path.c_str(), 3), 0);
        ASSERT_EQ(::truncate(other.c_str(), 3), 0);

        BucketGlobalAllocator recovered;
        ASSERT_TRUE(
            recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
        EXPECT_TRUE(recovered.TakeRecoveredReplicas().empty());
    }

    // Missing data file: committed metadata must not point at absent data.
    {
        TempDir tmp("bucket_recover_no_data");
        std::string data_path;
        {
            BucketGlobalAllocator alloc;
            ASSERT_TRUE(
                alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
            auto desc = alloc.Allocate("k", 100);
            ASSERT_TRUE(desc.has_value());
            ASSERT_TRUE(alloc.MarkCommitted("k", *desc));
            data_path = desc->file_path;
        }
        std::filesystem::remove(data_path);

        BucketGlobalAllocator recovered;
        ASSERT_TRUE(
            recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
        EXPECT_TRUE(recovered.TakeRecoveredReplicas().empty());
        EXPECT_FALSE(recovered.GetBucketIdForKey("k").has_value());
    }
}

TEST(BucketGlobalAllocatorTest, RecoveryRemovesOrphanDataFiles) {
    TempDir tmp("bucket_recover_orphan");
    const std::string orphan = tmp.file(
        "bucket_" + BucketGlobalAllocator::FormatBucketId(7) + ".data");
    {
        const int fd = ::open(orphan.c_str(), O_CREAT | O_WRONLY, 0644);
        ASSERT_GE(fd, 0);
        ::close(fd);
    }
    ASSERT_TRUE(std::filesystem::exists(orphan));

    BucketGlobalAllocator recovered;
    ASSERT_TRUE(
        recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    // A data file with no valid metadata is unreachable, so it is reclaimed.
    EXPECT_FALSE(std::filesystem::exists(orphan));
    // Bucket ids must not be reused after an orphan at a higher id.
    auto desc = recovered.Allocate("k", 100);
    ASSERT_TRUE(desc.has_value());
    EXPECT_GT(desc->shard_idx, 7);
}

TEST(BucketGlobalAllocatorTest, RecoveryContinuesInterruptedEviction) {
    TempDir tmp("bucket_recover_evicting");
    // Build a bucket, then rewrite its metadata with the evicting marker set,
    // simulating a crash between the marker and the file deletes.
    std::string meta_path;
    std::string data_path;
    {
        BucketGlobalAllocator alloc;
        ASSERT_TRUE(
            alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
        auto desc = alloc.Allocate("k", 100);
        ASSERT_TRUE(desc.has_value());
        ASSERT_TRUE(alloc.MarkCommitted("k", *desc));
        data_path = desc->file_path;
        const std::string base = tmp.file(
            "bucket_" + BucketGlobalAllocator::FormatBucketId(0) + ".meta");
        meta_path = base + ".1";
    }

    {
        // Load, set evicting, recompute the checksum, and write the marker to
        // both slots. This models a marker publication that has completed a
        // durable slot write while making either slot independently decisive.
        for (int slot = 0; slot < 2; ++slot) {
            const std::string slot_path =
                tmp.file("bucket_" + BucketGlobalAllocator::FormatBucketId(0) +
                         ".meta." + std::to_string(slot));
            const auto size = std::filesystem::file_size(slot_path);
            std::string payload(size, '\0');
            const int fd = ::open(slot_path.c_str(), O_RDWR);
            ASSERT_GE(fd, 0);
            ASSERT_EQ(::pread(fd, payload.data(), size, 0),
                      static_cast<ssize_t>(size));
            PersistedBucketMetadata snapshot;
            struct_pb::from_pb(snapshot, payload);
            snapshot.evicting = true;
            snapshot.checksum = 0;
            std::string rechecked;
            struct_pb::to_pb(snapshot, rechecked);
            snapshot.checksum = Crc32cValue(rechecked.data(), rechecked.size());
            std::string final_payload;
            struct_pb::to_pb(snapshot, final_payload);
            ASSERT_EQ(::ftruncate(fd, 0), 0);
            ASSERT_EQ(::pwrite(fd, final_payload.data(), final_payload.size(), 0),
                      static_cast<ssize_t>(final_payload.size()));
            ::close(fd);
        }
    }

    BucketGlobalAllocator recovered;
    ASSERT_TRUE(
        recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    // The eviction is finished rather than the entries being resurrected.
    EXPECT_TRUE(recovered.TakeRecoveredReplicas().empty());
    EXPECT_FALSE(std::filesystem::exists(data_path));
    EXPECT_FALSE(std::filesystem::exists(meta_path));
    EXPECT_FALSE(std::filesystem::exists(meta_path.substr(0, meta_path.size() - 1) +
                                         (meta_path.back() == '0' ? "1" : "0")));
}

TEST(BucketGlobalAllocatorTest, RecoveryPreservesBucketIdSequence) {
    TempDir tmp("bucket_recover_ids");
    const uint64_t capacity = kAlignment;  // one entry per bucket
    {
        BucketGlobalAllocator alloc;
        ASSERT_TRUE(alloc.Init(MakeBucketConfig(tmp.path(), capacity, 8)));
        for (int i = 0; i < 3; ++i) {
            auto desc = alloc.Allocate("k" + std::to_string(i), 100);
            ASSERT_TRUE(desc.has_value());
            ASSERT_TRUE(alloc.MarkCommitted("k" + std::to_string(i), *desc));
        }
    }

    BucketGlobalAllocator recovered;
    ASSERT_TRUE(recovered.Init(MakeBucketConfig(tmp.path(), capacity, 8)));
    EXPECT_EQ(recovered.GetBucketCount(), 3u);
    EXPECT_EQ(recovered.TakeRecoveredReplicas().size(), 3u);

    // New allocations must not reuse an existing bucket id.
    auto desc = recovered.Allocate("after", 100);
    ASSERT_TRUE(desc.has_value());
    EXPECT_EQ(desc->shard_idx, 3);
}

// ---------------------------------------------------------------------------
// BucketGlobalAllocator: eviction
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

TEST(BucketGlobalAllocatorTest, PrepareEvictionSkipsActiveBucket) {
    TempDir tmp("bucket_evict_active");
    const uint64_t capacity = kAlignment;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    // Only one bucket exists and it is the active one, so nothing is evictable
    // even though usage is above the high watermark.
    auto desc = alloc.Allocate("k0", 100);
    ASSERT_TRUE(desc.has_value());
    auto pending = alloc.PrepareEviction();
    EXPECT_TRUE(pending.Empty());
    EXPECT_LT(pending.bucket_id(), 0);
}

TEST(BucketGlobalAllocatorTest, CommitEvictionDeletesBucketAndFrees) {
    TempDir tmp("bucket_evict_commit");
    const uint64_t capacity = kAlignment;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    std::vector<DistributedFSDescriptor> descs;
    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value()) << "allocation " << i;
        ASSERT_TRUE(alloc.MarkCommitted("k" + std::to_string(i), *desc));
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

TEST(BucketGlobalAllocatorTest, AbortEvictionRestoresBucketUnchanged) {
    TempDir tmp("bucket_evict_abort");
    const uint64_t capacity = kAlignment;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
        ASSERT_TRUE(alloc.MarkCommitted("k" + std::to_string(i), *desc));
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

TEST(BucketGlobalAllocatorTest, UnresolvedPendingEvictionSelfAborts) {
    TempDir tmp("bucket_evict_dtor");
    const uint64_t capacity = kAlignment;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
        ASSERT_TRUE(alloc.MarkCommitted("k" + std::to_string(i), *desc));
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

TEST(BucketGlobalAllocatorTest, CommitAndAbortEvictionAreIdempotent) {
    TempDir tmp("bucket_evict_idempotent");
    const uint64_t capacity = kAlignment;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
        ASSERT_TRUE(alloc.MarkCommitted("k" + std::to_string(i), *desc));
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

TEST(BucketGlobalAllocatorTest, EvictionCandidatesExcludeFreedKeys) {
    TempDir tmp("bucket_evict_freed");
    const uint64_t capacity = 2 * kAlignment;
    BucketGlobalAllocator alloc;
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

TEST(BucketGlobalAllocatorTest, EvictionUsesFixedCapacityDenominator) {
    TempDir tmp("bucket_evict_watermark");
    const uint64_t capacity = kAlignment;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 8)));

    // Total capacity stays max_bucket_count * bucket_capacity regardless of how
    // many buckets exist, so deleting one lowers usage instead of shrinking the
    // denominator in lockstep (which would never converge).
    EXPECT_EQ(alloc.GetTotalCapacity(), 8u * capacity);
    EXPECT_EQ(alloc.GetUsedBytes(), 0u);

    for (int i = 0; i < 5; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
        ASSERT_TRUE(alloc.MarkCommitted("k" + std::to_string(i), *desc));
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

TEST(BucketGlobalAllocatorTest, EvictionStopsAtLowWatermark) {
    TempDir tmp("bucket_evict_low");
    const uint64_t capacity = kAlignment;
    BucketGlobalAllocator alloc;
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

TEST(BucketGlobalAllocatorTest, EvictionDisabledYieldsNoCandidates) {
    TempDir tmp("bucket_evict_disabled");
    const uint64_t capacity = kAlignment;
    auto config = MakeEvictionConfig(tmp.path(), capacity, 2);
    config.eviction_enabled = false;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(config));
    EXPECT_FALSE(alloc.IsEvictionEnabled());
}

TEST(BucketGlobalAllocatorTest, ReadableDescriptorSurvivesConcurrentEviction) {
    TempDir tmp("bucket_evict_inflight");
    const uint64_t capacity = kAlignment;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(MakeEvictionConfig(tmp.path(), capacity, 4)));

    std::vector<DistributedFSDescriptor> descs;
    for (int i = 0; i < 3; ++i) {
        auto desc = alloc.Allocate("k" + std::to_string(i), 100);
        ASSERT_TRUE(desc.has_value());
        ASSERT_TRUE(alloc.MarkCommitted("k" + std::to_string(i), *desc));
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

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

TEST(BucketGlobalAllocatorTest, ConcurrentAllocateProducesDisjointRegions) {
    TempDir tmp("bucket_concurrent_alloc");
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(
        alloc.Init(MakeBucketConfig(tmp.path(), 4 * 1024 * 1024, 32)));

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
            const uint64_t begin = EntryStartOf(desc, key);
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

TEST(BucketGlobalAllocatorTest, ConcurrentFreeAndUpdateAccessAreSafe) {
    TempDir tmp("bucket_concurrent_free");
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(
        alloc.Init(MakeBucketConfig(tmp.path(), 1024 * 1024, 16)));

    constexpr int kKeys = 64;
    std::vector<std::pair<std::string, DistributedFSDescriptor>> allocations;
    for (int i = 0; i < kKeys; ++i) {
        const std::string key = "k" + std::to_string(i);
        auto desc = alloc.Allocate(key, 512);
        ASSERT_TRUE(desc.has_value());
        ASSERT_TRUE(alloc.MarkCommitted(key, *desc));
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

TEST(GlobalAllocatorInterfaceTest, BothImplementationsWorkPolymorphically) {
    TempDir bucket_dir("iface_bucket");
    TempDir shard_dir("iface_shard");

    std::vector<std::unique_ptr<GlobalAllocatorInterface>> allocators;
    {
        auto bucket = std::make_unique<BucketGlobalAllocator>();
        ASSERT_TRUE(bucket->Init(
            MakeBucketConfig(bucket_dir.path(), kBucketCapacity, 8)));
        allocators.push_back(std::move(bucket));
    }
    {
        auto shard = std::make_unique<DfsGlobalAllocator>();
        ASSERT_TRUE(shard->Init(
            MakeShardConfig(shard_dir.path(), 4, 1024 * 1024)));
        allocators.push_back(std::move(shard));
    }

    for (auto& allocator : allocators) {
        EXPECT_TRUE(allocator->IsInitialized());
        auto desc = allocator->Allocate("poly_key", 100);
        ASSERT_TRUE(desc.has_value())
            << "type=" << ToString(allocator->Type());
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

    const std::string formatted = bucket.FormatStr();
    EXPECT_NE(formatted.find("allocator_type=bucket"), std::string::npos);
    EXPECT_NE(formatted.find("bucket_capacity="), std::string::npos);
}

// ---------------------------------------------------------------------------
// Entry layout helper
// ---------------------------------------------------------------------------

TEST(BucketEntryLayoutTest, ComputesAlignedEntriesAndRejectsOverflow) {
    const uint64_t alignment = 4096;

    auto layout = ComputeBucketEntryLayout(0, 4, 100, alignment);
    ASSERT_TRUE(layout.has_value());
    EXPECT_EQ(layout->entry_start, 0u);
    EXPECT_EQ(layout->value_offset, 8u + 4u);
    EXPECT_EQ(layout->entry_size, 8u + 4u + 100u);
    EXPECT_EQ(layout->reserved_size, alignment);
    EXPECT_EQ(layout->entry_end(), alignment);

    // A non-aligned cursor is rounded up to the next boundary.
    auto next = ComputeBucketEntryLayout(1, 4, 100, alignment);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next->entry_start, alignment);

    // Invalid inputs.
    EXPECT_FALSE(ComputeBucketEntryLayout(0, 0, 100, alignment).has_value());
    EXPECT_FALSE(ComputeBucketEntryLayout(0, 4, 0, alignment).has_value());
    EXPECT_FALSE(ComputeBucketEntryLayout(0, 4, 100, 0).has_value());
    EXPECT_FALSE(ComputeBucketEntryLayout(0, 4, 100, 3000).has_value());

    // Overflow must be rejected rather than wrapping around.
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    EXPECT_FALSE(ComputeBucketEntryLayout(0, 4, kMax, alignment).has_value());
    EXPECT_FALSE(ComputeBucketEntryLayout(kMax, 4, 100, alignment).has_value());
    EXPECT_FALSE(ComputeBucketEntryLayout(0, kMax, 100, alignment).has_value());

    EXPECT_FALSE(CheckedAlignUp(kMax, alignment).has_value());
    EXPECT_TRUE(IsValidBucketAlignment(4096));
    EXPECT_FALSE(IsValidBucketAlignment(0));
    EXPECT_FALSE(IsValidBucketAlignment(3000));
}

TEST(BucketEntryLayoutTest, RebuildMatchesComputeAndRejectsMisalignment) {
    const uint64_t alignment = 4096;
    auto computed = ComputeBucketEntryLayout(alignment, 6, 200, alignment);
    ASSERT_TRUE(computed.has_value());

    auto rebuilt = RebuildBucketEntryLayout(computed->entry_start, 6, 200,
                                            alignment);
    ASSERT_TRUE(rebuilt.has_value());
    EXPECT_EQ(rebuilt->entry_start, computed->entry_start);
    EXPECT_EQ(rebuilt->value_offset, computed->value_offset);
    EXPECT_EQ(rebuilt->reserved_size, computed->reserved_size);

    // A recorded entry start that is not aligned indicates corrupt metadata.
    EXPECT_FALSE(RebuildBucketEntryLayout(1, 6, 200, alignment).has_value());
}

TEST(BucketEntryLayoutTest, DescriptorConstructionIsCentralized) {
    const uint64_t alignment = 4096;
    const std::string key = "abcd";
    auto layout = ComputeBucketEntryLayout(0, key.size(), 100, alignment);
    ASSERT_TRUE(layout.has_value());

    auto desc = MakeBucketDescriptor("/tmp/bucket_000001.data", *layout, 100, 1);
    EXPECT_EQ(desc.file_path, "/tmp/bucket_000001.data");
    EXPECT_EQ(desc.offset, layout->value_offset);
    EXPECT_EQ(desc.object_size, 100u);
    EXPECT_EQ(desc.aligned_size, layout->reserved_size);
    EXPECT_EQ(desc.shard_idx, 1);
    // The value offset is intentionally not alignment-aligned in bucket mode.
    EXPECT_NE(desc.offset % alignment, 0u);
}


// ---------------------------------------------------------------------------
// Append-only metadata log: crash replay and concurrent compaction
// ---------------------------------------------------------------------------

TEST(BucketMetadataLogTest, ThresholdCompactionStaysOffPutHotPath) {
    TempDir tmp("bucket_log_deferred_compaction");
    auto config = MakeBucketConfig(tmp.path(), 4 * 1024 * 1024, 8);
    config.bucket_meta_log_threshold = 1 << 20;

    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(config));

    // Two records for a large key cross the 1 MiB minimum threshold. Neither
    // Allocate nor MarkCommitted may compact synchronously: PutEnd must remain
    // one append plus fdatasync and perform no snapshot namespace operation.
    const std::string key(600 * 1024, 'k');
    auto desc = alloc.Allocate(key, 100);
    ASSERT_TRUE(desc.has_value());
    ASSERT_TRUE(alloc.MarkCommitted(key, *desc));
    EXPECT_GE(alloc.GetLogBytes(0), config.bucket_meta_log_threshold);

    const std::string base =
        tmp.file("bucket_" + BucketGlobalAllocator::FormatBucketId(0) + ".meta");
    EXPECT_FALSE(std::filesystem::exists(base));
    EXPECT_NE(std::filesystem::exists(base + ".0"),
              std::filesystem::exists(base + ".1"));

    // The maintenance call notices an already-durable oversized log, writes the
    // inactive stable slot, and clears the log without rename.
    alloc.FlushDirtyMetadata();
    EXPECT_EQ(alloc.GetLogBytes(0), 0u);
    EXPECT_TRUE(std::filesystem::exists(base + ".0"));
    EXPECT_TRUE(std::filesystem::exists(base + ".1"));
}

TEST(BucketMetadataLogTest, AbruptRestartReplaysCommittedLog) {
    TempDir tmp("bucket_log_replay");
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        BucketGlobalAllocator alloc;
        if (!alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8))) {
            ::_exit(10);
        }
        auto first = alloc.Allocate("replay_a", 100);
        auto second = alloc.Allocate("replay_b", 100);
        if (!first || !second || !alloc.MarkCommitted("replay_a", *first) ||
            !alloc.MarkCommitted("replay_b", *second)) {
            ::_exit(11);
        }
        // Simulate a process crash: the append-only log was fdatasync'd, but
        // the destructor never gets a chance to compact it into the snapshot.
        ::_exit(0);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);

    BucketGlobalAllocator recovered;
    ASSERT_TRUE(recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    auto replicas = recovered.TakeRecoveredReplicas();
    ASSERT_EQ(replicas.size(), 2u);
    EXPECT_TRUE(recovered.GetBucketIdForKey("replay_a").has_value());
    EXPECT_TRUE(recovered.GetBucketIdForKey("replay_b").has_value());
    // Recovery immediately compacts the replayed state and clears the log.
    EXPECT_EQ(recovered.GetLogBytes(0), 0u);
}

TEST(BucketMetadataLogTest, CorruptRecordDoesNotDiscardFollowingRecords) {
    TempDir tmp("bucket_log_corrupt_record");
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        BucketGlobalAllocator alloc;
        if (!alloc.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8))) {
            ::_exit(20);
        }
        for (const char* key : {"log_a", "log_b", "log_c"}) {
            auto desc = alloc.Allocate(key, 100);
            if (!desc || !alloc.MarkCommitted(key, *desc)) ::_exit(21);
        }
        ::_exit(0);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);

    const auto log_path = tmp.file(
        "bucket_" + BucketGlobalAllocator::FormatBucketId(0) + ".meta.log");
    ASSERT_TRUE(std::filesystem::exists(log_path));
    const auto size = std::filesystem::file_size(log_path);
    ASSERT_GT(size, static_cast<uintmax_t>(kMetaLogHeaderSize));
    // Damage the first record's body while leaving subsequent magic/framing
    // intact. Per-record CRC and magic resynchronization must preserve log_b/c.
    const int fd = ::open(log_path.c_str(), O_RDWR);
    ASSERT_GE(fd, 0);
    unsigned char byte = 0;
    ASSERT_EQ(::pread(fd, &byte, 1, kMetaLogHeaderSize - 1), 1);
    byte ^= 0x5a;
    ASSERT_EQ(::pwrite(fd, &byte, 1, kMetaLogHeaderSize - 1), 1);
    ASSERT_EQ(::fsync(fd), 0);
    ::close(fd);

    BucketGlobalAllocator recovered;
    ASSERT_TRUE(recovered.Init(MakeBucketConfig(tmp.path(), kBucketCapacity, 8)));
    // The damaged ADD_PENDING may be reconstructed by its later valid
    // MARK_COMMITTED record. What matters here is that corruption is isolated:
    // records after it remain replayable rather than the whole log being lost.
    EXPECT_TRUE(recovered.GetBucketIdForKey("log_b").has_value());
    EXPECT_TRUE(recovered.GetBucketIdForKey("log_c").has_value());
}

TEST(BucketMetadataLogTest, CompactionRacesWithConcurrentAllocateAndCommit) {
    TempDir tmp("bucket_log_compaction_race");
    auto config = MakeBucketConfig(tmp.path(), 4 * 1024 * 1024, 16);
    // Force the compaction path to run repeatedly, including while writers are
    // producing new records. A unique temp suffix is required for this test.
    config.bucket_meta_log_threshold = 1 << 20;
    BucketGlobalAllocator alloc;
    ASSERT_TRUE(alloc.Init(config));

    constexpr int kThreads = 6;
    constexpr int kPerThread = 20;
    std::atomic<bool> start{false};
    std::vector<std::thread> writers;
    for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([&alloc, &start, t]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int i = 0; i < kPerThread; ++i) {
                const std::string key = "race_" + std::to_string(t) + "_" +
                                        std::to_string(i);
                auto desc = alloc.Allocate(key, 100);
                if (desc) {
                    EXPECT_TRUE(alloc.MarkCommitted(key, *desc));
                }
            }
        });
    }
    std::thread compactor([&alloc, &start]() {
        start.store(true, std::memory_order_release);
        for (int i = 0; i < 80; ++i) {
            (void)alloc.CompactAllBuckets();
            std::this_thread::yield();
        }
    });
    for (auto& writer : writers) writer.join();
    compactor.join();
    (void)alloc.CompactAllBuckets();

    // Every successful writer's committed entry must remain indexed and every
    // published snapshot/log pair must be recoverable.
    BucketGlobalAllocator recovered;
    ASSERT_TRUE(recovered.Init(config));
    EXPECT_EQ(recovered.TakeRecoveredReplicas().size(),
              static_cast<size_t>(kThreads * kPerThread));
}

TEST(BucketMetadataLogTest, RecoveryReusesActiveBucketTail) {
    TempDir tmp("bucket_log_active_tail");
    const auto config = MakeBucketConfig(tmp.path(), 8 * kAlignment, 8);
    DistributedFSDescriptor old_desc;
    {
        const pid_t child = ::fork();
        ASSERT_GE(child, 0);
        if (child == 0) {
            BucketGlobalAllocator alloc;
            if (!alloc.Init(config)) ::_exit(30);
            auto desc = alloc.Allocate("tail_old", 100);
            if (!desc || !alloc.MarkCommitted("tail_old", *desc)) ::_exit(31);
            ::_exit(0);
        }
        int status = 0;
        ASSERT_EQ(::waitpid(child, &status, 0), child);
        ASSERT_TRUE(WIFEXITED(status));
        ASSERT_EQ(WEXITSTATUS(status), 0);
    }
    BucketGlobalAllocator recovered;
    ASSERT_TRUE(recovered.Init(config));
    auto next = recovered.Allocate("tail_new", 100);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next->shard_idx, 0);
    EXPECT_GE(EntryStartOf(*next, "tail_new"), kAlignment);
}

}  // namespace mooncake::test
