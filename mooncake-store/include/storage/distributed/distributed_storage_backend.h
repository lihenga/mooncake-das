#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "fs_adapter.h"
#include "replica.h"
#include "storage/distributed/global_allocator_interface.h"
#include "storage_backend.h"

namespace mooncake {

class ThreadPool;

struct DistributedStorageConfig {
    std::string fsdir = "/mnt/3fs/mooncake";
    std::string fs_adapter_type = "hf3fs";
    bool enable_health_check = false;
    int shard_count = 64;
    uint64_t shard_capacity = 4ULL * 1024 * 1024 * 1024;
    uint64_t alignment = 4096;
    bool single_tenant = true;
    bool eviction_enabled = true;
    double eviction_high_watermark = 0.9;
    double eviction_low_watermark = 0.7;
    std::chrono::seconds deferred_free_duration{30};
    std::chrono::seconds eviction_check_interval{5};

    // Which space-management strategy the master uses. SHARD is the default so
    // existing deployments are unaffected.
    DfsAllocatorType allocator_type = DfsAllocatorType::SHARD;
    // BUCKET mode only: size of each append-only bucket data file.
    uint64_t bucket_capacity = 256ULL * 1024 * 1024;
    // BUCKET mode only: upper bound on live buckets. Also the fixed
    // denominator for eviction watermarks, so it must be > 0.
    int64_t max_bucket_count = 256;
    // BUCKET mode only: number of threads for parallel batch reads across
    // buckets. Set to 1 to disable parallelism. Must be in
    // [1, kMaxBatchReadThreads].
    int batch_read_threads = 4;
    // Cleared by FromEnvironment() when MOONCAKE_DFS_ALLOCATOR_TYPE names an
    // unknown allocator, so the master can reject the configuration instead of
    // silently defaulting to SHARD.
    bool allocator_type_valid = true;

    bool Validate() const;
    bool ValidateForAllocator() const;
    bool ValidateForBucketAllocator() const;
    static DistributedStorageConfig FromEnvironment();
    std::string FormatStr() const;
};

struct DfsWriteRequest {
    std::string key;
    DistributedFSDescriptor descriptor;
    std::vector<Slice> slices;
};

struct DfsReadRequest {
    std::string key;
    DistributedFSDescriptor descriptor;
    std::vector<Slice> slices;
};

/**
 * @brief Distributed filesystem storage backend.
 *
 * Implements StorageBackendInterface, delegating I/O to a FileSystemAdapter.
 * Does not handle eviction (DFS manages its own space).
 */
class DistributedStorageBackend : public StorageBackendInterface {
   public:
    DistributedStorageBackend(
        const FileStorageConfig& file_storage_config,
        const DistributedStorageConfig& distributed_config,
        std::unique_ptr<FileSystemAdapter> fs_adapter);
    ~DistributedStorageBackend() override;

    tl::expected<void, ErrorCode> Init() override;

    tl::expected<int64_t, ErrorCode> BatchOffload(
        const std::unordered_map<std::string, std::vector<Slice>>& batch_object,
        std::function<ErrorCode(const std::vector<std::string>& keys,
                                std::vector<StorageObjectMetadata>& metadatas)>
            complete_handler,
        EvictionHandler eviction_handler = nullptr) override;

    std::vector<tl::expected<void, ErrorCode>> BatchWrite(
        const std::vector<DfsWriteRequest>& requests);

    std::vector<tl::expected<void, ErrorCode>> BatchRead(
        const std::vector<DfsReadRequest>& requests);

    // Key-only storage backend operations cannot safely address DFS objects;
    // callers must use BatchRead/BatchWrite with request-scoped descriptors.
    tl::expected<void, ErrorCode> BatchLoad(
        std::unordered_map<std::string, Slice>& batched_slices) override;

    tl::expected<bool, ErrorCode> IsExist(const std::string& key) override;

    tl::expected<bool, ErrorCode> IsEnableOffloading() override;

    tl::expected<void, ErrorCode> ScanMeta(
        const std::function<ErrorCode(
            const std::vector<std::string>& keys,
            std::vector<StorageObjectMetadata>& metadatas)>& handler) override;

    /**
     * @brief Which DFS space-management mode this backend was configured for.
     *
     * The client uses this to choose between the synchronous SHARD write path
     * and the asynchronous BUCKET one.
     */
    DfsAllocatorType GetAllocatorType() const {
        return distributed_config_.allocator_type;
    }

   private:
    struct ShardFile {
        std::string path;
        int fd = -1;
        std::mutex mutex;
    };

    // BUCKET mode opens bucket data files on demand and caches the handles.
    // Handles are shared_ptr so an in-flight read/write keeps the fd alive even
    // if the cache entry is dropped (e.g. after the bucket is evicted).
    struct OpenFileHandle {
        std::string path;
        int fd = -1;
        FileSystemAdapter* adapter = nullptr;
        std::mutex mutex;

        ~OpenFileHandle() {
            if (fd >= 0 && adapter != nullptr) {
                (void)adapter->CloseFile(fd);
                fd = -1;
            }
        }
    };

    bool IsBucketMode() const {
        return distributed_config_.allocator_type == DfsAllocatorType::BUCKET;
    }

    /**
     * @brief A validated, ready-to-use target for one DFS request.
     *
     * `fd` and `mutex` are borrowed: in SHARD mode they belong to the
     * long-lived ShardFile, in BUCKET mode to `keepalive`, whose shared_ptr
     * guarantees the fd stays open for the duration of the I/O even if the
     * cache entry is dropped concurrently.
     */
    struct ResolvedTarget {
        int fd = -1;
        std::mutex* mutex = nullptr;
        std::shared_ptr<OpenFileHandle> keepalive;
    };

    /**
     * @brief Resolve `descriptor` to a validated I/O target.
     *
     * SHARD mode validates against the fixed shard table. BUCKET mode
     * canonicalizes the descriptor path, verifies it names the expected bucket
     * data file under the configured DFS root, and then opens/caches it.
     */
    tl::expected<ResolvedTarget, ErrorCode> ResolveTarget(
        const DistributedFSDescriptor& descriptor, const std::string& key);

    tl::expected<std::shared_ptr<OpenFileHandle>, ErrorCode> GetOrOpenBucket(
        const std::string& path);

    /**
     * @brief One validated BUCKET-mode read, resolved but not yet issued.
     */
    struct PreparedRead {
        size_t request_index = 0;
        ResolvedTarget target;
        uint64_t entry_start = 0;
        uint64_t reserved_size = 0;
    };

    /**
     * @brief Every read of one batch that targets the same bucket data file.
     *
     * Keyed on the handle's mutex rather than its fd: an fd number can be
     * reused once a handle is closed, while the mutex address uniquely
     * identifies the open handle for as long as a PreparedRead in the group
     * keeps it alive through `ResolvedTarget::keepalive`.
     */
    struct BucketReadGroup {
        std::mutex* mutex = nullptr;
        std::vector<PreparedRead> reads;
    };

    /**
     * @brief A run of contiguous entries collapsed into a single read.
     */
    struct MergedIo {
        uint64_t entry_start = 0;
        uint64_t total_size = 0;
        std::vector<const PreparedRead*> reads;
    };

    static ErrorCode ReadFully(FileSystemAdapter* fs_adapter,
                               const ResolvedTarget& target, uint64_t offset,
                               std::span<char> output);

    /// Scatter `value` (object_size bytes) across the request's slices.
    static void CopyToSlices(const DfsReadRequest& request, const char* value);

    /// Consumes `prepared`, bucketing its entries by open file handle.
    static std::unordered_map<std::mutex*, BucketReadGroup> GroupReadsByBucket(
        std::vector<PreparedRead>&& prepared);

    static void SortGroupByOffset(BucketReadGroup& group);

    static std::vector<MergedIo> BuildMergedIos(const BucketReadGroup& group);

    static void ExecuteMergedRead(
        const MergedIo& io, const std::vector<DfsReadRequest>& requests,
        std::vector<tl::expected<void, ErrorCode>>& results,
        const ResolvedTarget& target, std::mutex* mutex,
        FileSystemAdapter* fs_adapter, std::vector<char>& staging);

    static void FailGroupReads(
        const BucketReadGroup& group,
        std::vector<tl::expected<void, ErrorCode>>& results, ErrorCode error);

    static void ProcessBucketGroup(
        BucketReadGroup& group, const std::vector<DfsReadRequest>& requests,
        std::vector<tl::expected<void, ErrorCode>>& results,
        FileSystemAdapter* fs_adapter);

    static void DispatchParallelReads(
        std::unordered_map<std::mutex*, BucketReadGroup>& groups,
        const std::vector<DfsReadRequest>& requests,
        std::vector<tl::expected<void, ErrorCode>>& results, ThreadPool& pool,
        FileSystemAdapter* fs_adapter);

    std::unique_ptr<FileSystemAdapter> fs_adapter_;
    DistributedStorageConfig distributed_config_;
    std::string root_dir_;
    // Canonical form of root_dir_, used to reject descriptor paths that try to
    // escape the configured DFS root.
    std::string canonical_root_dir_;
    std::vector<std::unique_ptr<ShardFile>> shard_files_;

    mutable std::mutex bucket_cache_mutex_;
    std::unordered_map<std::string, std::shared_ptr<OpenFileHandle>>
        bucket_cache_;
    std::unique_ptr<ThreadPool> batch_read_pool_;

    bool initialized_ = false;
};

}  // namespace mooncake
