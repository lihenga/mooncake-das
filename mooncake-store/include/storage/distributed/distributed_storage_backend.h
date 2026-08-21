#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "fs_adapter.h"
#include "replica.h"
#include "storage/distributed/global_allocator_interface.h"
#include "storage_backend.h"

namespace mooncake {

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
    // BUCKET mode only: how large a bucket's metadata log may grow before it is
    // folded into a fresh snapshot. 0 selects the default derived from
    // `bucket_capacity` (see ResolveBucketMetaLogThreshold).
    uint64_t bucket_meta_log_threshold = 0;
    // Cleared by FromEnvironment() when MOONCAKE_DFS_ALLOCATOR_TYPE names an
    // unknown allocator, so the master can reject the configuration instead of
    // silently defaulting to SHARD.
    bool allocator_type_valid = true;

    bool Validate() const;
    bool ValidateForAllocator() const;
    bool ValidateForBucketAllocator() const;
    static DistributedStorageConfig FromEnvironment();
    std::string FormatStr() const;

    /**
     * @brief Effective metadata-log compaction threshold in bytes.
     *
     * Returns `bucket_meta_log_threshold` when set explicitly, otherwise a
     * fraction of the bucket capacity capped at a few megabytes: large enough
     * that compaction is rare, small enough that replay after a crash stays
     * cheap.
     */
    uint64_t ResolveBucketMetaLogThreshold() const;
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

    bool initialized_ = false;
};

}  // namespace mooncake
