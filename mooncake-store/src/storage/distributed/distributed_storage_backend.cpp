#include "storage/distributed/distributed_storage_backend.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <sstream>

#include "environ.h"
#include "storage/distributed/bucket_entry_layout.h"
#include "storage/distributed/bucket_global_allocator.h"
#include "storage/distributed/dfs_global_allocator.h"
#include "thread_pool.h"
#include "types.h"
#include "utils.h"

namespace mooncake {

namespace {

// Upper bound on MOONCAKE_DFS_BATCH_READ_THREADS. Each parallel bucket read
// can hold a staging buffer of up to kMaxMergedIo, so an unbounded thread
// count would blow up both the thread and the memory budget.
constexpr int kMaxBatchReadThreads = 256;

// Bound temporary memory and I/O latency while still collapsing the common
// BatchAllocate output to one read. Larger batches become a few contiguous
// reads rather than one read per object.
constexpr uint64_t kMaxMergedIo = 4ULL * 1024 * 1024;

bool IsDfsDescriptorRangeValid(const DistributedFSDescriptor& desc,
                               const DistributedStorageConfig& config) {
    if (config.alignment == 0 || desc.object_size == 0 ||
        desc.aligned_size < desc.object_size ||
        desc.offset % config.alignment != 0 ||
        desc.aligned_size % config.alignment != 0) {
        return false;
    }
    if (desc.offset > config.shard_capacity ||
        desc.aligned_size > config.shard_capacity - desc.offset) {
        return false;
    }

    constexpr uint64_t kMaxFileOffset =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    return desc.offset <= kMaxFileOffset &&
           desc.aligned_size <= kMaxFileOffset - desc.offset;
}

/**
 * @brief Validate a BUCKET-mode descriptor.
 *
 * Unlike SHARD, the *value* offset of a bucket entry is generally not
 * alignment-aligned (it sits after an 8-byte header plus the key), so the
 * alignment check applies to the entry start and reserved size instead.
 */
bool IsBucketDescriptorRangeValid(const DistributedFSDescriptor& desc,
                                  const std::string& key,
                                  const DistributedStorageConfig& config) {
    if (desc.object_size == 0 || desc.shard_idx < 0) return false;
    if (key.empty()) return false;

    const uint64_t header_and_key =
        BucketEntryLayout::kHeaderSize + key.size();
    if (desc.offset < header_and_key) return false;
    const uint64_t entry_start = desc.offset - header_and_key;

    auto layout = RebuildBucketEntryLayout(entry_start, key.size(),
                                           desc.object_size, config.alignment);
    if (!layout) return false;
    if (layout->value_offset != desc.offset) return false;
    if (layout->reserved_size != desc.aligned_size) return false;
    if (layout->entry_end() > config.bucket_capacity) return false;

    constexpr uint64_t kMaxFileOffset =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    return layout->entry_start <= kMaxFileOffset &&
           layout->reserved_size <= kMaxFileOffset - layout->entry_start;
}

/**
 * @brief Canonicalize `path` without requiring it to exist yet.
 *
 * `weakly_canonical` collapses `..` and resolves the existing prefix through
 * symlinks, which is what we need to reject descriptor paths pointing outside
 * the configured DFS root.
 */
std::string CanonicalizePath(const std::string& path) {
    std::error_code ec;
    auto canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec) return {};
    return canonical.lexically_normal().string();
}

/**
 * @brief True when `canonical_path` lies inside `canonical_root`.
 */
bool IsPathWithinRoot(const std::string& canonical_path,
                      const std::string& canonical_root) {
    if (canonical_path.empty() || canonical_root.empty()) return false;
    const std::filesystem::path path(canonical_path);
    const std::filesystem::path root(canonical_root);
    auto path_it = path.begin();
    for (auto root_it = root.begin(); root_it != root.end(); ++root_it) {
        if (path_it == path.end() || *path_it != *root_it) return false;
        ++path_it;
    }
    // Require at least one component below the root so the root directory
    // itself is never accepted as a data file.
    return path_it != path.end();
}

/**
 * @brief Check that `path` is the bucket data file of `bucket_id`.
 */
bool MatchesBucketDataFileName(const std::string& path, int64_t bucket_id) {
    const std::string expected =
        "bucket_" + BucketGlobalAllocator::FormatBucketId(bucket_id) + ".data";
    return std::filesystem::path(path).filename().string() == expected;
}

}  // namespace

std::optional<DfsAllocatorType> ParseDfsAllocatorType(std::string_view name) {
    if (name == "shard" || name == "SHARD") return DfsAllocatorType::SHARD;
    if (name == "bucket" || name == "BUCKET") return DfsAllocatorType::BUCKET;
    return std::nullopt;
}

const char* ToString(DfsAllocatorType type) {
    switch (type) {
        case DfsAllocatorType::SHARD:
            return "shard";
        case DfsAllocatorType::BUCKET:
            return "bucket";
    }
    return "unknown";
}

bool DistributedStorageConfig::Validate() const {
    if (fsdir.empty()) {
        LOG(ERROR) << "DistributedStorageConfig: fsdir is empty";
        return false;
    }
    if (!std::filesystem::path(fsdir).is_absolute()) {
        LOG(ERROR)
            << "DistributedStorageConfig: fsdir must be an absolute path: "
            << fsdir;
        return false;
    }
    if (fs_adapter_type != "hf3fs" && fs_adapter_type != "posix") {
        LOG(ERROR) << "DistributedStorageConfig: unsupported fs_adapter_type: "
                   << fs_adapter_type;
        return false;
    }
    if (shard_count <= 0) {
        LOG(ERROR) << "DistributedStorageConfig: shard_count must > 0";
        return false;
    }
    if (shard_capacity == 0) {
        LOG(ERROR) << "DistributedStorageConfig: shard_capacity must > 0";
        return false;
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        LOG(ERROR) << "DistributedStorageConfig: alignment must be power of 2";
        return false;
    }
    if (shard_capacity % alignment != 0) {
        LOG(ERROR) << "DistributedStorageConfig: shard_capacity must align";
        return false;
    }
    if (!single_tenant) {
        LOG(ERROR) << "DistributedStorageConfig: Currently, DFS requires "
                      "single_tenant=true";
        return false;
    }
    return true;
}

bool DistributedStorageConfig::ValidateForAllocator() const {
    if (!Validate()) return false;

    if (eviction_low_watermark < 0.0 || eviction_low_watermark > 1.0 ||
        eviction_high_watermark < 0.0 || eviction_high_watermark > 1.0 ||
        eviction_low_watermark >= eviction_high_watermark) {
        LOG(ERROR) << "DistributedStorageConfig: eviction watermarks must "
                      "satisfy 0 <= low < high <= 1, low="
                   << eviction_low_watermark
                   << ", high=" << eviction_high_watermark;
        return false;
    }
    if (deferred_free_duration.count() < 0) {
        LOG(ERROR) << "DistributedStorageConfig: deferred_free_duration must "
                      "be non-negative, seconds="
                   << deferred_free_duration.count();
        return false;
    }
    if (eviction_enabled && eviction_check_interval.count() <= 0) {
        LOG(ERROR) << "DistributedStorageConfig: eviction_check_interval must "
                      "be positive when eviction is enabled, seconds="
                   << eviction_check_interval.count();
        return false;
    }
    return true;
}

bool DistributedStorageConfig::ValidateForBucketAllocator() const {
    if (bucket_capacity == 0) {
        LOG(ERROR) << "DistributedStorageConfig: bucket_capacity must > 0";
        return false;
    }
    if (!IsValidBucketAlignment(alignment)) {
        LOG(ERROR) << "DistributedStorageConfig: alignment must be a power of "
                      "2, alignment="
                   << alignment;
        return false;
    }
    if (bucket_capacity % alignment != 0) {
        LOG(ERROR) << "DistributedStorageConfig: bucket_capacity ("
                   << bucket_capacity << ") must be a multiple of alignment ("
                   << alignment << ")";
        return false;
    }
    // A single entry must be able to fit: header + key + a non-empty value.
    if (bucket_capacity <= BucketEntryLayout::kHeaderSize) {
        LOG(ERROR) << "DistributedStorageConfig: bucket_capacity is too small "
                      "to hold a single entry, bucket_capacity="
                   << bucket_capacity;
        return false;
    }
    if (max_bucket_count <= 0) {
        // A fixed denominator is required for stable eviction watermarks.
        LOG(ERROR) << "DistributedStorageConfig: max_bucket_count must > 0 in "
                      "bucket allocator mode, max_bucket_count="
                   << max_bucket_count;
        return false;
    }
    if (max_bucket_count > kMaxBucketId) {
        LOG(ERROR) << "DistributedStorageConfig: max_bucket_count ("
                   << max_bucket_count << ") exceeds the addressable maximum ("
                   << kMaxBucketId << ")";
        return false;
    }
    if (static_cast<uint64_t>(max_bucket_count) >
        std::numeric_limits<uint64_t>::max() / bucket_capacity) {
        LOG(ERROR) << "DistributedStorageConfig: max_bucket_count * "
                      "bucket_capacity overflows";
        return false;
    }
    if (batch_read_threads < 1 || batch_read_threads > kMaxBatchReadThreads) {
        LOG(ERROR) << "DistributedStorageConfig: batch_read_threads must be in "
                      "[1, "
                   << kMaxBatchReadThreads
                   << "], batch_read_threads=" << batch_read_threads;
        return false;
    }
    return true;
}

bool DistributedStorageConfig::IsReplicaEnabledFromEnvironment() {
    const bool enabled = Environ::GetBool("MOONCAKE_DFS_FORCE_ONE_REPLICA", false);
    static std::once_flag log_once;
    std::call_once(log_once, [enabled] {
        LOG(INFO) << "MOONCAKE_DFS_FORCE_ONE_REPLICA is "
                  << (enabled ? "true" : "false");
    });
    return enabled;
}

DistributedStorageConfig DistributedStorageConfig::FromEnvironment() {
    DistributedStorageConfig config;
    config.fsdir = Environ::GetString(
        "MOONCAKE_DFS_ROOT_DIR",
        Environ::GetString("MOONCAKE_DISTRIBUTED_ROOT_DIR", config.fsdir));
    if (!std::filesystem::path(config.fsdir).is_absolute()) {
        config.fsdir = std::filesystem::absolute(config.fsdir).string();
    }
    config.fs_adapter_type =
        Environ::GetString("MOONCAKE_DFS_FS_ADAPTER",
                           Environ::GetString("MOONCAKE_DISTRIBUTED_FS_TYPE",
                                              config.fs_adapter_type));
    config.enable_health_check =
        Environ::GetBool("MOONCAKE_DISTRIBUTED_HEALTH_CHECK", false);
    config.shard_count =
        Environ::GetInt("MOONCAKE_DFS_SHARD_COUNT", config.shard_count);
    config.shard_capacity = Environ::GetUInt64("MOONCAKE_DFS_SHARD_CAPACITY",
                                               config.shard_capacity);
    config.alignment =
        Environ::GetUInt64("MOONCAKE_DFS_ALIGNMENT", config.alignment);
    config.single_tenant =
        Environ::GetBool("MOONCAKE_DFS_SINGLE_TENANT", config.single_tenant);
    config.eviction_enabled = Environ::GetBool("MOONCAKE_DFS_EVICTION_ENABLED",
                                               config.eviction_enabled);
    config.eviction_high_watermark = Environ::GetDouble(
        "MOONCAKE_DFS_EVICTION_HIGH_WATERMARK", config.eviction_high_watermark);
    config.eviction_low_watermark = Environ::GetDouble(
        "MOONCAKE_DFS_EVICTION_LOW_WATERMARK", config.eviction_low_watermark);
    config.deferred_free_duration = std::chrono::seconds(Environ::GetInt(
        "MOONCAKE_DFS_DEFERRED_FREE_SECONDS",
        static_cast<int>(config.deferred_free_duration.count())));
    config.eviction_check_interval = std::chrono::seconds(Environ::GetInt(
        "MOONCAKE_DFS_EVICTION_CHECK_INTERVAL",
        static_cast<int>(config.eviction_check_interval.count())));

    // An unknown allocator type name is recorded rather than corrected, so the
    // master surfaces a configuration error instead of silently using SHARD.
    const std::string allocator_type_name = Environ::GetString(
        "MOONCAKE_DFS_ALLOCATOR_TYPE", ToString(config.allocator_type));
    if (auto parsed = ParseDfsAllocatorType(allocator_type_name)) {
        config.allocator_type = *parsed;
    } else {
        LOG(ERROR) << "Unknown MOONCAKE_DFS_ALLOCATOR_TYPE '"
                   << allocator_type_name << "', expected 'shard' or 'bucket'";
        config.allocator_type_valid = false;
    }
    config.bucket_capacity = Environ::GetUInt64("MOONCAKE_DFS_BUCKET_CAPACITY",
                                                config.bucket_capacity);
    config.max_bucket_count = static_cast<int64_t>(
        Environ::GetInt("MOONCAKE_DFS_MAX_BUCKET_COUNT",
                        static_cast<int>(config.max_bucket_count)));
    config.batch_read_threads = Environ::GetInt(
        "MOONCAKE_DFS_BATCH_READ_THREADS", config.batch_read_threads);
    config.batch_read_merge_enabled =
        Environ::GetBool("MOONCAKE_DFS_BATCH_READ_MERGE_ENABLED",
                         config.batch_read_merge_enabled);
    config.direct_read_enabled = Environ::GetBool(
        "MOONCAKE_DFS_DIRECT_READ_ENABLED", config.direct_read_enabled);
    LOG(INFO) << config.FormatStr();
    return config;
}

std::string DistributedStorageConfig::FormatStr() const {
    std::ostringstream oss;
    oss << "fsdir=" << fsdir << ", fs_adapter_type=" << fs_adapter_type
        << ", enable_health_check=" << enable_health_check
        << ", shard_count=" << shard_count
        << ", shard_capacity=" << shard_capacity << ", alignment=" << alignment
        << ", single_tenant=" << single_tenant
        << ", eviction_enabled=" << eviction_enabled
        << ", eviction_high_watermark=" << eviction_high_watermark
        << ", eviction_low_watermark=" << eviction_low_watermark
        << ", deferred_free_seconds=" << deferred_free_duration.count()
        << ", eviction_check_interval_seconds="
        << eviction_check_interval.count()
        << ", allocator_type=" << ToString(allocator_type)
        << ", bucket_capacity=" << bucket_capacity
        << ", max_bucket_count=" << max_bucket_count
        << ", batch_read_threads=" << batch_read_threads
        << ", batch_read_merge_enabled=" << batch_read_merge_enabled
        << ", direct_read_enabled=" << direct_read_enabled;
    return oss.str();
}

DistributedStorageBackend::DistributedStorageBackend(
    const FileStorageConfig& file_storage_config,
    const DistributedStorageConfig& distributed_config,
    std::unique_ptr<FileSystemAdapter> fs_adapter)
    : StorageBackendInterface(file_storage_config),
      fs_adapter_(std::move(fs_adapter)),
      distributed_config_(distributed_config),
      root_dir_(distributed_config.fsdir) {}

DistributedStorageBackend::~DistributedStorageBackend() {
    for (auto& shard : shard_files_) {
        if (shard && fs_adapter_) {
            if (shard->direct_fd >= 0) {
                fs_adapter_->CloseFile(shard->direct_fd);
                shard->direct_fd = -1;
            }
            if (shard->fd >= 0) {
                fs_adapter_->CloseFile(shard->fd);
                shard->fd = -1;
            }
        }
    }
    // Drop cached bucket handles before the adapter goes away; each handle
    // closes its own fd in its destructor via the adapter pointer it holds.
    {
        std::lock_guard<std::mutex> lock(bucket_cache_mutex_);
        bucket_cache_.clear();
        bucket_direct_cache_.clear();
    }
    batch_read_pool_.reset();
    if (fs_adapter_) fs_adapter_->Shutdown();
}

tl::expected<void, ErrorCode> DistributedStorageBackend::Init() {
    if (initialized_) {
        LOG(WARNING) << "DistributedStorageBackend is already initialized";
        return {};
    }
    std::error_code ec;
    std::filesystem::create_directories(root_dir_, ec);
    if (ec) {
        LOG(ERROR) << "Failed to create DFS root directory " << root_dir_
                   << ": " << ec.message();
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }

    auto init_result = fs_adapter_->Init(root_dir_);
    if (!init_result) return init_result;

    canonical_root_dir_ = CanonicalizePath(root_dir_);
    if (canonical_root_dir_.empty()) {
        LOG(ERROR) << "Failed to canonicalize DFS root directory " << root_dir_;
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    if (IsBucketMode()) {
        // Bucket data files are created by the master's allocator and opened
        // on demand here, so there is no fixed shard table to preopen.
        if (!distributed_config_.ValidateForBucketAllocator()) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        if (distributed_config_.batch_read_threads > 1) {
            batch_read_pool_ = std::make_unique<ThreadPool>(
                static_cast<size_t>(distributed_config_.batch_read_threads));
        }
        initialized_ = true;
        LOG(INFO) << "DistributedStorageBackend initialized in bucket mode, "
                     "fsdir="
                  << root_dir_ << ", bucket_capacity="
                  << distributed_config_.bucket_capacity;
        return {};
    }

    // SHARD mode has no bucket-mode config validation, so clamp the pool size
    // here instead of trusting the environment.
    if (distributed_config_.batch_read_threads > 1) {
        const int threads = std::min(distributed_config_.batch_read_threads,
                                     kMaxBatchReadThreads);
        batch_read_pool_ =
            std::make_unique<ThreadPool>(static_cast<size_t>(threads));
    }

    shard_files_.reserve(distributed_config_.shard_count);
    for (int i = 0; i < distributed_config_.shard_count; ++i) {
        std::string path = root_dir_ + "/dfs_shard_" +
                           DfsGlobalAllocator::FormatShardIdx(
                               i, distributed_config_.shard_count) +
                           ".data";
        auto fd_result = fs_adapter_->OpenFile(path);
        if (!fd_result) {
            LOG(ERROR) << "Failed to open DFS shard " << path << ": "
                       << fd_result.error();
            return tl::make_unexpected(fd_result.error());
        }
        auto shard = std::make_unique<ShardFile>();
        shard->path = std::move(path);
        shard->fd = *fd_result;
        if (distributed_config_.direct_read_enabled) {
            auto direct_result = fs_adapter_->OpenFileDirect(shard->path);
            if (direct_result) {
                shard->direct_fd = *direct_result;
            } else if (direct_result.error() != ErrorCode::NOT_SUPPORTED) {
                LOG(WARNING) << "Failed to open direct read handle for DFS "
                                "shard "
                             << shard->path << ": " << direct_result.error();
            }
        }
        shard_files_.push_back(std::move(shard));
    }

    initialized_ = true;
    return {};
}

tl::expected<std::shared_ptr<DistributedStorageBackend::OpenFileHandle>,
             ErrorCode>
DistributedStorageBackend::GetOrOpenBucket(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(bucket_cache_mutex_);
        auto it = bucket_cache_.find(path);
        if (it != bucket_cache_.end()) return it->second;
    }

    auto fd_result = fs_adapter_->OpenFile(path);
    if (!fd_result) {
        LOG(ERROR) << "Failed to open DFS bucket file " << path << ": "
                   << fd_result.error();
        return tl::make_unexpected(fd_result.error());
    }

    auto handle = std::make_shared<OpenFileHandle>();
    handle->path = path;
    handle->fd = *fd_result;
    handle->adapter = fs_adapter_.get();

    std::lock_guard<std::mutex> lock(bucket_cache_mutex_);
    // Another thread may have populated the cache while we were opening; keep
    // the winner and let our handle close its own fd on destruction.
    auto [it, inserted] = bucket_cache_.emplace(path, handle);
    (void)inserted;
    return it->second;
}

tl::expected<std::shared_ptr<DistributedStorageBackend::OpenFileHandle>,
             ErrorCode>
DistributedStorageBackend::GetOrOpenBucketDirect(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(bucket_cache_mutex_);
        auto it = bucket_direct_cache_.find(path);
        if (it != bucket_direct_cache_.end()) return it->second;
    }

    auto fd_result = fs_adapter_->OpenFileDirect(path);
    if (!fd_result) {
        if (fd_result.error() != ErrorCode::NOT_SUPPORTED) {
            LOG(WARNING) << "Failed to open direct read handle for DFS bucket "
                            "file "
                         << path << ": " << fd_result.error();
        }
        return tl::make_unexpected(fd_result.error());
    }

    auto handle = std::make_shared<OpenFileHandle>();
    handle->path = path;
    handle->fd = *fd_result;
    handle->adapter = fs_adapter_.get();

    std::lock_guard<std::mutex> lock(bucket_cache_mutex_);
    // Another thread may have populated the cache while we were opening; keep
    // the winner and let our handle close its own fd on destruction.
    auto [it, inserted] = bucket_direct_cache_.emplace(path, handle);
    (void)inserted;
    return it->second;
}

tl::expected<DistributedStorageBackend::ResolvedTarget, ErrorCode>
DistributedStorageBackend::ResolveTarget(
    const DistributedFSDescriptor& descriptor, const std::string& key,
    std::string* resolved_path) {
    if (!IsBucketMode()) {
        if (descriptor.shard_idx < 0 ||
            descriptor.shard_idx >= static_cast<int>(shard_files_.size())) {
            LOG(ERROR) << "Invalid DFS shard_idx " << descriptor.shard_idx
                       << " for key " << key;
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        auto& shard = *shard_files_[descriptor.shard_idx];
        if (descriptor.file_path != shard.path) {
            LOG(ERROR) << "DFS path mismatch for key " << key
                       << ", descriptor=" << descriptor.file_path
                       << ", configured=" << shard.path;
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        if (!IsDfsDescriptorRangeValid(descriptor, distributed_config_)) {
            LOG(ERROR) << "Invalid DFS descriptor range for key " << key
                       << ", offset=" << descriptor.offset
                       << ", object_size=" << descriptor.object_size
                       << ", aligned_size=" << descriptor.aligned_size
                       << ", shard_capacity="
                       << distributed_config_.shard_capacity;
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        return ResolvedTarget{shard.fd, &shard.mutex, nullptr};
    }

    // BUCKET mode: the descriptor carries an allocator-chosen path, so it must
    // be validated before it is used to open anything.
    if (descriptor.shard_idx < 0) {
        LOG(ERROR) << "Invalid DFS bucket id " << descriptor.shard_idx
                   << " for key " << key;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (!IsBucketDescriptorRangeValid(descriptor, key, distributed_config_)) {
        LOG(ERROR) << "Invalid DFS bucket descriptor for key " << key
                   << ", offset=" << descriptor.offset
                   << ", object_size=" << descriptor.object_size
                   << ", aligned_size=" << descriptor.aligned_size
                   << ", bucket_capacity="
                   << distributed_config_.bucket_capacity;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    const std::string canonical = CanonicalizePath(descriptor.file_path);
    if (canonical.empty() ||
        !IsPathWithinRoot(canonical, canonical_root_dir_)) {
        LOG(ERROR) << "DFS bucket path " << descriptor.file_path << " for key "
                   << key << " resolves outside the configured DFS root "
                   << canonical_root_dir_;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (!MatchesBucketDataFileName(canonical, descriptor.shard_idx)) {
        LOG(ERROR) << "DFS bucket path " << descriptor.file_path << " for key "
                   << key << " does not name bucket " << descriptor.shard_idx;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    if (resolved_path != nullptr) *resolved_path = canonical;
    auto handle = GetOrOpenBucket(canonical);
    if (!handle) return tl::make_unexpected(handle.error());
    auto& shared = handle.value();
    return ResolvedTarget{shared->fd, &shared->mutex, shared};
}

tl::expected<int64_t, ErrorCode> DistributedStorageBackend::BatchOffload(
    const std::unordered_map<std::string, std::vector<Slice>>& /*batch_object*/,
    std::function<ErrorCode(const std::vector<std::string>& keys,
                            std::vector<StorageObjectMetadata>& metadatas)>
    /*complete_handler*/,
    EvictionHandler /*eviction_handler*/) {
    return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
}

std::vector<tl::expected<void, ErrorCode>>
DistributedStorageBackend::BatchWrite(
    const std::vector<DfsWriteRequest>& requests) {
    std::vector<tl::expected<void, ErrorCode>> results(
        requests.size(), tl::make_unexpected(ErrorCode::INVALID_PARAMS));
    if (!initialized_) {
        LOG(ERROR) << "DistributedStorageBackend is not initialized";
        std::fill(results.begin(), results.end(),
                  tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE));
        return results;
    }

    // BUCKET entries are physically contiguous when BatchAllocate succeeded.
    // Build validated entry payloads first, including the reserved zero padding,
    // then issue one write per contiguous bounded-size run. SHARD keeps the
    // original one-request path because its allocator does not promise
    // contiguity.
    if (!IsBucketMode()) {
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto& request = requests[i];
            auto target = ResolveTarget(request.descriptor, request.key);
            if (!target) { results[i] = tl::make_unexpected(target.error()); continue; }
            std::vector<iovec> iovs;
            uint64_t total = 0, value_size = 0;
            if (request.descriptor.object_size == 0) continue;
            for (const auto& slice : request.slices) {
                if ((!slice.ptr && slice.size > 0) ||
                    slice.size > std::numeric_limits<uint64_t>::max() - total) {
                    total = 0; break;
                }
                if (slice.size) iovs.push_back({slice.ptr, slice.size});
                total += slice.size; value_size += slice.size;
            }
            if (total == 0 || value_size != request.descriptor.object_size) continue;
            std::lock_guard<std::mutex> lock(*target->mutex);
            uint64_t done = 0;
            size_t index = 0;
            uint64_t consumed = 0;
            ErrorCode error = ErrorCode::OK;
            while (done < total && index < iovs.size()) {
                std::vector<iovec> pending;
                pending.push_back({static_cast<char*>(iovs[index].iov_base) + consumed,
                                   iovs[index].iov_len - consumed});
                for (size_t j = index + 1; j < iovs.size(); ++j) pending.push_back(iovs[j]);
                auto r = fs_adapter_->WriteAt(target->fd, pending.data(),
                                              static_cast<int>(pending.size()),
                                              static_cast<int64_t>(request.descriptor.offset + done));
                if (!r) {
                    error = r.error();
                    break;
                }
                if (*r == 0) {
                    error = ErrorCode::FILE_WRITE_FAIL;
                    break;
                }
                uint64_t advanced = *r; done += advanced;
                while (advanced && index < iovs.size()) {
                    auto available = iovs[index].iov_len - consumed;
                    auto step = std::min<uint64_t>(advanced, available);
                    consumed += step; advanced -= step;
                    if (consumed == iovs[index].iov_len) { ++index; consumed = 0; }
                }
            }
            if (error == ErrorCode::OK && done == total) results[i] = {};
            else results[i] = tl::make_unexpected(
                error == ErrorCode::OK ? ErrorCode::FILE_WRITE_FAIL : error);
        }
        return results;
    }

    struct Prepared {
        size_t index;
        ResolvedTarget target;
        uint64_t offset;
        std::vector<char> payload;
    };
    std::vector<Prepared> prepared;
    prepared.reserve(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        auto target = ResolveTarget(request.descriptor, request.key);
        if (!target) { results[i] = tl::make_unexpected(target.error()); continue; }
        const uint64_t header_size = BucketEntryLayout::kHeaderSize + request.key.size();
        if (request.descriptor.offset < header_size) continue;
        const uint64_t entry_size = header_size + request.descriptor.object_size;
        if (request.descriptor.aligned_size < entry_size) continue;
        uint64_t value_size = 0;
        std::vector<char> payload(static_cast<size_t>(request.descriptor.aligned_size), 0);
        for (size_t j = 0; j < BucketEntryLayout::kHeaderSize; ++j)
            payload[j] = static_cast<char>((request.key.size() >> (8 * j)) & 0xff);
        std::memcpy(payload.data() + BucketEntryLayout::kHeaderSize,
                    request.key.data(), request.key.size());
        size_t payload_offset = static_cast<size_t>(header_size);
        bool invalid = false;
        for (const auto& slice : request.slices) {
            if ((!slice.ptr && slice.size > 0) ||
                slice.size > request.descriptor.object_size - value_size) {
                invalid = true; break;
            }
            if (slice.size) {
                std::memcpy(payload.data() + payload_offset, slice.ptr, slice.size);
                payload_offset += slice.size; value_size += slice.size;
            }
        }
        if (invalid || value_size != request.descriptor.object_size) continue;
        prepared.push_back({i, std::move(*target),
                            request.descriptor.offset - header_size,
                            std::move(payload)});
    }

    auto write_run = [&](size_t begin, size_t end) {
        size_t pos = begin;
        while (pos < end) {
            size_t stop = pos;
            uint64_t total_size = 0;
            constexpr size_t kMaxIov = 1024;
            while (stop < end && stop - pos < kMaxIov &&
                   (stop == pos || total_size + prepared[stop].payload.size() <=
                                       kMaxMergedIo)) {
                total_size += prepared[stop].payload.size();
                ++stop;
            }

            std::vector<iovec> iovs;
            iovs.reserve(stop - pos);
            for (size_t j = pos; j < stop; ++j) {
                iovs.push_back(
                    {prepared[j].payload.data(), prepared[j].payload.size()});
            }

            std::lock_guard<std::mutex> lock(*prepared[pos].target.mutex);
            uint64_t written = 0;
            size_t iov_index = 0;
            uint64_t iov_consumed = 0;
            ErrorCode error = ErrorCode::OK;
            while (written < total_size) {
                std::vector<iovec> pending;
                pending.reserve(iovs.size() - iov_index);
                pending.push_back(
                    {static_cast<char*>(iovs[iov_index].iov_base) + iov_consumed,
                     iovs[iov_index].iov_len - iov_consumed});
                for (size_t j = iov_index + 1; j < iovs.size(); ++j) {
                    pending.push_back(iovs[j]);
                }
                auto result = fs_adapter_->WriteAt(
                    prepared[pos].target.fd, pending.data(),
                    static_cast<int>(pending.size()),
                    static_cast<int64_t>(prepared[pos].offset + written));
                if (!result) {
                    error = result.error();
                    break;
                }
                if (*result == 0) {
                    error = ErrorCode::FILE_WRITE_FAIL;
                    break;
                }
                uint64_t advanced = *result;
                written += advanced;
                while (advanced != 0 && iov_index < iovs.size()) {
                    const uint64_t available =
                        iovs[iov_index].iov_len - iov_consumed;
                    const uint64_t step =
                        std::min<uint64_t>(advanced, available);
                    iov_consumed += step;
                    advanced -= step;
                    if (iov_consumed == iovs[iov_index].iov_len) {
                        ++iov_index;
                        iov_consumed = 0;
                    }
                }
            }
            if (error == ErrorCode::OK && written == total_size) {
                for (size_t j = pos; j < stop; ++j) {
                    results[prepared[j].index] = {};
                }
            } else {
                if (error == ErrorCode::OK) error = ErrorCode::FILE_WRITE_FAIL;
                for (size_t j = pos; j < stop; ++j) {
                    results[prepared[j].index] = tl::make_unexpected(error);
                }
            }
            pos = stop;
        }
    };
    size_t run = 0;
    for (size_t i = 1; i <= prepared.size(); ++i) {
        bool contiguous = i < prepared.size() &&
            prepared[i-1].target.fd == prepared[i].target.fd &&
            prepared[i-1].target.mutex == prepared[i].target.mutex &&
            prepared[i].offset == prepared[i-1].offset + prepared[i-1].payload.size();
        if (!contiguous) { if (run < i) write_run(run, i); run = i; }
    }
    return results;
}

ErrorCode DistributedStorageBackend::ReadFully(FileSystemAdapter* fs_adapter,
                                               const ResolvedTarget& target,
                                               uint64_t offset,
                                               std::span<char> output) {
    const auto start = std::chrono::steady_clock::now();
    uint64_t done = 0;
    ErrorCode error = ErrorCode::OK;
    while (done < output.size()) {
        iovec iov{output.data() + done, output.size() - done};
        auto read = fs_adapter->ReadAt(target.fd, &iov, 1,
                                       static_cast<int64_t>(offset + done));
        if (!read) {
            error = read.error();
            break;
        }
        if (*read == 0) {
            error = ErrorCode::FILE_READ_FAIL;
            break;
        }
        done += *read;
    }
    const int64_t duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    if (dfs_read_trace_enabled()) {
        LOG(INFO) << "ReadFully: bytes=" << done << "/" << output.size()
                  << " duration_us=" << duration_us
                  << " error=" << static_cast<int>(error);
    }
    return error;
}

void DistributedStorageBackend::CopyToSlices(const DfsReadRequest& request,
                                             const char* value) {
    uint64_t remaining = request.descriptor.object_size;
    for (const auto& slice : request.slices) {
        const size_t size =
            static_cast<size_t>(std::min<uint64_t>(slice.size, remaining));
        if (size != 0) {
            std::memcpy(slice.ptr, value, size);
            value += size;
            remaining -= size;
        }
        if (remaining == 0) break;
    }
}

std::unordered_map<std::mutex*, DistributedStorageBackend::BucketReadGroup>
DistributedStorageBackend::GroupReadsByBucket(
    std::vector<PreparedRead>&& prepared) {
    std::unordered_map<std::mutex*, BucketReadGroup> groups;
    for (auto& pr : prepared) {
        auto& group = groups[pr.target.mutex];
        if (group.mutex == nullptr) group.mutex = pr.target.mutex;
        group.reads.push_back(std::move(pr));
    }
    return groups;
}

void DistributedStorageBackend::SortGroupByOffset(BucketReadGroup& group) {
    std::sort(group.reads.begin(), group.reads.end(),
              [](const PreparedRead& a, const PreparedRead& b) {
                  return a.entry_start < b.entry_start;
              });
}

/**
 * @brief Collapse runs of contiguous entries in a sorted group into one read.
 */
std::vector<DistributedStorageBackend::MergedIo>
DistributedStorageBackend::BuildMergedIos(const BucketReadGroup& group) {
    std::vector<MergedIo> merged;
    size_t run_begin = 0;

    while (run_begin < group.reads.size()) {
        size_t run_end = run_begin + 1;
        while (run_end < group.reads.size()) {
            const auto& previous = group.reads[run_end - 1];
            const auto& current = group.reads[run_end];
            const bool contiguous =
                current.entry_start ==
                previous.entry_start + previous.reserved_size;
            const uint64_t merged_size = current.entry_start +
                                         current.reserved_size -
                                         group.reads[run_begin].entry_start;
            if (!contiguous || merged_size > kMaxMergedIo) break;
            ++run_end;
        }

        MergedIo io;
        io.entry_start = group.reads[run_begin].entry_start;
        io.total_size = group.reads[run_end - 1].entry_start +
                        group.reads[run_end - 1].reserved_size - io.entry_start;
        io.reads.reserve(run_end - run_begin);
        for (size_t j = run_begin; j < run_end; ++j) {
            io.reads.push_back(&group.reads[j]);
        }
        merged.push_back(std::move(io));
        run_begin = run_end;
    }
    return merged;
}

void DistributedStorageBackend::ExecuteMergedRead(
    const MergedIo& io, const std::vector<DfsReadRequest>& requests,
    std::vector<tl::expected<void, ErrorCode>>& results,
    const ResolvedTarget& target, std::mutex* mutex,
    FileSystemAdapter* fs_adapter, std::vector<char>& staging) {
    staging.resize(static_cast<size_t>(io.total_size));
    ErrorCode error;
    {
        std::lock_guard<std::mutex> lock(*mutex);
        error = ReadFully(fs_adapter, target, io.entry_start, staging);
    }
    if (error != ErrorCode::OK) {
        for (const auto* pr : io.reads) {
            results[pr->request_index] = tl::make_unexpected(error);
        }
        return;
    }
    for (const auto* pr : io.reads) {
        const auto& request = requests[pr->request_index];
        const uint64_t value_offset = pr->entry_start - io.entry_start +
                                      BucketEntryLayout::kHeaderSize +
                                      request.key.size();
        CopyToSlices(request, staging.data() + value_offset);
        results[pr->request_index] = {};
    }
}

void DistributedStorageBackend::FailGroupReads(
    const BucketReadGroup& group,
    std::vector<tl::expected<void, ErrorCode>>& results, ErrorCode error) {
    for (const auto& pr : group.reads) {
        results[pr.request_index] = tl::make_unexpected(error);
    }
}

void DistributedStorageBackend::ProcessBucketGroup(
    BucketReadGroup& group, const std::vector<DfsReadRequest>& requests,
    std::vector<tl::expected<void, ErrorCode>>& results,
    FileSystemAdapter* fs_adapter, MergeStats& stats) {
    const auto& target = group.reads.front().target;
    std::vector<char> staging;
    const auto merged = BuildMergedIos(group);
    uint64_t aggregated_reads = 0;
    uint64_t merged_ios = 0;
    uint64_t merged_bytes = 0;
    for (const auto& io : merged) {
        if (io.reads.size() > 1) {
            aggregated_reads += io.reads.size();
            ++merged_ios;
            merged_bytes += io.total_size;
        }
    }
    stats.aggregated_reads.fetch_add(aggregated_reads,
                                     std::memory_order_relaxed);
    stats.merged_ios.fetch_add(merged_ios, std::memory_order_relaxed);
    stats.merged_bytes.fetch_add(merged_bytes, std::memory_order_relaxed);
    for (const auto& io : merged) {
        ExecuteMergedRead(io, requests, results, target, group.mutex,
                          fs_adapter, staging);
    }
}

/**
 * @brief Run one bucket group per pool task and block until all have finished.
 *
 * Every task must decrement the pending counter exactly once, including on
 * failure: a task that escaped without doing so would both hang this call and
 * terminate the worker thread, since the pool does not catch exceptions.
 */
void DistributedStorageBackend::DispatchParallelReads(
    std::unordered_map<std::mutex*, BucketReadGroup>& groups,
    const std::vector<DfsReadRequest>& requests,
    std::vector<tl::expected<void, ErrorCode>>& results, ThreadPool& pool,
    FileSystemAdapter* fs_adapter, MergeStats& stats) {
    std::vector<BucketReadGroup*> group_ptrs;
    group_ptrs.reserve(groups.size());
    for (auto& [mutex, group] : groups) {
        group_ptrs.push_back(&group);
    }

    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    size_t pending_groups = 0;

    auto mark_done = [&completion_mutex, &completion_cv, &pending_groups]() {
        // Notify while still holding the lock: otherwise a spurious wakeup
        // could let the waiter observe 0, return, and destroy completion_cv
        // out from under this notify_one().
        std::lock_guard<std::mutex> lock(completion_mutex);
        --pending_groups;
        completion_cv.notify_one();
    };

    for (auto* g : group_ptrs) {
        {
            std::lock_guard<std::mutex> lock(completion_mutex);
            ++pending_groups;
        }
        try {
            pool.enqueue([g, &requests, &results, fs_adapter, &stats,
                          &mark_done]() {
                try {
                    ProcessBucketGroup(*g, requests, results, fs_adapter,
                                       stats);
                } catch (const std::exception& e) {
                    LOG(ERROR) << "Bucket batch read task failed: " << e.what();
                    FailGroupReads(*g, results, ErrorCode::FILE_READ_FAIL);
                } catch (...) {
                    LOG(ERROR) << "Bucket batch read task failed";
                    FailGroupReads(*g, results, ErrorCode::FILE_READ_FAIL);
                }
                mark_done();
            });
        } catch (const std::exception& e) {
            // The pool rejected the task, so nothing will decrement for it.
            LOG(ERROR) << "Failed to enqueue bucket batch read: " << e.what();
            FailGroupReads(*g, results, ErrorCode::FILE_READ_FAIL);
            mark_done();
        }
    }

    std::unique_lock<std::mutex> lock(completion_mutex);
    completion_cv.wait(lock, [&pending_groups] { return pending_groups == 0; });
}

void DistributedStorageBackend::ExecuteKeyRead(
    const PreparedKeyRead& read, const std::vector<DfsReadRequest>& requests,
    std::vector<tl::expected<void, ErrorCode>>& results) {
    const auto& request = requests[read.request_index];
    const uint64_t object_size = request.descriptor.object_size;

    // Scatter the value straight into the caller's slices (clamped to
    // object_size) so there is no staging buffer or extra copy on this path.
    std::vector<iovec> iovs;
    iovs.reserve(request.slices.size());
    uint64_t remaining = object_size;
    for (const auto& slice : request.slices) {
        if (remaining == 0) break;
        const uint64_t size = std::min<uint64_t>(slice.size, remaining);
        if (size != 0) {
            iovs.push_back({slice.ptr, static_cast<size_t>(size)});
            remaining -= size;
        }
    }

    // preadv-style interfaces reject iov lists longer than IOV_MAX, so issue
    // the read in bounded chunks. Short reads resume where they stopped.
    constexpr size_t kMaxIovChunk = 1024;
    ErrorCode error = ErrorCode::OK;
    uint64_t done = 0;
    size_t index = 0;
    size_t consumed = 0;

    // Direct handles are read-only with offset-explicit I/O and arrive here
    // with a null mutex; the regular fallback handle is shared with writers,
    // so reads on it keep the legacy serialization.
    std::unique_lock<std::mutex> target_lock;
    if (read.target.mutex != nullptr) {
        target_lock = std::unique_lock<std::mutex>(*read.target.mutex);
    }

    while (done < object_size && index < iovs.size()) {
        std::vector<iovec> pending;
        pending.push_back(
            {static_cast<char*>(iovs[index].iov_base) + consumed,
             iovs[index].iov_len - consumed});
        for (size_t j = index + 1;
             j < iovs.size() && pending.size() < kMaxIovChunk; ++j) {
            pending.push_back(iovs[j]);
        }
        auto read_result = fs_adapter_->DirectReadAt(
            read.target.fd, pending.data(), static_cast<int>(pending.size()),
            static_cast<int64_t>(read.value_offset + done));
        if (!read_result) {
            error = read_result.error();
            break;
        }
        if (*read_result == 0) {
            error = ErrorCode::FILE_READ_FAIL;
            break;
        }
        uint64_t advanced = *read_result;
        done += advanced;
        while (advanced != 0 && index < iovs.size()) {
            const size_t available = iovs[index].iov_len - consumed;
            const size_t step = std::min<uint64_t>(advanced, available);
            consumed += step;
            advanced -= step;
            if (consumed == iovs[index].iov_len) {
                ++index;
                consumed = 0;
            }
        }
    }
    if (error == ErrorCode::OK && done != object_size) {
        error = ErrorCode::FILE_READ_FAIL;
    }

    if (error == ErrorCode::OK) {
        results[read.request_index] = {};
    } else {
        results[read.request_index] = tl::make_unexpected(error);
    }
}

/**
 * @brief Per-key read flow: no bucketing or merging, each key is read on its
 * own and the reads are fanned out over the batch read pool.
 */
std::vector<tl::expected<void, ErrorCode>>
DistributedStorageBackend::BatchReadDirect(
    const std::vector<DfsReadRequest>& requests) {
    const auto timing_start = std::chrono::steady_clock::now();
    std::vector<tl::expected<void, ErrorCode>> results(
        requests.size(), tl::make_unexpected(ErrorCode::INVALID_PARAMS));

    std::vector<PreparedKeyRead> prepared;
    prepared.reserve(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        std::string resolved_path;
        auto target =
            ResolveTarget(request.descriptor, request.key, &resolved_path);
        if (!target) {
            results[i] = tl::make_unexpected(target.error());
            continue;
        }

        uint64_t capacity = 0;
        bool invalid = false;
        for (const auto& slice : request.slices) {
            if ((!slice.ptr && slice.size != 0) ||
                slice.size > std::numeric_limits<uint64_t>::max() - capacity) {
                invalid = true;
                break;
            }
            capacity += slice.size;
        }
        if (invalid || capacity < request.descriptor.object_size) continue;

        // Switch to the direct read handle when the adapter offers one. The
        // direct handle is read-only and every read carries an explicit
        // offset, so it needs no serialization against other reads and the
        // handle mutex is dropped on purpose.
        if (distributed_config_.direct_read_enabled) {
            ResolvedTarget direct;
            if (IsBucketMode()) {
                auto handle = GetOrOpenBucketDirect(resolved_path);
                if (handle) {
                    direct = ResolvedTarget{(*handle)->fd, nullptr, *handle};
                }
            } else {
                auto& shard = *shard_files_[request.descriptor.shard_idx];
                if (shard.direct_fd >= 0) {
                    direct = ResolvedTarget{shard.direct_fd, nullptr, nullptr};
                }
            }
            if (direct.fd >= 0) {
                target->fd = direct.fd;
                target->mutex = nullptr;
                target->keepalive = std::move(direct.keepalive);
            }
        }
        prepared.push_back(
            {i, std::move(*target), request.descriptor.offset});
    }

    if (batch_read_pool_ != nullptr && prepared.size() > 1) {
        std::mutex completion_mutex;
        std::condition_variable completion_cv;
        size_t pending_reads = 0;

        auto mark_done = [&completion_mutex, &completion_cv,
                          &pending_reads]() {
            // Notify while still holding the lock: otherwise a spurious
            // wakeup could let the waiter observe 0, return, and destroy
            // completion_cv out from under this notify_one().
            std::lock_guard<std::mutex> lock(completion_mutex);
            --pending_reads;
            completion_cv.notify_one();
        };

        for (const auto& read : prepared) {
            {
                std::lock_guard<std::mutex> lock(completion_mutex);
                ++pending_reads;
            }
            try {
                batch_read_pool_->enqueue(
                    [this, read, &requests, &results, &mark_done]() {
                        try {
                            ExecuteKeyRead(read, requests, results);
                        } catch (const std::exception& e) {
                            LOG(ERROR)
                                << "Direct batch read task failed: "
                                << e.what();
                            results[read.request_index] = tl::make_unexpected(
                                ErrorCode::FILE_READ_FAIL);
                        } catch (...) {
                            LOG(ERROR) << "Direct batch read task failed";
                            results[read.request_index] = tl::make_unexpected(
                                ErrorCode::FILE_READ_FAIL);
                        }
                        mark_done();
                    });
            } catch (const std::exception& e) {
                // The pool rejected the task, so nothing will decrement for
                // it.
                LOG(ERROR) << "Failed to enqueue direct batch read: "
                           << e.what();
                results[read.request_index] =
                    tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
                mark_done();
            }
        }

        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock,
                           [&pending_reads] { return pending_reads == 0; });
    } else {
        // No pool (or a single read): nothing to overlap with, run inline.
        for (const auto& read : prepared) {
            ExecuteKeyRead(read, requests, results);
        }
    }

    if (dfs_read_trace_enabled()) {
        const int64_t duration_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - timing_start)
                .count();
        uint64_t total_bytes = 0;
        for (const auto& request : requests) {
            total_bytes += request.descriptor.object_size;
        }
        double gbps = 0.0;
        if (duration_us > 0) {
            // bytes / us == MB/s; /1000 -> GB/s.
            gbps = static_cast<double>(total_bytes) / duration_us / 1000.0;
        }
        LOG(INFO) << "BatchReadDirect: requests=" << requests.size()
                  << " bytes=" << total_bytes
                  << " duration_us=" << duration_us
                  << " bandwidth_GBps=" << gbps;
    }
    return results;
}

std::vector<tl::expected<void, ErrorCode>> DistributedStorageBackend::BatchRead(
    const std::vector<DfsReadRequest>& requests) {
    const auto timing_start = std::chrono::steady_clock::now();
    auto elapsed_us = [](const auto& a, const auto& b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a)
            .count();
    };
    std::vector<tl::expected<void, ErrorCode>> results(
        requests.size(), tl::make_unexpected(ErrorCode::INVALID_PARAMS));
    if (!initialized_) {
        std::fill(results.begin(), results.end(),
                  tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE));
        return results;
    }

    // Default flow: read each key directly, in parallel, with no bucketing.
    // The merge flow below is kept and can be re-enabled through
    // MOONCAKE_DFS_BATCH_READ_MERGE_ENABLED.
    if (!distributed_config_.batch_read_merge_enabled) {
        return BatchReadDirect(requests);
    }

    std::vector<PreparedRead> prepared;
    prepared.reserve(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        auto target = ResolveTarget(request.descriptor, request.key);
        if (!target) {
            results[i] = tl::make_unexpected(target.error());
            continue;
        }

        uint64_t capacity = 0;
        bool invalid = false;
        for (const auto& slice : request.slices) {
            if ((!slice.ptr && slice.size != 0) ||
                slice.size > std::numeric_limits<uint64_t>::max() - capacity) {
                invalid = true;
                break;
            }
            capacity += slice.size;
        }
        if (invalid || capacity < request.descriptor.object_size) continue;

        if (!IsBucketMode()) {
            std::vector<char> value(request.descriptor.object_size);
            std::lock_guard<std::mutex> lock(*target->mutex);
            const ErrorCode error = ReadFully(
                fs_adapter_.get(), *target, request.descriptor.offset, value);
            if (error != ErrorCode::OK) {
                results[i] = tl::make_unexpected(error);
                continue;
            }
            CopyToSlices(request, value.data());
            results[i] = {};
            continue;
        }

        const uint64_t header_size =
            BucketEntryLayout::kHeaderSize + request.key.size();
        if (request.descriptor.offset < header_size) continue;
        prepared.push_back({i, std::move(*target),
                            request.descriptor.offset - header_size,
                            request.descriptor.aligned_size});
    }

    if (!IsBucketMode()) {
        if (dfs_read_trace_enabled()) {
            LOG(INFO) << "BatchRead: requests=" << requests.size()
                      << " duration_us="
                      << elapsed_us(timing_start,
                                    std::chrono::steady_clock::now());
        }
        return results;
    }

    // Bucket the reads by open file handle, then sort each bucket by entry
    // offset so runs of contiguous entries collapse into a single read
    // regardless of the order the caller passed them in.
    auto groups = GroupReadsByBucket(std::move(prepared));
    for (auto& [mutex, group] : groups) {
        SortGroupByOffset(group);
    }

    MergeStats stats;
    if (batch_read_pool_ != nullptr && groups.size() > 1) {
        DispatchParallelReads(groups, requests, results, *batch_read_pool_,
                              fs_adapter_.get(), stats);
    } else {
        // A single bucket has nothing to overlap with, so skip the pool.
        for (auto& [mutex, group] : groups) {
            ProcessBucketGroup(group, requests, results, fs_adapter_.get(),
                               stats);
        }
    }

    if (dfs_read_trace_enabled()) {
        const int64_t duration_us =
            elapsed_us(timing_start, std::chrono::steady_clock::now());
        uint64_t total_bytes = 0;
        for (const auto& request : requests) {
            total_bytes += request.descriptor.object_size;
        }
        double gbps = 0.0;
        if (duration_us > 0) {
            // bytes / us == MB/s; /1000 -> GB/s.
            gbps = static_cast<double>(total_bytes) / duration_us / 1000.0;
        }
        const uint64_t aggregated_reads = stats.aggregated_reads.load();
        const uint64_t merged_ios = stats.merged_ios.load();
        const uint64_t merged_bytes = stats.merged_bytes.load();
        double avg_merged_size = 0.0;
        if (merged_ios > 0) {
            avg_merged_size = static_cast<double>(merged_bytes) / merged_ios;
        }
        LOG(INFO) << "BatchRead: requests=" << requests.size()
                  << " buckets=" << groups.size()
                  << " bytes=" << total_bytes
                  << " duration_us=" << duration_us
                  << " bandwidth_GBps=" << gbps
                  << " merged_ios=" << merged_ios
                  << " saved_ios=" << aggregated_reads - merged_ios
                  << " avg_merged_size=" << avg_merged_size;
    }

    return results;
}

tl::expected<void, ErrorCode> DistributedStorageBackend::BatchLoad(
    std::unordered_map<std::string, Slice>& /*batched_slices*/) {
    return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
}

tl::expected<bool, ErrorCode> DistributedStorageBackend::IsExist(
    const std::string& /*key*/) {
    return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
}

tl::expected<bool, ErrorCode> DistributedStorageBackend::IsEnableOffloading() {
    return false;
}

tl::expected<void, ErrorCode> DistributedStorageBackend::ScanMeta(
    const std::function<ErrorCode(
        const std::vector<std::string>& keys,
        std::vector<StorageObjectMetadata>& metadatas)>& /*handler*/) {
    return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
}

}  // namespace mooncake
