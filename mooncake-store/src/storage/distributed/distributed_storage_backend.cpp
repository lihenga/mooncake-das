#include "storage/distributed/distributed_storage_backend.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>

#include "environ.h"
#include "storage/distributed/bucket_entry_layout.h"
#include "storage/distributed/bucket_global_allocator.h"
#include "storage/distributed/dfs_global_allocator.h"
#include "types.h"
#include "utils.h"

namespace mooncake {

namespace {

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
    // A log must be able to hold at least one maximum-size record, otherwise
    // compaction would trigger on every single append.
    if (bucket_meta_log_threshold != 0 &&
        bucket_meta_log_threshold < kMetaLogMaxRecordSize) {
        LOG(ERROR) << "DistributedStorageConfig: bucket_meta_log_threshold ("
                   << bucket_meta_log_threshold
                   << ") must be at least the maximum log record size ("
                   << kMetaLogMaxRecordSize << ")";
        return false;
    }
    return true;
}

uint64_t DistributedStorageConfig::ResolveBucketMetaLogThreshold() const {
    if (bucket_meta_log_threshold != 0) return bucket_meta_log_threshold;

    // A 64th of the bucket capacity keeps the log small relative to the data it
    // describes, and the 4 MiB ceiling bounds crash-recovery replay cost for
    // large buckets. The floor keeps the threshold above one maximum-size
    // record so a single append can never force a compaction.
    constexpr uint64_t kMaxDefaultThreshold = 4ULL * 1024 * 1024;
    const uint64_t derived = bucket_capacity / 64;
    return std::max<uint64_t>(kMetaLogMaxRecordSize,
                              std::min(derived, kMaxDefaultThreshold));
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
    config.bucket_meta_log_threshold =
        Environ::GetUInt64("MOONCAKE_DFS_BUCKET_META_LOG_THRESHOLD",
                           config.bucket_meta_log_threshold);
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
        << ", bucket_meta_log_threshold=" << ResolveBucketMetaLogThreshold();
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
        if (shard && shard->fd >= 0 && fs_adapter_) {
            fs_adapter_->CloseFile(shard->fd);
            shard->fd = -1;
        }
    }
    // Drop cached bucket handles before the adapter goes away; each handle
    // closes its own fd in its destructor via the adapter pointer it holds.
    {
        std::lock_guard<std::mutex> lock(bucket_cache_mutex_);
        bucket_cache_.clear();
    }
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
        initialized_ = true;
        LOG(INFO) << "DistributedStorageBackend initialized in bucket mode, "
                     "fsdir="
                  << root_dir_ << ", bucket_capacity="
                  << distributed_config_.bucket_capacity;
        return {};
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

tl::expected<DistributedStorageBackend::ResolvedTarget, ErrorCode>
DistributedStorageBackend::ResolveTarget(
    const DistributedFSDescriptor& descriptor, const std::string& key) {
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
    // Exactly one result per request, in request order, so callers can always
    // pair a request with its outcome.
    std::vector<tl::expected<void, ErrorCode>> results;
    results.reserve(requests.size());

    if (!initialized_) {
        LOG(ERROR) << "DistributedStorageBackend is not initialized";
        results.assign(requests.size(),
                       tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE));
        return results;
    }

    const bool bucket_mode = IsBucketMode();

    for (const auto& request : requests) {
        const auto& desc = request.descriptor;
        auto target = ResolveTarget(desc, request.key);
        if (!target) {
            results.emplace_back(tl::make_unexpected(target.error()));
            continue;
        }

        // In bucket mode the entry header and key are written together with
        // the value, so the on-disk record is self-describing and recovery can
        // cross-check it.
        std::string header;
        uint64_t write_offset = desc.offset;
        if (bucket_mode) {
            const uint64_t key_size = request.key.size();
            header.resize(BucketEntryLayout::kHeaderSize + request.key.size());
            for (size_t i = 0; i < BucketEntryLayout::kHeaderSize; ++i) {
                header[i] = static_cast<char>((key_size >> (8 * i)) & 0xFF);
            }
            std::memcpy(header.data() + BucketEntryLayout::kHeaderSize,
                        request.key.data(), request.key.size());
            write_offset = desc.offset - header.size();
        }

        std::vector<iovec> iovs;
        iovs.reserve(request.slices.size() + (bucket_mode ? 1 : 0));
        uint64_t total_size = 0;
        if (bucket_mode) {
            iovs.push_back({header.data(), header.size()});
            total_size += header.size();
        }

        uint64_t value_size = 0;
        bool invalid = false;
        for (const auto& slice : request.slices) {
            if ((!slice.ptr && slice.size > 0) ||
                slice.size >
                    std::numeric_limits<uint64_t>::max() - total_size) {
                invalid = true;
                break;
            }
            if (slice.size == 0) continue;
            total_size += slice.size;
            value_size += slice.size;
            iovs.push_back({slice.ptr, slice.size});
        }
        if (invalid || value_size != desc.object_size) {
            LOG(WARNING) << "Invalid DFS write request for key " << request.key
                         << ", expected=" << desc.object_size
                         << ", actual=" << value_size;
            results.emplace_back(
                tl::make_unexpected(ErrorCode::INVALID_PARAMS));
            continue;
        }
        if (iovs.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
            results.emplace_back(
                tl::make_unexpected(ErrorCode::INVALID_PARAMS));
            continue;
        }

        // Resume across partial writes rather than reporting a short write:
        // pwritev may legitimately stop early on large vectors.
        std::lock_guard<std::mutex> lock(*target->mutex);
        uint64_t written = 0;
        size_t iov_index = 0;
        uint64_t iov_consumed = 0;
        ErrorCode write_error = ErrorCode::OK;
        while (written < total_size && iov_index < iovs.size()) {
            std::vector<iovec> pending;
            pending.reserve(iovs.size() - iov_index);
            pending.push_back(
                {static_cast<char*>(iovs[iov_index].iov_base) + iov_consumed,
                 iovs[iov_index].iov_len - iov_consumed});
            for (size_t i = iov_index + 1; i < iovs.size(); ++i) {
                pending.push_back(iovs[i]);
            }

            auto write_result = fs_adapter_->WriteAt(
                target->fd, pending.data(), static_cast<int>(pending.size()),
                static_cast<int64_t>(write_offset + written));
            if (!write_result) {
                write_error = write_result.error();
                break;
            }
            if (*write_result == 0) {
                // No progress: treat as a hard failure instead of spinning.
                write_error = ErrorCode::FILE_WRITE_FAIL;
                break;
            }

            uint64_t advanced = *write_result;
            written += advanced;
            while (advanced > 0 && iov_index < iovs.size()) {
                const uint64_t available =
                    iovs[iov_index].iov_len - iov_consumed;
                const uint64_t step = std::min<uint64_t>(advanced, available);
                iov_consumed += step;
                advanced -= step;
                if (iov_consumed == iovs[iov_index].iov_len) {
                    ++iov_index;
                    iov_consumed = 0;
                }
            }
        }

        if (write_error != ErrorCode::OK) {
            LOG(WARNING) << "DFS write failed for key " << request.key
                         << ", error=" << write_error << ", written=" << written
                         << "/" << total_size;
            results.emplace_back(tl::make_unexpected(write_error));
            continue;
        }
        if (written != total_size) {
            LOG(WARNING) << "DFS short write for key " << request.key
                         << ", expected=" << total_size
                         << ", actual=" << written;
            results.emplace_back(
                tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL));
            continue;
        }
        results.emplace_back();
    }
    return results;
}

std::vector<tl::expected<void, ErrorCode>> DistributedStorageBackend::BatchRead(
    const std::vector<DfsReadRequest>& requests) {
    std::vector<tl::expected<void, ErrorCode>> results;
    results.reserve(requests.size());

    if (!initialized_) {
        LOG(ERROR) << "DistributedStorageBackend is not initialized";
        results.assign(requests.size(),
                       tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE));
        return results;
    }

    for (const auto& request : requests) {
        const auto& desc = request.descriptor;
        auto target = ResolveTarget(desc, request.key);
        if (!target) {
            results.emplace_back(tl::make_unexpected(target.error()));
            continue;
        }
        if (desc.object_size > std::numeric_limits<size_t>::max() ||
            request.slices.size() >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
            results.emplace_back(
                tl::make_unexpected(ErrorCode::INVALID_PARAMS));
            continue;
        }

        // Read exactly object_size bytes starting at the value offset; the
        // entry header and key are never handed back to the caller.
        std::vector<iovec> iovs;
        iovs.reserve(request.slices.size());
        size_t remaining = static_cast<size_t>(desc.object_size);
        bool invalid = false;
        for (const auto& slice : request.slices) {
            if (!slice.ptr && slice.size > 0) {
                invalid = true;
                break;
            }
            if (remaining == 0 || slice.size == 0) {
                continue;
            }
            const size_t read_size = std::min(slice.size, remaining);
            iovs.push_back({slice.ptr, read_size});
            remaining -= read_size;
        }
        if (invalid || remaining != 0) {
            LOG(WARNING) << "Invalid DFS read request for key " << request.key
                         << ", expected capacity at least=" << desc.object_size;
            results.emplace_back(
                tl::make_unexpected(ErrorCode::INVALID_PARAMS));
            continue;
        }

        std::lock_guard<std::mutex> lock(*target->mutex);
        uint64_t read_total = 0;
        size_t iov_index = 0;
        uint64_t iov_consumed = 0;
        ErrorCode read_error = ErrorCode::OK;
        while (read_total < desc.object_size && iov_index < iovs.size()) {
            std::vector<iovec> pending;
            pending.reserve(iovs.size() - iov_index);
            pending.push_back(
                {static_cast<char*>(iovs[iov_index].iov_base) + iov_consumed,
                 iovs[iov_index].iov_len - iov_consumed});
            for (size_t i = iov_index + 1; i < iovs.size(); ++i) {
                pending.push_back(iovs[i]);
            }

            auto read_result = fs_adapter_->ReadAt(
                target->fd, pending.data(), static_cast<int>(pending.size()),
                static_cast<int64_t>(desc.offset + read_total));
            if (!read_result) {
                read_error = read_result.error();
                break;
            }
            if (*read_result == 0) {
                // Hit EOF before the value was fully read.
                break;
            }

            uint64_t advanced = *read_result;
            read_total += advanced;
            while (advanced > 0 && iov_index < iovs.size()) {
                const uint64_t available =
                    iovs[iov_index].iov_len - iov_consumed;
                const uint64_t step = std::min<uint64_t>(advanced, available);
                iov_consumed += step;
                advanced -= step;
                if (iov_consumed == iovs[iov_index].iov_len) {
                    ++iov_index;
                    iov_consumed = 0;
                }
            }
        }

        if (read_error != ErrorCode::OK) {
            LOG(WARNING) << "DFS read failed for key " << request.key
                         << ", error=" << read_error;
            results.emplace_back(tl::make_unexpected(read_error));
            continue;
        }
        if (read_total != desc.object_size) {
            LOG(WARNING) << "DFS short read for key " << request.key
                         << ", expected=" << desc.object_size
                         << ", actual=" << read_total;
            results.emplace_back(
                tl::make_unexpected(ErrorCode::FILE_READ_FAIL));
            continue;
        }
        results.emplace_back();
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
