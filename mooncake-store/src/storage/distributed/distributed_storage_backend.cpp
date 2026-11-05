#include "l4/distributed_storage_internal.h"
#include "l4/dfs_descriptor_internal.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/dfs_log_limiter.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unordered_set>

#include "environ.h"
#include "storage/distributed/bucket_entry_layout.h"
#include "storage/distributed/dfs_io_timeout.h"
#include "storage/distributed/immutable_bucket_allocator.h"
#include "storage/distributed/dfs_root_identity.h"
#include "storage/distributed/dfs_read_probe.h"
#include "storage/distributed/dfs_global_allocator.h"
#include "thread_pool.h"
#include <map>
#include <optional>
#include <string_view>

#include "types.h"

namespace mooncake {

using namespace dfs_descriptor_detail;
constexpr char kDfsHealthProbeFilePrefix[] = ".mooncake_dfs_health_probe_";

namespace {

std::optional<int> ParseShardFileName(std::string_view name) {
    constexpr std::string_view prefix = "dfs_shard_";
    constexpr std::string_view suffix = ".data";
    if (!name.starts_with(prefix) || !name.ends_with(suffix)) {
        return std::nullopt;
    }
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    if (name.size() < 2) return std::nullopt;

    int index = 0;
    for (char digit : name) {
        if (digit < '0' || digit > '9' ||
            index > (std::numeric_limits<int>::max() - (digit - '0')) / 10) {
            return std::nullopt;
        }
        index = index * 10 + (digit - '0');
    }
    return index;
}

std::optional<int> ParseBucketFileName(std::string_view name) {
    constexpr std::string_view prefix = "bucket_";
    constexpr std::string_view suffix = ".data";
    if (!name.starts_with(prefix) || !name.ends_with(suffix)) {
        return std::nullopt;
    }
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    if (name.empty()) return std::nullopt;

    int index = 0;
    for (char digit : name) {
        if (digit < '0' || digit > '9' ||
            index > (std::numeric_limits<int>::max() - (digit - '0')) / 10) {
            return std::nullopt;
        }
        index = index * 10 + (digit - '0');
    }
    return index;
}

using l4::kBucketDataSuffix;
using l4::kBucketFilePrefix;

/**
 * @brief Validate a BUCKET-mode descriptor.
 *
 * A bucket descriptor points directly at an alignment-aligned object start;
 *
 * the reserved size covers the object plus trailing alignment padding.
 */

bool IsBucketDescriptorRangeValid(const DistributedFSDescriptor& desc,
                                  const DistributedStorageConfig& config) {
    if (config.alignment == 0 || desc.shard_idx < 0 || desc.object_size == 0 ||
        desc.aligned_size < desc.object_size ||
        desc.offset % config.alignment != 0 ||
        desc.aligned_size % config.alignment != 0 ||
        desc.offset > config.bucket_capacity ||
        desc.aligned_size > config.bucket_capacity - desc.offset ||
        desc.object_size > config.bucket_capacity - desc.offset) {
        return false;
    }
    constexpr uint64_t kMaxOffset =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    return desc.offset <= kMaxOffset &&
           desc.object_size <= kMaxOffset - desc.offset &&
           desc.aligned_size <= kMaxOffset - desc.offset;
}

// Completes a positional transfer of `total` bytes at `offset` whose first
// `transferred` bytes are already done, retrying short transfers.
tl::expected<void, ErrorCode> TransferAll(FileSystemAdapter& adapter, int fd,
                                          std::vector<iovec> iovs,
                                          int64_t offset, uint64_t total,
                                          uint64_t transferred, bool write) {
    size_t index = 0;
    size_t consumed = transferred;
    while (true) {
        while (consumed > 0 && index < iovs.size()) {
            if (consumed >= iovs[index].iov_len) {
                consumed -= iovs[index].iov_len;
                ++index;
            } else {
                iovs[index].iov_base =
                    static_cast<char*>(iovs[index].iov_base) + consumed;
                iovs[index].iov_len -= consumed;
                consumed = 0;
            }
        }
        if (transferred >= total) return {};

        while (index < iovs.size() && iovs[index].iov_len == 0) ++index;
        if (index == iovs.size() ||
            iovs.size() - index >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        const auto count = static_cast<int>(iovs.size() - index);
        const auto position = offset + static_cast<int64_t>(transferred);
        auto result =
            write ? adapter.WriteAt(fd, iovs.data() + index, count, position)
                  : adapter.ReadAt(fd, iovs.data() + index, count, position);
        if (!result) return tl::make_unexpected(result.error());
        if (*result == 0 || *result > total - transferred) {
            return tl::make_unexpected(write ? ErrorCode::FILE_WRITE_FAIL
                                             : ErrorCode::FILE_READ_FAIL);
        }
        consumed = *result;
        transferred += *result;
    }
}

tl::expected<uint64_t, ErrorCode> CalculateSliceBytes(
    std::span<const Slice> slices) {
    uint64_t total = 0;
    for (const auto& slice : slices) {
        if ((slice.ptr == nullptr && slice.size != 0) ||
            slice.size > std::numeric_limits<uint64_t>::max() - total) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        total += slice.size;
    }
    return total;
}

}  // namespace

DistributedStorageBackend::DistributedStorageBackend(
    const FileStorageConfig& file_storage_config,
    const DistributedStorageConfig& distributed_config,
    std::unique_ptr<FileSystemAdapter> fs_adapter)
    : DistributedStorageBackend(file_storage_config, distributed_config,
                                std::move(fs_adapter), nullptr, {}) {}

DistributedStorageBackend::DistributedStorageBackend(
    const FileStorageConfig& file_storage_config,
    const DistributedStorageConfig& distributed_config,
    std::unique_ptr<FileSystemAdapter> fs_adapter,
    std::string expected_root_identity)
    : DistributedStorageBackend(file_storage_config, distributed_config,
                                std::move(fs_adapter), nullptr,
                                std::move(expected_root_identity)) {}

DistributedStorageBackend::DistributedStorageBackend(
    const FileStorageConfig& file_storage_config,
    const DistributedStorageConfig& distributed_config,
    std::unique_ptr<FileSystemAdapter> fs_adapter,
    std::unique_ptr<ObjectStorageAdapter> object_storage_adapter,
    std::string expected_root_identity)
    : StorageBackendInterface(file_storage_config),
      fs_adapter_(std::move(fs_adapter)),
      object_storage_adapter_(std::move(object_storage_adapter)),
      distributed_config_(distributed_config),
      batch_timeout_fixed_ms_(distributed_config.batch_timeout_fixed_ms),
      expected_root_identity_(std::move(expected_root_identity)),
      root_dir_(distributed_config.fsdir),
      descriptor_root_dir_(distributed_config.fsdir) {
    CHECK((fs_adapter_ != nullptr) != (object_storage_adapter_ != nullptr))
        << "DistributedStorageBackend: exactly one I/O adapter is required";
    if (object_storage_adapter_) {
        storage_mode_ = DistributedStorageMode::kObjectStorage;
    }
    health_probe_path_prefix_ = root_dir_ + "/" + kDfsHealthProbeFilePrefix +
                                UuidToString(generate_uuid());
}

DistributedStorageBackend::~DistributedStorageBackend() {
    for (auto& [_, shard] : shard_files_) {
        if (shard && shard->fd >= 0 && fs_adapter_) {
            if (shard->read_fd >= 0 && shard->read_fd != shard->fd)
                fs_adapter_->CloseFile(shard->read_fd);
            fs_adapter_->CloseFile(shard->fd);
            shard->fd = -1;
            shard->read_fd = -1;
        }
    }
    if (fs_adapter_ && !active_health_probe_path_.empty()) {
        (void)fs_adapter_->DeleteFile(active_health_probe_path_);
    }
    // Drop cached bucket handles before the adapter goes away; each handle
    // closes its own fd in its destructor via the adapter pointer it holds.
    {
        std::lock_guard<std::mutex> lock(bucket_cache_mutex_);
        bucket_id_cache_.clear();
        bucket_id_direct_cache_.clear();
    }
    batch_read_pool_.reset();
    if (fs_adapter_) fs_adapter_->Shutdown();
}

tl::expected<void, ErrorCode> DistributedStorageBackend::Init() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    return InitLocked();
}

tl::expected<void, ErrorCode> DistributedStorageBackend::InitLocked() {
    if (initialized_.load(std::memory_order_acquire)) {
        LOG(WARNING) << "DistributedStorageBackend is already initialized";
        return ValidateRootIdentity();
    }

    if (UsesObjectStorage()) {
        auto init_result = object_storage_adapter_->Init();
        if (!init_result) return init_result;
        if (distributed_config_.enable_health_check) {
            auto health_result = object_storage_adapter_->CheckHealth();
            if (!health_result) {
                LOG(ERROR) << "Object storage health check failed, adapter="
                           << object_storage_adapter_->GetName() << ", error="
                           << static_cast<int>(health_result.error());
                return health_result;
            }
        }
        initialized_ = true;
        LOG(INFO) << "DistributedStorageBackend initialized, object adapter="
                  << object_storage_adapter_->GetName();
        return {};
    }

    std::error_code ec;
    if (!std::filesystem::is_directory(root_dir_, ec) || ec) {
        LOG(ERROR) << "DFS root directory is unavailable: " << root_dir_;
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }

    auto init_result = fs_adapter_->Init(root_dir_);
    if (!init_result) {
        const auto error = init_result.error();
        ResetFailedInitializationLocked();
        return tl::make_unexpected(error);
    }

    // Initialization is serialized and no batch I/O can run until it succeeds.
    // Retry the control plane when DFS was disabled at client startup, without
    // adding RPCs to business I/O or trusting the local marker as authority.
    if (expected_root_identity_.empty() && root_identity_provider_) {
        auto identity = root_identity_provider_();
        if (identity && !identity->empty() &&
            identity->size() <= kMaxDfsRootIdentitySize) {
            expected_root_identity_ = std::move(*identity);
        }
    }
    auto identity_result = ValidateRootIdentity();
    if (!identity_result) {
        LOG(ERROR) << "DFS root identity validation failed for " << root_dir_;
        ResetFailedInitializationLocked();
        return identity_result;
    }

    if (IsBucketMode()) {
        return InitBucketModeLocked();
    }

    // Only descriptors returned by the master identify published shards.
    // Opening files from the configured count or a directory scan can retain
    // handles to staged shards that a failed expansion subsequently unlinks.
    initialized_ = true;
    return {};
}

tl::expected<DistributedStorageBackend::ShardFile*, ErrorCode>
DistributedStorageBackend::GetOrOpenShard(
    const DistributedFSDescriptor& descriptor, bool read_only) {
    if (descriptor.shard_idx < 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    ShardFile* shard = nullptr;
    {
        std::lock_guard cache_lock(shard_files_mutex_);
        auto existing = shard_files_.find(descriptor.shard_idx);
        if (existing != shard_files_.end()) shard = existing->second.get();
    }
    if (shard) {
        std::lock_guard shard_lock(shard->mutex);
        // Only the exact path validated when this fd was opened can bypass
        // normalization. Aliases and changed descriptors take the full
        // validation path below. Cache entries live until backend teardown.
        if ((shard->fd >= 0 || (read_only && shard->read_fd >= 0)) &&
            shard->path == descriptor.file_path)
            return shard;
    }

    const auto path =
        std::filesystem::path(descriptor.file_path).lexically_normal();
    auto root = std::filesystem::path(root_dir_).lexically_normal();
    if (root.filename().empty()) root = root.parent_path();
    auto parent = path.parent_path();
    if (parent.empty()) parent = ".";
    const auto index = ParseShardFileName(path.filename().string());
    if (!index || *index != descriptor.shard_idx || parent != root) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    if (!shard) {
        std::lock_guard cache_lock(shard_files_mutex_);
        auto existing = shard_files_.find(descriptor.shard_idx);
        if (existing == shard_files_.end()) {
            existing = shard_files_
                           .emplace(descriptor.shard_idx,
                                    std::make_unique<ShardFile>())
                           .first;
        }
        shard = existing->second.get();
    }

    // File validation and opening may block. Serialize them only with this
    // shard's initialization and non-batched I/O, leaving other cached shards
    // available.
    std::lock_guard shard_lock(shard->mutex);
    auto file_path = path.string();
    if (shard->fd >= 0 || (read_only && shard->read_fd >= 0)) {
        if (shard->path != file_path) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        return shard;
    }

    auto fd = ValidateAndOpenShardFile(file_path);
    if (!fd && read_only) {
        fd = fs_adapter_->OpenFileReadOnly(file_path);
        if (!fd) return tl::make_unexpected(fd.error());
        shard->path = std::move(file_path);
        shard->read_fd = *fd;
        return shard;
    }
    if (!fd) return tl::make_unexpected(fd.error());
    // Failed attempts keep fd == -1 so a later request can retry
    // initialization.
    shard->path = std::move(file_path);
    shard->fd = *fd;
    if (shard->read_fd < 0) shard->read_fd = *fd;
    return shard;
}

tl::expected<int, ErrorCode>
DistributedStorageBackend::ValidateAndOpenShardFile(
    const std::string& file_path) {
    // The master prepares new shards before publishing their descriptors.
    // OpenFile may create a file, so reject missing/unprepared shards first.
    auto file_size = fs_adapter_->GetFileSize(file_path);
    if (!file_size) return tl::make_unexpected(file_size.error());
    if (*file_size != distributed_config_.shard_capacity) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    auto root = std::filesystem::path(root_dir_).lexically_normal();
    if (root.filename().empty()) root = root.parent_path();
    const auto path = std::filesystem::path(file_path);
    std::error_code ec;
    const auto canonical_root = std::filesystem::canonical(root, ec);
    if (ec) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    const auto canonical_path = std::filesystem::canonical(path, ec);
    if (ec) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    if (canonical_path.parent_path() != canonical_root ||
        canonical_path.filename() != path.filename()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    return fs_adapter_->OpenFileReadWriteExisting(file_path);
}

tl::expected<void, ErrorCode> DistributedStorageBackend::ReopenShardLocked(
    ShardFile& shard) {
    if (shard.path.empty()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    // Drop the stale descriptor first. The caller holds shard.mutex, which
    // serializes all I/O on this shard, so no other thread can be reading or
    // writing through the old fd while we close it.
    if (shard.fd >= 0) {
        if (shard.read_fd == shard.fd) shard.read_fd = -1;
        (void)fs_adapter_->CloseFile(shard.fd);
        shard.fd = -1;
    }
    auto fd = ValidateAndOpenShardFile(shard.path);
    if (!fd) return tl::make_unexpected(fd.error());
    shard.fd = *fd;
    if (shard.read_fd < 0) shard.read_fd = *fd;
    return {};
}

bool DistributedStorageBackend::UsesBucketAllocator() const {
    return distributed_config_.allocator_type == "bucket";
}

tl::expected<int, ErrorCode> DistributedStorageBackend::OpenBucket(
    const DistributedFSDescriptor& descriptor) {
    const auto path =
        std::filesystem::path(descriptor.file_path).lexically_normal();
    auto root = std::filesystem::path(root_dir_).lexically_normal();
    if (root.filename().empty()) root = root.parent_path();
    auto parent = path.parent_path();
    if (parent.empty()) parent = ".";
    const auto bucket_id = ParseBucketFileName(path.filename().string());
    if (!bucket_id || *bucket_id != descriptor.shard_idx || parent != root) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    const std::string file_path = path.string();
    auto file_size = fs_adapter_->GetFileSize(file_path);
    if (!file_size) return tl::make_unexpected(file_size.error());
    if (*file_size != distributed_config_.bucket_capacity) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    std::error_code error;
    const auto canonical_root = std::filesystem::canonical(root, error);
    if (error) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    const auto canonical_path = std::filesystem::canonical(path, error);
    if (error) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    if (canonical_path.parent_path() != canonical_root ||
        canonical_path.filename() != path.filename()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    return fs_adapter_->OpenExistingFile(file_path);
}

tl::expected<int64_t, ErrorCode> DistributedStorageBackend::BatchOffload(
    const std::unordered_map<std::string, std::vector<Slice>>& batch_object,
    std::function<ErrorCode(const std::vector<std::string>& keys,
                            std::vector<StorageObjectMetadata>& metadatas)>
        complete_handler,
    EvictionHandler eviction_handler) {
    if (!UsesObjectStorage()) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }
    if (!initialized_.load(std::memory_order_acquire)) {
        LOG(ERROR) << "DistributedStorageBackend is not initialized";
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    if (eviction_handler) {
        LOG_FIRST_N(WARNING, 1)
            << "DistributedStorageBackend does not support eviction, "
               "eviction_handler ignored";
    }

    std::vector<std::string> success_keys;
    std::vector<StorageObjectMetadata> success_metas;
    std::vector<ObjectPutRequest> requests;
    // Own descriptors through PutBatch; each request only borrows an array.
    std::vector<std::vector<iovec>> iov_storage;
    std::vector<size_t> request_sizes;
    requests.reserve(batch_object.size());
    iov_storage.reserve(batch_object.size());
    request_sizes.reserve(batch_object.size());
    for (const auto& [key, slices] : batch_object) {
        if (slices.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
            LOG(WARNING) << "Failed to offload key " << key
                         << ": slice count exceeds INT_MAX";
            continue;
        }
        std::vector<iovec> iovs;
        iovs.reserve(slices.size());
        size_t total_size = 0;
        bool total_size_overflow = false;
        for (const auto& slice : slices) {
            if (slice.size > std::numeric_limits<size_t>::max() - total_size) {
                total_size_overflow = true;
                break;
            }
            iovs.push_back({slice.ptr, slice.size});
            total_size += slice.size;
        }
        if (total_size_overflow) {
            LOG(WARNING) << "Failed to offload key " << key
                         << ": total slice size overflows size_t";
            continue;
        }

        iov_storage.push_back(std::move(iovs));
        const auto& stored_iovs = iov_storage.back();
        requests.push_back(
            {key, stored_iovs.data(), static_cast<int>(stored_iovs.size())});
        request_sizes.push_back(total_size);
    }

    auto results = object_storage_adapter_->PutBatch(requests);
    if (results.size() != requests.size()) {
        LOG(ERROR)
            << "Object storage returned an invalid PUT batch result count";
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i]) {
            LOG(WARNING) << "Failed to offload key " << requests[i].logical_key
                         << ": " << static_cast<int>(results[i].error());
            continue;
        }
        success_keys.push_back(requests[i].logical_key);
        success_metas.emplace_back(
            -1, 0, static_cast<int64_t>(requests[i].logical_key.size()),
            static_cast<int64_t>(request_sizes[i]), "");
    }

    if (complete_handler && !success_keys.empty()) {
        auto err = complete_handler(success_keys, success_metas);
        if (err != ErrorCode::OK) {
            return tl::make_unexpected(err);
        }
    }
    return static_cast<int64_t>(success_keys.size());
}

struct DistributedStorageBackend::DfsIoOp {
    size_t index;  // Position in the caller's request batch.
    const std::string& key;
    const DistributedFSDescriptor& descriptor;
    std::vector<iovec> iovs;
    uint64_t size;
    int64_t file_offset = 0;
};

tl::expected<void, ErrorCode> DistributedStorageBackend::ExecuteSingleIo(
    DfsIoOp& op, bool write) {
    if (UsesBucketAllocator()) {
        auto fd = OpenBucket(op.descriptor);
        if (!fd) return tl::make_unexpected(fd.error());
        auto result = TransferAll(*fs_adapter_, *fd, std::move(op.iovs),
                                  op.file_offset, op.size, 0, write);
        auto close_result = fs_adapter_->CloseFile(*fd);
        if (!result) return result;
        return close_result;
    }

    auto shard = GetOrOpenShard(op.descriptor);
    if (!shard) {
        LOG(ERROR) << "Failed to open DFS shard for key " << op.key << ": "
                   << shard.error();
        return tl::make_unexpected(shard.error());
    }
    std::lock_guard lock((*shard)->mutex);
    const auto count = static_cast<int>(op.iovs.size());
    const auto offset = op.file_offset;
    auto result =
        write
            ? fs_adapter_->WriteAt((*shard)->fd, op.iovs.data(), count, offset)
            : fs_adapter_->ReadAt((*shard)->fd, op.iovs.data(), count, offset);
    const char* op_name = write ? "write" : "read";
    if (!result && fs_adapter_->SupportsFdReopen()) {
        // A cached descriptor can go stale (fd invalidated, shard file
        // rotated). Reopen once under the shard lock and retry before
        // surfacing the failure. A short transfer below is treated as genuine
        // corruption and not retried. The lock already serializes shard I/O, so
        // the reopen adds no cost to the success path. Only adapters whose I/O
        // faults are fd-recoverable (POSIX) opt in via SupportsFdReopen.
        LOG(WARNING) << "DFS " << op_name << " failed for key " << op.key
                     << ", error=" << result.error() << ", reopening shard "
                     << op.descriptor.shard_idx;
        auto reopen = ReopenShardLocked(**shard);
        if (reopen) {
            result = write ? fs_adapter_->WriteAt((*shard)->fd, op.iovs.data(),
                                                  count, offset)
                           : fs_adapter_->ReadAt((*shard)->fd, op.iovs.data(),
                                                 count, offset);
        } else {
            LOG(ERROR) << "DFS shard " << op.descriptor.shard_idx
                       << " reopen failed for key " << op.key
                       << ", error=" << reopen.error();
        }
    }
    if (!result) {
        LOG(WARNING) << "DFS " << op_name << " failed for key " << op.key
                     << ", error=" << result.error();
        return tl::make_unexpected(result.error());
    }
    if (*result != op.size) {
        LOG(WARNING) << "DFS short " << op_name << " for key " << op.key
                     << ", expected=" << op.size << ", actual=" << *result;
        return tl::make_unexpected(write ? ErrorCode::FILE_WRITE_FAIL
                                         : ErrorCode::FILE_READ_FAIL);
    }
    return {};
}

void DistributedStorageBackend::ExecuteShardBatchIo(
    std::vector<DfsIoOp>& ops,
    std::vector<tl::expected<void, ErrorCode>>& results, bool write) {
    std::vector<FdIoRequest> io_requests;
    std::vector<DfsIoOp*> io_ops;
    io_requests.reserve(ops.size());
    io_ops.reserve(ops.size());
    for (auto& op : ops) {
        auto shard = GetOrOpenShard(op.descriptor);
        if (!shard) {
            LOG(ERROR) << "Failed to open DFS shard for key " << op.key << ": "
                       << shard.error();
            results[op.index] = tl::make_unexpected(shard.error());
            continue;
        }
        io_requests.push_back({(*shard)->fd, op.iovs.data(),
                               static_cast<int>(op.iovs.size()),
                               op.file_offset});
        io_ops.push_back(&op);
    }
    // Shard I/O fails short transfers.
    SubmitBatchIo(io_requests, io_ops, results, write, false);
}

void DistributedStorageBackend::ExecuteBucketBatchIo(
    std::vector<DfsIoOp>& ops,
    std::vector<tl::expected<void, ErrorCode>>& results, bool write) {
    // Group ops by the full descriptor identity OpenBucket validates, so each
    // bucket is opened once per batch.
    std::map<std::pair<std::string, int>, std::vector<DfsIoOp*>> buckets;
    for (auto& op : ops) {
        buckets[{op.descriptor.file_path, op.descriptor.shard_idx}].push_back(
            &op);
    }

    auto next = buckets.begin();
    while (next != buckets.end()) {
        std::vector<std::pair<int, const std::vector<DfsIoOp*>*>> opened;
        std::vector<FdIoRequest> io_requests;
        std::vector<DfsIoOp*> io_ops;
        for (; next != buckets.end() && opened.size() < kMaxOpenBucketsPerBatch;
             ++next) {
            const auto& bucket_ops = next->second;
            auto fd = OpenBucket(bucket_ops.front()->descriptor);
            if (!fd) {
                LOG(ERROR) << "Failed to open DFS bucket " << next->first.first
                           << ": " << fd.error();
                for (auto* op : bucket_ops) {
                    results[op->index] = tl::make_unexpected(fd.error());
                }
                continue;
            }
            opened.emplace_back(*fd, &bucket_ops);
            for (auto* op : bucket_ops) {
                io_requests.push_back({*fd, op->iovs.data(),
                                       static_cast<int>(op->iovs.size()),
                                       op->file_offset});
                io_ops.push_back(op);
            }
        }

        // Bucket I/O completes short transfers.
        SubmitBatchIo(io_requests, io_ops, results, write, true);

        // A failed close fails the bucket's otherwise successful requests.
        for (const auto& [fd, bucket_ops] : opened) {
            auto close_result = fs_adapter_->CloseFile(fd);
            if (close_result) continue;
            for (auto* op : *bucket_ops) {
                auto& result = results[op->index];
                if (result) result = tl::make_unexpected(close_result.error());
            }
        }
    }
}

void DistributedStorageBackend::SubmitBatchIo(
    const std::vector<FdIoRequest>& io_requests,
    const std::vector<DfsIoOp*>& io_ops,
    std::vector<tl::expected<void, ErrorCode>>& results, bool write,
    bool complete_short_io) {
    if (io_requests.empty()) return;
    const char* op_name = write ? "write" : "read";
    auto io_results = write ? fs_adapter_->BatchWriteAt(io_requests)
                            : fs_adapter_->BatchReadAt(io_requests);
    if (io_results.size() != io_requests.size()) {
        LOG(ERROR) << "Filesystem adapter returned an invalid batch " << op_name
                   << " result count";
        for (auto* op : io_ops) {
            results[op->index] = tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        return;
    }
    for (size_t i = 0; i < io_results.size(); ++i) {
        auto& op = *io_ops[i];
        const auto& result = io_results[i];
        if (!result) {
            LOG(WARNING) << "DFS " << op_name << " failed for key " << op.key
                         << ", error=" << result.error();
            results[op.index] = tl::make_unexpected(result.error());
        } else if (*result == op.size) {
            continue;
        } else if (complete_short_io && *result != 0 && *result < op.size) {
            auto rest =
                TransferAll(*fs_adapter_, io_requests[i].fd, std::move(op.iovs),
                            io_requests[i].offset, op.size, *result, write);
            if (!rest) results[op.index] = tl::make_unexpected(rest.error());
        } else {
            LOG(WARNING) << "DFS short " << op_name << " for key " << op.key
                         << ", expected=" << op.size << ", actual=" << *result;
            results[op.index] = tl::make_unexpected(
                write ? ErrorCode::FILE_WRITE_FAIL : ErrorCode::FILE_READ_FAIL);
        }
    }
}

std::vector<tl::expected<void, ErrorCode>>
DistributedStorageBackend::BatchWrite(
    const std::vector<DfsWriteRequest>& requests) {
    if (UsesObjectStorage()) {
        auto initialized = EnsureInitialized();
        if (!initialized)
            return std::vector<tl::expected<void, ErrorCode>>(
                requests.size(), tl::make_unexpected(initialized.error()));
    }
    {
        std::vector<tl::expected<void, ErrorCode>> results(requests.size());
        if (UsesObjectStorage()) {
            std::vector<ObjectPutRequest> object_requests;
            std::vector<std::vector<iovec>> iov_storage;
            std::vector<size_t> request_indices;
            object_requests.reserve(requests.size());
            iov_storage.reserve(requests.size());
            request_indices.reserve(requests.size());
            for (size_t index = 0; index < requests.size(); ++index) {
                const auto& request = requests[index];
                const auto& desc = request.descriptor;
                auto total_size = CalculateSliceBytes(request.slices);
                if (request.slices.empty() || !total_size ||
                    request.slices.size() >
                        static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    desc.object_size == 0 || *total_size != desc.object_size) {
                    results[index] =
                        tl::make_unexpected(ErrorCode::INVALID_PARAMS);
                    continue;
                }
                auto& iovs = iov_storage.emplace_back();
                iovs.reserve(request.slices.size());
                for (const auto& slice : request.slices)
                    iovs.push_back({slice.ptr, slice.size});
                object_requests.push_back(
                    {request.key, iovs.data(), static_cast<int>(iovs.size())});
                request_indices.push_back(index);
            }

            auto object_results =
                object_storage_adapter_->PutBatch(object_requests);
            if (object_results.size() != object_requests.size()) {
                for (size_t index : request_indices) {
                    results[index] =
                        tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
                }
                return results;
            }
            for (size_t index = 0; index < object_results.size(); ++index) {
                results[request_indices[index]] =
                    std::move(object_results[index]);
            }
            return results;
        }
    }

    return BatchWriteBlocking(requests);
}

std::vector<tl::expected<void, ErrorCode>> DistributedStorageBackend::BatchRead(
    const std::vector<DfsReadRequest>& requests) {
    if (UsesObjectStorage()) {
        auto initialized = EnsureInitialized();
        if (!initialized)
            return std::vector<tl::expected<void, ErrorCode>>(
                requests.size(), tl::make_unexpected(initialized.error()));
    }
    {
        std::vector<tl::expected<void, ErrorCode>> results(requests.size());
        if (UsesObjectStorage()) {
            std::vector<ObjectGetRequest> object_requests;
            std::vector<std::vector<char>> staging_buffers;
            std::vector<size_t> staging_request_positions;
            std::vector<size_t> request_indices;
            object_requests.reserve(requests.size());
            staging_buffers.reserve(requests.size());
            staging_request_positions.reserve(requests.size());
            request_indices.reserve(requests.size());
            for (size_t index = 0; index < requests.size(); ++index) {
                const auto& request = requests[index];
                const auto& desc = request.descriptor;
                auto capacity = CalculateSliceBytes(request.slices);
                if (request.slices.empty() || !capacity ||
                    desc.object_size > std::numeric_limits<size_t>::max() ||
                    desc.object_size == 0 || *capacity < desc.object_size) {
                    results[index] =
                        tl::make_unexpected(ErrorCode::INVALID_PARAMS);
                    continue;
                }
                void* buffer = request.slices.front().ptr;
                size_t staging_position = std::numeric_limits<size_t>::max();
                if (request.slices.size() > 1) {
                    staging_position = staging_buffers.size();
                    staging_buffers.emplace_back(
                        static_cast<size_t>(desc.object_size));
                    buffer = staging_buffers.back().data();
                }
                object_requests.push_back(
                    {request.key, buffer,
                     static_cast<size_t>(desc.object_size)});
                staging_request_positions.push_back(staging_position);
                request_indices.push_back(index);
            }

            auto object_results =
                object_storage_adapter_->GetBatch(object_requests);
            if (object_results.size() != object_requests.size()) {
                for (size_t index : request_indices) {
                    results[index] =
                        tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
                }
                return results;
            }
            for (size_t i = 0; i < object_results.size(); ++i) {
                const size_t request_index = request_indices[i];
                if (!object_results[i]) {
                    results[request_index] =
                        tl::make_unexpected(object_results[i].error());
                    continue;
                }
                if (*object_results[i] != object_requests[i].size) {
                    results[request_index] =
                        tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
                    continue;
                }
                const size_t staging_position = staging_request_positions[i];
                if (staging_position != std::numeric_limits<size_t>::max()) {
                    const auto& buffer = staging_buffers[staging_position];
                    size_t offset = 0;
                    for (const auto& slice : requests[request_index].slices) {
                        if (offset == buffer.size()) break;
                        const size_t length =
                            std::min(slice.size, buffer.size() - offset);
                        std::memcpy(slice.ptr, buffer.data() + offset, length);
                        offset += length;
                    }
                }
                results[request_index] = {};
            }
            return results;
        }
    }

    if (requests.empty()) return {};
    const auto request_start = std::chrono::steady_clock::now();
    uint64_t total_bytes = 0;
    uint64_t inflight_bytes = 0;
    bool all_buffers_owned = true;
    for (const auto& request : requests) {
        SaturatingAddBytes(request.descriptor.object_size, &total_bytes);
        for (const auto& slice : request.slices) {
            SaturatingAddBytes(slice.size, &inflight_bytes);
            if (slice.size != 0 && !request.buffer_owner) {
                all_buffers_owned = false;
            }
        }
    }
    if (dfs_read_timed_out_.load(std::memory_order_acquire)) {
        LOG_EVERY_N(WARNING, 64)
            << "DFS BatchGet rejected by timeout latch: "
            << "reason=previous_read_timeout_latched, requests="
            << requests.size() << ", total_bytes=" << total_bytes
            << ", inflight_bytes=" << inflight_bytes
            << ", configured_timeout_ms="
            << distributed_config_.batch_get_timeout_ms
            << ", max_inflight_batches="
            << distributed_config_.batch_get_max_inflight_batches
            << ", max_inflight_bytes="
            << distributed_config_.batch_get_max_inflight_bytes
            << ", error=" << ErrorCode::DFS_BATCH_GET_TIMEOUT;
        return std::vector<tl::expected<void, ErrorCode>>(
            requests.size(),
            tl::make_unexpected(ErrorCode::DFS_BATCH_GET_TIMEOUT));
    }
    const auto base_timeout_ms = distributed_config_.batch_get_timeout_ms;
    const auto timeout_ms = DfsBatchGetTimeoutMs(
        base_timeout_ms, total_bytes,
        batch_timeout_fixed_ms_.load(std::memory_order_acquire));
    const auto admission_deadline =
        base_timeout_ms == 0 ? std::chrono::steady_clock::time_point::max()
                             : DfsIoDeadlineFromNow(timeout_ms);
    const ErrorCode admission_result =
        AcquireInflightReadUntil(inflight_bytes, admission_deadline);
    if (admission_result != ErrorCode::OK) {
        LOG_EVERY_N(WARNING, 64)
            << "DFS BatchGet admission rejected: reason="
            << (admission_result == ErrorCode::DFS_BATCH_GET_TIMEOUT
                    ? "dfs_read_timeout_latched"
                    : "inflight_capacity_deadline_expired")
            << ", request_bytes=" << inflight_bytes << ", max_batches="
            << distributed_config_.batch_get_max_inflight_batches
            << ", max_bytes="
            << distributed_config_.batch_get_max_inflight_bytes
            << ", error=" << admission_result;
        return std::vector<tl::expected<void, ErrorCode>>(
            requests.size(), tl::make_unexpected(admission_result));
    }

    if (base_timeout_ms == 0 || !all_buffers_owned) {
        if (base_timeout_ms != 0 && !all_buffers_owned) {
            LOG_EVERY_N(WARNING, 64)
                << "DFS BatchGet destination has no lifetime owner; using "
                   "blocking I/O to prevent buffer reuse while preadv is "
                   "still running";
        }
        try {
            auto results = BatchReadBlocking(requests);
            ReleaseInflightRead(inflight_bytes);
            return results;
        } catch (...) {
            ReleaseInflightRead(inflight_bytes);
            throw;
        }
    }
    // Admission waiting and DFS I/O have different failure semantics. Start
    // the I/O deadline only after capacity has been acquired so queueing behind
    // healthy reads can never set the backend-wide DFS timeout latch.
    const auto deadline = DfsIoDeadlineFromNow(timeout_ms);
    struct TimedReadContext {
        std::mutex mutex;
        std::condition_variable cv;
        bool done = false;
        std::chrono::steady_clock::time_point completed_at;
        std::vector<DfsReadRequest> requests;
        std::vector<tl::expected<void, ErrorCode>> results;
        std::function<void()> release_inflight;

        ~TimedReadContext() {
            if (release_inflight) release_inflight();
        }
    };
    std::shared_ptr<TimedReadContext> context;
    try {
        context = std::make_shared<TimedReadContext>();
        // Preserve the request metadata for the detached worker, but read
        // directly into the caller-provided slices. This keeps aligned pinned
        // buffers on the O_DIRECT zero-copy path and avoids a private buffer
        // allocation plus a completion-time memcpy for every object. Production
        // callers attach the staging allocation through buffer_owner, and this
        // copied request quarantines it until the underlying read really ends.
        context->requests = requests;
    } catch (const std::bad_alloc&) {
        ReleaseInflightRead(inflight_bytes);
        return std::vector<tl::expected<void, ErrorCode>>(
            requests.size(),
            tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE));
    }

    std::shared_ptr<DistributedStorageBackend> self;
    try {
        self = shared_from_this();
    } catch (const std::bad_weak_ptr&) {
        // Test-only/legacy stack ownership cannot safely outlive this call.
        // Keep the old blocking lifetime contract in that configuration.
        try {
            auto results = BatchReadBlocking(requests);
            ReleaseInflightRead(inflight_bytes);
            return results;
        } catch (...) {
            ReleaseInflightRead(inflight_bytes);
            throw;
        }
    }
    // Keep the backend alive through quota release too; relying on the
    // destruction order of separate thread-lambda captures would leave a
    // narrow use-after-free window during shutdown.
    try {
        context->release_inflight = [owner = self, inflight_bytes]() {
            owner->ReleaseInflightRead(inflight_bytes);
        };
    } catch (const std::bad_alloc&) {
        ReleaseInflightRead(inflight_bytes);
        return std::vector<tl::expected<void, ErrorCode>>(
            requests.size(),
            tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE));
    }

    if (std::chrono::steady_clock::now() >= deadline) {
        SetDfsReadTimedOut(true);
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - request_start)
                .count();
        LOG_EVERY_N(WARNING, 64)
            << "DFS BatchGet timed out before worker launch: "
            << "reason=deadline_expired_after_admission, requests="
            << requests.size() << ", total_bytes=" << total_bytes
            << ", inflight_bytes=" << inflight_bytes
            << ", timeout_ms=" << timeout_ms << ", elapsed_ms=" << elapsed_ms
            << ", max_inflight_batches="
            << distributed_config_.batch_get_max_inflight_batches
            << ", max_inflight_bytes="
            << distributed_config_.batch_get_max_inflight_bytes
            << ", error=" << ErrorCode::DFS_BATCH_GET_TIMEOUT;
        return std::vector<tl::expected<void, ErrorCode>>(
            requests.size(),
            tl::make_unexpected(ErrorCode::DFS_BATCH_GET_TIMEOUT));
    }
    try {
        std::thread([self = std::move(self), context]() {
            std::vector<tl::expected<void, ErrorCode>> results;
            try {
                results = self->BatchReadBlocking(context->requests);
            } catch (const std::bad_alloc&) {
                results.assign(
                    context->requests.size(),
                    tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE));
            } catch (const std::exception& e) {
                LOG(ERROR) << "Timed DFS BatchGet worker failed: " << e.what();
                results.assign(context->requests.size(),
                               tl::make_unexpected(ErrorCode::INTERNAL_ERROR));
            } catch (...) {
                LOG(ERROR) << "Timed DFS BatchGet worker failed";
                results.assign(context->requests.size(),
                               tl::make_unexpected(ErrorCode::INTERNAL_ERROR));
            }
            {
                std::lock_guard lock(context->mutex);
                context->results = std::move(results);
                context->completed_at = std::chrono::steady_clock::now();
                context->done = true;
            }
            context->cv.notify_all();
        }).detach();
    } catch (const std::system_error&) {
        return std::vector<tl::expected<void, ErrorCode>>(
            requests.size(),
            tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE));
    }

    std::unique_lock lock(context->mutex);
    const bool completed_before_deadline = context->cv.wait_until(
        lock, deadline, [&context] { return context->done; });
    const auto finished_at = std::chrono::steady_clock::now();
    if (!completed_before_deadline || context->completed_at > deadline) {
        SetDfsReadTimedOut(true);
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(finished_at -
                                                                  request_start)
                .count();
        const auto worker_elapsed_ms =
            context->done
                ? std::chrono::duration_cast<std::chrono::milliseconds>(
                      context->completed_at - request_start)
                      .count()
                : -1;
        LOG_EVERY_N(WARNING, 64)
            << "DFS BatchGet I/O timed out: "
            << "reason="
            << (completed_before_deadline ? "worker_completed_after_deadline"
                                          : "worker_not_completed_by_deadline")
            << ", requests=" << requests.size()
            << ", total_bytes=" << total_bytes
            << ", inflight_bytes=" << inflight_bytes
            << ", timeout_ms=" << timeout_ms << ", elapsed_ms=" << elapsed_ms
            << ", worker_elapsed_ms=" << worker_elapsed_ms
            << ", max_inflight_batches="
            << distributed_config_.batch_get_max_inflight_batches
            << ", max_inflight_bytes="
            << distributed_config_.batch_get_max_inflight_bytes
            << ", error=" << ErrorCode::DFS_BATCH_GET_TIMEOUT;
        return std::vector<tl::expected<void, ErrorCode>>(
            requests.size(),
            tl::make_unexpected(ErrorCode::DFS_BATCH_GET_TIMEOUT));
    }
    return context->results;
}

ObjectStorageIoResults DistributedStorageBackend::BatchDeleteObjects(
    std::span<const std::string> logical_keys) {
    if (!initialized_ || !UsesObjectStorage() || !object_storage_adapter_) {
        return ObjectStorageIoResults(
            logical_keys.size(),
            tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE));
    }
    return object_storage_adapter_->DeleteBatch(logical_keys);
}

tl::expected<void, ErrorCode> DistributedStorageBackend::BatchLoad(
    std::unordered_map<std::string, Slice>& batched_slices) {
    if (!UsesObjectStorage()) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }
    if (!initialized_.load(std::memory_order_acquire)) {
        LOG(ERROR) << "DistributedStorageBackend is not initialized";
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    std::vector<ObjectGetRequest> requests;
    requests.reserve(batched_slices.size());
    for (auto& [key, slice] : batched_slices) {
        requests.push_back({key, slice.ptr, slice.size});
    }
    auto results = object_storage_adapter_->GetBatch(requests);
    if (results.size() != requests.size()) {
        LOG(ERROR)
            << "Object storage returned an invalid GET batch result count";
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i]) {
            return tl::make_unexpected(results[i].error());
        }
        if (*results[i] != requests[i].size) {
            return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
        }
    }
    return {};
}

tl::expected<bool, ErrorCode> DistributedStorageBackend::IsExist(
    const std::string& key) {
    if (!UsesObjectStorage()) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }
    if (!initialized_.load(std::memory_order_acquire)) {
        LOG(ERROR) << "DistributedStorageBackend is not initialized";
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return object_storage_adapter_->Exists(key);
}

tl::expected<bool, ErrorCode> DistributedStorageBackend::IsEnableOffloading() {
    return UsesObjectStorage();
}

tl::expected<void, ErrorCode> DistributedStorageBackend::ScanMeta(
    const std::function<
        ErrorCode(const std::vector<std::string>& keys,
                  std::vector<StorageObjectMetadata>& metadatas)>& handler) {
    if (!UsesObjectStorage()) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }
    if (!initialized_.load(std::memory_order_acquire)) {
        LOG(ERROR) << "DistributedStorageBackend is not initialized";
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    std::vector<std::string> batch_keys;
    std::vector<StorageObjectMetadata> batch_metas;
    const size_t batch_limit = static_cast<size_t>(std::max<int64_t>(
        1, file_storage_config_.scanmeta_iterator_keys_limit));

    auto key_infos = object_storage_adapter_->ListKeys();
    if (!key_infos) {
        LOG(ERROR) << "Failed to list keys from object storage adapter: "
                   << static_cast<int>(key_infos.error());
        return tl::make_unexpected(key_infos.error());
    }

    for (const auto& info : *key_infos) {
        batch_keys.push_back(info.logical_key);
        batch_metas.emplace_back(-1, 0,
                                 static_cast<int64_t>(info.logical_key.size()),
                                 static_cast<int64_t>(info.size), "");
        if (batch_keys.size() >= batch_limit) {
            auto err = handler(batch_keys, batch_metas);
            if (err != ErrorCode::OK) return tl::make_unexpected(err);
            batch_keys.clear();
            batch_metas.clear();
        }
    }
    if (!batch_keys.empty()) {
        auto err = handler(batch_keys, batch_metas);
        if (err != ErrorCode::OK) return tl::make_unexpected(err);
    }
    return {};
}

}  // namespace mooncake
