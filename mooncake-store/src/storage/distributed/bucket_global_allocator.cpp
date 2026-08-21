#include "storage/distributed/bucket_global_allocator.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <utility>

#include <ylt/struct_pb.hpp>

#include "crc32c.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/posix_fs_adapter.h"
#ifdef USE_3FS
#include "storage/distributed/hf3fs_adapter.h"
#endif

namespace mooncake {

namespace {

constexpr const char* kBucketFilePrefix = "bucket_";
constexpr const char* kBucketDataSuffix = ".data";
constexpr const char* kBucketMetaSuffix = ".meta";
constexpr const char* kBucketMetaLogSuffix = ".meta.log";

int64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int64_t WallClockNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

bool IsLive(BucketEntryState state) {
    return state == BucketEntryState::PENDING ||
           state == BucketEntryState::COMMITTED;
}

bool IsKnownEntryState(int32_t state) {
    return state == static_cast<int32_t>(BucketEntryState::PENDING) ||
           state == static_cast<int32_t>(BucketEntryState::COMMITTED) ||
           state == static_cast<int32_t>(BucketEntryState::TOMBSTONE);
}

uint32_t ComputeMetadataChecksum(PersistedBucketMetadata snapshot) {
    // The checksum covers the serialized form with the checksum field zeroed,
    // so verification can recompute it the same way after loading.
    snapshot.checksum = 0;
    std::string payload;
    struct_pb::to_pb(snapshot, payload);
    return Crc32cValue(payload.data(), payload.size());
}

uint32_t ComputeLegacyMetadataChecksum(LegacyPersistedBucketMetadata snapshot) {
    snapshot.checksum = 0;
    std::string payload;
    struct_pb::to_pb(snapshot, payload);
    return Crc32cValue(payload.data(), payload.size());
}

PersistedBucketMetadata UpgradeLegacyMetadata(
    const LegacyPersistedBucketMetadata& legacy) {
    PersistedBucketMetadata snapshot;
    snapshot.version = kBucketMetadataVersion;
    snapshot.bucket_id = legacy.bucket_id;
    snapshot.bucket_generation = legacy.bucket_generation;
    snapshot.capacity = legacy.capacity;
    snapshot.alignment = legacy.alignment;
    snapshot.append_offset = legacy.append_offset;
    snapshot.log_seq = legacy.log_seq;
    snapshot.snapshot_epoch = 0;
    snapshot.evicting = legacy.evicting;
    snapshot.entries = legacy.entries;
    snapshot.checksum = ComputeMetadataChecksum(snapshot);
    return snapshot;
}

// --- little-endian fixed-width encode/decode for the metadata log ---

void PutU32(std::string& out, uint32_t value) {
    char buffer[4];
    for (int i = 0; i < 4; ++i) {
        buffer[i] = static_cast<char>((value >> (8 * i)) & 0xFFu);
    }
    out.append(buffer, sizeof(buffer));
}

void PutU64(std::string& out, uint64_t value) {
    char buffer[8];
    for (int i = 0; i < 8; ++i) {
        buffer[i] = static_cast<char>((value >> (8 * i)) & 0xFFu);
    }
    out.append(buffer, sizeof(buffer));
}

uint32_t GetU32(const char* data) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(static_cast<unsigned char>(data[i]))
                 << (8 * i);
    }
    return value;
}

uint64_t GetU64(const char* data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(data[i]))
                 << (8 * i);
    }
    return value;
}

/**
 * @brief Find the next record boundary after a corrupt one.
 *
 * A single damaged record must not cost the rest of the log, so replay scans
 * forward for the magic that starts every record and resumes there. The CRC on
 * the next record is what confirms the guess.
 */
size_t FindNextMagic(std::string_view payload, size_t from) {
    char magic[4];
    std::string encoded;
    encoded.reserve(4);
    PutU32(encoded, kMetaLogMagic);
    std::memcpy(magic, encoded.data(), sizeof(magic));

    if (payload.size() < sizeof(magic)) return std::string_view::npos;
    for (size_t pos = from; pos + sizeof(magic) <= payload.size(); ++pos) {
        if (std::memcmp(payload.data() + pos, magic, sizeof(magic)) == 0) {
            return pos;
        }
    }
    return std::string_view::npos;
}

/**
 * @brief Extract the bucket id from a `bucket_<id><suffix>` file name.
 */
std::optional<int64_t> ParseBucketFileName(const std::string& name,
                                           const char* suffix) {
    const std::string prefix(kBucketFilePrefix);
    const std::string tail(suffix);
    if (name.size() <= prefix.size() + tail.size()) return std::nullopt;
    if (name.compare(0, prefix.size(), prefix) != 0) return std::nullopt;
    if (name.compare(name.size() - tail.size(), tail.size(), tail) != 0) {
        return std::nullopt;
    }

    const std::string digits =
        name.substr(prefix.size(), name.size() - prefix.size() - tail.size());
    if (digits.empty() ||
        !std::all_of(digits.begin(), digits.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return std::nullopt;
    }
    try {
        const long long value = std::stoll(digits);
        if (value < 0 || value > kMaxBucketId) return std::nullopt;
        return static_cast<int64_t>(value);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

}  // namespace

const char* ToString(MetaLogOp op) {
    switch (op) {
        case MetaLogOp::ADD_PENDING:
            return "add_pending";
        case MetaLogOp::MARK_COMMITTED:
            return "mark_committed";
        case MetaLogOp::TOMBSTONE:
            return "tombstone";
    }
    return "unknown";
}

void SerializeMetaLogRecord(const MetaLogRecord& record, std::string& out) {
    const size_t start = out.size();
    const uint64_t key_size = record.key.size();
    const uint64_t record_size = kMetaLogHeaderSize + key_size;

    PutU32(out, kMetaLogMagic);
    PutU32(out, 0);  // crc placeholder, filled in below
    PutU32(out, static_cast<uint32_t>(record_size));
    PutU32(out, static_cast<uint32_t>(record.op));
    PutU64(out, record.seq);
    PutU64(out, static_cast<uint64_t>(record.bucket_id));
    PutU64(out, record.bucket_generation);
    PutU64(out, static_cast<uint64_t>(record.timestamp_ns));
    PutU64(out, record.entry_offset);
    PutU64(out, key_size);
    PutU64(out, record.value_size);
    PutU64(out, record.reserved_size);
    PutU64(out, record.entry_generation);
    PutU64(out, record.append_offset);
    out.append(record.key);

    // CRC covers everything after the magic and the CRC slot itself, so a
    // record whose body was torn mid-write fails verification even though its
    // framing still parses.
    const size_t crc_begin = start + 8;
    const uint32_t crc =
        Crc32cValue(out.data() + crc_begin, out.size() - crc_begin);
    std::string encoded;
    PutU32(encoded, crc);
    std::memcpy(out.data() + start + 4, encoded.data(), encoded.size());
}

std::optional<MetaLogRecord> DeserializeMetaLogRecord(std::string_view payload,
                                                      size_t pos,
                                                      size_t& consumed) {
    consumed = 0;
    if (pos >= payload.size()) return std::nullopt;
    if (payload.size() - pos < kMetaLogHeaderSize) return std::nullopt;

    const char* data = payload.data() + pos;
    if (GetU32(data) != kMetaLogMagic) return std::nullopt;

    const uint32_t crc = GetU32(data + 4);
    const uint32_t record_size = GetU32(data + 8);
    if (record_size < kMetaLogHeaderSize ||
        record_size > kMetaLogMaxRecordSize) {
        return std::nullopt;
    }
    if (payload.size() - pos < record_size) {
        // A torn tail record: the writer was interrupted before the append
        // completed. Everything before it is still valid.
        return std::nullopt;
    }

    const uint32_t actual_crc = Crc32cValue(data + 8, record_size - 8);
    if (actual_crc != crc) {
        // Report the length so the caller can skip exactly this record; the
        // framing itself verified, only the body is bad.
        consumed = record_size;
        return std::nullopt;
    }

    const uint32_t op_value = GetU32(data + 12);
    if (op_value > static_cast<uint32_t>(MetaLogOp::TOMBSTONE)) {
        consumed = record_size;
        return std::nullopt;
    }

    MetaLogRecord record;
    record.op = static_cast<MetaLogOp>(op_value);
    record.seq = GetU64(data + 16);
    record.bucket_id = static_cast<int64_t>(GetU64(data + 24));
    record.bucket_generation = GetU64(data + 32);
    record.timestamp_ns = static_cast<int64_t>(GetU64(data + 40));
    record.entry_offset = GetU64(data + 48);
    record.key_size = GetU64(data + 56);
    record.value_size = GetU64(data + 64);
    record.reserved_size = GetU64(data + 72);
    record.entry_generation = GetU64(data + 80);
    record.append_offset = GetU64(data + 88);

    if (record.key_size != record_size - kMetaLogHeaderSize) {
        consumed = record_size;
        return std::nullopt;
    }
    record.key.assign(data + kMetaLogHeaderSize, record.key_size);

    consumed = record_size;
    return record;
}

BucketGlobalAllocator::PendingEviction::~PendingEviction() {
    // An unresolved transaction must not leave the bucket frozen forever.
    // Note: `owner_` must still be set when AbortEviction is entered, because
    // that is how it recognises the transaction as its own; clearing it here
    // first would make the abort a silent no-op. AbortEviction clears it.
    if (owner_ != nullptr) {
        owner_->AbortEviction(std::move(*this), /*demote=*/false);
    }
}

BucketGlobalAllocator::PendingEviction::PendingEviction(
    PendingEviction&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      bucket_id_(std::exchange(other.bucket_id_, -1)),
      bucket_generation_(std::exchange(other.bucket_generation_, 0)),
      candidates_(std::move(other.candidates_)) {
    other.candidates_.clear();
}

BucketGlobalAllocator::PendingEviction&
BucketGlobalAllocator::PendingEviction::operator=(
    PendingEviction&& other) noexcept {
    if (this != &other) {
        if (owner_ != nullptr) {
            PendingEviction discarded;
            discarded.owner_ = std::exchange(owner_, nullptr);
            discarded.bucket_id_ = bucket_id_;
            discarded.bucket_generation_ = bucket_generation_;
            discarded.candidates_ = std::move(candidates_);
            discarded.owner_->AbortEviction(std::move(discarded),
                                            /*demote=*/false);
        }
        owner_ = std::exchange(other.owner_, nullptr);
        bucket_id_ = std::exchange(other.bucket_id_, -1);
        bucket_generation_ = std::exchange(other.bucket_generation_, 0);
        candidates_ = std::move(other.candidates_);
        other.candidates_.clear();
    }
    return *this;
}

BucketGlobalAllocator::~BucketGlobalAllocator() {
    if (initialized_.load(std::memory_order_acquire)) {
        // A clean shutdown should not leave deferred tombstones unflushed, and
        // compacting here means the next start reads a snapshot instead of
        // replaying the whole log.
        FlushDirtyMetadata();
        CompactAllBuckets();
    }
    {
        std::lock_guard<std::mutex> log_lock(log_mutex_);
        for (auto& [bucket_id, fd] : log_fds_) {
            (void)bucket_id;
            if (fd >= 0 && fs_adapter_) (void)fs_adapter_->CloseFile(fd);
        }
        log_fds_.clear();
        log_bytes_.clear();
        log_synced_seq_.clear();
    }
    if (fs_adapter_) fs_adapter_->Shutdown();
}

std::string BucketGlobalAllocator::FormatBucketId(int64_t bucket_id) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << bucket_id;
    return oss.str();
}

std::string BucketGlobalAllocator::BucketDataPath(int64_t bucket_id) const {
    return fsdir_ + "/" + kBucketFilePrefix + FormatBucketId(bucket_id) +
           kBucketDataSuffix;
}

std::string BucketGlobalAllocator::BucketMetaPath(int64_t bucket_id) const {
    return fsdir_ + "/" + kBucketFilePrefix + FormatBucketId(bucket_id) +
           kBucketMetaSuffix;
}

std::string BucketGlobalAllocator::BucketMetaSlotPath(int64_t bucket_id,
                                                       int slot) const {
    return BucketMetaPath(bucket_id) + "." + std::to_string(slot);
}

std::string BucketGlobalAllocator::BucketMetaLogPath(int64_t bucket_id) const {
    return fsdir_ + "/" + kBucketFilePrefix + FormatBucketId(bucket_id) +
           kBucketMetaLogSuffix;
}

tl::expected<void, ErrorCode> BucketGlobalAllocator::Init(
    const DistributedStorageConfig& config) {
    if (initialized_.load(std::memory_order_acquire)) return {};

    if (!config.ValidateForAllocator()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (!config.ValidateForBucketAllocator()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    fsdir_ = config.fsdir;
    fs_adapter_type_ = config.fs_adapter_type;
    bucket_capacity_ = config.bucket_capacity;
    alignment_ = config.alignment;
    max_bucket_count_ = config.max_bucket_count;
    eviction_enabled_ = config.eviction_enabled;
    eviction_high_watermark_ = config.eviction_high_watermark;
    eviction_low_watermark_ = config.eviction_low_watermark;
    eviction_check_interval_ = config.eviction_check_interval;
    log_compaction_threshold_ = config.bucket_meta_log_threshold;

    std::error_code ec;
    std::filesystem::create_directories(fsdir_, ec);
    if (ec) {
        LOG(ERROR) << "Failed to create DFS bucket directory " << fsdir_ << ": "
                   << ec.message();
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }

    if (config.fs_adapter_type == "posix") {
        fs_adapter_ = std::make_unique<PosixFsAdapter>();
    } else if (config.fs_adapter_type == "hf3fs") {
#ifdef USE_3FS
        fs_adapter_ = std::make_unique<Hf3fsAdapter>();
#else
        LOG(ERROR) << "The hf3fs DFS adapter requires Mooncake to be built "
                      "with the USE_3FS compile-time option (-DUSE_3FS=ON)";
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
#endif
    }
    if (!fs_adapter_) {
        LOG(ERROR) << "Unsupported DFS fs adapter type "
                   << config.fs_adapter_type;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    auto adapter_init = fs_adapter_->Init(fsdir_);
    if (!adapter_init) {
        LOG(ERROR) << "Failed to initialize DFS fs adapter "
                   << config.fs_adapter_type << " for fsdir=" << fsdir_
                   << ", error=" << adapter_init.error();
        fs_adapter_.reset();
        return tl::make_unexpected(adapter_init.error());
    }

    auto recovered = RecoverFromDisk();
    if (!recovered) {
        fs_adapter_->Shutdown();
        fs_adapter_.reset();
        return recovered;
    }

    initialized_.store(true, std::memory_order_release);

    // Recovery folded each log into memory; publish that as a snapshot right
    // away so a second restart starts from a clean snapshot slot and empty log.
    const size_t compacted = CompactAllBuckets();

    LOG(INFO) << "DFS bucket allocator initialized, fsdir=" << fsdir_
              << ", bucket_capacity=" << bucket_capacity_
              << ", alignment=" << alignment_
              << ", max_bucket_count=" << max_bucket_count_
              << ", meta_log_threshold=" << log_compaction_threshold_
              << ", recovered_buckets=" << buckets_.size()
              << ", recovered_replicas=" << recovered_replicas_.size()
              << ", compacted_buckets=" << compacted;
    return {};
}

PersistedBucketMetadata BucketGlobalAllocator::SnapshotLocked(
    BucketState& bucket, bool evicting) {
    PersistedBucketMetadata snapshot;
    snapshot.version = kBucketMetadataVersion;
    snapshot.bucket_id = bucket.bucket_id;
    snapshot.bucket_generation = bucket.generation;
    snapshot.capacity = bucket.capacity;
    snapshot.alignment = alignment_;
    snapshot.append_offset = bucket.append_offset;
    // Every sequence number issued so far is either already in this snapshot's
    // in-memory state or belongs to another bucket, so the snapshot supersedes
    // all of them for this bucket. Recording the global watermark (rather than
    // this bucket's last queued seq) is what lets compaction clear the log
    // without inspecting its contents.
    snapshot.log_seq = next_log_seq_ - 1;
    snapshot.snapshot_epoch =
        bucket.snapshot_epoch + 1;
    bucket.snapshot_epoch = snapshot.snapshot_epoch;
    snapshot.evicting = evicting;
    snapshot.entries.reserve(bucket.entries.size());
    for (const auto& [key, entry] : bucket.entries) {
        PersistedBucketEntry persisted;
        persisted.key = key;
        persisted.entry_offset = entry.entry_offset;
        persisted.key_size = entry.key_size;
        persisted.value_size = entry.value_size;
        persisted.reserved_size = entry.reserved_size;
        persisted.generation = entry.generation;
        persisted.state = static_cast<int32_t>(entry.state);
        snapshot.entries.push_back(std::move(persisted));
    }
    // Deterministic order keeps the serialized bytes (and hence the checksum)
    // stable for identical logical state, which makes tests reproducible.
    std::sort(
        snapshot.entries.begin(), snapshot.entries.end(),
        [](const PersistedBucketEntry& lhs, const PersistedBucketEntry& rhs) {
            if (lhs.entry_offset != rhs.entry_offset) {
                return lhs.entry_offset < rhs.entry_offset;
            }
            return lhs.key < rhs.key;
        });
    snapshot.checksum = ComputeMetadataChecksum(snapshot);
    return snapshot;
}

uint64_t BucketGlobalAllocator::QueueLogRecordLocked(BucketState& bucket,
                                                     MetaLogOp op,
                                                     const std::string& key,
                                                     const BucketEntry& entry) {
    MetaLogRecord record;
    record.op = op;
    record.seq = next_log_seq_++;
    record.bucket_id = bucket.bucket_id;
    record.bucket_generation = bucket.generation;
    record.timestamp_ns = WallClockNs();
    record.entry_offset = entry.entry_offset;
    record.key_size = key.size();
    record.value_size = entry.value_size;
    record.reserved_size = entry.reserved_size;
    record.entry_generation = entry.generation;
    record.append_offset = bucket.append_offset;
    record.key = key;

    pending_log_.push_back(std::move(record));
    const uint64_t seq = pending_log_.back().seq;
    bucket.last_queued_seq = seq;
    return seq;
}

tl::expected<void, ErrorCode> BucketGlobalAllocator::PersistMetadata(
    const PersistedBucketMetadata& snapshot) {
    if (!fs_adapter_) {
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }

    std::string payload;
    try {
        struct_pb::to_pb(snapshot, payload);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to serialize bucket metadata, bucket_id="
                   << snapshot.bucket_id << ", error=" << e.what();
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    const int slot = static_cast<int>(snapshot.snapshot_epoch & 1u);
    const std::string slot_path =
        BucketMetaSlotPath(snapshot.bucket_id, slot);
    bool slot_existed = false;
    auto slot_exists_result = fs_adapter_->FileExists(slot_path);
    if (slot_exists_result) slot_existed = *slot_exists_result;

    // Rewrite only the inactive stable slot. The other slot remains a complete,
    // checksummed recovery point until this write and sync finish, so atomic
    // namespace replacement is unnecessary.
    auto write_result = fs_adapter_->WriteFile(
        slot_path, std::span<const char>(payload.data(), payload.size()));
    if (!write_result) {
        LOG(ERROR) << "Failed to write bucket metadata slot " << slot_path
                   << ", error=" << write_result.error();
        return tl::make_unexpected(write_result.error());
    }
    if (*write_result != payload.size()) {
        LOG(ERROR) << "Short write of bucket metadata slot " << slot_path
                   << ", expected=" << payload.size()
                   << ", actual=" << *write_result;
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }

    auto sync_file = fs_adapter_->SyncFile(slot_path);
    if (!sync_file) {
        LOG(ERROR) << "Failed to fsync bucket metadata slot " << slot_path
                   << ", error=" << sync_file.error();
        return tl::make_unexpected(sync_file.error());
    }

    // Creating a directory entry needs one directory sync. Rewriting an existing
    // slot changes no namespace metadata and deliberately avoids that DFS lock.
    if (!slot_existed) {
        auto sync_dir = fs_adapter_->SyncDirectory(fsdir_);
        if (!sync_dir) {
            LOG(ERROR) << "Failed to fsync DFS bucket directory " << fsdir_
                       << ", error=" << sync_dir.error();
            return tl::make_unexpected(sync_dir.error());
        }
    }
    return {};
}

void BucketGlobalAllocator::DeleteBucketFiles(int64_t bucket_id) {
    if (!fs_adapter_) return;
    {
        std::lock_guard<std::mutex> log_lock(log_mutex_);
        CloseLogLogLocked(bucket_id);
    }
    auto meta_result = fs_adapter_->DeleteFile(BucketMetaPath(bucket_id));
    if (!meta_result && meta_result.error() != ErrorCode::FILE_NOT_FOUND) {
        LOG(ERROR) << "Failed to delete legacy bucket metadata file for bucket_id="
                   << bucket_id << ", error=" << meta_result.error();
    }
    for (int slot = 0; slot < 2; ++slot) {
        auto slot_result =
            fs_adapter_->DeleteFile(BucketMetaSlotPath(bucket_id, slot));
        if (!slot_result && slot_result.error() != ErrorCode::FILE_NOT_FOUND) {
            LOG(ERROR) << "Failed to delete bucket metadata slot " << slot
                       << " for bucket_id=" << bucket_id
                       << ", error=" << slot_result.error();
        }
    }
    auto log_result = fs_adapter_->DeleteFile(BucketMetaLogPath(bucket_id));
    if (!log_result && log_result.error() != ErrorCode::FILE_NOT_FOUND) {
        LOG(ERROR) << "Failed to delete bucket metadata log for bucket_id="
                   << bucket_id << ", error=" << log_result.error();
    }
    auto data_result = fs_adapter_->DeleteFile(BucketDataPath(bucket_id));
    if (!data_result && data_result.error() != ErrorCode::FILE_NOT_FOUND) {
        LOG(ERROR) << "Failed to delete bucket data file for bucket_id="
                   << bucket_id << ", error=" << data_result.error();
    }
    // Sweep obsolete temp files left by the version-2 rename publication
    // protocol. Version 3 uses stable slots and never creates new temp files.
    const std::string temp_prefix =
        std::string(kBucketFilePrefix) + FormatBucketId(bucket_id) +
        kBucketMetaSuffix + ".tmp.";
    if (auto files = fs_adapter_->ListFiles(fsdir_)) {
        for (const auto& name : *files) {
            if (name.rfind(temp_prefix, 0) != 0) continue;
            (void)fs_adapter_->DeleteFile(fsdir_ + "/" + name);
        }
    }
}

// === metadata log I/O ===

tl::expected<int, ErrorCode> BucketGlobalAllocator::GetOrOpenLogLogLocked(
    int64_t bucket_id) {
    auto it = log_fds_.find(bucket_id);
    if (it != log_fds_.end()) return it->second;
    if (!fs_adapter_) {
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }

    const std::string path = BucketMetaLogPath(bucket_id);
    auto fd = fs_adapter_->OpenAppendFile(path);
    if (!fd) {
        LOG(ERROR) << "Failed to open DFS bucket metadata log " << path
                   << ", error=" << fd.error();
        return tl::make_unexpected(fd.error());
    }
    // The fd stays open for the bucket's lifetime, so the hot path never pays
    // an open(2) and O_APPEND keeps concurrent record writes from interleaving.
    log_fds_[bucket_id] = *fd;
    if (log_bytes_.find(bucket_id) == log_bytes_.end()) {
        auto size = fs_adapter_->GetFileSize(path);
        log_bytes_[bucket_id] = size ? static_cast<uint64_t>(*size) : 0;
    }
    return *fd;
}

void BucketGlobalAllocator::CloseLogLogLocked(int64_t bucket_id) {
    auto it = log_fds_.find(bucket_id);
    if (it != log_fds_.end()) {
        if (it->second >= 0 && fs_adapter_) {
            (void)fs_adapter_->CloseFile(it->second);
        }
        log_fds_.erase(it);
    }
    log_bytes_.erase(bucket_id);
    log_synced_seq_.erase(bucket_id);
}

tl::expected<void, ErrorCode>
BucketGlobalAllocator::DrainPendingLogLogLocked() {
    std::vector<MetaLogRecord> records;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        records.swap(pending_log_);
    }
    if (records.empty()) return {};

    // Group by bucket so each log is appended once and synced once, and keep
    // sequence order within a bucket so a replay that stops early still sees a
    // prefix of the history rather than an arbitrary subset.
    std::sort(records.begin(), records.end(),
              [](const MetaLogRecord& lhs, const MetaLogRecord& rhs) {
                  if (lhs.bucket_id != rhs.bucket_id) {
                      return lhs.bucket_id < rhs.bucket_id;
                  }
                  return lhs.seq < rhs.seq;
              });

    tl::expected<void, ErrorCode> outcome;
    std::vector<MetaLogRecord> failed;
    size_t index = 0;
    while (index < records.size()) {
        const int64_t bucket_id = records[index].bucket_id;
        size_t end = index;
        std::string payload;
        uint64_t highest_seq = 0;
        while (end < records.size() && records[end].bucket_id == bucket_id) {
            SerializeMetaLogRecord(records[end], payload);
            highest_seq = std::max(highest_seq, records[end].seq);
            ++end;
        }

        auto fail_group = [&](ErrorCode error) {
            // Put the records back so a later flush can retry them, and report
            // the first error to the caller.
            for (size_t i = index; i < end; ++i) {
                failed.push_back(std::move(records[i]));
            }
            if (outcome) outcome = tl::make_unexpected(error);
        };

        auto fd = GetOrOpenLogLogLocked(bucket_id);
        if (!fd) {
            fail_group(fd.error());
            index = end;
            continue;
        }

        auto appended = fs_adapter_->AppendData(
            *fd, std::span<const char>(payload.data(), payload.size()));
        if (!appended) {
            LOG(ERROR) << "Failed to append DFS bucket metadata log, bucket_id="
                       << bucket_id << ", error=" << appended.error();
            fail_group(appended.error());
            index = end;
            continue;
        }
        // A short append would leave a torn record behind. It cannot be undone,
        // but the per-record CRC makes replay drop it, and reporting the failure
        // keeps the caller from treating the operation as durable.
        if (*appended != payload.size()) {
            LOG(ERROR) << "Short append of DFS bucket metadata log, bucket_id="
                       << bucket_id << ", expected=" << payload.size()
                       << ", actual=" << *appended;
            log_bytes_[bucket_id] += *appended;
            fail_group(ErrorCode::FILE_WRITE_FAIL);
            index = end;
            continue;
        }

        auto synced = fs_adapter_->SyncFileData(*fd);
        if (!synced) {
            LOG(ERROR) << "Failed to fdatasync DFS bucket metadata log, "
                          "bucket_id="
                       << bucket_id << ", error=" << synced.error();
            log_bytes_[bucket_id] += *appended;
            fail_group(synced.error());
            index = end;
            continue;
        }

        log_bytes_[bucket_id] += payload.size();
        auto& synced_seq = log_synced_seq_[bucket_id];
        synced_seq = std::max(synced_seq, highest_seq);
        index = end;
    }

    if (!failed.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Retried records go back in front of anything queued since, so the
        // vector stays sorted by sequence number.
        failed.insert(failed.end(), pending_log_.begin(), pending_log_.end());
        pending_log_.swap(failed);
    }
    return outcome;
}

tl::expected<void, ErrorCode> BucketGlobalAllocator::FlushLog() {
    std::lock_guard<std::mutex> log_lock(log_mutex_);
    return DrainPendingLogLogLocked();
}

tl::expected<void, ErrorCode> BucketGlobalAllocator::SyncLogUpTo(
    int64_t bucket_id, uint64_t seq) {
    std::lock_guard<std::mutex> log_lock(log_mutex_);
    auto synced_it = log_synced_seq_.find(bucket_id);
    if (synced_it != log_synced_seq_.end() && synced_it->second >= seq) {
        // Another thread's flush already covered this record, including its
        // fdatasync, so there is nothing left to make durable.
        return {};
    }
    auto drained = DrainPendingLogLogLocked();
    if (!drained) return drained;

    synced_it = log_synced_seq_.find(bucket_id);
    if (synced_it == log_synced_seq_.end() || synced_it->second < seq) {
        // The drain reported success yet the sequence number is still not
        // durable, which means the record was never queued for this bucket.
        // Treat it as a write failure rather than claiming durability.
        LOG(ERROR) << "DFS bucket metadata log sequence " << seq
                   << " is not durable for bucket_id=" << bucket_id;
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return {};
}

bool BucketGlobalAllocator::ShouldCompact(int64_t bucket_id) const {
    uint64_t log_bytes = 0;
    {
        std::lock_guard<std::mutex> log_lock(log_mutex_);
        auto it = log_bytes_.find(bucket_id);
        if (it != log_bytes_.end()) log_bytes = it->second;
    }
    if (log_compaction_threshold_ > 0 && log_bytes >= log_compaction_threshold_) {
        return true;
    }
    if (log_bytes == 0) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buckets_.find(bucket_id);
    if (it == buckets_.end()) return false;
    const auto& bucket = *it->second;
    // More than half the entries dead means the snapshot is mostly wasted
    // bytes; folding now keeps both the snapshot and the log small.
    return bucket.tombstones * 2 > bucket.entries.size();
}

void BucketGlobalAllocator::MaybeCompact(int64_t bucket_id) {
    if (!ShouldCompact(bucket_id)) return;
    auto compacted = CompactBucket(bucket_id);
    if (!compacted) {
        // Not fatal: the log still holds every delta, so the bucket stays
        // correct and the next trigger retries.
        LOG(WARNING) << "Failed to compact DFS bucket metadata, bucket_id="
                     << bucket_id << ", error=" << compacted.error();
    }
}

tl::expected<void, ErrorCode> BucketGlobalAllocator::CompactBucket(
    int64_t bucket_id) {
    if (!fs_adapter_) {
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }

    // Hold `log_mutex_` across the whole compaction. That is what makes the
    // truncation safe: no record can be appended between taking the snapshot
    // and clearing the log, so everything the log holds is either in the
    // snapshot or was already superseded by it.
    std::lock_guard<std::mutex> log_lock(log_mutex_);

    // Anything still queued has to reach the log first, otherwise the snapshot
    // would claim (via its log_seq watermark) to cover records that are neither
    // on disk nor in the snapshot's own in-memory source of truth.
    auto drained = DrainPendingLogLogLocked();
    if (!drained) return drained;

    PersistedBucketMetadata snapshot;
    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = buckets_.find(bucket_id);
        if (it == buckets_.end()) {
            // Evicted meanwhile; its files are gone or going.
            return {};
        }
        generation = it->second->generation;
        snapshot = SnapshotLocked(*it->second, /*evicting=*/false);
    }

    auto persisted = PersistMetadata(snapshot);
    if (!persisted) return persisted;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = buckets_.find(bucket_id);
        if (it == buckets_.end() || it->second->generation != generation) {
            // The bucket changed identity while we published; the new owner
            // will publish its own snapshot. Leave the log alone: it may
            // already carry the new generation's records.
            return {};
        }
    }

    // The snapshot is durable and covers every sequence number issued so far,
    // so the log can start over. Truncating through the live fd keeps it valid
    // and avoids a rename.
    auto fd = GetOrOpenLogLogLocked(bucket_id);
    if (!fd) return tl::make_unexpected(fd.error());
    auto truncated = fs_adapter_->TruncateFile(*fd, 0);
    if (!truncated) {
        LOG(ERROR) << "Failed to truncate DFS bucket metadata log, bucket_id="
                   << bucket_id << ", error=" << truncated.error();
        return truncated;
    }
    // Make the truncation itself durable, so a crash cannot resurrect records
    // the snapshot already absorbed. Replaying them would be harmless (they are
    // superseded by log_seq) but the log would never actually shrink.
    auto synced = fs_adapter_->SyncFileData(*fd);
    if (!synced) {
        LOG(ERROR) << "Failed to fdatasync truncated DFS bucket metadata log, "
                      "bucket_id="
                   << bucket_id << ", error=" << synced.error();
        return synced;
    }
    log_bytes_[bucket_id] = 0;
    auto& synced_seq = log_synced_seq_[bucket_id];
    synced_seq = std::max(synced_seq, snapshot.log_seq);
    return {};
}

size_t BucketGlobalAllocator::CompactAllBuckets() {
    std::vector<int64_t> bucket_ids;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        bucket_ids.reserve(buckets_.size());
        for (const auto& [bucket_id, bucket] : buckets_) {
            (void)bucket;
            bucket_ids.push_back(bucket_id);
        }
    }
    // Deterministic order keeps concurrent compactions from deadlocking on the
    // filesystem and makes test output stable.
    std::sort(bucket_ids.begin(), bucket_ids.end());

    size_t compacted = 0;
    for (const int64_t bucket_id : bucket_ids) {
        auto result = CompactBucket(bucket_id);
        if (!result) {
            LOG(WARNING) << "Failed to compact DFS bucket metadata, bucket_id="
                         << bucket_id << ", error=" << result.error();
            continue;
        }
        ++compacted;
    }
    return compacted;
}

uint64_t BucketGlobalAllocator::GetLogBytes(int64_t bucket_id) const {
    std::lock_guard<std::mutex> log_lock(log_mutex_);
    auto it = log_bytes_.find(bucket_id);
    return it == log_bytes_.end() ? 0 : it->second;
}

// === bucket lifecycle ===

tl::expected<BucketGlobalAllocator::BucketPtr, ErrorCode>
BucketGlobalAllocator::CreateBucketUnlocked(std::unique_lock<std::mutex>& lock) {
    if (max_bucket_count_ > 0 &&
        static_cast<int64_t>(buckets_.size()) >= max_bucket_count_) {
        return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
    }
    if (next_bucket_id_ > kMaxBucketId) {
        LOG(ERROR) << "DFS bucket id space exhausted, next_bucket_id="
                   << next_bucket_id_;
        return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
    }

    // Only one thread may be creating a bucket at a time. Without this, N
    // threads that all find the active bucket full each reserve a *different*
    // id, then race in the unlocked I/O section below: the last one to relock
    // wins `active_bucket_id_`, and the losers' buckets are published but
    // orphaned. Worse, each loser's rollback path can delete files belonging to
    // an id another thread has already published, which is what made concurrent
    // Allocate() fail with FILE_NOT_FOUND.
    //
    // EnsureActiveBucket guarantees `bucket_creation_in_flight_` is false here:
    // it waits out an in-flight creation and re-checks the active bucket first,
    // so a waiter reuses the new bucket instead of creating a redundant one.
    bucket_creation_in_flight_ = true;
    // Clears the flag and wakes the next waiter on every exit path.
    struct CreationGuard {
        BucketGlobalAllocator* self;
        ~CreationGuard() {
            self->bucket_creation_in_flight_ = false;
            self->bucket_creation_cv_.notify_all();
        }
    } creation_guard{this};

    const int64_t bucket_id = next_bucket_id_++;
    const uint64_t generation = next_generation_++;

    auto bucket = std::make_shared<BucketState>();
    bucket->bucket_id = bucket_id;
    bucket->generation = generation;
    bucket->capacity = bucket_capacity_;
    bucket->append_offset = 0;
    bucket->live_bytes = 0;
    bucket->last_access_ns = NowNs();

    PersistedBucketMetadata snapshot =
        SnapshotLocked(*bucket, /*evicting=*/false);

    // Preallocation and the initial `.meta` write are DFS I/O, so drop the
    // allocator lock around them.
    lock.unlock();
    auto prealloc = fs_adapter_->PreallocateFile(BucketDataPath(bucket_id),
                                                 bucket_capacity_);
    tl::expected<void, ErrorCode> persisted;
    if (prealloc) {
        persisted = PersistMetadata(snapshot);
    }
    lock.lock();

    if (!prealloc || !persisted) {
        const ErrorCode error =
            !prealloc ? prealloc.error() : persisted.error();
        LOG(ERROR) << "Failed to create DFS bucket " << bucket_id
                   << ", error=" << error;
        // Roll back the id reservation when nothing else claimed it meanwhile,
        // and remove any partially created files.
        if (next_bucket_id_ == bucket_id + 1 && !buckets_.count(bucket_id)) {
            next_bucket_id_ = bucket_id;
        }
        lock.unlock();
        DeleteBucketFiles(bucket_id);
        lock.lock();
        return tl::make_unexpected(error);
    }

    // A concurrent creator may have published this id while we were unlocked.
    auto existing = buckets_.find(bucket_id);
    if (existing != buckets_.end()) {
        LOG(WARNING) << "DFS bucket " << bucket_id
                     << " was published concurrently; reusing existing state";
        return existing->second;
    }

    buckets_.emplace(bucket_id, bucket);
    active_bucket_id_ = bucket_id;
    TouchLruLocked(bucket_id, bucket->last_access_ns);
    return bucket;
}

tl::expected<BucketGlobalAllocator::BucketPtr, ErrorCode>
BucketGlobalAllocator::EnsureActiveBucket(std::unique_lock<std::mutex>& lock,
                                          uint64_t required) {
    if (required > bucket_capacity_) {
        // Refuse rather than spill across buckets: the caller asked for one
        // contiguous region and no bucket can ever satisfy it.
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    // Each iteration re-checks the active bucket, so a thread that waited for
    // someone else's creation reuses that bucket instead of adding another. The
    // bound keeps a pathological interleaving from looping forever; in practice
    // one wait plus one creation is the worst case.
    constexpr int kMaxAttempts = 8;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        if (active_bucket_id_ >= 0) {
            auto it = buckets_.find(active_bucket_id_);
            if (it != buckets_.end() && !it->second->frozen) {
                auto& bucket = *it->second;
                auto entry_start =
                    CheckedAlignUp(bucket.append_offset, alignment_);
                if (entry_start && *entry_start <= bucket.capacity &&
                    required <= bucket.capacity - *entry_start) {
                    return it->second;
                }
            }
        }
        if (bucket_creation_in_flight_) {
            // Someone else is already creating one. Wait for it and loop, so we
            // consume their bucket rather than creating a competing one.
            bucket_creation_cv_.wait(lock);
            continue;
        }
        auto created = CreateBucketUnlocked(lock);
        if (!created) return tl::make_unexpected(created.error());
        // Loop once more so the freshly created bucket goes through the same
        // capacity check instead of being trusted blindly.
    }
    return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
}

void BucketGlobalAllocator::TouchLruLocked(int64_t bucket_id, int64_t now_ns) {
    auto index_it = lru_index_.find(bucket_id);
    if (index_it != lru_index_.end()) {
        lru_list_.splice(lru_list_.begin(), lru_list_, index_it->second);
    } else {
        lru_list_.push_front(bucket_id);
        lru_index_[bucket_id] = lru_list_.begin();
    }
    auto bucket_it = buckets_.find(bucket_id);
    if (bucket_it != buckets_.end()) {
        bucket_it->second->last_access_ns = now_ns;
    }
}

void BucketGlobalAllocator::RemoveFromLruLocked(int64_t bucket_id) {
    auto index_it = lru_index_.find(bucket_id);
    if (index_it == lru_index_.end()) return;
    lru_list_.erase(index_it->second);
    lru_index_.erase(index_it);
}

tl::expected<DistributedFSDescriptor, ErrorCode>
BucketGlobalAllocator::ReserveInBucketLocked(BucketState& bucket,
                                             const std::string& key,
                                             uint64_t size,
                                             uint64_t* out_seq) {
    auto layout = ComputeBucketEntryLayout(bucket.append_offset, key.size(),
                                           size, alignment_);
    if (!layout) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (layout->entry_end() > bucket.capacity) {
        return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
    }

    BucketEntry entry;
    entry.entry_offset = layout->entry_start;
    entry.key_size = key.size();
    entry.value_size = size;
    entry.reserved_size = layout->reserved_size;
    entry.generation = next_generation_++;
    entry.state = BucketEntryState::PENDING;

    bucket.entries[key] = entry;
    bucket.append_offset = layout->entry_end();
    bucket.live_bytes += layout->reserved_size;
    key_index_[key] = bucket.bucket_id;

    // Queue the delta while the append offset it records is the one this
    // reservation produced; the append itself happens after the lock is
    // dropped.
    const uint64_t seq =
        QueueLogRecordLocked(bucket, MetaLogOp::ADD_PENDING, key, entry);
    if (out_seq) *out_seq = seq;

    return MakeBucketDescriptor(BucketDataPath(bucket.bucket_id), *layout, size,
                                bucket.bucket_id);
}

void BucketGlobalAllocator::UnreserveInBucketLocked(
    BucketState& bucket, const std::string& key,
    const DistributedFSDescriptor& descriptor) {
    auto entry_it = bucket.entries.find(key);
    if (entry_it == bucket.entries.end()) return;
    const auto& entry = entry_it->second;
    if (entry.value_size != descriptor.object_size ||
        entry.reserved_size != descriptor.aligned_size) {
        return;
    }

    const uint64_t entry_end = entry.entry_offset + entry.reserved_size;
    if (bucket.append_offset == entry_end) {
        // Only the most recent reservation can give its space back, which is
        // exactly how BatchAllocate unwinds (reverse order).
        bucket.append_offset = entry.entry_offset;
    }
    if (bucket.live_bytes >= entry.reserved_size) {
        bucket.live_bytes -= entry.reserved_size;
    } else {
        bucket.live_bytes = 0;
    }

    // The ADD_PENDING record for this reservation may already be queued or even
    // durable. Record a tombstone so the two cannot disagree: replay applies
    // both in sequence order and ends at TOMBSTONE, which recovery never
    // revives. Dropping the queued record instead would be wrong once it has
    // reached disk.
    BucketEntry rolled_back = entry;
    rolled_back.state = BucketEntryState::TOMBSTONE;
    QueueLogRecordLocked(bucket, MetaLogOp::TOMBSTONE, key, rolled_back);

    bucket.entries.erase(entry_it);

    auto index_it = key_index_.find(key);
    if (index_it != key_index_.end() && index_it->second == bucket.bucket_id) {
        key_index_.erase(index_it);
    }
}

BucketGlobalAllocator::BucketEntry*
BucketGlobalAllocator::FindMatchingEntryLocked(
    const std::string& key, const DistributedFSDescriptor& desc,
    BucketPtr* out_bucket) {
    if (desc.shard_idx < 0) return nullptr;
    const int64_t bucket_id = static_cast<int64_t>(desc.shard_idx);

    auto index_it = key_index_.find(key);
    if (index_it == key_index_.end() || index_it->second != bucket_id) {
        return nullptr;
    }
    auto bucket_it = buckets_.find(bucket_id);
    if (bucket_it == buckets_.end()) return nullptr;

    auto entry_it = bucket_it->second->entries.find(key);
    if (entry_it == bucket_it->second->entries.end()) return nullptr;

    // Match on every layout-defining field so a descriptor from a superseded
    // allocation cannot address the current one.
    auto& entry = entry_it->second;
    if (entry.value_size != desc.object_size ||
        entry.reserved_size != desc.aligned_size ||
        entry.key_size != key.size()) {
        return nullptr;
    }
    auto layout = RebuildBucketEntryLayout(entry.entry_offset, entry.key_size,
                                           entry.value_size, alignment_);
    if (!layout || layout->value_offset != desc.offset) return nullptr;

    if (out_bucket) *out_bucket = bucket_it->second;
    return &entry;
}

// === allocation ===

tl::expected<DistributedFSDescriptor, ErrorCode>
BucketGlobalAllocator::Allocate(const std::string& key, uint64_t size) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }
    if (key.empty() || size == 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    BatchAllocateRequest request{key, size};
    auto results = BatchAllocate({request});
    if (results.size() != 1) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    if (!results[0].success) {
        return tl::make_unexpected(results[0].error);
    }
    return results[0].descriptor;
}

std::vector<BatchAllocateResult> BucketGlobalAllocator::BatchAllocate(
    const std::vector<BatchAllocateRequest>& requests) {
    std::vector<BatchAllocateResult> results;
    results.reserve(requests.size());
    for (const auto& request : requests) {
        results.push_back(
            BatchAllocateResult{request.key, {}, false, ErrorCode::OK});
    }
    if (requests.empty()) return results;

    auto fail_all = [&results](ErrorCode error) {
        for (auto& result : results) {
            result.success = false;
            result.error = error;
            result.descriptor = DistributedFSDescriptor{};
        }
    };
    if (!initialized_.load(std::memory_order_acquire)) {
        fail_all(ErrorCode::DFS_SERVICE_UNAVAILABLE);
        return results;
    }

    // Validate request shape before changing allocator state. A batch may span
    // buckets, but one object must always fit in one bucket.
    std::vector<size_t> allocatable;
    allocatable.reserve(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        if (request.key.empty() || request.size == 0) {
            fail_all(ErrorCode::INVALID_PARAMS);
            return results;
        }
        for (size_t j = 0; j < i; ++j) {
            if (requests[j].key == request.key) {
                LOG(ERROR) << "Duplicate key " << request.key
                           << " in DFS batch allocate request";
                fail_all(ErrorCode::INVALID_PARAMS);
                return results;
            }
        }
        auto layout = ComputeBucketEntryLayout(0, request.key.size(),
                                               request.size, alignment_);
        if (!layout || layout->reserved_size > bucket_capacity_) {
            LOG(ERROR) << "DFS object for key " << request.key
                       << " exceeds bucket capacity, object_size="
                       << request.size << ", reserved_size="
                       << (layout ? layout->reserved_size : 0)
                       << ", bucket_capacity=" << bucket_capacity_;
            fail_all(ErrorCode::INVALID_PARAMS);
            return results;
        }
        allocatable.push_back(i);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    // Existing live keys retain their per-key OBJECT_ALREADY_EXISTS outcome;
    // the remaining requests are packed in their original order.
    allocatable.clear();
    for (size_t i = 0; i < requests.size(); ++i) {
        if (key_index_.count(requests[i].key) != 0) {
            LOG(WARNING) << "DFS batch allocate skipped key " << requests[i].key
                         << ": it already has a live allocation";
            results[i].error = ErrorCode::OBJECT_ALREADY_EXISTS;
            continue;
        }
        allocatable.push_back(i);
    }
    if (allocatable.empty()) return results;

    struct Reservation {
        size_t request_index = 0;
        int64_t bucket_id = -1;
        uint64_t generation = 0;
        uint64_t seq = 0;
    };
    std::vector<Reservation> reserved;
    reserved.reserve(allocatable.size());

    auto fail_allocatable = [&]() {
        for (const size_t index : allocatable) {
            results[index].success = false;
            results[index].descriptor = DistributedFSDescriptor{};
            if (results[index].error == ErrorCode::OK) {
                results[index].error = ErrorCode::NO_AVAILABLE_HANDLE;
            }
        }
    };

    // Append-pack in request order. EnsureActiveBucket receives only the
    // current object's reserved size, so a partially filled active bucket is
    // used before a new bucket is created. A bucket boundary can therefore
    // occur only between two objects, never inside one object.
    for (const size_t index : allocatable) {
        const auto& request = requests[index];
        auto object_layout = ComputeBucketEntryLayout(
            0, request.key.size(), request.size, alignment_);
        if (!object_layout) {
            fail_allocatable();
            break;
        }
        auto bucket_result = EnsureActiveBucket(lock,
                                                object_layout->reserved_size);
        if (!bucket_result) {
            fail_allocatable();
            results[index].error = bucket_result.error();
            break;
        }
        auto bucket = bucket_result.value();
        uint64_t seq = 0;
        auto descriptor = ReserveInBucketLocked(*bucket, request.key,
                                                 request.size, &seq);
        if (!descriptor) {
            fail_allocatable();
            results[index].error = descriptor.error();
            break;
        }
        results[index].descriptor = std::move(descriptor.value());
        results[index].success = true;
        results[index].error = ErrorCode::OK;
        reserved.push_back({index, bucket->bucket_id, bucket->generation, seq});
        TouchLruLocked(bucket->bucket_id, NowNs());
    }

    if (reserved.size() != allocatable.size()) {
        // Roll back in reverse reservation order, across every bucket touched
        // by this batch. Tombstone records make already-durable ADD_PENDING
        // records safe to replay on restart.
        for (auto it = reserved.rbegin(); it != reserved.rend(); ++it) {
            auto bucket_it = buckets_.find(it->bucket_id);
            if (bucket_it != buckets_.end() &&
                bucket_it->second->generation == it->generation) {
                UnreserveInBucketLocked(*bucket_it->second,
                                        requests[it->request_index].key,
                                        results[it->request_index].descriptor);
            }
        }
        lock.unlock();
        fail_allocatable();
        return results;
    }

    // The batch may touch multiple metadata logs. Every ADD_PENDING record in
    // every touched bucket must be fdatasync'd before the reservation is
    // exposed to the master.
    std::map<int64_t, std::pair<uint64_t, uint64_t>> bucket_sequences;
    for (const auto& item : reserved) {
        auto& state = bucket_sequences[item.bucket_id];
        state.first = item.generation;
        state.second = std::max(state.second, item.seq);
    }
    lock.unlock();

    for (const auto& [bucket_id, sequence] : bucket_sequences) {
        auto synced = SyncLogUpTo(bucket_id, sequence.second);
        if (synced) continue;

        LOG(ERROR) << "Failed to persist DFS metadata for multi-bucket batch, "
                   << "bucket_id=" << bucket_id << ", error=" << synced.error();
        lock.lock();
        for (auto it = reserved.rbegin(); it != reserved.rend(); ++it) {
            auto bucket_it = buckets_.find(it->bucket_id);
            if (bucket_it != buckets_.end() &&
                bucket_it->second->generation == it->generation) {
                UnreserveInBucketLocked(*bucket_it->second,
                                        requests[it->request_index].key,
                                        results[it->request_index].descriptor);
            }
        }
        lock.unlock();
        // Flush the compensating tombstones. If this second flush fails, the
        // tombstones remain queued and maintenance will retry them; recovery
        // still never exposes PENDING entries as committed objects.
        (void)FlushLog();
        fail_allocatable();
        return results;
    }

    return results;
}

bool BucketGlobalAllocator::MarkCommitted(
    const std::string& key, const DistributedFSDescriptor& descriptor) {
    if (!initialized_.load(std::memory_order_acquire)) return false;

    int64_t bucket_id = -1;
    uint64_t generation = 0;
    uint64_t seq = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        BucketPtr bucket;
        auto* entry = FindMatchingEntryLocked(key, descriptor, &bucket);
        if (entry == nullptr) return false;
        if (entry->state == BucketEntryState::COMMITTED) {
            // Idempotent: a duplicate PutEnd for the same generation succeeds.
            // The original transition is already durable, so there is nothing
            // to append.
            return true;
        }
        if (entry->state != BucketEntryState::PENDING) return false;
        entry->state = BucketEntryState::COMMITTED;
        bucket_id = bucket->bucket_id;
        generation = bucket->generation;
        seq = QueueLogRecordLocked(*bucket, MetaLogOp::MARK_COMMITTED, key,
                                   *entry);
    }

    auto synced = SyncLogUpTo(bucket_id, seq);
    if (!synced) {
        LOG(ERROR) << "Failed to persist DFS bucket commit for key=" << key
                   << ", bucket_id=" << bucket_id
                   << ", error=" << synced.error();
        // Roll the in-memory state back so the visible state keeps matching
        // what is durable: the caller then treats the write as failed. The
        // queued MARK_COMMITTED record stays queued, and a later flush that
        // succeeds would make it durable - which is why a compensating
        // TOMBSTONE is recorded instead of quietly dropping the transition.
        std::lock_guard<std::mutex> lock(mutex_);
        BucketPtr bucket;
        auto* entry = FindMatchingEntryLocked(key, descriptor, &bucket);
        if (entry != nullptr && bucket && bucket->generation == generation &&
            entry->state == BucketEntryState::COMMITTED) {
            entry->state = BucketEntryState::PENDING;
            BucketEntry rolled_back = *entry;
            rolled_back.state = BucketEntryState::TOMBSTONE;
            QueueLogRecordLocked(*bucket, MetaLogOp::TOMBSTONE, key,
                                 rolled_back);
        }
        return false;
    }

    // The maintenance thread observes the log size and compacts later; PutEnd
    // must never inherit a full-snapshot rewrite or DFS namespace operation.
    return true;
}

void BucketGlobalAllocator::Free(const std::string& key,
                                 const DistributedFSDescriptor& descriptor) {
    if (!initialized_.load(std::memory_order_acquire)) return;

    std::lock_guard<std::mutex> lock(mutex_);
    BucketPtr bucket;
    auto* entry = FindMatchingEntryLocked(key, descriptor, &bucket);
    if (entry == nullptr) {
        // Stale Free from a superseded generation: ignore it so it cannot
        // drop the allocation that replaced it.
        return;
    }
    if (!IsLive(entry->state)) return;

    // Buckets are append-only, so freeing leaves a tombstone rather than
    // reclaiming the middle of the file. Space comes back only when the
    // whole bucket is evicted.
    entry->state = BucketEntryState::TOMBSTONE;
    ++bucket->tombstones;
    if (bucket->live_bytes >= entry->reserved_size) {
        bucket->live_bytes -= entry->reserved_size;
    } else {
        bucket->live_bytes = 0;
    }
    auto index_it = key_index_.find(key);
    if (index_it != key_index_.end() && index_it->second == bucket->bucket_id) {
        key_index_.erase(index_it);
    }
    // The tombstone must reach disk so a restart cannot revive the key, but the
    // master calls Free() while holding a metadata shard lock, so appending and
    // syncing here would put I/O under that lock. Queue the record and let
    // FlushDirtyMetadata() write it.
    //
    // Losing the tombstone in a crash before the flush is safe: recovery only
    // revives COMMITTED entries, and the master's own metadata (which no longer
    // references the key) is the authority on visibility.
    QueueLogRecordLocked(*bucket, MetaLogOp::TOMBSTONE, key, *entry);
}

size_t BucketGlobalAllocator::FlushDirtyMetadata() {
    if (!initialized_.load(std::memory_order_acquire)) return 0;

    // Which buckets still have records that are queued but not durable. Read
    // before the flush, since the flush is what clears the gap.
    std::vector<std::pair<int64_t, uint64_t>> wanted;
    std::vector<int64_t> bucket_ids;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        bucket_ids.reserve(buckets_.size());
        for (const auto& [bucket_id, bucket] : buckets_) {
            bucket_ids.push_back(bucket_id);
            if (bucket->last_queued_seq == 0) continue;
            wanted.emplace_back(bucket_id, bucket->last_queued_seq);
        }
    }

    std::vector<std::pair<int64_t, uint64_t>> dirty;
    if (!wanted.empty()) {
        std::lock_guard<std::mutex> log_lock(log_mutex_);
        for (const auto& [bucket_id, queued_seq] : wanted) {
            auto it = log_synced_seq_.find(bucket_id);
            if (it != log_synced_seq_.end() && it->second >= queued_seq) {
                continue;
            }
            dirty.emplace_back(bucket_id, queued_seq);
        }
        if (!dirty.empty()) {
            auto drained = DrainPendingLogLogLocked();
            if (!drained) {
                LOG(ERROR) << "Failed to flush DFS bucket metadata log, error="
                           << drained.error();
            }
        }
    }

    size_t flushed = 0;
    {
        std::lock_guard<std::mutex> log_lock(log_mutex_);
        for (const auto& [bucket_id, queued_seq] : dirty) {
            auto it = log_synced_seq_.find(bucket_id);
            if (it != log_synced_seq_.end() && it->second >= queued_seq) {
                ++flushed;
            }
        }
    }

    // Hot paths may have already synced a log that crossed the threshold. Scan
    // every bucket here so compaction remains maintenance work rather than being
    // charged to the Put/Allocate request that happened to cross the limit.
    for (const int64_t bucket_id : bucket_ids) {
        MaybeCompact(bucket_id);
    }
    return flushed;
}

void BucketGlobalAllocator::UpdateAccess(
    const std::string& key, const DistributedFSDescriptor& descriptor) {
    if (!initialized_.load(std::memory_order_acquire)) return;

    std::lock_guard<std::mutex> lock(mutex_);
    BucketPtr bucket;
    auto* entry = FindMatchingEntryLocked(key, descriptor, &bucket);
    if (entry == nullptr || !IsLive(entry->state)) return;
    // A frozen bucket is mid-eviction; refreshing it would fight the
    // transaction the master is currently resolving.
    if (bucket->frozen) return;
    TouchLruLocked(bucket->bucket_id, NowNs());
}

uint64_t BucketGlobalAllocator::GetTotalCapacity() const {
    // A fixed denominator: watermarks must not move as buckets come and go,
    // or deleting a bucket would shrink capacity in lockstep with usage and
    // eviction would never converge.
    if (max_bucket_count_ > 0) {
        return static_cast<uint64_t>(max_bucket_count_) * bucket_capacity_;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<uint64_t>(buckets_.size()) * bucket_capacity_;
}

uint64_t BucketGlobalAllocator::UsedBytesLocked() const {
    // Physical reservation, not live bytes: an append-only bucket keeps its
    // whole reserved prefix until the bucket itself is evicted.
    uint64_t used = 0;
    for (const auto& [bucket_id, bucket] : buckets_) {
        (void)bucket_id;
        used += bucket->append_offset;
    }
    return used;
}

uint64_t BucketGlobalAllocator::GetUsedBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return UsedBytesLocked();
}

size_t BucketGlobalAllocator::GetBucketCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buckets_.size();
}

std::optional<int64_t> BucketGlobalAllocator::GetBucketIdForKey(
    const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = key_index_.find(key);
    if (it == key_index_.end()) return std::nullopt;
    return it->second;
}

std::vector<BucketGlobalAllocator::RecoveredReplica>
BucketGlobalAllocator::TakeRecoveredReplicas() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::move(recovered_replicas_);
}

// === eviction ===

BucketGlobalAllocator::PendingEviction
BucketGlobalAllocator::PrepareEviction() {
    return PrepareEvictionInternal(/*force_one=*/false);
}

BucketGlobalAllocator::PendingEviction
BucketGlobalAllocator::PrepareEvictionForAllocationFailure() {
    return PrepareEvictionInternal(/*force_one=*/true);
}

BucketGlobalAllocator::PendingEviction
BucketGlobalAllocator::PrepareEvictionInternal(bool force_one) {
    PendingEviction pending;
    if (!initialized_.load(std::memory_order_acquire)) return pending;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        const uint64_t capacity =
            max_bucket_count_ > 0
                ? static_cast<uint64_t>(max_bucket_count_) * bucket_capacity_
                : static_cast<uint64_t>(buckets_.size()) * bucket_capacity_;
        if (capacity == 0) return pending;

        const double usage = static_cast<double>(UsedBytesLocked()) /
                             static_cast<double>(capacity);
        if (!force_one) {
            if (usage >= eviction_high_watermark_) {
                eviction_active_ = true;
            }
            if (!eviction_active_) return pending;
            if (usage < eviction_low_watermark_) {
                eviction_active_ = false;
                return pending;
            }
        }

        // Walk the LRU from the cold end and take the first bucket that is
        // neither active nor already frozen.
        BucketPtr victim;
        for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
            const int64_t bucket_id = *it;
            if (bucket_id == active_bucket_id_) continue;
            auto bucket_it = buckets_.find(bucket_id);
            if (bucket_it == buckets_.end()) continue;
            if (bucket_it->second->frozen) continue;
            victim = bucket_it->second;
            break;
        }
        if (!victim) return pending;

        std::vector<GlobalAllocatorInterface::EvictionCandidate> candidates;
        for (const auto& [key, entry] : victim->entries) {
            // Tombstoned entries are already gone from the master's view; only
            // live entries need validating.
            if (!IsLive(entry.state)) continue;
            auto layout = RebuildBucketEntryLayout(
                entry.entry_offset, entry.key_size, entry.value_size,
                alignment_);
            if (!layout) {
                LOG(ERROR) << "Skipping DFS eviction of bucket "
                           << victim->bucket_id << ": entry for key " << key
                           << " has an inconsistent layout";
                return pending;
            }
            GlobalAllocatorInterface::EvictionCandidate candidate;
            candidate.key = key;
            candidate.shard_idx = static_cast<int>(victim->bucket_id);
            candidate.offset = layout->value_offset;
            // Byte-identical to what Allocate handed out, so the master can
            // match replica metadata field by field.
            candidate.descriptor = MakeBucketDescriptor(
                BucketDataPath(victim->bucket_id), *layout, entry.value_size,
                victim->bucket_id);
            candidates.push_back(std::move(candidate));
        }

        victim->frozen = true;
        RemoveFromLruLocked(victim->bucket_id);

        pending.owner_ = this;
        pending.bucket_id_ = victim->bucket_id;
        pending.bucket_generation_ = victim->generation;
        pending.candidates_ = std::move(candidates);
    }

    // Fold the log into a snapshot now that the bucket is frozen. The master
    // validates the candidates against its own metadata next, and if it aborts
    // the transaction the bucket returns to service with a clean, compact
    // snapshot rather than a log that keeps growing across rounds.
    auto compacted = CompactBucket(pending.bucket_id_);
    if (!compacted) {
        LOG(WARNING) << "Failed to compact DFS bucket metadata before eviction,"
                        " bucket_id="
                     << pending.bucket_id_ << ", error=" << compacted.error();
    }
    return pending;
}

void BucketGlobalAllocator::CommitEviction(PendingEviction&& pending) {
    // Detach first so the destructor of `pending` cannot abort what we commit.
    auto* owner = std::exchange(pending.owner_, nullptr);
    if (owner != this) return;

    const int64_t bucket_id = pending.bucket_id_;
    const uint64_t generation = pending.bucket_generation_;
    pending.candidates_.clear();
    if (bucket_id < 0) return;

    PersistedBucketMetadata marker;
    bool have_marker = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = buckets_.find(bucket_id);
        if (it == buckets_.end() || it->second->generation != generation) {
            return;
        }
        marker = SnapshotLocked(*it->second, /*evicting=*/true);
        have_marker = true;
    }

    // Publish a durable "this bucket is being evicted" marker before deleting
    // anything. If we crash between the marker and the deletes, recovery sees
    // the marker and treats the bucket as gone instead of resurrecting entries
    // whose data file may already be missing. The marker supersedes the log by
    // construction (its log_seq covers every record issued so far), so a
    // surviving log cannot revive the bucket either.
    if (have_marker) {
        auto persisted = PersistMetadata(marker);
        if (!persisted) {
            LOG(ERROR) << "Failed to persist DFS eviction marker for bucket "
                       << bucket_id << ", error=" << persisted.error()
                       << "; the bucket is already invisible to readers and "
                          "will be reclaimed on a later attempt";
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = buckets_.find(bucket_id);
        if (it == buckets_.end() || it->second->generation != generation) {
            return;
        }
        for (const auto& [key, entry] : it->second->entries) {
            if (!IsLive(entry.state)) continue;
            auto index_it = key_index_.find(key);
            if (index_it != key_index_.end() && index_it->second == bucket_id) {
                key_index_.erase(index_it);
            }
        }
        buckets_.erase(it);
        RemoveFromLruLocked(bucket_id);
        if (active_bucket_id_ == bucket_id) active_bucket_id_ = -1;
    }

    // Drop any records still queued for this bucket: their target is gone, and
    // appending them would recreate a log file for a bucket that no longer
    // exists.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_log_.erase(
            std::remove_if(pending_log_.begin(), pending_log_.end(),
                           [bucket_id](const MetaLogRecord& record) {
                               return record.bucket_id == bucket_id;
                           }),
            pending_log_.end());
    }

    // File deletion happens last and outside the lock. A failure here leaks
    // space but can no longer produce a dangling read, because the master has
    // already dropped every replica in this bucket.
    DeleteBucketFiles(bucket_id);
    LOG(INFO) << "Evicted DFS bucket " << bucket_id;
}

void BucketGlobalAllocator::AbortEviction(PendingEviction&& pending) {
    AbortEviction(std::move(pending), /*demote=*/true);
}

void BucketGlobalAllocator::AbortEviction(PendingEviction&& pending,
                                          bool demote) {
    auto* owner = std::exchange(pending.owner_, nullptr);
    if (owner != this) return;

    const int64_t bucket_id = pending.bucket_id_;
    const uint64_t generation = pending.bucket_generation_;
    pending.candidates_.clear();
    if (bucket_id < 0) return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buckets_.find(bucket_id);
    if (it == buckets_.end() || it->second->generation != generation) return;
    it->second->frozen = false;
    if (lru_index_.find(bucket_id) != lru_index_.end()) return;

    if (demote) {
        // The master actively rejected this bucket, so put it back at the warm
        // end: PrepareEviction scans from the cold end, and reinserting there
        // would hand back the same rejected bucket on every subsequent round,
        // never letting the scan reach another candidate.
        lru_list_.push_front(bucket_id);
        lru_index_[bucket_id] = lru_list_.begin();
    } else {
        // The transaction was dropped without a verdict (destructor or move
        // assignment). Nothing judged the bucket, so restore its original cold
        // position and let the next round reconsider it.
        lru_list_.push_back(bucket_id);
        lru_index_[bucket_id] = std::prev(lru_list_.end());
    }
}

// === recovery ===

void BucketGlobalAllocator::ReplayLogForRecovery(BucketState& bucket,
                                                 uint64_t snapshot_log_seq,
                                                 uint64_t& max_seq,
                                                 uint64_t& max_generation) {
    const std::string log_path = BucketMetaLogPath(bucket.bucket_id);
    auto file_size = fs_adapter_->GetFileSize(log_path);
    if (!file_size) {
        if (file_size.error() != ErrorCode::FILE_NOT_FOUND) {
            LOG(ERROR) << "Cannot stat DFS bucket metadata log " << log_path
                       << ", error=" << file_size.error()
                       << "; continuing from the snapshot alone";
        }
        return;
    }
    if (*file_size == 0) return;

    std::string payload(*file_size, '\0');
    auto read = fs_adapter_->ReadFile(log_path, payload.data(), payload.size());
    if (!read) {
        LOG(ERROR) << "Cannot read DFS bucket metadata log " << log_path
                   << ", error=" << read.error()
                   << "; continuing from the snapshot alone";
        return;
    }
    payload.resize(*read);

    // Collect first, then apply in sequence order. Records for one bucket are
    // appended in order, but the log can hold records written by different
    // flushes, and applying out of order would let an older state win.
    std::vector<MetaLogRecord> records;
    size_t pos = 0;
    size_t bad_records = 0;
    while (pos < payload.size()) {
        size_t consumed = 0;
        auto record = DeserializeMetaLogRecord(payload, pos, consumed);
        if (record) {
            pos += consumed;
            if (record->bucket_id != bucket.bucket_id) {
                // Not ours: the log name is authoritative, so a foreign record
                // means the file was corrupted or mixed up. Ignore it.
                ++bad_records;
                continue;
            }
            records.push_back(std::move(*record));
            continue;
        }
        if (consumed > 0) {
            // Framing was fine, the body was not. Skip exactly this record.
            ++bad_records;
            pos += consumed;
            continue;
        }
        // Unusable framing: either a torn tail (normal after a crash) or
        // corruption mid-log. Resynchronize on the next magic; if there is
        // none, the rest of the file is unusable.
        const size_t next = FindNextMagic(payload, pos + 1);
        if (next == std::string_view::npos) {
            if (pos + kMetaLogHeaderSize <= payload.size()) ++bad_records;
            break;
        }
        ++bad_records;
        pos = next;
    }

    std::sort(records.begin(), records.end(),
              [](const MetaLogRecord& lhs, const MetaLogRecord& rhs) {
                  return lhs.seq < rhs.seq;
              });

    size_t applied = 0;
    size_t skipped_stale = 0;
    for (const auto& record : records) {
        max_seq = std::max(max_seq, record.seq);
        max_generation = std::max(max_generation, record.entry_generation + 1);
        max_generation = std::max(max_generation, record.bucket_generation + 1);

        if (record.seq <= snapshot_log_seq) {
            // The snapshot already accounts for this delta.
            ++skipped_stale;
            continue;
        }
        if (record.bucket_generation != bucket.generation) {
            // A record from a previous incarnation of this bucket id.
            ++skipped_stale;
            continue;
        }
        if (record.key.empty() || record.key_size != record.key.size()) {
            continue;
        }

        auto layout = RebuildBucketEntryLayout(record.entry_offset,
                                               record.key_size,
                                               record.value_size, alignment_);
        if (!layout || layout->reserved_size != record.reserved_size ||
            layout->entry_end() > bucket.capacity) {
            LOG(ERROR) << "Ignoring DFS metadata log record for key "
                       << record.key << " in bucket " << bucket.bucket_id
                       << ": out-of-range or inconsistent layout";
            continue;
        }

        // The append cursor only ever moves forward: a record's append_offset
        // covers everything reserved up to and including itself, so taking the
        // maximum can never place the cursor inside live data.
        bucket.append_offset = std::max(bucket.append_offset,
                                        std::max(record.append_offset,
                                                 layout->entry_end()));

        BucketEntry entry;
        entry.entry_offset = record.entry_offset;
        entry.key_size = record.key_size;
        entry.value_size = record.value_size;
        entry.reserved_size = record.reserved_size;
        entry.generation = record.entry_generation;
        switch (record.op) {
            case MetaLogOp::ADD_PENDING:
                entry.state = BucketEntryState::PENDING;
                break;
            case MetaLogOp::MARK_COMMITTED:
                entry.state = BucketEntryState::COMMITTED;
                break;
            case MetaLogOp::TOMBSTONE:
                entry.state = BucketEntryState::TOMBSTONE;
                break;
        }

        auto existing = bucket.entries.find(record.key);
        if (existing != bucket.entries.end() &&
            existing->second.generation > entry.generation) {
            // A newer allocation already owns this key; an older record must
            // not overwrite it. Sequence order normally prevents this, but a
            // reordered or duplicated log makes the check worth keeping.
            ++skipped_stale;
            continue;
        }
        bucket.entries[record.key] = entry;
        ++applied;
    }

    if (bad_records > 0 || applied > 0) {
        LOG(INFO) << "Replayed DFS bucket metadata log for bucket "
                  << bucket.bucket_id << ": applied=" << applied
                  << ", superseded=" << skipped_stale
                  << ", damaged=" << bad_records
                  << ", snapshot_log_seq=" << snapshot_log_seq;
    }
}

tl::expected<void, ErrorCode> BucketGlobalAllocator::RecoverFromDisk() {
    auto files = fs_adapter_->ListFiles(fsdir_);
    if (!files) {
        if (files.error() == ErrorCode::FILE_NOT_FOUND) return {};
        LOG(ERROR) << "Failed to list DFS bucket directory " << fsdir_
                   << ", error=" << files.error();
        return tl::make_unexpected(files.error());
    }

    std::vector<int64_t> meta_ids;
    std::vector<int64_t> data_ids;
    std::vector<int64_t> log_ids;
    std::unordered_map<int64_t, std::vector<std::string>> snapshot_paths;
    for (const auto& name : *files) {
        // Order matters: slot and log suffixes include the legacy `.meta`
        // stem, so recognize them before the single-file version-2 snapshot.
        if (auto log_id = ParseBucketFileName(name, kBucketMetaLogSuffix)) {
            log_ids.push_back(*log_id);
        } else if (name.size() > 2 &&
                   (name.ends_with(".meta.0") || name.ends_with(".meta.1"))) {
            const std::string base = name.substr(0, name.size() - 2);
            if (auto slot_id = ParseBucketFileName(base, kBucketMetaSuffix)) {
                meta_ids.push_back(*slot_id);
                snapshot_paths[*slot_id].push_back(fsdir_ + "/" + name);
            }
        } else if (auto id = ParseBucketFileName(name, kBucketMetaSuffix)) {
            meta_ids.push_back(*id);
            snapshot_paths[*id].push_back(fsdir_ + "/" + name);
        } else if (auto data_id =
                       ParseBucketFileName(name, kBucketDataSuffix)) {
            data_ids.push_back(*data_id);
        } else if (name.find(".tmp.") != std::string::npos) {
            // Remove leftovers from the old rename-based publication protocol.
            (void)fs_adapter_->DeleteFile(fsdir_ + "/" + name);
        }
    }
    std::sort(meta_ids.begin(), meta_ids.end());
    meta_ids.erase(std::unique(meta_ids.begin(), meta_ids.end()), meta_ids.end());
    std::sort(log_ids.begin(), log_ids.end());

    int64_t max_seen_id = -1;
    uint64_t max_generation = 0;
    uint64_t max_log_seq = 0;
    // key -> (generation, bucket_id): resolves the same key appearing in more
    // than one bucket by keeping the newest committed generation.
    std::unordered_map<std::string, std::pair<uint64_t, int64_t>> winners;

    for (const int64_t bucket_id : meta_ids) {
        max_seen_id = std::max(max_seen_id, bucket_id);

        // Select the newest complete, checksummed slot. The legacy single
        // `.meta` snapshot has epoch zero and is considered only when no newer
        // version-3 slot is valid.
        std::optional<PersistedBucketMetadata> selected;
        const auto paths_it = snapshot_paths.find(bucket_id);
        if (paths_it != snapshot_paths.end()) {
            for (const auto& meta_path : paths_it->second) {
                auto file_size = fs_adapter_->GetFileSize(meta_path);
                if (!file_size) continue;
                std::string payload(*file_size, '\0');
                if (*file_size > 0) {
                    auto read = fs_adapter_->ReadFile(meta_path, payload.data(),
                                                      payload.size());
                    if (!read || *read != payload.size()) continue;
                }

                PersistedBucketMetadata candidate;
                bool valid = false;
                try {
                    struct_pb::from_pb(candidate, payload);
                    valid = candidate.version == kBucketMetadataVersion &&
                            ComputeMetadataChecksum(candidate) ==
                                candidate.checksum;
                } catch (...) {
                    valid = false;
                }

                if (!valid && meta_path == BucketMetaPath(bucket_id)) {
                    LegacyPersistedBucketMetadata legacy;
                    try {
                        struct_pb::from_pb(legacy, payload);
                        if (legacy.version == kLegacyBucketMetadataVersion &&
                            ComputeLegacyMetadataChecksum(legacy) ==
                                legacy.checksum) {
                            candidate = UpgradeLegacyMetadata(legacy);
                            valid = true;
                        }
                    } catch (...) {
                        valid = false;
                    }
                }

                if (!valid || candidate.bucket_id != bucket_id) {
                    LOG(WARNING) << "Ignoring invalid DFS bucket snapshot "
                                 << meta_path;
                    continue;
                }
                if (!selected || candidate.snapshot_epoch >
                                     selected->snapshot_epoch) {
                    selected = std::move(candidate);
                }
            }
        }
        if (!selected) {
            LOG(ERROR) << "Quarantining DFS bucket " << bucket_id
                       << ": no valid metadata snapshot";
            continue;
        }
        PersistedBucketMetadata snapshot = std::move(*selected);

        if (snapshot.alignment != alignment_) {
            LOG(ERROR) << "Quarantining DFS bucket " << bucket_id
                       << ": metadata alignment " << snapshot.alignment
                       << " does not match configured " << alignment_;
            continue;
        }
        if (snapshot.capacity == 0 || snapshot.capacity > bucket_capacity_) {
            LOG(ERROR) << "Quarantining DFS bucket " << bucket_id
                       << ": metadata capacity " << snapshot.capacity
                       << " is incompatible with configured "
                       << bucket_capacity_;
            continue;
        }
        if (snapshot.append_offset > snapshot.capacity) {
            LOG(ERROR) << "Quarantining DFS bucket " << bucket_id
                       << ": append_offset " << snapshot.append_offset
                       << " exceeds capacity " << snapshot.capacity;
            continue;
        }
        max_log_seq = std::max(max_log_seq, snapshot.log_seq);
        if (snapshot.evicting) {
            // The marker says the data file was being deleted; finish the job
            // rather than exposing entries whose data may already be gone.
            LOG(INFO) << "Completing interrupted eviction of DFS bucket "
                      << bucket_id;
            max_generation =
                std::max(max_generation, snapshot.bucket_generation + 1);
            DeleteBucketFiles(bucket_id);
            continue;
        }

        auto data_size = fs_adapter_->GetFileSize(BucketDataPath(bucket_id));
        if (!data_size) {
            LOG(ERROR) << "Quarantining DFS bucket " << bucket_id
                       << ": data file is missing or unreadable";
            continue;
        }

        auto bucket = std::make_shared<BucketState>();
        bucket->bucket_id = bucket_id;
        bucket->generation = snapshot.bucket_generation;
        bucket->capacity = snapshot.capacity;
        bucket->append_offset = 0;
        bucket->live_bytes = 0;
        bucket->last_access_ns = NowNs();

        bool bucket_ok = true;
        for (const auto& persisted : snapshot.entries) {
            max_generation = std::max(max_generation, persisted.generation + 1);
            if (!IsKnownEntryState(persisted.state)) {
                LOG(ERROR) << "Quarantining DFS bucket " << bucket_id
                           << ": entry for key " << persisted.key
                           << " has unknown state " << persisted.state;
                bucket_ok = false;
                break;
            }
            if (persisted.key.empty() ||
                persisted.key_size != persisted.key.size()) {
                LOG(ERROR) << "Quarantining DFS bucket " << bucket_id
                           << ": entry key size mismatch";
                bucket_ok = false;
                break;
            }
            auto layout = RebuildBucketEntryLayout(persisted.entry_offset,
                                                   persisted.key_size,
                                                   persisted.value_size,
                                                   alignment_);
            if (!layout || layout->reserved_size != persisted.reserved_size ||
                layout->entry_end() > snapshot.capacity) {
                LOG(ERROR) << "Quarantining DFS bucket " << bucket_id
                           << ": entry for key " << persisted.key
                           << " has an out-of-range or inconsistent layout";
                bucket_ok = false;
                break;
            }

            BucketEntry entry;
            entry.entry_offset = persisted.entry_offset;
            entry.key_size = persisted.key_size;
            entry.value_size = persisted.value_size;
            entry.reserved_size = persisted.reserved_size;
            entry.generation = persisted.generation;
            entry.state = static_cast<BucketEntryState>(persisted.state);

            // Keep the space reserved even for dead entries: the offsets of
            // later entries depend on it.
            bucket->append_offset =
                std::max(bucket->append_offset, layout->entry_end());
            bucket->entries[persisted.key] = entry;
        }
        if (!bucket_ok) continue;

        // The persisted append_offset is authoritative when it is at least as
        // large as what the entries imply (it also covers rolled-back space).
        bucket->append_offset =
            std::max(bucket->append_offset, snapshot.append_offset);

        // Fold in whatever the log recorded after the snapshot was published.
        ReplayLogForRecovery(*bucket, snapshot.log_seq, max_log_seq,
                             max_generation);
        if (bucket->append_offset > bucket->capacity) {
            LOG(ERROR) << "Quarantining DFS bucket " << bucket_id
                       << ": replayed append_offset " << bucket->append_offset
                       << " exceeds capacity " << bucket->capacity;
            continue;
        }

        // Settle the final state of every entry now that the snapshot and the
        // log have both been applied. A PENDING entry's data was never
        // confirmed durable, so it must not come back as readable; only
        // COMMITTED entries survive.
        for (auto& [key, entry] : bucket->entries) {
            if (entry.state == BucketEntryState::COMMITTED) {
                auto layout = RebuildBucketEntryLayout(entry.entry_offset,
                                                       entry.key_size,
                                                       entry.value_size,
                                                       alignment_);
                if (!layout ||
                    layout->entry_end() > static_cast<uint64_t>(*data_size)) {
                    LOG(ERROR) << "Quarantining committed DFS entry for key "
                               << key << " in bucket " << bucket_id
                               << ": it extends past the data file end";
                    entry.state = BucketEntryState::TOMBSTONE;
                    ++bucket->tombstones;
                    continue;
                }
                bucket->live_bytes += entry.reserved_size;
                continue;
            }
            entry.state = BucketEntryState::TOMBSTONE;
            ++bucket->tombstones;
        }

        max_generation = std::max(max_generation, bucket->generation + 1);

        for (const auto& [key, entry] : bucket->entries) {
            if (entry.state != BucketEntryState::COMMITTED) continue;
            auto winner_it = winners.find(key);
            if (winner_it == winners.end()) {
                winners[key] = {entry.generation, bucket_id};
            } else if (entry.generation > winner_it->second.first) {
                LOG(WARNING) << "DFS key " << key << " found in buckets "
                             << winner_it->second.second << " and " << bucket_id
                             << "; keeping the newer generation";
                winner_it->second = {entry.generation, bucket_id};
            } else {
                LOG(WARNING) << "DFS key " << key << " in bucket " << bucket_id
                             << " is superseded by bucket "
                             << winner_it->second.second;
            }
        }

        buckets_.emplace(bucket_id, std::move(bucket));
    }

    // Drop entries that lost the duplicate-key race, then index the winners.
    for (auto& [bucket_id, bucket] : buckets_) {
        for (auto& [key, entry] : bucket->entries) {
            if (entry.state != BucketEntryState::COMMITTED) continue;
            auto winner_it = winners.find(key);
            if (winner_it == winners.end() ||
                winner_it->second.second != bucket_id) {
                entry.state = BucketEntryState::TOMBSTONE;
                ++bucket->tombstones;
                if (bucket->live_bytes >= entry.reserved_size) {
                    bucket->live_bytes -= entry.reserved_size;
                } else {
                    bucket->live_bytes = 0;
                }
                continue;
            }
            key_index_[key] = bucket_id;

            auto layout = RebuildBucketEntryLayout(
                entry.entry_offset, entry.key_size, entry.value_size,
                alignment_);
            if (!layout) continue;
            recovered_replicas_.push_back(RecoveredReplica{
                key, MakeBucketDescriptor(BucketDataPath(bucket_id), *layout,
                                          entry.value_size, bucket_id)});
        }
    }

    // Orphaned data files (no valid `.meta`) can never be read, because every
    // descriptor is reconstructed from metadata. Remove them to reclaim space.
    for (const int64_t data_id : data_ids) {
        if (buckets_.count(data_id) > 0) continue;
        LOG(WARNING) << "Removing orphaned DFS bucket data file for bucket_id="
                     << data_id << " (no valid metadata)";
        DeleteBucketFiles(data_id);
        max_seen_id = std::max(max_seen_id, data_id);
    }

    // Snapshot-less logs are unreadable: a snapshot establishes the bucket
    // capacity/generation on which deltas are replayed.
    for (const int64_t log_id : log_ids) {
        if (buckets_.count(log_id) > 0) continue;
        LOG(WARNING) << "Removing orphaned DFS bucket metadata log for "
                        "bucket_id="
                     << log_id << " (no valid metadata)";
        (void)fs_adapter_->DeleteFile(BucketMetaLogPath(log_id));
        max_seen_id = std::max(max_seen_id, log_id);
    }

    next_bucket_id_ = max_seen_id + 1;
    next_generation_ = std::max<uint64_t>(1, max_generation);
    // Continue the sequence space past everything on disk, so a record written
    // after recovery can never compare equal to (or below) a replayed one.
    next_log_seq_ = max_log_seq + 1;

    // Resume appending into the newest recovered bucket that still has room,
    // rather than starting a fresh one and abandoning the tail of that bucket.
    // Recovery restored `append_offset` past every persisted entry (including
    // PENDING ones, whose space stays reserved), so continuing here appends
    // after the recovered data and can never overwrite it.
    active_bucket_id_ = -1;
    for (const auto& [bucket_id, bucket] : buckets_) {
        if (bucket->frozen) continue;
        auto entry_start = CheckedAlignUp(bucket->append_offset, alignment_);
        if (!entry_start || *entry_start >= bucket->capacity) continue;
        if (bucket_id > active_bucket_id_) active_bucket_id_ = bucket_id;
    }

    // Seed the LRU newest-first so recovered buckets have a defined order.
    std::vector<int64_t> ordered;
    ordered.reserve(buckets_.size());
    for (const auto& [bucket_id, bucket] : buckets_) {
        (void)bucket;
        ordered.push_back(bucket_id);
    }
    std::sort(ordered.begin(), ordered.end());
    for (const int64_t bucket_id : ordered) {
        lru_list_.push_front(bucket_id);
        lru_index_[bucket_id] = lru_list_.begin();
    }

    return {};
}

}  // namespace mooncake
