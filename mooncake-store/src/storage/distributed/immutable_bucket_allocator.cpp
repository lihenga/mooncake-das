#include "storage/distributed/immutable_bucket_allocator.h"

#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "master_metric_manager.h"
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
constexpr auto kBucketCreationRetryDelay = std::chrono::seconds(1);

int64_t CurrentTimeNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool IsLive(BucketEntryState state) {
    return state != BucketEntryState::TOMBSTONE;
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

ImmutableBucketAllocator::PendingEviction::~PendingEviction() {
    // An unresolved transaction must not leave the bucket frozen forever.
    // Note: `owner_` must still be set when AbortEviction is entered, because
    // that is how it recognises the transaction as its own; clearing it here
    // first would make the abort a silent no-op. AbortEviction clears it.
    if (owner_ != nullptr) {
        owner_->AbortEviction(std::move(*this), /*demote=*/false);
    }
}

ImmutableBucketAllocator::PendingEviction::PendingEviction(
    PendingEviction&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      bucket_id_(std::exchange(other.bucket_id_, -1)),
      bucket_identity_(std::move(other.bucket_identity_)),
      candidates_(std::move(other.candidates_)) {
    other.candidates_.clear();
}

ImmutableBucketAllocator::PendingEviction&
ImmutableBucketAllocator::PendingEviction::operator=(
    PendingEviction&& other) noexcept {
    if (this != &other) {
        if (owner_ != nullptr) {
            PendingEviction discarded;
            discarded.owner_ = std::exchange(owner_, nullptr);
            discarded.bucket_id_ = bucket_id_;
            discarded.bucket_identity_ = std::move(bucket_identity_);
            discarded.candidates_ = std::move(candidates_);
            discarded.owner_->AbortEviction(std::move(discarded),
                                            /*demote=*/false);
        }
        owner_ = std::exchange(other.owner_, nullptr);
        bucket_id_ = std::exchange(other.bucket_id_, -1);
        bucket_identity_ = std::move(other.bucket_identity_);
        candidates_ = std::move(other.candidates_);
        other.candidates_.clear();
    }
    return *this;
}

ImmutableBucketAllocator::~ImmutableBucketAllocator() {
    initialized_.store(false, std::memory_order_release);
    StopBucketWorkers();

    std::vector<int64_t> ready_bucket_ids;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_bucket_ids = DetachReadyBucketsLocked();
        UpdateBucketPoolMetricsLocked();
    }
    for (const int64_t bucket_id : ready_bucket_ids) {
        DeleteBucketFiles(bucket_id);
    }
    if (fs_adapter_) fs_adapter_->Shutdown();
}

std::string ImmutableBucketAllocator::FormatBucketId(int64_t bucket_id) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << bucket_id;
    return oss.str();
}

std::string ImmutableBucketAllocator::BucketDataPath(int64_t bucket_id) const {
    return fsdir_ + "/" + kBucketFilePrefix + FormatBucketId(bucket_id) +
           kBucketDataSuffix;
}

std::string ImmutableBucketAllocator::BucketMetaPath(int64_t bucket_id) const {
    return fsdir_ + "/" + kBucketFilePrefix + FormatBucketId(bucket_id) +
           kBucketMetaSuffix;
}

tl::expected<void, ErrorCode> ImmutableBucketAllocator::Init(
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
    ready_bucket_target_ = static_cast<size_t>(config.ready_bucket_target);
    bucket_create_concurrency_ =
        static_cast<size_t>(config.bucket_create_concurrency);
    eviction_enabled_ = config.eviction_enabled;
    eviction_high_watermark_ = config.eviction_high_watermark;
    eviction_low_watermark_ = config.eviction_low_watermark;
    eviction_check_interval_ = config.eviction_check_interval;

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

    auto purged = PurgeExistingBucketFiles();
    if (!purged) {
        fs_adapter_->Shutdown();
        fs_adapter_.reset();
        return purged;
    }

    size_t initial_ready_target = 0;
    ErrorCode initial_creation_error = ErrorCode::OK;
    if (ready_bucket_target_ > 0) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_bucket_workers_ = false;
            bucket_creation_retry_after_ =
                std::chrono::steady_clock::time_point::min();
            initial_ready_target = EffectiveReadyTargetLocked();
            initialization_creations_remaining_ = initial_ready_target;
            UpdateBucketPoolMetricsLocked();
        }

        try {
            bucket_create_workers_.reserve(bucket_create_concurrency_);
            for (size_t i = 0; i < bucket_create_concurrency_; ++i) {
                bucket_create_workers_.emplace_back(
                    &ImmutableBucketAllocator::BucketCreateWorker, this);
            }
        } catch (const std::system_error& e) {
            LOG(ERROR) << "Failed to start DFS bucket creation workers: "
                       << e.what();
            StopBucketWorkers();
            std::vector<int64_t> ready_bucket_ids;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ready_bucket_ids = DetachReadyBucketsLocked();
                UpdateBucketPoolMetricsLocked();
            }
            for (const int64_t bucket_id : ready_bucket_ids) {
                DeleteBucketFiles(bucket_id);
            }
            fs_adapter_->Shutdown();
            fs_adapter_.reset();
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }

        if (initial_ready_target > 0) {
            std::unique_lock<std::mutex> lock(mutex_);
            bucket_pool_cv_.notify_all();
            bucket_pool_cv_.wait(lock, [this, initial_ready_target] {
                return stop_bucket_workers_ ||
                       ready_bucket_ids_.size() >= initial_ready_target ||
                       (initialization_creations_remaining_ == 0 &&
                        bucket_creations_in_flight_ == 0);
            });
            initialization_creations_remaining_ = 0;
            if (ready_bucket_ids_.size() < initial_ready_target) {
                initial_creation_error = last_bucket_creation_error_;
                if (!ready_bucket_ids_.empty()) {
                    LOG(WARNING)
                        << "DFS ready bucket pool initialized below target, "
                        << "ready_count=" << ready_bucket_ids_.size()
                        << ", target=" << initial_ready_target
                        << ", last_error=" << initial_creation_error;
                }
            }
            bucket_pool_cv_.notify_all();
        }

        if (initial_ready_target > 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (ready_bucket_ids_.empty()) {
                if (initial_creation_error == ErrorCode::OK) {
                    initial_creation_error = ErrorCode::FILE_WRITE_FAIL;
                }
            } else {
                initial_creation_error = ErrorCode::OK;
            }
        }
    } else {
        std::lock_guard<std::mutex> lock(mutex_);
        UpdateBucketPoolMetricsLocked();
    }

    if (initial_creation_error != ErrorCode::OK) {
        LOG(ERROR) << "Failed to initialize DFS ready bucket pool, target="
                   << initial_ready_target
                   << ", error=" << initial_creation_error;
        StopBucketWorkers();
        std::vector<int64_t> ready_bucket_ids;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_bucket_ids = DetachReadyBucketsLocked();
            UpdateBucketPoolMetricsLocked();
        }
        for (const int64_t bucket_id : ready_bucket_ids) {
            DeleteBucketFiles(bucket_id);
        }
        fs_adapter_->Shutdown();
        fs_adapter_.reset();
        return tl::make_unexpected(initial_creation_error);
    }

    initialized_.store(true, std::memory_order_release);

    size_t ready_bucket_count = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_bucket_count = ready_bucket_ids_.size();
    }
    LOG(INFO) << "DFS bucket allocator initialized, fsdir=" << fsdir_
              << ", bucket_capacity=" << bucket_capacity_
              << ", alignment=" << alignment_
              << ", max_bucket_count=" << max_bucket_count_
              << ", ready_bucket_target=" << ready_bucket_target_
              << ", bucket_create_concurrency=" << bucket_create_concurrency_
              << ", ready_buckets=" << ready_bucket_count;
    return {};
}

tl::expected<void, ErrorCode>
ImmutableBucketAllocator::PurgeExistingBucketFiles() {
    if (!fs_adapter_) {
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }

    auto files = fs_adapter_->ListFiles(fsdir_);
    if (!files) {
        if (files.error() == ErrorCode::FILE_NOT_FOUND) return {};
        LOG(ERROR) << "Failed to list DFS bucket directory " << fsdir_
                   << ", error=" << files.error();
        return tl::make_unexpected(files.error());
    }

    std::unordered_set<int64_t> bucket_ids;
    for (const auto& name : *files) {
        if (auto id = ParseBucketFileName(name, kBucketDataSuffix)) {
            bucket_ids.insert(*id);
        } else if (auto meta_id = ParseBucketFileName(name, kBucketMetaSuffix)) {
            // Only an older build writes these; nothing reads them any more.
            bucket_ids.insert(*meta_id);
        }
        // Anything else in the directory belongs to someone else and is left
        // alone.
    }
    if (bucket_ids.empty()) return {};

    std::vector<int64_t> ordered(bucket_ids.begin(), bucket_ids.end());
    std::sort(ordered.begin(), ordered.end());
    for (const int64_t bucket_id : ordered) {
        DeleteBucketFiles(bucket_id);
    }
    LOG(INFO) << "Purged " << ordered.size()
              << " leftover DFS bucket(s) from a previous run, fsdir=" << fsdir_;
    return {};
}

void ImmutableBucketAllocator::SealActiveBucketLocked() {
    if (active_bucket_id_ < 0) return;
    auto it = buckets_.find(active_bucket_id_);
    active_bucket_id_ = -1;
    if (it == buckets_.end()) return;
    BucketState& bucket = *it->second;
    // The bucket accepts no further allocations, so its reserved prefix is
    // final. Nothing is persisted: the state lives and dies with this process.
    bucket.sealed = true;
}

void ImmutableBucketAllocator::DeleteBucketFiles(int64_t bucket_id) {
    if (!fs_adapter_) return;
    auto meta_result = fs_adapter_->DeleteFile(BucketMetaPath(bucket_id));
    if (!meta_result && meta_result.error() != ErrorCode::FILE_NOT_FOUND) {
        LOG(ERROR) << "Failed to delete bucket metadata file for bucket_id="
                   << bucket_id << ", error=" << meta_result.error();
    }
    auto data_result = fs_adapter_->DeleteFile(BucketDataPath(bucket_id));
    if (!data_result && data_result.error() != ErrorCode::FILE_NOT_FOUND) {
        LOG(ERROR) << "Failed to delete bucket data file for bucket_id="
                   << bucket_id << ", error=" << data_result.error();
    }
}

// === bucket lifecycle ===

size_t ImmutableBucketAllocator::EffectiveReadyTargetLocked() const {
    const size_t ready_count = ready_bucket_ids_.size();
    const size_t non_ready_count = buckets_.size() - ready_count;
    if (max_bucket_count_ > 0 &&
        non_ready_count >= static_cast<size_t>(max_bucket_count_)) {
        return 0;
    }
    const size_t available =
        max_bucket_count_ > 0
            ? static_cast<size_t>(max_bucket_count_) - non_ready_count
            : ready_bucket_target_;
    return std::min(ready_bucket_target_, available);
}

bool ImmutableBucketAllocator::CanReserveBucketCreationLocked() const {
    if (stop_bucket_workers_ || next_bucket_id_ > kMaxBucketId) return false;
    return max_bucket_count_ <= 0 ||
           buckets_.size() + bucket_creations_in_flight_ <
               static_cast<size_t>(max_bucket_count_);
}

std::optional<ImmutableBucketAllocator::BucketCreationReservation>
ImmutableBucketAllocator::ReserveBucketCreationLocked() {
    if (!CanReserveBucketCreationLocked()) return std::nullopt;
    BucketCreationReservation reservation;
    reservation.bucket_id = next_bucket_id_++;
    ++bucket_creations_in_flight_;
    try {
        UpdateBucketPoolMetricsLocked();
    } catch (...) {
        --bucket_creations_in_flight_;
        throw;
    }
    return reservation;
}

void ImmutableBucketAllocator::FinishBucketCreationLocked(bool synchronous) {
    CHECK_GT(bucket_creations_in_flight_, 0);
    --bucket_creations_in_flight_;
    if (synchronous) synchronous_creation_in_flight_ = false;
    bucket_pool_cv_.notify_all();
}

void ImmutableBucketAllocator::RollbackBucketPublicationLocked(
    int64_t bucket_id, const BucketPtr& bucket) {
    auto ready_it =
        std::find(ready_bucket_ids_.begin(), ready_bucket_ids_.end(), bucket_id);
    if (ready_it != ready_bucket_ids_.end()) ready_bucket_ids_.erase(ready_it);
    RemoveFromLruLocked(bucket_id);
    if (active_bucket_id_ == bucket_id) active_bucket_id_ = -1;
    auto it = buckets_.find(bucket_id);
    if (it != buckets_.end() && it->second == bucket) buckets_.erase(it);
}

bool ImmutableBucketAllocator::ActiveBucketHasSpaceLocked(
    uint64_t required, BucketPtr* bucket) const {
    if (active_bucket_id_ < 0) return false;
    auto it = buckets_.find(active_bucket_id_);
    if (it == buckets_.end() || it->second->ready || it->second->sealed ||
        it->second->frozen) {
        return false;
    }
    auto entry_start = CheckedAlignUp(it->second->append_offset, alignment_);
    if (!entry_start || *entry_start > it->second->capacity ||
        required > it->second->capacity - *entry_start) {
        return false;
    }
    if (bucket != nullptr) *bucket = it->second;
    return true;
}

void ImmutableBucketAllocator::RequestRefillLocked(bool urgent) {
    // One urgent retry may bypass the backoff for each observed failure wave.
    // A stream of waiting allocations cannot continuously reset the delay.
    if (urgent &&
        last_urgent_refill_sequence_ != bucket_creation_failure_sequence_) {
        last_urgent_refill_sequence_ = bucket_creation_failure_sequence_;
        bucket_creation_retry_after_ =
            std::chrono::steady_clock::time_point::min();
    }
    bucket_pool_cv_.notify_all();
}

ImmutableBucketAllocator::BucketPtr
ImmutableBucketAllocator::PromoteReadyBucketLocked() {
    while (!ready_bucket_ids_.empty()) {
        const int64_t bucket_id = ready_bucket_ids_.front();
        ready_bucket_ids_.pop_front();
        auto it = buckets_.find(bucket_id);
        if (it == buckets_.end() || !it->second->ready) continue;

        SealActiveBucketLocked();
        it->second->ready = false;
        active_bucket_id_ = bucket_id;
        TouchLruLocked(bucket_id, CurrentTimeNs());
        UpdateBucketPoolMetricsLocked();
        RequestRefillLocked();
        return it->second;
    }
    UpdateBucketPoolMetricsLocked();
    return nullptr;
}

std::vector<int64_t> ImmutableBucketAllocator::DetachReadyBucketsLocked() {
    std::vector<int64_t> bucket_ids;
    bucket_ids.reserve(ready_bucket_ids_.size());
    for (const int64_t bucket_id : ready_bucket_ids_) {
        auto it = buckets_.find(bucket_id);
        if (it == buckets_.end() || !it->second->ready) continue;
        bucket_ids.push_back(bucket_id);
        buckets_.erase(it);
    }
    ready_bucket_ids_.clear();
    return bucket_ids;
}

void ImmutableBucketAllocator::UpdateBucketPoolMetricsLocked() const {
    MasterMetricManager::instance().set_dfs_bucket_pool_state(
        ready_bucket_ids_.size(), bucket_creations_in_flight_,
        static_cast<uint64_t>(ready_bucket_ids_.size()) * bucket_capacity_);
}

void ImmutableBucketAllocator::StopBucketWorkers() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_bucket_workers_ = true;
        initialization_creations_remaining_ = 0;
        bucket_pool_cv_.notify_all();
    }
    for (auto& worker : bucket_create_workers_) {
        if (worker.joinable()) worker.join();
    }
    bucket_create_workers_.clear();
}

void ImmutableBucketAllocator::BucketCreateWorker() {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto needs_creation = [this] {
        return CanReserveBucketCreationLocked() &&
               ready_bucket_ids_.size() + bucket_creations_in_flight_ <
                   EffectiveReadyTargetLocked();
    };

    while (!stop_bucket_workers_) {
        while (!stop_bucket_workers_ && !needs_creation()) {
            bucket_pool_cv_.wait(lock);
        }
        if (stop_bucket_workers_) break;

        const auto now = std::chrono::steady_clock::now();
        if (initialization_creations_remaining_ == 0 &&
            now < bucket_creation_retry_after_) {
            const auto retry_after = bucket_creation_retry_after_;
            bucket_pool_cv_.wait_until(
                lock, retry_after, [this, retry_after, &needs_creation] {
                    return stop_bucket_workers_ ||
                           initialization_creations_remaining_ > 0 ||
                           bucket_creation_retry_after_ < retry_after ||
                           !needs_creation();
                });
            continue;
        }

        std::optional<BucketCreationReservation> reservation;
        int64_t reserved_bucket_id = -1;
        BucketPtr bucket;
        const bool initialization_attempt =
            initialization_creations_remaining_ > 0;
        bool initialization_consumed = false;
        try {
            reservation = ReserveBucketCreationLocked();
            if (!reservation) continue;
            reserved_bucket_id = reservation->bucket_id;
            if (initialization_attempt) {
                --initialization_creations_remaining_;
                initialization_consumed = true;
            }

            lock.unlock();
            const auto create_started = std::chrono::steady_clock::now();
            auto preallocated = fs_adapter_->PreallocateFile(
                BucketDataPath(reserved_bucket_id), bucket_capacity_);
            const auto create_latency_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - create_started)
                    .count();
            MasterMetricManager::instance()
                .observe_dfs_bucket_create_latency_us(create_latency_us);
            lock.lock();

            const bool within_capacity =
                max_bucket_count_ <= 0 ||
                buckets_.size() + bucket_creations_in_flight_ <=
                    static_cast<size_t>(max_bucket_count_);
            const bool within_target =
                ready_bucket_ids_.size() + bucket_creations_in_flight_ <=
                EffectiveReadyTargetLocked();
            const bool publish = preallocated && within_capacity &&
                                 within_target && !stop_bucket_workers_;
            bool delete_file = false;
            const char* result = "failure";
            if (publish) {
                bucket = std::make_shared<BucketState>();
                bucket->bucket_id = reserved_bucket_id;
                bucket->capacity = bucket_capacity_;
                bucket->last_access_ns = CurrentTimeNs();
                bucket->ready = true;
                const bool inserted =
                    buckets_.emplace(bucket->bucket_id, bucket).second;
                if (!inserted) throw std::runtime_error("duplicate bucket id");
                ready_bucket_ids_.push_back(bucket->bucket_id);
                bucket_creation_retry_after_ =
                    std::chrono::steady_clock::time_point::min();
                result = "success";
            } else if (preallocated) {
                delete_file = true;
                result = "discarded";
            } else {
                delete_file = true;
                ++bucket_creation_failure_sequence_;
                last_bucket_creation_error_ = preallocated.error();
                bucket_creation_retry_after_ = std::chrono::steady_clock::now() +
                                               kBucketCreationRetryDelay;
            }

            FinishBucketCreationLocked(/*synchronous=*/false);
            reservation.reset();
            const size_t ready_count = ready_bucket_ids_.size();
            const size_t in_flight = bucket_creations_in_flight_;
            const int64_t max_bucket_count = max_bucket_count_;
            UpdateBucketPoolMetricsLocked();
            lock.unlock();

            MasterMetricManager::instance().inc_dfs_bucket_create_total(result);
            if (!preallocated) {
                LOG(WARNING) << "Failed to precreate DFS bucket, bucket_id="
                             << reserved_bucket_id
                             << ", error=" << preallocated.error()
                             << ", ready_count=" << ready_count
                             << ", in_flight=" << in_flight
                             << ", max_bucket_count=" << max_bucket_count;
            } else if (!publish) {
                VLOG(1)
                    << "Discarded precreated DFS bucket after state change, "
                    << "bucket_id=" << reserved_bucket_id;
            }
            if (delete_file) DeleteBucketFiles(reserved_bucket_id);

            lock.lock();
        } catch (...) {
            if (!lock.owns_lock()) lock.lock();
            if (initialization_attempt && !initialization_consumed) {
                --initialization_creations_remaining_;
            }
            if (reservation) {
                RollbackBucketPublicationLocked(reserved_bucket_id, bucket);
                FinishBucketCreationLocked(/*synchronous=*/false);
            }
            ++bucket_creation_failure_sequence_;
            last_bucket_creation_error_ = ErrorCode::INTERNAL_ERROR;
            bucket_creation_retry_after_ =
                std::chrono::steady_clock::now() + kBucketCreationRetryDelay;
            bucket_pool_cv_.notify_all();
            lock.unlock();
            if (reservation) DeleteBucketFiles(reserved_bucket_id);
            LOG(ERROR) << "Unexpected exception while creating DFS bucket; "
                          "the ready bucket pool will retry";
            lock.lock();
        }
    }
}

tl::expected<ImmutableBucketAllocator::BucketPtr, ErrorCode>
ImmutableBucketAllocator::CreateBucketUnlocked(
    std::unique_lock<std::mutex>& lock) {
    std::optional<BucketCreationReservation> reservation;
    try {
        reservation = ReserveBucketCreationLocked();
    } catch (...) {
        LOG(ERROR) << "Unexpected exception while reserving a DFS bucket";
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    if (!reservation) {
        if (next_bucket_id_ > kMaxBucketId) {
            LOG(ERROR) << "DFS bucket id space exhausted, next_bucket_id="
                       << next_bucket_id_;
        }
        return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
    }
    synchronous_creation_in_flight_ = true;

    ErrorCode error = ErrorCode::OK;
    BucketPtr bucket;
    bool preallocated = false;
    bool publish = false;
    try {
        lock.unlock();
        const auto create_started = std::chrono::steady_clock::now();
        auto result = fs_adapter_->PreallocateFile(
            BucketDataPath(reservation->bucket_id), bucket_capacity_);
        preallocated = result.has_value();
        if (!result) error = result.error();
        const auto create_latency_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - create_started)
                .count();
        MasterMetricManager::instance().observe_dfs_bucket_create_latency_us(
            create_latency_us);
        lock.lock();

        const bool within_capacity =
            max_bucket_count_ <= 0 ||
            buckets_.size() + bucket_creations_in_flight_ <=
                static_cast<size_t>(max_bucket_count_);
        publish = preallocated && within_capacity && !stop_bucket_workers_;
        if (publish) {
            bucket = std::make_shared<BucketState>();
            bucket->bucket_id = reservation->bucket_id;
            bucket->capacity = bucket_capacity_;
            bucket->last_access_ns = CurrentTimeNs();
            const bool inserted =
                buckets_.emplace(bucket->bucket_id, bucket).second;
            if (!inserted) throw std::runtime_error("duplicate bucket id");
            SealActiveBucketLocked();
            active_bucket_id_ = bucket->bucket_id;
            TouchLruLocked(bucket->bucket_id, bucket->last_access_ns);
        } else if (preallocated) {
            error = ErrorCode::NO_AVAILABLE_HANDLE;
        }
    } catch (...) {
        if (!lock.owns_lock()) lock.lock();
        RollbackBucketPublicationLocked(reservation->bucket_id, bucket);
        FinishBucketCreationLocked(/*synchronous=*/true);
        UpdateBucketPoolMetricsLocked();
        lock.unlock();
        DeleteBucketFiles(reservation->bucket_id);
        lock.lock();
        LOG(ERROR) << "Unexpected exception while publishing DFS bucket "
                   << reservation->bucket_id;
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    FinishBucketCreationLocked(/*synchronous=*/true);
    UpdateBucketPoolMetricsLocked();

    lock.unlock();
    MasterMetricManager::instance().inc_dfs_bucket_create_total(
        publish ? "success" : (preallocated ? "discarded" : "failure"));
    if (!publish) DeleteBucketFiles(reservation->bucket_id);
    lock.lock();

    if (!publish) {
        if (!preallocated) {
            LOG(ERROR) << "Failed to create DFS bucket "
                       << reservation->bucket_id << ", error=" << error;
        }
        return tl::make_unexpected(error);
    }
    return bucket;
}

tl::expected<ImmutableBucketAllocator::BucketPtr, ErrorCode>
ImmutableBucketAllocator::EnsureActiveBucket(std::unique_lock<std::mutex>& lock,
                                          uint64_t required) {
    if (required > bucket_capacity_) {
        // Refuse rather than spill across buckets: the caller asked for one
        // contiguous region and no bucket can ever satisfy it.
        LOG(ERROR) << "DFS EnsureActiveBucket rejected: object exceeds "
                   << "bucket capacity, required=" << required
                   << ", bucket_capacity=" << bucket_capacity_;
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    std::optional<std::chrono::steady_clock::time_point> rollover_started;
    bool pool_exhaustion_recorded = false;
    const auto observe_rollover = [&rollover_started] {
        if (!rollover_started) return;
        const auto latency_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - *rollover_started)
                .count();
        MasterMetricManager::instance().observe_dfs_bucket_rollover_latency_us(
            latency_us);
    };

    while (true) {
        BucketPtr active;
        if (ActiveBucketHasSpaceLocked(required, &active)) {
            observe_rollover();
            return active;
        }
        if (!rollover_started && active_bucket_id_ >= 0) {
            rollover_started = std::chrono::steady_clock::now();
        }
        if (stop_bucket_workers_) {
            observe_rollover();
            return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
        }

        if (ready_bucket_target_ == 0) {
            if (synchronous_creation_in_flight_) {
                bucket_pool_cv_.wait(lock, [this] {
                    return stop_bucket_workers_ ||
                           !synchronous_creation_in_flight_;
                });
                continue;
            }
            auto created = CreateBucketUnlocked(lock);
            if (!created) {
                observe_rollover();
                return tl::make_unexpected(created.error());
            }
            continue;
        }

        if (!ready_bucket_ids_.empty()) {
            // Promoting a ready bucket seals the previous active one. There is
            // nothing to persist, so the allocation can continue immediately.
            if (PromoteReadyBucketLocked()) continue;
        }

        if (!CanReserveBucketCreationLocked() &&
            bucket_creations_in_flight_ == 0) {
            observe_rollover();
            return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
        }

        if (!pool_exhaustion_recorded) {
            pool_exhaustion_recorded = true;
            MasterMetricManager::instance()
                .inc_dfs_bucket_pool_exhausted_total();
            LOG_EVERY_N(WARNING, 100)
                << "DFS ready bucket pool exhausted, ready_count="
                << ready_bucket_ids_.size()
                << ", in_flight=" << bucket_creations_in_flight_
                << ", waiting_allocators=" << waiting_allocators_
                << ", active_bucket_id=" << active_bucket_id_
                << ", max_bucket_count=" << max_bucket_count_;
        }

        const uint64_t failure_sequence = bucket_creation_failure_sequence_;
        RequestRefillLocked(/*urgent=*/true);
        ++waiting_allocators_;
        const auto wait_started = std::chrono::steady_clock::now();
        bucket_pool_cv_.wait(lock, [this, required, failure_sequence] {
            return stop_bucket_workers_ ||
                   ActiveBucketHasSpaceLocked(required) ||
                   !ready_bucket_ids_.empty() ||
                   (bucket_creation_failure_sequence_ != failure_sequence &&
                    bucket_creations_in_flight_ == 0) ||
                   (!CanReserveBucketCreationLocked() &&
                    bucket_creations_in_flight_ == 0);
        });
        --waiting_allocators_;
        const auto wait_latency_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_started)
                .count();
        MasterMetricManager::instance().observe_dfs_bucket_pool_wait_latency_us(
            wait_latency_us);

        if (stop_bucket_workers_) {
            observe_rollover();
            return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
        }
        if (ready_bucket_ids_.empty() &&
            !ActiveBucketHasSpaceLocked(required) &&
            bucket_creations_in_flight_ == 0 &&
            bucket_creation_failure_sequence_ != failure_sequence) {
            const ErrorCode error = last_bucket_creation_error_ == ErrorCode::OK
                                        ? ErrorCode::FILE_WRITE_FAIL
                                        : last_bucket_creation_error_;
            observe_rollover();
            return tl::make_unexpected(error);
        }
    }
}

void ImmutableBucketAllocator::TouchLruLocked(int64_t bucket_id, int64_t now_ns) {
    auto index_it = lru_index_.find(bucket_id);
    if (index_it != lru_index_.end()) {
        lru_list_.splice(lru_list_.begin(), lru_list_, index_it->second);
    } else {
        lru_list_.push_front(bucket_id);
        try {
            lru_index_.emplace(bucket_id, lru_list_.begin());
        } catch (...) {
            lru_list_.pop_front();
            throw;
        }
    }
    auto bucket_it = buckets_.find(bucket_id);
    if (bucket_it != buckets_.end()) {
        bucket_it->second->last_access_ns = now_ns;
    }
}

void ImmutableBucketAllocator::RemoveFromLruLocked(int64_t bucket_id) {
    auto index_it = lru_index_.find(bucket_id);
    if (index_it == lru_index_.end()) return;
    lru_list_.erase(index_it->second);
    lru_index_.erase(index_it);
}

tl::expected<DistributedFSDescriptor, ErrorCode>
ImmutableBucketAllocator::ReserveInBucketLocked(BucketState& bucket,
                                             const std::string& key,
                                             uint64_t size) {
    auto layout =
        ComputeBucketEntryLayout(bucket.append_offset, size, alignment_);
    if (!layout) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (layout->end() > bucket.capacity) {
        return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
    }

    BucketEntry entry;
    entry.entry_offset = layout->offset;
    entry.key_size = key.size();
    entry.value_size = size;
    entry.reserved_size = layout->aligned_size;
    entry.state = BucketEntryState::PENDING;

    // A live entry for this key already sits somewhere. Assigning to
    // bucket.entries[key] here would either drop the prior allocation in this
    // bucket (no Free, no unreserve, no space reclaimed) or, if the prior
    // entry is in another bucket, create an orphan that key_index_ will point
    // away from. The per-key dedup above is meant to keep this from happening,
    // but key_index_ and bucket.entries can disagree across an
    // EnsureActiveBucket unlock window, so the guard must live here too.
    auto existing_it = bucket.entries.find(key);
    if (existing_it != bucket.entries.end() && IsLive(existing_it->second.state)) {
        return tl::make_unexpected(ErrorCode::OBJECT_ALREADY_EXISTS);
    }
    auto existing_index_it = key_index_.find(key);
    if (existing_index_it != key_index_.end() &&
        existing_index_it->second != bucket.bucket_id) {
        // The key already belongs to a different bucket. Reserving here would
        // overwrite key_index_ and leave the prior bucket holding an orphan.
        return tl::make_unexpected(ErrorCode::OBJECT_ALREADY_EXISTS);
    }

    bucket.entries[key] = entry;
    bucket.append_offset = layout->end();
    bucket.live_bytes += layout->aligned_size;
    key_index_[key] = bucket.bucket_id;

    auto descriptor = MakeBucketDescriptor(BucketDataPath(bucket.bucket_id),
                                          *layout, bucket.bucket_id);
    return descriptor;
}

void ImmutableBucketAllocator::UnreserveInBucketLocked(
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

    // The reservation existed only in memory, so undoing it is just erasing it.
    bucket.entries.erase(entry_it);

    auto index_it = key_index_.find(key);
    if (index_it != key_index_.end() && index_it->second == bucket.bucket_id) {
        key_index_.erase(index_it);
    }
}

ImmutableBucketAllocator::BucketEntry*
ImmutableBucketAllocator::FindMatchingEntryLocked(
    const std::string& key, const DistributedFSDescriptor& desc,
    BucketPtr* out_bucket) {
    auto fail = [&key, &desc](const char* reason) -> BucketEntry* {
        LOG(ERROR) << "DFS entry match failed: reason=" << reason
                   << ", key=" << key << ", bucket_id=" << desc.shard_idx
                   << ", offset=" << desc.offset;
        return nullptr;
    };

    if (desc.shard_idx < 0) return fail("invalid_bucket_id");
    const int64_t bucket_id = static_cast<int64_t>(desc.shard_idx);

    auto index_it = key_index_.find(key);
    if (index_it == key_index_.end()) return fail("key_not_indexed");
    if (index_it->second != bucket_id) return fail("key_index_bucket_mismatch");
    auto bucket_it = buckets_.find(bucket_id);
    if (bucket_it == buckets_.end()) return fail("bucket_not_found");

    auto entry_it = bucket_it->second->entries.find(key);
    if (entry_it == bucket_it->second->entries.end())
        return fail("bucket_entry_not_found");

    // Match on every layout-defining field so a descriptor from a superseded
    // allocation cannot address the current one.
    auto& entry = entry_it->second;
    if (entry.value_size != desc.object_size)
        return fail("object_size_mismatch");
    if (entry.reserved_size != desc.aligned_size)
        return fail("aligned_size_mismatch");
    if (entry.key_size != key.size()) return fail("key_size_mismatch");
    auto layout = RebuildBucketEntryLayout(entry.entry_offset, entry.value_size,
                                           alignment_);
    if (!layout) return fail("invalid_entry_layout");
    if (layout->offset != desc.offset)
        return fail("offset_mismatch");

    if (out_bucket) *out_bucket = bucket_it->second;
    return &entry;
}

// === allocation ===

tl::expected<DistributedFSDescriptor, ErrorCode>
ImmutableBucketAllocator::Allocate(const std::string& key, uint64_t size) {
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

std::vector<BatchAllocateResult> ImmutableBucketAllocator::BatchAllocate(
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
    // buckets, but one object must always fit in one bucket. Any failure here is
    // terminal for the whole batch, so the reserve loop below can iterate every
    // request that survived.
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
        auto layout = ComputeBucketEntryLayout(0, request.size, alignment_);
        if (!layout || layout->aligned_size > bucket_capacity_) {
            LOG(ERROR) << "DFS object for key " << request.key
                       << " exceeds bucket capacity, object_size="
                       << request.size << ", reserved_size="
                       << (layout ? layout->aligned_size : 0)
                       << ", bucket_capacity=" << bucket_capacity_;
            fail_all(ErrorCode::INVALID_PARAMS);
            return results;
        }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    struct Reservation {
        size_t request_index = 0;
        int64_t bucket_id = -1;
        BucketPtr bucket_identity;
    };
    std::vector<Reservation> reserved;
    reserved.reserve(requests.size());
    size_t already_exists_count = 0;

    // Mark every not-yet-resolved request as a generic failure so a later break
    // leaves no slot at its default OK.
    auto fail_remaining = [&](size_t from_index) {
        for (size_t i = from_index; i < requests.size(); ++i) {
            results[i].success = false;
            results[i].descriptor = DistributedFSDescriptor{};
            if (results[i].error == ErrorCode::OK) {
                results[i].error = ErrorCode::NO_AVAILABLE_HANDLE;
            }
        }
    };

    // Append-pack in request order. EnsureActiveBucket receives only the
    // current object's reserved size, so a partially filled active bucket is
    // used before a new bucket is created. A bucket boundary can therefore
    // occur only between two objects, never inside one object.
    for (size_t index = 0; index < requests.size(); ++index) {
        const auto& request = requests[index];
        // Dedup right before the reserve, under the same lock hold. A key can
        // be live from an earlier reservation in this bucket or from a
        // concurrent batch that reserved it during the previous iteration's
        // EnsureActiveBucket window; either way, skip it.
        if (key_index_.count(request.key) != 0) {
            results[index].success = false;
            results[index].descriptor = DistributedFSDescriptor{};
            results[index].error = ErrorCode::OBJECT_ALREADY_EXISTS;
            ++already_exists_count;
            continue;
        }
        auto object_layout =
            ComputeBucketEntryLayout(0, request.size, alignment_);
        if (!object_layout) {
            // Per-key failure: this key's shape is invalid, but subsequent
            // keys may still allocate successfully.
            LOG(ERROR) << "DFS batch allocate layout failed: key="
                       << request.key << ", size=" << request.size
                       << ", error=invalid_layout";
            results[index].success = false;
            results[index].descriptor = DistributedFSDescriptor{};
            results[index].error = ErrorCode::INVALID_PARAMS;
            fail_remaining(index + 1);
            break;
        }
        auto bucket_result =
            EnsureActiveBucket(lock, object_layout->aligned_size);
        if (!bucket_result) {
            // Global failure: no active bucket can be created or found, so
            // remaining keys will fail the same way. Mark them and stop.
            LOG(ERROR) << "DFS batch allocate active bucket failed: key="
                       << request.key << ", reserved_size="
                       << object_layout->aligned_size
                       << ", error=" << bucket_result.error();
            results[index].success = false;
            results[index].descriptor = DistributedFSDescriptor{};
            results[index].error = bucket_result.error();
            fail_remaining(index + 1);
            break;
        }
        auto bucket = bucket_result.value();
        auto descriptor =
            ReserveInBucketLocked(*bucket, request.key, request.size);
        if (!descriptor) {
            // Per-key failure: this key does not fit in the current active
            // bucket, but subsequent keys (which may be smaller) might.
            // OBJECT_ALREADY_EXISTS means a concurrent batch reserved the key
            // during this iteration's EnsureActiveBucket window; record it and
            // keep packing the rest instead of unwinding the whole batch.
            results[index].success = false;
            results[index].descriptor = DistributedFSDescriptor{};
            results[index].error = descriptor.error();
            if (descriptor.error() == ErrorCode::OBJECT_ALREADY_EXISTS) {
                // A concurrent batch reserved the key during this iteration's
                // EnsureActiveBucket window; keep packing the rest instead of
                // unwinding the whole batch.
                ++already_exists_count;
                continue;
            }
            LOG(ERROR) << "DFS batch allocate reserve failed: key="
                       << request.key << ", size=" << request.size
                       << ", bucket_id=" << bucket->bucket_id
                       << ", append_offset=" << bucket->append_offset
                       << ", error=" << descriptor.error();
            fail_remaining(index + 1);
            break;
        }
        results[index].descriptor = std::move(descriptor.value());
        results[index].success = true;
        results[index].error = ErrorCode::OK;
        reserved.push_back({index, bucket->bucket_id, bucket});
        TouchLruLocked(bucket->bucket_id, CurrentTimeNs());
    }

    if (reserved.size() + already_exists_count != requests.size()) {
        // Roll back in reverse reservation order, across every bucket touched
        // by this batch. Nothing was written to disk, so undoing the in-memory
        // state is all it takes.
        for (auto it = reserved.rbegin(); it != reserved.rend(); ++it) {
            auto bucket_it = buckets_.find(it->bucket_id);
            if (bucket_it != buckets_.end() &&
                bucket_it->second.get() == it->bucket_identity.get()) {
                UnreserveInBucketLocked(*bucket_it->second,
                                        requests[it->request_index].key,
                                        results[it->request_index].descriptor);
            }
            // The reservation was rolled back, so this key did not succeed
            // after all; surface a failure so the caller does not treat a
            // descriptor that no longer resolves as valid.
            results[it->request_index].success = false;
            results[it->request_index].descriptor = DistributedFSDescriptor{};
            if (results[it->request_index].error == ErrorCode::OK) {
                results[it->request_index].error =
                    ErrorCode::NO_AVAILABLE_HANDLE;
            }
        }
        lock.unlock();
        return results;
    }

    // Reservations live in memory only; there is no metadata to keep in sync.
    return results;
}

void ImmutableBucketAllocator::Free(const std::string& key,
                                 const DistributedFSDescriptor& descriptor) {
    if (!initialized_.load(std::memory_order_acquire)) return;

    std::lock_guard<std::mutex> lock(mutex_);
    BucketPtr bucket;
    auto* entry = FindMatchingEntryLocked(key, descriptor, &bucket);
    if (entry == nullptr) {
        // Stale Free for a region that has since been reallocated: ignore it so
        // it cannot drop the allocation that replaced it.
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
    // A tombstone only has to survive for this process: buckets are append-only
    // and the reserved bytes come back when the whole bucket is evicted, so
    // there is nothing to write.
}

void ImmutableBucketAllocator::UpdateAccess(
    const std::string& key, const DistributedFSDescriptor& descriptor) {
    if (!initialized_.load(std::memory_order_acquire)) return;

    std::lock_guard<std::mutex> lock(mutex_);
    BucketPtr bucket;
    auto* entry = FindMatchingEntryLocked(key, descriptor, &bucket);
    if (entry == nullptr || !IsLive(entry->state)) return;
    // A frozen bucket is mid-eviction; refreshing it would fight the
    // transaction the master is currently resolving.
    if (bucket->frozen) return;
    TouchLruLocked(bucket->bucket_id, CurrentTimeNs());
}

uint64_t ImmutableBucketAllocator::GetTotalCapacity() const {
    // A fixed denominator: watermarks must not move as buckets come and go,
    // or deleting a bucket would shrink capacity in lockstep with usage and
    // eviction would never converge.
    std::lock_guard<std::mutex> lock(mutex_);
    if (max_bucket_count_ > 0) {
        return static_cast<uint64_t>(max_bucket_count_) * bucket_capacity_;
    }
    return static_cast<uint64_t>(buckets_.size()) * bucket_capacity_;
}

uint64_t ImmutableBucketAllocator::UsedBytesLocked() const {
    // Physical reservation, not live bytes: an append-only bucket keeps its
    // whole reserved prefix until the bucket itself is evicted.
    uint64_t used = 0;
    for (const auto& [bucket_id, bucket] : buckets_) {
        (void)bucket_id;
        used += bucket->append_offset;
    }
    return used;
}

uint64_t ImmutableBucketAllocator::GetUsedBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return UsedBytesLocked();
}

int64_t ImmutableBucketAllocator::SetMaxBucketCount(int64_t new_max_bucket_count) {
    std::vector<int64_t> removed_ready_buckets;
    int64_t old = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        old = max_bucket_count_;
        max_bucket_count_ = new_max_bucket_count;

        while (ready_bucket_ids_.size() > EffectiveReadyTargetLocked()) {
            const int64_t bucket_id = ready_bucket_ids_.back();
            ready_bucket_ids_.pop_back();
            auto it = buckets_.find(bucket_id);
            if (it == buckets_.end() || !it->second->ready) continue;
            buckets_.erase(it);
            removed_ready_buckets.push_back(bucket_id);
        }
        if (new_max_bucket_count > old) {
            bucket_creation_retry_after_ =
                std::chrono::steady_clock::time_point::min();
        }
        UpdateBucketPoolMetricsLocked();
    }

    for (const int64_t bucket_id : removed_ready_buckets) {
        DeleteBucketFiles(bucket_id);
    }
    bucket_pool_cv_.notify_all();
    LOG(INFO) << "Dynamic max_bucket_count changed from " << old << " to "
              << new_max_bucket_count
              << ", removed_ready_buckets=" << removed_ready_buckets.size();
    return old;
}

size_t ImmutableBucketAllocator::GetBucketCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buckets_.size();
}

std::optional<int64_t> ImmutableBucketAllocator::GetBucketIdForKey(
    const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = key_index_.find(key);
    if (it == key_index_.end()) return std::nullopt;
    return it->second;
}

// === eviction ===

ImmutableBucketAllocator::PendingEviction
ImmutableBucketAllocator::PrepareEviction() {
    return PrepareEvictionInternal(/*force_one=*/false);
}

ImmutableBucketAllocator::PendingEviction
ImmutableBucketAllocator::PrepareInvalidation(int64_t bucket_id) {
    return PrepareInvalidationInternal(bucket_id);
}

bool ImmutableBucketAllocator::HasBucket(int64_t bucket_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buckets_.contains(bucket_id);
}

ImmutableBucketAllocator::PendingEviction
ImmutableBucketAllocator::PrepareInvalidationInternal(int64_t bucket_id) {
    PendingEviction pending;
    if (!initialized_.load(std::memory_order_acquire) || bucket_id < 0) {
        return pending;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto bucket_it = buckets_.find(bucket_id);
        if (bucket_it == buckets_.end() || bucket_it->second->frozen) {
            return pending;
        }
        const bool was_active = active_bucket_id_ == bucket_id;
        const BucketPtr& victim = bucket_it->second;

        std::vector<DfsAllocatorInterface::EvictionCandidate> candidates;
        for (const auto& [key, entry] : victim->entries) {
            if (!IsLive(entry.state)) continue;
            auto layout = RebuildBucketEntryLayout(entry.entry_offset,
                                                   entry.value_size, alignment_);
            if (!layout) {
                LOG(ERROR) << "Skipping DFS invalidation of bucket " << bucket_id
                           << ": entry for key " << key
                           << " has an inconsistent layout";
                return pending;
            }
            DfsAllocatorInterface::EvictionCandidate candidate;
            candidate.key = key;
            candidate.shard_idx = static_cast<int>(bucket_id);
            candidate.offset = layout->offset;
            candidate.descriptor =
                MakeBucketDescriptor(BucketDataPath(bucket_id), *layout, bucket_id);
            candidates.push_back(std::move(candidate));
        }

        victim->frozen = true;
        victim->sealed = true;
        RemoveFromLruLocked(bucket_id);
        if (was_active) {
            active_bucket_id_ = -1;
            RequestRefillLocked(/*urgent=*/true);
        }
        pending.owner_ = this;
        pending.bucket_id_ = bucket_id;
        pending.bucket_identity_ = victim;
        pending.candidates_ = std::move(candidates);
    }
    return pending;
}

ImmutableBucketAllocator::PendingEviction
ImmutableBucketAllocator::PrepareEvictionForAllocationFailure() {
    return PrepareEvictionInternal(/*force_one=*/true);
}

ImmutableBucketAllocator::PendingEviction
ImmutableBucketAllocator::PrepareEvictionInternal(bool force_one) {
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
        // neither active nor frozen. In-flight writes no longer mark an entry:
        // the master rejects candidates whose DFS replica is still PROCESSING
        // while validating the transaction, so that guard lives there now.
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

        std::vector<DfsAllocatorInterface::EvictionCandidate> candidates;
        for (const auto& [key, entry] : victim->entries) {
            // Tombstoned entries are already gone from the master's view; only
            // live entries need validating.
            if (!IsLive(entry.state)) continue;
            auto layout = RebuildBucketEntryLayout(
                entry.entry_offset, entry.value_size, alignment_);
            if (!layout) {
                LOG(ERROR) << "Skipping DFS eviction of bucket "
                           << victim->bucket_id << ": entry for key " << key
                           << " has an inconsistent layout";
                return pending;
            }
            DfsAllocatorInterface::EvictionCandidate candidate;
            candidate.key = key;
            candidate.shard_idx = static_cast<int>(victim->bucket_id);
            candidate.offset = layout->offset;
            // Byte-identical to what Allocate handed out, so the master can
            // match replica metadata field by field.
            candidate.descriptor =
                MakeBucketDescriptor(BucketDataPath(victim->bucket_id), *layout,
                                     victim->bucket_id);
            candidates.push_back(std::move(candidate));
        }

        // A frozen bucket takes no further appends, so its reserved prefix is
        // final for the rest of the transaction.
        victim->frozen = true;
        victim->sealed = true;
        RemoveFromLruLocked(victim->bucket_id);

        pending.owner_ = this;
        pending.bucket_id_ = victim->bucket_id;
        pending.bucket_identity_ = victim;
        pending.candidates_ = std::move(candidates);
    }
    return pending;
}

void ImmutableBucketAllocator::CommitEviction(PendingEviction&& pending) {
    // Detach first so the destructor of `pending` cannot abort what we commit.
    auto* owner = std::exchange(pending.owner_, nullptr);
    if (owner != this) return;

    const int64_t bucket_id = pending.bucket_id_;
    auto bucket_identity = std::move(pending.bucket_identity_);
    pending.candidates_.clear();
    if (bucket_id < 0 || !bucket_identity) return;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = buckets_.find(bucket_id);
        if (it == buckets_.end() || it->second.get() != bucket_identity.get()) {
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

    // File deletion happens last and outside the lock. A failure here leaks
    // space but can no longer produce a dangling read, because the master has
    // already dropped every replica in this bucket.
    DeleteBucketFiles(bucket_id);
    // The deleted bucket's quota slot is reusable only after its files are no
    // longer present. Wake both refill workers and allocation waiters now.
    bucket_pool_cv_.notify_all();
    LOG(INFO) << "Evicted DFS bucket " << bucket_id;
}

void ImmutableBucketAllocator::AbortEviction(PendingEviction&& pending) {
    AbortEviction(std::move(pending), /*demote=*/true);
}

void ImmutableBucketAllocator::AbortEviction(PendingEviction&& pending,
                                          bool demote) {
    auto* owner = std::exchange(pending.owner_, nullptr);
    if (owner != this) return;

    const int64_t bucket_id = pending.bucket_id_;
    auto bucket_identity = std::move(pending.bucket_identity_);
    pending.candidates_.clear();
    if (bucket_id < 0 || !bucket_identity) return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buckets_.find(bucket_id);
    if (it == buckets_.end() || it->second.get() != bucket_identity.get()) {
        return;
    }
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

}  // namespace mooncake
