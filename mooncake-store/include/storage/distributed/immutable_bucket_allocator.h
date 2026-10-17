#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "storage/distributed/bucket_entry_layout.h"
#include "storage/distributed/fs_adapter.h"
#include "storage/distributed/dfs_allocator_interface.h"
#include "types.h"

namespace mooncake {

struct DistributedStorageConfig;

/**
 * @brief In-memory state of one reservation inside a bucket.
 *
 * A reservation is created PENDING by `Allocate`/`BatchAllocate` and stays
 * PENDING: bucket metadata is never persisted, so there is no commit step to
 * record. `Free` turns it into a TOMBSTONE, which keeps the reserved bytes
 * accounted for until the whole bucket is evicted.
 */
enum class BucketEntryState : int32_t {
    PENDING = 0,
    TOMBSTONE = 2,
};

/**
 * @brief Append-only, bucket-based DFS space allocator.
 *
 * Space is handed out by appending into the currently active bucket file. Each
 * bucket has a fixed capacity; when a request no longer fits, a new bucket is
 * created. Eviction works at whole-bucket granularity through a two-phase
 * protocol driven by the master.
 *
 * No metadata durability
 * ----------------------
 * Bucket state lives in memory only. A bucket owns exactly one file:
 *
 *   bucket_NNNNNN.data     preallocated data file
 *
 * Nothing describes what is inside that file, so a master restart loses every
 * bucket: `Init()` deletes every `bucket_*.data` it finds (together with any
 * `bucket_*.meta` left behind by an older build) and allocation starts from an
 * empty table with bucket ids counting up from 0 again. The DFS tier behaves
 * as a pure cache that is cold after a restart.
 *
 * This is the deliberate trade-off: without a per-key metadata snapshot the
 * put path performs no metadata I/O at all and the allocator never keeps a
 * second copy of every key for durability, which is what made large key counts
 * expensive in both time and memory.
 *
 * Lock ownership
 * --------------
 * `mutex_` guards every mutable allocator member: `buckets_`, `key_index_`,
 * `active_bucket_id_`, `next_bucket_id_`, `lru_list_`/`lru_index_` and the
 * eviction bookkeeping. Bucket state lives behind shared_ptr and is only
 * mutated while `mutex_` is held, so no per-bucket mutex is needed.
 *
 * Slow DFS I/O (preallocation, deletes) is always performed with `mutex_`
 * released: the caller snapshots the state it needs under the lock, does the
 * I/O, then reacquires the lock and re-validates the captured BucketState
 * object identity before publishing the result. No RPC, callback or filesystem
 * call ever happens while `mutex_` is held.
 */
class ImmutableBucketAllocator final : public DfsAllocatorInterface {
   public:
    ImmutableBucketAllocator() = default;
    ~ImmutableBucketAllocator() override;

    ImmutableBucketAllocator(const ImmutableBucketAllocator&) = delete;
    ImmutableBucketAllocator& operator=(const ImmutableBucketAllocator&) = delete;

    DfsAllocatorType Type() const override { return DfsAllocatorType::BUCKET; }

    tl::expected<void, ErrorCode> Init(
        const DistributedStorageConfig& config) override;

    bool IsInitialized() const override {
        return initialized_.load(std::memory_order_acquire);
    }

    tl::expected<DistributedFSDescriptor, ErrorCode> Allocate(
        const std::string& key, uint64_t size) override;

    std::vector<BatchAllocateResult> BatchAllocate(
        const std::vector<BatchAllocateRequest>& requests) override;

    void Free(const std::string& key,
              const DistributedFSDescriptor& descriptor) override;

    void UpdateAccess(const std::string& key,
                      const DistributedFSDescriptor& descriptor) override;

    bool IsEvictionEnabled() const override { return eviction_enabled_; }

    std::chrono::seconds GetEvictionCheckInterval() const override {
        return eviction_check_interval_;
    }

    uint64_t GetTotalCapacity() const override;

    uint64_t GetUsedBytes() const override;

    /**
     * @brief Dynamically change the maximum number of buckets at runtime.
     *
     * A value of 0 means unlimited. The new limit takes effect on the next
     * allocation: no new bucket is created once the count is reached. It does
     * not force eviction of already-existing buckets, so it can be set below
     * the current bucket count without immediate effect.
     *
     * @return the previous value of max_bucket_count_.
     */
    int64_t SetMaxBucketCount(int64_t new_max_bucket_count);

    /**
     * @brief Bucket-granular eviction transaction.
     *
     * Prepare() freezes one non-active bucket and lists its live entries. The
     * master validates every candidate without mutating metadata; only if all
     * are acceptable does it Commit(), which removes the replicas and then
     * deletes the bucket files. Any rejection leads to Abort(), which restores
     * the bucket untouched.
     *
     * Move-only, and Commit/Abort are each idempotent: whichever runs first
     * wins and the destructor aborts an unresolved transaction.
     */
    class PendingEviction {
       public:
        PendingEviction() = default;
        ~PendingEviction();

        PendingEviction(const PendingEviction&) = delete;
        PendingEviction& operator=(const PendingEviction&) = delete;
        PendingEviction(PendingEviction&& other) noexcept;
        PendingEviction& operator=(PendingEviction&& other) noexcept;

        bool Empty() const { return candidates_.empty(); }
        int64_t bucket_id() const { return bucket_id_; }
        const std::vector<EvictionCandidate>& Candidates() const {
            return candidates_;
        }

       private:
        friend class ImmutableBucketAllocator;

        ImmutableBucketAllocator* owner_ = nullptr;
        int64_t bucket_id_ = -1;
        // Keeps the exact BucketState captured by PrepareEviction alive. The
        // allocator compares pointer identity before Commit/Abort can act.
        std::shared_ptr<void> bucket_identity_;
        std::vector<EvictionCandidate> candidates_;
    };

    /**
     * @brief Freeze the coldest evictable bucket and return its live entries.
     * Returns an empty transaction when nothing can be evicted.
     */
    PendingEviction PrepareEviction();

    /**
     * @brief Prepare one cold bucket regardless of byte watermarks.
     *
     * Used only after allocation reports that the bucket-count limit has been
     * reached. It allows the master to reclaim one bucket and retry without
     * turning low-utilization bucket tails into a permanent allocation
     * deadlock.
     */
    PendingEviction PrepareEvictionForAllocationFailure();

    /**
     * @brief Freeze one specific bucket after its data file is confirmed missing.
     */
    PendingEviction PrepareInvalidation(int64_t bucket_id);

    bool HasBucket(int64_t bucket_id) const;

    /**
     * @brief Accept the eviction: drop the bucket and delete its files.
     * Must only be called once the master has removed every candidate replica.
     */
    void CommitEviction(PendingEviction&& pending);

    /**
     * @brief Reject the eviction and return the bucket to service unchanged.
     *
     * Used when the master declined the candidates. The bucket goes back at the
     * warm end of the LRU so the next round can reach a different candidate
     * instead of re-offering this one forever.
     */
    void AbortEviction(PendingEviction&& pending);

    /**
     * @brief Number of buckets currently tracked (test/metrics helper).
     */
    size_t GetBucketCount() const;

    /**
     * @brief Bucket id an existing key currently lives in, if any.
     */
    std::optional<int64_t> GetBucketIdForKey(const std::string& key) const;

    static std::string FormatBucketId(int64_t bucket_id);

   private:
    friend class PendingEviction;

    // Shared implementation for watermark-driven and allocation-failure-driven
    // eviction. The latter bypasses only the watermark gate and still observes
    // active/frozen state plus the master's full validation protocol.
    PendingEviction PrepareEvictionInternal(bool force_one);
    PendingEviction PrepareInvalidationInternal(int64_t bucket_id);

    // Shared implementation of AbortEviction. `demote` distinguishes an
    // explicit master rejection (return the bucket at the warm end so the scan
    // can move on) from a transaction dropped without a verdict by the
    // destructor or move assignment (restore its cold position).
    void AbortEviction(PendingEviction&& pending, bool demote);

    struct BucketEntry {
        uint64_t entry_offset = 0;
        uint64_t key_size = 0;
        uint64_t value_size = 0;
        uint64_t reserved_size = 0;
        BucketEntryState state = BucketEntryState::PENDING;
    };

    struct BucketState {
        int64_t bucket_id = 0;
        uint64_t capacity = 0;
        uint64_t append_offset = 0;
        // Bytes reserved by entries that are still live (not TOMBSTONE).
        // Drives the "is this bucket worth evicting" decision.
        uint64_t live_bytes = 0;
        int64_t last_access_ns = 0;
        // Set between PrepareEviction and Commit/Abort. A frozen bucket
        // accepts no new allocations and cannot be selected again.
        bool frozen = false;
        // A ready bucket owns a fully preallocated data file but has never
        // accepted an allocation. Ready buckets are absent from the LRU.
        bool ready = false;
        // True once the bucket has been sealed: it is no longer the active
        // bucket and accepts no further allocations. Remains meaningful for
        // space accounting and eviction even though nothing is persisted.
        bool sealed = false;
        // Tombstoned entries, kept for eviction bookkeeping.
        uint64_t tombstones = 0;
        std::unordered_map<std::string, BucketEntry> entries;
    };

    using BucketPtr = std::shared_ptr<BucketState>;

    struct BucketCreationReservation {
        int64_t bucket_id = -1;
    };

    // --- helpers, called with mutex_ held only where the name says Locked ---

    std::string BucketDataPath(int64_t bucket_id) const;
    std::string BucketMetaPath(int64_t bucket_id) const;

    // Seals the currently active bucket - it stops accepting allocations.
    // Requires `mutex_`.
    void SealActiveBucketLocked();

    // Synchronous compatibility path used only when ready_bucket_target_ is 0.
    // It reserves identity under the lock, preallocates outside it, then
    // publishes the bucket as active without rolling back the monotonic id.
    tl::expected<BucketPtr, ErrorCode> CreateBucketUnlocked(
        std::unique_lock<std::mutex>& lock);

    bool CanReserveBucketCreationLocked() const;
    std::optional<BucketCreationReservation> ReserveBucketCreationLocked();
    void FinishBucketCreationLocked(bool synchronous);
    void RollbackBucketPublicationLocked(int64_t bucket_id,
                                         const BucketPtr& bucket);
    size_t EffectiveReadyTargetLocked() const;
    BucketPtr PromoteReadyBucketLocked();
    bool ActiveBucketHasSpaceLocked(uint64_t required,
                                    BucketPtr* bucket = nullptr) const;
    void RequestRefillLocked(bool urgent = false);
    void BucketCreateWorker();
    void StopBucketWorkers();
    std::vector<int64_t> DetachReadyBucketsLocked();
    void UpdateBucketPoolMetricsLocked() const;

    // Ensures an active bucket exists with at least `required` bytes free.
    // May temporarily release `lock` to create a bucket.
    tl::expected<BucketPtr, ErrorCode> EnsureActiveBucket(
        std::unique_lock<std::mutex>& lock, uint64_t required);

    void TouchLruLocked(int64_t bucket_id, int64_t now_ns);
    void RemoveFromLruLocked(int64_t bucket_id);

    // Applies one reservation to `bucket` and returns the descriptor.
    tl::expected<DistributedFSDescriptor, ErrorCode> ReserveInBucketLocked(
        BucketState& bucket, const std::string& key, uint64_t size);

    // Rolls a reservation back out of `bucket` (used when a later entry of the
    // same batch fails). Only valid while the reservation is the most recent
    // one, which BatchAllocate guarantees by unwinding in reverse order.
    void UnreserveInBucketLocked(BucketState& bucket, const std::string& key,
                                 const DistributedFSDescriptor& descriptor);

    // Matches `descriptor` against the recorded entry for `key`.
    // Returns nullptr when the descriptor is stale.
    BucketEntry* FindMatchingEntryLocked(const std::string& key,
                                         const DistributedFSDescriptor& desc,
                                         BucketPtr* out_bucket);

    /**
     * @brief Delete every leftover bucket file from a previous process.
     *
     * Called once by `Init()` before any bucket is preallocated. Bucket data is
     * only addressable through in-memory state, so anything a previous master
     * left behind is unreachable garbage; a `bucket_*.meta` written by an older
     * build goes the same way. Only names matching `bucket_<id>.data` or
     * `bucket_<id>.meta` are touched - other files in the directory belong to
     * someone else.
     */
    tl::expected<void, ErrorCode> PurgeExistingBucketFiles();

    void DeleteBucketFiles(int64_t bucket_id);

    uint64_t UsedBytesLocked() const;

    std::string fsdir_;
    std::string fs_adapter_type_;
    std::unique_ptr<FileSystemAdapter> fs_adapter_;

    uint64_t bucket_capacity_ = 0;
    uint64_t alignment_ = 4096;
    int64_t max_bucket_count_ = 0;
    size_t ready_bucket_target_ = 4;
    size_t bucket_create_concurrency_ = 2;

    bool eviction_enabled_ = true;
    double eviction_high_watermark_ = 0.9;
    double eviction_low_watermark_ = 0.7;
    std::chrono::seconds eviction_check_interval_{5};

    mutable std::mutex mutex_;
    std::deque<int64_t> ready_bucket_ids_;
    size_t bucket_creations_in_flight_ = 0;
    size_t waiting_allocators_ = 0;
    bool synchronous_creation_in_flight_ = false;
    bool stop_bucket_workers_ = false;
    size_t initialization_creations_remaining_ = 0;
    uint64_t bucket_creation_failure_sequence_ = 0;
    uint64_t last_urgent_refill_sequence_ = static_cast<uint64_t>(-1);
    ErrorCode last_bucket_creation_error_ = ErrorCode::OK;
    std::chrono::steady_clock::time_point bucket_creation_retry_after_{};
    std::condition_variable bucket_pool_cv_;
    std::vector<std::thread> bucket_create_workers_;
    std::unordered_map<int64_t, BucketPtr> buckets_;
    std::unordered_map<std::string, int64_t> key_index_;
    // Always starts at 0: a restart begins with an empty table, so there is no
    // previous id to continue from and no leftover file to collide with.
    int64_t next_bucket_id_ = 0;
    int64_t active_bucket_id_ = -1;
    // Once the high watermark is crossed, keep evicting until usage falls
    // below the low watermark; protected buckets may make that span several
    // prepare/resolve rounds.
    bool eviction_active_ = false;

    std::list<int64_t> lru_list_;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> lru_index_;

    std::atomic<bool> initialized_{false};
};

}  // namespace mooncake
