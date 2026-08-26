#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "storage/distributed/bucket_entry_layout.h"
#include "storage/distributed/fs_adapter.h"
#include "storage/distributed/global_allocator_interface.h"
#include "types.h"

namespace mooncake {

struct DistributedStorageConfig;

/**
 * @brief Persisted state of one entry inside a bucket.
 *
 * `entry_offset` is the aligned start of the entry, not the value offset; the
 * value offset is always derived via BucketEntryLayout so there is a single
 * definition of the layout.
 */
struct PersistedBucketEntry {
    std::string key;
    uint64_t entry_offset = 0;
    uint64_t key_size = 0;
    uint64_t value_size = 0;
    uint64_t reserved_size = 0;
    uint64_t generation = 0;
    // 0 = PENDING (space reserved, data not known to be durable)
    // 1 = COMMITTED (data fully written; safe to recover)
    // 2 = TOMBSTONE (freed/removed; must not be revived on restart)
    int32_t state = 0;
    YLT_REFL(PersistedBucketEntry, key, entry_offset, key_size, value_size,
             reserved_size, generation, state);
};

/**
 * @brief On-disk `.meta` snapshot of one bucket.
 *
 * A CRC-32C over the serialized payload (with `checksum` zeroed) detects torn
 * or corrupt metadata; `version` rejects formats this build cannot parse.
 *
 * The snapshot is only rewritten by compaction, never once per key. `log_seq`
 * is the highest metadata-log sequence number this snapshot already accounts
 * for, so recovery can replay `bucket_NNNNNN.meta.log` on top of it while
 * discarding records the snapshot covers.
 */
struct PersistedBucketMetadata {
    uint32_t version = 0;
    uint32_t checksum = 0;
    int64_t bucket_id = 0;
    uint64_t bucket_generation = 0;
    uint64_t capacity = 0;
    uint64_t alignment = 0;
    uint64_t append_offset = 0;
    // Highest metadata-log sequence number already reflected here.
    uint64_t log_seq = 0;
    // Monotonically increasing snapshot publication generation. Compaction
    // writes the inactive `.meta.0`/`.meta.1` slot and recovery chooses the
    // valid slot with the greatest epoch, so publication never needs rename.
    uint64_t snapshot_epoch = 0;
    // Set when the bucket has been chosen for eviction and its data file is
    // being deleted. Recovery treats such a bucket as gone rather than live.
    bool evicting = false;
    std::vector<PersistedBucketEntry> entries;
    YLT_REFL(PersistedBucketMetadata, version, checksum, bucket_id,
             bucket_generation, capacity, alignment, append_offset, log_seq,
             snapshot_epoch, evicting, entries);
};

struct LegacyPersistedBucketMetadata {
    uint32_t version = 0;
    uint32_t checksum = 0;
    int64_t bucket_id = 0;
    uint64_t bucket_generation = 0;
    uint64_t capacity = 0;
    uint64_t alignment = 0;
    uint64_t append_offset = 0;
    uint64_t log_seq = 0;
    bool evicting = false;
    std::vector<PersistedBucketEntry> entries;
    YLT_REFL(LegacyPersistedBucketMetadata, version, checksum, bucket_id,
             bucket_generation, capacity, alignment, append_offset, log_seq,
             evicting, entries);
};

// Bump when the layout of PersistedBucketMetadata changes incompatibly.
// Version 3 publishes snapshots through alternating stable slots and therefore
// never needs a metadata rename.
inline constexpr uint32_t kBucketMetadataVersion = 3;
inline constexpr uint32_t kLegacyBucketMetadataVersion = 2;

enum class BucketEntryState : int32_t {
    PENDING = 0,
    COMMITTED = 1,
    TOMBSTONE = 2,
};

/**
 * @brief Kind of metadata delta recorded in `bucket_NNNNNN.meta.log`.
 *
 * All three are idempotent, last-writer-wins updates of one (key, entry_offset)
 * slot, which is what lets recovery replay a log that partially overlaps the
 * snapshot it is replayed onto.
 */
enum class MetaLogOp : uint32_t {
    ADD_PENDING = 0,     // space reserved, data not yet known durable
    MARK_COMMITTED = 1,  // client reported the data write finished
    TOMBSTONE = 2,       // entry freed; must not be revived on restart
};

const char* ToString(MetaLogOp op);

/**
 * @brief In-memory form of one metadata-log record.
 *
 * Records are produced under the allocator lock, which assigns `seq`, and are
 * written to disk outside it. `seq` is globally monotonic, so it - not file
 * order - is what defines how recovery orders the replay.
 */
struct MetaLogRecord {
    MetaLogOp op = MetaLogOp::ADD_PENDING;
    uint64_t seq = 0;
    int64_t bucket_id = -1;
    uint64_t bucket_generation = 0;
    int64_t timestamp_ns = 0;
    uint64_t entry_offset = 0;
    uint64_t key_size = 0;
    uint64_t value_size = 0;
    uint64_t reserved_size = 0;
    uint64_t entry_generation = 0;
    // The bucket's append offset right after this operation was applied in
    // memory. Recovery takes the maximum over all replayed records, so the
    // append cursor can never end up inside data an earlier record reserved.
    uint64_t append_offset = 0;
    std::string key;
};

/**
 * @brief Serialized metadata-log record framing.
 *
 * Layout (little-endian, byte offsets), a fixed header followed by the key:
 *   0  magic             u32  kMetaLogMagic, also the resynchronization anchor
 *   4  crc               u32  CRC-32C over bytes [8, record_size)
 *   8  record_size       u32  total record length, header included
 *   12 op                u32
 *   16 seq               u64
 *   24 bucket_id         i64
 *   32 bucket_generation u64
 *   40 timestamp_ns      i64
 *   48 entry_offset      u64
 *   56 key_size          u64
 *   64 value_size        u64
 *   72 reserved_size     u64
 *   80 entry_generation  u64
 *   88 append_offset     u64
 *   96 key bytes         key_size bytes
 *
 * The CRC is per record rather than over the whole log, so one damaged record
 * costs exactly that record: replay drops it and resynchronizes on the next
 * magic.
 */
inline constexpr uint32_t kMetaLogMagic = 0x474C424Du;  // "MBLG"
inline constexpr uint32_t kMetaLogHeaderSize = 96;
// Refuse absurd lengths outright so a corrupt size field cannot make recovery
// allocate wildly. Keys are bounded far below this in practice.
inline constexpr uint32_t kMetaLogMaxRecordSize = 1u << 20;

void SerializeMetaLogRecord(const MetaLogRecord& record, std::string& out);

/**
 * @brief Parse one record from `payload` starting at `pos`.
 *
 * On success `consumed` receives the record length. Returns std::nullopt when
 * the record is unusable; `consumed` then holds the number of bytes to skip,
 * which is 0 when even the framing cannot be trusted and the caller must
 * resynchronize by scanning for the next magic.
 */
std::optional<MetaLogRecord> DeserializeMetaLogRecord(std::string_view payload,
                                                      size_t pos,
                                                      size_t& consumed);

/**
 * @brief Append-only, bucket-based DFS space allocator.
 *
 * Space is handed out by appending into the currently active bucket file. Each
 * bucket has a fixed capacity; when a request no longer fits, a new bucket is
 * created. Buckets carry persisted metadata so committed entries can be
 * recovered after a master restart, and eviction works at whole-bucket
 * granularity through a two-phase protocol driven by the master.
 *
 * Metadata durability
 * -------------------
 * Each bucket owns two metadata files:
 *
 *   bucket_NNNNNN.meta.0    alternating full snapshot slots
 *   bucket_NNNNNN.meta.1    (the valid slot with the greatest epoch wins)
 *   bucket_NNNNNN.meta.log  append-only deltas produced since that snapshot
 *
 * Compaction rewrites the inactive snapshot slot and syncs it before clearing
 * the log. A torn rewrite leaves the other slot valid, so neither hot paths nor
 * compaction need rename or temporary metadata files.
 *
 * The hot path - Allocate, BatchAllocate, MarkCommitted, Free - appends one
 * fixed-shape record and fdatasyncs the log. It never rewrites the snapshot and
 * never renames, so the cost of storing a key is constant instead of
 * proportional to the number of keys already in the bucket.
 *
 * Compaction folds the log back into a fresh snapshot when the log grows past
 * `log_compaction_threshold_`, when more than half the bucket's entries are
 * tombstones, on the master's DFS maintenance tick (which also drives eviction)
 * and at shutdown. Recovery loads the snapshot, replays the log records the
 * snapshot does not already cover, and compacts immediately so the next start
 * begins from a clean snapshot.
 *
 * Lock ownership
 * --------------
 * `mutex_` guards every mutable allocator member: `buckets_`, `key_index_`,
 * `active_bucket_id_`, `next_bucket_id_`, `next_log_seq_`, `pending_log_`,
 * `lru_list_`/`lru_index_` and the eviction bookkeeping. Bucket state lives
 * behind shared_ptr and is only mutated while `mutex_` is held, so no
 * per-bucket mutex is needed.
 *
 * `log_mutex_` guards the log descriptors and their counters and serializes log
 * I/O. It is always acquired *before* `mutex_`; no path takes them in the
 * opposite order, and no path takes `log_mutex_` while holding `mutex_`.
 *
 * Slow DFS I/O (preallocation, snapshot publication, log appends, deletes) is
 * always performed with `mutex_` released: the caller snapshots the state it
 * needs under the lock, does the I/O, then reacquires the lock and re-validates
 * the bucket generation before publishing the result. No RPC, callback or
 * filesystem call ever happens while `mutex_` is held.
 */
class BucketGlobalAllocator final : public GlobalAllocatorInterface {
   public:
    BucketGlobalAllocator() = default;
    ~BucketGlobalAllocator() override;

    BucketGlobalAllocator(const BucketGlobalAllocator&) = delete;
    BucketGlobalAllocator& operator=(const BucketGlobalAllocator&) = delete;

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
     * @brief Mark a reservation as durable so restart recovery may revive it.
     *
     * Called by the master once the client reports the DFS data write finished
     * (PutEnd). A mismatching descriptor or generation is ignored, so a late
     * completion from a superseded operation cannot commit the allocation that
     * replaced it.
     *
     * @return true when the entry is COMMITTED and that fact is durable.
     */
    bool MarkCommitted(const std::string& key,
                       const DistributedFSDescriptor& descriptor);

    /**
     * @brief All committed entries recovered from disk at Init() time.
     *
     * The master re-registers these as COMPLETE DFS replicas so recovered
     * objects are queryable and readable, then clears the list.
     */
    struct RecoveredReplica {
        std::string key;
        DistributedFSDescriptor descriptor;
    };
    std::vector<RecoveredReplica> TakeRecoveredReplicas();

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
        friend class BucketGlobalAllocator;

        BucketGlobalAllocator* owner_ = nullptr;
        int64_t bucket_id_ = -1;
        uint64_t bucket_generation_ = 0;
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
     * turning low-utilization bucket tails into a permanent allocation deadlock.
     */
    PendingEviction PrepareEvictionForAllocationFailure();

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
     * @brief Make every deferred metadata delta durable and compact as needed.
     *
     * `Free()` only updates in-memory state and queues a tombstone record,
     * because the master calls it while holding a metadata shard lock and DFS
     * I/O must never happen under that lock. This method performs the deferred
     * append and then compacts any bucket that has crossed a compaction
     * trigger, so it must be called with no master lock held: the master's DFS
     * maintenance tick drives it, and the destructor runs it once more so a
     * clean shutdown leaves no unflushed tombstones.
     *
     * @return the number of buckets whose pending metadata was made durable.
     */
    size_t FlushDirtyMetadata();

    /**
     * @brief Append every queued metadata record and fdatasync the logs.
     *
     * The hot-path durability primitive: one append plus one data sync per
     * touched log, no snapshot rewrite and no rename.
     */
    tl::expected<void, ErrorCode> FlushLog();

    /**
     * @brief Fold `bucket_id`'s log back into a fresh `.meta` snapshot.
     *
     * Rewrites the inactive stable snapshot slot, then clears the log.
     * `log_mutex_` is held throughout, so no record can be appended between
     * the truncation; combined with a snapshot `log_seq` of "every sequence
     * number issued so far", everything the truncation discards is covered.
     */
    tl::expected<void, ErrorCode> CompactBucket(int64_t bucket_id);

    /**
     * @brief Compact every bucket. Returns how many were compacted.
     */
    size_t CompactAllBuckets();

    /**
     * @brief Number of buckets currently tracked (test/metrics helper).
     */
    size_t GetBucketCount() const;

    /**
     * @brief Bytes currently held in `bucket_id`'s metadata log.
     */
    uint64_t GetLogBytes(int64_t bucket_id) const;

    /**
     * @brief Log size at which a bucket becomes eligible for compaction.
     */
    uint64_t GetLogCompactionThreshold() const {
        return log_compaction_threshold_;
    }

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
        uint64_t generation = 0;
        BucketEntryState state = BucketEntryState::PENDING;
    };

    struct BucketState {
        int64_t bucket_id = 0;
        // Bumped whenever the bucket is (re)created so a stale transaction
        // cannot resolve against a different bucket that reused the id.
        uint64_t generation = 0;
        uint64_t capacity = 0;
        uint64_t append_offset = 0;
        // Bytes reserved by entries that are still live (PENDING or
        // COMMITTED). Drives the "is this bucket worth evicting" decision.
        uint64_t live_bytes = 0;
        int64_t last_access_ns = 0;
        // Set between PrepareEviction and Commit/Abort. A frozen bucket
        // accepts no new allocations and cannot be selected again.
        bool frozen = false;
        // Highest metadata-log sequence number queued for this bucket. The
        // bucket is dirty exactly while this exceeds the durable sequence
        // number tracked in `log_synced_seq_`.
        uint64_t last_queued_seq = 0;
        // Epoch of the newest durable snapshot slot. The next publication uses
        // epoch + 1 and alternates slots by parity.
        uint64_t snapshot_epoch = 0;
        // Tombstoned entries, used by the tombstone-ratio compaction trigger.
        uint64_t tombstones = 0;
        std::unordered_map<std::string, BucketEntry> entries;
    };

    using BucketPtr = std::shared_ptr<BucketState>;

    // --- helpers, called with mutex_ held only where the name says Locked ---

    std::string BucketDataPath(int64_t bucket_id) const;
    std::string BucketMetaPath(int64_t bucket_id) const;
    std::string BucketMetaSlotPath(int64_t bucket_id, int slot) const;
    std::string BucketMetaLogPath(int64_t bucket_id) const;

    // Serializes `bucket` into a PersistedBucketMetadata snapshot covering
    // every sequence number issued so far. Taken under `mutex_`; the file write
    // happens outside it.
    PersistedBucketMetadata SnapshotLocked(BucketState& bucket,
                                           bool evicting);

    // Queues one metadata delta for `bucket`, assigning it the next sequence
    // number and returning it. The record reaches disk when the log is flushed.
    uint64_t QueueLogRecordLocked(BucketState& bucket, MetaLogOp op,
                                  const std::string& key,
                                  const BucketEntry& entry);

    // Writes a complete snapshot to one of two stable metadata slots and syncs
    // it. The previous slot remains the recovery fallback; no rename is used.
    tl::expected<void, ErrorCode> PersistMetadata(
        const PersistedBucketMetadata& snapshot);

    // Ensures `seq` is durable for `bucket_id`, appending and syncing whatever
    // is still queued if it is not. Call with no lock held.
    tl::expected<void, ErrorCode> SyncLogUpTo(int64_t bucket_id, uint64_t seq);

    // Appends everything queued and syncs each touched log. Requires
    // `log_mutex_`; must not be called with `mutex_` held.
    tl::expected<void, ErrorCode> DrainPendingLogLogLocked();

    // Opens (and caches) the append fd of `bucket_id`'s log. Requires
    // `log_mutex_`.
    tl::expected<int, ErrorCode> GetOrOpenLogLogLocked(int64_t bucket_id);

    // Closes and forgets `bucket_id`'s log fd. Requires `log_mutex_`.
    void CloseLogLogLocked(int64_t bucket_id);

    // True when `bucket_id` has crossed a compaction trigger. Takes both locks
    // internally; call with neither held.
    bool ShouldCompact(int64_t bucket_id) const;

    // Compacts `bucket_id` when a trigger fired, and logs but does not
    // propagate a compaction failure: the log is still authoritative, so the
    // only cost is that it stays large a little longer.
    void MaybeCompact(int64_t bucket_id);

    // Creates a fresh bucket: allocates the id under the lock, preallocates
    // the data file and writes the initial `.meta` outside the lock, then
    // publishes the bucket. Rolls back id/state/files on any failure.
    tl::expected<BucketPtr, ErrorCode> CreateBucketUnlocked(
        std::unique_lock<std::mutex>& lock);

    // Ensures an active bucket exists with at least `required` bytes free.
    // May temporarily release `lock` to create a bucket.
    tl::expected<BucketPtr, ErrorCode> EnsureActiveBucket(
        std::unique_lock<std::mutex>& lock, uint64_t required);

    void TouchLruLocked(int64_t bucket_id, int64_t now_ns);
    void RemoveFromLruLocked(int64_t bucket_id);

    // Applies one reservation to `bucket`, queues its ADD_PENDING record and
    // returns the descriptor. `out_seq` receives the record's sequence number.
    tl::expected<DistributedFSDescriptor, ErrorCode> ReserveInBucketLocked(
        BucketState& bucket, const std::string& key, uint64_t size,
        uint64_t* out_seq);

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

    tl::expected<void, ErrorCode> RecoverFromDisk();

    // Replays `bucket_NNNNNN.meta.log` on top of an already-loaded snapshot.
    // Single-threaded recovery context only.
    void ReplayLogForRecovery(BucketState& bucket, uint64_t snapshot_log_seq,
                              uint64_t& max_seq, uint64_t& max_generation);

    void DeleteBucketFiles(int64_t bucket_id);

    uint64_t UsedBytesLocked() const;

    std::string fsdir_;
    std::string fs_adapter_type_;
    std::unique_ptr<FileSystemAdapter> fs_adapter_;

    uint64_t bucket_capacity_ = 0;
    uint64_t alignment_ = 4096;
    int64_t max_bucket_count_ = 0;
    uint64_t log_compaction_threshold_ = 0;

    bool eviction_enabled_ = true;
    double eviction_high_watermark_ = 0.9;
    double eviction_low_watermark_ = 0.7;
    std::chrono::seconds eviction_check_interval_{5};

    mutable std::mutex mutex_;
    // Serializes bucket creation. Creating a bucket releases `mutex_` for the
    // file I/O, so without this flag several threads would each reserve a
    // distinct id and race to publish, orphaning all but one and letting a
    // loser's rollback delete a winner's files.
    bool bucket_creation_in_flight_ = false;
    std::condition_variable bucket_creation_cv_;
    std::unordered_map<int64_t, BucketPtr> buckets_;
    std::unordered_map<std::string, int64_t> key_index_;
    int64_t next_bucket_id_ = 0;
    int64_t active_bucket_id_ = -1;
    uint64_t next_generation_ = 1;
    // Metadata deltas produced but not yet appended, in sequence order, and
    // the globally monotonic counter that orders them.
    std::vector<MetaLogRecord> pending_log_;
    uint64_t next_log_seq_ = 1;
    // Once the high watermark is crossed, keep evicting until usage falls
    // below the low watermark; protected buckets may make that span several
    // prepare/resolve rounds.
    bool eviction_active_ = false;

    std::list<int64_t> lru_list_;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> lru_index_;

    std::vector<RecoveredReplica> recovered_replicas_;

    // Guards the log descriptors and counters below and serializes log I/O.
    // Always acquired before `mutex_`, never while holding it.
    mutable std::mutex log_mutex_;
    std::unordered_map<int64_t, int> log_fds_;
    std::unordered_map<int64_t, uint64_t> log_bytes_;
    // Highest sequence number that is durable for each bucket, either because
    // it reached the log or because a published snapshot covers it.
    std::unordered_map<int64_t, uint64_t> log_synced_seq_;

    std::atomic<bool> initialized_{false};
};

}  // namespace mooncake
