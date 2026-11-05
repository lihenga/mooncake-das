// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "file_storage.h"

#include <algorithm>
#include <memory>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "aligned_client_buffer.h"
#include "storage_backend.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "client_metric.h"
#include "utils.h"
#include "device/accelerator_registry.h"
#ifdef USE_URING
#include "file_interface.h"
#endif

namespace mooncake {

namespace {
bool UseMappedPinnedRestoreArena() {
#if defined(USE_HYGON)
    return true;
#else
    return false;
#endif
}

}  // namespace

void FileStorage::InitializeUnifiedDfsBuffer() {
    if (config_.enable_dfs && config_.unified_buffer_size > 0) {
        const size_t arena_size =
            static_cast<size_t>(config_.unified_buffer_size);
        bool mapped_required = UseMappedPinnedRestoreArena();
        auto buffer =
            PinnedBufferPool::AllocatePinned(arena_size, mapped_required);
        if (mapped_required && (!buffer.pinned_host.addr || !buffer.mapped ||
                                !buffer.device_data)) {
            LOG(WARNING)
                << "Mapped unified DFS buffer allocation failed; falling "
                   "back to pinned host memory, size="
                << arena_size;
            buffer = PinnedBufferPool::AllocatePinned(arena_size, false);
            mapped_required = false;
        }
        if (buffer.pinned_host.addr &&
            (!mapped_required ||
             (buffer.mapped && buffer.device_data != nullptr))) {
            pinned_restore_arena_ =
                std::make_shared<PinnedBufferPool::Buffer>(std::move(buffer));
            pinned_restore_arena_allocator_ = ClientBufferAllocator::create(
                pinned_restore_arena_->data, pinned_restore_arena_->capacity,
                client_ ? client_->GetProtocol() : "",
                pinned_restore_arena_->device_data, pinned_restore_arena_,
                usage_tracker_, DfsBufferUsageKind::kRead);
            const auto dfs_alignment =
                GetEnvOr<size_t>("MOONCAKE_DFS_ALIGNMENT", 4096);
            const auto read_chunk_allocation_bytes =
                pinned_restore_arena_allocator_
                    ->allocation_size_for_aligned_request(
                        static_cast<size_t>(
                            std::max<int64_t>(config_.read_chunk_size, 0)),
                        dfs_alignment);
            const auto write_chunk_allocation_bytes =
                pinned_restore_arena_allocator_
                    ->allocation_size_for_aligned_request(
                        static_cast<size_t>(
                            std::max<int64_t>(config_.write_chunk_size, 0)),
                        dfs_alignment);
            const auto read_min_protection_bytes =
                std::min(read_chunk_allocation_bytes,
                         pinned_restore_arena_->capacity / 2);
            if (read_chunk_allocation_bytes > read_min_protection_bytes) {
                LOG(WARNING)
                    << "DFS read free reserve was clamped to half of the "
                       "unified buffer: requested_read_min_protection_bytes="
                    << read_chunk_allocation_bytes
                    << ", effective_read_min_protection_bytes="
                    << read_min_protection_bytes << ", unified_capacity_bytes="
                    << pinned_restore_arena_->capacity;
            }
            // Carve one write chunk out of the prefetch budget so prefetch can
            // never fill the whole non-read region and leave DFS write staging
            // without space. This is a configuration-time reservation only:
            // writes wait for space (bounded by
            // MC_STORE_DFS_WRITE_ARENA_WAIT_MS, default 10s) instead of failing
            // immediately, so once reads release memory the write proceeds. No
            // per-request write protection is therefore needed in the admission
            // path.
            const auto write_min_protection_bytes = std::min(
                write_chunk_allocation_bytes,
                pinned_restore_arena_->capacity - read_min_protection_bytes);
            const auto prefetch_limit_bytes = static_cast<uint64_t>(
                static_cast<double>(pinned_restore_arena_->capacity) *
                config_.prefetch_capacity_ratio);
            const auto admission_limits = usage_tracker_->ConfigureAdmission(
                read_min_protection_bytes, write_min_protection_bytes,
                prefetch_limit_bytes);
            if (prefetch_limit_bytes > admission_limits.prefetch_limit_bytes) {
                LOG(WARNING)
                    << "DFS prefetch capacity was truncated to leave room for "
                       "one read chunk and one write chunk: "
                       "requested_prefetch_limit_bytes="
                    << prefetch_limit_bytes
                    << ", effective_prefetch_limit_bytes="
                    << admission_limits.prefetch_limit_bytes
                    << ", read_min_protection_bytes="
                    << admission_limits.read_min_protection_bytes
                    << ", write_min_protection_bytes="
                    << admission_limits.write_min_protection_bytes
                    << ", unified_capacity_bytes="
                    << pinned_restore_arena_->capacity
                    << (admission_limits.prefetch_limit_bytes == 0
                            ? "; prefetch is disabled, increase "
                              "MC_STORE_DFS_UNIFIED_BUFFER_SIZE_BYTES or "
                              "reduce "
                              "the read/write chunk sizes"
                            : "");
            }
            LOG(INFO) << "Initialized unified DFS buffer from restore arena, "
                      << "size=" << pinned_restore_arena_->capacity
                      << ", mapped=" << pinned_restore_arena_->mapped
                      << ", read_min_protection_bytes="
                      << admission_limits.read_min_protection_bytes
                      << ", write_min_protection_bytes="
                      << admission_limits.write_min_protection_bytes
                      << ", prefetch_limit_bytes="
                      << admission_limits.prefetch_limit_bytes;
            const auto validate_chunk = [this, read_min_protection_bytes,
                                         dfs_alignment](int64_t chunk_size,
                                                        const char* name,
                                                        bool is_write_chunk) {
                if (chunk_size <= 0) return;
                const auto requested_size = static_cast<size_t>(chunk_size);
                const auto allocation_size =
                    pinned_restore_arena_allocator_
                        ->allocation_size_for_aligned_request(requested_size,
                                                              dfs_alignment);
                const auto available_capacity =
                    is_write_chunk ? pinned_restore_arena_->capacity -
                                         read_min_protection_bytes
                                   : pinned_restore_arena_->capacity;
                if (allocation_size == 0 ||
                    allocation_size > available_capacity) {
                    LOG(ERROR)
                        << "Unified DFS buffer is too small for " << name
                        << ": chunk_size=" << chunk_size
                        << ", required_allocation_size=" << allocation_size
                        << ", capacity=" << pinned_restore_arena_->capacity;
                    throw std::invalid_argument(
                        "Unified DFS buffer is too small for DFS chunk size");
                }
            };
            validate_chunk(config_.read_chunk_size, "read chunk", false);
            validate_chunk(config_.write_chunk_size, "write chunk", true);
        } else {
            LOG(ERROR) << "Failed to allocate required pinned/mapped unified "
                       << "DFS buffer, size=" << arena_size;
            // Failing to allocate a configured unified buffer leaves no DFS
            // staging capacity at all: every read and write would fail at
            // request time instead of at startup. Fail fast so the
            // misconfiguration or memory shortage is visible immediately.
            throw std::runtime_error(
                "Unified DFS buffer allocation failed while "
                "MC_STORE_DFS_UNIFIED_BUFFER_SIZE_BYTES is non-zero");
        }
    } else if (config_.enable_dfs && config_.unified_buffer_size == 0) {
        // DFS is on but has no staging space. Reads and writes still work for
        // host data, but every path that needs a unified allocation degrades:
        // async DFS writes keep only the memory replica, and a synchronous
        // write of an object holding GPU slices loses its DFS replica. Warn
        // because the configuration looks valid yet silently weakens DFS.
        LOG(WARNING) << "Unified DFS buffer disabled by configuration "
                        "(MC_STORE_DFS_UNIFIED_BUFFER_SIZE_BYTES=0): DFS "
                        "writes have no staging space, so async DFS writes "
                        "keep only memory replicas and GPU-slice synchronous "
                        "writes lose their DFS replica; only host-only "
                        "synchronous writes still reach DFS";
    }
}

std::optional<BufferHandle> FileStorage::AllocatePinnedStagingBuffer(
    size_t size, size_t alignment) const {
    if (!pinned_restore_arena_allocator_) {
        LOG(ERROR) << "DFS_UNIFIED_BUFFER_ALLOCATION_FAILED kind=read size="
                   << size << " alignment=" << alignment;
        return std::nullopt;
    }
    auto allocation = pinned_restore_arena_allocator_->allocate_aligned(
        size, alignment, DfsBufferUsageKind::kRead);
    if (!allocation) {
        LOG(ERROR) << "DFS_UNIFIED_BUFFER_ALLOCATION_FAILED kind=read size="
                   << size << " alignment=" << alignment;
    }
    return allocation;
}

std::optional<BufferHandle> FileStorage::AllocatePinnedPrefetchBuffer(
    size_t size, size_t alignment) const {
    if (!pinned_restore_arena_allocator_ || size == 0 || alignment == 0) {
        LOG(ERROR) << "DFS_UNIFIED_BUFFER_ALLOCATION_FAILED kind=prefetch size="
                   << size << " alignment=" << alignment;
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(unified_prefetch_mutex_);
    auto allocation = pinned_restore_arena_allocator_->allocate_aligned(
        size, alignment, DfsBufferUsageKind::kPrefetch);
    if (!allocation) {
        LOG(ERROR) << "DFS_UNIFIED_BUFFER_ALLOCATION_FAILED kind=prefetch size="
                   << size << " alignment=" << alignment;
    }
    return allocation;
}

offset_allocator::OffsetAllocStorageReport FileStorage::UnifiedBufferReport()
    const {
    if (!pinned_restore_arena_allocator_) return {0, 0};
    return pinned_restore_arena_allocator_->storageReport();
}

tl::expected<void, ErrorCode> FileStorage::InitDfs() {
    dfs_heartbeat_interval_seconds_.store(300, std::memory_order_release);
    LOG(INFO) << "DFS storage mode enabled; starting DFS heartbeat thread";
    dfs_heartbeat_running_.store(true);
    dfs_heartbeat_thread_ =
        std::thread(&FileStorage::DfsHeartbeatThreadFunc, this);

    client_buffer_gc_running_.store(true);
    client_buffer_gc_thread_ =
        std::thread(&FileStorage::ClientBufferGCThreadFunc, this);
    return {};
}

tl::expected<void, ErrorCode> FileStorage::ProbeDfsRead() {
    if (!distributed_backend_) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }
    return distributed_backend_->ProbeRead();
}

tl::expected<void, ErrorCode> FileStorage::ProbeDfsWrite() {
    if (!distributed_backend_) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }
    return distributed_backend_->ProbeWrite();
}

bool FileStorage::IsDfsBackendReady() const {
    return distributed_backend_ && distributed_backend_->IsInitialized();
}

void FileStorage::DfsHeartbeatThreadFunc() {
    const auto initial_delay_seconds = generate_uuid().first % 6;
    for (uint64_t elapsed = 0;
         elapsed < initial_delay_seconds && dfs_heartbeat_running_.load();
         ++elapsed) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    LOG(INFO) << "Starting DFS heartbeat thread with interval="
              << dfs_heartbeat_interval_seconds_.load(std::memory_order_acquire)
              << "s";
    while (dfs_heartbeat_running_.load()) {
        if (distributed_backend_) {
            auto snapshot = client_->GetDfsBucketSnapshot(
                dfs_snapshot_epoch_, dfs_snapshot_version_);
            if (!snapshot) {
                LOG(WARNING) << "DFS bucket snapshot request failed: "
                             << snapshot.error();
            } else {
                auto epoch_guard =
                    distributed_backend_->AcquireBucketEpoch(snapshot->epoch);
                if (!epoch_guard) {
                    LOG(WARNING) << "Ignoring DFS snapshot from an unavailable "
                                    "or superseded Master epoch: "
                                 << epoch_guard.error();
                } else {
                    client_->SetDfsBatchTimeoutFixedMs(
                        snapshot->batch_timeout_fixed_ms);
                    const uint32_t heartbeat_interval = std::max<uint32_t>(
                        snapshot->heartbeat_interval_seconds, 1);
                    const uint32_t previous_interval =
                        dfs_heartbeat_interval_seconds_.exchange(
                            heartbeat_interval, std::memory_order_acq_rel);
                    if (previous_interval != heartbeat_interval) {
                        LOG(INFO) << "Updated DFS heartbeat interval from "
                                  << previous_interval << "s to "
                                  << heartbeat_interval << "s";
                    }
                    if (!snapshot->unchanged) {
                        dfs_snapshot_epoch_ = snapshot->epoch;
                        dfs_snapshot_version_ = snapshot->version;
                        auto result =
                            distributed_backend_->CleanupStaleBucketFds(
                                snapshot->bucket_ids);
                        if (!result) {
                            LOG(WARNING)
                                << "DFS FD cleanup failed: " << result.error();
                        } else if (*result > 0) {
                            LOG(INFO) << "DFS FD cleanup removed " << *result
                                      << " stale bucket handles";
                        }
                    }
                }
            }
        }
        const uint32_t heartbeat_interval = std::max<uint32_t>(
            dfs_heartbeat_interval_seconds_.load(std::memory_order_acquire), 1);
        for (uint32_t elapsed = 0;
             elapsed < heartbeat_interval && dfs_heartbeat_running_.load();
             ++elapsed) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

}  // namespace mooncake
