// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "client_service.h"
#include "transfer_task.h"

namespace mooncake {
namespace l4 {

[[nodiscard]] inline size_t CalculateSliceSize(
    const std::vector<Slice>& slices) {
    size_t slice_size = 0;
    for (const auto& slice : slices) {
        slice_size += slice.size;
    }
    return slice_size;
}

struct ReplicaTransferSummary {
    size_t allocated_memory_replicas = 0;
    size_t allocated_nof_replicas = 0;
    size_t allocated_dfs_replicas = 0;
    size_t successful_memory_transfers = 0;
    size_t successful_nof_transfers = 0;
    size_t successful_dfs_transfers = 0;
    size_t failed_memory_transfers = 0;
    size_t failed_nof_transfers = 0;
    size_t failed_dfs_transfers = 0;
    ErrorCode first_error = ErrorCode::OK;
    // Set when the DFS replica's PutEnd/PutRevoke has been handed to the
    // background write task (BUCKET mode). FinalizeBatchPut must then leave the
    // DFS replica alone: sending its own End would publish data that has not
    // been written yet, and a Revoke would fight the task's completion.
    bool dfs_completion_deferred = false;

    void RecordAllocatedReplica(const Replica::Descriptor& replica) {
        if (replica.is_memory_replica()) {
            ++allocated_memory_replicas;
        } else if (replica.is_nof_replica()) {
            ++allocated_nof_replicas;
        } else if (replica.is_dfs_replica()) {
            ++allocated_dfs_replicas;
        }
    }

    void RecordSuccess(ReplicaType replica_type) {
        if (replica_type == ReplicaType::MEMORY) {
            ++successful_memory_transfers;
        } else if (replica_type == ReplicaType::NOF_SSD) {
            ++successful_nof_transfers;
        } else if (replica_type == ReplicaType::DFS) {
            ++successful_dfs_transfers;
        }
    }

    void RecordFailure(ReplicaType replica_type, ErrorCode error) {
        if (replica_type == ReplicaType::MEMORY) {
            ++failed_memory_transfers;
        } else if (replica_type == ReplicaType::NOF_SSD) {
            ++failed_nof_transfers;
        } else if (replica_type == ReplicaType::DFS) {
            ++failed_dfs_transfers;
        }
        if (first_error == ErrorCode::OK) {
            first_error = error;
        }
    }
};

bool NonDfsTransfersSucceeded(const ReplicaTransferSummary& summary);

bool AllAllocatedTransfersSucceeded(const ReplicaTransferSummary& summary);

bool HasExpectedReplicaAllocation(const ReplicateConfig& config,
                                  const ReplicaTransferSummary& summary,
                                  bool allow_missing_dfs = false);

std::optional<ErrorCode> ValidatePreferSameNodeWriteConfig(
    const ReplicateConfig& config);

// success describes whether the overall put should succeed. Reliable modes
// require all allocated replicas to complete. Flexible dual-replica mode only
// requires one replica type to succeed, so success may be true while
// revoke_type is also set for the failed side.
struct FinalizeDecision {
    std::optional<ReplicaType> end_type;
    std::optional<ReplicaType> revoke_type;
    bool success = false;
    ErrorCode error = ErrorCode::OK;
};

FinalizeDecision DetermineFinalizeDecision(
    const ReplicateConfig& config, const ReplicaTransferSummary& summary,
    bool allow_missing_dfs = false, bool allow_memory_only_fallback = false);

}  // namespace l4

enum class PutOperationState {
    PENDING,
    MASTER_FAILED,
    TRANSFER_FAILED,
    FINALIZE_FAILED,
    SUCCESS
};

class PutOperation {
   public:
    struct PendingTransferRecord {
        ReplicaType replica_type;
        TransferFuture future;

        PendingTransferRecord(ReplicaType type,
                              TransferFuture&& transfer_future)
            : replica_type(type), future(std::move(transfer_future)) {}
    };

    PutOperation(std::string_view k, const std::vector<Slice>& s)
        : key(k), slices(s) {
        // Initialize with a pending error state to ensure result is always set
        result = tl::unexpected(ErrorCode::INTERNAL_ERROR);
    }

    std::string key;
    std::vector<Slice> slices;
    std::optional<uint64_t> object_checksum;
    std::vector<std::vector<Slice>> batched_slices;

    // Enhanced state tracking
    PutOperationState state = PutOperationState::PENDING;
    tl::expected<void, ErrorCode> result;
    std::vector<Replica::Descriptor> replicas;
    std::vector<PendingTransferRecord> pending_transfers;

    size_t requested_memory_replicas = 0;
    size_t requested_nof_replicas = 0;
    size_t requested_dfs_replicas = 0;
    bool service_dual_write_eligible = false;
    bool service_metric_recorded = false;
    bool service_metric_excluded = false;
    l4::ReplicaTransferSummary transfer_summary;

    // Error context for debugging
    std::optional<std::string> failure_context;
    // Helper methods for robust state management
    void SetSuccess() {
        state = PutOperationState::SUCCESS;
        result = {};
        failure_context.reset();
    }

    void SetTerminalError(ErrorCode error, PutOperationState terminal_state,
                          const std::string& context = "") {
        state = terminal_state;
        result = tl::unexpected(error);
        if (!context.empty()) {
            failure_context = context;
        }
    }

    void SetError(ErrorCode error, const std::string& context = "") {
        result = tl::unexpected(error);
        if (!context.empty()) {
            failure_context = toString(error) + ": " + context + "; " +
                              failure_context.value_or("");
        }

        // Update state based on current processing stage
        if (replicas.empty()) {
            state = PutOperationState::MASTER_FAILED;
        } else if (pending_transfers.empty()) {
            state = PutOperationState::TRANSFER_FAILED;
        } else {
            state = PutOperationState::FINALIZE_FAILED;
        }
    }

    void AppendFailureContext(const std::string& context) {
        if (context.empty()) {
            return;
        }
        if (!failure_context.has_value()) {
            failure_context = context;
            return;
        }
        failure_context = *failure_context + "; " + context;
    }

    void InitializeRequestedReplicas(const ReplicateConfig& config) {
        requested_memory_replicas = config.replica_num;
        requested_nof_replicas = config.nof_replica_num;
        requested_dfs_replicas = config.dfs_replica_num;
        service_dual_write_eligible =
            requested_memory_replicas > 0 && requested_dfs_replicas > 0;
    }

    ReplicateConfig ToReplicateConfig() const {
        ReplicateConfig config;
        config.replica_num = requested_memory_replicas;
        config.nof_replica_num = requested_nof_replicas;
        config.dfs_replica_num = requested_dfs_replicas;
        return config;
    }

    void RecordAllocatedReplicas() {
        transfer_summary = l4::ReplicaTransferSummary{};
        for (const auto& replica : replicas) {
            transfer_summary.RecordAllocatedReplica(replica);
        }
    }

    bool IsResolved() const { return state != PutOperationState::PENDING; }

    bool IsSuccessful() const {
        return state == PutOperationState::SUCCESS && result.has_value();
    }
};

}  // namespace mooncake
