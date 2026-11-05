#include "client_service.h"
#include "l4/async_dfs_write_context.h"
#include "l4/client_write_internal.h"
#include "storage/distributed/dfs_log_limiter.h"

#include <boost/algorithm/string.hpp>
#include <glog/logging.h>

#include "ascii_string.h"
#include "allocator.h"
#include "segment.h"

#include <csignal>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#ifdef USE_NOF
#include <numa.h>
#endif
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <sched.h>
#include <thread>
#include <set>
#include <utility>
#include <vector>
#include <ylt/struct_json/json_reader.h>
#include <boost/asio/ip/address.hpp>

#include "transfer_engine.h"
#include "topology.h"
#include "transfer_task.h"
#include "transport/transport.h"
#include "config.h"
#include "ha/leadership/leader_coordinator_factory.h"
#ifdef MOONCAKE_ENABLE_TEST_FAILPOINTS
#include "ha/oplog/oplog_test_failpoint.h"
#endif
#include "types.h"
#include "client_buffer.h"
#include "utils.h"
#include "rpc_types.h"
#include "local_hot_cache.h"
#include "device/accelerator_registry.h"
#ifdef USE_INTRA_NVLINK
#include "gpu_vendor/intra_nvlink.h"
#endif
#include "crc_checksum.h"
#include "environ.h"
#include "environment_variables.h"
#include "storage/distributed/dfs_io_timeout.h"
#include "storage/distributed/distributed_storage_backend.h"

namespace mooncake {

using l4::AllAllocatedTransfersSucceeded;
using l4::CalculateSliceSize;
using l4::DetermineFinalizeDecision;
using l4::FinalizeDecision;
using l4::HasExpectedReplicaAllocation;
using l4::NonDfsTransfersSucceeded;
using l4::ReplicaTransferSummary;
using l4::ValidatePreferSameNodeWriteConfig;

namespace l4 {

bool NonDfsTransfersSucceeded(const ReplicaTransferSummary& summary) {
    return summary.successful_memory_transfers ==
               summary.allocated_memory_replicas &&
           summary.successful_nof_transfers == summary.allocated_nof_replicas &&
           summary.failed_memory_transfers == 0 &&
           summary.failed_nof_transfers == 0;
}

bool AllAllocatedTransfersSucceeded(const ReplicaTransferSummary& summary) {
    return NonDfsTransfersSucceeded(summary) &&
           summary.successful_dfs_transfers == summary.allocated_dfs_replicas &&
           summary.failed_dfs_transfers == 0;
}

bool HasExpectedReplicaAllocation(const ReplicateConfig& config,
                                  const ReplicaTransferSummary& summary,
                                  bool allow_missing_dfs) {
    if (config.nof_replica_num == 0 && config.dfs_replica_num == 0) {
        return summary.allocated_memory_replicas > 0;
    }
    if (DetermineReplicaWriteMode(config) ==
        ReplicaWriteMode::FLEXIBLE_DUAL_REPLICA) {
        return summary.allocated_memory_replicas +
                   summary.allocated_nof_replicas >
               0;
    }
    if (allow_missing_dfs && config.dfs_replica_num == 1 &&
        summary.allocated_dfs_replicas == 0) {
        return summary.allocated_memory_replicas == config.replica_num &&
               summary.allocated_nof_replicas == config.nof_replica_num;
    }
    return summary.allocated_memory_replicas == config.replica_num &&
           summary.allocated_nof_replicas == config.nof_replica_num &&
           summary.allocated_dfs_replicas == config.dfs_replica_num;
}

std::optional<ErrorCode> ValidatePreferSameNodeWriteConfig(
    const ReplicateConfig& config) {
    if (config.nof_replica_num > 0) {
        LOG(ERROR) << "prefer_alloc_in_same_node is not supported with "
                      "NoF replicas";
        return ErrorCode::INVALID_PARAMS;
    }
    if (config.replica_num != 1) {
        LOG(ERROR) << "prefer_alloc_in_same_node is not supported with "
                      "replica_num != 1";
        return ErrorCode::INVALID_PARAMS;
    }

    return std::nullopt;
}

FinalizeDecision DetermineFinalizeDecision(
    const ReplicateConfig& config, const ReplicaTransferSummary& summary,
    bool allow_missing_dfs, bool allow_memory_only_fallback) {
    const auto write_mode = DetermineReplicaWriteMode(config);
    const bool allocation_satisfied =
        HasExpectedReplicaAllocation(config, summary, allow_missing_dfs);

    const bool dfs_deferred = summary.dfs_completion_deferred;
    if (dfs_deferred && summary.allocated_memory_replicas > 0 &&
        summary.allocated_nof_replicas > 0) {
        LOG(ERROR) << "Deferred DFS completion combined with both MEMORY and "
                      "NoF replicas is not supported";
    }
    // Preserve a completed local tier when an allocated DFS write fails on any
    // backend. Missing DFS allocation is degradable only for callers that
    // explicitly enable the bucket-mode memory-only fallback.
    const bool best_effort_dfs_failure =
        allow_missing_dfs && config.dfs_replica_num == 1 && !dfs_deferred &&
        NonDfsTransfersSucceeded(summary) &&
        ((allow_memory_only_fallback && summary.allocated_dfs_replicas == 0) ||
         (summary.allocated_dfs_replicas == 1 &&
          summary.successful_dfs_transfers == 0 &&
          summary.failed_dfs_transfers > 0));
    // Return the replica type(s) owned by this finalize call whose transfers
    // succeeded. Deferred DFS completion stays owned by the background task.
    const auto successful_type = [&]() -> std::optional<ReplicaType> {
        const bool memory_succeeded = summary.allocated_memory_replicas > 0 &&
                                      summary.successful_memory_transfers ==
                                          summary.allocated_memory_replicas &&
                                      summary.failed_memory_transfers == 0;
        const bool nof_succeeded = summary.allocated_nof_replicas > 0 &&
                                   summary.successful_nof_transfers ==
                                       summary.allocated_nof_replicas &&
                                   summary.failed_nof_transfers == 0;
        const bool dfs_succeeded = !dfs_deferred &&
                                   summary.allocated_dfs_replicas > 0 &&
                                   summary.successful_dfs_transfers ==
                                       summary.allocated_dfs_replicas &&
                                   summary.failed_dfs_transfers == 0;
        const size_t successful_types = static_cast<size_t>(memory_succeeded) +
                                        static_cast<size_t>(nof_succeeded) +
                                        static_cast<size_t>(dfs_succeeded);
        if (successful_types == 0) return std::nullopt;
        if (successful_types > 1) return ReplicaType::ALL;
        if (memory_succeeded) return ReplicaType::MEMORY;
        if (nof_succeeded) return ReplicaType::NOF_SSD;
        return ReplicaType::DFS;
    };

    // Return the replica type(s) still owned by this finalize call whose
    // transfers failed. A deferred DFS replica is excluded because the
    // background task owns its eventual End/Revoke.
    const auto failed_type = [&]() -> std::optional<ReplicaType> {
        const bool memory_failed = summary.allocated_memory_replicas > 0 &&
                                   summary.successful_memory_transfers <
                                       summary.allocated_memory_replicas;
        const bool nof_failed =
            summary.allocated_nof_replicas > 0 &&
            summary.successful_nof_transfers < summary.allocated_nof_replicas;
        const bool dfs_failed =
            !dfs_deferred && summary.allocated_dfs_replicas > 0 &&
            summary.successful_dfs_transfers < summary.allocated_dfs_replicas;
        const size_t failed_types = static_cast<size_t>(memory_failed) +
                                    static_cast<size_t>(nof_failed) +
                                    static_cast<size_t>(dfs_failed);
        if (failed_types == 0) return std::nullopt;
        if (failed_types > 1) return ReplicaType::ALL;
        if (memory_failed) return ReplicaType::MEMORY;
        if (nof_failed) return ReplicaType::NOF_SSD;
        return ReplicaType::DFS;
    };

    if (write_mode != ReplicaWriteMode::FLEXIBLE_DUAL_REPLICA) {
        const bool all_transfers_succeeded =
            AllAllocatedTransfersSucceeded(summary);
        if (allocation_satisfied && all_transfers_succeeded) {
            return {.end_type = successful_type(),
                    .revoke_type = std::nullopt,
                    .success = true,
                    .error = ErrorCode::OK};
        }
        if (best_effort_dfs_failure) {
            return {.end_type = successful_type(),
                    .revoke_type = failed_type(),
                    .success = true,
                    .error = ErrorCode::OK};
        }
        if (!dfs_deferred) {
            return {.end_type = std::nullopt,
                    .revoke_type = ReplicaType::ALL,
                    .success = false,
                    .error = allocation_satisfied
                                 ? (summary.first_error == ErrorCode::OK
                                        ? ErrorCode::TRANSFER_FAIL
                                        : summary.first_error)
                                 : ErrorCode::NO_AVAILABLE_HANDLE};
        }
        return {.end_type = successful_type(),
                .revoke_type = failed_type(),
                .success = false,
                .error = allocation_satisfied
                             ? (summary.first_error == ErrorCode::OK
                                    ? ErrorCode::TRANSFER_FAIL
                                    : summary.first_error)
                             : ErrorCode::NO_AVAILABLE_HANDLE};
    }

    const bool memory_succeeded = summary.successful_memory_transfers > 0;
    const bool nof_succeeded = summary.successful_nof_transfers > 0;

    if (memory_succeeded && nof_succeeded) {
        return {.end_type = successful_type(),
                .revoke_type = std::nullopt,
                .success = true,
                .error = ErrorCode::OK};
    }
    if (memory_succeeded) {
        return {.end_type = ReplicaType::MEMORY,
                .revoke_type = ReplicaType::NOF_SSD,
                .success = true,
                .error = ErrorCode::OK};
    }
    if (nof_succeeded) {
        return {.end_type = ReplicaType::NOF_SSD,
                .revoke_type = ReplicaType::MEMORY,
                .success = true,
                .error = ErrorCode::OK};
    }

    return {.end_type = successful_type(),
            .revoke_type = failed_type(),
            .success = false,
            .error = summary.first_error == ErrorCode::OK
                         ? ErrorCode::NO_AVAILABLE_HANDLE
                         : summary.first_error};
}

}  // namespace l4

namespace {

bool IsValidServiceEndpoint(const std::string& endpoint) {
    if (endpoint.empty()) return false;

    std::string host;
    std::string_view port_text;
    if (endpoint.front() == '[') {
        const auto closing_bracket = endpoint.find(']');
        if (closing_bracket == std::string::npos ||
            closing_bracket + 1 >= endpoint.size() ||
            endpoint[closing_bracket + 1] != ':') {
            return false;
        }
        host = endpoint.substr(1, closing_bracket - 1);
        port_text = std::string_view(endpoint).substr(closing_bracket + 2);
        boost::system::error_code error;
        const auto address = boost::asio::ip::make_address(host, error);
        if (error || !address.is_v6() || address.is_unspecified()) {
            return false;
        }
    } else {
        const auto separator = endpoint.rfind(':');
        if (separator == std::string::npos || endpoint.find(':') != separator) {
            return false;
        }
        host = endpoint.substr(0, separator);
        port_text = std::string_view(endpoint).substr(separator + 1);
        if (host.empty() || host == "*") return false;
        boost::system::error_code error;
        const auto address = boost::asio::ip::make_address(host, error);
        if (error || !address.is_v4() || address.is_unspecified()) {
            return false;
        }
    }

    if (port_text.empty()) return false;
    uint32_t port = 0;
    for (unsigned char c : port_text) {
        if (!std::isdigit(c)) return false;
        port = port * 10 + static_cast<uint32_t>(c - '0');
        if (port > 65535) return false;
    }
    return port != 0;
}

}  // namespace

void Client::ObserveL4Read(bool prefetch, bool success, uint64_t bytes,
                           double duration_seconds) {
    if (metrics_ != nullptr) {
        metrics_->ObserveL4Read(prefetch, success, bytes, duration_seconds);
    }
    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    if (service_metrics_snapshot_.service_endpoint.empty()) return;
    auto& l4 = service_metrics_snapshot_.l4;
    const size_t index = (prefetch ? 0 : 2) + (success ? 0 : 1);
    ++l4.read_operations[index];
    l4.read_bytes[index] += bytes;
    l4.read_duration_us[index] += static_cast<uint64_t>(duration_seconds * 1e6);
    const auto bucket =
        std::lower_bound(kL4ReadLatencySecondsBuckets.begin(),
                         kL4ReadLatencySecondsBuckets.end(), duration_seconds);
    ++l4.read_duration_buckets[index]
                              [bucket - kL4ReadLatencySecondsBuckets.begin()];
}

void Client::ObserveL4Restore(bool prefetched, uint64_t ranges,
                              uint64_t bytes) {
    if (metrics_ != nullptr) {
        metrics_->ObserveL4Restore(prefetched, ranges, bytes);
    }
    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    if (service_metrics_snapshot_.service_endpoint.empty()) return;
    const size_t index = prefetched ? 0 : 1;
    service_metrics_snapshot_.l4.restore_ranges[index] += ranges;
    service_metrics_snapshot_.l4.restore_bytes[index] += bytes;
}

int Client::SetServiceEndpoint(const std::string& endpoint) {
    if (!IsValidServiceEndpoint(endpoint)) {
        return toInt(ErrorCode::INVALID_PARAMS);
    }

    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    if (!service_metrics_snapshot_.service_endpoint.empty()) {
        return service_metrics_snapshot_.service_endpoint == endpoint
                   ? 0
                   : toInt(ErrorCode::INVALID_PARAMS);
    }
    service_metrics_snapshot_.service_endpoint = endpoint;
    service_metrics_report_running_.store(true, std::memory_order_release);
    try {
        service_metrics_report_thread_ =
            std::thread(&Client::ServiceMetricsReporterLoop, this);
    } catch (const std::exception& error) {
        service_metrics_report_running_.store(false, std::memory_order_release);
        service_metrics_snapshot_.service_endpoint.clear();
        LOG(ERROR) << "Failed to start service metrics reporter: "
                   << error.what();
        return toInt(ErrorCode::INTERNAL_ERROR);
    }
    return 0;
}

void Client::RecordServiceReadFailure(ReplicaType replica_type) {
    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    if (service_metrics_snapshot_.service_endpoint.empty()) return;
    if (replica_type == ReplicaType::MEMORY) {
        ++service_metrics_snapshot_.read_failures_memory;
    } else if (replica_type == ReplicaType::DFS) {
        ++service_metrics_snapshot_.read_failures_dfs;
    }
}

void Client::RecordServiceDualWrite(bool memory_succeeded, bool dfs_succeeded) {
    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    if (service_metrics_snapshot_.service_endpoint.empty()) return;
    if (memory_succeeded && dfs_succeeded) {
        ++service_metrics_snapshot_.dual_write_both_success;
    } else if (memory_succeeded) {
        ++service_metrics_snapshot_.dual_write_memory_only;
    } else if (dfs_succeeded) {
        ++service_metrics_snapshot_.dual_write_dfs_only;
    } else {
        ++service_metrics_snapshot_.dual_write_neither_success;
    }
}

void Client::BeginServiceDualWritePending() {
    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    if (!service_metrics_snapshot_.service_endpoint.empty()) {
        ++service_metrics_snapshot_.dual_write_pending;
    }
}

void Client::CancelServiceDualWritePending() {
    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    if (!service_metrics_snapshot_.service_endpoint.empty() &&
        service_metrics_snapshot_.dual_write_pending > 0) {
        --service_metrics_snapshot_.dual_write_pending;
    }
}

void Client::CompleteServiceDualWrite(bool memory_succeeded,
                                      bool dfs_succeeded) {
    std::lock_guard<std::mutex> lock(service_metrics_mutex_);
    if (service_metrics_snapshot_.service_endpoint.empty()) return;
    if (service_metrics_snapshot_.dual_write_pending > 0) {
        --service_metrics_snapshot_.dual_write_pending;
    }
    if (memory_succeeded && dfs_succeeded) {
        ++service_metrics_snapshot_.dual_write_both_success;
    } else if (memory_succeeded) {
        ++service_metrics_snapshot_.dual_write_memory_only;
    } else if (dfs_succeeded) {
        ++service_metrics_snapshot_.dual_write_dfs_only;
    } else {
        ++service_metrics_snapshot_.dual_write_neither_success;
    }
}

void Client::CompleteAsyncDfsWriteMetrics(
    const std::shared_ptr<AsyncDfsWriteContext>& context,
    const std::vector<ErrorCode>& outcomes) {
    if (context->service_metrics_settled.exchange(true)) return;
    for (size_t i = 0; i < context->service_dual_writes.size(); ++i) {
        const auto& metric = context->service_dual_writes[i];
        if (!metric.eligible) continue;
        const bool dfs_succeeded = i < outcomes.size() &&
                                   outcomes[i] == ErrorCode::OK &&
                                   metric.requested_dfs_replicas == 1;
        CompleteServiceDualWrite(metric.memory_succeeded, dfs_succeeded);
    }
}

void Client::ExpireAsyncDfsWriteMetrics() {
    std::vector<std::shared_ptr<AsyncDfsWriteContext>> expired;
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(service_metrics_mutex_);
        std::erase_if(
            service_pending_writes_, [&expired, now](const auto& item) {
                auto context = item.lock();
                if (!context || context->service_metrics_settled.load() ||
                    context->io_completed.load()) {
                    return true;
                }
                if (context->deadline > now) return false;
                expired.push_back(std::move(context));
                return true;
            });
    }
    for (const auto& context : expired) {
        // Publication is already past its deadline. Settle statistics only;
        // the worker still owns its buffers until the filesystem call ends.
        CompleteAsyncDfsWriteMetrics(context, {});
    }
}

void Client::ServiceMetricsReporterLoop() {
    while (service_metrics_report_running_.load(std::memory_order_acquire)) {
        for (int i = 0; i < 50 && service_metrics_report_running_.load(
                                      std::memory_order_acquire);
             ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!service_metrics_report_running_.load(std::memory_order_acquire)) {
            break;
        }

        ExpireAsyncDfsWriteMetrics();
        ServiceMetricsSnapshot snapshot;
        {
            std::lock_guard<std::mutex> lock(service_metrics_mutex_);
            if (service_metrics_snapshot_.service_endpoint.empty()) return;
            snapshot = service_metrics_snapshot_;
            snapshot.sequence = ++service_metrics_snapshot_.sequence;
        }
        auto result = master_client_.ReportServiceMetrics(snapshot);
        if (!result) {
            VLOG(1) << "Service metrics report failed: "
                    << toString(result.error());
        }
    }
}

void Client::StopServiceMetricsReporter() {
    service_metrics_report_running_.store(false, std::memory_order_release);
    if (service_metrics_report_thread_.joinable()) {
        service_metrics_report_thread_.join();
    }

    ServiceMetricsSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(service_metrics_mutex_);
        if (service_metrics_snapshot_.service_endpoint.empty()) return;
        snapshot = service_metrics_snapshot_;
        snapshot.sequence = ++service_metrics_snapshot_.sequence;
    }
    auto result = master_client_.ReportServiceMetrics(snapshot);
    if (!result) {
        VLOG(1) << "Final service metrics report failed: "
                << toString(result.error());
    }
}

tl::expected<QueryResult, ErrorCode> Client::PrepareDfsReadRetry(
    const QueryResult& original, const QueryResult& refreshed) {
    Replica::Descriptor selected;
    if (FindFirstCompleteReplica(original.replicas, selected) !=
            ErrorCode::OK ||
        !selected.is_dfs_replica()) {
        return tl::unexpected(ErrorCode::DFS_STALE_HANDLE);
    }
    // The caller still owns buffers and session state based on the old query.
    // Only retry the same replica type, size and checksum view. If the caller
    // omitted a checksum, do not silently introduce one from a new query.
    if (original.object_checksum != refreshed.object_checksum) {
        return tl::unexpected(ErrorCode::DFS_STALE_HANDLE);
    }
    for (const auto& replica : refreshed.replicas) {
        if (!replica.is_dfs_replica() ||
            replica.status != ReplicaStatus::COMPLETE) {
            continue;
        }
        if (calculate_total_size(replica) != calculate_total_size(selected)) {
            return tl::unexpected(ErrorCode::DFS_STALE_HANDLE);
        }
        return QueryResult({replica}, refreshed.lease_timeout,
                           original.object_checksum);
    }
    return tl::unexpected(ErrorCode::DFS_STALE_HANDLE);
}

void Client::AsyncDfsWriteContext::ReleaseStagingBuffer() {
    staging_buffer.reset();
}

Client::AsyncDfsWriteContext::~AsyncDfsWriteContext() {
    slices.clear();
    ReleaseStagingBuffer();
}

bool Client::StageDfsWriteData(
    AsyncDfsWriteContext& context,
    const std::vector<const std::vector<Slice>*>& slice_lists) {
    if (!context.staging_buffer || !context.staging_buffer->ptr()) return false;

    auto runtime_accelerator =
        device::GetAcceleratorRegistry().RuntimeAccelerators();

    size_t staging_offset = 0;
    context.slices.resize(slice_lists.size());
    for (size_t i = 0; i < slice_lists.size(); ++i) {
        if (slice_lists[i] == nullptr) {
            LOG(ERROR) << "Missing slices for async DFS write of key "
                       << context.keys[i];
            return false;
        }
        for (const auto& slice : *slice_lists[i]) {
            if (slice.size == 0) continue;
            if (slice.ptr == nullptr) {
                LOG(ERROR) << "Null slice for async DFS write of key "
                           << context.keys[i];
                return false;
            }

            if (staging_offset > context.staging_buffer->size() ||
                slice.size > context.staging_buffer->size() - staging_offset) {
                LOG(ERROR) << "DFS write staging arena exhausted for key "
                           << context.keys[i];
                return false;
            }
            char* destination =
                static_cast<char*>(context.staging_buffer->ptr()) +
                staging_offset;
            staging_offset += slice.size;

            device::PointerInfo info{};
            auto* device =
                runtime_accelerator.FindDeviceForPointer(slice.ptr, &info);
            if (device != nullptr) {
                device->SetContext(info.device_id);
                if (!device->Copy(destination, slice.ptr, slice.size,
                                  device::CopyDirection::kDeviceToHost)) {
                    LOG(ERROR)
                        << "DFS D2H staging failed for key " << context.keys[i];
                    return false;
                }
            } else {
                std::memcpy(destination, slice.ptr, slice.size);
            }
            context.slices[i].push_back(Slice{destination, slice.size});
        }
    }
    return true;
}

void Client::RunAsyncDfsWrite(std::shared_ptr<AsyncDfsWriteContext> context) {
    std::vector<DfsWriteRequest> requests;
    requests.reserve(context->keys.size());
    for (size_t i = 0; i < context->keys.size(); ++i) {
        requests.push_back(DfsWriteRequest{
            context->keys[i], context->descriptors[i], context->slices[i]});
    }

    // Every request gets a definite outcome: a missing or short result vector
    // is treated as failure for the affected keys rather than silently ignored.
    std::vector<ErrorCode> outcomes(requests.size(), ErrorCode::INTERNAL_ERROR);
    const auto write_started = std::chrono::steady_clock::now();
    std::vector<tl::expected<void, ErrorCode>> results;
    try {
        results = context->backend->BatchWrite(requests);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Async DFS BatchWrite failed: " << e.what();
        results.assign(requests.size(),
                       tl::make_unexpected(ErrorCode::INTERNAL_ERROR));
    } catch (...) {
        LOG(ERROR) << "Async DFS BatchWrite failed";
        results.assign(requests.size(),
                       tl::make_unexpected(ErrorCode::INTERNAL_ERROR));
    }
    // Once the worker observes a result, it owns outcome classification. The
    // reporter must not turn a timely result into a timeout while we process
    // it.
    context->io_completed.store(true);
    const double write_duration_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      write_started)
            .count();
    uint64_t successful_bytes = 0;
    bool all_succeeded = !requests.empty() && results.size() == requests.size();
    if (results.size() != requests.size()) {
        LOG(ERROR) << "Async DFS BatchWrite response size mismatch: expected "
                   << requests.size() << ", got " << results.size();
    } else {
        for (size_t i = 0; i < results.size(); ++i) {
            outcomes[i] = results[i] ? ErrorCode::OK : results[i].error();
            if (results[i]) {
                for (const auto& slice : requests[i].slices) {
                    successful_bytes += slice.size;
                }
            } else {
                all_succeeded = false;
            }
        }
    }

    const bool timed_out =
        context->deadline != std::chrono::steady_clock::time_point::max() &&
        std::chrono::steady_clock::now() > context->deadline;
    if (timed_out) {
        // The foreground has already published any successful memory replica.
        // Do not publish a DFS replica whose write missed its deadline; revoke
        // only that replica after the blocking I/O has actually returned.
        std::fill(outcomes.begin(), outcomes.end(),
                  ErrorCode::DFS_BATCH_PUT_TIMEOUT);
        all_succeeded = false;
        successful_bytes = 0;
    }

    CompleteAsyncDfsWriteMetrics(context, outcomes);

    ObserveDirectIo("write", DirectStorageMetricSource(ReplicaType::DFS),
                    all_succeeded, successful_bytes, write_duration_seconds);

    // BatchWrite is synchronous: once it returns, neither the backend nor the
    // local request views use the staging bytes. Release the slot before the
    // completion RPCs so another chunk can stage while PutEnd/PutRevoke runs.
    requests.clear();
    context->slices.clear();
    context->ReleaseStagingBuffer();

    // Report each key individually so one failure cannot revoke its neighbours.
    // The RPCs are idempotent on the master side, so a bounded retry is safe.
    constexpr int kMaxCompletionAttempts = 3;
    for (size_t i = 0; i < outcomes.size(); ++i) {
        const auto& key = context->keys[i];
        const bool succeeded = outcomes[i] == ErrorCode::OK;
        if (!succeeded) {
            static DfsLogLimiter log_limiter;
            if (auto suppressed = log_limiter.ShouldLog()) {
                LOG(ERROR) << "Async DFS write failed for key " << key
                           << ", error=" << toString(outcomes[i])
                           << "; revoking the DFS replica"
                           << ", suppressed=" << *suppressed;
            }
        }

        tl::expected<void, ErrorCode> completion =
            tl::make_unexpected(ErrorCode::RPC_FAIL);
        const auto object_checksum = context->object_checksums[i];
        for (int attempt = 0; attempt < kMaxCompletionAttempts; ++attempt) {
            if (succeeded) {
                completion = context->is_upsert
                                 ? master_client_.UpsertEnd(
                                       ObjectMeta{key, object_checksum},
                                       ReplicaType::DFS,
                                       context->descriptors[i].MasterEpoch())
                                 : master_client_.PutEnd(
                                       ObjectMeta{key, object_checksum},
                                       ReplicaType::DFS,
                                       context->descriptors[i].MasterEpoch());
            } else {
                completion = context->is_upsert
                                 ? master_client_.UpsertRevoke(
                                       key, ReplicaType::DFS,
                                       context->descriptors[i].MasterEpoch())
                                 : master_client_.PutRevoke(
                                       key, ReplicaType::DFS,
                                       context->descriptors[i].MasterEpoch());
            }
            if (completion) break;
            // OBJECT_NOT_FOUND / INVALID_WRITE mean the object already moved on
            // (removed, or replaced by a newer generation). Retrying cannot
            // help and must not disturb the new state.
            const auto error = completion.error();
            if (error == ErrorCode::OBJECT_NOT_FOUND ||
                error == ErrorCode::INVALID_WRITE ||
                error == ErrorCode::ILLEGAL_CLIENT) {
                break;
            }
        }
        if (!completion) {
            LOG(ERROR)
                << "Failed to finalize async DFS write for key " << key << " ("
                << (succeeded ? "PutEnd" : "PutRevoke")
                << "), error=" << toString(completion.error())
                << "; the master will reclaim the PROCESSING replica via "
                   "its put-start timeout";
        }
    }
}

void Client::DrainAsyncDfsWrites() {
    dfs_writes_shutting_down_.store(true, std::memory_order_release);
    std::unique_lock<std::mutex> lock(dfs_inflight_mutex_);
    // Wait for tasks that still use master_client_ / dfs_storage_backend_ so
    // they cannot outlive the Client and access freed state.
    dfs_inflight_cv_.wait(lock, [this] { return dfs_inflight_writes_ == 0; });
}

tl::expected<DfsBucketSnapshotResponse, ErrorCode> Client::GetDfsBucketSnapshot(
    const UUID& last_epoch, uint64_t last_version) {
    return master_client_.GetDfsBucketSnapshot(client_id_, last_epoch,
                                               last_version);
}

void Client::SetDfsWriteChunkSize(size_t chunk_size) {
    dfs_write_chunk_size_ = chunk_size;
}

void Client::SetDfsWriteAlignment(size_t alignment) {
    if (alignment != 0) {
        dfs_write_alignment_ = alignment;
    }
}

void Client::SetUnifiedDfsBuffer(
    std::shared_ptr<ClientBufferAllocator> allocator, size_t capacity) {
    unified_buffer_allocator_ = std::move(allocator);
    unified_buffer_capacity_ = capacity;
    if (!unified_buffer_allocator_ || unified_buffer_capacity_ == 0) return;
    if (!unified_buffer_log_running_.exchange(true)) {
        unified_buffer_log_thread_ =
            std::thread(&Client::LogUnifiedDfsBufferUsage, this);
    }
}

void Client::LogUnifiedDfsBufferUsage() {
    std::unique_lock<std::mutex> lock(unified_buffer_log_mutex_);
    while (unified_buffer_log_running_.load(std::memory_order_acquire)) {
        unified_buffer_log_cv_.wait_for(lock, std::chrono::seconds(5), [this] {
            return !unified_buffer_log_running_.load(std::memory_order_acquire);
        });
        if (!unified_buffer_log_running_.load(std::memory_order_acquire)) {
            break;
        }
        const auto allocator = unified_buffer_allocator_;
        const auto tracker = allocator ? allocator->usage_tracker() : nullptr;
        const auto snapshot =
            tracker ? tracker->Snapshot() : DfsBufferUsageSnapshot{};
        const double capacity = static_cast<double>(unified_buffer_capacity_);
        const auto utilization = [capacity](uint64_t bytes) {
            return capacity == 0.0 ? 0.0 : bytes * 100.0 / capacity;
        };
        size_t write_backlog = 0;
        {
            std::lock_guard<std::mutex> inflight_lock(dfs_inflight_mutex_);
            write_backlog = dfs_inflight_writes_;
        }
        if (metrics_) {
            metrics_->unified_buffer_metric.capacity_bytes.update(
                static_cast<int64_t>(unified_buffer_capacity_));
            metrics_->unified_buffer_metric.read_bytes.update(
                static_cast<int64_t>(snapshot.read_bytes));
            metrics_->unified_buffer_metric.prefetch_bytes.update(
                static_cast<int64_t>(snapshot.prefetch_bytes));
            metrics_->unified_buffer_metric.write_bytes.update(
                static_cast<int64_t>(snapshot.write_bytes));
            metrics_->unified_buffer_metric.total_bytes.update(
                static_cast<int64_t>(snapshot.total_bytes));
            metrics_->unified_buffer_metric.peak_total_bytes.update(
                static_cast<int64_t>(snapshot.peak_total_bytes));
            metrics_->unified_buffer_metric.write_backlog.update(
                static_cast<int64_t>(write_backlog));
        }
        if (!dfs_read_trace_enabled()) continue;
        LOG(INFO) << "action=dfs_unified_buffer_usage scope=process"
                  << " unified_capacity_bytes=" << unified_buffer_capacity_
                  << " read_bytes=" << snapshot.read_bytes
                  << " prefetch_bytes=" << snapshot.prefetch_bytes
                  << " write_bytes=" << snapshot.write_bytes
                  << " total_bytes=" << snapshot.total_bytes
                  << " read_utilization=" << std::fixed << std::setprecision(1)
                  << utilization(snapshot.read_bytes) << "%"
                  << " prefetch_utilization="
                  << utilization(snapshot.prefetch_bytes) << "%"
                  << " write_utilization=" << utilization(snapshot.write_bytes)
                  << "% total_utilization=" << utilization(snapshot.total_bytes)
                  << "% peak_total_bytes=" << snapshot.peak_total_bytes
                  << " peak_total_utilization="
                  << utilization(snapshot.peak_total_bytes) << "%"
                  << " write_backlog=" << write_backlog;
    }
}

void Client::SetDfsBatchTimeoutFixedMs(uint64_t timeout_ms) {
    const uint64_t old_timeout_ms = dfs_batch_timeout_fixed_ms_.exchange(
        timeout_ms, std::memory_order_acq_rel);
    if (dfs_storage_backend_) {
        dfs_storage_backend_->SetBatchTimeoutFixedMs(timeout_ms);
    }
    if (old_timeout_ms != timeout_ms) {
        LOG(INFO) << "Updated DFS batch timeout fixed budget from "
                  << old_timeout_ms << "ms to " << timeout_ms << "ms";
    }
}

tl::expected<std::string, ErrorCode> Client::FetchDfsRootIdentity() {
    auto config = master_client_.GetStorageConfig();
    if (!config) return tl::make_unexpected(config.error());
    if (config->dfs_root_identity.empty()) {
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }
    return config->dfs_root_identity;
}

tl::expected<std::shared_ptr<DistributedStorageBackend>, ErrorCode>
Client::GetDfsBackfillBackend() {
    if (dfs_storage_backend_) return dfs_storage_backend_;
    std::lock_guard<std::mutex> lock(dfs_backfill_backend_mutex_);
    if (dfs_backfill_backend_) return dfs_backfill_backend_;

    auto storage = master_client_.GetStorageConfig();
    if (!storage) return tl::make_unexpected(storage.error());
    if (storage->dfs_fsdir.empty() || storage->dfs_root_identity.empty()) {
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }
    FileStorageConfig config;
    config.storage_backend_type = StorageBackendType::kDistributed;
    auto backend = CreateStorageBackend(config, storage->dfs_root_identity,
                                        storage->dfs_fsdir);
    if (!backend) return tl::make_unexpected(backend.error());
    dfs_backfill_backend_ =
        std::dynamic_pointer_cast<DistributedStorageBackend>(*backend);
    if (dfs_backfill_backend_) {
        dfs_backfill_backend_->SetDescriptorRootDir(storage->dfs_fsdir);
        dfs_backfill_backend_->SetMasterEpochProvider(
            master_client_.MakeDfsMasterEpochProvider());
    }
    // The executor checks initialization before acquiring a source pin.
    return dfs_backfill_backend_;
}

bool Client::BeginDfsBackfillAttempt(const UUID& task_id) {
    std::lock_guard<std::mutex> lock(dfs_backfill_backend_mutex_);
    if (dfs_backfill_failures_ != 0) {
        if (std::chrono::steady_clock::now() < dfs_backfill_retry_after_ ||
            dfs_backfill_probe_task_ != UUID{})
            return false;
        // After a cooldown only one real write probes this holder's mount.
        dfs_backfill_probe_task_ = task_id;
    }
    return true;
}

void Client::FinishDfsBackfillAttempt(const UUID& task_id, ErrorCode result,
                                      bool storage_failure) {
    std::lock_guard<std::mutex> lock(dfs_backfill_backend_mutex_);
    const bool probe = dfs_backfill_probe_task_ == task_id;
    if (probe) dfs_backfill_probe_task_ = UUID{};
    if (result == ErrorCode::OK) {
        dfs_backfill_failures_ = 0;
        dfs_backfill_retry_after_ = {};
    } else if (storage_failure && (dfs_backfill_failures_ == 0 || probe)) {
        // Concurrent failures from the initial healthy batch do not keep
        // extending the pause. Missing keys/source races do not open it.
        dfs_backfill_failures_ = std::min(dfs_backfill_failures_ + 1, 4u);
        dfs_backfill_retry_after_ =
            std::chrono::steady_clock::now() +
            std::chrono::seconds(
                std::min(5u << (dfs_backfill_failures_ - 1), 30u));
    }
}

void Client::StopDfsWritesAndBufferLogging() {
    // Stop accepting new asynchronous DFS writes before draining tasks already
    // submitted. Their BufferHandles retain the unified allocator.
    unified_buffer_log_running_.store(false, std::memory_order_release);
    unified_buffer_log_cv_.notify_all();
    if (unified_buffer_log_thread_.joinable()) {
        unified_buffer_log_thread_.join();
    }
    if (unified_buffer_allocator_) {
        unified_buffer_allocator_->ShutdownAllocationWaiters();
    }
    // They report their result through master_client_ and write through
    // dfs_storage_backend_, so they must not outlive this object.
    DrainAsyncDfsWrites();
    write_thread_pool_.stop();
    StopServiceMetricsReporter();
}

void Client::RecordBatchServiceDualWriteMetrics(
    std::vector<PutOperation>& ops) {
    for (auto& op : ops) {
        if (!op.service_dual_write_eligible || op.service_metric_recorded ||
            op.service_metric_excluded ||
            op.transfer_summary.dfs_completion_deferred) {
            continue;
        }
        if (!op.result.has_value() &&
            op.result.error() == ErrorCode::OBJECT_ALREADY_EXISTS) {
            continue;
        }

        const bool memory_succeeded =
            op.transfer_summary.successful_memory_transfers ==
            op.requested_memory_replicas;
        const bool dfs_succeeded =
            op.transfer_summary.successful_dfs_transfers ==
            op.requested_dfs_replicas;
        RecordServiceDualWrite(memory_succeeded, dfs_succeeded);
        op.service_metric_recorded = true;
    }
}

void Client::SubmitDfsWriteChunk(std::vector<PutOperation>& ops,
                                 const std::vector<size_t>& chunk_indices,
                                 bool is_upsert, bool allow_async,
                                 std::chrono::steady_clock::time_point deadline,
                                 bool* staging_timed_out_out,
                                 bool* staging_exceeds_limit_out) {
    if (staging_timed_out_out) *staging_timed_out_out = false;
    if (staging_exceeds_limit_out) *staging_exceeds_limit_out = false;
    std::vector<std::string> keys;
    std::vector<const std::vector<Slice>*> slice_lists;
    std::vector<DistributedFSDescriptor> descriptors;
    std::vector<size_t> op_indices;
    const bool bucket_mode =
        dfs_storage_backend_ != nullptr &&
        dfs_storage_backend_->GetAllocatorType() == DfsAllocatorType::BUCKET;

    for (size_t i : chunk_indices) {
        auto& op = ops[i];
        if (op.IsResolved() ||
            op.transfer_summary.allocated_dfs_replicas == 0 ||
            !NonDfsTransfersSucceeded(op.transfer_summary)) {
            continue;
        }
        // In BUCKET mode, an incomplete requested topology will be revoked.
        // Skipping the DFS write keeps that revoke unambiguous instead of
        // racing a background completion.
        if (bucket_mode && !HasExpectedReplicaAllocation(op.ToReplicateConfig(),
                                                         op.transfer_summary)) {
            continue;
        }

        auto dfs_it = std::find_if(op.replicas.begin(), op.replicas.end(),
                                   [](const Replica::Descriptor& replica) {
                                       return replica.is_dfs_replica();
                                   });
        if (dfs_it == op.replicas.end()) {
            op.transfer_summary.RecordFailure(ReplicaType::DFS,
                                              ErrorCode::INVALID_REPLICA);
            op.AppendFailureContext("Allocated DFS replica has no descriptor");
            continue;
        }
        keys.push_back(op.key);
        slice_lists.push_back(&op.slices);
        descriptors.push_back(dfs_it->get_dfs_descriptor());
        op_indices.push_back(i);
    }

    if (keys.empty()) return;

    auto backend = dfs_storage_backend_;
    if (!backend) {
        // No backend: fail every affected key here instead of queueing a task
        // that would dereference a null pointer.
        LOG(ERROR) << "DFS backend is unavailable; failing " << keys.size()
                   << " DFS writes";
        for (const size_t index : op_indices) {
            auto& op = ops[index];
            op.transfer_summary.RecordFailure(
                ReplicaType::DFS, ErrorCode::DFS_SERVICE_UNAVAILABLE);
            op.AppendFailureContext("DFS backend is not initialized");
        }
        return;
    }

    // Restore the allocator's original contract: BUCKET writes are staged and
    // asynchronous, while SHARD writes remain synchronous. The configured
    // deadline limits publication of an asynchronous DFS replica; it does not
    // turn every allocator into a detached write path.
    const bool async_write =
        allow_async && backend->GetAllocatorType() == DfsAllocatorType::BUCKET;
    if (!async_write) {
        // The staging wait must fit inside the BatchPut deadline, otherwise
        // a short MOONCAKE_DFS_BATCH_PUT_TIMEOUT_MS would be ignored and the
        // caller would block for the full arena wait instead of returning
        // on time. Mirror the asynchronous path's min(arena_wait, left).
        const auto sync_staging_wait = [&]() -> std::chrono::milliseconds {
            if (deadline == std::chrono::steady_clock::time_point::max()) {
                return dfs_write_arena_wait_;
            }
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::max(deadline - std::chrono::steady_clock::now(),
                             std::chrono::steady_clock::duration::zero()));
            return std::min(dfs_write_arena_wait_, left);
        }();
        auto results = WriteDfsReplicas(keys, slice_lists, descriptors, nullptr,
                                        sync_staging_wait);
        for (size_t i = 0; i < results.size(); ++i) {
            auto& op = ops[op_indices[i]];
            if (results[i] == ErrorCode::OK) {
                op.transfer_summary.RecordSuccess(ReplicaType::DFS);
            } else {
                op.transfer_summary.RecordFailure(ReplicaType::DFS, results[i]);
                op.AppendFailureContext("Synchronous DFS write failed: " +
                                        toString(results[i]));
            }
        }
        return;
    }

    std::shared_ptr<AsyncDfsWriteContext> context;
    bool staged = false;
    bool staging_allocation_timed_out = false;
    bool staging_request_exceeds_limit = false;
    try {
        context = std::make_shared<AsyncDfsWriteContext>();
        context->deadline = deadline;
        context->keys = keys;
        context->descriptors = descriptors;
        context->backend = std::move(backend);
        context->unified_buffer_allocator = unified_buffer_allocator_;
        if (!context->unified_buffer_allocator) {
            LOG(ERROR) << "DFS_UNIFIED_BUFFER_ALLOCATION_FAILED kind=write "
                          "mode=asynchronous reason=allocator_unavailable";
            ObserveUnifiedDfsBufferAllocationFailure(
                DfsBufferUsageKind::kWrite);
            throw std::runtime_error("Unified DFS buffer is unavailable");
        }
        size_t payload_size = 0;
        for (const auto* slices : slice_lists) {
            if (!slices) {
                throw std::runtime_error("DFS write staging size overflow");
            }
            const size_t slice_size = CalculateSliceSize(*slices);
            if (payload_size >
                std::numeric_limits<size_t>::max() - slice_size) {
                throw std::runtime_error("DFS write staging size overflow");
            }
            payload_size += slice_size;
        }
        const bool deadline_expired =
            deadline != std::chrono::steady_clock::time_point::max() &&
            deadline <= std::chrono::steady_clock::now();
        const auto staging_wait =
            deadline_expired ? std::chrono::milliseconds::zero()
            : deadline == std::chrono::steady_clock::time_point::max()
                ? dfs_write_arena_wait_
                : std::min(
                      dfs_write_arena_wait_,
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline - std::chrono::steady_clock::now()));
        if (deadline_expired) staging_allocation_timed_out = true;
        context->staging_buffer =
            context->unified_buffer_allocator->allocate_aligned_wait(
                payload_size, dfs_write_alignment_, DfsBufferUsageKind::kWrite,
                staging_wait, &staging_allocation_timed_out,
                &staging_request_exceeds_limit);
        if (!context->staging_buffer) {
            LOG(ERROR) << "DFS_UNIFIED_BUFFER_ALLOCATION_FAILED kind=write "
                          "mode=asynchronous size="
                       << payload_size;
            ObserveUnifiedDfsBufferAllocationFailure(
                DfsBufferUsageKind::kWrite);
            throw std::runtime_error("Unified DFS write buffer unavailable");
        }
        context->is_upsert = is_upsert;
        context->object_checksums.resize(keys.size());
        context->service_dual_writes.resize(keys.size());
        for (size_t i = 0; i < op_indices.size(); ++i) {
            const auto& op = ops[op_indices[i]];
            auto& metric = context->service_dual_writes[i];
            metric.eligible = op.service_dual_write_eligible;
            metric.memory_succeeded =
                op.transfer_summary.successful_memory_transfers ==
                op.requested_memory_replicas;
            metric.requested_dfs_replicas = op.requested_dfs_replicas;
        }
        staged = StageDfsWriteData(*context, slice_lists);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to stage async DFS write: " << e.what();
    }
    if (!staged) {
        for (const size_t index : op_indices) {
            auto& op = ops[index];
            op.transfer_summary.RecordFailure(ReplicaType::DFS,
                                              ErrorCode::TRANSFER_FAIL);
            op.AppendFailureContext(
                staging_request_exceeds_limit
                    ? "DFS staging request exceeds the unified buffer limit"
                    : (staging_allocation_timed_out
                           ? "DFS staging allocation wait timed out"
                           : "Failed to stage DFS write data"));
        }
        if (staging_timed_out_out) {
            *staging_timed_out_out = staging_allocation_timed_out;
        }
        if (staging_exceeds_limit_out) {
            *staging_exceeds_limit_out = staging_request_exceeds_limit;
        }
        return;
    }

    // No End may run until the foreground CRC is ready, even if DFS I/O
    // finishes before the memory replicas are finalized.
    bool checksums_ready = true;
    for (size_t i = 0; i < op_indices.size(); ++i) {
        auto& op = ops[op_indices[i]];
        ComputePutObjectChecksum(op, context->slices[i], true);
        context->object_checksums[i] = op.object_checksum;
        if (op.IsResolved()) checksums_ready = false;
    }
    if (!checksums_ready) {
        for (const size_t index : op_indices) {
            ops[index].transfer_summary.RecordFailure(ReplicaType::DFS,
                                                      ErrorCode::TRANSFER_FAIL);
        }
        // FinalizeBatchPut revokes all allocations for terminal CRC errors.
        return;
    }

    {
        std::lock_guard<std::mutex> lock(dfs_inflight_mutex_);
        if (dfs_writes_shutting_down_.load(std::memory_order_acquire)) {
            LOG(ERROR) << "Client is shutting down; refusing " << keys.size()
                       << " async DFS writes";
            for (const size_t index : op_indices) {
                auto& op = ops[index];
                op.transfer_summary.RecordFailure(
                    ReplicaType::DFS, ErrorCode::DFS_SERVICE_UNAVAILABLE);
                op.AppendFailureContext("Client is shutting down");
            }
            return;
        }
        ++dfs_inflight_writes_;
    }

    auto release_inflight = [this]() {
        std::lock_guard<std::mutex> lock(dfs_inflight_mutex_);
        if (dfs_inflight_writes_ > 0) --dfs_inflight_writes_;
        dfs_inflight_cv_.notify_all();
    };

    for (const auto& metric : context->service_dual_writes) {
        if (metric.eligible) BeginServiceDualWritePending();
    }

    try {
        {
            std::lock_guard<std::mutex> lock(service_metrics_mutex_);
            if (!service_metrics_snapshot_.service_endpoint.empty() &&
                context->deadline !=
                    std::chrono::steady_clock::time_point::max()) {
                service_pending_writes_.push_back(context);
            }
        }
#ifdef MOONCAKE_ENABLE_TEST_FAILPOINTS
        TestFailPoint::Wait("dfs_write_before_enqueue");
#endif
        write_thread_pool_.enqueue([this, context, release_inflight]() {
            try {
                RunAsyncDfsWrite(context);
            } catch (const std::exception& e) {
                LOG(ERROR) << "Async DFS write task threw: " << e.what();
                CompleteAsyncDfsWriteMetrics(context, {});
            } catch (...) {
                LOG(ERROR) << "Async DFS write task threw an unknown exception";
                CompleteAsyncDfsWriteMetrics(context, {});
            }
            release_inflight();
        });
    } catch (const std::exception& e) {
        // Enqueue may fail after the reporter has already settled the deadline.
        LOG(ERROR) << "Failed to enqueue async DFS write: " << e.what();
        release_inflight();
        const bool metrics_already_settled =
            context->service_metrics_settled.exchange(true);
        if (!metrics_already_settled) {
            for (const auto& metric : context->service_dual_writes) {
                if (metric.eligible) CancelServiceDualWritePending();
            }
        }
        for (const size_t index : op_indices) {
            auto& op = ops[index];
            if (metrics_already_settled && op.service_dual_write_eligible) {
                // The reporter owns this completion; the foreground must not
                // count it again when it handles the rejected write.
                op.service_metric_recorded = true;
            }
            op.transfer_summary.RecordFailure(
                ReplicaType::DFS, ErrorCode::DFS_SERVICE_UNAVAILABLE);
            op.AppendFailureContext("Failed to enqueue async DFS write");
        }
        return;
    }

    // The foreground never waits for DFS I/O in BUCKET mode. Publish the
    // successful local tier now; the background task exclusively owns DFS
    // End/Revoke and applies the deadline only to DFS publication.
    for (size_t i = 0; i < op_indices.size(); ++i) {
        auto& op = ops[op_indices[i]];
        op.transfer_summary.RecordSuccess(ReplicaType::DFS);
        op.transfer_summary.dfs_completion_deferred = true;
    }
}

void Client::SubmitDfsWrites(std::vector<PutOperation>& ops, bool is_upsert,
                             bool allow_async) {
    auto backend = dfs_storage_backend_;
    uint64_t total_dfs_payload_bytes = 0;
    for (const auto& op : ops) {
        if (op.IsResolved() ||
            op.transfer_summary.allocated_dfs_replicas == 0 ||
            !NonDfsTransfersSucceeded(op.transfer_summary)) {
            continue;
        }
        for (const auto& slice : op.slices) {
            SaturatingAddBytes(slice.size, &total_dfs_payload_bytes);
        }
    }
    const auto timeout_ms = DfsBatchPutTimeoutMs(
        dfs_batch_put_timeout_ms_, total_dfs_payload_bytes,
        dfs_batch_timeout_fixed_ms_.load(std::memory_order_acquire));
    const auto deadline = dfs_batch_put_timeout_ms_ == 0
                              ? std::chrono::steady_clock::time_point::max()
                              : DfsIoDeadlineFromNow(timeout_ms);
    const bool async_write =
        allow_async && backend &&
        backend->GetAllocatorType() == DfsAllocatorType::BUCKET;
    if (!async_write) {
        std::vector<size_t> all_indices(ops.size());
        std::iota(all_indices.begin(), all_indices.end(), 0);
        SubmitDfsWriteChunk(ops, all_indices, is_upsert, allow_async, deadline,
                            nullptr);
        return;
    }

    const size_t chunk_limit = dfs_write_chunk_size_;
    std::vector<std::vector<size_t>> chunks;
    std::vector<size_t> chunk;
    size_t chunk_bytes = 0;
    uint64_t total_payload_bytes = 0;
    const auto flush_chunk = [&]() {
        if (!chunk.empty()) {
            chunks.push_back(std::move(chunk));
            chunk.clear();
            chunk_bytes = 0;
        }
    };

    for (size_t i = 0; i < ops.size(); ++i) {
        auto& op = ops[i];
        if (op.IsResolved() ||
            op.transfer_summary.allocated_dfs_replicas == 0 ||
            !NonDfsTransfersSucceeded(op.transfer_summary)) {
            continue;
        }
        size_t object_bytes = 0;
        bool overflow = false;
        for (const auto& slice : op.slices) {
            if (slice.size >
                std::numeric_limits<size_t>::max() - object_bytes) {
                overflow = true;
                break;
            }
            object_bytes += slice.size;
        }
        if (overflow || object_bytes > chunk_limit) {
            op.transfer_summary.RecordFailure(ReplicaType::DFS,
                                              ErrorCode::TRANSFER_FAIL);
            op.AppendFailureContext("DFS object exceeds staging chunk limit");
            continue;
        }
        if (total_payload_bytes <=
            std::numeric_limits<uint64_t>::max() - object_bytes) {
            total_payload_bytes += object_bytes;
        } else {
            total_payload_bytes = std::numeric_limits<uint64_t>::max();
        }
        if (!chunk.empty() && object_bytes > chunk_limit - chunk_bytes) {
            flush_chunk();
        }
        chunk.push_back(i);
        chunk_bytes += object_bytes;
    }
    flush_chunk();

    if (dfs_read_trace_enabled()) {
        LOG(INFO) << "Async DFS write batch: total_payload_bytes="
                  << total_payload_bytes << ", chunks=" << chunks.size()
                  << ", chunk_limit_bytes=" << chunk_limit
                  << ", timeout_ms=" << timeout_ms;
    }

    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        bool staging_timed_out = false;
        bool staging_exceeds_limit = false;
        SubmitDfsWriteChunk(ops, chunks[chunk_index], is_upsert, allow_async,
                            deadline, &staging_timed_out,
                            &staging_exceeds_limit);
        if (staging_exceeds_limit) {
            LOG(WARNING) << "DFS staging request exceeds the unified buffer "
                            "limit; refusing chunk "
                         << chunk_index;
            continue;
        }
        if (staging_timed_out) {
            LOG(WARNING) << "DFS write staging allocation wait timed out; "
                         << "refusing remaining "
                         << chunks.size() - chunk_index - 1 << " chunks";
            for (size_t remaining = chunk_index + 1; remaining < chunks.size();
                 ++remaining) {
                for (const size_t op_index : chunks[remaining]) {
                    auto& op = ops[op_index];
                    if (!op.IsResolved() &&
                        op.transfer_summary.allocated_dfs_replicas > 0) {
                        op.transfer_summary.RecordFailure(
                            ReplicaType::DFS, ErrorCode::TRANSFER_FAIL);
                        op.AppendFailureContext(
                            "DFS staging allocation wait timed out");
                    }
                }
            }
            break;
        }
    }
}

std::vector<tl::expected<bool, ErrorCode>> Client::BatchIsExistByReplicaType(
    const std::vector<std::string>& keys, ReplicaType replica_type) {
    auto response =
        master_client_.BatchExistKeyByReplicaType(keys, replica_type);

    if (response.size() != keys.size()) {
        LOG(ERROR) << "BatchExistKeyByReplicaType response size mismatch. "
                   << "Expected: " << keys.size()
                   << ", Got: " << response.size();
        std::vector<tl::expected<bool, ErrorCode>> results;
        results.reserve(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            results.emplace_back(tl::unexpected(ErrorCode::RPC_FAIL));
        }
        return results;
    }

    return response;
}

ReplicateConfig Client::AttachConfig(const ReplicateConfig& config) const {
    ReplicateConfig client_cfg = config;
    if (!host_id_.empty()) {
        client_cfg.host_id = host_id_;
    }
    if (!DfsWriteAllowed()) {
        client_cfg.dfs_replica_num = 0;
    } else if (DistributedStorageConfig::IsReplicaEnabledFromEnvironment()) {
        client_cfg.dfs_replica_num = 1;
    }
    return client_cfg;
}

void Client::EnterHaRuntimeMode() {
    if (!leader_coordinator_) {
        return;
    }

    master_client_.EnableHaConnectionPolicy();
    if (!leader_monitor_running_.exchange(true)) {
        leader_monitor_thread_ =
            std::thread([this]() { this->LeaderMonitorThreadMain(); });
    }
}

}  // namespace mooncake
