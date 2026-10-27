#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

namespace mooncake {

enum class DfsBufferUsageKind : uint8_t {
    kRead = 0,
    kLocal = 1,
    kPrefetch = 2,
    kWrite = 3,
};

struct DfsBufferUsageSnapshot {
    uint64_t read_bytes;
    uint64_t local_bytes;
    uint64_t prefetch_bytes;
    uint64_t write_bytes;
    uint64_t total_bytes;
    uint64_t read_peak_bytes;
    uint64_t local_peak_bytes;
    uint64_t prefetch_peak_bytes;
    uint64_t write_peak_bytes;
    uint64_t peak_total_bytes;
    uint64_t read_capacity_bytes;
    uint64_t local_capacity_bytes;
    uint64_t prefetch_capacity_bytes;
    uint64_t write_capacity_bytes;
    uint64_t read_peak_utilization_tenths;
    uint64_t local_peak_utilization_tenths;
    uint64_t prefetch_peak_utilization_tenths;
    uint64_t write_peak_utilization_tenths;
    uint64_t write_waiting_requests;
    uint64_t write_waiting_peak_requests;
    uint64_t write_wait_total_requests;
    uint64_t write_wait_timeout_total;
    uint64_t write_wait_time_ms_total;
};

class DfsBufferUsageTracker {
   public:
    static std::shared_ptr<DfsBufferUsageTracker> ProcessTracker() {
        static auto tracker = std::make_shared<DfsBufferUsageTracker>();
        return tracker;
    }

    void AddCapacity(DfsBufferUsageKind kind, size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        capacities_[static_cast<size_t>(kind)] += capacity;
    }

    void RecordWriteWaitStart() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++write_waiting_requests_;
        ++write_waiting_peak_requests_;
        ++write_wait_total_requests_;
    }

    void RecordWriteWaitEnd(bool timed_out, uint64_t wait_time_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (write_waiting_requests_ > 0) --write_waiting_requests_;
        if (timed_out) ++write_wait_timeout_total_;
        write_wait_time_ms_total_ += wait_time_ms;
    }

    void Add(DfsBufferUsageKind kind, size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto index = static_cast<size_t>(kind);
        bytes_[index] += bytes;
        peaks_[index] = std::max(peaks_[index], bytes_[index]);
        peak_utilization_tenths_[index] =
            std::max(peak_utilization_tenths_[index],
                     UtilizationTenths(bytes_[index], capacities_[index]));
        total_bytes_ += bytes;
        peak_total_bytes_ = std::max(peak_total_bytes_, total_bytes_);
    }

    void Remove(DfsBufferUsageKind kind, size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& current = bytes_[static_cast<size_t>(kind)];
        current -= bytes;
        total_bytes_ -= bytes;
    }

    void Adjust(DfsBufferUsageKind kind, size_t old_bytes, size_t new_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& current = bytes_[static_cast<size_t>(kind)];
        if (new_bytes >= old_bytes) {
            current += new_bytes - old_bytes;
            total_bytes_ += new_bytes - old_bytes;
        } else {
            const auto delta = old_bytes - new_bytes;
            current -= delta;
            total_bytes_ -= delta;
        }
        peaks_[static_cast<size_t>(kind)] =
            std::max(peaks_[static_cast<size_t>(kind)], current);
        peak_utilization_tenths_[static_cast<size_t>(kind)] = std::max(
            peak_utilization_tenths_[static_cast<size_t>(kind)],
            UtilizationTenths(current, capacities_[static_cast<size_t>(kind)]));
        peak_total_bytes_ = std::max(peak_total_bytes_, total_bytes_);
    }

    [[nodiscard]] DfsBufferUsageSnapshot Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {bytes_[0],
                bytes_[1],
                bytes_[2],
                bytes_[3],
                total_bytes_,
                peaks_[0],
                peaks_[1],
                peaks_[2],
                peaks_[3],
                peak_total_bytes_,
                capacities_[0],
                capacities_[1],
                capacities_[2],
                capacities_[3],
                peak_utilization_tenths_[0],
                peak_utilization_tenths_[1],
                peak_utilization_tenths_[2],
                peak_utilization_tenths_[3],
                write_waiting_requests_,
                write_waiting_peak_requests_,
                write_wait_total_requests_,
                write_wait_timeout_total_,
                write_wait_time_ms_total_};
    }

   private:
    static uint64_t UtilizationTenths(uint64_t bytes, uint64_t capacity) {
        return capacity == 0 ? 0 : (bytes * 1000) / capacity;
    }

    mutable std::mutex mutex_;
    std::array<uint64_t, 4> bytes_{};
    std::array<uint64_t, 4> peaks_{};
    std::array<uint64_t, 4> capacities_{};
    std::array<uint64_t, 4> peak_utilization_tenths_{};
    uint64_t total_bytes_{0};
    uint64_t peak_total_bytes_{0};
    uint64_t write_waiting_requests_{0};
    uint64_t write_waiting_peak_requests_{0};
    uint64_t write_wait_total_requests_{0};
    uint64_t write_wait_timeout_total_{0};
    uint64_t write_wait_time_ms_total_{0};
};

class DfsBufferAllocationUsage {
   public:
    DfsBufferAllocationUsage(std::shared_ptr<DfsBufferUsageTracker> tracker,
                             DfsBufferUsageKind kind, size_t bytes)
        : tracker_(std::move(tracker)), kind_(kind), bytes_(bytes) {
        if (tracker_) tracker_->Add(kind_, bytes_);
    }

    ~DfsBufferAllocationUsage() {
        if (tracker_) tracker_->Remove(kind_, bytes_);
    }

    void SetBytes(size_t bytes) {
        if (tracker_) tracker_->Adjust(kind_, bytes_, bytes);
        bytes_ = bytes;
    }

    DfsBufferAllocationUsage(const DfsBufferAllocationUsage&) = delete;
    DfsBufferAllocationUsage& operator=(const DfsBufferAllocationUsage&) =
        delete;

   private:
    std::shared_ptr<DfsBufferUsageTracker> tracker_;
    DfsBufferUsageKind kind_;
    size_t bytes_;
};

}  // namespace mooncake
