#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include <glog/logging.h>

namespace mooncake {

inline constexpr uint64_t kHighWatermarkNumerator = 8;
inline constexpr uint64_t kHighWatermarkDenominator = 10;

enum class DfsBufferUsageKind : uint8_t { kRead, kPrefetch, kWrite };

struct DfsBufferUsageSnapshot {
    uint64_t read_bytes = 0;
    uint64_t prefetch_bytes = 0;
    uint64_t write_bytes = 0;
    uint64_t total_bytes = 0;
    uint64_t peak_total_bytes = 0;
};

class DfsBufferUsageTracker {
   public:
    explicit DfsBufferUsageTracker(uint64_t capacity_bytes = 0)
        : capacity_bytes_(capacity_bytes) {}

    void Add(DfsBufferUsageKind kind, uint64_t bytes) {
        bool warn = false;
        uint64_t total = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& value = bytes_[static_cast<size_t>(kind)];
            value += bytes;
            total_bytes_ += bytes;
            peak_total_bytes_ = std::max(peak_total_bytes_, total_bytes_);
            if (capacity_bytes_ > 0) {
                const bool above_threshold =
                    static_cast<unsigned __int128>(total_bytes_) *
                        kHighWatermarkDenominator >=
                    static_cast<unsigned __int128>(capacity_bytes_) *
                        kHighWatermarkNumerator;
                warn = above_threshold && !high_watermark_warned_;
                high_watermark_warned_ = above_threshold;
                total = total_bytes_;
            }
        }
        if (warn) {
            LOG(WARNING) << "DFS_UNIFIED_BUFFER_HIGH_USAGE total_bytes="
                         << total << " capacity_bytes=" << capacity_bytes_
                         << " utilization="
                         << (static_cast<double>(total) * 100.0 /
                             static_cast<double>(capacity_bytes_))
                         << "%";
        }
    }

    void Remove(DfsBufferUsageKind kind, uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& value = bytes_[static_cast<size_t>(kind)];
        value = value >= bytes ? value - bytes : 0;
        total_bytes_ = total_bytes_ >= bytes ? total_bytes_ - bytes : 0;
        if (capacity_bytes_ > 0 &&
            static_cast<unsigned __int128>(total_bytes_) *
                    kHighWatermarkDenominator <
                static_cast<unsigned __int128>(capacity_bytes_) *
                    kHighWatermarkNumerator) {
            high_watermark_warned_ = false;
        }
    }

    [[nodiscard]] DfsBufferUsageSnapshot Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {bytes_[0], bytes_[1], bytes_[2], total_bytes_,
                peak_total_bytes_};
    }

   private:
    mutable std::mutex mutex_;
    std::array<uint64_t, 3> bytes_{};
    uint64_t total_bytes_ = 0;
    uint64_t peak_total_bytes_ = 0;
    uint64_t capacity_bytes_ = 0;
    bool high_watermark_warned_ = false;
};

class DfsBufferAllocationUsage {
   public:
    DfsBufferAllocationUsage(std::shared_ptr<DfsBufferUsageTracker> tracker,
                             DfsBufferUsageKind kind, uint64_t bytes)
        : tracker_(std::move(tracker)), kind_(kind), bytes_(bytes) {
        if (tracker_) tracker_->Add(kind_, bytes_);
    }

    ~DfsBufferAllocationUsage() {
        if (tracker_) tracker_->Remove(kind_, bytes_);
    }

    DfsBufferAllocationUsage(const DfsBufferAllocationUsage&) = delete;
    DfsBufferAllocationUsage& operator=(const DfsBufferAllocationUsage&) = delete;

   private:
    std::shared_ptr<DfsBufferUsageTracker> tracker_;
    DfsBufferUsageKind kind_;
    uint64_t bytes_;
};

}  // namespace mooncake
