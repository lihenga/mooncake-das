#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace mooncake {

enum class DfsBufferUsageKind : uint8_t { kRead, kPrefetch, kWrite };

struct DfsBufferUsageSnapshot {
    uint64_t read_bytes;
    uint64_t prefetch_bytes;
    uint64_t write_bytes;
    uint64_t total_bytes;
    uint64_t peak_total_bytes;
};

class DfsBufferUsageTracker {
   public:
    static std::shared_ptr<DfsBufferUsageTracker> ProcessTracker() {
        static auto tracker = std::make_shared<DfsBufferUsageTracker>();
        return tracker;
    }

    void Add(DfsBufferUsageKind kind, size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        bytes_[static_cast<size_t>(kind)] += bytes;
        total_bytes_ += bytes;
        peak_total_bytes_ = std::max(peak_total_bytes_, total_bytes_);
    }

    void Remove(DfsBufferUsageKind kind, size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        bytes_[static_cast<size_t>(kind)] -= bytes;
        total_bytes_ -= bytes;
    }

    [[nodiscard]] DfsBufferUsageSnapshot Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {bytes_[0], bytes_[1], bytes_[2], total_bytes_,
                peak_total_bytes_};
    }

   private:
    mutable std::mutex mutex_;
    std::array<uint64_t, 3> bytes_{};
    uint64_t total_bytes_{0};
    uint64_t peak_total_bytes_{0};
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

    DfsBufferAllocationUsage(const DfsBufferAllocationUsage&) = delete;
    DfsBufferAllocationUsage& operator=(const DfsBufferAllocationUsage&) =
        delete;

   private:
    std::shared_ptr<DfsBufferUsageTracker> tracker_;
    DfsBufferUsageKind kind_;
    size_t bytes_;
};

}  // namespace mooncake
