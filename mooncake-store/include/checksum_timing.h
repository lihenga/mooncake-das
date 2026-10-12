#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <time.h>
#include <unistd.h>

#include <glog/logging.h>

#include "environ.h"

namespace mooncake {

// Enabled by default; set MC_STORE_CHECKSUM_TIMING=0 to disable diagnostics.
// Process-wide counters, with no background thread or per-key logging.
// Counters are cumulative; stages nest, so their times must not be summed.
enum class ChecksumStage {
    WRITE_CHECKSUM,
    READ_CHECKSUM,
    PREFETCH_CHECKSUM,
    FOREGROUND_CHECKSUM,
    BATCH_PUT,
    BATCH_UPSERT,
    WRITE_ALLOCATION,
    WRITE_BUFFER_STAGE,
    WRITE_TRANSFER_SUBMIT,
    WRITE_TRANSFER_WAIT,
    WRITE_TRANSFER_MERGED,
    DFS_SUBMIT,
    DFS_STAGE,
    DFS_WRITE_IO,
    WRITE_FINALIZE,
    DFS_READ_IO,
    DFS_PREFETCH_READ_IO,
    DFS_FOREGROUND_READ_IO,
    COUNT,
};

class ChecksumTiming {
   public:
    // The existing DFS session path reads and verifies on this thread before
    // scattering to the GPU. Preserve the context across nested calls.
    class ReadScope {
       public:
        explicit ReadScope(bool prefetch) : previous_(ReadStage()) {
            ReadStage() = prefetch ? ChecksumStage::PREFETCH_CHECKSUM
                                   : ChecksumStage::FOREGROUND_CHECKSUM;
        }
        ~ReadScope() { ReadStage() = previous_; }
        ReadScope(const ReadScope&) = delete;
        ReadScope& operator=(const ReadScope&) = delete;

       private:
        const ChecksumStage previous_;
    };
    static ChecksumStage& ReadStage() {
        static thread_local ChecksumStage stage = ChecksumStage::READ_CHECKSUM;
        return stage;
    }
    static ChecksumStage ReadIoStage() {
        if (ReadStage() == ChecksumStage::PREFETCH_CHECKSUM) {
            return ChecksumStage::DFS_PREFETCH_READ_IO;
        }
        if (ReadStage() == ChecksumStage::FOREGROUND_CHECKSUM) {
            return ChecksumStage::DFS_FOREGROUND_READ_IO;
        }
        return ChecksumStage::DFS_READ_IO;
    }
    static bool Enabled() {
        static const bool enabled =
            Environ::GetBool("MC_STORE_CHECKSUM_TIMING", true);
        return enabled;
    }

    static uint64_t Now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    static uint64_t ThreadCpuNow() {
        timespec value{};
        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) return 0;
        return uint64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
    }

    explicit ChecksumTiming(ChecksumStage stage, uint64_t bytes = 0)
        : enabled_(Enabled()), stage_(stage), bytes_(bytes) {
        if (!enabled_) return;
        started_ = Now();
        cpu_started_ = ThreadCpuNow();
        auto& counters = Get().stages[static_cast<size_t>(stage_)];
        counters.active.fetch_add(1, std::memory_order_relaxed);
        counters.active_bytes.fetch_add(bytes_, std::memory_order_relaxed);
    }

    ~ChecksumTiming() { Finish(); }
    ChecksumTiming(const ChecksumTiming&) = delete;
    ChecksumTiming& operator=(const ChecksumTiming&) = delete;

    uint64_t StartPart() const { return enabled_ ? Now() : 0; }
    void AddCrc(uint64_t start) {
        if (enabled_) crc_ns_ += Now() - start;
    }
    void AddD2h(uint64_t start, uint64_t bytes) {
        if (!enabled_) return;
        d2h_ns_ += Now() - start;
        gpu_bytes_ += bytes;
    }
    void AddStaging(uint64_t start) {
        if (enabled_) staging_ns_ += Now() - start;
    }

    void Finish() {
        if (!enabled_ || finished_) return;
        finished_ = true;
        const auto end = Now();
        const auto cpu_end = ThreadCpuNow();
        auto& state = Get();
        auto& c = state.stages[static_cast<size_t>(stage_)];
        c.wall_ns.fetch_add(end - started_, std::memory_order_relaxed);
        if (cpu_started_ != 0 && cpu_end >= cpu_started_) {
            c.cpu_ns.fetch_add(cpu_end - cpu_started_,
                               std::memory_order_relaxed);
        }
        c.bytes.fetch_add(bytes_, std::memory_order_relaxed);
        c.crc_ns.fetch_add(crc_ns_, std::memory_order_relaxed);
        c.d2h_ns.fetch_add(d2h_ns_, std::memory_order_relaxed);
        c.staging_ns.fetch_add(staging_ns_, std::memory_order_relaxed);
        c.gpu_bytes.fetch_add(gpu_bytes_, std::memory_order_relaxed);
        auto previous = c.max_ns.load(std::memory_order_relaxed);
        while (previous < end - started_ &&
               !c.max_ns.compare_exchange_weak(previous, end - started_,
                                               std::memory_order_relaxed)) {
        }
        c.calls.fetch_add(1, std::memory_order_relaxed);
        c.active_bytes.fetch_sub(bytes_, std::memory_order_relaxed);
        c.active.fetch_sub(1, std::memory_order_relaxed);
        ReportIfDue(state, end);
    }

    static void ReadResult(bool present, bool verified) {
        if (!Enabled()) return;
        auto& state = Get();
        auto& counter =
            !present ? state.read_absent
                     : (verified ? state.read_verified : state.read_failed);
        counter.fetch_add(1, std::memory_order_relaxed);
    }

   private:
    struct Counters {
        std::atomic<uint64_t> calls{0}, bytes{0}, wall_ns{0}, cpu_ns{0},
            max_ns{0};
        std::atomic<uint64_t> crc_ns{0}, d2h_ns{0}, staging_ns{0}, gpu_bytes{0};
        std::atomic<uint64_t> active{0}, active_bytes{0};
    };
    struct State {
        const uint64_t started = Now();
        std::atomic<uint64_t> next_report{started + 10000000000ULL};
        std::atomic<uint64_t> read_absent{0}, read_verified{0}, read_failed{0};
        std::array<Counters, static_cast<size_t>(ChecksumStage::COUNT)> stages;
    };
    static State& Get() {
        static State state;
        return state;
    }
    static void ReportIfDue(State& state, uint64_t now) {
        auto deadline = state.next_report.load(std::memory_order_relaxed);
        if (now < deadline ||
            !state.next_report.compare_exchange_strong(
                deadline, now + 10000000000ULL, std::memory_order_relaxed)) {
            return;
        }
        static constexpr const char* names[] = {"write_checksum",
                                                "read_checksum",
                                                "prefetch_checksum",
                                                "foreground_checksum",
                                                "batch_put",
                                                "batch_upsert",
                                                "write_allocation",
                                                "write_buffer_stage",
                                                "write_transfer_submit",
                                                "write_transfer_wait",
                                                "write_transfer_merged",
                                                "dfs_submit",
                                                "dfs_stage",
                                                "dfs_write_io",
                                                "write_finalize",
                                                "dfs_read_io",
                                                "dfs_prefetch_read_io",
                                                "dfs_foreground_read_io"};
        static_assert(std::size(names) ==
                      static_cast<size_t>(ChecksumStage::COUNT));
        const auto elapsed_ms = (now - state.started) / 1000000;
        // Snapshots are approximate under concurrency. Interval costs can be
        // derived by subtracting successive cumulative totals for each stage.
        LOG(INFO) << "checksum_timing pid=" << getpid()
                  << " elapsed_ms=" << elapsed_ms
                  << " read_verified=" << state.read_verified.load()
                  << " read_absent=" << state.read_absent.load()
                  << " read_failed=" << state.read_failed.load();
        for (size_t i = 0; i < state.stages.size(); ++i) {
            const auto& c = state.stages[i];
            const auto calls = c.calls.load(std::memory_order_relaxed);
            const auto active = c.active.load(std::memory_order_relaxed);
            if (calls == 0 && active == 0) continue;
            LOG(INFO) << "checksum_timing pid=" << getpid()
                      << " elapsed_ms=" << elapsed_ms << " stage=" << names[i]
                      << " calls=" << calls
                      << " bytes=" << c.bytes.load(std::memory_order_relaxed)
                      << " wall_us=" << c.wall_ns.load() / 1000
                      << " cpu_us=" << c.cpu_ns.load() / 1000
                      << " max_us=" << c.max_ns.load() / 1000
                      << " crc_us=" << c.crc_ns.load() / 1000
                      << " d2h_us=" << c.d2h_ns.load() / 1000
                      << " staging_us=" << c.staging_ns.load() / 1000
                      << " gpu_bytes=" << c.gpu_bytes.load()
                      << " active_calls=" << active
                      << " active_bytes=" << c.active_bytes.load();
        }
    }

    const bool enabled_;
    const ChecksumStage stage_;
    const uint64_t bytes_;
    bool finished_ = false;
    uint64_t started_ = 0, cpu_started_ = 0;
    uint64_t crc_ns_ = 0, d2h_ns_ = 0, staging_ns_ = 0, gpu_bytes_ = 0;
};

}  // namespace mooncake
