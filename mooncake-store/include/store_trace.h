#pragma once

#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace mooncake {

// Diagnostics only: no GPU queries, events, synchronization, or payload reads.
// One process-wide watchdog also reports calls that never reach their end log.
// Phase names must be string literals. Counters belong to the executing thread.
class StoreTrace {
   public:
    using Clock = std::chrono::steady_clock;

    static bool Enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("MC_STORE_DFS_DIAGNOSTICS");
            return value == nullptr || std::string(value) != "0";
        }();
        return enabled;
    }

    StoreTrace(const char* operation, const void* owner, size_t objects,
               uint64_t parent = CurrentId(), bool bind_thread = true,
               bool log_fast_calls = true)
        : operation_(operation),
          owner_(owner),
          objects_(objects),
          parent_(parent),
          previous_(current_),
          bind_thread_(bind_thread),
          log_fast_calls_(log_fast_calls),
          started_(Clock::now()),
          phase_started_(started_),
          last_report_(started_) {
        if (!Enabled()) return;
        id_ = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (log_fast_calls_ && (parent_ == 0 || !bind_thread_)) {
            LOG(INFO) << "STORE_TRACE_BEGIN" << Identity();
        }
        auto& registry = GetRegistry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.active.push_back(this);
        if (bind_thread_) current_ = this;
    }

    ~StoreTrace() {
        if (id_ == 0) return;
        {
            auto& registry = GetRegistry();
            std::lock_guard<std::mutex> lock(registry.mutex);
            auto& active = registry.active;
            active.erase(std::remove(active.begin(), active.end(), this),
                         active.end());
        }
        Phase("done");
        const auto total_us = Elapsed(started_, phase_started_);
        completed_.fetch_add(1, std::memory_order_relaxed);
        if (log_fast_calls_ || total_us >= 100000 ||
            std::uncaught_exceptions()) {
            for (size_t i = 0; i < phase_count_; ++i) {
                fields_ << ' ' << phases_[i].name << "_us=" << phases_[i].us;
            }
            if (other_phase_us_ != 0)
                fields_ << " other_phases_us=" << other_phase_us_;
            LOG(INFO) << "STORE_TRACE_END" << Identity() << " result="
                      << (std::uncaught_exceptions() ? "exception" : result_)
                      << " total_us=" << total_us << fields_.str();
        }
        if (bind_thread_) current_ = previous_;
    }

    StoreTrace(const StoreTrace&) = delete;
    StoreTrace& operator=(const StoreTrace&) = delete;

    uint64_t id() const { return id_; }
    static uint64_t CurrentId() { return current_ ? current_->id_ : 0; }
    static StoreTrace* Current() { return current_; }

    void Phase(const char* phase) {
        if (id_ == 0) return;
        const auto now = Clock::now();
        std::lock_guard<std::mutex> lock(phase_mutex_);
        size_t index = 0;
        while (index < phase_count_ && phases_[index].name != phase_) ++index;
        if (index < phases_.size()) {
            if (index == phase_count_) phases_[phase_count_++].name = phase_;
            phases_[index].us += Elapsed(phase_started_, now);
        } else {
            other_phase_us_ += Elapsed(phase_started_, now);
        }
        phase_ = phase;
        phase_started_ = now;
        executor_ = std::this_thread::get_id();
    }

    template <typename T>
    void Field(const char* name, const T& value) {
        if (id_ != 0) fields_ << ' ' << name << '=' << value;
    }
    void Result(const char* result) { result_ = result; }
    void ForceSummary() { log_fast_calls_ = true; }
    template <typename T>
    void Results(const std::vector<T>& results) {
        size_t failures = 0;
        int first_error = 0;
        for (const auto& result : results) {
            if (!result.has_value()) {
                if (failures++ == 0)
                    first_error = static_cast<int>(result.error());
            }
        }
        Field("failures", failures);
        Field("first_error", first_error);
        Result(failures == 0 ? "success" : "failed");
    }
    void Keys(const std::vector<std::string>& keys) {
        if (keys.empty() || id_ == 0) return;
        auto hash = [](const std::string& key) {
            uint64_t value = 14695981039346656037ULL;
            for (unsigned char c : key) value = (value ^ c) * 1099511628211ULL;
            return value;
        };
        Field("first_key_hash", hash(keys.front()));
        Field("last_key_hash", hash(keys.back()));
    }
    void Codes(const std::vector<int>& codes) {
        size_t failures = 0;
        int first_error = 0;
        for (int code : codes) {
            if (code < 0 && failures++ == 0) first_error = code;
        }
        Field("failures", failures);
        Field("first_error", first_error);
        Result(failures == 0 ? "success" : "failed");
    }
    static void Mark(const char* phase) {
        if (current_) current_->Phase(phase);
    }
    template <typename T>
    static void Count(const char* name, const T& value) {
        if (current_) current_->Field(name, value);
    }

    // Public for deterministic tests; production is polled every two seconds.
    // At most one stall report per call per five seconds, not per slice/key.
    static void ReportStalls(Clock::time_point now = Clock::now()) {
        if (!Enabled()) return;
        auto& registry = GetRegistry();
        std::string heartbeat;
        std::vector<std::string> stalls;
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            if (now - registry.last_heartbeat >= std::chrono::seconds(10)) {
                registry.last_heartbeat = now;
                std::ostringstream out;
                out << "STORE_TRACE_HEARTBEAT pid=" << getpid()
                    << " active=" << registry.active.size() << " started="
                    << sequence_.load(std::memory_order_relaxed)
                    << " completed="
                    << completed_.load(std::memory_order_relaxed);
                heartbeat = out.str();
            }
            for (auto* trace : registry.active) {
                std::lock_guard<std::mutex> phase_lock(trace->phase_mutex_);
                if (now - trace->started_ < std::chrono::seconds(5) ||
                    now - trace->last_report_ < std::chrono::seconds(5))
                    continue;
                trace->last_report_ = now;
                std::ostringstream out;
                out << "STORE_TRACE_STALL" << trace->Identity()
                    << " executor=" << trace->executor_
                    << " phase=" << trace->phase_
                    << " phase_us=" << Elapsed(trace->phase_started_, now)
                    << " total_us=" << Elapsed(trace->started_, now);
                stalls.push_back(out.str());
            }
        }
        // Snapshot while traces are alive, but never hold their locks across
        // log I/O: a slow log sink must not stall every native API call.
        if (!heartbeat.empty()) LOG(INFO) << heartbeat;
        for (const auto& stall : stalls) LOG(WARNING) << stall;
    }

   private:
    struct Registry {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<StoreTrace*> active;
        Clock::time_point last_heartbeat = Clock::now();
        bool stopping = false;
        std::thread watchdog;
        Registry()
            : watchdog([this] {
                  std::unique_lock<std::mutex> lock(mutex);
                  while (!cv.wait_for(lock, std::chrono::seconds(2),
                                      [this] { return stopping; })) {
                      lock.unlock();
                      StoreTrace::ReportStalls();
                      lock.lock();
                  }
              }) {}
        ~Registry() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            cv.notify_one();
            watchdog.join();
        }
    };
    static Registry& GetRegistry() {
        static Registry registry;
        return registry;
    }
    static int64_t Elapsed(Clock::time_point start, Clock::time_point end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
            .count();
    }
    std::string Identity() const {
        std::ostringstream out;
        out << " pid=" << getpid() << " owner=" << owner_ << " trace=" << id_
            << " parent=" << parent_ << " op=" << operation_
            << " objects=" << objects_;
        return out.str();
    }

    inline static std::atomic<uint64_t> sequence_{0};
    inline static std::atomic<uint64_t> completed_{0};
    inline static thread_local StoreTrace* current_ = nullptr;
    const char* operation_;
    const void* owner_;
    size_t objects_;
    uint64_t parent_;
    StoreTrace* previous_;
    bool bind_thread_;
    bool log_fast_calls_;
    uint64_t id_ = 0;
    Clock::time_point started_;
    Clock::time_point phase_started_;
    Clock::time_point last_report_;
    std::mutex phase_mutex_;
    const char* phase_ = "entry";
    std::thread::id executor_ = std::this_thread::get_id();
    const char* result_ = "unfinished";
    struct PhaseTime {
        const char* name = nullptr;
        int64_t us = 0;
    };
    std::array<PhaseTime, 64> phases_{};
    size_t phase_count_ = 0;
    int64_t other_phase_us_ = 0;
    std::ostringstream fields_;
};

}  // namespace mooncake
