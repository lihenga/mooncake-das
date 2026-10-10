#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <thread>
#include <utility>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace mooncake {

inline bool KVSessionTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("MOONCAKE_KV_SESSION_TRACE");
        return !value || std::string(value) != "0";
    }();
    return enabled;
}

// Explicitly propagated into native read-plan worker threads.
// Defined in the store library so the Python extension and native workers share
// TLS.
std::string& KVSessionTraceContext();
class KVSessionTraceScope {
   public:
    explicit KVSessionTraceScope(const std::string& context)
        : previous_(std::exchange(KVSessionTraceContext(), context)) {}
    ~KVSessionTraceScope() { KVSessionTraceContext() = std::move(previous_); }

   private:
    std::string previous_;
};

// Default-on diagnostics. Hashes are stable across processes; never log keys
// or payloads. Per-operation event budgets bound diagnostic output.
inline uint64_t KVSessionKeyHash(const std::string& key) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : key) hash = (hash ^ c) * 1099511628211ULL;
    return hash;
}

class KVSessionDiagnosticBatch {
   public:
    using Clock = std::chrono::steady_clock;
    KVSessionDiagnosticBatch(const char* operation, const void* owner,
                             size_t keys)
        : operation_(operation),
          owner_(owner),
          keys_(keys),
          started_(Clock::now()),
          parent_(current_) {
        if (!KVSessionTraceEnabled()) return;
        id_ = ++sequence_;
        current_ = id_;
        const auto log_started = Clock::now();
        LOG(INFO) << "KV_SESSION_BEGIN" << Identity();
        Elapsed("begin_log_us", log_started);
    }
    ~KVSessionDiagnosticBatch() {
        if (!KVSessionTraceEnabled()) return;
        // Callers construct this before taking session_mutex_. Buffered events
        // are flushed after releasing that mutex, including on early returns.
        for (const auto& item : timings_) {
            details_ << ' ' << item.first << "_sum=" << item.second.first << ' '
                     << item.first << "_max=" << item.second.second;
        }
        const auto emit_started = Clock::now();
        for (const auto& line : events_)
            LOG(INFO) << "KV_SESSION_EVENT" << Identity() << line;
        Field("event_emit_us",
              std::chrono::duration_cast<std::chrono::microseconds>(
                  Clock::now() - emit_started)
                  .count());
        LOG(INFO) << "KV_SESSION_END" << Identity() << " total_us="
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - started_)
                         .count()
                  << " observed=" << observed_ << " failures=" << failures_
                  << " omitted_events=" << omitted_ << details_.str()
                  << " reasons=" << Reasons();
        current_ = parent_;
    }
    KVSessionDiagnosticBatch(const KVSessionDiagnosticBatch&) = delete;
    KVSessionDiagnosticBatch& operator=(const KVSessionDiagnosticBatch&) =
        delete;
    uint64_t id() const { return id_; }
    static uint64_t CurrentId() { return current_; }
    template <typename T>
    void Field(const char* name, const T& value) {
        if (KVSessionTraceEnabled()) details_ << ' ' << name << '=' << value;
    }
    // Account every event, but serialize only four examples per event/reason.
    // This is checked BEFORE formatting hashes/history strings under the lock.
    bool Admit(bool failed, const std::string& reason) {
        if (!KVSessionTraceEnabled()) return false;
        ++observed_;
        failures_ += failed;
        if (++reasons_[reason] <= 4) return true;
        ++omitted_;
        return false;
    }
    void Sample(const std::string& line) { events_.push_back(line); }
    void Elapsed(const char* name, Clock::time_point since) {
        if (!KVSessionTraceEnabled()) return;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                            Clock::now() - since)
                            .count();
        auto& timing = timings_[name];
        timing.first += us;
        timing.second = std::max(timing.second, static_cast<int64_t>(us));
    }

   private:
    std::string Reasons() const {
        std::ostringstream out;
        for (const auto& item : reasons_)
            out << item.first << ':' << item.second << ',';
        return out.str();
    }
    std::string Identity() const {
        std::ostringstream out;
        out << " schema=2 op_id=" << id_ << " parent_id=" << parent_
            << " operation=" << operation_ << " client=" << owner_
            << " keys=" << keys_ << " trace_id=" << KVSessionTraceContext()
            << " thread=" << std::this_thread::get_id() << " wall_us="
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()
            << " mono_us="
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   Clock::now().time_since_epoch())
                   .count();
        return out.str();
    }
    inline static std::atomic<uint64_t> sequence_{0};
    inline static thread_local uint64_t current_{0};
    const char* operation_;
    const void* owner_;
    size_t keys_;
    Clock::time_point started_;
    uint64_t parent_, id_{0};
    size_t observed_{0}, failures_{0}, omitted_{0};
    std::map<std::string, size_t> reasons_;
    std::map<std::string, std::pair<int64_t, int64_t>> timings_;
    std::vector<std::string> events_;
    std::ostringstream details_;
};

// Every access must hold the owning RealClient's session_mutex_. This bounded
// history is independent of get_sessions_; it never grants or extends a lease.
class KVSessionDiagnosticHistory {
   public:
    using Clock = KVSessionDiagnosticBatch::Clock;
    void Observe(const char* file, int line, KVSessionDiagnosticBatch& batch,
                 const std::string& key, size_t index, const char* event,
                 int rc, bool exists, Clock::time_point deadline = {},
                 bool prefetched = false, const std::string& source = "unknown",
                 const std::string& detail = "") {
        if (!KVSessionTraceEnabled()) return;
        const auto now = Clock::now();
        auto it = history_.find(key);
        if (it == history_.end()) {
            if (history_.size() >= 65536) {
                history_.erase(history_.begin());
                ++evicted_;
            }
            it = history_.emplace(key, State{}).first;
        }
        auto& state = it->second;
        const std::string action(event);
        if (action == "start_query" || action == "refresh_query")
            ++state.queries;
        if (action == "refresh_attempt") ++state.refresh_attempts;
        if (action == "start" && rc == 0) {
            state.generation = ++generation_;
            ++state.starts;
            state.created = state.refreshed = now;
            state.creator_trace = KVSessionTraceContext();
            state.generation_refresh_ok = 0;
            state.first_read = {};
            state.refresh_trace = "none";
        }
        if (action == "start" || action == "retire" ||
            (action == "end" && exists) || (action == "refresh" && rc == 0)) {
            state.last_op = batch.id();
            state.mutated = now;
            state.last_event = detail.empty() ? action : detail;
            state.last_trace = KVSessionTraceContext();
            if (action == "refresh") {
                ++state.refresh_ok;
                ++state.generation_refresh_ok;
                state.refresh_trace = KVSessionTraceContext();
                state.refreshed = now;
            }
        }
        if ((action == "read_admitted" || action == "read" ||
             detail == "range_lease_expired") &&
            state.first_read == Clock::time_point{}) {
            state.first_read = now;
        }
        if (deadline != Clock::time_point{}) state.deadline = deadline;
        if (!batch.Admit(rc < 0, detail.empty()
                                     ? action
                                     : detail.substr(0, detail.find(' '))))
            return;
        std::ostringstream out;
        out << " site=" << file << ":" << line
            << " key_hash=" << KVSessionKeyHash(key) << " key_index=" << index
            << " observed_mono_us="
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   now.time_since_epoch())
                   .count()
            << " observed_wall_us="
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()
            << " event=" << event << " rc=" << rc
            << " session_exists=" << exists << " source=" << source
            << " prefetch_cached=" << prefetched << " lease_remaining_ms="
            << (deadline == Clock::time_point{} ? 0 : Millis(deadline - now))
            << " deadline_source="
            << (deadline == Clock::time_point{} ? "history"
                                                : "current_observation")
            << " deadline_mono_us="
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   state.deadline.time_since_epoch())
                   .count()
            << " deadline_wall_us="
            << (state.deadline == Clock::time_point{}
                    ? 0
                    : std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                              .count() +
                          std::chrono::duration_cast<std::chrono::microseconds>(
                              state.deadline - now)
                              .count())
            << " history_known=" << (state.generation != 0)
            << " history_evictions=" << evicted_;
        if (it != history_.end()) {
            const auto& state = it->second;
            out << " generation=" << state.generation << " age_ms="
                << (state.generation ? Millis(now - state.created) : -1)
                << " since_refresh_ms="
                << (state.generation ? Millis(now - state.refreshed) : -1)
                << " since_mutation_ms="
                << (state.last_op ? Millis(now - state.mutated) : -1)
                << " last_mutation_op=" << state.last_op
                << " last_event=" << state.last_event
                << " creator_trace=" << state.creator_trace
                << " last_mutation_trace=" << state.last_trace
                << " starts=" << state.starts
                << " query_count=" << state.queries
                << " refresh_attempts=" << state.refresh_attempts
                << " refresh_ok=" << state.refresh_ok
                << " generation_refresh_ok=" << state.generation_refresh_ok
                << " last_refresh_trace=" << state.refresh_trace
                << " start_to_first_read_ms="
                << (state.generation && state.first_read != Clock::time_point{}
                        ? Millis(state.first_read - state.created)
                        : -1);
        }
        if (!detail.empty()) out << " reason=" << detail;
        batch.Sample(out.str());
    }

   private:
    static int64_t Millis(Clock::duration value) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(value)
            .count();
    }
    struct State {
        uint64_t generation{0}, last_op{0};
        uint64_t starts{0}, queries{0}, refresh_attempts{0}, refresh_ok{0},
            generation_refresh_ok{0};
        Clock::time_point created{}, refreshed{}, mutated{}, deadline{},
            first_read{};
        std::string creator_trace{"none"}, last_trace{"none"},
            refresh_trace{"none"};
        std::string last_event;
    };
    uint64_t generation_{0}, evicted_{0};
    std::unordered_map<std::string, State> history_;
};

}  // namespace mooncake
