#pragma once

#include <glog/logging.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace mooncake {

// Always-on diagnostics. Hashes are stable across processes; never log keys
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
        id_ = ++sequence_;
        current_ = id_;
        LOG(INFO) << "KV_SESSION_BEGIN" << Identity();
    }
    ~KVSessionDiagnosticBatch() {
        // Callers construct this before taking session_mutex_. Buffered events
        // are flushed after releasing that mutex, including on early returns.
        for (const auto& line : events_)
            LOG(INFO) << "KV_SESSION_EVENT" << Identity() << line;
        LOG(INFO) << "KV_SESSION_END" << Identity() << " total_us="
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - started_)
                         .count()
                  << " observed=" << observed_ << " failures=" << failures_
                  << " omitted_events=" << omitted_ << details_.str();
        current_ = parent_;
    }
    KVSessionDiagnosticBatch(const KVSessionDiagnosticBatch&) = delete;
    KVSessionDiagnosticBatch& operator=(const KVSessionDiagnosticBatch&) =
        delete;
    uint64_t id() const { return id_; }
    static uint64_t CurrentId() { return current_; }
    template <typename T>
    void Field(const char* name, const T& value) {
        details_ << ' ' << name << '=' << value;
    }
    void Event(bool failed, const std::string& line) {
        ++observed_;
        failures_ += failed;
        // Failure events have their own budget so normal events cannot hide
        // the failure that motivated this diagnostic run.
        size_t& count = failed ? failed_events_ : normal_events_;
        if (count++ < 64)
            events_.push_back(line);
        else
            ++omitted_;
    }

   private:
    std::string Identity() const {
        std::ostringstream out;
        out << " op_id=" << id_ << " parent_id=" << parent_
            << " operation=" << operation_ << " client=" << owner_
            << " keys=" << keys_;
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
    size_t failed_events_{0}, normal_events_{0};
    std::vector<std::string> events_;
    std::ostringstream details_;
};

// Every access must hold the owning RealClient's session_mutex_. This bounded
// history is independent of get_sessions_; it never grants or extends a lease.
class KVSessionDiagnosticHistory {
   public:
    using Clock = KVSessionDiagnosticBatch::Clock;
    void Observe(KVSessionDiagnosticBatch& batch, const std::string& key,
                 size_t index, const char* event, int rc, bool exists,
                 Clock::time_point deadline = {}, bool prefetched = false,
                 const std::string& source = "unknown",
                 const std::string& detail = "") {
        const auto now = Clock::now();
        auto it = history_.find(key);
        const std::string action(event);
        if (action == "start" && rc == 0) {
            if (it == history_.end()) {
                if (history_.size() >= 65536) {
                    history_.erase(history_.begin());
                    ++evicted_;
                }
                it = history_.emplace(key, State{}).first;
            }
            it->second =
                State{++generation_, batch.id(), now, now, now, "start"};
        } else if (it != history_.end() &&
                   (action == "retire" || (action == "end" && exists) ||
                    (action == "refresh" && rc == 0))) {
            it->second.last_op = batch.id();
            it->second.mutated = now;
            it->second.last_event = event;
            if (action == "refresh") it->second.refreshed = now;
            if (action == "retire") it->second.last_event = detail;
        }
        std::ostringstream out;
        out << " key_hash=" << KVSessionKeyHash(key) << " key_index=" << index
            << " observed_mono_us="
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   now.time_since_epoch())
                   .count()
            << " event=" << event << " rc=" << rc
            << " session_exists=" << exists << " source=" << source
            << " prefetch_cached=" << prefetched << " lease_remaining_ms="
            << (deadline == Clock::time_point{} ? 0 : Millis(deadline - now))
            << " history_known=" << (it != history_.end())
            << " history_evictions=" << evicted_;
        if (it != history_.end()) {
            const auto& state = it->second;
            out << " generation=" << state.generation
                << " age_ms=" << Millis(now - state.created)
                << " since_refresh_ms=" << Millis(now - state.refreshed)
                << " since_mutation_ms=" << Millis(now - state.mutated)
                << " last_mutation_op=" << state.last_op
                << " last_event=" << state.last_event;
        }
        if (!detail.empty()) out << " reason=" << detail;
        batch.Event(rc < 0, out.str());
    }

   private:
    static int64_t Millis(Clock::duration value) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(value)
            .count();
    }
    struct State {
        uint64_t generation{0}, last_op{0};
        Clock::time_point created{}, refreshed{}, mutated{};
        std::string last_event;
    };
    uint64_t generation_{0}, evicted_{0};
    std::unordered_map<std::string, State> history_;
};

}  // namespace mooncake
