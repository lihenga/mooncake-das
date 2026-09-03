#include "dfs_prefetcher.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <utility>

#include "client_metric.h"
#include "utils.h"

namespace mooncake {

bool dfs_prefetch_enabled() {
    static const bool enabled = [] {
        const char *val = std::getenv("MC_STORE_ENABLE_DFS_PREFETCH");
        if (!val) {
            return false;
        }
        std::string s(val);
        for (auto &c : s) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        const bool result =
            s == "1" || s == "true" || s == "yes" || s == "on" || s == "enable";
        if (result) {
            LOG(INFO) << "DFS prefetch enabled"
                      << " (MC_STORE_ENABLE_DFS_PREFETCH=" << val << ")";
        } else {
            LOG(INFO) << "DFS prefetch disabled"
                      << " (MC_STORE_ENABLE_DFS_PREFETCH=" << val << ")";
        }
        return result;
    }();
    return enabled;
}

DfsPrefetchConfig DfsPrefetchConfig::FromEnv() {
    DfsPrefetchConfig config;
    config.max_bytes =
        GetEnvOr<uint64_t>("MC_STORE_DFS_PREFETCH_MAX_BYTES", config.max_bytes);
    config.ttl_ms =
        GetEnvOr<uint64_t>("MC_STORE_DFS_PREFETCH_TTL_MS", config.ttl_ms);
    config.io_threads = GetEnvOr<uint32_t>("MC_STORE_DFS_PREFETCH_IO_THREADS",
                                           config.io_threads);
    config.wait_timeout_ms = GetEnvOr<uint64_t>(
        "MC_STORE_DFS_PREFETCH_WAIT_TIMEOUT_MS", config.wait_timeout_ms);
    config.retry_backoff_ms = GetEnvOr<uint64_t>(
        "MC_STORE_DFS_PREFETCH_RETRY_BACKOFF_MS", config.retry_backoff_ms);
    config.max_batch_keys = GetEnvOr<size_t>(
        "MC_STORE_DFS_PREFETCH_MAX_BATCH_KEYS", config.max_batch_keys);
    if (config.io_threads == 0) {
        config.io_threads = 1;
    }
    if (config.max_batch_keys == 0) {
        config.max_batch_keys = 1;
    }
    return config;
}

namespace {

constexpr std::chrono::milliseconds kQueueWaitSlice{50};
constexpr std::chrono::milliseconds kFailedTtl{1000};

inline void SetPromiseValue(const std::shared_ptr<std::promise<void>> &done) {
    if (done) {
        try {
            done->set_value();
        } catch (const std::future_error &e) {
            LOG(ERROR) << "DFS prefetch set_value failed: " << e.what();
        }
    }
}

}  // namespace

DfsPrefetcher::DfsPrefetcher(DfsPrefetchConfig config, BatchQueryFn query_fn,
                             BatchGetFn get_fn, LocalEndpointsFn endpoints_fn,
                             AllocateFn alloc_fn, DfsPrefetchMetric *metric)
    : config_(config),
      query_fn_(std::move(query_fn)),
      get_fn_(std::move(get_fn)),
      endpoints_fn_(std::move(endpoints_fn)),
      alloc_fn_(std::move(alloc_fn)),
      metric_(metric) {
    io_pool_ = std::make_unique<ThreadPool>(config_.io_threads);
    coordinator_thread_ = std::thread([this] { CoordinatorLoop(); });
    LOG(INFO) << "DfsPrefetcher started: max_bytes=" << config_.max_bytes
              << ", ttl_ms=" << config_.ttl_ms
              << ", io_threads=" << config_.io_threads
              << ", wait_timeout_ms=" << config_.wait_timeout_ms;
}

DfsPrefetcher::~DfsPrefetcher() { Shutdown(); }

void DfsPrefetcher::Shutdown() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) {
            return;
        }
        stop_ = true;
    }
    queue_cv_.notify_all();
    if (coordinator_thread_.joinable()) {
        coordinator_thread_.join();
    }
    // After the coordinator exits no new IO tasks can be submitted; draining
    // the pool waits for in-flight BatchGet calls to finish.
    io_pool_.reset();
    std::lock_guard<std::mutex> lock(entries_mutex_);
    entries_.clear();
    inflight_bytes_.store(0, std::memory_order_relaxed);
    if (metric_) {
        metric_->inflight_bytes.update(int64_t{0});
    }
    LOG(INFO) << "DfsPrefetcher stopped";
}

void DfsPrefetcher::NotifyExistTrue(std::vector<std::string> keys) {
    if (keys.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) {
            return;
        }
        for (auto &key : keys) {
            pending_keys_.push_back(std::move(key));
        }
    }
    queue_cv_.notify_all();
}

std::shared_ptr<BufferHandle> DfsPrefetcher::TryConsume(
    const std::string &key, bool &out_found) {
    out_found = false;
    std::shared_ptr<PrefetchEntry> entry;
    std::shared_ptr<std::promise<void>> done_ref;
    {
        std::lock_guard<std::mutex> lock(entries_mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            if (metric_) {
                metric_->consume_total.inc({"miss"});
            }
            return nullptr;
        }
        entry = it->second;
        if (!entry) {
            LOG(ERROR) << "DFS prefetch consume found null entry: key="
                       << key;
            entries_.erase(it);
            return nullptr;
        }
        out_found = true;
        const auto state = entry->state.load(std::memory_order_acquire);
        if (state == PrefetchState::READY) {
            auto handle = std::shared_ptr<BufferHandle>(
                std::move(entry->buffer));
            EraseEntryLocked(it);
            if (metric_) {
                metric_->consume_total.inc({"hit"});
            }
            return handle;
        }
        if (state == PrefetchState::FAILED) {
            // Keep the FAILED entry until its backoff window expires so
            // repeated Notify calls do not immediately retry a broken read.
            if (metric_) {
                metric_->consume_total.inc({"failed"});
            }
            return nullptr;
        }
        // READING: wait outside the lock.
        done_ref = entry->done;
    }

    if (!done_ref || !entry->done_future.valid()) {
        if (metric_) {
            metric_->consume_total.inc({"miss"});
        }
        return nullptr;
    }
    const auto wait_status =
        entry->done_future.wait_for(std::chrono::milliseconds(
            config_.wait_timeout_ms));
    if (wait_status != std::future_status::ready) {
        if (metric_) {
            metric_->consume_total.inc({"wait_timeout"});
        }
        LOG(WARNING) << "DFS prefetch wait timeout for key: " << key
                     << ", falling back to direct read";
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(entries_mutex_);
    auto it = entries_.find(key);
    if (it == entries_.end() || it->second.get() != entry.get()) {
        // GC removed it (or the key was replaced) while we waited.
        if (metric_) {
            metric_->consume_total.inc({"miss"});
        }
        return nullptr;
    }
    if (entry->state.load(std::memory_order_acquire) !=
        PrefetchState::READY) {
        if (metric_) {
            metric_->consume_total.inc({"failed"});
        }
        return nullptr;
    }
    auto handle = std::shared_ptr<BufferHandle>(std::move(entry->buffer));
    EraseEntryLocked(it);
    if (metric_) {
        metric_->consume_total.inc({"hit"});
    }
    return handle;
}

void DfsPrefetcher::CoordinatorLoop() {
    while (true) {
        std::deque<std::string> batch;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, kQueueWaitSlice, [this] {
                return stop_ || !pending_keys_.empty();
            });
            if (stop_ && pending_keys_.empty()) {
                return;
            }
            batch.swap(pending_keys_);
        }
        if (!batch.empty()) {
            ProcessKeys({std::make_move_iterator(batch.begin()),
                         std::make_move_iterator(batch.end())});
        }
        GcExpired();
    }
}

void DfsPrefetcher::ProcessKeys(const std::vector<std::string> &keys) {
    // 1. In-map dedup: skip keys already being tracked, and failed keys that
    // are still inside their retry backoff window.
    std::vector<std::string> query_keys;
    query_keys.reserve(keys.size());
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(entries_mutex_);
        for (const auto &key : keys) {
            auto it = entries_.find(key);
            if (it != entries_.end()) {
                const auto state =
                    it->second->state.load(std::memory_order_acquire);
                if (state == PrefetchState::FAILED) {
                    const auto backoff =
                        std::chrono::milliseconds(config_.retry_backoff_ms);
                    if (now - it->second->failed_at >= backoff) {
                        EraseEntryLocked(it);
                    } else {
                        if (metric_) {
                            metric_->skipped_total.inc({"backoff"});
                        }
                    }
                } else {
                    if (metric_) {
                        metric_->skipped_total.inc({"duplicate"});
                    }
                }
            }
            if (entries_.find(key) == entries_.end()) {
                query_keys.push_back(key);
            }
        }
    }
    if (query_keys.empty()) {
        return;
    }

    const auto local_endpoints = endpoints_fn_();

    // 2. BatchQuery (chunked), keep only keys whose best replica is DFS.
    std::vector<std::pair<std::string, std::shared_ptr<PrefetchEntry>>>
        to_read;
    for (size_t base = 0; base < query_keys.size();
         base += config_.max_batch_keys) {
        const size_t end =
            std::min(base + config_.max_batch_keys, query_keys.size());
        std::vector<std::string> chunk(query_keys.begin() + base,
                                       query_keys.begin() + end);
        std::vector<tl::expected<QueryResult, ErrorCode>> qrs;
        try {
            qrs = query_fn_(chunk);
        } catch (const std::exception &e) {
            LOG(WARNING) << "DFS prefetch BatchQuery failed: " << e.what();
            if (metric_) {
                metric_->query_rpc_total.inc({"fail"});
            }
            continue;
        }
        if (metric_) {
            metric_->query_rpc_total.inc({"ok"});
        }
        for (size_t i = 0; i < chunk.size(); ++i) {
            if (i >= qrs.size() || !qrs[i]) {
                if (metric_) {
                    metric_->skipped_total.inc({"query_error"});
                }
                continue;
            }
            const Replica::Descriptor *best =
                SelectBestReplica(qrs[i]->replicas, local_endpoints);
            if (best == nullptr || !best->is_dfs_replica()) {
                if (metric_) {
                    metric_->skipped_total.inc({"non_dfs"});
                }
                continue;
            }
            if (qrs[i]->IsLeaseExpired()) {
                if (metric_) {
                    metric_->skipped_total.inc({"lease_expired"});
                }
                continue;
            }

            auto entry = std::make_shared<PrefetchEntry>();
            entry->replica = *best;
            entry->lease_timeout = qrs[i]->lease_timeout;
            entry->total_size = calculate_total_size(*best);
            entry->done = std::make_shared<std::promise<void>>();
            entry->done_future = entry->done->get_future().share();
            entry->created_at = std::chrono::steady_clock::now();

            if (entry->total_size == 0 ||
                entry->total_size > config_.max_bytes) {
                if (metric_) {
                    metric_->skipped_total.inc({"capacity"});
                }
                continue;
            }
            const int64_t inflight =
                inflight_bytes_.load(std::memory_order_relaxed);
            if (inflight + static_cast<int64_t>(entry->total_size) >
                static_cast<int64_t>(config_.max_bytes)) {
                if (metric_) {
                    metric_->skipped_total.inc({"capacity"});
                }
                continue;
            }

            auto alloc = alloc_fn_(entry->total_size);
            if (!alloc) {
                // Transient arena pressure: skip without marking FAILED so a
                // later exist probe may retry.
                if (metric_) {
                    metric_->skipped_total.inc({"capacity"});
                }
                continue;
            }
            entry->buffer = std::make_unique<BufferHandle>(std::move(*alloc));

            {
                std::lock_guard<std::mutex> lock(entries_mutex_);
                if (entries_.find(chunk[i]) != entries_.end()) {
                    // Lost a race with another insertion path; drop ours.
                    if (metric_) {
                        metric_->skipped_total.inc({"duplicate"});
                    }
                    continue;
                }
                entries_.emplace(chunk[i], entry);
            }
            TrackInflight(static_cast<int64_t>(entry->total_size));
            if (metric_) {
                metric_->triggered_total.inc();
            }
            to_read.emplace_back(chunk[i], std::move(entry));
        }
    }

    // 3. Submit IO batches (chunked) to the driver pool.
    for (size_t base = 0; base < to_read.size();
         base += config_.max_batch_keys) {
        const size_t end =
            std::min(base + config_.max_batch_keys, to_read.size());
        std::vector<std::pair<std::string, std::shared_ptr<PrefetchEntry>>>
            io_batch(std::make_move_iterator(to_read.begin() + base),
                     std::make_move_iterator(to_read.begin() + end));
        if (io_batch.empty()) {
            continue;
        }
        auto task_batch = std::make_shared<
            std::vector<std::pair<std::string, std::shared_ptr<PrefetchEntry>>>>(
            std::move(io_batch));
        try {
            io_pool_->enqueue([this, task_batch]() mutable {
                RunIoBatch(std::move(*task_batch));
            });
        } catch (const std::exception &e) {
            LOG(WARNING) << "DFS prefetch enqueue failed (shutdown?): "
                         << e.what();
            for (auto &item : *task_batch) {
                auto &entry = item.second;
                // failed_at/last_error must be visible before the FAILED
                // state is published (release), otherwise a concurrent
                // ProcessKeys acquire-read can observe a default time_point.
                entry->failed_at = std::chrono::steady_clock::now();
                entry->last_error = ErrorCode::INTERNAL_ERROR;
                if (entry->buffer) {
                    entry->buffer.reset();
                    TrackInflight(-static_cast<int64_t>(entry->total_size));
                    entry->total_size = 0;
                }
                entry->state.store(PrefetchState::FAILED,
                                   std::memory_order_release);
                SetPromiseValue(entry->done);
            }
        }
    }
}

void DfsPrefetcher::RunIoBatch(
    std::vector<std::pair<std::string, std::shared_ptr<PrefetchEntry>>> batch) {
    std::vector<std::string> keys;
    std::vector<QueryResult> qrs;
    std::unordered_map<std::string, std::vector<Slice>> slices;
    std::unordered_set<size_t> invalid_entries;
    keys.reserve(batch.size());
    qrs.reserve(batch.size());
    for (const auto &[key, entry] : batch) {
        if (!entry) {
            LOG(ERROR) << "DFS prefetch has null entry: key=" << key;
            invalid_entries.insert(keys.size());
            keys.push_back(key);
            qrs.emplace_back(std::vector<Replica::Descriptor>{},
                             std::chrono::steady_clock::time_point{});
            slices[key] = {};
            continue;
        }
        if (!entry->buffer) {
            LOG(ERROR) << "DFS prefetch has null buffer: key=" << key
                       << ", entry=" << entry.get();
        }
        keys.push_back(key);
        qrs.emplace_back(std::vector<Replica::Descriptor>{entry->replica},
                         entry->lease_timeout);
        std::vector<Slice> key_slices;
        if (!entry->buffer ||
            allocateSlices(key_slices, entry->replica, entry->buffer->ptr()) !=
                0) {
            LOG(ERROR) << "DFS prefetch allocateSlices failed for key: "
                       << key;
            slices[key] = {};
        } else {
            slices[key] = std::move(key_slices);
        }
    }

    const auto io_start = std::chrono::steady_clock::now();
    std::vector<tl::expected<void, ErrorCode>> results;
    try {
        results = get_fn_(keys, qrs, slices);
    } catch (const std::exception &e) {
        LOG(WARNING) << "DFS prefetch BatchGet failed: " << e.what();
        results.clear();
    }
    const double io_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      io_start)
            .count();
    if (dfs_read_trace_enabled()) {
        uint64_t batch_bytes = 0;
        for (const auto &item : batch) {
            if (item.second) {
                batch_bytes += item.second->total_size;
            }
        }
        LOG(INFO) << "dfs_prefetch_batch: keys=" << batch.size()
                  << ", batch_bytes=" << batch_bytes
                  << ", batch_read_us="
                  << static_cast<uint64_t>(io_seconds * 1000000.0);
    }
    if (metric_) {
        metric_->read_latency_seconds.observe(io_seconds);
    }

    uint64_t ok_bytes = 0;
    uint64_t ok_keys = 0;
    for (size_t i = 0; i < batch.size(); ++i) {
        auto &entry = batch[i].second;
        if (!entry) {
            LOG(ERROR) << "DFS prefetch result has null entry: key="
                       << keys[i];
            continue;
        }
        const bool ok = invalid_entries.find(i) == invalid_entries.end() &&
                        i < results.size() && results[i].has_value() &&
                        !slices[keys[i]].empty();
        if (ok) {
            entry->state.store(PrefetchState::READY,
                               std::memory_order_release);
            ok_bytes += entry->total_size;
            ++ok_keys;
        } else {
            if (i < results.size() && !results[i].has_value()) {
                entry->last_error = results[i].error();
            } else {
                entry->last_error = ErrorCode::INTERNAL_ERROR;
            }
            // Publish failed_at/last_error before the FAILED state (release);
            // ProcessKeys reads failed_at under acquire for backoff.
            entry->failed_at = std::chrono::steady_clock::now();
            // The buffer cannot serve anyone; release its inflight share.
            if (entry->buffer) {
                entry->buffer.reset();
                TrackInflight(-static_cast<int64_t>(entry->total_size));
                entry->total_size = 0;
            }
            entry->state.store(PrefetchState::FAILED,
                               std::memory_order_release);
        }
        SetPromiseValue(entry->done);
    }
    if (metric_) {
        const char *result = "fail";
        if (ok_keys == batch.size()) {
            result = "ok";
        } else if (ok_keys > 0) {
            result = "partial";
        }
        metric_->read_total.inc({result});
        metric_->read_bytes.inc(ok_bytes);
    }
}

void DfsPrefetcher::GcExpired() {
    const auto now = std::chrono::steady_clock::now();
    const auto ttl = std::chrono::milliseconds(config_.ttl_ms);
    std::lock_guard<std::mutex> lock(entries_mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
        auto &entry = it->second;
        if (!entry) {
            LOG(ERROR) << "DFS prefetch GC found null entry";
            it = entries_.erase(it);
            continue;
        }
        const auto state = entry->state.load(std::memory_order_acquire);
        if (state == PrefetchState::READING) {
            // In-flight reads own their completion path.
            ++it;
            continue;
        }
        std::chrono::milliseconds age;
        if (state == PrefetchState::FAILED) {
            age = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - entry->failed_at);
            if (age < kFailedTtl) {
                ++it;
                continue;
            }
        } else {
            age = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - entry->created_at);
            if (age < ttl) {
                ++it;
                continue;
            }
            if (metric_) {
                metric_->expired_total.inc();
            }
        }
        it = EraseEntryLocked(it);
    }
}

std::unordered_map<std::string,
                   std::shared_ptr<PrefetchEntry>>::iterator
DfsPrefetcher::EraseEntryLocked(
    std::unordered_map<std::string,
                       std::shared_ptr<PrefetchEntry>>::iterator it) {
    auto &entry = it->second;
    // The capacity charge (total_size) is released exactly once per entry,
    // whichever path consumes it first; total_size is zeroed on release so
    // later cleanup paths do not double-count.
    if (entry->buffer) {
        entry->buffer.reset();
    }
    if (entry->total_size > 0) {
        TrackInflight(-static_cast<int64_t>(entry->total_size));
        entry->total_size = 0;
    }
    return entries_.erase(it);
}

void DfsPrefetcher::TrackInflight(int64_t delta) {
    const int64_t current =
        inflight_bytes_.fetch_add(delta, std::memory_order_relaxed) + delta;
    if (metric_) {
        metric_->inflight_bytes.update(current);
    }
}

}  // namespace mooncake
