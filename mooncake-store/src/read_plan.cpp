#include "pyclient.h"
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "read_plan.h"
#include <set>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <iterator>
#include <unordered_map>
namespace mooncake {
namespace {
// Scoped reservations protect plans sharing the legacy client's key-indexed
// session map. The lock is not held during RPC, transfers, or waits.
std::mutex active_mutex;
std::map<PyClient *, std::set<std::string>> active_keys;
class ActiveKeys {
    PyClient *client_;
    std::set<std::string> keys_;

   public:
    ActiveKeys(PyClient *client, const std::vector<std::string> &keys)
        : client_(client), keys_(keys.begin(), keys.end()) {
        std::lock_guard<std::mutex> lock(active_mutex);
        auto &active = active_keys[client_];
        for (const auto &key : keys_)
            if (active.count(key))
                throw std::runtime_error(
                    "overlapping read plan on the same client/key");
        // Roll back even if allocating a set node fails partway through.
        try {
            active.insert(keys_.begin(), keys_.end());
        } catch (...) {
            for (const auto &key : keys_) active.erase(key);
            if (active.empty()) active_keys.erase(client_);
            throw;
        }
    }
    ~ActiveKeys() {
        std::lock_guard<std::mutex> lock(active_mutex);
        auto it = active_keys.find(client_);
        if (it == active_keys.end()) return;
        for (const auto &key : keys_) it->second.erase(key);
        if (it->second.empty()) active_keys.erase(it);
    }
};
using Item = ReadComponent;
using Pool = ReadLayout;
struct Ranges {
    std::vector<std::string> keys;
    std::vector<std::vector<void *>> addresses;
    std::vector<std::vector<size_t>> sizes, offsets;
};

// ReadLayout instances are independent descriptions of destination memory, so
// the same object key may legitimately appear in more than one layout.  The
// range API represents that as one key with multiple destination ranges.  Keep
// the first-seen key order and merge all later ranges into that entry.
Ranges merge_ranges_by_key(Ranges input) {
    if (input.keys.size() != input.addresses.size() ||
        input.keys.size() != input.sizes.size() ||
        input.keys.size() != input.offsets.size()) {
        throw std::runtime_error("read plan range vector size mismatch");
    }

    Ranges output;
    output.keys.reserve(input.keys.size());
    output.addresses.reserve(input.addresses.size());
    output.sizes.reserve(input.sizes.size());
    output.offsets.reserve(input.offsets.size());
    std::unordered_map<std::string, size_t> key_indices;
    key_indices.reserve(input.keys.size());

    for (size_t i = 0; i < input.keys.size(); ++i) {
        auto [it, inserted] =
            key_indices.emplace(input.keys[i], output.keys.size());
        if (inserted) {
            output.keys.push_back(std::move(input.keys[i]));
            output.addresses.push_back(std::move(input.addresses[i]));
            output.sizes.push_back(std::move(input.sizes[i]));
            output.offsets.push_back(std::move(input.offsets[i]));
            continue;
        }

        const size_t index = it->second;
        output.addresses[index].insert(
            output.addresses[index].end(),
            std::make_move_iterator(input.addresses[i].begin()),
            std::make_move_iterator(input.addresses[i].end()));
        output.sizes[index].insert(
            output.sizes[index].end(),
            std::make_move_iterator(input.sizes[i].begin()),
            std::make_move_iterator(input.sizes[i].end()));
        output.offsets[index].insert(
            output.offsets[index].end(),
            std::make_move_iterator(input.offsets[i].begin()),
            std::make_move_iterator(input.offsets[i].end()));
    }
    return output;
}

size_t checked_add(size_t a, size_t b) {
    if (b > SIZE_MAX - a) throw std::overflow_error("range overflow");
    return a + b;
}
size_t address(const Item &i, size_t row) {
    if (i[1] && row > SIZE_MAX / i[1])
        throw std::overflow_error("row address overflow");
    auto p = checked_add(i[0], row * i[1]);
    checked_add(p, i[2]);
    checked_add(i[3], i[2]);
    return p;
}
}  // namespace
struct ReadPlan::Impl {
    std::shared_ptr<mooncake::PyClient> client;
    std::vector<Pool> layouts;
    int groups;
    bool reuse;
    bool page_wise;
    uint64_t built_count = 0, refreshed_count = 0;
    std::mutex mutex;
    std::condition_variable cv;
    int ready = -1;
    bool running = false, finished = false;
    std::exception_ptr failure;
    std::vector<uint64_t> stats{0, 0, 0};  // calls, keys, bytes
    void mark(int group) {
        {
            std::lock_guard lock(mutex);
            ready = group;
        }
        cv.notify_all();
    }
    void finish(std::exception_ptr error) {
        {
            std::lock_guard lock(mutex);
            failure = error;
            finished = true;
            if (!error) ready = groups - 1;
        }
        cv.notify_all();
    }

   public:
    Impl(std::shared_ptr<PyClient> c, std::vector<Pool> p, int n,
         bool reuse_ranges, bool page_wise_mode)
        : client(std::move(c)),
          layouts(std::move(p)),
          groups(n),
          reuse(reuse_ranges),
          page_wise(page_wise_mode) {
        if (!client) throw std::invalid_argument("read plan requires a client");
        if (n <= 0) throw std::invalid_argument("num_groups must be positive");
        if (reuse && page_wise)
            throw std::invalid_argument(
                "read plan page-wise mode does not support range reuse");
        for (const auto &p : layouts) {
            const auto &[keys, rows, packed, layout] = p;
            if (layout.size() != size_t(n))
                throw std::invalid_argument("group layout count mismatch");
            for (const auto &items : layout) {
                if (items.empty()) continue;
                if (rows.size() > SIZE_MAX / items.size())
                    throw std::overflow_error("key count overflow");
                if (keys.size() != rows.size() * (packed ? 1 : items.size()))
                    throw std::invalid_argument(
                        "key/row/component count mismatch");
                if (!rows.empty()) {
                    auto [lo, hi] =
                        std::minmax_element(rows.begin(), rows.end());
                    for (const auto &i : items) {
                        address(i, *lo);
                        address(i, *hi);
                    }
                }
            }
        }
    }
    Ranges build(int group) const {
        if (group < 0 || group >= groups)
            throw std::out_of_range("group out of range");
        Ranges out;
        size_t total = 0;
        for (const auto &[keys, rows, packed, layout] : layouts)
            if (!layout[group].empty()) total = checked_add(total, keys.size());
        out.keys.reserve(total);
        out.addresses.reserve(total);
        out.sizes.reserve(total);
        out.offsets.reserve(total);
        for (const auto &[keys, rows, packed, layout] : layouts) {
            const auto &items = layout[group];
            if (items.empty()) continue;
            out.keys.insert(out.keys.end(), keys.begin(), keys.end());
            for (auto row : rows) {
                if (packed) {
                    std::vector<void *> a;
                    std::vector<size_t> s, o;
                    a.reserve(items.size());
                    s.reserve(items.size());
                    o.reserve(items.size());
                    for (const auto &i : items) {
                        a.push_back(reinterpret_cast<void *>(address(i, row)));
                        s.push_back(i[2]);
                        o.push_back(i[3]);
                    }
                    out.addresses.push_back(std::move(a));
                    out.sizes.push_back(std::move(s));
                    out.offsets.push_back(std::move(o));
                } else
                    for (const auto &i : items) {
                        out.addresses.push_back(
                            {reinterpret_cast<void *>(address(i, row))});
                        out.sizes.push_back({i[2]});
                        out.offsets.push_back({i[3]});
                    }
            }
        }
        return merge_ranges_by_key(std::move(out));
    }
    // Aggregate every group's components into one range per key. This is the
    // page-wise path: a single batch_get reads all layers of every page,
    // giving up layer-wise overlap for minimum RPC count.
    Ranges build_all() const {
        Ranges out;
        size_t total = 0;
        for (const auto &[keys, rows, packed, layout] : layouts)
            total = checked_add(total, keys.size());
        out.keys.reserve(total);
        out.addresses.reserve(total);
        out.sizes.reserve(total);
        out.offsets.reserve(total);
        for (const auto &[keys, rows, packed, layout] : layouts) {
            out.keys.insert(out.keys.end(), keys.begin(), keys.end());
            for (auto row : rows) {
                if (packed) {
                    // One key per row; collect all groups' components into its
                    // range list, matching _load_page_wise's per-page extend.
                    std::vector<void *> a;
                    std::vector<size_t> s, o;
                    for (const auto &items : layout)
                        for (const auto &i : items) {
                            a.push_back(reinterpret_cast<void *>(address(i, row)));
                            s.push_back(i[2]);
                            o.push_back(i[3]);
                        }
                    out.addresses.push_back(std::move(a));
                    out.sizes.push_back(std::move(s));
                    out.offsets.push_back(std::move(o));
                } else {
                    // One key per (row, component); each key carries the ranges
                    // of that component across all groups. All non-empty groups
                    // share the same component count (enforced at construction).
                    size_t component_count = 0;
                    for (const auto &group_items : layout)
                        if (!group_items.empty()) {
                            component_count = group_items.size();
                            break;
                        }
                    for (size_t c = 0; c < component_count; ++c) {
                        std::vector<void *> a;
                        std::vector<size_t> s, o;
                        for (const auto &group_items : layout) {
                            if (group_items.size() <= c) continue;
                            const auto &i = group_items[c];
                            a.push_back(reinterpret_cast<void *>(address(i, row)));
                            s.push_back(i[2]);
                            o.push_back(i[3]);
                        }
                        out.addresses.push_back(std::move(a));
                        out.sizes.push_back(std::move(s));
                        out.offsets.push_back(std::move(o));
                    }
                }
            }
        }
        return merge_ranges_by_key(std::move(out));
    }
    std::vector<size_t> signature(int group) const {
        std::vector<size_t> out;
        out.reserve(layouts.size());
        for (const auto &p : layouts)
            out.push_back(std::get<3>(p)[group].size());
        return out;
    }
    void refresh(Ranges &out, int group) const {
        std::unordered_map<std::string, size_t> key_indices;
        key_indices.reserve(out.keys.size());
        for (size_t i = 0; i < out.keys.size(); ++i) {
            if (!key_indices.emplace(out.keys[i], i).second)
                throw std::runtime_error("reused ranges contain duplicate key");
        }
        std::vector<size_t> range_indices(out.keys.size(), 0);

        auto find_output = [&](const std::string &key, size_t count) {
            auto it = key_indices.find(key);
            if (it == key_indices.end())
                throw std::runtime_error("reused range key mismatch");
            const size_t index = it->second;
            const size_t begin = range_indices[index];
            if (begin > out.addresses[index].size() ||
                count > out.addresses[index].size() - begin ||
                out.addresses[index].size() != out.sizes[index].size() ||
                out.addresses[index].size() != out.offsets[index].size()) {
                throw std::runtime_error("reused range shape mismatch");
            }
            range_indices[index] += count;
            return std::pair<size_t, size_t>{index, begin};
        };

        for (const auto &[keys, rows, packed, layout] : layouts) {
            const auto &items = layout[group];
            if (items.empty()) continue;
            size_t key_index = 0;
            for (auto row : rows) {
                if (packed) {
                    if (key_index >= keys.size())
                        throw std::runtime_error("reused range key mismatch");
                    const auto [index, begin] =
                        find_output(keys[key_index++], items.size());
                    for (size_t j = 0; j < items.size(); ++j) {
                        const auto &i = items[j];
                        out.addresses[index][begin + j] =
                            reinterpret_cast<void *>(address(i, row));
                        out.sizes[index][begin + j] = i[2];
                        out.offsets[index][begin + j] = i[3];
                    }
                } else {
                    for (const auto &i : items) {
                        if (key_index >= keys.size())
                            throw std::runtime_error(
                                "reused range key mismatch");
                        const auto [index, begin] =
                            find_output(keys[key_index++], 1);
                        out.addresses[index][begin] =
                            reinterpret_cast<void *>(address(i, row));
                        out.sizes[index][begin] = i[2];
                        out.offsets[index][begin] = i[3];
                    }
                }
            }
            if (key_index != keys.size())
                throw std::runtime_error("reused range key mismatch");
        }
        for (size_t i = 0; i < out.keys.size(); ++i)
            if (range_indices[i] != out.addresses[i].size())
                throw std::runtime_error("reused range shape mismatch");
    }
    void check(const Ranges &r, const std::vector<int> &results, int group) {
        if (results.size() != r.keys.size())
            throw std::runtime_error(
                "Mooncake read plan result count mismatch");
        uint64_t bytes = 0;
        for (size_t k = 0; k < results.size(); ++k) {
            size_t expected = 0;
            for (auto s : r.sizes[k]) expected = checked_add(expected, s);
            if (results[k] < 0 || size_t(results[k]) != expected)
                throw std::runtime_error(
                    "Mooncake read plan range get failed at group=" +
                    std::to_string(group) + " key_index=" + std::to_string(k) +
                    " rc=" + std::to_string(results[k]) +
                    " expected=" + std::to_string(expected));
            bytes += expected;
        }
        ++stats[0];
        stats[1] += results.size();
        stats[2] += bytes;
    }

    // A fixed two-group window is opt-in via MOONCAKE_READ_PLAN_PIPELINE=1.
    // Conservative span check: false positives retain the sequential path.
    // Distinct groups must never write overlapping destination bytes in flight.
    bool disjoint_groups() const {
        struct Span {
            size_t begin, end;
            int group;
        };
        std::vector<Span> spans;
        for (const auto &[keys, rows, packed, layout] : layouts) {
            if (rows.empty()) continue;
            const auto [lo, hi] = std::minmax_element(rows.begin(), rows.end());
            for (int g = 0; g < groups; ++g)
                for (const auto &item : layout[g]) {
                    if (!item[2]) continue;
                    spans.push_back({address(item, *lo),
                                     checked_add(address(item, *hi), item[2]), g});
                }
        }
        std::sort(spans.begin(), spans.end(), [](const Span &a, const Span &b) {
            return a.begin < b.begin;
        });
        std::vector<size_t> ends(groups, 0);
        for (const auto &s : spans) {
            for (int g = 0; g < groups; ++g)
                if (g != s.group && ends[g] > s.begin) return false;
            ends[s.group] = std::max(ends[s.group], s.end);
        }
        return true;
    }
    void run_pipelined() {
        struct Slot {
            Ranges ranges;
            std::vector<int> result;
            std::exception_ptr error;
            bool done = false;
            bool built = false;
        };
        std::vector<Slot> slots(groups);
        std::mutex work_mutex;
        std::condition_variable work_cv;
        int next = 0, consumed = 0;
        bool stop = false;
        auto worker = [&] {
            while (true) {
                int g;
                {
                    std::unique_lock lock(work_mutex);
                    work_cv.wait(lock, [&] {
                        return stop || next >= groups || next < consumed + 2;
                    });
                    if (stop || next >= groups) return;
                    g = next++;
                }
                auto &slot = slots[g];
                try {
                    slot.ranges = build(g);
                    slot.built = true;
                    auto &r = slot.ranges;
                    if (!r.keys.empty())
                        slot.result = client->batch_get_into_multi_buffer_ranges(
                            r.keys, r.addresses, r.sizes, r.offsets);
                } catch (...) {
                    slot.error = std::current_exception();
                }
                {
                    std::lock_guard lock(work_mutex);
                    slot.done = true;
                }
                work_cv.notify_all();
            }
        };
        std::vector<std::thread> workers;
        auto drain = [&] {
            {
                std::lock_guard lock(work_mutex);
                stop = true;
            }
            work_cv.notify_all();
            for (auto &t : workers)
                if (t.joinable()) t.join();
            // Include ranges prepared speculatively, even on read failure.
            for (const auto &slot : slots)
                if (slot.built) ++built_count;
        };
        // reserve before creating threads so allocation cannot destroy a
        // joinable temporary. Always drain reads before session cleanup.
        workers.reserve(2);
        try {
            workers.emplace_back(worker);
            workers.emplace_back(worker);
            for (int g = 0; g < groups; ++g) {
                auto &slot = slots[g];
                {
                    std::unique_lock lock(work_mutex);
                    work_cv.wait(lock, [&] { return slot.done; });
                }
                if (slot.error) std::rethrow_exception(slot.error);
                if (!slot.ranges.keys.empty()) check(slot.ranges, slot.result, g);
                if (g < groups - 1) mark(g);
                slot.ranges = Ranges{};
                slot.result.clear();
                {
                    std::lock_guard lock(work_mutex);
                    consumed = g + 1;
                }
                work_cv.notify_all();
            }
        } catch (...) {
            drain();
            throw;
        }
        drain();
    }

    void run_impl() {
        {
            std::lock_guard lock(mutex);
            if (running) throw std::runtime_error("plan may only run once");
            running = true;
        }
        std::vector<std::string> session;
        bool started = false;
        std::exception_ptr error;
        std::unique_ptr<ActiveKeys> reservation;
        try {
            std::set<std::string> seen;
            for (const auto &p : layouts) {
                for (const auto &key : std::get<0>(p))
                    if (seen.insert(key).second) session.push_back(key);
            }
            reservation = std::make_unique<ActiveKeys>(client.get(), session);
            {
                started = true;
                auto result = client->batch_get_session_start(session);
                if (result.size() != session.size() ||
                    std::any_of(result.begin(), result.end(),
                                [](int x) { return x != 0; }))
                    throw std::runtime_error(
                        "Mooncake read plan session start failed");
            }
            const char *enabled = std::getenv("MOONCAKE_READ_PLAN_PIPELINE");
            const bool requested = enabled && std::string(enabled) == "1";
            const bool pipeline =
                requested && !reuse && !page_wise && groups > 1 &&
                disjoint_groups();
            if (requested) {
                static std::atomic<bool> logged_yes{false}, logged_no{false};
                auto &logged = pipeline ? logged_yes : logged_no;
                if (!logged.exchange(true))
                    std::fprintf(
                        stderr,
                        "MOONCAKE_READ_PLAN_PIPELINE enabled=%d groups=%d "
                        "keys=%zu\n",
                        int(pipeline), groups, session.size());
            }
            if (page_wise) {
                // One batch_get carries every group's ranges per key; readiness
                // is published only after the single transfer succeeds.
                auto r = build_all();
                ++built_count;
                if (!r.keys.empty()) {
                    auto result = client->batch_get_into_multi_buffer_ranges(
                        r.keys, r.addresses, r.sizes, r.offsets);
                    check(r, result, -1);
                }
            } else if (pipeline) {
                run_pipelined();
            } else {
                std::map<std::vector<size_t>, Ranges> cached_ranges;
                for (int group = 0; group < groups; ++group) {
                    Ranges fresh;
                    Ranges *selected = nullptr;
                    if (reuse) {
                        auto [it, inserted] =
                            cached_ranges.try_emplace(signature(group));
                        if (inserted) {
                            it->second = build(group);
                            ++built_count;
                        } else {
                            refresh(it->second, group);
                            ++refreshed_count;
                        }
                        selected = &it->second;
                    } else {
                        fresh = build(group);
                        selected = &fresh;
                        ++built_count;
                    }
                    auto &r = *selected;
                    if (!r.keys.empty()) {
                        auto result = client->batch_get_into_multi_buffer_ranges(
                            r.keys, r.addresses, r.sizes, r.offsets);
                        check(r, result, group);
                    }
                    if (group < groups - 1)
                        mark(group);  // Final readiness includes session cleanup.
                }
            }
        } catch (...) {
            error = std::current_exception();
        }
        if (started) {
            try {
                if (client->batch_get_session_end(session) != 0)
                    throw std::runtime_error(
                        "read plan session cleanup failed");
            } catch (...) {
                if (!error) error = std::current_exception();
            }
        }
        reservation.reset();
        finish(error);
        if (error) std::rethrow_exception(error);
    }
    void run() { run_impl(); }
    void wait(int group) {
        if (group < 0 || group >= groups)
            throw std::out_of_range("group out of range");
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return ready >= group || finished; });
        if (failure) std::rethrow_exception(failure);
        if (ready < group)
            throw std::runtime_error("plan ended before requested group");
    }
    std::vector<uint64_t> get_stats() {
        std::lock_guard lock(mutex);
        if (!finished) throw std::runtime_error("stats require completed plan");
        return stats;
    }
    std::vector<uint64_t> reuse_stats() {
        std::lock_guard lock(mutex);
        if (!finished)
            throw std::runtime_error("reuse stats require completed plan");
        return {built_count, refreshed_count};
    }
};
ReadPlan::ReadPlan(std::shared_ptr<PyClient> client,
                   std::vector<ReadLayout> layouts, int groups, bool reuse,
                   bool page_wise)
    : impl_(std::make_unique<Impl>(std::move(client), std::move(layouts),
                                   groups, reuse, page_wise)) {}
ReadPlan::~ReadPlan() = default;
void ReadPlan::run() { impl_->run(); }
void ReadPlan::wait(int group) { impl_->wait(group); }
std::vector<uint64_t> ReadPlan::stats() { return impl_->get_stats(); }
std::vector<uint64_t> ReadPlan::reuse_stats() { return impl_->reuse_stats(); }
}  // namespace mooncake
