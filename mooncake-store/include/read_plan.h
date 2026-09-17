#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace mooncake {
class PyClient;
// Destination base, destination row stride, byte count, object source offset.
using ReadComponent = std::array<size_t, 4>;
// Object keys, destination row indices, packed-object flag, components per
// group. Packed: one object per row. Unpacked: one object per row/component.
using ReadLayout = std::tuple<std::vector<std::string>, std::vector<size_t>,
                              bool, std::vector<std::vector<ReadComponent>>>;

// One-shot ordered range reads. Owns a strong client reference, not raw
// destination memory. Keep destination allocations registered/alive and do not
// close the client until run() ends. Do not mix legacy sessions on these keys
// with a plan. Concurrent plans on the same client with overlapping keys fail
// explicitly.
class ReadPlan {
   public:
    ReadPlan(std::shared_ptr<PyClient> client, std::vector<ReadLayout> layouts,
             int num_groups, bool reuse_ranges = false,
             bool page_wise = false);
    ~ReadPlan();
    ReadPlan(const ReadPlan&) = delete;
    ReadPlan& operator=(const ReadPlan&) = delete;
    void run();
    void wait(int group);
    std::vector<uint64_t> stats();
    std::vector<uint64_t> reuse_stats();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace mooncake