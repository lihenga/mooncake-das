// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Modified by Hygon Information Technology Co., Ltd., 2026.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "client_service.h"
#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/posix_fs_adapter.h"

namespace {

constexpr uint64_t kMiB = 1024ULL * 1024;
constexpr uint64_t kGiB = 1024ULL * kMiB;
constexpr uint64_t kMaxMergedIo = 4ULL * kMiB;
constexpr size_t kAlignment = 4096;

DEFINE_string(key_dump, "", "Path written by MOONCAKE_DFS_KEY_DUMP");
DEFINE_string(master_server, "127.0.0.1:50051", "Master server address");
DEFINE_string(local_host, "127.0.0.1:19001", "Local client host:port");
DEFINE_string(metadata_connstring, "P2PHANDSHAKE", "Metadata connection string");
DEFINE_string(protocol, "tcp", "Transfer protocol");
DEFINE_uint64(batch_size, 0,
              "Keys per replayed BatchGet; 0 replays each dumped batch intact");
DEFINE_uint64(requests, 0, "Measured requests; 0 runs until --duration");
DEFINE_uint64(duration, 60, "Measured duration in seconds when --requests=0");
DEFINE_uint64(warmup_requests, 3, "Unreported requests before measurement");
DEFINE_uint64(max_source_batches, 1,
              "Maximum batch_get lines to load; 0 loads every line");

struct FreeDeleter {
    void operator()(void* pointer) const { std::free(pointer); }
};
using AlignedBuffer = std::unique_ptr<void, FreeDeleter>;

struct ReplayRequest {
    std::vector<std::string> keys;
    std::vector<mooncake::QueryResult> queries;
    std::unordered_map<std::string, std::vector<mooncake::Slice>> slices;
    AlignedBuffer output;
    uint64_t value_bytes = 0;
    uint64_t disk_bytes = 0;
    uint64_t task_count = 0;
};

struct Stats {
    uint64_t requests = 0;
    uint64_t keys = 0;
    uint64_t value_bytes = 0;
    uint64_t disk_bytes = 0;
    std::vector<double> latency_ms;
};

std::vector<std::vector<std::string>> LoadBatches() {
    std::ifstream input(FLAGS_key_dump);
    if (!input) throw std::runtime_error("cannot open key dump: " + FLAGS_key_dump);

    std::vector<std::vector<std::string>> batches;
    std::string tag;
    while (input >> tag) {
        std::string pid;
        std::string batch_id;
        if (!(input >> pid >> batch_id)) {
            throw std::runtime_error("malformed key dump line");
        }
        std::string line;
        std::getline(input, line);
        if (tag != "batch_get") continue;

        std::vector<std::string> keys;
        size_t begin = 0;
        while (begin < line.size()) {
            begin = line.find_first_not_of(' ', begin);
            if (begin == std::string::npos) break;
            const size_t end = line.find(' ', begin);
            keys.push_back(line.substr(begin, end - begin));
            begin = end == std::string::npos ? line.size() : end + 1;
        }
        if (!keys.empty()) batches.push_back(std::move(keys));
        if (FLAGS_max_source_batches != 0 &&
            batches.size() >= FLAGS_max_source_batches) {
            break;
        }
    }
    return batches;
}

std::vector<std::vector<std::string>> BuildRequestKeys(
    const std::vector<std::vector<std::string>>& source_batches) {
    if (FLAGS_batch_size == 0) return source_batches;

    std::vector<std::string> all_keys;
    for (const auto& batch : source_batches) {
        all_keys.insert(all_keys.end(), batch.begin(), batch.end());
    }

    std::vector<std::vector<std::string>> requests;
    for (size_t begin = 0; begin < all_keys.size(); begin += FLAGS_batch_size) {
        const size_t end =
            std::min<size_t>(begin + FLAGS_batch_size, all_keys.size());
        requests.emplace_back(all_keys.begin() + begin, all_keys.begin() + end);
    }
    return requests;
}

std::optional<mooncake::QueryResult> QueryDfsOnly(
    mooncake::Client& client, const std::string& key) {
    auto query = client.Query(key);
    if (!query) {
        LOG(ERROR) << "Query failed for key " << key << ": "
                   << mooncake::toString(query.error());
        return std::nullopt;
    }

    std::vector<mooncake::Replica::Descriptor> replicas;
    for (const auto& replica : query->replicas) {
        if (replica.is_dfs_replica() &&
            replica.status == mooncake::ReplicaStatus::COMPLETE) {
            replicas.push_back(replica);
        }
    }
    if (replicas.empty()) {
        LOG(ERROR) << "No DFS replica for key " << key;
        return std::nullopt;
    }
    return mooncake::QueryResult(std::move(replicas), query->lease_timeout,
                                 query->object_checksum);
}

ReplayRequest PrepareRequest(mooncake::Client& client,
                             const std::vector<std::string>& source_keys) {
    ReplayRequest request;
    std::unordered_set<std::string> seen;
    for (const auto& key : source_keys) {
        if (!seen.insert(key).second) continue;
        auto query = QueryDfsOnly(client, key);
        if (!query) throw std::runtime_error("failed to prepare replay request");
        const auto& descriptor = query->replicas.front().get_dfs_descriptor();
        request.value_bytes += descriptor.object_size;
        request.disk_bytes += descriptor.aligned_size;
        request.keys.push_back(key);
        request.queries.push_back(std::move(*query));
    }

    if (request.disk_bytes == 0) {
        throw std::runtime_error("replay request has no DFS data");
    }
    void* raw = nullptr;
    if (posix_memalign(&raw, kAlignment, request.disk_bytes) != 0) {
        throw std::runtime_error("failed to allocate aligned output buffer");
    }
    request.output.reset(raw);

    uint64_t offset = 0;
    for (size_t i = 0; i < request.keys.size(); ++i) {
        const auto& descriptor =
            request.queries[i].replicas.front().get_dfs_descriptor();
        auto* output = static_cast<char*>(request.output.get()) + offset;
        request.slices.emplace(
            request.keys[i],
            std::vector<mooncake::Slice>{
                {output, static_cast<size_t>(descriptor.aligned_size)}});
        offset += descriptor.aligned_size;
    }
    return request;
}

uint64_t EstimateTaskCount(const ReplayRequest& request, bool merge_enabled) {
    if (!merge_enabled) return request.keys.size();

    struct Range {
        std::string file_path;
        uint64_t offset;
        uint64_t size;
    };
    std::vector<Range> ranges;
    ranges.reserve(request.queries.size());
    for (const auto& query : request.queries) {
        const auto& descriptor = query.replicas.front().get_dfs_descriptor();
        ranges.push_back(
            {descriptor.file_path, descriptor.offset, descriptor.aligned_size});
    }
    std::sort(ranges.begin(), ranges.end(),
              [](const Range& left, const Range& right) {
                  return left.file_path != right.file_path
                             ? left.file_path < right.file_path
                             : left.offset < right.offset;
              });

    uint64_t tasks = 0;
    uint64_t merged_size = 0;
    uint64_t merged_end = 0;
    std::string merged_path;
    for (const auto& range : ranges) {
        if (merged_path == range.file_path && merged_end == range.offset &&
            merged_size + range.size <= kMaxMergedIo) {
            merged_size += range.size;
            merged_end += range.size;
            continue;
        }
        ++tasks;
        merged_path = range.file_path;
        merged_size = range.size;
        merged_end = range.offset + range.size;
    }
    return tasks;
}

void RunRequest(mooncake::Client& client, ReplayRequest& request) {
    auto results =
        client.BatchGet(request.keys, request.queries, request.slices);
    if (results.size() != request.keys.size()) {
        throw std::runtime_error("BatchGet returned an unexpected result count");
    }
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i]) {
            throw std::runtime_error("BatchGet failed for key " + request.keys[i] +
                                     ": " + mooncake::toString(results[i].error()));
        }
    }
}

double Percentile(std::vector<double> values, double percentile) {
    if (values.empty()) return 0;
    const size_t index = static_cast<size_t>(percentile * (values.size() - 1));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

void PrintSummary(const Stats& stats, double seconds) {
    const double value_gib = static_cast<double>(stats.value_bytes) / kGiB;
    const double disk_gib = static_cast<double>(stats.disk_bytes) / kGiB;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "requests=" << stats.requests << " keys=" << stats.keys
              << " elapsed_s=" << seconds << '\n';
    std::cout << "value_GiB=" << value_gib << " disk_GiB=" << disk_gib
              << " value_GiBps=" << value_gib / seconds
              << " disk_GiBps=" << disk_gib / seconds
              << " key_iops=" << static_cast<double>(stats.keys) / seconds << '\n';
    std::cout << "request_p50_ms=" << Percentile(stats.latency_ms, 0.50)
              << " request_p95_ms=" << Percentile(stats.latency_ms, 0.95)
              << " request_p99_ms=" << Percentile(stats.latency_ms, 0.99) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    if (FLAGS_key_dump.empty() || (FLAGS_requests == 0 && FLAGS_duration == 0)) {
        std::cerr
            << "--key_dump and one of --requests or --duration are required\n";
        return 1;
    }

    try {
        auto source_batches = LoadBatches();
        auto request_keys = BuildRequestKeys(source_batches);
        if (request_keys.empty()) {
            throw std::runtime_error("no batch_get records found");
        }

        auto client_result = mooncake::Client::Create(
            FLAGS_local_host, FLAGS_metadata_connstring, FLAGS_protocol, std::nullopt,
            FLAGS_master_server);
        if (!client_result) throw std::runtime_error("failed to create client");
        auto client = *client_result;

        const auto config = mooncake::DistributedStorageConfig::FromEnvironment();
        mooncake::FileStorageConfig file_config;
        file_config.storage_backend_type = mooncake::StorageBackendType::kDistributed;
        file_config.storage_filepath = config.fsdir;
        auto backend = std::make_shared<mooncake::DistributedStorageBackend>(
            file_config, config, std::make_unique<mooncake::PosixFsAdapter>());
        auto init = backend->Init();
        if (!init) {
            throw std::runtime_error("failed to initialize DFS backend: " +
                                     mooncake::toString(init.error()));
        }
        client->SetDfsStorageBackend(backend);

        std::vector<ReplayRequest> requests;
        requests.reserve(request_keys.size());
        for (const auto& keys : request_keys) {
            requests.push_back(PrepareRequest(*client, keys));
        }
        uint64_t total_tasks = 0;
        for (auto& request : requests) {
            request.task_count =
                EstimateTaskCount(request, config.batch_read_merge_enabled);
            total_tasks += request.task_count;
        }
        const auto& first = requests.front();
        const double first_task_mib = first.task_count == 0
                                          ? 0
                                          : static_cast<double>(first.disk_bytes) /
                                                first.task_count / kMiB;
        std::cout << "loaded_source_batches=" << source_batches.size()
                  << " replay_requests=" << requests.size()
                  << " first_request_keys=" << first.keys.size()
                  << " first_request_tasks=" << first.task_count
                  << " first_task_avg_MiB=" << first_task_mib
                  << " total_estimated_tasks=" << total_tasks << '\n';

        for (uint64_t i = 0; i < FLAGS_warmup_requests; ++i) {
            RunRequest(*client, requests[i % requests.size()]);
        }

        Stats stats;
        const auto started = std::chrono::steady_clock::now();
        const auto deadline = started + std::chrono::seconds(FLAGS_duration);
        for (uint64_t i = 0;
             (FLAGS_requests != 0 && i < FLAGS_requests) ||
             (FLAGS_requests == 0 && std::chrono::steady_clock::now() < deadline);
             ++i) {
            auto& request = requests[i % requests.size()];
            const auto request_started = std::chrono::steady_clock::now();
            RunRequest(*client, request);
            const auto request_finished = std::chrono::steady_clock::now();
            stats.latency_ms.push_back(
                std::chrono::duration<double, std::milli>(request_finished - request_started)
                    .count());
            ++stats.requests;
            stats.keys += request.keys.size();
            stats.value_bytes += request.value_bytes;
            stats.disk_bytes += request.disk_bytes;
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
        PrintSummary(stats, seconds);
    } catch (const std::exception& error) {
        std::cerr << "dfs_batch_get_replay_bench: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
