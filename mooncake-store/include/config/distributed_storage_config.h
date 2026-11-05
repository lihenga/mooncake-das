#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include "storage/distributed/dfs_io_timeout.h"

namespace mooncake {

struct DistributedStorageConfig {
    std::string fsdir = "/mnt/3fs/mooncake";
    std::string fs_adapter_type = "hf3fs";
    std::string allocator_type = "shard";
    bool enable_health_check = false;
    int shard_count = 64;
    uint64_t shard_capacity = 4ULL * 1024 * 1024 * 1024;
    uint64_t bucket_capacity = 256ULL * 1024 * 1024;
    int64_t max_bucket_count = 256;
    uint64_t alignment = 4096;
    bool single_tenant = true;
    bool eviction_enabled = true;
    double eviction_high_watermark = 0.9;
    double eviction_low_watermark = 0.7;
    std::chrono::seconds deferred_free_duration{30};
    std::chrono::seconds eviction_check_interval{5};
    std::string object_storage_config_path;

    bool UsesObjectStorage() const {
        return fs_adapter_type == "oss" || fs_adapter_type == "s3" ||
               fs_adapter_type == "kvcs-lowlevel";
    }
    bool UsesKvcs() const { return fs_adapter_type == "kvcs-lowlevel"; }

    double low_utilization_threshold = 0.05;
    int low_utilization_evictions_per_tick = 1;
    int low_utilization_eviction_attempts_per_tick = 8;
    bool bucket_recovery_enabled = false;
    int ready_bucket_target = 4;
    int bucket_create_concurrency = 2;
    int batch_read_threads = 128;
    bool batch_read_merge_enabled = false;
    bool direct_read_enabled = true;
    uint64_t batch_get_timeout_ms = 10000;
    uint64_t batch_timeout_fixed_ms = kDefaultDfsBatchTimeoutFixedMs;
    uint64_t batch_get_max_inflight_batches = 8;
    uint64_t batch_get_max_inflight_bytes = 8ULL * 1024 * 1024 * 1024;
    bool allocator_type_valid = true;

    bool ValidateForBucketAllocator() const;
    static bool IsReplicaEnabledFromEnvironment();
    bool Validate() const;
    bool ValidateForAllocator() const;
    static DistributedStorageConfig FromEnvironment();
    std::string FormatStr() const;
};

}  // namespace mooncake
