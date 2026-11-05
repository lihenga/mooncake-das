#pragma once

namespace mooncake {
struct DistributedStorageConfig;
}

namespace mooncake::l4 {

void ReadDfsLowUtilizationEnvironment(DistributedStorageConfig& config);
void ReadDfsBucketEnvironment(DistributedStorageConfig& config);

// Must match the constants in immutable_bucket_allocator.cpp.
constexpr const char* kBucketFilePrefix = "bucket_";
constexpr const char* kBucketDataSuffix = ".data";

}  // namespace mooncake::l4
