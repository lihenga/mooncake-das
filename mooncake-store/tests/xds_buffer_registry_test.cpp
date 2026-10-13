#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>

#include "storage/distributed/immutable_bucket_allocator.h"
#include <memory>

#include "storage/distributed/distributed_storage_backend.h"
#include "storage/distributed/posix_fs_adapter.h"

namespace mooncake {
namespace {

class FakeAcceleratorFileIo final : public AcceleratorFileIo {
   public:
    DirectIoCapabilities Capabilities() const override {
        DirectIoCapabilities capabilities;
        capabilities.available = true;
        capabilities.supports_distributed_fs = true;
        capabilities.supports_gpu_staging = true;
        return capabilities;
    }
    DeviceXdsState DeviceState(int32_t) const override {
        return DeviceXdsState::kAvailable;
    }
    tl::expected<XdsFileRegistration, XdsError> RegisterFile(
        int fd, const XdsFileIdentity&) override {
        return XdsFileRegistration{++next_id_, fd};
    }
    tl::expected<XdsBufferRegistration, XdsError> RegisterBuffer(
        void* base, size_t length, int32_t device_id) override {
        ++buffer_register_calls;
        return XdsBufferRegistration{++next_id_, base, length, device_id};
    }
    tl::expected<DirectReadCompletion, XdsError> Read(
        const DirectReadOp& op) override {
        ++read_calls;
        return DirectReadCompletion{op.length};
    }
    tl::expected<DirectReadCompletion, XdsError> ReadStaged(
        const DirectReadOp& op) override {
        ++staged_calls;
        last_staged = op;
        return DirectReadCompletion{op.length};
    }
    tl::expected<void, XdsError> DeregisterFile(
        XdsFileRegistration) override {
        return {};
    }
    tl::expected<void, XdsError> DeregisterBuffer(
        XdsBufferRegistration) override {
        ++buffer_deregister_calls;
        return {};
    }

    uint64_t next_id_ = 0;
    int buffer_register_calls = 0;
    int buffer_deregister_calls = 0;
    int read_calls = 0;
    int staged_calls = 0;
    DirectReadOp last_staged;
};

TEST(XdsBufferRegistryTest, DuplicateRegistrationUsesLogicalReference) {
    FileStorageConfig file_config;
    DistributedStorageConfig dfs_config;
    auto backend = std::make_shared<DistributedStorageBackend>(
        file_config, dfs_config, std::make_unique<PosixFsAdapter>());
    auto fake = std::make_shared<FakeAcceleratorFileIo>();
    backend->SetAcceleratorFileIo(fake);

    void* memory = nullptr;
    ASSERT_EQ(posix_memalign(&memory, 4096, 8192), 0);
    ASSERT_TRUE(backend->RegisterXdsBuffer(memory, 8192, 0));
    ASSERT_TRUE(backend->RegisterXdsBuffer(memory, 8192, 0));
    EXPECT_EQ(fake->buffer_register_calls, 1);

    ASSERT_TRUE(backend->UnregisterXdsBuffer(memory));
    EXPECT_EQ(fake->buffer_deregister_calls, 0);
    ASSERT_TRUE(backend->UnregisterXdsBuffer(memory));
    EXPECT_EQ(fake->buffer_deregister_calls, 1);
    std::free(memory);
}

TEST(XdsBufferRegistryTest, RejectsPartiallyOverlappingRegistration) {
    FileStorageConfig file_config;
    DistributedStorageConfig dfs_config;
    auto backend = std::make_shared<DistributedStorageBackend>(
        file_config, dfs_config, std::make_unique<PosixFsAdapter>());
    auto fake = std::make_shared<FakeAcceleratorFileIo>();
    backend->SetAcceleratorFileIo(fake);

    void* memory = nullptr;
    ASSERT_EQ(posix_memalign(&memory, 4096, 12288), 0);
    ASSERT_TRUE(backend->RegisterXdsBuffer(memory, 8192, 0));
    auto* overlap = static_cast<char*>(memory) + 4096;
    auto result = backend->RegisterXdsBuffer(overlap, 8192, 0);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().error_class, XdsErrorClass::kContractViolation);
    ASSERT_TRUE(backend->UnregisterXdsBuffer(memory));
    std::free(memory);
}

TEST(XdsBufferRegistryTest, RejectsOverlappingBatchBeforeIo) {
    FileStorageConfig file_config;
    DistributedStorageConfig dfs_config;
    auto backend = std::make_shared<DistributedStorageBackend>(
        file_config, dfs_config, std::make_unique<PosixFsAdapter>());
    auto fake = std::make_shared<FakeAcceleratorFileIo>();
    backend->SetAcceleratorFileIo(fake);

    void* memory = nullptr;
    ASSERT_EQ(posix_memalign(&memory, 4096, 8192), 0);
    DfsRangeReadRequest first;
    first.key = "first";
    first.ranges.push_back(DfsReadRange{memory, 4096, 0, 0, 0});
    DfsRangeReadRequest second;
    second.key = "second";
    second.ranges.push_back(DfsReadRange{
        static_cast<char*>(memory) + 2048, 4096, 0, 0, 0});

    auto results = backend->BatchReadRanges({first, second});
    ASSERT_EQ(results.size(), 2);
    ASSERT_FALSE(results[0]);
    ASSERT_FALSE(results[1]);
    EXPECT_EQ(results[0].error().error_class,
              XdsErrorClass::kContractViolation);
    EXPECT_EQ(results[1].error().error_class,
              XdsErrorClass::kContractViolation);
    EXPECT_EQ(fake->read_calls, 0);
    std::free(memory);
}

TEST(XdsBufferRegistryTest, PlansStagingWithinBucketAndRejectsTruncatedCover) {
    char directory[] = "/tmp/mooncake-xds-plan-XXXXXX";
    ASSERT_NE(::mkdtemp(directory), nullptr);
    struct Cleanup {
        std::string path;
        ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{directory};
    DistributedStorageConfig config;
    config.fsdir = directory;
    config.allocator_type = DfsAllocatorType::BUCKET;
    config.batch_read_threads = 1;
    config.xds_mode = XdsMode::kRequired;
    config.xds_allow_request_fallback = false;
    auto backend = std::make_shared<DistributedStorageBackend>(
        FileStorageConfig{}, config, std::make_unique<PosixFsAdapter>());
    ASSERT_TRUE(backend->Init());
    auto fake = std::make_shared<FakeAcceleratorFileIo>();
    backend->SetAcceleratorFileIo(fake);
    const std::string path = std::string(directory) + "/bucket_" +
        ImmutableBucketAllocator::FormatBucketId(0) + ".data";
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::ftruncate(fd, 8192), 0);
    (void)::close(fd);
    const int direct = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (direct < 0) GTEST_SKIP() << "O_DIRECT unavailable";
    (void)::close(direct);
    void* allocation = nullptr;
    ASSERT_EQ(::posix_memalign(&allocation, 4096, 8192), 0);
    std::unique_ptr<void, decltype(&std::free)> memory(allocation, &std::free);
    ASSERT_TRUE(backend->RegisterXdsBuffer(allocation, 8192, 0));
    DfsRangeReadRequest request;
    request.key = "key";
    request.descriptor.shard_idx = 0;
    request.descriptor.file_path = path;
    request.descriptor.offset = 4096;
    request.descriptor.object_size = 4095;
    request.descriptor.aligned_size = 4096;
    request.ranges.push_back(DfsReadRange{
        static_cast<char*>(allocation) + 1, 4094, 1, 0, 0});
    DistributedStorageBackend::XdsReadStats stats;
    auto results = backend->BatchReadRanges({request}, &stats);
    ASSERT_TRUE(results[0]);
    EXPECT_EQ(fake->staged_calls, 1);
    EXPECT_EQ(fake->read_calls, 0);
    EXPECT_EQ(stats.staging_bytes, 4094u);
    EXPECT_EQ(fake->last_staged.file_offset, 4097u);
    EXPECT_EQ(fake->last_staged.device_offset, 1u);
    EXPECT_EQ(fake->last_staged.length, 4094u);
    EXPECT_EQ(fake->last_staged.extent_begin, 4096u);
    EXPECT_EQ(fake->last_staged.extent_end, 8192u);

    // Already aligned ranges retain the direct path, even below min-read in
    // strict required mode.
    auto aligned = request;
    aligned.descriptor.object_size = 4096;
    aligned.ranges = {DfsReadRange{allocation, 4096, 0, 0, 0}};
    results = backend->BatchReadRanges({aligned}, &stats);
    ASSERT_TRUE(results[0]);
    EXPECT_EQ(fake->read_calls, 1);
    EXPECT_EQ(stats.direct_bytes, 4096u);
    EXPECT_EQ(stats.staging_bytes, 0u);

    // A later invalid logical range aborts the batch before the valid first
    // request can write anything.
    auto invalid = request;
    invalid.key = "invalid";
    invalid.ranges[0].dst = static_cast<char*>(allocation) + 4096;
    invalid.ranges[0].object_offset = 4095;
    results = backend->BatchReadRanges({request, invalid});
    EXPECT_FALSE(results[0]);
    EXPECT_FALSE(results[1]);
    EXPECT_EQ(fake->staged_calls, 1);

    // The logical last byte exists, but its covering 4096-byte read does not.
    ASSERT_EQ(::truncate(path.c_str(), 8191), 0);
    results = backend->BatchReadRanges({request});
    ASSERT_FALSE(results[0]);
    EXPECT_EQ(results[0].error().error_class,
              XdsErrorClass::kContractViolation);
    EXPECT_EQ(fake->staged_calls, 1);
    ASSERT_TRUE(backend->UnregisterXdsBuffer(allocation));
}

}  // namespace
}  // namespace mooncake
