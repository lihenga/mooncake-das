#include <gtest/gtest.h>

#include <cstdlib>
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

}  // namespace
}  // namespace mooncake
