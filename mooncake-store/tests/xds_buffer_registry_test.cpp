#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
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
        capabilities.staging_buffer_bytes = scratch_bytes;
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
        operations.push_back(op);
        if (on_read) on_read();
        return DirectReadCompletion{op.length, op.length, 1, 0};
    }
    tl::expected<DirectReadCompletion, XdsError> ReadStaged(
        const DirectReadOp& op) override {
        ++staged_calls;
        operations.push_back(op);
        if (on_read) on_read();
        if (fail_staged) {
            return tl::make_unexpected(XdsError{
                XdsErrorClass::kBackendFatal,
                XdsTargetState::kStoppedSafeToOverwrite, -1, "fake-failure"});
        }
        last_staged = op;
        const uint64_t physical_begin = op.file_offset / 4096 * 4096;
        const uint64_t physical_end =
            (op.file_offset + op.length + 4095) / 4096 * 4096;
        return DirectReadCompletion{op.length, physical_end - physical_begin,
                                    1, 1};
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

    uint64_t scratch_bytes = 8192;
    bool fail_staged = false;
    std::function<void()> on_read;
    std::vector<DirectReadOp> operations;
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
    EXPECT_EQ(stats.staging_physical_bytes, 4096u);
    EXPECT_EQ(stats.staging_hipfile_read_calls, 1u);
    EXPECT_EQ(stats.staging_chunks, 1u);
    EXPECT_EQ(stats.max_staged_op_bytes, 4094u);
    EXPECT_EQ(stats.staging_completed_ops, 1u);
    EXPECT_EQ(stats.direct_completed_ops, 0u);
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
    EXPECT_EQ(stats.direct_physical_bytes, 4096u);
    EXPECT_EQ(stats.direct_hipfile_read_calls, 1u);
    EXPECT_EQ(stats.staging_bytes, 0u);
    EXPECT_EQ(stats.staging_physical_bytes, 0u);
    EXPECT_EQ(stats.staging_hipfile_read_calls, 0u);
    EXPECT_EQ(stats.staging_chunks, 0u);
    EXPECT_EQ(stats.max_staged_op_bytes, 0u);
    EXPECT_EQ(stats.direct_completed_ops, 1u);
    EXPECT_EQ(stats.staging_completed_ops, 0u);

    // A later invalid logical range aborts the batch before the valid first
    // request can write anything.
    auto invalid = request;
    invalid.key = "invalid";
    invalid.ranges[0].dst = static_cast<char*>(allocation) + 4096;
    invalid.ranges[0].object_offset = 4095;
    results = backend->BatchReadRanges({request, invalid}, &stats);
    EXPECT_FALSE(results[0]);
    EXPECT_FALSE(results[1]);
    EXPECT_EQ(stats.direct_completed_ops, 0u);
    EXPECT_EQ(stats.staging_completed_ops, 0u);
    EXPECT_EQ(stats.direct_bytes, 0u);
    EXPECT_EQ(stats.staging_bytes, 0u);
    EXPECT_EQ(stats.direct_physical_bytes, 0u);
    EXPECT_EQ(stats.staging_physical_bytes, 0u);
    EXPECT_EQ(stats.direct_hipfile_read_calls, 0u);
    EXPECT_EQ(stats.staging_hipfile_read_calls, 0u);
    EXPECT_EQ(stats.staging_chunks, 0u);
    EXPECT_EQ(stats.max_staged_op_bytes, 0u);
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

class XdsAggregatePlanTest : public ::testing::Test {
   protected:
    void SetUp() override {
        char directory[] = "/tmp/mooncake-xds-aggregate-XXXXXX";
        ASSERT_NE(::mkdtemp(directory), nullptr);
        directory_ = directory;
        config_.fsdir = directory_;
        config_.allocator_type = DfsAllocatorType::BUCKET;
        config_.batch_read_threads = 1;
        config_.xds_mode = XdsMode::kRequired;
        config_.xds_allow_request_fallback = false;
        path_ = directory_ + "/bucket_" +
                ImmutableBucketAllocator::FormatBucketId(0) + ".data";
        const int fd = ::open(path_.c_str(), O_CREAT | O_RDWR, 0600);
        ASSERT_GE(fd, 0);
        ASSERT_EQ(::ftruncate(fd, 65536), 0);
        (void)::close(fd);
        const int direct = ::open(path_.c_str(), O_RDONLY | O_DIRECT);
        if (direct < 0) GTEST_SKIP() << "O_DIRECT unavailable";
        (void)::close(direct);
        ASSERT_EQ(::posix_memalign(&memory_, 4096, 131072), 0);
        Start(true);
    }
    void TearDown() override {
        backend_.reset();
        std::free(memory_);
        if (!directory_.empty()) std::filesystem::remove_all(directory_);
    }
    void Start(bool enabled) {
        backend_.reset();
        config_.xds_staged_aggregate_dry_run = enabled;
        backend_ = std::make_shared<DistributedStorageBackend>(
            FileStorageConfig{}, config_, std::make_unique<PosixFsAdapter>());
        ASSERT_TRUE(backend_->Init());
        fake_ = std::make_shared<FakeAcceleratorFileIo>();
        backend_->SetAcceleratorFileIo(fake_);
        ASSERT_TRUE(backend_->RegisterXdsBuffer(memory_, 131072, 0));
    }
    DfsRangeReadRequest Request(
        std::initializer_list<std::pair<uint64_t, uint64_t>> ranges) {
        DfsRangeReadRequest request;
        request.key = "object";
        request.descriptor.file_path = path_;
        request.descriptor.shard_idx = 0;
        request.descriptor.object_size = 65536;
        request.descriptor.aligned_size = 65536;
        size_t index = 0;
        for (auto [offset, length] : ranges) {
            request.ranges.push_back({static_cast<char*>(memory_) +
                                          index++ * 32768 + 1,
                                      length, offset, 0, 0});
        }
        return request;
    }
    DistributedStorageBackend::XdsReadStats Run(
        const std::vector<DfsRangeReadRequest>& requests) {
        DistributedStorageBackend::XdsReadStats stats;
        auto results = backend_->BatchReadRanges(requests, &stats);
        for (const auto& result : results) EXPECT_TRUE(result);
        return stats;
    }
    std::string directory_, path_;
    void* memory_ = nullptr;
    DistributedStorageConfig config_;
    std::shared_ptr<DistributedStorageBackend> backend_;
    std::shared_ptr<FakeAcceleratorFileIo> fake_;
};

TEST_F(XdsAggregatePlanTest, CoversOverlapTouchGapCapacityAndLargeOps) {
    struct Case {
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        uint64_t calls, candidate_calls, bytes, candidate_bytes;
        uint64_t windows, gaps, capacity, oversized;
    };
    const std::vector<Case> cases = {
        {{{1, 10}}, 1, 1, 4096, 4096, 0, 0, 0, 0},
        {{{100, 10}, {1, 10}}, 2, 1, 8192, 4096, 1, 0, 0, 0},
        {{{4097, 10}, {1, 10}}, 2, 1, 8192, 8192, 1, 0, 0, 0},
        {{{1, 10}, {8193, 10}}, 2, 2, 8192, 8192, 0, 1, 0, 0},
        {{{1, 4096}, {8193, 10}}, 2, 2, 12288, 12288, 0, 0, 1, 0},
        {{{1, 8192}, {1, 10}}, 3, 3, 16384, 16384, 0, 0, 0, 1},
        {{{1, 16383}}, 2, 2, 16384, 16384, 0, 0, 0, 1},
        // Two final windows overlap; count each physical read, not their union.
        {{{1, 4096}, {4097, 4096}}, 2, 2, 16384, 16384, 0, 0, 1, 0},
    };
    for (const auto& test : cases) {
        auto request = Request({});
        size_t index = 0;
        for (auto [offset, length] : test.ranges) {
            request.ranges.push_back({static_cast<char*>(memory_) +
                                          index++ * 32768 + 1,
                                      length, offset, 0, 0});
        }
        const auto stats = Run({request});
        const auto& p = stats.staged_aggregate_plan;
        EXPECT_TRUE(p.enabled);
        EXPECT_TRUE(p.estimate_valid);
        EXPECT_EQ(p.evaluated_objects, 1u);
        EXPECT_EQ(p.skipped_objects, 0u);
        EXPECT_EQ(p.staged_ops, test.ranges.size());
        EXPECT_EQ(p.baseline_calls, test.calls);
        EXPECT_EQ(p.candidate_calls, test.candidate_calls);
        EXPECT_EQ(p.baseline_physical_bytes, test.bytes);
        EXPECT_EQ(p.candidate_physical_bytes, test.candidate_bytes);
        EXPECT_EQ(p.candidate_windows, test.windows);
        EXPECT_EQ(p.gap_rejections, test.gaps);
        EXPECT_EQ(p.capacity_rejections, test.capacity);
        EXPECT_EQ(p.oversized_ops, test.oversized);
    }
}

TEST_F(XdsAggregatePlanTest, DisabledAndEnabledKeepRealCallsAndPinsIdentical) {
    const auto request = Request({{4097, 10}, {1, 10}});
    std::vector<DirectReadOp> original;
    for (bool enabled : {false, true}) {
        Start(enabled);
        // Drop a duplicate logical reference during I/O, then drain the final
        // registration after completion. Neither mode may leak planner pins.
        ASSERT_TRUE(backend_->RegisterXdsBuffer(memory_, 131072, 0));
        fake_->on_read = [&] {
            if (fake_->staged_calls == 1)
                EXPECT_TRUE(backend_->UnregisterXdsBuffer(memory_));
            EXPECT_EQ(fake_->buffer_deregister_calls, 0);
        };
        const auto stats = Run({request});
        EXPECT_EQ(fake_->buffer_deregister_calls, 0);
        ASSERT_TRUE(backend_->UnregisterXdsBuffer(memory_));
        EXPECT_EQ(fake_->buffer_deregister_calls, 1);
        EXPECT_EQ(stats.staging_completed_ops, 2u);
        EXPECT_EQ(stats.staging_bytes, 20u);
        EXPECT_EQ(stats.staging_physical_bytes, 8192u);
        EXPECT_EQ(stats.staging_hipfile_read_calls, 2u);
        EXPECT_EQ(stats.staged_aggregate_plan.enabled, enabled);
        ASSERT_EQ(fake_->operations.size(), 2u);
        if (!enabled) {
            original = fake_->operations;
            EXPECT_EQ(stats.staged_aggregate_plan.staged_ops, 0u);
        } else {
            EXPECT_EQ(stats.staged_aggregate_plan.candidate_calls, 1u);
            for (size_t i = 0; i < original.size(); ++i) {
                const auto& a = original[i];
                const auto& b = fake_->operations[i];
                EXPECT_EQ(a.file.id, b.file.id);
                EXPECT_EQ(a.buffer.id, b.buffer.id);
                EXPECT_EQ(a.buffer.base, b.buffer.base);
                EXPECT_EQ(a.buffer.length, b.buffer.length);
                EXPECT_EQ(a.file_offset, b.file_offset);
                EXPECT_EQ(a.device_offset, b.device_offset);
                EXPECT_EQ(a.length, b.length);
                EXPECT_EQ(a.device_id, b.device_id);
                EXPECT_EQ(a.staged, b.staged);
                EXPECT_EQ(a.extent_begin, b.extent_begin);
                EXPECT_EQ(a.extent_end, b.extent_end);
            }
        }
    }
    Start(true);
    EXPECT_TRUE(backend_->BatchReadRanges({request})[0]);
    EXPECT_EQ(fake_->operations.size(), 2u);  // No stats pointer is supported.
}

TEST_F(XdsAggregatePlanTest, UsesMergedBaselineAndIgnoresDirectOperations) {
    auto request = Request({{1, 10}, {11, 10}, {8192, 4096}});
    request.ranges[1].dst = static_cast<char*>(memory_) + 11;
    request.ranges[2].dst = static_cast<char*>(memory_) + 65536;
    auto stats = Run({request});
    EXPECT_EQ(stats.staged_aggregate_plan.staged_ops, 1u);
    EXPECT_EQ(stats.staged_aggregate_plan.baseline_calls, 1u);
    EXPECT_EQ(stats.staged_aggregate_plan.candidate_windows, 0u);
    EXPECT_EQ(stats.direct_completed_ops, 1u);
    EXPECT_EQ(fake_->operations.size(), 2u);
}

TEST_F(XdsAggregatePlanTest, UnknownOrUnalignedScratchSkipsWithoutChangingIo) {
    const auto request = Request({{1, 10}, {4097, 10}});
    for (uint64_t scratch : {0u, 4097u}) {
        fake_->scratch_bytes = scratch;
        const auto stats = Run({request});
        const auto& p = stats.staged_aggregate_plan;
        EXPECT_FALSE(p.estimate_valid);
        EXPECT_TRUE(p.scratch_unavailable);
        EXPECT_EQ(p.evaluated_objects, 0u);
        EXPECT_EQ(p.skipped_objects, 1u);
        EXPECT_EQ(p.baseline_calls, 0u);
        EXPECT_EQ(stats.staging_completed_ops, 2u);
    }
}

TEST_F(XdsAggregatePlanTest, ObjectsAreIndependentAndFailuresRemainPlanned) {
    auto first = Request({{1, 10}});
    auto second = first;
    second.key = "second";
    second.ranges[0].dst = static_cast<char*>(memory_) + 32769;
    auto stats = Run({first, second});
    EXPECT_EQ(stats.staged_aggregate_plan.evaluated_objects, 2u);
    EXPECT_EQ(stats.staged_aggregate_plan.candidate_windows, 0u);
    EXPECT_EQ(stats.staged_aggregate_plan.candidate_calls, 2u);
    fake_->fail_staged = true;
    auto results = backend_->BatchReadRanges({first, second}, &stats);
    EXPECT_FALSE(results[0]);
    EXPECT_FALSE(results[1]);
    EXPECT_EQ(stats.staged_aggregate_plan.evaluated_objects, 2u);
    EXPECT_EQ(stats.staged_aggregate_plan.baseline_calls, 2u);
    EXPECT_EQ(stats.staging_completed_ops, 0u);
    EXPECT_EQ(stats.staging_physical_bytes, 0u);
}

TEST_F(XdsAggregatePlanTest, FilesDevicesAndExtentsStayIndependent) {
    const std::string other_path = directory_ + "/bucket_" +
        ImmutableBucketAllocator::FormatBucketId(1) + ".data";
    const int fd = ::open(other_path.c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::ftruncate(fd, 65536), 0);
    (void)::close(fd);
    ASSERT_TRUE(backend_->UnregisterXdsBuffer(memory_));
    ASSERT_TRUE(backend_->RegisterXdsBuffer(memory_, 32768, 0));
    ASSERT_TRUE(backend_->RegisterXdsBuffer(
        static_cast<char*>(memory_) + 32768, 32768, 1));
    auto first = Request({{1, 10}});
    auto second = first;
    second.key = "other";
    second.descriptor.shard_idx = 1;
    second.descriptor.file_path = other_path;
    second.descriptor.offset = 4096;
    second.descriptor.object_size = 4096;
    second.descriptor.aligned_size = 4096;
    second.ranges[0].dst = static_cast<char*>(memory_) + 32769;
    second.ranges[0].device_id = 1;
    auto stats = Run({first, second});
    EXPECT_EQ(stats.staged_aggregate_plan.evaluated_objects, 2u);
    EXPECT_EQ(stats.staged_aggregate_plan.candidate_calls, 2u);
    EXPECT_EQ(stats.staged_aggregate_plan.candidate_windows, 0u);
    ASSERT_EQ(fake_->operations.size(), 2u);
    EXPECT_NE(fake_->operations[0].file.id, fake_->operations[1].file.id);
    EXPECT_NE(fake_->operations[0].device_id, fake_->operations[1].device_id);
    EXPECT_EQ(fake_->operations[1].extent_begin, 4096u);
    EXPECT_EQ(fake_->operations[1].extent_end, 8192u);

    // Mixed devices in one object remain ineligible, not a candidate.
    first.ranges.push_back(second.ranges[0]);
    auto results = backend_->BatchReadRanges({first}, &stats);
    EXPECT_FALSE(results[0]);
    EXPECT_EQ(stats.staged_aggregate_plan.skipped_objects, 1u);
    EXPECT_EQ(stats.staged_aggregate_plan.evaluated_objects, 0u);
    EXPECT_EQ(fake_->operations.size(), 2u);
}

TEST_F(XdsAggregatePlanTest, CoverAtExtentEndIsValidButTruncatedEofIsNot) {
    auto request = Request({{4095, 1}});
    request.descriptor.offset = 4096;
    request.descriptor.object_size = 4096;
    request.descriptor.aligned_size = 4096;
    auto stats = Run({request});
    EXPECT_EQ(stats.staged_aggregate_plan.baseline_physical_bytes, 4096u);
    EXPECT_EQ(stats.staged_aggregate_plan.candidate_calls, 1u);
    ASSERT_EQ(::truncate(path_.c_str(), 8191), 0);
    auto results = backend_->BatchReadRanges({request}, &stats);
    EXPECT_FALSE(results[0]);
    EXPECT_EQ(stats.staged_aggregate_plan.evaluated_objects, 0u);
    EXPECT_EQ(stats.staged_aggregate_plan.skipped_objects, 1u);
    EXPECT_EQ(fake_->operations.size(), 1u);
}

TEST_F(XdsAggregatePlanTest, InvalidExtentAndOverflowNeverBecomeCandidates) {
    for (bool overflow : {false, true}) {
        auto request = Request({{1, 10}});
        if (overflow) {
            request.ranges[0].object_offset =
                std::numeric_limits<uint64_t>::max();
        } else {
            request.descriptor.aligned_size = 4095;
        }
        DistributedStorageBackend::XdsReadStats stats;
        auto results = backend_->BatchReadRanges({request}, &stats);
        EXPECT_FALSE(results[0]);
        EXPECT_EQ(stats.staged_aggregate_plan.evaluated_objects, 0u);
        EXPECT_EQ(stats.staged_aggregate_plan.skipped_objects, 1u);
        EXPECT_FALSE(stats.staged_aggregate_plan.estimate_valid);
        EXPECT_TRUE(fake_->operations.empty());
    }
}

}  // namespace
}  // namespace mooncake
