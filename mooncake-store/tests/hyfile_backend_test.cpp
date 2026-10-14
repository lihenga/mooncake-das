#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <memory>
#include <string>

#include "storage/distributed/xds/hyfile_abi.h"
#include "storage/distributed/xds/hyfile_backend.h"

namespace mooncake {
namespace {

namespace abi = xds::hyfile_abi;
constexpr size_t kDefaultStagingBufferBytes = 4 * 1024 * 1024;

struct FakeState {
    abi::Error driver_open_result{abi::kSuccess, 0};
    abi::Error buffer_register_result{abi::kSuccess, 0};
    abi::Error buffer_deregister_result{abi::kSuccess, 0};
    abi::Error handle_register_result{abi::kSuccess, 0};
    int get_device_count_result = 0;
    int set_device_result = 0;
    int synchronize_result = 0;
    int device_count = 1;
    int malloc_calls = 0;
    int malloc_result = 0;
    bool misalign_allocation = false;
    int free_calls = 0;
    int copy_calls = 0;
    int copy_result = 0;
    int fail_sync_call = 0;
    bool fill_reads = false;
    std::vector<void*> allocations;
    bool return_requested_length = true;
    ssize_t read_result = 0;
    size_t expected_malloc_size = kDefaultStagingBufferBytes;

    int driver_open_calls = 0;
    int driver_close_calls = 0;
    int handle_register_calls = 0;
    int handle_deregister_calls = 0;
    int buffer_register_calls = 0;
    int buffer_deregister_calls = 0;
    int read_calls = 0;
    int set_device_calls = 0;
    int synchronize_calls = 0;

    void* last_buffer = nullptr;
    size_t last_size = 0;
    int last_flags = 0;
    int last_device = -1;
    unsigned long long last_file_offset = 0;
    unsigned long long last_device_offset = 0;
};

FakeState* fake_state = nullptr;

abi::Error DriverOpen() {
    ++fake_state->driver_open_calls;
    return fake_state->driver_open_result;
}

abi::Error DriverClose() {
    ++fake_state->driver_close_calls;
    return {abi::kSuccess, 0};
}

abi::Error HandleRegister(abi::FileHandle* handle,
                          abi::FileDescriptor* descriptor) {
    ++fake_state->handle_register_calls;
    if (descriptor->type != abi::kFileHandleTypeOpaqueFd ||
        descriptor->handle.fd < 0 || descriptor->fs_ops != nullptr) {
        return {5022, 0};
    }
    *handle = reinterpret_cast<void*>(0x1234);
    return fake_state->handle_register_result;
}

void HandleDeregister(abi::FileHandle) {
    ++fake_state->handle_deregister_calls;
}

abi::Error BufferRegister(void* base, size_t size, int flags) {
    ++fake_state->buffer_register_calls;
    fake_state->last_buffer = base;
    fake_state->last_size = size;
    fake_state->last_flags = flags;
    return fake_state->buffer_register_result;
}

abi::Error BufferDeregister(void* base) {
    ++fake_state->buffer_deregister_calls;
    fake_state->last_buffer = base;
    return fake_state->buffer_deregister_result;
}

ssize_t Read(abi::FileHandle, void* base, size_t size,
             unsigned long long file_offset,
             unsigned long long device_offset) {
    ++fake_state->read_calls;
    EXPECT_EQ(reinterpret_cast<uintptr_t>(base) % 4096, 0u);
    EXPECT_EQ(size % 4096, 0u);
    EXPECT_EQ(file_offset % 4096, 0u);
    EXPECT_EQ(device_offset % 4096, 0u);
    if (fake_state->fill_reads) {
        for (size_t i = 0; i < size; ++i) {
            static_cast<unsigned char*>(base)[device_offset + i] =
                static_cast<unsigned char>((file_offset + i) % 251);
        }
    }
    fake_state->last_size = size;
    fake_state->last_file_offset = file_offset;
    fake_state->last_device_offset = device_offset;
    return fake_state->return_requested_length
               ? static_cast<ssize_t>(size)
               : fake_state->read_result;
}

int GetDeviceCount(int* count) {
    *count = fake_state->device_count;
    return fake_state->get_device_count_result;
}

int SetDevice(int device_id) {
    ++fake_state->set_device_calls;
    fake_state->last_device = device_id;
    return fake_state->set_device_result;
}

int DeviceSynchronize() {
    ++fake_state->synchronize_calls;
    return fake_state->fail_sync_call == fake_state->synchronize_calls
               ? 9 : fake_state->synchronize_result;
}

int MallocDevice(void** base, size_t length) {
    ++fake_state->malloc_calls;
    EXPECT_EQ(length, fake_state->expected_malloc_size);
    if (fake_state->malloc_result != 0) return fake_state->malloc_result;
    const int result = ::posix_memalign(base, 4096, length + 4096);
    if (result == 0) {
        fake_state->allocations.push_back(*base);
        if (fake_state->misalign_allocation) {
            *base = static_cast<char*>(*base) + 1;
        }
    }
    return result;
}

int FreeDevice(void*) {
    ++fake_state->free_calls;
    // The fixture owns fake allocations, including quarantined ones.
    return 0;
}

int CopyDevice(void* dst, const void* src, size_t length) {
    ++fake_state->copy_calls;
    if (fake_state->copy_result == 0) std::memcpy(dst, src, length);
    return fake_state->copy_result;
}

abi::Api CompleteApi() {
    return abi::Api{DriverOpen,       DriverClose,     HandleRegister,
                    HandleDeregister, BufferRegister, BufferDeregister,
                    Read};
}

xds::testing::HipRuntimeApi CompleteHipApi() {
    return xds::testing::HipRuntimeApi{GetDeviceCount, SetDevice,
                                       DeviceSynchronize, 0, MallocDevice,
                                       FreeDevice, CopyDevice};
}

std::string RuntimeIdentity() {
    const auto* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    return std::string("hyfile-test:") + info->test_suite_name() + "." +
           info->name();
}

class AlignedMemory {
   public:
    explicit AlignedMemory(size_t size) : size_(size) {
        if (::posix_memalign(&data_, 4096, size_) != 0) data_ = nullptr;
    }
    ~AlignedMemory() { std::free(data_); }

    void* data() const { return data_; }
    size_t size() const { return size_; }

   private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

class DirectFile {
   public:
    DirectFile() {
        char path[] = "/tmp/mooncake-hyfile-test-XXXXXX";
        const int temporary_fd = ::mkstemp(path);
        if (temporary_fd < 0) return;
        path_ = path;
        (void)::ftruncate(temporary_fd, 16 * 1024 * 1024);
        (void)::close(temporary_fd);
        fd_ = ::open(path_.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    }

    ~DirectFile() {
        if (fd_ >= 0) (void)::close(fd_);
        if (!path_.empty()) (void)::unlink(path_.c_str());
    }

    bool valid() const { return fd_ >= 0; }
    int fd() const { return fd_; }

    XdsFileIdentity identity() const {
        struct stat info {};
        EXPECT_EQ(::fstat(fd_, &info), 0);
        return XdsFileIdentity{path_, static_cast<uint64_t>(info.st_dev),
                               static_cast<uint64_t>(info.st_ino)};
    }

   private:
    std::string path_;
    int fd_ = -1;
};

class HyFileBackendTest : public ::testing::Test {
   protected:
    void SetUp() override { fake_state = &state_; }
    void TearDown() override {
        xds::testing::ResetHyFileProcessRuntimeForTesting();
        for (void* allocation : state_.allocations) std::free(allocation);
        fake_state = nullptr;
    }

    tl::expected<std::shared_ptr<AcceleratorFileIo>, XdsError> Create(
        size_t staging_buffer_bytes = kDefaultStagingBufferBytes) {
        state_.expected_malloc_size = staging_buffer_bytes;
        return xds::testing::CreateHyFileAcceleratorFileIoForTesting(
            RuntimeIdentity(), CompleteApi(), CompleteHipApi(),
            staging_buffer_bytes);
    }

    FakeState state_;
};

TEST(HyFileBackendProductionTest, MissingLibraryFailsClosed) {
    auto backend = CreateHyFileAcceleratorFileIo(
        "/path/that/does/not/exist/libhyfile.so",
        kDefaultStagingBufferBytes);
    ASSERT_FALSE(backend);
    EXPECT_EQ(backend.error().error_class, XdsErrorClass::kNotEligible);
    EXPECT_EQ(backend.error().target_state, XdsTargetState::kUntouched);
    EXPECT_EQ(backend.error().operation, "dlopen(libhyfile)");
}

TEST_F(HyFileBackendTest, MissingRequiredFunctionFailsBeforeDriverOpen) {
    auto api = CompleteApi();
    api.read = nullptr;
    auto backend = xds::testing::CreateHyFileAcceleratorFileIoForTesting(
        RuntimeIdentity(), api, CompleteHipApi(),
        kDefaultStagingBufferBytes);
    ASSERT_FALSE(backend);
    EXPECT_EQ(backend.error().operation, "hyfile-required-symbol-missing");
    EXPECT_EQ(state_.driver_open_calls, 0);
}

TEST_F(HyFileBackendTest, CapabilitiesReportActualScratchCapacity) {
    EXPECT_EQ(DirectIoCapabilities{}.staging_buffer_bytes, 0u);
    auto backend = Create(8192);
    ASSERT_TRUE(backend);
    EXPECT_TRUE((*backend)->Capabilities().supports_gpu_staging);
    EXPECT_EQ((*backend)->Capabilities().staging_buffer_bytes, 8192u);
}

TEST_F(HyFileBackendTest, ProcessRuntimeOpensAndClosesDriverOnce) {
    auto first = Create();
    ASSERT_TRUE(first);
    auto second = Create();
    ASSERT_TRUE(second);
    EXPECT_EQ(state_.driver_open_calls, 1);

    first->reset();
    EXPECT_EQ(state_.driver_close_calls, 0);
    second->reset();
    EXPECT_EQ(state_.driver_close_calls, 0);
    xds::testing::ResetHyFileProcessRuntimeForTesting();
    EXPECT_EQ(state_.driver_close_calls, 1);
}

TEST_F(HyFileBackendTest, RejectsInvalidStagingBufferSizeBeforeDriverOpen) {
    auto backend = Create(4097);
    ASSERT_FALSE(backend);
    EXPECT_EQ(backend.error().error_class, XdsErrorClass::kContractViolation);
    EXPECT_EQ(backend.error().operation, "hyfile-staging-buffer-size");
    EXPECT_EQ(state_.driver_open_calls, 0);
}

TEST_F(HyFileBackendTest, BufferLifecycleSynchronizesBeforeDeregister) {
    AlignedMemory memory(8192);
    ASSERT_NE(memory.data(), nullptr);
    auto backend = Create();
    ASSERT_TRUE(backend);

    auto registration = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(registration);
    EXPECT_EQ(state_.buffer_register_calls, 1);
    EXPECT_EQ(state_.last_flags, abi::kRdmaRegister);

    auto result = (*backend)->DeregisterBuffer(*registration);
    ASSERT_TRUE(result);
    EXPECT_EQ(state_.synchronize_calls, 1);
    EXPECT_EQ(state_.buffer_deregister_calls, 1);
}

TEST_F(HyFileBackendTest, DeregisterSyncFailureKeepsBufferQuarantined) {
    AlignedMemory memory(4096);
    ASSERT_NE(memory.data(), nullptr);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto registration = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(registration);

    state_.synchronize_result = 7;
    auto result = (*backend)->DeregisterBuffer(*registration);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().error_class, XdsErrorClass::kBackendFatal);
    EXPECT_EQ(result.error().target_state,
              XdsTargetState::kUnknownMayStillBeWritten);
    EXPECT_EQ(state_.buffer_deregister_calls, 0);

    // Permit best-effort destructor cleanup without touching real HIP.
    state_.synchronize_result = 0;
}

TEST_F(HyFileBackendTest, ReadsExactRangeAndSynchronizes) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(8192);
    ASSERT_NE(memory.data(), nullptr);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto file_registration =
        (*backend)->RegisterFile(file.fd(), file.identity());
    ASSERT_TRUE(file_registration);
    auto buffer_registration =
        (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(buffer_registration);

    DirectReadOp op{*file_registration, *buffer_registration,
                    4096, 4096, 4096, 0};
    auto result = (*backend)->Read(op);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->bytes, 4096u);
    EXPECT_EQ(state_.read_calls, 1);
    EXPECT_EQ(state_.last_file_offset, 4096u);
    EXPECT_EQ(state_.last_device_offset, 4096u);
    EXPECT_EQ(state_.synchronize_calls, 1);

    ASSERT_TRUE((*backend)->DeregisterFile(*file_registration));
    ASSERT_TRUE((*backend)->DeregisterBuffer(*buffer_registration));
}

TEST_F(HyFileBackendTest, ShortReadIsSafeToOverwriteAfterSync) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    ASSERT_NE(memory.data(), nullptr);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto file_registration =
        (*backend)->RegisterFile(file.fd(), file.identity());
    ASSERT_TRUE(file_registration);
    auto buffer_registration =
        (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(buffer_registration);

    state_.return_requested_length = false;
    state_.read_result = 2048;
    DirectReadOp op{*file_registration, *buffer_registration, 0, 0, 4096, 0};
    auto result = (*backend)->Read(op);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().error_class, XdsErrorClass::kTransientIo);
    EXPECT_EQ(result.error().target_state,
              XdsTargetState::kStoppedSafeToOverwrite);

    ASSERT_TRUE((*backend)->DeregisterFile(*file_registration));
    ASSERT_TRUE((*backend)->DeregisterBuffer(*buffer_registration));
}

TEST_F(HyFileBackendTest, ReadSyncFailureMarksTargetStateUnknown) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    ASSERT_NE(memory.data(), nullptr);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto file_registration =
        (*backend)->RegisterFile(file.fd(), file.identity());
    ASSERT_TRUE(file_registration);
    auto buffer_registration =
        (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(buffer_registration);

    state_.synchronize_result = 9;
    DirectReadOp op{*file_registration, *buffer_registration, 0, 0, 4096, 0};
    auto result = (*backend)->Read(op);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().error_class, XdsErrorClass::kBackendFatal);
    EXPECT_EQ(result.error().target_state,
              XdsTargetState::kUnknownMayStillBeWritten);

    state_.synchronize_result = 0;
    ASSERT_TRUE((*backend)->DeregisterFile(*file_registration));
    ASSERT_TRUE((*backend)->DeregisterBuffer(*buffer_registration));
}

TEST_F(HyFileBackendTest, StagingChunksExactLogicalBytesAndReusesScratch) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    constexpr size_t capacity = 8192;
    AlignedMemory memory(capacity + 8192);
    ASSERT_NE(memory.data(), nullptr);
    std::memset(memory.data(), 0xff, memory.size());
    auto backend = Create(capacity);
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto br = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(br);
    state_.fill_reads = true;
    DirectReadOp op{*fr, *br, 4095, 13, capacity + 7, 0,
                    true, 0, 16 * 1024 * 1024};
    auto result = (*backend)->ReadStaged(op);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->bytes, op.length);
    EXPECT_EQ(result->physical_bytes, capacity + 8192);
    EXPECT_EQ(result->hipfile_read_calls, 2u);
    EXPECT_EQ(result->staging_chunks, 2u);
    EXPECT_EQ(state_.read_calls, 2);
    EXPECT_EQ(state_.copy_calls, 2);
    EXPECT_EQ(state_.malloc_calls, 1);
    const auto* bytes = static_cast<unsigned char*>(memory.data());
    for (size_t i = 0; i < memory.size(); ++i) {
        const unsigned char expected = i >= 13 && i < 13 + op.length
            ? static_cast<unsigned char>((op.file_offset + i - 13) % 251)
            : 0xff;
        ASSERT_EQ(bytes[i], expected) << i;
    }
    op.file_offset = 0;
    op.length = 1;
    ASSERT_TRUE((*backend)->ReadStaged(op));
    op.file_offset = op.extent_end - 1;
    ASSERT_TRUE((*backend)->ReadStaged(op));
    EXPECT_EQ(state_.malloc_calls, 1);
    backend->reset();
    EXPECT_EQ(state_.free_calls, 1);
    EXPECT_EQ(state_.buffer_deregister_calls, 2);
}

TEST_F(HyFileBackendTest, StagingRejectsExtentFileBoundsAndUnknownTarget) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto br = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(br);
    DirectReadOp op{*fr, *br, 1, 0, 1, 0, true, 1, 4096};
    EXPECT_FALSE((*backend)->ReadStaged(op));
    op.extent_begin = 0;
    op.extent_end = 4095;
    EXPECT_FALSE((*backend)->ReadStaged(op));
    op.extent_end = 32 * 1024 * 1024;
    op.file_offset = 16 * 1024 * 1024;
    EXPECT_FALSE((*backend)->ReadStaged(op));
    op.file_offset = 1;
    ++op.buffer.id;
    EXPECT_FALSE((*backend)->ReadStaged(op));
    EXPECT_EQ(state_.malloc_calls, 0);
    EXPECT_EQ(state_.read_calls, 0);
}

TEST_F(HyFileBackendTest, StagingShortReadNeverCopiesTarget) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto br = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(br);
    state_.return_requested_length = false;
    state_.read_result = 2048;
    DirectReadOp op{*fr, *br, 1, 1, 2000, 0, true, 0, 4096};
    auto result = (*backend)->ReadStaged(op);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().target_state, XdsTargetState::kUntouched);
    EXPECT_EQ(state_.copy_calls, 0);
}

TEST_F(HyFileBackendTest, StagingCopyFailureSynchronizesBeforeReturning) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto br = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(br);
    state_.copy_result = 7;
    DirectReadOp op{*fr, *br, 1, 1, 2000, 0, true, 0, 4096};
    auto result = (*backend)->ReadStaged(op);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().operation, "hipMemcpy:staging");
    EXPECT_EQ(result.error().target_state,
              XdsTargetState::kStoppedSafeToOverwrite);
    EXPECT_EQ(state_.synchronize_calls, 2);
}

TEST_F(HyFileBackendTest, StagingReadSyncFailureQuarantinesScratch) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto br = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(br);
    state_.synchronize_result = 9;
    DirectReadOp op{*fr, *br, 1, 1, 2000, 0, true, 0, 4096};
    auto result = (*backend)->ReadStaged(op);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().target_state, XdsTargetState::kUntouched);
    EXPECT_EQ(state_.copy_calls, 0);
    EXPECT_FALSE((*backend)->ReadStaged(op));
    EXPECT_EQ(state_.read_calls, 1);
    EXPECT_FALSE((*backend)->DeregisterFile(*fr));
    backend->reset();
    EXPECT_EQ(state_.free_calls, 0);
    EXPECT_EQ(state_.handle_deregister_calls, 0);
}

TEST_F(HyFileBackendTest, StagingCopySyncFailureMarksTargetUnknown) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto br = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(br);
    state_.fail_sync_call = 2;
    DirectReadOp op{*fr, *br, 1, 1, 2000, 0, true, 0, 4096};
    auto result = (*backend)->ReadStaged(op);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().target_state,
              XdsTargetState::kUnknownMayStillBeWritten);
    EXPECT_FALSE((*backend)->ReadStaged(op));
    EXPECT_EQ(state_.read_calls, 1);
    // A later successful sync permits safe destructor cleanup.
    backend->reset();
    EXPECT_EQ(state_.free_calls, 1);
}

TEST_F(HyFileBackendTest, StagingAllocationsAreBoundedPerDevice) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory first(4096);
    AlignedMemory second(4096);
    state_.device_count = 2;
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto a = (*backend)->RegisterBuffer(first.data(), first.size(), 0);
    auto b = (*backend)->RegisterBuffer(second.data(), second.size(), 1);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    DirectReadOp op{*fr, *a, 1, 1, 1, 0, true, 0, 4096};
    state_.malloc_result = 7;
    EXPECT_FALSE((*backend)->ReadStaged(op));
    EXPECT_EQ(state_.read_calls, 0);
    state_.malloc_result = 0;
    ASSERT_TRUE((*backend)->ReadStaged(op));
    ASSERT_TRUE((*backend)->ReadStaged(op));
    op.buffer = *b;
    op.device_id = 1;
    ASSERT_TRUE((*backend)->ReadStaged(op));
    ASSERT_TRUE((*backend)->ReadStaged(op));
    EXPECT_EQ(state_.allocations.size(), 2u);
    backend->reset();
    EXPECT_EQ(state_.free_calls, 2);
}

TEST_F(HyFileBackendTest, StagingChecksActualAllocationAlignment) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto br = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(br);
    state_.misalign_allocation = true;
    DirectReadOp op{*fr, *br, 1, 1, 1, 0, true, 0, 4096};
    auto result = (*backend)->ReadStaged(op);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().operation, "hipMalloc:staging-alignment");
    EXPECT_EQ(state_.buffer_register_calls, 1);  // Only the target.
    EXPECT_EQ(state_.read_calls, 0);
    backend->reset();
    EXPECT_EQ(state_.free_calls, 1);
}

TEST_F(HyFileBackendTest, StagingRegistrationFailureRetainsUncertainMapping) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto br = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(br);
    state_.buffer_register_result = {5011, 0};
    DirectReadOp op{*fr, *br, 1, 1, 1, 0, true, 0, 4096};
    EXPECT_FALSE((*backend)->ReadStaged(op));
    EXPECT_FALSE((*backend)->ReadStaged(op));
    EXPECT_EQ(state_.malloc_calls, 1);
    EXPECT_EQ(state_.read_calls, 0);
    backend->reset();
    EXPECT_EQ(state_.free_calls, 0);
}

TEST_F(HyFileBackendTest, StagingCleanupFailureDoesNotFreeRegisteredMemory) {
    DirectFile file;
    if (!file.valid()) GTEST_SKIP() << "O_DIRECT is unavailable";
    AlignedMemory memory(4096);
    auto backend = Create();
    ASSERT_TRUE(backend);
    auto fr = (*backend)->RegisterFile(file.fd(), file.identity());
    auto br = (*backend)->RegisterBuffer(memory.data(), memory.size(), 0);
    ASSERT_TRUE(fr);
    ASSERT_TRUE(br);
    DirectReadOp op{*fr, *br, 1, 1, 1, 0, true, 0, 4096};
    ASSERT_TRUE((*backend)->ReadStaged(op));
    state_.buffer_deregister_result = {5011, 0};
    backend->reset();
    EXPECT_EQ(state_.free_calls, 0);
    EXPECT_EQ(state_.handle_deregister_calls, 0);
}

TEST_F(HyFileBackendTest, InheritedRuntimeIsUnavailableAfterFork) {
    auto backend = Create();
    ASSERT_TRUE(backend);

    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        ::_exit((*backend)->Capabilities().available ? 1 : 0);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

}  // namespace
}  // namespace mooncake
