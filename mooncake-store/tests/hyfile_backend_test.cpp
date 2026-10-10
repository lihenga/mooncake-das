#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <limits>
#include <memory>
#include <string>

#include "storage/distributed/xds/hyfile_abi.h"
#include "storage/distributed/xds/hyfile_backend.h"

namespace mooncake {
namespace {

namespace abi = xds::hyfile_abi;

struct FakeState {
    abi::Error driver_open_result{abi::kSuccess, 0};
    abi::Error buffer_register_result{abi::kSuccess, 0};
    abi::Error buffer_deregister_result{abi::kSuccess, 0};
    abi::Error handle_register_result{abi::kSuccess, 0};
    int get_device_count_result = 0;
    int set_device_result = 0;
    int synchronize_result = 0;
    int device_count = 1;
    bool return_requested_length = true;
    ssize_t read_result = 0;

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

ssize_t Read(abi::FileHandle, void*, size_t size,
             unsigned long long file_offset,
             unsigned long long device_offset) {
    ++fake_state->read_calls;
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
    return fake_state->synchronize_result;
}

abi::Api CompleteApi() {
    return abi::Api{DriverOpen,       DriverClose,     HandleRegister,
                    HandleDeregister, BufferRegister, BufferDeregister,
                    Read};
}

xds::testing::HipRuntimeApi CompleteHipApi() {
    return xds::testing::HipRuntimeApi{GetDeviceCount, SetDevice,
                                       DeviceSynchronize, 0};
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
        fake_state = nullptr;
    }

    tl::expected<std::shared_ptr<AcceleratorFileIo>, XdsError> Create() {
        return xds::testing::CreateHyFileAcceleratorFileIoForTesting(
            RuntimeIdentity(), CompleteApi(), CompleteHipApi());
    }

    FakeState state_;
};

TEST(HyFileBackendProductionTest, MissingLibraryFailsClosed) {
    auto backend = CreateHyFileAcceleratorFileIo(
        "/path/that/does/not/exist/libhyfile.so");
    ASSERT_FALSE(backend);
    EXPECT_EQ(backend.error().error_class, XdsErrorClass::kNotEligible);
    EXPECT_EQ(backend.error().target_state, XdsTargetState::kUntouched);
    EXPECT_EQ(backend.error().operation, "dlopen(libhyfile)");
}

TEST_F(HyFileBackendTest, MissingRequiredFunctionFailsBeforeDriverOpen) {
    auto api = CompleteApi();
    api.read = nullptr;
    auto backend = xds::testing::CreateHyFileAcceleratorFileIoForTesting(
        RuntimeIdentity(), api, CompleteHipApi());
    ASSERT_FALSE(backend);
    EXPECT_EQ(backend.error().operation, "hyfile-required-symbol-missing");
    EXPECT_EQ(state_.driver_open_calls, 0);
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
