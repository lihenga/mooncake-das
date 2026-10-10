#include "storage/distributed/xds/hyfile_backend.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <hip/hip_runtime_api.h>
#include <hipfile.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mooncake {
namespace {

constexpr int kHipFileRdmaRegister = 1;
constexpr uint64_t kDirectIoAlignment = 4096;

XdsError Failure(XdsErrorClass error_class, XdsTargetState target_state,
                 std::string operation, int64_t raw_code = 0) {
    return XdsError{error_class, target_state, raw_code, std::move(operation)};
}

template <typename Function>
bool LoadSymbol(void* library, const char* name, Function* function) {
    ::dlerror();
    *function = reinterpret_cast<Function>(::dlsym(library, name));
    return *function != nullptr && ::dlerror() == nullptr;
}

struct HyFileApi {
    decltype(&hipFileDriverOpen) driver_open = nullptr;
    decltype(&hipFileDriverClose) driver_close = nullptr;
    decltype(&hipFileHandleRegister) handle_register = nullptr;
    decltype(&hipFileHandleDeregister) handle_deregister = nullptr;
    decltype(&hipFileBufRegister) buffer_register = nullptr;
    decltype(&hipFileBufDeregister) buffer_deregister = nullptr;
    decltype(&hipFileRead) read = nullptr;
};

bool IsSuccess(hipFileError_t result) {
    return result.err == hipFileSuccess;
}

int64_t RawCode(hipFileError_t result) {
    if (result.err != hipFileSuccess) {
        return static_cast<int64_t>(result.err);
    }
    return static_cast<int64_t>(result.hip_drv_err);
}

class HyFileAcceleratorFileIo final : public AcceleratorFileIo {
   public:
    HyFileAcceleratorFileIo(void* library, HyFileApi api,
                            std::vector<DeviceXdsState> device_states)
        : library_(library),
          api_(api),
          device_states_(std::move(device_states)) {}

    ~HyFileAcceleratorFileIo() override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [_, buffer] : buffers_) {
            if (buffer.device_id >= 0) {
                (void)::hipSetDevice(buffer.device_id);
            }
            (void)api_.buffer_deregister(buffer.base);
        }
        for (const auto& [_, file] : files_) {
            api_.handle_deregister(file.handle);
        }
        buffers_.clear();
        files_.clear();
        if (driver_opened_) {
            (void)api_.driver_close();
            driver_opened_ = false;
        }
        if (library_) {
            (void)::dlclose(library_);
            library_ = nullptr;
        }
    }

    HyFileAcceleratorFileIo(const HyFileAcceleratorFileIo&) = delete;
    HyFileAcceleratorFileIo& operator=(const HyFileAcceleratorFileIo&) = delete;

    DirectIoCapabilities Capabilities() const override {
        DirectIoCapabilities capabilities;
        capabilities.available = driver_opened_;
        capabilities.supports_distributed_fs = driver_opened_;
        capabilities.file_offset_alignment = kDirectIoAlignment;
        capabilities.device_address_alignment = kDirectIoAlignment;
        capabilities.length_alignment = kDirectIoAlignment;
        return capabilities;
    }

    DeviceXdsState DeviceState(int32_t device_id) const override {
        if (device_id < 0 ||
            static_cast<size_t>(device_id) >= device_states_.size()) {
            return DeviceXdsState::kUnavailable;
        }
        return device_states_[device_id];
    }

    tl::expected<XdsFileRegistration, XdsError> RegisterFile(
        int fd, const XdsFileIdentity& identity) override {
        const int flags = fd >= 0 ? ::fcntl(fd, F_GETFL) : -1;
        if (fd < 0 || flags < 0 ||
#ifdef O_DIRECT
            (flags & O_DIRECT) == 0 ||
#endif
            identity.canonical_path.empty()) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched, "hipFileHandleRegister:fd"));
        }
        struct stat file_info {};
        if (::fstat(fd, &file_info) != 0 ||
            static_cast<uint64_t>(file_info.st_dev) != identity.device ||
            static_cast<uint64_t>(file_info.st_ino) != identity.inode) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched,
                "hipFileHandleRegister:file-identity", errno));
        }

        hipFileDescr_t descriptor{};
        descriptor.type = hipFileHandleTypeOpaqueFD;
        descriptor.handle.fd = fd;
        hipFileHandle_t handle = nullptr;

        std::lock_guard<std::mutex> lock(mutex_);
        const hipFileError_t result =
            api_.handle_register(&handle, &descriptor);
        if (!IsSuccess(result)) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kTransientIo, XdsTargetState::kUntouched,
                "hipFileHandleRegister", RawCode(result)));
        }
        const uint64_t id = next_id_++;
        files_.emplace(id, FileRegistration{handle, fd});
        return XdsFileRegistration{id, fd};
    }

    tl::expected<XdsBufferRegistration, XdsError> RegisterBuffer(
        void* base, size_t length, int32_t device_id) override {
        if (!base || length == 0 ||
            reinterpret_cast<uintptr_t>(base) % kDirectIoAlignment != 0 ||
            length % kDirectIoAlignment != 0 ||
            DeviceState(device_id) != DeviceXdsState::kAvailable) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kNotEligible, XdsTargetState::kUntouched,
                "hipFileBufRegister:arguments"));
        }

        std::lock_guard<std::mutex> lock(mutex_);
        const hipError_t set_device_result = ::hipSetDevice(device_id);
        if (set_device_result != hipSuccess) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal, XdsTargetState::kUntouched,
                "hipSetDevice:register", set_device_result));
        }
        const hipFileError_t result =
            api_.buffer_register(base, length, kHipFileRdmaRegister);
        if (!IsSuccess(result)) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kTransientIo, XdsTargetState::kUntouched,
                "hipFileBufRegister", RawCode(result)));
        }
        const uint64_t id = next_id_++;
        const XdsBufferRegistration registration{id, base, length, device_id};
        buffers_.emplace(id, registration);
        return registration;
    }

    tl::expected<DirectReadCompletion, XdsError> Read(
        const DirectReadOp& op) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto file = files_.find(op.file.id);
        auto buffer = buffers_.find(op.buffer.id);
        if (file == files_.end() || buffer == buffers_.end() ||
            op.file.fd != file->second.fd ||
            op.buffer.base != buffer->second.base ||
            op.buffer.length != buffer->second.length ||
            op.buffer.device_id != buffer->second.device_id ||
            op.device_id != buffer->second.device_id || op.length == 0 ||
            op.file_offset % kDirectIoAlignment != 0 ||
            op.device_offset % kDirectIoAlignment != 0 ||
            op.length % kDirectIoAlignment != 0 ||
            op.device_offset > buffer->second.length ||
            op.length > buffer->second.length - op.device_offset ||
            op.file_offset >
                static_cast<uint64_t>(std::numeric_limits<hoff_t>::max()) ||
            op.length >
                static_cast<uint64_t>(std::numeric_limits<hoff_t>::max()) -
                    op.file_offset ||
            op.device_offset >
                static_cast<uint64_t>(std::numeric_limits<hoff_t>::max())) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched, "hipFileRead:arguments"));
        }
        const hipError_t set_device_result = ::hipSetDevice(op.device_id);
        if (set_device_result != hipSuccess) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal, XdsTargetState::kUntouched,
                "hipSetDevice:read", set_device_result));
        }

        errno = 0;
        const ssize_t bytes = api_.read(
            file->second.handle, buffer->second.base, op.length,
            static_cast<hoff_t>(op.file_offset),
            static_cast<hoff_t>(op.device_offset));
        const int saved_errno = errno;
        const hipError_t sync_result = ::hipDeviceSynchronize();
        if (sync_result != hipSuccess) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal,
                XdsTargetState::kUnknownMayStillBeWritten,
                "hipDeviceSynchronize", sync_result));
        }
        if (bytes < 0 || static_cast<uint64_t>(bytes) != op.length) {
            const int64_t raw = bytes == -1 && saved_errno != 0
                                    ? -static_cast<int64_t>(saved_errno)
                                    : static_cast<int64_t>(bytes);
            return tl::make_unexpected(Failure(
                XdsErrorClass::kTransientIo,
                XdsTargetState::kStoppedSafeToOverwrite, "hipFileRead", raw));
        }
        return DirectReadCompletion{static_cast<uint64_t>(bytes)};
    }

    tl::expected<void, XdsError> DeregisterFile(
        XdsFileRegistration registration) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto file = files_.find(registration.id);
        if (file == files_.end()) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched,
                "hipFileHandleDeregister:unknown-registration"));
        }
        if (registration.fd != file->second.fd) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched,
                "hipFileHandleDeregister:registration-mismatch"));
        }
        api_.handle_deregister(file->second.handle);
        files_.erase(file);
        return {};
    }

    tl::expected<void, XdsError> DeregisterBuffer(
        XdsBufferRegistration registration) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto buffer = buffers_.find(registration.id);
        if (buffer == buffers_.end() ||
            buffer->second.base != registration.base ||
            buffer->second.length != registration.length ||
            buffer->second.device_id != registration.device_id) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched,
                "hipFileBufDeregister:unknown-registration"));
        }
        const hipError_t set_device_result =
            ::hipSetDevice(buffer->second.device_id);
        if (set_device_result != hipSuccess) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal, XdsTargetState::kUntouched,
                "hipSetDevice:deregister", set_device_result));
        }
        const hipFileError_t result =
            api_.buffer_deregister(buffer->second.base);
        if (!IsSuccess(result)) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal, XdsTargetState::kUntouched,
                "hipFileBufDeregister", RawCode(result)));
        }
        buffers_.erase(buffer);
        return {};
    }

    void MarkDriverOpened() { driver_opened_ = true; }

   private:
    struct FileRegistration {
        hipFileHandle_t handle = nullptr;
        int fd = -1;
    };

    void* library_ = nullptr;
    HyFileApi api_;
    std::vector<DeviceXdsState> device_states_;
    bool driver_opened_ = false;
    uint64_t next_id_ = 1;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, FileRegistration> files_;
    std::unordered_map<uint64_t, XdsBufferRegistration> buffers_;
};

}  // namespace

tl::expected<std::shared_ptr<AcceleratorFileIo>, XdsError>
CreateHyFileAcceleratorFileIo(const std::string& library_path) {
    void* library = ::dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        return tl::make_unexpected(Failure(
            XdsErrorClass::kNotEligible, XdsTargetState::kUntouched,
            "dlopen(libhyfile)"));
    }

    HyFileApi api;
    bool loaded = LoadSymbol(library, "hipFileDriverOpen", &api.driver_open) &&
                  LoadSymbol(library, "hipFileHandleRegister",
                             &api.handle_register) &&
                  LoadSymbol(library, "hipFileHandleDeregister",
                             &api.handle_deregister) &&
                  LoadSymbol(library, "hipFileBufRegister",
                             &api.buffer_register) &&
                  LoadSymbol(library, "hipFileBufDeregister",
                             &api.buffer_deregister) &&
                  LoadSymbol(library, "hipFileRead", &api.read);
    // Hygon's deployed library exports the cuFile-compatible _v2 close name;
    // newer hipFile releases export hipFileDriverClose instead.
    loaded = loaded &&
             (LoadSymbol(library, "hipFileDriverClose_v2", &api.driver_close) ||
              LoadSymbol(library, "hipFileDriverClose", &api.driver_close));
    if (!loaded) {
        (void)::dlclose(library);
        return tl::make_unexpected(Failure(
            XdsErrorClass::kNotEligible, XdsTargetState::kUntouched,
            "dlsym(libhyfile-required-symbol)"));
    }

    const hipFileError_t open_result = api.driver_open();
    if (!IsSuccess(open_result)) {
        (void)::dlclose(library);
        return tl::make_unexpected(Failure(
            XdsErrorClass::kBackendFatal, XdsTargetState::kUntouched,
            "hipFileDriverOpen", RawCode(open_result)));
    }

    int device_count = 0;
    if (::hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
        (void)api.driver_close();
        (void)::dlclose(library);
        return tl::make_unexpected(Failure(
            XdsErrorClass::kNotEligible, XdsTargetState::kUntouched,
            "hipGetDeviceCount"));
    }
    std::vector<DeviceXdsState> device_states(
        static_cast<size_t>(device_count), DeviceXdsState::kAvailable);
    auto backend = std::make_shared<HyFileAcceleratorFileIo>(
        library, api, std::move(device_states));
    backend->MarkDriverOpened();
    return std::static_pointer_cast<AcceleratorFileIo>(std::move(backend));
}

}  // namespace mooncake
