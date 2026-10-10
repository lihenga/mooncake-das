#include "storage/distributed/xds/hyfile_backend.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <glog/logging.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "storage/distributed/xds/hyfile_abi.h"

namespace mooncake {
namespace {

namespace abi = xds::hyfile_abi;

constexpr uint64_t kDirectIoAlignment = 4096;

XdsError Failure(XdsErrorClass error_class, XdsTargetState target_state,
                 std::string operation, int64_t raw_code = 0) {
    return XdsError{error_class, target_state, raw_code, std::move(operation)};
}

XdsError ForkFailure(std::string operation) {
    return Failure(XdsErrorClass::kBackendFatal,
                   XdsTargetState::kUntouched, std::move(operation));
}

bool IsSuccess(abi::Error result) { return result.err == abi::kSuccess; }

int64_t RawCode(abi::Error result) {
    if (result.err != abi::kSuccess) {
        return static_cast<int64_t>(result.err);
    }
    return static_cast<int64_t>(result.hip_error);
}

XdsErrorClass ClassifyVendorError(int error) {
    switch (error) {
        case 5007:  // platform not supported
        case 5008:  // I/O not supported on this file
        case 5009:  // device not supported
        case 5034:  // I/O disabled
            return XdsErrorClass::kNotEligible;

        case 5012:  // invalid device pointer
        case 5013:  // invalid memory type
        case 5014:  // pointer range error
        case 5016:  // invalid mapping size
        case 5017:  // invalid mapping range
        case 5018:  // invalid file type
        case 5019:  // invalid file open flags
        case 5020:  // O_DIRECT not set
        case 5022:  // invalid value
        case 5023:  // memory already registered
        case 5024:  // memory not registered
        case 5027:  // handle not registered
        case 5028:  // handle already registered
            return XdsErrorClass::kContractViolation;

        case 5001:  // driver not initialized
        case 5002:  // invalid driver properties
        case 5003:  // unsupported driver limit
        case 5004:  // driver version mismatch
        case 5005:  // driver version read error
        case 5006:  // driver closing
        case 5010:  // filesystem driver error
        case 5011:  // HIP runtime/driver error
        case 5015:  // context mismatch
        case 5029:  // device not found
        case 5030:  // internal error
        case 5033:  // filesystem setup error
        case 5035:  // batch submit failure
        case 5036:  // GPU memory pinning failure
            return XdsErrorClass::kBackendFatal;

        default:
            return XdsErrorClass::kTransientIo;
    }
}

XdsError VendorFailure(abi::Error result, XdsTargetState target_state,
                       std::string operation) {
    return Failure(ClassifyVendorError(result.err), target_state,
                   std::move(operation), RawCode(result));
}

bool IsApiComplete(const abi::Api& api) {
    return api.driver_open && api.driver_close && api.handle_register &&
           api.handle_deregister && api.buffer_register &&
           api.buffer_deregister && api.read;
}

template <typename Function>
bool LoadSymbol(void* library, const char* name, Function* function) {
    ::dlerror();
    *function = reinterpret_cast<Function>(::dlsym(library, name));
    return *function != nullptr && ::dlerror() == nullptr;
}

struct HipApi {
    int (*get_device_count)(int*) = nullptr;
    int (*set_device)(int) = nullptr;
    int (*device_synchronize)() = nullptr;
    int success = 0;
};

int HipGetDeviceCount(int* count) {
    return static_cast<int>(::hipGetDeviceCount(count));
}

int HipSetDevice(int device_id) {
    return static_cast<int>(::hipSetDevice(device_id));
}

int HipDeviceSynchronize() {
    return static_cast<int>(::hipDeviceSynchronize());
}

HipApi ProductionHipApi() {
    return HipApi{HipGetDeviceCount, HipSetDevice, HipDeviceSynchronize,
                  static_cast<int>(hipSuccess)};
}

bool IsHipApiComplete(const HipApi& api) {
    return api.get_device_count && api.set_device && api.device_synchronize;
}

std::mutex process_runtime_mutex;

class HyFileProcessRuntime {
   public:
    HyFileProcessRuntime(std::string identity, void* library,
                         bool owns_library, abi::Api api, HipApi hip_api)
        : identity_(std::move(identity)),
          library_(library),
          owns_library_(owns_library),
          api_(api),
          hip_api_(hip_api),
          owner_pid_(::getpid()) {}

    ~HyFileProcessRuntime() {
        // Serialize teardown against creation of the next process-wide runtime.
        std::lock_guard<std::mutex> process_lock(process_runtime_mutex);
        if (!IsCurrentProcess()) {
            // Vendor state inherited across fork is not safe to close or reuse.
            // The child process will reclaim the mappings when it exits.
            return;
        }
        std::lock_guard<std::mutex> api_lock(api_mutex_);
        if (driver_opened_) {
            const abi::Error result = api_.driver_close();
            if (!IsSuccess(result)) {
                LOG(ERROR) << "hipFileDriverClose_v2 failed during shutdown, "
                           << "code=" << RawCode(result);
            }
            driver_opened_ = false;
        }
        if (owns_library_ && library_) {
            if (::dlclose(library_) != 0) {
                LOG(ERROR) << "dlclose(libhyfile) failed during shutdown";
            }
            library_ = nullptr;
        }
    }

    HyFileProcessRuntime(const HyFileProcessRuntime&) = delete;
    HyFileProcessRuntime& operator=(const HyFileProcessRuntime&) = delete;

    bool IsCurrentProcess() const { return owner_pid_ == ::getpid(); }
    const std::string& Identity() const { return identity_; }
    const abi::Api& Api() const { return api_; }
    const HipApi& Hip() const { return hip_api_; }
    std::mutex& ApiMutex() { return api_mutex_; }
    void MarkDriverOpened() { driver_opened_ = true; }

   private:
    std::string identity_;
    void* library_ = nullptr;
    bool owns_library_ = false;
    abi::Api api_;
    HipApi hip_api_;
    pid_t owner_pid_ = 0;
    bool driver_opened_ = false;
    std::mutex api_mutex_;
};

// The driver is process-global and remains open until process shutdown. This
// also prevents a new hipFileDriverOpen from racing the previous close when
// Store clients are created and destroyed concurrently.
std::shared_ptr<HyFileProcessRuntime> process_runtime;

struct RuntimeSpec {
    void* library = nullptr;
    bool owns_library = false;
    abi::Api api;
    HipApi hip_api;
};

using RuntimeLoader =
    std::function<tl::expected<RuntimeSpec, XdsError>()>;

tl::expected<std::shared_ptr<HyFileProcessRuntime>, XdsError>
AcquireProcessRuntime(const std::string& identity,
                      const RuntimeLoader& loader) {
    std::lock_guard<std::mutex> process_lock(process_runtime_mutex);
    if (process_runtime) {
        auto existing = process_runtime;
        if (!existing->IsCurrentProcess()) {
            return tl::make_unexpected(
                ForkFailure("hyfile-runtime-inherited-after-fork"));
        }
        if (existing->Identity() != identity) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched,
                "hyfile-runtime-library-mismatch"));
        }
        return existing;
    }

    auto loaded = loader();
    if (!loaded) return tl::make_unexpected(loaded.error());
    RuntimeSpec spec = std::move(*loaded);
    if (!IsApiComplete(spec.api) || !IsHipApiComplete(spec.hip_api)) {
        if (spec.owns_library && spec.library) (void)::dlclose(spec.library);
        return tl::make_unexpected(Failure(
            XdsErrorClass::kNotEligible, XdsTargetState::kUntouched,
            "hyfile-required-symbol-missing"));
    }

    const abi::Error open_result = spec.api.driver_open();
    if (!IsSuccess(open_result)) {
        if (spec.owns_library && spec.library) (void)::dlclose(spec.library);
        return tl::make_unexpected(VendorFailure(
            open_result, XdsTargetState::kUntouched, "hipFileDriverOpen"));
    }

    auto runtime = std::make_shared<HyFileProcessRuntime>(
        identity, spec.library, spec.owns_library, spec.api, spec.hip_api);
    runtime->MarkDriverOpened();
    process_runtime = runtime;
    return runtime;
}

tl::expected<RuntimeSpec, XdsError> LoadProductionRuntime(
    const std::string& library_path) {
    void* library = ::dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        return tl::make_unexpected(Failure(
            XdsErrorClass::kNotEligible, XdsTargetState::kUntouched,
            "dlopen(libhyfile)"));
    }

    abi::Api api;
    auto require = [&](const char* name, auto* function)
        -> tl::expected<void, XdsError> {
        if (LoadSymbol(library, name, function)) return {};
        (void)::dlclose(library);
        library = nullptr;
        return tl::make_unexpected(Failure(
            XdsErrorClass::kNotEligible, XdsTargetState::kUntouched,
            std::string("dlsym(") + name + ")"));
    };

    auto result = require("hipFileDriverOpen", &api.driver_open);
    if (!result) return tl::make_unexpected(result.error());
    result = require("hipFileDriverClose_v2", &api.driver_close);
    if (!result) return tl::make_unexpected(result.error());
    result = require("hipFileHandleRegister", &api.handle_register);
    if (!result) return tl::make_unexpected(result.error());
    result = require("hipFileHandleDeregister", &api.handle_deregister);
    if (!result) return tl::make_unexpected(result.error());
    result = require("hipFileBufRegister", &api.buffer_register);
    if (!result) return tl::make_unexpected(result.error());
    result = require("hipFileBufDeregister", &api.buffer_deregister);
    if (!result) return tl::make_unexpected(result.error());
    result = require("hipFileRead", &api.read);
    if (!result) return tl::make_unexpected(result.error());

    return RuntimeSpec{library, true, api, ProductionHipApi()};
}

class HyFileAcceleratorFileIo final : public AcceleratorFileIo {
   public:
    HyFileAcceleratorFileIo(
        std::shared_ptr<HyFileProcessRuntime> runtime,
        std::vector<DeviceXdsState> device_states)
        : runtime_(std::move(runtime)),
          device_states_(std::move(device_states)) {}

    ~HyFileAcceleratorFileIo() override {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        if (!runtime_->IsCurrentProcess()) {
            buffers_.clear();
            files_.clear();
            return;
        }

        std::lock_guard<std::mutex> api_lock(runtime_->ApiMutex());
        const auto& hip = runtime_->Hip();
        const auto& api = runtime_->Api();
        for (const auto& [_, buffer] : buffers_) {
            const int set_result = hip.set_device(buffer.device_id);
            const int sync_result = set_result == hip.success
                                        ? hip.device_synchronize()
                                        : set_result;
            if (set_result != hip.success || sync_result != hip.success) {
                LOG(ERROR) << "Unable to synchronize xDS buffer during "
                              "shutdown, device="
                           << buffer.device_id << ", code=" << sync_result;
                continue;
            }
            const abi::Error result = api.buffer_deregister(buffer.base);
            if (!IsSuccess(result)) {
                LOG(ERROR) << "hipFileBufDeregister failed during shutdown, "
                           << "code=" << RawCode(result);
            }
        }
        for (const auto& [_, file] : files_) {
            api.handle_deregister(file.handle);
        }
        buffers_.clear();
        files_.clear();
    }

    HyFileAcceleratorFileIo(const HyFileAcceleratorFileIo&) = delete;
    HyFileAcceleratorFileIo& operator=(const HyFileAcceleratorFileIo&) =
        delete;

    DirectIoCapabilities Capabilities() const override {
        DirectIoCapabilities capabilities;
        capabilities.available = runtime_->IsCurrentProcess();
        capabilities.supports_distributed_fs = capabilities.available;
        capabilities.file_offset_alignment = kDirectIoAlignment;
        capabilities.device_address_alignment = kDirectIoAlignment;
        capabilities.length_alignment = kDirectIoAlignment;
        return capabilities;
    }

    DeviceXdsState DeviceState(int32_t device_id) const override {
        if (!runtime_->IsCurrentProcess()) return DeviceXdsState::kFatal;
        if (device_id < 0 ||
            static_cast<size_t>(device_id) >= device_states_.size()) {
            return DeviceXdsState::kUnavailable;
        }
        return device_states_[device_id];
    }

    tl::expected<XdsFileRegistration, XdsError> RegisterFile(
        int fd, const XdsFileIdentity& identity) override {
        if (!runtime_->IsCurrentProcess()) {
            return tl::make_unexpected(
                ForkFailure("hipFileHandleRegister:fork"));
        }
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
        if (::fstat(fd, &file_info) != 0) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kTransientIo, XdsTargetState::kUntouched,
                "hipFileHandleRegister:fstat", errno));
        }
        if (static_cast<uint64_t>(file_info.st_dev) != identity.device ||
            static_cast<uint64_t>(file_info.st_ino) != identity.inode) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched,
                "hipFileHandleRegister:file-identity"));
        }

        abi::FileDescriptor descriptor{};
        descriptor.type = abi::kFileHandleTypeOpaqueFd;
        descriptor.handle.fd = fd;
        descriptor.fs_ops = nullptr;
        abi::FileHandle handle = nullptr;

        std::lock_guard<std::mutex> state_lock(state_mutex_);
        std::lock_guard<std::mutex> api_lock(runtime_->ApiMutex());
        const abi::Error result =
            runtime_->Api().handle_register(&handle, &descriptor);
        if (!IsSuccess(result)) {
            return tl::make_unexpected(VendorFailure(
                result, XdsTargetState::kUntouched,
                "hipFileHandleRegister"));
        }
        const uint64_t id = next_id_++;
        files_.emplace(id, FileRegistration{handle, fd});
        return XdsFileRegistration{id, fd};
    }

    tl::expected<XdsBufferRegistration, XdsError> RegisterBuffer(
        void* base, size_t length, int32_t device_id) override {
        if (!runtime_->IsCurrentProcess()) {
            return tl::make_unexpected(ForkFailure("hipFileBufRegister:fork"));
        }
        if (!base || length == 0) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched,
                "hipFileBufRegister:arguments"));
        }
        if (reinterpret_cast<uintptr_t>(base) % kDirectIoAlignment != 0 ||
            length % kDirectIoAlignment != 0 ||
            DeviceState(device_id) != DeviceXdsState::kAvailable) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kNotEligible, XdsTargetState::kUntouched,
                "hipFileBufRegister:eligibility"));
        }

        std::lock_guard<std::mutex> state_lock(state_mutex_);
        std::lock_guard<std::mutex> api_lock(runtime_->ApiMutex());
        const auto& hip = runtime_->Hip();
        const int set_device_result = hip.set_device(device_id);
        if (set_device_result != hip.success) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal, XdsTargetState::kUntouched,
                "hipSetDevice:register", set_device_result));
        }
        const abi::Error result = runtime_->Api().buffer_register(
            base, length, abi::kRdmaRegister);
        if (!IsSuccess(result)) {
            return tl::make_unexpected(VendorFailure(
                result, XdsTargetState::kUntouched,
                "hipFileBufRegister"));
        }
        const uint64_t id = next_id_++;
        const XdsBufferRegistration registration{id, base, length, device_id};
        buffers_.emplace(id, registration);
        return registration;
    }

    tl::expected<DirectReadCompletion, XdsError> Read(
        const DirectReadOp& op) override {
        if (!runtime_->IsCurrentProcess()) {
            return tl::make_unexpected(ForkFailure("hipFileRead:fork"));
        }
        std::lock_guard<std::mutex> state_lock(state_mutex_);
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
            op.length >
                static_cast<uint64_t>(std::numeric_limits<ssize_t>::max()) ||
            op.file_offset > std::numeric_limits<uint64_t>::max() -
                                 op.length) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kContractViolation,
                XdsTargetState::kUntouched, "hipFileRead:arguments"));
        }

        std::lock_guard<std::mutex> api_lock(runtime_->ApiMutex());
        const auto& hip = runtime_->Hip();
        const int set_device_result = hip.set_device(op.device_id);
        if (set_device_result != hip.success) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal, XdsTargetState::kUntouched,
                "hipSetDevice:read", set_device_result));
        }

        errno = 0;
        const ssize_t bytes = runtime_->Api().read(
            file->second.handle, buffer->second.base, op.length,
            static_cast<unsigned long long>(op.file_offset),
            static_cast<unsigned long long>(op.device_offset));
        const int saved_errno = errno;
        const int sync_result = hip.device_synchronize();
        if (sync_result != hip.success) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal,
                XdsTargetState::kUnknownMayStillBeWritten,
                "hipDeviceSynchronize:read", sync_result));
        }
        if (bytes < 0 || static_cast<uint64_t>(bytes) != op.length) {
            XdsErrorClass error_class = XdsErrorClass::kTransientIo;
            if (bytes < -abi::kOperationErrorBase &&
                bytes >= -static_cast<ssize_t>(
                             std::numeric_limits<int>::max())) {
                error_class = ClassifyVendorError(static_cast<int>(-bytes));
            }
            const int64_t raw = bytes == -1 && saved_errno != 0
                                    ? -static_cast<int64_t>(saved_errno)
                                    : static_cast<int64_t>(bytes);
            return tl::make_unexpected(Failure(
                error_class, XdsTargetState::kStoppedSafeToOverwrite,
                "hipFileRead", raw));
        }
        return DirectReadCompletion{static_cast<uint64_t>(bytes)};
    }

    tl::expected<void, XdsError> DeregisterFile(
        XdsFileRegistration registration) override {
        if (!runtime_->IsCurrentProcess()) {
            return tl::make_unexpected(
                ForkFailure("hipFileHandleDeregister:fork"));
        }
        std::lock_guard<std::mutex> state_lock(state_mutex_);
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
        std::lock_guard<std::mutex> api_lock(runtime_->ApiMutex());
        runtime_->Api().handle_deregister(file->second.handle);
        files_.erase(file);
        return {};
    }

    tl::expected<void, XdsError> DeregisterBuffer(
        XdsBufferRegistration registration) override {
        if (!runtime_->IsCurrentProcess()) {
            return tl::make_unexpected(
                ForkFailure("hipFileBufDeregister:fork"));
        }
        std::lock_guard<std::mutex> state_lock(state_mutex_);
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

        std::lock_guard<std::mutex> api_lock(runtime_->ApiMutex());
        const auto& hip = runtime_->Hip();
        const int set_device_result =
            hip.set_device(buffer->second.device_id);
        if (set_device_result != hip.success) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal,
                XdsTargetState::kUnknownMayStillBeWritten,
                "hipSetDevice:deregister", set_device_result));
        }
        const int sync_result = hip.device_synchronize();
        if (sync_result != hip.success) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal,
                XdsTargetState::kUnknownMayStillBeWritten,
                "hipDeviceSynchronize:deregister", sync_result));
        }
        const abi::Error result =
            runtime_->Api().buffer_deregister(buffer->second.base);
        if (!IsSuccess(result)) {
            return tl::make_unexpected(Failure(
                XdsErrorClass::kBackendFatal,
                XdsTargetState::kStoppedSafeToOverwrite,
                "hipFileBufDeregister", RawCode(result)));
        }
        buffers_.erase(buffer);
        return {};
    }

   private:
    struct FileRegistration {
        abi::FileHandle handle = nullptr;
        int fd = -1;
    };

    std::shared_ptr<HyFileProcessRuntime> runtime_;
    std::vector<DeviceXdsState> device_states_;
    uint64_t next_id_ = 1;
    mutable std::mutex state_mutex_;
    std::unordered_map<uint64_t, FileRegistration> files_;
    std::unordered_map<uint64_t, XdsBufferRegistration> buffers_;
};

tl::expected<std::shared_ptr<AcceleratorFileIo>, XdsError> BuildAdapter(
    const std::shared_ptr<HyFileProcessRuntime>& runtime) {
    if (!runtime->IsCurrentProcess()) {
        return tl::make_unexpected(
            ForkFailure("hyfile-runtime-inherited-after-fork"));
    }
    std::lock_guard<std::mutex> api_lock(runtime->ApiMutex());
    int device_count = 0;
    const int result = runtime->Hip().get_device_count(&device_count);
    if (result != runtime->Hip().success || device_count <= 0) {
        return tl::make_unexpected(Failure(
            XdsErrorClass::kNotEligible, XdsTargetState::kUntouched,
            "hipGetDeviceCount", result));
    }
    std::vector<DeviceXdsState> device_states(
        static_cast<size_t>(device_count), DeviceXdsState::kAvailable);
    return std::static_pointer_cast<AcceleratorFileIo>(
        std::make_shared<HyFileAcceleratorFileIo>(runtime,
                                                  std::move(device_states)));
}

}  // namespace

tl::expected<std::shared_ptr<AcceleratorFileIo>, XdsError>
CreateHyFileAcceleratorFileIo(const std::string& library_path) {
    auto runtime = AcquireProcessRuntime(
        library_path,
        [&] { return LoadProductionRuntime(library_path); });
    if (!runtime) return tl::make_unexpected(runtime.error());
    return BuildAdapter(*runtime);
}

#ifdef MOONCAKE_ENABLE_XDS_TEST_HOOKS
namespace xds::testing {

tl::expected<std::shared_ptr<AcceleratorFileIo>, XdsError>
CreateHyFileAcceleratorFileIoForTesting(
    const std::string& runtime_identity, const hyfile_abi::Api& hyfile_api,
    const HipRuntimeApi& hip_api) {
    const HipApi internal_hip_api{hip_api.get_device_count,
                                  hip_api.set_device,
                                  hip_api.device_synchronize,
                                  hip_api.success};
    auto runtime = AcquireProcessRuntime(
        runtime_identity, [&]() -> tl::expected<RuntimeSpec, XdsError> {
            return RuntimeSpec{nullptr, false, hyfile_api, internal_hip_api};
        });
    if (!runtime) return tl::make_unexpected(runtime.error());
    return BuildAdapter(*runtime);
}

void ResetHyFileProcessRuntimeForTesting() {
    std::shared_ptr<HyFileProcessRuntime> runtime;
    {
        std::lock_guard<std::mutex> process_lock(process_runtime_mutex);
        runtime = std::move(process_runtime);
    }
    runtime.reset();
}

}  // namespace xds::testing
#endif

}  // namespace mooncake
