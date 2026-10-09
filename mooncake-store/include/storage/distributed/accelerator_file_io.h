#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <ylt/util/tl/expected.hpp>

namespace mooncake {

enum class XdsMode { kPosix, kAuto, kRequired };

enum class XdsErrorClass {
    kNotEligible,
    kContractViolation,
    kTransientIo,
    kBackendFatal,
};

enum class XdsTargetState {
    kUntouched,
    kStoppedSafeToOverwrite,
    kUnknownMayStillBeWritten,
};

struct XdsError {
    XdsErrorClass error_class = XdsErrorClass::kBackendFatal;
    XdsTargetState target_state = XdsTargetState::kUntouched;
    int64_t raw_code = 0;
    std::string operation;
};

enum class DeviceXdsState { kUntested, kAvailable, kUnavailable, kFatal };

struct DirectIoCapabilities {
    bool available = false;
    bool supports_distributed_fs = false;
    bool supports_async = false;
    bool supports_scatter_gather = false;
    uint64_t file_offset_alignment = 4096;
    uint64_t device_address_alignment = 4096;
    uint64_t length_alignment = 4096;
};

struct XdsFileIdentity {
    std::string canonical_path;
    uint64_t device = 0;
    uint64_t inode = 0;
};

struct XdsFileRegistration {
    uint64_t id = 0;
    int fd = -1;
};

struct XdsBufferRegistration {
    uint64_t id = 0;
    void* base = nullptr;
    size_t length = 0;
    int32_t device_id = -1;
};

struct DirectReadOp {
    XdsFileRegistration file;
    XdsBufferRegistration buffer;
    uint64_t file_offset = 0;
    uint64_t device_offset = 0;
    uint64_t length = 0;
    int32_t device_id = -1;
};

struct DirectReadCompletion {
    uint64_t bytes = 0;
};

class AcceleratorFileIo {
   public:
    virtual ~AcceleratorFileIo() = default;
    virtual DirectIoCapabilities Capabilities() const = 0;
    virtual DeviceXdsState DeviceState(int32_t device_id) const = 0;
    virtual tl::expected<XdsFileRegistration, XdsError> RegisterFile(
        int fd, const XdsFileIdentity& identity) = 0;
    virtual tl::expected<XdsBufferRegistration, XdsError> RegisterBuffer(
        void* base, size_t length, int32_t device_id) = 0;
    virtual tl::expected<DirectReadCompletion, XdsError> Read(
        const DirectReadOp& op) = 0;
    virtual tl::expected<void, XdsError> DeregisterFile(
        XdsFileRegistration registration) = 0;
    virtual tl::expected<void, XdsError> DeregisterBuffer(
        XdsBufferRegistration registration) = 0;
};

}  // namespace mooncake
