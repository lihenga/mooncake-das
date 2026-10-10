#pragma once

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

namespace mooncake::xds::hyfile_abi {

// Minimal ABI used by the deployed libhyfile.so. Keep this header independent
// from the optional vendor SDK: xDS is loaded at runtime through dlopen/dlsym.
struct Error {
    int err;
    int hip_error;
};

using FileHandle = void*;

union FileHandleValue {
    int fd;
    void* handle;
};

struct FileDescriptor {
    int type;
    FileHandleValue handle;
    void* fs_ops;
};

constexpr int kSuccess = 0;
constexpr int kOperationErrorBase = 5000;
constexpr int kFileHandleTypeOpaqueFd = 1;
constexpr int kRdmaRegister = 1;

using DriverOpenFn = Error (*)();
using DriverCloseFn = Error (*)();
using HandleRegisterFn = Error (*)(FileHandle*, FileDescriptor*);
using HandleDeregisterFn = void (*)(FileHandle);
using BufferRegisterFn = Error (*)(void*, size_t, int);
using BufferDeregisterFn = Error (*)(void*);
using ReadFn = ssize_t (*)(FileHandle, void*, size_t, unsigned long long,
                           unsigned long long);

struct Api {
    DriverOpenFn driver_open = nullptr;
    DriverCloseFn driver_close = nullptr;
    HandleRegisterFn handle_register = nullptr;
    HandleDeregisterFn handle_deregister = nullptr;
    BufferRegisterFn buffer_register = nullptr;
    BufferDeregisterFn buffer_deregister = nullptr;
    ReadFn read = nullptr;
};

// The embedded ABI is intentionally limited to 64-bit Linux. These checks
// have no runtime cost and prevent silently compiling a different C layout.
static_assert(sizeof(void*) == 8);
static_assert(sizeof(size_t) == 8);
static_assert(sizeof(ssize_t) == 8);
static_assert(sizeof(Error) == 8);
static_assert(alignof(Error) == 4);
static_assert(sizeof(FileHandleValue) == 8);
static_assert(alignof(FileHandleValue) == 8);
static_assert(sizeof(FileDescriptor) == 24);
static_assert(alignof(FileDescriptor) == 8);
static_assert(offsetof(FileDescriptor, type) == 0);
static_assert(offsetof(FileDescriptor, handle) == 8);
static_assert(offsetof(FileDescriptor, fs_ops) == 16);

}  // namespace mooncake::xds::hyfile_abi
