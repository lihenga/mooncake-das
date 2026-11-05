// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "storage/distributed/posix_fs_adapter.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>

#include <glog/logging.h>

namespace mooncake::posix_detail {

// O_DIRECT alignment assumed for direct reads: offset, buffer address and
// length must all be multiples of the filesystem block size. 4096 covers the
// common 512-byte and 4K-block cases.
constexpr uint64_t kDirectIoAlignment = 4096;

// Single pread chunk cap so a large window never overflows ssize_t.
constexpr uint64_t kDirectIoMaxChunk = 1ULL << 30;

// Upper bound on pooled O_DIRECT staging buffers. Sized to cover the batch
// read fan-out so a fully busy pool is the common steady state; beyond this a
// read falls back to a transient allocation rather than blocking on a slot.
constexpr size_t kMaxDirectStagingBuffers = 128;

inline tl::expected<void, ErrorCode> ValidateIov(const iovec* iov, int iovcnt) {
    if (iovcnt < 0 || (iovcnt > 0 && iov == nullptr)) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    for (int i = 0; i < iovcnt; ++i) {
        if (!iov[i].iov_base && iov[i].iov_len > 0) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
    }
    return {};
}

inline ErrorCode ReadOpenError() {
    return errno == ENOENT ? ErrorCode::FILE_NOT_FOUND
                           : ErrorCode::FILE_OPEN_FAIL;
}

}  // namespace mooncake::posix_detail
