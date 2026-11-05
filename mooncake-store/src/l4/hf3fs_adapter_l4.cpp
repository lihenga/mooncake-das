// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "storage/distributed/hf3fs_adapter.h"

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>

#include <glog/logging.h>

#include "hf3fs/hf3fs.h"

namespace mooncake {

tl::expected<size_t, ErrorCode> Hf3fsAdapter::OverwriteFile(
    const std::string& path, std::span<const char> data) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);

    if (hf3fs_reg_fd(fd, 0) > 0) {
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }
    auto* resource = resource_manager_->getThreadResource();
    if (!resource || !resource->initialized) {
        hf3fs_dereg_fd(fd);
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }

    auto& threefs_iov = resource->iov_;
    auto& ior_write = resource->ior_write_;
    size_t total_written = 0;
    while (total_written < data.size()) {
        const size_t chunk =
            std::min(data.size() - total_written, resource->config_.iov_size);
        memcpy(threefs_iov.base, data.data() + total_written, chunk);
        int ret =
            hf3fs_prep_io(&ior_write, &threefs_iov, false, threefs_iov.base, fd,
                          static_cast<off_t>(total_written), chunk, nullptr);
        if (ret < 0) break;
        ret = hf3fs_submit_ios(&ior_write);
        if (ret < 0) break;
        struct hf3fs_cqe cqe;
        ret = hf3fs_wait_for_ios(&ior_write, &cqe, 1, 1, nullptr);
        if (ret < 0 || cqe.result < 0) break;
        const size_t bytes_written = cqe.result;
        total_written += bytes_written;
        if (bytes_written < chunk) break;
    }
    if (total_written == data.size() &&
        ::ftruncate(fd, static_cast<off_t>(data.size())) != 0) {
        total_written = 0;
    }
    hf3fs_dereg_fd(fd);
    const int close_result = close(fd);
    if (close_result != 0 || total_written != data.size()) {
        return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
    }
    return total_written;
}

tl::expected<int, ErrorCode> Hf3fsAdapter::OpenFileReadOnly(
    const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    if (hf3fs_reg_fd(fd, 0) > 0) {
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }
    return fd;
}

tl::expected<int, ErrorCode> Hf3fsAdapter::OpenFileReadWriteExisting(
    const std::string& path) {
    int fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return tl::make_unexpected(errno == ENOENT ? ErrorCode::FILE_NOT_FOUND
                                                   : ErrorCode::FILE_OPEN_FAIL);
    }
    if (hf3fs_reg_fd(fd, 0) > 0) {
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }
    return fd;
}

tl::expected<int, ErrorCode> Hf3fsAdapter::OpenFileDirectStrict(
    const std::string& path) {
#ifdef O_DIRECT
    const int fd =
        open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW);
#else
    errno = EOPNOTSUPP;
    const int fd = -1;
#endif
    if (fd < 0) {
        return tl::make_unexpected(errno == ENOENT ? ErrorCode::FILE_NOT_FOUND
                                                   : ErrorCode::FILE_OPEN_FAIL);
    }
    if (hf3fs_reg_fd(fd, 0) > 0) {
        close(fd);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }
    return fd;
}

}  // namespace mooncake
