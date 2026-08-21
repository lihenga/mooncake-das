#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cerrno>
#include <span>
#include <string>
#include <vector>

#include "types.h"

namespace mooncake {

struct FileInfo {
    std::string name;
    size_t size;
};

/**
 * @brief Abstract interface for distributed filesystem adapters.
 *
 * Encapsulates file-level I/O differences across DFS implementations
 * (3FS, CephFS, JuiceFS, etc.). DistributedStorageBackend depends only
 * on this interface and is unaware of the concrete DFS.
 */
class FileSystemAdapter {
   public:
    virtual ~FileSystemAdapter() = default;

    // === File I/O ===

    virtual tl::expected<size_t, ErrorCode> WriteFile(
        const std::string& path, std::span<const char> data) = 0;

    // Read file into a pre-allocated buffer (zero-copy into Slice.ptr)
    virtual tl::expected<size_t, ErrorCode> ReadFile(const std::string& path,
                                                     void* buf, size_t len) = 0;

    virtual tl::expected<size_t, ErrorCode> VectorWriteFile(
        const std::string& path, const iovec* iov, int iovcnt,
        off_t offset) = 0;

    virtual tl::expected<size_t, ErrorCode> VectorReadFile(
        const std::string& path, const iovec* iov, int iovcnt,
        off_t offset) = 0;

    // === File management ===

    virtual tl::expected<void, ErrorCode> DeleteFile(
        const std::string& path) = 0;

    virtual tl::expected<bool, ErrorCode> FileExists(
        const std::string& path) = 0;

    virtual tl::expected<std::vector<std::string>, ErrorCode> ListFiles(
        const std::string& dir) = 0;

    virtual tl::expected<size_t, ErrorCode> GetFileSize(
        const std::string& path) {
        struct stat st;
        if (::stat(path.c_str(), &st) != 0) {
            if (errno == ENOENT) {
                return tl::make_unexpected(ErrorCode::FILE_NOT_FOUND);
            }
            return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
        }
        return static_cast<size_t>(st.st_size);
    }

    // === Batch operations (default implementations, adapters may override) ===

    virtual tl::expected<void, ErrorCode> DeleteFiles(
        const std::vector<std::string>& paths) {
        for (const auto& path : paths) {
            auto result = DeleteFile(path);
            if (!result) return result;
        }
        return {};
    }

    virtual tl::expected<std::vector<FileInfo>, ErrorCode> ListFilesWithInfo(
        const std::string& dir) {
        auto files = ListFiles(dir);
        if (!files) return tl::make_unexpected(files.error());

        std::vector<FileInfo> result;
        result.reserve(files->size());
        for (const auto& name : *files) {
            std::string full_path = dir + "/" + name;
            auto size = GetFileSize(full_path);
            if (size) {
                result.push_back({name, *size});
            } else if (size.error() != ErrorCode::FILE_NOT_FOUND) {
                return tl::make_unexpected(size.error());
            }
        }
        return result;
    }

    // === fd-based I/O for shard-offset DFS mode ===

    virtual tl::expected<int, ErrorCode> OpenFile(const std::string& /*path*/) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }

    virtual tl::expected<void, ErrorCode> CloseFile(int /*fd*/) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }

    virtual tl::expected<void, ErrorCode> PreallocateFile(
        const std::string& /*path*/, uint64_t /*size*/) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }

    virtual tl::expected<size_t, ErrorCode> WriteAt(int /*fd*/,
                                                    const iovec* /*iov*/,
                                                    int /*iovcnt*/,
                                                    int64_t /*offset*/) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }

    virtual tl::expected<size_t, ErrorCode> ReadAt(int /*fd*/, iovec* /*iov*/,
                                                   int /*iovcnt*/,
                                                   int64_t /*offset*/) {
        return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
    }

    // === Durability primitives ===
    //
    // Used to publish metadata sidecars atomically: write a temp file, fsync
    // it, rename it over the live name, then fsync the directory so the rename
    // itself is durable. Defaults are POSIX implementations, which also hold
    // for FUSE-mounted distributed filesystems; adapters with a native
    // metadata path may override them.

    virtual tl::expected<void, ErrorCode> SyncFile(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return tl::make_unexpected(errno == ENOENT
                                           ? ErrorCode::FILE_NOT_FOUND
                                           : ErrorCode::FILE_OPEN_FAIL);
        }
        const int rc = ::fsync(fd);
        const int saved_errno = errno;
        ::close(fd);
        if (rc != 0) {
            errno = saved_errno;
            return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
        }
        return {};
    }

    virtual tl::expected<void, ErrorCode> SyncDirectory(
        const std::string& dir) {
        const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd < 0) {
            // Some filesystems refuse to open directories for fsync. Treat
            // that as "nothing further to flush" rather than a hard failure.
            if (errno == EACCES || errno == EPERM || errno == EINVAL) {
                return {};
            }
            return tl::make_unexpected(errno == ENOENT
                                           ? ErrorCode::FILE_NOT_FOUND
                                           : ErrorCode::FILE_OPEN_FAIL);
        }
        const int rc = ::fsync(fd);
        const int saved_errno = errno;
        ::close(fd);
        if (rc != 0 && saved_errno != EINVAL) {
            errno = saved_errno;
            return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
        }
        return {};
    }

    virtual tl::expected<void, ErrorCode> RenameFile(const std::string& from,
                                                     const std::string& to) {
        if (::rename(from.c_str(), to.c_str()) != 0) {
            return tl::make_unexpected(errno == ENOENT
                                           ? ErrorCode::FILE_NOT_FOUND
                                           : ErrorCode::FILE_WRITE_FAIL);
        }
        return {};
    }

    // === Append-only metadata log primitives ===
    //
    // The bucket allocator records each metadata delta as a single appended
    // record instead of rewriting the whole sidecar, so the hot path costs one
    // append plus one data sync and never renames. These calls operate on a
    // long-lived fd opened once per bucket and kept open until the bucket goes
    // away.

    /**
     * @brief Open `path` for appending, creating it when absent.
     *
     * Every write through the returned fd lands at the current end of file,
     * which is what makes concurrent appends of whole records safe.
     */
    virtual tl::expected<int, ErrorCode> OpenAppendFile(
        const std::string& path) {
        const int fd = ::open(path.c_str(),
                              O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) {
            return tl::make_unexpected(errno == ENOENT
                                           ? ErrorCode::FILE_NOT_FOUND
                                           : ErrorCode::FILE_OPEN_FAIL);
        }
        return fd;
    }

    /**
     * @brief Append `data` to the file behind `fd`.
     *
     * Loops over short writes so a record is never left half-written by a
     * partial `write(2)`. Returns the number of bytes appended.
     */
    virtual tl::expected<size_t, ErrorCode> AppendData(
        int fd, std::span<const char> data) {
        size_t written = 0;
        while (written < data.size()) {
            const ssize_t rc =
                ::write(fd, data.data() + written, data.size() - written);
            if (rc < 0) {
                if (errno == EINTR) continue;
                return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
            }
            if (rc == 0) {
                return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
            }
            written += static_cast<size_t>(rc);
        }
        return written;
    }

    /**
     * @brief Flush the data written through `fd` to stable storage.
     *
     * `fdatasync` is enough here: the log file already exists and only its
     * contents and size change, both of which fdatasync covers.
     */
    virtual tl::expected<void, ErrorCode> SyncFileData(int fd) {
        while (::fdatasync(fd) != 0) {
            if (errno == EINTR) continue;
            // Some filesystems only implement full fsync.
            if (errno == EINVAL || errno == ENOSYS) {
                if (::fsync(fd) == 0) return {};
            }
            return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
        }
        return {};
    }

    /**
     * @brief Resize the file behind `fd` without closing it.
     *
     * Used to clear a metadata log after its contents were folded into a fresh
     * snapshot, so the fd stays valid across compactions.
     */
    virtual tl::expected<void, ErrorCode> TruncateFile(int fd, uint64_t size) {
        while (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
            if (errno == EINTR) continue;
            return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
        }
        return {};
    }

    // === Lifecycle ===

    virtual tl::expected<void, ErrorCode> Init(
        const std::string& mount_path) = 0;

    virtual tl::expected<void, ErrorCode> Shutdown() = 0;

    // === Identity ===

    virtual const char* GetName() const = 0;
};

}  // namespace mooncake
