#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <string>

namespace mooncake {

// Keeps pathname-based adapter operations on the directory that was verified,
// even if the configured absolute path is renamed, unmounted or covered.
struct DfsRootDirectory {
    explicit DfsRootDirectory(const std::string& path)
        : fd(::open(path.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC)) {}
    ~DfsRootDirectory() {
        if (fd >= 0) ::close(fd);
    }
    DfsRootDirectory(const DfsRootDirectory&) = delete;
    DfsRootDirectory& operator=(const DfsRootDirectory&) = delete;

    std::string PinnedPath() const {
        return "/proc/self/fd/" + std::to_string(fd);
    }

    bool SameDirectory(const DfsRootDirectory& other) const {
        return mount_id == other.mount_id && inode == other.inode;
    }

    const int fd;
    uint64_t mount_id = 0;
    uint64_t inode = 0;
    std::string fs_type;
    std::string mount_point;
    std::string source;
};

}  // namespace mooncake
