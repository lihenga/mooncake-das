#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>

namespace mooncake {

inline std::string ReadDfsFilesystemType(int root_fd) {
    struct stat st{};
    if (::fstat(root_fd, &st) != 0 || !S_ISDIR(st.st_mode)) return {};
    // Identify the opened mount, not whatever the pathname resolves to now.
    std::ifstream fdinfo("/proc/self/fdinfo/" + std::to_string(root_fd));
    std::string line;
    std::string mount_id;
    while (std::getline(fdinfo, line)) {
        std::istringstream row(line);
        std::string field;
        if (row >> field && field == "mnt_id:") {
            row >> mount_id;
            break;
        }
    }
    if (mount_id.empty()) return {};
    std::ifstream mountinfo("/proc/self/mountinfo");
    while (std::getline(mountinfo, line)) {
        std::istringstream row(line);
        std::string field;
        if (!(row >> field) || field != mount_id) continue;
        while (row >> field) {
            if (field == "-") {
                return row >> field ? field : std::string{};
            }
        }
    }
    return {};
}

inline std::string ReadDfsFilesystemType(const std::string& root) {
    const int fd = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return {};
    const auto type = ReadDfsFilesystemType(fd);
    ::close(fd);
    return type;
}

}  // namespace mooncake
