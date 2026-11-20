#pragma once

#include <sys/stat.h>
#include <sys/sysmacros.h>

#include <fstream>
#include <sstream>
#include <string>

namespace mooncake {

// Bind mounts of the same filesystem share its device and filesystem type.
inline std::string ReadDfsFilesystemType(const std::string& root) {
    struct stat st{};
    if (::stat(root.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return {};
    const std::string device = std::to_string(major(st.st_dev)) + ":" +
                               std::to_string(minor(st.st_dev));
    std::ifstream mountinfo("/proc/self/mountinfo");
    std::string line;
    while (std::getline(mountinfo, line)) {
        std::istringstream row(line);
        std::string field;
        std::string mounted_device;
        if (!(row >> field >> field >> mounted_device) ||
            mounted_device != device) {
            continue;
        }
        while (row >> field) {
            if (field == "-") {
                return row >> field ? field : std::string{};
            }
        }
    }
    return {};
}

}  // namespace mooncake
