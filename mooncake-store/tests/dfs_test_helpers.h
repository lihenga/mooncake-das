#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "storage/distributed/dfs_root_directory.h"

namespace mooncake::testing {

// Allow exactly the filesystem backing the test root, including bind mounts.
inline std::string GetDfsTestFilesystemType(const std::string& root) {
    DfsRootDirectory directory(root);
    if (directory.fd >= 0) {
        std::ifstream fdinfo("/proc/self/fdinfo/" +
                             std::to_string(directory.fd));
        std::string line;
        uint64_t mount_id = 0;
        while (std::getline(fdinfo, line)) {
            if (line.starts_with("mnt_id:")) {
                std::istringstream field(line.substr(7));
                field >> mount_id;
                break;
            }
        }
        std::ifstream mountinfo("/proc/self/mountinfo");
        while (mount_id != 0 && std::getline(mountinfo, line)) {
            std::istringstream row(line);
            uint64_t id = 0;
            if (!(row >> id) || id != mount_id) continue;
            std::string field;
            while (row >> field) {
                if (field == "-") {
                    std::string type;
                    if (row >> type) return type;
                    break;
                }
            }
        }
    }
    throw std::runtime_error("Cannot identify DFS test filesystem for " + root);
}

}  // namespace mooncake::testing
