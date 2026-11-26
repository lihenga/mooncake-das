#pragma once

#include <glog/logging.h>

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <memory>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

namespace mooncake::dfs_probe_gc {

// Configuration uses whole seconds and must be strictly greater than a minute.
inline constexpr int kMinimumDurationSeconds = 61;

inline int NormalizeDurationSeconds(const char* name, int configured) {
    if (configured < kMinimumDurationSeconds) {
        LOG(WARNING) << "DFS write-probe cleanup configuration below minimum, "
                     << "config=" << name << ", configured_seconds=" << configured
                     << ", effective_seconds=" << kMinimumDurationSeconds;
        return kMinimumDurationSeconds;
    }
    return configured;
}

// UuidToString emits two canonical decimal uint64_t values separated by '-'.
// Do not use StringToUuid here: it accepts partially parsed numeric strings.
inline bool IsProbeFileName(std::string_view name) {
    constexpr std::string_view prefix = ".mooncake_dfs_health_probe_";
    if (!name.starts_with(prefix)) return false;
    name.remove_prefix(prefix.size());
    const auto consume_number = [](std::string_view number, bool nonzero) {
        if (number.empty() || (number.size() > 1 && number.front() == '0')) {
            return false;
        }
        uint64_t value = 0;
        const auto result = std::from_chars(
            number.data(), number.data() + number.size(), value);
        return result.ec == std::errc{} &&
               result.ptr == number.data() + number.size() &&
               (!nonzero || value != 0);
    };
    const auto dash = name.find('-');
    const auto underscore = name.find('_');
    if (dash == std::string_view::npos ||
        underscore == std::string_view::npos || dash >= underscore) {
        return false;
    }
    return consume_number(name.substr(0, dash), false) &&
           consume_number(name.substr(dash + 1, underscore - dash - 1), false) &&
           consume_number(name.substr(underscore + 1), true);
}

inline bool IsExpired(const struct stat& info, std::chrono::seconds ttl) {
    // Compare the stat timestamp directly: converting a far-future mtime to a
    // nanosecond system_clock time_point can overflow and look expired.
    const auto cutoff = std::chrono::system_clock::now() - ttl;
    const auto seconds = std::chrono::floor<std::chrono::seconds>(
        cutoff.time_since_epoch());
    const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
        cutoff.time_since_epoch() - seconds);
    return S_ISREG(info.st_mode) &&
           (info.st_mtim.tv_sec < seconds.count() ||
            (info.st_mtim.tv_sec == seconds.count() &&
             info.st_mtim.tv_nsec < nanoseconds.count()));
}

inline void LogFailure(const std::string& path, const char* operation,
                       int error) {
    LOG(WARNING) << "DFS write-probe cleanup failed, file=" << path
                 << ", operation=" << operation << ", errno=" << error
                 << ", error=" << std::strerror(error);
}

enum class DeleteResult { kSkipped, kDeleted, kAlreadyAbsent, kFailed };

// Re-stat at deletion time instead of trusting the scan's cached mtime/type.
// All operations are relative to the opened directory and never follow links.
inline DeleteResult DeleteIfExpired(int directory_fd, const std::string& root,
                                    const std::string& name,
                                    std::chrono::seconds ttl) {
    if (!IsProbeFileName(name)) return DeleteResult::kSkipped;
    const std::string path = root + "/" + name;
    struct stat current {};
    if (::fstatat(directory_fd, name.c_str(), &current,
                  AT_SYMLINK_NOFOLLOW) != 0) {
        const int error = errno;
        if (error == ENOENT) {
            return DeleteResult::kAlreadyAbsent;
        }
        LogFailure(path, "restat", error);
        return DeleteResult::kFailed;
    }
    if (!IsExpired(current, ttl)) return DeleteResult::kSkipped;
    const int result = ::unlinkat(directory_fd, name.c_str(), 0);
    const int error = result == 0 ? 0 : errno;
    if (result == 0) return DeleteResult::kDeleted;
    if (error == ENOENT) return DeleteResult::kAlreadyAbsent;
    LogFailure(path, "unlink", error);
    return DeleteResult::kFailed;
}

// Return the number of files actually unlinked; an already absent file is not
// an error, but does not increment the deletion count.
inline size_t Cleanup(const std::string& root, std::chrono::seconds ttl) {
    // Allow deployments whose configured DFS root is a directory symlink.
    // Individual entries are still inspected with AT_SYMLINK_NOFOLLOW.
    const int fd = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        LogFailure(root, "open directory", errno);
        return 0;
    }
    DIR* opened = ::fdopendir(fd);
    if (!opened) {
        const int error = errno;
        ::close(fd);
        LogFailure(root, "fdopendir", error);
        return 0;
    }
    const auto close_directory = [](DIR* directory) { ::closedir(directory); };
    std::unique_ptr<DIR, decltype(close_directory)> directory(opened,
                                                             close_directory);
    size_t deleted_count = 0;
    while (true) {
        errno = 0;
        const auto* entry = ::readdir(directory.get());
        if (!entry) {
            if (errno != 0) LogFailure(root, "readdir", errno);
            break;
        }
        const std::string name(entry->d_name);
        if (!IsProbeFileName(name)) continue;
        struct stat info {};
        if (::fstatat(fd, name.c_str(), &info, AT_SYMLINK_NOFOLLOW) != 0) {
            const int error = errno;
            if (error != ENOENT) {
                LogFailure(root + "/" + name, "stat", error);
            }
            continue;
        }
        if (IsExpired(info, ttl) &&
            DeleteIfExpired(fd, root, name, ttl) == DeleteResult::kDeleted) {
            ++deleted_count;
        }
    }
    LOG(INFO) << "DFS write-probe cleanup finished, deleted_count="
              << deleted_count;
    return deleted_count;
}

}  // namespace mooncake::dfs_probe_gc
