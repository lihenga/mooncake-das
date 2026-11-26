#include "storage/distributed/dfs_probe_gc.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>

namespace mooncake::test {
namespace {

constexpr char kProbe[] = ".mooncake_dfs_health_probe_123-456_1";
constexpr auto kTtl = std::chrono::seconds(7200);

class DfsProbeGcTest : public ::testing::Test {
   protected:
    void SetUp() override {
        char pattern[] = "/tmp/mooncake_probe_gc_XXXXXX";
        const char* created = ::mkdtemp(pattern);
        ASSERT_NE(created, nullptr);
        root_ = created;
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    void Create(const std::string& name, int age_seconds) {
        std::ofstream output(root_ + "/" + name);
        output << "mooncake-dfs-health-v1\n";
        output.close();
        ASSERT_TRUE(output.good());
        SetAge(name, age_seconds);
    }

    void SetAge(const std::string& name, int age_seconds) {
        const auto modified = std::time(nullptr) - age_seconds;
        const timespec times[2] = {{modified, 0}, {modified, 0}};
        ASSERT_EQ(::utimensat(AT_FDCWD, (root_ + "/" + name).c_str(), times,
                              AT_SYMLINK_NOFOLLOW),
                  0);
    }

    bool Exists(const std::string& name) const {
        return std::filesystem::exists(root_ + "/" + name);
    }

    std::string root_;
};

TEST(DfsProbeGcNameTest, AcceptsOnlyCanonicalUuidAndPositiveGeneration) {
    EXPECT_TRUE(dfs_probe_gc::IsProbeFileName(kProbe));
    EXPECT_TRUE(dfs_probe_gc::IsProbeFileName(
        ".mooncake_dfs_health_probe_0-18446744073709551615_42"));
    for (const char* suffix : {"123-456_0", "123-456_01", "01-456_1",
                               "123-456_+1", "123-456_-1", "123-456_1.tmp",
                               "123x-456_1", "123-456_1_2", "123--456_1",
                               "18446744073709551616-456_1", "123-456_",
                               "123-456_18446744073709551616"}) {
        EXPECT_FALSE(dfs_probe_gc::IsProbeFileName(
            std::string(".mooncake_dfs_health_probe_") + suffix))
            << suffix;
    }
    EXPECT_FALSE(dfs_probe_gc::IsProbeFileName(
        std::string("subdirectory/") + kProbe));
    EXPECT_FALSE(dfs_probe_gc::IsProbeFileName(".mooncake_dfs_read_probe"));
}

TEST(DfsProbeGcExpiryTest, FarFutureTimestampCannotOverflowIntoThePast) {
    struct stat info {};
    info.st_mode = S_IFREG | 0600;
    info.st_mtim.tv_sec = std::numeric_limits<time_t>::max();
    EXPECT_FALSE(dfs_probe_gc::IsExpired(info, kTtl));
}

TEST(DfsProbeGcConfigTest, BothDurationsMustExceedOneMinute) {
    for (const char* name : {
             "MOONCAKE_DFS_HEALTH_PROBE_TTL_SECONDS",
             "MOONCAKE_DFS_HEALTH_PROBE_CLEANUP_INTERVAL_SECONDS"}) {
        for (const int configured : {-100, 0, 1, 59, 60}) {
            EXPECT_EQ(dfs_probe_gc::NormalizeDurationSeconds(name, configured),
                      61);
        }
        for (const int configured : {61, 62, 3600, 7200}) {
            EXPECT_EQ(dfs_probe_gc::NormalizeDurationSeconds(name, configured),
                      configured);
        }
    }
}

TEST_F(DfsProbeGcTest, DeletesOnlyExpiredRegularMatchingFiles) {
    Create(kProbe, 7201);
    Create(".mooncake_dfs_health_probe_123-456_2", 0);
    Create(".mooncake_dfs_health_probe_123-456_3", -3600);
    Create(".mooncake_dfs_health_probe_bad_1", 14400);
    Create(".mooncake_dfs_read_probe", 14400);
    Create(".mooncake_dfs_root_identity", 14400);
    EXPECT_EQ(dfs_probe_gc::Cleanup(root_, kTtl), 1u);
    EXPECT_FALSE(Exists(kProbe));
    EXPECT_TRUE(Exists(".mooncake_dfs_health_probe_123-456_2"));
    EXPECT_TRUE(Exists(".mooncake_dfs_health_probe_123-456_3"));
    EXPECT_TRUE(Exists(".mooncake_dfs_health_probe_bad_1"));
    EXPECT_TRUE(Exists(".mooncake_dfs_read_probe"));
    EXPECT_TRUE(Exists(".mooncake_dfs_root_identity"));
    EXPECT_EQ(dfs_probe_gc::Cleanup(root_, kTtl), 0u);
}

TEST_F(DfsProbeGcTest, ReportsTotalDeletionCountAcrossMultipleFiles) {
    Create(kProbe, 14400);
    Create(".mooncake_dfs_health_probe_123-456_2", 14400);
    Create(".mooncake_dfs_health_probe_123-456_3", 0);
    EXPECT_EQ(dfs_probe_gc::Cleanup(root_, kTtl), 2u);
    EXPECT_EQ(dfs_probe_gc::Cleanup(root_, kTtl), 0u);
}

TEST_F(DfsProbeGcTest, DoesNotRecurseOrFollowSymlinks) {
    ASSERT_TRUE(std::filesystem::create_directory(root_ + "/nested"));
    Create(std::string("nested/") + kProbe, 14400);
    ASSERT_EQ(::symlink((root_ + "/nested/" + kProbe).c_str(),
                        (root_ + "/" + kProbe).c_str()),
              0);
    SetAge(kProbe, 14400);
    const std::string directory = ".mooncake_dfs_health_probe_123-456_2";
    ASSERT_TRUE(std::filesystem::create_directory(root_ + "/" + directory));
    SetAge(directory, 14400);
    const std::string fifo = ".mooncake_dfs_health_probe_123-456_3";
    ASSERT_EQ(::mkfifo((root_ + "/" + fifo).c_str(), 0600), 0);
    SetAge(fifo, 14400);
    dfs_probe_gc::Cleanup(root_, kTtl);
    EXPECT_TRUE(Exists(kProbe));
    EXPECT_TRUE(Exists(std::string("nested/") + kProbe));
    EXPECT_TRUE(Exists(directory));
    EXPECT_TRUE(Exists(fifo));
}

TEST_F(DfsProbeGcTest, AllowsSymlinkAsConfiguredRoot) {
    ASSERT_TRUE(std::filesystem::create_directory(root_ + "/nested"));
    Create(std::string("nested/") + kProbe, 14400);
    ASSERT_EQ(::symlink((root_ + "/nested").c_str(),
                        (root_ + "/alias").c_str()),
              0);
    dfs_probe_gc::Cleanup(root_ + "/alias", kTtl);
    EXPECT_FALSE(Exists(std::string("nested/") + kProbe));
}

TEST_F(DfsProbeGcTest, RechecksMtimeAndTypeBeforeDeletion) {
    Create(kProbe, 14400);
    const int fd = ::open(root_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    ASSERT_GE(fd, 0);
    struct stat scanned {};
    ASSERT_EQ(::fstatat(fd, kProbe, &scanned, AT_SYMLINK_NOFOLLOW), 0);
    ASSERT_TRUE(dfs_probe_gc::IsExpired(scanned, kTtl));
    SetAge(kProbe, 0);
    dfs_probe_gc::DeleteIfExpired(fd, root_, kProbe, kTtl);
    EXPECT_TRUE(Exists(kProbe));
    SetAge(kProbe, -3600);
    dfs_probe_gc::DeleteIfExpired(fd, root_, kProbe, kTtl);
    EXPECT_TRUE(Exists(kProbe));
    ASSERT_EQ(::unlink((root_ + "/" + kProbe).c_str()), 0);
    Create("target", 14400);
    ASSERT_EQ(::symlink("target", (root_ + "/" + kProbe).c_str()), 0);
    dfs_probe_gc::DeleteIfExpired(fd, root_, kProbe, kTtl);
    EXPECT_TRUE(Exists(kProbe));
    EXPECT_TRUE(Exists("target"));
    ::close(fd);
}

TEST_F(DfsProbeGcTest, MissingFileAndFailedScanDoNotPreventLaterCleanup) {
    const int fd = ::open(root_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    ASSERT_GE(fd, 0);
    EXPECT_EQ(dfs_probe_gc::DeleteIfExpired(fd, root_, kProbe, kTtl),
              dfs_probe_gc::DeleteResult::kAlreadyAbsent);
    ::close(fd);
    EXPECT_EQ(dfs_probe_gc::DeleteIfExpired(-1, root_, kProbe, kTtl),
              dfs_probe_gc::DeleteResult::kFailed);
    EXPECT_NO_THROW(dfs_probe_gc::Cleanup(root_ + "/missing", kTtl));
    Create(kProbe, 14400);
    dfs_probe_gc::Cleanup(root_, kTtl);
    EXPECT_FALSE(Exists(kProbe));
}

TEST_F(DfsProbeGcTest, FailedUnlinkCanBeRetriedOnTheNextScan) {
    if (::geteuid() == 0) GTEST_SKIP() << "root bypasses directory permissions";
    Create(kProbe, 14400);
    const std::string second = ".mooncake_dfs_health_probe_123-456_2";
    Create(second, 14400);
    ASSERT_EQ(::chmod(root_.c_str(), 0500), 0);
    EXPECT_EQ(dfs_probe_gc::Cleanup(root_, kTtl), 0u);
    EXPECT_TRUE(Exists(kProbe));
    EXPECT_TRUE(Exists(second));
    // Restore permissions before assertions that could abort the test.
    const int restored = ::chmod(root_.c_str(), 0700);
    ASSERT_EQ(restored, 0);
    EXPECT_EQ(dfs_probe_gc::Cleanup(root_, kTtl), 2u);
    EXPECT_FALSE(Exists(kProbe));
    EXPECT_FALSE(Exists(second));
}

}  // namespace
}  // namespace mooncake::test
