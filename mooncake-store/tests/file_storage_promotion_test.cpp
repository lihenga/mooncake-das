// Unit tests for FileStorage::ProcessPromotionTasks. Uses a stub Client
// that overrides the 4 promotion-RPC entry points so we can drive the
// orchestration loop deterministically without standing up a master.

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <optional>
#include <set>
#include <thread>

#include "client_service.h"
#include "file_storage.h"
#include "storage_backend.h"

namespace mooncake {

namespace fs_test {

// Programmable fake. Each Set* method records what the next call should
// return; the call itself records that it happened so tests can assert on
// invocation count and ordering.
class FakeClient : public Client {
   public:
    FakeClient()
        : Client(/*local_hostname=*/"localhost:9003",
                 /*metadata_connstring=*/"",
                 /*protocol=*/"tcp",
                 /*labels=*/{}) {}

    // Drives the queue returned to the heartbeat caller.
    std::vector<PromotionTaskItem> heartbeat_queue;
    tl::expected<void, ErrorCode> heartbeat_result =
        tl::expected<void, ErrorCode>{};

    tl::expected<void, ErrorCode> PromotionObjectHeartbeat(
        std::vector<PromotionTaskItem>& promotion_objects) override {
        heartbeat_calls.fetch_add(1);
        if (!heartbeat_result.has_value()) {
            return tl::make_unexpected(heartbeat_result.error());
        }
        // Mirror production master: PromotionObjectHeartbeat returns at
        // most kMaxPerHeartbeat keys per call (see
        // MasterService::PromotionObjectHeartbeat) and leaves the rest
        // queued for subsequent calls. Tests iterate by calling
        // ProcessPromotionTasks multiple times until heartbeat_queue is
        // empty.
        constexpr size_t kMaxPerHeartbeat = 1;
        promotion_objects.clear();
        while (promotion_objects.size() < kMaxPerHeartbeat &&
               !heartbeat_queue.empty()) {
            promotion_objects.push_back(std::move(heartbeat_queue.back()));
            heartbeat_queue.pop_back();
        }
        return {};
    }

    // PromotionAllocStart: per-key dispatch table. Keys absent from
    // alloc_overrides return the default response.
    std::unordered_map<std::string, ErrorCode> alloc_overrides;
    PromotionAllocStartResponse default_alloc_response;
    bool default_alloc_succeeds = true;

    tl::expected<PromotionAllocStartResponse, ErrorCode> PromotionAllocStart(
        const std::string& key, uint64_t size,
        const std::vector<std::string>& preferred_segments) override {
        return PromotionAllocStart(key, "default", size, preferred_segments);
    }

    tl::expected<PromotionAllocStartResponse, ErrorCode> PromotionAllocStart(
        const std::string& key, const std::string& tenant_id, uint64_t size,
        const std::vector<std::string>& preferred_segments) override {
        (void)tenant_id;
        (void)size;
        (void)preferred_segments;
        alloc_calls.fetch_add(1);
        last_alloc_key = key;
        auto it = alloc_overrides.find(key);
        if (it != alloc_overrides.end()) {
            return tl::make_unexpected(it->second);
        }
        if (!default_alloc_succeeds) {
            return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
        }
        return default_alloc_response;
    }

    // PromotionWrite: per-key dispatch.
    std::unordered_map<std::string, ErrorCode> write_overrides;
    ErrorCode default_write_result = ErrorCode::OK;

    ErrorCode PromotionWrite(const Replica::Descriptor&,
                             std::vector<Slice>&) override {
        write_calls.fetch_add(1);
        events.push_back("write");
        auto it = write_overrides.find(last_alloc_key);
        if (it != write_overrides.end()) {
            return it->second;
        }
        return default_write_result;
    }

    std::unordered_map<std::string, ErrorCode> notify_overrides;
    tl::expected<void, ErrorCode> default_notify_result =
        tl::expected<void, ErrorCode>{};

    tl::expected<void, ErrorCode> NotifyPromotionSuccess(
        const std::string& key) override {
        return NotifyPromotionSuccess(key, "default");
    }

    tl::expected<void, ErrorCode> NotifyPromotionSuccess(
        const std::string& key, const std::string& tenant_id) override {
        (void)tenant_id;
        notify_calls.fetch_add(1);
        notify_keys.push_back(key);
        events.push_back("notify_success");
        auto it = notify_overrides.find(key);
        if (it != notify_overrides.end()) {
            return tl::make_unexpected(it->second);
        }
        if (!default_notify_result.has_value()) {
            return tl::make_unexpected(default_notify_result.error());
        }
        return {};
    }

    // NotifyPromotionFailure: records calls so tests can assert that
    // post-AllocStart failure paths in ProcessPromotionTasks actually
    // notify the master.
    tl::expected<void, ErrorCode> NotifyPromotionFailure(
        const std::string& key) override {
        return NotifyPromotionFailure(key, "default");
    }

    tl::expected<void, ErrorCode> NotifyPromotionFailure(
        const std::string& key, const std::string& tenant_id) override {
        (void)tenant_id;
        notify_failure_calls.fetch_add(1);
        notify_failure_keys.push_back(key);
        events.push_back("notify_failure");
        return {};
    }

    // ---- DFS promotion channel. Same shape as the SSD overrides
    // above, but for Client::DfsPromotionObjectHeartbeat /
    // HasDfsStorageBackend / ReadDfsReplicaForPromotion. ----

    // A client only serves the DFS channel when it wired a distributed
    // backend at construction; the fake flips this instead.
    bool dfs_capable = false;

    bool HasDfsStorageBackend() const override { return dfs_capable; }

    std::vector<PromotionTaskItem> dfs_heartbeat_queue;
    tl::expected<void, ErrorCode> dfs_heartbeat_result =
        tl::expected<void, ErrorCode>{};

    tl::expected<void, ErrorCode> DfsPromotionObjectHeartbeat(
        std::vector<PromotionTaskItem>& promotion_objects) override {
        dfs_heartbeat_calls.fetch_add(1);
        if (!dfs_heartbeat_result.has_value()) {
            return tl::make_unexpected(dfs_heartbeat_result.error());
        }
        // Mirror the production master: at most kMaxPerHeartbeat keys per
        // call, remainder stays queued for subsequent ticks.
        constexpr size_t kMaxPerHeartbeat = 1;
        promotion_objects.clear();
        while (promotion_objects.size() < kMaxPerHeartbeat &&
               !dfs_heartbeat_queue.empty()) {
            promotion_objects.push_back(std::move(dfs_heartbeat_queue.back()));
            dfs_heartbeat_queue.pop_back();
        }
        return {};
    }

    // ReadDfsReplicaForPromotion: per-key dispatch. `key` is the
    // tenant-scoped storage key the executor derives, so overrides are
    // registered against TenantId(...).MakeScopedKey(local_key).
    std::unordered_map<std::string, ErrorCode> dfs_read_overrides;
    ErrorCode default_dfs_read_result = ErrorCode::OK;

    ErrorCode ReadDfsReplicaForPromotion(
        const std::string& key, const Replica::Descriptor& replica_descriptor,
        std::vector<Slice>& slices) override {
        (void)replica_descriptor;
        (void)slices;
        dfs_read_calls.fetch_add(1);
        last_dfs_read_key = key;
        events.push_back("dfs_read");
        auto it = dfs_read_overrides.find(key);
        if (it != dfs_read_overrides.end()) {
            return it->second;
        }
        return default_dfs_read_result;
    }

    std::atomic<int> heartbeat_calls{0};
    std::atomic<int> alloc_calls{0};
    std::atomic<int> write_calls{0};
    std::atomic<int> notify_calls{0};
    std::atomic<int> notify_failure_calls{0};
    std::atomic<int> dfs_heartbeat_calls{0};
    std::atomic<int> dfs_read_calls{0};
    std::vector<std::string> notify_keys;
    std::vector<std::string> notify_failure_keys;
    std::string last_alloc_key;
    std::string last_dfs_read_key;

    // Ordered log of the RPCs the executor issued. Lets tests assert the
    // read -> TE-write -> commit sequence (and that no write/commit ran when
    // the read failed). Single-threaded use only.
    std::vector<std::string> events;
};

}  // namespace fs_test

class FileStoragePromotionTest : public ::testing::Test {
   protected:
    std::string data_path;
    std::shared_ptr<fs_test::FakeClient> fake;
    std::unique_ptr<FileStorage> file_storage;

    void SetUp() override {
        google::InitGoogleLogging("FileStoragePromotionTest");
        FLAGS_logtostderr = true;
        unsetenv("MOONCAKE_OFFLOAD_FILE_STORAGE_PATH");

        data_path = std::filesystem::current_path().string() + "/data_prom";
        std::filesystem::create_directories(data_path);
        for (const auto& entry :
             std::filesystem::directory_iterator(data_path)) {
            if (entry.is_regular_file()) std::filesystem::remove(entry.path());
        }

        FileStorageConfig cfg = FileStorageConfig::FromEnvironment();
        cfg.storage_filepath = data_path;
        cfg.local_buffer_size = 4 * 1024 * 1024;
        fake = std::make_shared<fs_test::FakeClient>();
        file_storage =
            std::make_unique<FileStorage>(cfg, fake, "localhost:9003");
    }

    void TearDown() override {
        file_storage.reset();
        fake.reset();
        google::ShutdownGoogleLogging();
        for (const auto& entry :
             std::filesystem::directory_iterator(data_path)) {
            if (entry.is_regular_file()) std::filesystem::remove(entry.path());
        }
    }

    tl::expected<void, ErrorCode> CallProcessPromotionTasks() {
        return file_storage->ProcessPromotionTasks();
    }

    tl::expected<void, ErrorCode> CallProcessDfsPromotionTasks() {
        return file_storage->ProcessDfsPromotionTasks();
    }

    // Build a DFS-channel task carrying a COMPLETE DFS source descriptor, the
    // shape MasterService::DfsPromotionObjectHeartbeat hands to a capable
    // client.
    static PromotionTaskItem MakeDfsTask(
        const std::string& key, int64_t size,
        const std::string& file_path = "/dfs/obj", uint64_t offset = 0) {
        PromotionTaskItem task;
        task.tenant_id = std::string(TenantId::kDefaultValue);
        task.key = key;
        task.size = size;
        DistributedFSDescriptor dfs_desc;
        dfs_desc.file_path = file_path;
        dfs_desc.offset = offset;
        dfs_desc.object_size = static_cast<uint64_t>(size);
        dfs_desc.aligned_size = static_cast<uint64_t>(size);
        dfs_desc.shard_idx = 0;
        Replica::Descriptor desc{};
        desc.id = 0;
        desc.status = ReplicaStatus::COMPLETE;
        desc.descriptor_variant = dfs_desc;
        task.source_dfs = std::move(desc);
        return task;
    }

    // The tenant-scoped storage key the executor derives for a local key.
    static std::string ScopedKey(const std::string& local_key) {
        return TenantId(std::string(TenantId::kDefaultValue))
            .MakeScopedKey(local_key);
    }

    // Drain a multi-key queue across multiple ticks. ProcessPromotionTasks
    // caps work at 1 task/tick (heartbeat-safety) and the FakeClient's
    // PromotionObjectHeartbeat mirrors production by clearing the queue on
    // drain. So the test fixture has to push the remaining keys back
    // between ticks, the same way the master would re-push if the gate
    // re-fires. We track which key was processed via fake->last_alloc_key
    // and remove it from the working set.
    tl::expected<void, ErrorCode> DrainAllPromotionTasks(
        std::unordered_map<std::string, int64_t> remaining) {
        tl::expected<void, ErrorCode> last_res{};
        while (!remaining.empty()) {
            std::string before = fake->last_alloc_key;
            fake->heartbeat_queue.clear();
            fake->heartbeat_queue.reserve(remaining.size());
            for (const auto& [key, size] : remaining) {
                fake->heartbeat_queue.push_back(PromotionTaskItem{
                    .tenant_id = "default",
                    .key = key,
                    .size = size,
                    .source_dfs = std::nullopt});
            }
            last_res = CallProcessPromotionTasks();
            if (!last_res.has_value()) return last_res;
            if (fake->last_alloc_key == before ||
                !remaining.contains(fake->last_alloc_key)) {
                // No forward progress (e.g., empty effective queue / size<=0
                // skip on every key); avoid infinite loop.
                break;
            }
            remaining.erase(fake->last_alloc_key);
        }
        return last_res;
    }
};

// Empty heartbeat queue: do nothing past the heartbeat call.
TEST_F(FileStoragePromotionTest, EmptyQueueIsNoOp) {
    fake->heartbeat_queue = {};
    auto res = CallProcessPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->heartbeat_calls.load(), 1);
    EXPECT_EQ(fake->alloc_calls.load(), 0);
    EXPECT_EQ(fake->write_calls.load(), 0);
    EXPECT_EQ(fake->notify_calls.load(), 0);
}

// Heartbeat returns SEGMENT_NOT_FOUND (transient post-restart):
// swallow silently, no error to caller.
TEST_F(FileStoragePromotionTest, HeartbeatSegmentNotFoundIsBenign) {
    fake->heartbeat_result = tl::make_unexpected(ErrorCode::SEGMENT_NOT_FOUND);
    auto res = CallProcessPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 0);
}

// Heartbeat returns a non-benign error: propagate it.
TEST_F(FileStoragePromotionTest, HeartbeatHardErrorPropagates) {
    fake->heartbeat_result = tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    auto res = CallProcessPromotionTasks();
    ASSERT_FALSE(res.has_value());
    EXPECT_EQ(res.error(), ErrorCode::INTERNAL_ERROR);
    EXPECT_EQ(fake->alloc_calls.load(), 0);
}

// Non-positive size in queue: skip that key, continue.
TEST_F(FileStoragePromotionTest, NonPositiveSizeSkipped) {
    fake->heartbeat_queue = {
        {.tenant_id = "default",
         .key = "k_bad",
         .size = 0,
         .source_dfs = std::nullopt},
        {.tenant_id = "default",
         .key = "k_good",
         .size = 1024,
         .source_dfs = std::nullopt}};
    auto res = CallProcessPromotionTasks();
    EXPECT_TRUE(res.has_value());
    // Only k_good should reach AllocStart.
    EXPECT_EQ(fake->alloc_calls.load(), 1);
    EXPECT_EQ(fake->last_alloc_key, "k_good");
}

// PromotionAllocStart fails (e.g. master out of DRAM): skip key, no
// write, no notify, advance to next.
TEST_F(FileStoragePromotionTest, AllocStartFailureSkipsKey) {
    fake->alloc_overrides["k1"] = ErrorCode::NO_AVAILABLE_HANDLE;
    auto res = DrainAllPromotionTasks({{"k1", 1024}, {"k2", 1024}});
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 2);
    // k1's failure prevents its write+notify; k2 succeeds (then fails at
    // BatchLoad because the SSD file doesn't exist).
    EXPECT_LE(fake->write_calls.load(), 1);
    EXPECT_LE(fake->notify_calls.load(), 1);
}

// BatchLoad failure (SSD file missing): no PromotionWrite, no Notify.
// Master-side reaper handles the orphaned PROCESSING replica.
TEST_F(FileStoragePromotionTest, BatchLoadFailureLeavesNoNotify) {
    fake->heartbeat_queue = {
        {.tenant_id = "default",
         .key = "k_missing",
         .size = 1024,
         .source_dfs = std::nullopt}};
    // Default alloc succeeds; BatchLoad will fail because there's no file
    // at data_path/k_missing for the storage backend to read.
    auto res = CallProcessPromotionTasks();
    EXPECT_TRUE(res.has_value());  // ProcessPromotionTasks itself is best-
                                   // effort; per-key failures are warnings.
    EXPECT_EQ(fake->alloc_calls.load(), 1);
    EXPECT_EQ(fake->write_calls.load(), 0)
        << "TransferWrite must not run if BatchLoad failed";
    EXPECT_EQ(fake->notify_calls.load(), 0)
        << "NotifyPromotionSuccess must not run if BatchLoad failed";
}

// PromotionWrite failure: no Notify.
TEST_F(FileStoragePromotionTest, TransferWriteFailureLeavesNoNotify) {
    fake->heartbeat_queue = {
        {.tenant_id = "default",
         .key = "k_te_fail",
         .size = 1024,
         .source_dfs = std::nullopt}};
    fake->default_write_result = ErrorCode::TRANSFER_FAIL;
    auto res = CallProcessPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 1);
    // Whether write_calls is 0 or 1 depends on whether BatchLoad succeeded
    // first; either way Notify must not fire.
    EXPECT_EQ(fake->notify_calls.load(), 0)
        << "NotifyPromotionSuccess must not run on TransferWrite failure";
}

// NotifyPromotionSuccess failure: logged, but processing of remaining
// keys continues.
TEST_F(FileStoragePromotionTest, NotifyFailureDoesNotAbortBatch) {
    fake->notify_overrides["k1"] = ErrorCode::OBJECT_NOT_FOUND;
    auto res = DrainAllPromotionTasks({{"k1", 1024}, {"k2", 1024}});
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 2)
        << "Both keys must be alloc-attempted regardless of k1's notify "
        << "failure";
}

// Per-key independence: failures on one key don't prevent attempts on
// others.
TEST_F(FileStoragePromotionTest, PerKeyFailuresAreIndependent) {
    fake->alloc_overrides["k_alloc_fail"] = ErrorCode::NO_AVAILABLE_HANDLE;
    fake->write_overrides["k_write_fail"] = ErrorCode::TRANSFER_FAIL;
    auto res = DrainAllPromotionTasks({
        {"k_alloc_fail", 1024},
        {"k_write_fail", 1024},
        {"k_normal", 1024},
    });
    EXPECT_TRUE(res.has_value());
    // All three reach AllocStart.
    EXPECT_EQ(fake->alloc_calls.load(), 3);
}

// Every post-admission failure path (including PromotionAllocStart's
// own failure) must call NotifyPromotionFailure so the master releases
// the task slot immediately. Without this, the slot stays pinned until
// put_start_release_timeout_sec_ (~10 min default), turning a transient
// DRAM-pressure spike into a sustained outage of promotion_queue_limit_.
TEST_F(FileStoragePromotionTest, AllocStartFailureNotifiesMaster) {
    fake->heartbeat_queue = {
        {.tenant_id = "default",
         .key = "k_alloc_fail",
         .size = 1024,
         .source_dfs = std::nullopt}};
    fake->alloc_overrides["k_alloc_fail"] = ErrorCode::NO_AVAILABLE_HANDLE;
    auto res = CallProcessPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 1);
    EXPECT_EQ(fake->notify_calls.load(), 0)
        << "Success-notify must not fire on AllocStart failure.";
    EXPECT_EQ(fake->notify_failure_calls.load(), 1)
        << "AllocStart failure must invoke NotifyPromotionFailure so the "
        << "master can release the task slot immediately. Without this "
        << "the slot is pinned for ~10 min and transient DRAM pressure "
        << "saturates promotion_queue_limit_ for the same window.";
    ASSERT_EQ(fake->notify_failure_keys.size(), 1u);
    EXPECT_EQ(fake->notify_failure_keys[0], "k_alloc_fail");
}

// Same invariant on the post-AllocStart paths (BatchLoad /
// TransferWrite / Notify-Success). Each failure mode must release the
// master slot. Exercises three modes (AllocStart itself, missing-file
// BatchLoad, override on Notify) by draining across successive
// heartbeats (the FakeClient mirrors the master's kMaxPerHeartbeat = 1
// cap) and verifies each one releases.
TEST_F(FileStoragePromotionTest, PostAllocFailuresAllNotifyMaster) {
    fake->alloc_overrides["k_alloc_fail"] = ErrorCode::NO_AVAILABLE_HANDLE;
    // k_load_fail: no override -> AllocStart succeeds, but BatchLoad
    // will fail because no SSD file exists for this key in data_path.
    // k_notify_fail: AllocStart and BatchLoad and TransferWrite all
    // succeed; Notify is overridden to fail.
    fake->notify_overrides["k_notify_fail"] = ErrorCode::OBJECT_NOT_FOUND;

    auto res = DrainAllPromotionTasks({
        {"k_alloc_fail", 1024},
        {"k_load_fail", 1024},
        {"k_notify_fail", 1024},
    });
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 3);

    // Every failed key must have its slot released via Notify-Failure.
    // k_notify_fail also counts: Notify-Success failed, so we still
    // need to free the slot.
    EXPECT_EQ(fake->notify_failure_calls.load(), 3)
        << "All three failed promotions must each invoke "
        << "NotifyPromotionFailure. If this fires with <3, one of the "
        << "failure paths is still leaking the master slot.";
    std::set<std::string> got(fake->notify_failure_keys.begin(),
                              fake->notify_failure_keys.end());
    EXPECT_EQ(got.count("k_alloc_fail"), 1u);
    EXPECT_EQ(got.count("k_load_fail"), 1u);
    EXPECT_EQ(got.count("k_notify_fail"), 1u);
}

// ================= DFS promotion executor =================
// FileStorage::ProcessDfsPromotionTasks mirrors the SSD executor above but
// reads the source bytes from the DFS backend
// (Client::ReadDfsReplicaForPromotion) instead of local SSD, and only runs on
// DFS-capable clients (those with a distributed backend).

// A client without a distributed backend must not poll the DFS channel at
// all: the whole tick is a no-op.
TEST_F(FileStoragePromotionTest, DfsNonCapableClientSkipsChannel) {
    fake->dfs_capable = false;
    fake->dfs_heartbeat_queue = {MakeDfsTask("k_dfs", 1024)};

    auto res = CallProcessDfsPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->dfs_heartbeat_calls.load(), 0)
        << "Non-DFS-capable clients must not poll DfsPromotionObjectHeartbeat";
    EXPECT_EQ(fake->alloc_calls.load(), 0);
}

// Capable client, empty queue: heartbeat runs, nothing else.
TEST_F(FileStoragePromotionTest, DfsEmptyQueueIsNoOp) {
    fake->dfs_capable = true;
    auto res = CallProcessDfsPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->dfs_heartbeat_calls.load(), 1);
    EXPECT_EQ(fake->alloc_calls.load(), 0);
    EXPECT_EQ(fake->dfs_read_calls.load(), 0);
}

// Heartbeat hard error propagates to the caller (mirrors the SSD channel).
TEST_F(FileStoragePromotionTest, DfsHeartbeatHardErrorPropagates) {
    fake->dfs_capable = true;
    fake->dfs_heartbeat_result = tl::make_unexpected(ErrorCode::INTERNAL_ERROR);

    auto res = CallProcessDfsPromotionTasks();
    ASSERT_FALSE(res.has_value());
    EXPECT_EQ(res.error(), ErrorCode::INTERNAL_ERROR);
    EXPECT_EQ(fake->alloc_calls.load(), 0);
}

// A DFS task missing its source descriptor cannot be executed: release the
// claimed slot immediately instead of pinning it for the reaper TTL.
TEST_F(FileStoragePromotionTest, DfsTaskWithoutSourceNotifiesFailure) {
    fake->dfs_capable = true;
    PromotionTaskItem no_source;
    no_source.tenant_id = std::string(TenantId::kDefaultValue);
    no_source.key = "k_no_source";
    no_source.size = 1024;
    no_source.source_dfs.reset();
    fake->dfs_heartbeat_queue = {std::move(no_source)};

    auto res = CallProcessDfsPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 0);
    EXPECT_EQ(fake->dfs_read_calls.load(), 0);
    EXPECT_EQ(fake->notify_calls.load(), 0);
    EXPECT_EQ(fake->notify_failure_calls.load(), 1);
    ASSERT_EQ(fake->notify_failure_keys.size(), 1u);
    EXPECT_EQ(fake->notify_failure_keys[0], "k_no_source");
}

// Non-positive size is skipped before any RPC: nothing to release.
TEST_F(FileStoragePromotionTest, DfsNonPositiveSizeSkipped) {
    fake->dfs_capable = true;
    fake->dfs_heartbeat_queue = {MakeDfsTask("k_bad", 0)};

    auto res = CallProcessDfsPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 0);
    EXPECT_EQ(fake->dfs_read_calls.load(), 0);
    EXPECT_EQ(fake->notify_failure_calls.load(), 0);
}

// Happy path: AllocStart -> read DFS source -> TE-write -> commit, in that
// order, using the tenant-scoped storage key for the DFS read.
TEST_F(FileStoragePromotionTest, DfsHappyPathReadsWritesCommits) {
    fake->dfs_capable = true;
    fake->dfs_heartbeat_queue = {MakeDfsTask("k_dfs_ok", 1024, "/dfs/x", 4096)};

    auto res = CallProcessDfsPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 1);
    EXPECT_EQ(fake->dfs_read_calls.load(), 1);
    EXPECT_EQ(fake->write_calls.load(), 1);
    EXPECT_EQ(fake->notify_calls.load(), 1);
    EXPECT_EQ(fake->notify_failure_calls.load(), 0);
    EXPECT_EQ(fake->last_dfs_read_key, ScopedKey("k_dfs_ok"));
    EXPECT_EQ(fake->events,
              (std::vector<std::string>{"dfs_read", "write",
                                        "notify_success"}));
}

// DFS read failure: no TE write, no commit, slot released.
TEST_F(FileStoragePromotionTest, DfsReadFailureNotifiesFailure) {
    fake->dfs_capable = true;
    fake->dfs_heartbeat_queue = {MakeDfsTask("k_dfs_read_fail", 1024)};
    fake->dfs_read_overrides[ScopedKey("k_dfs_read_fail")] =
        ErrorCode::DFS_SERVICE_UNAVAILABLE;

    auto res = CallProcessDfsPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 1);
    EXPECT_EQ(fake->dfs_read_calls.load(), 1);
    EXPECT_EQ(fake->write_calls.load(), 0)
        << "TE write must not run when the DFS read failed";
    EXPECT_EQ(fake->notify_calls.load(), 0);
    EXPECT_EQ(fake->notify_failure_calls.load(), 1);
}

// AllocStart failure (e.g. no free DRAM): no read, slot released.
TEST_F(FileStoragePromotionTest, DfsAllocStartFailureNotifiesFailure) {
    fake->dfs_capable = true;
    fake->dfs_heartbeat_queue = {MakeDfsTask("k_dfs_alloc_fail", 1024)};
    fake->alloc_overrides["k_dfs_alloc_fail"] = ErrorCode::NO_AVAILABLE_HANDLE;

    auto res = CallProcessDfsPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->alloc_calls.load(), 1);
    EXPECT_EQ(fake->dfs_read_calls.load(), 0);
    EXPECT_EQ(fake->notify_calls.load(), 0);
    EXPECT_EQ(fake->notify_failure_calls.load(), 1);
}

// TE write failure: no commit, slot released.
TEST_F(FileStoragePromotionTest, DfsTransferWriteFailureNotifiesFailure) {
    fake->dfs_capable = true;
    fake->dfs_heartbeat_queue = {MakeDfsTask("k_dfs_write_fail", 1024)};
    fake->write_overrides["k_dfs_write_fail"] = ErrorCode::TRANSFER_FAIL;

    auto res = CallProcessDfsPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->dfs_read_calls.load(), 1);
    EXPECT_EQ(fake->write_calls.load(), 1);
    EXPECT_EQ(fake->notify_calls.load(), 0);
    EXPECT_EQ(fake->notify_failure_calls.load(), 1);
}

// Commit failure: the bytes landed but the commit did not. The slot must
// still be released; the orphaned PROCESSING replica is reaped by the
// master's DFS promotion reaper.
TEST_F(FileStoragePromotionTest, DfsNotifySuccessFailureNotifiesFailure) {
    fake->dfs_capable = true;
    fake->dfs_heartbeat_queue = {MakeDfsTask("k_dfs_notify_fail", 1024)};
    fake->notify_overrides["k_dfs_notify_fail"] = ErrorCode::OBJECT_NOT_FOUND;

    auto res = CallProcessDfsPromotionTasks();
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(fake->dfs_read_calls.load(), 1);
    EXPECT_EQ(fake->write_calls.load(), 1);
    EXPECT_EQ(fake->notify_calls.load(), 1);
    EXPECT_EQ(fake->notify_failure_calls.load(), 1);
}

}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
