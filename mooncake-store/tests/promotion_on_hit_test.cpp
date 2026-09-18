// Unit tests for the L2->L1 promotion-on-hit master-side path. Exercises
// the master-service entry points directly without going through the RPC
// layer.

#include "master_service.h"

// master_service.h only forward-declares the heat sketch (the real header is
// confined to the master_service.cpp TU), but these tests drive the sketch
// directly through the ...ForTesting funnels, so the full definition is needed.
#include "decaying_quantile/decaying_ddsketch.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "tenant_quota_policy_store.h"
#include "types.h"

namespace mooncake::test {

size_t CountPromotionTask(const std::vector<PromotionTaskItem>& tasks,
                          const std::string& key) {
    return std::count_if(
        tasks.begin(), tasks.end(),
        [&key](const PromotionTaskItem& task) { return task.key == key; });
}

// RAII override for a process environment variable. The DFS allocator is
// configured exclusively from the environment at MasterService construction
// time, so the tests that need a real DFS backend (eviction) have to flip these
// before building the service and restore them afterwards.
class ScopedEnv {
   public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        ::setenv(name, value, /*overwrite=*/1);
    }

    ~ScopedEnv() {
        if (had_previous_) {
            ::setenv(name_.c_str(), previous_.c_str(), /*overwrite=*/1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

   private:
    std::string name_;
    bool had_previous_ = false;
    std::string previous_;
};

class PromotionOnHitTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("PromotionOnHitTest");
        FLAGS_logtostderr = true;
    }

    void TearDown() override {
        for (const auto& path : policy_files_) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
        google::ShutdownGoogleLogging();
    }

    // Friend access to MasterService::promotion_admission_threshold_, which
    // is otherwise private. PromotionOnHitTest is friended; TEST_F-generated
    // subclasses are not, hence this static funnel.
    static uint32_t GetPromotionAdmissionThresholdForTesting(
        MasterService* service) {
        return service->promotion_admission_threshold_;
    }

    static size_t CountPromotionCandidatesForTesting(MasterService* service,
                                                     const TenantId& tenant) {
        return service->CountCandidatesForTesting(tenant);
    }

    static constexpr uint32_t MaxPromotionCandidateRetriesForTesting() {
        return MasterService::kPromotionCandidateMaxRetries;
    }

    static constexpr uint32_t MaxPromotionExecutionFailuresForTesting() {
        return MasterService::kMaxPromotionExecutionFailures;
    }

    static void ResetCandidateBackoffsForTesting(MasterService* service) {
        service->ResetCandidateBackoffsForTesting();
    }

    static size_t RunPromotionCandidateRetryForTesting(MasterService* service) {
        return service->RunPromotionCandidateRetryForTesting();
    }

    static size_t RunPromotionCandidateRetryForTesting(MasterService* service,
                                                       size_t shards_to_scan) {
        return service->RunPromotionCandidateRetry(shards_to_scan);
    }

    static void ClearCandidatesForReloadForTesting(MasterService* service) {
        service->ClearCandidatesForReload();
    }

    static uint64_t GetPromotionCandidateCountForTesting(
        MasterService* service) {
        return service->promotion_candidate_count_.load(
            std::memory_order_relaxed);
    }

    static uint64_t GetPromotionInFlightForTesting(MasterService* service) {
        return service->promotion_in_flight_.load(std::memory_order_relaxed);
    }

    static bool HasPromotionTaskForTesting(MasterService* service,
                                           const TenantId& tenant_id,
                                           const std::string& key) {
        MasterService::MetadataAccessorRO accessor(
            service, MasterService::ObjectIdentity{.tenant_id = tenant_id,
                                                   .user_key = key});
        const auto* tenant_state = accessor.GetTenantState();
        return tenant_state != nullptr &&
               tenant_state->promotion_tasks.contains(key);
    }

    // std::nullopt when the key has no in-flight promotion task.
    static std::optional<uint32_t> GetPromotionTaskExecutionFailuresForTesting(
        MasterService* service, const TenantId& tenant_id,
        const std::string& key) {
        MasterService::MetadataAccessorRO accessor(
            service, MasterService::ObjectIdentity{.tenant_id = tenant_id,
                                                   .user_key = key});
        const auto* tenant_state = accessor.GetTenantState();
        if (tenant_state == nullptr) {
            return std::nullopt;
        }
        auto it = tenant_state->promotion_tasks.find(key);
        if (it == tenant_state->promotion_tasks.end()) {
            return std::nullopt;
        }
        return it->second.execution_failures;
    }

    static bool PromotionAdmissionBlockedByPrimaryWriteForTesting(
        MasterService* service, const TenantId& tenant_id,
        const std::string& key) {
        const auto result =
            service->TryPushPromotionQueue(MasterService::ObjectIdentity{
                .tenant_id = tenant_id, .user_key = key});
        return result == MasterService::PromotionQueueResult::kAlreadyInFlight;
    }

    static void MarkReplicaCompleteForTesting(MasterService* service,
                                              const TenantId& tenant_id,
                                              const std::string& key,
                                              ReplicaID replica_id) {
        MasterService::MetadataAccessorRW accessor(
            service, MasterService::ObjectIdentity{.tenant_id = tenant_id,
                                                   .user_key = key});
        ASSERT_TRUE(accessor.Exists());
        auto* replica = accessor.Get().GetReplicaByID(replica_id);
        ASSERT_NE(replica, nullptr);
        replica->mark_complete();
    }

    static std::unique_ptr<AllocatedBuffer> AllocateOnSegmentForTesting(
        MasterService* service, const UUID& segment_id, size_t size) {
        std::shared_ptr<BufferAllocatorBase> allocator;
        {
            auto segment_access = service->segment_manager_.getSegmentAccess();
            allocator = segment_access.GetAllocator(segment_id);
        }
        if (!allocator) {
            return nullptr;
        }
        return allocator->allocate(size);
    }

    static constexpr size_t kDefaultSegmentBase = 0x300000000;

    std::string WriteTenantQuotaPolicyFile(
        const std::map<std::string, uint64_t>& tenant_quotas) {
        TenantQuotaPolicySnapshot snapshot;
        snapshot.tenant_quotas = tenant_quotas;
        auto path =
            std::filesystem::temp_directory_path() /
            ("mooncake_promotion_tenant_policy_" + std::to_string(::getpid()) +
             "_" + std::to_string(next_policy_file_++) + ".yaml");
        std::ofstream out(path);
        out << FormatTenantQuotaPolicyYaml(snapshot);
        out.close();
        policy_files_.push_back(path.string());
        return path.string();
    }

    Segment MakeSegment(std::string name, size_t base, size_t size) const {
        Segment segment;
        segment.id = generate_uuid();
        segment.name = std::move(name);
        segment.base = base;
        segment.size = size;
        segment.te_endpoint = segment.name;
        return segment;
    }

    struct MountedSegmentContext {
        UUID segment_id;
        UUID client_id;
        std::string segment_name;
    };

    MountedSegmentContext PrepareSegment(MasterService& service,
                                         std::string name, size_t base,
                                         size_t size) const {
        Segment segment = MakeSegment(std::move(name), base, size);
        UUID client_id = generate_uuid();
        auto mount_result = service.MountSegment(segment, client_id);
        EXPECT_TRUE(mount_result.has_value());
        auto mount_ld = service.MountLocalDiskSegment(client_id, true);
        EXPECT_TRUE(mount_ld.has_value());
        return {.segment_id = segment.id,
                .client_id = client_id,
                .segment_name = segment.name};
    }

    // Put an object and complete it (creates a MEMORY replica).
    void PutObject(MasterService& service, const UUID& client_id,
                   const std::string& key, size_t size = 1024) {
        ReplicateConfig config;
        config.replica_num = 1;
        auto put_start =
            service.PutStart(client_id, key, TenantId::Default(), size, config);
        ASSERT_TRUE(put_start.has_value()) << "PutStart failed for key=" << key;
        auto put_end = service.PutEnd(client_id, key, TenantId::Default(),
                                      ReplicaType::MEMORY);
        ASSERT_TRUE(put_end.has_value()) << "PutEnd failed for key=" << key;
    }

    // Inject a synthetic LOCAL_DISK replica for `key` on `client_id`'s
    // segment via NotifyOffloadSuccess. Lets tests put a key into
    // LOCAL_DISK-only state without running the full offload pipeline.
    bool InjectLocalDiskReplica(
        MasterService& service, const UUID& client_id, const std::string& key,
        int64_t size, const std::string& transport_endpoint,
        std::string tenant_id = std::string(TenantId::kDefaultValue)) {
        // NotifyOffloadSuccess registers replicas only for a client with a
        // mounted LOCAL_DISK segment, the way the real offload pipeline
        // mounts before registering. Idempotent, so repeat injections for
        // the same client are fine.
        auto mount = service.MountLocalDiskSegment(client_id, true);
        if (!mount.has_value()) {
            return false;
        }
        std::vector<OffloadTaskItem> tasks{
            OffloadTaskItem{.tenant_id = tenant_id, .key = key, .size = size}};
        StorageObjectMetadata sm;
        sm.bucket_id = 0;
        sm.offset = 0;
        sm.key_size = static_cast<int64_t>(key.size());
        sm.data_size = size;
        sm.transport_endpoint = transport_endpoint;
        std::vector<StorageObjectMetadata> metas{sm};
        auto res = service.NotifyOffloadSuccess(client_id, tasks, metas);
        return res.has_value();
    }

    // ---- DFS promotion channel helpers. PromotionOnHitTest is
    // friended by MasterService, so these funnels can drive the DFS
    // master-side queue/heartbeat logic directly without the full Put/DFS
    // pipeline (which would need the DFS backend environment + a real client
    // to later read the source). ----

    // Inject a synthetic DFS-only object directly into the metadata shard.
    // The DFS replica is COMPLETE and no MEMORY replica exists, so admission
    // sees exactly the "top COMPLETE replica is DFS" state.
    bool InjectDfsOnlyObject(MasterService& service, const std::string& key,
                             size_t size = 1024,
                             const std::string& file_path = "/dfs/test/obj",
                             uint64_t offset = 0, bool hard_pinned = false,
                             ReplicaStatus status = ReplicaStatus::COMPLETE) {
        const TenantId tenant = TenantId::Default();
        const size_t shard_index = service.getMetadataShardIndex(tenant, key);
        DistributedFSDescriptor dfs_desc;
        dfs_desc.file_path = file_path;
        dfs_desc.offset = offset;
        dfs_desc.object_size = size;
        dfs_desc.aligned_size = size;
        dfs_desc.shard_idx = static_cast<int>(shard_index);
        std::vector<Replica> replicas;
        replicas.emplace_back(std::move(dfs_desc), status);
        // Hold the shard's write lock for the injection. The master starts its
        // background threads (eviction discard sweep, DFS reconcile scan, ghost
        // census) as soon as it is constructed and they walk these very maps
        // under this lock; poking them unlocked raced those scans and could
        // make the injected object invisible to the admit call right after it.
        MasterService::MetadataShardAccessorRW shard_accessor(&service,
                                                              shard_index);
        auto& tenant_shard = shard_accessor.get();
        const auto [it, inserted] = tenant_shard.tenants[tenant].metadata.emplace(
            std::piecewise_construct, std::forward_as_tuple(key),
            std::forward_as_tuple(
                generate_uuid(), std::chrono::system_clock::now(), size,
                std::move(replicas),
                std::nullopt /*committed_soft_pin_timeout: no soft pin*/,
                hard_pinned /*enable_hard_pin*/, ObjectDataType::UNKNOWN,
                std::string() /*group_id*/, tenant, key));
        (void)it;
        EXPECT_TRUE(inserted) << "duplicate DFS-only key: " << key;
        return inserted;
    }

    // Run one admission pass for an injected DFS-only key (the same entry
    // point the read-hit / candidate-retry paths funnel into).
    static bool AdmitDfsPromotionForTesting(MasterService* service,
                                            const std::string& key,
                                            double heat = 100.0) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return false;
        return service->TryAdmitDfsPromotion(
                   accessor.GetTenantState(), accessor.Get(), heat,
                   service->DfsNowEpochMin(), false) ==
               MasterService::DfsAdmissionResult::kAdmitted;
    }

    // Mark `key` as a protected DFS source without running the full
    // admission path, so the DFS eviction acceptance can be driven exactly.
    static void ProtectDfsSourceForTesting(MasterService* service,
                                           const std::string& key) {
        service->ProtectDfsSource(
            MasterService::ObjectIdentity{TenantId::Default(), key});
    }

    static uint32_t GetDfsPromotionInFlightForTesting(MasterService* service) {
        return service->dfs_promotion_in_flight_.load(
            std::memory_order_relaxed);
    }

    static size_t GetDfsPromotionQueueSizeForTesting(MasterService* service) {
        std::lock_guard<std::mutex> lock(service->dfs_promotion_queue_mutex_);
        return service->dfs_promotion_queue_.size();
    }

    static size_t GetDfsPromotionTaskTableSizeForTesting(
        MasterService* service) {
        size_t total = 0;
        for (size_t i = 0; i < service->metadata_shards_.size(); ++i) {
            // Read every shard under its lock: the background DFS maintenance
            // pass mutates these tables concurrently.
            MasterService::MetadataShardAccessorRO shard(service, i);
            for (const auto& [tenant, state] : shard->tenants) {
                (void)tenant;
                total += state.dfs_promotion_tasks.size();
            }
        }
        return total;
    }

    static bool IsDfsSourceProtectedForTesting(MasterService* service,
                                               const std::string& key) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return false;
        return accessor.GetTenantState().dfs_protected_keys.count(key) > 0;
    }

    // Cancel a queued/claimed task exactly like a completion/cancel would:
    // erase the record, release the DFS source protection, decrement the
    // in-flight counter. The physical queue node is intentionally left behind
    // (design: cancellation never removes queue nodes; a later claim or the
    // oversize rebuild drops them as stale).
    static bool CancelDfsPromotionTaskForTesting(MasterService* service,
                                                 const std::string& key) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return false;
        auto& tenant_state = accessor.GetTenantState();
        if (tenant_state.dfs_promotion_tasks.erase(key) == 0) return false;
        tenant_state.dfs_protected_keys.erase(key);
        service->dfs_promotion_in_flight_.fetch_sub(1,
                                                    std::memory_order_relaxed);
        return true;
    }

    // Age a task's enqueued/claimed timestamps so the TTL reaper sees it as
    // expired without sleeping.
    static void BackdateDfsPromotionTaskForTesting(
        MasterService* service, const std::string& key,
        std::chrono::minutes age) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return;
        auto& record = accessor.GetTenantState().dfs_promotion_tasks.at(key);
        const auto past = std::chrono::steady_clock::now() - age;
        record.enqueued_at = past;
        record.claimed_at = past;
    }

    // ReapDfsPromotionTasks only sweeps a fixed batch of shards per call
    // (round-robin over the whole array); loop enough passes to cover every
    // shard deterministically.
    static void RunDfsPromotionReaperForTesting(MasterService* service) {
        constexpr int kPassesToCoverAllShards = 8;
        for (int i = 0; i < kPassesToCoverAllShards; ++i) {
            service->ReapDfsPromotionTasks();
        }
    }

    static void OverrideDfsPromotionQueueLimitForTesting(
        MasterService* service, uint32_t limit) {
        service->dfs_promotion_queue_limit_ = limit;
    }

    // ---- Metrics / reconcile self-healing / ghost census /
    // anti-thrash cooldown helpers. ----

    // Same funnel as AdmitDfsPromotionForTesting, but returns the raw verdict
    // so tests can tell kCooling apart from the other rejection paths.
    static MasterService::DfsAdmissionResult AdmitDfsPromotionResultForTesting(
        MasterService* service, const std::string& key, double heat = 100.0) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) {
            return MasterService::DfsAdmissionResult::kDisabled;
        }
        return service->TryAdmitDfsPromotion(accessor.GetTenantState(),
                                             accessor.Get(), heat,
                                             service->DfsNowEpochMin(), false);
    }

    // Expose the cached admission threshold so a failing expectation can show
    // the value the gate actually compared against.
    static double GetDfsHeatThresholdForTesting(MasterService* service) {
        return service->GetDfsHeatThreshold(service->DfsNowEpochMin());
    }

    // Drive the DFS heat state machine directly; the read-hit path would need
    // a full GetReplicaList round trip through the RPC layer.
    static void ReconcileDfsHeatForTesting(MasterService* service,
                                           const std::string& key,
                                           bool access_hit) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return;
        service->ReconcileDfsHeat(accessor.Get(), service->DfsNowEpochMin(),
                                  access_hit);
    }

    // Run `fn` on the DFS replica data of `key` while the shard lock is held.
    // Returning a bare DfsReplicaData* (as this helper used to) dropped the
    // lock at the call site, yet the master's background eviction / DFS
    // maintenance threads mutate the very same fields under that lock.
    template <typename Fn>
    static auto WithDfsReplicaDataForTesting(MasterService* service,
                                             const std::string& key, Fn&& fn) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        DfsReplicaData* data = nullptr;
        if (accessor.Exists()) {
            Replica* const dfs =
                accessor.Get().GetFirstReplica(&Replica::fn_is_dfs_replica);
            if (dfs != nullptr) data = dfs->dfs_data();
        }
        return fn(data);
    }

    static void SetDfsLastPromotedMinForTesting(MasterService* service,
                                                const std::string& key,
                                                uint32_t value) {
        WithDfsReplicaDataForTesting(
            service, key, [value](DfsReplicaData* data) {
                if (data != nullptr) data->dfs_last_promoted_min = value;
            });
    }

    static uint32_t GetDfsLastPromotedMinForTesting(MasterService* service,
                                                    const std::string& key) {
        return WithDfsReplicaDataForTesting(
            service, key, [](const DfsReplicaData* data) -> uint32_t {
                return data != nullptr ? data->dfs_last_promoted_min : 0;
            });
    }

    static uint32_t GetDfsLastAccessMinForTesting(MasterService* service,
                                                  const std::string& key) {
        return WithDfsReplicaDataForTesting(
            service, key, [](const DfsReplicaData* data) -> uint32_t {
                return data != nullptr ? data->dfs_last_access_min : 0;
            });
    }

    // Self-healing sweep, ghost census, cooldown config hooks and the
    // private DfsAdmissionResult verdicts (TEST_F bodies are not friended, so
    // they must funnel through this class).
    static void RunDfsPromotionReconcileScanForTesting(MasterService* service) {
        service->RunDfsPromotionReconcileScanForTesting();
    }

    static void RunDfsPromotionGhostMetricsForTesting(MasterService* service) {
        service->UpdateDfsPromotionGhostMetricsForTesting();
    }

    static uint64_t DfsNowEpochMinForTesting(MasterService* service) {
        return service->DfsNowEpochMin();
    }

    static uint32_t GetDfsPromotionCooldownMinForTesting(
        MasterService* service) {
        return service->dfs_promotion_cooldown_min_;
    }

    static MasterService::DfsAdmissionResult DfsAdmissionResultBelowThreshold() {
        return MasterService::DfsAdmissionResult::kBelowThreshold;
    }
    static MasterService::DfsAdmissionResult DfsAdmissionResultCooling() {
        return MasterService::DfsAdmissionResult::kCooling;
    }
    static MasterService::DfsAdmissionResult DfsAdmissionResultAdmitted() {
        return MasterService::DfsAdmissionResult::kAdmitted;
    }

    // Register a client as a LOCAL_DISK holder only (no DRAM segment).
    // This simulates the cross-host case where the LOCAL_DISK source lives
    // on a different node than the DRAM target chosen for promotion.
    UUID PrepareLocalDiskOnlyClient(MasterService& service) const {
        UUID client_id = generate_uuid();
        auto mount_ld = service.MountLocalDiskSegment(client_id, true);
        EXPECT_TRUE(mount_ld.has_value());
        return client_id;
    }

    // Count replicas of `key` matching `pred`. Used by the execution-layer
    // tests to
    // observe PROCESSING / COMPLETE MEMORY replicas and the DFS source.
    static size_t CountReplicasForTesting(
        MasterService* service, const std::string& key,
        const std::function<bool(const Replica&)>& pred) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return 0;
        return accessor.Get().CountReplicas(pred);
    }

    // ---- Coverage hardening helpers for the DFS channel. ----
    // The behaviours that have to be locked down go beyond the happy paths:
    // the heat state machine, decay monotonicity, threshold provenance, every
    // admission gate, queue dispatch and eviction protection. These funnels
    // expose the remaining private state so a TEST_F body can drive it
    // directly.

    static double GetDfsHeatForTesting(MasterService* service,
                                       const std::string& key) {
        return WithDfsReplicaDataForTesting(
            service, key, [](const DfsReplicaData* data) -> double {
                return data != nullptr ? static_cast<double>(data->dfs_heat)
                                       : -1.0;
            });
    }

    // Drive the real read-hit entry point — what GetReplicaList calls once it
    // has decided the read was served by DFS — without going through
    // replica-list construction, so a test can pin the reconcile behaviour
    // itself (heat sampling + admission attempt) rather than the dfs_served
    // predicate that decides whether it runs at all.
    static void ReconcileDfsHeatOnReadForTesting(MasterService* service,
                                                 const std::string& key) {
        service->ReconcileDfsHeatOnRead(
            MasterService::ObjectIdentity{TenantId::Default(), key});
    }

    // Simulate an already-registered sketch member carrying `heat` as of
    // `last_access_min`. The production state machine keeps the metadata and
    // the sketch in lockstep (Add on first membership, Replace on later hits),
    // so this helper registers the matching sample too: otherwise the next
    // ReconcileDfsHeat hit would Replace a sample the sketch never held and the
    // assertion would be testing an impossible state.
    static void SetDfsHeatStateForTesting(MasterService* service,
                                          const std::string& key, float heat,
                                          uint32_t last_access_min) {
        WithDfsReplicaDataForTesting(service, key, [&](DfsReplicaData* data) {
            if (data == nullptr) return;
            if (service->dfs_heat_sketch_ != nullptr && last_access_min != 0) {
                const uint64_t key_hash = MasterService::DfsHeatKeyHash(
                    TenantId::Default().MakeScopedKey(key));
                if (data->dfs_last_access_min == 0) {
                    service->dfs_heat_sketch_->Add(
                        key_hash, static_cast<double>(heat),
                        static_cast<uint64_t>(last_access_min));
                } else {
                    service->dfs_heat_sketch_->Replace(
                        key_hash, static_cast<double>(data->dfs_heat),
                        static_cast<uint64_t>(data->dfs_last_access_min),
                        static_cast<double>(heat),
                        static_cast<uint64_t>(last_access_min));
                }
                // Mirror the production invariant: a registered member always
                // carries the hash cached at registration, so the Remove /
                // Replace paths read it instead of re-hashing the scoped key.
                data->dfs_key_hash = key_hash;
            }
            data->dfs_heat = heat;
            data->dfs_last_access_min = last_access_min;
        });
    }

    // Feed the sketch directly: shaping the P90 / total-weight inputs through
    // real read hits would need hundreds of metadata round trips.
    static bool AddDfsHeatSampleForTesting(MasterService* service,
                                           const std::string& scoped_key,
                                           double heat) {
        return service->dfs_heat_sketch_->Add(
            MasterService::DfsHeatKeyHash(scoped_key), heat,
            service->DfsNowEpochMin());
    }

    static void RefreshDfsHeatThresholdForTesting(MasterService* service) {
        service->RefreshDfsHeatThreshold(service->DfsNowEpochMin());
    }

    // Total weight of the heat sketch: the observable proxy for "how many
    // samples does the collection still hold", used to prove the active-leave
    // path really erased the sample instead of only zeroing the metadata.
    static double GetDfsSketchTotalWeightForTesting(MasterService* service) {
        const auto weight = service->dfs_heat_sketch_->GetTotalWeight(
            service->DfsNowEpochMin());
        return weight.has_value() ? weight.value() : -1.0;
    }

    static bool GetDfsThresholdLowWeightForTesting(MasterService* service) {
        return service->dfs_threshold_low_weight_.load(
            std::memory_order_relaxed);
    }

    static void SetDfsThresholdRefreshMinForTesting(MasterService* service,
                                                    uint32_t value) {
        service->dfs_promotion_threshold_refresh_min_ = value;
    }

    static void SetDfsMaxPerHeartbeatForTesting(MasterService* service,
                                                uint32_t value) {
        service->dfs_promotion_max_per_heartbeat_ = value;
    }

    static void GrantObjectLeaseForTesting(MasterService* service,
                                           const std::string& key,
                                           uint64_t ttl_ms) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return;
        accessor.Get().GrantReadLease(ttl_ms);
    }

    // ReconcileDfsHeat decides "DFS-served" from the *first* COMPLETE replica
    // only, so pushing the DFS replica to the back of the vector is how a test
    // forces the active-leave transition without running a real promotion.
    static bool MoveDfsReplicaLastForTesting(MasterService* service,
                                             const std::string& key) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return false;
        std::vector<Replica> moved =
            accessor.Get().PopReplicas(&Replica::fn_is_dfs_replica);
        if (moved.empty()) return false;
        accessor.Get().AddReplicas(std::move(moved));
        return true;
    }

    // The SSD channel shares the per-key in-flight dedup with DFS; planting a
    // record is enough to exercise the cross-channel branch.
    static void AddSsdPromotionTaskForTesting(MasterService* service,
                                              const std::string& key) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return;
        accessor.GetTenantState().promotion_tasks.emplace(
            key, MasterService::PromotionTask{});
    }

    // Reverse direction of the "one in-flight task per key" rule: attach a
    // COMPLETE LOCAL_DISK replica to an object
    // that already exists, so a DFS-only object becomes eligible for the SSD
    // channel while its DFS task is still in flight.
    static bool AttachLocalDiskReplicaForTesting(MasterService* service,
                                                 const UUID& client_id,
                                                 const std::string& key,
                                                 int64_t size,
                                                 const std::string& endpoint) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return false;
        std::vector<Replica> replicas;
        replicas.emplace_back(client_id, size, endpoint, ReplicaStatus::COMPLETE);
        accessor.Get().AddReplicas(std::move(replicas));
        accessor.GetShard().OnDiskReplicaAdded(accessor.Get());
        return true;
    }

    static size_t GetDfsCandidateCountForTesting(MasterService* service,
                                                 const std::string& key) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) return 0;
        return accessor.GetTenantState().dfs_promotion_candidates.count(key);
    }

    static MasterService::PromotionCandidateReason
    GetDfsCandidateReasonForTesting(MasterService* service,
                                    const std::string& key) {
        const MasterService::ObjectIdentity object_id{TenantId::Default(), key};
        MasterService::MetadataAccessorRW accessor(service, object_id);
        if (!accessor.Exists()) {
            return MasterService::PromotionCandidateReason::kPushFailed;
        }
        auto& candidates = accessor.GetTenantState().dfs_promotion_candidates;
        const auto it = candidates.find(key);
        if (it == candidates.end()) {
            return MasterService::PromotionCandidateReason::kPushFailed;
        }
        return it->second.last_reason;
    }

    static MasterService::DfsAdmissionResult DfsAdmissionResultDisabled() {
        return MasterService::DfsAdmissionResult::kDisabled;
    }
    static MasterService::DfsAdmissionResult DfsAdmissionResultMemoryPresent() {
        return MasterService::DfsAdmissionResult::kMemoryPresent;
    }
    static MasterService::DfsAdmissionResult DfsAdmissionResultNoDfsSource() {
        return MasterService::DfsAdmissionResult::kNoDfsSource;
    }
    static MasterService::DfsAdmissionResult DfsAdmissionResultHardPinned() {
        return MasterService::DfsAdmissionResult::kHardPinned;
    }
    static MasterService::DfsAdmissionResult DfsAdmissionResultLeaseActive() {
        return MasterService::DfsAdmissionResult::kLeaseActive;
    }
    static MasterService::DfsAdmissionResult DfsAdmissionResultQueueCap() {
        return MasterService::DfsAdmissionResult::kQueueCapRejected;
    }
    static MasterService::DfsAdmissionResult DfsAdmissionResultWatermark() {
        return MasterService::DfsAdmissionResult::kWatermarkRejected;
    }
    static MasterService::DfsAdmissionResult DfsAdmissionResultDupInFlight() {
        return MasterService::DfsAdmissionResult::kDupInFlight;
    }
    static MasterService::PromotionCandidateReason DfsCandidateReasonWatermark() {
        return MasterService::PromotionCandidateReason::kWatermark;
    }
    static MasterService::PromotionCandidateReason DfsCandidateReasonQueueCap() {
        return MasterService::PromotionCandidateReason::kQueueCap;
    }

    std::vector<std::string> policy_files_;
    size_t next_policy_file_ = 0;
};

// Sanity: with promotion disabled, no path mutates promotion_objects.
TEST_F(PromotionOnHitTest, DefaultOffNoPromotion) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = false;  // explicitly off
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "test_segment", kDefaultSegmentBase, seg_size);

    PutObject(*service, ctx.client_id, "k1");
    // GetReplicaList many times. With promotion_on_hit=false, nothing should
    // appear in promotion_objects regardless of access count.
    for (int i = 0; i < 5; ++i) {
        auto resp = service->GetReplicaList("k1", TenantId::Default());
        ASSERT_TRUE(resp.has_value());
    }

    auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->size(), 0u);

    service->RemoveAll();
}

// With promotion enabled but no LOCAL_DISK replica present, no promotion
// should fire (the trigger gate any_local_disk is false).
TEST_F(PromotionOnHitTest, NoLocalDiskNoPromotion) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;  // promote on first touch
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "test_segment", kDefaultSegmentBase, seg_size);

    PutObject(*service, ctx.client_id, "k_mem_only");
    for (int i = 0; i < 5; ++i) {
        auto resp = service->GetReplicaList("k_mem_only", TenantId::Default());
        ASSERT_TRUE(resp.has_value());
    }

    auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->size(), 0u)
        << "Memory-only key should not trigger promotion";

    service->RemoveAll();
}

// Single-shot Get on a key with both MEMORY and LOCAL_DISK should not
// promote (any_memory=true → trigger gate fails).
TEST_F(PromotionOnHitTest, MemoryReplicaPresentNoPromotion) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;  // first-touch
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "test_segment", kDefaultSegmentBase, seg_size);

    PutObject(*service, ctx.client_id, "k_dual", 1024);
    // Add a LOCAL_DISK replica next to the MEMORY one.
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, "k_dual", 1024,
                                       ctx.segment_name));

    for (int i = 0; i < 5; ++i) {
        auto resp = service->GetReplicaList("k_dual", TenantId::Default());
        ASSERT_TRUE(resp.has_value());
    }

    auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->size(), 0u)
        << "MEMORY replica still present: should not promote";

    service->RemoveAll();
}

TEST_F(PromotionOnHitTest, BatchGetReplicaListPromotesLocalDiskOnlyObject) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_max_per_heartbeat = 2;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "test_segment", kDefaultSegmentBase, seg_size);

    const std::string single_key = "k_single_get_promote";
    const std::string batch_key = "k_batch_get_promote";
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, single_key,
                                       1024, ctx.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, batch_key, 1024,
                                       ctx.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t admitted_pre = mm.get_promotion_admitted();
    const int64_t in_flight_pre = mm.get_promotion_in_flight();

    auto single_result =
        service->GetReplicaList(single_key, TenantId::Default());
    ASSERT_TRUE(single_result.has_value());
    ASSERT_EQ(single_result->replicas.size(), 1u);
    EXPECT_TRUE(single_result->replicas[0].is_local_disk_replica());

    auto batch_result = service->BatchGetReplicaList(
        std::vector<std::string>{batch_key}, TenantId::Default());
    ASSERT_EQ(batch_result.size(), 1u);
    ASSERT_TRUE(batch_result[0].has_value());
    ASSERT_EQ(batch_result[0]->replicas.size(), 1u);
    EXPECT_TRUE(batch_result[0]->replicas[0].is_local_disk_replica());

    EXPECT_EQ(mm.get_promotion_admitted() - admitted_pre, 2);
    EXPECT_EQ(mm.get_promotion_in_flight() - in_flight_pre, 2);

    auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->size(), 2u);
    EXPECT_EQ(CountPromotionTask(*pending, single_key), 1u);
    EXPECT_EQ(CountPromotionTask(*pending, batch_key), 1u);
    // H12: the SSD channel never fills the DFS-only source snapshot, so a task
    // produced by it must carry no source_dfs at all.
    for (const auto& task : *pending) {
        EXPECT_FALSE(task.source_dfs.has_value()) << "key=" << task.key;
    }

    service->RemoveAll();
}

// The read-only admin batch query must NOT trigger promotion-on-hit, even for a
// LOCAL_DISK-only key. Contrast with BatchGetReplicaListPromotesLocalDiskOnly
// Object above, where the client-facing BatchGetReplicaList admits the key for
// promotion.
TEST_F(PromotionOnHitTest,
       BatchGetReplicaListForAdminDoesNotPromoteLocalDiskOnlyObject) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_max_per_heartbeat = 2;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "test_segment_admin",
                              kDefaultSegmentBase, seg_size);

    const std::string key = "k_batch_admin_no_promote";
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, key, 1024,
                                       ctx.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t admitted_pre = mm.get_promotion_admitted();
    const int64_t in_flight_pre = mm.get_promotion_in_flight();

    auto result = service->BatchGetReplicaListForAdmin(
        std::vector<std::string>{key}, TenantId::Default());
    ASSERT_EQ(result.size(), 1u);
    ASSERT_TRUE(result[0].has_value());
    ASSERT_EQ(result[0]->replicas.size(), 1u);
    EXPECT_TRUE(result[0]->replicas[0].is_local_disk_replica());

    // No promotion admitted or enqueued by the read-only admin path.
    EXPECT_EQ(mm.get_promotion_admitted() - admitted_pre, 0);
    EXPECT_EQ(mm.get_promotion_in_flight() - in_flight_pre, 0);
    auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->size(), 0u);
    EXPECT_EQ(CountPromotionTask(*pending, key), 0u);

    service->RemoveAll();
}

// The read-only admin batch query must NOT update the store-observed cache-hit
// counters. Contrast with the client-facing BatchGetReplicaList, which bumps
// the memory-cache-hit counter for a MEMORY replica.
TEST_F(PromotionOnHitTest,
       BatchGetReplicaListForAdminDoesNotUpdateCacheHitMetrics) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "metrics_segment", kDefaultSegmentBase,
                              seg_size);

    const std::string key = "k_admin_no_metric";
    PutObject(*service, ctx.client_id, key, 1024);

    using CacheHitStat = MasterMetricManager::CacheHitStat;
    auto mem_hits = []() {
        auto stats = MasterMetricManager::instance().calculate_cache_stats();
        return stats[CacheHitStat::MEMORY_HITS];
    };

    const double before_admin = mem_hits();

    // Read-only admin query must leave the memory-cache-hit counter untouched.
    auto admin_result = service->BatchGetReplicaListForAdmin(
        std::vector<std::string>{key}, TenantId::Default());
    ASSERT_EQ(admin_result.size(), 1u);
    ASSERT_TRUE(admin_result[0].has_value());
    EXPECT_EQ(mem_hits(), before_admin);

    // The client-facing path does bump it, proving the assertion above is
    // meaningful rather than a counter that never moves.
    (void)service->BatchGetReplicaList(std::vector<std::string>{key},
                                       TenantId::Default());
    EXPECT_GT(mem_hits(), before_admin);

    service->RemoveAll();
}

// PromotionObjectHeartbeat returns an empty task list when called against a
// client that has no LocalDiskSegment registered.
TEST_F(PromotionOnHitTest, HeartbeatReturnsErrorForUnknownClient) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    auto service = std::make_unique<MasterService>(config);

    UUID unknown_client = generate_uuid();
    auto pending = service->PromotionObjectHeartbeat(unknown_client);
    ASSERT_FALSE(pending.has_value());
    EXPECT_EQ(pending.error(), ErrorCode::SEGMENT_NOT_FOUND);
}

// PromotionAllocStart on a non-existent key returns OBJECT_NOT_FOUND.
TEST_F(PromotionOnHitTest, AllocStartUnknownKey) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    auto service = std::make_unique<MasterService>(config);

    auto resp = service->PromotionAllocStart(generate_uuid(), "nonexistent",
                                             TenantId::Default(), 1024, {});
    ASSERT_FALSE(resp.has_value());
    EXPECT_EQ(resp.error(), ErrorCode::OBJECT_NOT_FOUND);
}

TEST_F(PromotionOnHitTest, InvalidPrimaryEndCannotCompletePromotionReplica) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    Segment holder_segment =
        MakeSegment("issue_3203_end", kDefaultSegmentBase, seg_size);
    const UUID holder_id = generate_uuid();
    ASSERT_TRUE(service->MountSegment(holder_segment, holder_id).has_value());
    ASSERT_TRUE(service->MountLocalDiskSegment(holder_id, true).has_value());
    ASSERT_TRUE(
        service->ReMountSegment({holder_segment}, holder_id).has_value());
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder_id, "k_cold", 1024,
                                       holder_segment.name));

    ReplicateConfig upsert_config;
    upsert_config.replica_num = 1;
    ASSERT_TRUE(service
                    ->UpsertStart(holder_id, "k_cold", TenantId::Default(),
                                  1024, upsert_config)
                    .has_value());
    ASSERT_TRUE(service
                    ->UpsertEnd(holder_id, "k_cold", TenantId::Default(),
                                ReplicaType::LOCAL_DISK)
                    .has_value());

    auto read = service->GetReplicaList("k_cold", TenantId::Default());
    ASSERT_TRUE(read.has_value());
    auto alloc = service->PromotionAllocStart(holder_id, "k_cold",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value());

    // Retrying the completed End is a no-op: it succeeds, and the promotion
    // still owns its PROCESSING MEMORY replica and task.
    ASSERT_TRUE(service
                    ->UpsertEnd(holder_id, "k_cold", TenantId::Default(),
                                ReplicaType::LOCAL_DISK)
                    .has_value());

    // There was no primary PutStart. PutEnd must not complete the staged
    // promotion replica even though the metadata client_id matches.
    auto invalid_put_end = service->PutEnd(
        holder_id, "k_cold", TenantId::Default(), ReplicaType::MEMORY);
    ASSERT_FALSE(invalid_put_end.has_value());
    EXPECT_EQ(invalid_put_end.error(), ErrorCode::INVALID_WRITE);

    auto rejected_upsert = service->UpsertStart(
        holder_id, "k_cold", TenantId::Default(), 1024, upsert_config);
    ASSERT_FALSE(rejected_upsert.has_value());
    EXPECT_EQ(rejected_upsert.error(), ErrorCode::OBJECT_HAS_REPLICATION_TASK);

    // A caller must not be able to finalize after the rejected UpsertStart.
    auto invalid_upsert_end = service->UpsertEnd(
        holder_id, "k_cold", TenantId::Default(), ReplicaType::MEMORY);
    ASSERT_FALSE(invalid_upsert_end.has_value());
    EXPECT_EQ(invalid_upsert_end.error(), ErrorCode::INVALID_WRITE);

    // Only the promotion completion path owns the staged replica.
    auto notify = service->NotifyPromotionSuccess(holder_id, "k_cold",
                                                  TenantId::Default());
    ASSERT_TRUE(notify.has_value());

    service->RemoveAll();
}

TEST_F(PromotionOnHitTest, PrimaryWriteBlocksPromotionAdmission) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder = PrepareSegment(*service, "issue_3203_primary",
                                 kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_cold",
                                       1024, holder.segment_name));

    ReplicateConfig upsert_config;
    upsert_config.replica_num = 1;
    auto upsert = service->UpsertStart(
        holder.client_id, "k_cold", TenantId::Default(), 1024, upsert_config);
    ASSERT_TRUE(upsert.has_value());

    EXPECT_TRUE(PromotionAdmissionBlockedByPrimaryWriteForTesting(
        service.get(), TenantId::Default(), "k_cold"));
    EXPECT_EQ(GetPromotionInFlightForTesting(service.get()), 0u);

    auto end = service->UpsertEnd(holder.client_id, "k_cold",
                                  TenantId::Default(), ReplicaType::LOCAL_DISK);
    ASSERT_TRUE(end.has_value());

    service->RemoveAll();
}

TEST_F(PromotionOnHitTest, StalePromotionReplicaCleanupErasesTask) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    Segment holder_segment =
        MakeSegment("issue_3203_holder", kDefaultSegmentBase, seg_size);
    const UUID holder_id = generate_uuid();
    ASSERT_TRUE(service->MountSegment(holder_segment, holder_id).has_value());
    ASSERT_TRUE(service->MountLocalDiskSegment(holder_id, true).has_value());
    ASSERT_TRUE(
        service->ReMountSegment({holder_segment}, holder_id).has_value());

    Segment target_segment = MakeSegment(
        "issue_3203_target", kDefaultSegmentBase + seg_size, seg_size);
    const UUID target_id = generate_uuid();
    ASSERT_TRUE(service->MountSegment(target_segment, target_id).has_value());

    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder_id, "k_cold", 1024,
                                       holder_segment.name));
    auto read = service->GetReplicaList("k_cold", TenantId::Default());
    ASSERT_TRUE(read.has_value());
    auto alloc = service->PromotionAllocStart(
        holder_id, "k_cold", TenantId::Default(), 1024, {target_segment.name});
    ASSERT_TRUE(alloc.has_value());
    ASSERT_EQ(alloc->memory_descriptor.get_memory_descriptor()
                  .buffer_descriptor.transport_endpoint_,
              target_segment.name);

    // Recreate the legacy bad state: an invalid primary End used to mark the
    // promotion-owned replica COMPLETE before stale-handle cleanup removed it.
    MarkReplicaCompleteForTesting(service.get(), TenantId::Default(), "k_cold",
                                  alloc->memory_descriptor.id);

    auto& metrics = MasterMetricManager::instance();
    const int64_t cancelled_before = metrics.get_promotion_cancelled();
    ASSERT_EQ(GetPromotionInFlightForTesting(service.get()), 1u);

    ASSERT_TRUE(
        service->UnmountSegment(target_segment.id, target_id).has_value());

    constexpr auto kCleanupTimeout = std::chrono::seconds(5);
    constexpr auto kPollInterval = std::chrono::milliseconds(10);
    const auto cleanup_deadline =
        std::chrono::steady_clock::now() + kCleanupTimeout;
    while (HasPromotionTaskForTesting(service.get(), TenantId::Default(),
                                      "k_cold") &&
           std::chrono::steady_clock::now() < cleanup_deadline) {
        std::this_thread::sleep_for(kPollInterval);
    }
    ASSERT_FALSE(HasPromotionTaskForTesting(service.get(), TenantId::Default(),
                                            "k_cold"));
    EXPECT_EQ(GetPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_EQ(metrics.get_promotion_cancelled() - cancelled_before, 1);
    auto pending = service->PromotionObjectHeartbeat(holder_id);
    ASSERT_TRUE(pending.has_value());
    EXPECT_TRUE(pending->empty());

    auto notify = service->NotifyPromotionSuccess(holder_id, "k_cold",
                                                  TenantId::Default());
    ASSERT_FALSE(notify.has_value());
    EXPECT_EQ(notify.error(), ErrorCode::REPLICA_IS_NOT_READY);

    auto remaining =
        service->GetReplicaListForAdmin("k_cold", TenantId::Default());
    ASSERT_TRUE(remaining.has_value());
    ASSERT_EQ(remaining->replicas.size(), 1u);
    EXPECT_TRUE(remaining->replicas[0].is_local_disk_replica());

    service->RemoveAll();
}

// NotifyPromotionSuccess on a non-existent key returns OBJECT_NOT_FOUND.
TEST_F(PromotionOnHitTest, NotifyUnknownKey) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    auto service = std::make_unique<MasterService>(config);

    UUID client_id = generate_uuid();
    auto resp = service->NotifyPromotionSuccess(client_id, "nonexistent",
                                                TenantId::Default());
    ASSERT_FALSE(resp.has_value());
    EXPECT_EQ(resp.error(), ErrorCode::OBJECT_NOT_FOUND);
}

// Concurrent readers racing into TryPushPromotionQueue must dedupe to a
// single PromotionTask. Without dedup, the source LOCAL_DISK replica's
// refcnt would be incremented N times and the per-segment promotion_objects
// map would either error on the second insert (current behavior) or
// double-queue.
TEST_F(PromotionOnHitTest, RacingReadersDedup) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;  // first-touch fires the gate
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "test_segment", kDefaultSegmentBase, seg_size);

    // LOCAL_DISK-only key: NotifyOffloadSuccess on a never-PUT key creates
    // metadata with only the LOCAL_DISK replica (AddReplica's Create-on-
    // missing path).
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, "k_cold", 1024,
                                       ctx.segment_name));

    constexpr int kThreads = 32;
    constexpr int kReadsPerThread = 10;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&service]() {
            for (int j = 0; j < kReadsPerThread; ++j) {
                auto r = service->GetReplicaList("k_cold", TenantId::Default());
                EXPECT_TRUE(r.has_value());
            }
        });
    }
    for (auto& t : threads) t.join();

    auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->size(), 1u)
        << "Concurrent readers (" << kThreads << " x " << kReadsPerThread
        << ") must dedupe to a single promotion task";
    ASSERT_TRUE(CountPromotionTask(*pending, "k_cold"));

    service->RemoveAll();
}

// A PromotionTask that never receives NotifyPromotionSuccess must be
// reaped after `put_start_release_timeout_sec` so that:
//   (a) the source LOCAL_DISK replica's refcnt is decremented (no
//       permanent pin → eviction can still free it later);
//   (b) the dedup gate is unblocked, so a subsequent read can re-enqueue
//       the same key for retry.
TEST_F(PromotionOnHitTest, StalePromotionReaper) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    // Short staleness window: the reaper uses put_start_release_timeout_sec
    // for promotion tasks. The master enforces release > discard, so set
    // discard to a still-smaller value.
    config.put_start_discard_timeout_sec = 0;
    config.put_start_release_timeout_sec = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "test_segment", kDefaultSegmentBase, seg_size);

    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, "k_cold", 1024,
                                       ctx.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t expired_pre = mm.get_promotion_expired();

    // Trigger #1: enqueue, then drain the per-segment queue. Drain leaves
    // the per-shard PromotionTask intact (the heartbeat is best-effort GC,
    // not the authoritative state).
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
        auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
        ASSERT_TRUE(pending.has_value());
        EXPECT_EQ(pending->size(), 1u);
    }

    // Without reap, dedup blocks re-enqueue. Confirm: GetReplicaList again,
    // heartbeat must be empty because PromotionTask still pins the slot.
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
        auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
        ASSERT_TRUE(pending.has_value());
        EXPECT_EQ(pending->size(), 0u)
            << "Dedup gate should block re-enqueue while task is in flight";
    }

    // Wait past the staleness window. The eviction thread reaps every
    // kEvictionThreadSleepMs (10 ms) but gated by
    // `now - last_discard_time > put_start_release_timeout_sec_` (strict).
    // With release=1s a 2s sleep leaves only ~1s margin between reaper
    // firing and the wake-up, which has flaked under CI load. 3s gives
    // ~2s of margin and matches the schedule's strict-greater comparison.
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Trigger #3: with the task reaped, dedup is unblocked and a fresh
    // GetReplicaList must enqueue again.
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
        auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
        ASSERT_TRUE(pending.has_value());
        EXPECT_EQ(pending->size(), 1u)
            << "After reap, a fresh read must re-enqueue the same key";
    }

    EXPECT_EQ(mm.get_promotion_expired() - expired_pre, 1)
        << "Reaper expiry must bump promotion_expired";

    service->RemoveAll();
}

// Force-Remove on a key with a queued PromotionTask must not corrupt
// state. The PromotionTask references a Replica*; once the metadata is
// gone, NotifyPromotionSuccess and the reaper must both tolerate the
// missing entry.
TEST_F(PromotionOnHitTest, RemoveDuringPromotion) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 100;  // short lease: Remove won't block
    config.put_start_discard_timeout_sec = 0;
    config.put_start_release_timeout_sec = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "test_segment", kDefaultSegmentBase, seg_size);

    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, "k_cold", 1024,
                                       ctx.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t cancelled_pre = mm.get_promotion_cancelled();

    // Queue a promotion task.
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // Lease must expire before we can call non-force Remove (or use force).
    auto rm = service->Remove("k_cold", TenantId::Default(), /*force=*/true);
    // Remove returns REPLICA_IS_NOT_READY if any replica is non-COMPLETE.
    // The injected LOCAL_DISK replica is COMPLETE, so this should succeed.
    ASSERT_TRUE(rm.has_value())
        << "Remove on a LOCAL_DISK-only key with a queued promotion should "
        << "succeed (all replicas COMPLETE); error=" << rm.error();
    EXPECT_EQ(mm.get_promotion_cancelled() - cancelled_pre, 1)
        << "Remove of a key with an in-flight promotion task must bump "
        << "promotion_cancelled";

    // NotifyPromotionSuccess on the now-removed key must surface the missing
    // metadata cleanly, not crash.
    auto notify = service->NotifyPromotionSuccess(ctx.client_id, "k_cold",
                                                  TenantId::Default());
    ASSERT_FALSE(notify.has_value());
    EXPECT_EQ(notify.error(), ErrorCode::OBJECT_NOT_FOUND);

    // Wait for the reaper; it must tolerate the missing metadata entry
    // (the source replica it would dec_refcnt is already gone). 3s for
    // CI-safe margin over the 1s release timeout.
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Re-injecting the key and re-triggering must work end-to-end, proving
    // the per-shard PromotionTask was reaped (not stuck).
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, "k_cold", 1024,
                                       ctx.segment_name));
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
        auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
        ASSERT_TRUE(pending.has_value());
        EXPECT_EQ(pending->size(), 1u)
            << "After Remove + reap, the same key must re-enqueue cleanly";
    }

    service->RemoveAll();
}

// Cross-host promotion: the LOCAL_DISK source's holder client has no DRAM
// segment of its own. The reader's DRAM segment (segment_b) is the only
// allocator candidate, so PromotionAllocStart must place the new MEMORY
// replica there — proving the master's allocation strategy crosses the
// host boundary cleanly.
TEST_F(PromotionOnHitTest, MultiSegmentAllocPicksAvailableSegment) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    // Holder: LOCAL_DISK only, no DRAM. Stands in for "host A" that has the
    // SSD copy but no free RAM.
    UUID holder_client_id = PrepareLocalDiskOnlyClient(*service);

    // Reader-side DRAM target: "host B" with a fresh DRAM segment.
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto reader_ctx =
        PrepareSegment(*service, "segment_b", kDefaultSegmentBase, seg_size);

    // The cold key: only replica is LOCAL_DISK on the holder client.
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder_client_id, "k_cold",
                                       1024, "segment_a_endpoint"));

    // Seed the PromotionTask through the gate — PromotionAllocStart now
    // requires an in-flight task to exist (rejects orphaned-stage path).
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // PromotionAllocStart — test the segment-selection logic on top of
    // the gate-seeded task.
    auto resp = service->PromotionAllocStart(holder_client_id, "k_cold",
                                             TenantId::Default(), 1024, {});
    ASSERT_TRUE(resp.has_value())
        << "PromotionAllocStart should succeed when any DRAM segment has "
        << "capacity; error=" << resp.error();

    const auto& mem_desc = resp.value().memory_descriptor;
    ASSERT_TRUE(mem_desc.is_memory_replica());
    EXPECT_EQ(
        mem_desc.get_memory_descriptor().buffer_descriptor.transport_endpoint_,
        reader_ctx.segment_name)
        << "New MEMORY replica must be allocated on the only DRAM segment "
        << "(segment_b), not on the LOCAL_DISK holder which has no DRAM";

    // Sanity: the staged MEMORY replica is PROCESSING (visible only after
    // NotifyPromotionSuccess flips it COMPLETE).
    EXPECT_EQ(mem_desc.status, ReplicaStatus::PROCESSING);

    service->RemoveAll();
}

// preferred_segments is honored: promotion can be steered to a specific
// DRAM segment (e.g., the reader's local one). TryPushPromotionQueue
// currently doesn't pass preferred_segments through, but
// PromotionAllocStart respects them for clients biasing toward locality.
TEST_F(PromotionOnHitTest, MultiSegmentAllocRespectsPreferred) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg_a =
        PrepareSegment(*service, "segment_a", kDefaultSegmentBase, seg_size);
    auto seg_b = PrepareSegment(*service, "segment_b",
                                kDefaultSegmentBase + seg_size, seg_size);

    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg_a.client_id, "k_cold",
                                       1024, seg_a.segment_name));

    // Seed the PromotionTask through the gate so AllocStart's
    // task-existence check passes.
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    auto resp = service->PromotionAllocStart(seg_a.client_id, "k_cold",
                                             TenantId::Default(), 1024,
                                             {seg_b.segment_name});
    ASSERT_TRUE(resp.has_value());
    const auto& mem_desc = resp.value().memory_descriptor;
    EXPECT_EQ(
        mem_desc.get_memory_descriptor().buffer_descriptor.transport_endpoint_,
        seg_b.segment_name)
        << "preferred_segments={segment_b} should pin the new MEMORY "
        << "replica to segment_b";

    service->RemoveAll();
}

// promotion_queue_limit caps total in-flight tasks cluster-wide
// (gate: promotion_in_flight_ >= limit). With limit=1 the very first
// queued task saturates the cap, so a second LOCAL_DISK-only read —
// even on a key in the same shard — must be silently dropped by the
// cap gate (reads still succeed; just no new task is enqueued).
// QueueLimitRejectsCrossShard covers the same-cap-across-different-
// shards case.
TEST_F(PromotionOnHitTest, QueueLimitRejectsBeyondCap) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;  // any 1 task saturates a shard
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);

    // Find two keys that hash to the same shard. MasterService::
    // getShardIndex is private but the formula is deterministic
    // (std::hash<std::string>{}(key) % kNumShards), so we can mirror
    // it here. kNumShards=1024 (master_service.h:889).
    constexpr size_t kNumShardsLocal = 1024;
    auto shard_of = [](const std::string& k) {
        return std::hash<std::string>{}(k) % kNumShardsLocal;
    };
    const std::string k1 = "qlim_first";
    std::string k2;
    for (int i = 0; i < 100000 && k2.empty(); ++i) {
        std::string candidate = "qlim_collide_" + std::to_string(i);
        if (shard_of(candidate) == shard_of(k1)) {
            k2 = candidate;
        }
    }
    ASSERT_FALSE(k2.empty())
        << "could not find a same-shard collision for " << k1;

    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, k1, 1024,
                                       seg.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, k2, 1024,
                                       seg.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t cap_rej_pre = mm.get_promotion_rejected_cap();

    // First read on k1 enqueues a task in shard S.
    auto r1 = service->GetReplicaList(k1, TenantId::Default());
    ASSERT_TRUE(r1.has_value());

    // Second read on k2 (same shard S, different key, so no dedup) must
    // be dropped by the cap gate: the cluster-wide in-flight counter is
    // already 1, which meets promotion_queue_limit_ = 1.
    auto r2 = service->GetReplicaList(k2, TenantId::Default());
    ASSERT_TRUE(r2.has_value()) << "read itself must still succeed; "
                                << "queue gate is silent";

    // Drain the holder's queue: only k1 should appear.
    auto heartbeat = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(heartbeat.has_value());
    EXPECT_EQ(heartbeat->size(), 1u)
        << "promotion_queue_limit=1 should admit only the first task "
        << "globally; k2's enqueue must be dropped";
    EXPECT_EQ(CountPromotionTask(*heartbeat, k1), 1u)
        << "k1 was read first and should be the surviving task";
    EXPECT_EQ(CountPromotionTask(*heartbeat, k2), 0u)
        << "k2 was rejected by the cap gate; should not appear";
    EXPECT_EQ(mm.get_promotion_rejected_cap() - cap_rej_pre, 1)
        << "k2's rejection must increment promotion_rejected_cap";

    service->RemoveAll();
}

// PromotionObjectHeartbeat caps the per-call response at kMaxPerHeartbeat
// (1) and leaves the remainder queued for subsequent heartbeats. This is
// the master-side bound that keeps the client's heartbeat thread inside
// the liveness window when many keys are queued, while guaranteeing no
// task is dropped — a regression from the prior client-side cap that
// silently discarded leftovers.
TEST_F(PromotionOnHitTest, HeartbeatBoundedBatchPreservesLeftovers) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);

    // Push three keys into the holder's promotion_objects via the trigger
    // path. Each is a LOCAL_DISK-only key (no MEMORY replica), so the
    // first GetReplicaList per key crosses the admission threshold (=1)
    // and TryPushPromotionQueue enqueues a task.
    const std::vector<std::string> keys{"hb_k1", "hb_k2", "hb_k3"};
    for (const auto& k : keys) {
        ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, k, 1024,
                                           seg.segment_name));
        auto r = service->GetReplicaList(k, TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // First heartbeat returns at most 1 key.
    auto tick1 = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(tick1.has_value());
    EXPECT_EQ(tick1->size(), 1u)
        << "heartbeat should return at most kMaxPerHeartbeat=1 entry";
    std::string first_key = tick1->begin()->key;
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), first_key) != keys.end());

    // Second heartbeat returns another key (a different one).
    auto tick2 = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(tick2.has_value());
    EXPECT_EQ(tick2->size(), 1u);
    std::string second_key = tick2->begin()->key;
    EXPECT_NE(second_key, first_key)
        << "second heartbeat must drain a different leftover key, not "
        << "re-return the one already extracted";

    // Third heartbeat returns the third key.
    auto tick3 = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(tick3.has_value());
    EXPECT_EQ(tick3->size(), 1u);
    std::string third_key = tick3->begin()->key;
    EXPECT_NE(third_key, first_key);
    EXPECT_NE(third_key, second_key);

    // Fourth heartbeat: queue is now empty.
    auto tick4 = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(tick4.has_value());
    EXPECT_TRUE(tick4->empty())
        << "after draining all queued keys, heartbeat must return empty";

    // Sanity: master-side promotion_tasks records are intact for all keys
    // (they're cleared by NotifyPromotionSuccess, not by Heartbeat), so the
    // source refcnts remain pinned until processed.
    for (const auto& k : keys) {
        auto rl = service->GetReplicaList(k, TenantId::Default());
        ASSERT_TRUE(rl.has_value()) << "key " << k << " should still exist";
    }

    service->RemoveAll();
}

// The promotion task reaper must pop the staged PROCESSING MEMORY
// replica added by PromotionAllocStart. The staged replica is not in
// shard->processing_keys, so DiscardExpiredProcessingReplicas's main
// sweep can't see it, and the reaper for promotion tasks is the only
// place that knows the replica exists. Without this path the orphan
// holds its allocator buffer indefinitely (until the object is removed
// or evicted).
//
// We can't observe the staged replica via GetReplicaList because the
// master filters out PROCESSING entries (clients can only read COMPLETE
// replicas), so we use QuerySegments to watch the DRAM allocator's used
// bytes: AllocStart bumps it, and the reaper must return it to baseline.
// NotifyPromotionSuccess on a reaped task must also fail cleanly.
TEST_F(PromotionOnHitTest, ReaperPopsStagedMemoryReplicaOnExpiry) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_multi_tenants = true;
    config.tenant_quota_connector_type = "file";
    config.tenant_quota_connector_uri =
        WriteTenantQuotaPolicyFile({{TenantId::Default().value(), 4096}});
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    config.put_start_discard_timeout_sec = 0;
    config.put_start_release_timeout_sec = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, "k_cold", 1024,
                                       ctx.segment_name));

    // Baseline allocator usage on the DRAM segment.
    auto seg_baseline = service->QuerySegments(ctx.segment_name);
    ASSERT_TRUE(seg_baseline.has_value());
    const size_t used_baseline = seg_baseline->first;

    // Trigger the gate to enqueue a PromotionTask.
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    // Drive the AllocStart side so alloc_id != 0 — this is the exact
    // setup that produces an orphaned PROCESSING MEMORY replica if the
    // reaper does not pop it.
    auto alloc = service->PromotionAllocStart(ctx.client_id, "k_cold",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value());
    ASSERT_EQ(
        service->GetTenantQuotaSnapshot(TenantId::Default())->charged_bytes,
        1024);

    // After AllocStart, the DRAM allocator must have committed bytes for
    // the staged PROCESSING MEMORY replica.
    auto seg_after_alloc = service->QuerySegments(ctx.segment_name);
    ASSERT_TRUE(seg_after_alloc.has_value());
    EXPECT_GT(seg_after_alloc->first, used_baseline)
        << "PromotionAllocStart should bump segment used bytes "
        << "(allocator-tracked PROCESSING MEMORY replica)";

    // Wait past the staleness window; the eviction thread reaps the
    // task and (with the fix) pops the staged replica via
    // EraseReplicaByID, which releases the buffer back to the allocator.
    // Poll instead of a fixed sleep — the eviction thread cadence and
    // the test runner's scheduling jitter (especially under suite load)
    // both vary, so a hard 2s sleep is flaky here.
    constexpr auto kPollDeadline = std::chrono::seconds(5);
    constexpr auto kPollInterval = std::chrono::milliseconds(50);
    const auto poll_start = std::chrono::steady_clock::now();
    size_t used_after_reap = 0;
    while (std::chrono::steady_clock::now() - poll_start < kPollDeadline) {
        auto q = service->QuerySegments(ctx.segment_name);
        ASSERT_TRUE(q.has_value());
        used_after_reap = q->first;
        if (used_after_reap == used_baseline) break;
        std::this_thread::sleep_for(kPollInterval);
    }
    EXPECT_EQ(used_after_reap, used_baseline)
        << "after reap: staged PROCESSING MEMORY replica's buffer must "
        << "be freed back to the DRAM allocator. If this fires, the "
        << "reaper is not popping the staged replica and the buffer "
        << "leaks until the object itself is removed or evicted.";
    EXPECT_EQ(
        service->GetTenantQuotaSnapshot(TenantId::Default())->charged_bytes, 0);

    // NotifyPromotionSuccess for a reaped task must not commit anything
    // and must return REPLICA_IS_NOT_READY (the task entry is gone, so
    // the alloc_id lookup at the top of NotifyPromotionSuccess fails
    // fast).
    auto notify = service->NotifyPromotionSuccess(ctx.client_id, "k_cold",
                                                  TenantId::Default());
    ASSERT_FALSE(notify.has_value());
    EXPECT_EQ(notify.error(), ErrorCode::REPLICA_IS_NOT_READY);

    service->RemoveAll();
}

// The cap gate must be cluster-wide, not per-shard. Promotion targets
// skewed hot keys, which by definition cluster into a small number of
// shards; a `shard->size() * kNumShards >= limit` heuristic fires
// roughly kNumShards-times too eagerly on that workload. With a global
// atomic counter, a task in shard A counts toward the cap that gates a
// task in shard B.
TEST_F(PromotionOnHitTest, QueueLimitRejectsCrossShard) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;  // 1 in-flight task globally
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);

    // Find two keys hashing to *different* shards. With the old per-shard
    // heuristic this would let both through (each shard's count is 0
    // independently). With the global counter, only the first goes in.
    constexpr size_t kNumShardsLocal = 1024;
    auto shard_of = [](const std::string& k) {
        return std::hash<std::string>{}(k) % kNumShardsLocal;
    };
    const std::string k1 = "xshard_first";
    std::string k2;
    for (int i = 0; i < 100000 && k2.empty(); ++i) {
        std::string candidate = "xshard_other_" + std::to_string(i);
        if (shard_of(candidate) != shard_of(k1)) {
            k2 = candidate;
        }
    }
    ASSERT_FALSE(k2.empty()) << "couldn't find a different-shard key";
    ASSERT_NE(shard_of(k1), shard_of(k2));

    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, k1, 1024,
                                       seg.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, k2, 1024,
                                       seg.segment_name));

    auto r1 = service->GetReplicaList(k1, TenantId::Default());
    ASSERT_TRUE(r1.has_value());

    // k2 lives in a different shard, but the global cap is already met
    // by k1's task — k2 must be rejected.
    auto r2 = service->GetReplicaList(k2, TenantId::Default());
    ASSERT_TRUE(r2.has_value()) << "read itself still succeeds";

    auto heartbeat = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(heartbeat.has_value());
    EXPECT_EQ(heartbeat->size(), 1u)
        << "with global cap=1 and one task already in shard " << shard_of(k1)
        << ", a key hashing to shard " << shard_of(k2)
        << " must be rejected by the global gate. A per-shard heuristic "
        << "would admit it here since the destination shard's local "
        << "count is 0.";
    EXPECT_EQ(CountPromotionTask(*heartbeat, k1), 1u);
    EXPECT_EQ(CountPromotionTask(*heartbeat, k2), 0u);

    service->RemoveAll();
}

// PromotionAllocStart must reset PromotionTask.start_time so the reaper
// TTL covers the active-transfer phase on its own and is not consumed by
// the queue-wait phase. Without the reset, a task that waited in the
// holder's promotion_objects queue for most of the original TTL could
// enter active transfer with little budget left; if the SSD read + RDMA
// write of a large object then ran past expiry, the reaper would
// EraseReplicaByID on the staged MEMORY replica mid-flight and the
// allocator could hand the freed buffer to a concurrent Put while the
// client's RDMA write is still landing into it (use-after-free in the
// allocator + silent data corruption for the concurrent Put).
//
// We can't directly observe start_time from a black-box test, so we
// arrange the timing so the reset is the only thing that distinguishes
// "task still alive" from "task reaped" at the assertion points:
//
//   T=0    : admit task   (original start_time = T=0)
//   T=Wq   : AllocStart   (resets start_time = T=Wq)
//   T=Wq+Wa: assertion 1  -- alive iff the reset happened
//                            (Wq + Wa > TTL,  so without reset the
//                             original start_time has aged past TTL)
//                            (Wa < TTL,        so with the reset the
//                             new start_time has NOT aged past TTL)
//   T=Wq+Wb: assertion 2  -- reaped
//                            (Wb > TTL, so even with the reset the
//                             active-transfer phase has now exceeded its
//                             own full window)
//
// Concretely with TTL = 2s, Wq = 1.5s, Wa = 1.5s (-> 3.0s elapsed,
// 1.5s since AllocStart), Wb = 3.0s (-> 4.5s elapsed, 3.0s since
// AllocStart). The "alive" assertion at Wq+Wa is the one that proves
// the reset is wired up correctly.
//
// QuerySegments(seg).first (used bytes) is the observable: AllocStart
// bumps it, the reaper's EraseReplicaByID returns it to baseline. We
// can't use GetReplicaList because the master filters out PROCESSING
// replicas from the response.
TEST_F(PromotionOnHitTest, AllocStartResetsTaskDeadline) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 8000;
    config.put_start_discard_timeout_sec = 0;
    config.put_start_release_timeout_sec = 2;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, "k_late", 1024,
                                       ctx.segment_name));

    auto seg_baseline = service->QuerySegments(ctx.segment_name);
    ASSERT_TRUE(seg_baseline.has_value());
    const size_t used_baseline = seg_baseline->first;

    // T=0 : admit. start_time = T=0.
    {
        auto r = service->GetReplicaList("k_late", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // T=Wq : simulate the holder's queue having been backlogged. With
    // TTL = 2s and Wq = 1.5s the task is still alive at AllocStart
    // (the queue-wait phase's own window hasn't expired yet — 1.5 < 2).
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    auto alloc = service->PromotionAllocStart(ctx.client_id, "k_late",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value())
        << "AllocStart must succeed before the queue-wait phase's TTL "
        << "expires (1.5s elapsed, TTL is 2s). If this fires the test "
        << "timing is wrong, not the feature.";

    // Buffer has been staged.
    auto seg_after_alloc = service->QuerySegments(ctx.segment_name);
    ASSERT_TRUE(seg_after_alloc.has_value());
    EXPECT_GT(seg_after_alloc->first, used_baseline)
        << "PromotionAllocStart should bump segment used bytes "
        << "(allocator-tracked PROCESSING MEMORY replica)";

    // T=Wq+Wa : total elapsed since admission is 3.0s > TTL=2s. If the
    // reset is missing, the reaper sees the original start_time = 0
    // and expires the task here, calling EraseReplicaByID which would
    // free the staged buffer mid-RDMA-write in production. With the
    // reset, start_time was bumped at AllocStart so age-since-reset is
    // only 1.5s < TTL=2s and the task stays alive. This is the
    // assertion that proves the reset is in place.
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    auto seg_after_wait = service->QuerySegments(ctx.segment_name);
    ASSERT_TRUE(seg_after_wait.has_value());
    EXPECT_GT(seg_after_wait->first, used_baseline)
        << "post-AllocStart staged MEMORY buffer must still be alive: "
        << "queue-wait + active-transfer wall time (3.0s) has exceeded "
        << "the bare TTL (2s), but the start_time reset at AllocStart "
        << "gives the active-transfer phase its own fresh TTL window. "
        << "If this fires, the reset in PromotionAllocStart is missing "
        << "or broken — without it the reaper frees the staged buffer "
        << "here while a client's RDMA write could still be in flight.";

    // T=Wq+Wb : sleep another 3s for a total of 4.5s since AllocStart.
    // The active-transfer phase's own TTL has now expired regardless
    // of the reset; the reaper must fire and free the staged buffer.
    std::this_thread::sleep_for(std::chrono::seconds(3));
    auto seg_after_reap = service->QuerySegments(ctx.segment_name);
    ASSERT_TRUE(seg_after_reap.has_value());
    EXPECT_EQ(seg_after_reap->first, used_baseline)
        << "after the active-transfer phase's own TTL expires "
        << "(4.5s > 2s since AllocStart), the reaper must fire and "
        << "free the staged buffer via EraseReplicaByID";

    service->RemoveAll();
}

// NotifyPromotionSuccess must decrement the cluster-wide in-flight
// counter so future admissions can use the slot; otherwise queue_limit
// remains saturated forever after the first successful promotion. Also
// exercises the only success-path end-to-end coverage of
// NotifyPromotionSuccess in the suite.
TEST_F(PromotionOnHitTest, NotifySuccessDecrementsCounter) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;  // 1 in-flight task globally
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);

    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_first", 1024,
                                       seg.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_second",
                                       1024, seg.segment_name));

    // Admit task 1. promotion_in_flight_ goes from 0 -> 1.
    {
        auto r = service->GetReplicaList("k_first", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // Drive the full success path: AllocStart stages the PROCESSING MEMORY
    // replica and records alloc_id; NotifyPromotionSuccess flips it
    // COMPLETE, drops the source LOCAL_DISK refcnt, erases the task, and
    // decrements the counter.
    auto alloc = service->PromotionAllocStart(seg.client_id, "k_first",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value())
        << "AllocStart should succeed; error=" << alloc.error();
    auto notify = service->NotifyPromotionSuccess(seg.client_id, "k_first",
                                                  TenantId::Default());
    ASSERT_TRUE(notify.has_value())
        << "NotifyPromotionSuccess happy path should succeed; if this "
        << "fires, AllocStart did not record alloc_id, or the staged "
        << "replica is missing/non-PROCESSING; error=" << notify.error();

    // Counter must now be 0. A second admission on a *different* key must
    // be allowed. If fetch_sub is missing on the success path, the cap is
    // still saturated at 1 and TryPushPromotionQueue silently drops this
    // attempt.
    {
        auto r = service->GetReplicaList("k_second", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto pending = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->size(), 1u)
        << "k_second must be admitted after k_first succeeds — the global "
        << "in-flight counter must decrement on the NotifyPromotionSuccess "
        << "success path. Without fetch_sub, the cap stays saturated and "
        << "k_second is silently dropped.";
    EXPECT_EQ(CountPromotionTask(*pending, "k_second"), 1u);

    service->RemoveAll();
}

TEST_F(PromotionOnHitTest, TenantQuotaChargesAtAllocStartAndSettlesLifecycle) {
    const TenantId tenant_id("tenant-a");
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_multi_tenants = true;
    config.tenant_quota_connector_type = "file";
    config.tenant_quota_connector_uri =
        WriteTenantQuotaPolicyFile({{tenant_id.value(), 4096}});
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg =
        PrepareSegment(*service, "seg_quota", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "success", 1024,
                                       seg.segment_name, tenant_id.value()));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "failure", 1024,
                                       seg.segment_name, tenant_id.value()));

    ASSERT_TRUE(service->GetReplicaList("success", tenant_id));
    ASSERT_TRUE(service->PromotionAllocStart(seg.client_id, "success",
                                             tenant_id, 1024, {}));
    ASSERT_EQ(service->GetTenantQuotaSnapshot(tenant_id)->charged_bytes, 1024);
    ASSERT_TRUE(
        service->NotifyPromotionSuccess(seg.client_id, "success", tenant_id));
    EXPECT_EQ(service->GetTenantQuotaSnapshot(tenant_id)->charged_bytes, 1024);

    ASSERT_TRUE(service->GetReplicaList("failure", tenant_id));
    ASSERT_TRUE(service->PromotionAllocStart(seg.client_id, "failure",
                                             tenant_id, 1024, {}));
    ASSERT_EQ(service->GetTenantQuotaSnapshot(tenant_id)->charged_bytes, 2048);
    ASSERT_TRUE(
        service->NotifyPromotionFailure(seg.client_id, "failure", tenant_id));
    EXPECT_EQ(service->GetTenantQuotaSnapshot(tenant_id)->charged_bytes, 1024);

    ASSERT_TRUE(service->Remove("success", tenant_id, /*force=*/true));
    EXPECT_EQ(service->GetTenantQuotaSnapshot(tenant_id)->charged_bytes, 0);
    service->RemoveAll();
}

TEST_F(PromotionOnHitTest, UpsertStartRejectsActivePromotionTask) {
    const TenantId tenant_id("tenant-a");
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_multi_tenants = true;
    config.tenant_quota_connector_type = "file";
    config.tenant_quota_connector_uri =
        WriteTenantQuotaPolicyFile({{tenant_id.value(), 4096}});
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    constexpr uint64_t object_size = 1024;
    auto seg =
        PrepareSegment(*service, "seg_quota", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "key",
                                       object_size, seg.segment_name,
                                       tenant_id.value()));

    ASSERT_TRUE(service->GetReplicaList("key", tenant_id));
    ASSERT_TRUE(service->PromotionAllocStart(seg.client_id, "key", tenant_id,
                                             object_size, {}));
    ASSERT_EQ(service->GetTenantQuotaSnapshot(tenant_id)->charged_bytes,
              object_size);

    ReplicateConfig replicate_config;
    replicate_config.replica_num = 1;
    auto upsert = service->UpsertStart(seg.client_id, "key", tenant_id,
                                       object_size, replicate_config);
    ASSERT_FALSE(upsert.has_value());
    EXPECT_EQ(upsert.error(), ErrorCode::OBJECT_HAS_REPLICATION_TASK);

    ASSERT_TRUE(
        service->NotifyPromotionSuccess(seg.client_id, "key", tenant_id));
    EXPECT_EQ(service->GetTenantQuotaSnapshot(tenant_id)->charged_bytes,
              object_size);
    service->RemoveAll();
}

// PromotionAllocStart must reject when the in-flight task has been
// reaped between the holder's heartbeat and the AllocStart RPC arriving
// (e.g. client stall past put_start_release_timeout_sec_). Without the
// check, AllocStart would allocate + AddReplicas a PROCESSING MEMORY
// replica with nothing tracking it: the generic PROCESSING reaper only
// iterates shard->processing_keys (never populated by promotion) and
// the promotion-task reaper has nothing left to iterate, so the buffer
// leaks until the object is removed or evicted.
TEST_F(PromotionOnHitTest, AllocStartRejectsReapedTask) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    config.put_start_discard_timeout_sec = 0;
    config.put_start_release_timeout_sec = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, "k_cold", 1024,
                                       ctx.segment_name));

    // Baseline allocator usage.
    auto seg_baseline = service->QuerySegments(ctx.segment_name);
    ASSERT_TRUE(seg_baseline.has_value());
    const size_t used_baseline = seg_baseline->first;

    // Admit the task.
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // Wait past the TTL so the reaper sweeps the task. AllocStart has
    // not yet been called, so alloc_id is 0; the reaper's
    // EraseReplicaByID branch is a no-op and only the task entry is
    // removed. We can't easily poll for reap externally (the only
    // user-facing observable would be re-admitting through the gate,
    // which would create a fresh task and defeat the test), so use a
    // fixed sleep with margin. 3s sleep over 1s release timeout matches
    // StalePromotionReaper's CI-safe margin.
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // AllocStart on a reaped task must reject without allocating.
    // Allocating would leave an orphaned PROCESSING MEMORY replica
    // attached to the object.
    auto alloc = service->PromotionAllocStart(ctx.client_id, "k_cold",
                                              TenantId::Default(), 1024, {});
    ASSERT_FALSE(alloc.has_value())
        << "AllocStart must reject when the task has been reaped — "
        << "otherwise the staged PROCESSING MEMORY replica is orphaned";
    EXPECT_EQ(alloc.error(), ErrorCode::REPLICA_IS_NOT_READY);

    // The DRAM allocator must be at baseline: no buffer was staged.
    auto seg_after = service->QuerySegments(ctx.segment_name);
    ASSERT_TRUE(seg_after.has_value());
    EXPECT_EQ(seg_after->first, used_baseline)
        << "AllocStart on a reaped task must not allocate. If used bytes "
        << "grew here, AllocStart got to AddReplicas before the task-"
        << "existence check and the staged buffer is now orphaned (no "
        << "reaper iterates it).";

    service->RemoveAll();
}

// NotifyPromotionSuccess must reject calls from a client that is not
// the holder of the source LOCAL_DISK replica. Without the check, any
// client knowing the key could flip the staged PROCESSING MEMORY replica
// to COMPLETE before the holder's RDMA write has landed, exposing torn
// data to readers.
TEST_F(PromotionOnHitTest, NotifyRejectsNonHolder) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_cold",
                                       1024, holder.segment_name));

    // Admit + stage.
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto alloc = service->PromotionAllocStart(holder.client_id, "k_cold",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value());

    // An unrelated client tries to Notify. Must be rejected as
    // INVALID_PARAMS so the staged replica stays PROCESSING.
    UUID intruder_id = generate_uuid();
    ASSERT_NE(intruder_id, holder.client_id);
    auto bad_notify = service->NotifyPromotionSuccess(intruder_id, "k_cold",
                                                      TenantId::Default());
    ASSERT_FALSE(bad_notify.has_value())
        << "Notify from a non-holder client must be rejected — otherwise "
        << "any client knowing the key can commit someone else's "
        << "still-being-written replica";
    EXPECT_EQ(bad_notify.error(), ErrorCode::INVALID_PARAMS);

    // The staged replica must still be PROCESSING (not committed by the
    // rejected call). Readers must not see it via GetReplicaList yet —
    // GetReplicaList filters PROCESSING replicas, and pre-AllocStart
    // there are no COMPLETE replicas to read for a LOCAL_DISK-only key
    // (only the LOCAL_DISK descriptor itself).
    auto r_check = service->GetReplicaList("k_cold", TenantId::Default());
    ASSERT_TRUE(r_check.has_value());
    bool saw_complete_memory = false;
    for (const auto& d : r_check.value().replicas) {
        if (d.is_memory_replica() && d.status == ReplicaStatus::COMPLETE) {
            saw_complete_memory = true;
            break;
        }
    }
    EXPECT_FALSE(saw_complete_memory)
        << "Rejected Notify must not have committed the staged replica";

    // The legitimate holder must still be able to Notify successfully.
    auto good_notify = service->NotifyPromotionSuccess(
        holder.client_id, "k_cold", TenantId::Default());
    ASSERT_TRUE(good_notify.has_value())
        << "Holder Notify on the same task must succeed after a rejected "
        << "intruder Notify — the task entry should be untouched by the "
        << "rejection path; error=" << good_notify.error();

    service->RemoveAll();
}

// NotifyPromotionFailure must immediately release the task slot and the
// staged buffer so that transient client-side errors (SSD throttling,
// RDMA flakes) do not pin promotion_queue_limit_ for the full reaper
// TTL. With queue_limit=1 and a holder client that fails after a
// successful AllocStart, a second admission on a different key would
// be silently dropped until reaper TTL if Notify-Failure didn't
// fast-release. The TTL is set high enough here that any test pass
// can be attributed to the explicit release, not to reaper expiry.
//
// Note: failure also re-records the failed key as a retry candidate
// (transient execution errors are retried, not dropped). With queue_limit=1
// the retried key and a fresh admission on another key race for the freed
// slot; the assertions below accept either winner — what they must NOT
// accept is an undeliverable heartbeat, which is what a saturated slot
// would produce.
TEST_F(PromotionOnHitTest, NotifyFailureReleasesStateImmediately) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;  // cap=1 makes the slot observable
    config.default_kv_lease_ttl = 5000;
    // Long TTL so the test's pass-fail signal cannot be attributed to
    // reaper sweep; only NotifyPromotionFailure could plausibly release.
    config.put_start_release_timeout_sec = 300;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);

    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_a", 1024,
                                       seg.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_b", 1024,
                                       seg.segment_name));

    auto seg_baseline = service->QuerySegments(seg.segment_name);
    ASSERT_TRUE(seg_baseline.has_value());
    const size_t used_baseline = seg_baseline->first;

    auto& mm = MasterMetricManager::instance();
    const int64_t failed_pre = mm.get_promotion_failed();
    const int64_t recorded_pre = mm.get_promotion_candidate_recorded();

    // Admit + stage k_a. promotion_in_flight_ goes 0 -> 1.
    {
        auto r = service->GetReplicaList("k_a", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto alloc = service->PromotionAllocStart(seg.client_id, "k_a",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value());

    // The staged PROCESSING MEMORY buffer is allocated.
    auto seg_after_alloc = service->QuerySegments(seg.segment_name);
    ASSERT_TRUE(seg_after_alloc.has_value());
    EXPECT_GT(seg_after_alloc->first, used_baseline)
        << "AllocStart should commit a buffer in the DRAM allocator";

    // Holder reports failure (simulating SSD read error after AllocStart
    // succeeded). Master must immediately reap the staged replica and
    // decrement the slot counter.
    auto failure = service->NotifyPromotionFailure(seg.client_id, "k_a",
                                                   TenantId::Default());
    ASSERT_TRUE(failure.has_value())
        << "NotifyPromotionFailure on a valid in-flight task from the "
        << "legitimate holder must succeed; error=" << failure.error();
    EXPECT_EQ(mm.get_promotion_failed() - failed_pre, 1)
        << "holder-reported failure must bump promotion_failed";
    // The failure must also re-record a retry candidate for k_a (metric
    // moves synchronously inside NotifyPromotionFailure; the background
    // retry loop can only consume the candidate, never un-record it).
    EXPECT_EQ(mm.get_promotion_candidate_recorded() - recorded_pre, 1)
        << "NotifyPromotionFailure must re-record k_a as a retry candidate "
        << "so transient execution errors are retried, not dropped";

    // The staged buffer must be freed back to the DRAM allocator. If
    // this fires, NotifyPromotionFailure did not pop the staged replica
    // via EraseReplicaByID — same orphan-replica shape that originally
    // motivated the reaper fix.
    auto seg_after_release = service->QuerySegments(seg.segment_name);
    ASSERT_TRUE(seg_after_release.has_value());
    EXPECT_EQ(seg_after_release->first, used_baseline)
        << "NotifyPromotionFailure must release the staged buffer back "
        << "to the allocator; otherwise it leaks until the object is "
        << "removed or evicted.";

    // The slot must be freed. Two outcomes prove it, and which one occurs
    // depends on a benign race with the eviction thread's retry loop:
    //   (a) k_b's read admits k_b while the freed slot is still open, or
    //   (b) k_a — re-recorded as a retry candidate by the failure above —
    //       is re-admitted first, so k_b is cap-rejected and becomes a
    //       candidate itself.
    // Either way the heartbeat must deliver exactly one task (queue_limit=1)
    // and it must be one of the two keys; before the fast-release fix the
    // slot would have stayed saturated and the heartbeat would be empty.
    {
        auto r = service->GetReplicaList("k_b", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto heartbeat = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(heartbeat.has_value());
    ASSERT_EQ(heartbeat->size(), 1u)
        << "after k_a's failure released the slot, exactly one promotion "
        << "must be deliverable (queue_limit=1); an empty result means "
        << "NotifyPromotionFailure did not decrement promotion_in_flight_, "
        << "and transient client-side errors would saturate the queue "
        << "limit for the full reaper TTL";
    const std::string& winner = heartbeat->front().key;
    EXPECT_TRUE(winner == "k_a" || winner == "k_b")
        << "delivered task must be the retried k_a or the newly admitted "
        << "k_b, got: " << winner;

    // Idempotency: repeated failure notification on the same key must be
    // safe (return OK without underflowing the counter).
    auto failure_again = service->NotifyPromotionFailure(seg.client_id, "k_a",
                                                         TenantId::Default());
    EXPECT_TRUE(failure_again.has_value())
        << "Repeated NotifyPromotionFailure should be idempotent (return "
        << "OK on already-released task), not error.";

    service->RemoveAll();
}

// NotifyPromotionFailure must re-record a retry candidate for the failed
// key so transient execution errors (DRAM pressure at AllocStart, TE write
// flake, SSD throttle) are retried by the eviction-thread retry loop rather
// than silently dropped. The re-admission can be driven by the background
// retry loop or by an explicit scan — the final state is identical either
// way: candidate consumed, key re-queued for the holder.
TEST_F(PromotionOnHitTest, NotifyFailureRecordsCandidateAndRetries) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 5000;
    config.put_start_release_timeout_sec = 300;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_a", 1024,
                                       seg.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t recorded_pre = mm.get_promotion_candidate_recorded();

    {
        auto r = service->GetReplicaList("k_a", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto alloc = service->PromotionAllocStart(seg.client_id, "k_a",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value());

    // Holder reports failure after a successful AllocStart (simulating an
    // SSD read error / TE write flake past the staging point).
    auto failure = service->NotifyPromotionFailure(seg.client_id, "k_a",
                                                   TenantId::Default());
    ASSERT_TRUE(failure.has_value());
    EXPECT_EQ(mm.get_promotion_candidate_recorded() - recorded_pre, 1)
        << "NotifyPromotionFailure must re-record a retry candidate";

    // Whether the eviction thread's retry loop or this explicit scan wins
    // the race to re-admit, the resulting state is identical: the candidate
    // is consumed and k_a is queued again for the holder.
    ResetCandidateBackoffsForTesting(service.get());
    RunPromotionCandidateRetryForTesting(service.get());
    EXPECT_EQ(GetPromotionInFlightForTesting(service.get()), 1u)
        << "k_a must be re-admitted after its transient failure";
    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        0u)
        << "the re-admitted candidate must be consumed";

    auto heartbeat = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(heartbeat.has_value());
    ASSERT_EQ(heartbeat->size(), 1u);
    EXPECT_EQ(heartbeat->front().key, "k_a");

    service->RemoveAll();
}

// A persistently-failing key (e.g. a broken SSD file that still has a
// LOCAL_DISK replica) must not re-record -> re-admit -> fail -> re-record
// forever: each NotifyPromotionFailure re-records with execution_failures+1
// (propagated candidate -> task at admission), and once the chain reaches
// kMaxPromotionExecutionFailures the failure stops re-recording. The
// give-up kills only the self-sustaining cycle — a genuine read afterwards
// re-admits with a fresh count.
TEST_F(PromotionOnHitTest, NotifyFailureGivesUpAfterMaxExecutionFailures) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 5000;
    config.put_start_release_timeout_sec = 300;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_a", 1024,
                                       seg.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t recorded_pre = mm.get_promotion_candidate_recorded();
    const int64_t gave_up_pre = mm.get_promotion_execution_gave_up();

    // First admission is driven by a real read (fresh chain, count 0).
    {
        auto r = service->GetReplicaList("k_a", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // Fail kMaxPromotionExecutionFailures times. Each failure must re-record
    // with an incremented count, and each re-admission must show the count
    // propagated across the candidate's consumption. The background eviction
    // thread may win the re-admission race against the explicit retry call —
    // assertions only target the resulting terminal state, which is
    // identical under both interleavings.
    for (uint32_t i = 1; i <= MaxPromotionExecutionFailuresForTesting(); ++i) {
        ASSERT_EQ(GetPromotionInFlightForTesting(service.get()), 1u)
            << "chain must be in-flight at iteration " << i;
        EXPECT_EQ(GetPromotionTaskExecutionFailuresForTesting(
                      service.get(), TenantId::Default(), "k_a"),
                  std::optional<uint32_t>(i - 1))
            << "execution_failures must propagate candidate -> task";
        auto alloc = service->PromotionAllocStart(
            seg.client_id, "k_a", TenantId::Default(), 1024, {});
        ASSERT_TRUE(alloc.has_value());
        auto failure = service->NotifyPromotionFailure(seg.client_id, "k_a",
                                                       TenantId::Default());
        ASSERT_TRUE(failure.has_value());
        EXPECT_EQ(mm.get_promotion_candidate_recorded() - recorded_pre,
                  static_cast<int64_t>(i))
            << "failure " << i << " must re-record (still below the bound)";
        ResetCandidateBackoffsForTesting(service.get());
        RunPromotionCandidateRetryForTesting(service.get());
    }

    // The next failure hits the bound: no re-record, no re-admission.
    ASSERT_EQ(GetPromotionInFlightForTesting(service.get()), 1u);
    EXPECT_EQ(
        GetPromotionTaskExecutionFailuresForTesting(service.get(),
                                                    TenantId::Default(), "k_a"),
        std::optional<uint32_t>(MaxPromotionExecutionFailuresForTesting()));
    auto alloc = service->PromotionAllocStart(seg.client_id, "k_a",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value());
    auto failure = service->NotifyPromotionFailure(seg.client_id, "k_a",
                                                   TenantId::Default());
    ASSERT_TRUE(failure.has_value());
    EXPECT_EQ(mm.get_promotion_candidate_recorded() - recorded_pre,
              static_cast<int64_t>(MaxPromotionExecutionFailuresForTesting()))
        << "the bound-crossing failure must NOT re-record";
    EXPECT_EQ(mm.get_promotion_execution_gave_up() - gave_up_pre, 1)
        << "give-up must bump promotion_execution_gave_up";
    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        0u);
    EXPECT_EQ(RunPromotionCandidateRetryForTesting(service.get()), 0u);
    EXPECT_EQ(GetPromotionInFlightForTesting(service.get()), 0u)
        << "the self-sustaining cycle must be dead";

    // Give-up kills only the self-sustaining cycle: a genuine read re-admits
    // with a fresh chain (count 0).
    {
        auto r = service->GetReplicaList("k_a", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    EXPECT_EQ(GetPromotionInFlightForTesting(service.get()), 1u);
    EXPECT_EQ(GetPromotionTaskExecutionFailuresForTesting(
                  service.get(), TenantId::Default(), "k_a"),
              std::optional<uint32_t>(0u));

    service->RemoveAll();
}

// NotifyPromotionFailure must reject calls from a client that is not the
// holder. Without this gate, any client knowing the key could prematurely
// release a legitimate in-flight promotion's master state and free the
// staged buffer mid-RDMA-write. Mirror of NotifyRejectsNonHolder for the
// Notify-Success path.
TEST_F(PromotionOnHitTest, NotifyFailureRejectsNonHolder) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 5000;
    config.put_start_release_timeout_sec = 300;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_cold",
                                       1024, holder.segment_name));

    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto alloc = service->PromotionAllocStart(holder.client_id, "k_cold",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value());

    // Intruder calls Failure with the wrong client_id.
    UUID intruder_id = generate_uuid();
    ASSERT_NE(intruder_id, holder.client_id);
    auto bad_failure = service->NotifyPromotionFailure(intruder_id, "k_cold",
                                                       TenantId::Default());
    ASSERT_FALSE(bad_failure.has_value())
        << "Failure from a non-holder client must be rejected.";
    EXPECT_EQ(bad_failure.error(), ErrorCode::INVALID_PARAMS);

    // The legitimate holder must still be able to either commit via
    // Notify-Success or release via Notify-Failure on the same task. Use
    // Notify-Failure here to exercise the surviving-task path.
    auto good_failure = service->NotifyPromotionFailure(
        holder.client_id, "k_cold", TenantId::Default());
    ASSERT_TRUE(good_failure.has_value())
        << "Holder Failure must succeed after a rejected intruder "
        << "Failure — the task entry should be untouched by the "
        << "rejection path; error=" << good_failure.error();

    service->RemoveAll();
}

// PromotionAllocStart must reject callers that aren't the holder. Mirror
// of the Notify gate. Without this gate a client that drained another's
// promotion_objects queue via PromotionObjectHeartbeat could call
// AllocStart to stage arbitrary DRAM allocations on the destination
// segment, with no path to commit (Notify rejects on holder_id mismatch)
// — they'd just sit pinned until reaper TTL.
TEST_F(PromotionOnHitTest, AllocStartRejectsNonHolder) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 5000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_cold",
                                       1024, holder.segment_name));

    auto seg_baseline = service->QuerySegments(holder.segment_name);
    ASSERT_TRUE(seg_baseline.has_value());
    const size_t used_baseline = seg_baseline->first;

    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    UUID intruder_id = generate_uuid();
    ASSERT_NE(intruder_id, holder.client_id);
    auto bad_alloc = service->PromotionAllocStart(
        intruder_id, "k_cold", TenantId::Default(), 1024, {});
    ASSERT_FALSE(bad_alloc.has_value())
        << "AllocStart from a non-holder client must be rejected — "
        << "otherwise an attacker that drained another's queue could "
        << "stage arbitrary DRAM allocations.";
    EXPECT_EQ(bad_alloc.error(), ErrorCode::INVALID_PARAMS);

    // No buffer was allocated.
    auto seg_after_bad = service->QuerySegments(holder.segment_name);
    ASSERT_TRUE(seg_after_bad.has_value());
    EXPECT_EQ(seg_after_bad->first, used_baseline)
        << "Rejected AllocStart must not have allocated any DRAM.";

    // The legitimate holder must still be able to AllocStart (task
    // untouched by the rejection).
    auto good_alloc = service->PromotionAllocStart(
        holder.client_id, "k_cold", TenantId::Default(), 1024, {});
    ASSERT_TRUE(good_alloc.has_value())
        << "Holder AllocStart on the same task must succeed after a "
        << "rejected intruder AllocStart; error=" << good_alloc.error();

    service->RemoveAll();
}

// PromotionAllocStart must reject size that doesn't match the task's
// recorded object_size. The size is captured from the source LOCAL_DISK
// descriptor at task admission; mismatch indicates a buggy caller or
// malicious request and would let the caller request an arbitrary-size
// DRAM buffer (smaller → eventual RDMA-write overflow risk; larger →
// wasted DRAM pinned until reaper TTL).
TEST_F(PromotionOnHitTest, AllocStartRejectsSizeMismatch) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 5000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    constexpr int64_t kRealSize = 1024;
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_cold",
                                       kRealSize, holder.segment_name));

    auto seg_baseline = service->QuerySegments(holder.segment_name);
    ASSERT_TRUE(seg_baseline.has_value());
    const size_t used_baseline = seg_baseline->first;

    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // Mismatched-size requests must be rejected, regardless of direction.
    for (uint64_t bad_size : {static_cast<uint64_t>(kRealSize) / 2,
                              static_cast<uint64_t>(kRealSize) * 4}) {
        auto bad_alloc = service->PromotionAllocStart(
            holder.client_id, "k_cold", TenantId::Default(), bad_size, {});
        ASSERT_FALSE(bad_alloc.has_value())
            << "AllocStart with size=" << bad_size
            << " (task.object_size=" << kRealSize << ") must be rejected.";
        EXPECT_EQ(bad_alloc.error(), ErrorCode::INVALID_PARAMS);

        // No buffer must have been staged.
        auto seg_after_bad = service->QuerySegments(holder.segment_name);
        ASSERT_TRUE(seg_after_bad.has_value());
        EXPECT_EQ(seg_after_bad->first, used_baseline)
            << "Rejected AllocStart (size=" << bad_size
            << ") must not have allocated any DRAM.";
    }

    // Correct size must still work — the task was not consumed by the
    // rejections.
    auto good_alloc = service->PromotionAllocStart(
        holder.client_id, "k_cold", TenantId::Default(),
        static_cast<uint64_t>(kRealSize), {});
    ASSERT_TRUE(good_alloc.has_value())
        << "AllocStart with the correct size must succeed after rejected "
        << "size-mismatch attempts; error=" << good_alloc.error();

    service->RemoveAll();
}

// When a holder client expires, ClientMonitorFunc must clean up its
// dangling promotion_tasks entries and decrement the global in-flight
// counter. Without this cleanup, the entries stay pinned until reaper
// TTL — and on a rolling restart of many holders the cluster-wide cap
// promotion_queue_limit_ saturates and blocks all new admissions for
// the full TTL.
//
// Test mechanism: short client_live_ttl_sec, admit a promotion, stop
// pinging, wait for ClientMonitorFunc to expire the client and call
// ClearInvalidHandles. Then assert promotion_in_flight_ is back to 0
// by attempting a second admission with queue_limit=1 on a fresh
// client.
TEST_F(PromotionOnHitTest, ClientExpiryClearsPromotionTask) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;  // cap=1 makes the slot observable
    config.default_kv_lease_ttl = 5000;
    // Long task TTL so that any clearing we see must come from
    // ClearInvalidHandles, not from the promotion-task reaper.
    config.put_start_release_timeout_sec = 300;
    // Short client TTL so expiration is fast.
    config.client_live_ttl_sec = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_cold",
                                       1024, holder.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t cancelled_pre = mm.get_promotion_cancelled();

    // Admit the promotion. promotion_in_flight_ goes 0 -> 1.
    {
        auto r = service->GetReplicaList("k_cold", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // Sanity: with queue_limit=1 the cap is saturated. A second
    // different-shard admission must be rejected right now.
    auto second_holder = PrepareSegment(
        *service, "seg_b", kDefaultSegmentBase + seg_size, seg_size);
    // Promote second_holder into ok_client_ via ReMountSegment so its
    // LOCAL_DISK replicas survive any ClearInvalidHandles run triggered
    // by the first holder's expiry. MountSegment alone does not register
    // the client as alive (only ReMountSegment does), and
    // CleanupStaleHandles uses ok_client_ to decide which LOCAL_DISK
    // replicas to erase — without this, second_holder's k_other replica
    // would be wiped alongside the first holder's k_cold replica when
    // ClearInvalidHandles runs.
    {
        Segment seg_b =
            MakeSegment("seg_b", kDefaultSegmentBase + seg_size, seg_size);
        seg_b.id = second_holder.segment_id;
        std::vector<Segment> segs{seg_b};
        auto remount = service->ReMountSegment(segs, second_holder.client_id);
        ASSERT_TRUE(remount.has_value()) << "ReMount failed";
    }
    ASSERT_TRUE(InjectLocalDiskReplica(*service, second_holder.client_id,
                                       "k_other", 1024,
                                       second_holder.segment_name));
    {
        auto r = service->GetReplicaList("k_other", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto pending_pre =
        service->PromotionObjectHeartbeat(second_holder.client_id);
    ASSERT_TRUE(pending_pre.has_value());
    EXPECT_EQ(CountPromotionTask(*pending_pre, "k_other"), 0u)
        << "Sanity: queue_limit=1 should block the second admission "
        << "while the first task is in flight.";

    // Wait for the first holder to expire while keeping the second
    // holder alive via periodic Pings. ClientMonitorFunc runs every
    // kClientMonitorSleepMs (1s) and expires clients whose last ping is
    // older than client_live_ttl_sec (1s); we ping second_holder every
    // 200ms so it stays alive across the 4s wait. Without these pings
    // both holders would expire together and the second holder's
    // LOCAL_DISK segment would be unmounted — making "k_other" un-
    // admittable for reasons unrelated to the promotion-task slot.
    {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (std::chrono::steady_clock::now() < deadline) {
            (void)service->Ping(second_holder.client_id);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    // ClearInvalidHandles should have erased the holder's LOCAL_DISK
    // source replica AND (with the fix) the promotion_tasks entry,
    // decrementing the global in-flight counter. Re-admit a promotion
    // on the second holder; with queue_limit=1 this can only succeed if
    // the slot was freed.
    {
        auto r = service->GetReplicaList("k_other", TenantId::Default());
        ASSERT_TRUE(r.has_value())
            << "GetReplicaList(k_other) failed with error=" << r.error();
    }
    auto pending_post =
        service->PromotionObjectHeartbeat(second_holder.client_id);
    ASSERT_TRUE(pending_post.has_value());
    EXPECT_EQ(CountPromotionTask(*pending_post, "k_other"), 1u)
        << "After the holder expired, ClearInvalidHandles must have "
        << "erased its promotion_tasks entry and decremented "
        << "promotion_in_flight_. Otherwise the global cap remains "
        << "saturated by the dead holder's task for "
        << "put_start_release_timeout_sec_ seconds, and this admission "
        << "is dropped.";
    EXPECT_EQ(mm.get_promotion_cancelled() - cancelled_pre, 1)
        << "Holder-client expiry mid-promotion must bump "
        << "promotion_cancelled";

    service->RemoveAll();
}

// promotion_admission_threshold=0 would silently bypass the frequency
// gate (`freq < 0` is never true for uint8_t freq), letting every Get
// on a LOCAL_DISK-only key admit a promotion. master.cpp clamps the
// flag at parse time, and MasterService's constructor adds a
// defense-in-depth clamp for direct-construction paths (tests,
// embedded users). Verify the clamp lifts 0 → 1.
TEST_F(PromotionOnHitTest, AdmissionThresholdZeroClampsToOne) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 0;  // would bypass the gate
    auto service = std::make_unique<MasterService>(config);
    EXPECT_EQ(GetPromotionAdmissionThresholdForTesting(service.get()), 1u)
        << "threshold=0 must be clamped to 1; otherwise every Get-on-"
        << "LOCAL_DISK admits a promotion and the frequency gate is "
        << "silently disabled.";
}

// promotion_admission_threshold above the CountMinSketch saturating
// max (255) would make the gate unreachable (freq saturates at 255,
// `255 < 256` is true forever → no admissions). Verify the constructor
// clamps high too.
TEST_F(PromotionOnHitTest, AdmissionThresholdAboveMaxClampsToMax) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1000;  // > 255
    auto service = std::make_unique<MasterService>(config);
    EXPECT_EQ(GetPromotionAdmissionThresholdForTesting(service.get()), 255u)
        << "threshold above the CountMinSketch counter max (255) must "
        << "be clamped to 255; otherwise the gate is unreachable.";
}

// Remove(force=true) on a key with an in-flight PromotionTask must
// drop the task entry alongside the metadata, so promotion_in_flight_
// is decremented immediately rather than pinned for ~10 min until the
// reaper sweeps. With queue_limit=1, the test admits one task, removes
// the key, then admits a second task on a different key — only
// succeeds if the slot was freed.
TEST_F(PromotionOnHitTest, RemoveErasesPromotionTask) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;  // makes the slot observable
    config.default_kv_lease_ttl = 5000;
    // Long task TTL so any cap-slot reclaim must come from the
    // metadata-erase path, not the reaper.
    config.put_start_release_timeout_sec = 300;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_first",
                                       1024, holder.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_second",
                                       1024, holder.segment_name));

    // Admit task 1.
    {
        auto r = service->GetReplicaList("k_first", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    {
        auto pending = service->PromotionObjectHeartbeat(holder.client_id);
        ASSERT_TRUE(pending.has_value());
        EXPECT_EQ(CountPromotionTask(*pending, "k_first"), 1u);
    }

    // Remove k_first with force=true. With the fix, this also wipes
    // k_first's promotion_tasks entry and decrements
    // promotion_in_flight_ back to 0.
    auto rm = service->Remove("k_first", TenantId::Default(), /*force=*/true);
    ASSERT_TRUE(rm.has_value())
        << "Remove should succeed; error=" << rm.error();

    // Now admit a different key. With queue_limit=1, this only succeeds
    // if the slot was freed by Remove. Without the in-flight cleanup,
    // it would stay pinned for the full 300s TTL.
    {
        auto r = service->GetReplicaList("k_second", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto pending_post = service->PromotionObjectHeartbeat(holder.client_id);
    ASSERT_TRUE(pending_post.has_value());
    EXPECT_EQ(CountPromotionTask(*pending_post, "k_second"), 1u)
        << "k_second must be admittable after Remove of k_first — Remove "
        << "must erase the in-flight promotion_tasks entry and decrement "
        << "promotion_in_flight_, otherwise queue_limit=1 stays saturated.";

    service->RemoveAll();
}

// RemoveByRegex on a key with an in-flight PromotionTask must drop the
// task entry, same as Remove. Mirror of RemoveErasesPromotionTask, but
// exercises the regex path which iterates shards directly without an
// accessor object.
TEST_F(PromotionOnHitTest, RemoveByRegexErasesPromotionTask) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;
    config.default_kv_lease_ttl = 5000;
    config.put_start_release_timeout_sec = 300;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "regex_k1",
                                       1024, holder.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "other_k2",
                                       1024, holder.segment_name));

    // Admit task on regex_k1.
    {
        auto r = service->GetReplicaList("regex_k1", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // RemoveByRegex matches regex_k1 only.
    auto removed =
        service->RemoveByRegex("^regex_", TenantId::Default(), /*force=*/true);
    ASSERT_TRUE(removed.has_value())
        << "RemoveByRegex should succeed; error=" << removed.error();
    EXPECT_EQ(removed.value(), 1) << "exactly one key (regex_k1) should match";

    // Slot must be free — admit on other_k2 (different shard or same,
    // doesn't matter because counter is global).
    {
        auto r = service->GetReplicaList("other_k2", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto pending_post = service->PromotionObjectHeartbeat(holder.client_id);
    ASSERT_TRUE(pending_post.has_value());
    EXPECT_EQ(CountPromotionTask(*pending_post, "other_k2"), 1u)
        << "other_k2 must be admittable after RemoveByRegex of regex_k1 "
        << "— RemoveByRegex must erase the in-flight promotion_tasks "
        << "entry. Otherwise queue_limit=1 stays saturated.";

    service->RemoveAll();
}

// RemoveAll on a key with an in-flight PromotionTask must drop the
// task entry alongside the metadata. Same shape as
// RemoveErasesPromotionTask but exercises the bulk-erase loop in
// MasterService::RemoveAll, which iterates every shard and erases
// metadata entries directly.
TEST_F(PromotionOnHitTest, RemoveAllErasesPromotionTask) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;
    config.default_kv_lease_ttl = 5000;
    config.put_start_release_timeout_sec = 300;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_first",
                                       1024, holder.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t cancelled_pre = mm.get_promotion_cancelled();

    {
        auto r = service->GetReplicaList("k_first", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    {
        auto pending = service->PromotionObjectHeartbeat(holder.client_id);
        ASSERT_TRUE(pending.has_value());
        EXPECT_EQ(CountPromotionTask(*pending, "k_first"), 1u);
    }

    auto removed = service->RemoveAll(/*force=*/true);
    EXPECT_GE(removed, 1) << "RemoveAll should erase k_first";

    EXPECT_EQ(mm.get_promotion_cancelled() - cancelled_pre, 1)
        << "RemoveAll on a key with an in-flight promotion must route "
        << "through EraseMetadataEntry and bump promotion_cancelled_total.";

    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_second",
                                       1024, holder.segment_name));
    {
        auto r = service->GetReplicaList("k_second", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto pending_post = service->PromotionObjectHeartbeat(holder.client_id);
    ASSERT_TRUE(pending_post.has_value());
    EXPECT_EQ(CountPromotionTask(*pending_post, "k_second"), 1u)
        << "k_second must be admittable after RemoveAll of k_first — "
        << "otherwise queue_limit=1 stays saturated until reaper TTL.";

    service->RemoveAll(/*force=*/true);
}

// BatchRemove normal-completion path on a key with an in-flight
// PromotionTask must drop the task entry. ReMountSegment registers the
// holder in ok_client_ so CleanupStaleHandles returns false and
// BatchRemove takes the non-stale branch.
TEST_F(PromotionOnHitTest, BatchRemoveErasesPromotionTask) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;
    config.default_kv_lease_ttl = 5000;
    config.put_start_release_timeout_sec = 300;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    {
        Segment seg_a = MakeSegment("seg_a", kDefaultSegmentBase, seg_size);
        seg_a.id = holder.segment_id;
        std::vector<Segment> segs{seg_a};
        auto remount = service->ReMountSegment(segs, holder.client_id);
        ASSERT_TRUE(remount.has_value()) << "ReMount failed";
    }
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_first",
                                       1024, holder.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_second",
                                       1024, holder.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t cancelled_pre = mm.get_promotion_cancelled();

    {
        auto r = service->GetReplicaList("k_first", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    {
        auto pending = service->PromotionObjectHeartbeat(holder.client_id);
        ASSERT_TRUE(pending.has_value());
        EXPECT_EQ(CountPromotionTask(*pending, "k_first"), 1u);
    }

    auto results =
        service->BatchRemove({"k_first"}, TenantId::Default(), /*force=*/true);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].has_value())
        << "BatchRemove should succeed; error=" << results[0].error();

    EXPECT_EQ(mm.get_promotion_cancelled() - cancelled_pre, 1)
        << "BatchRemove normal path on a key with an in-flight promotion "
        << "must bump promotion_cancelled_total.";

    {
        auto r = service->GetReplicaList("k_second", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto pending_post = service->PromotionObjectHeartbeat(holder.client_id);
    ASSERT_TRUE(pending_post.has_value());
    EXPECT_EQ(CountPromotionTask(*pending_post, "k_second"), 1u)
        << "k_second must be admittable after BatchRemove of k_first — "
        << "otherwise queue_limit=1 stays saturated until reaper TTL.";

    service->RemoveAll(/*force=*/true);
}

// BatchRemove stale-handle path on a key with an in-flight
// PromotionTask must drop the task entry. The holder is mounted via
// PrepareSegment only (no ReMount), so its client is absent from
// ok_client_; BatchRemove's CleanupStaleHandles then erases the
// LOCAL_DISK replica and the stale-handle branch fires.
TEST_F(PromotionOnHitTest, BatchRemoveStaleHandleErasesPromotionTask) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_queue_limit = 1;
    config.default_kv_lease_ttl = 5000;
    config.put_start_release_timeout_sec = 300;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto holder =
        PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, holder.client_id, "k_first",
                                       1024, holder.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t cancelled_pre = mm.get_promotion_cancelled();

    {
        auto r = service->GetReplicaList("k_first", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    {
        auto pending = service->PromotionObjectHeartbeat(holder.client_id);
        ASSERT_TRUE(pending.has_value());
        EXPECT_EQ(CountPromotionTask(*pending, "k_first"), 1u);
    }

    auto results =
        service->BatchRemove({"k_first"}, TenantId::Default(), /*force=*/true);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_FALSE(results[0].has_value())
        << "stale-handle path should report OBJECT_NOT_FOUND once the "
        << "LOCAL_DISK replica is wiped.";

    EXPECT_EQ(mm.get_promotion_cancelled() - cancelled_pre, 1)
        << "BatchRemove stale-handle path must also bump "
        << "promotion_cancelled_total.";

    auto second_holder = PrepareSegment(
        *service, "seg_b", kDefaultSegmentBase + seg_size, seg_size);
    {
        Segment seg_b =
            MakeSegment("seg_b", kDefaultSegmentBase + seg_size, seg_size);
        seg_b.id = second_holder.segment_id;
        std::vector<Segment> segs{seg_b};
        auto remount = service->ReMountSegment(segs, second_holder.client_id);
        ASSERT_TRUE(remount.has_value()) << "ReMount failed";
    }
    ASSERT_TRUE(InjectLocalDiskReplica(*service, second_holder.client_id,
                                       "k_second", 1024,
                                       second_holder.segment_name));
    {
        auto r = service->GetReplicaList("k_second", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    auto pending_post =
        service->PromotionObjectHeartbeat(second_holder.client_id);
    ASSERT_TRUE(pending_post.has_value());
    EXPECT_EQ(CountPromotionTask(*pending_post, "k_second"), 1u)
        << "k_second must be admittable after the stale-handle "
        << "BatchRemove — otherwise queue_limit=1 stays saturated until "
        << "reaper TTL.";

    service->RemoveAll(/*force=*/true);
}

// The funnel metrics (admitted, completed, completed_bytes, in_flight)
// track a successful promotion lifecycle end-to-end: a single
// admission bumps admitted+1 and in_flight+1; NotifyPromotionSuccess
// bumps completed+1 and completed_bytes+object_size and brings
// in_flight back to 0.
TEST_F(PromotionOnHitTest, MetricsFunnelTracksSuccessfulPromotion) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    constexpr int64_t kObjBytes = 4096;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_hot",
                                       kObjBytes, seg.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t admitted_pre = mm.get_promotion_admitted();
    const int64_t completed_pre = mm.get_promotion_completed();
    const int64_t bytes_pre = mm.get_promotion_completed_bytes();
    const int64_t in_flight_pre = mm.get_promotion_in_flight();

    // Admit.
    {
        auto r = service->GetReplicaList("k_hot", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    EXPECT_EQ(mm.get_promotion_admitted() - admitted_pre, 1);
    EXPECT_EQ(mm.get_promotion_in_flight() - in_flight_pre, 1);

    // Drive AllocStart + NotifyPromotionSuccess.
    auto alloc = service->PromotionAllocStart(
        seg.client_id, "k_hot", TenantId::Default(), kObjBytes, {});
    ASSERT_TRUE(alloc.has_value());
    auto notify = service->NotifyPromotionSuccess(seg.client_id, "k_hot",
                                                  TenantId::Default());
    ASSERT_TRUE(notify.has_value());

    EXPECT_EQ(mm.get_promotion_completed() - completed_pre, 1);
    EXPECT_EQ(mm.get_promotion_completed_bytes() - bytes_pre, kObjBytes);
    EXPECT_EQ(mm.get_promotion_in_flight() - in_flight_pre, 0);

    service->RemoveAll();
}

// Each rejection gate (frequency / watermark / cap) increments its
// own counter when its branch fires.
TEST_F(PromotionOnHitTest, MetricsRejectionCountersIncrementOnGateMiss) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    // Threshold > 1 so the first Get is rejected on frequency.
    config.promotion_admission_threshold = 2;
    config.promotion_queue_limit = 1;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_a", 1024,
                                       seg.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_b", 1024,
                                       seg.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t freq_pre = mm.get_promotion_rejected_frequency();
    const int64_t cap_pre = mm.get_promotion_rejected_cap();

    // 1st Get on k_a: freq=1, threshold=2 → rejected on frequency.
    {
        auto r = service->GetReplicaList("k_a", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    EXPECT_EQ(mm.get_promotion_rejected_frequency() - freq_pre, 1);

    // 2nd Get on k_a: freq=2, admits. Now in-flight = 1 == limit.
    {
        auto r = service->GetReplicaList("k_a", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    // Get on k_b: freq=1, threshold=2 → rejected on frequency.
    {
        auto r = service->GetReplicaList("k_b", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    // Get on k_b again: freq=2, gets past frequency, but cap=1
    // saturated → rejected on cap.
    {
        auto r = service->GetReplicaList("k_b", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    EXPECT_EQ(mm.get_promotion_rejected_cap() - cap_pre, 1);

    service->RemoveAll();

    // Watermark gate uses a fresh service configured with
    // eviction_high_watermark_ratio = 0.0 so any non-negative DRAM
    // usage trips it. threshold = 1 makes the first Get clear the
    // frequency gate and reach the watermark check.
    MasterServiceConfig wm_config;
    wm_config.enable_offload = true;
    wm_config.promotion_on_hit = true;
    wm_config.promotion_admission_threshold = 1;
    wm_config.promotion_queue_limit = 50;
    wm_config.eviction_high_watermark_ratio = 0.0;
    wm_config.default_kv_lease_ttl = 2000;
    auto wm_service = std::make_unique<MasterService>(wm_config);

    auto wm_seg =
        PrepareSegment(*wm_service, "wm_seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*wm_service, wm_seg.client_id, "k_w",
                                       1024, wm_seg.segment_name));

    const int64_t wm_pre = mm.get_promotion_rejected_watermark();
    {
        auto r = wm_service->GetReplicaList("k_w", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    EXPECT_EQ(mm.get_promotion_rejected_watermark() - wm_pre, 1);

    wm_service->RemoveAll();
}

TEST_F(PromotionOnHitTest, WatermarkUsesAllocatorStateAfterMetricsReset) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.eviction_high_watermark_ratio = 0.5;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t kSegmentSize = 16 * 1024 * 1024;
    constexpr size_t kAllocationSize = 12 * 1024 * 1024;
    const std::string segment_name = "allocator_watermark_segment";
    auto segment = PrepareSegment(*service, segment_name, kDefaultSegmentBase,
                                  kSegmentSize);
    auto allocated = AllocateOnSegmentForTesting(
        service.get(), segment.segment_id, kAllocationSize);
    ASSERT_NE(allocated, nullptr);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, segment.client_id,
                                       "allocator_watermark_key", 1024,
                                       segment.segment_name));

    auto& metrics = MasterMetricManager::instance();
    metrics.reset_allocated_mem_size();
    metrics.reset_total_mem_capacity();
    metrics.reset_segment_allocated_mem_size(segment_name);
    metrics.reset_segment_total_mem_capacity(segment_name);
    EXPECT_EQ(metrics.get_allocated_mem_size(), 0);
    EXPECT_EQ(metrics.get_total_mem_capacity(), 0);

    const int64_t rejected_before = metrics.get_promotion_rejected_watermark();
    auto get_result =
        service->GetReplicaList("allocator_watermark_key", TenantId::Default());
    EXPECT_TRUE(get_result.has_value());
    EXPECT_EQ(metrics.get_promotion_rejected_watermark() - rejected_before, 1);

    // Restore the observable gauges before normal teardown. The business
    // assertion above intentionally reset them, but allocator and service
    // destructors still emit their matching decrements.
    metrics.inc_allocated_mem_size(segment_name, kAllocationSize);
    metrics.inc_total_mem_capacity(segment_name, kSegmentSize);
    allocated.reset();
    service->RemoveAll();
}

TEST_F(PromotionOnHitTest, AdmissionFrequencyIsTenantScoped) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.enable_multi_tenants = true;
    config.tenant_quota_connector_type = "file";
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 2;
    config.default_kv_lease_ttl = 2000;
    const std::string key = "shared_hot_key";
    const TenantId tenant_a("tenant_promotion_a");
    const TenantId tenant_b("tenant_promotion_b");
    config.tenant_quota_connector_uri =
        WriteTenantQuotaPolicyFile({{tenant_a.value(), 64 * 1024 * 1024},
                                    {tenant_b.value(), 64 * 1024 * 1024}});
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg =
        PrepareSegment(*service, "seg_tenant", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, key, 1024,
                                       seg.segment_name, tenant_a.value()));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, key, 1024,
                                       seg.segment_name, tenant_b.value()));

    {
        auto r = service->GetReplicaList(key, tenant_a);
        ASSERT_TRUE(r.has_value());
    }
    {
        auto r = service->GetReplicaList(key, tenant_b);
        ASSERT_TRUE(r.has_value());
    }
    auto pending_after_one_each =
        service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(pending_after_one_each.has_value());
    EXPECT_TRUE(pending_after_one_each->empty())
        << "same user key in two tenants must not share promotion heat";

    {
        auto r = service->GetReplicaList(key, tenant_a);
        ASSERT_TRUE(r.has_value());
    }
    auto pending_after_tenant_a_second =
        service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(pending_after_tenant_a_second.has_value());
    ASSERT_EQ(pending_after_tenant_a_second->size(), 1u);
    EXPECT_EQ((*pending_after_tenant_a_second)[0].tenant_id, tenant_a.value());
    EXPECT_EQ((*pending_after_tenant_a_second)[0].key, key);

    service->RemoveAll();
}

// promotion_max_per_heartbeat controls how many tasks
// PromotionObjectHeartbeat returns per call. Set the knob to 3 and
// verify the master returns up to 3 tasks per heartbeat.
TEST_F(PromotionOnHitTest, MaxPerHeartbeatKnobControlsBatchSize) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_max_per_heartbeat = 3;  // raise from default 1
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);

    // Admit 5 tasks on the same holder.
    for (int i = 0; i < 5; ++i) {
        const auto key = "k_" + std::to_string(i);
        ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, key, 1024,
                                           seg.segment_name));
        auto r = service->GetReplicaList(key, TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // First heartbeat: cap=3.
    auto first = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->size(), 3u)
        << "expected promotion_max_per_heartbeat=3 keys, got " << first->size();

    // Second heartbeat: 2 leftover.
    auto second = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->size(), 2u);

    // Third heartbeat: empty.
    auto third = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(third.has_value());
    EXPECT_EQ(third->size(), 0u);

    service->RemoveAll();
}

// promotion_max_per_heartbeat = 0 must be clamped to 1 by the
// MasterService constructor; otherwise PromotionObjectHeartbeat would
// return an empty batch every call and silently disable promotion
// delivery.
TEST_F(PromotionOnHitTest, MaxPerHeartbeatZeroClampsToOne) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.promotion_max_per_heartbeat = 0;  // pathological
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k1", 1024,
                                       seg.segment_name));
    {
        auto r = service->GetReplicaList("k1", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    // If max_per_heartbeat=0 weren't clamped, this would return empty.
    auto hb = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(hb.has_value());
    EXPECT_EQ(hb->size(), 1u)
        << "max_per_heartbeat=0 must clamp to 1 so promotion delivery "
        << "isn't silently disabled";

    service->RemoveAll();
}

// Remove on a key mid-promotion must bump promotion_cancelled so the
// funnel invariant
// admitted = completed + failed + expired + cancelled + in_flight
// holds. Exercises the EraseMetadataEntry path.
TEST_F(PromotionOnHitTest, MetricsRemoveMidPromotionCountsAsCancelled) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "seg_a", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_drop", 1024,
                                       seg.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t admitted_pre = mm.get_promotion_admitted();
    const int64_t cancelled_pre = mm.get_promotion_cancelled();
    const int64_t in_flight_pre = mm.get_promotion_in_flight();

    // Admit a promotion. in_flight goes from 0 to 1.
    {
        auto r = service->GetReplicaList("k_drop", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    ASSERT_EQ(mm.get_promotion_admitted() - admitted_pre, 1);
    ASSERT_EQ(mm.get_promotion_in_flight() - in_flight_pre, 1);

    auto rm = service->Remove("k_drop", TenantId::Default(), /*force=*/true);
    ASSERT_TRUE(rm.has_value()) << "error=" << rm.error();

    EXPECT_EQ(mm.get_promotion_in_flight() - in_flight_pre, 0);
    EXPECT_EQ(mm.get_promotion_cancelled() - cancelled_pre, 1);

    service->RemoveAll();
}

// --- Promotion retry candidate tests ---

// A transient watermark rejection records a candidate; the retry scheduler
// can later see and process it.
TEST_F(PromotionOnHitTest, RetryCandidate_WatermarkRejectionRecordsCandidate) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    config.eviction_high_watermark_ratio = 0.0;  // all promotions rejected
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg =
        PrepareSegment(*service, "retry_seg", kDefaultSegmentBase, seg_size);
    // LOCAL_DISK-only key: inject without a prior PutStart.
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_wm", 1024,
                                       seg.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t recorded_pre = mm.get_promotion_candidate_recorded();

    // Get triggers frequency gate (threshold=1 → passes) then hits watermark=0.
    {
        auto r = service->GetReplicaList("k_wm", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    EXPECT_EQ(mm.get_promotion_candidate_recorded() - recorded_pre, 1)
        << "Expected one candidate recorded on watermark rejection";
    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        1u);

    // A second Get on the same key should update last_seen but not add a
    // duplicate candidate.
    {
        auto r = service->GetReplicaList("k_wm", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        1u)
        << "Duplicate candidate must not be created";

    service->RemoveAll();
}

// A candidate recorded while the promotion queue is full is admitted once the
// active slot is released and the retry scanner reaches it.
TEST_F(PromotionOnHitTest, RetryCandidate_CapRejectedThenQueuedOnRetry) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    config.promotion_queue_limit = 1;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "retry_cap_seg", kDefaultSegmentBase,
                              seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_busy", 1024,
                                       seg.segment_name));
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_retry", 1024,
                                       seg.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t candidate_admitted_pre =
        mm.get_promotion_candidate_admitted();
    const int64_t promotion_admitted_pre = mm.get_promotion_admitted();

    {
        auto r = service->GetReplicaList("k_busy", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    ASSERT_EQ(mm.get_promotion_admitted() - promotion_admitted_pre, 1);
    ASSERT_EQ(GetPromotionInFlightForTesting(service.get()), 1u);

    {
        auto r = service->GetReplicaList("k_retry", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    ASSERT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        1u);

    // Release the slot via a *successful* promotion of k_busy. Using
    // NotifyPromotionFailure here would change the candidate set: failure
    // deliberately re-records a retry candidate for the failed key
    // (transient execution errors are retried, not dropped), so k_busy
    // would then compete with k_retry for the freed slot.
    auto alloc = service->PromotionAllocStart(seg.client_id, "k_busy",
                                              TenantId::Default(), 1024, {});
    ASSERT_TRUE(alloc.has_value());
    auto done = service->NotifyPromotionSuccess(seg.client_id, "k_busy",
                                                TenantId::Default());
    ASSERT_TRUE(done.has_value());
    ASSERT_EQ(GetPromotionInFlightForTesting(service.get()), 0u);

    ResetCandidateBackoffsForTesting(service.get());
    EXPECT_EQ(RunPromotionCandidateRetryForTesting(service.get()), 1u);

    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        0u);
    EXPECT_EQ(GetPromotionInFlightForTesting(service.get()), 1u);
    EXPECT_EQ(mm.get_promotion_candidate_admitted() - candidate_admitted_pre,
              1);

    auto heartbeat = service->PromotionObjectHeartbeat(seg.client_id);
    ASSERT_TRUE(heartbeat.has_value());
    ASSERT_EQ(heartbeat->size(), 1u);
    EXPECT_EQ(heartbeat->front().key, "k_retry");

    service->RemoveAll();
}

TEST_F(PromotionOnHitTest, RetryCandidate_NoCandidatesOrNoShardBudgetNoops) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    config.eviction_high_watermark_ratio = 0.0;
    auto service = std::make_unique<MasterService>(config);

    EXPECT_EQ(RunPromotionCandidateRetryForTesting(service.get()), 0u);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg = PrepareSegment(*service, "retry_noop_seg", kDefaultSegmentBase,
                              seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_noop", 1024,
                                       seg.segment_name));

    {
        auto r = service->GetReplicaList("k_noop", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    ASSERT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        1u);

    EXPECT_EQ(RunPromotionCandidateRetryForTesting(service.get(),
                                                   /*shards_to_scan=*/0),
              0u);
    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        1u);

    service->RemoveAll();
}

// Candidate is cleaned up after kPromotionCandidateMaxRetries scans.
TEST_F(PromotionOnHitTest, RetryCandidate_ExhaustedAfterMaxRetries) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    config.eviction_high_watermark_ratio = 0.0;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg =
        PrepareSegment(*service, "exhaust_seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_exhaust",
                                       1024, seg.segment_name));

    // Record a candidate via a Get (watermark=0 rejects).
    {
        auto r = service->GetReplicaList("k_exhaust", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    ASSERT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        1u);

    auto& mm = MasterMetricManager::instance();
    const int64_t expired_pre = mm.get_promotion_candidate_expired_evaluated();

    // Drive retries; each scan increments retry_count until exhausted.
    // Reset backoff timestamps before each scan so wall-clock time doesn't
    // gate retries and the loop completes without sleeping. The scan count
    // is derived from kPromotionCandidateMaxRetries (one scan to reach the
    // limit, one more to observe it and erase).
    for (uint32_t i = 0; i <= MaxPromotionCandidateRetriesForTesting(); ++i) {
        ResetCandidateBackoffsForTesting(service.get());
        RunPromotionCandidateRetryForTesting(service.get());
    }

    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        0u)
        << "Candidate must be erased after max retries";
    EXPECT_GT(mm.get_promotion_candidate_expired_evaluated() - expired_pre, 0);

    service->RemoveAll();
}

// When an object is removed while a candidate is pending, the retry scan
// cleans up the candidate without promoting.
TEST_F(PromotionOnHitTest, RetryCandidate_ObjectRemovedMidRetry) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    config.eviction_high_watermark_ratio = 0.0;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg =
        PrepareSegment(*service, "rm_seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_rm", 1024,
                                       seg.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t admitted_pre = mm.get_promotion_candidate_admitted();

    // Record candidate.
    {
        auto r = service->GetReplicaList("k_rm", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    ASSERT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        1u);

    // Remove the object while candidate is pending.
    auto rm = service->Remove("k_rm", TenantId::Default(), /*force=*/true);
    ASSERT_TRUE(rm.has_value());

    // Retry scan should find object gone and erase candidate.
    RunPromotionCandidateRetryForTesting(service.get());
    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        0u);

    EXPECT_EQ(mm.get_promotion_candidate_admitted() - admitted_pre, 0);

    service->RemoveAll();
}

// Records multiple candidates and verifies per-key tracking is correct.
TEST_F(PromotionOnHitTest, RetryCandidate_MultipleKeysTracked) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    config.eviction_high_watermark_ratio = 0.0;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 64;
    auto seg =
        PrepareSegment(*service, "multi_seg", kDefaultSegmentBase, seg_size);

    auto& mm = MasterMetricManager::instance();
    const int64_t recorded_pre = mm.get_promotion_candidate_recorded();
    const int64_t unevaluated_pre =
        mm.get_promotion_candidate_expired_unevaluated();

    constexpr int kKeys = 5;
    for (int i = 0; i < kKeys; i++) {
        std::string key = "k_multi_" + std::to_string(i);
        ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, key, 512,
                                           seg.segment_name));
        auto r = service->GetReplicaList(key, TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }

    EXPECT_EQ(mm.get_promotion_candidate_recorded() - recorded_pre, kKeys);
    EXPECT_EQ(
        mm.get_promotion_candidate_expired_unevaluated() - unevaluated_pre, 0);
    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        static_cast<size_t>(kKeys));

    service->RemoveAll();
}

// ClearCandidatesForReload resets all candidate state and the global count.
TEST_F(PromotionOnHitTest, RetryCandidate_ClearOnReload) {
    MasterServiceConfig config;
    config.enable_offload = true;
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    config.eviction_high_watermark_ratio = 0.0;
    auto service = std::make_unique<MasterService>(config);

    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto seg =
        PrepareSegment(*service, "reload_seg", kDefaultSegmentBase, seg_size);
    ASSERT_TRUE(InjectLocalDiskReplica(*service, seg.client_id, "k_reload",
                                       1024, seg.segment_name));

    // Record a candidate.
    {
        auto r = service->GetReplicaList("k_reload", TenantId::Default());
        ASSERT_TRUE(r.has_value());
    }
    ASSERT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        1u);

    // Simulate metadata reload.
    ClearCandidatesForReloadForTesting(service.get());

    EXPECT_EQ(
        CountPromotionCandidatesForTesting(service.get(), TenantId::Default()),
        0u);
    EXPECT_EQ(GetPromotionCandidateCountForTesting(service.get()), 0u);
    EXPECT_EQ(GetPromotionInFlightForTesting(service.get()), 0u);

    service->RemoveAll();
}

// ---------------- Admission and dispatch of the DFS promotion channel --------
// The DFS promotion channel copies hot DFS-only objects back into MEMORY:
// admission (read-hit / retry) enqueues a task carrying a snapshot of the
// COMPLETE DFS source replica; a DFS-capable client claims tasks via the
// DfsPromotionObjectHeartbeat; claimed/uncompleted tasks are reclaimed by the
// TTL reaper. Tests here drive the master-side logic directly through the
// friend funnels above (no DFS backend, no real executor). The client-side
// executor (read source -> write MEMORY -> Notify success) is covered by the
// file_storage promotion tests.

MasterServiceConfig MakeDfsPromotionConfig() {
    MasterServiceConfig config;
    config.dfs_promotion.enable = true;
    config.dfs_promotion.task_ttl_min = 1;  // fast TTL for the reaper tests
    // Pin the P90 trust gate explicitly: the threshold-provenance tests shape
    // the sketch by hand with tens of samples, so they must not silently start
    // depending on how the production default is tuned.
    config.dfs_promotion.min_total_weight = 8.0;
    // The execution-layer tests stage the promoted MEMORY replica on a mounted
    // DRAM segment, so MountLocalDiskSegment must be permitted (it gates on
    // enable_offload_); the admission/dispatch tests never mount a segment.
    config.enable_offload = true;
    // Keep the DRAM watermark gate open regardless of other tests' segment
    // allocations (metrics are process-global).
    config.eviction_high_watermark_ratio = 1.0;
    return config;
}

TEST_F(PromotionOnHitTest, DfsPromotionAdmissionEnqueuesAndHeartbeatDispatches) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs", 1024, "/dfs/obj/k_dfs",
                                    4096));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs"));
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 1u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 1u);
    EXPECT_TRUE(IsDfsSourceProtectedForTesting(service.get(), "k_dfs"));

    // First heartbeat dispatches the task with the source DFS descriptor.
    const UUID executor = generate_uuid();
    auto resp = service->DfsPromotionObjectHeartbeat(executor);
    ASSERT_TRUE(resp.has_value());
    ASSERT_EQ(resp->size(), 1u);
    const auto& task = resp->front();
    EXPECT_EQ(task.key, "k_dfs");
    EXPECT_EQ(task.size, 1024);
    ASSERT_TRUE(task.source_dfs.has_value());
    ASSERT_TRUE(task.source_dfs->is_dfs_replica());
    const auto& src = task.source_dfs->get_dfs_descriptor();
    EXPECT_EQ(src.file_path, "/dfs/obj/k_dfs");
    EXPECT_EQ(src.offset, 4096);
    EXPECT_EQ(src.object_size, 1024);

    // The queue node was popped at claim and the record is claimed, so no
    // re-dispatch (even from a different client id).
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 0u);
    auto again = service->DfsPromotionObjectHeartbeat(generate_uuid());
    ASSERT_TRUE(again.has_value());
    EXPECT_TRUE(again->empty());
}

TEST_F(PromotionOnHitTest, DfsPromotionQueuedTaskBlocksReAdmission) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dup", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dup"));
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);

    // While the task is still queued (not yet claimed), re-admission of the
    // same key must be rejected by the per-key dedup gate shared with the SSD
    // channel, and must not create a second task/record.
    EXPECT_FALSE(AdmitDfsPromotionForTesting(service.get(), "k_dup", 500.0));
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 1u);
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 1u);

    // A different key is unaffected.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_other", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_other"));
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 2u);
}

TEST_F(PromotionOnHitTest, DfsPromotionHeartbeatIsFifoForEqualHeat) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    const std::vector<std::string> keys{"k_a", "k_b", "k_c"};
    for (const auto& key : keys) {
        ASSERT_TRUE(InjectDfsOnlyObject(*service, key, 1024));
        ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), key));
    }
    // All tasks carry the same admission heat, so the comparator orders by
    // enqueue sequence => dispatch order equals admission order (FIFO).
    std::vector<std::string> dispatched;
    const UUID executor = generate_uuid();
    for (size_t i = 0; i < keys.size(); ++i) {
        auto resp = service->DfsPromotionObjectHeartbeat(executor);
        ASSERT_TRUE(resp.has_value());
        ASSERT_EQ(resp->size(), 1u) << "dispatch " << i;
        dispatched.push_back(resp->front().key);
    }
    EXPECT_EQ(dispatched[0], "k_a");
    EXPECT_EQ(dispatched[1], "k_b");
    EXPECT_EQ(dispatched[2], "k_c");
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionClaimDropsCancelledStaleNode) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_cancel", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_cancel"));
    // Cancellation erases the record but deliberately leaves the queue node.
    ASSERT_TRUE(CancelDfsPromotionTaskForTesting(service.get(), "k_cancel"));
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_live", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_live"));
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 2u);

    // The heartbeat pops the stale (cancelled) node first and drops it, then
    // dispatches the live task.
    auto resp = service->DfsPromotionObjectHeartbeat(generate_uuid());
    ASSERT_TRUE(resp.has_value());
    ASSERT_EQ(resp->size(), 1u);
    EXPECT_EQ(resp->front().key, "k_live");
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);
}

// A burst of cancelled tasks leaves a pile of stale nodes behind (cancellation
// never removes a node physically). Each stale node costs a metadata shard
// lock to discard, so a single claim must not walk the whole backlog while
// holding snapshot_mutex_ shared; it drops up to a fixed cap and leaves the
// rest to later heartbeats (or the oversized-queue rebuild).
TEST_F(PromotionOnHitTest, DfsPromotionClaimBoundsStaleDropsPerHeartbeat) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    constexpr size_t kStale = 100;
    constexpr size_t kStaleCap = 64;  // mirrors kMaxStaleDropsPerClaim
    for (size_t i = 0; i < kStale; ++i) {
        const std::string key = "k_stale_" + std::to_string(i);
        ASSERT_TRUE(InjectDfsOnlyObject(*service, key, 1024));
        ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), key));
        ASSERT_TRUE(CancelDfsPromotionTaskForTesting(service.get(), key));
    }
    // One live task sits behind the stale ones: equal heat, so the heap orders
    // by enqueue sequence and the live node is popped last.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_live", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_live"));
    ASSERT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), kStale + 1);

    // First heartbeat drains only the cap and returns nothing, without yet
    // reaching the live task buried underneath.
    auto first = service->DfsPromotionObjectHeartbeat(generate_uuid());
    ASSERT_TRUE(first.has_value());
    EXPECT_TRUE(first->empty());
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()),
              kStale + 1 - kStaleCap);

    // The next heartbeat clears the remainder and dispatches the live task.
    auto second = service->DfsPromotionObjectHeartbeat(generate_uuid());
    ASSERT_TRUE(second.has_value());
    ASSERT_EQ(second->size(), 1u);
    EXPECT_EQ(second->front().key, "k_live");
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionReaperReclaimsExpiredClaimedTask) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_expire", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_expire"));
    const UUID executor = generate_uuid();
    auto resp = service->DfsPromotionObjectHeartbeat(executor);
    ASSERT_TRUE(resp.has_value());
    ASSERT_EQ(resp->size(), 1u);
    ASSERT_TRUE(IsDfsSourceProtectedForTesting(service.get(), "k_expire"));
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);

    // A claimed-but-never-acknowledged task ages past its TTL: the reaper
    // erases the record, releases the DFS source protection and decrements
    // the in-flight counter.
    BackdateDfsPromotionTaskForTesting(service.get(), "k_expire",
                                       std::chrono::minutes(5));
    RunDfsPromotionReaperForTesting(service.get());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 0u);
    EXPECT_FALSE(IsDfsSourceProtectedForTesting(service.get(), "k_expire"));

    // The record is gone: re-admission is allowed again.
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_expire"));
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);
}

TEST_F(PromotionOnHitTest, DfsPromotionQueueRebuildDropsStaleNodes) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_s1", 1024));
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_s2", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_s1"));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_s2"));
    ASSERT_TRUE(CancelDfsPromotionTaskForTesting(service.get(), "k_s1"));
    // Physical queue still holds both nodes but the logical state has one task.
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 2u);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);

    // Force the oversize trigger (physical > limit): the rebuild drains the
    // heap, keeps only nodes that still match an unclaimed record, and pushes
    // them back - dropping the cancelled k_s1 node.
    OverrideDfsPromotionQueueLimitForTesting(service.get(), 1);
    RunDfsPromotionReaperForTesting(service.get());
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 1u);

    auto resp = service->DfsPromotionObjectHeartbeat(generate_uuid());
    ASSERT_TRUE(resp.has_value());
    ASSERT_EQ(resp->size(), 1u);
    EXPECT_EQ(resp->front().key, "k_s2");
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 0u);
}

// ---------------- DFS promotion execution layer ----------------
// The DFS channel reuses the SSD channel's RPCs (PromotionAllocStart /
// NotifyPromotionSuccess / NotifyPromotionFailure) but keeps its own global
// task table. These tests drive the master side of that reuse: staging a
// PROCESSING MEMORY replica for a claimed DFS task, committing it, aborting
// it, and reclaiming it on TTL expiry. The client-side executor
// (FileStorage::ProcessDfsPromotionTasks) is exercised separately.

TEST_F(PromotionOnHitTest,
       DfsPromotionAllocStartAndCommitCompletesMemoryReplica) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_commit_seg", kDefaultSegmentBase,
                              seg_size);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_commit", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_commit"));

    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);
    ASSERT_EQ(claimed->front().key, "k_dfs_commit");
    ASSERT_TRUE(claimed->front().source_dfs.has_value());

    // Stage a PROCESSING MEMORY replica through the shared AllocStart RPC.
    const std::vector<std::string> preferred;
    auto alloc = service->PromotionAllocStart(
        ctx.client_id, "k_dfs_commit", TenantId::Default(), 1024, preferred);
    ASSERT_TRUE(alloc.has_value()) << "AllocStart rejected the DFS task";
    EXPECT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_commit", [](const Replica& replica) {
                      return Replica::fn_is_memory_replica(replica) &&
                             Replica::fn_is_processing(replica);
                  }),
              1u);

    // Commit: the staged replica flips to COMPLETE and every DFS bookkeeping
    // entry is released (task table, in-flight, source protection).
    auto notify = service->NotifyPromotionSuccess(
        ctx.client_id, "k_dfs_commit", TenantId::Default());
    ASSERT_TRUE(notify.has_value());
    EXPECT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_commit", [](const Replica& replica) {
                      return Replica::fn_is_memory_replica(replica) &&
                             Replica::fn_is_completed(replica);
                  }),
              1u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_FALSE(IsDfsSourceProtectedForTesting(service.get(), "k_dfs_commit"));
}

TEST_F(PromotionOnHitTest, DfsPromotionNotifyFailureReleasesStagedReplica) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "dfs_fail_seg", kDefaultSegmentBase, seg_size);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_fail", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_fail"));
    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);

    const std::vector<std::string> preferred;
    auto alloc = service->PromotionAllocStart(
        ctx.client_id, "k_dfs_fail", TenantId::Default(), 1024, preferred);
    ASSERT_TRUE(alloc.has_value());
    ASSERT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_fail", [](const Replica& replica) {
                      return Replica::fn_is_memory_replica(replica);
                  }),
              1u);

    // Failure: the staged MEMORY replica is erased and all bookkeeping is
    // released, while the DFS source stays so the object remains retryable.
    auto notify = service->NotifyPromotionFailure(
        ctx.client_id, "k_dfs_fail", TenantId::Default());
    ASSERT_TRUE(notify.has_value());
    EXPECT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_fail", [](const Replica& replica) {
                      return Replica::fn_is_memory_replica(replica);
                  }),
              0u);
    EXPECT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_fail", [](const Replica& replica) {
                      return Replica::fn_is_dfs_replica(replica);
                  }),
              1u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_FALSE(IsDfsSourceProtectedForTesting(service.get(), "k_dfs_fail"));

    // Releasing a failed task must not poison the key: once the task record
    // is gone the object can be admitted again on a later hit.
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_fail"));
}

TEST_F(PromotionOnHitTest, DfsPromotionReaperReclaimsStagedProcessingReplica) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "dfs_reap_seg", kDefaultSegmentBase, seg_size);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_reap", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_reap"));
    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);

    const std::vector<std::string> preferred;
    auto alloc = service->PromotionAllocStart(
        ctx.client_id, "k_dfs_reap", TenantId::Default(), 1024, preferred);
    ASSERT_TRUE(alloc.has_value());

    // The holder stalls after staging: age the task past TTL and reap. The
    // staged PROCESSING MEMORY replica must be reclaimed with it, otherwise
    // the buffer leaks (no task record would point at it anymore).
    BackdateDfsPromotionTaskForTesting(service.get(), "k_dfs_reap",
                                       std::chrono::minutes(5));
    RunDfsPromotionReaperForTesting(service.get());
    EXPECT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_reap", [](const Replica& replica) {
                      return Replica::fn_is_memory_replica(replica);
                  }),
              0u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
}

// ---------------- Metrics / self-healing / anti-thrash cooldown --------------
// Covers the reconcile self-healing scan and ghost census plus the per-key
// anti-thrash cooldown that stops a key from being promoted repeatedly, and
// the dfs_promotion_* metric wiring. Metrics are process-global, so every
// assertion is a delta.

TEST_F(PromotionOnHitTest, DfsPromotionMetricsTrackAdmitCommitAndInFlight) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_metrics_seg", kDefaultSegmentBase,
                              seg_size);
    auto& mm = MasterMetricManager::instance();
    const int64_t admitted_pre = mm.get_dfs_promotion_admitted();
    const int64_t completed_pre = mm.get_dfs_promotion_completed();
    const int64_t bytes_pre = mm.get_dfs_promotion_completed_bytes();
    const int64_t in_flight_pre = mm.get_dfs_promotion_in_flight();

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_metric", 2048));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_metric"));
    EXPECT_EQ(mm.get_dfs_promotion_admitted(), admitted_pre + 1);
    EXPECT_EQ(mm.get_dfs_promotion_in_flight(), in_flight_pre + 1);

    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);
    const std::vector<std::string> preferred;
    ASSERT_TRUE(service
                    ->PromotionAllocStart(ctx.client_id, "k_dfs_metric",
                                          TenantId::Default(), 2048, preferred)
                    .has_value());
    ASSERT_TRUE(service
                    ->NotifyPromotionSuccess(ctx.client_id, "k_dfs_metric",
                                             TenantId::Default())
                    .has_value());
    EXPECT_EQ(mm.get_dfs_promotion_completed(), completed_pre + 1);
    EXPECT_EQ(mm.get_dfs_promotion_completed_bytes(), bytes_pre + 2048);
    EXPECT_EQ(mm.get_dfs_promotion_in_flight(), in_flight_pre);
}

TEST_F(PromotionOnHitTest, DfsPromotionMetricsTrackFailureAndExpiry) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_metrics_seg2", kDefaultSegmentBase,
                              seg_size);
    auto& mm = MasterMetricManager::instance();
    const int64_t failed_pre = mm.get_dfs_promotion_failed();
    const int64_t expired_pre = mm.get_dfs_promotion_expired();
    const int64_t in_flight_pre = mm.get_dfs_promotion_in_flight();

    // Failure path: the staged replica is released and the task is counted as
    // failed so success + failure accounts for every terminal task.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_metric_fail", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_metric_fail"));
    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);
    const std::vector<std::string> preferred;
    ASSERT_TRUE(
        service
            ->PromotionAllocStart(ctx.client_id, "k_dfs_metric_fail",
                                  TenantId::Default(), 1024, preferred)
            .has_value());
    ASSERT_TRUE(service
                    ->NotifyPromotionFailure(ctx.client_id, "k_dfs_metric_fail",
                                             TenantId::Default())
                    .has_value());
    EXPECT_EQ(mm.get_dfs_promotion_failed(), failed_pre + 1);
    EXPECT_EQ(mm.get_dfs_promotion_in_flight(), in_flight_pre);

    // Expiry path: a claimed-but-silent task ages past TTL and is counted as
    // expired (not as a failure).
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_metric_exp", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_metric_exp"));
    auto claimed2 = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed2.has_value());
    ASSERT_EQ(claimed2->size(), 1u);
    BackdateDfsPromotionTaskForTesting(service.get(), "k_dfs_metric_exp",
                                       std::chrono::minutes(5));
    RunDfsPromotionReaperForTesting(service.get());
    EXPECT_EQ(mm.get_dfs_promotion_expired(), expired_pre + 1);
    EXPECT_EQ(mm.get_dfs_promotion_in_flight(), in_flight_pre);
}

TEST_F(PromotionOnHitTest, DfsPromotionMetricsTrackCancellationOnRemoval) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    auto& mm = MasterMetricManager::instance();
    const int64_t cancelled_pre = mm.get_dfs_promotion_cancelled();
    const int64_t in_flight_pre = mm.get_dfs_promotion_in_flight();

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_metric_cancel", 1024));
    ASSERT_TRUE(
        AdmitDfsPromotionForTesting(service.get(), "k_dfs_metric_cancel"));
    // Erasing the object cancels the in-flight task through
    // ErasePromotionTaskIfPresent.
    ASSERT_TRUE(service->Remove("k_dfs_metric_cancel", TenantId::Default(),
                                /*force=*/true)
                    .has_value());
    EXPECT_EQ(mm.get_dfs_promotion_cancelled(), cancelled_pre + 1);
    EXPECT_EQ(mm.get_dfs_promotion_in_flight(), in_flight_pre);
}

TEST_F(PromotionOnHitTest, DfsPromotionLowWeightRejectionIsAttributed) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    auto& mm = MasterMetricManager::instance();
    const int64_t low_weight_pre = mm.get_dfs_promotion_rejected_low_weight();
    const int64_t frequency_pre = mm.get_dfs_promotion_rejected_frequency();

    // A fresh sketch has total weight 0, so the threshold degrades to the
    // absolute floor. A heat below that floor must be attributed to the
    // low-weight (cold start) bucket, not to the frequency bucket.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_loww", 1024));
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_loww",
                                                /*heat=*/1.0),
              DfsAdmissionResultBelowThreshold())
        << "threshold=" << GetDfsHeatThresholdForTesting(service.get());
    EXPECT_EQ(mm.get_dfs_promotion_rejected_low_weight(), low_weight_pre + 1);
    EXPECT_EQ(mm.get_dfs_promotion_rejected_frequency(), frequency_pre);
}

TEST_F(PromotionOnHitTest, DfsPromotionReconcileScanRegistersMissedSample) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    // The injected object holds a COMPLETE DFS replica but was never
    // registered in the heat sketch: exactly the "missed Add" drift the
    // self-healing sweep is meant to repair.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_scan", 1024));
    EXPECT_EQ(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_scan"), 0u);

    // One pass walks at most 128 of the 1024 shards (kMaxShardsPerPass), so
    // drive the sweep until its cursor wraps around and reaches this object's
    // shard. The scan is idempotent, so extra passes are harmless.
    for (int pass = 0;
         pass < 16 &&
         GetDfsLastAccessMinForTesting(service.get(), "k_dfs_scan") == 0;
         ++pass) {
        RunDfsPromotionReconcileScanForTesting(service.get());
    }
    EXPECT_NE(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_scan"), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionGhostCensusMatchesSketchWeight) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    auto& mm = MasterMetricManager::instance();

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_ghost", 1024));
    // Register + heat the sample through the read-hit state machine.
    for (int i = 0; i < 3; ++i) {
        ReconcileDfsHeatForTesting(service.get(), "k_dfs_ghost",
                                   /*access_hit=*/true);
    }
    RunDfsPromotionGhostMetricsForTesting(service.get());

    // No drift here: the weight reachable from live objects must be positive
    // and match the sketch's own total weight (a ghost would show as a gap).
    EXPECT_GT(mm.get_dfs_promotion_members_weight(), 0);
    EXPECT_NEAR(mm.get_dfs_promotion_members_weight(),
                mm.get_dfs_promotion_sketch_weight(), 1);
}

TEST_F(PromotionOnHitTest, DfsPromotionCooldownStampedOnCompletion) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx =
        PrepareSegment(*service, "dfs_cool_seg", kDefaultSegmentBase, seg_size);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_cool", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_cool"));
    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);
    const std::vector<std::string> preferred;
    ASSERT_TRUE(service
                    ->PromotionAllocStart(ctx.client_id, "k_dfs_cool",
                                          TenantId::Default(), 1024, preferred)
                    .has_value());
    EXPECT_EQ(GetDfsLastPromotedMinForTesting(service.get(), "k_dfs_cool"), 0u);

    ASSERT_TRUE(service
                    ->NotifyPromotionSuccess(ctx.client_id, "k_dfs_cool",
                                             TenantId::Default())
                    .has_value());
    // A completed promotion stamps the cooldown on the DFS source replica.
    EXPECT_NE(GetDfsLastPromotedMinForTesting(service.get(), "k_dfs_cool"), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionCooldownBlocksReAdmission) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_cool_block", 1024));
    // Pretend the key completed a promotion "now": the object is DFS-served
    // again (its MEMORY replica was evicted), so without the cooldown it would
    // be admitted immediately.
    const uint32_t now_min =
        static_cast<uint32_t>(DfsNowEpochMinForTesting(service.get()));
    SetDfsLastPromotedMinForTesting(service.get(), "k_dfs_cool_block", now_min);
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(),
                                                "k_dfs_cool_block"),
              DfsAdmissionResultCooling());

    // Once the window has elapsed the same key is admissible again.
    SetDfsLastPromotedMinForTesting(
        service.get(), "k_dfs_cool_block",
        now_min - GetDfsPromotionCooldownMinForTesting(service.get()) - 1);
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(),
                                                "k_dfs_cool_block"),
              DfsAdmissionResultAdmitted());
}

TEST_F(PromotionOnHitTest, DfsPromotionCooldownDisabledAllowsReAdmission) {
    auto config = MakeDfsPromotionConfig();
    config.dfs_promotion.cooldown_min = 0;  // disable the anti-thrash window
    auto service = std::make_unique<MasterService>(config);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_cool_off", 1024));
    const uint32_t now_min =
        static_cast<uint32_t>(DfsNowEpochMinForTesting(service.get()));
    SetDfsLastPromotedMinForTesting(service.get(), "k_dfs_cool_off", now_min);
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(),
                                                "k_dfs_cool_off"),
              DfsAdmissionResultAdmitted());
}

// ===================== DFS channel coverage hardening =====================
// The DFS channel has to be locked down well beyond the happy paths covered
// above: the heat state machine (membership / idempotence / active leave /
// generation swap), decay monotonicity, threshold provenance, every admission
// gate, queue dispatch ordering, holder authorization, and concurrent
// bookkeeping consistency. Every case below is deterministic: the epoch minute
// is injected through the friend funnels instead of sleeping.

// ---- Heat state machine + decay ----

TEST_F(PromotionOnHitTest, DfsPromotionHeatRegistersMemberOnNonHitTransition) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_reg", 1024));
    EXPECT_EQ(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_reg"), 0u);

    // A non-hit transition (MEMORY evicted / fresh generation) still has to
    // register the sample with heat 0, otherwise the object silently drops out
    // of the sketch and its later hits start from an unknown baseline.
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_reg", /*access_hit=*/false);
    EXPECT_NE(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_reg"), 0u);
    EXPECT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_reg"), 0.0);
    EXPECT_NEAR(GetDfsSketchTotalWeightForTesting(service.get()), 1.0, 0.05);

    // Already a member: further non-hit transitions must be no-ops, so the
    // collection does not grow without bound on every eviction sweep.
    const uint32_t registered_at =
        GetDfsLastAccessMinForTesting(service.get(), "k_dfs_reg");
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_reg", false);
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_reg", false);
    EXPECT_EQ(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_reg"),
              registered_at);
    EXPECT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_reg"), 0.0);
    EXPECT_NEAR(GetDfsSketchTotalWeightForTesting(service.get()), 1.0, 0.05);
}

TEST_F(PromotionOnHitTest, DfsPromotionHeatGrowsByOnePerDenseHit) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_dense", 1024));

    // No idle gap between hits (same epoch minute), so the decay factor is 1
    // and heat degenerates into a plain hit counter.
    for (int hit = 1; hit <= 4; ++hit) {
        ReconcileDfsHeatForTesting(service.get(), "k_dfs_dense",
                                   /*access_hit=*/true);
        EXPECT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_dense"),
                         static_cast<double>(hit));
    }
}

TEST_F(PromotionOnHitTest, DfsPromotionHeatDecaysByExactlyOneHalfLife) {
    auto config = MakeDfsPromotionConfig();
    auto service = std::make_unique<MasterService>(config);
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_halflife", 1024));

    const uint32_t now_min =
        static_cast<uint32_t>(DfsNowEpochMinForTesting(service.get()));
    const uint32_t half_life = config.dfs_promotion.half_life_min;
    ASSERT_GT(half_life, 0u);

    // heat 8 exactly one half-life ago -> 8 * 2^-1 + 1 = 5 after this hit.
    SetDfsHeatStateForTesting(service.get(), "k_dfs_halflife", /*heat=*/8.0f,
                              now_min - half_life);
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_halflife", true);
    EXPECT_NEAR(GetDfsHeatForTesting(service.get(), "k_dfs_halflife"), 5.0,
                1e-6);
    EXPECT_EQ(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_halflife"),
              now_min);
}

TEST_F(PromotionOnHitTest, DfsPromotionHeatSparseHitsStayColderThanDenseHits) {
    auto config = MakeDfsPromotionConfig();
    auto service = std::make_unique<MasterService>(config);
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_dense_cmp", 1024));
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_sparse_cmp", 1024));

    const uint32_t now_min =
        static_cast<uint32_t>(DfsNowEpochMinForTesting(service.get()));
    const uint32_t half_life = config.dfs_promotion.half_life_min;

    // Identical starting heat (16) and one hit each: the key that idled a full
    // half-life before the hit must come out strictly colder. This pins decay
    // monotonicity: idling can only cool a key, never heat it.
    SetDfsHeatStateForTesting(service.get(), "k_dfs_dense_cmp", 16.0f, now_min);
    SetDfsHeatStateForTesting(service.get(), "k_dfs_sparse_cmp", 16.0f,
                              now_min - half_life);
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_dense_cmp", true);
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_sparse_cmp", true);

    const double dense = GetDfsHeatForTesting(service.get(), "k_dfs_dense_cmp");
    const double sparse =
        GetDfsHeatForTesting(service.get(), "k_dfs_sparse_cmp");
    EXPECT_DOUBLE_EQ(dense, 17.0);
    EXPECT_DOUBLE_EQ(sparse, 9.0);
    EXPECT_LT(sparse, dense);
}

TEST_F(PromotionOnHitTest, DfsPromotionHeatSaturatesAtSketchMaxValue) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_clamp", 1024));

    const uint32_t now_min =
        static_cast<uint32_t>(DfsNowEpochMinForTesting(service.get()));
    // Heat doubles as the sketch's indexed value, whose upper bound is 1e7.
    // A hit on a saturated key must clamp instead of overflowing the indexed
    // range, which would make the sketch reject the sample outright.
    SetDfsHeatStateForTesting(service.get(), "k_dfs_clamp", 1e7f, now_min);
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_clamp", true);

    EXPECT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_clamp"), 1e7);
    EXPECT_EQ(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_clamp"),
              now_min);
    // Still exactly one sample in the collection: the clamped replace must not
    // have inserted a second bucket.
    EXPECT_NEAR(GetDfsSketchTotalWeightForTesting(service.get()), 1.0, 0.05);
}

TEST_F(PromotionOnHitTest, DfsPromotionHeatActiveLeaveClearsRegisteredSample) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_leave_seg", kDefaultSegmentBase,
                              seg_size);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_leave", 1024));
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_leave", true);
    ASSERT_NE(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_leave"), 0u);
    ASSERT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_leave"), 1.0);
    ASSERT_NEAR(GetDfsSketchTotalWeightForTesting(service.get()), 1.0, 0.05);

    // A MEMORY/LOCAL_DISK replica takes over as the top COMPLETE replica: the
    // object left the DFS collection, so the registered sample must be removed
    // from the sketch and zeroed in the metadata instead of being left to decay
    // and skew the P90 threshold.
    ASSERT_TRUE(InjectLocalDiskReplica(*service, ctx.client_id, "k_dfs_leave",
                                       1024, ctx.segment_name));
    ASSERT_TRUE(MoveDfsReplicaLastForTesting(service.get(), "k_dfs_leave"));
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_leave", true);

    EXPECT_EQ(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_leave"), 0u);
    EXPECT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_leave"), 0.0);
    EXPECT_NEAR(GetDfsSketchTotalWeightForTesting(service.get()), 0.0, 0.05);

    // The leave path runs on every later reconcile; a DFS replica that no
    // longer carries a sample must not be removed a second time (which would
    // drive the sketch weight negative).
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_leave", false);
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_leave", false);
    EXPECT_NEAR(GetDfsSketchTotalWeightForTesting(service.get()), 0.0, 0.05);
}

// H5 destroy-style sink: erasing the object destroys the replica records that
// carry the registered heat/timestamp, so the sample has to be Removed *first*.
// Unlike the active-leave case above this runs through FreeDfsReplicas, so it
// also pins down that the heat cleanup is not coupled to the DFS allocator
// being present (the unit harness never initializes one).
TEST_F(PromotionOnHitTest, DfsPromotionHeatDestructionRemovesRegisteredSample) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_destroy", 1024));
    ReconcileDfsHeatForTesting(service.get(), "k_dfs_destroy", true);
    ASSERT_NE(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_destroy"), 0u);
    ASSERT_NEAR(GetDfsSketchTotalWeightForTesting(service.get()), 1.0, 0.05);

    auto rm =
        service->Remove("k_dfs_destroy", TenantId::Default(), /*force=*/true);
    ASSERT_TRUE(rm.has_value()) << "Remove failed; error=" << rm.error();

    // No ghost: the sample left the sketch together with the replica records.
    EXPECT_NEAR(GetDfsSketchTotalWeightForTesting(service.get()), 0.0, 0.05);

    // A second Remove has nothing to erase and must not touch the sketch again.
    auto rm_again =
        service->Remove("k_dfs_destroy", TenantId::Default(), /*force=*/true);
    EXPECT_FALSE(rm_again.has_value());
    EXPECT_NEAR(GetDfsSketchTotalWeightForTesting(service.get()), 0.0, 0.05);
}

TEST_F(PromotionOnHitTest, DfsPromotionHeatIgnoresObjectWithoutCompleteReplica) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    // Only a PROCESSING DFS replica exists: the object is not readable yet, so
    // it is not part of the DFS collection and must not enter the sketch.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_no_complete", 1024,
                                    "/dfs/obj/pending", 0, /*hard_pinned=*/false,
                                    ReplicaStatus::PROCESSING));

    ReconcileDfsHeatForTesting(service.get(), "k_dfs_no_complete", true);
    EXPECT_EQ(GetDfsLastAccessMinForTesting(service.get(), "k_dfs_no_complete"),
              0u);
    EXPECT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_no_complete"),
                     0.0);
    EXPECT_DOUBLE_EQ(GetDfsSketchTotalWeightForTesting(service.get()), 0.0);
}

// ---- Read-hit hook parity ----
// GetReplicaList is the natural hook for observing the serving layer, and the
// SSD channel (promotion_on_hit) hooks both the single-key and the batch path.
// The DFS channel used to hook only the single-key path, so DFS-only objects
// read through BatchGetReplicaList never accumulated dfs_heat and were never
// promoted. These two tests pin the resulting symmetry.

TEST_F(PromotionOnHitTest, BatchGetReplicaListBumpsDfsHeat) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_single", 1024,
                                    "/dfs/obj/single", 4096));
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_batch", 1024,
                                    "/dfs/obj/batch", 4096));

    // Baseline: the single-key path has always bumped the heat.
    auto single = service->GetReplicaList("k_dfs_single", TenantId::Default());
    ASSERT_TRUE(single.has_value());
    ASSERT_FALSE(single->replicas.empty());
    ASSERT_TRUE(single->replicas[0].is_dfs_replica())
        << "precondition: DFS must be the serving layer";
    EXPECT_GT(GetDfsHeatForTesting(service.get(), "k_dfs_single"), 0.0);

    // The batch path must behave identically: one DFS-served read -> heat 1.
    auto batch = service->BatchGetReplicaList(
        std::vector<std::string>{"k_dfs_batch"}, TenantId::Default());
    ASSERT_EQ(batch.size(), 1u);
    ASSERT_TRUE(batch[0].has_value());
    ASSERT_FALSE(batch[0]->replicas.empty());
    ASSERT_TRUE(batch[0]->replicas[0].is_dfs_replica())
        << "precondition: DFS must be the serving layer";
    EXPECT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_batch"), 1.0);

    service->RemoveAll();
}

TEST_F(PromotionOnHitTest, BatchGetReplicaListForAdminDoesNotBumpDfsHeat) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_admin", 1024,
                                    "/dfs/obj/admin", 4096));

    auto result = service->BatchGetReplicaListForAdmin(
        std::vector<std::string>{"k_dfs_admin"}, TenantId::Default());
    ASSERT_EQ(result.size(), 1u);
    ASSERT_TRUE(result[0].has_value());
    ASSERT_FALSE(result[0]->replicas.empty());
    ASSERT_TRUE(result[0]->replicas[0].is_dfs_replica())
        << "precondition: DFS must be the serving layer";

    // The read-only admin query observes state; it must not feed the heat
    // sketch (same contract as the SSD channel's admin path).
    EXPECT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_admin"), 0.0);
    EXPECT_DOUBLE_EQ(GetDfsSketchTotalWeightForTesting(service.get()), 0.0);

    service->RemoveAll();
}

// ---- Threshold provenance ----

TEST_F(PromotionOnHitTest, DfsPromotionThresholdStartsAtAbsoluteFloor) {
    auto config = MakeDfsPromotionConfig();
    auto service = std::make_unique<MasterService>(config);

    // Regression for the threshold-cache race: T must be published as the
    // absolute floor
    // before any refresh runs, so a reader that loses the refresh CAS can never
    // observe 0.0 (which would admit every DFS object on the node).
    EXPECT_DOUBLE_EQ(GetDfsHeatThresholdForTesting(service.get()),
                     config.dfs_promotion.absolute_hot_threshold);
    EXPECT_TRUE(GetDfsThresholdLowWeightForTesting(service.get()));
    EXPECT_DOUBLE_EQ(GetDfsHeatThresholdForTesting(service.get()),
                     config.dfs_promotion.absolute_hot_threshold);
}

TEST_F(PromotionOnHitTest, DfsPromotionThresholdFallsBackBelowMinTotalWeight) {
    auto config = MakeDfsPromotionConfig();
    auto service = std::make_unique<MasterService>(config);
    const TenantId tenant = TenantId::Default();

    // The fixture pins min_total_weight to 8.0: seven very hot samples are
    // still not enough to trust a P90, so the floor must win and be attributed
    // as low-weight.
    for (int i = 0; i < 7; ++i) {
        ASSERT_TRUE(AddDfsHeatSampleForTesting(
            service.get(), tenant.MakeScopedKey("hot_" + std::to_string(i)),
            100.0));
    }
    RefreshDfsHeatThresholdForTesting(service.get());
    EXPECT_TRUE(GetDfsThresholdLowWeightForTesting(service.get()));
    EXPECT_DOUBLE_EQ(GetDfsHeatThresholdForTesting(service.get()),
                     config.dfs_promotion.absolute_hot_threshold);
}

TEST_F(PromotionOnHitTest, DfsPromotionLowWeightFlagTracksJitterSkippedRefresh) {
    auto config = MakeDfsPromotionConfig();
    auto service = std::make_unique<MasterService>(config);
    const TenantId tenant = TenantId::Default();
    const double absolute_floor = config.dfs_promotion.absolute_hot_threshold;

    // Cold start publishes the floor and marks the sketch low-weight.
    EXPECT_DOUBLE_EQ(GetDfsHeatThresholdForTesting(service.get()),
                     absolute_floor);
    ASSERT_TRUE(GetDfsThresholdLowWeightForTesting(service.get()));

    // Enough samples to clear min_total_weight (8.0), yet every heat stays
    // below the floor: p90 < floor, so the candidate T equals the cached T and
    // the minimal-change filter skips the refresh altogether. This is exactly
    // the steady state of a node that only ever sees cold DFS keys.
    for (int i = 0; i < 16; ++i) {
        ASSERT_TRUE(AddDfsHeatSampleForTesting(
            service.get(), tenant.MakeScopedKey("warm_" + std::to_string(i)),
            /*heat=*/1.0));
    }
    RefreshDfsHeatThresholdForTesting(service.get());

    // The jitter filter guards T only. T stays on the floor, but the low-weight
    // flag must still be recomputed: leaving it at its cold-start value makes
    // every later below-threshold rejection attribute to "cold start" instead
    // of "frequency", so dfs_promotion_rejected_low_weight stays inflated and
    // dfs_promotion_rejected_frequency stays pinned at zero.
    EXPECT_DOUBLE_EQ(GetDfsHeatThresholdForTesting(service.get()),
                     absolute_floor);
    EXPECT_FALSE(GetDfsThresholdLowWeightForTesting(service.get()));
}

TEST_F(PromotionOnHitTest, DfsPromotionThresholdUsesP90OnceWeightIsSufficient) {
    auto config = MakeDfsPromotionConfig();
    auto service = std::make_unique<MasterService>(config);
    const TenantId tenant = TenantId::Default();

    // 64 samples at heat 100 give weight 64 >= min_total_weight, so the P90
    // becomes trustworthy and T has to jump from the floor to ~100.
    for (int i = 0; i < 64; ++i) {
        ASSERT_TRUE(AddDfsHeatSampleForTesting(
            service.get(), tenant.MakeScopedKey("p90_" + std::to_string(i)),
            100.0));
    }
    RefreshDfsHeatThresholdForTesting(service.get());
    EXPECT_FALSE(GetDfsThresholdLowWeightForTesting(service.get()));

    const double threshold = GetDfsHeatThresholdForTesting(service.get());
    EXPECT_GT(threshold, config.dfs_promotion.absolute_hot_threshold);
    // The sketch is lossy (relative_accuracy 0.05) and returns a bucket
    // representative, so only the documented tolerance can be asserted.
    EXPECT_NEAR(threshold, 100.0, 15.0);

    // Heat below T is rejected and attributed to the frequency path; heat at T
    // passes the heat gate and reaches the queue.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_below_t", 1024));
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_at_t", 1024));
    const int64_t frequency_pre =
        MasterMetricManager::instance().get_dfs_promotion_rejected_frequency();
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_below_t",
                                                threshold - 10.0),
              DfsAdmissionResultBelowThreshold());
    // T came from the P90 path (not the low-weight fallback), so this rejection
    // belongs to the frequency counter rather than the low-weight one.
    EXPECT_EQ(
        MasterMetricManager::instance().get_dfs_promotion_rejected_frequency(),
        frequency_pre + 1);
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_at_t",
                                                threshold),
              DfsAdmissionResultAdmitted());
}

TEST_F(PromotionOnHitTest, DfsPromotionThresholdRefreshZeroTracksSketch) {
    auto config = MakeDfsPromotionConfig();
    config.dfs_promotion.threshold_refresh_min = 0;  // dev mode: no cache
    auto service = std::make_unique<MasterService>(config);
    const TenantId tenant = TenantId::Default();

    for (int i = 0; i < 8; ++i) {
        ASSERT_TRUE(AddDfsHeatSampleForTesting(
            service.get(), tenant.MakeScopedKey("cool_" + std::to_string(i)),
            50.0));
    }
    const double cool = GetDfsHeatThresholdForTesting(service.get());
    EXPECT_GT(cool, config.dfs_promotion.absolute_hot_threshold);
    EXPECT_NEAR(cool, 50.0, 10.0);

    // refresh_min == 0 means the next evaluation must pick the hotter
    // distribution up immediately, without waiting for a cache window.
    for (int i = 0; i < 64; ++i) {
        ASSERT_TRUE(AddDfsHeatSampleForTesting(
            service.get(), tenant.MakeScopedKey("hot_" + std::to_string(i)),
            1000.0));
    }
    const double hot = GetDfsHeatThresholdForTesting(service.get());
    EXPECT_GT(hot, cool);
    EXPECT_GT(hot, 800.0);
    EXPECT_LT(hot, 1500.0);
}

TEST_F(PromotionOnHitTest, DfsPromotionThresholdCacheHoldsWithinRefreshWindow) {
    auto config = MakeDfsPromotionConfig();
    config.dfs_promotion.threshold_refresh_min = 60;  // one hour window
    auto service = std::make_unique<MasterService>(config);
    const TenantId tenant = TenantId::Default();

    // The first evaluation publishes the cold-start floor and claims the
    // window.
    const double initial = GetDfsHeatThresholdForTesting(service.get());
    EXPECT_DOUBLE_EQ(initial, config.dfs_promotion.absolute_hot_threshold);

    // The sketch turns hot, but the window has not elapsed: T must stay put so
    // a burst of new samples cannot silently re-open the gate.
    for (int i = 0; i < 64; ++i) {
        ASSERT_TRUE(AddDfsHeatSampleForTesting(
            service.get(), tenant.MakeScopedKey("burst_" + std::to_string(i)),
            1000.0));
    }
    EXPECT_DOUBLE_EQ(GetDfsHeatThresholdForTesting(service.get()), initial);

    // An explicit refresh (the maintenance tick / window expiry) publishes the
    // new estimate.
    RefreshDfsHeatThresholdForTesting(service.get());
    EXPECT_GT(GetDfsHeatThresholdForTesting(service.get()), initial);
}

// ---- Admission gate coverage ----

TEST_F(PromotionOnHitTest, DfsPromotionAdmissionRejectedWhenChannelDisabled) {
    auto config = MakeDfsPromotionConfig();
    config.dfs_promotion.enable = false;
    auto service = std::make_unique<MasterService>(config);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_off", 1024));
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_off"),
              DfsAdmissionResultDisabled());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionAdmissionRejectedWhenMemoryReplicaPresent) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_mem_seg", kDefaultSegmentBase,
                              seg_size);

    // A COMPLETE MEMORY replica means the object already left the DFS
    // collection; admission must refuse before touching the queue.
    PutObject(*service, ctx.client_id, "k_dfs_mem");
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_mem"),
              DfsAdmissionResultMemoryPresent());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionAdmissionRejectedWithoutCompleteDfsSource) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    // A PROCESSING DFS replica is not a usable copy source: no COMPLETE replica
    // exists yet, so the source gate must reject.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_processing", 1024,
                                    "/dfs/obj/proc", 0, /*hard_pinned=*/false,
                                    ReplicaStatus::PROCESSING));
    EXPECT_EQ(
        AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_processing"),
        DfsAdmissionResultNoDfsSource());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionAdmissionRejectedForHardPinnedObject) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    // Hard-pinned objects must never be moved, regardless of how hot they are.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_hard_pin", 1024,
                                    "/dfs/obj/hard_pin", 0,
                                    /*hard_pinned=*/true));
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_hard_pin"),
              DfsAdmissionResultHardPinned());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionAdmissionRejectedWhileWriterLeaseActive) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    // An active writer lease means the object is being written right now; the
    // admission must not race the writer.
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_lease", 1024));
    GrantObjectLeaseForTesting(service.get(), "k_dfs_lease", 60000);
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_lease"),
              DfsAdmissionResultLeaseActive());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionAdmissionRejectedAtInFlightCap) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    auto& metrics = MasterMetricManager::instance();
    const int64_t rejected_pre = metrics.get_dfs_promotion_rejected_cap();

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_cap", 1024));
    // in_flight (0) >= limit (0) -> the queue is saturated.
    OverrideDfsPromotionQueueLimitForTesting(service.get(), 0);
    EXPECT_EQ(AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_cap"),
              DfsAdmissionResultQueueCap());
    EXPECT_EQ(metrics.get_dfs_promotion_rejected_cap(), rejected_pre + 1);
    // A cap rejection records *no* candidate. The cap is a property of the
    // global queue, not of this key: the key is not colder, its source is not
    // broken and waiting changes nothing about it, so there is no per-key state
    // for the retry sweep to act on — the next read hit re-attempts admission
    // for free. Recording one would only spend a kDfsPromotionCandidateLimit
    // slot that a genuinely retryable key (kLeaseActive, kNoDfsSource, ...)
    // needs, and make the sweep take a shard lock to re-test a verdict that is
    // already known.
    EXPECT_EQ(GetDfsCandidateCountForTesting(service.get(), "k_dfs_cap"), 0u);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 0u);
}

// A saturated queue must not make the read path pay for an admission attempt it
// cannot win — but it must also not freeze heat sampling, because the sketch
// (not the queue) is what decides admission once space frees up.
TEST_F(PromotionOnHitTest, DfsPromotionReadHitSkipsAdmissionWhileQueueSaturated) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    auto& metrics = MasterMetricManager::instance();

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_sat", 1024));
    // in_flight (0) >= limit (0) -> the queue is saturated for every key.
    OverrideDfsPromotionQueueLimitForTesting(service.get(), 0);

    const int64_t cap_pre = metrics.get_dfs_promotion_rejected_cap();

    ReconcileDfsHeatOnReadForTesting(service.get(), "k_dfs_sat");

    // Heat sampling keeps running: the first DFS-served hit registers the key
    // with heat=1. Freezing the samples while the queue is full would let every
    // key decay away and stall the channel long after the backlog drained.
    EXPECT_DOUBLE_EQ(GetDfsHeatForTesting(service.get(), "k_dfs_sat"), 1.0);
    // The admission attempt is short-circuited before the shard lock, so the
    // read never reaches the cap gate and is not counted as a cap rejection.
    EXPECT_EQ(metrics.get_dfs_promotion_rejected_cap(), cap_pre);
    EXPECT_EQ(GetDfsCandidateCountForTesting(service.get(), "k_dfs_sat"), 0u);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionAdmissionRejectedAboveDramWatermark) {
    // The high watermark is a const member fixed at construction, so force the
    // gate through the config: ratio 0 means the current DRAM usage (0.0 with
    // no mounted capacity) is already "above" it, and promoting more into DRAM
    // would fight the eviction thread.
    auto config = MakeDfsPromotionConfig();
    config.eviction_high_watermark_ratio = 0.0;
    auto service = std::make_unique<MasterService>(config);
    auto& metrics = MasterMetricManager::instance();
    const int64_t rejected_pre = metrics.get_dfs_promotion_rejected_watermark();

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_watermark", 1024));
    EXPECT_EQ(
        AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_watermark"),
        DfsAdmissionResultWatermark());
    EXPECT_EQ(metrics.get_dfs_promotion_rejected_watermark(), rejected_pre + 1);
    EXPECT_EQ(GetDfsCandidateCountForTesting(service.get(), "k_dfs_watermark"),
              1u);
    EXPECT_EQ(GetDfsCandidateReasonForTesting(service.get(), "k_dfs_watermark"),
              DfsCandidateReasonWatermark());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionAdmissionRejectedWhenSsdChannelTaskInFlight) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_cross_dup", 1024));
    // The SSD channel's task table is authoritative for the shared per-key
    // dedup: a queued or claimed SSD promotion must block a DFS one, otherwise
    // both channels would race to publish the same MEMORY replica.
    AddSsdPromotionTaskForTesting(service.get(), "k_dfs_cross_dup");
    EXPECT_EQ(
        AdmitDfsPromotionResultForTesting(service.get(), "k_dfs_cross_dup"),
        DfsAdmissionResultDupInFlight());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 0u);
}

// The reverse of the case above: the shared per-key in-flight gate has to be
// symmetric. Once a key's DFS promotion is queued, a later LOCAL_DISK replica
// must not let the SSD channel stage a second PROCESSING MEMORY replica for the
// same key (at most one in-flight MEMORY promotion per key).
TEST_F(PromotionOnHitTest, LocalDiskChannelRejectedWhenDfsTaskInFlight) {
    auto config = MakeDfsPromotionConfig();
    config.promotion_on_hit = true;
    config.promotion_admission_threshold = 1;
    config.default_kv_lease_ttl = 2000;
    auto service = std::make_unique<MasterService>(config);
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_ssd_dedup_seg", kDefaultSegmentBase,
                              seg_size);

    const std::string key = "k_dual_dedup";
    ASSERT_TRUE(InjectDfsOnlyObject(*service, key, 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), key));
    ASSERT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);

    // Attach a LOCAL_DISK replica while the DFS task is still in flight so the
    // SSD channel becomes eligible (!any_memory && any_local_disk).
    ASSERT_TRUE(AttachLocalDiskReplicaForTesting(service.get(), ctx.client_id,
                                                 key, 1024, ctx.segment_name));

    auto& mm = MasterMetricManager::instance();
    const int64_t admitted_pre = mm.get_promotion_admitted();

    auto result = service->GetReplicaList(key, TenantId::Default());
    ASSERT_TRUE(result.has_value());

    // No second (SSD) task may appear, and the DFS task must be untouched.
    EXPECT_EQ(mm.get_promotion_admitted(), admitted_pre)
        << "SSD channel must dedup against the in-flight DFS task";
    auto pending = service->PromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(pending->size(), 0u);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);
}

// ---- Queue dispatch ----

TEST_F(PromotionOnHitTest, DfsPromotionHeartbeatDispatchesHottestTaskFirst) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_q_cold", 1024));
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_q_warm", 1024));
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_q_hot", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_q_cold", 3.0));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_q_warm", 30.0));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_q_hot", 300.0));

    // The heap is a max-heap on the admission-time heat snapshot, so the
    // hottest task must come out first even though it was enqueued last.
    const UUID executor = generate_uuid();
    for (const char* expected : {"k_dfs_q_hot", "k_dfs_q_warm",
                                 "k_dfs_q_cold"}) {
        auto batch = service->DfsPromotionObjectHeartbeat(executor);
        ASSERT_TRUE(batch.has_value());
        ASSERT_EQ(batch->size(), 1u) << "expected " << expected;
        EXPECT_EQ(batch->front().key, expected);
    }
    auto drained = service->DfsPromotionObjectHeartbeat(executor);
    ASSERT_TRUE(drained.has_value());
    EXPECT_TRUE(drained->empty());
}

TEST_F(PromotionOnHitTest, DfsPromotionHeartbeatHonorsMaxPerHeartbeat) {
    auto config = MakeDfsPromotionConfig();
    config.dfs_promotion.max_per_heartbeat = 3;
    auto service = std::make_unique<MasterService>(config);

    constexpr int kTasks = 5;
    for (int i = 0; i < kTasks; ++i) {
        const std::string key = "k_dfs_batch_" + std::to_string(i);
        ASSERT_TRUE(InjectDfsOnlyObject(*service, key, 1024));
        ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), key, 10.0 + i));
    }

    const UUID executor = generate_uuid();
    auto first = service->DfsPromotionObjectHeartbeat(executor);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->size(), 3u);  // exactly max_per_heartbeat
    auto second = service->DfsPromotionObjectHeartbeat(executor);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->size(), 2u);  // remainder
    auto third = service->DfsPromotionObjectHeartbeat(executor);
    ASSERT_TRUE(third.has_value());
    EXPECT_TRUE(third->empty());

    // No task may be handed out twice across the batches.
    std::set<std::string> keys;
    for (const auto& task : *first) {
        EXPECT_TRUE(keys.insert(task.key).second);
    }
    for (const auto& task : *second) {
        EXPECT_TRUE(keys.insert(task.key).second);
    }
    EXPECT_EQ(keys.size(), static_cast<size_t>(kTasks));
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()),
              static_cast<uint32_t>(kTasks));
}

// ---- Execution-layer authorization ----

TEST_F(PromotionOnHitTest, DfsPromotionAllocStartRejectsNonHolder) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_holder_seg", kDefaultSegmentBase,
                              seg_size);
    const UUID other_client = PrepareLocalDiskOnlyClient(*service);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_holder", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_holder"));
    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);

    // Only the claiming client may stage the promotion target; anyone else
    // would publish a replica it does not own.
    const std::vector<std::string> preferred;
    auto stolen = service->PromotionAllocStart(
        other_client, "k_dfs_holder", TenantId::Default(), 1024, preferred);
    EXPECT_FALSE(stolen.has_value());
    EXPECT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_holder",
                  [](const Replica& replica) {
                      return Replica::fn_is_memory_replica(replica);
                  }),
              0u);

    // The task is untouched, so the real holder can still stage it.
    auto owned = service->PromotionAllocStart(
        ctx.client_id, "k_dfs_holder", TenantId::Default(), 1024, preferred);
    ASSERT_TRUE(owned.has_value());
}

TEST_F(PromotionOnHitTest, DfsPromotionAllocStartRejectsSizeMismatch) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_size_seg", kDefaultSegmentBase,
                              seg_size);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_size", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_size"));
    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);

    // A target whose size does not match the source descriptor would produce a
    // replica that can never be committed.
    const std::vector<std::string> preferred;
    auto mismatched = service->PromotionAllocStart(
        ctx.client_id, "k_dfs_size", TenantId::Default(), 2048, preferred);
    EXPECT_FALSE(mismatched.has_value());
    EXPECT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_size",
                  [](const Replica& replica) {
                      return Replica::fn_is_memory_replica(replica);
                  }),
              0u);

    auto matched = service->PromotionAllocStart(
        ctx.client_id, "k_dfs_size", TenantId::Default(), 1024, preferred);
    ASSERT_TRUE(matched.has_value());
}

TEST_F(PromotionOnHitTest, DfsPromotionAllocStartRejectsReapedTask) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_reaped_seg", kDefaultSegmentBase,
                              seg_size);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_reaped", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_reaped"));
    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);

    // The client went away; the reaper reclaims the task, after which a late
    // alloc-start must be refused instead of resurrecting a dead promotion.
    BackdateDfsPromotionTaskForTesting(service.get(), "k_dfs_reaped",
                                       std::chrono::minutes(5));
    RunDfsPromotionReaperForTesting(service.get());
    ASSERT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 0u);

    const std::vector<std::string> preferred;
    auto late = service->PromotionAllocStart(
        ctx.client_id, "k_dfs_reaped", TenantId::Default(), 1024, preferred);
    EXPECT_FALSE(late.has_value());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_FALSE(IsDfsSourceProtectedForTesting(service.get(), "k_dfs_reaped"));
}

TEST_F(PromotionOnHitTest, DfsPromotionAllocStartRejectsSecondStage) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_double_seg", kDefaultSegmentBase,
                              seg_size);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_double", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_double"));
    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);

    const std::vector<std::string> preferred;
    ASSERT_TRUE(service
                    ->PromotionAllocStart(ctx.client_id, "k_dfs_double",
                                          TenantId::Default(), 1024, preferred)
                    .has_value());
    // A retry (client lost the response) must not stage a second replica.
    auto again = service->PromotionAllocStart(
        ctx.client_id, "k_dfs_double", TenantId::Default(), 1024, preferred);
    EXPECT_FALSE(again.has_value());
    EXPECT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_double",
                  [](const Replica& replica) {
                      return Replica::fn_is_memory_replica(replica);
                  }),
              1u);
}

TEST_F(PromotionOnHitTest, DfsPromotionNotifyRejectsNonHolder) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());
    constexpr size_t seg_size = 1024 * 1024 * 16;
    auto ctx = PrepareSegment(*service, "dfs_notify_seg", kDefaultSegmentBase,
                              seg_size);
    const UUID other_client = PrepareLocalDiskOnlyClient(*service);

    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_notify", 1024));
    ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), "k_dfs_notify"));
    auto claimed = service->DfsPromotionObjectHeartbeat(ctx.client_id);
    ASSERT_TRUE(claimed.has_value());
    ASSERT_EQ(claimed->size(), 1u);

    const std::vector<std::string> preferred;
    ASSERT_TRUE(service
                    ->PromotionAllocStart(ctx.client_id, "k_dfs_notify",
                                          TenantId::Default(), 1024, preferred)
                    .has_value());

    // Neither terminal notification may be accepted from a non-holder: a bogus
    // success would publish an unverified replica, a bogus failure would tear
    // down a healthy promotion.
    auto bogus_success = service->NotifyPromotionSuccess(
        other_client, "k_dfs_notify", TenantId::Default());
    EXPECT_FALSE(bogus_success.has_value());
    auto bogus_failure = service->NotifyPromotionFailure(
        other_client, "k_dfs_notify", TenantId::Default());
    EXPECT_FALSE(bogus_failure.has_value());

    // The task survived both: still in flight, still owned by the holder.
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 1u);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);
    EXPECT_EQ(CountReplicasForTesting(
                  service.get(), "k_dfs_notify",
                  [](const Replica& replica) {
                      return Replica::fn_is_memory_replica(replica) &&
                             Replica::fn_is_completed(replica);
                  }),
              0u);
}

// ---- Reload / leader takeover ----

TEST_F(PromotionOnHitTest, DfsPromotionReloadDropsAllInMemoryTaskState) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    constexpr int kTasks = 3;
    for (int i = 0; i < kTasks; ++i) {
        const std::string key = "k_dfs_reload_" + std::to_string(i);
        ASSERT_TRUE(InjectDfsOnlyObject(*service, key, 1024));
        ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), key, 50.0));
    }
    ASSERT_EQ(GetDfsPromotionInFlightForTesting(service.get()),
              static_cast<uint32_t>(kTasks));
    ASSERT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()),
              static_cast<size_t>(kTasks));
    ASSERT_TRUE(IsDfsSourceProtectedForTesting(service.get(), "k_dfs_reload_0"));
    ASSERT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()),
              static_cast<size_t>(kTasks));

    // No DFS task survives a reload: the new leader rebuilds the collection
    // from reads, so every in-memory record must go, including the source
    // protection and the published threshold.
    ClearCandidatesForReloadForTesting(service.get());

    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 0u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 0u);
    EXPECT_FALSE(IsDfsSourceProtectedForTesting(service.get(), "k_dfs_reload_0"));
    EXPECT_EQ(GetDfsCandidateCountForTesting(service.get(), "k_dfs_reload_0"),
              0u);
    // The physical queue is process-local and cannot survive the takeover
    // either: it is dropped eagerly alongside the records it referenced, so no
    // dead node is left for a later claim to walk (each stale node costs a
    // metadata shard lock). Without the eager clear the residual nodes would
    // only drain through the claim path or the oversized-queue rebuild.
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 0u);
    // The conservative floor is re-published together with the cache stamp.
    EXPECT_DOUBLE_EQ(GetDfsHeatThresholdForTesting(service.get()),
                     MakeDfsPromotionConfig().dfs_promotion.absolute_hot_threshold);

    // A stale task must never be handed to a client after the takeover.
    auto batch = service->DfsPromotionObjectHeartbeat(generate_uuid());
    ASSERT_TRUE(batch.has_value());
    EXPECT_TRUE(batch->empty());
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 0u);
}

// ---- Concurrency / stress: bookkeeping under contention ----

TEST_F(PromotionOnHitTest, DfsPromotionConcurrentAdmissionsKeepBookkeepingConsistent) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    constexpr int kThreads = 8;
    constexpr int kKeysPerThread = 64;
    constexpr int kKeys = kThreads * kKeysPerThread;
    for (int i = 0; i < kKeys; ++i) {
        ASSERT_TRUE(
            InjectDfsOnlyObject(*service, "k_dfs_race_" + std::to_string(i),
                                1024));
    }

    std::atomic<int> admitted{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int i = 0; i < kKeysPerThread; ++i) {
                const std::string key =
                    "k_dfs_race_" + std::to_string(t * kKeysPerThread + i);
                if (AdmitDfsPromotionForTesting(service.get(), key, 100.0)) {
                    admitted.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
        thread.join();
    }

    // Every admission has to show up exactly once in all four bookkeeping
    // structures: the counter, the task table, the heap and the protection set.
    EXPECT_EQ(admitted.load(std::memory_order_relaxed), kKeys);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()),
              static_cast<uint32_t>(kKeys));
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()),
              static_cast<size_t>(kKeys));
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()),
              static_cast<size_t>(kKeys));
    for (int i = 0; i < kKeys; ++i) {
        ASSERT_TRUE(IsDfsSourceProtectedForTesting(
            service.get(), "k_dfs_race_" + std::to_string(i)))
            << "key " << i;
    }
}

TEST_F(PromotionOnHitTest, DfsPromotionConcurrentHeartbeatsNeverDoubleClaim) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    constexpr int kKeys = 256;
    for (int i = 0; i < kKeys; ++i) {
        const std::string key = "k_dfs_claim_" + std::to_string(i);
        ASSERT_TRUE(InjectDfsOnlyObject(*service, key, 1024));
        ASSERT_TRUE(AdmitDfsPromotionForTesting(service.get(), key, 100.0));
    }

    constexpr int kClients = 4;
    std::mutex collect_mutex;
    std::vector<std::string> claimed_keys;
    std::atomic<int> empty_batches{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(kClients);
    for (int c = 0; c < kClients; ++c) {
        threads.emplace_back([&]() {
            const UUID executor = generate_uuid();
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int attempt = 0; attempt < kKeys; ++attempt) {
                auto batch = service->DfsPromotionObjectHeartbeat(executor);
                if (!batch.has_value()) {
                    ADD_FAILURE() << "heartbeat failed";
                    return;
                }
                if (batch->empty()) {
                    empty_batches.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                std::lock_guard<std::mutex> lock(collect_mutex);
                for (const auto& task : *batch) {
                    claimed_keys.push_back(task.key);
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
        thread.join();
    }

    // The heap pop and the holder stamp are not atomic with respect to each
    // other, so the claim loop is the only thing preventing two clients from
    // staging the same promotion: assert it holds under contention.
    EXPECT_EQ(claimed_keys.size(), static_cast<size_t>(kKeys));
    const std::set<std::string> unique_keys(claimed_keys.begin(),
                                            claimed_keys.end());
    EXPECT_EQ(unique_keys.size(), claimed_keys.size());
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()),
              static_cast<uint32_t>(kKeys));
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()),
              static_cast<size_t>(kKeys));
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 0u);
}

TEST_F(PromotionOnHitTest, DfsPromotionConcurrentSameKeyAdmitsExactlyOnce) {
    auto service = std::make_unique<MasterService>(MakeDfsPromotionConfig());

    constexpr int kThreads = 8;
    ASSERT_TRUE(InjectDfsOnlyObject(*service, "k_dfs_same_key", 1024));

    std::atomic<int> admitted{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (AdmitDfsPromotionForTesting(service.get(), "k_dfs_same_key",
                                            100.0)) {
                admitted.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
        thread.join();
    }

    // The per-key in-flight dedup must be race-free: one admission, one task,
    // one heap node, one protection entry.
    EXPECT_EQ(admitted.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(GetDfsPromotionInFlightForTesting(service.get()), 1u);
    EXPECT_EQ(GetDfsPromotionTaskTableSizeForTesting(service.get()), 1u);
    EXPECT_EQ(GetDfsPromotionQueueSizeForTesting(service.get()), 1u);
    EXPECT_TRUE(IsDfsSourceProtectedForTesting(service.get(), "k_dfs_same_key"));
}

// Source-protection end-to-end: a DFS replica that is the source of a
// protected (in-flight)
// promotion must be vetoed by the DFS eviction acceptance even when the
// allocator has picked it as the coldest candidate, and the veto must not stop
// the scan from reclaiming capacity through the other candidates. This is the
// only DFS path that needs a real allocator, hence the environment setup
// mirroring the master_service_test DFS-eviction fixture.
TEST_F(PromotionOnHitTest, DfsPromotionSourceProtectedFromDfsEviction) {
    const auto dfs_root = (std::filesystem::temp_directory_path() /
                           ("promo_dfs_protect_" + std::to_string(::getpid())))
                              .string();
    std::filesystem::create_directories(dfs_root);
    ScopedEnv enable_dfs("MOONCAKE_ENABLE_DFS", "1");
    ScopedEnv fs_adapter("MOONCAKE_DFS_FS_ADAPTER", "posix");
    ScopedEnv root_dir("MOONCAKE_DFS_ROOT_DIR", dfs_root.c_str());
    ScopedEnv shard_count("MOONCAKE_DFS_SHARD_COUNT", "1");
    ScopedEnv shard_capacity("MOONCAKE_DFS_SHARD_CAPACITY", "32768");
    ScopedEnv alignment("MOONCAKE_DFS_ALIGNMENT", "4096");
    // Keep the background path disabled and drive one exact pass by hand.
    ScopedEnv eviction("MOONCAKE_DFS_EVICTION_ENABLED", "0");
    ScopedEnv high_watermark("MOONCAKE_DFS_EVICTION_HIGH_WATERMARK", "0.9");
    ScopedEnv low_watermark("MOONCAKE_DFS_EVICTION_LOW_WATERMARK", "0.7");
    ScopedEnv deferred_free("MOONCAKE_DFS_DEFERRED_FREE_SECONDS", "0");
    ScopedEnv single_tenant("MOONCAKE_DFS_SINGLE_TENANT", "true");

    {
        MasterService service(MakeDfsPromotionConfig());
        constexpr size_t seg_size = 1024 * 1024 * 16;
        auto ctx = PrepareSegment(service, "dfs_protect_seg",
                                  kDefaultSegmentBase, seg_size);

        ReplicateConfig config;
        config.replica_num = 1;
        config.dfs_replica_num = 1;

        // Four 100-byte objects fill the 32768-byte / 4096-aligned shard, so
        // one eviction pass is forced (same shape as the master_service_test
        // DFS eviction cases).
        std::vector<std::string> keys;
        for (int i = 0; i < 4; ++i) {
            keys.push_back("dfs_protect_" + std::to_string(i));
            auto start = service.PutStart(ctx.client_id, keys.back(),
                                          TenantId::Default(), 100, config);
            ASSERT_TRUE(start.has_value()) << "allocation " << i;
            ASSERT_TRUE(service
                            .PutEnd(ctx.client_id, keys.back(),
                                    TenantId::Default(), ReplicaType::ALL)
                            .has_value());
        }

        // The first allocation is the coldest candidate; protect it and evict.
        ProtectDfsSourceForTesting(&service, keys.front());
        service.RunDfsEvictionForTesting();

        auto protected_result =
            service.GetReplicaList(keys.front(), TenantId::Default());
        ASSERT_TRUE(protected_result.has_value());
        EXPECT_EQ(protected_result->replicas.size(), 2u)
            << "protected DFS source must survive the eviction pass";

        // The veto must not stall the scan: at least one unprotected key still
        // lost its DFS replica (candidates behind the protected one are
        // revisited because the rejection moves it to the MRU side).
        size_t reclaimed_elsewhere = 0;
        for (size_t i = 1; i < keys.size(); ++i) {
            auto result = service.GetReplicaList(keys[i], TenantId::Default());
            if (result.has_value() && result->replicas.size() < 2u) {
                ++reclaimed_elsewhere;
            }
        }
        EXPECT_GT(reclaimed_elsewhere, 0u)
            << "eviction must still reclaim capacity elsewhere";
    }

    std::error_code ec;
    std::filesystem::remove_all(dfs_root, ec);
}

}  // namespace mooncake::test

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
