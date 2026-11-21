#include "device/accelerator_registry.h"
#include "pinned_host_buffer.h"

#include "cuda_alike.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(USE_HIP)

#include <dlfcn.h>
#include <filesystem>
#include <glog/logging.h>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include "shared_segment/host_memfd.h"

namespace mooncake {
namespace device {
namespace {

std::shared_mutex &HipRuntimeStateMutex() {
    // Pinned-buffer deleters may run during static teardown after this device
    // singleton starts destructing, so the lock must remain valid until exit.
    static auto *mutex = new std::shared_mutex();
    return *mutex;
}

hipError_t AllocateHipPinnedHost(void **addr, size_t size,
                                 unsigned int flags) {
    std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
    return hipHostMalloc(addr, size, flags);
}

hipError_t GetHipHostDevicePointer(void **device_addr, void *host_addr) {
    std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
    return hipHostGetDevicePointer(device_addr, host_addr, 0);
}

void FreeHipPinnedHost(void *addr) {
    std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
    (void)hipHostFree(addr);
}

void FreeHipPinnedHostBuffer(void *addr) { FreeHipPinnedHost(addr); }

// Mapped length of each buffer AllocateMappedPinnedHost registered; the
// deleter only receives the address. Guarded by HipRuntimeStateMutex() and,
// like it, never destroyed so that teardown-time deleters stay valid.
std::unordered_map<void *, size_t> &RegisteredHostSpans() {
    static auto *spans = new std::unordered_map<void *, size_t>();
    return *spans;
}

void FreeHipRegisteredHostBuffer(void *addr) {
    size_t span = 0;
    {
        std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
        auto it = RegisteredHostSpans().find(addr);
        if (it == RegisteredHostSpans().end()) return;
        const hipError_t rc = hipHostUnregister(addr);
        if (rc != hipSuccess) {
            // Unmapping pages the device may still map is worse than a leak.
            LOG(ERROR) << "hipHostUnregister failed, leaking " << it->second
                       << " bytes at " << addr << ": "
                       << hipGetErrorString(rc);
            (void)hipGetLastError();
            return;
        }
        span = it->second;
        RegisteredHostSpans().erase(it);
    }
    // munmap drops the lock too; an explicit munlock would only add a failure
    // mode between the unregister and the unmap.
    if (munmap(addr, span) != 0) {
        PLOG(WARNING) << "munmap of " << span << " unregistered bytes at "
                      << addr << " failed";
    }
}

// NUMA node of the current device, or -1 when sysfs does not report one.
int CurrentDeviceNumaNode() {
    int device = -1;
    char bus_id[32] = {};
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) != hipSuccess) {
        (void)hipGetLastError();
        return -1;
    }
    for (char *c = bus_id; *c; ++c) {
        *c = static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
    }
    int node = -1;
    if (!(std::ifstream(std::string("/sys/bus/pci/devices/") + bus_id +
                        "/numa_node") >>
          node)) {
        return -1;
    }
    return node;
}

// AnonHugePages and Locked bytes of the mapping that starts exactly at `addr`
// and is roughly `size` long, read from /proc/self/smaps. Diagnostics only:
// the numbers describe the current state, not a guarantee that the span stays
// huge-page backed. A missing entry reports 0/0.
struct MappingStats {
    uint64_t anon_huge_bytes = 0;
    uint64_t locked_bytes = 0;
    uint64_t rss_bytes = 0;
    bool found = false;
};

MappingStats ReadMappingStats(void *addr, size_t size) {
    MappingStats stats;
    FILE *fp = std::fopen("/proc/self/smaps", "r");
    if (fp == nullptr) return stats;
    const auto want_start = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t want_end = want_start + size;
    char line[256];
    bool in_range = false;
    while (std::fgets(line, sizeof(line), fp) != nullptr) {
        unsigned long long start = 0, end = 0;
        if (std::sscanf(line, "%llx-%llx", &start, &end) == 2) {
            in_range = static_cast<uintptr_t>(start) == want_start &&
                       static_cast<uintptr_t>(end) == want_end;
            if (in_range) stats.found = true;
            continue;
        }
        if (!in_range) continue;
        unsigned long long kb = 0;
        if (std::sscanf(line, "AnonHugePages: %llu kB", &kb) == 1) {
            stats.anon_huge_bytes = kb * 1024ULL;
        } else if (std::sscanf(line, "Locked: %llu kB", &kb) == 1) {
            stats.locked_bytes = kb * 1024ULL;
        } else if (std::sscanf(line, "Rss: %llu kB", &kb) == 1) {
            stats.rss_bytes = kb * 1024ULL;
        }
    }
    std::fclose(fp);
    return stats;
}

double MappedPinnedThpThreshold() {
    static const double kThreshold = []() -> double {
        const char *value = std::getenv("MC_STORE_DFS_THP_MIN_PERCENT");
        if (value != nullptr && *value != '\0') {
            char *end = nullptr;
            const double parsed = std::strtod(value, &end);
            if (end != value && parsed >= 0.0 && parsed <= 100.0) {
                return parsed;
            }
        }
        // Requiring THP without naming a threshold means "mostly huge pages";
        // a value of 0 would silently accept a span that is entirely 4 KiB.
        const char *require = std::getenv("MC_STORE_DFS_REQUIRE_THP");
        const bool require_thp =
            require != nullptr && *require != '\0' && std::strcmp(require, "0") != 0;
        return require_thp ? 90.0 : 0.0;
    }();
    return kThreshold;
}

bool MappedPinnedRequireThp() {
    static const bool kRequire = []() -> bool {
        const char *value = std::getenv("MC_STORE_DFS_REQUIRE_THP");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return kRequire;
}

class HipAcceleratorDevice final : public ProbeCachedAcceleratorDevice {
   public:
    AcceleratorVendor Vendor() const override {
        return AcceleratorVendor::kHip;
    }

    bool ProbeAvailable() const override {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    PointerInfo QueryPointer(const void* ptr) const override {
        hipPointerAttribute_t attr{};
        if (hipPointerGetAttributes(&attr, ptr) == hipSuccess &&
            attr.type == hipMemoryTypeDevice) {
            return PointerInfo{.kind = MemoryKind::kDevice,
                               .device_id = attr.device};
        }
        hipGetLastError();
        return PointerInfo{.kind = MemoryKind::kHost, .device_id = -1};
    }

    int32_t CurrentDeviceId() const override {
        int device_id = -1;
        return hipGetDevice(&device_id) == hipSuccess ? device_id : -1;
    }

    void SetContext(int32_t device_id) const override {
        if (device_id < 0) return;
        std::unique_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
        hipSetDevice(device_id);
    }

    bool Copy(void* dst, const void* src, size_t size,
              CopyDirection direction) const override {
        hipMemcpyKind kind = hipMemcpyDefault;
        switch (direction) {
            case CopyDirection::kHostToDevice:
                kind = hipMemcpyHostToDevice;
                break;
            case CopyDirection::kDeviceToHost:
                kind = hipMemcpyDeviceToHost;
                break;
            case CopyDirection::kDeviceToDevice:
                kind = hipMemcpyDeviceToDevice;
                break;
            case CopyDirection::kHostToHost:
            case CopyDirection::kAuto:
                kind = hipMemcpyDefault;
                break;
        }
        return hipMemcpy(dst, src, size, kind) == hipSuccess;
    }

    bool CopyFromHostAsync(void* dst, const void* src, size_t size,
                           void* stream) const override {
        return hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice,
                              static_cast<hipStream_t>(stream)) == hipSuccess;
    }

    bool CopyFromHostBatchAsync(std::span<const HostCopyRange> ranges,
                                void* stream,
                                bool* used_batch_kernel = nullptr) const
        override {
        if (used_batch_kernel) *used_batch_kernel = false;
        constexpr size_t kKernelMinRanges = 32;
        if (ranges.size() < kKernelMinRanges) {
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }

        for (const auto &range : ranges) {
            // The scatter kernel reads the source through its device-visible
            // mapped-host alias. If any source is ordinary pinned memory, keep
            // the whole batch on the DMA path so submission order is preserved.
            if (!range.dst || !range.src || range.size == 0 ||
                !range.src_device) {
                return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
            }
        }

        hipFunction_t function = nullptr;
        if (!LoadCopyFunction(CurrentDeviceId(), &function)) {
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }

        struct DeviceRange {
            uint64_t dst_addr;
            uint64_t src_addr;
            uint64_t size;
        };
        if (ranges.size() > std::numeric_limits<size_t>::max() /
                                sizeof(DeviceRange) ||
            ranges.size() > std::numeric_limits<unsigned int>::max()) {
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }

        // Descriptors come from a per-stream arena that is allocated once and
        // reused until the stream is synchronized, so a batch no longer calls
        // hipHostMalloc. Under HSA_USERPTR_FOR_PAGED_MEM=0 that call is a GTT
        // allocation, which can run a direct compaction in the calling thread.
        const auto hip_stream = static_cast<hipStream_t>(stream);
        const size_t descriptor_bytes = ranges.size() * sizeof(DeviceRange);
        void *host_descriptors = nullptr;
        void *device_descriptors = nullptr;
        size_t arena_offset = 0;
        bool from_arena = false;
        if (ReserveDescriptorArena(hip_stream, descriptor_bytes,
                                   &host_descriptors, &device_descriptors,
                                   &arena_offset)) {
            from_arena = true;
        } else {
            LOG_EVERY_N(WARNING, 64)
                << "DFS H2D descriptor arena could not serve "
                << descriptor_bytes
                << " bytes, falling back to a per-batch pinned allocation "
                   "(count="
                << descriptor_arena_fallbacks_.fetch_add(1) + 1 << ")";
            if (AllocateHipPinnedHost(&host_descriptors, descriptor_bytes,
                                      hipHostMallocMapped) != hipSuccess) {
                hipGetLastError();
                return AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                 stream);
            }
            if (GetHipHostDevicePointer(&device_descriptors,
                                        host_descriptors) != hipSuccess) {
                hipGetLastError();
                FreeHipPinnedHost(host_descriptors);
                return AcceleratorDevice::CopyFromHostBatchAsync(ranges,
                                                                 stream);
            }
        }
        auto *descriptors = static_cast<DeviceRange *>(host_descriptors);
        for (size_t i = 0; i < ranges.size(); ++i) {
            descriptors[i] = DeviceRange{
                reinterpret_cast<uint64_t>(ranges[i].dst),
                reinterpret_cast<uint64_t>(ranges[i].src_device),
                ranges[i].size};
        }

        uint64_t range_count = ranges.size();
        void *args[] = {&device_descriptors, &range_count};
        constexpr unsigned int kThreads = 256;
        const hipError_t launch_result = hipModuleLaunchKernel(
            function, static_cast<unsigned int>(ranges.size()), 1, 1, kThreads,
            1, 1, 0, hip_stream, args, nullptr);
        if (launch_result != hipSuccess) {
            hipGetLastError();
            if (from_arena) {
                ReleaseDescriptorArena(hip_stream, arena_offset,
                                       descriptor_bytes);
            } else {
                FreeHipPinnedHost(host_descriptors);
            }
            return AcceleratorDevice::CopyFromHostBatchAsync(ranges, stream);
        }

        if (!from_arena) {
            // The stream has not consumed a one-off allocation yet; keep it
            // until SynchronizeStream reports the kernel has finished.
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            pending_descriptors_[hip_stream].push_back(host_descriptors);
        }
        if (used_batch_kernel) *used_batch_kernel = true;
        return true;
    }

    bool CreateStream(void** stream) const override {
        hipStream_t hip_stream = nullptr;
        if (hipStreamCreate(&hip_stream) != hipSuccess) {
            hipGetLastError();
            return false;
        }
        *stream = static_cast<void*>(hip_stream);
        return true;
    }

    bool SynchronizeStream(void* stream) const override {
        const auto hip_stream = static_cast<hipStream_t>(stream);
        bool success = false;
        {
            std::shared_lock<std::shared_mutex> lock(HipRuntimeStateMutex());
            success = hipStreamSynchronize(hip_stream) == hipSuccess;
        }
        // Synchronization establishes that the kernel no longer dereferences
        // the mapped descriptor array. Free one-off allocations even when the
        // stream reports a device-side error; the stream has completed in
        // either case. The arena is not freed, only rewound: every descriptor
        // it held has now been consumed.
        std::vector<void *> descriptors;
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto it = pending_descriptors_.find(hip_stream);
            if (it != pending_descriptors_.end()) {
                descriptors = std::move(it->second);
                pending_descriptors_.erase(it);
            }
            auto arena = descriptor_arenas_.find(hip_stream);
            if (arena != descriptor_arenas_.end()) arena->second.used = 0;
        }
        for (void *descriptor : descriptors) {
            FreeHipPinnedHost(descriptor);
        }
        return success;
    }

    void DestroyStream(void* stream) const override {
        const auto hip_stream = static_cast<hipStream_t>(stream);
        // The caller has synchronized the stream (or is tearing the device
        // down); a kernel still reading the arena would be a use-after-free.
        hipStreamDestroy(hip_stream);
        void *arena_host = nullptr;
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto it = descriptor_arenas_.find(hip_stream);
            if (it != descriptor_arenas_.end()) {
                arena_host = it->second.host;
                descriptor_arenas_.erase(it);
            }
        }
        if (arena_host != nullptr) FreeHipPinnedHost(arena_host);
    }

    PinnedHostBuffer AllocatePinnedHost(size_t size) const override {
        void* addr = nullptr;
        if (AllocateHipPinnedHost(&addr, size, 0) != hipSuccess) {
            hipGetLastError();
            return PinnedHostBuffer();
        }
        return PinnedHostBuffer(addr, size, FreeHipPinnedHostBuffer);
    }

    // Ordinary anonymous memory registered with hipHostRegister instead of
    // hipHostMalloc. Upstream KFD does not charge a registration (a userptr
    // BO, or an SVM range on kernels with the SVM API) to the TTM limit.
    // hipHostMalloc becomes a GTT BO, charged to the TTM limit (half of RAM
    // by default), once the process sets HSA_USERPTR_FOR_PAGED_MEM=0, as the
    // DeepSeek-V4.1 engram shared host table requires. Before the first touch
    // the mapping gets, best effort as libhsakmt gives a paged hipHostMalloc,
    // a preferred NUMA policy for the device's node, huge-page advice and
    // MADV_DONTFORK. Its base is aligned to the THP size unless the aligned
    // reservation fails.
    PinnedHostBuffer AllocateMappedPinnedHost(size_t size) const override {
        const size_t granularity = host_memfd::Granularity();
        // ReserveAlignedWindow adds one more granularity to the rounded span.
        if (size == 0 || size > SIZE_MAX - 2 * granularity) {
            return PinnedHostBuffer();
        }
        const size_t span =
            (size + granularity - 1) / granularity * granularity;
        // Both may allocate (the mutex on first use), so they run before
        // there is a mapping to lose.
        std::shared_mutex &runtime_mutex = HipRuntimeStateMutex();
        const int node = CurrentDeviceNumaNode();
        void *addr = host_memfd::ReserveAlignedWindow(span, granularity);
        if (addr == MAP_FAILED) {
            PLOG(ERROR) << "Mapped pinned host: reserving " << span
                        << " bytes failed";
            return PinnedHostBuffer();
        }
        // Make the reservation itself writable: on failure it is still ours
        // to unmap, unlike a MAP_FIXED replacement that may already have
        // dropped it.
        if (mprotect(addr, span, PROT_READ | PROT_WRITE) != 0) {
            PLOG(ERROR) << "Mapped pinned host: mprotect of " << span
                        << " bytes failed";
            munmap(addr, span);
            return PinnedHostBuffer();
        }
        int preferred_node = -1;
        if (node >= 0 && node < 64) {
            const unsigned long mask = 1UL << node;
            if (syscall(SYS_mbind, addr, span, MPOL_PREFERRED, &mask,
                        sizeof(mask) * 8 + 1, 0) == 0) {
                preferred_node = node;
            } else {
                PLOG(WARNING) << "Mapped pinned host: mbind to NUMA node "
                              << node << " failed";
            }
        }
        host_memfd::AdviseTransparentHugePages(addr, span);
        host_memfd::AdviseDontFork(addr, span);
        host_memfd::PrefaultRange(addr, span, host_memfd::kPrefaultStride);

        // Compaction generally skips compound pages, so a span that is mostly
        // huge-page backed is far less likely to be migrated and to evict the
        // GPU queues that map it. The coverage is what this allocation got,
        // not what it will keep: log it so a low value is visible.
        const MappingStats stats = ReadMappingStats(addr, span);
        const double thp_percent =
            span == 0 ? 0.0
                      : 100.0 * static_cast<double>(stats.anon_huge_bytes) /
                            static_cast<double>(span);
        LOG(INFO) << "Mapped pinned host: thp_coverage=" << thp_percent
                  << "% (" << stats.anon_huge_bytes / 1024 / 1024 << "/"
                  << span / 1024 / 1024 << " MiB) node=" << preferred_node;
        const double min_percent = MappedPinnedThpThreshold();
        if (thp_percent < min_percent) {
            const std::string message =
                "Mapped pinned host: thp_coverage " +
                std::to_string(thp_percent) + "% is below the required " +
                std::to_string(min_percent) + "%";
            if (MappedPinnedRequireThp()) {
                LOG(ERROR) << message << "; refusing the allocation";
                munmap(addr, span);
                return PinnedHostBuffer();
            }
            LOG(WARNING) << message;
        }

        // A migrated page is only one problem; a reclaimed or split one is
        // another. Locking keeps the span resident and out of the swap path.
        // It does not by itself stop compaction, which is why the coverage
        // above is the first line of defence.
        if (mlock(addr, span) != 0) {
            PLOG(WARNING) << "Mapped pinned host: mlock of " << span
                          << " bytes failed; the buffer stays migratable";
        }

        // The span is recorded before registering, so a failed insert leaves
        // nothing registered and a registered span is never unrecorded.
        bool recorded = false;
        hipError_t rc = hipSuccess;
        {
            std::unique_lock<std::shared_mutex> lock(runtime_mutex);
            try {
                recorded = RegisteredHostSpans().emplace(addr, span).second;
            } catch (const std::bad_alloc &) {
            }
            if (recorded) {
                rc = hipHostRegister(addr, span, hipHostRegisterMapped);
                if (rc != hipSuccess) RegisteredHostSpans().erase(addr);
            }
        }
        if (!recorded || rc != hipSuccess) {
            LOG(ERROR) << "Mapped pinned host: registering " << span
                       << " bytes failed: "
                       << (recorded ? hipGetErrorString(rc)
                                    : "cannot record the span");
            (void)hipGetLastError();
            munmap(addr, span);
            return PinnedHostBuffer();
        }

        void *device_addr = nullptr;
        rc = GetHipHostDevicePointer(&device_addr, addr);
        if (rc != hipSuccess) {
            LOG(ERROR) << "Mapped pinned host: hipHostGetDevicePointer "
                          "failed: "
                       << hipGetErrorString(rc);
            (void)hipGetLastError();
            FreeHipRegisteredHostBuffer(addr);
            return PinnedHostBuffer();
        }
        LOG(INFO) << "Mapped pinned host: registered " << span << " bytes at "
                  << addr << ", device alias " << device_addr
                  << ", preferred NUMA node " << preferred_node;
        return PinnedHostBuffer(addr, size, FreeHipRegisteredHostBuffer,
                                device_addr);
    }

    bool WarmMappedHostCopyPath(const std::vector<void *> &streams,
                                void *host_buffer, void *host_device_alias,
                                size_t buffer_size) const override {
        // Module load and descriptor arenas are the parts that touch the
        // driver; both are idempotent, so a second warm-up is cheap.
        hipFunction_t function = nullptr;
        const bool kernel_ready = LoadCopyFunction(CurrentDeviceId(), &function);
        if (!kernel_ready) {
            LOG(WARNING) << "DFS H2D warm-up: scatter kernel is unavailable; "
                            "batches will use DMA";
        }
        bool arenas_ready = true;
        for (void *stream : streams) {
            if (stream == nullptr) continue;
            if (!CreateDescriptorArena(static_cast<hipStream_t>(stream))) {
                arenas_ready = false;
                LOG(WARNING) << "DFS H2D warm-up: descriptor arena for stream "
                             << stream << " failed";
            }
        }
        if (!kernel_ready || !arenas_ready || streams.empty() ||
            host_buffer == nullptr || host_device_alias == nullptr ||
            buffer_size == 0) {
            return false;
        }

        constexpr size_t kWarmupRanges = 32;
        constexpr size_t kRangeBytes = 1;
        void *warmup_device = nullptr;
        if (hipMalloc(reinterpret_cast<void **>(&warmup_device),
                      kWarmupRanges * kRangeBytes) != hipSuccess) {
            hipGetLastError();
            LOG(WARNING) << "DFS H2D warm-up: temporary device allocation "
                            "failed";
            return false;
        }
        std::vector<HostCopyRange> ranges;
        ranges.reserve(kWarmupRanges);
        const size_t stride = std::max<size_t>(buffer_size / kWarmupRanges, 1);
        for (size_t i = 0; i < kWarmupRanges; ++i) {
            const size_t src_offset =
                std::min(i * stride, buffer_size - kRangeBytes);
            ranges.push_back({static_cast<char *>(warmup_device) + i,
                              static_cast<char *>(host_buffer) + src_offset,
                              kRangeBytes,
                              static_cast<char *>(host_device_alias) +
                                  src_offset});
        }
        bool used_kernel = false;
        void *stream = streams.front();
        const bool warmed =
            CopyFromHostBatchAsync(ranges, stream, &used_kernel) &&
            used_kernel && SynchronizeStream(stream);
        hipFree(warmup_device);
        if (!warmed) {
            LOG(WARNING) << "DFS H2D warm-up: scatter launch or synchronization "
                            "failed";
            return false;
        }
        return true;
    }

    ~HipAcceleratorDevice() override {
        std::vector<void *> descriptors;
        {
            std::lock_guard<std::mutex> descriptor_lock(descriptor_mutex_);
            for (auto &[_, pending] : pending_descriptors_) {
                descriptors.insert(descriptors.end(), pending.begin(),
                                   pending.end());
            }
            pending_descriptors_.clear();
        }
        for (void *descriptor : descriptors) {
            FreeHipPinnedHost(descriptor);
        }

        std::vector<void *> arenas;
        {
            std::lock_guard<std::mutex> descriptor_lock(descriptor_mutex_);
            for (auto &[_, arena] : descriptor_arenas_) {
                arenas.push_back(arena.host);
            }
            descriptor_arenas_.clear();
        }
        for (void *arena : arenas) {
            FreeHipPinnedHost(arena);
        }

        std::lock_guard<std::mutex> lock(module_mutex_);
        for (const auto &[_, module] : copy_modules_) {
            hipModuleUnload(module);
        }
    }

   private:
    // Descriptor staging for one stream: a single mapped pinned allocation,
    // handed out by bump allocation between two synchronizations. Mapped host
    // memory stays valid for the device until the deleter runs, so the arena
    // must outlive the kernel; the offset only resets once the stream has
    // finished. A batch larger than the whole arena still makes progress
    // through the one-off fallback in the caller.
    struct DescriptorArena {
        void *host = nullptr;
        // Device alias of host. A mapped host allocation keeps a constant
        // host/device address offset across its whole range, so a slot's alias
        // is base+offset. Deriving it this way avoids calling
        // hipHostGetDevicePointer with an interior (non-base) address, which is
        // not guaranteed to be accepted.
        void *device = nullptr;
        size_t bytes = 0;
        size_t used = 0;
    };

    size_t DescriptorArenaBytes() const {
        static const size_t kBytes = []() -> size_t {
            constexpr size_t kDefault = 4ULL * 1024 * 1024;
            const char *value = std::getenv("MC_STORE_DFS_H2D_DESC_ARENA_BYTES");
            if (value == nullptr || *value == '\0') return kDefault;
            char *end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (end == value || *end != '\0' || parsed == 0) return kDefault;
            return static_cast<size_t>(parsed);
        }();
        return kBytes;
    }

    // Bump-allocates descriptor_bytes from the stream's arena, creating it on
    // first use. Returns false when the remaining space does not fit, so the
    // caller can fall back to a one-off allocation. The HIP calls run outside
    // descriptor_mutex_, so this path never nests it around
    // HipRuntimeStateMutex().
    bool ReserveDescriptorArena(hipStream_t stream, size_t descriptor_bytes,
                                void **host, void **device,
                                size_t *arena_offset) const {
        if (descriptor_bytes == 0) return false;
        if (!TryBumpDescriptorArena(stream, descriptor_bytes, host, device,
                                    arena_offset)) {
            if (!CreateDescriptorArena(stream)) return false;
            if (!TryBumpDescriptorArena(stream, descriptor_bytes, host, device,
                                        arena_offset)) {
                return false;
            }
        }
        return true;
    }

    bool TryBumpDescriptorArena(hipStream_t stream, size_t descriptor_bytes,
                                void **host, void **device,
                                size_t *arena_offset) const {
        std::lock_guard<std::mutex> lock(descriptor_mutex_);
        auto it = descriptor_arenas_.find(stream);
        if (it == descriptor_arenas_.end()) return false;
        DescriptorArena &arena = it->second;
        if (descriptor_bytes > arena.bytes - arena.used) return false;
        const size_t offset = arena.used;
        arena.used += descriptor_bytes;
        *host = static_cast<char *>(arena.host) + offset;
        *device = static_cast<char *>(arena.device) + offset;
        if (arena_offset) *arena_offset = offset;
        return true;
    }

    bool CreateDescriptorArena(hipStream_t stream) const {
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            if (descriptor_arenas_.count(stream) != 0) return true;
        }
        const size_t bytes = DescriptorArenaBytes();
        void *host_addr = nullptr;
        if (AllocateHipPinnedHost(&host_addr, bytes, hipHostMallocMapped) !=
            hipSuccess) {
            hipGetLastError();
            return false;
        }
        void *device_addr = nullptr;
        if (GetHipHostDevicePointer(&device_addr, host_addr) != hipSuccess) {
            hipGetLastError();
            FreeHipPinnedHost(host_addr);
            return false;
        }
        bool keep = false;
        {
            std::lock_guard<std::mutex> lock(descriptor_mutex_);
            auto [it, inserted] =
                descriptor_arenas_.try_emplace(stream, DescriptorArena{});
            if (inserted) {
                it->second.host = host_addr;
                it->second.device = device_addr;
                it->second.bytes = bytes;
                it->second.used = 0;
                keep = true;
            }
        }
        if (!keep) {
            // Another submission created it first; keep the existing arena.
            FreeHipPinnedHost(host_addr);
            return true;
        }
        LOG(INFO) << "DFS H2D descriptor arena allocated: stream=" << stream
                  << " bytes=" << bytes;
        return true;
    }

    // Restores the reservation made for a launch that never reached the
    // stream, so the space is not lost until the next synchronization. Only
    // the tail reservation can be taken back; anything earlier is still
    // waiting in the queue and stays accounted.
    void ReleaseDescriptorArena(hipStream_t stream, size_t arena_offset,
                                size_t descriptor_bytes) const {
        std::lock_guard<std::mutex> lock(descriptor_mutex_);
        auto it = descriptor_arenas_.find(stream);
        if (it == descriptor_arenas_.end()) return;
        if (it->second.used == arena_offset + descriptor_bytes) {
            it->second.used = arena_offset;
        }
    }

    static std::string CopyKernelPath() {
        const char *directory = std::getenv("MC_COPY_KERNEL_PATH");
        if (directory && directory[0] != '\0') {
            return std::string(directory) + "/mc_copy_kernel.co";
        }

        // Wheels and native installs keep the kernel beside the Store library.
        // auditwheel may move that library into a sibling .libs directory.
        static int library_anchor;
        Dl_info library_info{};
        if (dladdr(&library_anchor, &library_info) && library_info.dli_fname) {
            const auto library_dir =
                std::filesystem::path(library_info.dli_fname).parent_path();
            for (const auto &path : {library_dir / "mc_copy_kernel.co",
                                     library_dir.parent_path() / "mooncake" /
                                         "mc_copy_kernel.co"}) {
                if (std::ifstream(path).good()) return path.string();
            }
        }
        return "/usr/local/lib/python3.10/dist-packages/mooncake/mc_copy_kernel.co";
    }

    bool LoadCopyFunction(int device_id, hipFunction_t *function) const {
        if (device_id < 0) return false;
        std::lock_guard<std::mutex> lock(module_mutex_);
        auto it = copy_functions_.find(device_id);
        if (it != copy_functions_.end()) {
            *function = it->second;
            return true;
        }

        const std::string path = CopyKernelPath();
        std::ifstream kernel_file(path);
        if (!kernel_file.good()) return false;

        hipModule_t module = nullptr;
        if (hipModuleLoad(&module, path.c_str()) != hipSuccess) {
            hipGetLastError();
            return false;
        }
        hipFunction_t loaded_function = nullptr;
        if (hipModuleGetFunction(&loaded_function, module,
                                 "MCStoreH2DScatterKernel") != hipSuccess) {
            hipGetLastError();
            hipModuleUnload(module);
            return false;
        }
        copy_modules_.emplace(device_id, module);
        copy_functions_.emplace(device_id, loaded_function);
        *function = loaded_function;
        return true;
    }

    mutable std::mutex module_mutex_;
    mutable std::mutex descriptor_mutex_;
    mutable std::unordered_map<int, hipModule_t> copy_modules_;
    mutable std::unordered_map<int, hipFunction_t> copy_functions_;
    mutable std::unordered_map<hipStream_t, std::vector<void *>>
        pending_descriptors_;
    mutable std::unordered_map<hipStream_t, DescriptorArena> descriptor_arenas_;
    mutable std::atomic<uint64_t> descriptor_arena_fallbacks_{0};
};

const AcceleratorDevice& HipDeviceInstance() {
    static HipAcceleratorDevice device;
    return device;
}

const AcceleratorDeviceRegistrar registered_hip_device(HipDeviceInstance());

}  // namespace
}  // namespace device
}  // namespace mooncake

#endif
