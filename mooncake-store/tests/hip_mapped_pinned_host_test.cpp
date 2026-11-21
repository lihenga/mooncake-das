// Hygon mapped pinned host memory behind the unified DFS buffer, on a real
// DCU: H2D batches through the device alias (scatter kernel) and through the
// host pointers (DMA) move the exact bytes, D2H copies land at internal
// offsets, the mapping prefers the device's NUMA node, releasing it unmaps
// the whole rounded span, and under HSA_USERPTR_FOR_PAGED_MEM=0 it is charged
// to system memory rather than to the TTM limit.
//
// Written for a deployment-like node: mbind and get_mempolicy allowed and
// room for a THP-aligned reservation. Where a container refuses them the
// allocator degrades by design, but the NUMA or alignment check fails here,
// so that a lost policy or alignment does not pass unnoticed.

#include <hip/hip_runtime.h>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "device/accelerator_registry.h"
#include "file_storage.h"
#include "pinned_buffer_pool.h"
#include "prepared_unified_buffer.h"
#include "real_client.h"
#include "shared_segment/host_memfd.h"

namespace mooncake {
namespace {

bool HaveDcu() {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

const device::AcceleratorDevice* RegisteredHip() {
    if (hipSetDevice(0) != hipSuccess) return nullptr;
    return device::GetAcceleratorRegistry().GetDevice(
        device::AcceleratorVendor::kHip);
}

// Skips without a DCU. Fails when a DCU is present but the HIP accelerator is
// not registered, e.g. because a static link dropped its object file.
#define GET_HIP_OR_SKIP(var)                                     \
    if (!HaveDcu()) GTEST_SKIP() << "no DCU";                    \
    const device::AcceleratorDevice* var = RegisteredHip();      \
    ASSERT_NE(var, nullptr) << "a DCU is present but no HIP "    \
                               "accelerator is registered"

// Bytes that depend on the absolute position and a seed, so a shifted,
// swapped or misdirected range cannot match.
void Fill(char* p, size_t n, uint32_t seed) {
    const uint32_t mix = seed * 0x9E3779B9u;
    for (size_t i = 0; i < n; ++i) {
        p[i] = static_cast<char>(
            ((static_cast<uint32_t>(i) ^ mix) * 2654435761u) >> 24);
    }
}

::testing::AssertionResult SameBytes(const char* got, const char* want,
                                     size_t n) {
    const auto diff = std::mismatch(got, got + n, want);
    if (diff.first == got + n) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "first difference at offset " << (diff.first - got);
}

int DeviceNumaNode(int device) {
    char bus_id[32] = {};
    if (hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) != hipSuccess) {
        return -1;
    }
    for (char* c = bus_id; *c; ++c) {
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

// The scatter kernel the HIP device loads, by the same rule.
std::string KernelPath() {
    const char* dir = std::getenv("MC_COPY_KERNEL_PATH");
    if (dir != nullptr && *dir != '\0') {
        return std::string(dir) + "/mc_copy_kernel.co";
    }
    return "/usr/local/lib/python3.10/dist-packages/mooncake/"
           "mc_copy_kernel.co";
}

// One H2D batch of n ranges out of a mapped buffer. The sources start just
// before a huge-page boundary and cross it, destinations run in reverse
// order from another base, every fifth range is odd sized and not 16-byte
// aligned, and the gaps around the ranges must stay untouched.
void CheckH2dBatch(const device::AcceleratorDevice* hip, size_t n,
                   bool expect_kernel) {
    constexpr size_t kSlot = 32 * 1024, kDevHead = 8192, kTail = 123;
    const size_t kHostHead = host_memfd::Granularity() - 40 * 1024;
    const size_t host_size = kHostHead + n * kSlot + kTail;
    const size_t dev_size = kDevHead + n * kSlot + kTail;
    auto buf = PinnedBufferPool::AllocatePinned(host_size, /*mapped=*/true);
    ASSERT_NE(buf.data, nullptr);
    ASSERT_TRUE(buf.mapped);
    EXPECT_EQ(buf.capacity, host_size);
    Fill(buf.data, host_size, 1);
    const std::vector<char> host_before(buf.data, buf.data + host_size);

    char* dev = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&dev), dev_size),
              hipSuccess);
    ASSERT_EQ(hipMemset(dev, 0, dev_size), hipSuccess);
    std::vector<char> expected(dev_size, 0);
    std::vector<device::HostCopyRange> ranges;
    for (size_t r = 0; r < n; ++r) {
        const bool odd = r % 5 == 4;
        const size_t size = kSlot / 2 + (odd ? 13 : 0);
        const size_t src = kHostHead + r * kSlot + (odd ? 3 : 0);
        const size_t dst = kDevHead + (n - 1 - r) * kSlot + (odd ? 3 : 0);
        ranges.push_back({dev + dst, buf.data + src, size,
                          static_cast<char*>(buf.device_data) + src});
        std::memcpy(expected.data() + dst, host_before.data() + src, size);
    }

    void* stream = nullptr;
    ASSERT_TRUE(hip->CreateStream(&stream));
    bool used_kernel = false;
    ASSERT_TRUE(hip->CopyFromHostBatchAsync(ranges, stream, &used_kernel));
    ASSERT_TRUE(hip->SynchronizeStream(stream));
    hip->DestroyStream(stream);
    std::cout << n << " ranges: used_batch_kernel=" << used_kernel
              << std::endl;
    if (!expect_kernel) {
        EXPECT_FALSE(used_kernel) << "fewer than 32 ranges must use DMA";
    } else if (std::ifstream(KernelPath()).good()) {
        EXPECT_TRUE(used_kernel) << "the scatter kernel did not run";
    } else {
        std::cout << KernelPath() << " is missing: the batch may use DMA"
                  << std::endl;
    }

    std::vector<char> got(dev_size);
    ASSERT_EQ(hipMemcpy(got.data(), dev, dev_size, hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_TRUE(SameBytes(got.data(), expected.data(), dev_size))
        << "device bytes";
    EXPECT_TRUE(SameBytes(buf.data, host_before.data(), host_size))
        << "the host source changed";
    EXPECT_EQ(hipFree(dev), hipSuccess);
}

// Only the result is observable: without the early size guard these sizes
// would still fail further down (in the reservation) and return empty.
TEST(HipMappedPinnedHostTest, RejectsEmptyAndOverflowingSizes) {
    GET_HIP_OR_SKIP(hip);
    (void)hip;
    const size_t granularity = host_memfd::Granularity();
    EXPECT_EQ(PinnedBufferPool::AllocatePinned(0, true).data, nullptr);
    EXPECT_EQ(PinnedBufferPool::AllocatePinned(SIZE_MAX - granularity, true)
                  .data,
              nullptr);
    EXPECT_EQ(PinnedBufferPool::AllocatePinned(SIZE_MAX, true).data, nullptr);
}

TEST(HipMappedPinnedHostTest, FileStorageOwnsLateAndPreparedUnifiedBuffers) {
    GET_HIP_OR_SKIP(hip);
    const int32_t device_id = hip->CurrentDeviceId();
    constexpr size_t kSize = 16 * 1024 * 1024;
    for (bool prepare : {false, true}) {
        SCOPED_TRACE(prepare ? "prepared" : "late allocation");
        ASSERT_FALSE(PreparedUnifiedBuffer::Claim(device_id, kSize));
        if (prepare) {
            ASSERT_EQ(PreparedUnifiedBuffer::Prepare(device_id, kSize), 0);
        }
        FileStorageConfig config;
        config.storage_filepath = "/tmp";
        config.local_buffer_size = 1024 * 1024;
        config.unified_buffer_size = kSize;
        config.read_chunk_size = 1024 * 1024;
        config.write_chunk_size = 1024 * 1024;
        config.enable_dfs = true;
        FileStorage storage(config, nullptr, "127.0.0.1:0");
        ASSERT_TRUE(storage.HasPinnedRestoreArena());
        EXPECT_EQ(storage.UnifiedBufferCapacity(), kSize);
        EXPECT_NE(storage.UnifiedBufferHostAddress(), nullptr);
        EXPECT_NE(storage.UnifiedBufferDeviceAddress(), nullptr);
        EXPECT_FALSE(PreparedUnifiedBuffer::Claim(device_id, kSize));
    }
}

TEST(HipMappedPinnedHostTest, ClientPreparationBeforeSetupHasNoFileStorage) {
    GET_HIP_OR_SKIP(hip);
    RealClient client;
    EXPECT_EQ(client.prepare_dfs_unified_buffer(hip->CurrentDeviceId(),
                                                16 * 1024 * 1024),
              0);
    // A DFS-enabled caller prepares a buffer; take it so this test releases it.
    auto buffer =
        PreparedUnifiedBuffer::Claim(hip->CurrentDeviceId(), 16 * 1024 * 1024);
}

TEST(HipMappedPinnedHostTest, ScatterKernelBatchCarriesExactBytes) {
    GET_HIP_OR_SKIP(hip);
    CheckH2dBatch(hip, 40, /*expect_kernel=*/true);
}

TEST(HipMappedPinnedHostTest, DmaBatchCarriesExactBytes) {
    GET_HIP_OR_SKIP(hip);
    CheckH2dBatch(hip, 8, /*expect_kernel=*/false);
}

// D2H into the buffer at an internal offset across a huge-page boundary, as
// DFS write staging does; the rest of the buffer must stay untouched.
TEST(HipMappedPinnedHostTest, DeviceToHostLandsAtInternalOffset) {
    GET_HIP_OR_SKIP(hip);
    constexpr size_t kSize = 3 * 1024 * 1024 + 123;
    constexpr size_t kSrc = 4096 + 5, kDst = 2 * 1024 * 1024 - 7;
    constexpr size_t kLen = 1024 * 1024 + 11;
    auto buf = PinnedBufferPool::AllocatePinned(kSize, /*mapped=*/true);
    ASSERT_NE(buf.data, nullptr);
    Fill(buf.data, kSize, 1);
    std::vector<char> expected(buf.data, buf.data + kSize);
    std::vector<char> image(kSize);
    Fill(image.data(), kSize, 2);
    std::memcpy(expected.data() + kDst, image.data() + kSrc, kLen);

    char* dev = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&dev), kSize), hipSuccess);
    ASSERT_EQ(hipMemcpy(dev, image.data(), kSize, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_TRUE(hip->Copy(buf.data + kDst, dev + kSrc, kLen,
                          device::CopyDirection::kDeviceToHost));
    EXPECT_TRUE(SameBytes(buf.data, expected.data(), kSize));
    EXPECT_EQ(hipFree(dev), hipSuccess);
}

TEST(HipMappedPinnedHostTest, PrefersDeviceNumaNodeAndUnmapsWholeSpan) {
    GET_HIP_OR_SKIP(hip);
    (void)hip;
    const size_t granularity = host_memfd::Granularity();
    char* addr = nullptr;
    {
        // One byte past a granule: the mapping spans two.
        auto buf = PinnedBufferPool::AllocatePinned(granularity + 1, true);
        ASSERT_NE(buf.data, nullptr);
        addr = buf.data;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(addr) % granularity, 0u)
            << "the aligned reservation fell back to page alignment";
        const int node = DeviceNumaNode(0);
        if (node < 0 || node >= 64) {
            // The allocator sets a policy only for nodes 0..63.
            std::cout << "device 0 NUMA node " << node
                      << ": no policy expected, not checked" << std::endl;
        } else {
            int mode = -1;
            unsigned long mask[16] = {};  // room for 1024 nodes
            const long rc =
                syscall(SYS_get_mempolicy, &mode, mask, sizeof(mask) * 8,
                        addr + granularity, MPOL_F_ADDR);
            EXPECT_EQ(rc, 0) << std::strerror(errno);
            if (rc == 0) {
                unsigned long want[16] = {};
                want[node / 64] = 1UL << (node % 64);
                EXPECT_EQ(mode, MPOL_PREFERRED);
                EXPECT_EQ(0, std::memcmp(mask, want, sizeof(mask)))
                    << "preferred node is not " << node;
            }
        }
    }
    // Released: every base page of the rounded span is unmapped.
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    size_t still_mapped = 0;
    for (size_t off = 0; off < 2 * granularity; off += page) {
        unsigned char resident = 0;
        errno = 0;
        if (mincore(addr + off, 1, &resident) == 0 || errno != ENOMEM) {
            ++still_mapped;
        }
    }
    EXPECT_EQ(still_mapped, 0u) << "pages of the span are still mapped";
}

// KFD counters from debugfs, in MiB; false when unreadable (needs root).
bool ReadKfdMemUsed(long* system_mb, long* ttm_mb) {
    std::ifstream file("/sys/kernel/debug/kfd/mem_limit");
    std::string line;
    bool got_system = false, got_ttm = false;
    while (std::getline(file, line)) {
        long value = 0;
        if (std::sscanf(line.c_str(), "System mem used %ldM", &value) == 1) {
            *system_mb = value;
            got_system = true;
        } else if (std::sscanf(line.c_str(), "TTM mem used %ldM", &value) ==
                   1) {
            *ttm_mb = value;
            got_ttm = true;
        }
    }
    return got_system && got_ttm;
}

// The point of the change. With HSA_USERPTR_FOR_PAGED_MEM=0 hipHostMalloc is
// charged to the TTM limit; the mapped buffer must not be. The counters are
// node-wide, so run this on a node without other DCU work; the slack only
// absorbs small background changes. The system-memory checks describe hycu
// today: a registration is charged to system memory, and the charge comes
// back asynchronously once the pages are unmapped (hipHostUnregister alone
// keeps it), so the release check polls.
TEST(HipMappedPinnedHostTest, StaysOutOfTtmWhenPagedMemoryIsGtt) {
    const char* paged = std::getenv("HSA_USERPTR_FOR_PAGED_MEM");
    if (paged == nullptr || std::string(paged) != "0") {
        GTEST_SKIP() << "run with HSA_USERPTR_FOR_PAGED_MEM=0";
    }
    GET_HIP_OR_SKIP(hip);
    (void)hip;
    long sys0 = 0, ttm0 = 0;
    if (!ReadKfdMemUsed(&sys0, &ttm0)) {
        GTEST_SKIP() << "/sys/kernel/debug/kfd/mem_limit is not readable";
    }
    constexpr size_t kBytes = 1ULL << 30;
    constexpr long kMb = 1024, kSlackMb = 128;

    // Reads the counters every 100 ms, for up to 5 s, until done() holds;
    // false if a read fails, so stale values are never judged.
    long sys = 0, ttm = 0;
    auto poll = [&](auto done) {
        for (int i = 0; i < 50; ++i) {
            if (!ReadKfdMemUsed(&sys, &ttm)) return false;
            if (done()) return true;
            usleep(100 * 1000);
        }
        return true;
    };

    void* gtt = nullptr;
    ASSERT_EQ(hipHostMalloc(&gtt, kBytes, hipHostMallocMapped), hipSuccess);
    long sys1 = 0, ttm1 = 0;
    ASSERT_TRUE(ReadKfdMemUsed(&sys1, &ttm1));
    ASSERT_EQ(hipHostFree(gtt), hipSuccess);
    EXPECT_GE(ttm1 - ttm0, kMb - kSlackMb)
        << "hipHostMalloc was not charged to TTM; the comparison is void";
    // The control allocation's charge must be back before the baseline.
    ASSERT_TRUE(poll([&] { return ttm - ttm0 < kSlackMb; }));
    ASSERT_LT(ttm - ttm0, kSlackMb) << "the hipHostMalloc charge stayed";
    const long sys2 = sys, ttm2 = ttm;

    long sys3 = 0, ttm3 = 0;
    {
        auto buf = PinnedBufferPool::AllocatePinned(kBytes, /*mapped=*/true);
        ASSERT_NE(buf.data, nullptr);
        ASSERT_TRUE(ReadKfdMemUsed(&sys3, &ttm3));
    }
    ASSERT_TRUE(poll([&] { return sys - sys3 <= -(kMb - kSlackMb); }));
    std::cout << "hipHostMalloc: system +" << sys1 - sys0 << "M, TTM +"
              << ttm1 - ttm0 << "M; mapped buffer: system +" << sys3 - sys2
              << "M, TTM +" << ttm3 - ttm2 << "M; after release: system "
              << sys - sys3 << "M, TTM " << ttm - ttm3 << "M" << std::endl;
    EXPECT_LT(ttm3 - ttm2, kSlackMb);
    EXPECT_GE(sys3 - sys2, kMb - kSlackMb);
    EXPECT_LE(sys - sys3, -(kMb - kSlackMb))
        << "the charge did not come back within 5 s of the release";
    // A system-only drop: what came back was not some GTT allocation's.
    EXPECT_LT(std::labs(ttm - ttm3), kSlackMb);
}

}  // namespace
}  // namespace mooncake
