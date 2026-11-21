// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "device/accelerator_registry.h"

// Linker wrappers inject failures before any work is queued. The sync wrapper
// deliberately returns before waiting, so ownership must survive an error.
static std::atomic<bool> fail_alloc{false}, fail_launch{false},
    fail_sync{false};
static std::atomic<int> host_allocs{0}, host_frees{0};
extern "C" hipError_t __real_hipHostMalloc(void **, size_t, unsigned int);
extern "C" hipError_t __wrap_hipHostMalloc(void **p, size_t n, unsigned int f) {
    if (fail_alloc.exchange(false)) return hipErrorOutOfMemory;
    auto result = __real_hipHostMalloc(p, n, f);
    if (result == hipSuccess) ++host_allocs;
    return result;
}
extern "C" hipError_t __real_hipHostFree(void *);
extern "C" hipError_t __wrap_hipHostFree(void *p) {
    auto result = __real_hipHostFree(p);
    if (result == hipSuccess) ++host_frees;
    return result;
}
extern "C" hipError_t __real_hipStreamSynchronize(hipStream_t);
extern "C" hipError_t __wrap_hipStreamSynchronize(hipStream_t s) {
    return fail_sync.exchange(false) ? hipErrorInvalidValue
                                     : __real_hipStreamSynchronize(s);
}
extern "C" hipError_t __real_hipModuleLaunchKernel(hipFunction_t, unsigned,
                                                   unsigned, unsigned, unsigned,
                                                   unsigned, unsigned, unsigned,
                                                   hipStream_t, void **,
                                                   void **);
extern "C" hipError_t __wrap_hipModuleLaunchKernel(hipFunction_t f, unsigned x,
                                                   unsigned y, unsigned z,
                                                   unsigned bx, unsigned by,
                                                   unsigned bz, unsigned shared,
                                                   hipStream_t s, void **args,
                                                   void **extra) {
    if (fail_launch.exchange(false)) return hipErrorInvalidValue;
    return __real_hipModuleLaunchKernel(f, x, y, z, bx, by, bz, shared, s, args,
                                        extra);
}
void Check(bool ok, int line) {
    if (!ok) {
        std::fprintf(stderr, "HIP H2D regression failed at line %d\n", line);
        std::exit(3);
    }
}
#define Require(ok) Check((ok), __LINE__)
int main(int argc, char **argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    Require(hipSetDevice(device) == hipSuccess);
    const auto *acc = mooncake::device::GetAcceleratorRegistry().GetDevice(
        mooncake::device::AcceleratorVendor::kHip);
    Require(acc != nullptr);
    auto source = acc->AllocateMappedPinnedHost(32 * 1024 * 1024);
    Require(source.addr && source.device_addr);
    void *target;
    Require(hipMalloc(&target, source.size) == hipSuccess);
    for (size_t i = 0; i < source.size; ++i)
        static_cast<uint8_t *>(source.addr)[i] = (i * 29 + (i >> 8)) % 251;
    auto make_ranges = [&](int count, size_t bytes, int shift = 0) {
        std::vector<mooncake::device::HostCopyRange> ranges;
        const size_t stride = (bytes + 255) / 256 * 256 + 256;
        for (int i = 0; i < count; ++i) {
            const size_t offset = i * stride + shift;
            ranges.push_back(
                {static_cast<char *>(target) + offset,
                 static_cast<char *>(source.addr) + offset, bytes,
                 static_cast<char *>(source.device_addr) + offset});
        }
        return ranges;
    };
    auto validate = [&](const auto &ranges) {
        for (auto range : ranges) {
            std::vector<uint8_t> got(range.size + 1);
            Require(hipMemcpy(got.data(), range.dst, got.size(),
                              hipMemcpyDeviceToHost) == hipSuccess);
            Require(std::memcmp(got.data(), range.src, range.size) == 0);
            Require(got.back() == 0xcd);
        }
    };
    void *stream;
    Require(acc->CreateStream(&stream));
    int checks = 0;
    for (int count : {0, 1, 31, 32, 127, 128, 129, 255, 256, 257}) {
        for (size_t bytes :
             {size_t{656}, size_t{16384}, size_t{41984}, size_t{65536}}) {
            auto ranges = make_ranges(count, bytes);
            Require(hipMemset(target, 0xcd, source.size) == hipSuccess);
            bool used = false;
            Require(acc->CopyFromHostBatchAsync(ranges, stream, &used));
            Require(used == (count >= 32));
            Require(acc->SynchronizeStream(stream));
            validate(ranges);
            ++checks;
        }
    }
    for (int shift : {0, 1, 3, 8, 16}) {
        for (size_t bytes :
             {size_t{1}, size_t{7}, size_t{15}, size_t{17}, size_t{41991}}) {
            auto ranges = make_ranges(256, bytes, shift);
            Require(hipMemset(target, 0xcd, source.size) == hipSuccess);
            Require(acc->CopyFromHostBatchAsync(ranges, stream));
            Require(acc->SynchronizeStream(stream));
            validate(ranges);
            ++checks;
        }
    }
    Require(hipMemset(target, 0xcd, source.size) == hipSuccess);
    auto ranges = make_ranges(256, 41984);
    const int allocations = host_allocs;
    Require(acc->CopyFromHostBatchAsync(ranges, stream));
    Require(acc->SynchronizeStream(stream));
    Require(host_allocs == allocations);
    ++checks;
    {
        auto first = make_ranges(128, 656);
        auto second = make_ranges(128, 41991);
        constexpr size_t offset = 8 * 1024 * 1024;
        for (auto &range : second) {
            range.dst = static_cast<char *>(range.dst) + offset;
            range.src = static_cast<const char *>(range.src) + offset;
            range.src_device =
                static_cast<const char *>(range.src_device) + offset;
        }
        Require(hipMemset(target, 0xcd, source.size) == hipSuccess);
        Require(acc->CopyFromHostBatchAsync(first, stream));
        Require(acc->CopyFromHostBatchAsync(second, stream));
        Require(acc->SynchronizeStream(stream));
        validate(first);
        validate(second);
        ++checks;
    }
    Require(hipMemset(target, 0xcd, source.size) == hipSuccess);
    for (auto *flag : {&fail_alloc, &fail_launch}) {
        void *fresh;
        Require(acc->CreateStream(&fresh));
        *flag = true;
        bool used = true;
        Require(acc->CopyFromHostBatchAsync(ranges, fresh, &used));
        Require(!used);
        Require(acc->SynchronizeStream(fresh));
        validate(ranges);
        acc->DestroyStream(fresh);
        ++checks;
    }
    Require(acc->CopyFromHostBatchAsync(ranges, stream));
    const int frees = host_frees;
    fail_sync = true;
    Require(!acc->SynchronizeStream(stream));
    Require(host_frees == frees);
    Require(acc->CopyFromHostBatchAsync(ranges, stream));
    Require(acc->SynchronizeStream(stream));
    validate(ranges);
    ++checks;
    std::atomic<bool> running{true};
    std::thread syncer([&] {
        Require(hipSetDevice(device) == hipSuccess);
        while (running) {
            Require(acc->SynchronizeStream(stream));
            std::this_thread::yield();
        }
    });
    for (int i = 0; i < 120; ++i)
        Require(acc->CopyFromHostBatchAsync(ranges, stream));
    running = false;
    syncer.join();
    Require(acc->SynchronizeStream(stream));
    validate(ranges);
    ++checks;
    for (void *special :
         {static_cast<void *>(nullptr), static_cast<void *>(hipStreamPerThread),
          static_cast<void *>(hipStreamLegacy)}) {
        bool used = true;
        Require(acc->CopyFromHostBatchAsync(ranges, special, &used));
        Require(!used);
        Require(acc->SynchronizeStream(special));
        ++checks;
    }
    auto large = make_ranges(3000, 656);
    for (int i = 0; i < 2; ++i) {
        Require(hipMemset(target, 0xcd, source.size) == hipSuccess);
        Require(acc->CopyFromHostBatchAsync(large, stream));
        Require(acc->SynchronizeStream(stream));
        validate(large);
        ++checks;
    }
    acc->DestroyStream(stream);
    Require(hipFree(target) == hipSuccess);
    std::printf(
        "PASS device=%d checks=%d (geometry, tails, warm reuse, failures, "
        "concurrency, DMA, ceiling)\n",
        device, checks);
}
