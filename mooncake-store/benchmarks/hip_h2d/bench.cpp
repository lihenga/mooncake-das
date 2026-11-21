// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "device/accelerator_registry.h"

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        const auto error = (call);                                             \
        if (error != hipSuccess) {                                             \
            std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error)); \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

using Clock = std::chrono::steady_clock;
struct DeviceRange {
    uint64_t dst, src, bytes;
};
struct Shape {
    int count;
    size_t bytes;
    int shift;
};
double Percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(fraction * (values.size() - 1))];
}

int main(int argc, char **argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 100;
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    const std::string mode = argc > 3 ? argv[3] : "chain";
    if (repeats <= 0 ||
        (mode != "chain" && mode != "scatter" && mode != "grouped"))
        return 2;
    HIP_CHECK(hipSetDevice(device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::fprintf(stderr, "%s %s, %d CUs, wave=%d\n", properties.name,
                 properties.gcnArchName, properties.multiProcessorCount,
                 properties.warpSize);
    const auto *accelerator =
        mooncake::device::GetAcceleratorRegistry().GetDevice(
            mooncake::device::AcceleratorVendor::kHip);
    if (!accelerator) return 2;
    void *stream;
    if (!accelerator->CreateStream(&stream)) return 2;
    auto hip_stream = static_cast<hipStream_t>(stream);
    hipEvent_t start, finish;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&finish));
    hipModule_t module{};
    hipFunction_t function{};
    if (mode != "chain") {
        const char *directory = std::getenv("MC_COPY_KERNEL_PATH");
        if (!directory) return 2;
        const auto path = std::string(directory) + "/mc_copy_kernel.co";
        HIP_CHECK(hipModuleLoad(&module, path.c_str()));
        HIP_CHECK(hipModuleGetFunction(&function, module,
                                       mode == "grouped"
                                           ? "MCStoreH2DGroupedKernel"
                                           : "MCStoreH2DScatterKernel"));
    }
    std::puts(
        "device,count,range_bytes,shift,mode,payload_bytes,repeats,"
        "wall_p50_us,wall_p95_us,event_p50_us,effective_GBs,event_GBs,correct");
    for (const Shape shape : {Shape{32, 656, 0},
                              {128, 16384, 0},
                              {256, 41984, 0},
                              {1024, 41984, 0},
                              {256, 65536, 0},
                              {256, 41991, 0},
                              {256, 41984, 8},
                              {1, 16 * 1024 * 1024, 0}}) {
        if (mode == "grouped" && (shape.count < 128 || shape.bytes > 65536))
            continue;
        const size_t stride = (shape.bytes + 255) / 256 * 256 + 256;
        const size_t span = stride * shape.count;
        const size_t slots =
            std::max(size_t{3}, (32 * 1024 * 1024 + span - 1) / span);
        const size_t capacity = span * slots + 256;
        uint8_t *host, *alias, *gpu;
        HIP_CHECK(hipHostMalloc(&host, capacity, hipHostMallocMapped));
        HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void **>(&alias),
                                          host, 0));
        HIP_CHECK(hipMalloc(&gpu, capacity));
        for (size_t i = 0; i < capacity; ++i)
            host[i] = static_cast<uint8_t>((i * 17 + (i >> 9) * 13) % 251);
        HIP_CHECK(hipMemset(gpu, 0xcd, capacity));
        DeviceRange *descriptors, *descriptor_alias;
        HIP_CHECK(hipHostMalloc(&descriptors, shape.count * sizeof(DeviceRange),
                                hipHostMallocMapped));
        HIP_CHECK(hipHostGetDevicePointer(
            reinterpret_cast<void **>(&descriptor_alias), descriptors, 0));
        std::vector<mooncake::device::HostCopyRange> ranges(shape.count);
        std::vector<double> walls, events;
        size_t slot = 0;
        for (int iteration = -5; iteration < repeats; ++iteration) {
            const auto wall_start = Clock::now();
            slot = static_cast<size_t>(iteration + 5) % slots;
            for (int j = 0; j < shape.count; ++j) {
                const size_t src = slot * span + j * stride + shape.shift;
                const size_t dst = slot * span +
                                   ((j * 17) % shape.count) * stride +
                                   shape.shift;
                ranges[j] = {gpu + dst, host + src, shape.bytes, alias + src};
                descriptors[j] = {reinterpret_cast<uint64_t>(gpu + dst),
                                  reinterpret_cast<uint64_t>(alias + src),
                                  shape.bytes};
            }
            HIP_CHECK(hipEventRecord(start, hip_stream));
            if (mode == "chain") {
                if (!accelerator->CopyFromHostBatchAsync(ranges, stream))
                    return 2;
            } else {
                uint64_t count = shape.count;
                void *arguments[] = {&descriptor_alias, &count};
                const unsigned blocks =
                    mode == "grouped" ? std::min((shape.count + 3) / 4, 32)
                                      : shape.count;
                HIP_CHECK(hipModuleLaunchKernel(function, blocks, 1, 1, 256, 1,
                                                1, 0, hip_stream, arguments,
                                                nullptr));
            }
            HIP_CHECK(hipEventRecord(finish, hip_stream));
            HIP_CHECK(hipEventSynchronize(finish));
            if (mode == "chain" && !accelerator->SynchronizeStream(stream))
                return 2;
            const auto wall_finish = Clock::now();
            float milliseconds;
            HIP_CHECK(hipEventElapsedTime(&milliseconds, start, finish));
            if (iteration >= 0) {
                walls.push_back(std::chrono::duration<double, std::micro>(
                                    wall_finish - wall_start)
                                    .count());
                events.push_back(milliseconds * 1000);
            }
        }
        std::vector<uint8_t> got(span + 256);
        HIP_CHECK(hipMemcpy(got.data(), gpu + slot * span, got.size(),
                            hipMemcpyDeviceToHost));
        bool correct = true;
        for (int j = 0; j < shape.count; ++j) {
            const size_t src = slot * span + j * stride + shape.shift;
            const size_t dst = ((j * 17) % shape.count) * stride + shape.shift;
            correct &=
                std::memcmp(got.data() + dst, host + src, shape.bytes) == 0 &&
                got[dst + shape.bytes] == 0xcd;
        }
        const double wall = Percentile(walls, .5),
                     event = Percentile(events, .5);
        const size_t payload = shape.count * shape.bytes;
        std::printf("%d,%d,%zu,%d,%s,%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
                    device, shape.count, shape.bytes, shape.shift, mode.c_str(),
                    payload, repeats, wall, Percentile(walls, .95), event,
                    payload / wall / 1000, payload / event / 1000, correct);
        std::fflush(stdout);
        if (!correct) return 3;
        HIP_CHECK(hipHostFree(descriptors));
        HIP_CHECK(hipFree(gpu));
        HIP_CHECK(hipHostFree(host));
    }
    accelerator->DestroyStream(stream);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(finish));
    if (module) HIP_CHECK(hipModuleUnload(module));
}
