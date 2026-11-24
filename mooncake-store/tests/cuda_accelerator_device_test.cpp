#include "device/accelerator_registry.h"

#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace mooncake::device {
namespace {

class CudaAcceleratorDeviceTest : public ::testing::Test {
   protected:
    static constexpr size_t kBytes = 64 * 1024;

    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "A CUDA device is required";
        }
        ASSERT_EQ(cudaGetDevice(&original_device_), cudaSuccess);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        device_ =
            GetAcceleratorRegistry().GetDevice(AcceleratorVendor::kNvidia);
        ASSERT_NE(device_, nullptr);
        source_ = device_->AllocatePinnedHost(kBytes);
        ASSERT_NE(source_.addr, nullptr);
        ASSERT_EQ(cudaMalloc(&destination_, kBytes), cudaSuccess);
        ASSERT_TRUE(device_->CreateStream(&stream_));
    }

    void TearDown() override {
        if (stream_) {
            device_->SynchronizeStream(stream_);
            device_->DestroyStream(stream_);
        }
        if (destination_) cudaFree(destination_);
        source_.reset();
        if (original_device_ >= 0) cudaSetDevice(original_device_);
    }

    const AcceleratorDevice *device_ = nullptr;
    PinnedHostBuffer source_;
    void *destination_ = nullptr;
    void *stream_ = nullptr;
    int original_device_ = -1;
};

TEST_F(CudaAcceleratorDeviceTest, BatchCopiesUnalignedRangesAndPreservesGaps) {
    auto *source = static_cast<unsigned char *>(source_.addr);
    for (size_t i = 0; i < kBytes; ++i) source[i] = (i * 17 + 3) & 0xff;

    for (size_t count : {0, 1, 31, 32, 65}) {
        SCOPED_TRACE(count);
        ASSERT_EQ(cudaMemset(destination_, 0xa5, kBytes), cudaSuccess);
        std::vector<unsigned char> expected(kBytes, 0xa5);
        std::vector<HostCopyRange> ranges;
        for (size_t i = 0; i < count; ++i) {
            const size_t src_offset = i * 512 + 7;
            const size_t dst_offset = i * 512 + 3;
            const size_t size = 129 + i % 73;
            ranges.push_back(
                {static_cast<unsigned char *>(destination_) + dst_offset,
                 source + src_offset, size});
            std::memcpy(expected.data() + dst_offset, source + src_offset,
                        size);
        }
        bool used_kernel = true;
        ASSERT_TRUE(
            device_->CopyFromHostBatchAsync(ranges, stream_, &used_kernel));
        // A native CUDA batch is not the Hygon mapped-host scatter kernel.
        EXPECT_FALSE(used_kernel);
        ASSERT_TRUE(device_->SynchronizeStream(stream_));
        std::vector<unsigned char> actual(kBytes);
        ASSERT_TRUE(device_->Copy(actual.data(), destination_, kBytes,
                                  CopyDirection::kDeviceToHost));
        EXPECT_EQ(actual, expected);
    }
}

TEST_F(CudaAcceleratorDeviceTest,
       MappedBatchesRetainDescriptorsUntilCompletion) {
    constexpr size_t count = 65;
    constexpr size_t stride = 8192;
    constexpr size_t bytes = 2 * count * stride;
    auto source = device_->AllocateMappedPinnedHost(bytes);
    ASSERT_NE(source.addr, nullptr);
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
    const bool expect_kernel = CUDART_VERSION >= 13000 && source.device_addr &&
                               properties.major == 10 && properties.minor == 3;
    void* destination = nullptr;
    ASSERT_EQ(cudaMalloc(&destination, bytes), cudaSuccess);
    auto* src = static_cast<unsigned char*>(source.addr);
    auto* dst = static_cast<unsigned char*>(destination);
    for (size_t i = 0; i < bytes; ++i) src[i] = (i * 17 + i / stride) & 0xff;

    for (int round = 0; round < 3; ++round) {
        ASSERT_EQ(cudaMemset(destination, 0xa5, bytes), cudaSuccess);
        std::vector<unsigned char> expected(bytes, 0xa5);
        // Two different descriptor arrays must survive together on one stream.
        for (size_t batch = 0; batch < 2; ++batch) {
            std::vector<HostCopyRange> ranges;
            for (size_t i = 0; i < count; ++i) {
                size_t offset = (batch * count + i) * stride + 1;
                size_t size = 4097 + i % 17;
                const void* alias =
                    source.device_addr
                        ? static_cast<unsigned char*>(source.device_addr) +
                              offset
                        : nullptr;
                ranges.push_back({dst + offset, src + offset, size, alias});
                std::memcpy(expected.data() + offset, src + offset, size);
            }
            bool used_kernel = false;
            ASSERT_TRUE(
                device_->CopyFromHostBatchAsync(ranges, stream_, &used_kernel));
            EXPECT_EQ(used_kernel, expect_kernel);
        }
        ASSERT_TRUE(device_->SynchronizeStream(stream_));
        std::vector<unsigned char> actual(bytes);
        ASSERT_EQ(cudaMemcpy(actual.data(), destination, bytes,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(actual, expected);
    }
    ASSERT_EQ(cudaFree(destination), cudaSuccess);
}

TEST_F(CudaAcceleratorDeviceTest, DestroyStreamDrainsPendingMappedBatch) {
    constexpr size_t count = 32;
    constexpr size_t size = 4096;
    constexpr size_t bytes = count * size;
    auto source = device_->AllocateMappedPinnedHost(bytes);
    ASSERT_NE(source.addr, nullptr);
    std::memset(source.addr, 0x3c, bytes);
    void* destination = nullptr;
    ASSERT_EQ(cudaMalloc(&destination, bytes), cudaSuccess);
    std::vector<HostCopyRange> ranges;
    for (size_t i = 0; i < count; ++i) {
        auto* src = static_cast<unsigned char*>(source.addr) + i * size;
        const void* alias =
            source.device_addr
                ? static_cast<unsigned char*>(source.device_addr) + i * size
                : nullptr;
        ranges.push_back({static_cast<unsigned char*>(destination) + i * size,
                          src, size, alias});
    }
    ASSERT_TRUE(device_->CopyFromHostBatchAsync(ranges, stream_));
    device_->DestroyStream(stream_);
    stream_ = nullptr;
    std::vector<unsigned char> actual(bytes);
    ASSERT_EQ(
        cudaMemcpy(actual.data(), destination, bytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(actual, std::vector<unsigned char>(bytes, 0x3c));
    ASSERT_EQ(cudaFree(destination), cudaSuccess);
}

}  // namespace
}  // namespace mooncake::device
