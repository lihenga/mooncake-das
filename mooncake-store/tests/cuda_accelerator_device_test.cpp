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

}  // namespace
}  // namespace mooncake::device
