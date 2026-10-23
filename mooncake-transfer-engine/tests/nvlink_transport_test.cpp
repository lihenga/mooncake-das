#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <memory>
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cuda_alike.h"
#include "transfer_engine.h"
#include "transport/transport.h"

using namespace mooncake;

// Select protocol based on build configuration
#ifdef USE_HIP
#define MNNVL_PROTOCOL "hip"
#else
#define MNNVL_PROTOCOL "nvlink"
#endif

DEFINE_string(metadata_server, P2PHANDSHAKE, "Metadata server or P2PHANDSHAKE");
DEFINE_string(local_server_name, "127.0.0.1:12345", "Local server name");
DEFINE_string(segment_id, "", "Remote segment; empty uses the test server");
DEFINE_int32(gpu_id, 0, "GPU ID to use");

static void checkCudaError(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        LOG(ERROR) << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")";
        exit(EXIT_FAILURE);
    }
}

static void* allocateCudaBuffer(size_t size, int gpu_id) {
    checkCudaError(cudaSetDevice(gpu_id), "Failed to set device");
    void* d_buf = nullptr;
    checkCudaError(cudaMalloc(&d_buf, size),
                   "Failed to allocate device memory");
    return d_buf;
}

static void freeCudaBuffer(void* addr) {
    checkCudaError(cudaFree(addr), "Failed to free device memory");
}

namespace {
constexpr size_t kDataLength = 4096000;
struct ServerInfo {
    uint64_t address;
    char endpoint[256];
};

// Fork before initializing either accelerator runtime: IPC handles must be
// imported in a different process from the one that exported the allocation.
int runServer(int fd) {
    auto engine = std::make_unique<TransferEngine>(false);
    if (engine->init(FLAGS_metadata_server, FLAGS_local_server_name) != 0 ||
        !engine->installTransport(MNNVL_PROTOCOL, nullptr))
        return 1;
    void* buffer = allocateCudaBuffer(kDataLength * 2, FLAGS_gpu_id);
    const std::string location = GPU_PREFIX + std::to_string(FLAGS_gpu_id);
    if (engine->registerLocalMemory(buffer, kDataLength * 2, location) != 0) {
        freeCudaBuffer(buffer);
        return 1;
    }
    ServerInfo info{};
    info.address = reinterpret_cast<uint64_t>(buffer);
    const auto endpoint = engine->getLocalIpAndPort();
    int rc = 1;
    if (endpoint.size() < sizeof(info.endpoint)) {
        std::snprintf(info.endpoint, sizeof(info.endpoint), "%s",
                      endpoint.c_str());
        if (send(fd, &info, sizeof(info), MSG_NOSIGNAL) == sizeof(info)) {
            char stop;
            rc = recv(fd, &stop, 1, 0) == 0 ? 0 : 1;
        }
    }
    engine->unregisterLocalMemory(buffer);
    freeCudaBuffer(buffer);
    return rc;
}

struct ServerProcess {
    pid_t pid;
    int fd;
    int finish() {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        int status = 0;
        pid_t reaped;
        do {
            reaped = waitpid(pid, &status, 0);
        } while (reaped < 0 && errno == EINTR);
        pid = -1;
        return reaped > 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    ~ServerProcess() {
        if (pid > 0) finish();
    }
};
}  // namespace

TEST(NvlinkTransportTest, WriteAndRead) {
    int gpu_id = FLAGS_gpu_id;
    const std::string location = GPU_PREFIX + std::to_string(gpu_id);

    int sockets[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sockets), 0);
    const pid_t pid = fork();
    if (pid < 0) {
        close(sockets[0]);
        close(sockets[1]);
        FAIL() << "fork failed";
    }
    if (pid == 0) {
        close(sockets[0]);
        alarm(90);
        const int rc = runServer(sockets[1]);
        close(sockets[1]);
        _exit(rc);
    }
    close(sockets[1]);
    ServerProcess server{pid, sockets[0]};
    const timeval receive_timeout{30, 0};
    ASSERT_EQ(setsockopt(server.fd, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
                         sizeof(receive_timeout)),
              0);
    ServerInfo info{};
    ASSERT_EQ(recv(server.fd, &info, sizeof(info), 0), sizeof(info));

    // Client (initiator) setup
    auto client_engine = std::make_unique<TransferEngine>(false);
    ASSERT_EQ(client_engine->init(FLAGS_metadata_server, "127.0.0.1:12346"), 0);

    // Install MNNVL transport (nvlink or hip) on client
    Transport* client_transport =
        client_engine->installTransport(MNNVL_PROTOCOL, nullptr);
    ASSERT_NE(client_transport, nullptr);
    const auto segment_id = client_engine->openSegment(
        FLAGS_segment_id.empty() ? info.endpoint : FLAGS_segment_id);
    ASSERT_GE(segment_id, 0);

    void* client_buffer = allocateCudaBuffer(kDataLength * 2, gpu_id);
    int rc = client_engine->registerLocalMemory(client_buffer, kDataLength * 2,
                                                location);
    ASSERT_EQ(rc, 0);

    // Write: client -> server
    {
        // Fill client buffer with data
        std::vector<char> host_data(kDataLength, 'A');
        checkCudaError(cudaMemcpy(client_buffer, host_data.data(), kDataLength,
                                  cudaMemcpyHostToDevice),
                       "Memcpy to client_buffer");

        auto batch_id = client_engine->allocateBatchID(1);
        TransferRequest entry;
        entry.opcode = TransferRequest::WRITE;
        entry.length = kDataLength;
        entry.source = client_buffer;
        entry.target_id = segment_id;
        entry.target_offset = info.address;
        Status s = client_engine->submitTransfer(batch_id, {entry});
        ASSERT_TRUE(s.ok());

        // Wait for completion
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        TransferStatus status;
        do {
            s = client_engine->getTransferStatus(batch_id, 0, status);
            ASSERT_TRUE(s.ok());
            ASSERT_LT(std::chrono::steady_clock::now(), deadline);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (status.s == TransferStatusEnum::WAITING ||
                 status.s == TransferStatusEnum::PENDING);

        ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
        s = client_engine->freeBatchID(batch_id);
        ASSERT_TRUE(s.ok());
    }

    // Read: server -> client
    {
        auto batch_id = client_engine->allocateBatchID(1);
        TransferRequest entry;
        entry.opcode = TransferRequest::READ;
        entry.length = kDataLength;
        entry.source = (char*)client_buffer + kDataLength;
        entry.target_id = segment_id;
        entry.target_offset = info.address;
        Status s = client_engine->submitTransfer(batch_id, {entry});
        ASSERT_TRUE(s.ok());

        // Wait for completion
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        TransferStatus status;
        do {
            s = client_engine->getTransferStatus(batch_id, 0, status);
            ASSERT_TRUE(s.ok());
            ASSERT_LT(std::chrono::steady_clock::now(), deadline);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (status.s == TransferStatusEnum::WAITING ||
                 status.s == TransferStatusEnum::PENDING);

        ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
        s = client_engine->freeBatchID(batch_id);
        ASSERT_TRUE(s.ok());
    }

    // Check data
    std::vector<char> host_check(kDataLength);
    checkCudaError(
        cudaMemcpy(host_check.data(), (char*)client_buffer + kDataLength,
                   kDataLength, cudaMemcpyDeviceToHost),
        "Memcpy from client_buffer");
    for (size_t i = 0; i < kDataLength; ++i) {
        ASSERT_EQ(host_check[i], 'A');
    }

    // Cleanup
    client_engine->unregisterLocalMemory(client_buffer);
    freeCudaBuffer(client_buffer);
    // Close the importing engine before allowing the exporter to free memory.
    client_engine.reset();
    EXPECT_EQ(server.finish(), 0);
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
