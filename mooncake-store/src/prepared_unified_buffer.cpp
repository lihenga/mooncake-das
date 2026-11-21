#include "prepared_unified_buffer.h"

#include <glog/logging.h>

#include <map>
#include <mutex>
#include <utility>

#include "device/accelerator_registry.h"
#include "config/distributed_storage_config.h"
#include "config/dfs_enablement_config.h"
#include "storage_backend.h"

namespace mooncake {
namespace {

struct Entry {
    PinnedBufferPool::Buffer buffer;
    size_t requested_size = 0;
};

// Process-wide and never destroyed, so a buffer still inside the table at exit
// releases through the runtime after the static destructors that own the HIP
// deleter have already run.
std::map<int32_t, Entry> &PreparedBuffers() {
    static auto *table = new std::map<int32_t, Entry>();
    return *table;
}

std::mutex &PreparedMutex() {
    static auto *mutex = new std::mutex();
    return *mutex;
}

}  // namespace

int PreparedUnifiedBuffer::Prepare(int32_t device_id, size_t size) {
    if (device_id < 0) {
        LOG(WARNING) << "prepare_dfs_unified_buffer: invalid device id "
                     << device_id;
        return -1;
    }
    if (size == 0) {
        LOG(WARNING) << "prepare_dfs_unified_buffer: size must be positive";
        return -1;
    }
    {
        std::lock_guard<std::mutex> lock(PreparedMutex());
        auto it = PreparedBuffers().find(device_id);
        if (it != PreparedBuffers().end()) {
            if (it->second.requested_size >= size) {
                LOG(INFO) << "prepare_dfs_unified_buffer: device " << device_id
                          << " already prepared, size="
                          << it->second.requested_size;
                return 0;
            }
            LOG(WARNING) << "prepare_dfs_unified_buffer: device " << device_id
                         << " is prepared with " << it->second.requested_size
                         << " bytes, smaller than the requested " << size
                         << "; replacing it";
            PreparedBuffers().erase(it);
        }
    }

    // Allocate outside the lock: the pool may block in the runtime.
    auto buffer = PinnedBufferPool::AllocatePinned(size, /*mapped=*/true);
    if (!buffer.pinned_host.addr || !buffer.mapped ||
        buffer.device_data == nullptr) {
        LOG(ERROR) << "prepare_dfs_unified_buffer: mapped allocation failed, "
                      "device="
                   << device_id << " size=" << size;
        return -1;
    }

    std::lock_guard<std::mutex> lock(PreparedMutex());
    auto it = PreparedBuffers().find(device_id);
    if (it != PreparedBuffers().end() && it->second.requested_size >= size) {
        // Another thread prepared an adequate buffer first; keep it and
        // release ours at scope exit.
        LOG(INFO) << "prepare_dfs_unified_buffer: device " << device_id
                  << " prepared concurrently; keeping the existing buffer";
        return 0;
    }
    if (it != PreparedBuffers().end()) {
        // The concurrent entry is smaller than this request. Replace it with
        // the buffer just allocated, so a successful return always promises
        // at least the requested capacity.
        PreparedBuffers().erase(it);
    }
    Entry entry;
    entry.buffer = std::move(buffer);
    entry.requested_size = size;
    PreparedBuffers()[device_id] = std::move(entry);
    LOG(INFO) << "prepare_dfs_unified_buffer: device " << device_id
              << " size=" << size << " ready";
    return 0;
}

std::optional<PinnedBufferPool::Buffer> PreparedUnifiedBuffer::Claim(
    int32_t device_id, size_t size) {
    std::lock_guard<std::mutex> lock(PreparedMutex());
    auto it = PreparedBuffers().find(device_id);
    if (it == PreparedBuffers().end()) return std::nullopt;
    if (it->second.requested_size < size) return std::nullopt;
    // Ownership leaves the table: the caller frees the buffer through its
    // deleter, so a second Claim for the same device must not see it again.
    PinnedBufferPool::Buffer buffer = std::move(it->second.buffer);
    PreparedBuffers().erase(it);
    return buffer;
}

int PrepareDfsUnifiedBuffer(int32_t device_id, int64_t size_bytes) {
    const auto file_storage_config = FileStorageConfig::FromEnvironment();
    const auto distributed_config = DistributedStorageConfig::FromEnvironment();
    const bool distributed_dfs =
        file_storage_config.storage_backend_type ==
            StorageBackendType::kDistributed &&
        (!distributed_config.UsesObjectStorage() ||
         distributed_config.UsesKvcs());
    const bool dfs_available =
        DfsEnablementConfig::FromEnvironment().enabled || distributed_dfs;
    if (!dfs_available) {
        LOG(INFO) << "prepare_dfs_unified_buffer: DFS is disabled, skipping";
        return 0;
    }

    size_t request = 0;
    if (size_bytes > 0) {
        request = static_cast<size_t>(size_bytes);
    } else {
        request = static_cast<size_t>(
            FileStorageConfig::FromEnvironment().unified_buffer_size);
    }
    if (request == 0) {
        LOG(INFO) << "prepare_dfs_unified_buffer: unified buffer is disabled "
                     "(size 0), nothing to prepare";
        return 0;
    }

    // Make the requested card current before allocating: the NUMA policy is
    // derived from the current device.
    auto accelerators =
        device::GetAcceleratorRegistry().RuntimeAccelerators(true).Devices();
    for (auto *accelerator : accelerators) {
        if (accelerator != nullptr && device_id >= 0) {
            accelerator->SetContext(device_id);
        }
    }
    if (device_id < 0) {
        for (auto *accelerator : accelerators) {
            if (accelerator == nullptr) continue;
            device_id = accelerator->CurrentDeviceId();
            if (device_id >= 0) break;
        }
    }
    return PreparedUnifiedBuffer::Prepare(device_id, request);
}

}  // namespace mooncake
