#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "pinned_buffer_pool.h"

namespace mooncake {

// A mapped unified DFS buffer allocated before the model is loaded, while
// system memory is still free. Allocating at that point is what makes the span
// mostly huge-page backed: once the model and its page cache have filled
// memory, the same allocation falls back to 4 KiB pages, and every 4 KiB page
// is a migration candidate that evicts the GPU queues mapping it.
//
// The flow is prepare -> claim -> own:
//   PreparedUnifiedBuffer::Prepare() allocates and registers, keeping the
//                                   buffer in a process-wide table keyed by
//                                   device id.
//   PreparedUnifiedBuffer::Claim()   hands ownership to FileStorage, which owns
//                                   it through its pinned_restore_arena_ from
//                                   then on.
// A buffer that is never claimed is freed when the process exits.
class PreparedUnifiedBuffer {
   public:
    // Allocates `size` mapped host bytes for `device_id` and stores them under
    // that device. Idempotent: a second call for the same device with the same
    // or a smaller size returns 0; a larger request replaces the table entry.
    // Returns 0 on success, -1 on failure.
    static int Prepare(int32_t device_id, size_t size);

    // Takes the prepared buffer for `device_id` if it is present and large
    // enough (`size` is the requested capacity). On success the caller owns it
    // and it is removed from the table.
    static std::optional<PinnedBufferPool::Buffer> Claim(int32_t device_id,
                                                         size_t size);
};

// Process-level entry point for callers that run before any Mooncake client
// exists (the model loader does). Prepares the buffer for `device_id` and,
// best effort, loads the H2D scatter kernel so the first business batch does
// not pay for it. size_bytes <= 0 reads MC_STORE_DFS_UNIFIED_BUFFER_SIZE_BYTES.
// Returns 0 on success, -1 on failure.
int PrepareDfsUnifiedBuffer(int32_t device_id, int64_t size_bytes);

}  // namespace mooncake
