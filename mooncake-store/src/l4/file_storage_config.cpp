// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "storage_backend.h"
#include "l4/file_storage_config.h"

#include <cmath>
#include <filesystem>
#include <locale>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "bool_parser.h"
#include "environ.h"
#include "environment_variables.h"

namespace mooncake::l4 {

void ReadDfsBufferEnvironment(FileStorageConfig& config) {
    using Variables = FileStorageEnvironmentVariables;
    config.unified_buffer_size =
        Environ::ReadOr(Variables::MC_STORE_DFS_UNIFIED_BUFFER_SIZE_BYTES,
                        config.unified_buffer_size);
    config.read_chunk_size = Environ::ReadOr(
        Variables::MC_STORE_DFS_READ_CHUNK_SIZE_BYTES, config.read_chunk_size);
    config.write_chunk_size =
        Environ::ReadOr(Variables::MC_STORE_DFS_WRITE_CHUNK_SIZE_BYTES,
                        config.write_chunk_size);
    constexpr int64_t kDefaultDfsChunkSize = 1LL * 1024 * 1024 * 1024;
    if (config.read_chunk_size <= 0)
        config.read_chunk_size = kDefaultDfsChunkSize;
    if (config.write_chunk_size <= 0) {
        config.write_chunk_size = kDefaultDfsChunkSize;
    }
    const auto prefetch_ratio =
        Environ::ReadOr(Variables::MC_STORE_DFS_PREFETCH_CAPACITY_RATIO,
                        std::to_string(config.prefetch_capacity_ratio));
    std::istringstream ratio_stream(prefetch_ratio);
    ratio_stream.imbue(std::locale::classic());
    double parsed_ratio = 0.0;
    ratio_stream >> parsed_ratio;
    if (ratio_stream.fail() || !ratio_stream.eof() ||
        !std::isfinite(parsed_ratio) || parsed_ratio < 0.0 ||
        parsed_ratio > 1.0) {
        LOG(ERROR) << Variables::MC_STORE_DFS_PREFETCH_CAPACITY_RATIO.name
                   << " must be a number in [0, 1], got " << prefetch_ratio;
    } else {
        config.prefetch_capacity_ratio = parsed_ratio;
    }
}

}  // namespace mooncake::l4
