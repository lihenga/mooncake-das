#include "storage_backend.h"
#include <cstring>
namespace mooncake {
tl::expected<void, ErrorCode> StorageBackendInterface::BatchLoadSpans(
    std::unordered_map<std::string, StorageValueSpan>& spans) {
    bool partial = false;
    std::unordered_map<std::string, Slice> full;
    full.reserve(spans.size());
    for (const auto& [key, span] : spans) {
        if (span.offset != 0 ||
            (span.object_size != 0 && span.object_size != span.slice.size)) {
            partial = true;
            break;
        }
        full.emplace(key, span.slice);
    }
    if (!partial) {
        auto loaded = BatchLoad(full);
        if (!loaded) return loaded;
        for (auto& [key, slice] : full) {
            spans.at(key).slice.ptr = slice.ptr;
        }
        return {};
    }

    // Backends without a seek override still return the requested window.
    // They read the whole value first; bucket and offset-allocator overrides
    // avoid that.
    for (auto& [key, span] : spans) {
        if (span.slice.ptr == nullptr || span.object_size == 0 ||
            span.offset > span.object_size ||
            span.slice.size > span.object_size - span.offset) {
            LOG(ERROR) << "Invalid value span for key: " << key;
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        std::vector<char> object(static_cast<size_t>(span.object_size));
        std::unordered_map<std::string, Slice> one{
            {key, Slice{object.data(), object.size()}}};
        auto loaded = BatchLoad(one);
        if (!loaded) return loaded;
        std::memcpy(span.slice.ptr,
                    static_cast<const char*>(one.at(key).ptr) + span.offset,
                    span.slice.size);
    }
    return {};
}
}  // namespace mooncake
