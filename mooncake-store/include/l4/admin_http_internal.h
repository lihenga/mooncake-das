// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "master_admin_service.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <ylt/reflection/user_reflect_macro.hpp>
#include <ylt/struct_json/json_reader.h>
#include <ylt/struct_json/json_writer.h>
#include <ylt/util/tl/expected.hpp>

#include "allocator.h"
#include "ha_metric_manager.h"
#include "master_metric_manager.h"
#include "runtime_log_config.h"
#include "rpc_service.h"
#include "types.h"

namespace mooncake::admin_detail {

inline std::string AppendMetricSections(std::string primary,
                                        std::string secondary) {
    if (!primary.empty() && primary.back() != '\n') {
        primary.push_back('\n');
    }
    primary += secondary;
    return primary;
}

struct HttpErrorResponse {
    bool success{false};
    int32_t error_code{0};
    std::string error_message;
};
YLT_REFL(HttpErrorResponse, success, error_code, error_message);

struct HttpSimpleErrorResponse {
    bool success{false};
    std::string error;
};
YLT_REFL(HttpSimpleErrorResponse, success, error);

template <typename T>
inline void WriteJsonResponse(coro_http::coro_http_response& resp,
                              coro_http::status_type status, const T& payload) {
    std::string json;
    struct_json::to_json(payload, json);
    resp.add_header("Content-Type", "application/json; charset=utf-8");
    resp.set_status_and_content(status, std::move(json));
}

template <typename T>
inline std::string EnumToString(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

inline coro_http::status_type ErrorCodeToHttpStatus(ErrorCode error) {
    switch (error) {
        case ErrorCode::INVALID_PARAMS:
            return coro_http::status_type::bad_request;
        case ErrorCode::JOB_NOT_FOUND:
        case ErrorCode::SEGMENT_NOT_FOUND:
        case ErrorCode::OBJECT_NOT_FOUND:
        case ErrorCode::TENANT_NOT_REGISTERED:
            return coro_http::status_type::not_found;
        case ErrorCode::UNAVAILABLE_IN_CURRENT_MODE:
        case ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS:
        case ErrorCode::TENANT_NOT_EMPTY:
            return coro_http::status_type::conflict;
        default:
            return coro_http::status_type::internal_server_error;
    }
}

inline void WriteErrorResponse(coro_http::coro_http_response& resp,
                               coro_http::status_type status, ErrorCode error,
                               std::string message = {}) {
    HttpErrorResponse payload;
    payload.error_code = toInt(error);
    payload.error_message =
        message.empty() ? toString(error) : std::move(message);
    WriteJsonResponse(resp, status, payload);
}

inline void WriteSimpleErrorResponse(coro_http::coro_http_response& resp,
                                     coro_http::status_type status,
                                     std::string error) {
    HttpSimpleErrorResponse payload;
    payload.error = std::move(error);
    WriteJsonResponse(resp, status, payload);
}

inline void SetServiceUnavailable(coro_http::coro_http_response& resp,
                                  std::string message) {
    WriteErrorResponse(resp, coro_http::status_type::service_unavailable,
                       ErrorCode::UNAVAILABLE_IN_CURRENT_MODE,
                       std::move(message));
}

inline tl::expected<UUID, ErrorCode> ParseJobId(std::string_view job_id_view) {
    if (job_id_view.empty()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    UUID job_id;
    if (!StringToUuid(std::string(job_id_view), job_id)) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    return job_id;
}

inline std::string EscapeJson(std::string_view input) {
    std::string escaped;
    escaped.reserve(input.size());
    for (char ch : input) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }
    return escaped;
}

inline std::string EscapePrometheusLabel(std::string_view input) {
    std::string escaped;
    escaped.reserve(input.size());
    for (char ch : input) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }
    return escaped;
}

struct HttpTenantQuotaSnapshot {
    std::string tenant_id;
    uint64_t requested_quota_bytes{0};
    uint64_t effective_quota_bytes{0};
    uint64_t charged_bytes{0};
    bool admission_closed{true};
    bool over_quota{false};
    bool has_explicit_policy{false};
};
YLT_REFL(HttpTenantQuotaSnapshot, tenant_id, requested_quota_bytes,
         effective_quota_bytes, charged_bytes, admission_closed, over_quota,
         has_explicit_policy);

inline HttpTenantQuotaSnapshot ToHttpTenantQuotaSnapshot(
    const TenantQuotaSnapshot& snapshot) {
    return HttpTenantQuotaSnapshot{
        .tenant_id = snapshot.tenant_id.value(),
        .requested_quota_bytes = snapshot.requested_quota_bytes,
        .effective_quota_bytes = snapshot.effective_quota_bytes,
        .charged_bytes = snapshot.charged_bytes,
        .admission_closed = snapshot.admission_closed,
        .over_quota = snapshot.over_quota,
        .has_explicit_policy = snapshot.has_explicit_policy,
    };
}

struct HttpTenantQuotaListResponse {
    bool success{true};
    std::vector<HttpTenantQuotaSnapshot> data;
};
YLT_REFL(HttpTenantQuotaListResponse, success, data);

struct HttpTenantQuotaResponse {
    bool success{true};
    HttpTenantQuotaSnapshot data;
};
YLT_REFL(HttpTenantQuotaResponse, success, data);

struct HttpTenantQuotaDeleteResponse {
    bool success{true};
    std::optional<HttpTenantQuotaSnapshot> data;
};
YLT_REFL(HttpTenantQuotaDeleteResponse, success, data);

struct HttpTenantQuotaPolicyRequest {
    uint64_t requested_quota_bytes{0};
};
YLT_REFL(HttpTenantQuotaPolicyRequest, requested_quota_bytes);

struct HttpDfsMaxBucketCountRequest {
    int64_t max_bucket_count{0};
};
YLT_REFL(HttpDfsMaxBucketCountRequest, max_bucket_count);

struct HttpDfsMaxBucketCountResponse {
    bool success{true};
    int64_t old_value{0};
    int64_t new_value{0};
};
YLT_REFL(HttpDfsMaxBucketCountResponse, success, old_value, new_value);

struct HttpDfsHeartbeatIntervalRequest {
    uint32_t interval_seconds{0};
};
YLT_REFL(HttpDfsHeartbeatIntervalRequest, interval_seconds);

struct HttpDfsHeartbeatIntervalResponse {
    bool success{true};
    uint32_t old_value{0};
    uint32_t new_value{0};
};
YLT_REFL(HttpDfsHeartbeatIntervalResponse, success, old_value, new_value);

struct HttpDfsBatchTimeoutFixedMsRequest {
    std::optional<uint64_t> timeout_ms;
};
YLT_REFL(HttpDfsBatchTimeoutFixedMsRequest, timeout_ms);

struct HttpDfsBatchTimeoutFixedMsResponse {
    bool success{true};
    uint64_t old_value{0};
    uint64_t new_value{0};
};
YLT_REFL(HttpDfsBatchTimeoutFixedMsResponse, success, old_value, new_value);

struct HttpDfsFilterModeRequest {
    std::optional<uint32_t> filter_mode;
};
YLT_REFL(HttpDfsFilterModeRequest, filter_mode);

struct HttpDfsFilterModeResponse {
    bool success{true};
    uint32_t old_value{0};
    uint32_t new_value{0};
};
YLT_REFL(HttpDfsFilterModeResponse, success, old_value, new_value);

struct HttpLoggingUpdateRequest {
    std::optional<std::string> minloglevel;
    std::optional<int> vlog_level;
    std::optional<std::string> ylt_severity;
    std::optional<bool> dfs_read_trace;
};
YLT_REFL(HttpLoggingUpdateRequest, minloglevel, vlog_level, ylt_severity,
         dfs_read_trace);

struct HttpLoggingResponse {
    bool success{true};
    std::string minloglevel;
    int vlog_level{0};
    std::string ylt_severity;
    bool dfs_read_trace{false};
};
YLT_REFL(HttpLoggingResponse, success, minloglevel, vlog_level, ylt_severity,
         dfs_read_trace);

inline void FillLoggingResponse(HttpLoggingResponse* response) {
    const auto config = GetRuntimeLogConfig();
    response->minloglevel = config.minloglevel;
    response->vlog_level = config.vlog_level;
    response->ylt_severity = config.ylt_severity;
    response->dfs_read_trace = config.dfs_read_trace;
}

inline tl::expected<std::string, ErrorCode> ParseAdminTenantId(
    coro_http::coro_http_request& req) {
    auto tenant_id_view = req.get_decode_query_value("tenant_id");
    if (tenant_id_view.empty()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    TenantId tenant_id{std::string(tenant_id_view)};
    if (!tenant_id.IsValid()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    return tenant_id.value();
}

inline tl::expected<HttpTenantQuotaPolicyRequest, std::string>
ParseQuotaPolicyBody(coro_http::coro_http_request& req) {
    HttpTenantQuotaPolicyRequest request;
    try {
        struct_json::from_json(request, req.get_body());
    } catch (const std::exception& e) {
        return tl::make_unexpected(std::string("Invalid JSON body: ") +
                                   e.what());
    }
    return request;
}

// Buckets are addressed through DistributedFSDescriptor::shard_idx, an int
// serialized as int32, so the count must stay within int32 range. Keep in
// sync with kMaxBucketId in storage/distributed/bucket_entry_layout.h.
constexpr int64_t kDfsMaxBucketCountLimit =
    static_cast<int64_t>(std::numeric_limits<int32_t>::max());

inline tl::expected<HttpDfsMaxBucketCountRequest, std::string>
ParseDfsMaxBucketCountBody(coro_http::coro_http_request& req) {
    HttpDfsMaxBucketCountRequest request;
    try {
        struct_json::from_json(request, req.get_body());
    } catch (const std::exception& e) {
        return tl::make_unexpected(std::string("Invalid JSON body: ") +
                                   e.what());
    }
    if (request.max_bucket_count <= 0) {
        return tl::make_unexpected("max_bucket_count must be positive (>0)");
    }
    if (request.max_bucket_count > kDfsMaxBucketCountLimit) {
        return tl::make_unexpected(
            "max_bucket_count exceeds maximum addressable bucket id (" +
            std::to_string(kDfsMaxBucketCountLimit) + ")");
    }
    return request;
}

inline tl::expected<HttpDfsHeartbeatIntervalRequest, std::string>
ParseDfsHeartbeatIntervalBody(coro_http::coro_http_request& req) {
    HttpDfsHeartbeatIntervalRequest request;
    try {
        struct_json::from_json(request, req.get_body());
    } catch (const std::exception& e) {
        return tl::make_unexpected(std::string("Invalid JSON body: ") +
                                   e.what());
    }
    if (request.interval_seconds == 0) {
        return tl::make_unexpected("interval_seconds must be positive (>0)");
    }
    return request;
}

inline tl::expected<HttpDfsBatchTimeoutFixedMsRequest, std::string>
ParseDfsBatchTimeoutFixedMsBody(coro_http::coro_http_request& req) {
    HttpDfsBatchTimeoutFixedMsRequest request;
    try {
        struct_json::from_json(request, req.get_body());
    } catch (const std::exception& e) {
        return tl::make_unexpected(std::string("Invalid JSON body: ") +
                                   e.what());
    }
    if (!request.timeout_ms.has_value()) {
        return tl::make_unexpected("timeout_ms is required");
    }
    return request;
}

inline tl::expected<HttpDfsFilterModeRequest, std::string>
ParseDfsFilterModeBody(coro_http::coro_http_request& req) {
    HttpDfsFilterModeRequest request;
    try {
        struct_json::from_json(request, req.get_body());
    } catch (const std::exception& e) {
        return tl::make_unexpected(std::string("Invalid JSON body: ") +
                                   e.what());
    }
    if (!request.filter_mode.has_value()) {
        return tl::make_unexpected("filter_mode is required");
    }
    if (*request.filter_mode > 1) {
        return tl::make_unexpected("filter_mode must be 0 or 1");
    }
    return request;
}

}  // namespace mooncake::admin_detail

namespace mooncake {

template <typename Handler>
void MasterAdminServer::WithActiveService(coro_http::coro_http_response& resp,
                                          Handler&& handler) const {
    auto service = GetActiveService();
    if (!service) {
        admin_detail::SetServiceUnavailable(resp,
                                            "service plane is not active");
        return;
    }
    std::forward<Handler>(handler)(service);
}

}  // namespace mooncake
