// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "master_admin_service.h"
#include "l4/admin_http_internal.h"

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

namespace mooncake {

using namespace admin_detail;

void MasterAdminServer::HandleSetDfsMaxBucketCount(
    coro_http::coro_http_request& req, coro_http::coro_http_response& resp) {
    auto body_result = ParseDfsMaxBucketCountBody(req);
    if (!body_result.has_value()) {
        WriteErrorResponse(resp, coro_http::status_type::bad_request,
                           ErrorCode::INVALID_PARAMS, body_result.error());
        return;
    }

    WithActiveService(resp, [&](auto service) {
        auto result =
            service->SetDfsMaxBucketCount(body_result->max_bucket_count);
        if (!result.has_value()) {
            WriteErrorResponse(resp, ErrorCodeToHttpStatus(result.error()),
                               result.error());
            return;
        }
        HttpDfsMaxBucketCountResponse response;
        response.success = true;
        response.old_value = result.value();
        response.new_value = body_result->max_bucket_count;
        WriteJsonResponse(resp, coro_http::status_type::ok, response);
    });
}

void MasterAdminServer::HandleSetDfsHeartbeatInterval(
    coro_http::coro_http_request& req, coro_http::coro_http_response& resp) {
    auto body_result = ParseDfsHeartbeatIntervalBody(req);
    if (!body_result.has_value()) {
        WriteErrorResponse(resp, coro_http::status_type::bad_request,
                           ErrorCode::INVALID_PARAMS, body_result.error());
        return;
    }

    WithActiveService(resp, [&](auto service) {
        auto result =
            service->SetDfsHeartbeatInterval(body_result->interval_seconds);
        if (!result.has_value()) {
            WriteErrorResponse(resp, ErrorCodeToHttpStatus(result.error()),
                               result.error());
            return;
        }
        HttpDfsHeartbeatIntervalResponse response;
        response.old_value = result.value();
        response.new_value = body_result->interval_seconds;
        WriteJsonResponse(resp, coro_http::status_type::ok, response);
    });
}

void MasterAdminServer::HandleSetDfsBatchTimeoutFixedMs(
    coro_http::coro_http_request& req, coro_http::coro_http_response& resp) {
    auto body_result = ParseDfsBatchTimeoutFixedMsBody(req);
    if (!body_result.has_value()) {
        WriteErrorResponse(resp, coro_http::status_type::bad_request,
                           ErrorCode::INVALID_PARAMS, body_result.error());
        return;
    }

    WithActiveService(resp, [&](auto service) {
        HttpDfsBatchTimeoutFixedMsResponse response;
        response.old_value =
            service->SetDfsBatchTimeoutFixedMs(*body_result->timeout_ms);
        response.new_value = *body_result->timeout_ms;
        WriteJsonResponse(resp, coro_http::status_type::ok, response);
    });
}

void MasterAdminServer::HandleGetDfsFilterMode(
    coro_http::coro_http_request&, coro_http::coro_http_response& resp) {
    WithActiveService(resp, [&](auto service) {
        const auto current = service->GetDfsFilterMode();
        WriteJsonResponse(resp, coro_http::status_type::ok,
                          HttpDfsFilterModeResponse{.old_value = current,
                                                    .new_value = current});
    });
}

void MasterAdminServer::HandleSetDfsFilterMode(
    coro_http::coro_http_request& req, coro_http::coro_http_response& resp) {
    auto body_result = ParseDfsFilterModeBody(req);
    if (!body_result.has_value()) {
        WriteErrorResponse(resp, coro_http::status_type::bad_request,
                           ErrorCode::INVALID_PARAMS, body_result.error());
        return;
    }

    WithActiveService(resp, [&](auto service) {
        auto result = service->SetDfsFilterMode(*body_result->filter_mode);
        if (!result.has_value()) {
            WriteErrorResponse(resp, ErrorCodeToHttpStatus(result.error()),
                               result.error());
            return;
        }
        WriteJsonResponse(
            resp, coro_http::status_type::ok,
            HttpDfsFilterModeResponse{.old_value = result.value(),
                                      .new_value = *body_result->filter_mode});
    });
}

void MasterAdminServer::HandleGetLogging(coro_http::coro_http_request&,
                                         coro_http::coro_http_response& resp) {
    HttpLoggingResponse response;
    FillLoggingResponse(&response);
    WriteJsonResponse(resp, coro_http::status_type::ok, response);
}

void MasterAdminServer::HandleUpdateLogging(
    coro_http::coro_http_request& req, coro_http::coro_http_response& resp) {
    HttpLoggingUpdateRequest request;
    try {
        struct_json::from_json(request, req.get_body());
    } catch (const std::exception& e) {
        WriteSimpleErrorResponse(resp, coro_http::status_type::bad_request,
                                 std::string("Invalid JSON body: ") + e.what());
        return;
    }

    RuntimeLogConfigUpdate update{
        .minloglevel = request.minloglevel,
        .vlog_level = request.vlog_level,
        .ylt_severity = request.ylt_severity,
        .dfs_read_trace = request.dfs_read_trace,
    };
    std::string error;
    if (!UpdateRuntimeLogConfig(update, &error)) {
        WriteSimpleErrorResponse(resp, coro_http::status_type::bad_request,
                                 std::move(error));
        return;
    }

    HttpLoggingResponse response;
    FillLoggingResponse(&response);
    WriteJsonResponse(resp, coro_http::status_type::ok, response);
}

}  // namespace mooncake
