// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glog/logging.h>
#include "read_layout.h"
#include "read_plan.h"
#include "types.h"
#include "version.h"

namespace mooncake::das {

namespace py = pybind11;

template <typename Store>
void RegisterStoreExtensions(py::class_<Store> &store) {
    store
        .def(
            "get_capabilities",
            [](const Store &self) {
                // Local API support only. DummyClient forwards health/metrics
                // RPCs to a daemon built from the same Mooncake package.
                const bool embedded = !self.use_dummy_client_;
                py::dict features;
                features["dfs_health"] = true;
                features["dfs_prefetch"] = embedded;
                features["read_plan"] = embedded;
                features["read_layout"] = embedded;
                features["service_metrics"] = true;
                features["client_http_auto_port"] = true;
                py::dict result;
                result["api_version"] = 1;
                result["scope"] = "local_package";
                result["client_mode"] = embedded ? "embedded" : "standalone";
                result["rpc_protocol_version"] = GetMooncakeStoreVersion();
                result["features"] = std::move(features);
                result["setup_parameters"] =
                    std::vector<std::string>{"local_hostname",
                                             "metadata_server",
                                             "global_segment_size",
                                             "local_buffer_size",
                                             "protocol",
                                             "rdma_devices",
                                             "master_server_addr",
                                             "engine",
                                             "enable_ssd_offload",
                                             "ssd_offload_path",
                                             "tenant_id",
                                             "enable_client_http_server",
                                             "client_http_port",
                                             "enable_dfs_prefetch"};
                return result;
            },
            "Describe optional local APIs and setup parameters without an RPC. "
            "Callable before setup; does not probe a standalone daemon. "
            "Legacy callers do not need to call this method.")
        .def("probe_dfs_health", &Store::probe_dfs_health,
             "Probe DFS read/write health. Returns "
             "[read_error_code, write_error_code, backend_ready].")
        .def("probe_dfs_read", &Store::probe_dfs_read,
             "Probe DFS read capability and return an error code.")
        .def("probe_dfs_write", &Store::probe_dfs_write,
             "Probe DFS write capability and return an error code.")
        .def(
            "set_service_endpoint",
            [](Store &self, const std::string &endpoint) {
                if (!self.store_) return toInt(ErrorCode::INVALID_PARAMS);
                py::gil_scoped_release release;
                return self.store_->set_service_endpoint(endpoint);
            },
            py::arg("endpoint"),
            "Enable master-reported service metrics for this immutable service "
            "endpoint.")
        .def("set_dfs_access_policy", &Store::set_dfs_access_policy,
             py::arg("allow_read"), py::arg("allow_write"),
             "Atomically update DFS read/write access policy.")
        .def(
            "batch_get_into_layouts",
            [](Store &self,
               const std::vector<std::shared_ptr<ReadLayoutDescriptor>>
                   &layouts) {
                if (!self.is_client_initialized())
                    throw std::runtime_error("Client is not initialized");
                auto client = self.store_;
                py::gil_scoped_release release;
                return mooncake::ReadIntoLayouts(*client, layouts);
            },
            py::arg("layouts"),
            "Ranged get into strided buffers defined by read layouts, using "
            "active get sessions (no Master RPC).")
        .def(
            "create_read_plan",
            [](Store &self, std::vector<mooncake::ReadLayout> layouts,
               int num_groups, bool reuse_ranges, py::object buffer_owners,
               bool page_wise, bool borrowed_sessions) {
                if (!self.is_client_initialized())
                    throw std::runtime_error("Client is not initialized");
                std::shared_ptr<mooncake::ReadPlan> plan;
                {
                    py::gil_scoped_release release;
                    plan = std::make_shared<mooncake::ReadPlan>(
                        self.store_, std::move(layouts), num_groups,
                        reuse_ranges, page_wise, borrowed_sessions);
                }
                auto object = py::cast(plan);
                object.attr("_buffer_owners") = std::move(buffer_owners);
                return object;
            },
            py::arg("layouts"), py::arg("num_groups"),
            py::arg("reuse_ranges") = false,
            py::arg("buffer_owners") = py::none(), py::arg("page_wise") = false,
            py::arg("borrowed_sessions") = false, py::keep_alive<0, 1>(),
            "Create ordered range reads; retain registered destination memory "
            "until run completes. By default the plan starts and ends get "
            "sessions; borrowed_sessions=True requires the caller to keep "
            "sessions for all keys active until run completes.")
        .def(
            "batch_get_session_refresh",
            [](Store &self, const std::vector<std::string> &keys) {
                if (!self.is_client_initialized()) {
                    LOG(ERROR) << "Client is not initialized";
                    return std::vector<int>{};
                }
                py::gil_scoped_release release;
                return self.store_->batch_get_session_refresh(keys);
            },
            py::arg("keys"),
            "Refresh leases for active get sessions while preserving cached "
            "session objects; incompatible refreshed replicas fail and "
            "invalidate that session. Expiry returns "
            "SESSION_REFRESH_QUERY_LEASE_EXPIRED (-717) before selection or "
            "SESSION_REFRESH_COMMIT_LEASE_EXPIRED (-718) before commit.")
        .def(
            "batch_get_session_start_with_sources",
            [](Store &self, const std::vector<std::string> &keys) {
                if (!self.is_client_initialized()) {
                    LOG(ERROR) << "Client is not initialized";
                    return std::pair<std::vector<int>,
                                     std::vector<std::string>>{};
                }
                py::gil_scoped_release release;
                auto result =
                    self.store_->batch_get_session_start_with_sources(keys);
                return std::make_pair(std::move(result.codes),
                                      std::move(result.sources));
            },
            py::arg("keys"),
            "Start get sessions and return each result with its selected "
            "source; expired queries return SESSION_START_LEASE_EXPIRED "
            "(-716).")
        .def(
            "batch_get_session_prefetch",
            [](Store &self, const std::vector<std::string> &keys) {
                if (!self.is_client_initialized()) {
                    LOG(ERROR) << "Client is not initialized";
                    return std::vector<int>{};
                }
                py::gil_scoped_release release;
                return self.store_->batch_get_session_prefetch(keys);
            },
            py::arg("keys"),
            "Synchronously read DFS objects in active get sessions into "
            "Mooncake's dedicated prefetch arena without copying to caller "
            "buffers. A successful DFS status means the complete object bytes "
            "remain owned by a compatible active session; non-DFS entries use "
            "the ordinary range-get path. Returns per-key status; "
            "SESSION_PREFETCH_LEASE_EXPIRED (-719) checks entry only. "
            "Successful validated BUCKET reads may finish after lease expiry; "
            "range-get still requires a live or refreshed lease.")
        .def(
            "dfs_prefetch_arena_available",
            [](Store &self) {
                if (!self.is_client_initialized()) return false;
                return self.store_->dfs_prefetch_arena_available();
            },
            "Whether batch_get_session_prefetch has a DFS staging arena "
            "(MC_STORE_DFS_PREFETCH_CAPACITY_RATIO). When false every DFS "
            "prefetch read fails, so callers should not submit prefetches.")
        .def(
            "disable_dfs_prefetch",
            [](Store &self) {
                if (!self.is_client_initialized()) return false;
                py::gil_scoped_release release;
                return self.store_->disable_dfs_prefetch();
            },
            "Startup-only: disable DFS prefetch and release its unused arena. "
            "Returns false if a session or arena allocation is active.")
        .def(
            "dfs_prefetch_arena_status",
            [](Store &self) {
                if (!self.is_client_initialized()) {
                    return std::string("client not initialized");
                }
                return self.store_->dfs_prefetch_arena_status();
            },
            "\"ready\" or the reason the DFS prefetch arena is unavailable.")
        .def(
            "record_prefetched_tokens",
            [](Store &self, uint64_t tokens) {
                if (!self.is_client_initialized()) return;
                py::gil_scoped_release release;
                self.store_->record_prefetched_tokens(tokens);
            },
            py::arg("tokens"),
            "Record prompt tokens successfully prefetched by the direct "
            "linker");
}

}  // namespace mooncake::das
