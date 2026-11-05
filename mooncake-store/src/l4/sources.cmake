# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

list(APPEND MOONCAKE_STORE_SHARED_SOURCES l4/utils_l4.cpp
     l4/etcd_helper_das.cpp l4/distributed_storage_backend_l4.cpp
     l4/posix_fs_adapter_l4.cpp)

list(
  APPEND
  MOONCAKE_STORE_MASTER_SOURCES
  l4/master_service_l4.cpp
  l4/master_metric_manager_l4.cpp
  l4/segment_metrics.cpp
  l4/rpc_service_l4.cpp
  l4/master_admin_service_l4.cpp
  l4/dfs_global_allocator_l4.cpp
  l4/etcd_service_ready.cpp)

list(
  APPEND
  MOONCAKE_STORE_CLIENT_SOURCES
  l4/client_service_l4.cpp
  l4/master_client_l4.cpp
  l4/client_buffer_l4.cpp
  l4/pinned_buffer_pool_l4.cpp
  l4/real_client_l4.cpp
  l4/dummy_client_l4.cpp
  l4/file_storage_l4.cpp
  l4/file_storage_config.cpp
  l4/device_copy.cpp)

if(USE_3FS)
  list(APPEND MOONCAKE_STORE_SHARED_SOURCES l4/hf3fs_adapter_l4.cpp)
endif()
