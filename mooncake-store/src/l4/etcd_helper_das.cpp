// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "etcd_helper.h"

#ifdef STORE_USE_ETCD
#include "libetcd_wrapper.h"
#endif

#include <glog/logging.h>

namespace mooncake {
#ifdef STORE_USE_ETCD
ErrorCode EtcdHelper::TxnCompareAndPutWithLease(
    const std::vector<TxnCompare>& compares, const std::vector<TxnPut>& puts,
    EtcdLeaseId lease_id) {
    std::vector<char*> compare_keys;
    std::vector<int> compare_key_sizes;
    std::vector<int> compare_kinds;
    std::vector<char*> compare_values;
    std::vector<int> compare_value_sizes;
    for (const auto& compare : compares) {
        compare_keys.push_back(const_cast<char*>(compare.key.data()));
        compare_key_sizes.push_back(static_cast<int>(compare.key.size()));
        compare_kinds.push_back(static_cast<int>(compare.kind));
        compare_values.push_back(
            const_cast<char*>(compare.expected_value.data()));
        compare_value_sizes.push_back(
            static_cast<int>(compare.expected_value.size()));
    }
    std::vector<char*> put_keys;
    std::vector<int> put_key_sizes;
    std::vector<char*> put_values;
    std::vector<int> put_value_sizes;
    for (const auto& put : puts) {
        put_keys.push_back(const_cast<char*>(put.key.data()));
        put_key_sizes.push_back(static_cast<int>(put.key.size()));
        put_values.push_back(const_cast<char*>(put.value.data()));
        put_value_sizes.push_back(static_cast<int>(put.value.size()));
    }
    char* err_msg = nullptr;
    int ret = EtcdStoreTxnCompareAndPutWithLeaseWrapper(
        compare_keys.data(), compare_key_sizes.data(), compare_kinds.data(),
        compare_values.data(), compare_value_sizes.data(),
        static_cast<int>(compares.size()), put_keys.data(),
        put_key_sizes.data(), put_values.data(), put_value_sizes.data(),
        static_cast<int>(puts.size()), static_cast<int64_t>(lease_id),
        &err_msg);
    if (ret == -2) {
        if (err_msg != nullptr) free(err_msg);
        return ErrorCode::ETCD_TRANSACTION_FAIL;
    }
    if (ret != 0) {
        LOG(ERROR) << "TxnCompareAndPutWithLease failed: "
                   << (err_msg == nullptr ? "" : err_msg);
        if (err_msg != nullptr) free(err_msg);
        return ErrorCode::ETCD_OPERATION_ERROR;
    }
    return ErrorCode::OK;
}
#else
ErrorCode EtcdHelper::TxnCompareAndPutWithLease(
    const std::vector<TxnCompare>& compares, const std::vector<TxnPut>& puts,
    EtcdLeaseId lease_id) {
    (void)compares;
    (void)puts;
    (void)lease_id;
    LOG(FATAL) << "Etcd is not enabled in compilation";
    return ErrorCode::ETCD_OPERATION_ERROR;
}
#endif
}  // namespace mooncake
