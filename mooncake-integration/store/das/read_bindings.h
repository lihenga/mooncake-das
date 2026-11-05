// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <pybind11/pybind11.h>

namespace mooncake::das {
void RegisterReadTypes(pybind11::module_& m);
}
