// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "read_bindings.h"
#include <pybind11/stl.h>
#include "read_layout.h"
#include "read_plan.h"

namespace py = pybind11;
namespace mooncake::das {

void RegisterReadTypes(py::module_& m) {
    using mooncake::ReadKeyMapping;
    using mooncake::ReadLayoutComponent;
    using mooncake::ReadLayoutDescriptor;
    auto layout_integer = [](py::handle value) {
        if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr()))
            throw py::type_error("layout fields and rows require integers");
        const int negative =
            PyObject_RichCompareBool(value.ptr(), py::int_(0).ptr(), Py_LT);
        if (negative < 0) throw py::error_already_set();
        if (negative)
            throw py::value_error("layout fields and rows must be nonnegative");
        const size_t result = PyLong_AsSize_t(value.ptr());
        if (PyErr_Occurred()) throw py::error_already_set();
        return result;
    };
    py::enum_<ReadKeyMapping>(m, "ReadKeyMapping")
        .value("PER_ROW", ReadKeyMapping::PER_ROW)
        .value("PER_ROW_COMPONENT", ReadKeyMapping::PER_ROW_COMPONENT);
    py::class_<ReadLayoutComponent>(m, "ReadComponent")
        .def(py::init([layout_integer](py::object base, py::object stride,
                                       py::object bytes, py::object offset) {
                 return ReadLayoutComponent(
                     layout_integer(base), layout_integer(stride),
                     layout_integer(bytes), layout_integer(offset));
             }),
             py::kw_only(), py::arg("destination_base"),
             py::arg("destination_stride"), py::arg("byte_count"),
             py::arg("source_offset") = py::int_(0))
        .def_readonly("destination_base",
                      &ReadLayoutComponent::destination_base)
        .def_readonly("destination_stride",
                      &ReadLayoutComponent::destination_stride)
        .def_readonly("byte_count", &ReadLayoutComponent::byte_count)
        .def_readonly("source_offset", &ReadLayoutComponent::source_offset);
    py::class_<ReadLayoutDescriptor, std::shared_ptr<ReadLayoutDescriptor>>(
        m, "ReadLayout")
        .def(
            py::init([layout_integer](
                         std::vector<std::string> keys, py::sequence rows,
                         py::object mapping,
                         std::vector<ReadLayoutComponent> components) {
                if (!py::isinstance<ReadKeyMapping>(mapping))
                    throw py::type_error("key_mapping requires ReadKeyMapping");
                std::vector<size_t> indices;
                indices.reserve(py::len(rows));
                for (py::handle row : rows)
                    indices.push_back(layout_integer(row));
                return std::make_shared<ReadLayoutDescriptor>(
                    std::move(keys), std::move(indices),
                    mapping.cast<ReadKeyMapping>(), std::move(components));
            }),
            py::kw_only(), py::arg("keys"), py::arg("rows"),
            py::arg("key_mapping"), py::arg("components"))
        .def_readonly("keys", &ReadLayoutDescriptor::keys)
        .def_readonly("rows", &ReadLayoutDescriptor::rows)
        .def_readonly("key_mapping", &ReadLayoutDescriptor::key_mapping)
        .def_readonly("components", &ReadLayoutDescriptor::components);
    py::class_<mooncake::ReadPlan, std::shared_ptr<mooncake::ReadPlan>>(
        m, "ReadPlan", py::dynamic_attr())
        .def("run", &mooncake::ReadPlan::run,
             py::call_guard<py::gil_scoped_release>())
        .def("wait", &mooncake::ReadPlan::wait, py::arg("group"),
             py::call_guard<py::gil_scoped_release>())
        .def("stats", &mooncake::ReadPlan::stats)
        .def("reuse_stats", &mooncake::ReadPlan::reuse_stats);
}

}  // namespace mooncake::das
