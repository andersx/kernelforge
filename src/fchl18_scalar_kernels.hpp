#pragma once
// FCHL18 scalar kernel variants not in fchl18_kernel.hpp.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Expose to module — implementation in fchl18_scalar_kernels.cpp
py::array_t<double> kernel_gaussian_symm_rfp_py(
    const py::list &coords_list,
    const py::list &z_list,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int    fourier_order,
    bool   use_atm
);
