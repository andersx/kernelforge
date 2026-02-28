#pragma once
// FCHL18 Jacobian kernel (dK/dR_A).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// kernel_gaussian_jacobian:
//   Shape (N_A*D_A, N_B) where D_A = n_atoms_A * 3.
//   Takes raw coords+charges for query molecules A, pre-computed repr for training B.
py::array_t<double> kernel_gaussian_jacobian_py(
    const py::list &coords_A_list,
    const py::list &z_A_list,
    const py::array_t<double,  py::array::c_style | py::array::forcecast> &x2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &nn2,
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
