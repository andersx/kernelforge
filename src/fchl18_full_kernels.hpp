#pragma once
// FCHL18 full combined energy+force kernels.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// kernel_gaussian_full: asymmetric combined kernel, shape (N_A*(1+D_A), N_B*(1+D_B)).
// Block layout:
//   [0:N_A,  0:N_B ]  scalar block    K[a,b]
//   [0:N_A,  N_B:  ]  jac_t block    dK/dR_B
//   [N_A:,   0:N_B ]  jac block      dK/dR_A
//   [N_A:,   N_B:  ]  hessian block  d²K/dR_A dR_B
py::array_t<double> kernel_gaussian_full_py(
    const py::list &coords_A_list,
    const py::list &z_A_list,
    const py::list &coords_B_list,
    const py::list &z_B_list,
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

// kernel_gaussian_full_symm: symmetric combined kernel (X==X training set).
// Shape (N*(1+D), N*(1+D)), lower triangle filled, upper mirrored.
py::array_t<double> kernel_gaussian_full_symm_py(
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

// kernel_gaussian_full_symm_rfp: symmetric combined kernel in RFP format.
// Output length = N*(1+D) * (N*(1+D)+1) / 2.
py::array_t<double> kernel_gaussian_full_symm_rfp_py(
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
