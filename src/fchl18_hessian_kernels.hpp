#pragma once
// FCHL18 Hessian kernel — symmetric full matrix variant.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// kernel_gaussian_hessian_symm:
//   Shape (N*D, N*D) where D = sum_i n_atoms_i * 3.
//   Same as kernel_gaussian_hessian(mols, mols, ...) but exploits symmetry:
//   only lower triangle blocks b <= a are computed, then mirrored.
py::array_t<double> kernel_gaussian_hessian_symm_py(
    const py::list &coords_list, const py::list &z_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm
);
