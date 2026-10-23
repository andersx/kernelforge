// FCHL18 Jacobian kernel — dK/dR_A
//
// kernel_gaussian_jacobian:
//   Shape (D_A, N_B) where D_A = sum_i n_atoms_i * 3.
//
//   Takes raw coords+charges for query molecules A + pre-computed repr for B.
//   Delegates to kf::fchl18::kernel_gaussian_jacobian which hoists B-side
//   precompute across all A molecules.

#include "fchl18_kernel.hpp"
#include "fchl18_kernel_common.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

static_assert(sizeof(int) == sizeof(int32_t), "int must be 32-bit for zero-copy int32 arrays");

static inline const int *i32_data(
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &arr
) {
    return reinterpret_cast<const int *>(arr.data());
}

// ---------------------------------------------------------------------------
// kernel_gaussian_jacobian:
//
// coords_A_list : list of N_A arrays, each (n_atoms_i, 3), float64
// z_A_list      : list of N_A arrays, each (n_atoms_i,),   int32
// x2            : (N_B, max_size2, 5, max_size2) float64 — pre-computed repr
// n2            : (N_B,) int32
// nn2           : (N_B, max_size2) int32
//
// Returns ndarray shape (D_A, N_B), float64
//   D_A = sum_i n_atoms_i * 3
//   J[row, b] = dK(A_i, B_b) / dR_{A_i}[flat_coord]
//   where row = row_offset[i] + atom_alpha * 3 + mu
// ---------------------------------------------------------------------------
py::array_t<double> kernel_gaussian_jacobian_py(
    const py::list &coords_A_list, const py::list &z_A_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &x2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &nn2, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm,
    bool use_alchemy, double alchemy_period_width, double alchemy_group_width
) {
    if (x2.ndim() != 4 || x2.shape(2) != 5)
        throw std::invalid_argument("x2 must be 4-D with shape (N_B, max_size, 5, max_size)");

    const int N_A = static_cast<int>(coords_A_list.size());
    const int N_B = static_cast<int>(x2.shape(0));
    const int max_size2 = static_cast<int>(x2.shape(1));

    if (static_cast<int>(z_A_list.size()) != N_A)
        throw std::invalid_argument("coords_A_list and z_A_list must have the same length");
    if (N_A == 0) throw std::invalid_argument("kernel_gaussian_jacobian: empty query set");

    // Parse A molecules
    std::vector<std::vector<double>> coords_A(static_cast<std::size_t>(N_A));
    std::vector<std::vector<int>> z_A(static_cast<std::size_t>(N_A));
    int D_A = 0;
    for (int i = 0; i < N_A; ++i) {
        kf::fchl18::MolData mi = kf::fchl18::parse_mol(coords_A_list[i], z_A_list[i]);
        D_A += mi.n_atoms * 3;
        coords_A[static_cast<std::size_t>(i)] = std::move(mi.coords);
        z_A[static_cast<std::size_t>(i)] = std::move(mi.z);
    }

    py::array_t<double> J({(py::ssize_t)D_A, (py::ssize_t)N_B});

    {
        py::gil_scoped_release release;
        kf::fchl18::kernel_gaussian_jacobian(
            coords_A,
            z_A,
            x2.data(),
            i32_data(n2),
            i32_data(nn2),
            N_B,
            max_size2,
            sigma,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm,
            J.mutable_data(),
            use_alchemy,
            alchemy_period_width,
            alchemy_group_width
        );
    }

    return J;
}
