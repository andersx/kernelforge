// FCHL18 Hessian kernel — symmetric full matrix variant
//
// kernel_gaussian_hessian_symm:
//   Computes d²K[a,b]/dR_a dR_b for all pairs (a,b) in the training set,
//   exploiting symmetry (only lower triangle b <= a computed, then mirrored).
//   Returns shape (D, D) where D = sum_i n_atoms_i * 3.

#include <cstring>
#include <stdexcept>
#include <vector>

#include "fchl18_kernel.hpp"
#include "fchl18_kernel_common.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// kernel_gaussian_hessian_symm
//
// coords_list : list of N arrays, each (n_atoms_i, 3), float64
// z_list      : list of N arrays, each (n_atoms_i,),   int32
//
// Returns ndarray shape (D, D) where D = sum_i n_atoms_i * 3.
//   H[row, col] = d²K[a,b] / dR_a[flat_row] dR_b[flat_col]
//
// Symmetry: H[row, col] == H[col, row].
// ---------------------------------------------------------------------------
py::array_t<double> kernel_gaussian_hessian_symm_py(
    const py::list &coords_list, const py::list &z_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm
) {
    const int nm = static_cast<int>(coords_list.size());
    if (static_cast<int>(z_list.size()) != nm)
        throw std::invalid_argument("coords_list and z_list must have the same length");
    if (nm == 0) throw std::invalid_argument("kernel_gaussian_hessian_symm: empty molecule list");

    std::vector<kf::fchl18::MolData> mols(nm);
    for (int a = 0; a < nm; ++a)
        mols[a] = kf::fchl18::parse_mol(coords_list[a], z_list[a]);

    int D = 0;
    for (int a = 0; a < nm; ++a)
        D += mols[static_cast<std::size_t>(a)].n_atoms * 3;

    py::array_t<double> H({(py::ssize_t)D, (py::ssize_t)D});

    std::vector<std::vector<double>> coords_vecs(static_cast<std::size_t>(nm));
    std::vector<std::vector<int>> z_vecs(static_cast<std::size_t>(nm));
    for (int a = 0; a < nm; ++a) {
        coords_vecs[static_cast<std::size_t>(a)] = mols[static_cast<std::size_t>(a)].coords;
        z_vecs[static_cast<std::size_t>(a)] = mols[static_cast<std::size_t>(a)].z;
    }

    {
        py::gil_scoped_release release;
        kf::fchl18::kernel_gaussian_hessian_symm_blocks(
            coords_vecs,
            z_vecs,
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
            H.mutable_data()
        );
    }

    return H;
}
