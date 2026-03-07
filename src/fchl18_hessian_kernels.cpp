// FCHL18 Hessian kernel — symmetric full matrix variant
//
// kernel_gaussian_hessian_symm:
//   Computes d²K[a,b]/dR_a dR_b for all pairs (a,b) in the training set,
//   exploiting symmetry (only lower triangle b <= a computed, then mirrored).
//   Returns shape (D, D) where D = sum_i n_atoms_i * 3.

#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

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

    // Parse molecules
    std::vector<kf::fchl18::MolData> mols(nm);
    for (int a = 0; a < nm; ++a)
        mols[a] = kf::fchl18::parse_mol(coords_list[a], z_list[a]);

    // Compute offsets: offset[i] = sum_{j<i} n_atoms_j * 3
    std::vector<int> offset(nm + 1, 0);
    for (int a = 0; a < nm; ++a)
        offset[a + 1] = offset[a] + mols[a].n_atoms * 3;
    const int D = offset[nm];  // total flattened coord dim

    // Allocate output (D, D), zero-initialised
    py::array_t<double> H({(py::ssize_t)D, (py::ssize_t)D});
    std::memset(H.mutable_data(), 0, sizeof(double) * D * D);

    {
        py::gil_scoped_release release;

// Loop over lower triangle of molecule blocks: b <= a.
// Each (a,b) block is disjoint in output rows/cols, safe to parallelise.
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int a = 0; a < nm; ++a) {
            for (int b = 0; b < nm; ++b) {
                if (b > a) continue;

                const kf::fchl18::MolData &ma = mols[a];
                const int na3A = ma.n_atoms * 3;
                const int r0 = offset[a];
                const kf::fchl18::MolData &mb = mols[b];
                const int na3B = mb.n_atoms * 3;
                const int c0 = offset[b];

                // Compute hessian block H[a,b]: shape (na3A, na3B)
                std::vector<double> block(static_cast<std::size_t>(na3A) * na3B, 0.0);
                kf::fchl18::kernel_gaussian_hessian(
                    ma.coords,
                    ma.z,
                    mb.coords,
                    mb.z,
                    ma.n_atoms,
                    mb.n_atoms,
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
                    block.data()
                );

                double *H_ptr = H.mutable_data();

                if (a == b) {
                    // Diagonal block: symmetrize in-block, write both triangles.
                    for (int amu = 0; amu < na3A; ++amu) {
                        for (int bnu = 0; bnu < na3B; ++bnu) {
                            double v;
                            if (amu == bnu) {
                                v = block[static_cast<std::size_t>(amu) * na3B + bnu];
                            } else {
                                v = 0.5 * (block[static_cast<std::size_t>(amu) * na3B + bnu] +
                                           block[static_cast<std::size_t>(bnu) * na3B + amu]);
                            }
                            H_ptr[(r0 + amu) * D + (c0 + bnu)] = v;
                            H_ptr[(c0 + bnu) * D + (r0 + amu)] = v;
                        }
                    }
                } else {
                    // Off-diagonal block (a > b): fill block and its transpose.
                    // H[a*block, b*block] = block[amu, bnu]
                    // H[b*block, a*block] = block[amu, bnu]^T
                    for (int amu = 0; amu < na3A; ++amu) {
                        for (int bnu = 0; bnu < na3B; ++bnu) {
                            const double v = block[static_cast<std::size_t>(amu) * na3B + bnu];
                            H_ptr[(r0 + amu) * D + (c0 + bnu)] = v;
                            H_ptr[(c0 + bnu) * D + (r0 + amu)] = v;
                        }
                    }
                }
            }
        }
    }

    return H;
}
