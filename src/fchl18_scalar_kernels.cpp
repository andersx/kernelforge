// FCHL18 scalar kernel — RFP variant
//
// kernel_gaussian_symm_rfp: takes raw coords+charges lists, generates FCHL18
// representations internally, computes the symmetric scalar kernel packed into
// RFP format.  Output length = N*(N+1)/2, compatible with cho_solve_rfp.

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "fchl18_kernel.hpp"
#include "fchl18_kernel_common.hpp"
#include "fchl18_repr.hpp"
#include "rfp_utils.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// kernel_gaussian_symm_rfp
//
// Takes raw coords+charges for N molecules, generates FCHL18 representations
// internally, then computes the symmetric scalar kernel packed into RFP format.
//
// Returns ndarray shape (N*(N+1)//2,), upper-triangle RFP (TRANSR='N', UPLO='U').
// ---------------------------------------------------------------------------
py::array_t<double> kernel_gaussian_symm_rfp_py(
    const py::list &coords_list, const py::list &z_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm
) {
    const int nm = static_cast<int>(coords_list.size());
    if (static_cast<int>(z_list.size()) != nm)
        throw std::invalid_argument("coords_list and z_list must have the same length");
    if (nm == 0) throw std::invalid_argument("kernel_gaussian_symm_rfp: empty molecule list");

    // Parse all molecules
    std::vector<kf::fchl18::MolData> mols(nm);
    for (int a = 0; a < nm; ++a)
        mols[a] = kf::fchl18::parse_mol(coords_list[a], z_list[a]);

    // Find max_size (max atom count across molecules)
    int max_size = 0;
    for (int a = 0; a < nm; ++a)
        max_size = std::max(max_size, mols[a].n_atoms);

    // Generate representation for all molecules:
    //   x_all  : (nm, max_size, 5, max_size)
    //   n_all  : (nm,)
    //   nn_all : (nm, max_size)
    const std::size_t repr_stride = static_cast<std::size_t>(max_size) * 5 * max_size;
    std::vector<double> x_all(static_cast<std::size_t>(nm) * repr_stride, 1e100);
    std::vector<int> n_all(nm, 0);
    std::vector<int> nn_all(static_cast<std::size_t>(nm) * max_size, 0);

    for (int a = 0; a < nm; ++a) {
        std::vector<double> x_mol;
        std::vector<int> nn_mol;
        kf::fchl18::generate_fchl18(
            mols[a].coords,
            mols[a].z,
            max_size,
            cut_distance,
            x_mol,
            nn_mol
        );
        n_all[a] = mols[a].n_atoms;
        // Copy into stacked arrays
        std::copy(
            x_mol.begin(),
            x_mol.end(),
            x_all.begin() + static_cast<std::ptrdiff_t>(a) * repr_stride
        );
        for (int i = 0; i < max_size; ++i)
            nn_all[static_cast<std::size_t>(a) * max_size + i] = nn_mol[i];
    }

    // Allocate full (nm x nm) temporary matrix for kernel_gaussian_symm
    const std::size_t N = static_cast<std::size_t>(nm);
    std::vector<double> K_full(N * N, 0.0);

    {
        py::gil_scoped_release release;

        kf::fchl18::kernel_gaussian_symm(
            x_all,
            n_all,
            nn_all,
            nm,
            max_size,
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
            K_full.data()
        );
    }

    // Allocate RFP output and pack upper triangle
    const std::size_t rfp_len = N * (N + 1) / 2;
    py::array_t<double> K_rfp(static_cast<py::ssize_t>(rfp_len));
    {
        double *rfp_ptr = K_rfp.mutable_data();
        // kernel_gaussian_symm fills the full matrix (lower + upper).
        // We read the lower triangle (b <= a) and write to RFP upper convention:
        //   rfp_index_upper_N(col=b, row=a) with b <= a
        for (std::size_t a = 0; a < N; ++a) {
            for (std::size_t b = 0; b <= a; ++b) {
                rfp_ptr[kf::rfp_index_upper_N(N, b, a)] = K_full[a * N + b];
            }
        }
    }

    return K_rfp;
}
