// FCHL18 full combined energy+force kernels.
//
// Three variants:
//   kernel_gaussian_full          — asymmetric (N_A*(1+D_A), N_B*(1+D_B))
//   kernel_gaussian_full_symm     — symmetric  (N*(1+D), N*(1+D)), full matrix
//   kernel_gaussian_full_symm_rfp — symmetric  RFP packed, length N*(1+D)*(N*(1+D)+1)/2
//
// Block layout for the full kernel:
//   [0:N_A,   0:N_B ]  scalar   K[a,b]
//   [0:N_A,   N_B:  ]  jac_t    dK/dR_B (= kernel_gradient w.r.t. B)
//   [N_A:,    0:N_B ]  jac      dK/dR_A (= kernel_gradient w.r.t. A)
//   [N_A:,    N_B:  ]  hessian  d²K/dR_A dR_B
//
// Row ordering (for asymmetric):
//   rows [0 : N_A]              — molecules 0..N_A-1 (scalar energy block)
//   rows [N_A : N_A + N_A*D_A] — atoms, interleaved: mol 0 first (jac/hessian rows)
//   cols [0 : N_B]              — molecules (scalar/jac_t cols)
//   cols [N_B : N_B + N_B*D_B] — atoms (jac_t/hessian cols)
//
// For variable-atom-count molecules D_A is heterogeneous; offsets are stored.

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
// Internal helper: build stacked repr for a list of molecules.
// Returns x_all, n_all, nn_all ready for kernel_gaussian_symm / kernel_gaussian.
// ---------------------------------------------------------------------------
static void build_repr(
    const std::vector<kf::fchl18::MolData> &mols, int nm, double cut_distance, int &max_size_out,
    std::vector<double> &x_all, std::vector<int> &n_all, std::vector<int> &nn_all
) {
    max_size_out = 0;
    for (int a = 0; a < nm; ++a)
        max_size_out = std::max(max_size_out, mols[a].n_atoms);

    const std::size_t repr_stride = static_cast<std::size_t>(max_size_out) * 5 * max_size_out;
    x_all.assign(static_cast<std::size_t>(nm) * repr_stride, 1e100);
    n_all.resize(nm, 0);
    nn_all.resize(static_cast<std::size_t>(nm) * max_size_out, 0);

    for (int a = 0; a < nm; ++a) {
        std::vector<double> x_mol;
        std::vector<int> nn_mol;
        kf::fchl18::generate_fchl18(
            mols[a].coords,
            mols[a].z,
            max_size_out,
            cut_distance,
            x_mol,
            nn_mol
        );
        n_all[a] = mols[a].n_atoms;
        std::copy(
            x_mol.begin(),
            x_mol.end(),
            x_all.begin() + static_cast<std::ptrdiff_t>(a) * repr_stride
        );
        for (int i = 0; i < max_size_out; ++i)
            nn_all[static_cast<std::size_t>(a) * max_size_out + i] = nn_mol[i];
    }
}

// ---------------------------------------------------------------------------
// kernel_gaussian_full (asymmetric)
//
// coords_A_list, z_A_list : N_A molecules
// coords_B_list, z_B_list : N_B molecules
//
// Output shape: (N_A + D_A_total, N_B + D_B_total)
//   D_A_total = sum_i n_atoms_i_A * 3
//   D_B_total = sum_j n_atoms_j_B * 3
// ---------------------------------------------------------------------------
py::array_t<double> kernel_gaussian_full_py(
    const py::list &coords_A_list, const py::list &z_A_list, const py::list &coords_B_list,
    const py::list &z_B_list, double sigma, double two_body_scaling, double two_body_width,
    double two_body_power, double three_body_scaling, double three_body_width,
    double three_body_power, double cut_start, double cut_distance, int fourier_order, bool use_atm
) {
    const int N_A = static_cast<int>(coords_A_list.size());
    const int N_B = static_cast<int>(coords_B_list.size());
    if (static_cast<int>(z_A_list.size()) != N_A)
        throw std::invalid_argument("coords_A_list and z_A_list must have the same length");
    if (static_cast<int>(z_B_list.size()) != N_B)
        throw std::invalid_argument("coords_B_list and z_B_list must have the same length");
    if (N_A == 0 || N_B == 0)
        throw std::invalid_argument("kernel_gaussian_full: empty molecule list");

    // Parse molecules
    std::vector<kf::fchl18::MolData> mols_A(N_A), mols_B(N_B);
    for (int a = 0; a < N_A; ++a)
        mols_A[a] = kf::fchl18::parse_mol(coords_A_list[a], z_A_list[a]);
    for (int b = 0; b < N_B; ++b)
        mols_B[b] = kf::fchl18::parse_mol(coords_B_list[b], z_B_list[b]);

    // Coordinate offsets (into jac/hessian rows/cols)
    std::vector<int> row_offset(N_A + 1, 0);
    for (int a = 0; a < N_A; ++a)
        row_offset[a + 1] = row_offset[a] + mols_A[a].n_atoms * 3;
    const int D_A = row_offset[N_A];

    std::vector<int> col_offset(N_B + 1, 0);
    for (int b = 0; b < N_B; ++b)
        col_offset[b + 1] = col_offset[b] + mols_B[b].n_atoms * 3;
    const int D_B = col_offset[N_B];

    // Full output shape: (N_A + D_A, N_B + D_B)
    const int full_rows = N_A + D_A;
    const int full_cols = N_B + D_B;

    py::array_t<double> K({(py::ssize_t)full_rows, (py::ssize_t)full_cols});
    std::memset(K.mutable_data(), 0, sizeof(double) * full_rows * full_cols);

    // Build representations for both sets
    int max_size_A, max_size_B;
    std::vector<double> x_A, x_B;
    std::vector<int> n_A_v, n_B_v, nn_A, nn_B;
    build_repr(mols_A, N_A, cut_distance, max_size_A, x_A, n_A_v, nn_A);
    build_repr(mols_B, N_B, cut_distance, max_size_B, x_B, n_B_v, nn_B);

    {
        py::gil_scoped_release release;

        // ---- Scalar block: K[0:N_A, 0:N_B] ----
        // kernel_gaussian writes (N_A, N_B) with stride N_B, but our output has stride
        // full_cols. Write into a temp buffer then scatter.
        std::vector<double> K_scalar(N_A * N_B, 0.0);
        kf::fchl18::kernel_gaussian(
            x_A,
            x_B,
            n_A_v,
            n_B_v,
            nn_A,
            nn_B,
            N_A,
            N_B,
            max_size_A,
            max_size_B,
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
            K_scalar.data()
        );

        double *K_ptr = K.mutable_data();
        // Scatter scalar block
        for (int a = 0; a < N_A; ++a) {
            for (int b = 0; b < N_B; ++b) {
                K_ptr[a * full_cols + b] = K_scalar[a * N_B + b];
            }
        }

        // ---- Jacobian block (dK/dR_A): rows [N_A:, 0:N_B] ----
        // Batch path hoists B-side precompute and picks a single OpenMP level.
        {
            std::vector<std::vector<double>> coords_A_vecs(static_cast<std::size_t>(N_A));
            std::vector<std::vector<int>> z_A_vecs(static_cast<std::size_t>(N_A));
            for (int i = 0; i < N_A; ++i) {
                coords_A_vecs[static_cast<std::size_t>(i)] = mols_A[static_cast<std::size_t>(i)].coords;
                z_A_vecs[static_cast<std::size_t>(i)] = mols_A[static_cast<std::size_t>(i)].z;
            }
            std::vector<double> J(static_cast<std::size_t>(D_A) * N_B, 0.0);
            kf::fchl18::kernel_gaussian_jacobian(
                coords_A_vecs,
                z_A_vecs,
                x_B,
                n_B_v,
                nn_B,
                N_B,
                max_size_B,
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
                J.data()
            );
            for (int row = 0; row < D_A; ++row) {
                for (int b = 0; b < N_B; ++b) {
                    K_ptr[static_cast<std::ptrdiff_t>(N_A + row) * full_cols + b] =
                        J[static_cast<std::size_t>(row) * N_B + b];
                }
            }
        }

        // ---- Jacobian-T block (dK/dR_B): rows [0:N_A, N_B:] ----
        // Differentiate B against repr_A: K_jt[a, flat_B] = dK(B, A_a)/dR_B[flat]
        // = dK(A_a, B)/dR_B by kernel symmetry.
        {
            std::vector<std::vector<double>> coords_B_vecs(static_cast<std::size_t>(N_B));
            std::vector<std::vector<int>> z_B_vecs(static_cast<std::size_t>(N_B));
            for (int j = 0; j < N_B; ++j) {
                coords_B_vecs[static_cast<std::size_t>(j)] = mols_B[static_cast<std::size_t>(j)].coords;
                z_B_vecs[static_cast<std::size_t>(j)] = mols_B[static_cast<std::size_t>(j)].z;
            }
            std::vector<double> Jt(static_cast<std::size_t>(N_A) * D_B, 0.0);
            kf::fchl18::kernel_gaussian_jacobian_t(
                coords_B_vecs,
                z_B_vecs,
                x_A,
                n_A_v,
                nn_A,
                N_A,
                max_size_A,
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
                Jt.data()
            );
            for (int a = 0; a < N_A; ++a) {
                for (int flat = 0; flat < D_B; ++flat) {
                    K_ptr[static_cast<std::ptrdiff_t>(a) * full_cols + N_B + flat] =
                        Jt[static_cast<std::size_t>(a) * D_B + flat];
                }
            }
        }

        // ---- Hessian block: rows [N_A:, N_B:] ----
        // Batch path hoists per-molecule AtomDataGrad across all pairs.
        {
            std::vector<std::vector<double>> coords_A_vecs(static_cast<std::size_t>(N_A));
            std::vector<std::vector<int>> z_A_vecs(static_cast<std::size_t>(N_A));
            std::vector<std::vector<double>> coords_B_vecs(static_cast<std::size_t>(N_B));
            std::vector<std::vector<int>> z_B_vecs(static_cast<std::size_t>(N_B));
            for (int i = 0; i < N_A; ++i) {
                coords_A_vecs[static_cast<std::size_t>(i)] = mols_A[static_cast<std::size_t>(i)].coords;
                z_A_vecs[static_cast<std::size_t>(i)] = mols_A[static_cast<std::size_t>(i)].z;
            }
            for (int j = 0; j < N_B; ++j) {
                coords_B_vecs[static_cast<std::size_t>(j)] = mols_B[static_cast<std::size_t>(j)].coords;
                z_B_vecs[static_cast<std::size_t>(j)] = mols_B[static_cast<std::size_t>(j)].z;
            }
            std::vector<double> H(static_cast<std::size_t>(D_A) * D_B, 0.0);
            kf::fchl18::kernel_gaussian_hessian_rect(
                coords_A_vecs,
                z_A_vecs,
                coords_B_vecs,
                z_B_vecs,
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
                H.data()
            );
            for (int row = 0; row < D_A; ++row) {
                for (int col = 0; col < D_B; ++col) {
                    K_ptr[static_cast<std::ptrdiff_t>(N_A + row) * full_cols + (N_B + col)] =
                        H[static_cast<std::size_t>(row) * D_B + col];
                }
            }
        }
    }

    return K;
}

// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm (symmetric training set)
//
// coords_list, z_list : N molecules (training set)
//
// Output shape: (N + D, N + D) where D = sum_i n_atoms_i * 3.
// Full matrix filled (lower triangle computed, upper mirrored).
// ---------------------------------------------------------------------------
py::array_t<double> kernel_gaussian_full_symm_py(
    const py::list &coords_list, const py::list &z_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm
) {
    const int nm = static_cast<int>(coords_list.size());
    if (static_cast<int>(z_list.size()) != nm)
        throw std::invalid_argument("coords_list and z_list must have the same length");
    if (nm == 0) throw std::invalid_argument("kernel_gaussian_full_symm: empty molecule list");

    std::vector<kf::fchl18::MolData> mols(nm);
    for (int a = 0; a < nm; ++a)
        mols[a] = kf::fchl18::parse_mol(coords_list[a], z_list[a]);

    std::vector<int> offset(nm + 1, 0);
    for (int a = 0; a < nm; ++a)
        offset[a + 1] = offset[a] + mols[a].n_atoms * 3;
    const int D = offset[nm];

    const int BIG = nm + D;

    py::array_t<double> K({(py::ssize_t)BIG, (py::ssize_t)BIG});
    std::memset(K.mutable_data(), 0, sizeof(double) * BIG * BIG);

    int max_size;
    std::vector<double> x_all;
    std::vector<int> n_all, nn_all;
    build_repr(mols, nm, cut_distance, max_size, x_all, n_all, nn_all);

    {
        py::gil_scoped_release release;

        // ---- Scalar block (symmetric): K[0:nm, 0:nm] ----
        std::vector<double> K_scalar(nm * nm, 0.0);
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
            K_scalar.data()
        );

        double *K_ptr = K.mutable_data();
        for (int a = 0; a < nm; ++a) {
            for (int b = 0; b < nm; ++b) {
                K_ptr[a * BIG + b] = K_scalar[a * nm + b];
            }
        }

        // ---- Jacobian + Jacobian-T blocks ----
        // One batch jacobian (B-hoisted) fills both by symmetry.
        {
            std::vector<std::vector<double>> coords_vecs(static_cast<std::size_t>(nm));
            std::vector<std::vector<int>> z_vecs(static_cast<std::size_t>(nm));
            for (int i = 0; i < nm; ++i) {
                coords_vecs[static_cast<std::size_t>(i)] = mols[static_cast<std::size_t>(i)].coords;
                z_vecs[static_cast<std::size_t>(i)] = mols[static_cast<std::size_t>(i)].z;
            }
            std::vector<double> J(static_cast<std::size_t>(D) * nm, 0.0);
            kf::fchl18::kernel_gaussian_jacobian(
                coords_vecs,
                z_vecs,
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
                J.data()
            );
            for (int flat = 0; flat < D; ++flat) {
                const int row_jac = nm + flat;
                for (int j = 0; j < nm; ++j) {
                    const double v = J[static_cast<std::size_t>(flat) * nm + j];
                    K_ptr[static_cast<std::ptrdiff_t>(row_jac) * BIG + j] = v;
                    K_ptr[static_cast<std::ptrdiff_t>(j) * BIG + row_jac] = v;
                }
            }
        }

        // ---- Hessian block: rows+cols [nm:, nm:], symmetric ----
        {
            std::vector<std::vector<double>> coords_vecs(static_cast<std::size_t>(nm));
            std::vector<std::vector<int>> z_vecs(static_cast<std::size_t>(nm));
            for (int a = 0; a < nm; ++a) {
                coords_vecs[static_cast<std::size_t>(a)] = mols[static_cast<std::size_t>(a)].coords;
                z_vecs[static_cast<std::size_t>(a)] = mols[static_cast<std::size_t>(a)].z;
            }
            std::vector<double> H(static_cast<std::size_t>(D) * D, 0.0);
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
                H.data()
            );
            for (int row = 0; row < D; ++row) {
                for (int col = 0; col < D; ++col) {
                    K_ptr[static_cast<std::ptrdiff_t>(nm + row) * BIG + (nm + col)] =
                        H[static_cast<std::size_t>(row) * D + col];
                }
            }
        }
    }

    return K;
}

// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm_rfp (symmetric, RFP packed)
//
// Same as kernel_gaussian_full_symm but packs into RFP format.
// Output length = BIG*(BIG+1)/2 where BIG = N + D.
// ---------------------------------------------------------------------------
py::array_t<double> kernel_gaussian_full_symm_rfp_py(
    const py::list &coords_list, const py::list &z_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm
) {
    const int nm = static_cast<int>(coords_list.size());
    if (static_cast<int>(z_list.size()) != nm)
        throw std::invalid_argument("coords_list and z_list must have the same length");
    if (nm == 0) throw std::invalid_argument("kernel_gaussian_full_symm_rfp: empty molecule list");

    std::vector<kf::fchl18::MolData> mols(nm);
    for (int a = 0; a < nm; ++a)
        mols[a] = kf::fchl18::parse_mol(coords_list[a], z_list[a]);

    std::vector<int> offset(nm + 1, 0);
    for (int a = 0; a < nm; ++a)
        offset[a + 1] = offset[a] + mols[a].n_atoms * 3;
    const int D = offset[nm];
    const std::size_t BIG = static_cast<std::size_t>(nm + D);
    const std::size_t rfp_len = BIG * (BIG + 1) / 2;

    py::array_t<double> K_rfp(static_cast<py::ssize_t>(rfp_len));
    std::memset(K_rfp.mutable_data(), 0, sizeof(double) * rfp_len);

    int max_size;
    std::vector<double> x_all;
    std::vector<int> n_all, nn_all;
    build_repr(mols, nm, cut_distance, max_size, x_all, n_all, nn_all);

    {
        py::gil_scoped_release release;

        double *rfp = K_rfp.mutable_data();

        // ---- Scalar block: K[a, b] with a >= b (lower triangle, col <= row) ----
        // Only upper triangle RFP: rfp_index_upper_N(BIG, col=b, row=a) with b <= a.
        std::vector<double> K_scalar(nm * nm, 0.0);
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
            K_scalar.data()
        );

        for (std::size_t a = 0; a < static_cast<std::size_t>(nm); ++a) {
            for (std::size_t b = 0; b <= a; ++b) {
                // col=b, row=a, b <= a
                rfp[kf::rfp_index_upper_N(BIG, b, a)] = K_scalar[a * nm + b];
            }
        }

        // ---- Jacobian + Jacobian-T blocks (same RFP slot by symmetry) ----
        {
            std::vector<std::vector<double>> coords_vecs(static_cast<std::size_t>(nm));
            std::vector<std::vector<int>> z_vecs(static_cast<std::size_t>(nm));
            for (int i = 0; i < nm; ++i) {
                coords_vecs[static_cast<std::size_t>(i)] = mols[static_cast<std::size_t>(i)].coords;
                z_vecs[static_cast<std::size_t>(i)] = mols[static_cast<std::size_t>(i)].z;
            }
            std::vector<double> J(static_cast<std::size_t>(D) * nm, 0.0);
            kf::fchl18::kernel_gaussian_jacobian(
                coords_vecs,
                z_vecs,
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
                J.data()
            );
            for (int flat = 0; flat < D; ++flat) {
                const std::size_t jac_row = static_cast<std::size_t>(nm + flat);
                for (int j = 0; j < nm; ++j) {
                    const double v = J[static_cast<std::size_t>(flat) * nm + j];
                    // jac (row=jac_row, col=j) and jac_t (row=j, col=jac_row) share one RFP slot
                    rfp[kf::rfp_index_upper_N(BIG, static_cast<std::size_t>(j), jac_row)] = v;
                }
            }
        }

        // ---- Hessian block: rows+cols [nm:, nm:] ----
        {
            std::vector<std::vector<double>> coords_vecs(static_cast<std::size_t>(nm));
            std::vector<std::vector<int>> z_vecs(static_cast<std::size_t>(nm));
            for (int a = 0; a < nm; ++a) {
                coords_vecs[static_cast<std::size_t>(a)] = mols[static_cast<std::size_t>(a)].coords;
                z_vecs[static_cast<std::size_t>(a)] = mols[static_cast<std::size_t>(a)].z;
            }
            std::vector<double> H(static_cast<std::size_t>(D) * D, 0.0);
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
                H.data()
            );
            // Pack upper triangle of the FF block into RFP:
            // BIG indices (nm+row, nm+col) with col <= row.
            for (int row = 0; row < D; ++row) {
                const std::size_t big_row = static_cast<std::size_t>(nm + row);
                for (int col = 0; col <= row; ++col) {
                    const std::size_t big_col = static_cast<std::size_t>(nm + col);
                    rfp[kf::rfp_index_upper_N(BIG, big_col, big_row)] =
                        H[static_cast<std::size_t>(row) * D + col];
                }
            }
        }
    }

    return K_rfp;
}
