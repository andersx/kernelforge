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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fchl18_kernel.hpp"
#include "fchl18_kernel_common.hpp"
#include "fchl18_repr.hpp"
#include "rfp_utils.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Internal helper: build stacked repr for a list of molecules.
// Returns x_all, n_all, nn_all ready for kernel_gaussian_symm / kernel_gaussian.
// ---------------------------------------------------------------------------
static void build_repr(
    const std::vector<kf::fchl18::MolData> &mols,
    int nm,
    double cut_distance,
    int &max_size_out,
    std::vector<double> &x_all,
    std::vector<int>    &n_all,
    std::vector<int>    &nn_all
) {
    max_size_out = 0;
    for (int a = 0; a < nm; ++a)
        max_size_out = std::max(max_size_out, mols[a].n_atoms);

    const std::size_t repr_stride =
        static_cast<std::size_t>(max_size_out) * 5 * max_size_out;
    x_all.assign(static_cast<std::size_t>(nm) * repr_stride, 1e100);
    n_all.resize(nm, 0);
    nn_all.resize(static_cast<std::size_t>(nm) * max_size_out, 0);

    for (int a = 0; a < nm; ++a) {
        std::vector<double> x_mol;
        std::vector<int>    nn_mol;
        kf::fchl18::generate_fchl18(
            mols[a].coords, mols[a].z,
            max_size_out, cut_distance,
            x_mol, nn_mol
        );
        n_all[a] = mols[a].n_atoms;
        std::copy(x_mol.begin(), x_mol.end(),
                  x_all.begin() + static_cast<std::ptrdiff_t>(a) * repr_stride);
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
) {
    if (use_atm)
        throw std::invalid_argument(
            "kernel_gaussian_full: use_atm=True is not yet supported.");
    if (cut_start < 1.0)
        throw std::invalid_argument(
            "kernel_gaussian_full: cutoff damping (cut_start < 1.0) is not yet supported.");

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
    std::vector<int>    n_A_v, n_B_v, nn_A, nn_B;
    build_repr(mols_A, N_A, cut_distance, max_size_A, x_A, n_A_v, nn_A);
    build_repr(mols_B, N_B, cut_distance, max_size_B, x_B, n_B_v, nn_B);

    {
        py::gil_scoped_release release;

        // ---- Scalar block: K[0:N_A, 0:N_B] ----
        // kernel_gaussian writes (N_A, N_B) with stride N_B, but our output has stride
        // full_cols. Write into a temp buffer then scatter.
        std::vector<double> K_scalar(N_A * N_B, 0.0);
        kf::fchl18::kernel_gaussian(
            x_A, x_B, n_A_v, n_B_v, nn_A, nn_B,
            N_A, N_B, max_size_A, max_size_B,
            sigma,
            two_body_scaling, two_body_width, two_body_power,
            three_body_scaling, three_body_width, three_body_power,
            cut_start, cut_distance, fourier_order, use_atm,
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
        // For each A molecule i, call kernel_gaussian_gradient(A_i, repr_B).
        // Result grad[alpha, mu, b] goes into K[N_A + row_offset[i] + alpha*3 + mu, b].
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_A; ++i) {
            const kf::fchl18::MolData &mi = mols_A[i];
            const int na_i = mi.n_atoms;
            std::vector<double> grad(static_cast<std::size_t>(na_i) * 3 * N_B, 0.0);

            kf::fchl18::kernel_gaussian_gradient(
                mi.coords, mi.z,
                x_B, n_B_v, nn_B,
                na_i, N_B, max_size_B,
                sigma,
                two_body_scaling, two_body_width, two_body_power,
                three_body_scaling, three_body_width, three_body_power,
                cut_start, cut_distance, fourier_order, use_atm,
                grad.data()
            );
            // grad[alpha, mu, b] -> K[N_A + row_offset[i] + alpha*3+mu, b]
            for (int alpha = 0; alpha < na_i; ++alpha) {
                for (int mu = 0; mu < 3; ++mu) {
                    const int row = N_A + row_offset[i] + alpha * 3 + mu;
                    for (int b = 0; b < N_B; ++b) {
                        K_ptr[static_cast<std::ptrdiff_t>(row) * full_cols + b] =
                            grad[static_cast<std::size_t>(alpha) * 3 * N_B + mu * N_B + b];
                    }
                }
            }
        }

        // ---- Jacobian-T block (dK/dR_B): rows [0:N_A, N_B:] ----
        // For each B molecule j, call kernel_gaussian_gradient(B_j, repr_A).
        // By K(A,B)=K(B,A): dK(A_i,B_j)/dR_B = dK(B_j,A_i)/dR_B = grad_j[beta,nu,i]
        // -> K[i, N_B + col_offset[j] + beta*3+nu]
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < N_B; ++j) {
            const kf::fchl18::MolData &mj = mols_B[j];
            const int na_j = mj.n_atoms;
            std::vector<double> grad(static_cast<std::size_t>(na_j) * 3 * N_A, 0.0);

            kf::fchl18::kernel_gaussian_gradient(
                mj.coords, mj.z,
                x_A, n_A_v, nn_A,
                na_j, N_A, max_size_A,
                sigma,
                two_body_scaling, two_body_width, two_body_power,
                three_body_scaling, three_body_width, three_body_power,
                cut_start, cut_distance, fourier_order, use_atm,
                grad.data()
            );
            // grad[beta, nu, i] = dK(B_j, A_i)/dR_B[beta,nu]
            //   = dK(A_i, B_j)/dR_B[beta,nu]  (kernel symmetry)
            // -> K[i, N_B + col_offset[j] + beta*3+nu]
            for (int beta = 0; beta < na_j; ++beta) {
                for (int nu = 0; nu < 3; ++nu) {
                    const int col = N_B + col_offset[j] + beta * 3 + nu;
                    for (int i = 0; i < N_A; ++i) {
                        K_ptr[static_cast<std::ptrdiff_t>(i) * full_cols + col] =
                            grad[static_cast<std::size_t>(beta) * 3 * N_A + nu * N_A + i];
                    }
                }
            }
        }

        // ---- Hessian block: rows [N_A:, N_B:] ----
        // H[N_A+row_offset[i]+amu, N_B+col_offset[j]+bnu] = d²K(A_i,B_j)/dR_A dR_B
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < N_A; ++i) {
            for (int j = 0; j < N_B; ++j) {
                const kf::fchl18::MolData &mi = mols_A[i];
                const int na3A = mi.n_atoms * 3;
                const kf::fchl18::MolData &mj = mols_B[j];
                const int na3B = mj.n_atoms * 3;

                std::vector<double> block(static_cast<std::size_t>(na3A) * na3B, 0.0);
                kf::fchl18::kernel_gaussian_hessian(
                    mi.coords, mi.z,
                    mj.coords, mj.z,
                    mi.n_atoms, mj.n_atoms,
                    sigma,
                    two_body_scaling, two_body_width, two_body_power,
                    three_body_scaling, three_body_width, three_body_power,
                    cut_start, cut_distance, fourier_order, use_atm,
                    block.data()
                );

                for (int amu = 0; amu < na3A; ++amu) {
                    const int row = N_A + row_offset[i] + amu;
                    for (int bnu = 0; bnu < na3B; ++bnu) {
                        const int col = N_B + col_offset[j] + bnu;
                        K_ptr[static_cast<std::ptrdiff_t>(row) * full_cols + col] =
                            block[static_cast<std::size_t>(amu) * na3B + bnu];
                    }
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
) {
    if (use_atm)
        throw std::invalid_argument(
            "kernel_gaussian_full_symm: use_atm=True is not yet supported.");
    if (cut_start < 1.0)
        throw std::invalid_argument(
            "kernel_gaussian_full_symm: cutoff damping (cut_start < 1.0) is not yet supported.");

    const int nm = static_cast<int>(coords_list.size());
    if (static_cast<int>(z_list.size()) != nm)
        throw std::invalid_argument("coords_list and z_list must have the same length");
    if (nm == 0)
        throw std::invalid_argument("kernel_gaussian_full_symm: empty molecule list");

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
    std::vector<int>    n_all, nn_all;
    build_repr(mols, nm, cut_distance, max_size, x_all, n_all, nn_all);

    {
        py::gil_scoped_release release;

        // ---- Scalar block (symmetric): K[0:nm, 0:nm] ----
        std::vector<double> K_scalar(nm * nm, 0.0);
        kf::fchl18::kernel_gaussian_symm(
            x_all, n_all, nn_all,
            nm, max_size,
            sigma,
            two_body_scaling, two_body_width, two_body_power,
            three_body_scaling, three_body_width, three_body_power,
            cut_start, cut_distance, fourier_order, use_atm,
            K_scalar.data()
        );

        double *K_ptr = K.mutable_data();
        for (int a = 0; a < nm; ++a) {
            for (int b = 0; b < nm; ++b) {
                K_ptr[a * BIG + b] = K_scalar[a * nm + b];
            }
        }

        // ---- Jacobian block (dK/dR_A): rows [nm:, 0:nm] and mirror [0:nm, nm:] ----
        // For each mol i: grad[alpha,mu, j] = dK(mol_i, mol_j)/dR_i[alpha,mu]
        // -> K[nm+offset[i]+alpha*3+mu, j]  (jac block)
        // -> K[j, nm+offset[i]+alpha*3+mu]  (jac_t block, by symmetry = same value)
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nm; ++i) {
            const kf::fchl18::MolData &mi = mols[i];
            const int na_i = mi.n_atoms;
            std::vector<double> grad(static_cast<std::size_t>(na_i) * 3 * nm, 0.0);

            kf::fchl18::kernel_gaussian_gradient(
                mi.coords, mi.z,
                x_all, n_all, nn_all,
                na_i, nm, max_size,
                sigma,
                two_body_scaling, two_body_width, two_body_power,
                three_body_scaling, three_body_width, three_body_power,
                cut_start, cut_distance, fourier_order, use_atm,
                grad.data()
            );

            for (int alpha = 0; alpha < na_i; ++alpha) {
                for (int mu = 0; mu < 3; ++mu) {
                    const int row_jac = nm + offset[i] + alpha * 3 + mu;
                    for (int j = 0; j < nm; ++j) {
                        const double v =
                            grad[static_cast<std::size_t>(alpha) * 3 * nm + mu * nm + j];
                        K_ptr[static_cast<std::ptrdiff_t>(row_jac) * BIG + j] = v;
                        K_ptr[static_cast<std::ptrdiff_t>(j) * BIG + row_jac] = v;
                    }
                }
            }
        }

        // ---- Hessian block: rows+cols [nm:, nm:], symmetric ----
        // Only compute lower triangle b <= a, then mirror.
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int a = 0; a < nm; ++a) {
            for (int b = 0; b < nm; ++b) {
                if (b > a) continue;

                const kf::fchl18::MolData &ma = mols[a];
                const int na3A = ma.n_atoms * 3;
                const kf::fchl18::MolData &mb = mols[b];
                const int na3B = mb.n_atoms * 3;

                std::vector<double> block(static_cast<std::size_t>(na3A) * na3B, 0.0);
                kf::fchl18::kernel_gaussian_hessian(
                    ma.coords, ma.z,
                    mb.coords, mb.z,
                    ma.n_atoms, mb.n_atoms,
                    sigma,
                    two_body_scaling, two_body_width, two_body_power,
                    three_body_scaling, three_body_width, three_body_power,
                    cut_start, cut_distance, fourier_order, use_atm,
                    block.data()
                );

                if (a == b) {
                    // Diagonal block: symmetrize
                    for (int amu = 0; amu < na3A; ++amu) {
                        for (int bnu = 0; bnu < na3B; ++bnu) {
                            double v;
                            if (amu == bnu) {
                                v = block[static_cast<std::size_t>(amu) * na3B + bnu];
                            } else {
                                v = 0.5 * (block[static_cast<std::size_t>(amu) * na3B + bnu]
                                         + block[static_cast<std::size_t>(bnu) * na3B + amu]);
                            }
                            const int row = nm + offset[a] + amu;
                            const int col = nm + offset[b] + bnu;
                            K_ptr[static_cast<std::ptrdiff_t>(row) * BIG + col] = v;
                            K_ptr[static_cast<std::ptrdiff_t>(col) * BIG + row] = v;
                        }
                    }
                } else {
                    // Off-diagonal (a > b): fill block and transpose
                    for (int amu = 0; amu < na3A; ++amu) {
                        for (int bnu = 0; bnu < na3B; ++bnu) {
                            const double v = block[static_cast<std::size_t>(amu) * na3B + bnu];
                            const int row = nm + offset[a] + amu;
                            const int col = nm + offset[b] + bnu;
                            K_ptr[static_cast<std::ptrdiff_t>(row) * BIG + col] = v;
                            K_ptr[static_cast<std::ptrdiff_t>(col) * BIG + row] = v;
                        }
                    }
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
) {
    if (use_atm)
        throw std::invalid_argument(
            "kernel_gaussian_full_symm_rfp: use_atm=True is not yet supported.");
    if (cut_start < 1.0)
        throw std::invalid_argument(
            "kernel_gaussian_full_symm_rfp: cutoff damping (cut_start < 1.0) is not yet supported.");

    const int nm = static_cast<int>(coords_list.size());
    if (static_cast<int>(z_list.size()) != nm)
        throw std::invalid_argument("coords_list and z_list must have the same length");
    if (nm == 0)
        throw std::invalid_argument("kernel_gaussian_full_symm_rfp: empty molecule list");

    std::vector<kf::fchl18::MolData> mols(nm);
    for (int a = 0; a < nm; ++a)
        mols[a] = kf::fchl18::parse_mol(coords_list[a], z_list[a]);

    std::vector<int> offset(nm + 1, 0);
    for (int a = 0; a < nm; ++a)
        offset[a + 1] = offset[a] + mols[a].n_atoms * 3;
    const int D   = offset[nm];
    const std::size_t BIG = static_cast<std::size_t>(nm + D);
    const std::size_t rfp_len = BIG * (BIG + 1) / 2;

    py::array_t<double> K_rfp(static_cast<py::ssize_t>(rfp_len));
    std::memset(K_rfp.mutable_data(), 0, sizeof(double) * rfp_len);

    int max_size;
    std::vector<double> x_all;
    std::vector<int>    n_all, nn_all;
    build_repr(mols, nm, cut_distance, max_size, x_all, n_all, nn_all);

    {
        py::gil_scoped_release release;

        double *rfp = K_rfp.mutable_data();

        // ---- Scalar block: K[a, b] with a >= b (lower triangle, col <= row) ----
        // Only upper triangle RFP: rfp_index_upper_N(BIG, col=b, row=a) with b <= a.
        std::vector<double> K_scalar(nm * nm, 0.0);
        kf::fchl18::kernel_gaussian_symm(
            x_all, n_all, nn_all,
            nm, max_size,
            sigma,
            two_body_scaling, two_body_width, two_body_power,
            three_body_scaling, three_body_width, three_body_power,
            cut_start, cut_distance, fourier_order, use_atm,
            K_scalar.data()
        );

        for (std::size_t a = 0; a < static_cast<std::size_t>(nm); ++a) {
            for (std::size_t b = 0; b <= a; ++b) {
                // col=b, row=a, b <= a
                rfp[kf::rfp_index_upper_N(BIG, b, a)] = K_scalar[a * nm + b];
            }
        }

        // ---- Jacobian block + Jacobian-T block ----
        // For mol i: grad[alpha,mu,j] = dK(mol_i,mol_j)/dR_i[alpha,mu]
        // Jac row:  BIG row = nm + offset[i] + alpha*3+mu,  BIG col = j
        //   -> col <= row condition: j < nm + offset[i]+alpha*3+mu   always true (j < nm)
        //   -> rfp_index_upper_N(BIG, col=j, row=nm+offset[i]+alpha*3+mu)
        // Jac_t col: BIG row = j,  BIG col = nm + offset[i] + alpha*3+mu
        //   -> col > row (jac_t is in upper triangle of block matrix, row=j < nm <= col)
        //   -> rfp_index_upper_N(BIG, col=j, row=nm+...) — same index as jac by symmetry!
        // So we write each entry once at rfp_index_upper_N(BIG, min(r,c), max(r,c)).
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nm; ++i) {
            const kf::fchl18::MolData &mi = mols[i];
            const int na_i = mi.n_atoms;
            std::vector<double> grad(static_cast<std::size_t>(na_i) * 3 * nm, 0.0);

            kf::fchl18::kernel_gaussian_gradient(
                mi.coords, mi.z,
                x_all, n_all, nn_all,
                na_i, nm, max_size,
                sigma,
                two_body_scaling, two_body_width, two_body_power,
                three_body_scaling, three_body_width, three_body_power,
                cut_start, cut_distance, fourier_order, use_atm,
                grad.data()
            );

            for (int alpha = 0; alpha < na_i; ++alpha) {
                for (int mu = 0; mu < 3; ++mu) {
                    const std::size_t jac_row =
                        static_cast<std::size_t>(nm + offset[i] + alpha * 3 + mu);
                    for (int j = 0; j < nm; ++j) {
                        const double v =
                            grad[static_cast<std::size_t>(alpha) * 3 * nm + mu * nm + j];
                        // jac:   (row=jac_row, col=j)   jac_row > j always
                        // jac_t: (row=j, col=jac_row)   same RFP position (col=j, row=jac_row)
                        rfp[kf::rfp_index_upper_N(BIG,
                                                   static_cast<std::size_t>(j),
                                                   jac_row)] = v;
                    }
                }
            }
        }

        // ---- Hessian block: rows+cols [nm:, nm:] ----
        // Lower triangle b <= a, then symmetry gives the transpose.
        // For off-diagonal: (row=nm+offset[a]+amu, col=nm+offset[b]+bnu), col <= row since b<=a.
        //   rfp_index_upper_N(BIG, col, row)
        // For diagonal (a==b): symmetrize element-wise.
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int a = 0; a < nm; ++a) {
            for (int b = 0; b < nm; ++b) {
                if (b > a) continue;

                const kf::fchl18::MolData &ma = mols[a];
                const int na3A = ma.n_atoms * 3;
                const kf::fchl18::MolData &mb = mols[b];
                const int na3B = mb.n_atoms * 3;

                std::vector<double> block(static_cast<std::size_t>(na3A) * na3B, 0.0);
                kf::fchl18::kernel_gaussian_hessian(
                    ma.coords, ma.z,
                    mb.coords, mb.z,
                    ma.n_atoms, mb.n_atoms,
                    sigma,
                    two_body_scaling, two_body_width, two_body_power,
                    three_body_scaling, three_body_width, three_body_power,
                    cut_start, cut_distance, fourier_order, use_atm,
                    block.data()
                );

                if (a == b) {
                    // Diagonal hessian block: symmetrize
                    for (int amu = 0; amu < na3A; ++amu) {
                        const std::size_t row = static_cast<std::size_t>(nm + offset[a] + amu);
                        for (int bnu = 0; bnu <= amu; ++bnu) {
                            const std::size_t col = static_cast<std::size_t>(nm + offset[b] + bnu);
                            double v;
                            if (amu == bnu) {
                                v = block[static_cast<std::size_t>(amu) * na3B + bnu];
                            } else {
                                v = 0.5 * (block[static_cast<std::size_t>(amu) * na3B + bnu]
                                         + block[static_cast<std::size_t>(bnu) * na3B + amu]);
                            }
                            rfp[kf::rfp_index_upper_N(BIG, col, row)] = v;
                        }
                    }
                } else {
                    // Off-diagonal (a > b): full block, col < row guaranteed
                    for (int amu = 0; amu < na3A; ++amu) {
                        const std::size_t row = static_cast<std::size_t>(nm + offset[a] + amu);
                        for (int bnu = 0; bnu < na3B; ++bnu) {
                            const std::size_t col = static_cast<std::size_t>(nm + offset[b] + bnu);
                            rfp[kf::rfp_index_upper_N(BIG, col, row)] =
                                block[static_cast<std::size_t>(amu) * na3B + bnu];
                        }
                    }
                }
            }
        }
    }

    return K_rfp;
}
