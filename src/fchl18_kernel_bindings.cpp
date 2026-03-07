// C++ standard library
#include <cstring>
#include <stdexcept>
#include <vector>

// OpenMP
#ifdef _OPENMP
    #include <omp.h>
#endif

// RFP index helper
#include "rfp_utils.hpp"

// Third-party libraries
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Project headers
#include "fchl18_kernel.hpp"
#include "fchl18_kernel_common.hpp"

// New kernel implementations
#include "fchl18_full_kernels.hpp"
#include "fchl18_hessian_kernels.hpp"
#include "fchl18_jacobian_kernels.hpp"
#include "fchl18_scalar_kernels.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: convert a 2-D contiguous int32 NumPy array to std::vector<int>
// ---------------------------------------------------------------------------
static std::vector<int> as_int_vector_2d(
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &arr
) {
    const std::size_t n = static_cast<std::size_t>(arr.size());
    std::vector<int> out(n);
    const int32_t *src = arr.data();
    for (std::size_t i = 0; i < n; ++i)
        out[i] = static_cast<int>(src[i]);
    return out;
}

static std::vector<int> as_int_vector_1d(
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &arr
) {
    const std::size_t n = static_cast<std::size_t>(arr.size());
    std::vector<int> out(n);
    const int32_t *src = arr.data();
    for (std::size_t i = 0; i < n; ++i)
        out[i] = static_cast<int>(src[i]);
    return out;
}

static std::vector<double> as_double_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &arr
) {
    const std::size_t n = static_cast<std::size_t>(arr.size());
    return std::vector<double>(arr.data(), arr.data() + n);
}

// ---------------------------------------------------------------------------
// kernel_gaussian: asymmetric (nm1, nm2) kernel matrix
//
// x1, x2    : (nm, max_size, 5, max_size) float64
// n1, n2    : (nm,) int32 — number of real atoms
// nn1, nn2  : (nm, max_size) int32 — neighbour counts per atom
// ---------------------------------------------------------------------------
static py::array_t<double> kernel_gaussian_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &x1,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &x2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n1,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &nn1,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &nn2, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm
) {
    if (x1.ndim() != 4) throw std::invalid_argument("x1 must be 4-D");
    if (x2.ndim() != 4) throw std::invalid_argument("x2 must be 4-D");
    if (x1.shape(2) != 5) throw std::invalid_argument("x1 dim 2 must be 5");
    if (x2.shape(2) != 5) throw std::invalid_argument("x2 dim 2 must be 5");

    const int nm1 = static_cast<int>(x1.shape(0));
    const int max_size1 = static_cast<int>(x1.shape(1));
    const int nm2 = static_cast<int>(x2.shape(0));
    const int max_size2 = static_cast<int>(x2.shape(1));

    auto x1_v = as_double_vector(x1);
    auto x2_v = as_double_vector(x2);
    auto n1_v = as_int_vector_1d(n1);
    auto n2_v = as_int_vector_1d(n2);
    auto nn1_v = as_int_vector_2d(nn1);
    auto nn2_v = as_int_vector_2d(nn2);

    py::array_t<double> K({(py::ssize_t)nm1, (py::ssize_t)nm2});

    {
        py::gil_scoped_release release;
        kf::fchl18::kernel_gaussian(
            x1_v,
            x2_v,
            n1_v,
            n2_v,
            nn1_v,
            nn2_v,
            nm1,
            nm2,
            max_size1,
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
            K.mutable_data()
        );
    }

    return K;
}

// ---------------------------------------------------------------------------
// kernel_gaussian_symm: symmetric (nm, nm) kernel matrix
// ---------------------------------------------------------------------------
static py::array_t<double> kernel_gaussian_symm_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &x,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &nn, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm
) {
    if (x.ndim() != 4) throw std::invalid_argument("x must be 4-D");
    if (x.shape(2) != 5) throw std::invalid_argument("x dim 2 must be 5");

    const int nm = static_cast<int>(x.shape(0));
    const int max_size = static_cast<int>(x.shape(1));

    auto x_v = as_double_vector(x);
    auto n_v = as_int_vector_1d(n);
    auto nn_v = as_int_vector_2d(nn);

    py::array_t<double> K({(py::ssize_t)nm, (py::ssize_t)nm});

    {
        py::gil_scoped_release release;
        kf::fchl18::kernel_gaussian_symm(
            x_v,
            n_v,
            nn_v,
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
            K.mutable_data()
        );
    }

    return K;
}

// ---------------------------------------------------------------------------
// kernel_gaussian_gradient: dK[A,b]/dR_A[alpha,mu]
//
// coords_A : (n_atoms_A, 3) float64 — Cartesian coordinates of query molecule A
// z_A      : (n_atoms_A,)   int32   — nuclear charges of A
// x2       : (nm2, max_size2, 5, max_size2) float64
// n2       : (nm2,)          int32
// nn2      : (nm2, max_size2) int32
//
// Returns ndarray shape (n_atoms_A, 3, nm2), float64
// ---------------------------------------------------------------------------
static py::array_t<double> kernel_gaussian_gradient_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &coords_A,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &z_A,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &x2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &nn2, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm
) {
    // Validate coords_A
    if (coords_A.ndim() != 2 || coords_A.shape(1) != 3)
        throw std::invalid_argument("coords_A must have shape (n_atoms_A, 3)");
    if (z_A.ndim() != 1 || z_A.shape(0) != coords_A.shape(0))
        throw std::invalid_argument("z_A must have shape (n_atoms_A,)");
    if (x2.ndim() != 4) throw std::invalid_argument("x2 must be 4-D");
    if (x2.shape(2) != 5) throw std::invalid_argument("x2 dim 2 must be 5");

    const int n_atoms_A = static_cast<int>(coords_A.shape(0));
    const int nm2 = static_cast<int>(x2.shape(0));
    const int max_size2 = static_cast<int>(x2.shape(1));

    // Convert coords_A to flat vector (n_atoms_A * 3)
    std::vector<double> coords_v(static_cast<std::size_t>(n_atoms_A) * 3);
    {
        auto r = coords_A.unchecked<2>();
        for (int i = 0; i < n_atoms_A; ++i) {
            coords_v[i * 3 + 0] = r(i, 0);
            coords_v[i * 3 + 1] = r(i, 1);
            coords_v[i * 3 + 2] = r(i, 2);
        }
    }

    // Convert z_A
    std::vector<int> z_v(n_atoms_A);
    {
        auto r = z_A.unchecked<1>();
        for (int i = 0; i < n_atoms_A; ++i)
            z_v[i] = static_cast<int>(r(i));
    }

    auto x2_v = as_double_vector(x2);
    auto n2_v = as_int_vector_1d(n2);
    auto nn2_v = as_int_vector_2d(nn2);

    py::array_t<double> grad({(py::ssize_t)n_atoms_A, (py::ssize_t)3, (py::ssize_t)nm2});

    {
        py::gil_scoped_release release;
        kf::fchl18::kernel_gaussian_gradient(
            coords_v,
            z_v,
            x2_v,
            n2_v,
            nn2_v,
            n_atoms_A,
            nm2,
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
            grad.mutable_data()
        );
    }

    return grad;
}

// ---------------------------------------------------------------------------
// kernel_gaussian_hessian: d²K[A,B]/dR_A dR_B, contracted block matrix
//
// coords_A_list : list of nm_A arrays, each (n_atoms_i, 3), float64
// z_A_list      : list of nm_A arrays, each (n_atoms_i,),  int32
// coords_B_list : list of nm_B arrays, each (n_atoms_j, 3), float64
// z_B_list      : list of nm_B arrays, each (n_atoms_j,),  int32
//
// Returns ndarray shape (D_A, D_B) where D_i = sum_mol n_atoms_i * 3.
// ---------------------------------------------------------------------------
static py::array_t<double> kernel_gaussian_hessian_py(
    const py::list &coords_A_list, const py::list &z_A_list, const py::list &coords_B_list,
    const py::list &z_B_list, double sigma, double two_body_scaling, double two_body_width,
    double two_body_power, double three_body_scaling, double three_body_width,
    double three_body_power, double cut_start, double cut_distance, int fourier_order, bool use_atm
) {
    const int nm_A = static_cast<int>(coords_A_list.size());
    const int nm_B = static_cast<int>(coords_B_list.size());

    if (static_cast<int>(z_A_list.size()) != nm_A)
        throw std::invalid_argument("coords_A_list and z_A_list must have the same length");
    if (static_cast<int>(coords_B_list.size()) != nm_B)
        throw std::invalid_argument("coords_B_list and z_B_list must have the same length");
    if (static_cast<int>(z_B_list.size()) != nm_B)
        throw std::invalid_argument("coords_B_list and z_B_list must have the same length");

    // Parse molecules using shared helper
    std::vector<kf::fchl18::MolData> mols_A(nm_A), mols_B(nm_B);
    for (int a = 0; a < nm_A; ++a)
        mols_A[a] = kf::fchl18::parse_mol(coords_A_list[a], z_A_list[a]);
    for (int b = 0; b < nm_B; ++b)
        mols_B[b] = kf::fchl18::parse_mol(coords_B_list[b], z_B_list[b]);

    // Compute offsets
    std::vector<int> row_offset(nm_A + 1, 0);
    for (int a = 0; a < nm_A; ++a)
        row_offset[a + 1] = row_offset[a] + mols_A[a].n_atoms * 3;
    const int D_A = row_offset[nm_A];

    std::vector<int> col_offset(nm_B + 1, 0);
    for (int b = 0; b < nm_B; ++b)
        col_offset[b + 1] = col_offset[b] + mols_B[b].n_atoms * 3;
    const int D_B = col_offset[nm_B];

    // Allocate output (D_A, D_B), zero-initialised
    py::array_t<double> H({(py::ssize_t)D_A, (py::ssize_t)D_B});
    std::memset(H.mutable_data(), 0, sizeof(double) * D_A * D_B);

    {
        py::gil_scoped_release release;

#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int a = 0; a < nm_A; ++a) {
            for (int b = 0; b < nm_B; ++b) {
                const kf::fchl18::MolData &ma = mols_A[a];
                const int na3A = ma.n_atoms * 3;
                const int r0 = row_offset[a];
                const kf::fchl18::MolData &mb = mols_B[b];
                const int na3B = mb.n_atoms * 3;
                const int c0 = col_offset[b];

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
                for (int amu = 0; amu < na3A; ++amu) {
                    const int row = r0 + amu;
                    for (int bnu = 0; bnu < na3B; ++bnu) {
                        H_ptr[row * D_B + (c0 + bnu)] =
                            block[static_cast<std::size_t>(amu) * na3B + bnu];
                    }
                }
            }
        }
    }

    return H;
}

// ---------------------------------------------------------------------------
// kernel_gaussian_hessian_symm_rfp:
//   Symmetric force-force kernel in RFP (Rectangular Full Packed) format.
// ---------------------------------------------------------------------------
static py::array_t<double> kernel_gaussian_hessian_symm_rfp_py(
    const py::list &coords_list, const py::list &z_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm
) {
    const int nm = static_cast<int>(coords_list.size());
    if (static_cast<int>(z_list.size()) != nm)
        throw std::invalid_argument("coords_list and z_list must have the same length");

    std::vector<kf::fchl18::MolData> mols(nm);
    for (int a = 0; a < nm; ++a)
        mols[a] = kf::fchl18::parse_mol(coords_list[a], z_list[a]);

    // Offsets into the flattened coordinate dimension
    std::vector<int> offset(nm + 1, 0);
    for (int a = 0; a < nm; ++a)
        offset[a + 1] = offset[a] + mols[a].n_atoms * 3;
    const std::size_t BIG = static_cast<std::size_t>(offset[nm]);  // N*D

    // Allocate RFP output (upper triangle packed, length = BIG*(BIG+1)/2)
    const std::size_t rfp_len = BIG * (BIG + 1) / 2;
    py::array_t<double> H_rfp(static_cast<py::ssize_t>(rfp_len));
    std::memset(H_rfp.mutable_data(), 0, sizeof(double) * rfp_len);

    {
        py::gil_scoped_release release;

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

                double *rfp_ptr = H_rfp.mutable_data();

                if (a == b) {
                    for (int amu = 0; amu < na3A; ++amu) {
                        const std::size_t row = static_cast<std::size_t>(r0 + amu);
                        for (int bnu = 0; bnu <= amu; ++bnu) {
                            const std::size_t col = static_cast<std::size_t>(c0 + bnu);
                            double val;
                            if (amu == bnu) {
                                val = block[static_cast<std::size_t>(amu) * na3B + bnu];
                            } else {
                                val = 0.5 * (block[static_cast<std::size_t>(amu) * na3B + bnu] +
                                             block[static_cast<std::size_t>(bnu) * na3B + amu]);
                            }
                            rfp_ptr[kf::rfp_index_upper_N(BIG, col, row)] = val;
                        }
                    }
                } else {
                    for (int amu = 0; amu < na3A; ++amu) {
                        const std::size_t row = static_cast<std::size_t>(r0 + amu);
                        for (int bnu = 0; bnu < na3B; ++bnu) {
                            const std::size_t col = static_cast<std::size_t>(c0 + bnu);
                            rfp_ptr[kf::rfp_index_upper_N(BIG, col, row)] =
                                block[static_cast<std::size_t>(amu) * na3B + bnu];
                        }
                    }
                }
            }
        }
    }

    return H_rfp;
}

// ---------------------------------------------------------------------------
// kernel_gaussian_jacobian_t:
//   Jacobian-transpose kernel: K_jt[i, j*D+d] = dK(test_i, train_j)/dR_{train_j}[d]
//   Shape: (N_test, N_train * D)
// ---------------------------------------------------------------------------
static py::array_t<double> kernel_gaussian_jacobian_t_py(
    const py::list &coords_train_list, const py::list &z_train_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &x_test,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n_test,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &nn_test, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm
) {
    const int N_train = static_cast<int>(coords_train_list.size());
    if (static_cast<int>(z_train_list.size()) != N_train)
        throw std::invalid_argument("coords_train_list and z_train_list must have the same length");
    if (x_test.ndim() != 4 || x_test.shape(2) != 5)
        throw std::invalid_argument(
            "x_test must be 4-D with shape (N_test, max_size, 5, max_size)"
        );

    const int N_test = static_cast<int>(x_test.shape(0));
    const int max_size = static_cast<int>(x_test.shape(1));

    auto x_te_flat = as_double_vector(x_test);
    auto n_te_flat = as_int_vector_1d(n_test);
    auto nn_te_flat = as_int_vector_2d(nn_test);

    // Parse training molecules using shared helper
    std::vector<kf::fchl18::MolData> train_mols(N_train);
    for (int j = 0; j < N_train; ++j)
        train_mols[j] = kf::fchl18::parse_mol(coords_train_list[j], z_train_list[j]);

    if (N_train == 0) throw std::invalid_argument("kernel_gaussian_jacobian_t: empty training set");

    // Compute per-training-mol column offsets (supports variable atom counts).
    std::vector<int> col_offset(N_train + 1, 0);
    for (int j = 0; j < N_train; ++j)
        col_offset[j + 1] = col_offset[j] + train_mols[j].n_atoms * 3;
    const int D_total = col_offset[N_train];  // sum_j n_atoms_j * 3

    // Output shape: (N_test, D_total)
    py::array_t<double> K_jt({(py::ssize_t)N_test, (py::ssize_t)D_total});
    std::memset(K_jt.mutable_data(), 0, sizeof(double) * N_test * D_total);

    {
        py::gil_scoped_release release;

// Parallelise over training molecules j.
// Each j writes to disjoint columns [col_offset[j], col_offset[j+1]).
#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < N_train; ++j) {
            const kf::fchl18::MolData &mj = train_mols[j];
            const int na_j = mj.n_atoms;

            // grad_j[alpha, mu, i] = dK(train_j, test_i) / dR_{train_j}[alpha,mu]
            // shape: (na_j, 3, N_test)
            std::vector<double> grad_j(static_cast<std::size_t>(na_j) * 3 * N_test, 0.0);

            kf::fchl18::kernel_gaussian_gradient(
                mj.coords,
                mj.z,
                x_te_flat,
                n_te_flat,
                nn_te_flat,
                na_j,
                N_test,
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
                grad_j.data()
            );

            // Scatter: K_jt[i, col_offset[j] + alpha*3 + mu] = grad_j[alpha, mu, i]
            double *K_ptr = K_jt.mutable_data();
            for (int alpha = 0; alpha < na_j; ++alpha) {
                for (int mu = 0; mu < 3; ++mu) {
                    const int col = col_offset[j] + alpha * 3 + mu;
                    for (int i = 0; i < N_test; ++i) {
                        K_ptr[static_cast<std::ptrdiff_t>(i) * D_total + col] =
                            grad_j[static_cast<std::size_t>(alpha) * 3 * N_test + mu * N_test + i];
                    }
                }
            }
        }
    }

    return K_jt;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(fchl18_kernel, m) {
    m.doc() = "FCHL18 Gaussian kernel functions";

    const char *kg_doc = R"pbdoc(
Compute the FCHL18 Gaussian kernel matrix K[a, b] between two sets of molecules.

Parameters
----------
x1, x2 : ndarray, shape (nm, max_size, 5, max_size), float64
    Representations from fchl18_repr.generate().
n1, n2 : ndarray, shape (nm,), int32
n1, n2 : ndarray, shape (nm,), int32
nn1, nn2 : ndarray, shape (nm, max_size), int32
sigma : float

Returns
-------
ndarray, shape (nm1, nm2), float64
)pbdoc";

    m.def(
        "kernel_gaussian",
        &kernel_gaussian_py,
        py::arg("x1"),
        py::arg("x2"),
        py::arg("n1"),
        py::arg("n2"),
        py::arg("nn1"),
        py::arg("nn2"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        kg_doc
    );

    m.def(
        "kernel_gaussian_symm",
        &kernel_gaussian_symm_py,
        py::arg("x"),
        py::arg("n"),
        py::arg("nn"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        "Compute the symmetric FCHL18 Gaussian kernel matrix K[a, b] = K[b, a]."
    );

    m.def(
        "kernel_gaussian_gradient",
        &kernel_gaussian_gradient_py,
        py::arg("coords_A"),
        py::arg("z_A"),
        py::arg("x2"),
        py::arg("n2"),
        py::arg("nn2"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        "Compute dK[A,b]/dR_A[alpha,mu]. Returns shape (n_atoms_A, 3, nm2)."
    );

    m.def(
        "kernel_gaussian_hessian",
        &kernel_gaussian_hessian_py,
        py::arg("coords_A_list"),
        py::arg("z_A_list"),
        py::arg("coords_B_list"),
        py::arg("z_B_list"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        "Compute d²K[A,B]/dR_A dR_B. Returns shape (D_A, D_B)."
    );

    m.def(
        "kernel_gaussian_hessian_symm_rfp",
        &kernel_gaussian_hessian_symm_rfp_py,
        py::arg("coords_list"),
        py::arg("z_list"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        "Compute symmetric Hessian kernel in RFP format. Output length N*D*(N*D+1)//2."
    );

    m.def(
        "kernel_gaussian_jacobian_t",
        &kernel_gaussian_jacobian_t_py,
        py::arg("coords_train_list"),
        py::arg("z_train_list"),
        py::arg("x_test"),
        py::arg("n_test"),
        py::arg("nn_test"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        R"pbdoc(
Compute the FCHL18 Jacobian-transpose kernel for energy prediction from force coefficients.

K_jt[i, col_offset[j] + d] = dK(test_i, train_j) / dR_{train_j}[d]

Shape: (N_test, D_total) where D_total = sum_j n_atoms_j * 3.
For homogeneous training sets (all same atom count), D_total = N_train * D.

Given force KRR coefficients alpha (shape D_total) solved from the Hessian kernel,
energy predictions are:

    E_pred = K_jt @ alpha       (shape N_test,)

Parameters
----------
coords_train_list : list of ndarray, each shape (n_atoms_j, 3), float64
    Cartesian coordinates of the N_train training molecules.
z_train_list : list of ndarray, each shape (n_atoms_j,), int32
    Nuclear charges of the training molecules.
x_test : ndarray, shape (N_test, max_size, 5, max_size), float64
    Pre-computed FCHL18 representations of the test molecules.
n_test : ndarray, shape (N_test,), int32
nn_test : ndarray, shape (N_test, max_size), int32
sigma : float

Returns
-------
ndarray, shape (N_test, D_total), float64
    K_jt[i, col_offset[j]+d] = dK(test_i, train_j) / dR_{train_j}[d]
)pbdoc"
    );

    // ---- New kernels ----

    m.def(
        "kernel_gaussian_symm_rfp",
        &kernel_gaussian_symm_rfp_py,
        py::arg("coords_list"),
        py::arg("z_list"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        R"pbdoc(
Compute the symmetric FCHL18 scalar kernel in RFP (Rectangular Full Packed) format.

Takes raw Cartesian coordinates and nuclear charges; generates representations internally.
Equivalent to kernel_gaussian_symm but packed into upper-triangle RFP format for use
with cho_solve_rfp.

Parameters
----------
coords_list : list of ndarray, each shape (n_atoms_i, 3), float64
z_list : list of ndarray, each shape (n_atoms_i,), int32
sigma : float
(remaining hyperparameters same as kernel_gaussian)

Returns
-------
ndarray, shape (N*(N+1)//2,), float64
    Upper-triangle RFP-packed scalar kernel (TRANSR='N', UPLO='U').
)pbdoc"
    );

    m.def(
        "kernel_gaussian_jacobian",
        &kernel_gaussian_jacobian_py,
        py::arg("coords_A_list"),
        py::arg("z_A_list"),
        py::arg("x2"),
        py::arg("n2"),
        py::arg("nn2"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        R"pbdoc(
Compute the FCHL18 Jacobian kernel dK/dR_A.

J[row, b] = dK(A_i, B_b) / dR_{A_i}[flat_coord]
where row = row_offset_i + atom_alpha*3 + mu.

Parameters
----------
coords_A_list : list of ndarray, each shape (n_atoms_i, 3), float64
    Raw Cartesian coordinates for query molecules.
z_A_list : list of ndarray, each shape (n_atoms_i,), int32
    Nuclear charges for query molecules.
x2 : ndarray, shape (N_B, max_size, 5, max_size), float64
    Pre-computed FCHL18 representations of training molecules B.
n2 : ndarray, shape (N_B,), int32
nn2 : ndarray, shape (N_B, max_size), int32
sigma : float

Returns
-------
ndarray, shape (D_A, N_B), float64
    D_A = sum_i n_atoms_i * 3
)pbdoc"
    );

    m.def(
        "kernel_gaussian_hessian_symm",
        &kernel_gaussian_hessian_symm_py,
        py::arg("coords_list"),
        py::arg("z_list"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"pbdoc(
Compute the symmetric FCHL18 Hessian kernel (full matrix).

Equivalent to kernel_gaussian_hessian(mols, mols, ...) but exploits symmetry:
only lower triangle of molecule blocks is computed, then mirrored.

Parameters
----------
coords_list : list of ndarray, each shape (n_atoms_i, 3), float64
z_list : list of ndarray, each shape (n_atoms_i,), int32
sigma : float
cut_start : float, default 1.0
use_atm : bool, default False

Returns
-------
ndarray, shape (D, D), float64
    D = sum_i n_atoms_i * 3.  Full symmetric matrix.
)pbdoc"
    );

    m.def(
        "kernel_gaussian_full",
        &kernel_gaussian_full_py,
        py::arg("coords_A_list"),
        py::arg("z_A_list"),
        py::arg("coords_B_list"),
        py::arg("z_B_list"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"pbdoc(
Compute the full combined energy+force FCHL18 kernel (asymmetric).

Shape: (N_A*(1+D_A), N_B*(1+D_B)) where D_i = sum_mol n_atoms_i * 3.

Block layout:
  [0:N_A,  0:N_B ]  scalar   K[a,b]
  [0:N_A,  N_B:  ]  jac_t    dK/dR_B
  [N_A:,   0:N_B ]  jac      dK/dR_A
  [N_A:,   N_B:  ]  hessian  d²K/dR_A dR_B

Parameters
----------
coords_A_list, z_A_list : N_A query molecules (raw coords + charges)
coords_B_list, z_B_list : N_B training molecules
sigma : float
cut_start : float, default 1.0
use_atm : bool, default False

Returns
-------
ndarray, shape (N_A+D_A_total, N_B+D_B_total), float64
)pbdoc"
    );

    m.def(
        "kernel_gaussian_full_symm",
        &kernel_gaussian_full_symm_py,
        py::arg("coords_list"),
        py::arg("z_list"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"pbdoc(
Compute the full combined energy+force FCHL18 kernel (symmetric training set).

Shape: (N*(1+D), N*(1+D)) where D = sum_i n_atoms_i * 3. Full matrix filled.

Parameters
----------
coords_list, z_list : N training molecules
sigma : float
cut_start : float, default 1.0
use_atm : bool, default False

Returns
-------
ndarray, shape (N+D, N+D), float64
)pbdoc"
    );

    m.def(
        "kernel_gaussian_full_symm_rfp",
        &kernel_gaussian_full_symm_rfp_py,
        py::arg("coords_list"),
        py::arg("z_list"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"pbdoc(
Compute the full combined energy+force FCHL18 kernel (symmetric, RFP format).

Output length = BIG*(BIG+1)//2 where BIG = N + D.
Compatible with cho_solve_rfp.

Parameters
----------
coords_list, z_list : N training molecules
sigma : float
cut_start : float, default 1.0
use_atm : bool, default False

Returns
-------
ndarray, shape (BIG*(BIG+1)//2,), float64
    Upper-triangle RFP-packed full kernel.
)pbdoc"
    );
}
