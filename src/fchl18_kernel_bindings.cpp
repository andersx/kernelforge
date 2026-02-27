// C++ standard library
#include <stdexcept>
#include <vector>

// Third-party libraries
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Project headers
#include "fchl18_kernel.hpp"

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
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<int>(src[i]);
    return out;
}

static std::vector<int> as_int_vector_1d(
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &arr
) {
    const std::size_t n = static_cast<std::size_t>(arr.size());
    std::vector<int> out(n);
    const int32_t *src = arr.data();
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<int>(src[i]);
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
    const py::array_t<double,  py::array::c_style | py::array::forcecast> &x1,
    const py::array_t<double,  py::array::c_style | py::array::forcecast> &x2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n1,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n2,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &nn1,
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
    int    fourier_order
) {
    if (x1.ndim() != 4) throw std::invalid_argument("x1 must be 4-D");
    if (x2.ndim() != 4) throw std::invalid_argument("x2 must be 4-D");
    if (x1.shape(2) != 5) throw std::invalid_argument("x1 dim 2 must be 5");
    if (x2.shape(2) != 5) throw std::invalid_argument("x2 dim 2 must be 5");

    const int nm1       = static_cast<int>(x1.shape(0));
    const int max_size1 = static_cast<int>(x1.shape(1));
    const int nm2       = static_cast<int>(x2.shape(0));
    const int max_size2 = static_cast<int>(x2.shape(1));

    auto x1_v   = as_double_vector(x1);
    auto x2_v   = as_double_vector(x2);
    auto n1_v   = as_int_vector_1d(n1);
    auto n2_v   = as_int_vector_1d(n2);
    auto nn1_v  = as_int_vector_2d(nn1);
    auto nn2_v  = as_int_vector_2d(nn2);

    py::array_t<double> K({(py::ssize_t)nm1, (py::ssize_t)nm2});

    {
        py::gil_scoped_release release;
        kf::fchl18::kernel_gaussian(
            x1_v, x2_v, n1_v, n2_v, nn1_v, nn2_v,
            nm1, nm2, max_size1, max_size2,
            sigma,
            two_body_scaling, two_body_width, two_body_power,
            three_body_scaling, three_body_width, three_body_power,
            cut_start, cut_distance, fourier_order,
            K.mutable_data()
        );
    }

    return K;
}

// ---------------------------------------------------------------------------
// kernel_gaussian_symm: symmetric (nm, nm) kernel matrix
// ---------------------------------------------------------------------------
static py::array_t<double> kernel_gaussian_symm_py(
    const py::array_t<double,  py::array::c_style | py::array::forcecast> &x,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &n,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &nn,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int    fourier_order
) {
    if (x.ndim() != 4) throw std::invalid_argument("x must be 4-D");
    if (x.shape(2) != 5) throw std::invalid_argument("x dim 2 must be 5");

    const int nm       = static_cast<int>(x.shape(0));
    const int max_size = static_cast<int>(x.shape(1));

    auto x_v  = as_double_vector(x);
    auto n_v  = as_int_vector_1d(n);
    auto nn_v = as_int_vector_2d(nn);

    py::array_t<double> K({(py::ssize_t)nm, (py::ssize_t)nm});

    {
        py::gil_scoped_release release;
        kf::fchl18::kernel_gaussian_symm(
            x_v, n_v, nn_v,
            nm, max_size,
            sigma,
            two_body_scaling, two_body_width, two_body_power,
            three_body_scaling, three_body_width, three_body_power,
            cut_start, cut_distance, fourier_order,
            K.mutable_data()
        );
    }

    return K;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(fchl18_kernel, m) {
    m.doc() = "FCHL18 Gaussian kernel functions";

    const char *kg_doc = R"pbdoc(
Compute the FCHL18 Gaussian kernel matrix K[a, b] between two sets of molecules.

K[a, b] = sum_{i in mol_a, j in mol_b: Z_i == Z_j}
              exp( -(s_ii + s_jj - 2*s_ij) / sigma^2 )

where s_ij is the FCHL18 scalar product between centre-atoms i and j.
No alchemical interpolation (only matching nuclear charges contribute).

Parameters
----------
x1, x2 : ndarray, shape (nm, max_size, 5, max_size), float64
    Representations from fchl18_repr.generate().
n1, n2 : ndarray, shape (nm,), int32
    Number of real atoms per molecule.
nn1, nn2 : ndarray, shape (nm, max_size), int32
    Number of neighbours per atom (including self at index 0).
sigma : float
    Gaussian kernel width.
two_body_scaling : float, default 2.0
    Overall weight for two-body (radial) terms.
two_body_width : float, default 0.1
    Gaussian width for the radial distance kernel [Angstrom].
two_body_power : float, default 6.0
    Power-law exponent for two-body weights: ksi = cut(r) / r^power.
three_body_scaling : float, default 2.0
    Overall weight for three-body (angular) terms.
three_body_width : float, default 3.0
    Angular Gaussian width for Fourier expansion normalisation.
three_body_power : float, default 3.0
    Power-law exponent in the Axilrod-Teller-Muto three-body weight.
cut_start : float, default 0.5
    Fraction of cut_distance at which the cutoff damping starts.
cut_distance : float, default 1e6
    Neighbour cutoff radius (must match representation) [Angstrom].
fourier_order : int, default 2
    Truncation order for the three-body Fourier expansion.

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
        py::arg("two_body_scaling")   = 2.0,
        py::arg("two_body_width")     = 0.1,
        py::arg("two_body_power")     = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width")   = 3.0,
        py::arg("three_body_power")   = 3.0,
        py::arg("cut_start")          = 0.5,
        py::arg("cut_distance")       = 1e6,
        py::arg("fourier_order")      = 2,
        kg_doc
    );

    const char *kgs_doc = R"pbdoc(
Compute the symmetric FCHL18 Gaussian kernel matrix K[a, b] = K[b, a].

Same kernel as kernel_gaussian but for K(X, X). Exploits symmetry:
only the upper triangle is computed and the result is mirrored.

Parameters
----------
x : ndarray, shape (nm, max_size, 5, max_size), float64
n : ndarray, shape (nm,), int32
nn : ndarray, shape (nm, max_size), int32
sigma : float
(remaining hyperparameters same as kernel_gaussian)

Returns
-------
ndarray, shape (nm, nm), float64
)pbdoc";

    m.def(
        "kernel_gaussian_symm",
        &kernel_gaussian_symm_py,
        py::arg("x"),
        py::arg("n"),
        py::arg("nn"),
        py::arg("sigma"),
        py::arg("two_body_scaling")   = 2.0,
        py::arg("two_body_width")     = 0.1,
        py::arg("two_body_power")     = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width")   = 3.0,
        py::arg("three_body_power")   = 3.0,
        py::arg("cut_start")          = 0.5,
        py::arg("cut_distance")       = 1e6,
        py::arg("fourier_order")      = 2,
        kgs_doc
    );
}
