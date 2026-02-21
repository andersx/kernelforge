// C++ standard library
#include <stdexcept>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// Project headers
#include "aligned_alloc64.hpp"
#include "rff_features.hpp"

namespace py = pybind11;

// Forward declaration — defined in rff_elemental_bindings.cpp
void register_rff_elemental(py::module_ &m);

// ---- rff_features binding ---------------------------------------------------

static py::array_t<double> py_rff_features(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr) {

    if (X_arr.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N, rep_size)");
    if (W_arr.ndim() != 2)
        throw std::invalid_argument("W must be 2D (rep_size, D)");
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("b must be 1D (D,)");

    const auto N        = static_cast<std::size_t>(X_arr.shape(0));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(1));
    const auto D        = static_cast<std::size_t>(b_arr.shape(0));

    if (static_cast<std::size_t>(W_arr.shape(0)) != rep_size)
        throw std::invalid_argument("W.shape[0] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(1)) != D)
        throw std::invalid_argument("W.shape[1] must equal b.shape[0] (D)");

    double *Zptr = aligned_alloc_64(N * D);
    auto capsule = py::capsule(Zptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_features(
        X_arr.data(), W_arr.data(), b_arr.data(),
        N, rep_size, D, Zptr);

    return py::array_t<double>(
        {static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        Zptr, capsule);
}

// ---- Module definition ------------------------------------------------------

PYBIND11_MODULE(kitchen_sinks, m) {
    m.doc() = "Random Fourier Features (Kitchen Sinks) module";

    m.def("rff_features", &py_rff_features,
          R"doc(
Compute Random Fourier Features.

Z = sqrt(2/D) * cos(X @ W + b)

Parameters
----------
X : ndarray, shape (N, rep_size)
    Input feature matrix.
W : ndarray, shape (rep_size, D)
    Random weight matrix.
b : ndarray, shape (D,)
    Random bias vector.

Returns
-------
Z : ndarray, shape (N, D)
    Random Fourier feature matrix.
)doc",
          py::arg("X"), py::arg("W"), py::arg("b"));

    // Elemental functions registered from rff_elemental_bindings.cpp
    register_rff_elemental(m);
}
