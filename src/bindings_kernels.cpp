#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "aligned_alloc64.hpp"
#include "kernels.hpp"

#include <stdexcept>

namespace py = pybind11;

static void check_2d(const py::array& X) {
    if (X.ndim() != 2) {
        throw std::runtime_error("X must be 2D (n, rep_size) in row-major (C) order.");
    }
}

py::array_t<double> kernel_symm_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    double alpha
) {
    check_2d(X);
    auto bufX = X.request();

    int n        = static_cast<int>(bufX.shape[0]); // rows
    int rep_size = static_cast<int>(bufX.shape[1]); // cols
    double* Xptr = static_cast<double*>(bufX.ptr);

    // Allocate aligned K (row-major, n x n)
    std::size_t nelems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    double* Kptr = aligned_alloc_64(nelems);

    // Capsule to free aligned memory when NumPy array is GC’d
    auto capsule = py::capsule(Kptr, [](void* p){ aligned_free_64(p); });

    // Make a NumPy view over Kptr (row-major)
    py::array_t<double> K(
        {n, n},
        {static_cast<py::ssize_t>(n) * static_cast<py::ssize_t>(sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        Kptr,
        capsule
    );

    // Compute
    kernel_symm(Xptr, n, rep_size, alpha, Kptr);

    return K;
}

py::array_t<double> kernel_asymm_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X1,  // (n1,d)
    py::array_t<double, py::array::c_style | py::array::forcecast> X2,  // (n2,d)
    double alpha
) {
    check_2d(X1);
    check_2d(X2);

    const std::size_t n1 = static_cast<std::size_t>(X1.shape(0));
    const std::size_t d1 = static_cast<std::size_t>(X1.shape(1));
    const std::size_t n2 = static_cast<std::size_t>(X2.shape(0));
    const std::size_t d2 = static_cast<std::size_t>(X2.shape(1));
    if (d1 != d2) throw std::runtime_error("X1.shape[1] must equal X2.shape[1] (feature dimension d).");

    // Make aligned (n2 x n1) output
    const std::size_t nelems = n2 * n1;
    double* Kptr = aligned_alloc_64(nelems);
    auto capsule = py::capsule(Kptr, [](void* p){ aligned_free_64(p); });

    py::array_t<double> K(
        { static_cast<py::ssize_t>(n1), static_cast<py::ssize_t>(n2) },
        { static_cast<py::ssize_t>(n2 * sizeof(double)), static_cast<py::ssize_t>(sizeof(double)) },
        Kptr,
        capsule
    );

    auto X1v = X1.unchecked<2>();
    auto X2v = X2.unchecked<2>();

    kernel_asymm(
        X1v.data(0,0), X2v.data(0,0),
        n1, n2, d1,
        alpha,
        Kptr
    );

    return K;
}


PYBIND11_MODULE(_kernels, m) {
    m.doc() = "Symmetric Gaussian kernel via BLAS (row-major), with 64-byte aligned output buffer.";
    m.def("kernel_symm", &kernel_symm_py,
          py::arg("X"), py::arg("alpha"),
          "Compute K = exp(alpha*(||x_i||^2 + ||x_j||^2 - 2 x_i·x_j)) over the lower triangle.\n"
          "X is (n, rep_size) in row-major; returns K as an (n,n) NumPy array.");
    m.def("kernel_asymm", &kernel_asymm_py,
          py::arg("X1"), py::arg("X2"), py::arg("alpha"),
          "Return K (n2, n1) where K[i2,i1] = exp(alpha*(||x2||^2 + ||x1||^2 - 2 x2·x1)).");
}

