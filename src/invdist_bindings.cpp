// invdist_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "invdist.hpp"
#include <vector>
#include <stdexcept>

namespace py = pybind11;

static void check_R(const py::array& R) {
    if (R.ndim() != 2 || R.shape(1) != 3) {
        throw std::runtime_error("R must be a 2D array of shape (N,3).");
    }
    if (R.shape(0) < 2) {
        throw std::runtime_error("Need at least 2 atoms (N>=2).");
    }
}

py::array_t<double> py_inverse_distance_upper(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    double eps = 1e-12)
{
    check_R(R);
    const std::size_t N = static_cast<std::size_t>(R.shape(0));
    const std::size_t M = invdist::num_pairs(N);

    // flatten coords
    std::vector<double> Rflat(3 * N);
    auto Rv = R.unchecked<2>();
    for (std::size_t i = 0; i < N; ++i) {
        Rflat[3*i+0] = Rv(i,0);
        Rflat[3*i+1] = Rv(i,1);
        Rflat[3*i+2] = Rv(i,2);
    }

    py::array_t<double> x((py::ssize_t)M);
    auto xv = x.mutable_unchecked<1>();

    invdist::inverse_distance_upper(
        Rflat.data(), N, eps,
        xv.mutable_data(0)
    );

    return x;
}

py::tuple py_inverse_distance_upper_and_jacobian(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    double eps = 1e-12)
{
    check_R(R);
    const std::size_t N = static_cast<std::size_t>(R.shape(0));
    const std::size_t M = invdist::num_pairs(N);
    const std::size_t D = 3 * N;

    std::vector<double> Rflat(3 * N);
    auto Rv = R.unchecked<2>();
    for (std::size_t i = 0; i < N; ++i) {
        Rflat[3*i+0] = Rv(i,0);
        Rflat[3*i+1] = Rv(i,1);
        Rflat[3*i+2] = Rv(i,2);
    }

    py::array_t<double> x((py::ssize_t)M);
    py::array_t<double> J({(py::ssize_t)M, (py::ssize_t)D});

    auto xv = x.mutable_unchecked<1>();
    auto Jv = J.mutable_unchecked<2>();

    invdist::inverse_distance_upper_and_jacobian(
        Rflat.data(), N, eps,
        xv.mutable_data(0),
        Jv.mutable_data(0,0)
    );

    return py::make_tuple(x, J);
}

PYBIND11_MODULE(_invdist, m) {
    m.doc() = "Upper-triangle inverse distances and Jacobians";

    m.def("num_pairs", &invdist::num_pairs);

    // 1) x only (shape M)
    m.def("inverse_distance_upper",
          &py_inverse_distance_upper,
          py::arg("R"), py::arg("eps") = 1e-12,
          "Return x (M,), the strict upper-triangle inverse distances.");

    // 2) x and J (shapes M, and M x 3N)
    m.def("inverse_distance_upper_and_jacobian",
          &py_inverse_distance_upper_and_jacobian,
          py::arg("R"), py::arg("eps") = 1e-12,
          "Return (x, J) where x is (M,) upper-triangle and J is (M, 3N).");
}

