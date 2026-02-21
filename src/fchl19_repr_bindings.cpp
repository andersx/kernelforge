// C++ standard library
#include <stdexcept>
#include <vector>

// Third-party libraries
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Project headers
#include "fchl19_repr.hpp"

namespace py = pybind11;

// Convert a 2D NumPy array (n,3) or 1D (n*3,) to std::vector<double>
static std::vector<double> as_coords_vector(const py::array &arr, size_t natoms) {
    if (arr.ndim() == 2) {
        if (static_cast<size_t>(arr.shape(0)) != natoms || arr.shape(1) != 3)
            throw std::invalid_argument("coords must have shape (n_atoms, 3)");
        std::vector<double> out(natoms * 3);
        auto buf = arr.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto r = buf.unchecked<2>();
        for (size_t i = 0; i < natoms; ++i) {
            out[3 * i + 0] = r(i, 0);
            out[3 * i + 1] = r(i, 1);
            out[3 * i + 2] = r(i, 2);
        }
        return out;
    } else if (arr.ndim() == 1) {
        if (static_cast<size_t>(arr.shape(0)) != natoms * 3)
            throw std::invalid_argument("coords 1D length must be n_atoms*3");
        auto buf = arr.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto r = buf.unchecked<1>();
        return std::vector<double>(r.data(0), r.data(0) + natoms * 3);
    } else {
        throw std::invalid_argument("coords must be a 1D or 2D NumPy array");
    }
}

static std::vector<int> as_int_vector(const py::array &arr, size_t expected = SIZE_MAX) {
    auto buf = arr.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    if (buf.ndim() != 1)
        throw std::invalid_argument("array must be 1D");
    if (expected != SIZE_MAX && static_cast<size_t>(buf.shape(0)) != expected)
        throw std::invalid_argument("unexpected array length");
    auto r = buf.unchecked<1>();
    return std::vector<int>(r.data(0), r.data(0) + r.shape(0));
}

py::array_t<double> generate_fchl_acsf_py(const py::array &coords,     // (n,3) or (3n,)
                                          const py::array &nuclear_z,  // (n,)
                                          std::vector<int> elements, int nRs2, int nRs3,
                                          int nFourier, double eta2, double eta3, double zeta,
                                          double rcut, double acut, double two_body_decay,
                                          double three_body_decay, double three_body_weight) {
    const size_t natoms = static_cast<size_t>(nuclear_z.cast<py::array>().shape(0));
    if (natoms == 0)
        throw std::invalid_argument("n_atoms must be > 0");

    std::vector<double> coords_v = as_coords_vector(coords, natoms);
    std::vector<int> z_v = as_int_vector(nuclear_z, natoms);

    // Default elements if not provided
    if (elements.empty()) {
        elements = {1, 6, 7, 8, 16};
    }

    // Construct Rs2, Rs3, Ts as in Python version
    std::vector<double> Rs2_v, Rs3_v, Ts_v;
    Rs2_v.reserve(nRs2);
    Rs3_v.reserve(nRs3);

    for (int i = 1; i <= nRs2; ++i) {
        Rs2_v.push_back(rcut * static_cast<double>(i) / static_cast<double>(nRs2 + 1));
    }
    for (int i = 1; i <= nRs3; ++i) {
        Rs3_v.push_back(acut * static_cast<double>(i) / static_cast<double>(nRs3 + 1));
    }
    Ts_v.reserve(2 * nFourier);
    for (int i = 0; i < 2 * nFourier; ++i) {
        Ts_v.push_back(M_PI * static_cast<double>(i) / static_cast<double>(2 * nFourier - 1));
    }

    // Normalization constant for three-body
    double norm_three_body_weight = std::sqrt(eta3 / M_PI) * three_body_weight;

    const size_t rep_size =
        kf::fchl19::compute_rep_size(elements.size(), Rs2_v.size(), Rs3_v.size(), Ts_v.size());

    py::array_t<double> out({static_cast<py::ssize_t>(natoms), static_cast<py::ssize_t>(rep_size)});
    auto out_mut = out.mutable_unchecked<2>();

    std::vector<double> rep;
    {
        py::gil_scoped_release release;
        kf::fchl19::generate_fchl_acsf(coords_v, z_v, elements, Rs2_v, Rs3_v, Ts_v, eta2, eta3, zeta,
                                       rcut, acut, two_body_decay, three_body_decay,
                                       norm_three_body_weight, rep);
    }

    if (rep.size() != natoms * rep_size)
        throw std::runtime_error("internal error: unexpected rep size");
    const double *src = rep.data();
    for (size_t i = 0; i < natoms; ++i)
        for (size_t j = 0; j < rep_size; ++j)
            out_mut(i, j) = src[i * rep_size + j];

    return out;
}

// Build basis arrays like in your Python: linspace(0,rcut,1+n)[1:]
static void build_basis_from_sizes(int nRs2, int nRs3, int nFourier, double rcut, double acut,
                                   std::vector<double> &Rs2, std::vector<double> &Rs3,
                                   std::vector<double> &Ts) {
    Rs2.clear();
    Rs3.clear();
    Ts.clear();
    Rs2.reserve(nRs2);
    Rs3.reserve(nRs3);
    Ts.reserve(2 * nFourier);
    for (int i = 1; i <= nRs2; ++i)
        Rs2.push_back(rcut * double(i) / double(nRs2 + 1));
    for (int i = 1; i <= nRs3; ++i)
        Rs3.push_back(acut * double(i) / double(nRs3 + 1));
    const int nT = std::max(2 * nFourier, 2);  // ensure even >=2
    for (int i = 0; i < nT; ++i) {
        double x = (nT == 1 ? 0.0 : (double(i) / double(nT - 1)));
        Ts.push_back(M_PI * x);
    }
}

static py::tuple generate_fchl_acsf_rep_and_grad_py(const py::array &coords,     // (n,3)
                                                    const py::array &nuclear_z,  // (n,)
                                                    std::vector<int> elements, int nRs2, int nRs3,
                                                    int nFourier, double eta2, double eta3,
                                                    double zeta, double rcut, double acut,
                                                    double two_body_decay, double three_body_decay,
                                                    double three_body_weight) {
    const std::size_t natoms = static_cast<std::size_t>(nuclear_z.cast<py::array>().shape(0));
    if (natoms == 0)
        throw std::invalid_argument("n_atoms must be > 0");

    std::vector<double> coords_v = as_coords_vector(coords, natoms);
    std::vector<int> z_v = as_int_vector(nuclear_z, natoms);
    if (elements.empty())
        elements = {1, 6, 7, 8, 16};

    std::vector<double> Rs2_v, Rs3_v, Ts_v;
    build_basis_from_sizes(nRs2, nRs3, nFourier, rcut, acut, Rs2_v, Rs3_v, Ts_v);

    // rescale three-body weight
    const double w3 = std::sqrt(eta3 / M_PI) * three_body_weight;

    const std::size_t rep_size =
        kf::fchl19::compute_rep_size(elements.size(), Rs2_v.size(), Rs3_v.size(), Ts_v.size());

    std::vector<double> rep, grad;
    {
        py::gil_scoped_release release;
        kf::fchl19::generate_fchl_acsf_and_gradients(coords_v, z_v, elements, Rs2_v, Rs3_v, Ts_v, eta2,
                                                 eta3, zeta, rcut, acut, two_body_decay,
                                                 three_body_decay, w3, rep, grad);
    }

    // Wrap outputs
    py::array_t<double> rep_arr(
        py::array::ShapeContainer{(py::ssize_t)natoms, (py::ssize_t)rep_size});
    auto R = rep_arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < natoms; ++i)
        for (std::size_t j = 0; j < rep_size; ++j)
            R(i, j) = rep[i * rep_size + j];

    py::array_t<double> grad_arr(py::array::ShapeContainer{
        (py::ssize_t)natoms, (py::ssize_t)rep_size, (py::ssize_t)(3 * natoms)});
    auto G = grad_arr.mutable_unchecked<3>();
    std::size_t idx = 0;
    for (std::size_t i = 0; i < natoms; ++i)
        for (std::size_t j = 0; j < rep_size; ++j)
            for (std::size_t a = 0; a < natoms; ++a)
                for (int d = 0; d < 3; ++d)
                    G(i, j, 3 * a + d) = grad[idx++];

    return py::make_tuple(rep_arr, grad_arr);
}

PYBIND11_MODULE(fchl19_repr, m) {
    m.doc() = "FCHL19 representation and gradients";

    m.def("compute_rep_size", &kf::fchl19::compute_rep_size, py::arg("nelements"), py::arg("nbasis2"),
          py::arg("nbasis3"), py::arg("nabasis"),
          "Compute the total representation size per-atom.");

    m.def("generate_fchl_acsf", &generate_fchl_acsf_py, py::arg("coords"), py::arg("nuclear_z"),
          py::arg("elements") = std::vector<int>{1, 6, 7, 8, 16}, py::arg("nRs2") = 24,
          py::arg("nRs3") = 20, py::arg("nFourier") = 1, py::arg("eta2") = 0.32,
          py::arg("eta3") = 2.7, py::arg("zeta") = M_PI, py::arg("rcut") = 8.0,
          py::arg("acut") = 8.0, py::arg("two_body_decay") = 1.8,
          py::arg("three_body_decay") = 0.57, py::arg("three_body_weight") = 13.4,
          R"pbdoc(
Generate FCHL-like ACSF representation.


Parameters
----------
coords : ndarray, shape (n_atoms, 3) or (3*n_atoms,)
Cartesian coordinates.
nuclear_z : ndarray of int, shape (n_atoms,)
Nuclear charges.
elements : list[int], default [1,6,7,8,16]
Unique element types present.
nRs2 : int, default 24
nRs3 : int, default 20
nFourier : int, default 1
Rs2 = linspace(0, rcut, 1+nRs2)[1:]
Rs3 = linspace(0, acut, 1+nRs3)[1:]
Ts = linspace(0, pi, 2*nFourier)
eta2 : float, default 0.32
eta3 : float, default 2.7
zeta : float, default pi
rcut : float, default 8.0
acut : float, default 8.0
two_body_decay : float, default 1.8
three_body_decay : float, default 0.57
three_body_weight : float, default 13.4, rescaled internally by sqrt(eta3/pi)


Returns
-------
ndarray, shape (n_atoms, rep_size)
Per-atom representation.
)pbdoc");

    m.def("generate_fchl_acsf_and_gradients", &generate_fchl_acsf_rep_and_grad_py,
          py::arg("coords"), py::arg("nuclear_z"),
          py::arg("elements") = std::vector<int>{1, 6, 7, 8, 16}, py::arg("nRs2") = 24,
          py::arg("nRs3") = 20, py::arg("nFourier") = 1, py::arg("eta2") = 0.32,
          py::arg("eta3") = 2.7, py::arg("zeta") = M_PI, py::arg("rcut") = 8.0,
          py::arg("acut") = 8.0, py::arg("two_body_decay") = 1.8,
          py::arg("three_body_decay") = 0.57, py::arg("three_body_weight") = 13.4,
          R"pbdoc(
Generate ACSF representation and its Jacobian with respect to atomic coordinates.

This version computes gradients by **central finite differences** over coordinates
(step size `h`, default 1e-6). It matches the representation built from internal
Rs2/Rs3/Ts (linspace) and rescales the three-body weight by sqrt(eta3/pi).

Returns
-------
(rep, grad)
  rep : (n_atoms, rep_size)
  grad: (n_atoms, rep_size, n_atoms, 3)
)pbdoc");
}
