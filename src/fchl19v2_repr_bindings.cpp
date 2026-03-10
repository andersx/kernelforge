// C++ standard library
#include <stdexcept>
#include <vector>

// Third-party libraries
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Project headers
#include "fchl19v2_repr.hpp"

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
    if (buf.ndim() != 1) throw std::invalid_argument("array must be 1D");
    if (expected != SIZE_MAX && static_cast<size_t>(buf.shape(0)) != expected)
        throw std::invalid_argument("unexpected array length");
    auto r = buf.unchecked<1>();
    return std::vector<int>(r.data(0), r.data(0) + r.shape(0));
}

// Build Rs2, Rs3 grids.
// Rs2: linspace(0, rcut, 1+nRs2)[1:] = rcut*i/nRs2 for i=1..nRs2
// Rs3: linspace(0, acut, 1+nRs3)[1:] = acut*i/nRs3 for i=1..nRs3
// Rs3_minus: linspace(0, acut, 1+nRs3_minus)[1:] (only for SplitR variants)
static void build_grids(
    int nRs2, int nRs3, int nRs3_minus, double rcut, double acut,
    std::vector<double> &Rs2, std::vector<double> &Rs3, std::vector<double> &Rs3_minus
) {
    Rs2.clear();
    Rs3.clear();
    Rs3_minus.clear();
    Rs2.reserve(nRs2);
    Rs3.reserve(nRs3);
    for (int i = 1; i <= nRs2; ++i)
        Rs2.push_back(rcut * double(i) / double(nRs2));
    for (int i = 1; i <= nRs3; ++i)
        Rs3.push_back(acut * double(i) / double(nRs3));
    if (nRs3_minus > 0) {
        Rs3_minus.reserve(nRs3_minus);
        for (int i = 1; i <= nRs3_minus; ++i)
            Rs3_minus.push_back(acut * double(i) / double(nRs3_minus));
    }
}

// Compute nabasis from type and nFourier/nCosine
static int compute_nabasis(
    const std::string &three_body_type, int nFourier, int nCosine
) {
    // Cosine variants use nCosine (if >0), else nFourier as fallback
    if (three_body_type == "odd_fourier_rbar" || three_body_type == "odd_fourier_split_r") {
        return 2 * nFourier;
    }
    // CosineSeries variants
    return (nCosine > 0) ? nCosine : 2 * nFourier;
}

// ---- generate (forward only) ----
static py::array_t<double> generate_py(
    const py::array &coords,
    const py::array &nuclear_z,
    std::vector<int> elements,
    int nRs2,
    int nRs3,
    int nRs3_minus,
    int nFourier,
    int nCosine,
    double eta2,
    double eta3,
    double eta3_minus,
    double zeta,
    double rcut,
    double acut,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight,
    const std::string &two_body_type_str,
    const std::string &three_body_type_str
) {
    const size_t natoms = static_cast<size_t>(nuclear_z.cast<py::array>().shape(0));
    if (natoms == 0) throw std::invalid_argument("n_atoms must be > 0");

    std::vector<double> coords_v = as_coords_vector(coords, natoms);
    std::vector<int> z_v = as_int_vector(nuclear_z, natoms);
    if (elements.empty()) elements = {1, 6, 7, 8, 16};

    std::vector<double> Rs2_v, Rs3_v, Rs3_minus_v;
    build_grids(nRs2, nRs3, nRs3_minus, rcut, acut, Rs2_v, Rs3_v, Rs3_minus_v);

    const auto tb_type = kf::fchl19v2::two_body_type_from_string(two_body_type_str);
    const auto ab_type = kf::fchl19v2::three_body_type_from_string(three_body_type_str);

    const int nabasis = compute_nabasis(three_body_type_str, nFourier, nCosine);
    const double w3 = std::sqrt(eta3 / M_PI) * three_body_weight;

    const size_t rep_size = kf::fchl19v2::compute_rep_size(
        elements.size(), Rs2_v.size(), Rs3_v.size(), static_cast<size_t>(nabasis), ab_type,
        Rs3_minus_v.size()
    );

    std::vector<double> rep;
    {
        py::gil_scoped_release release;
        kf::fchl19v2::generate(
            coords_v, z_v, elements, Rs2_v, Rs3_v, Rs3_minus_v, eta2, eta3, eta3_minus, zeta,
            rcut, acut, two_body_decay, three_body_decay, w3, tb_type, ab_type, nabasis, rep
        );
    }

    py::array_t<double> out(
        {static_cast<py::ssize_t>(natoms), static_cast<py::ssize_t>(rep_size)}
    );
    auto o = out.mutable_unchecked<2>();
    for (size_t i = 0; i < natoms; ++i)
        for (size_t j = 0; j < rep_size; ++j)
            o(i, j) = rep[i * rep_size + j];
    return out;
}

// ---- generate_and_gradients ----
static py::tuple generate_and_gradients_py(
    const py::array &coords,
    const py::array &nuclear_z,
    std::vector<int> elements,
    int nRs2,
    int nRs3,
    int nRs3_minus,
    int nFourier,
    int nCosine,
    double eta2,
    double eta3,
    double eta3_minus,
    double zeta,
    double rcut,
    double acut,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight,
    const std::string &two_body_type_str,
    const std::string &three_body_type_str
) {
    const size_t natoms = static_cast<size_t>(nuclear_z.cast<py::array>().shape(0));
    if (natoms == 0) throw std::invalid_argument("n_atoms must be > 0");

    std::vector<double> coords_v = as_coords_vector(coords, natoms);
    std::vector<int> z_v = as_int_vector(nuclear_z, natoms);
    if (elements.empty()) elements = {1, 6, 7, 8, 16};

    std::vector<double> Rs2_v, Rs3_v, Rs3_minus_v;
    build_grids(nRs2, nRs3, nRs3_minus, rcut, acut, Rs2_v, Rs3_v, Rs3_minus_v);

    const auto tb_type = kf::fchl19v2::two_body_type_from_string(two_body_type_str);
    const auto ab_type = kf::fchl19v2::three_body_type_from_string(three_body_type_str);

    const int nabasis = compute_nabasis(three_body_type_str, nFourier, nCosine);
    const double w3 = std::sqrt(eta3 / M_PI) * three_body_weight;

    const size_t rep_size = kf::fchl19v2::compute_rep_size(
        elements.size(), Rs2_v.size(), Rs3_v.size(), static_cast<size_t>(nabasis), ab_type,
        Rs3_minus_v.size()
    );

    std::vector<double> rep, grad;
    {
        py::gil_scoped_release release;
        kf::fchl19v2::generate_and_gradients(
            coords_v, z_v, elements, Rs2_v, Rs3_v, Rs3_minus_v, eta2, eta3, eta3_minus, zeta,
            rcut, acut, two_body_decay, three_body_decay, w3, tb_type, ab_type, nabasis, rep, grad
        );
    }

    py::array_t<double> rep_arr(
        {static_cast<py::ssize_t>(natoms), static_cast<py::ssize_t>(rep_size)}
    );
    auto R = rep_arr.mutable_unchecked<2>();
    for (size_t i = 0; i < natoms; ++i)
        for (size_t j = 0; j < rep_size; ++j)
            R(i, j) = rep[i * rep_size + j];

    py::array_t<double> grad_arr(
        {static_cast<py::ssize_t>(natoms), static_cast<py::ssize_t>(rep_size),
         static_cast<py::ssize_t>(3 * natoms)}
    );
    auto G = grad_arr.mutable_unchecked<3>();
    size_t idx = 0;
    for (size_t i = 0; i < natoms; ++i)
        for (size_t j = 0; j < rep_size; ++j)
            for (size_t a = 0; a < natoms; ++a)
                for (int d = 0; d < 3; ++d)
                    G(i, j, 3 * a + d) = grad[idx++];

    return py::make_tuple(rep_arr, grad_arr);
}

// ---- compute_rep_size (Python-visible) ----
static size_t compute_rep_size_py(
    size_t nelements, size_t nbasis2, size_t nbasis3, size_t nabasis,
    const std::string &three_body_type_str, size_t nbasis3_minus
) {
    const auto ab_type = kf::fchl19v2::three_body_type_from_string(three_body_type_str);
    return kf::fchl19v2::compute_rep_size(nelements, nbasis2, nbasis3, nabasis, ab_type,
                                          nbasis3_minus);
}

PYBIND11_MODULE(fchl19v2_repr, m) {
    m.doc() = "FCHL19v2 representation with selectable two-body and three-body basis functions";

    m.def(
        "compute_rep_size",
        &compute_rep_size_py,
        py::arg("nelements"),
        py::arg("nbasis2"),
        py::arg("nbasis3"),
        py::arg("nabasis"),
        py::arg("three_body_type") = "odd_fourier_rbar",
        py::arg("nbasis3_minus") = 0,
        "Compute the total representation size per atom."
    );

    m.def(
        "generate",
        &generate_py,
        py::arg("coords"),
        py::arg("nuclear_z"),
        py::arg("elements") = std::vector<int>{1, 6, 7, 8, 16},
        py::arg("nRs2") = 24,
        py::arg("nRs3") = 20,
        py::arg("nRs3_minus") = 0,
        py::arg("nFourier") = 1,
        py::arg("nCosine") = 0,
        py::arg("eta2") = 0.32,
        py::arg("eta3") = 2.7,
        py::arg("eta3_minus") = 2.7,
        py::arg("zeta") = M_PI,
        py::arg("rcut") = 8.0,
        py::arg("acut") = 8.0,
        py::arg("two_body_decay") = 1.8,
        py::arg("three_body_decay") = 0.57,
        py::arg("three_body_weight") = 13.4,
        py::arg("two_body_type") = "log_normal",
        py::arg("three_body_type") = "odd_fourier_rbar",
        R"pbdoc(
Generate FCHL19v2 representation.

two_body_type: "log_normal" | "gaussian_r" | "gaussian_log_r" | "gaussian_r_no_pow" | "bessel"
three_body_type: "odd_fourier_rbar" | "cosine_rbar" | "odd_fourier_split_r" |
                 "cosine_split_r" | "cosine_split_r_no_atm"

Returns ndarray shape (n_atoms, rep_size).
)pbdoc"
    );

    m.def(
        "generate_and_gradients",
        &generate_and_gradients_py,
        py::arg("coords"),
        py::arg("nuclear_z"),
        py::arg("elements") = std::vector<int>{1, 6, 7, 8, 16},
        py::arg("nRs2") = 24,
        py::arg("nRs3") = 20,
        py::arg("nRs3_minus") = 0,
        py::arg("nFourier") = 1,
        py::arg("nCosine") = 0,
        py::arg("eta2") = 0.32,
        py::arg("eta3") = 2.7,
        py::arg("eta3_minus") = 2.7,
        py::arg("zeta") = M_PI,
        py::arg("rcut") = 8.0,
        py::arg("acut") = 8.0,
        py::arg("two_body_decay") = 1.8,
        py::arg("three_body_decay") = 0.57,
        py::arg("three_body_weight") = 13.4,
        py::arg("two_body_type") = "log_normal",
        py::arg("three_body_type") = "odd_fourier_rbar",
        R"pbdoc(
Generate FCHL19v2 representation and its Jacobian wrt atomic coordinates.

Returns (rep, grad) with shapes (n_atoms, rep_size) and (n_atoms, rep_size, n_atoms*3).
)pbdoc"
    );
}
