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

// ---- rff_gradient binding ---------------------------------------------------

static py::array_t<double> py_rff_gradient(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr) {

    if (X_arr.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N, rep_size)");
    if (dX_arr.ndim() != 3)
        throw std::invalid_argument("dX must be 3D (N, rep_size, ncoords)");
    if (W_arr.ndim() != 2)
        throw std::invalid_argument("W must be 2D (rep_size, D)");
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("b must be 1D (D,)");

    const auto N        = static_cast<std::size_t>(X_arr.shape(0));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(1));
    const auto D        = static_cast<std::size_t>(b_arr.shape(0));
    const auto ncoords  = static_cast<std::size_t>(dX_arr.shape(2));

    if (static_cast<std::size_t>(dX_arr.shape(0)) != N)
        throw std::invalid_argument("dX.shape[0] must equal X.shape[0] (N)");
    if (static_cast<std::size_t>(dX_arr.shape(1)) != rep_size)
        throw std::invalid_argument("dX.shape[1] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(0)) != rep_size)
        throw std::invalid_argument("W.shape[0] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(1)) != D)
        throw std::invalid_argument("W.shape[1] must equal b.shape[0] (D)");

    const std::size_t total_grads = N * ncoords;
    double *Gptr = aligned_alloc_64(D * total_grads);
    auto capsule = py::capsule(Gptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gradient(
        X_arr.data(), dX_arr.data(), W_arr.data(), b_arr.data(),
        N, rep_size, D, ncoords, Gptr);

    return py::array_t<double>(
        {static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(total_grads)},
        {static_cast<py::ssize_t>(total_grads * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        Gptr, capsule);
}

// ---- rff_full binding -------------------------------------------------------

static py::array_t<double> py_rff_full(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr) {

    if (X_arr.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N, rep_size)");
    if (dX_arr.ndim() != 3)
        throw std::invalid_argument("dX must be 3D (N, rep_size, ncoords)");
    if (W_arr.ndim() != 2)
        throw std::invalid_argument("W must be 2D (rep_size, D)");
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("b must be 1D (D,)");

    const auto N        = static_cast<std::size_t>(X_arr.shape(0));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(1));
    const auto D        = static_cast<std::size_t>(b_arr.shape(0));
    const auto ncoords  = static_cast<std::size_t>(dX_arr.shape(2));

    if (static_cast<std::size_t>(dX_arr.shape(0)) != N)
        throw std::invalid_argument("dX.shape[0] must equal X.shape[0] (N)");
    if (static_cast<std::size_t>(dX_arr.shape(1)) != rep_size)
        throw std::invalid_argument("dX.shape[1] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(0)) != rep_size)
        throw std::invalid_argument("W.shape[0] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(1)) != D)
        throw std::invalid_argument("W.shape[1] must equal b.shape[0] (D)");

    const std::size_t total_rows = N + N * ncoords;
    double *Zptr = aligned_alloc_64(total_rows * D);
    auto capsule = py::capsule(Zptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_full(
        X_arr.data(), dX_arr.data(), W_arr.data(), b_arr.data(),
        N, rep_size, D, ncoords, Zptr);

    return py::array_t<double>(
        {static_cast<py::ssize_t>(total_rows), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        Zptr, capsule);
}

// ---- rff_gramian_symm binding -----------------------------------------------

static py::tuple py_rff_gramian_symm(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &Y_arr,
    std::size_t chunk_size) {

    if (X_arr.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N, rep_size)");
    if (W_arr.ndim() != 2)
        throw std::invalid_argument("W must be 2D (rep_size, D)");
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("b must be 1D (D,)");
    if (Y_arr.ndim() != 1)
        throw std::invalid_argument("Y must be 1D (N,)");

    const auto N        = static_cast<std::size_t>(X_arr.shape(0));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(1));
    const auto D        = static_cast<std::size_t>(b_arr.shape(0));

    if (static_cast<std::size_t>(W_arr.shape(0)) != rep_size)
        throw std::invalid_argument("W.shape[0] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(1)) != D)
        throw std::invalid_argument("W.shape[1] must equal b.shape[0] (D)");
    if (static_cast<std::size_t>(Y_arr.shape(0)) != N)
        throw std::invalid_argument("Y.shape[0] must equal X.shape[0] (N)");

    double *ZtZ_ptr = aligned_alloc_64(D * D);
    double *ZtY_ptr = aligned_alloc_64(D);
    auto cap_gram = py::capsule(ZtZ_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(ZtY_ptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gramian_symm(
        X_arr.data(), W_arr.data(), b_arr.data(), Y_arr.data(),
        N, rep_size, D, chunk_size,
        ZtZ_ptr, ZtY_ptr);

    auto ZtZ_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        ZtZ_ptr, cap_gram);

    auto ZtY_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtY_ptr, cap_proj);

    return py::make_tuple(ZtZ_out, ZtY_out);
}

// ---- rff_gradient_gramian_symm binding --------------------------------------

static py::tuple py_rff_gradient_gramian_symm(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &F_arr,
    std::size_t chunk_size) {

    if (X_arr.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N, rep_size)");
    if (dX_arr.ndim() != 3)
        throw std::invalid_argument("dX must be 3D (N, rep_size, ncoords)");
    if (W_arr.ndim() != 2)
        throw std::invalid_argument("W must be 2D (rep_size, D)");
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("b must be 1D (D,)");
    if (F_arr.ndim() != 1)
        throw std::invalid_argument("F must be 1D (N*ncoords,)");

    const auto N        = static_cast<std::size_t>(X_arr.shape(0));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(1));
    const auto D        = static_cast<std::size_t>(b_arr.shape(0));
    const auto ncoords  = static_cast<std::size_t>(dX_arr.shape(2));

    if (static_cast<std::size_t>(dX_arr.shape(0)) != N)
        throw std::invalid_argument("dX.shape[0] must equal X.shape[0] (N)");
    if (static_cast<std::size_t>(dX_arr.shape(1)) != rep_size)
        throw std::invalid_argument("dX.shape[1] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(0)) != rep_size)
        throw std::invalid_argument("W.shape[0] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(1)) != D)
        throw std::invalid_argument("W.shape[1] must equal b.shape[0] (D)");

    const std::size_t expected_F = N * ncoords;
    if (static_cast<std::size_t>(F_arr.shape(0)) != expected_F)
        throw std::invalid_argument(
            "F.shape[0] must equal N * ncoords = " + std::to_string(expected_F));

    double *GtG_ptr = aligned_alloc_64(D * D);
    double *GtF_ptr = aligned_alloc_64(D);
    auto cap_gram = py::capsule(GtG_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(GtF_ptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gradient_gramian_symm(
        X_arr.data(), dX_arr.data(), W_arr.data(), b_arr.data(),
        F_arr.data(),
        N, rep_size, D, ncoords, chunk_size,
        GtG_ptr, GtF_ptr);

    auto GtG_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        GtG_ptr, cap_gram);

    auto GtF_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        GtF_ptr, cap_proj);

    return py::make_tuple(GtG_out, GtF_out);
}

// ---- rff_full_gramian_symm binding ------------------------------------------

static py::tuple py_rff_full_gramian_symm(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &Y_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &F_arr,
    std::size_t energy_chunk,
    std::size_t force_chunk) {

    if (X_arr.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N, rep_size)");
    if (dX_arr.ndim() != 3)
        throw std::invalid_argument("dX must be 3D (N, rep_size, ncoords)");
    if (W_arr.ndim() != 2)
        throw std::invalid_argument("W must be 2D (rep_size, D)");
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("b must be 1D (D,)");
    if (Y_arr.ndim() != 1)
        throw std::invalid_argument("Y must be 1D (N,)");
    if (F_arr.ndim() != 1)
        throw std::invalid_argument("F must be 1D (N*ncoords,)");

    const auto N        = static_cast<std::size_t>(X_arr.shape(0));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(1));
    const auto D        = static_cast<std::size_t>(b_arr.shape(0));
    const auto ncoords  = static_cast<std::size_t>(dX_arr.shape(2));

    if (static_cast<std::size_t>(dX_arr.shape(0)) != N)
        throw std::invalid_argument("dX.shape[0] must equal X.shape[0] (N)");
    if (static_cast<std::size_t>(dX_arr.shape(1)) != rep_size)
        throw std::invalid_argument("dX.shape[1] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(0)) != rep_size)
        throw std::invalid_argument("W.shape[0] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(1)) != D)
        throw std::invalid_argument("W.shape[1] must equal b.shape[0] (D)");
    if (static_cast<std::size_t>(Y_arr.shape(0)) != N)
        throw std::invalid_argument("Y.shape[0] must equal X.shape[0] (N)");

    const std::size_t expected_F = N * ncoords;
    if (static_cast<std::size_t>(F_arr.shape(0)) != expected_F)
        throw std::invalid_argument(
            "F.shape[0] must equal N * ncoords = " + std::to_string(expected_F));

    double *ZtZ_ptr = aligned_alloc_64(D * D);
    double *ZtY_ptr = aligned_alloc_64(D);
    auto cap_gram = py::capsule(ZtZ_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(ZtY_ptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_full_gramian_symm(
        X_arr.data(), dX_arr.data(), W_arr.data(), b_arr.data(),
        Y_arr.data(), F_arr.data(),
        N, rep_size, D, ncoords,
        energy_chunk, force_chunk,
        ZtZ_ptr, ZtY_ptr);

    auto ZtZ_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        ZtZ_ptr, cap_gram);

    auto ZtY_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtY_ptr, cap_proj);

    return py::make_tuple(ZtZ_out, ZtY_out);
}

// ---- rff_gramian_symm_rfp binding -------------------------------------------

static py::tuple py_rff_gramian_symm_rfp(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &Y_arr,
    std::size_t chunk_size) {

    if (X_arr.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N, rep_size)");
    if (W_arr.ndim() != 2)
        throw std::invalid_argument("W must be 2D (rep_size, D)");
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("b must be 1D (D,)");
    if (Y_arr.ndim() != 1)
        throw std::invalid_argument("Y must be 1D (N,)");

    const auto N        = static_cast<std::size_t>(X_arr.shape(0));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(1));
    const auto D        = static_cast<std::size_t>(b_arr.shape(0));

    if (static_cast<std::size_t>(W_arr.shape(0)) != rep_size)
        throw std::invalid_argument("W.shape[0] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(1)) != D)
        throw std::invalid_argument("W.shape[1] must equal b.shape[0] (D)");
    if (static_cast<std::size_t>(Y_arr.shape(0)) != N)
        throw std::invalid_argument("Y.shape[0] must equal X.shape[0] (N)");

    const std::size_t nt = D * (D + 1) / 2;
    double *ZtZ_rfp_ptr = aligned_alloc_64(nt);
    double *ZtY_ptr     = aligned_alloc_64(D);
    auto cap_rfp  = py::capsule(ZtZ_rfp_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(ZtY_ptr,     [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gramian_symm_rfp(
        X_arr.data(), W_arr.data(), b_arr.data(), Y_arr.data(),
        N, rep_size, D, chunk_size,
        ZtZ_rfp_ptr, ZtY_ptr);

    auto ZtZ_rfp_out = py::array_t<double>(
        {static_cast<py::ssize_t>(nt)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtZ_rfp_ptr, cap_rfp);

    auto ZtY_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtY_ptr, cap_proj);

    return py::make_tuple(ZtZ_rfp_out, ZtY_out);
}

// ---- rff_gradient_gramian_symm_rfp binding ----------------------------------

static py::tuple py_rff_gradient_gramian_symm_rfp(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &F_arr,
    std::size_t chunk_size) {

    if (X_arr.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N, rep_size)");
    if (dX_arr.ndim() != 3)
        throw std::invalid_argument("dX must be 3D (N, rep_size, ncoords)");
    if (W_arr.ndim() != 2)
        throw std::invalid_argument("W must be 2D (rep_size, D)");
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("b must be 1D (D,)");
    if (F_arr.ndim() != 1)
        throw std::invalid_argument("F must be 1D (N*ncoords,)");

    const auto N        = static_cast<std::size_t>(X_arr.shape(0));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(1));
    const auto D        = static_cast<std::size_t>(b_arr.shape(0));
    const auto ncoords  = static_cast<std::size_t>(dX_arr.shape(2));

    if (static_cast<std::size_t>(dX_arr.shape(0)) != N)
        throw std::invalid_argument("dX.shape[0] must equal X.shape[0] (N)");
    if (static_cast<std::size_t>(dX_arr.shape(1)) != rep_size)
        throw std::invalid_argument("dX.shape[1] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(0)) != rep_size)
        throw std::invalid_argument("W.shape[0] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(1)) != D)
        throw std::invalid_argument("W.shape[1] must equal b.shape[0] (D)");

    const std::size_t expected_F = N * ncoords;
    if (static_cast<std::size_t>(F_arr.shape(0)) != expected_F)
        throw std::invalid_argument(
            "F.shape[0] must equal N * ncoords = " + std::to_string(expected_F));

    const std::size_t nt = D * (D + 1) / 2;
    double *GtG_rfp_ptr = aligned_alloc_64(nt);
    double *GtF_ptr     = aligned_alloc_64(D);
    auto cap_rfp  = py::capsule(GtG_rfp_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(GtF_ptr,     [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gradient_gramian_symm_rfp(
        X_arr.data(), dX_arr.data(), W_arr.data(), b_arr.data(),
        F_arr.data(),
        N, rep_size, D, ncoords, chunk_size,
        GtG_rfp_ptr, GtF_ptr);

    auto GtG_rfp_out = py::array_t<double>(
        {static_cast<py::ssize_t>(nt)},
        {static_cast<py::ssize_t>(sizeof(double))},
        GtG_rfp_ptr, cap_rfp);

    auto GtF_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        GtF_ptr, cap_proj);

    return py::make_tuple(GtG_rfp_out, GtF_out);
}

// ---- rff_full_gramian_symm_rfp binding --------------------------------------

static py::tuple py_rff_full_gramian_symm_rfp(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &Y_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &F_arr,
    std::size_t energy_chunk,
    std::size_t force_chunk) {

    if (X_arr.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N, rep_size)");
    if (dX_arr.ndim() != 3)
        throw std::invalid_argument("dX must be 3D (N, rep_size, ncoords)");
    if (W_arr.ndim() != 2)
        throw std::invalid_argument("W must be 2D (rep_size, D)");
    if (b_arr.ndim() != 1)
        throw std::invalid_argument("b must be 1D (D,)");
    if (Y_arr.ndim() != 1)
        throw std::invalid_argument("Y must be 1D (N,)");
    if (F_arr.ndim() != 1)
        throw std::invalid_argument("F must be 1D (N*ncoords,)");

    const auto N        = static_cast<std::size_t>(X_arr.shape(0));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(1));
    const auto D        = static_cast<std::size_t>(b_arr.shape(0));
    const auto ncoords  = static_cast<std::size_t>(dX_arr.shape(2));

    if (static_cast<std::size_t>(dX_arr.shape(0)) != N)
        throw std::invalid_argument("dX.shape[0] must equal X.shape[0] (N)");
    if (static_cast<std::size_t>(dX_arr.shape(1)) != rep_size)
        throw std::invalid_argument("dX.shape[1] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(0)) != rep_size)
        throw std::invalid_argument("W.shape[0] must equal X.shape[1] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(1)) != D)
        throw std::invalid_argument("W.shape[1] must equal b.shape[0] (D)");
    if (static_cast<std::size_t>(Y_arr.shape(0)) != N)
        throw std::invalid_argument("Y.shape[0] must equal X.shape[0] (N)");

    const std::size_t expected_F = N * ncoords;
    if (static_cast<std::size_t>(F_arr.shape(0)) != expected_F)
        throw std::invalid_argument(
            "F.shape[0] must equal N * ncoords = " + std::to_string(expected_F));

    const std::size_t nt = D * (D + 1) / 2;
    double *ZtZ_rfp_ptr = aligned_alloc_64(nt);
    double *ZtY_ptr     = aligned_alloc_64(D);
    auto cap_rfp  = py::capsule(ZtZ_rfp_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(ZtY_ptr,     [](void *p) { aligned_free_64(p); });

    kf::rff::rff_full_gramian_symm_rfp(
        X_arr.data(), dX_arr.data(), W_arr.data(), b_arr.data(),
        Y_arr.data(), F_arr.data(),
        N, rep_size, D, ncoords,
        energy_chunk, force_chunk,
        ZtZ_rfp_ptr, ZtY_ptr);

    auto ZtZ_rfp_out = py::array_t<double>(
        {static_cast<py::ssize_t>(nt)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtZ_rfp_ptr, cap_rfp);

    auto ZtY_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtY_ptr, cap_proj);

    return py::make_tuple(ZtZ_rfp_out, ZtY_out);
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

    m.def("rff_gradient", &py_rff_gradient,
          R"doc(
Compute gradient of RFF features w.r.t. atomic coordinates.

G[d, i*ncoords + g] = d(Z[i,d]) / d(coord_g)
                    = -sqrt(2/D) * sin(z[d]) * (W^T @ dX[i])_{d,g}

Parameters
----------
X : ndarray, shape (N, rep_size)
    Input feature matrix.
dX : ndarray, shape (N, rep_size, ncoords)
    Jacobians of features w.r.t. atomic coordinates.
    dX[i, r, g] = d(X[i, r]) / d(coord_g).
    ncoords = 3 * natoms, same for all N molecules.
W : ndarray, shape (rep_size, D)
    Random weight matrix.
b : ndarray, shape (D,)
    Random bias vector.

Returns
-------
G : ndarray, shape (D, N*ncoords)
    Gradient matrix. G[d, g] = derivative of feature d w.r.t. gradient g.
)doc",
          py::arg("X"), py::arg("dX"), py::arg("W"), py::arg("b"));

    m.def("rff_full", &py_rff_full,
          R"doc(
Compute combined energy+force RFF feature matrix (stacked).

Z_full[0:N, :]           = rff_features(X, W, b)          (energy rows)
Z_full[N:N+N*ncoords, :] = rff_gradient(X, dX, W, b).T    (force rows)

Satisfies: Z_full.T @ Z_full = Z.T@Z + G@G.T

Parameters
----------
X : ndarray, shape (N, rep_size)
    Input feature matrix.
dX : ndarray, shape (N, rep_size, ncoords)
    Jacobians of features w.r.t. atomic coordinates.
W : ndarray, shape (rep_size, D)
    Random weight matrix.
b : ndarray, shape (D,)
    Random bias vector.

Returns
-------
Z_full : ndarray, shape (N + N*ncoords, D)
    Combined feature matrix.
)doc",
          py::arg("X"), py::arg("dX"), py::arg("W"), py::arg("b"));

    m.def("rff_gramian_symm", &py_rff_gramian_symm,
          R"doc(
Compute chunked symmetric Gramian for energy-only RFF training.

Processes molecules in chunks, computing RFF features and accumulating
the Gram matrix (Z^T @ Z) and projection (Z^T @ Y).

Parameters
----------
X : ndarray, shape (N, rep_size)
    Input feature matrix.
W : ndarray, shape (rep_size, D)
    Random weight matrix.
b : ndarray, shape (D,)
    Random bias vector.
Y : ndarray, shape (N,)
    Target energies.
chunk_size : int
    Number of molecules per chunk (default: 1000).

Returns
-------
ZtZ : ndarray, shape (D, D)
    Symmetric Gram matrix Z^T @ Z.
ZtY : ndarray, shape (D,)
    Projection vector Z^T @ Y.
)doc",
          py::arg("X"), py::arg("W"), py::arg("b"),
          py::arg("Y"), py::arg("chunk_size") = 1000);

    m.def("rff_gradient_gramian_symm", &py_rff_gradient_gramian_symm,
          R"doc(
Compute chunked symmetric Gramian for force-only RFF training.

Processes molecules in chunks, computing RFF gradients and accumulating
the Gram matrix (G @ G^T) and projection (G @ F).

Parameters
----------
X : ndarray, shape (N, rep_size)
    Input feature matrix.
dX : ndarray, shape (N, rep_size, ncoords)
    Jacobians of features w.r.t. atomic coordinates.
W : ndarray, shape (rep_size, D)
    Random weight matrix.
b : ndarray, shape (D,)
    Random bias vector.
F : ndarray, shape (N*ncoords,)
    Target forces (flattened).
chunk_size : int
    Number of molecules per chunk (default: 100).

Returns
-------
GtG : ndarray, shape (D, D)
    Symmetric Gram matrix G @ G^T.
GtF : ndarray, shape (D,)
    Projection vector G @ F.
)doc",
          py::arg("X"), py::arg("dX"), py::arg("W"), py::arg("b"),
          py::arg("F"), py::arg("chunk_size") = 100);

    m.def("rff_full_gramian_symm", &py_rff_full_gramian_symm,
          R"doc(
Compute chunked symmetric Gramian for energy+force RFF training.

Combines energy (RFF features) and force (gradient of RFF) terms
into a single Gram matrix and projection vector:
  ZtZ = Z^T @ Z + G @ G^T
  ZtY  = Z^T @ Y + G @ F

Parameters
----------
X : ndarray, shape (N, rep_size)
    Input feature matrix.
dX : ndarray, shape (N, rep_size, ncoords)
    Jacobians of features w.r.t. atomic coordinates.
    ncoords = 3 * natoms, same for all N molecules.
W : ndarray, shape (rep_size, D)
    Random weight matrix.
b : ndarray, shape (D,)
    Random bias vector.
Y : ndarray, shape (N,)
    Target energies.
F : ndarray, shape (N*ncoords,)
    Target forces (flattened).
energy_chunk : int
    Chunk size for energy loop (default: 1000).
force_chunk : int
    Chunk size for force loop (default: 100).

Returns
-------
ZtZ : ndarray, shape (D, D)
    Symmetric Gram matrix (energy + force contributions).
ZtY : ndarray, shape (D,)
    Projection vector (energy + force contributions).
)doc",
          py::arg("X"), py::arg("dX"), py::arg("W"), py::arg("b"),
          py::arg("Y"), py::arg("F"),
          py::arg("energy_chunk") = 1000,
          py::arg("force_chunk") = 100);

    m.def("rff_gramian_symm_rfp", &py_rff_gramian_symm_rfp,
          R"doc(
Compute chunked Gramian for energy-only RFF training, RFP-packed output.

Same as rff_gramian_symm but returns ZtZ in LAPACK RFP format
(TRANSR='N', UPLO='U'), length D*(D+1)//2.

Parameters
----------
X : ndarray, shape (N, rep_size)
W : ndarray, shape (rep_size, D)
b : ndarray, shape (D,)
Y : ndarray, shape (N,)
chunk_size : int

Returns
-------
ZtZ_rfp : ndarray, shape (D*(D+1)//2,)
    Packed upper triangle of Z^T @ Z.
ZtY : ndarray, shape (D,)
)doc",
          py::arg("X"), py::arg("W"), py::arg("b"),
          py::arg("Y"), py::arg("chunk_size") = 1000);

    m.def("rff_gradient_gramian_symm_rfp", &py_rff_gradient_gramian_symm_rfp,
          R"doc(
Compute chunked Gramian for force-only RFF training, RFP-packed output.

Same as rff_gradient_gramian_symm but returns GtG in LAPACK RFP format.

Parameters
----------
X : ndarray, shape (N, rep_size)
dX : ndarray, shape (N, rep_size, ncoords)
W : ndarray, shape (rep_size, D)
b : ndarray, shape (D,)
F : ndarray, shape (N*ncoords,)
chunk_size : int

Returns
-------
GtG_rfp : ndarray, shape (D*(D+1)//2,)
GtF : ndarray, shape (D,)
)doc",
          py::arg("X"), py::arg("dX"), py::arg("W"), py::arg("b"),
          py::arg("F"), py::arg("chunk_size") = 100);

    m.def("rff_full_gramian_symm_rfp", &py_rff_full_gramian_symm_rfp,
          R"doc(
Compute chunked Gramian for energy+force RFF training, RFP-packed output.

Same as rff_full_gramian_symm but returns ZtZ in LAPACK RFP format.

Parameters
----------
X : ndarray, shape (N, rep_size)
dX : ndarray, shape (N, rep_size, ncoords)
W : ndarray, shape (rep_size, D)
b : ndarray, shape (D,)
Y : ndarray, shape (N,)
F : ndarray, shape (N*ncoords,)
energy_chunk : int
force_chunk : int

Returns
-------
ZtZ_rfp : ndarray, shape (D*(D+1)//2,)
ZtY : ndarray, shape (D,)
)doc",
          py::arg("X"), py::arg("dX"), py::arg("W"), py::arg("b"),
          py::arg("Y"), py::arg("F"),
          py::arg("energy_chunk") = 1000,
          py::arg("force_chunk") = 100);

    // Elemental functions registered from rff_elemental_bindings.cpp
    register_rff_elemental(m);
}
