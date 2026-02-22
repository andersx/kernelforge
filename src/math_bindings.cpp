// C++ standard library
#include <stdexcept>

// Third-party libraries
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// Project headers
#include "aligned_alloc64.hpp"
#include "blas_int.h"
#include "math.hpp"

namespace py = pybind11;

py::array_t<double> solve_cholesky_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> K_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_in, double regularize = 0.0) {
    py::buffer_info Kbuf = K_in.request(true);
    py::buffer_info ybuf = y_in.request();

    if (Kbuf.ndim != 2 || Kbuf.shape[0] != Kbuf.shape[1])
        throw std::runtime_error("K must be n x n.");
    if (ybuf.ndim != 1 || ybuf.shape[0] != Kbuf.shape[0])
        throw std::runtime_error("y must have length n.");

    const blas_int n = static_cast<blas_int>(Kbuf.shape[0]);
    auto *K = static_cast<double *>(Kbuf.ptr);
    const auto *y = static_cast<const double *>(ybuf.ptr);

    // 64-byte aligned buffer that NumPy will own
    const std::size_t n_size = static_cast<std::size_t>(n);
    double *alpha_ptr = aligned_alloc_64(n_size);

    // Compute directly into the aligned buffer
    kf::math::solve_cholesky(K, y, n, alpha_ptr, regularize);

    auto free_capsule =
        py::capsule(alpha_ptr, [](void *p) { aligned_free_64(static_cast<double *>(p)); });

    return py::array_t<double>({n}, {sizeof(double)}, alpha_ptr, free_capsule);
}

static py::array_t<double> solve_cholesky_rfp_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> K_arf_in,  // 1-D RFP
    py::array_t<double, py::array::c_style | py::array::forcecast> y_in,      // 1-D
    double regularize,
    char uplo = 'U',  // <-- pass the ACTUAL mapping of K_arf ('U' if your kernel wrote upper)
    char transr = 'N') {
    auto Kbuf = K_arf_in.request(true);  // writable: factorization is in-place
    auto ybuf = y_in.request();

    if (Kbuf.ndim != 1 || ybuf.ndim != 1)
        throw std::runtime_error("K_arf must be 1D RFP and y must be 1D");
    const blas_int n = static_cast<blas_int>(ybuf.shape[0]);
    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t need = n_size * (n_size + 1) / 2;
    if (static_cast<std::size_t>(Kbuf.shape[0]) != need)
        throw std::runtime_error("K_arf length must be n*(n+1)/2");

    auto *K_arf = static_cast<double *>(Kbuf.ptr);
    const auto *y = static_cast<const double *>(ybuf.ptr);

    double *alpha_ptr = aligned_alloc_64(n_size);
    if (!alpha_ptr)
        throw std::bad_alloc();

    kf::math::solve_cholesky_rfp(K_arf, y, n, alpha_ptr, regularize, uplo, transr);

    auto cap = py::capsule(alpha_ptr, [](void *p) { aligned_free_64(p); });
    return py::array_t<double>({(py::ssize_t)n}, {(py::ssize_t)sizeof(double)}, alpha_ptr, cap);
}

// Helper to normalize flags (inline to avoid extra helpers)
static inline char norm_uplo(char u) {
    return (u == 'l' || u == 'L') ? 'L' : 'U';
}
static inline char norm_tr(char t) {
    return (t == 't' || t == 'T') ? 'T' : 'N';
}
static inline char swap_uplo(char u) {
    return (u == 'U') ? 'L' : 'U';
}

// Full (n×n, **C-order**) -> RFP (length n*(n+1)/2), no copy of A.
// We pass the raw C-order pointer straight to Fortran, but **swap UPLO**.
py::array_t<double> full_to_rfp_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> A_in, char uplo = 'L',
    char transr = 'N') {
    py::buffer_info Abuf = A_in.request();  // C-order; no copy
    if (Abuf.ndim != 2 || Abuf.shape[0] != Abuf.shape[1])
        throw std::runtime_error("A must be square (n x n).");

    uplo = norm_uplo(uplo);
    transr = norm_tr(transr);

    const blas_int n = static_cast<blas_int>(Abuf.shape[0]);
    const double *Arow = static_cast<const double *>(Abuf.ptr);

    // Output RFP buffer
    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t nt = n_size * (n_size + 1) / 2;
    double *ARF = aligned_alloc_64(nt);
    if (!ARF)
        throw std::bad_alloc();

    // Treat Arow as column-major A^T; swap UPLO so the intended triangle is used
    const blas_int lda = n;
    const blas_int info = kf::math::full_to_rfp(transr, swap_uplo(uplo), n, Arow, lda, ARF);
    if (info != 0) {
        aligned_free_64(ARF);
        throw std::runtime_error("dtrttf_ failed, info=" + std::to_string(info));
    }

    auto cap = py::capsule(ARF, [](void *p) { aligned_free_64(static_cast<double *>(p)); });
    return py::array_t<double>({(py::ssize_t)nt}, {(py::ssize_t)sizeof(double)}, ARF, cap);
}

// RFP (length n*(n+1)/2) -> Full (n×n, **C-order**), no extra copies.
// We let Fortran write into our buffer (interpreted as column-major),
// then **expose the same buffer as C-order**. Because the matrix is symmetric,
// the transpose view is identical, and swapping UPLO ensures the expected triangle is filled.
py::array_t<double> rfp_to_full_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> ARF_in, blas_int n,
    char uplo = 'L', char transr = 'N') {
    if (n <= 0)
        throw std::runtime_error("n must be > 0");
    py::buffer_info Rbuf = ARF_in.request();
    if (Rbuf.ndim != 1)
        throw std::runtime_error("ARF must be a 1-D array.");
    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t need = n_size * (n_size + 1) / 2;
    if (static_cast<std::size_t>(Rbuf.shape[0]) != need)
        throw std::runtime_error("ARF length must be n*(n+1)/2.");

    uplo = norm_uplo(uplo);
    transr = norm_tr(transr);

    const double *ARF = static_cast<const double *>(Rbuf.ptr);

    // Allocate full matrix buffer once; we will *return it as C-order*
    const std::size_t nn = n_size * n_size;
    double *A = aligned_alloc_64(nn);
    if (!A)
        throw std::bad_alloc();

    // Fortran writes treating A as column-major; swap UPLO to compensate
    const blas_int lda = n;
    const blas_int info = kf::math::rfp_to_full(transr, swap_uplo(uplo), n, ARF, A, lda);
    if (info != 0) {
        aligned_free_64(A);
        throw std::runtime_error("dtfttr_ failed, info=" + std::to_string(info));
    }

    // Expose as C-order (row-major) without copying
    auto cap = py::capsule(A, [](void *p) { aligned_free_64(static_cast<double *>(p)); });
    return py::array_t<double>(
        /* shape   */ {(py::ssize_t)n, (py::ssize_t)n},
        /* strides */ {(py::ssize_t)(sizeof(double) * n), (py::ssize_t)sizeof(double)},
        /* ptr     */ A,
        /* base    */ cap);
}

// Simplified solver: internally copies K_rfp so the user never needs to .copy().
// Always uses TRANSR='N', UPLO='U' (the convention all kernelforge RFP kernels output).
// l2 is added to the diagonal before Cholesky factorization (L2 regularization).
static py::array_t<double> cho_solve_rfp_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> K_rfp_in,  // 1-D RFP
    py::array_t<double, py::array::c_style | py::array::forcecast> y_in,      // 1-D
    double l2) {
    auto Kbuf = K_rfp_in.request();
    auto ybuf = y_in.request();

    if (Kbuf.ndim != 1 || ybuf.ndim != 1)
        throw std::runtime_error("K_rfp must be 1D and y must be 1D");
    const blas_int n = static_cast<blas_int>(ybuf.shape[0]);
    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t need = n_size * (n_size + 1) / 2;
    if (static_cast<std::size_t>(Kbuf.shape[0]) != need)
        throw std::runtime_error("K_rfp length must be n*(n+1)/2");

    // Make an aligned working copy of K_rfp (factorization is destructive)
    double *K_copy = aligned_alloc_64(need);
    if (!K_copy) throw std::bad_alloc();
    std::memcpy(K_copy, static_cast<const double *>(Kbuf.ptr), need * sizeof(double));

    const auto *y = static_cast<const double *>(ybuf.ptr);

    double *alpha_ptr = aligned_alloc_64(n_size);
    if (!alpha_ptr) { aligned_free_64(K_copy); throw std::bad_alloc(); }

    try {
        kf::math::solve_cholesky_rfp(K_copy, y, n, alpha_ptr, l2, 'U', 'N');
    } catch (...) {
        aligned_free_64(K_copy);
        aligned_free_64(alpha_ptr);
        throw;
    }

    aligned_free_64(K_copy);  // factored copy no longer needed

    auto cap = py::capsule(alpha_ptr, [](void *p) { aligned_free_64(p); });
    return py::array_t<double>({(py::ssize_t)n}, {(py::ssize_t)sizeof(double)}, alpha_ptr, cap);
}

PYBIND11_MODULE(kernelmath, m) {
    m.doc() = "Mathematical utilities (Cholesky solvers, etc.)";
    m.def("solve_cholesky", &solve_cholesky_py, py::arg("K"), py::arg("y"),
          py::arg("regularize") = 0.0,
          "Solve Kx=y using Cholesky factorization.\n"
          "- K is overwritten with factorization\n"
          "- y is preserved\n"
          "- regularize is added to diagonal of K\n"
          "- alpha is returned");
    m.def("solve_cholesky_rfp_L", &solve_cholesky_rfp_py, py::arg("K_arf"), py::arg("y"),
          py::arg("regularize") = 0.0, py::arg("uplo") = 'U', py::arg("transr") = 'N',
          "Solve (K * alpha = y) where K is SPD in RFP format (TRANSR='N', UPLO='L').\n"
          "Overwrites K_arf during factorization (diagonal restored afterwards).");
    m.def("cho_solve_rfp", &cho_solve_rfp_py, py::arg("K_rfp"), py::arg("y"),
          py::arg("l2") = 0.0,
          "Solve (K + l2*I) @ alpha = y where K is in RFP packed format.\n"
          "Internally copies K_rfp before factorization — no need to pass K_rfp.copy().\n"
          "Uses TRANSR='N', UPLO='U' (the convention all kernelforge RFP kernels use).\n"
          "l2: L2 regularization added to the diagonal (default 0.0).\n"
          "Returns alpha as a 1D numpy array of length n.");
    m.def("full_to_rfp", &full_to_rfp_py, py::arg("A"), py::arg("uplo") = 'U',
          py::arg("transr") = 'N',
          "Full (n×n, Fortran-order) -> RFP (1-D). No copies; A must be F-contiguous.");
    m.def("rfp_to_full", &rfp_to_full_py, py::arg("ARF"), py::arg("n"), py::arg("uplo") = 'U',
          py::arg("transr") = 'N',
          "RFP (1-D) -> Full (n×n, Fortran-order). No extra copies; returns F-contiguous.");
}
