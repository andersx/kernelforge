// C++ standard library
#include <stdexcept>

// Third-party libraries
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// Project headers
#include "aligned_alloc64.hpp"
#include "blas_int.h"
#include "global_kernels.hpp"

namespace py = pybind11;

static void check_2d(const py::array &X) {
    if (X.ndim() != 2) {
        throw std::runtime_error("X must be 2D (n, rep_size) in row-major (C) order.");
    }
}

py::array_t<double> kernel_symm_py(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                                   double alpha) {
    check_2d(X);
    auto bufX = X.request();

    blas_int n = static_cast<blas_int>(bufX.shape[0]);         // rows
    blas_int rep_size = static_cast<blas_int>(bufX.shape[1]);  // cols
    double *Xptr = static_cast<double *>(bufX.ptr);

    // Allocate aligned K (row-major, n x n)
    std::size_t nelems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    double *Kptr = aligned_alloc_64(nelems);

    // Capsule to free aligned memory when NumPy array is GC’d
    auto capsule = py::capsule(Kptr, [](void *p) { aligned_free_64(p); });

    // Make a NumPy view over Kptr (row-major)
    py::array_t<double> K({n, n},
                          {static_cast<py::ssize_t>(n) * static_cast<py::ssize_t>(sizeof(double)),
                           static_cast<py::ssize_t>(sizeof(double))},
                          Kptr, capsule);

    // Compute
    kf::kernel_gaussian_symm(Xptr, n, rep_size, alpha, Kptr);

    return K;
}

py::array_t<double> kernel_asymm_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X1,  // (n1,d)
    py::array_t<double, py::array::c_style | py::array::forcecast> X2,  // (n2,d)
    double alpha) {
    check_2d(X1);
    check_2d(X2);

    const std::size_t n1 = static_cast<std::size_t>(X1.shape(0));
    const std::size_t d1 = static_cast<std::size_t>(X1.shape(1));
    const std::size_t n2 = static_cast<std::size_t>(X2.shape(0));
    const std::size_t d2 = static_cast<std::size_t>(X2.shape(1));
    if (d1 != d2)
        throw std::runtime_error("X1.shape[1] must equal X2.shape[1] (feature dimension d).");

    // Make aligned (n2 x n1) output
    const std::size_t nelems = n2 * n1;
    double *Kptr = aligned_alloc_64(nelems);
    auto capsule = py::capsule(Kptr, [](void *p) { aligned_free_64(p); });

    py::array_t<double> K(
        {static_cast<py::ssize_t>(n1), static_cast<py::ssize_t>(n2)},
        {static_cast<py::ssize_t>(n2 * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        Kptr, capsule);

    auto X1v = X1.unchecked<2>();
    auto X2v = X2.unchecked<2>();

    kf::kernel_gaussian(X1v.data(0, 0), X2v.data(0, 0), n1, n2, d1, alpha, Kptr);

    return K;
}

static py::array_t<double> gaussian_jacobian_batch_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X1,   // (N1, M)
    py::array_t<double, py::array::c_style | py::array::forcecast> dX1,  // (N1, M, D)
    py::array_t<double, py::array::c_style | py::array::forcecast> X2,   // (N2, M)
    double sigma) {
    if (X1.ndim() != 2)
        throw std::invalid_argument("X1 must be 2D (N1,M).");
    if (dX1.ndim() != 3)
        throw std::invalid_argument("dX1 must be 3D (N1,M,D).");
    if (X2.ndim() != 2)
        throw std::invalid_argument("X2 must be 2D (N2,M).");
    if (sigma <= 0.0)
        throw std::invalid_argument("sigma must be > 0.");

    const auto N1 = static_cast<std::size_t>(X1.shape(0));
    const auto M = static_cast<std::size_t>(X1.shape(1));

    if (static_cast<std::size_t>(dX1.shape(0)) != N1)
        throw std::invalid_argument("dX1.shape[0] must equal X1.shape[0] (N1).");
    if (static_cast<std::size_t>(dX1.shape(1)) != M)
        throw std::invalid_argument("dX1.shape[1] must equal X1.shape[1] (M).");

    const auto D = static_cast<std::size_t>(dX1.shape(2));  // = 3N (query)
    if (D == 0)
        throw std::invalid_argument("D (last dim of dX1) must be > 0.");

    const auto N2 = static_cast<std::size_t>(X2.shape(0));
    const auto Mx = static_cast<std::size_t>(X2.shape(1));
    if (Mx != M)
        throw std::invalid_argument("X2.shape[1] must equal M.");

    // Raw pointers (contiguous due to c_style|forcecast)
    auto x1 = X1.unchecked<2>();    // (N1,M)
    auto dx1 = dX1.unchecked<3>();  // (N1,M,D)
    auto x2 = X2.unchecked<2>();    // (N2,M)

    const double *X1p = x1.data(0, 0);
    const double *dX1p = dx1.data(0, 0, 0);
    const double *X2p = x2.data(0, 0);

    // Output (N1*D, N2)
    py::array_t<double> K({static_cast<py::ssize_t>(N1 * D), static_cast<py::ssize_t>(N2)});
    auto Kv = K.mutable_unchecked<2>();
    double *Kp = Kv.mutable_data(0, 0);

    kf::kernel_gaussian_jacobian(X1p, dX1p, X2p, N1, N2, M, D, sigma, Kp);
    return K;
}

static py::array_t<double> rbf_hessian_full_tiled_gemm_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X1,   // (N1, M)
    py::array_t<double, py::array::c_style | py::array::forcecast> dX1,  // (N1, M, D1)
    py::array_t<double, py::array::c_style | py::array::forcecast> X2,   // (N2, M)
    py::array_t<double, py::array::c_style | py::array::forcecast> dX2,  // (N2, M, D2)
    double sigma, py::object tile_B_obj                                  /* int or None */
) {
    if (X1.ndim() != 2)
        throw std::invalid_argument("X1 must be 2D (N1,M).");
    if (X2.ndim() != 2)
        throw std::invalid_argument("X2 must be 2D (N2,M).");
    if (dX1.ndim() != 3)
        throw std::invalid_argument("dX1 must be 3D (N1,M,D1).");
    if (dX2.ndim() != 3)
        throw std::invalid_argument("dX2 must be 3D (N2,M,D2).");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0.");

    const std::size_t N1 = static_cast<std::size_t>(X1.shape(0));
    const std::size_t M = static_cast<std::size_t>(X1.shape(1));
    const std::size_t N2 = static_cast<std::size_t>(X2.shape(0));
    const std::size_t Mx = static_cast<std::size_t>(X2.shape(1));

    if (Mx != M)
        throw std::invalid_argument("X2.shape[1] must equal X1.shape[1] (M).");

    if (static_cast<std::size_t>(dX1.shape(0)) != N1)
        throw std::invalid_argument("dX1.shape[0] must equal X1.shape[0] (N1).");
    if (static_cast<std::size_t>(dX1.shape(1)) != M)
        throw std::invalid_argument("dX1.shape[1] must equal X1.shape[1] (M).");

    if (static_cast<std::size_t>(dX2.shape(0)) != N2)
        throw std::invalid_argument("dX2.shape[0] must equal X2.shape[0] (N2).");
    if (static_cast<std::size_t>(dX2.shape(1)) != M)
        throw std::invalid_argument("dX2.shape[1] must equal X2.shape[1] (M).");

    const std::size_t D1 = static_cast<std::size_t>(dX1.shape(2));
    const std::size_t D2 = static_cast<std::size_t>(dX2.shape(2));
    if (D1 == 0 || D2 == 0)
        throw std::invalid_argument("D1 and D2 must be > 0.");

    // tile_B: default 0 means "choose heuristic" inside the core
    std::size_t tile_B = 0;
    if (!tile_B_obj.is_none()) {
        long tb = tile_B_obj.cast<long>();
        if (tb < 0)
            throw std::invalid_argument("tile_B must be >= 0 (0 means auto).");
        tile_B = static_cast<std::size_t>(tb);
    }

    // Prepare output ((N1*D1) x (N2*D2))
    // py::array_t<double> H({ static_cast<py::ssize_t>(N1*D1),
    //                         static_cast<py::ssize_t>(N2*D2) });
    // auto Hv = H.mutable_unchecked<2>();

    // Raw pointers (arrays are C-contiguous due to c_style|forcecast)
    const double *X1p = X1.unchecked<2>().data(0, 0);
    const double *dX1p = dX1.unchecked<3>().data(0, 0, 0);
    const double *X2p = X2.unchecked<2>().data(0, 0);
    const double *dX2p = dX2.unchecked<3>().data(0, 0, 0);
    // double* Hout       = Hv.mutable_data(0,0);

    // Number of elements
    const std::size_t nelems = static_cast<std::size_t>(N1) * D1 * N2 * D2;

    // Allocate aligned memory (e.g. 64-byte aligned)
    double *Hptr = aligned_alloc_64(nelems);

    // Capsule to free when Python GC runs
    auto capsule = py::capsule(Hptr, [](void *p) { aligned_free_64(p); });

    // Build NumPy array with shape (N1*D1, N2*D2)
    py::array_t<double> H({static_cast<py::ssize_t>(N1 * D1), static_cast<py::ssize_t>(N2 * D2)},
                          {static_cast<py::ssize_t>(N2 * D2 * sizeof(double)),  // row stride
                           static_cast<py::ssize_t>(sizeof(double))},           // col stride
                          Hptr, capsule);

    kf::kernel_gaussian_hessian(X1p, dX1p, X2p, dX2p, N1, N2, M, D1, D2, sigma, tile_B, Hptr);
    return H;
}

static py::array_t<double> rbf_hessian_full_tiled_gemm_sym_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,   // (N, M)
    py::array_t<double, py::array::c_style | py::array::forcecast> dX,  // (N, M, D)
    double sigma, py::object tile_B_obj                                 /* int or None */
) {
    if (X.ndim() != 2)
        throw std::invalid_argument("X must be 2D (N,M).");
    if (dX.ndim() != 3)
        throw std::invalid_argument("dX must be 3D (N,M,D).");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0.");

    const std::size_t N = static_cast<std::size_t>(X.shape(0));
    const std::size_t M = static_cast<std::size_t>(X.shape(1));
    if (static_cast<std::size_t>(dX.shape(0)) != N)
        throw std::invalid_argument("dX.shape[0] must equal X.shape[0] (N).");
    if (static_cast<std::size_t>(dX.shape(1)) != M)
        throw std::invalid_argument("dX.shape[1] must equal X.shape[1] (M).");

    const std::size_t D = static_cast<std::size_t>(dX.shape(2));
    if (D == 0)
        throw std::invalid_argument("D must be > 0.");

    // tile_B: default 0 means "auto"
    std::size_t tile_B = 0;
    if (!tile_B_obj.is_none()) {
        long tb = tile_B_obj.cast<long>();
        if (tb < 0)
            throw std::invalid_argument("tile_B must be >= 0 (0 means auto).");
        tile_B = static_cast<std::size_t>(tb);
    }

    // Number of elements in symmetric Hessian
    const std::size_t BIG = N * D;
    const std::size_t nelems = BIG * BIG;

    // Allocate aligned memory for the result
    double *Hptr = aligned_alloc_64(nelems);

    // Capsule to free the memory when Python GC runs
    auto capsule = py::capsule(Hptr, [](void *p) { aligned_free_64(p); });

    // Wrap into NumPy array (BIG x BIG)
    py::array_t<double> H({static_cast<py::ssize_t>(BIG), static_cast<py::ssize_t>(BIG)},
                          {static_cast<py::ssize_t>(BIG * sizeof(double)),  // row stride
                           static_cast<py::ssize_t>(sizeof(double))},       // col stride
                          Hptr, capsule);

    // Raw pointers
    const double *Xp = X.unchecked<2>().data(0, 0);
    const double *dXp = dX.unchecked<3>().data(0, 0, 0);

    // Call the C++ core
    kf::kernel_gaussian_hessian_symm(Xp, dXp, N, M, D, sigma, tile_B, Hptr);

    return H;
}

py::array_t<double> kernel_symm_rfp_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> X, double alpha) {
    check_2d(X);
    auto bufX = X.request();

    blas_int n = static_cast<blas_int>(bufX.shape[0]);         // rows
    blas_int rep_size = static_cast<blas_int>(bufX.shape[1]);  // cols
    double *Xptr = static_cast<double *>(bufX.ptr);

    // RFP output size: n*(n+1)/2
    const std::size_t nt =
        static_cast<std::size_t>(n) * (static_cast<std::size_t>(n) + 1) / 2;
    double *arf = aligned_alloc_64(nt);

    // Capsule to free aligned memory when NumPy array is GC'd
    auto capsule = py::capsule(arf, [](void *p) { aligned_free_64(p); });

    // Make a NumPy 1D array over arf (C-contiguous)
    py::array_t<double> K_rfp({static_cast<py::ssize_t>(nt)}, {static_cast<py::ssize_t>(sizeof(double))},
                              arf, capsule);

    // Compute
    kf::kernel_gaussian_symm_rfp(Xptr, n, rep_size, alpha, arf);

    return K_rfp;
}

PYBIND11_MODULE(global_kernels, m) {
    m.doc() = "Global (structure-wise) Gaussian kernels via BLAS (row-major), with 64-byte aligned output buffer.";
    m.def("kernel_gaussian_symm", &kernel_symm_py, py::arg("X"), py::arg("alpha"),
          "Compute K = exp(alpha*(||x_i||^2 + ||x_j||^2 - 2 x_i·x_j)) over the lower triangle.\n"
          "X is (n, rep_size) in row-major; returns K as an (n,n) NumPy array.");
    m.def("kernel_gaussian_symm_rfp", &kernel_symm_rfp_py, py::arg("X"), py::arg("alpha"),
          "Compute symmetric Gaussian kernel directly into RFP format (TRANSR='N', UPLO='U').\n"
          "X is (n, rep_size) in row-major; returns 1D array of length n*(n+1)/2 in RFP packed layout.");
    m.def("kernel_gaussian", &kernel_asymm_py, py::arg("X1"), py::arg("X2"), py::arg("alpha"),
          "Return K (n2, n1) where K[i2,i1] = exp(alpha*(||x2||^2 + ||x1||^2 - 2 x2·x1)).");
    m.def("kernel_gaussian_jacobian", &gaussian_jacobian_batch_py, py::arg("X1"), py::arg("dX1"),
          py::arg("X2"), py::arg("sigma"));
    m.def("kernel_gaussian_hessian", &rbf_hessian_full_tiled_gemm_py, py::arg("X1"),
          py::arg("dX1"), py::arg("X2"), py::arg("dX2"), py::arg("sigma"),
          py::arg("tile_B") = py::none(),
          "Compute the full Gaussian Hessian/GDML kernel with DGEMM tiling.\n"
          "Shapes: X1(N1,M), dX1(N1,M,D1), X2(N2,M), dX2(N2,M,D2) -> H((N1*D1),(N2*D2)).\n"
          "tile_B: refs per tile (0 = auto).");
    m.def("kernel_gaussian_hessian_symm", &rbf_hessian_full_tiled_gemm_sym_py, py::arg("X"),
          py::arg("dX"), py::arg("sigma"), py::arg("tile_B") = py::none(),
          "Compute symmetric Gaussian Hessian/GDML kernel (training version).");
}
