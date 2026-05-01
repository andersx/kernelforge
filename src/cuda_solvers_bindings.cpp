// cuda_solvers_bindings.cpp — pybind11 bindings for cuda_solvers

#include <pybind11/pybind11.h>

#include "cuda_solvers.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_solvers, m) {
    m.doc() = "CUDA least-squares solvers via cuSOLVER/cuBLAS (FP32).";

    m.def(
        "cuda_solve_svd",
        &kf::solvers::cuda_solve_svd,
        py::arg("Z"),
        py::arg("y"),
        py::arg("rcond") = 0.0,
        py::arg("z_col_major") = false,
        "Solve min_w ||Z @ w - y||_2 via truncated SVD (GPU, FP32).\n"
        "Z is (m, n) float32 torch.Tensor, y is (m,) float32 torch.Tensor.\n"
        "If z_col_major=True, Z is (n, m) col-major (avoids internal transpose).\n"
        "rcond: singular values < rcond * S_max are treated as zero.\n"
        "rcond <= 0 uses machine-epsilon heuristic (numpy default).\n"
        "Returns w: (n,) float32 CPU tensor."
    );

    m.def(
        "cuda_solve_qr",
        &kf::solvers::cuda_solve_qr,
        py::arg("Z"),
        py::arg("y"),
        py::arg("z_col_major") = false,
        "Solve min_w ||Z @ w - y||_2 via QR (GPU, FP32).\n"
        "Z is (m, n) float32 torch.Tensor, y is (m,) float32 torch.Tensor, m >= n.\n"
        "If z_col_major=True, Z is (n, m) col-major (avoids internal transpose).\n"
        "Returns w: (n,) float32 CPU tensor."
    );

    m.def(
        "cuda_solve_gels",
        &kf::solvers::cuda_solve_gels,
        py::arg("Z"),
        py::arg("y"),
        py::arg("z_col_major") = false,
        "Solve min_w ||Z @ w - y||_2 via cusolverDnSSgels IRS (GPU, FP32).\n"
        "Z is (m, n) float32 torch.Tensor, y is (m,) float32 torch.Tensor, m >= n.\n"
        "If z_col_major=True, Z is (n, m) col-major (avoids internal transpose).\n"
        "No rcond truncation — use cuda_solve_svd for rank-deficient systems.\n"
        "Returns w: (n,) float32 CPU tensor."
    );
}

