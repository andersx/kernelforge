#pragma once
#include <cstddef>

namespace kf {

void kernel_symm(
    const double* Xptr,
    int n,
    int rep_size,
    double alpha,
    double* Kptr
);

void kernel_asymm(
    const double* X1,
    const double* X2,
    std::size_t n1, std::size_t n2, std::size_t d,
    double alpha,
    double* K
);

void gaussian_jacobian_batch(
    const double* X1,
    const double* dX1,
    const double* X2,
    std::size_t N1,
    std::size_t N2,
    std::size_t M,
    std::size_t D,     // = 3N for the query molecules
    double sigma,
    double* K_out
);

void rbf_hessian_full_tiled_gemm(
    const double* X1,  const double* dX1,
    const double* X2,  const double* dX2,
    std::size_t N1, std::size_t N2,
    std::size_t M,  std::size_t D1, std::size_t D2,
    double sigma,
    std::size_t tile_B,
    double* H_out);

void rbf_hessian_full_tiled_gemm_sym_fast(
    const double* __restrict X,      // (N x M), row-major
    const double* __restrict dX,     // per-sample Jacobians, (N blocks) of (M x D), row-major
    std::size_t N, std::size_t M, std::size_t D,
    double sigma,
    std::size_t tile_B,
    double* __restrict H_out);       // (N*D x N*D), row-major

}  // namespace kf
