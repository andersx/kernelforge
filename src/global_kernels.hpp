#pragma once
#include <cstddef>
#include "blas_int.h"

namespace kf {

void kernel_gaussian_symm(const double *Xptr, blas_int n, blas_int rep_size, double alpha,
                          double *Kptr);

void kernel_gaussian_symm_rfp(const double *Xptr, blas_int n, blas_int rep_size, double alpha,
                              double *arf);

void kernel_gaussian(const double *X1, const double *X2, std::size_t n1, std::size_t n2, std::size_t d,
                     double alpha, double *K);

void kernel_gaussian_jacobian(const double *X1, const double *dX1, const double *X2, std::size_t N1,
                              std::size_t N2, std::size_t M,
                              std::size_t D,  // = 3N for the query molecules
                              double sigma, double *K_out);

void kernel_gaussian_jacobian_t(const double *X1, const double *X2, const double *dX2,
                                std::size_t N1, std::size_t N2, std::size_t M,
                                std::size_t D,  // = 3N for the reference molecules
                                double sigma, double *K_out);

void kernel_gaussian_hessian(const double *X1, const double *dX1, const double *X2,
                             const double *dX2, std::size_t N1, std::size_t N2, std::size_t M,
                             std::size_t D1, std::size_t D2, double sigma, std::size_t tile_B,
                             double *H_out);

void kernel_gaussian_hessian_symm(
    const double *__restrict X,   // (N x M), row-major
    const double *__restrict dX,  // per-sample Jacobians, (N blocks) of (M x D), row-major
    std::size_t N, std::size_t M, std::size_t D, double sigma, std::size_t tile_B,
    double *__restrict H_out);  // (N*D x N*D), row-major

void kernel_gaussian_hessian_symm_rfp(
    const double *__restrict X,    // (N x M), row-major
    const double *__restrict dX,   // per-sample Jacobians, (N blocks) of (M x D), row-major
    std::size_t N, std::size_t M, std::size_t D, double sigma, std::size_t tile_B,
    double *__restrict H_rfp);  // RFP packed format, length N*D*(N*D+1)/2

// Full combined kernel for energy+force KRR (asymmetric).
// Output shape: (N1*(1+D1)) x (N2*(1+D2)), row-major, stride = N2*(1+D2).
// Layout:
//   [0:N1,         0:N2       ] = scalar kernel  K(a,b) = exp(-||x1a-x2b||^2 / (2*sigma^2))
//   [0:N1,         N2:N2+N2*D2] = jacobian_t     K_jt[a, b*D2+d]
//   [N1:N1+N1*D1,  0:N2       ] = jacobian        K_j[a*D1+d, b]
//   [N1:N1+N1*D1,  N2:N2+N2*D2] = hessian         H[a*D1+d1, b*D2+d2]
void kernel_gaussian_full(
    const double *__restrict X1,   // (N1 x M), row-major
    const double *__restrict dX1,  // (N1 blocks) of (M x D1), row-major
    const double *__restrict X2,   // (N2 x M), row-major
    const double *__restrict dX2,  // (N2 blocks) of (M x D2), row-major
    std::size_t N1, std::size_t N2, std::size_t M,
    std::size_t D1, std::size_t D2,
    double sigma, std::size_t tile_B,
    double *__restrict K_full);  // ((N1*(1+D1)) x (N2*(1+D2))), row-major

// Full combined kernel for energy+force KRR (symmetric, X1==X2, D1==D2).
// Output shape: (N*(1+D)) x (N*(1+D)), row-major, lower triangle only.
void kernel_gaussian_full_symm(
    const double *__restrict X,    // (N x M), row-major
    const double *__restrict dX,   // (N blocks) of (M x D), row-major
    std::size_t N, std::size_t M, std::size_t D,
    double sigma, std::size_t tile_B,
    double *__restrict K_full);  // ((N*(1+D)) x (N*(1+D))), row-major, lower triangle

// Full combined kernel for energy+force KRR (symmetric RFP format).
// Output: 1D RFP array of length BIG*(BIG+1)/2 where BIG = N*(1+D).
void kernel_gaussian_full_symm_rfp(
    const double *__restrict X,    // (N x M), row-major
    const double *__restrict dX,   // (N blocks) of (M x D), row-major
    std::size_t N, std::size_t M, std::size_t D,
    double sigma, std::size_t tile_B,
    double *__restrict K_rfp);  // RFP packed, length N*(1+D)*(N*(1+D)+1)/2

}  // namespace kf
