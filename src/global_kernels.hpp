#pragma once
#include "blas_int.h"

#include <cstddef>

namespace kf {

void kernel_gaussian_symm(
    const double *Xptr, blas_int n, blas_int rep_size, double alpha, double *Kptr
);

void kernel_gaussian_symm_rfp(
    const double *Xptr, blas_int n, blas_int rep_size, double alpha, double *arf
);

void kernel_gaussian(
    const double *X1, const double *X2, std::size_t n1, std::size_t n2, std::size_t d, double alpha,
    double *K
);

void kernel_gaussian_jacobian(
    const double *X1, const double *dX1, const double *X2, std::size_t N1, std::size_t N2,
    std::size_t M,
    std::size_t D,  // = 3N for the query molecules
    double sigma, double *K_out
);

void kernel_gaussian_jacobian_t(
    const double *X1, const double *X2, const double *dX2, std::size_t N1, std::size_t N2,
    std::size_t M,
    std::size_t D,  // = 3N for the reference molecules
    double sigma, double *K_out
);

void kernel_gaussian_hessian(
    const double *X1, const double *dX1, const double *X2, const double *dX2, std::size_t N1,
    std::size_t N2, std::size_t M, std::size_t D1, std::size_t D2, double sigma, std::size_t tile_B,
    double *H_out
);

void kernel_gaussian_hessian_symm(
    const double *__restrict X,   // (N x M), row-major
    const double *__restrict dX,  // per-sample Jacobians, (N blocks) of (M x D), row-major
    std::size_t N, std::size_t M, std::size_t D, double sigma, std::size_t tile_B,
    double *__restrict H_out
);  // (N*D x N*D), row-major

void kernel_gaussian_hessian_symm_rfp(
    const double *__restrict X,   // (N x M), row-major
    const double *__restrict dX,  // per-sample Jacobians, (N blocks) of (M x D), row-major
    std::size_t N, std::size_t M, std::size_t D, double sigma, std::size_t tile_B,
    double *__restrict H_rfp
);  // RFP packed format, length N*D*(N*D+1)/2

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
    std::size_t N1, std::size_t N2, std::size_t M, std::size_t D1, std::size_t D2, double sigma,
    std::size_t tile_B,
    double *__restrict K_full
);  // ((N1*(1+D1)) x (N2*(1+D2))), row-major

// Full combined kernel for energy+force KRR (symmetric, X1==X2, D1==D2).
// Output shape: (N*(1+D)) x (N*(1+D)), row-major, lower triangle only.
void kernel_gaussian_full_symm(
    const double *__restrict X,   // (N x M), row-major
    const double *__restrict dX,  // (N blocks) of (M x D), row-major
    std::size_t N, std::size_t M, std::size_t D, double sigma, std::size_t tile_B,
    double *__restrict K_full
);  // ((N*(1+D)) x (N*(1+D))), row-major, lower triangle

// Full combined kernel for energy+force KRR (symmetric RFP format).
// Output: 1D RFP array of length BIG*(BIG+1)/2 where BIG = N*(1+D).
void kernel_gaussian_full_symm_rfp(
    const double *__restrict X,   // (N x M), row-major
    const double *__restrict dX,  // (N blocks) of (M x D), row-major
    std::size_t N, std::size_t M, std::size_t D, double sigma, std::size_t tile_B,
    double *__restrict K_rfp
);  // RFP packed, length N*(1+D)*(N*(1+D)+1)/2

// ============================================================================
// J^T·α Trick: Efficient force prediction via descriptor-space coefficients.
//
// Pre-compute (once after training):
//   alpha_desc[m] = J(r_train[m])^T · α_m  ∈ ℝ^D
// At prediction time (for each query r*):
//   G(d*) = Σ_m H_k(d*, d_m) · alpha_desc[m]   [O(M·D) in descriptor space]
//   F(r*) = J(r*)^T · G(d*)                     [O(3N·D) back-projection, once per query]
// Cost reduction: ~26× speedup + ~28× memory savings vs full matrix approach.
// ============================================================================

// Compute descriptor-space force coefficients: α̃[m,k] = J[m,d,k]^T · α[m,d]
// Shapes: dX(N,D,M), alpha(N,D) -> alpha_desc(N,M)
void kernel_gaussian_compute_alpha_desc(
    const double *dX,  // (N, D, M) training Jacobians, row-major
    const double *alpha,    // (N, D) KRR force coefficients, row-major
    std::size_t N, std::size_t D, std::size_t M,
    double *alpha_desc  // (N, M) output descriptor-space coefficients
);

// Efficient force prediction via Hessian kernel matvec using J^T·α trick.
// Cost: O(N_q·N_t·M + N_q·D·M) vs O(N_q·N_t·D·M) for full matrix.
// Shapes: X_q(N_q,M), dX_q(N_q,D,M), X_t(N_t,M), alpha_desc(N_t,M) -> F_out(N_q,D)
void kernel_gaussian_hessian_matvec(
    const double *__restrict X_q,         // (N_q, M) query descriptors
    const double *__restrict dX_q,        // (N_q, D, M) query Jacobians
    const double *__restrict X_t,         // (N_t, M) training descriptors
    const double *__restrict alpha_desc,  // (N_t, M) pre-computed J^T·α coefficients
    std::size_t N_q, std::size_t N_t, std::size_t M, std::size_t D, double sigma,
    double *__restrict F_out  // (N_q, D) output forces
);

}  // namespace kf
