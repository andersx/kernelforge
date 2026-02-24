#pragma once
#include <cstddef>
#include "blas_int.h"

namespace kf {
namespace math {

void solve_cholesky(double *K, const double *y, blas_int n, double *alpha, const double regularize);

void solve_cholesky_rfp(double *K_arf, const double *y, blas_int n, double *alpha,
                        double regularize, char uplo /* 'U' or 'L' */, char transr = 'N');

// Fortran LAPACK interface (column-major).
// A is full, column-major with lda >= n; only triangle 'uplo' is referenced.
// transr: 'N' or 'T' (RFP storage option). Returns LAPACK info.
blas_int full_to_rfp(char transr, char uplo, blas_int n, const double *A_colmaj, blas_int lda,
                     double *ARF);

// RFP (length n*(n+1)/2) -> Full (column-major A with lda >= n). Returns LAPACK info.
blas_int rfp_to_full(char transr, char uplo, blas_int n, const double *ARF, double *A_colmaj,
                     blas_int lda);

// Least-squares solve via QR/LQ decomposition (DGELS).
// A is C-order m×n; y is length m; x is length n on output.
// A must have full rank.
void solve_qr(const double *A, const double *y, blas_int m, blas_int n, double *x);

// Least-squares solve via divide-and-conquer SVD (DGELSD).
// A is C-order m×n; y is length m; x is length n on output.
// Singular values < rcond*sigma_max are treated as zero (rcond=0 uses machine epsilon).
void solve_svd(const double *A, const double *y, blas_int m, blas_int n, double *x, double rcond);

// 1-norm condition number of a square n×n matrix via LU factorization (DLANGE+DGETRF+DGECON).
// A is C-order n×n; it is not modified (an internal copy is used).
double condition_number_ge(const double *A, blas_int n);

}  // namespace math
}  // namespace kf
