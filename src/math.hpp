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

}  // namespace math
}  // namespace kf
