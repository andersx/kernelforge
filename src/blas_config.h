#pragma once

// Unified BLAS/LAPACK header inclusion
// Handles: Apple Accelerate, Intel MKL, OpenBLAS
//
// This header is included by all source files that use BLAS/LAPACK functions.
// It selects the appropriate backend based on compile-time defines:
//   - __APPLE__:   Use Accelerate.framework
//   - KF_USE_MKL:  Use Intel MKL (Linux/Windows)
//   - Otherwise:   Use generic BLAS (OpenBLAS, etc.)
//
// ILP64 support (64-bit integers):
//   - Accelerate:  Define ACCELERATE_BLAS_ILP64 and ACCELERATE_LAPACK_ILP64
//   - MKL:         Define MKL_ILP64
//   - OpenBLAS:    Define OPENBLAS_USE64BITINT

#if defined(__APPLE__)
    // Apple Accelerate framework (macOS)
    // ILP64 defines must be set BEFORE including Accelerate.h
    #ifdef KF_BLAS_ILP64
        #define ACCELERATE_BLAS_ILP64
        #define ACCELERATE_LAPACK_ILP64
    #endif
    #include <Accelerate/Accelerate.h>

#elif defined(KF_USE_MKL)
    // Intel MKL (Linux/Windows)
    // ILP64 define must be set BEFORE including mkl.h
    #ifdef KF_BLAS_ILP64
        #define MKL_ILP64
    #endif
    #include <mkl.h>

#else
    // Generic BLAS (OpenBLAS, ATLAS, reference BLAS, etc.)
    // For OpenBLAS ILP64, OPENBLAS_USE64BITINT must be defined before cblas.h
    #ifdef KF_BLAS_ILP64
        #define OPENBLAS_USE64BITINT
    #endif
    #include <cblas.h>

    // OpenBLAS does not export LAPACKE_dsfrk; declare the Fortran symbol directly.
    // (MKL and Accelerate already declare dsfrk_ in their own headers.)
    // blas_int.h has not been included yet, so use the concrete type directly:
    //   LP64 → int,  ILP64 → long (int64_t on Linux x86_64)
    extern "C" {
    #ifdef KF_BLAS_ILP64
    void dsfrk_(const char *transr, const char *uplo, const char *trans,
                const long *n, const long *k,
                const double *alpha, const double *a, const long *lda,
                const double *beta, double *c);
    #else
    void dsfrk_(const char *transr, const char *uplo, const char *trans,
                const int *n, const int *k,
                const double *alpha, const double *a, const int *lda,
                const double *beta, double *c);
    #endif
    }

#endif

// Include our integer type definition (must precede the kf_dsfrk wrapper below)
#include "blas_int.h"

// kf_dsfrk: portable wrapper for LAPACK's DSFRK (RFP symmetric rank-k update).
// Replaces LAPACKE_dsfrk which is not exported by all OpenBLAS builds.
inline void kf_dsfrk(char transr, char uplo, char trans,
                     blas_int n, blas_int k,
                     double alpha, const double *a, blas_int lda,
                     double beta, double *c) {
#if defined(KF_USE_MKL) && defined(KF_BLAS_ILP64)
    // MKL ILP64: MKL_INT is 'long long' but blas_int is int64_t = 'long' on Linux.
    // Both are 64-bit; cast to match MKL's declaration and suppress the type mismatch.
    const long long n_ll = n, k_ll = k, lda_ll = lda;
    dsfrk_(&transr, &uplo, &trans, &n_ll, &k_ll, &alpha, a, &lda_ll, &beta, c);
#else
    dsfrk_(&transr, &uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c);
#endif
}
