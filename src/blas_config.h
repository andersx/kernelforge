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

#endif

// Include our integer type definition
#include "blas_int.h"
