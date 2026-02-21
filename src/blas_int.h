#pragma once
#include <cstdint>

// Integer type for BLAS/LAPACK interfaces
// - LP64 (default): 32-bit integers (int)
// - ILP64: 64-bit integers
//   - macOS: uses 'long' (64-bit on arm64)
//   - Linux: uses 'int64_t' (equivalent to 'long long' on x86-64)

#ifdef KF_BLAS_ILP64
    #if defined(__APPLE__)
        // Apple Accelerate ILP64: CBLAS uses '__CBLAS_INT' = 'long'
        // LAPACK uses '__LAPACK_int' = 'long' when ACCELERATE_*_ILP64 are defined
        using blas_int = long;
    #else
        // OpenBLAS ILP64: 'blasint' = BLASLONG = 'long long' (int64_t)
        // Intel MKL ILP64: 'MKL_INT' = 'long long int' (equivalent to int64_t on x86-64)
        using blas_int = std::int64_t;
    #endif
#else
    // LP64: standard 32-bit integers
    using blas_int = int;
#endif
