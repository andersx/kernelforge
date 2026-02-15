#pragma once
#include "constants.hpp"

#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(_MSC_VER)
    #include <malloc.h>
#endif

// Aligned allocation for double[] using optimal SIMD/cache alignment
static inline double *aligned_alloc_64(std::size_t nelems) {
#if defined(_MSC_VER)
    void *p = _aligned_malloc(nelems * sizeof(double), kf::ALIGNMENT_BYTES);
    if (!p)
        throw std::bad_alloc();
    return static_cast<double *>(p);
#else
    void *p = nullptr;
    if (posix_memalign(&p, kf::ALIGNMENT_BYTES, nelems * sizeof(double)) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<double *>(p);
#endif
}

static inline void aligned_free_64(void *p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}
