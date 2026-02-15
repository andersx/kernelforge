#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(_MSC_VER)
    #include <malloc.h>
#endif

// 64-byte aligned allocation for double[]
static inline double *aligned_alloc_64(std::size_t nelems) {
#if defined(_MSC_VER)
    void *p = _aligned_malloc(nelems * sizeof(double), 64);
    if (!p)
        throw std::bad_alloc();
    return static_cast<double *>(p);
#else
    void *p = nullptr;
    if (posix_memalign(&p, 64, nelems * sizeof(double)) != 0) {
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
