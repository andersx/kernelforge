#pragma once
#include "constants.hpp"

#include <cstddef>
#include <cstdlib>
#include <new>

// Aligned allocation for double[] using optimal SIMD/cache alignment
// Uses POSIX posix_memalign (Linux/macOS)
static inline double *aligned_alloc_64(std::size_t nelems) {
    void *p = nullptr;
    if (posix_memalign(&p, kf::ALIGNMENT_BYTES, nelems * sizeof(double)) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<double *>(p);
}

// Free aligned memory allocated by aligned_alloc_64
static inline void aligned_free_64(void *p) {
    std::free(p);
}
