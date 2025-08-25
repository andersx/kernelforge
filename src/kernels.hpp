#pragma once
#include <cstddef>

// Simple 64-byte aligned alloc/free (POSIX)
double* aligned_alloc_64(std::size_t nelems);
void aligned_free_64(void* p);

void kernel_symm(
    const double* Xptr,
    int n,
    int rep_size,
    double alpha,
    double* Kptr
);


void kernel_asymm(
    const double* X1,
    const double* X2,
    std::size_t n1, std::size_t n2, std::size_t d,
    double alpha,
    double* K
);
