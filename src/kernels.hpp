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

void gaussian_jacobian_batch(
    const double* X1,
    const double* dX1,
    const double* X2,
    std::size_t N1,
    std::size_t N2,
    std::size_t M,
    std::size_t D,     // = 3N for the query molecules
    double sigma,
    double* K_out
);

void rbf_hessian_full_tiled_gemm(
    const double* X1,  const double* dX1,
    const double* X2,  const double* dX2,
    std::size_t N1, std::size_t N2,
    std::size_t M,  std::size_t D1, std::size_t D2,
    double sigma,
    std::size_t tile_B,
    double* H_out);

std::vector<double> solve_cholesky(double* K, const double* y, int n);
