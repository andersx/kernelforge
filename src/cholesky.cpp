#include <iostream>
#include <vector>
#include <stdexcept>

// Declare Fortran LAPACK symbols (all vendors provide these)
extern "C" {
    void dpotrf_(const char* uplo, const int* n, double* a,
                 const int* lda, int* info);

    void dpotrs_(const char* uplo, const int* n, const int* nrhs,
                 const double* a, const int* lda, double* b,
                 const int* ldb, int* info);
}

// Solve K * alpha = y using Cholesky factorization.
// K is symmetric positive-definite (will be overwritten).
std::vector<double> solve_cholesky(std::vector<double> K,
                                   const std::vector<double>& y, int n) {
    if ((int)y.size() != n) {
        throw std::runtime_error("Size mismatch: y must have length n");
    }

    std::vector<double> alpha = y; // copy RHS

    const char uplo = 'U';  // use upper triangle
    int info;

    // Factorization: K = U^T * U
    dpotrf_(&uplo, &n, K.data(), &n, &info);
    if (info != 0) {
        throw std::runtime_error("Cholesky decomposition failed, info=" +
                                 std::to_string(info));
    }

    int nrhs = 1;
    // Solve system
    dpotrs_(&uplo, &n, &nrhs, K.data(), &n, alpha.data(), &n, &info);
    if (info != 0) {
        throw std::runtime_error("Cholesky solve failed, info=" +
                                 std::to_string(info));
    }

    return alpha;
}
