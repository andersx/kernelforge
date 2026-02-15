// C++ standard library
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

// Declare Fortran LAPACK symbols (all vendors provide these)
extern "C" {
    void dpotrf_(const char* uplo, const int* n, double* a,
                 const int* lda, int* info);

    void dpotrs_(const char* uplo, const int* n, const int* nrhs,
                 const double* a, const int* lda, double* b,
                 const int* ldb, int* info);

    void dpftrf_(const char* TRANSR, const char* UPLO, const int* N, double* A, int* INFO);

    void dpftrs_(const char* TRANSR, const char* UPLO, const int* N, const int* NRHS,
             const double* A, double* B, const int* LDB, int* INFO);

    void dtrttf_(const char* TRANSR, const char* UPLO, const int* N,
             const double* A, const int* LDA, double* ARF, int* INFO);

    void dtfttr_(const char* TRANSR, const char* UPLO, const int* N,
             const double* ARF, double* A, const int* LDA, int* INFO);


}

// Solve K * alpha = y using Cholesky factorization.
// K is symmetric positive-definite (will be overwritten).
void solve_cholesky(double* K, const double* y, int n, double* alpha, double regularize)
{
    if (n <= 0) throw std::runtime_error("n must be > 0");
    if (!K || !y) throw std::runtime_error("K and y must be non-null");

    // copy RHS: y -> alpha
    std::memcpy(alpha, y, n * sizeof(double));
    std::vector<double> diagonal(n);
    for (size_t i = 0; i < n; ++i) diagonal[i] = K[i * n + i];

    // jitter on the diagonal for stability
    for (size_t i = 0; i < n; ++i) K[i * n + i] += regularize;

    // factor/use upper triangle in (FORTRAN terms/column-major)
    // This is the same as LAPACK_ROW_MAJOR with 'L' (lower`) for C/C++/Python
    // So contraintuitively we use 'U' (upper) here for lower triangular in C/C++/Python ok
    const char uplo = 'U';
    int info = 0;

    // Cholesky factorization: K = U^T * U (upper)
    dpotrf_(&uplo, &n, K, &n, &info);
    if (info != 0) {
        throw std::runtime_error("Cholesky decomposition failed, info=" + std::to_string(info));
    }

    // Solve U^T * U * alpha = y
    const int nrhs = 1;
    dpotrs_(&uplo, &n, &nrhs, K, &n, alpha, &n, &info);
    if (info != 0) {
        throw std::runtime_error("Cholesky solve failed, info=" + std::to_string(info));
    }

    // reset diagonal
    for (int i = 0; i < n; ++i) K[i * n + i] = diagonal[i];

    // Restore symmetry by mirroring upper triangle to lower triangle
    #pragma omp parallel for schedule(static)
    for (size_t j = 1; j < n; ++j) {
        for (size_t i = 0; i < j; ++i) {
             K[j * n + i] = K[i * n + j];
        }
    }
}


// Diagonal index for TRANSR='N', UPLO='U' (0-based j)
static inline std::size_t rfp_diag_index_upper_N(int n, int j0) {
    const int k = n / 2;
    const int stride = (n % 2 == 0) ? (n + 1) : n;
    if (j0 > k) { // strict >
        return (std::size_t)(j0 - k) * (std::size_t)stride + (std::size_t)j0;
    } else {
        return (std::size_t)j0 * (std::size_t)stride + (std::size_t)(j0 + k + 1);
    }
}

// Map to lower via 180° rotation
static inline std::size_t rfp_diag_index_lower_N(int n, int j0) {
    const int ju = n - 1 - j0;
    return rfp_diag_index_upper_N(n, ju);
}

void solve_cholesky_rfp(double* K_arf, const double* y, int n,
                        double* alpha, double regularize,
                        char uplo, char transr)
{
    if (n <= 0) throw std::runtime_error("n must be > 0");
    if (!K_arf || !y || !alpha) throw std::runtime_error("null pointer");
    if (!(uplo=='U'||uplo=='u'||uplo=='L'||uplo=='l'))
        throw std::runtime_error("uplo must be 'U' or 'L'");
    if (!(transr=='N'||transr=='n')) // keep this simple; extend if you use 'T'
        throw std::runtime_error("Only TRANSR='N' supported here");

    uplo   = (uplo=='l'||uplo=='L') ? 'L' : 'U';
    transr = 'N';

    // Copy RHS
    std::memcpy(alpha, y, (std::size_t)n * sizeof(double));

    // Save + regularize diagonal in RFP (matching the chosen UPLO)
    std::vector<double> diag(n);
    for (int j = 0; j < n; ++j) {
        const std::size_t idx = (uplo=='U')
            ? rfp_diag_index_upper_N(n, j)
            : rfp_diag_index_lower_N(n, j);
        diag[j]      = K_arf[idx];
        K_arf[idx]  += regularize;
    }

    int info = 0;
    dpftrf_(&transr, &uplo, &n, K_arf, &info);
    if (info != 0) {
        for (int j = 0; j < n; ++j) {
            const std::size_t idx = (uplo=='U') ? rfp_diag_index_upper_N(n, j)
                                                : rfp_diag_index_lower_N(n, j);
            K_arf[idx] = diag[j];
        }
        throw std::runtime_error("DPFTRF failed, info=" + std::to_string(info));
    }

    const int nrhs = 1, ldb = n;
    dpftrs_(&transr, &uplo, &n, &nrhs, K_arf, alpha, &ldb, &info);
    if (info != 0) {
        for (int j = 0; j < n; ++j) {
            const std::size_t idx = (uplo=='U') ? rfp_diag_index_upper_N(n, j)
                                                : rfp_diag_index_lower_N(n, j);
            K_arf[idx] = diag[j];
        }
        throw std::runtime_error("DPFTRS failed, info=" + std::to_string(info));
    }

    // Restore diagonal
    for (int j = 0; j < n; ++j) {
        const std::size_t idx = (uplo=='U') ? rfp_diag_index_upper_N(n, j)
                                            : rfp_diag_index_lower_N(n, j);
        K_arf[idx] = diag[j];
    }
}


int full_to_rfp(char transr, char uplo,
                int n, const double* A_colmaj, int lda,
                double* ARF)
{
    int info = 0;
    dtrttf_(&transr, &uplo, &n, A_colmaj, &lda, ARF, &info);
    return info;
}

int rfp_to_full(char transr, char uplo,
                int n, const double* ARF,
                double* A_colmaj, int lda)
{
    int info = 0;
    dtfttr_(&transr, &uplo, &n, ARF, A_colmaj, &lda, &info);
    if (info != 0) return info;

    // Normalize UPLO
    const bool upper = (uplo == 'U' || uplo == 'u');

    if (upper) {
        // Upper triangle is valid -> copy to lower: A(j,i) = A(i,j) for i < j
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < j; ++i) {
                A_colmaj[(size_t)j + (size_t)i * lda] = A_colmaj[(size_t)i + (size_t)j * lda];
            }
        }
    } else {
        // Lower triangle is valid -> copy to upper: A(j,i) = A(i,j) for i > j
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; ++j) {
            for (int i = j + 1; i < n; ++i) {
                A_colmaj[(size_t)j + (size_t)i * lda] = A_colmaj[(size_t)i + (size_t)j * lda];
            }
        }
    }
    return 0;
}
