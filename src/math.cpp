// C++ standard library
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// Project headers
#include "blas_int.h"

namespace kf {
namespace math {

// Declare Fortran LAPACK symbols (all vendors provide these)
extern "C" {
void dpotrf_(const char *uplo, const blas_int *n, double *a, const blas_int *lda, blas_int *info);

void dgels_(
    const char *trans, const blas_int *m, const blas_int *n, const blas_int *nrhs, double *a,
    const blas_int *lda, double *b, const blas_int *ldb, double *work, const blas_int *lwork,
    blas_int *info
);

void dgelsd_(
    const blas_int *m, const blas_int *n, const blas_int *nrhs, double *a, const blas_int *lda,
    double *b, const blas_int *ldb, double *s, const double *rcond, blas_int *rank, double *work,
    const blas_int *lwork, blas_int *iwork, blas_int *info
);

double dlange_(
    const char *norm, const blas_int *m, const blas_int *n, const double *a, const blas_int *lda,
    double *work
);

void dgetrf_(
    const blas_int *m, const blas_int *n, double *a, const blas_int *lda, blas_int *ipiv,
    blas_int *info
);

void dgecon_(
    const char *norm, const blas_int *n, const double *a, const blas_int *lda, const double *anorm,
    double *rcond, double *work, blas_int *iwork, blas_int *info
);

void dpotrs_(
    const char *uplo, const blas_int *n, const blas_int *nrhs, const double *a, const blas_int *lda,
    double *b, const blas_int *ldb, blas_int *info
);

void dpftrf_(const char *TRANSR, const char *UPLO, const blas_int *N, double *A, blas_int *INFO);

void dpftrs_(
    const char *TRANSR, const char *UPLO, const blas_int *N, const blas_int *NRHS, const double *A,
    double *B, const blas_int *LDB, blas_int *INFO
);

void dtrttf_(
    const char *TRANSR, const char *UPLO, const blas_int *N, const double *A, const blas_int *LDA,
    double *ARF, blas_int *INFO
);

void dtfttr_(
    const char *TRANSR, const char *UPLO, const blas_int *N, const double *ARF, double *A,
    const blas_int *LDA, blas_int *INFO
);
}

// Solve K * alpha = y using Cholesky factorization.
// K is symmetric positive-definite (will be overwritten).
void solve_cholesky(double *K, const double *y, blas_int n, double *alpha, double regularize) {
    if (n <= 0) throw std::runtime_error("n must be > 0");
    if (!K || !y) throw std::runtime_error("K and y must be non-null");

    const std::size_t n_size = static_cast<std::size_t>(n);

    // copy RHS: y -> alpha
    std::memcpy(alpha, y, n_size * sizeof(double));
    std::vector<double> diagonal(n_size);
    for (std::size_t i = 0; i < n_size; ++i)
        diagonal[i] = K[i * n_size + i];

    // jitter on the diagonal for stability
    for (std::size_t i = 0; i < n_size; ++i)
        K[i * n_size + i] += regularize;

    // factor/use upper triangle in (FORTRAN terms/column-major)
    // This is the same as LAPACK_ROW_MAJOR with 'L' (lower`) for C/C++/Python
    // So contraintuitively we use 'U' (upper) here for lower triangular in C/C++/Python ok
    const char uplo = 'U';
    blas_int info = 0;

    // Cholesky factorization: K = U^T * U (upper)
    dpotrf_(&uplo, &n, K, &n, &info);
    if (info != 0) {
        throw std::runtime_error("Cholesky decomposition failed, info=" + std::to_string(info));
    }

    // Solve U^T * U * alpha = y
    const blas_int nrhs = 1;
    dpotrs_(&uplo, &n, &nrhs, K, &n, alpha, &n, &info);
    if (info != 0) {
        throw std::runtime_error("Cholesky solve failed, info=" + std::to_string(info));
    }

    // reset diagonal
    for (std::size_t i = 0; i < n_size; ++i)
        K[i * n_size + i] = diagonal[i];

// Restore symmetry by mirroring upper triangle to lower triangle
#pragma omp parallel for schedule(static)
    for (std::size_t j = 1; j < n_size; ++j) {
        for (std::size_t i = 0; i < j; ++i) {
            K[j * n_size + i] = K[i * n_size + j];
        }
    }
}

// Diagonal index for TRANSR='N', UPLO='U' (0-based j)
static inline std::size_t rfp_diag_index_upper_N(blas_int n, blas_int j0) {
    const blas_int k = n / 2;
    const blas_int stride = (n % 2 == 0) ? (n + 1) : n;
    if (j0 >= k) {  // strict >
        return (std::size_t)(j0 - k) * (std::size_t)stride + (std::size_t)j0;
    } else {
        return (std::size_t)j0 * (std::size_t)stride + (std::size_t)(j0 + k + 1);
    }
}

// Map to lower via 180° rotation
static inline std::size_t rfp_diag_index_lower_N(blas_int n, blas_int j0) {
    const blas_int ju = n - 1 - j0;
    return rfp_diag_index_upper_N(n, ju);
}

void solve_cholesky_rfp(
    double *K_arf, const double *y, blas_int n, double *alpha, double regularize, char uplo,
    char transr
) {
    if (n <= 0) throw std::runtime_error("n must be > 0");
    if (!K_arf || !y || !alpha) throw std::runtime_error("null pointer");
    if (!(uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l'))
        throw std::runtime_error("uplo must be 'U' or 'L'");
    if (!(transr == 'N' || transr == 'n'))
        throw std::runtime_error("Only TRANSR='N' supported here");

    uplo = (uplo == 'l' || uplo == 'L') ? 'L' : 'U';
    transr = 'N';

    const std::size_t n_size = static_cast<std::size_t>(n);

    // Copy RHS into alpha
    std::memcpy(alpha, y, n_size * sizeof(double));

    // Add regularization to the diagonal of K_arf in-place.
    // The caller is responsible for providing a working copy if preservation is needed
    // (cho_solve_rfp_py always does this; solve_cholesky_rfp_L documents the overwrite).
    if (regularize != 0.0) {
        for (blas_int j = 0; j < n; ++j) {
            const std::size_t idx =
                (uplo == 'U') ? rfp_diag_index_upper_N(n, j) : rfp_diag_index_lower_N(n, j);
            K_arf[idx] += regularize;
        }
    }

    blas_int info = 0;
    dpftrf_(&transr, &uplo, &n, K_arf, &info);
    if (info != 0) throw std::runtime_error("DPFTRF failed, info=" + std::to_string(info));

    const blas_int nrhs = 1, ldb = n;
    dpftrs_(&transr, &uplo, &n, &nrhs, K_arf, alpha, &ldb, &info);
    if (info != 0) throw std::runtime_error("DPFTRS failed, info=" + std::to_string(info));
}

blas_int full_to_rfp(
    char transr, char uplo, blas_int n, const double *A_colmaj, blas_int lda, double *ARF
) {
    blas_int info = 0;
    dtrttf_(&transr, &uplo, &n, A_colmaj, &lda, ARF, &info);
    return info;
}

blas_int rfp_to_full(
    char transr, char uplo, blas_int n, const double *ARF, double *A_colmaj, blas_int lda
) {
    blas_int info = 0;
    dtfttr_(&transr, &uplo, &n, ARF, A_colmaj, &lda, &info);
    if (info != 0) return info;

    // Normalize UPLO
    const bool upper = (uplo == 'U' || uplo == 'u');
    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t lda_size = static_cast<std::size_t>(lda);

    if (upper) {
// Upper triangle is valid -> copy to lower: A(j,i) = A(i,j) for i < j
#pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < n_size; ++j) {
            for (std::size_t i = 0; i < j; ++i) {
                A_colmaj[j + i * lda_size] = A_colmaj[i + j * lda_size];
            }
        }
    } else {
// Lower triangle is valid -> copy to upper: A(j,i) = A(i,j) for i > j
#pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < n_size; ++j) {
            for (std::size_t i = j + 1; i < n_size; ++i) {
                A_colmaj[j + i * lda_size] = A_colmaj[i + j * lda_size];
            }
        }
    }
    return 0;
}

// Solve min||A x - y||_2 using QR/LQ decomposition (DGELS, 'N' transpose).
// A is C-order (row-major) m×n; a temporary column-major copy is made internally.
void solve_qr(const double *A, const double *y, blas_int m, blas_int n, double *x) {
    if (m <= 0 || n <= 0) throw std::runtime_error("m and n must be > 0");
    if (!A || !y || !x) throw std::runtime_error("null pointer");

    const std::size_t m_sz = static_cast<std::size_t>(m);
    const std::size_t n_sz = static_cast<std::size_t>(n);
    const blas_int ldb = std::max(m, n);
    const std::size_t ldb_sz = static_cast<std::size_t>(ldb);

    // Transpose A from C-order (row-major) to column-major for LAPACK
    std::vector<double> A_f(m_sz * n_sz);
    for (std::size_t i = 0; i < m_sz; ++i)
        for (std::size_t j = 0; j < n_sz; ++j)
            A_f[j * m_sz + i] = A[i * n_sz + j];

    // RHS buffer: size max(m,n), zero-padded; first m elements = y
    std::vector<double> B(ldb_sz, 0.0);
    std::memcpy(B.data(), y, m_sz * sizeof(double));

    const char trans = 'N';
    const blas_int nrhs = 1;
    blas_int info = 0;

    // Workspace query
    blas_int lwork = -1;
    double work_query = 0.0;
    dgels_(&trans, &m, &n, &nrhs, A_f.data(), &m, B.data(), &ldb, &work_query, &lwork, &info);
    lwork = static_cast<blas_int>(work_query);

    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgels_(&trans, &m, &n, &nrhs, A_f.data(), &m, B.data(), &ldb, work.data(), &lwork, &info);

    if (info != 0) throw std::runtime_error("DGELS failed, info=" + std::to_string(info));

    // Solution is in B[0:n]
    std::memcpy(x, B.data(), n_sz * sizeof(double));
}

// Solve min||A x - y||_2 using divide-and-conquer SVD (DGELSD).
// A is C-order (row-major) m×n; a temporary column-major copy is made internally.
void solve_svd(const double *A, const double *y, blas_int m, blas_int n, double *x, double rcond) {
    if (m <= 0 || n <= 0) throw std::runtime_error("m and n must be > 0");
    if (!A || !y || !x) throw std::runtime_error("null pointer");

    const std::size_t m_sz = static_cast<std::size_t>(m);
    const std::size_t n_sz = static_cast<std::size_t>(n);
    const blas_int ldb = std::max(m, n);
    const std::size_t ldb_sz = static_cast<std::size_t>(ldb);
    const std::size_t s_sz = static_cast<std::size_t>(std::min(m, n));

    // Transpose A from C-order (row-major) to column-major for LAPACK
    std::vector<double> A_f(m_sz * n_sz);
    for (std::size_t i = 0; i < m_sz; ++i)
        for (std::size_t j = 0; j < n_sz; ++j)
            A_f[j * m_sz + i] = A[i * n_sz + j];

    // RHS buffer: size max(m,n), zero-padded; first m elements = y
    std::vector<double> B(ldb_sz, 0.0);
    std::memcpy(B.data(), y, m_sz * sizeof(double));

    std::vector<double> S(s_sz);
    blas_int rank = 0;
    blas_int info = 0;
    const blas_int nrhs = 1;

    // Workspace query
    blas_int lwork_q = -1;
    double work_q = 0.0;
    blas_int iwork_q = 0;
    dgelsd_(
        &m,
        &n,
        &nrhs,
        A_f.data(),
        &m,
        B.data(),
        &ldb,
        S.data(),
        &rcond,
        &rank,
        &work_q,
        &lwork_q,
        &iwork_q,
        &info
    );

    const blas_int lwork = static_cast<blas_int>(work_q);
    const blas_int liwork = iwork_q;

    std::vector<double> work(static_cast<std::size_t>(lwork));
    std::vector<blas_int> iwork(static_cast<std::size_t>(liwork));

    dgelsd_(
        &m,
        &n,
        &nrhs,
        A_f.data(),
        &m,
        B.data(),
        &ldb,
        S.data(),
        &rcond,
        &rank,
        work.data(),
        &lwork,
        iwork.data(),
        &info
    );

    if (info != 0) throw std::runtime_error("DGELSD failed, info=" + std::to_string(info));

    // Solution is in B[0:n]
    std::memcpy(x, B.data(), n_sz * sizeof(double));
}

// 1-norm condition number of square n×n matrix via LU factorization.
// A is C-order; no transposition needed since cond(A) = cond(A^T).
double condition_number_ge(const double *A, blas_int n) {
    if (n <= 0) throw std::runtime_error("n must be > 0");
    if (!A) throw std::runtime_error("null pointer");

    const std::size_t n_sz = static_cast<std::size_t>(n);

    // Copy A (DGETRF overwrites the matrix)
    std::vector<double> A_copy(n_sz * n_sz);
    std::memcpy(A_copy.data(), A, n_sz * n_sz * sizeof(double));

    const char norm = '1';

    // Compute 1-norm of A (Fortran sees A^T, but DLANGE('1') on A^T = DLANGE('I') on A)
    // For condition number: cond(A) = cond(A^T), so this is fine.
    std::vector<double> work_norm(n_sz);
    const double anorm = dlange_(&norm, &n, &n, A_copy.data(), &n, work_norm.data());

    // LU factorization
    std::vector<blas_int> ipiv(n_sz);
    blas_int info = 0;
    dgetrf_(&n, &n, A_copy.data(), &n, ipiv.data(), &info);
    if (info != 0) throw std::runtime_error("DGETRF failed, info=" + std::to_string(info));

    // Estimate reciprocal condition number
    double rcond = 0.0;
    std::vector<double> work_con(static_cast<std::size_t>(4 * n));
    std::vector<blas_int> iwork_con(n_sz);
    dgecon_(&norm, &n, A_copy.data(), &n, &anorm, &rcond, work_con.data(), iwork_con.data(), &info);
    if (info != 0) throw std::runtime_error("DGECON failed, info=" + std::to_string(info));

    if (rcond == 0.0) return std::numeric_limits<double>::infinity();

    return 1.0 / rcond;
}

}  // namespace math
}  // namespace kf
