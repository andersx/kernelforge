// C++ standard library
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

// Third-party libraries
#include <omp.h>
#if defined(__APPLE__)
    #ifdef KF_BLAS_ILP64
        #define ACCELERATE_BLAS_ILP64
        #define ACCELERATE_LAPACK_ILP64
    #endif
    #include <Accelerate/Accelerate.h>
#else
    #ifdef KF_BLAS_ILP64
        #define OPENBLAS_USE64BITINT
    #endif
    #include <cblas.h>
#endif

// Project headers
#include "aligned_alloc64.hpp"
#include "blas_int.h"
#include "constants.hpp"

namespace kf {

void kernel_gaussian_symm(const double *Xptr, blas_int n, blas_int rep_size, double alpha,
                          double *Kptr) {
    // 1) DSYRK (RowMajor, lower triangle): K = (-2*alpha) * X * X^T + 0*K
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, n, rep_size, -2.0 * alpha, Xptr, rep_size,
                0.0, Kptr, n);

    // 2) diag = -0.5 * diag(K)
    const std::size_t n_size = static_cast<std::size_t>(n);
    std::vector<double> diag(n_size);
    for (blas_int i = 0; i < n; ++i) {
        diag[i] = -0.5 * Kptr[i * n + i];
    }

    // 3) K += 1 * (1*diag^T + diag*1^T) on LOWER via dsyr2
    std::vector<double> onevec(n_size, 1.0);
    cblas_dsyr2(CblasRowMajor, CblasLower, n, 1.0, onevec.data(), 1, diag.data(), 1, Kptr, n);

// 4) Elementwise exp on the lower triangle (i <= j)
#pragma omp parallel for schedule(guided)
    for (blas_int j = 0; j < n; ++j) {
        for (blas_int i = 0; i <= j; ++i) {
            Kptr[j * n + i] = std::exp(Kptr[j * n + i]);
        }
    }
}

static inline void rowwise_self_norms(const double *X, std::size_t n, std::size_t d, double *out) {
    // out[i] = sum_k X[i, k]^2  (row-major: row i is contiguous)
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = X + i * d;
        double acc = 0.0;
        for (std::size_t k = 0; k < d; ++k)
            acc += row[k] * row[k];
        out[i] = acc;
    }
}

void kernel_gaussian(const double *X1, const double *X2, std::size_t n1, std::size_t n2, std::size_t d,
                     double alpha, double *K) {
    // 1) K = (-2*alpha) * X1 * X2^T
    // RowMajor: A=X1 (n1 x d), B=X2 (n2 x d) but we pass Trans(B) => (d x n2)
    // lda = d, ldb = d, ldc = n1
    const double gemm_alpha = -2.0 * alpha;
    const double gemm_beta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(n1),                 // M
                static_cast<int>(n2),                 // N
                static_cast<int>(d),                  // K
                gemm_alpha, X1, static_cast<int>(d),  // A, lda
                X2, static_cast<int>(d),              // B, ldb  (Trans)
                gemm_beta, K, static_cast<int>(n2));  // C

    // 2) Rowwise self norms
    std::vector<double> nrm1(n1), nrm2(n2);
    rowwise_self_norms(X1, n1, d, nrm1.data());
    rowwise_self_norms(X2, n2, d, nrm2.data());

    // Ones
    std::vector<double> one_n1(n1, 1.0), one_n2(n2, 1.0);

    // 3) K += alpha * (ones_n2 * nrm1^T)   => GER(m=n1, n=n2)
    cblas_dger(CblasRowMajor, static_cast<int>(n1), static_cast<int>(n2), alpha, one_n1.data(), 1,
               nrm2.data(), 1, K, static_cast<int>(n2));

    // 4) K += alpha * (nrm2 * ones_n1^T)
    cblas_dger(CblasRowMajor, static_cast<int>(n1), static_cast<int>(n2), alpha, nrm1.data(), 1,
               one_n2.data(), 1, K, static_cast<int>(n2));

    // 5) Elementwise exp
    const std::size_t total = n2 * n1;
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(total); ++idx) {
        K[static_cast<std::size_t>(idx)] = std::exp(K[static_cast<std::size_t>(idx)]);
    }
}

void kernel_gaussian_jacobian(const double *X1, const double *dX1, const double *X2, std::size_t N1,
                             std::size_t N2, std::size_t M, std::size_t D, double sigma,
                             double *K_out) {
    if (!X1 || !dX1 || !X2 || !K_out)
        throw std::invalid_argument("null pointer");
    if (N1 == 0 || N2 == 0 || M == 0 || D == 0)
        throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0");

    const double inv_s2 = 1.0 / (sigma * sigma);

    // scratch for W_a: (M x N2) row-major, and a column diff
    double *W = aligned_alloc_64(static_cast<size_t>(M) * N2);
    std::vector<double> diff(M);

    for (std::size_t a = 0; a < N1; ++a) {
        const double *X1a = X1 + a * M;
        const double *J1a = dX1 + (a * M) * D;  // (M,D) block base
        // Build W(:,b) for all b
        for (std::size_t b = 0; b < N2; ++b) {
            const double *X2b = X2 + b * M;

            double sq = 0.0;
            for (std::size_t i = 0; i < M; ++i) {
                const double di = X2b[i] - X1a[i];
                diff[i] = di;
                sq += di * di;
            }

            const double k = std::exp(-0.5 * inv_s2 * sq);
            const double coeff = k * inv_s2;

            // W(i,b) with row-major (M x N2): index = i*N2 + b
            for (std::size_t i = 0; i < M; ++i) {
                W[i * N2 + b] = coeff * diff[i];
            }
        }

        // Output block for this 'a': rows [a*D : (a+1)*D), all columns b=0..N2-1
        double *Kblock = K_out + (a * D) * N2;

        // Kblock(D x N2) = (J1a^T)(D x M) * W(M x N2)
        // J1a is (M x D) row-major in memory.
        // Row-major DGEMM: op(A)=A^T ⇒ (D x M), lda = D; B=W (M x N2), ldb=N2; C(Kblock) (D x N2),
        // ldc=N2
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(D),
                    static_cast<int>(N2), static_cast<int>(M), 1.0, J1a, static_cast<int>(D),
                    W, static_cast<int>(N2), 0.0, Kblock, static_cast<int>(N2));
    }

    aligned_free_64(W);
}

static inline void row_axpy(std::size_t n, double alpha, const double *x, double *y) {
    for (std::size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}
// If using MKL and you want local control of threads around level-1/2 ops:
// #include <mkl.h>

void kernel_gaussian_hessian(const double *__restrict X1, const double *__restrict dX1,
                                 const double *__restrict X2, const double *__restrict dX2,
                                 std::size_t N1, std::size_t N2, std::size_t M, std::size_t D1,
                                 std::size_t D2, double sigma,
                                 std::size_t tile_B,  // now used
                                 double *__restrict H_out) {
    if (!X1 || !dX1 || !X2 || !dX2 || !H_out)
        throw std::invalid_argument("null pointer");
    if (N1 == 0 || N2 == 0 || M == 0 || D1 == 0 || D2 == 0)
        throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0");

    const std::size_t big_rows = N1 * D1;
    const std::size_t big_cols = N2 * D2;

    const double inv_s2 = 1.0 / (sigma * sigma);
    const double inv_s4 = inv_s2 * inv_s2;

    // 1) Distances → C = K/s^2 and C4 = K/s^4
    std::vector<double> n1(N1), n2(N2);

#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        const double *__restrict x = X1 + a * M;
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (std::size_t i = 0; i < M; ++i)
            s += x[i] * x[i];
        n1[a] = s;
    }

#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        const double *__restrict x = X2 + b * M;
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (std::size_t i = 0; i < M; ++i)
            s += x[i] * x[i];
        n2[b] = s;
    }

    // S = X1 @ X2^T  (N1 x N2)  ← let BLAS thread this
    double *S = aligned_alloc_64(N1 * N2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N1, (int)N2, (int)M, 1.0, X1, (int)M,
                X2, (int)M, 0.0, S, (int)N2);

    // C, C4
    double *C = aligned_alloc_64(N1 * N2);
    double *C4 = aligned_alloc_64(N1 * N2);
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        for (std::size_t b = 0; b < N2; ++b) {
            const double sq = n1[a] + n2[b] - 2.0 * S[a * N2 + b];
            const double k = std::exp(-0.5 * inv_s2 * sq);
            C[a * N2 + b] = k * inv_s2;
            C4[a * N2 + b] = k * inv_s4;
        }
    }

    // 2) Pack J1_hat (N1*D1 x M),  J2_all_cat (M x N2*D2)
    double *J1_hat = aligned_alloc_64(big_rows * M);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        const double *__restrict J1 = dX1 + (a * M) * D1;  // (M x D1)
        for (std::size_t dj = 0; dj < D1; ++dj) {
            double *__restrict row = J1_hat + (a * D1 + dj) * M;
// transpose column dj of (M x D1) → row of length M
#pragma omp simd
            for (std::size_t i = 0; i < M; ++i)
                row[i] = J1[i * D1 + dj];
        }
    }

    double *J2_all_cat = aligned_alloc_64(M * big_cols);
// Parallelize outer over b to keep writes mostly contiguous per thread
#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        const double *__restrict J2 = dX2 + (b * M) * D2;  // (M x D2)
        for (std::size_t i = 0; i < M; ++i) {
            std::memcpy(J2_all_cat + i * big_cols + b * D2, J2 + i * D2,
                        D2 * sizeof(double));
        }
    }
    const int lda_J1 = (int)M;
    const int ldb_J2cat = (int)big_cols;
    const int ldc_H = (int)big_cols;

    // 3) Base Gram for ALL blocks once: H_out = J1_hat @ J2_all_cat  (threaded BLAS)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)big_rows, (int)big_cols, (int)M,
                1.0, J1_hat, lda_J1, J2_all_cat, ldb_J2cat, 0.0, H_out, ldc_H);

    // 4) Projection tables: V1X2_all and V2X1_all  (threaded BLAS)
    double *V1X2_all = aligned_alloc_64(big_rows * N2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)big_rows, (int)N2, (int)M, 1.0,
                J1_hat, (int)M, X2, (int)M, 0.0, V1X2_all, (int)N2);

    double *V2X1_all = aligned_alloc_64(big_cols * N1);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, (int)big_cols, (int)N1, (int)M, 1.0,
                J2_all_cat, (int)big_cols, X1, (int)M, 0.0, V2X1_all, (int)N1);

    // 5) Self projections: U1[a]=J1_a^T x1_a, U2[b]=J2_b^T x2_b  (parallelize outer loops)
    std::vector<double> U1(N1 * D1), U2(N2 * D2);

#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        const double *__restrict x1 = X1 + a * M;
        const double *__restrict J1 = dX1 + (a * M) * D1;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D1, 1.0, J1, (int)D1, x1, 1, 0.0,
                    U1.data() + a * D1, 1);
    }

#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        const double *__restrict x2 = X2 + b * M;
        const double *__restrict J2 = dX2 + (b * M) * D2;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D2, 1.0, J2, (int)D2, x2, 1, 0.0,
                    U2.data() + b * D2, 1);
    }

    // 6) Per-block: scale Gram by C[a,b], then rank-1 correction via GER
    if (tile_B == 0)
        tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N2);  // reasonable default

    // If you can, make L1/L2 BLAS single-threaded here to avoid oversubscription:
    // int saved = mkl_get_max_threads();
    // mkl_set_num_threads_local(1);

#pragma omp parallel
    {
        std::vector<double> v1(D1), v2(D2);  // private buffers per thread

#pragma omp for schedule(static)
        for (std::size_t a = 0; a < N1; ++a) {
            const double *__restrict U1a = U1.data() + a * D1;
            const double *__restrict V1X2a =
                V1X2_all + (a * D1) * N2;  // (D1 x N2), row-major

            for (std::size_t b0 = 0; b0 < N2; b0 += tile_B) {
                const std::size_t bend = std::min(N2, b0 + tile_B);

                for (std::size_t b = b0; b < bend; ++b) {
                    const double cab = C[a * N2 + b];
                    const double c4ab = C4[a * N2 + b];

                    double *__restrict Hblk = H_out + (a * D1) * big_cols + b * D2;

                    // Scale Gram block by cab: D1 rows × D2 cols within the big row-major matrix
                    for (std::size_t i = 0; i < D1; ++i) {
                        cblas_dscal((int)D2, cab, Hblk + i * big_cols, 1);
                    }

// v1 = U1[a] - V1X2_all[aD1:(a+1)D1, b]
#pragma omp simd
                    for (std::size_t i = 0; i < D1; ++i) {
                        v1[i] = U1a[i] - V1X2a[i * N2 + b];
                    }

                    // v2 = V2X1_all[bD2:(b+1)D2, a] - U2[b]
                    const double *__restrict U2b = U2.data() + b * D2;
                    const double *__restrict V2X1col =
                        V2X1_all + (b * D2) * N1 + a;  // column 'a' in row-major
#pragma omp simd
                    for (std::size_t j = 0; j < D2; ++j) {
                        v2[j] = V2X1col[j * N1] - U2b[j];
                    }

                    // Rank-1 correction: H(a,b) -= c4ab * v1 v2^T  (Level-2 BLAS)
                    // Probably single-threaded
                    cblas_dger(CblasRowMajor, (int)D1, (int)D2, -c4ab, v1.data(), 1, v2.data(), 1,
                               Hblk, (int)big_cols);
                }
            }
        }
    }

    // mkl_set_num_threads_local(saved);

    // Free aligned allocations
    aligned_free_64(V2X1_all);
    aligned_free_64(V1X2_all);
    aligned_free_64(J2_all_cat);
    aligned_free_64(J1_hat);
    aligned_free_64(C4);
    aligned_free_64(C);
    aligned_free_64(S);
}

// Symmetric (training) RBF Hessian kernel, Gaussian.
// X: (N, M) row-major          – descriptor vectors
// dX: (N, M, D) row-major      – Jacobians wrt descriptor coords (M x D) per sample
// Output H_out: (N*D, N*D) row-major, symmetric
void kernel_gaussian_hessian_symm(
    // Symmetric (training) RBF Hessian, Gaussian kernel.
    // Assumes: X2==X1 (same set), dX2==dX1, N1==N2, D1==D2.
    // Builds only the lower triangle and mirrors to upper.
    // void rbf_hessian_full_tiled_gemm_symmetric(
    const double *__restrict X1, const double *__restrict dX1, std::size_t N1, std::size_t M,
    std::size_t D1, double sigma, std::size_t tile_B, double *__restrict H_out) {
    if (!X1 || !dX1 || !H_out)
        throw std::invalid_argument("null pointer");
    if (N1 == 0 || M == 0 || D1 == 0)
        throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0");

    const std::size_t N = N1;
    const std::size_t D = D1;
    const std::size_t BIG = N * D;

    const double inv_s2 = 1.0 / (sigma * sigma);
    const double inv_s4 = inv_s2 * inv_s2;

    // Zero output once (we'll fill lower and mirror to upper at the end)
    // std::fill(H_out, H_out + BIG*BIG, 0.0);

    // 1) Distances → C = K/s^2 and C4 = K/s^4
    std::vector<double> n(N);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *__restrict x = X1 + a * M;
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (std::size_t i = 0; i < M; ++i)
            s += x[i] * x[i];
        n[a] = s;
    }

    // S = X1 @ X1^T (lower) via DSYRK, then mirror to upper for convenient reads
    double *S = aligned_alloc_64(N * N);
    std::memset(S, 0, N * N * sizeof(double));
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, (int)N, (int)M, 1.0, X1, (int)M, 0.0,
                S, (int)N);
#pragma omp parallel for
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            S[i * N + j] = S[j * N + i];
        }
    }

    double *C = aligned_alloc_64(N * N);
    double *C4 = aligned_alloc_64(N * N);
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            const double sq = n[a] + n[b] - 2.0 * S[a * N + b];
            const double k = std::exp(-0.5 * inv_s2 * sq);
            C[a * N + b] = k * inv_s2;
            C4[a * N + b] = k * inv_s4;
        }
    }

    // 2) Pack J_hat (N*D x M) from dX1 (N,M,D): row i=(a*D+dj) is column dj of (M x D)
    double *J_hat = aligned_alloc_64(BIG * M);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *__restrict J = dX1 + (a * M) * D;  // (M x D)
        for (std::size_t dj = 0; dj < D; ++dj) {
            double *__restrict row = J_hat + (a * D + dj) * M;
#pragma omp simd
            for (std::size_t i = 0; i < M; ++i)
                row[i] = J[i * D + dj];
        }
    }

    // 3) Base Gram only lower: H_out := J_hat @ J_hat^T  via DSYRK (lower)
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, (int)BIG, (int)M, 1.0, J_hat,
                (int)M, 0.0, H_out, (int)BIG);

    // 4) Projection table V = J_hat @ X1^T  (BIG x N)
    double *V = aligned_alloc_64(BIG * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)BIG, (int)N, (int)M, 1.0,
                J_hat, (int)M, X1, (int)M, 0.0, V, (int)N);

    // 5) Self projections: U[a] = J_a^T x_a  (N x D)
    std::vector<double> U(N * D);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *__restrict x = X1 + a * M;
        const double *__restrict Ja = dX1 + (a * M) * D;  // (M x D)
        double *__restrict Ua = U.data() + a * D;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D, 1.0, Ja, (int)D, x, 1, 0.0, Ua, 1);
    }

    // 6) Per-block (lower triangle only): scale Gram by C[a,b], then rank-1 correction via GER
    if (tile_B == 0)
        tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N);  // sane default tile width

#pragma omp parallel
    {
        std::vector<double> v1(D), v2(D);  // private buffers

#pragma omp for schedule(static)
        for (std::size_t a = 0; a < N; ++a) {
            const double *__restrict Ua = U.data() + a * D;
            const double *__restrict Va = V + (a * D) * N;  // (D x N), row-major

            for (std::size_t b0 = 0; b0 < a + 1; b0 += tile_B) {
                const std::size_t bend = std::min<std::size_t>(a + 1, b0 + tile_B);

                for (std::size_t b = b0; b < bend; ++b) {
                    const double cab = C[a * N + b];
                    const double c4ab = C4[a * N + b];

                    // H(a,b) submatrix top-left
                    double *__restrict Hblk = H_out + (a * D) * BIG + b * D;

                    // Scale Gram block by cab (row by row)
                    for (std::size_t i = 0; i < D; ++i) {
                        cblas_dscal((int)D, cab, Hblk + i * BIG, 1);
                    }

// v1 = U[a] - V[aD:(a+1)D, b]
#pragma omp simd
                    for (std::size_t i = 0; i < D; ++i) {
                        v1[i] = Ua[i] - Va[i * N + b];
                    }

                    // v2 = V[bD:(b+1)D, a] - U[b]
                    const double *__restrict Ub = U.data() + b * D;
                    const double *__restrict Vb = V + (b * D) * N;
#pragma omp simd
                    for (std::size_t j = 0; j < D; ++j) {
                        v2[j] = Vb[j * N + a] - Ub[j];
                    }

                    // Rank-1 correction: H(a,b) -= c4ab * v1 v2^T
                    cblas_dger(CblasRowMajor, (int)D, (int)D, -c4ab, v1.data(), 1, v2.data(), 1,
                               Hblk, (int)BIG);
                }
            }
        }
    }

    // // 7) Mirror lower → upper
    // for (std::size_t i = 0; i < BIG; ++i) {
    //     const double* __restrict src = H_out + i*BIG;
    //     for (std::size_t j = 0; j < i; ++j) {
    //         H_out[j*BIG + i] = src[j];
    //     }
    // }

    // Free aligned allocations
    aligned_free_64(V);
    aligned_free_64(J_hat);
    aligned_free_64(C4);
    aligned_free_64(C);
    aligned_free_64(S);
}

// Helper: RFP index for TRANSR='N', UPLO='U' (upper triangle, i <= j)
static inline std::size_t rfp_index_upper_N(blas_int n, blas_int i, blas_int j) {
    // Precondition: 0 <= i <= j < n
    const blas_int k = n / 2;
    const blas_int stride = (n % 2 == 0) ? (n + 1) : n;
    if (j >= k) {
        return static_cast<std::size_t>(j - k) * static_cast<std::size_t>(stride) +
               static_cast<std::size_t>(i);
    } else {
        return static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) +
               static_cast<std::size_t>(j + k + 1);
    }
}

void kernel_gaussian_symm_rfp(const double *Xptr, blas_int n, blas_int rep_size, double alpha,
                              double *arf) {
    // Compute symmetric Gaussian kernel directly into RFP format
    // Output: arf[n*(n+1)/2] in Rectangular Full Packed (RFP) format
    // 
    // Internal storage: Fortran TRANSR='N', UPLO='U' (upper triangle packed)
    // Python API usage: rfp_to_full(arf, n, uplo='L') — note uplo='L' due to swap_uplo trick
    // 
    // This avoids allocating the full n×n matrix (saves 2× memory for large n)

    if (n <= 0 || rep_size <= 0)
        throw std::invalid_argument("n and rep_size must be > 0");
    if (!Xptr || !arf)
        throw std::invalid_argument("Xptr and arf must be non-null");

    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t rep_size_size = static_cast<std::size_t>(rep_size);
    const std::size_t nt = n_size * (n_size + 1) / 2;

    // Zero the RFP output
    std::memset(arf, 0, nt * sizeof(double));

    // 1) Precompute squared norms: sq[i] = alpha * ||X[i]||^2
    std::vector<double> sq(n_size);
    for (blas_int i = 0; i < n; ++i) {
        const double *row = Xptr + static_cast<std::size_t>(i) * rep_size_size;
        double acc = 0.0;
        for (blas_int k = 0; k < rep_size; ++k)
            acc += row[k] * row[k];
        sq[i] = alpha * acc;
    }

    // 2) Tiled DGEMM to compute kernel entries directly into RFP
    // Tile size — tuned for L2/L3 cache
    const blas_int TILE = 512;

    // Allocate scratch buffer for DGEMM tile results
    double *G_tile = aligned_alloc_64(static_cast<std::size_t>(TILE) * TILE);
    if (!G_tile)
        throw std::bad_alloc();

    // Process lower triangle tiles (i0 <= j0)
    for (blas_int j0 = 0; j0 < n; j0 += TILE) {
        const blas_int jb = std::min(TILE, n - j0);
        const double *Xj = Xptr + static_cast<std::size_t>(j0) * rep_size_size;

        for (blas_int i0 = 0; i0 <= j0; i0 += TILE) {
            const blas_int ib = std::min(TILE, n - i0);
            const double *Xi = Xptr + static_cast<std::size_t>(i0) * rep_size_size;

            if (i0 == j0) {
                // Diagonal tile: use DSYRK (upper triangle)
                cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, ib, rep_size, -2.0 * alpha, Xi,
                            rep_size, 0.0, G_tile, ib);

                // Write upper triangle of this tile to RFP
                for (blas_int j_loc = 0; j_loc < jb; ++j_loc) {
                    const blas_int j = j0 + j_loc;
                    for (blas_int i_loc = 0; i_loc <= j_loc; ++i_loc) {
                        const blas_int i = i0 + i_loc;
                        const double g_ij = G_tile[static_cast<std::size_t>(i_loc) * ib + j_loc];
                        const double k_ij = std::exp(sq[i] + sq[j] + g_ij);
                        const std::size_t rfp_idx = rfp_index_upper_N(n, i, j);
                        arf[rfp_idx] = k_ij;
                    }
                }
            } else {
                // Off-diagonal tile: use DGEMM (full rectangle)
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ib, jb, rep_size, -2.0 * alpha,
                            Xi, rep_size, Xj, rep_size, 0.0, G_tile, jb);

                // Write all entries (i < j) from this tile to RFP
                for (blas_int j_loc = 0; j_loc < jb; ++j_loc) {
                    const blas_int j = j0 + j_loc;
                    for (blas_int i_loc = 0; i_loc < ib; ++i_loc) {
                        const blas_int i = i0 + i_loc;
                        const double g_ij = G_tile[static_cast<std::size_t>(i_loc) * jb + j_loc];
                        const double k_ij = std::exp(sq[i] + sq[j] + g_ij);
                        const std::size_t rfp_idx = rfp_index_upper_N(n, i, j);
                        arf[rfp_idx] = k_ij;
                    }
                }
            }
        }
    }

    aligned_free_64(G_tile);
}

}  // namespace kf
