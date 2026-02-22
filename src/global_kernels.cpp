// C++ standard library
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// Third-party libraries
#include <omp.h>

// BLAS threading control (MKL or OpenBLAS)
#if !defined(__APPLE__)
    #if defined(KF_USE_MKL)
        // Intel MKL threading control
        extern "C" {
            void mkl_set_num_threads(int num_threads);
            int mkl_get_max_threads(void);
        }
        #define blas_set_num_threads mkl_set_num_threads
        #define blas_get_num_threads mkl_get_max_threads
    #else
        // OpenBLAS threading control
        extern "C" {
            void blas_set_num_threads(int num_threads);
            int blas_get_num_threads(void);
        }
        #define blas_set_num_threads blas_set_num_threads
        #define blas_get_num_threads blas_get_num_threads
    #endif
#endif

// Project headers
#include "aligned_alloc64.hpp"
#include "blas_config.h"
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

// Helper: RFP index for TRANSR='N', UPLO='U' (upper triangle, i <= j)
// Overload for blas_int (used by kernel_gaussian_symm_rfp)
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

    // Profiling timers
    double t_start = omp_get_wtime();

    const double inv_s2 = 1.0 / (sigma * sigma);

    // Disable BLAS threading for small DGEMMs in parallel region
#if !defined(__APPLE__)
    int saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

// Parallelize over N1 queries
#pragma omp parallel
    {
        // Thread-private scratch buffers
        double *W_local = aligned_alloc_64(static_cast<size_t>(M) * N2);
        std::vector<double> diff(M);

#pragma omp for schedule(dynamic, 4)
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
                    W_local[i * N2 + b] = coeff * diff[i];
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
                        W_local, static_cast<int>(N2), 0.0, Kblock, static_cast<int>(N2));
        }
        
        // Free thread-local buffer
        aligned_free_64(W_local);
    }

    // Restore BLAS threading
#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads);
#endif

    // Profiling output
    const char* profile_env = std::getenv("KERNELFORGE_PROFILE");
    if (profile_env && std::atoi(profile_env) != 0) {
        double t_total = omp_get_wtime() - t_start;
        printf("\n=== kernel_gaussian_jacobian profiling ===\n");
        printf("Problem size: N1=%zu, N2=%zu, M=%zu, D=%zu\n", N1, N2, M, D);
        printf("Output size: %zu x %zu\n", N1*D, N2);
        printf("TOTAL (parallelized over N1):     %8.4f ms\n", t_total*1000);
        printf("==========================================\n\n");
    }
}

// Transposed Jacobian kernel: Jacobians on reference side (X2, dX2) instead of query side
// Output: K_t(N1, N2*D) where K_t[a, b*D+d] = sum_i (k_ab/σ²) * (x1[a,i] - x2[b,i]) * J2[b,i,d]
// Relationship: kernel_gaussian_jacobian_t(X2, X1, dX1, σ) == kernel_gaussian_jacobian(X1, dX1, X2, σ).T
void kernel_gaussian_jacobian_t(const double *X1, const double *X2, const double *dX2, 
                                std::size_t N1, std::size_t N2, std::size_t M, std::size_t D,
                                double sigma, double *K_out) {
    if (!X1 || !X2 || !dX2 || !K_out)
        throw std::invalid_argument("null pointer");
    if (N1 == 0 || N2 == 0 || M == 0 || D == 0)
        throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0");

    // Profiling timers
    double t_start = omp_get_wtime();

    const double inv_s2 = 1.0 / (sigma * sigma);

    // Disable BLAS threading for small DGEMMs in parallel region
#if !defined(__APPLE__)
    int saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

    // Parallelize over N2 (reference structures with Jacobians)
#pragma omp parallel
    {
        // Thread-private scratch buffers
        double *W_local = aligned_alloc_64(static_cast<size_t>(M) * N1);  // (M × N1)
        std::vector<double> diff(M);

#pragma omp for schedule(dynamic, 4)
        for (std::size_t b = 0; b < N2; ++b) {
            const double *X2b = X2 + b * M;
            const double *J2b = dX2 + (b * M) * D;  // (M, D) block base
            
            // Build W(:, a) for all a
            // W[i, a] = (k_ab / σ²) * (x1[a, i] - x2[b, i])
            for (std::size_t a = 0; a < N1; ++a) {
                const double *X1a = X1 + a * M;

                double sq = 0.0;
                for (std::size_t i = 0; i < M; ++i) {
                    const double di = X1a[i] - X2b[i];  // Query - Reference
                    diff[i] = di;
                    sq += di * di;
                }

                const double k = std::exp(-0.5 * inv_s2 * sq);
                const double coeff = k * inv_s2;

                // W(i, a) with row-major (M × N1): index = i*N1 + a
                for (std::size_t i = 0; i < M; ++i) {
                    W_local[i * N1 + a] = coeff * diff[i];
                }
            }

            // Output block for this 'b': columns [b*D : (b+1)*D), all rows a=0..N1-1
            // K_out is (N1, N2*D) row-major, so K_out[a, b*D+d] = K_out[a*(N2*D) + b*D + d]
            // We want to write column-blocks, which means writing to K_out + b*D with stride N2*D
            
            // Kblock(N1 × D) = W^T(N1 × M) @ J2b(M × D)
            // W_local is (M × N1) row-major, so W^T is accessed via transpose
            // Output: starting at column b*D of K_out
            
            // We need to write to K_out in a way that respects row-major layout
            // K_out[a, b*D+d] is at offset a*(N2*D) + b*D + d
            // For DGEMM, we'll compute into a temporary block then copy, OR
            // use DGEMM with appropriate strides
            
            // Actually, let's compute into a temp block and copy
            std::vector<double> Kblock(N1 * D);  // (N1 × D)
            
            // DGEMM: Kblock(N1 × D) = W_local^T(N1 × M) @ J2b(M × D)
            // W_local is (M × N1) row-major, J2b is (M × D) row-major
            // CblasTrans on W_local gives us (N1 × M)
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                        static_cast<int>(N1),
                        static_cast<int>(D), 
                        static_cast<int>(M), 
                        1.0, 
                        W_local, static_cast<int>(N1),  // W is (M × N1), transposed to (N1 × M)
                        J2b, static_cast<int>(D),        // J2b is (M × D)
                        0.0, 
                        Kblock.data(), static_cast<int>(D));  // Kblock is (N1 × D)
            
            // Copy Kblock to the appropriate columns of K_out
            // K_out[a, b*D : (b+1)*D] = Kblock[a, :]
            for (std::size_t a = 0; a < N1; ++a) {
                double *K_row = K_out + a * (N2 * D) + b * D;
                const double *Kblock_row = Kblock.data() + a * D;
                std::memcpy(K_row, Kblock_row, D * sizeof(double));
            }
        }
        
        // Free thread-local buffer
        aligned_free_64(W_local);
    }

    // Restore BLAS threading
#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads);
#endif

    // Profiling output
    const char* profile_env = std::getenv("KERNELFORGE_PROFILE");
    if (profile_env && std::atoi(profile_env) != 0) {
        double t_total = omp_get_wtime() - t_start;
        printf("\n=== kernel_gaussian_jacobian_t profiling ===\n");
        printf("Problem size: N1=%zu, N2=%zu, M=%zu, D=%zu\n", N1, N2, M, D);
        printf("Output size: %zu x %zu\n", N1, N2*D);
        printf("TOTAL (parallelized over N2):     %8.4f ms\n", t_total*1000);
        printf("==========================================\n\n");
    }
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

    // Profiling timers
    double t_start = omp_get_wtime();
    double t_phase1, t_phase2, t_phase3, t_phase4, t_phase5, t_phase6;

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
    t_phase1 = omp_get_wtime() - t_start;

    // 2) Pack J1_hat (N1*D1 x M),  J2_all_cat (M x N2*D2)
    double t2_start = omp_get_wtime();
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
    t_phase2 = omp_get_wtime() - t2_start;

    // 3) Base Gram for ALL blocks once: H_out = J1_hat @ J2_all_cat  (threaded BLAS)
    double t3_start = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)big_rows, (int)big_cols, (int)M,
                1.0, J1_hat, lda_J1, J2_all_cat, ldb_J2cat, 0.0, H_out, ldc_H);
    t_phase3 = omp_get_wtime() - t3_start;

    // 4) Projection tables: V1X2_all and V2X1_all  (threaded BLAS)
    double t4_start = omp_get_wtime();
    double *V1X2_all = aligned_alloc_64(big_rows * N2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)big_rows, (int)N2, (int)M, 1.0,
                J1_hat, (int)M, X2, (int)M, 0.0, V1X2_all, (int)N2);

    double *V2X1_all = aligned_alloc_64(big_cols * N1);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, (int)big_cols, (int)N1, (int)M, 1.0,
                J2_all_cat, (int)big_cols, X1, (int)M, 0.0, V2X1_all, (int)N1);
    t_phase4 = omp_get_wtime() - t4_start;

    // 5) Self projections: U1[a]=J1_a^T x1_a, U2[b]=J2_b^T x2_b  (parallelize outer loops)
    double t5_start = omp_get_wtime();
    std::vector<double> U1(N1 * D1), U2(N2 * D2);

    // Disable BLAS threading for small DGEMV calls
#if !defined(__APPLE__)
    int saved_blas_threads_phase5 = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

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

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads_phase5);
#endif

    t_phase5 = omp_get_wtime() - t5_start;

    // 6) Per-block: scale Gram by C[a,b], then rank-1 correction via GER
    double t6_start = omp_get_wtime();
    if (tile_B == 0)
        tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N2);  // reasonable default

    // Disable BLAS threading to avoid oversubscription with small operations
#if !defined(__APPLE__)
    int saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
    const char* debug_env = std::getenv("KERNELFORGE_DEBUG");
    if (debug_env && std::atoi(debug_env) != 0) {
        printf("[DEBUG] Phase 6: Disabled BLAS threading (was %d threads, now 1)\n", saved_blas_threads);
    }
#endif

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
    t_phase6 = omp_get_wtime() - t6_start;

    // Restore BLAS threading
#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads);
#endif

    // Profiling output (can be controlled via environment variable)
    const char* profile_env = std::getenv("KERNELFORGE_PROFILE");
    if (profile_env && std::atoi(profile_env) != 0) {
        double t_total = omp_get_wtime() - t_start;
        #pragma omp critical
        {
            printf("\n=== kernel_gaussian_hessian profiling ===\n");
            printf("Problem size: N1=%zu, N2=%zu, M=%zu, D1=%zu, D2=%zu\n", N1, N2, M, D1, D2);
            printf("Output size: %zu x %zu\n", big_rows, big_cols);
            printf("Phase 1 (distance coefficients):  %8.4f ms  (%5.1f%%)\n", t_phase1*1000, 100*t_phase1/t_total);
            printf("Phase 2 (pack Jacobians):          %8.4f ms  (%5.1f%%)\n", t_phase2*1000, 100*t_phase2/t_total);
            printf("Phase 3 (base Gram DGEMM):         %8.4f ms  (%5.1f%%)\n", t_phase3*1000, 100*t_phase3/t_total);
            printf("Phase 4 (projection tables):       %8.4f ms  (%5.1f%%)\n", t_phase4*1000, 100*t_phase4/t_total);
            printf("Phase 5 (self projections):        %8.4f ms  (%5.1f%%)\n", t_phase5*1000, 100*t_phase5/t_total);
            printf("Phase 6 (per-block corrections):   %8.4f ms  (%5.1f%%)\n", t_phase6*1000, 100*t_phase6/t_total);
            printf("TOTAL:                             %8.4f ms\n", t_total*1000);
            printf("=========================================\n\n");
        }
    }

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

    // Profiling timers
    double t_start = omp_get_wtime();
    double t_phase1, t_phase2, t_phase3, t_phase4, t_phase5, t_phase6;

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
    t_phase1 = omp_get_wtime() - t_start;

    // 2) Pack J_hat (N*D x M) from dX1 (N,M,D): row i=(a*D+dj) is column dj of (M x D)
    double t2_start = omp_get_wtime();
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
    t_phase2 = omp_get_wtime() - t2_start;

    // 3) Base Gram only lower: H_out := J_hat @ J_hat^T  via DSYRK (lower)
    double t3_start = omp_get_wtime();
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, (int)BIG, (int)M, 1.0, J_hat,
                (int)M, 0.0, H_out, (int)BIG);
    t_phase3 = omp_get_wtime() - t3_start;

    // 4) Projection table V = J_hat @ X1^T  (BIG x N)
    double t4_start = omp_get_wtime();
    double *V = aligned_alloc_64(BIG * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)BIG, (int)N, (int)M, 1.0,
                J_hat, (int)M, X1, (int)M, 0.0, V, (int)N);
    t_phase4 = omp_get_wtime() - t4_start;

    // 5) Self projections: U[a] = J_a^T x_a  (N x D)
    double t5_start = omp_get_wtime();
    std::vector<double> U(N * D);

    // Disable BLAS threading for small DGEMV calls
#if !defined(__APPLE__)
    int saved_blas_threads_phase5_symm = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *__restrict x = X1 + a * M;
        const double *__restrict Ja = dX1 + (a * M) * D;  // (M x D)
        double *__restrict Ua = U.data() + a * D;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D, 1.0, Ja, (int)D, x, 1, 0.0, Ua, 1);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads_phase5_symm);
#endif

    t_phase5 = omp_get_wtime() - t5_start;

    // 6) Per-block (lower triangle only): scale Gram by C[a,b], then rank-1 correction via GER
    double t6_start = omp_get_wtime();
    if (tile_B == 0)
        tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N);  // sane default tile width

    // Disable BLAS threading to avoid oversubscription with small operations
#if !defined(__APPLE__)
    int saved_blas_threads_symm = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

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
    t_phase6 = omp_get_wtime() - t6_start;

    // Restore BLAS threading
#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads_symm);
#endif

    // // 7) Mirror lower → upper
    // for (std::size_t i = 0; i < BIG; ++i) {
    //     const double* __restrict src = H_out + i*BIG;
    //     for (std::size_t j = 0; j < i; ++j) {
    //         H_out[j*BIG + i] = src[j];
    //     }
    // }

    // Profiling output (can be controlled via environment variable)
    const char* profile_env = std::getenv("KERNELFORGE_PROFILE");
    if (profile_env && std::atoi(profile_env) != 0) {
        double t_total = omp_get_wtime() - t_start;
        #pragma omp critical
        {
            printf("\n=== kernel_gaussian_hessian_symm profiling ===\n");
            printf("Problem size: N=%zu, M=%zu, D=%zu\n", N, M, D);
            printf("Output size: %zu x %zu (lower triangle only)\n", BIG, BIG);
            printf("Phase 1 (distance coefficients):  %8.4f ms  (%5.1f%%)\n", t_phase1*1000, 100*t_phase1/t_total);
            printf("Phase 2 (pack Jacobians):          %8.4f ms  (%5.1f%%)\n", t_phase2*1000, 100*t_phase2/t_total);
            printf("Phase 3 (base Gram DSYRK):         %8.4f ms  (%5.1f%%)\n", t_phase3*1000, 100*t_phase3/t_total);
            printf("Phase 4 (projection table):        %8.4f ms  (%5.1f%%)\n", t_phase4*1000, 100*t_phase4/t_total);
            printf("Phase 5 (self projections):        %8.4f ms  (%5.1f%%)\n", t_phase5*1000, 100*t_phase5/t_total);
            printf("Phase 6 (per-block corrections):   %8.4f ms  (%5.1f%%)\n", t_phase6*1000, 100*t_phase6/t_total);
            printf("TOTAL:                             %8.4f ms\n", t_total*1000);
            printf("===============================================\n\n");
        }
    }

    // Free aligned allocations
    aligned_free_64(V);
    aligned_free_64(J_hat);
    aligned_free_64(C4);
    aligned_free_64(C);
    aligned_free_64(S);
}

// ============================================================================
// RFP (Row-First Packed) version - saves ~50% memory for symmetric matrices
// ============================================================================

// RFP indexing for upper triangle (TRANSR='N', UPLO='U')
// Maps (i,j) with i<=j to packed index in range [0, N*(N+1)/2)
static inline std::size_t rfp_index_upper_N(std::size_t n, std::size_t i, std::size_t j) {
    // Precondition: i <= j < n
    const std::size_t k = n / 2;
    const std::size_t stride = (n % 2 == 0) ? (n + 1) : n;
    
    if (j >= k) {
        // Top zone
        return (j - k) * stride + i;
    } else {
        // Bottom zone
        return i * stride + j + k + 1;
    }
}

// Symmetric Hessian kernel with RFP output (memory-efficient)
// Output H_rfp: packed lower triangle, length BIG*(BIG+1)/2 where BIG=N*D
// Memory savings: ~50% compared to full matrix
void kernel_gaussian_hessian_symm_rfp(
    const double *__restrict X1, const double *__restrict dX1, std::size_t N1, std::size_t M,
    std::size_t D1, double sigma, std::size_t tile_B, double *__restrict H_rfp) {
    if (!X1 || !dX1 || !H_rfp)
        throw std::invalid_argument("null pointer");
    if (N1 == 0 || M == 0 || D1 == 0)
        throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0");

    const std::size_t N = N1;
    const std::size_t D = D1;
    const std::size_t BIG = N * D;
    const std::size_t rfp_size = BIG * (BIG + 1) / 2;

    const double inv_s2 = 1.0 / (sigma * sigma);
    const double inv_s4 = inv_s2 * inv_s2;

    // Profiling timers
    double t_start = omp_get_wtime();
    double t_phase1, t_phase2, t_phase3, t_phase4, t_phase5, t_phase6;

    // Zero output
    std::memset(H_rfp, 0, rfp_size * sizeof(double));

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
    t_phase1 = omp_get_wtime() - t_start;

    // 2) Pack J_hat (N*D x M) from dX1 (N,M,D): row i=(a*D+dj) is column dj of (M x D)
    double t2_start = omp_get_wtime();
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
    t_phase2 = omp_get_wtime() - t2_start;

    // 3) Base Gram lower triangle via DSYRK, but we compute to temp full matrix first
    // (DSYRK only writes lower, we need both for the loop below)
    double t3_start = omp_get_wtime();
    double *H_temp = aligned_alloc_64(BIG * BIG);
    std::memset(H_temp, 0, BIG * BIG * sizeof(double));
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, (int)BIG, (int)M, 1.0, J_hat,
                (int)M, 0.0, H_temp, (int)BIG);
    t_phase3 = omp_get_wtime() - t3_start;

    // 4) Projection table V = J_hat @ X1^T  (BIG x N)
    double t4_start = omp_get_wtime();
    double *V = aligned_alloc_64(BIG * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)BIG, (int)N, (int)M, 1.0,
                J_hat, (int)M, X1, (int)M, 0.0, V, (int)N);
    t_phase4 = omp_get_wtime() - t4_start;

    // 5) Self projections: U[a] = J_a^T x_a  (N x D)
    double t5_start = omp_get_wtime();
    std::vector<double> U(N * D);

    // Disable BLAS threading for small DGEMV calls
#if !defined(__APPLE__)
    int saved_blas_threads_phase5_symm = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *__restrict x = X1 + a * M;
        const double *__restrict Ja = dX1 + (a * M) * D;  // (M x D)
        double *__restrict Ua = U.data() + a * D;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D, 1.0, Ja, (int)D, x, 1, 0.0, Ua, 1);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads_phase5_symm);
#endif

    t_phase5 = omp_get_wtime() - t5_start;

    // 6) Per-block (lower triangle only): scale Gram by C[a,b], then rank-1 correction via GER
    // Then write to RFP format
    double t6_start = omp_get_wtime();
    if (tile_B == 0)
        tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N);  // sane default tile width

    // Disable BLAS threading to avoid oversubscription with small operations
#if !defined(__APPLE__)
    int saved_blas_threads_symm = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

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

                    // H(a,b) submatrix from temp matrix
                    const double *__restrict Hblk_src = H_temp + (a * D) * BIG + b * D;

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

                    // Apply scaling and rank-1 correction, write to RFP
                    for (std::size_t i = 0; i < D; ++i) {
                        for (std::size_t j = 0; j < D; ++j) {
                            const std::size_t row = a * D + i;
                            const std::size_t col = b * D + j;
                            
                            // Read from temp (only lower triangle is valid from DSYRK)
                            double val;
                            if (row >= col) {
                                val = Hblk_src[i * BIG + j];
                            } else {
                                // Transpose access
                                val = H_temp[col * BIG + row];
                            }
                            
                            // Apply transformations
                            val = val * cab - c4ab * v1[i] * v2[j];
                            
                            // Write to RFP (store lower triangle only)
                            if (row >= col) {
                                const std::size_t idx = rfp_index_upper_N(BIG, col, row);
                                H_rfp[idx] = val;
                            }
                        }
                    }
                }
            }
        }
    }
    t_phase6 = omp_get_wtime() - t6_start;

    // Restore BLAS threading
#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads_symm);
#endif

    // Profiling output
    const char* profile_env = std::getenv("KERNELFORGE_PROFILE");
    if (profile_env && std::atoi(profile_env) != 0) {
        double t_total = omp_get_wtime() - t_start;
        #pragma omp critical
        {
            printf("\n=== kernel_gaussian_hessian_symm_rfp profiling ===\n");
            printf("Problem size: N=%zu, M=%zu, D=%zu\n", N, M, D);
            printf("Output size: %zu (RFP packed, saves %.1f%% memory)\n", 
                   rfp_size, 100.0 * (1.0 - 0.5*(BIG+1)/BIG));
            printf("Phase 1 (distance coefficients):  %8.4f ms  (%5.1f%%)\n", t_phase1*1000, 100*t_phase1/t_total);
            printf("Phase 2 (pack Jacobians):          %8.4f ms  (%5.1f%%)\n", t_phase2*1000, 100*t_phase2/t_total);
            printf("Phase 3 (base Gram DSYRK):         %8.4f ms  (%5.1f%%)\n", t_phase3*1000, 100*t_phase3/t_total);
            printf("Phase 4 (projection table):        %8.4f ms  (%5.1f%%)\n", t_phase4*1000, 100*t_phase4/t_total);
            printf("Phase 5 (self projections):        %8.4f ms  (%5.1f%%)\n", t_phase5*1000, 100*t_phase5/t_total);
            printf("Phase 6 (per-block corrections):   %8.4f ms  (%5.1f%%)\n", t_phase6*1000, 100*t_phase6/t_total);
            printf("TOTAL:                             %8.4f ms\n", t_total*1000);
            printf("====================================================\n\n");
        }
    }

    // Free allocations
    aligned_free_64(H_temp);
    aligned_free_64(V);
    aligned_free_64(J_hat);
    aligned_free_64(C4);
    aligned_free_64(C);
    aligned_free_64(S);
}

}  // namespace kf
