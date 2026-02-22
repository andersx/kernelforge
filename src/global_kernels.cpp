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
            void openblas_set_num_threads(int num_threads);
            int openblas_get_num_threads(void);
        }
        #define blas_set_num_threads openblas_set_num_threads
        #define blas_get_num_threads openblas_get_num_threads
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

// 4) Elementwise exp on the lower triangle (row j >= col i)
#pragma omp parallel for schedule(guided)
    for (blas_int j = 0; j < n; ++j) {
        for (blas_int i = 0; i <= j; ++i) {
            Kptr[j * n + i] = std::exp(Kptr[j * n + i]);
        }
    }

    // 5) Mirror lower triangle to upper triangle
#pragma omp parallel for schedule(static)
    for (blas_int j = 0; j < n; ++j) {
        for (blas_int i = 0; i < j; ++i) {
            Kptr[i * n + j] = Kptr[j * n + i];
        }
    }
}

// RFP index map: (i, j) 0-based, i <= j -> linear RFP position (0-based)
// Valid for Fortran TRANSR='N', UPLO='U', any n.
// Used to scatter kernel values directly into RFP layout.
static inline std::size_t rfp_index_upper_N(blas_int n, blas_int i, blas_int j) {
    const blas_int k      = n / 2;
    const blas_int stride = (n % 2 == 0) ? (n + 1) : n;
    if (j >= k) {
        return static_cast<std::size_t>(j - k) * static_cast<std::size_t>(stride) +
               static_cast<std::size_t>(i);
    } else {
        return static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) +
               static_cast<std::size_t>(j + k + 1);
    }
}

// =============================================================================
// LEGACY: DSFRK-based implementation of kernel_gaussian_symm_rfp.
//
// This version uses LAPACKE_dsfrk to write the Gram matrix directly into RFP
// without any temporary buffer (pure O(N²/2) memory). It is a faithful C++
// port of the Fortran reference in examples/dsfrk_kernel.f90.
//
// WHY IT IS NOT THE PRIMARY IMPLEMENTATION:
//   LAPACKE_dsfrk for double precision is ~2.5× slower than cblas_dsyrk on MKL
//   (N=10000: ~0.35 s vs ~0.14 s). MKL's DSFRK is not as well optimised as
//   DSYRK for double; the Fortran reference uses single-precision ssfrk where
//   the gap is smaller. The tiled DSYRK approach below achieves 1.28× overhead
//   while still avoiding the full N×N allocation, so it dominates on all counts.
//
// The code is kept here for reference and in case a future MKL release improves
// DSFRK performance, or for use on platforms where DSYRK tile memory is tight.
// =============================================================================
#if 0

// Inverse RFP map: linear RFP index (0-based) -> (i, j) 0-based, i <= j
// Valid for TRANSR='N', UPLO='U', any n. Ported from Fortran rfp_ij_n_u.
static inline void rfp_ij_n_u(blas_int n, std::size_t idx, blas_int *i_out, blas_int *j_out) {
    const blas_int k      = n / 2;
    const bool     even   = (n % 2 == 0);
    const blas_int stride = even ? (n + 1) : n;
    const blas_int p  = static_cast<blas_int>(idx);
    const blas_int c0 = p / stride;
    const blas_int r0 = p - c0 * stride;
    const blas_int iA = r0, jA = k + c0;
    const bool     useA = (iA <= jA);
    const blas_int q  = p - (k + 1);
    const blas_int iB = (q >= 0) ? (q / stride) : 0;
    const blas_int jB = (q >= 0) ? (q - iB * stride) : 0;
    *i_out = useA ? iA : iB;
    *j_out = useA ? jA : jB;
}

void kernel_gaussian_symm_rfp_dsfrk(const double *Xptr, blas_int n, blas_int rep_size,
                                     double alpha, double *arf) {
    if (n <= 0 || rep_size <= 0) throw std::invalid_argument("n and rep_size must be > 0");
    if (!Xptr || !arf)           throw std::invalid_argument("Xptr and arf must be non-null");

    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t nt     = n_size * (n_size + 1) / 2;

    std::memset(arf, 0, nt * sizeof(double));  // beta=0 may skip writes on uninit buffer

    LAPACKE_dsfrk(LAPACK_COL_MAJOR, 'N', 'U', 'T',
                  n, rep_size, -2.0 * alpha, Xptr, rep_size, 0.0, arf);

    std::vector<double> sq(n_size);
#pragma omp parallel for schedule(static)
    for (blas_int i = 0; i < n; ++i) {
        const double *row = Xptr + static_cast<std::size_t>(i) * static_cast<std::size_t>(rep_size);
        double acc = 0.0;
        for (blas_int kk = 0; kk < rep_size; ++kk) acc += row[kk] * row[kk];
        sq[i] = alpha * acc;
    }

    const blas_int k_rfp      = n / 2;
    const blas_int stride_rfp = (n % 2 == 0) ? (n + 1) : n;
#pragma omp parallel for schedule(guided)
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(nt); ++idx) {
        blas_int ii, jj;
        rfp_ij_n_u(n, static_cast<std::size_t>(idx), &ii, &jj);
        arf[idx] = std::exp(arf[idx] + sq[ii] + sq[jj]);
    }
}

#endif  // LEGACY DSFRK implementation

// =============================================================================
// Tile size for kernel_gaussian_symm_rfp (number of rows/cols per tile).
// Temporary buffer is TILE_RFP × TILE_RFP doubles = 512 MB for 8192.
// For N <= TILE_RFP the whole matrix fits in one tile (single DSYRK + scatter).
static constexpr blas_int TILE_RFP = 8192;

void kernel_gaussian_symm_rfp(const double *Xptr, blas_int n, blas_int rep_size,
                               double alpha, double *arf) {
    // Compute symmetric Gaussian kernel directly into RFP format using tiled DGEMM/DSYRK.
    // Output: arf[n*(n+1)/2] in Rectangular Full Packed (RFP) format
    //   (Fortran TRANSR='N', UPLO='U').
    //
    // Memory: one TILE_RFP×TILE_RFP temporary buffer (~512 MB for T=8192).
    // For N >> TILE_RFP this is far less than N×N.
    // For N <= TILE_RFP the whole matrix fits in one tile (single DSYRK + scatter).

    if (n <= 0 || rep_size <= 0)
        throw std::invalid_argument("n and rep_size must be > 0");
    if (!Xptr || !arf)
        throw std::invalid_argument("Xptr and arf must be non-null");

    const std::size_t n_sz = static_cast<std::size_t>(n);
    const std::size_t d_sz = static_cast<std::size_t>(rep_size);

    // 1) Squared norms: sq[i] = alpha * ||X[i]||^2
    std::vector<double> sq(n_sz);
#pragma omp parallel for schedule(static)
    for (blas_int i = 0; i < n; ++i) {
        const double *row = Xptr + static_cast<std::size_t>(i) * d_sz;
        double acc = 0.0;
        for (blas_int kk = 0; kk < rep_size; ++kk)
            acc += row[kk] * row[kk];
        sq[i] = alpha * acc;
    }

    // 2) Allocate one reusable tile buffer (TILE_RFP × TILE_RFP or smaller if n < TILE_RFP)
    const blas_int T  = std::min<blas_int>(TILE_RFP, n);
    const std::size_t tile_buf_sz = static_cast<std::size_t>(T) * static_cast<std::size_t>(T);
    double *tile = aligned_alloc_64(tile_buf_sz);
    if (!tile) throw std::bad_alloc();

    // 3) Tile over the upper triangle.
    //    Outer loop: tile-columns tc (j range [j0, j1))
    //    Inner loop: tile-rows tr <= tc (i range [i0, i1))
    //    Diagonal tiles (tr==tc): DSYRK + scatter upper-triangle entries to RFP
    //    Off-diagonal tiles (tr<tc): DGEMM + scatter all entries to RFP
    for (blas_int j0 = 0; j0 < n; j0 += T) {
        const blas_int j1  = std::min<blas_int>(j0 + T, n);
        const blas_int Tj  = j1 - j0;  // width of this column tile
        const double *Xj   = Xptr + static_cast<std::size_t>(j0) * d_sz;

        for (blas_int i0 = 0; i0 <= j0; i0 += T) {
            const blas_int i1  = std::min<blas_int>(i0 + T, j1);  // i1 <= j1 (upper tri only)
            const blas_int Ti  = i1 - i0;
            const double *Xi   = Xptr + static_cast<std::size_t>(i0) * d_sz;

            if (i0 == j0) {
                // ---- Diagonal tile: only upper triangle needed ----
                // DSYRK (lower, row-major) into tile[Ti×Ti]
                // tile[p*Ti + q] = -2*alpha * X[i0+p] · X[i0+q]  for q <= p (lower)
                cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                            Ti, rep_size,
                            -2.0 * alpha, Xi, rep_size,
                            0.0, tile, Ti);

                // Scatter lower triangle of tile → RFP (with exp).
                // DSYRK row-major lower: tile[p*Ti + q] valid for q <= p.
                // In global indexing: row = i0+p, col = i0+q, with row >= col (lower tri),
                // which maps to rfp_index_upper_N(n, col, row) = rfp_index_upper_N(n, i0+q, i0+p).
#pragma omp parallel for schedule(guided)
                for (blas_int p = 0; p < Ti; ++p) {
                    for (blas_int q = 0; q <= p; ++q) {
                        const blas_int gi = i0 + q;  // global i (upper-tri: i <= j)
                        const blas_int gj = i0 + p;  // global j
                        const double val = std::exp(
                            sq[gi] + sq[gj] +
                            tile[static_cast<std::size_t>(p) * static_cast<std::size_t>(Ti) +
                                 static_cast<std::size_t>(q)]);
                        arf[rfp_index_upper_N(n, gi, gj)] = val;
                    }
                }
            } else {
                // ---- Off-diagonal tile: i range is strictly left of j range ----
                // tile[Ti × Tj] = (-2*alpha) * X[i0:i1] @ X[j0:j1]^T
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            Ti, Tj, rep_size,
                            -2.0 * alpha, Xi, rep_size,
                            Xj, rep_size,
                            0.0, tile, Tj);

                // Scatter all Ti×Tj entries → RFP (with exp).
                // tile[p*Tj + q] corresponds to global (i=i0+p, j=j0+q), i < j always.
#pragma omp parallel for schedule(guided)
                for (blas_int p = 0; p < Ti; ++p) {
                    for (blas_int q = 0; q < Tj; ++q) {
                        const blas_int gi = i0 + p;
                        const blas_int gj = j0 + q;
                        const double val = std::exp(
                            sq[gi] + sq[gj] +
                            tile[static_cast<std::size_t>(p) * static_cast<std::size_t>(Tj) +
                                 static_cast<std::size_t>(q)]);
                        arf[rfp_index_upper_N(n, gi, gj)] = val;
                    }
                }
            }
        }
    }

    aligned_free_64(tile);
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
        // Thread-private scratch buffers: allocate once per thread, reuse across iterations
        double *W_local = aligned_alloc_64(static_cast<size_t>(M) * N1);  // (M × N1)
        double *Kblock  = aligned_alloc_64(N1 * D);                        // (N1 × D)
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

            // DGEMM: Kblock(N1 × D) = W_local^T(N1 × M) @ J2b(M × D)
            // W_local is (M × N1) row-major, transposed to (N1 × M)
            // J2b is (M × D) row-major
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        static_cast<int>(N1),
                        static_cast<int>(D),
                        static_cast<int>(M),
                        1.0,
                        W_local, static_cast<int>(N1),
                        J2b, static_cast<int>(D),
                        0.0,
                        Kblock, static_cast<int>(D));
            
            // Copy Kblock to the appropriate columns of K_out
            // K_out is (N1, N2*D) row-major: K_out[a, b*D : (b+1)*D] = Kblock[a, :]
            for (std::size_t a = 0; a < N1; ++a) {
                double *K_row = K_out + a * (N2 * D) + b * D;
                const double *Kblock_row = Kblock + a * D;
                std::memcpy(K_row, Kblock_row, D * sizeof(double));
            }
        }
        
        // Free thread-local buffers
        aligned_free_64(W_local);
        aligned_free_64(Kblock);
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

    // 2) Pack J1_hat (N1*D1 × M) and J2_hat (N2*D2 × M): same row-major layout.
    //    Each row (a*D1+dj) of J1_hat is column dj of the (M×D1) Jacobian for molecule a.
    double t2_start = omp_get_wtime();
    double *J1_hat = aligned_alloc_64(big_rows * M);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        const double *__restrict J1 = dX1 + (a * M) * D1;
        for (std::size_t dj = 0; dj < D1; ++dj) {
            double *__restrict row = J1_hat + (a * D1 + dj) * M;
#pragma omp simd
            for (std::size_t i = 0; i < M; ++i)
                row[i] = J1[i * D1 + dj];
        }
    }

    double *J2_hat = aligned_alloc_64(big_cols * M);
#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        const double *__restrict J2 = dX2 + (b * M) * D2;
        for (std::size_t dj = 0; dj < D2; ++dj) {
            double *__restrict row = J2_hat + (b * D2 + dj) * M;
#pragma omp simd
            for (std::size_t i = 0; i < M; ++i)
                row[i] = J2[i * D2 + dj];
        }
    }
    t_phase2 = omp_get_wtime() - t2_start;

    // 3) No global Gram matrix. D1×D2 Gram blocks computed on-the-fly in phase 6.
    //    Avoids writing/reading the big_rows × big_cols = (N1*D1)×(N2*D2) H_out,
    //    which at N=1000, D=27 is 5.8 GB and bottlenecked by cold-page memory bandwidth.
    double t3_start = omp_get_wtime();
    t_phase3 = omp_get_wtime() - t3_start;  // nothing to do

    // 4) Projection tables: V1X2 = J1_hat @ X2^T (big_rows × N2),
    //                        V2X1 = J2_hat @ X1^T (big_cols × N1)
    double t4_start = omp_get_wtime();
    double *V1X2_all = aligned_alloc_64(big_rows * N2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)big_rows, (int)N2, (int)M, 1.0,
                J1_hat, (int)M, X2, (int)M, 0.0, V1X2_all, (int)N2);

    double *V2X1_all = aligned_alloc_64(big_cols * N1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)big_cols, (int)N1, (int)M, 1.0,
                J2_hat, (int)M, X1, (int)M, 0.0, V2X1_all, (int)N1);
    t_phase4 = omp_get_wtime() - t4_start;

    // 5) Self projections: U1[a]=J1_a^T x1_a, U2[b]=J2_b^T x2_b
    double t5_start = omp_get_wtime();
    std::vector<double> U1(N1 * D1), U2(N2 * D2);

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

    // 6) All blocks: compute D1×D2 Gram on-the-fly per (a,b), fuse scale + rank-1 + write H_out.
    //    Avoids ever writing the cold N1*D1 × N2*D2 global Gram matrix; each D1×D2 Gblk
    //    (D=27: 5832 bytes) stays hot in L1 cache.
    double t6_start = omp_get_wtime();
    if (tile_B == 0)
        tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N2);

#if !defined(__APPLE__)
    int saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel
    {
        double *Gblk = aligned_alloc_64(D1 * D2);  // thread-local D1×D2 Gram scratch
        std::vector<double> v1(D1), v2(D2);

#pragma omp for schedule(static)
        for (std::size_t a = 0; a < N1; ++a) {
            const double *__restrict U1a    = U1.data() + a * D1;
            const double *__restrict V1X2a  = V1X2_all + (a * D1) * N2;  // (D1×N2)
            const double *__restrict J1a_hat = J1_hat + a * D1 * M;       // (D1×M)

            for (std::size_t b0 = 0; b0 < N2; b0 += tile_B) {
                const std::size_t bend = std::min(N2, b0 + tile_B);

                for (std::size_t b = b0; b < bend; ++b) {
                    const double cab  = C[a * N2 + b];
                    const double c4ab = C4[a * N2 + b];

                    // Gram block: Gblk(D1×D2) = J1_hat[a*D1:, :] @ J2_hat[b*D2:, :]^T
                    const double *__restrict J2b_hat = J2_hat + b * D2 * M;
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                (int)D1, (int)D2, (int)M,
                                1.0, J1a_hat, (int)M, J2b_hat, (int)M,
                                0.0, Gblk, (int)D2);

                    // v1 = U1[a] - V1X2_all[aD1:(a+1)D1, b]
#pragma omp simd
                    for (std::size_t i = 0; i < D1; ++i)
                        v1[i] = U1a[i] - V1X2a[i * N2 + b];

                    // v2 = V2X1_all[bD2:(b+1)D2, a] - U2[b]
                    const double *__restrict U2b    = U2.data() + b * D2;
                    const double *__restrict V2X1ba = V2X1_all + (b * D2) * N1 + a;  // col a
#pragma omp simd
                    for (std::size_t j = 0; j < D2; ++j)
                        v2[j] = V2X1ba[j * N1] - U2b[j];

                    // Fused scale + rank-1 + write to H_out block
                    double *__restrict Hblk = H_out + (a * D1) * big_cols + b * D2;
                    for (std::size_t i = 0; i < D1; ++i) {
                        double *__restrict Hrow = Hblk + i * big_cols;
#pragma omp simd
                        for (std::size_t j = 0; j < D2; ++j)
                            Hrow[j] = Gblk[i * D2 + j] * cab - c4ab * v1[i] * v2[j];
                    }
                }
            }
        }
        aligned_free_64(Gblk);
    }
    t_phase6 = omp_get_wtime() - t6_start;

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
    aligned_free_64(J2_hat);
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

    // 3) No global Gram matrix. D×D Gram blocks computed on-the-fly in phase 6.
    //    Avoids allocating/writing/reading the BIG×BIG = (N*D)² H_out in phase 3/6,
    //    which at N=1000, D=27 is 5.8 GB and bottlenecked by cold-page memory bandwidth.
    double t3_start = omp_get_wtime();
    t_phase3 = omp_get_wtime() - t3_start;  // nothing to do

    // 4) Projection table V = J_hat @ X1^T  (BIG x N)
    double t4_start = omp_get_wtime();
    double *V = aligned_alloc_64(BIG * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)BIG, (int)N, (int)M, 1.0,
                J_hat, (int)M, X1, (int)M, 0.0, V, (int)N);
    t_phase4 = omp_get_wtime() - t4_start;

    // 5) Self projections: U[a] = J_a^T x_a  (N x D)
    double t5_start = omp_get_wtime();
    std::vector<double> U(N * D);

#if !defined(__APPLE__)
    int saved_blas_threads_phase5_symm = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *__restrict x = X1 + a * M;
        const double *__restrict Ja = dX1 + (a * M) * D;
        double *__restrict Ua = U.data() + a * D;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D, 1.0, Ja, (int)D, x, 1, 0.0, Ua, 1);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads_phase5_symm);
#endif

    t_phase5 = omp_get_wtime() - t5_start;

    // 6) Lower-triangle blocks: compute D×D Gram on-the-fly per (a,b), fuse scale + rank-1.
    //    Avoids ever writing the cold BIG×BIG global Gram matrix; each D×D Gblk
    //    (D=27: 5832 bytes) stays hot in L1 cache.
    double t6_start = omp_get_wtime();
    if (tile_B == 0)
        tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N);

#if !defined(__APPLE__)
    int saved_blas_threads_symm = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel
    {
        double *Gblk = aligned_alloc_64(D * D);  // thread-local D×D Gram scratch
        std::vector<double> v1(D), v2(D);

#pragma omp for schedule(static)
        for (std::size_t a = 0; a < N; ++a) {
            const double *__restrict Ua = U.data() + a * D;
            const double *__restrict Va = V + (a * D) * N;
            const double *__restrict Ja_hat = J_hat + a * D * M;  // (D×M)

            for (std::size_t b0 = 0; b0 < a + 1; b0 += tile_B) {
                const std::size_t bend = std::min<std::size_t>(a + 1, b0 + tile_B);

                for (std::size_t b = b0; b < bend; ++b) {
                    const double cab  = C[a * N + b];
                    const double c4ab = C4[a * N + b];

                    // Gram block: Gblk(D×D) = J_hat[a*D:(a+1)*D, :] @ J_hat[b*D:(b+1)*D, :]^T
                    const double *__restrict Jb_hat = J_hat + b * D * M;
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                (int)D, (int)D, (int)M,
                                1.0, Ja_hat, (int)M, Jb_hat, (int)M,
                                0.0, Gblk, (int)D);

                    // v1 = U[a] - V[aD:(a+1)D, b]
#pragma omp simd
                    for (std::size_t i = 0; i < D; ++i)
                        v1[i] = Ua[i] - Va[i * N + b];

                    // v2 = V[bD:(b+1)D, a] - U[b]
                    const double *__restrict Ub = U.data() + b * D;
                    const double *__restrict Vb = V + (b * D) * N;
#pragma omp simd
                    for (std::size_t j = 0; j < D; ++j)
                        v2[j] = Vb[j * N + a] - Ub[j];

                    // Fused scale + rank-1 correction, write lower-triangle block of H_out.
                    // H_out[aD+i, bD+j] = Gblk[i,j]*cab - c4ab*v1[i]*v2[j]
                    // Only the lower triangle is filled (consistent with original behaviour;
                    // the caller/test only reads H_out[i,j] for j<=i).
                    double *__restrict Hblk = H_out + (a * D) * BIG + b * D;
                    for (std::size_t i = 0; i < D; ++i) {
                        double *__restrict Hrow = Hblk + i * BIG;
#pragma omp simd
                        for (std::size_t j = 0; j < D; ++j)
                            Hrow[j] = Gblk[i * D + j] * cab - c4ab * v1[i] * v2[j];
                    }
                }
            }
        }
        aligned_free_64(Gblk);
    }
    t_phase6 = omp_get_wtime() - t6_start;

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads_symm);
#endif

    // 7) Mirror lower-triangle D×D blocks to upper triangle.
    //    For each (a, b) with a > b, copy the D×D block at [a*D:, b*D:] → [b*D:, a*D:].
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < a; ++b) {
            for (std::size_t d1 = 0; d1 < D; ++d1) {
                for (std::size_t d2 = 0; d2 < D; ++d2) {
                    H_out[(b * D + d2) * BIG + (a * D + d1)] =
                        H_out[(a * D + d1) * BIG + (b * D + d2)];
                }
            }
        }
        // Mirror diagonal block lower→upper within the D×D block at [a*D:, a*D:]
        for (std::size_t d1 = 0; d1 < D; ++d1) {
            for (std::size_t d2 = 0; d2 < d1; ++d2) {
                H_out[(a * D + d2) * BIG + (a * D + d1)] =
                    H_out[(a * D + d1) * BIG + (a * D + d2)];
            }
        }
    }

    // Profiling output
    const char* profile_env = std::getenv("KERNELFORGE_PROFILE");
    if (profile_env && std::atoi(profile_env) != 0) {
        double t_total = omp_get_wtime() - t_start;
        #pragma omp critical
        {
            printf("\n=== kernel_gaussian_hessian_symm profiling ===\n");
            printf("Problem size: N=%zu, M=%zu, D=%zu\n", N, M, D);
            printf("Output size: %zu x %zu (full symmetric)\n", BIG, BIG);
            printf("Phase 1 (distance coefficients):  %8.4f ms  (%5.1f%%)\n", t_phase1*1000, 100*t_phase1/t_total);
            printf("Phase 2 (pack Jacobians):          %8.4f ms  (%5.1f%%)\n", t_phase2*1000, 100*t_phase2/t_total);
            printf("Phase 3 (base Gram):               %8.4f ms  (%5.1f%%)  [on-the-fly per block in phase 6]\n", t_phase3*1000, 100*t_phase3/t_total);
            printf("Phase 4 (projection table):        %8.4f ms  (%5.1f%%)\n", t_phase4*1000, 100*t_phase4/t_total);
            printf("Phase 5 (self projections):        %8.4f ms  (%5.1f%%)\n", t_phase5*1000, 100*t_phase5/t_total);
            printf("Phase 6 (per-block Gram+correct):  %8.4f ms  (%5.1f%%)\n", t_phase6*1000, 100*t_phase6/t_total);
            printf("TOTAL:                             %8.4f ms\n", t_total*1000);
            printf("===============================================\n\n");
        }
    }

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

    // NOTE: H_rfp is NOT zeroed here. Every element is written exactly once in
    // phase 6, so a prior memset would only waste time on a cold 2.9 GB buffer.

    // 1) Distances → C = K/s^2 and C4 = K/s^4  (identical to hessian_symm)
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

    // 2) Pack J_hat (N*D x M): row (a*D+dj) = column dj of J_a  (identical to hessian_symm)
    double t2_start = omp_get_wtime();
    double *J_hat = aligned_alloc_64(BIG * M);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *__restrict J = dX1 + (a * M) * D;
        for (std::size_t dj = 0; dj < D; ++dj) {
            double *__restrict row = J_hat + (a * D + dj) * M;
#pragma omp simd
            for (std::size_t i = 0; i < M; ++i)
                row[i] = J[i * D + dj];
        }
    }
    t_phase2 = omp_get_wtime() - t2_start;

    // 3) No global Gram matrix allocated here.
    //    The D×D Gram block for each (a,b) pair is computed on-the-fly in phase 6
    //    via a small DGEMM (D×M) @ (M×D) → (D×D), using a thread-local scratch buffer.
    //    This avoids allocating the BIG×BIG = (N*D)² temp matrix (~5.8 GB for N=1000,D=27)
    //    that the previous implementation used, and eliminates its ~2.4 GB/s-limited memset.
    double t3_start = omp_get_wtime();
    t_phase3 = omp_get_wtime() - t3_start;  // nothing to do

    // 4) Projection table V = J_hat @ X1^T  (BIG x N)  (identical to hessian_symm)
    double t4_start = omp_get_wtime();
    double *V = aligned_alloc_64(BIG * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)BIG, (int)N, (int)M, 1.0,
                J_hat, (int)M, X1, (int)M, 0.0, V, (int)N);
    t_phase4 = omp_get_wtime() - t4_start;

    // 5) Self projections: U[a] = J_a^T x_a  (N x D)  (identical to hessian_symm)
    double t5_start = omp_get_wtime();
    std::vector<double> U(N * D);

#if !defined(__APPLE__)
    int saved_blas_threads_phase5 = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *__restrict x = X1 + a * M;
        const double *__restrict Ja = dX1 + (a * M) * D;
        double *__restrict Ua = U.data() + a * D;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D, 1.0, Ja, (int)D, x, 1, 0.0, Ua, 1);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads_phase5);
#endif

    t_phase5 = omp_get_wtime() - t5_start;

    // 6) Per-block lower-triangle: compute D×D Gram on-the-fly, apply scaling +
    //    rank-1 correction, write directly to RFP. No H_temp needed.
    double t6_start = omp_get_wtime();
    if (tile_B == 0)
        tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N);

#if !defined(__APPLE__)
    int saved_blas_threads_symm = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel
    {
        // Thread-local scratch: D×D Gram block + v1/v2 correction vectors
        double *Gblk = aligned_alloc_64(D * D);
        std::vector<double> v1(D), v2(D);

#pragma omp for schedule(static)
        for (std::size_t a = 0; a < N; ++a) {
            const double *__restrict Ua = U.data() + a * D;
            const double *__restrict Va = V + (a * D) * N;
            // J_hat rows for block a: J_hat + a*D*M, shape (D × M)
            const double *__restrict Ja_hat = J_hat + a * D * M;

            for (std::size_t b0 = 0; b0 < a + 1; b0 += tile_B) {
                const std::size_t bend = std::min<std::size_t>(a + 1, b0 + tile_B);

                for (std::size_t b = b0; b < bend; ++b) {
                    const double cab  = C[a * N + b];
                    const double c4ab = C4[a * N + b];

                    // Compute Gram block: Gblk(D×D) = J_hat[a*D:(a+1)*D, :] @ J_hat[b*D:(b+1)*D, :]^T
                    // J_hat rows are (D×M) for each molecule block.
                    const double *__restrict Jb_hat = J_hat + b * D * M;
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                (int)D, (int)D, (int)M,
                                1.0, Ja_hat, (int)M, Jb_hat, (int)M,
                                0.0, Gblk, (int)D);

                    // v1 = U[a] - V[aD:(a+1)D, b]
#pragma omp simd
                    for (std::size_t i = 0; i < D; ++i)
                        v1[i] = Ua[i] - Va[i * N + b];

                    // v2 = V[bD:(b+1)D, a] - U[b]
                    const double *__restrict Ub = U.data() + b * D;
                    const double *__restrict Vb = V + (b * D) * N;
#pragma omp simd
                    for (std::size_t j = 0; j < D; ++j)
                        v2[j] = Vb[j * N + a] - Ub[j];

                    // Apply scaling + rank-1 correction and write to RFP.
                    // Global indices: row = a*D+i, col = b*D+j, row >= col for lower tri.
                    // For a > b all entries are lower-tri (row >= col always).
                    // For a == b only lower triangle (i >= j) is written.
                    for (std::size_t i = 0; i < D; ++i) {
                        const std::size_t row = a * D + i;
                        const std::size_t j_max = (a == b) ? (i + 1) : D;
                        for (std::size_t j = 0; j < j_max; ++j) {
                            const std::size_t col = b * D + j;
                            const double val = Gblk[i * D + j] * cab - c4ab * v1[i] * v2[j];
                            // RFP upper: rfp_index_upper_N(BIG, col, row) with col <= row
                            H_rfp[rfp_index_upper_N(BIG, col, row)] = val;
                        }
                    }
                }
            }
        }
        aligned_free_64(Gblk);
    }
    t_phase6 = omp_get_wtime() - t6_start;

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
            printf("Phase 3 (base Gram):               %8.4f ms  (%5.1f%%)  [on-the-fly per block in phase 6]\n", t_phase3*1000, 100*t_phase3/t_total);
            printf("Phase 4 (projection table):        %8.4f ms  (%5.1f%%)\n", t_phase4*1000, 100*t_phase4/t_total);
            printf("Phase 5 (self projections):        %8.4f ms  (%5.1f%%)\n", t_phase5*1000, 100*t_phase5/t_total);
            printf("Phase 6 (per-block Gram+correct):  %8.4f ms  (%5.1f%%)\n", t_phase6*1000, 100*t_phase6/t_total);
            printf("TOTAL:                             %8.4f ms\n", t_total*1000);
            printf("====================================================\n\n");
        }
    }

    aligned_free_64(V);
    aligned_free_64(J_hat);
    aligned_free_64(C4);
    aligned_free_64(C);
    aligned_free_64(S);
}

// ============================================================================
// Full combined energy+force kernel (asymmetric).
// Output K_full is (N1*(1+D1)) x (N2*(1+D2)), row-major (stride = full_cols).
//
// Block layout (full_cols = N2 + N2*D2 = N2*(1+D2)):
//   [0:N1,        0:N2]          scalar  K[a,b] = exp(-||x1a-x2b||^2/(2σ²))
//   [0:N1,        N2:full_cols]  jac_t   K_jt[a, b*D2+d] = C[a,b]*(U2[b,d] - V2X1[b,d,a])
//   [N1:,         0:N2]          jac     K_j[a*D1+d, b]  = C[a,b]*(V1X2[a,d,b] - U1[a,d])
//   [N1:,         N2:full_cols]  hessian H[a*D1+d1, b*D2+d2] (on-the-fly Gram)
//
// where:
//   C[a,b]  = exp(-0.5*inv_s2*sq) * inv_s2  (= K[a,b] / σ²)
//   C4[a,b] = exp(-0.5*inv_s2*sq) * inv_s4  (= K[a,b] / σ⁴)
//   V1X2[a,d,b] = (J1_hat row a*D1+d) · X2[b]   [rows of J1_hat dotted with X2 cols]
//   V2X1[b,d,a] = (J2_hat row b*D2+d) · X1[a]   [rows of J2_hat dotted with X1 cols]
//   U1[a,d]     = (J1_hat row a*D1+d) · X1[a]
//   U2[b,d]     = (J2_hat row b*D2+d) · X2[b]
// ============================================================================
void kernel_gaussian_full(
    const double *__restrict X1, const double *__restrict dX1,
    const double *__restrict X2, const double *__restrict dX2,
    std::size_t N1, std::size_t N2, std::size_t M,
    std::size_t D1, std::size_t D2,
    double sigma, std::size_t tile_B,
    double *__restrict K_full) {

    if (!X1 || !dX1 || !X2 || !dX2 || !K_full)
        throw std::invalid_argument("null pointer");
    if (N1 == 0 || N2 == 0 || M == 0 || D1 == 0 || D2 == 0)
        throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0");

    const double inv_s2 = 1.0 / (sigma * sigma);
    const double inv_s4 = inv_s2 * inv_s2;

    const std::size_t big_rows = N1 * D1;   // rows of hessian block
    const std::size_t big_cols = N2 * D2;   // cols of hessian block
    const std::size_t full_rows = N1 + big_rows;   // total rows of K_full
    const std::size_t full_cols = N2 + big_cols;   // total cols of K_full (= stride)

    double t_start = omp_get_wtime();

    // -------------------------------------------------------------------------
    // Phase 1: Distance coefficients C[a,b] = K/s^2, C4[a,b] = K/s^4
    // -------------------------------------------------------------------------
    std::vector<double> n1v(N1), n2v(N2);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        const double *x = X1 + a * M;
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (std::size_t i = 0; i < M; ++i) s += x[i] * x[i];
        n1v[a] = s;
    }
#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        const double *x = X2 + b * M;
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (std::size_t i = 0; i < M; ++i) s += x[i] * x[i];
        n2v[b] = s;
    }

    double *S = aligned_alloc_64(N1 * N2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)N1, (int)N2, (int)M,
                1.0, X1, (int)M, X2, (int)M, 0.0, S, (int)N2);

    double *C  = aligned_alloc_64(N1 * N2);
    double *C4 = aligned_alloc_64(N1 * N2);
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        for (std::size_t b = 0; b < N2; ++b) {
            const double sq = n1v[a] + n2v[b] - 2.0 * S[a * N2 + b];
            const double k  = std::exp(-0.5 * inv_s2 * sq);
            C [a * N2 + b] = k * inv_s2;
            C4[a * N2 + b] = k * inv_s4;
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2: Pack Jacobians into J1_hat (big_rows x M) and J2_hat (big_cols x M)
    // J_hat[a*D+d, i] = J[a, i, d]  (transpose (M,D) -> (D,M) per molecule)
    // -------------------------------------------------------------------------
    double *J1_hat = aligned_alloc_64(big_rows * M);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        const double *J1 = dX1 + a * M * D1;
        for (std::size_t d = 0; d < D1; ++d) {
            double *row = J1_hat + (a * D1 + d) * M;
#pragma omp simd
            for (std::size_t i = 0; i < M; ++i) row[i] = J1[i * D1 + d];
        }
    }

    double *J2_hat = aligned_alloc_64(big_cols * M);
#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        const double *J2 = dX2 + b * M * D2;
        for (std::size_t d = 0; d < D2; ++d) {
            double *row = J2_hat + (b * D2 + d) * M;
#pragma omp simd
            for (std::size_t i = 0; i < M; ++i) row[i] = J2[i * D2 + d];
        }
    }

    // -------------------------------------------------------------------------
    // Phase 3: Projection tables
    //   V1X2 (big_rows x N2) = J1_hat @ X2^T
    //   V2X1 (big_cols x N1) = J2_hat @ X1^T
    // -------------------------------------------------------------------------
    double *V1X2 = aligned_alloc_64(big_rows * N2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)big_rows, (int)N2, (int)M,
                1.0, J1_hat, (int)M, X2, (int)M, 0.0, V1X2, (int)N2);

    double *V2X1 = aligned_alloc_64(big_cols * N1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)big_cols, (int)N1, (int)M,
                1.0, J2_hat, (int)M, X1, (int)M, 0.0, V2X1, (int)N1);

    // -------------------------------------------------------------------------
    // Phase 4: Self projections U1[a*D1+d] = J1_hat[a,d,:] · X1[a,:]
    //                           U2[b*D2+d] = J2_hat[b,d,:] · X2[b,:]
    // -------------------------------------------------------------------------
    std::vector<double> U1(big_rows), U2(big_cols);

#if !defined(__APPLE__)
    int saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    (int)M, (int)D1, 1.0,
                    dX1 + a * M * D1, (int)D1,
                    X1  + a * M, 1,
                    0.0, U1.data() + a * D1, 1);
    }
#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    (int)M, (int)D2, 1.0,
                    dX2 + b * M * D2, (int)D2,
                    X2  + b * M, 1,
                    0.0, U2.data() + b * D2, 1);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads);
#endif

    // -------------------------------------------------------------------------
    // Phase 5: Fill K_full — all four blocks in a single pass over (a, b).
    //
    // Memory layout of K_full (stride = full_cols = N2 + N2*D2):
    //
    //   Scalar [a, b]:              K_full[a * full_cols + b]
    //   Jac_t  [a, N2 + b*D2 + d]: K_full[a * full_cols + N2 + b*D2 + d]
    //   Jac    [N1+a*D1+d, b]:      K_full[(N1 + a*D1 + d) * full_cols + b]
    //   Hess   [N1+a*D1+d1, N2+b*D2+d2]:
    //                               K_full[(N1+a*D1+d1)*full_cols + N2+b*D2+d2]
    // -------------------------------------------------------------------------
    if (tile_B == 0) tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N2);

#if !defined(__APPLE__)
    saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel
    {
        double *Gblk = aligned_alloc_64(D1 * D2);
        std::vector<double> v1(D1), v2(D2);

#pragma omp for schedule(static)
        for (std::size_t a = 0; a < N1; ++a) {
            const double *U1a     = U1.data() + a * D1;
            const double *V1X2a   = V1X2 + (a * D1) * N2;   // (D1 x N2) block
            const double *J1a_hat = J1_hat + a * D1 * M;     // (D1 x M) block

            // Row pointers in K_full for this 'a'
            double *Kscalar_row = K_full + a * full_cols;          // scalar and jac_t row
            double *Kjac_base   = K_full + (N1 + a * D1) * full_cols; // base for jac+hess rows

            for (std::size_t b0 = 0; b0 < N2; b0 += tile_B) {
                const std::size_t bend = std::min(N2, b0 + tile_B);

                for (std::size_t b = b0; b < bend; ++b) {
                    const double cab  = C [a * N2 + b];
                    const double c4ab = C4[a * N2 + b];
                    // Recover raw kernel K = cab * sigma^2
                    const double kab  = cab * sigma * sigma;

                    // ----- Scalar block [a, b] -----
                    Kscalar_row[b] = kab;

                    // ----- Jacobian_t block [a, N2 + b*D2 : N2 + (b+1)*D2] -----
                    // K_jt[a, b*D2+d] = C[a,b] * (V2X1[b*D2+d, a] - U2[b*D2+d])
                    // Matches standalone: diff = X1a - X2b, K_jt = coeff * J2b^T * diff
                    {
                        const double *U2b    = U2.data() + b * D2;
                        const double *V2X1ba = V2X1 + (b * D2) * N1 + a;  // stride N1
                        double *Kjt_dest     = Kscalar_row + N2 + b * D2;
#pragma omp simd
                        for (std::size_t d = 0; d < D2; ++d)
                            Kjt_dest[d] = cab * (V2X1ba[d * N1] - U2b[d]);
                    }

                    // ----- Jacobian block [N1+a*D1 : N1+(a+1)*D1, b] -----
                    // K_j[a*D1+d, b] = C[a,b] * (V1X2[a*D1+d, b] - U1[a*D1+d])
                    {
                        for (std::size_t d = 0; d < D1; ++d) {
                            Kjac_base[d * full_cols + b] =
                                cab * (V1X2a[d * N2 + b] - U1a[d]);
                        }
                    }

                    // ----- Hessian block [N1+a*D1+d1, N2+b*D2+d2] -----
                    // Gram block: Gblk(D1 x D2) = J1_hat[a*D1:,:] @ J2_hat[b*D2:,:]^T
                    const double *J2b_hat = J2_hat + b * D2 * M;
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                (int)D1, (int)D2, (int)M,
                                1.0, J1a_hat, (int)M, J2b_hat, (int)M,
                                0.0, Gblk, (int)D2);

                    // v1[d] = U1[a*D1+d] - V1X2[a*D1+d, b]
#pragma omp simd
                    for (std::size_t d = 0; d < D1; ++d)
                        v1[d] = U1a[d] - V1X2a[d * N2 + b];

                    // v2[d] = V2X1[b*D2+d, a] - U2[b*D2+d]
                    const double *U2b    = U2.data() + b * D2;
                    const double *V2X1ba = V2X1 + (b * D2) * N1 + a;
#pragma omp simd
                    for (std::size_t d = 0; d < D2; ++d)
                        v2[d] = V2X1ba[d * N1] - U2b[d];

                    // Write hessian block
                    for (std::size_t d1 = 0; d1 < D1; ++d1) {
                        double *Hrow = Kjac_base + d1 * full_cols + N2 + b * D2;
#pragma omp simd
                        for (std::size_t d2 = 0; d2 < D2; ++d2)
                            Hrow[d2] = Gblk[d1 * D2 + d2] * cab - c4ab * v1[d1] * v2[d2];
                    }
                }
            }
        }
        aligned_free_64(Gblk);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads);
#endif

    // Profiling
    const char *profile_env = std::getenv("KERNELFORGE_PROFILE");
    if (profile_env && std::atoi(profile_env) != 0) {
        double t_total = omp_get_wtime() - t_start;
        printf("\n=== kernel_gaussian_full profiling ===\n");
        printf("Problem size: N1=%zu, N2=%zu, M=%zu, D1=%zu, D2=%zu\n", N1, N2, M, D1, D2);
        printf("Output size: %zu x %zu\n", full_rows, full_cols);
        printf("TOTAL: %8.4f ms\n", t_total * 1000);
        printf("======================================\n\n");
    }

    aligned_free_64(V2X1);
    aligned_free_64(V1X2);
    aligned_free_64(J2_hat);
    aligned_free_64(J1_hat);
    aligned_free_64(C4);
    aligned_free_64(C);
    aligned_free_64(S);
}

// ============================================================================
// Full combined energy+force kernel (symmetric: X1==X2, D1==D2).
// Output K_full is (N*(1+D)) x (N*(1+D)), row-major, lower triangle only.
//
// Block layout (BIG = N*(1+D), full_cols = BIG):
//   [0:N,    0:N]     scalar block   (lower triangle)
//   [N:,     0:N]     jacobian block (all entries)
//   [0:N,    N:]      jacobian_t block (upper-left transpose of jacobian block — NOT stored,
//                      caller fills from lower triangle)
//   [N:,     N:]      hessian block  (lower triangle only)
//
// Note: This fills the lower triangle of K_full (row >= col).
// The scalar diagonal is K[a,a] = 1.0 (exp(0)).
// The jacobian_t upper block is the transpose of the jacobian lower-left block.
// The hessian block is symmetric; only lower triangle is filled.
// ============================================================================
void kernel_gaussian_full_symm(
    const double *__restrict X,
    const double *__restrict dX,
    std::size_t N, std::size_t M, std::size_t D,
    double sigma, std::size_t tile_B,
    double *__restrict K_full) {

    if (!X || !dX || !K_full)
        throw std::invalid_argument("null pointer");
    if (N == 0 || M == 0 || D == 0)
        throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0");

    const double inv_s2 = 1.0 / (sigma * sigma);
    const double inv_s4 = inv_s2 * inv_s2;

    const std::size_t big  = N * D;           // rows/cols of hessian block
    const std::size_t BIG  = N + big;         // total rows/cols of K_full
    const std::size_t full_cols = BIG;        // stride

    double t_start = omp_get_wtime();

    // ---- Phase 1: Distance coefficients (symmetric, lower triangle only) ----
    std::vector<double> nv(N);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *x = X + a * M;
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (std::size_t i = 0; i < M; ++i) s += x[i] * x[i];
        nv[a] = s;
    }

    // S = X @ X^T  (symmetric, N x N)
    double *S = aligned_alloc_64(N * N);
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                (int)N, (int)M, 1.0, X, (int)M, 0.0, S, (int)N);
    // Fill upper from lower for distance calculation convenience
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = a + 1; b < N; ++b)
            S[a * N + b] = S[b * N + a];

    double *C  = aligned_alloc_64(N * N);
    double *C4 = aligned_alloc_64(N * N);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b <= a; ++b) {
            const double sq = nv[a] + nv[b] - 2.0 * S[a * N + b];
            const double k  = std::exp(-0.5 * inv_s2 * sq);
            C [a * N + b] = k * inv_s2;
            C4[a * N + b] = k * inv_s4;
            // Mirror for projection table phases
            C [b * N + a] = C [a * N + b];
            C4[b * N + a] = C4[a * N + b];
        }
    }

    // ---- Phase 2: Pack Jacobians J_hat (big x M) ----
    double *J_hat = aligned_alloc_64(big * M);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *J = dX + a * M * D;
        for (std::size_t d = 0; d < D; ++d) {
            double *row = J_hat + (a * D + d) * M;
#pragma omp simd
            for (std::size_t i = 0; i < M; ++i) row[i] = J[i * D + d];
        }
    }

    // ---- Phase 3: Projection table V (big x N) = J_hat @ X^T ----
    double *V = aligned_alloc_64(big * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)big, (int)N, (int)M,
                1.0, J_hat, (int)M, X, (int)M, 0.0, V, (int)N);

    // ---- Phase 4: Self projections U[a*D+d] = J_hat[a,d,:] · X[a,:] ----
    std::vector<double> U(big);

#if !defined(__APPLE__)
    int saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    (int)M, (int)D, 1.0,
                    dX + a * M * D, (int)D,
                    X  + a * M, 1,
                    0.0, U.data() + a * D, 1);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads);
#endif

    // ---- Phase 5: Fill K_full lower triangle ----
    // For (a, b) pairs with a >= b:
    //   Scalar:   [a, b]               C[a,b]*s^2  (and [b,a] by symmetry)
    //   Jac:      [N+a*D+d, b]         C[a,b]*(V[a,d,b] - U[a,d])
    //   Jac_t:    [b, N+a*D+d]         = Jac^T (upper triangle — don't fill)
    //             [a, N+b*D+d] lower   C[a,b]*(U[b,d] - V[b,d,a])
    //   Hessian:  [N+a*D+d1, N+b*D+d2] (lower: a>=b only full rows, within a==b lower tri)

    if (tile_B == 0) tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N);

#if !defined(__APPLE__)
    saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel
    {
        double *Gblk = aligned_alloc_64(D * D);
        std::vector<double> v1(D), v2(D);

#pragma omp for schedule(static)
        for (std::size_t a = 0; a < N; ++a) {
            const double *Ua   = U.data() + a * D;
            const double *Va   = V + (a * D) * N;  // row group for a: (D x N)
            const double *Ja_hat = J_hat + a * D * M;

            double *row_scalar_a  = K_full + a * full_cols;                // row a (scalar+jact)
            double *rows_hess_a   = K_full + (N + a * D) * full_cols;     // rows N+a*D.. of K_full

            for (std::size_t b = 0; b <= a; ++b) {
                const double cab  = C [a * N + b];
                const double c4ab = C4[a * N + b];
                const double kab  = cab * sigma * sigma;

                const double *Ub   = U.data() + b * D;
                const double *Vb   = V + (b * D) * N;  // (D x N)

                // ---- Scalar block (lower triangle) ----
                // [a, b] and diagonal
                row_scalar_a[b] = kab;
                if (a == b) {
                    // Self kernel = exp(0) = 1.0; also fill diagonal of scalar block
                    row_scalar_a[b] = 1.0;
                }
                if (b < a) {
                    // [b, a] = [a, b] (mirror into upper scalar block)
                    K_full[b * full_cols + a] = kab;
                }

                // ---- Jacobian block: rows N+a*D..N+(a+1)*D, col b ----
                // K_j[a*D+d, b] = C[a,b] * (V[a,d,b] - U[a,d])
                // Also fill jacobian_t: [b, N+a*D+d] = same value (transpose)
                for (std::size_t d = 0; d < D; ++d) {
                    const double val_j = cab * (Va[d * N + b] - Ua[d]);
                    rows_hess_a[(long)(d * full_cols) + (long)b] = val_j;
                    // Jacobian_t (upper triangle): [b, N+a*D+d]
                    K_full[b * full_cols + N + a * D + d] = val_j;
                }

                // ---- If a > b: also fill jacobian for b at col a ----
                // K_j[b*D+d, a] = C[b,a] * (V[b,d,a] - U[b,d])
                if (b < a) {
                    double *rows_hess_b = K_full + (N + b * D) * full_cols;
                    for (std::size_t d = 0; d < D; ++d) {
                        const double val_j = cab * (Vb[d * N + a] - Ub[d]);
                        rows_hess_b[d * full_cols + a] = val_j;
                        // Jacobian_t [a, N+b*D+d]
                        K_full[a * full_cols + N + b * D + d] = val_j;
                    }
                }

                // ---- Hessian block [N+a*D+d1, N+b*D+d2] (lower triangle: b <= a) ----
                const double *Jb_hat = J_hat + b * D * M;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            (int)D, (int)D, (int)M,
                            1.0, Ja_hat, (int)M, Jb_hat, (int)M,
                            0.0, Gblk, (int)D);

                // v1[d] = U[a,d] - V[a,d,b]
#pragma omp simd
                for (std::size_t d = 0; d < D; ++d) v1[d] = Ua[d] - Va[d * N + b];
                // v2[d] = V[b,d,a] - U[b,d]
#pragma omp simd
                for (std::size_t d = 0; d < D; ++d) v2[d] = Vb[d * N + a] - Ub[d];

                if (a == b) {
                    // Diagonal hessian block: fill lower triangle then mirror to upper
                    for (std::size_t d1 = 0; d1 < D; ++d1) {
                        double *Hrow = rows_hess_a + d1 * full_cols + N + b * D;
                        for (std::size_t d2 = 0; d2 <= d1; ++d2)
                            Hrow[d2] = Gblk[d1 * D + d2] * cab - c4ab * v1[d1] * v2[d2];
                    }
                    // Mirror lower→upper within diagonal D×D block
                    for (std::size_t d1 = 0; d1 < D; ++d1) {
                        for (std::size_t d2 = d1 + 1; d2 < D; ++d2) {
                            rows_hess_a[d1 * full_cols + N + b * D + d2] =
                                rows_hess_a[d2 * full_cols + N + b * D + d1];
                        }
                    }
                    // Also mirror the off-diagonal hessian blocks to upper-right
                    // (already handled by the b<a else branch for off-diagonal blocks,
                    //  but the block [N+a*D:, N+a*D:] upper portion also needs mirroring
                    //  into [N+b*D:, N+a*D:] — not needed since b==a here)
                } else {
                    // Off-diagonal: fill all D×D entries for [N+a*D:, N+b*D:]
                    for (std::size_t d1 = 0; d1 < D; ++d1) {
                        double *Hrow = rows_hess_a + d1 * full_cols + N + b * D;
#pragma omp simd
                        for (std::size_t d2 = 0; d2 < D; ++d2)
                            Hrow[d2] = Gblk[d1 * D + d2] * cab - c4ab * v1[d1] * v2[d2];
                    }
                    // Mirror transposed block [N+b*D:, N+a*D:] = [N+a*D:, N+b*D:]^T
                    // (Done in a separate post-loop pass to avoid OpenMP race conditions)
                }
            }
        }
        aligned_free_64(Gblk);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads);
#endif

    // ---- Mirror off-diagonal hessian blocks to upper triangle ----
    // For each pair (a > b), the block at [N+a*D:, N+b*D:] (D×D) has been filled.
    // We now copy it transposed to [N+b*D:, N+a*D:].
#pragma omp parallel for schedule(static)
    for (std::size_t a = 1; a < N; ++a) {
        for (std::size_t b = 0; b < a; ++b) {
            for (std::size_t d1 = 0; d1 < D; ++d1) {
                for (std::size_t d2 = 0; d2 < D; ++d2) {
                    // Source: K_full[N+a*D+d1, N+b*D+d2]
                    // Dest:   K_full[N+b*D+d2, N+a*D+d1]
                    K_full[(N + b * D + d2) * full_cols + (N + a * D + d1)] =
                        K_full[(N + a * D + d1) * full_cols + (N + b * D + d2)];
                }
            }
        }
    }

    // Profiling
    const char *profile_env = std::getenv("KERNELFORGE_PROFILE");
    if (profile_env && std::atoi(profile_env) != 0) {
        double t_total = omp_get_wtime() - t_start;
        printf("\n=== kernel_gaussian_full_symm profiling ===\n");
        printf("Problem size: N=%zu, M=%zu, D=%zu\n", N, M, D);
        printf("Output size: %zu x %zu (full symmetric)\n", BIG, BIG);
        printf("TOTAL: %8.4f ms\n", t_total * 1000);
        printf("===========================================\n\n");
    }

    aligned_free_64(V);
    aligned_free_64(J_hat);
    aligned_free_64(C4);
    aligned_free_64(C);
    aligned_free_64(S);
}

// ============================================================================
// Full combined energy+force kernel (symmetric RFP format).
// Output: 1D RFP array of length BIG*(BIG+1)/2, where BIG = N*(1+D).
// Uses the same RFP indexing as kernel_gaussian_hessian_symm_rfp (TRANSR='N', UPLO='U').
// ============================================================================
void kernel_gaussian_full_symm_rfp(
    const double *__restrict X,
    const double *__restrict dX,
    std::size_t N, std::size_t M, std::size_t D,
    double sigma, std::size_t tile_B,
    double *__restrict K_rfp) {

    if (!X || !dX || !K_rfp)
        throw std::invalid_argument("null pointer");
    if (N == 0 || M == 0 || D == 0)
        throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0))
        throw std::invalid_argument("sigma must be > 0");

    const double inv_s2 = 1.0 / (sigma * sigma);
    const double inv_s4 = inv_s2 * inv_s2;

    const std::size_t big  = N * D;
    const std::size_t BIG  = N + big;         // = N*(1+D)
    const std::size_t rfp_size = BIG * (BIG + 1) / 2;

    double t_start = omp_get_wtime();

    // ---- Phase 1: Distance coefficients ----
    std::vector<double> nv(N);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *x = X + a * M;
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (std::size_t i = 0; i < M; ++i) s += x[i] * x[i];
        nv[a] = s;
    }

    double *S = aligned_alloc_64(N * N);
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                (int)N, (int)M, 1.0, X, (int)M, 0.0, S, (int)N);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = a + 1; b < N; ++b)
            S[a * N + b] = S[b * N + a];

    double *C  = aligned_alloc_64(N * N);
    double *C4 = aligned_alloc_64(N * N);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b <= a; ++b) {
            const double sq = nv[a] + nv[b] - 2.0 * S[a * N + b];
            const double k  = std::exp(-0.5 * inv_s2 * sq);
            C [a * N + b] = C [b * N + a] = k * inv_s2;
            C4[a * N + b] = C4[b * N + a] = k * inv_s4;
        }
    }

    // ---- Phase 2: Pack Jacobians ----
    double *J_hat = aligned_alloc_64(big * M);
#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double *J = dX + a * M * D;
        for (std::size_t d = 0; d < D; ++d) {
            double *row = J_hat + (a * D + d) * M;
#pragma omp simd
            for (std::size_t i = 0; i < M; ++i) row[i] = J[i * D + d];
        }
    }

    // ---- Phase 3: Projection table V (big x N) ----
    double *V = aligned_alloc_64(big * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)big, (int)N, (int)M,
                1.0, J_hat, (int)M, X, (int)M, 0.0, V, (int)N);

    // ---- Phase 4: Self projections ----
    std::vector<double> U(big);

#if !defined(__APPLE__)
    int saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    (int)M, (int)D, 1.0,
                    dX + a * M * D, (int)D, X + a * M, 1,
                    0.0, U.data() + a * D, 1);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads);
#endif

    // ---- Phase 5: Fill RFP ----
    // RFP index for (row, col) in the upper triangle of a BIG×BIG symmetric matrix.
    // We use the same rfp_index_upper_N helper (for std::size_t).
    // The matrix is indexed as: K_full[row, col] with row <= col (upper convention).
    //
    // Our block structure in K_full (global row/col 0-based):
    //   Scalar block:    rows [0,N),  cols [0,N)
    //   Jac_t block:     rows [0,N),  cols [N, BIG)  = [0,N) x [N+a*D, N+(a+1)*D)
    //   Hessian block:   rows [N,BIG), cols [N,BIG)
    //
    // The Jacobian block (lower-left) has row>=col in upper-triangle sense only when
    // row >= N (hessian rows) and col < N (scalar cols) — that's the lower-left,
    // which is NOT in upper triangle. We store the UPPER triangle (row<=col), so:
    //   Scalar:   row=a, col=b, a<=b  => rfp(a, b)
    //   Jac_t:    row=a, col=N+b*D+d, a < N+b*D+d  => rfp(a, N+b*D+d)
    //   Hessian:  row=N+a*D+d1, col=N+b*D+d2, d1<=d2 when a==b, any when a<b
    //             => rfp(N+a*D+d1, N+b*D+d2) for a<=b

    if (tile_B == 0) tile_B = std::min<std::size_t>(DEFAULT_TILE_SIZE, N);

#if !defined(__APPLE__)
    saved_blas_threads = blas_get_num_threads();
    blas_set_num_threads(1);
#endif

#pragma omp parallel
    {
        double *Gblk = aligned_alloc_64(D * D);
        std::vector<double> v1(D), v2(D);

#pragma omp for schedule(static)
        for (std::size_t a = 0; a < N; ++a) {
            const double *Ua     = U.data() + a * D;
            const double *Va     = V + (a * D) * N;
            const double *Ja_hat = J_hat + a * D * M;

            for (std::size_t b = a; b < N; ++b) {  // upper triangle: b >= a
                const double cab  = C [a * N + b];
                const double c4ab = C4[a * N + b];
                const double kab  = cab * sigma * sigma;

                const double *Ub     = U.data() + b * D;
                const double *Vb     = V + (b * D) * N;
                const double *Jb_hat = J_hat + b * D * M;

                // ---- Scalar block [a, b] (upper: a <= b) ----
                K_rfp[rfp_index_upper_N(BIG, a, b)] = (a == b) ? 1.0 : kab;

                // ---- Jac_t block: [a, N+b*D+d] (always upper since N+b*D+d > a for any a<N) ----
                // K_full[a, N+b*D+d] = K_j[N+b*D+d, a] = C[a,b] * (V[b,d,a] - U[b,d])
                for (std::size_t d = 0; d < D; ++d) {
                    const std::size_t col = N + b * D + d;
                    K_rfp[rfp_index_upper_N(BIG, a, col)] = cab * (Vb[d * N + a] - Ub[d]);
                }

                if (a < b) {
                    // ---- Jac_t block: [b, N+a*D+d] (b < N+a*D+d always) ----
                    // K_full[b, N+a*D+d] = K_j[N+a*D+d, b] = C[a,b] * (V[a,d,b] - U[a,d])
                    for (std::size_t d = 0; d < D; ++d) {
                        const std::size_t col = N + a * D + d;
                        K_rfp[rfp_index_upper_N(BIG, b, col)] = cab * (Va[d * N + b] - Ua[d]);
                    }
                }

                // ---- Hessian block [N+a*D+d1, N+b*D+d2] ----
                // For a < b: all D*D entries (both d1<d2 and d1>=d2 positions)
                // For a == b: upper triangle d1 <= d2
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            (int)D, (int)D, (int)M,
                            1.0, Ja_hat, (int)M, Jb_hat, (int)M,
                            0.0, Gblk, (int)D);

#pragma omp simd
                for (std::size_t d = 0; d < D; ++d) v1[d] = Ua[d] - Va[d * N + b];
#pragma omp simd
                for (std::size_t d = 0; d < D; ++d) v2[d] = Vb[d * N + a] - Ub[d];

                if (a == b) {
                    for (std::size_t d1 = 0; d1 < D; ++d1) {
                        for (std::size_t d2 = d1; d2 < D; ++d2) {
                            const std::size_t row = N + a * D + d1;
                            const std::size_t col = N + b * D + d2;
                            K_rfp[rfp_index_upper_N(BIG, row, col)] =
                                Gblk[d1 * D + d2] * cab - c4ab * v1[d1] * v2[d2];
                        }
                    }
                } else {
                    for (std::size_t d1 = 0; d1 < D; ++d1) {
                        for (std::size_t d2 = 0; d2 < D; ++d2) {
                            const std::size_t row = N + a * D + d1;
                            const std::size_t col = N + b * D + d2;
                            K_rfp[rfp_index_upper_N(BIG, row, col)] =
                                Gblk[d1 * D + d2] * cab - c4ab * v1[d1] * v2[d2];
                        }
                    }
                }
            }
        }
        aligned_free_64(Gblk);
    }

#if !defined(__APPLE__)
    blas_set_num_threads(saved_blas_threads);
#endif

    // Profiling
    const char *profile_env = std::getenv("KERNELFORGE_PROFILE");
    if (profile_env && std::atoi(profile_env) != 0) {
        double t_total = omp_get_wtime() - t_start;
        printf("\n=== kernel_gaussian_full_symm_rfp profiling ===\n");
        printf("Problem size: N=%zu, M=%zu, D=%zu, BIG=%zu\n", N, M, D, BIG);
        printf("RFP size: %zu\n", rfp_size);
        printf("TOTAL: %8.4f ms\n", t_total * 1000);
        printf("===============================================\n\n");
    }

    aligned_free_64(V);
    aligned_free_64(J_hat);
    aligned_free_64(C4);
    aligned_free_64(C);
    aligned_free_64(S);
}

}  // namespace kf
