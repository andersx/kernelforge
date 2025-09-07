#include <algorithm>
#include <cmath>
#include <cstring>   // std::memcpy
#include <omp.h>
#include <iostream>
#include <vector>
#include <stdexcept>

#if defined(__APPLE__)
  #include <Accelerate/Accelerate.h>   // brings in clapack.h too
  #define LAPACK_CHAR_ARG char
#else
  #include <cblas.h>
  #define LAPACK_CHAR_ARG const char
// Fortran LAPACK declarations
extern "C" {
    void dpotrf_(LAPACK_CHAR_ARG* uplo, const int* n, double* a,
                 const int* lda, int* info);

    void dpotrs_(LAPACK_CHAR_ARG* uplo, const int* n, const int* nrhs,
                 const double* a, const int* lda, double* b,
                 const int* ldb, int* info);
}
#endif

void kernel_symm(
    const double* Xptr,
    int n,
    int rep_size,
    double alpha,
    double* Kptr
) {
    // 1) DSYRK (RowMajor, lower triangle): K = (-2*alpha) * X * X^T + 0*K
    double t0 = omp_get_wtime();
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, rep_size,
                -2.0 * alpha,
                Xptr, rep_size,
                0.0, Kptr, n);
    double t1 = omp_get_wtime();

    // 2) diag = -0.5 * diag(K)
    std::vector<double> diag(n);
    for (int i = 0; i < n; ++i) {
        diag[i] = -0.5 * Kptr[i * n + i];
    }

    // 3) K += 1 * (1*diag^T + diag*1^T) on LOWER via dsyr2
    std::vector<double> onevec(n, 1.0);
    cblas_dsyr2(CblasRowMajor, CblasLower, n, 1.0,
                onevec.data(), 1,
                diag.data(), 1,
                Kptr, n);

    // 4) Elementwise exp on the lower triangle (i <= j)
    #pragma omp parallel for schedule(guided)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            Kptr[j * n + i] = std::exp(Kptr[j * n + i]);
        }
    }
}

static inline void rowwise_self_norms(
    const double* X,
    std::size_t n,
    std::size_t d,
    double* out
) {
    // out[i] = sum_k X[i, k]^2  (row-major: row i is contiguous)
    for (std::size_t i = 0; i < n; ++i) {
        const double* row = X + i * d;
        double acc = 0.0;
        for (std::size_t k = 0; k < d; ++k) acc += row[k] * row[k];
        out[i] = acc;
    }
}

void kernel_asymm(const double* X1,
                  const double* X2,
                  std::size_t n1,
                  std::size_t n2,
                  std::size_t d,
                  double alpha,
                  double* K)
{
    // 1) K = (-2*alpha) * X1 * X2^T
    // RowMajor: A=X1 (n1 x d), B=X2 (n2 x d) but we pass Trans(B) => (d x n2)
    // lda = d, ldb = d, ldc = n1
    const double gemm_alpha = -2.0 * alpha;
    const double gemm_beta  = 0.0;
    double t0 = omp_get_wtime();
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                static_cast<int>(n1),       // M
                static_cast<int>(n2),       // N
                static_cast<int>(d),        // K
                gemm_alpha,
                X1, static_cast<int>(d),    // A, lda
                X2, static_cast<int>(d),    // B, ldb  (Trans)
                gemm_beta,
                K,  static_cast<int>(n2));  // C
    double t1 = omp_get_wtime();

    // 2) Rowwise self norms
    std::vector<double> nrm1(n1), nrm2(n2);
    rowwise_self_norms(X1, n1, d, nrm1.data());
    rowwise_self_norms(X2, n2, d, nrm2.data());

    // Ones
    std::vector<double> one_n1(n1, 1.0), one_n2(n2, 1.0);

    // 3) K += alpha * (ones_n2 * nrm1^T)   => GER(m=n1, n=n2)
    cblas_dger(CblasRowMajor,
               static_cast<int>(n1), static_cast<int>(n2),
               alpha,
               one_n1.data(), 1,
               nrm2.data(),   1,
               K, static_cast<int>(n2));

    // 4) K += alpha * (nrm2 * ones_n1^T)
    cblas_dger(CblasRowMajor,
               static_cast<int>(n1), static_cast<int>(n2),
               alpha,
               nrm1.data(),   1,
               one_n2.data(), 1,
               K, static_cast<int>(n2));

    // 5) Elementwise exp
    const std::size_t total = n2 * n1;
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(total); ++idx) {
        K[static_cast<std::size_t>(idx)] = std::exp(K[static_cast<std::size_t>(idx)]);
    }
}

// void gaussian_jacobian_batch(
//     const double* X1, const double* dX1, const double* X2,
//     std::size_t N1, std::size_t N2, std::size_t M, std::size_t D,
//     double sigma, double* K_out)
// {
//     if (!X1 || !dX1 || !X2 || !K_out) throw std::invalid_argument("null pointer");
//     if (N1==0 || N2==0 || M==0 || D==0) throw std::invalid_argument("empty dimension");
//     if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be > 0");
// 
//     const double inv_s2 = 1.0 / (sigma * sigma);
//     std::vector<double> diff(M), w(M);
// 
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* X2b = X2 + b * M;
// 
//         for (std::size_t a = 0; a < N1; ++a) {
//             const double* X1a = X1 + a * M;
//             const double* J1a = dX1 + (a * M) * D;  // (M,D) block base
// 
//             double sq = 0.0;
//             for (std::size_t i = 0; i < M; ++i) {
//                 const double di = X2b[i] - X1a[i];
//                 diff[i] = di;
//                 sq += di * di;
//             }
// 
//             const double k = std::exp(-0.5 * inv_s2 * sq);
//             const double coeff = k * inv_s2;
//             for (std::size_t i = 0; i < M; ++i) w[i] = coeff * diff[i];
// 
//             const std::size_t row_base = a * D;           // start row of this query block
//             for (std::size_t dj = 0; dj < D; ++dj) {
//                 double acc = 0.0;
//                 const double* Jcol = J1a + dj;            // J[i,dj] at Jcol[i*D]
//                 for (std::size_t i = 0; i < M; ++i) {
//                     acc += Jcol[i * D] * w[i];
//                 }
//                 // row-major write: index = row * N2 + col
//                 const std::size_t row = row_base + dj;
//                 K_out[row * N2 + b] = acc;                // ✅ correct row-major addressing
//             }
//         }
//     }
// }


void gaussian_jacobian_batch(
    const double* X1,
    const double* dX1,
    const double* X2,
    std::size_t N1,
    std::size_t N2,
    std::size_t M,
    std::size_t D,
    double sigma,
    double* K_out
) {
    if (!X1 || !dX1 || !X2 || !K_out) throw std::invalid_argument("null pointer");
    if (N1==0 || N2==0 || M==0 || D==0) throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be > 0");

    const double inv_s2 = 1.0 / (sigma * sigma);

    // scratch for W_a: (M x N2) row-major, and a column diff
    std::vector<double> W(static_cast<size_t>(M) * N2);
    std::vector<double> diff(M);

    for (std::size_t a = 0; a < N1; ++a) {
        const double* X1a = X1 + a * M;
        const double* J1a = dX1 + (a * M) * D;   // (M,D) block base
        // Build W(:,b) for all b
        for (std::size_t b = 0; b < N2; ++b) {
            const double* X2b = X2 + b * M;

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
        double* Kblock = K_out + (a * D) * N2;

        // Kblock(D x N2) = (J1a^T)(D x M) * W(M x N2)
        // J1a is (M x D) row-major in memory.
        // Row-major DGEMM: op(A)=A^T ⇒ (D x M), lda = D; B=W (M x N2), ldb=N2; C(Kblock) (D x N2), ldc=N2
        cblas_dgemm(
            CblasRowMajor,
            CblasTrans, CblasNoTrans,
            static_cast<int>(D), static_cast<int>(N2), static_cast<int>(M),
            1.0,
            J1a, static_cast<int>(D),
            W.data(), static_cast<int>(N2),
            0.0,
            Kblock, static_cast<int>(N2)
       );
    }
}

static inline void row_axpy(std::size_t n, double alpha, const double* x, double* y) {
    for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}
// If using MKL and you want local control of threads around level-1/2 ops:
// #include <mkl.h>

void rbf_hessian_full_tiled_gemm(
    const double* __restrict X1,  const double* __restrict dX1,
    const double* __restrict X2,  const double* __restrict dX2,
    std::size_t N1, std::size_t N2,
    std::size_t M,  std::size_t D1, std::size_t D2,
    double sigma,
    std::size_t tile_B,                   // now used
    double* __restrict H_out)
{
    if (!X1 || !dX1 || !X2 || !dX2 || !H_out) throw std::invalid_argument("null pointer");
    if (N1==0 || N2==0 || M==0 || D1==0 || D2==0) throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be > 0");

    const std::size_t big_rows = N1 * D1;
    const std::size_t big_cols = N2 * D2;

    const double inv_s2 = 1.0 / (sigma * sigma);
    const double inv_s4 = inv_s2 * inv_s2;

    // 1) Distances → C = K/s^2 and C4 = K/s^4
    std::vector<double> n1(N1), n2(N2);

    #pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        const double* __restrict x = X1 + a*M;
        double s = 0.0;
        #pragma omp simd reduction(+:s)
        for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
        n1[a] = s;
    }

    #pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        const double* __restrict x = X2 + b*M;
        double s = 0.0;
        #pragma omp simd reduction(+:s)
        for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
        n2[b] = s;
    }

    // S = X1 @ X2^T  (N1 x N2)  ← let BLAS thread this
    std::vector<double> S(N1 * N2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)N1, (int)N2, (int)M,
                1.0, X1, (int)M,
                     X2, (int)M,
                0.0, S.data(), (int)N2);

    // C, C4
    std::vector<double> C(N1 * N2), C4(N1 * N2);
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        for (std::size_t b = 0; b < N2; ++b) {
            const double sq = n1[a] + n2[b] - 2.0 * S[a*N2 + b];
            const double k  = std::exp(-0.5 * inv_s2 * sq);
            C [a*N2 + b] = k * inv_s2;
            C4[a*N2 + b] = k * inv_s4;
        }
    }

    // 2) Pack J1_hat (N1*D1 x M),  J2_all_cat (M x N2*D2)
    std::vector<double> J1_hat(big_rows * M);
    #pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        const double* __restrict J1 = dX1 + (a*M)*D1; // (M x D1)
        for (std::size_t dj = 0; dj < D1; ++dj) {
            double* __restrict row = J1_hat.data() + (a*D1 + dj) * M;
            // transpose column dj of (M x D1) → row of length M
            #pragma omp simd
            for (std::size_t i = 0; i < M; ++i)
                row[i] = J1[i*D1 + dj];
        }
    }

    std::vector<double> J2_all_cat(M * big_cols);
    // Parallelize outer over b to keep writes mostly contiguous per thread
    #pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        const double* __restrict J2 = dX2 + (b*M)*D2; // (M x D2)
        for (std::size_t i = 0; i < M; ++i) {
            std::memcpy(J2_all_cat.data() + i*big_cols + b*D2,
                        J2 + i*D2,
                        D2 * sizeof(double));
        }
    }
    const int lda_J1   = (int)M;
    const int ldb_J2cat= (int)big_cols;
    const int ldc_H    = (int)big_cols;

    // 3) Base Gram for ALL blocks once: H_out = J1_hat @ J2_all_cat  (threaded BLAS)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)big_rows, (int)big_cols, (int)M,
                1.0, J1_hat.data(), lda_J1,
                     J2_all_cat.data(), ldb_J2cat,
                0.0, H_out, ldc_H);

    // 4) Projection tables: V1X2_all and V2X1_all  (threaded BLAS)
    std::vector<double> V1X2_all(big_rows * N2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)big_rows, (int)N2, (int)M,
                1.0, J1_hat.data(), (int)M,
                     X2,             (int)M,
                0.0, V1X2_all.data(), (int)N2);

    std::vector<double> V2X1_all(big_cols * N1);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                (int)big_cols, (int)N1, (int)M,
                1.0, J2_all_cat.data(), (int)big_cols,
                     X1,               (int)M,
                0.0, V2X1_all.data(), (int)N1);

    // 5) Self projections: U1[a]=J1_a^T x1_a, U2[b]=J2_b^T x2_b  (parallelize outer loops)
    std::vector<double> U1(N1 * D1), U2(N2 * D2);

    #pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N1; ++a) {
        const double* __restrict x1 = X1 + a*M;
        const double* __restrict J1 = dX1 + (a*M)*D1;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D1,
                    1.0, J1, (int)D1, x1, 1, 0.0, U1.data() + a*D1, 1);
    }

    #pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < N2; ++b) {
        const double* __restrict x2 = X2 + b*M;
        const double* __restrict J2 = dX2 + (b*M)*D2;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D2,
                    1.0, J2, (int)D2, x2, 1, 0.0, U2.data() + b*D2, 1);
    }

    // 6) Per-block: scale Gram by C[a,b], then rank-1 correction via GER
    if (tile_B == 0) tile_B = std::min<std::size_t>(64, N2);  // reasonable default

    // If you can, make L1/L2 BLAS single-threaded here to avoid oversubscription:
    // int saved = mkl_get_max_threads();
    // mkl_set_num_threads_local(1);

    #pragma omp parallel
    {
        std::vector<double> v1(D1), v2(D2);  // private buffers per thread

        #pragma omp for schedule(static)
        for (std::size_t a = 0; a < N1; ++a) {

            const double* __restrict U1a   = U1.data() + a*D1;
            const double* __restrict V1X2a = V1X2_all.data() + (a*D1) * N2; // (D1 x N2), row-major

            for (std::size_t b0 = 0; b0 < N2; b0 += tile_B) {
                const std::size_t bend = std::min(N2, b0 + tile_B);

                for (std::size_t b = b0; b < bend; ++b) {
                    const double cab  = C [a*N2 + b];
                    const double c4ab = C4[a*N2 + b];

                    double* __restrict Hblk = H_out + (a*D1) * big_cols + b*D2;

                    // Scale Gram block by cab: D1 rows × D2 cols within the big row-major matrix
                    for (std::size_t i = 0; i < D1; ++i) {
                        cblas_dscal((int)D2, cab, Hblk + i*big_cols, 1);
                    }

                    // v1 = U1[a] - V1X2_all[aD1:(a+1)D1, b]
                    #pragma omp simd
                    for (std::size_t i = 0; i < D1; ++i) {
                        v1[i] = U1a[i] - V1X2a[i*N2 + b];
                    }

                    // v2 = V2X1_all[bD2:(b+1)D2, a] - U2[b]
                    const double* __restrict U2b     = U2.data() + b*D2;
                    const double* __restrict V2X1col = V2X1_all.data() + (b*D2) * N1 + a; // column 'a' in row-major
                    #pragma omp simd
                    for (std::size_t j = 0; j < D2; ++j) {
                        v2[j] = V2X1col[j*N1] - U2b[j];
                    }

                    // Rank-1 correction: H(a,b) -= c4ab * v1 v2^T  (Level-2 BLAS)
                    // Probably single-threaded
                    cblas_dger(CblasRowMajor,
                               (int)D1, (int)D2,
                               -c4ab,
                               v1.data(), 1,
                               v2.data(), 1,
                               Hblk, (int)big_cols);
                }
            }
        }
    }

    // mkl_set_num_threads_local(saved);
}



// Drop-in replacement: computes V1X2_all = J1_hat @ X2^T once
// void rbf_hessian_full_tiled_gemm(
//     const double* X1,  const double* dX1,
//     const double* X2,  const double* dX2,
//     std::size_t N1, std::size_t N2,
//     std::size_t M,  std::size_t D1, std::size_t D2,
//     double sigma,
//     std::size_t /*tile_B*/,
//     double* H_out)
// {
//     if (!X1 || !dX1 || !X2 || !dX2 || !H_out) throw std::invalid_argument("null pointer");
//     if (N1==0 || N2==0 || M==0 || D1==0 || D2==0) throw std::invalid_argument("empty dimension");
//     if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be > 0");
// 
//     const std::size_t big_rows = N1 * D1;
//     const std::size_t big_cols = N2 * D2;
// 
//     const double inv_s2 = 1.0 / (sigma * sigma);
//     const double inv_s4 = inv_s2 * inv_s2;
// 
//     // 1) Distances → C = K/s^2 and C4 = K/s^4
//     std::vector<double> n1(N1), n2(N2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x = X1 + a*M;
//         double s = 0.0; for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
//         n1[a] = s;
//     }
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* x = X2 + b*M;
//         double s = 0.0; for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
//         n2[b] = s;
//     }
// 
//     // S = X1 @ X2^T  (N1 x N2)
//     std::vector<double> S(N1 * N2);
//     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//                 (int)N1, (int)N2, (int)M,
//                 1.0, X1, (int)M,
//                      X2, (int)M,
//                 0.0, S.data(), (int)N2);
// 
//     // C, C4
//     std::vector<double> C(N1 * N2), C4(N1 * N2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         for (std::size_t b = 0; b < N2; ++b) {
//             const double sq = n1[a] + n2[b] - 2.0 * S[a*N2 + b];
//             const double k  = std::exp(-0.5 * inv_s2 * sq);
//             C [a*N2 + b] = k * inv_s2;
//             C4[a*N2 + b] = k * inv_s4;
//         }
//     }
// 
//     // 2) Pack J1_hat (N1*D1 x M),  J2_all_cat (M x N2*D2)
//     std::vector<double> J1_hat(big_rows * M);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* J1 = dX1 + (a*M)*D1; // (M x D1)
//         for (std::size_t dj = 0; dj < D1; ++dj) {
//             double* row = J1_hat.data() + (a*D1 + dj) * M;
//             for (std::size_t i = 0; i < M; ++i)
//                 row[i] = J1[i*D1 + dj];
//         }
//     }
// 
//     std::vector<double> J2_all_cat(M * big_cols);
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* J2 = dX2 + (b*M)*D2; // (M x D2)
//         for (std::size_t i = 0; i < M; ++i) {
//             std::memcpy(J2_all_cat.data() + i*big_cols + b*D2,
//                         J2 + i*D2,
//                         D2 * sizeof(double));
//         }
//     }
//     const int lda_J1 = (int)M;
//     const int ldb_J2cat = (int)big_cols;
//     const int ldc_H = (int)big_cols;
// 
//     // 3) Base Gram for ALL blocks once: H_out = J1_hat @ J2_all_cat
//     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                 (int)big_rows, (int)big_cols, (int)M,
//                 1.0, J1_hat.data(), lda_J1,
//                      J2_all_cat.data(), ldb_J2cat,
//                 0.0, H_out, ldc_H);
// 
//     // 4) Projection tables: V1X2_all and V2X1_all
//     //    V1X2_all = J1_hat @ X2^T  (N1*D1 x N2)
//     //    V2X1_all = (J2_all_cat)^T @ X1^T  (N2*D2 x N1) using Trans,Trans (no extra pack)
//     std::vector<double> V1X2_all(big_rows * N2);
//     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//                 (int)big_rows, (int)N2, (int)M,
//                 1.0, J1_hat.data(), (int)M,
//                      X2,             (int)M,
//                 0.0, V1X2_all.data(), (int)N2);
// 
//     std::vector<double> V2X1_all(big_cols * N1);
//     cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
//                 (int)big_cols, (int)N1, (int)M,
//                 1.0, J2_all_cat.data(), (int)big_cols, // A^T is (big_cols x M)
//                      X1,               (int)M,        // B^T is (M x N1)
//                 0.0, V2X1_all.data(), (int)N1);
// 
//     // Optionally free J1_hat to reduce peak (uncomment if needed):
//     // { std::vector<double>().swap(J1_hat); }
// 
//     // 5) Self projections: U1[a]=J1_a^T x1_a, U2[b]=J2_b^T x2_b
//     std::vector<double> U1(N1 * D1), U2(N2 * D2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x1 = X1 + a*M;
//         const double* J1 = dX1 + (a*M)*D1;
//         cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D1,
//                     1.0, J1, (int)D1, x1, 1, 0.0, U1.data() + a*D1, 1);
//     }
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* x2 = X2 + b*M;
//         const double* J2 = dX2 + (b*M)*D2;
//         cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D2,
//                     1.0, J2, (int)D2, x2, 1, 0.0, U2.data() + b*D2, 1);
//     }
// 
//     // 6) Per-block: scale Gram by C[a,b], then rank-1 correction via DGER
//     std::vector<double> v1(D1), v2(D2);
// 
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* U1a = U1.data() + a*D1;
//         const double* V1X2_a = V1X2_all.data() + (a*D1) * N2; // (D1 x N2), row-major
// 
//         for (std::size_t b = 0; b < N2; ++b) {
//             const double cab  = C [a*N2 + b];
//             const double c4ab = C4[a*N2 + b];
// 
//             // Submatrix H(a,b): top-left pointer
//             double* Hblk = H_out + (a*D1) * big_cols + b*D2;
// 
//             // Scale Gram block by cab (row-wise dscal)
//             for (std::size_t i = 0; i < D1; ++i) {
//                 cblas_dscal((int)D2, cab, Hblk + i*big_cols, 1);
//             }
// 
//             // v1 = U1[a] - V1X2_all[aD1:(a+1)D1, b]
//             for (std::size_t i = 0; i < D1; ++i) {
//                 v1[i] = U1a[i] - V1X2_a[i*N2 + b];
//             }
// 
//             // v2 = V2X1_all[bD2:(b+1)D2, a] - U2[b]
//             const double* U2b = U2.data() + b*D2;
//             const double* V2X1_bcol = V2X1_all.data() + (b*D2) * N1 + a; // column 'a' inside row-major
//             for (std::size_t j = 0; j < D2; ++j) {
//                 v2[j] = V2X1_bcol[j*N1] - U2b[j];
//             }
// 
//             // Rank-1 correction: H(a,b) -= c4ab * v1 v2^T
//             cblas_dger(CblasRowMajor,
//                        (int)D1, (int)D2,
//                        -c4ab,
//                        v1.data(), 1,
//                        v2.data(), 1,
//                        Hblk, (int)big_cols);
//         }
//     }
// }
// void rbf_hessian_full_tiled_gemm(
//     const double* X1,  const double* dX1,
//     const double* X2,  const double* dX2,
//     std::size_t N1, std::size_t N2,
//     std::size_t M,  std::size_t D1, std::size_t D2,
//     double sigma,
//     std::size_t tile_B,
//     double* H_out)
// {
//     if (!X1 || !dX1 || !X2 || !dX2 || !H_out) throw std::invalid_argument("null pointer");
//     if (N1==0 || N2==0 || M==0 || D1==0 || D2==0) throw std::invalid_argument("empty dimension");
//     if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be > 0");
// 
//     const std::size_t big_cols = N2 * D2;
//     const double inv_s2 = 1.0 / (sigma * sigma);
//     const double inv_s4 = inv_s2 * inv_s2;
// 
//     // 1) Norms of X1, X2
//     std::vector<double> n1(N1), n2(N2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x = X1 + a*M;
//         double s = 0.0; for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
//         n1[a] = s;
//     }
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* x = X2 + b*M;
//         double s = 0.0; for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
//         n2[b] = s;
//     }
// 
//     // 2) Cross-dots S = X1 @ X2^T  (N1 x N2)
//     std::vector<double> S(N1 * N2);
//     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//                 (int)N1, (int)N2, (int)M,
//                 1.0, X1, (int)M,
//                      X2, (int)M,
//                 0.0, S.data(), (int)N2);
// 
//     // 3) Build C = K/s^2 and C4 = K/s^4
//     std::vector<double> C (N1 * N2), C4(N1 * N2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         for (std::size_t b = 0; b < N2; ++b) {
//             const double sq = n1[a] + n2[b] - 2.0 * S[a*N2 + b];
//             const double k  = std::exp(-0.5 * inv_s2 * sq);
//             C [a*N2 + b] = k * inv_s2;
//             C4[a*N2 + b] = k * inv_s4;
//         }
//     }
// 
//     // 4) U1[a,:] = J1_a^T x1_a,  U2[b,:] = J2_b^T x2_b
//     std::vector<double> U1(N1 * D1), U2(N2 * D2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x1 = X1 + a*M;
//         const double* J1 = dX1 + (a*M)*D1; // (M x D1)
//         cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D1,
//                     1.0, J1, (int)D1,
//                     x1, 1, 0.0, U1.data() + a*D1, 1);
//     }
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* x2 = X2 + b*M;
//         const double* J2 = dX2 + (b*M)*D2; // (M x D2)
//         cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D2,
//                     1.0, J2, (int)D2,
//                     x2, 1, 0.0, U2.data() + b*D2, 1);
//     }
// 
//     // 5) Tile width (refs)
//     if (tile_B == 0) {
//         tile_B = std::max<std::size_t>(1, 32768 / std::max<std::size_t>(1, D2 * sizeof(double)));
//         tile_B = std::min<std::size_t>(tile_B, N2);
//     }
// 
//     // 6) Prepack ALL J2 into (M x (N2*D2)) once → no per-tile packing
//     std::vector<double> J2_all_cat(M * big_cols);
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* J2 = dX2 + (b*M)*D2; // (M x D2)
//         for (std::size_t i = 0; i < M; ++i) {
//             std::memcpy(J2_all_cat.data() + i*big_cols + b*D2,
//                         J2 + i*D2,
//                         D2 * sizeof(double));
//         }
//     }
//     const int ldcJ2 = (int)big_cols; // leading dimension for row-major view
// 
//     // 7) NEW: Precompute V1X2_all = J1_hat @ X2^T
//     // Build J1_hat (N1*D1 x M), where row (a*D1 + dj) = (J1_a[:, dj])^T
//     std::vector<double> J1_hat((size_t)N1 * D1 * M);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* J1 = dX1 + (a*M)*D1; // (M x D1)
//         for (std::size_t dj = 0; dj < D1; ++dj) {
//             double* row = J1_hat.data() + (a*D1 + dj) * M;
//             for (std::size_t i = 0; i < M; ++i) {
//                 row[i] = J1[i*D1 + dj];
//             }
//         }
//     }
//     std::vector<double> V1X2_all((size_t)N1 * D1 * N2); // (N1*D1 x N2)
//     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//                 (int)(N1*D1), (int)N2, (int)M,
//                 1.0, J1_hat.data(), (int)M,
//                      X2,            (int)M,
//                 0.0, V1X2_all.data(), (int)N2);
// 
//     // Workspaces reused per query/tile
//     std::vector<double> G;                 // (D1 x (B*D2))
//     std::vector<double> v2_tile;           // (B*D2)
//     std::vector<double> v1(D1), v2(D2);
//     G.resize(D1 * tile_B * D2);
//     v2_tile.resize(tile_B * D2);
// 
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x1  = X1  + a*M;
//         const double* J1  = dX1 + (a*M)*D1; // (M x D1)
//         const double* U1a = U1.data() + a*D1;
// 
//         // Pointer to the D1×N2 slice in V1X2_all for this a
//         const double* V1X2_a = V1X2_all.data() + (a*D1) * N2;
// 
//         for (std::size_t b0 = 0; b0 < N2; b0 += tile_B) {
//             const std::size_t B = std::min<std::size_t>(tile_B, N2 - b0);
// 
//             // Views into prepacked J2_all_cat tile (M x (B*D2))
//             const double* J2_tile = J2_all_cat.data() + b0*D2;
// 
//             // G = J1_a^T * J2_tile  → (D1 x (B*D2))
//             cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
//                         (int)D1, (int)(B*D2), (int)M,
//                         1.0, J1, (int)D1,
//                              J2_tile, ldcJ2,
//                         0.0, G.data(), (int)(B*D2));
// 
//             // v2_tile = J2_tile^T * x1_a  → (B*D2)
//             cblas_dgemv(CblasRowMajor, CblasTrans,
//                         (int)M, (int)(B*D2),
//                         1.0, J2_tile, ldcJ2,
//                         x1, 1,
//                         0.0, v2_tile.data(), 1);
// 
//             // Finish each block in the tile, writing directly into H_out
//             for (std::size_t t = 0; t < B; ++t) {
//                 const std::size_t b = b0 + t;
//                 const double cab  = C [a*N2 + b];
//                 const double c4ab = C4[a*N2 + b];
// 
//                 // H(a,b) top-left in the big matrix
//                 double* Hdst = H_out + (a*D1) * big_cols + b*D2;
// 
//                 // Copy Gram rows for this t into H_out and scale by cab
//                 for (std::size_t i = 0; i < D1; ++i) {
//                     const double* gsrc = G.data() + i*(B*D2) + t*D2;
//                     double* row = Hdst + i*big_cols; // row-major stride across full width
//                     std::memcpy(row, gsrc, D2 * sizeof(double));
//                     cblas_dscal((int)D2, cab, row, 1);
//                 }
// 
//                 // Rank-1 vectors:
//                 // v1_ab = U1[a,:] - V1X2_a[:, b]
//                 for (std::size_t i = 0; i < D1; ++i)
//                     v1[i] = U1a[i] - V1X2_a[i*N2 + b];
// 
//                 // v2_ab = (J2_tile^T x1_a)[t*D2:(t+1)*D2] - U2[b,:]
//                 const double* U2b = U2.data() + b*D2;
//                 for (std::size_t j = 0; j < D2; ++j)
//                     v2[j] = v2_tile[t*D2 + j] - U2b[j];
// 
//                 // H(a,b) -= c4ab * v1 v2^T   (rank-1 update directly on submatrix)
//                 cblas_dger(CblasRowMajor,
//                            (int)D1, (int)D2,
//                            -c4ab,
//                            v1.data(), 1,
//                            v2.data(), 1,
//                            Hdst, (int)big_cols);
//             }
//         }
//     }
// }


// void rbf_hessian_full_tiled_gemm(
//     const double* X1,  const double* dX1,
//     const double* X2,  const double* dX2,
//     std::size_t N1, std::size_t N2,
//     std::size_t M,  std::size_t D1, std::size_t D2,
//     double sigma,
//     std::size_t tile_B,
//     double* H_out)
// {
//     if (!X1 || !dX1 || !X2 || !dX2 || !H_out) throw std::invalid_argument("null pointer");
//     if (N1==0 || N2==0 || M==0 || D1==0 || D2==0) throw std::invalid_argument("empty dimension");
//     if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be > 0");
// 
//     const std::size_t big_cols = N2 * D2;
//     const double inv_s2 = 1.0 / (sigma * sigma);
//     const double inv_s4 = inv_s2 * inv_s2;
// 
//     // 1) Precompute norms of X1, X2
//     std::vector<double> n1(N1), n2(N2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x = X1 + a*M;
//         double s = 0.0; for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
//         n1[a] = s;
//     }
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* x = X2 + b*M;
//         double s = 0.0; for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
//         n2[b] = s;
//     }
// 
//     // 2) Cross-dots S = X1 @ X2^T  (N1 x N2)
//     std::vector<double> S(N1 * N2);
//     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//                 (int)N1, (int)N2, (int)M,
//                 1.0, X1, (int)M,
//                      X2, (int)M,
//                 0.0, S.data(), (int)N2);
// 
//     // 3) Build C = K/s^2 and C4 = K/s^4
//     std::vector<double> C(N1 * N2), C4(N1 * N2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         for (std::size_t b = 0; b < N2; ++b) {
//             const double sq = n1[a] + n2[b] - 2.0 * S[a*N2 + b];
//             const double k  = std::exp(-0.5 * inv_s2 * sq);
//             C [a*N2 + b] = k * inv_s2;
//             C4[a*N2 + b] = k * inv_s4;
//         }
//     }
// 
//     // 4) Precompute U1[a,:] = J1_a^T x1_a, U2[b,:] = J2_b^T x2_b
//     std::vector<double> U1(N1 * D1), U2(N2 * D2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x1 = X1 + a*M;
//         const double* J1 = dX1 + (a*M)*D1; // (M x D1)
//         cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D1,
//                     1.0, J1, (int)D1,
//                     x1, 1, 0.0, U1.data() + a*D1, 1);
//     }
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* x2 = X2 + b*M;
//         const double* J2 = dX2 + (b*M)*D2; // (M x D2)
//         cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D2,
//                     1.0, J2, (int)D2,
//                     x2, 1, 0.0, U2.data() + b*D2, 1);
//     }
// 
//     // 5) Default tile size
//     if (tile_B == 0) {
//         tile_B = std::max<std::size_t>(1, 32768 / std::max<std::size_t>(1, D2 * sizeof(double)));
//         tile_B = std::min<std::size_t>(tile_B, N2);
//     }
// 
//     // **New**: 6) Prepack ALL J2 into one (M x N2*D2) matrix once → no per-tile packing
//     std::vector<double> J2_all_cat(M * big_cols);
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* J2 = dX2 + (b*M)*D2; // (M x D2)
//         for (std::size_t i = 0; i < M; ++i) {
//             std::memcpy(J2_all_cat.data() + i*big_cols + b*D2,
//                         J2 + i*D2,
//                         D2 * sizeof(double));
//         }
//     }
//     const int ldcJ2 = (int)big_cols; // leading "columns" for row-major view
// 
//     // Workspaces reused per query
//     std::vector<double> V1X2_a(D1 * N2);   // (D1 x N2)
//     std::vector<double> G;                 // (D1 x (B*D2)), allocated once per max tile
//     std::vector<double> v2_tile;           // (B*D2)
//     std::vector<double> v1(D1), v2(D2);
// 
//     G.resize(D1 * tile_B * D2);
//     v2_tile.resize(tile_B * D2);
// 
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x1  = X1  + a*M;
//         const double* J1  = dX1 + (a*M)*D1; // (M x D1)
//         const double* U1a = U1.data() + a*D1;
// 
//         // 6a) V1X2_a = J1_a^T X2^T  (D1 x N2)  [one DGEMM per a]
//         cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
//                     (int)D1, (int)N2, (int)M,
//                     1.0, J1, (int)D1,
//                          X2, (int)M,
//                     0.0, V1X2_a.data(), (int)N2);
// 
//         for (std::size_t b0 = 0; b0 < N2; b0 += tile_B) {
//             const std::size_t B = std::min<std::size_t>(tile_B, N2 - b0);
// 
//             // Views into the prepacked J2_all_cat tile (M x (B*D2))
//             const double* J2_tile = J2_all_cat.data() + b0*D2;
// 
//             // 6b) G = J1_a^T * J2_tile (D1 x (B*D2))  [one DGEMM]
//             cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
//                         (int)D1, (int)(B*D2), (int)M,
//                         1.0, J1, (int)D1,
//                              J2_tile, ldcJ2,
//                         0.0, G.data(), (int)(B*D2));
// 
//             // 6c) v2_tile = J2_tile^T * x1_a (B*D2)   [one DGEMV]
//             cblas_dgemv(CblasRowMajor, CblasTrans,
//                         (int)M, (int)(B*D2),
//                         1.0, J2_tile, ldcJ2,
//                         x1, 1,
//                         0.0, v2_tile.data(), 1);
// 
//             // 6d) Finish each block (write directly into H_out)
//             for (std::size_t t = 0; t < B; ++t) {
//                 const std::size_t b = b0 + t;
//                 const double cab  = C [a*N2 + b];
//                 const double c4ab = C4[a*N2 + b];
// 
//                 // Base pointer of H(a,b) block (top-left)
//                 double* Hdst = H_out + (a*D1) * big_cols + b*D2;
// 
//                 // Copy Gram rows for this t into H_out and scale by cab
//                 for (std::size_t i = 0; i < D1; ++i) {
//                     const double* gsrc = G.data() + i*(B*D2) + t*D2;
//                     double* row = Hdst + i*big_cols;                // row-major stride
//                     std::memcpy(row, gsrc, D2 * sizeof(double));
//                     cblas_dscal((int)D2, cab, row, 1);
//                 }
// 
//                 // Build rank-1 vectors: v1 = U1[a] - V1X2_a[:, b]
//                 for (std::size_t i = 0; i < D1; ++i)
//                     v1[i] = U1a[i] - V1X2_a[i*N2 + b];
// 
//                 // v2 = (J2_tile^T x1)[t*D2:(t+1)*D2] - U2[b]
//                 const double* U2b = U2.data() + b*D2;
//                 for (std::size_t j = 0; j < D2; ++j)
//                     v2[j] = v2_tile[t*D2 + j] - U2b[j];
// 
//                 // Rank-1 correction directly into H_out submatrix:
//                 // Row-major dger on submatrix with lda = big_cols (full row stride)
//                 cblas_dger(CblasRowMajor,
//                            (int)D1, (int)D2,
//                            -c4ab,
//                            v1.data(), 1,
//                            v2.data(), 1,
//                            Hdst, (int)big_cols);
//             }
//         }
//     }
// }



// void rbf_hessian_full_tiled_gemm(
//     const double* X1,  const double* dX1,
//     const double* X2,  const double* dX2,
//     std::size_t N1, std::size_t N2,
//     std::size_t M,  std::size_t D1, std::size_t D2,
//     double sigma,
//     std::size_t tile_B,
//     double* H_out)
// {
//     if (!X1 || !dX1 || !X2 || !dX2 || !H_out) throw std::invalid_argument("null pointer");
//     if (N1==0 || N2==0 || M==0 || D1==0 || D2==0) throw std::invalid_argument("empty dimension");
//     if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be > 0");
// 
//     const std::size_t big_cols = N2 * D2;
//     const double inv_s2 = 1.0 / (sigma * sigma);
//     const double inv_s4 = inv_s2 * inv_s2;
// 
//     // 0) Output zero (optional; we overwrite every block anyway)
//     // std::fill(H_out, H_out + (N1*D1)*(N2*D2), 0.0);
// 
//     // 1) Precompute norms of X1 and X2
//     std::vector<double> n1(N1), n2(N2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x = X1 + a*M;
//         double s = 0.0; for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
//         n1[a] = s;
//     }
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* x = X2 + b*M;
//         double s = 0.0; for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
//         n2[b] = s;
//     }
// 
//     // 2) Cross-dots S = X1 @ X2^T  (N1 x N2)
//     std::vector<double> S(N1 * N2);
//     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
//                 (int)N1, (int)N2, (int)M,
//                 1.0, X1, (int)M,
//                      X2, (int)M,
//                 0.0, S.data(), (int)N2);
// 
//     // 3) Build C = K/s^2 and C4 = K/s^4 from distances
//     std::vector<double> C(N1 * N2), C4(N1 * N2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         for (std::size_t b = 0; b < N2; ++b) {
//             const double sq = n1[a] + n2[b] - 2.0 * S[a*N2 + b];
//             const double k  = std::exp(-0.5 * inv_s2 * sq);
//             C [a*N2 + b] = k * inv_s2;
//             C4[a*N2 + b] = k * inv_s4;
//         }
//     }
// 
//     // 4) Precompute U1[a,:] = J1_a^T x1_a, U2[b,:] = J2_b^T x2_b
//     std::vector<double> U1(N1 * D1), U2(N2 * D2);
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x1 = X1 + a*M;
//         const double* J1 = dX1 + (a*M)*D1; // (M x D1)
//         double* u1 = U1.data() + a*D1;
//         cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D1,
//                     1.0, J1, (int)D1,
//                     x1, 1, 0.0, u1, 1);
//     }
//     for (std::size_t b = 0; b < N2; ++b) {
//         const double* x2 = X2 + b*M;
//         const double* J2 = dX2 + (b*M)*D2; // (M x D2)
//         double* u2 = U2.data() + b*D2;
//         cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D2,
//                     1.0, J2, (int)D2,
//                     x2, 1, 0.0, u2, 1);
//     }
// 
//     // 5) Choose tile size for references
//     if (tile_B == 0) {
//         // heuristic: keep a few rows of G in L2; tweak as needed
//         tile_B = std::max<std::size_t>(1, 32768 / std::max<std::size_t>(1, D2 * sizeof(double)));
//         tile_B = std::min<std::size_t>(tile_B, N2);
//     }
// 
//     // Workspace reused per query a
//     std::vector<double> V1X2_a(D1 * N2);             // J1_a^T X2^T  (D1 x N2)
//     std::vector<double> J2_cat;                      // (M x (Bt*D2))
//     std::vector<double> G;                           // (D1 x (Bt*D2))
//     std::vector<double> v2_tile;                     // (Bt*D2)
//     std::vector<double> Hblk(D1 * D2);               // one output block
//     std::vector<double> v1(D1), v2(D2);              // rank-1 vectors
// 
//     for (std::size_t a = 0; a < N1; ++a) {
//         const double* x1 = X1  + a*M;
//         const double* J1 = dX1 + (a*M)*D1; // (M x D1)
//         const double* U1a = U1.data() + a*D1;
// 
//         // 5a) V1X2_a = J1_a^T X2^T  (D1 x N2)  [one DGEMM per a]
//         cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
//                     (int)D1, (int)N2, (int)M,
//                     1.0, J1, (int)D1,
//                          X2, (int)M,
//                     0.0, V1X2_a.data(), (int)N2);
// 
//         for (std::size_t b0 = 0; b0 < N2; b0 += tile_B) {
//             const std::size_t B = std::min<std::size_t>(tile_B, N2 - b0);
// 
//             // 5b) Pack J2_cat = [J2_b0 | J2_b0+1 | ...]  (M x (B*D2))
//             J2_cat.assign(M * B * D2, 0.0);
//             for (std::size_t t = 0; t < B; ++t) {
//                 const std::size_t b = b0 + t;
//                 const double* J2 = dX2 + (b*M)*D2; // (M x D2)
//                 for (std::size_t i = 0; i < M; ++i) {
//                     // row i: copy D2 doubles
//                     std::memcpy(J2_cat.data() + i*(B*D2) + t*D2,
//                                 J2 + i*D2,
//                                 D2 * sizeof(double));
//                 }
//             }
// 
//             // 5c) G = J1_a^T * J2_cat   → (D1 x (B*D2))   [one DGEMM per tile]
//             G.assign(D1 * B * D2, 0.0);
//             cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
//                         (int)D1, (int)(B*D2), (int)M,
//                         1.0, J1, (int)D1,
//                              J2_cat.data(), (int)(B*D2),
//                         0.0, G.data(), (int)(B*D2));
// 
//             // 5d) v2_tile = J2_cat^T * x1_a   → (B*D2)     [one DGEMV per tile]
//             v2_tile.assign(B * D2, 0.0);
//             cblas_dgemv(CblasRowMajor, CblasTrans,
//                         (int)M, (int)(B*D2),
//                         1.0, J2_cat.data(), (int)(B*D2),
//                         x1, 1,
//                         0.0, v2_tile.data(), 1);
// 
//             // 5e) Finish each block in the tile
//             for (std::size_t t = 0; t < B; ++t) {
//                 const std::size_t b = b0 + t;
// 
//                 // -- start H_ab from Gram block: Hblk = C[a,b] * G[:, t-block]
//                 const double cab = C[a*N2 + b];
//                 for (std::size_t i = 0; i < D1; ++i) {
//                     const double* gsrc = G.data() + i*(B*D2) + t*D2;
//                     double*       hdst = Hblk.data() + i*D2;
//                     // copy row and scale by cab
//                     for (std::size_t j = 0; j < D2; ++j) hdst[j] = cab * gsrc[j];
//                 }
// 
//                 // -- build rank-1 vectors
//                 // v1_ab = U1[a,:] - V1X2_a[:, b]
//                 for (std::size_t i = 0; i < D1; ++i) {
//                     v1[i] = U1a[i] - V1X2_a[i*N2 + b];
//                 }
//                 // v2_ab = (J2_cat^T x1_a)[t*D2:(t+1)*D2] - U2[b,:]
//                 const double* U2b = U2.data() + b*D2;
//                 for (std::size_t j = 0; j < D2; ++j) {
//                     v2[j] = v2_tile[t*D2 + j] - U2b[j];
//                 }
// 
//                 // -- rank-1 correction: Hblk -= C4[a,b] * (v1 v2^T)
//                 const double c4ab = C4[a*N2 + b];
//                 cblas_dger(CblasRowMajor,
//                            (int)D1, (int)D2,
//                            -c4ab,
//                            v1.data(), 1,
//                            v2.data(), 1,
//                            Hblk.data(), (int)D2);
// 
//                 // -- scatter Hblk into big output at (a,b)
//                 for (std::size_t i = 0; i < D1; ++i) {
//                     double* dst = H_out + (a*D1 + i)*big_cols + b*D2;
//                     const double* src = Hblk.data() + i*D2;
//                     std::memcpy(dst, src, D2 * sizeof(double));
//                 }
//             } // tile t
//         } // tiles of b
//     } // for a
// }

// Symmetric (training) RBF Hessian kernel, Gaussian.
// X: (N, M) row-major          – descriptor vectors
// dX: (N, M, D) row-major      – Jacobians wrt descriptor coords (M x D) per sample
// Output H_out: (N*D, N*D) row-major, symmetric
void rbf_hessian_full_tiled_gemm_sym_fast(
// Symmetric (training) RBF Hessian, Gaussian kernel.
// Assumes: X2==X1 (same set), dX2==dX1, N1==N2, D1==D2.
// Builds only the lower triangle and mirrors to upper.
// void rbf_hessian_full_tiled_gemm_symmetric(
    const double* __restrict X1,  const double* __restrict dX1,
    std::size_t N1, 
    std::size_t M,  std::size_t D1,
    double sigma,
    std::size_t tile_B,
    double* __restrict H_out)
{
    if (!X1 || !dX1 || !H_out) throw std::invalid_argument("null pointer");
    if (N1==0 || M==0 || D1==0 ) throw std::invalid_argument("empty dimension");
    if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be > 0");

    const std::size_t N   = N1;
    const std::size_t D   = D1;
    const std::size_t BIG = N * D;

    const double inv_s2 = 1.0 / (sigma * sigma);
    const double inv_s4 = inv_s2 * inv_s2;

    // Zero output once (we'll fill lower and mirror to upper at the end)
    // std::fill(H_out, H_out + BIG*BIG, 0.0);

    // 1) Distances → C = K/s^2 and C4 = K/s^4
    std::vector<double> n(N);
    #pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double* __restrict x = X1 + a*M;
        double s = 0.0;
        #pragma omp simd reduction(+:s)
        for (std::size_t i = 0; i < M; ++i) s += x[i]*x[i];
        n[a] = s;
    }

    // S = X1 @ X1^T (lower) via DSYRK, then mirror to upper for convenient reads
    std::vector<double> S(N * N, 0.0);
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                (int)N, (int)M,
                1.0, X1, (int)M,
                0.0, S.data(), (int)N);
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i+1; j < N; ++j)
            S[i*N + j] = S[j*N + i];

    std::vector<double> C(N * N), C4(N * N);
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            const double sq = n[a] + n[b] - 2.0 * S[a*N + b];
            const double k  = std::exp(-0.5 * inv_s2 * sq);
            C [a*N + b] = k * inv_s2;
            C4[a*N + b] = k * inv_s4;
        }
    }

    // 2) Pack J_hat (N*D x M) from dX1 (N,M,D): row i=(a*D+dj) is column dj of (M x D)
    std::vector<double> J_hat(BIG * M);
    #pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double* __restrict J = dX1 + (a*M)*D; // (M x D)
        for (std::size_t dj = 0; dj < D; ++dj) {
            double* __restrict row = J_hat.data() + (a*D + dj) * M;
            #pragma omp simd
            for (std::size_t i = 0; i < M; ++i)
                row[i] = J[i*D + dj];
        }
    }

    // 3) Base Gram only lower: H_out := J_hat @ J_hat^T  via DSYRK (lower)
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                (int)BIG, (int)M,
                1.0, J_hat.data(), (int)M,
                0.0, H_out, (int)BIG);

    // 4) Projection table V = J_hat @ X1^T  (BIG x N)
    std::vector<double> V(BIG * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)BIG, (int)N, (int)M,
                1.0, J_hat.data(), (int)M,
                     X1,            (int)M,
                0.0, V.data(),     (int)N);

    // 5) Self projections: U[a] = J_a^T x_a  (N x D)
    std::vector<double> U(N * D);
    #pragma omp parallel for schedule(static)
    for (std::size_t a = 0; a < N; ++a) {
        const double* __restrict x  = X1 + a*M;
        const double* __restrict Ja = dX1 + (a*M)*D;  // (M x D)
        double* __restrict Ua       = U.data() + a*D;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)M, (int)D,
                    1.0, Ja, (int)D, x, 1, 0.0, Ua, 1);
    }

    // 6) Per-block (lower triangle only): scale Gram by C[a,b], then rank-1 correction via GER
    if (tile_B == 0) tile_B = std::min<std::size_t>(64, N);  // sane default tile width

    #pragma omp parallel
    {
        std::vector<double> v1(D), v2(D);  // private buffers

        #pragma omp for schedule(static)
        for (std::size_t a = 0; a < N; ++a) {

            const double* __restrict Ua   = U.data() + a*D;
            const double* __restrict Va   = V.data() + (a*D) * N; // (D x N), row-major

            for (std::size_t b0 = 0; b0 < a+1; b0 += tile_B) {
                const std::size_t bend = std::min<std::size_t>(a+1, b0 + tile_B);

                for (std::size_t b = b0; b < bend; ++b) {
                    const double cab  = C [a*N + b];
                    const double c4ab = C4[a*N + b];

                    // H(a,b) submatrix top-left
                    double* __restrict Hblk = H_out + (a*D) * BIG + b*D;

                    // Scale Gram block by cab (row by row)
                    for (std::size_t i = 0; i < D; ++i) {
                        cblas_dscal((int)D, cab, Hblk + i*BIG, 1);
                    }

                    // v1 = U[a] - V[aD:(a+1)D, b]
                    #pragma omp simd
                    for (std::size_t i = 0; i < D; ++i) {
                        v1[i] = Ua[i] - Va[i*N + b];
                    }

                    // v2 = V[bD:(b+1)D, a] - U[b]
                    const double* __restrict Ub = U.data() + b*D;
                    const double* __restrict Vb = V.data() + (b*D) * N;
                    #pragma omp simd
                    for (std::size_t j = 0; j < D; ++j) {
                        v2[j] = Vb[j*N + a] - Ub[j];
                    }

                    // Rank-1 correction: H(a,b) -= c4ab * v1 v2^T
                    cblas_dger(CblasRowMajor,
                               (int)D, (int)D,
                               -c4ab,
                               v1.data(), 1,
                               v2.data(), 1,
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
}

