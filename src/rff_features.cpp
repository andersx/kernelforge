// C++ standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

// Third-party libraries
#include <omp.h>

// Project headers
#include "aligned_alloc64.hpp"
#include "blas_config.h"
#include "rff_features.hpp"
#include "rfp_utils.hpp"

namespace kf::rff {

void rff_features(const double *X, const double *W, const double *b,
                  std::size_t N, std::size_t rep_size, std::size_t D,
                  double *Z) {
    if (!X || !W || !b || !Z)
        throw std::invalid_argument("rff_features: null pointer");
    if (N == 0 || rep_size == 0 || D == 0)
        throw std::invalid_argument("rff_features: zero dimension");

    // Z = X @ W  via DGEMM
    // X is (N, rep_size), W is (rep_size, D), Z is (N, D)
    // RowMajor: M=N, N=D, K=rep_size
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(N),         // M
                static_cast<int>(D),         // N
                static_cast<int>(rep_size),  // K
                1.0, X, static_cast<int>(rep_size),  // A, lda
                W, static_cast<int>(D),              // B, ldb
                0.0, Z, static_cast<int>(D));        // C, ldc

    // Z[:, d] = cos(Z[:, d] + b[d]) * sqrt(2/D)
    const double normalization = std::sqrt(2.0 / static_cast<double>(D));

    for (std::size_t i = 0; i < N; ++i) {
        #pragma omp simd
        for (std::size_t d = 0; d < D; ++d) {
            Z[i * D + d] = std::cos(Z[i * D + d] + b[d]) * normalization;
        }
    }
}

void rff_gradient(const double *X, const double *dX,
                  const double *W, const double *b,
                  std::size_t N, std::size_t rep_size,
                  std::size_t D, std::size_t ncoords,
                  double *G) {
    if (!X || !dX || !W || !b || !G)
        throw std::invalid_argument("rff_gradient: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_gradient: zero dimension");

    const double normalization = -std::sqrt(2.0 / static_cast<double>(D));
    const std::size_t total_grads = N * ncoords;  // total columns of G

    // Zero output G (D, N*ncoords) row-major
    std::memset(G, 0, D * total_grads * sizeof(double));

    // Parallelize over molecules. Each thread writes to its own non-overlapping
    // column block of G (molecule i → columns [i*ncoords, (i+1)*ncoords)), so
    // there are no race conditions.
    // Note: BLAS is called inside the OMP region on per-molecule sub-problems.
    // Set MKL_NUM_THREADS=1 or OPENBLAS_NUM_THREADS=1 to avoid nested threading.
    #pragma omp parallel
    {
        std::vector<double> z_i(D);
        double *dg = aligned_alloc_64(D * rep_size);

        #pragma omp for schedule(dynamic)
        for (std::size_t i = 0; i < N; ++i) {
            // z_i = b + W^T @ X[i]   (DGEMV: W is rep_size×D → W^T is D×rep_size)
            std::memcpy(z_i.data(), b, D * sizeof(double));
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        static_cast<int>(rep_size), static_cast<int>(D),
                        1.0, W, static_cast<int>(D),
                        X + i * rep_size, 1,
                        1.0, z_i.data(), 1);

            // dg[d, r] = sin(z_i[d]) * normalization * W[r, d]
            for (std::size_t d = 0; d < D; ++d) {
                const double factor = std::sin(z_i[d]) * normalization;
                for (std::size_t r = 0; r < rep_size; ++r) {
                    dg[d * rep_size + r] = factor * W[r * D + d];
                }
            }

            // G[:, i*ncoords:(i+1)*ncoords] = dg @ dX[i]
            // dg: (D, rep_size), dX[i]: (rep_size, ncoords) → result: (D, ncoords)
            // G is (D, total_grads) row-major; ldc = total_grads, base = G + i*ncoords
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(D),
                        static_cast<int>(ncoords),
                        static_cast<int>(rep_size),
                        1.0,
                        dg,                           // A: (D, rep_size)
                        static_cast<int>(rep_size),
                        dX + i * rep_size * ncoords,  // B: dX[i] (rep_size, ncoords)
                        static_cast<int>(ncoords),
                        0.0,                          // beta: overwrite (G already zeroed)
                        G + i * ncoords,              // C col offset i*ncoords in G
                        static_cast<int>(total_grads));
        }

        aligned_free_64(dg);
    }  // end omp parallel
}

void rff_gramian_symm(const double *X, const double *W, const double *b,
                      const double *Y,
                      std::size_t N, std::size_t rep_size, std::size_t D,
                      std::size_t chunk_size,
                      double *ZtZ, double *ZtY) {
    if (!X || !W || !b || !Y || !ZtZ || !ZtY)
        throw std::invalid_argument("rff_gramian_symm: null pointer");
    if (N == 0 || rep_size == 0 || D == 0)
        throw std::invalid_argument("rff_gramian_symm: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gramian_symm: zero chunk_size");

    std::memset(ZtZ, 0, D * D * sizeof(double));
    std::memset(ZtY, 0, D * sizeof(double));

    const std::size_t nchunks = (N + chunk_size - 1) / chunk_size;

    // Parallelize over chunks using thread-local accumulators, then reduce.
    // rff_features calls BLAS (not OpenMP), so there is no nested parallelism.
    // Set MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1 to avoid BLAS thread contention.
    #pragma omp parallel
    {
        std::vector<double> loc_ZtZ(D * D, 0.0);
        std::vector<double> loc_ZtY(D, 0.0);

        #pragma omp for schedule(dynamic)
        for (std::size_t ci = 0; ci < nchunks; ++ci) {
            const std::size_t cs = ci * chunk_size;
            const std::size_t ce = std::min(cs + chunk_size, N);
            const std::size_t nc = ce - cs;

            double *Z = aligned_alloc_64(nc * D);

            rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);

            // loc_ZtZ += Z^T @ Z  (upper triangle via DSYRK)
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        static_cast<int>(D), static_cast<int>(nc),
                        1.0, Z, static_cast<int>(D),
                        1.0, loc_ZtZ.data(), static_cast<int>(D));

            // loc_ZtY += Z^T @ Y_chunk
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        static_cast<int>(nc), static_cast<int>(D),
                        1.0, Z, static_cast<int>(D),
                        Y + cs, 1,
                        1.0, loc_ZtY.data(), 1);

            aligned_free_64(Z);
        }

        // Reduce thread-local accumulators into global output
        #pragma omp critical
        {
            for (std::size_t k = 0; k < D * D; ++k) ZtZ[k] += loc_ZtZ[k];
            for (std::size_t k = 0; k < D;     ++k) ZtY[k] += loc_ZtY[k];
        }
    }

    // Symmetrize: copy upper triangle to lower
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            ZtZ[j * D + i] = ZtZ[i * D + j];
        }
    }
}

void rff_full_gramian_symm(const double *X, const double *dX,
                           const double *W, const double *b,
                           const double *Y, const double *F,
                           std::size_t N, std::size_t rep_size, std::size_t D,
                           std::size_t ncoords,
                           std::size_t energy_chunk, std::size_t force_chunk,
                           double *ZtZ, double *ZtY) {
    if (!X || !dX || !W || !b || !Y || !F || !ZtZ || !ZtY)
        throw std::invalid_argument("rff_full_gramian_symm: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_full_gramian_symm: zero dimension");
    if (energy_chunk == 0 || force_chunk == 0)
        throw std::invalid_argument("rff_full_gramian_symm: zero chunk size");

    std::memset(ZtZ, 0, D * D * sizeof(double));
    std::memset(ZtY, 0, D * sizeof(double));

    // ---- Energy loop (chunked, parallelized) ----
    // rff_features calls BLAS (not OpenMP) → safe to parallelize here.
    {
        const std::size_t nchunks = (N + energy_chunk - 1) / energy_chunk;

        #pragma omp parallel
        {
            std::vector<double> loc_ZtZ(D * D, 0.0);
            std::vector<double> loc_ZtY(D, 0.0);

            #pragma omp for schedule(dynamic)
            for (std::size_t ci = 0; ci < nchunks; ++ci) {
                const std::size_t cs = ci * energy_chunk;
                const std::size_t ce = std::min(cs + energy_chunk, N);
                const std::size_t nc = ce - cs;

                double *Z = aligned_alloc_64(nc * D);

                rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);

                cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            static_cast<int>(D), static_cast<int>(nc),
                            1.0, Z, static_cast<int>(D),
                            1.0, loc_ZtZ.data(), static_cast<int>(D));

                cblas_dgemv(CblasRowMajor, CblasTrans,
                            static_cast<int>(nc), static_cast<int>(D),
                            1.0, Z, static_cast<int>(D),
                            Y + cs, 1,
                            1.0, loc_ZtY.data(), 1);

                aligned_free_64(Z);
            }

            #pragma omp critical
            {
                for (std::size_t k = 0; k < D * D; ++k) ZtZ[k] += loc_ZtZ[k];
                for (std::size_t k = 0; k < D;     ++k) ZtY[k] += loc_ZtY[k];
            }
        }
    }

    // ---- Force loop (chunked, sequential outer loop) ----
    // rff_gradient uses OpenMP internally; keep this loop sequential
    // to avoid nested parallelism.
    for (std::size_t cs = 0; cs < N; cs += force_chunk) {
        const std::size_t ce = std::min(cs + force_chunk, N);
        const std::size_t nc = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        double *G = aligned_alloc_64(D * ngrads_chunk);

        rff_gradient(X  + cs * rep_size,
                     dX + cs * rep_size * ncoords,
                     W, b, nc, rep_size, D, ncoords, G);

        // ZtZ += G @ G^T  (upper triangle, G is D×ngrads_chunk)
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    1.0, ZtZ, static_cast<int>(D));

        // ZtY += G @ F_chunk
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + cs * ncoords, 1,
                    1.0, ZtY, 1);

        aligned_free_64(G);
    }

    // Symmetrize: copy upper triangle to lower
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            ZtZ[j * D + i] = ZtZ[i * D + j];
        }
    }
}

void rff_full(const double *X, const double *dX,
              const double *W, const double *b,
              std::size_t N, std::size_t rep_size,
              std::size_t D, std::size_t ncoords,
              double *Z_full) {
    if (!X || !dX || !W || !b || !Z_full)
        throw std::invalid_argument("rff_full: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_full: zero dimension");

    const std::size_t total_grads = N * ncoords;

    // Top half: energy features Z[0:N, :]
    rff_features(X, W, b, N, rep_size, D, Z_full);

    // Bottom half: G^T where G is (D, total_grads)
    double *G = aligned_alloc_64(D * total_grads);
    rff_gradient(X, dX, W, b, N, rep_size, D, ncoords, G);

    // Transpose G (D, total_grads) -> Z_full[N:] (total_grads, D)
    // Z_full[(N + g) * D + d] = G[d * total_grads + g]
    #pragma omp parallel for schedule(static)
    for (std::size_t g = 0; g < total_grads; ++g) {
        double *row = Z_full + (N + g) * D;
        for (std::size_t d = 0; d < D; ++d) {
            row[d] = G[d * total_grads + g];
        }
    }

    aligned_free_64(G);
}

void rff_gradient_gramian_symm(const double *X, const double *dX,
                               const double *W, const double *b,
                               const double *F,
                               std::size_t N, std::size_t rep_size,
                               std::size_t D, std::size_t ncoords,
                               std::size_t chunk_size,
                               double *GtG, double *GtF) {
    if (!X || !dX || !W || !b || !F || !GtG || !GtF)
        throw std::invalid_argument("rff_gradient_gramian_symm: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_gradient_gramian_symm: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gradient_gramian_symm: zero chunk_size");

    std::memset(GtG, 0, D * D * sizeof(double));
    std::memset(GtF, 0, D * sizeof(double));

    // rff_gradient uses OMP internally; keep outer loop sequential
    for (std::size_t cs = 0; cs < N; cs += chunk_size) {
        const std::size_t ce = std::min(cs + chunk_size, N);
        const std::size_t nc = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        double *G = aligned_alloc_64(D * ngrads_chunk);

        rff_gradient(X  + cs * rep_size,
                     dX + cs * rep_size * ncoords,
                     W, b, nc, rep_size, D, ncoords, G);

        // GtG += G @ G^T (upper triangle, G is D×ngrads_chunk)
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    1.0, GtG, static_cast<int>(D));

        // GtF += G @ F_chunk
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + cs * ncoords, 1,
                    1.0, GtF, 1);

        aligned_free_64(G);
    }

    // Symmetrize: copy upper triangle to lower
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            GtG[j * D + i] = GtG[i * D + j];
        }
    }
}

void rff_gramian_symm_rfp(const double *X, const double *W, const double *b,
                          const double *Y,
                          std::size_t N, std::size_t rep_size, std::size_t D,
                          std::size_t chunk_size,
                          double *ZtZ_rfp, double *ZtY) {
    if (!X || !W || !b || !Y || !ZtZ_rfp || !ZtY)
        throw std::invalid_argument("rff_gramian_symm_rfp: null pointer");
    if (N == 0 || rep_size == 0 || D == 0)
        throw std::invalid_argument("rff_gramian_symm_rfp: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gramian_symm_rfp: zero chunk_size");

    // Compute full symmetric ZtZ, then pack upper triangle into RFP
    double *ZtZ_full = aligned_alloc_64(D * D);
    rff_gramian_symm(X, W, b, Y, N, rep_size, D, chunk_size, ZtZ_full, ZtY);

    const std::size_t nt = D * (D + 1) / 2;
    std::memset(ZtZ_rfp, 0, nt * sizeof(double));
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i; j < D; ++j) {
            ZtZ_rfp[kf::rfp_index_upper_N(D, i, j)] = ZtZ_full[i * D + j];
        }
    }

    aligned_free_64(ZtZ_full);
}

void rff_gradient_gramian_symm_rfp(const double *X, const double *dX,
                                   const double *W, const double *b,
                                   const double *F,
                                   std::size_t N, std::size_t rep_size,
                                   std::size_t D, std::size_t ncoords,
                                   std::size_t chunk_size,
                                   double *GtG_rfp, double *GtF) {
    if (!X || !dX || !W || !b || !F || !GtG_rfp || !GtF)
        throw std::invalid_argument("rff_gradient_gramian_symm_rfp: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_gradient_gramian_symm_rfp: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gradient_gramian_symm_rfp: zero chunk_size");

    // Compute full symmetric GtG, then pack upper triangle into RFP
    double *GtG_full = aligned_alloc_64(D * D);
    rff_gradient_gramian_symm(X, dX, W, b, F, N, rep_size, D, ncoords, chunk_size,
                              GtG_full, GtF);

    const std::size_t nt = D * (D + 1) / 2;
    std::memset(GtG_rfp, 0, nt * sizeof(double));
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i; j < D; ++j) {
            GtG_rfp[kf::rfp_index_upper_N(D, i, j)] = GtG_full[i * D + j];
        }
    }

    aligned_free_64(GtG_full);
}

void rff_full_gramian_symm_rfp(const double *X, const double *dX,
                               const double *W, const double *b,
                               const double *Y, const double *F,
                               std::size_t N, std::size_t rep_size, std::size_t D,
                               std::size_t ncoords,
                               std::size_t energy_chunk, std::size_t force_chunk,
                               double *ZtZ_rfp, double *ZtY) {
    if (!X || !dX || !W || !b || !Y || !F || !ZtZ_rfp || !ZtY)
        throw std::invalid_argument("rff_full_gramian_symm_rfp: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_full_gramian_symm_rfp: zero dimension");
    if (energy_chunk == 0 || force_chunk == 0)
        throw std::invalid_argument("rff_full_gramian_symm_rfp: zero chunk size");

    // Compute full symmetric ZtZ, then pack upper triangle into RFP
    double *ZtZ_full = aligned_alloc_64(D * D);
    rff_full_gramian_symm(X, dX, W, b, Y, F, N, rep_size, D, ncoords,
                          energy_chunk, force_chunk, ZtZ_full, ZtY);

    const std::size_t nt = D * (D + 1) / 2;
    std::memset(ZtZ_rfp, 0, nt * sizeof(double));
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i; j < D; ++j) {
            ZtZ_rfp[kf::rfp_index_upper_N(D, i, j)] = ZtZ_full[i * D + j];
        }
    }

    aligned_free_64(ZtZ_full);
}

}  // namespace kf::rff
