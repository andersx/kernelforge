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

    // Outer loop parallelized: when called standalone this uses all OMP threads;
    // when called from inside an OMP parallel region (e.g. gramian functions),
    // nested parallelism is disabled by default so it serializes — same as before.
    #pragma omp parallel for schedule(static)
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

    // Parallelize over molecules. Each thread writes to its own non-overlapping
    // column block of G (molecule i → columns [i*ncoords, (i+1)*ncoords)).
    //
    // Per-molecule computation:
    //   G[:, i*nc:(i+1)*nc] = diag(sin(z_i) * norm) @ W^T @ dX[i]
    // Split as:
    //   1. DGEMM: G_chunk = W^T @ dX[i]   (beta=0, initializes the chunk)
    //   2. Scale row d in-place by sin(z_i[d]) * norm
    //
    // This avoids constructing the intermediate dg (D×rep_size) matrix whose
    // dg[d,r] = sin(z_i[d])*norm*W[r,d] loop reads W with stride-D (column-major
    // access into a row-major array). BLAS DGEMM with CblasTrans handles the
    // W transposition via optimised blocked algorithms.
    //
    // No initial memset needed: every element of G is initialised by the
    // per-molecule DGEMM with beta=0.
    //
    // Serialise BLAS inside the OMP region to prevent thread oversubscription.
    // MKL does this automatically; OpenBLAS requires an explicit API call.
    const int blas_nt = kf_blas_get_num_threads();
    kf_blas_set_num_threads(1);
    #pragma omp parallel
    {
        std::vector<double> z_i(D);

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < N; ++i) {
            // z_i = b + W^T @ X[i]
            std::memcpy(z_i.data(), b, D * sizeof(double));
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        static_cast<int>(rep_size), static_cast<int>(D),
                        1.0, W, static_cast<int>(D),
                        X + i * rep_size, 1,
                        1.0, z_i.data(), 1);

            // G[:, i*ncoords:(i+1)*ncoords] = W^T @ dX[i]
            // W is (rep_size, D) row-major → W^T is (D, rep_size); lda = D.
            // dX[i] is (rep_size, ncoords) row-major.
            // Result (D, ncoords) written at column offset i*ncoords; ldc = total_grads.
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        static_cast<int>(D),
                        static_cast<int>(ncoords),
                        static_cast<int>(rep_size),
                        1.0,
                        W,                            // A: (rep_size, D), lda = D
                        static_cast<int>(D),
                        dX + i * rep_size * ncoords,  // B: dX[i] (rep_size, ncoords)
                        static_cast<int>(ncoords),
                        0.0,                          // beta = 0: initialise chunk
                        G + i * ncoords,              // C col offset i*ncoords in G
                        static_cast<int>(total_grads));

            // Scale row d in-place by sin(z_i[d]) * normalization
            for (std::size_t d = 0; d < D; ++d) {
                const double factor = std::sin(z_i[d]) * normalization;
                double *row = G + d * total_grads + i * ncoords;
                #pragma omp simd
                for (std::size_t c = 0; c < ncoords; ++c) {
                    row[c] *= factor;
                }
            }
        }
    }  // end omp parallel
    kf_blas_set_num_threads(blas_nt);
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

    const std::size_t nchunks = (N + chunk_size - 1) / chunk_size;

    // Accumulate directly into ZtZ using DSYRK (no thread-local buffers).
    // BLAS itself is multi-threaded; large chunks keep utilisation high.
    for (std::size_t ci = 0; ci < nchunks; ++ci) {
        const std::size_t cs = ci * chunk_size;
        const std::size_t ce = std::min(cs + chunk_size, N);
        const std::size_t nc = ce - cs;

        double *Z = aligned_alloc_64(nc * D);

        // zero accum on first chunk, then sum
        double clear_or_sum = (ci == 0) ? 0.0 : 1.0;

        rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);

        // ZtZ += Z^T @ Z  (upper triangle via DSYRK)
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    static_cast<int>(D), static_cast<int>(nc),
                    1.0, Z, static_cast<int>(D),
                    clear_or_sum, ZtZ, static_cast<int>(D));

        // ZtY += Z^T @ Y_chunk
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    static_cast<int>(nc), static_cast<int>(D),
                    1.0, Z, static_cast<int>(D),
                    Y + cs, 1,
                    clear_or_sum, ZtY, 1);

        aligned_free_64(Z);
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

    // ---- Energy loop (chunked, sequential) ----
    // Accumulate directly into ZtZ/ZtY; clear_or_sum avoids memset.
    {
        const std::size_t nchunks = (N + energy_chunk - 1) / energy_chunk;

        for (std::size_t ci = 0; ci < nchunks; ++ci) {
            const std::size_t cs = ci * energy_chunk;
            const std::size_t ce = std::min(cs + energy_chunk, N);
            const std::size_t nc = ce - cs;

            double *Z = aligned_alloc_64(nc * D);

            // zero accum on first chunk, then sum
            double clear_or_sum = (ci == 0) ? 0.0 : 1.0;

            rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);

            // ZtZ += Z^T @ Z  (upper triangle via DSYRK)
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        static_cast<int>(D), static_cast<int>(nc),
                        1.0, Z, static_cast<int>(D),
                        clear_or_sum, ZtZ, static_cast<int>(D));

            // ZtY += Z^T @ Y_chunk
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        static_cast<int>(nc), static_cast<int>(D),
                        1.0, Z, static_cast<int>(D),
                        Y + cs, 1,
                        clear_or_sum, ZtY, 1);

            aligned_free_64(Z);
        }
    }

    // ---- Force loop (chunked, sequential outer loop) ----
    // rff_gradient uses OpenMP internally; keep this loop sequential
    // to avoid nested parallelism. Energy loop already ran so beta=1.0 always.
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

    // rff_gradient uses OMP internally; keep outer loop sequential.
    // Accumulate directly into GtG/GtF; clear_or_sum avoids memset.
    std::size_t chunk_idx = 0;
    for (std::size_t cs = 0; cs < N; cs += chunk_size, ++chunk_idx) {
        const std::size_t ce = std::min(cs + chunk_size, N);
        const std::size_t nc = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        double *G = aligned_alloc_64(D * ngrads_chunk);

        // zero accum on first chunk, then sum
        double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

        rff_gradient(X  + cs * rep_size,
                     dX + cs * rep_size * ncoords,
                     W, b, nc, rep_size, D, ncoords, G);

        // GtG += G @ G^T (upper triangle, G is D×ngrads_chunk)
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    clear_or_sum, GtG, static_cast<int>(D));

        // GtF += G @ F_chunk
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + cs * ncoords, 1,
                    clear_or_sum, GtF, 1);

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

    const std::size_t nt      = D * (D + 1) / 2;
    const std::size_t nchunks = (N + chunk_size - 1) / chunk_size;

    // // Accumulate directly into RFP using DSFRK
    for (std::size_t ci = 0; ci < nchunks; ++ci) {
        const std::size_t cs = ci * chunk_size;
        const std::size_t ce = std::min(cs + chunk_size, N);
        const std::size_t nc = ce - cs;

        double *Z = aligned_alloc_64(nc * D);

        // zero accum on first chunk, then sum
        double clear_or_sum = (ci == 0) ? 0.0 : 1.0;

        rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);

        // ZtZ_rfp += Z_chunk^T @ Z_chunk
        kf_dsfrk('N', 'U', 'N',
                 static_cast<blas_int>(D), static_cast<blas_int>(nc),
                 1.0, Z, static_cast<blas_int>(D),
                 clear_or_sum, ZtZ_rfp);

        // ZtY += Z_chunk^T @ Y_chunk
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    static_cast<int>(nc), static_cast<int>(D),
                    1.0, Z, static_cast<int>(D),
                    Y + cs, 1,
                    clear_or_sum, ZtY, 1);

        aligned_free_64(Z);
    }

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

    // rff_gradient uses OMP internally; outer loop is sequential.
    // Accumulate directly into GtG_rfp/GtF; clear_or_sum avoids memset.
    std::size_t chunk_idx = 0;
    for (std::size_t cs = 0; cs < N; cs += chunk_size, ++chunk_idx) {
        const std::size_t ce           = std::min(cs + chunk_size, N);
        const std::size_t nc           = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        double *G = aligned_alloc_64(D * ngrads_chunk);

        // zero accum on first chunk, then sum
        double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

        rff_gradient(X  + cs * rep_size,
                     dX + cs * rep_size * ncoords,
                     W, b, nc, rep_size, D, ncoords, G);

        // GtG_rfp += G @ G^T
        // G is (D, ngrads_chunk) row-major ≡ (ngrads_chunk, D) col-major with LDA=ngrads_chunk.
        // DSFRK TRANS='T': C += A^T*A where A is (ngrads_chunk, D) → D×D ✓
        kf_dsfrk('N', 'U', 'T',
                 static_cast<blas_int>(D), static_cast<blas_int>(ngrads_chunk),
                 1.0, G, static_cast<blas_int>(ngrads_chunk),
                 clear_or_sum, GtG_rfp);

        // GtF += G @ F_chunk
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + cs * ncoords, 1,
                    clear_or_sum, GtF, 1);

        aligned_free_64(G);
    }
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

    // ---- Energy loop (chunked, sequential) ----
    // Accumulate directly into ZtZ_rfp/ZtY; clear_or_sum avoids memset.
    {
        const std::size_t nchunks = (N + energy_chunk - 1) / energy_chunk;

        for (std::size_t ci = 0; ci < nchunks; ++ci) {
            const std::size_t cs = ci * energy_chunk;
            const std::size_t ce = std::min(cs + energy_chunk, N);
            const std::size_t nc = ce - cs;

            double *Z = aligned_alloc_64(nc * D);

            // zero accum on first chunk, then sum
            double clear_or_sum = (ci == 0) ? 0.0 : 1.0;

            rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);

            // ZtZ_rfp += Z^T @ Z
            kf_dsfrk('N', 'U', 'N',
                     static_cast<blas_int>(D), static_cast<blas_int>(nc),
                     1.0, Z, static_cast<blas_int>(D),
                     clear_or_sum, ZtZ_rfp);

            // ZtY += Z^T @ Y_chunk
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        static_cast<int>(nc), static_cast<int>(D),
                        1.0, Z, static_cast<int>(D),
                        Y + cs, 1,
                        clear_or_sum, ZtY, 1);

            aligned_free_64(Z);
        }
    }

    // ---- Force loop (rff_gradient uses OMP internally → sequential outer loop) ----
    // Energy loop already ran so beta=1.0 always.
    for (std::size_t cs = 0; cs < N; cs += force_chunk) {
        const std::size_t ce           = std::min(cs + force_chunk, N);
        const std::size_t nc           = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        double *G = aligned_alloc_64(D * ngrads_chunk);
        rff_gradient(X  + cs * rep_size,
                     dX + cs * rep_size * ncoords,
                     W, b, nc, rep_size, D, ncoords, G);

        // ZtZ_rfp += G @ G^T
        kf_dsfrk('N', 'U', 'T',
                 static_cast<blas_int>(D), static_cast<blas_int>(ngrads_chunk),
                 1.0, G, static_cast<blas_int>(ngrads_chunk),
                 1.0, ZtZ_rfp);

        // ZtY += G @ F_chunk
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + cs * ncoords, 1,
                    1.0, ZtY, 1);

        aligned_free_64(G);
    }
}

}  // namespace kf::rff
