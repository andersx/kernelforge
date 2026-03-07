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

void rff_features(
    const double *X, const double *W, const double *b, std::size_t N, std::size_t rep_size,
    std::size_t D, double *Z
) {
    if (!X || !W || !b || !Z) throw std::invalid_argument("rff_features: null pointer");
    if (N == 0 || rep_size == 0 || D == 0)
        throw std::invalid_argument("rff_features: zero dimension");

    // Z = X @ W  via DGEMM
    // X is (N, rep_size), W is (rep_size, D), Z is (N, D)
    // RowMajor: M=N, N=D, K=rep_size
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(N),         // M
        static_cast<int>(D),         // N
        static_cast<int>(rep_size),  // K
        1.0,
        X,
        static_cast<int>(rep_size),  // A, lda
        W,
        static_cast<int>(D),  // B, ldb
        0.0,
        Z,
        static_cast<int>(D)
    );  // C, ldc

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

void rff_gradient(
    const double *X, const double *dX_T, const double *W, const double *b, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t dX_T_stride, double *G
) {
    if (!X || !dX_T || !W || !b || !G) throw std::invalid_argument("rff_gradient: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_gradient: zero dimension");

    const double normalization = -std::sqrt(2.0 / static_cast<double>(D));
    const std::size_t total_grads = N * ncoords;

    // --- Step 1: Compute Z (D, N): z values for all molecules ---
    // Z[d, i] = b[d] + (W^T @ X^T)[d, i]
    double *Z = aligned_alloc_64(D * N);

    // Initialize Z with broadcast b: Z[d, i] = b[d] for all i
#pragma omp parallel for schedule(static)
    for (std::size_t d = 0; d < D; ++d) {
        for (std::size_t i = 0; i < N; ++i) {
            Z[d * N + i] = b[d];
        }
    }

    // Z += W^T @ X^T
    // W: (rep_size, D) row-major → W^T via CblasTrans
    // X: (N, rep_size) row-major → X^T via CblasTrans
    // Result: Z (D, N) row-major
    cblas_dgemm(
        CblasRowMajor,
        CblasTrans,                  // W^T
        CblasTrans,                  // X^T
        static_cast<int>(D),         // M = D
        static_cast<int>(N),         // N = N
        static_cast<int>(rep_size),  // K = rep_size
        1.0,
        W,
        static_cast<int>(D),  // lda for W (rep_size, D)
        X,
        static_cast<int>(rep_size),  // ldb for X (N, rep_size)
        1.0,                         // beta = 1.0 (add to b broadcast)
        Z,
        static_cast<int>(N)  // ldc for Z (D, N)
    );

    // --- Step 2: Compute G (total_grads, D) = dX_T^T @ W ---
    // dX_T: (rep_size, total_grads) with stride dX_T_stride
    // W:    (rep_size, D) row-major
    // G:    (total_grads, D) row-major OUTPUT
    cblas_dgemm(
        CblasRowMajor,
        CblasTrans,                     // dX_T^T
        CblasNoTrans,                   // W
        static_cast<int>(total_grads),  // M = total_grads
        static_cast<int>(D),            // N = D
        static_cast<int>(rep_size),     // K = rep_size
        1.0,
        dX_T,
        static_cast<int>(dX_T_stride),  // lda: stride of dX_T
        W,
        static_cast<int>(D),  // ldb: W (rep_size, D)
        0.0,                  // beta = 0: initialize G
        G,
        static_cast<int>(D)  // ldc: G (total_grads, D)
    );

    // --- Step 3: Scale G in-place: G[i*nc+c, d] *= sin(Z[d, i]) * norm ---
    // Iterate over each gradient row sequentially for cache-friendly access
#pragma omp parallel for schedule(static)
    for (std::size_t g = 0; g < total_grads; ++g) {
        const std::size_t i = g / ncoords;  // molecule index
        double *row = G + g * D;            // G[g, :] — contiguous D elements
#pragma omp simd
        for (std::size_t d = 0; d < D; ++d) {
            row[d] *= std::sin(Z[d * N + i]) * normalization;
        }
    }

    aligned_free_64(Z);
}

void rff_gramian_symm(
    const double *X, const double *W, const double *b, const double *Y, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t chunk_size, double *ZtZ, double *ZtY
) {
    if (!X || !W || !b || !Y || !ZtZ || !ZtY)
        throw std::invalid_argument("rff_gramian_symm: null pointer");
    if (N == 0 || rep_size == 0 || D == 0)
        throw std::invalid_argument("rff_gramian_symm: zero dimension");
    if (chunk_size == 0) throw std::invalid_argument("rff_gramian_symm: zero chunk_size");

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
        cblas_dsyrk(
            CblasRowMajor,
            CblasUpper,
            CblasTrans,
            static_cast<int>(D),
            static_cast<int>(nc),
            1.0,
            Z,
            static_cast<int>(D),
            clear_or_sum,
            ZtZ,
            static_cast<int>(D)
        );

        // ZtY += Z^T @ Y_chunk
        cblas_dgemv(
            CblasRowMajor,
            CblasTrans,
            static_cast<int>(nc),
            static_cast<int>(D),
            1.0,
            Z,
            static_cast<int>(D),
            Y + cs,
            1,
            clear_or_sum,
            ZtY,
            1
        );

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

void rff_full_gramian_symm(
    const double *X, const double *dX_T, const double *W, const double *b, const double *Y,
    const double *F, std::size_t N, std::size_t rep_size, std::size_t D, std::size_t ncoords,
    std::size_t energy_chunk, std::size_t force_chunk, std::size_t dX_T_stride, double *ZtZ,
    double *ZtY
) {
    if (!X || !dX_T || !W || !b || !Y || !F || !ZtZ || !ZtY)
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
            cblas_dsyrk(
                CblasRowMajor,
                CblasUpper,
                CblasTrans,
                static_cast<int>(D),
                static_cast<int>(nc),
                1.0,
                Z,
                static_cast<int>(D),
                clear_or_sum,
                ZtZ,
                static_cast<int>(D)
            );

            // ZtY += Z^T @ Y_chunk
            cblas_dgemv(
                CblasRowMajor,
                CblasTrans,
                static_cast<int>(nc),
                static_cast<int>(D),
                1.0,
                Z,
                static_cast<int>(D),
                Y + cs,
                1,
                clear_or_sum,
                ZtY,
                1
            );

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

        double *G = aligned_alloc_64(ngrads_chunk * D);

        rff_gradient(
            X + cs * rep_size,
            dX_T + cs * ncoords,
            W,
            b,
            nc,
            rep_size,
            D,
            ncoords,
            dX_T_stride,
            G
        );

        // ZtZ += G^T @ G  (upper triangle, G is ngrads_chunk×D)
        cblas_dsyrk(
            CblasRowMajor,
            CblasUpper,
            CblasTrans,
            static_cast<int>(D),
            static_cast<int>(ngrads_chunk),
            1.0,
            G,
            static_cast<int>(D),
            1.0,
            ZtZ,
            static_cast<int>(D)
        );

        // ZtY += G^T @ F_chunk
        cblas_dgemv(
            CblasRowMajor,
            CblasTrans,
            static_cast<int>(ngrads_chunk),
            static_cast<int>(D),
            1.0,
            G,
            static_cast<int>(D),
            F + cs * ncoords,
            1,
            1.0,
            ZtY,
            1
        );

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

void rff_full(
    const double *X, const double *dX_T, const double *W, const double *b, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t dX_T_stride,
    double *Z_full
) {
    if (!X || !dX_T || !W || !b || !Z_full) throw std::invalid_argument("rff_full: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_full: zero dimension");

    // Top half: energy features Z[0:N, :] — (N, D) row-major
    rff_features(X, W, b, N, rep_size, D, Z_full);

    // Bottom half: gradient features — write directly into Z_full[N:, :]
    // rff_gradient outputs (total_grads, D) row-major — exactly what we need.
    rff_gradient(
        X,
        dX_T,
        W,
        b,
        N,
        rep_size,
        D,
        ncoords,
        dX_T_stride,
        Z_full + N * D  // write directly at Z_full[N:, :], no transpose needed
    );
}

void rff_gradient_gramian_symm(
    const double *X, const double *dX_T, const double *W, const double *b, const double *F,
    std::size_t N, std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t chunk_size,
    std::size_t dX_T_stride, double *GtG, double *GtF
) {
    if (!X || !dX_T || !W || !b || !F || !GtG || !GtF)
        throw std::invalid_argument("rff_gradient_gramian_symm: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_gradient_gramian_symm: zero dimension");
    if (chunk_size == 0) throw std::invalid_argument("rff_gradient_gramian_symm: zero chunk_size");

    // rff_gradient now uses full BLAS threading; keep outer loop sequential.
    // Accumulate directly into GtG/GtF; clear_or_sum avoids memset.
    std::size_t chunk_idx = 0;
    for (std::size_t cs = 0; cs < N; cs += chunk_size, ++chunk_idx) {
        const std::size_t ce = std::min(cs + chunk_size, N);
        const std::size_t nc = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        // G is (ngrads_chunk, D) row-major
        double *G = aligned_alloc_64(ngrads_chunk * D);

        // zero accum on first chunk, then sum
        double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

        rff_gradient(
            X + cs * rep_size,
            dX_T + cs * ncoords,  // column offset into transposed dX
            W,
            b,
            nc,
            rep_size,
            D,
            ncoords,
            dX_T_stride,  // stride of full dX_T array
            G
        );

        // GtG += G^T @ G (upper triangle, G is ngrads_chunk × D)
        cblas_dsyrk(
            CblasRowMajor,
            CblasUpper,
            CblasTrans,
            static_cast<int>(D),
            static_cast<int>(ngrads_chunk),
            1.0,
            G,
            static_cast<int>(D),  // lda = D (row stride of G)
            clear_or_sum,
            GtG,
            static_cast<int>(D)
        );

        // GtF += G^T @ F_chunk
        cblas_dgemv(
            CblasRowMajor,
            CblasTrans,
            static_cast<int>(ngrads_chunk),
            static_cast<int>(D),
            1.0,
            G,
            static_cast<int>(D),
            F + cs * ncoords,
            1,
            clear_or_sum,
            GtF,
            1
        );

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

void rff_gramian_symm_rfp(
    const double *X, const double *W, const double *b, const double *Y, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t chunk_size, double *ZtZ_rfp, double *ZtY
) {
    if (!X || !W || !b || !Y || !ZtZ_rfp || !ZtY)
        throw std::invalid_argument("rff_gramian_symm_rfp: null pointer");
    if (N == 0 || rep_size == 0 || D == 0)
        throw std::invalid_argument("rff_gramian_symm_rfp: zero dimension");
    if (chunk_size == 0) throw std::invalid_argument("rff_gramian_symm_rfp: zero chunk_size");

    const std::size_t nchunks = (N + chunk_size - 1) / chunk_size;

    // Accumulate directly into RFP using DSFRK
    for (std::size_t ci = 0; ci < nchunks; ++ci) {
        const std::size_t cs = ci * chunk_size;
        const std::size_t ce = std::min(cs + chunk_size, N);
        const std::size_t nc = ce - cs;

        double *Z = aligned_alloc_64(nc * D);

        // zero accum on first chunk, then sum
        double clear_or_sum = (ci == 0) ? 0.0 : 1.0;

        rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);

        // ZtZ_rfp += Z_chunk^T @ Z_chunk
        kf_dsfrk(
            'N',
            'U',
            'N',
            static_cast<blas_int>(D),
            static_cast<blas_int>(nc),
            1.0,
            Z,
            static_cast<blas_int>(D),
            clear_or_sum,
            ZtZ_rfp
        );

        // ZtY += Z_chunk^T @ Y_chunk
        cblas_dgemv(
            CblasRowMajor,
            CblasTrans,
            static_cast<int>(nc),
            static_cast<int>(D),
            1.0,
            Z,
            static_cast<int>(D),
            Y + cs,
            1,
            clear_or_sum,
            ZtY,
            1
        );

        aligned_free_64(Z);
    }
}

void rff_gradient_gramian_symm_rfp(
    const double *X, const double *dX_T, const double *W, const double *b, const double *F,
    std::size_t N, std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t chunk_size,
    std::size_t dX_T_stride, double *GtG_rfp, double *GtF
) {
    if (!X || !dX_T || !W || !b || !F || !GtG_rfp || !GtF)
        throw std::invalid_argument("rff_gradient_gramian_symm_rfp: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_gradient_gramian_symm_rfp: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gradient_gramian_symm_rfp: zero chunk_size");

    // rff_gradient uses OMP internally; outer loop is sequential.
    // Accumulate directly into GtG_rfp/GtF; clear_or_sum avoids memset.
    std::size_t chunk_idx = 0;
    for (std::size_t cs = 0; cs < N; cs += chunk_size, ++chunk_idx) {
        const std::size_t ce = std::min(cs + chunk_size, N);
        const std::size_t nc = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        double *G = aligned_alloc_64(ngrads_chunk * D);

        // zero accum on first chunk, then sum
        double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

        rff_gradient(
            X + cs * rep_size,
            dX_T + cs * ncoords,
            W,
            b,
            nc,
            rep_size,
            D,
            ncoords,
            dX_T_stride,
            G
        );

        // GtG_rfp += G^T @ G
        // G is (ngrads_chunk, D) row-major ≡ (D, ngrads_chunk) col-major with LDA=D.
        // DSFRK TRANS='N': C += A^T*A where A is (D, ngrads_chunk) col-major → D×D ✓
        kf_dsfrk(
            'N',
            'U',
            'N',
            static_cast<blas_int>(D),
            static_cast<blas_int>(ngrads_chunk),
            1.0,
            G,
            static_cast<blas_int>(D),
            clear_or_sum,
            GtG_rfp
        );

        // GtF += G^T @ F_chunk
        cblas_dgemv(
            CblasRowMajor,
            CblasTrans,
            static_cast<int>(ngrads_chunk),
            static_cast<int>(D),
            1.0,
            G,
            static_cast<int>(D),
            F + cs * ncoords,
            1,
            clear_or_sum,
            GtF,
            1
        );

        aligned_free_64(G);
    }
}

void rff_full_gramian_symm_rfp(
    const double *X, const double *dX_T, const double *W, const double *b, const double *Y,
    const double *F, std::size_t N, std::size_t rep_size, std::size_t D, std::size_t ncoords,
    std::size_t energy_chunk, std::size_t force_chunk, std::size_t dX_T_stride, double *ZtZ_rfp,
    double *ZtY
) {
    if (!X || !dX_T || !W || !b || !Y || !F || !ZtZ_rfp || !ZtY)
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
            kf_dsfrk(
                'N',
                'U',
                'N',
                static_cast<blas_int>(D),
                static_cast<blas_int>(nc),
                1.0,
                Z,
                static_cast<blas_int>(D),
                clear_or_sum,
                ZtZ_rfp
            );

            // ZtY += Z^T @ Y_chunk
            cblas_dgemv(
                CblasRowMajor,
                CblasTrans,
                static_cast<int>(nc),
                static_cast<int>(D),
                1.0,
                Z,
                static_cast<int>(D),
                Y + cs,
                1,
                clear_or_sum,
                ZtY,
                1
            );

            aligned_free_64(Z);
        }
    }

    // ---- Force loop (rff_gradient uses OMP internally → sequential outer loop) ----
    // Energy loop already ran so beta=1.0 always.
    for (std::size_t cs = 0; cs < N; cs += force_chunk) {
        const std::size_t ce = std::min(cs + force_chunk, N);
        const std::size_t nc = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        double *G = aligned_alloc_64(ngrads_chunk * D);

        rff_gradient(
            X + cs * rep_size,
            dX_T + cs * ncoords,
            W,
            b,
            nc,
            rep_size,
            D,
            ncoords,
            dX_T_stride,
            G
        );

        // ZtZ_rfp += G^T @ G
        kf_dsfrk(
            'N',
            'U',
            'N',
            static_cast<blas_int>(D),
            static_cast<blas_int>(ngrads_chunk),
            1.0,
            G,
            static_cast<blas_int>(D),
            1.0,
            ZtZ_rfp
        );

        // ZtY += G^T @ F_chunk
        cblas_dgemv(
            CblasRowMajor,
            CblasTrans,
            static_cast<int>(ngrads_chunk),
            static_cast<int>(D),
            1.0,
            G,
            static_cast<int>(D),
            F + cs * ncoords,
            1,
            1.0,
            ZtY,
            1
        );

        aligned_free_64(G);
    }
}

}  // namespace kf::rff
