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
#include "profiling.h"
#include "rff_features.hpp"

namespace kf::rff {

void rff_features(
    const double *X, const double *W, const double *b, std::size_t N, std::size_t rep_size,
    std::size_t D, double *Z
) {
    if (!X || !W || !b || !Z) throw std::invalid_argument("rff_features: null pointer");
    if (N == 0 || rep_size == 0 || D == 0)
        throw std::invalid_argument("rff_features: zero dimension");

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_dgemm = 0.0, t_cos_scaling = 0.0;
#endif

    // Z = X @ W  via DGEMM
    // X is (N, rep_size), W is (rep_size, D), Z is (N, D)
    // RowMajor: M=N, N=D, K=rep_size
    PROFILE_START(dgemm);
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
    PROFILE_END(dgemm, t_dgemm);

    // Z[:, d] = cos(Z[:, d] + b[d]) * sqrt(2/D)
    const double normalization = std::sqrt(2.0 / static_cast<double>(D));

    // Outer loop parallelized: when called standalone this uses all OMP threads;
    // when called from inside an OMP parallel region (e.g. gramian functions),
    // nested parallelism is disabled by default so it serializes — same as before.
    PROFILE_START(cos_scaling);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
#pragma omp simd
        for (std::size_t d = 0; d < D; ++d) {
            Z[i * D + d] = std::cos(Z[i * D + d] + b[d]) * normalization;
        }
    }
    PROFILE_END(cos_scaling, t_cos_scaling);

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_total = t_dgemm + t_cos_scaling;
    std::printf(
        "[PROFILE] rff_features(N=%zu, rep_size=%zu, D=%zu):\n"
        "  Phase 1 (DGEMM):        %8.4fs (%5.1f%%)\n"
        "  Phase 2 (cos scaling):  %8.4fs (%5.1f%%)\n"
        "  Total:                  %8.4fs (100.0%%)\n",
        N,
        rep_size,
        D,
        t_dgemm,
        100.0 * t_dgemm / t_total,
        t_cos_scaling,
        100.0 * t_cos_scaling / t_total,
        t_total
    );
#endif
}

void rff_gradient(
    const double *X, const double *dX_T, const double *W, const double *b, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t dX_T_stride, double *G
) {
    if (!X || !dX_T || !W || !b || !G) throw std::invalid_argument("rff_gradient: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_gradient: zero dimension");

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_dgemm_z = 0.0, t_dgemm_g = 0.0, t_sin_scaling = 0.0;
#endif

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
    PROFILE_START(dgemm_z);
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
    PROFILE_END(dgemm_z, t_dgemm_z);

    // --- Step 2: Compute G (total_grads, D) = dX_T^T @ W ---
    // dX_T: (rep_size, total_grads) with stride dX_T_stride
    // W:    (rep_size, D) row-major
    // G:    (total_grads, D) row-major OUTPUT
    PROFILE_START(dgemm_g);
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
    PROFILE_END(dgemm_g, t_dgemm_g);

    // --- Step 3: Scale G in-place: G[i*nc+c, d] *= sin(Z[d, i]) * norm ---
    // Iterate over each gradient row sequentially for cache-friendly access
    PROFILE_START(sin_scaling);
#pragma omp parallel for schedule(static)
    for (std::size_t g = 0; g < total_grads; ++g) {
        const std::size_t i = g / ncoords;  // molecule index
        double *row = G + g * D;            // G[g, :] — contiguous D elements
#pragma omp simd
        for (std::size_t d = 0; d < D; ++d) {
            row[d] *= std::sin(Z[d * N + i]) * normalization;
        }
    }
    PROFILE_END(sin_scaling, t_sin_scaling);

    aligned_free_64(Z);

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_total = t_dgemm_z + t_dgemm_g + t_sin_scaling;
    std::printf(
        "[PROFILE] rff_gradient(N=%zu, rep_size=%zu, D=%zu, ncoords=%zu):\n"
        "  Phase 1 (DGEMM Z):      %8.4fs (%5.1f%%)\n"
        "  Phase 2 (DGEMM G):      %8.4fs (%5.1f%%)\n"
        "  Phase 3 (sin scaling):  %8.4fs (%5.1f%%)\n"
        "  Total:                  %8.4fs (100.0%%)\n",
        N,
        rep_size,
        D,
        ncoords,
        t_dgemm_z,
        100.0 * t_dgemm_z / t_total,
        t_dgemm_g,
        100.0 * t_dgemm_g / t_total,
        t_sin_scaling,
        100.0 * t_sin_scaling / t_total,
        t_total
    );
#endif
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

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_rff_features = 0.0, t_dsyrk = 0.0, t_dgemv = 0.0, t_symmetrize = 0.0;
#endif

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

        PROFILE_START(rff_features);
        rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);
        PROFILE_END(rff_features, t_rff_features);

        // ZtZ += Z^T @ Z  (upper triangle via DSYRK)
        PROFILE_START(dsyrk);
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
        PROFILE_END(dsyrk, t_dsyrk);

        // ZtY += Z^T @ Y_chunk
        PROFILE_START(dgemv);
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
        PROFILE_END(dgemv, t_dgemv);

        aligned_free_64(Z);
    }

    // Symmetrize: copy upper triangle to lower
    PROFILE_START(symmetrize);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            ZtZ[j * D + i] = ZtZ[i * D + j];
        }
    }
    PROFILE_END(symmetrize, t_symmetrize);

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_total = t_rff_features + t_dsyrk + t_dgemv + t_symmetrize;
    std::printf(
        "[PROFILE] rff_gramian_symm(N=%zu, rep_size=%zu, D=%zu, chunks=%zu):\n"
        "  Phase 1 (rff_features): %8.4fs (%5.1f%%)\n"
        "  Phase 2 (DSYRK):        %8.4fs (%5.1f%%)\n"
        "  Phase 3 (DGEMV ZtY):    %8.4fs (%5.1f%%)\n"
        "  Phase 4 (symmetrize):   %8.4fs (%5.1f%%)\n"
        "  Total:                  %8.4fs (100.0%%)\n",
        N,
        rep_size,
        D,
        nchunks,
        t_rff_features,
        100.0 * t_rff_features / t_total,
        t_dsyrk,
        100.0 * t_dsyrk / t_total,
        t_dgemv,
        100.0 * t_dgemv / t_total,
        t_symmetrize,
        100.0 * t_symmetrize / t_total,
        t_total
    );
#endif
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

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_energy_rff_features = 0.0, t_energy_dsyrk = 0.0, t_energy_dgemv = 0.0;
    double t_force_rff_gradient = 0.0, t_force_dsyrk = 0.0, t_force_dgemv = 0.0;
    double t_symmetrize = 0.0;
    std::size_t energy_chunks = 0, force_chunks = 0;
#endif

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

            PROFILE_START(energy_rff_features);
            rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);
            PROFILE_END(energy_rff_features, t_energy_rff_features);

            // ZtZ += Z^T @ Z  (upper triangle via DSYRK)
            PROFILE_START(energy_dsyrk);
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
            PROFILE_END(energy_dsyrk, t_energy_dsyrk);

            // ZtY += Z^T @ Y_chunk
            PROFILE_START(energy_dgemv);
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
            PROFILE_END(energy_dgemv, t_energy_dgemv);

            aligned_free_64(Z);
#ifdef KERNELFORGE_ENABLE_PROFILING
            energy_chunks++;
#endif
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

        PROFILE_START(force_rff_gradient);
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
        PROFILE_END(force_rff_gradient, t_force_rff_gradient);

        // ZtZ += G^T @ G  (upper triangle, G is ngrads_chunk×D)
        PROFILE_START(force_dsyrk);
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
        PROFILE_END(force_dsyrk, t_force_dsyrk);

        // ZtY += G^T @ F_chunk
        PROFILE_START(force_dgemv);
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
        PROFILE_END(force_dgemv, t_force_dgemv);

        aligned_free_64(G);
#ifdef KERNELFORGE_ENABLE_PROFILING
        force_chunks++;
#endif
    }

    // Symmetrize: copy upper triangle to lower
    PROFILE_START(symmetrize);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            ZtZ[j * D + i] = ZtZ[i * D + j];
        }
    }
    PROFILE_END(symmetrize, t_symmetrize);

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_energy_total = t_energy_rff_features + t_energy_dsyrk + t_energy_dgemv;
    double t_force_total = t_force_rff_gradient + t_force_dsyrk + t_force_dgemv;
    double t_total = t_energy_total + t_force_total + t_symmetrize;
    std::printf(
        "[PROFILE] rff_full_gramian_symm(N=%zu, D=%zu, ncoords=%zu, e_chunks=%zu, f_chunks=%zu):\n"
        "  === Energy Loop ===\n"
        "    rff_features:       %8.4fs (%5.1f%%)\n"
        "    DSYRK (energy):     %8.4fs (%5.1f%%)\n"
        "    DGEMV (ZtY):        %8.4fs (%5.1f%%)\n"
        "    Subtotal:           %8.4fs (%5.1f%%)\n"
        "  === Force Loop ===\n"
        "    rff_gradient:       %8.4fs (%5.1f%%)\n"
        "    DSYRK (force):      %8.4fs (%5.1f%%)\n"
        "    DGEMV (GtF):        %8.4fs (%5.1f%%)\n"
        "    Subtotal:           %8.4fs (%5.1f%%)\n"
        "  === Finalization ===\n"
        "    Symmetrization:     %8.4fs (%5.1f%%)\n"
        "  === Total ===\n"
        "    Total time:         %8.4fs (100.0%%)\n",
        N,
        D,
        ncoords,
        energy_chunks,
        force_chunks,
        t_energy_rff_features,
        100.0 * t_energy_rff_features / t_total,
        t_energy_dsyrk,
        100.0 * t_energy_dsyrk / t_total,
        t_energy_dgemv,
        100.0 * t_energy_dgemv / t_total,
        t_energy_total,
        100.0 * t_energy_total / t_total,
        t_force_rff_gradient,
        100.0 * t_force_rff_gradient / t_total,
        t_force_dsyrk,
        100.0 * t_force_dsyrk / t_total,
        t_force_dgemv,
        100.0 * t_force_dgemv / t_total,
        t_force_total,
        100.0 * t_force_total / t_total,
        t_symmetrize,
        100.0 * t_symmetrize / t_total,
        t_total
    );
#endif
}

void rff_full(
    const double *X, const double *dX_T, const double *W, const double *b, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t dX_T_stride,
    double *Z_full
) {
    if (!X || !dX_T || !W || !b || !Z_full) throw std::invalid_argument("rff_full: null pointer");
    if (N == 0 || rep_size == 0 || D == 0 || ncoords == 0)
        throw std::invalid_argument("rff_full: zero dimension");

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_rff_features = 0.0, t_rff_gradient = 0.0;
#endif

    const std::size_t total_grads = N * ncoords;

    // Top half: energy features Z[0:N, :] — (N, D) row-major
    PROFILE_START(rff_features);
    rff_features(X, W, b, N, rep_size, D, Z_full);
    PROFILE_END(rff_features, t_rff_features);

    // Bottom half: gradient features — write directly into Z_full[N:, :]
    // rff_gradient now outputs (total_grads, D) row-major — exactly what we need!
    PROFILE_START(rff_gradient);
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
        Z_full + N * D  // write directly at Z_full[N:, :], no transpose needed!
    );
    PROFILE_END(rff_gradient, t_rff_gradient);

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_total = t_rff_features + t_rff_gradient;
    std::printf(
        "[PROFILE] rff_full(N=%zu, rep_size=%zu, D=%zu, ncoords=%zu):\n"
        "  Phase 1 (rff_features): %8.4fs (%5.1f%%)\n"
        "  Phase 2 (rff_gradient): %8.4fs (%5.1f%%)\n"
        "  Total:                  %8.4fs (100.0%%) *** TRANSPOSE ELIMINATED! ***\n",
        N,
        rep_size,
        D,
        ncoords,
        t_rff_features,
        100.0 * t_rff_features / t_total,
        t_rff_gradient,
        100.0 * t_rff_gradient / t_total,
        t_total
    );
#endif
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

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_rff_gradient = 0.0, t_dsyrk = 0.0, t_dgemv = 0.0, t_symmetrize = 0.0;
#endif

    // rff_gradient now uses full BLAS threading; keep outer loop sequential.
    // Accumulate directly into GtG/GtF; clear_or_sum avoids memset.
    std::size_t chunk_idx = 0;
    for (std::size_t cs = 0; cs < N; cs += chunk_size, ++chunk_idx) {
        const std::size_t ce = std::min(cs + chunk_size, N);
        const std::size_t nc = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        // G is now (ngrads_chunk, D) row-major instead of (D, ngrads_chunk)
        double *G = aligned_alloc_64(ngrads_chunk * D);

        // zero accum on first chunk, then sum
        double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

        PROFILE_START(rff_gradient);
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
        PROFILE_END(rff_gradient, t_rff_gradient);

        // GtG += G^T @ G (upper triangle, G is ngrads_chunk × D)
        PROFILE_START(dsyrk);
        cblas_dsyrk(
            CblasRowMajor,
            CblasUpper,
            CblasTrans,  // Changed: G^T @ G
            static_cast<int>(D),
            static_cast<int>(ngrads_chunk),
            1.0,
            G,
            static_cast<int>(D),  // Changed: lda = D (row stride of G)
            clear_or_sum,
            GtG,
            static_cast<int>(D)
        );
        PROFILE_END(dsyrk, t_dsyrk);

        // GtF += G^T @ F_chunk
        PROFILE_START(dgemv);
        cblas_dgemv(
            CblasRowMajor,
            CblasTrans,                      // Changed: G^T @ F
            static_cast<int>(ngrads_chunk),  // Changed: M = ngrads_chunk
            static_cast<int>(D),             // Changed: N = D
            1.0,
            G,
            static_cast<int>(D),  // Changed: lda = D
            F + cs * ncoords,
            1,
            clear_or_sum,
            GtF,
            1
        );
        PROFILE_END(dgemv, t_dgemv);

        aligned_free_64(G);
    }

    // Symmetrize: copy upper triangle to lower
    PROFILE_START(symmetrize);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            GtG[j * D + i] = GtG[i * D + j];
        }
    }
    PROFILE_END(symmetrize, t_symmetrize);

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_total = t_rff_gradient + t_dsyrk + t_dgemv + t_symmetrize;
    std::printf(
        "[PROFILE] rff_gradient_gramian_symm(N=%zu, D=%zu, ncoords=%zu, chunks=%zu):\n"
        "  Phase 1 (rff_gradient): %8.4fs (%5.1f%%)\n"
        "  Phase 2 (DSYRK GtG):    %8.4fs (%5.1f%%)\n"
        "  Phase 3 (DGEMV GtF):    %8.4fs (%5.1f%%)\n"
        "  Phase 4 (symmetrize):   %8.4fs (%5.1f%%)\n"
        "  Total:                  %8.4fs (100.0%%)\n",
        N,
        D,
        ncoords,
        chunk_idx,
        t_rff_gradient,
        100.0 * t_rff_gradient / t_total,
        t_dsyrk,
        100.0 * t_dsyrk / t_total,
        t_dgemv,
        100.0 * t_dgemv / t_total,
        t_symmetrize,
        100.0 * t_symmetrize / t_total,
        t_total
    );
#endif
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

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_rff_features = 0.0, t_dsfrk = 0.0, t_dgemv = 0.0;
#endif

    const std::size_t nt = D * (D + 1) / 2;
    const std::size_t nchunks = (N + chunk_size - 1) / chunk_size;

    // // Accumulate directly into RFP using DSFRK
    for (std::size_t ci = 0; ci < nchunks; ++ci) {
        const std::size_t cs = ci * chunk_size;
        const std::size_t ce = std::min(cs + chunk_size, N);
        const std::size_t nc = ce - cs;

        double *Z = aligned_alloc_64(nc * D);

        // zero accum on first chunk, then sum
        double clear_or_sum = (ci == 0) ? 0.0 : 1.0;

        PROFILE_START(rff_features);
        rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);
        PROFILE_END(rff_features, t_rff_features);

        // ZtZ_rfp += Z_chunk^T @ Z_chunk
        PROFILE_START(dsfrk);
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
        PROFILE_END(dsfrk, t_dsfrk);

        // ZtY += Z_chunk^T @ Y_chunk
        PROFILE_START(dgemv);
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
        PROFILE_END(dgemv, t_dgemv);

        aligned_free_64(Z);
    }

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_total = t_rff_features + t_dsfrk + t_dgemv;
    std::printf(
        "[PROFILE] rff_gramian_symm_rfp(N=%zu, rep_size=%zu, D=%zu, chunks=%zu):\n"
        "  Phase 1 (rff_features): %8.4fs (%5.1f%%)\n"
        "  Phase 2 (DSFRK):        %8.4fs (%5.1f%%)\n"
        "  Phase 3 (DGEMV ZtY):    %8.4fs (%5.1f%%)\n"
        "  Total:                  %8.4fs (100.0%%)\n",
        N,
        rep_size,
        D,
        nchunks,
        t_rff_features,
        100.0 * t_rff_features / t_total,
        t_dsfrk,
        100.0 * t_dsfrk / t_total,
        t_dgemv,
        100.0 * t_dgemv / t_total,
        t_total
    );
#endif
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

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_rff_gradient = 0.0, t_dsfrk = 0.0, t_dgemv = 0.0;
#endif

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

        PROFILE_START(rff_gradient);
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
        PROFILE_END(rff_gradient, t_rff_gradient);

        // GtG_rfp += G^T @ G
        // G is (ngrads_chunk, D) row-major ≡ (D, ngrads_chunk) col-major with LDA=D.
        // DSFRK TRANS='N': C += A^T*A where A is (D, ngrads_chunk) col-major → D×D ✓
        PROFILE_START(dsfrk);
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
        PROFILE_END(dsfrk, t_dsfrk);

        // GtF += G^T @ F_chunk
        PROFILE_START(dgemv);
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
        PROFILE_END(dgemv, t_dgemv);

        aligned_free_64(G);
    }

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_total = t_rff_gradient + t_dsfrk + t_dgemv;
    std::printf(
        "[PROFILE] rff_gradient_gramian_symm_rfp(N=%zu, D=%zu, ncoords=%zu, chunks=%zu):\n"
        "  Phase 1 (rff_gradient): %8.4fs (%5.1f%%)\n"
        "  Phase 2 (DSFRK GtG):    %8.4fs (%5.1f%%)\n"
        "  Phase 3 (DGEMV GtF):    %8.4fs (%5.1f%%)\n"
        "  Total:                  %8.4fs (100.0%%)\n",
        N,
        D,
        ncoords,
        chunk_idx,
        t_rff_gradient,
        100.0 * t_rff_gradient / t_total,
        t_dsfrk,
        100.0 * t_dsfrk / t_total,
        t_dgemv,
        100.0 * t_dgemv / t_total,
        t_total
    );
#endif
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

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_energy_rff_features = 0.0, t_energy_dsfrk = 0.0, t_energy_dgemv = 0.0;
    double t_force_rff_gradient = 0.0, t_force_dsfrk = 0.0, t_force_dgemv = 0.0;
    std::size_t energy_chunks = 0, force_chunks = 0;
#endif

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

            PROFILE_START(energy_rff_features);
            rff_features(X + cs * rep_size, W, b, nc, rep_size, D, Z);
            PROFILE_END(energy_rff_features, t_energy_rff_features);

            // ZtZ_rfp += Z^T @ Z
            PROFILE_START(energy_dsfrk);
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
            PROFILE_END(energy_dsfrk, t_energy_dsfrk);

            // ZtY += Z^T @ Y_chunk
            PROFILE_START(energy_dgemv);
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
            PROFILE_END(energy_dgemv, t_energy_dgemv);

            aligned_free_64(Z);
#ifdef KERNELFORGE_ENABLE_PROFILING
            energy_chunks++;
#endif
        }
    }

    // ---- Force loop (rff_gradient uses OMP internally → sequential outer loop) ----
    // Energy loop already ran so beta=1.0 always.
    for (std::size_t cs = 0; cs < N; cs += force_chunk) {
        const std::size_t ce = std::min(cs + force_chunk, N);
        const std::size_t nc = ce - cs;
        const std::size_t ngrads_chunk = nc * ncoords;

        double *G = aligned_alloc_64(ngrads_chunk * D);

        PROFILE_START(force_rff_gradient);
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
        PROFILE_END(force_rff_gradient, t_force_rff_gradient);

        // ZtZ_rfp += G^T @ G
        PROFILE_START(force_dsfrk);
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
        PROFILE_END(force_dsfrk, t_force_dsfrk);

        // ZtY += G^T @ F_chunk
        PROFILE_START(force_dgemv);
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
        PROFILE_END(force_dgemv, t_force_dgemv);

        aligned_free_64(G);
#ifdef KERNELFORGE_ENABLE_PROFILING
        force_chunks++;
#endif
    }

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_energy_total = t_energy_rff_features + t_energy_dsfrk + t_energy_dgemv;
    double t_force_total = t_force_rff_gradient + t_force_dsfrk + t_force_dgemv;
    double t_total = t_energy_total + t_force_total;
    std::printf(
        "[PROFILE] rff_full_gramian_symm_rfp(N=%zu, D=%zu, ncoords=%zu, e_chunks=%zu, "
        "f_chunks=%zu):\n"
        "  === Energy Loop ===\n"
        "    rff_features:       %8.4fs (%5.1f%%)\n"
        "    DSFRK (energy):     %8.4fs (%5.1f%%)\n"
        "    DGEMV (ZtY):        %8.4fs (%5.1f%%)\n"
        "    Subtotal:           %8.4fs (%5.1f%%)\n"
        "  === Force Loop ===\n"
        "    rff_gradient:       %8.4fs (%5.1f%%)\n"
        "    DSFRK (force):      %8.4fs (%5.1f%%)\n"
        "    DGEMV (GtF):        %8.4fs (%5.1f%%)\n"
        "    Subtotal:           %8.4fs (%5.1f%%)\n"
        "  === Total ===\n"
        "    Total time:         %8.4fs (100.0%%)\n",
        N,
        D,
        ncoords,
        energy_chunks,
        force_chunks,
        t_energy_rff_features,
        100.0 * t_energy_rff_features / t_total,
        t_energy_dsfrk,
        100.0 * t_energy_dsfrk / t_total,
        t_energy_dgemv,
        100.0 * t_energy_dgemv / t_total,
        t_energy_total,
        100.0 * t_energy_total / t_total,
        t_force_rff_gradient,
        100.0 * t_force_rff_gradient / t_total,
        t_force_dsfrk,
        100.0 * t_force_dsfrk / t_total,
        t_force_dgemv,
        100.0 * t_force_dgemv / t_total,
        t_force_total,
        100.0 * t_force_total / t_total,
        t_total
    );
#endif
}

}  // namespace kf::rff
