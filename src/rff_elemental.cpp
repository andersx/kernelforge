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
#include "rff_elemental.hpp"

namespace kf::rff {

void rff_features_elemental(
    const double *X, const int *Q, const int *sizes,
    const double *W, const double *b,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    double *LZ) {

    if (!X || !Q || !sizes || !W || !b || !LZ)
        throw std::invalid_argument("rff_features_elemental: null pointer");
    if (nmol == 0 || max_atoms == 0 || rep_size == 0 || nelements == 0 || D == 0)
        throw std::invalid_argument("rff_features_elemental: zero dimension");

    const double normalization = std::sqrt(2.0 / static_cast<double>(D));

    // Zero the output
    std::memset(LZ, 0, nmol * D * sizeof(double));

    // For each element e:
    //   1. Count atoms of element e per molecule, build per-mol start/count
    //   2. Gather those atoms into Xsort (total_e, rep_size) — OMP over molecules
    //   3. Ze = Xsort @ W[e] + b[e]  via one DGEMM from the main thread
    //   4. cos + normalization — OMP over total_e * D elements
    //   5. Scatter-add Ze rows back into LZ per molecule — OMP over molecules
    //
    // BLAS (step 3) is called from the main thread only, never inside an OMP
    // parallel region, so any threaded BLAS backend works without conflict.

    for (std::size_t e = 0; e < nelements; ++e) {

        // --- Step 1: count atoms of type e per molecule (cheap, keep serial) ---
        std::vector<std::size_t> mol_start(nmol);
        std::vector<std::size_t> mol_count(nmol);
        std::size_t total_e = 0;

        for (std::size_t i = 0; i < nmol; ++i) {
            mol_start[i] = total_e;
            std::size_t cnt = 0;
            const int natoms_i = sizes[i];
            for (int j = 0; j < natoms_i; ++j) {
                if (Q[i * max_atoms + j] == static_cast<int>(e)) ++cnt;
            }
            mol_count[i] = cnt;
            total_e += cnt;
        }

        if (total_e == 0) continue;

        // --- Step 2: Gather Xsort (total_e, rep_size) — OMP over molecules ---
        double *Xsort = aligned_alloc_64(total_e * rep_size);

        #pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nmol; ++i) {
            if (mol_count[i] == 0) continue;
            const int natoms_i = sizes[i];
            std::size_t local_idx = mol_start[i];
            for (int j = 0; j < natoms_i; ++j) {
                if (Q[i * max_atoms + j] == static_cast<int>(e)) {
                    std::memcpy(Xsort + local_idx * rep_size,
                                X     + (i * max_atoms + j) * rep_size,
                                rep_size * sizeof(double));
                    ++local_idx;
                }
            }
        }

        // --- Step 3: Ze = Xsort @ W[e]  (main thread, let BLAS use its own threads) ---
        double *Ze = aligned_alloc_64(total_e * D);
        const double *We = W + e * rep_size * D;
        const double *be = b + e * D;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(total_e),
                    static_cast<int>(D),
                    static_cast<int>(rep_size),
                    1.0, Xsort, static_cast<int>(rep_size),
                    We,    static_cast<int>(D),
                    0.0, Ze,    static_cast<int>(D));

        aligned_free_64(Xsort);

        // --- Step 4: Ze[row,d] = cos(Ze[row,d] + be[d]) * norm — OMP over rows ---
        #pragma omp parallel for schedule(static)
        for (std::size_t row = 0; row < total_e; ++row) {
            double *ze = Ze + row * D;
            for (std::size_t d = 0; d < D; ++d) {
                ze[d] = std::cos(ze[d] + be[d]) * normalization;
            }
        }

        // --- Step 5: Scatter-add Ze rows into LZ per molecule — OMP over molecules ---
        #pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nmol; ++i) {
            if (mol_count[i] == 0) continue;
            const std::size_t start = mol_start[i];
            const std::size_t count = mol_count[i];
            double *lz_row = LZ + i * D;
            for (std::size_t k = start; k < start + count; ++k) {
                const double *ze_row = Ze + k * D;
                #pragma omp simd
                for (std::size_t d = 0; d < D; ++d) {
                    lz_row[d] += ze_row[d];
                }
            }
        }

        aligned_free_64(Ze);
    }
}

void rff_gramian_elemental(
    const double *X, const int *Q, const int *sizes,
    const double *W, const double *b,
    const double *Y,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t chunk_size,
    double *ZtZ, double *ZtY) {

    if (!X || !Q || !sizes || !W || !b || !Y || !ZtZ || !ZtY)
        throw std::invalid_argument("rff_gramian_elemental: null pointer");
    if (nmol == 0 || D == 0)
        throw std::invalid_argument("rff_gramian_elemental: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gramian_elemental: zero chunk_size");

    // Accumulate directly into ZtZ/ZtY; clear_or_sum avoids memset.
    std::size_t chunk_idx = 0;
    for (std::size_t chunk_start = 0; chunk_start < nmol; chunk_start += chunk_size, ++chunk_idx) {
        const std::size_t chunk_end  = std::min(chunk_start + chunk_size, nmol);
        const std::size_t this_chunk = chunk_end - chunk_start;

        double *LZ = aligned_alloc_64(this_chunk * D);

        // zero accum on first chunk, then sum
        double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

        rff_features_elemental(
            X     + chunk_start * max_atoms * rep_size,
            Q     + chunk_start * max_atoms,
            sizes + chunk_start,
            W, b,
            this_chunk, max_atoms, rep_size, nelements, D,
            LZ);

        // ZtZ += LZ^T @ LZ  (upper triangle via DSYRK)
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    static_cast<int>(D),
                    static_cast<int>(this_chunk),
                    1.0, LZ, static_cast<int>(D),
                    clear_or_sum, ZtZ, static_cast<int>(D));

        // ZtY += LZ^T @ Y_chunk
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    static_cast<int>(this_chunk),
                    static_cast<int>(D),
                    1.0, LZ, static_cast<int>(D),
                    Y + chunk_start, 1,
                    clear_or_sum, ZtY, 1);

        aligned_free_64(LZ);
    }

    // Symmetrize: copy upper triangle to lower
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            ZtZ[j * D + i] = ZtZ[i * D + j];
        }
    }
}

void rff_full_gramian_elemental(
    const double *X, const double *dX,
    const int *Q, const int *sizes,
    const double *W, const double *b,
    const double *Y, const double *F,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t energy_chunk, std::size_t force_chunk,
    double *ZtZ, double *ZtY) {

    if (!X || !dX || !Q || !sizes || !W || !b || !Y || !F || !ZtZ || !ZtY)
        throw std::invalid_argument("rff_full_gramian_elemental: null pointer");
    if (nmol == 0 || D == 0)
        throw std::invalid_argument("rff_full_gramian_elemental: zero dimension");

    // ---- Energy loop (chunked) ----
    // Accumulate directly into ZtZ/ZtY; clear_or_sum avoids memset.
    {
        std::size_t chunk_idx = 0;
        for (std::size_t cs = 0; cs < nmol; cs += energy_chunk, ++chunk_idx) {
            const std::size_t ce = std::min(cs + energy_chunk, nmol);
            const std::size_t nc = ce - cs;

            double *LZ = aligned_alloc_64(nc * D);

            // zero accum on first chunk, then sum
            double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

            rff_features_elemental(
                X     + cs * max_atoms * rep_size,
                Q     + cs * max_atoms,
                sizes + cs,
                W, b,
                nc, max_atoms, rep_size, nelements, D,
                LZ);

            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        static_cast<int>(D), static_cast<int>(nc),
                        1.0, LZ, static_cast<int>(D),
                        clear_or_sum, ZtZ, static_cast<int>(D));

            cblas_dgemv(CblasRowMajor, CblasTrans,
                        static_cast<int>(nc), static_cast<int>(D),
                        1.0, LZ, static_cast<int>(D),
                        Y + cs, 1,
                        clear_or_sum, ZtY, 1);

            aligned_free_64(LZ);
        }
    }

    // ---- Force loop (chunked) ----
    // Energy loop already ran so beta=1.0 always.
    // Precompute cumulative atom offsets for F indexing
    std::vector<std::size_t> cum_atoms(nmol + 1);
    cum_atoms[0] = 0;
    for (std::size_t i = 0; i < nmol; ++i)
        cum_atoms[i + 1] = cum_atoms[i] + static_cast<std::size_t>(sizes[i]);

    for (std::size_t cs = 0; cs < nmol; cs += force_chunk) {
        const std::size_t ce = std::min(cs + force_chunk, nmol);
        const std::size_t nc = ce - cs;

        std::size_t ngrads_chunk = 0;
        for (std::size_t i = cs; i < ce; ++i)
            ngrads_chunk += 3 * static_cast<std::size_t>(sizes[i]);
        if (ngrads_chunk == 0) continue;

        double *G = aligned_alloc_64(D * ngrads_chunk);

        rff_gradient_elemental(
            X  + cs * max_atoms * rep_size,
            dX + cs * max_atoms * rep_size * max_atoms * 3,
            Q  + cs * max_atoms,
            sizes + cs,
            W, b,
            nc, max_atoms, rep_size, nelements, D,
            ngrads_chunk,
            G);

        // ZtZ += G @ G^T  (upper triangle, main thread)
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    1.0, ZtZ, static_cast<int>(D));

        // ZtY += G @ F_chunk
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + 3 * cum_atoms[cs], 1,
                    1.0, ZtY, 1);

        aligned_free_64(G);
    }

    // Symmetrize
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            ZtZ[j * D + i] = ZtZ[i * D + j];
        }
    }
}

void rff_gradient_elemental(
    const double *X, const double *dX,
    const int *Q, const int *sizes,
    const double *W, const double *b,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t ngrads,
    double *G) {

    if (!X || !dX || !Q || !sizes || !W || !b || !G)
        throw std::invalid_argument("rff_gradient_elemental: null pointer");
    if (nmol == 0 || max_atoms == 0 || rep_size == 0 || nelements == 0 || D == 0)
        throw std::invalid_argument("rff_gradient_elemental: zero dimension");

    const double normalization = -std::sqrt(2.0 / static_cast<double>(D));
    const double neg_norm     =  std::sqrt(2.0 / static_cast<double>(D));  // = -normalization

    // Zero output G (D, ngrads) row-major
    std::memset(G, 0, D * ngrads * sizeof(double));

    // Precompute g_start[i] = 3 * sum(sizes[0:i])
    std::vector<std::size_t> g_start(nmol);
    {
        std::size_t cumsum = 0;
        for (std::size_t i = 0; i < nmol; ++i) {
            g_start[i] = 3 * cumsum;
            cumsum += static_cast<std::size_t>(sizes[i]);
        }
    }

    // Strides for dX (nmol, max_atoms, rep_size, max_atoms, 3) row-major
    const std::size_t dX_mol_stride  = max_atoms * rep_size * max_atoms * 3;
    const std::size_t dX_atom_stride = rep_size  * max_atoms * 3;
    const std::size_t max_ncols      = 3 * max_atoms;

    // Parallelize over molecules. Each thread gets its own scratch buffers.
    // BLAS (DGEMV + DGEMM) is called inside the OMP region on small per-atom
    // sub-problems. Set MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1 to avoid
    // nested threading.
    //
    // Per-atom computation (replaces the old dg-fill + DGEMM approach):
    //   G_chunk += sqrt(2/D) * diag(sin(dZ)) @ We^T @ dX_atom
    // Split as:
    //   1. DGEMM: G_tmp = We^T @ dX_atom  (beta=0, D×ncols result, efficient transpose)
    //   2. Scale-accumulate: G_chunk[d,:] += sin(dZ[d]) * sqrt(2/D) * G_tmp[d,:]
    //
    // This avoids the dg (D×rep_size) buffer whose construction reads We column-by-
    // column (stride D) — a severe cache miss pattern. G_tmp is D×ncols (much
    // smaller) and stays warm in L2/L3.
    //
    // dX_atom and G_tmp are allocated once per thread (with max_ncols columns)
    // rather than once per molecule, eliminating inner-loop malloc overhead.
    //
    // Serialise BLAS inside the OMP region to prevent thread oversubscription.
    // MKL does this automatically; OpenBLAS requires an explicit API call.
    const int blas_nt = kf_blas_get_num_threads();
    kf_blas_set_num_threads(1);
    #pragma omp parallel
    {
        std::vector<double> dZ(D);
        double *G_tmp   = aligned_alloc_64(D * max_ncols);
        double *dX_atom = aligned_alloc_64(rep_size * max_ncols);

        #pragma omp for schedule(dynamic)
        for (std::size_t i = 0; i < nmol; ++i) {
            const int natoms = sizes[i];
            if (natoms == 0) continue;

            const std::size_t ncols = 3 * static_cast<std::size_t>(natoms);

            for (int j = 0; j < natoms; ++j) {
                const int e       = Q[i * max_atoms + j];
                const double *We  = W + static_cast<std::size_t>(e) * rep_size * D;
                const double *be  = b + static_cast<std::size_t>(e) * D;
                const double *Xij = X + (i * max_atoms + j) * rep_size;

                // dZ = be + We^T @ Xij
                std::memcpy(dZ.data(), be, D * sizeof(double));
                cblas_dgemv(CblasRowMajor, CblasTrans,
                            static_cast<int>(rep_size), static_cast<int>(D),
                            1.0, We, static_cast<int>(D),
                            Xij, 1,
                            1.0, dZ.data(), 1);

                // Gather dX_atom (rep_size, ncols).
                // dX[i, j, r, 0:natoms, 0:3] is natoms*3 = ncols contiguous doubles,
                // so each row r of dX_atom is a single memcpy.
                for (std::size_t r = 0; r < rep_size; ++r) {
                    std::memcpy(dX_atom + r * ncols,
                                dX + i * dX_mol_stride +
                                     static_cast<std::size_t>(j) * dX_atom_stride +
                                     r * (max_atoms * 3),
                                ncols * sizeof(double));
                }

                // G_tmp (D, ncols) = We^T @ dX_atom
                // We is (rep_size, D) row-major → We^T is (D, rep_size); lda = D.
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            static_cast<int>(D),
                            static_cast<int>(ncols),
                            static_cast<int>(rep_size),
                            1.0,
                            We,      static_cast<int>(D),
                            dX_atom, static_cast<int>(ncols),
                            0.0,
                            G_tmp,   static_cast<int>(ncols));

                // G[:, g_start[i]:+ncols] += sin(dZ[d]) * sqrt(2/D) * G_tmp[d,:]
                for (std::size_t d = 0; d < D; ++d) {
                    const double factor   = std::sin(dZ[d]) * neg_norm;
                    const double *tmp_row = G_tmp + d * ncols;
                    double       *g_row   = G + d * ngrads + g_start[i];
                    #pragma omp simd
                    for (std::size_t c = 0; c < ncols; ++c) {
                        g_row[c] += factor * tmp_row[c];
                    }
                }
            }
        }

        aligned_free_64(dX_atom);
        aligned_free_64(G_tmp);
    } // end omp parallel
    kf_blas_set_num_threads(blas_nt);
}

void rff_full_elemental(
    const double *X, const double *dX,
    const int *Q, const int *sizes,
    const double *W, const double *b,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t ngrads,
    double *Z_full) {

    if (!X || !dX || !Q || !sizes || !W || !b || !Z_full)
        throw std::invalid_argument("rff_full_elemental: null pointer");
    if (nmol == 0 || max_atoms == 0 || rep_size == 0 || nelements == 0 || D == 0)
        throw std::invalid_argument("rff_full_elemental: zero dimension");

    // Top half: energy features LZ[0:nmol, :]
    rff_features_elemental(X, Q, sizes, W, b, nmol, max_atoms, rep_size, nelements, D, Z_full);

    // Bottom half: G^T where G is (D, ngrads)
    double *G = aligned_alloc_64(D * ngrads);
    rff_gradient_elemental(X, dX, Q, sizes, W, b, nmol, max_atoms, rep_size, nelements, D,
                           ngrads, G);

    // Transpose G (D, ngrads) -> Z_full[nmol:] (ngrads, D)
    // Z_full[(nmol + g) * D + d] = G[d * ngrads + g]
    #pragma omp parallel for schedule(static)
    for (std::size_t g = 0; g < ngrads; ++g) {
        double *row = Z_full + (nmol + g) * D;
        for (std::size_t d = 0; d < D; ++d) {
            row[d] = G[d * ngrads + g];
        }
    }

    aligned_free_64(G);
}

void rff_gradient_gramian_elemental(
    const double *X, const double *dX,
    const int *Q, const int *sizes,
    const double *W, const double *b,
    const double *F,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t chunk_size,
    double *GtG, double *GtF) {

    if (!X || !dX || !Q || !sizes || !W || !b || !F || !GtG || !GtF)
        throw std::invalid_argument("rff_gradient_gramian_elemental: null pointer");
    if (nmol == 0 || D == 0)
        throw std::invalid_argument("rff_gradient_gramian_elemental: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gradient_gramian_elemental: zero chunk_size");

    // Precompute cumulative atom offsets for F indexing
    std::vector<std::size_t> cum_atoms(nmol + 1);
    cum_atoms[0] = 0;
    for (std::size_t i = 0; i < nmol; ++i)
        cum_atoms[i + 1] = cum_atoms[i] + static_cast<std::size_t>(sizes[i]);

    // Accumulate directly into GtG/GtF; clear_or_sum avoids memset.
    // chunk_idx only incremented when a chunk actually does work (ngrads_chunk > 0).
    std::size_t chunk_idx = 0;
    for (std::size_t cs = 0; cs < nmol; cs += chunk_size) {
        const std::size_t ce = std::min(cs + chunk_size, nmol);
        const std::size_t nc = ce - cs;

        std::size_t ngrads_chunk = 0;
        for (std::size_t i = cs; i < ce; ++i)
            ngrads_chunk += 3 * static_cast<std::size_t>(sizes[i]);
        if (ngrads_chunk == 0) continue;

        double *G = aligned_alloc_64(D * ngrads_chunk);

        // zero accum on first non-empty chunk, then sum
        double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

        rff_gradient_elemental(
            X  + cs * max_atoms * rep_size,
            dX + cs * max_atoms * rep_size * max_atoms * 3,
            Q  + cs * max_atoms,
            sizes + cs,
            W, b,
            nc, max_atoms, rep_size, nelements, D,
            ngrads_chunk,
            G);

        // GtG += G @ G^T (upper triangle)
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    clear_or_sum, GtG, static_cast<int>(D));

        // GtF += G @ F_chunk
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + 3 * cum_atoms[cs], 1,
                    clear_or_sum, GtF, 1);

        aligned_free_64(G);
        ++chunk_idx;
    }

    // Symmetrize
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            GtG[j * D + i] = GtG[i * D + j];
        }
    }
}

void rff_gramian_elemental_rfp(
    const double *X, const int *Q, const int *sizes,
    const double *W, const double *b,
    const double *Y,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t chunk_size,
    double *ZtZ_rfp, double *ZtY) {

    if (!X || !Q || !sizes || !W || !b || !Y || !ZtZ_rfp || !ZtY)
        throw std::invalid_argument("rff_gramian_elemental_rfp: null pointer");
    if (nmol == 0 || D == 0)
        throw std::invalid_argument("rff_gramian_elemental_rfp: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gramian_elemental_rfp: zero chunk_size");

    // rff_features_elemental uses OMP internally → serial outer loop.
    // Accumulate directly into ZtZ_rfp/ZtY; clear_or_sum avoids memset.
    std::size_t chunk_idx = 0;
    for (std::size_t cs = 0; cs < nmol; cs += chunk_size, ++chunk_idx) {
        const std::size_t ce = std::min(cs + chunk_size, nmol);
        const std::size_t nc = ce - cs;

        double *LZ = aligned_alloc_64(nc * D);

        // zero accum on first chunk, then sum
        double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

        rff_features_elemental(
            X     + cs * max_atoms * rep_size,
            Q     + cs * max_atoms,
            sizes + cs,
            W, b,
            nc, max_atoms, rep_size, nelements, D,
            LZ);

        // LZ is (nc, D) row-major ≡ (D, nc) col-major with LDA=D.
        // DSFRK TRANS='N': C += A*A^T where A is (D, nc) → D×D ✓
        kf_dsfrk('N', 'U', 'N',
                 static_cast<blas_int>(D), static_cast<blas_int>(nc),
                 1.0, LZ, static_cast<blas_int>(D),
                 clear_or_sum, ZtZ_rfp);

        cblas_dgemv(CblasRowMajor, CblasTrans,
                    static_cast<int>(nc), static_cast<int>(D),
                    1.0, LZ, static_cast<int>(D),
                    Y + cs, 1,
                    clear_or_sum, ZtY, 1);

        aligned_free_64(LZ);
    }
}

void rff_gradient_gramian_elemental_rfp(
    const double *X, const double *dX,
    const int *Q, const int *sizes,
    const double *W, const double *b,
    const double *F,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t chunk_size,
    double *GtG_rfp, double *GtF) {

    if (!X || !dX || !Q || !sizes || !W || !b || !F || !GtG_rfp || !GtF)
        throw std::invalid_argument("rff_gradient_gramian_elemental_rfp: null pointer");
    if (nmol == 0 || D == 0)
        throw std::invalid_argument("rff_gradient_gramian_elemental_rfp: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gradient_gramian_elemental_rfp: zero chunk_size");

    // Precompute cumulative atom offsets for F indexing
    std::vector<std::size_t> cum_atoms(nmol + 1);
    cum_atoms[0] = 0;
    for (std::size_t i = 0; i < nmol; ++i)
        cum_atoms[i + 1] = cum_atoms[i] + static_cast<std::size_t>(sizes[i]);

    // rff_gradient_elemental uses OMP internally → serial outer loop.
    // Accumulate directly into GtG_rfp/GtF; clear_or_sum avoids memset.
    // chunk_idx only incremented when a chunk actually does work (ngrads_chunk > 0).
    std::size_t chunk_idx = 0;
    for (std::size_t cs = 0; cs < nmol; cs += chunk_size) {
        const std::size_t ce = std::min(cs + chunk_size, nmol);
        const std::size_t nc = ce - cs;

        std::size_t ngrads_chunk = 0;
        for (std::size_t i = cs; i < ce; ++i)
            ngrads_chunk += 3 * static_cast<std::size_t>(sizes[i]);
        if (ngrads_chunk == 0) continue;

        double *G = aligned_alloc_64(D * ngrads_chunk);

        // zero accum on first non-empty chunk, then sum
        double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

        rff_gradient_elemental(
            X  + cs * max_atoms * rep_size,
            dX + cs * max_atoms * rep_size * max_atoms * 3,
            Q  + cs * max_atoms,
            sizes + cs,
            W, b,
            nc, max_atoms, rep_size, nelements, D,
            ngrads_chunk, G);

        // G is (D, ngrads_chunk) row-major ≡ (ngrads_chunk, D) col-major with LDA=ngrads_chunk.
        // DSFRK TRANS='T': C += A^T*A where A is (ngrads_chunk, D) → D×D ✓
        kf_dsfrk('N', 'U', 'T',
                 static_cast<blas_int>(D), static_cast<blas_int>(ngrads_chunk),
                 1.0, G, static_cast<blas_int>(ngrads_chunk),
                 clear_or_sum, GtG_rfp);

        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + 3 * cum_atoms[cs], 1,
                    clear_or_sum, GtF, 1);

        aligned_free_64(G);
        ++chunk_idx;
    }
}

void rff_full_gramian_elemental_rfp(
    const double *X, const double *dX,
    const int *Q, const int *sizes,
    const double *W, const double *b,
    const double *Y, const double *F,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t energy_chunk, std::size_t force_chunk,
    double *ZtZ_rfp, double *ZtY) {

    if (!X || !dX || !Q || !sizes || !W || !b || !Y || !F || !ZtZ_rfp || !ZtY)
        throw std::invalid_argument("rff_full_gramian_elemental_rfp: null pointer");
    if (nmol == 0 || D == 0)
        throw std::invalid_argument("rff_full_gramian_elemental_rfp: zero dimension");
    if (energy_chunk == 0 || force_chunk == 0)
        throw std::invalid_argument("rff_full_gramian_elemental_rfp: zero chunk size");

    // ---- Energy loop (serial: rff_features_elemental uses OMP) ----
    // Accumulate directly into ZtZ_rfp/ZtY; clear_or_sum avoids memset.
    {
        std::size_t chunk_idx = 0;
        for (std::size_t cs = 0; cs < nmol; cs += energy_chunk, ++chunk_idx) {
            const std::size_t ce = std::min(cs + energy_chunk, nmol);
            const std::size_t nc = ce - cs;

            double *LZ = aligned_alloc_64(nc * D);

            // zero accum on first chunk, then sum
            double clear_or_sum = (chunk_idx == 0) ? 0.0 : 1.0;

            rff_features_elemental(
                X     + cs * max_atoms * rep_size,
                Q     + cs * max_atoms,
                sizes + cs,
                W, b,
                nc, max_atoms, rep_size, nelements, D,
                LZ);

            kf_dsfrk('N', 'U', 'N',
                     static_cast<blas_int>(D), static_cast<blas_int>(nc),
                     1.0, LZ, static_cast<blas_int>(D),
                     clear_or_sum, ZtZ_rfp);

            cblas_dgemv(CblasRowMajor, CblasTrans,
                        static_cast<int>(nc), static_cast<int>(D),
                        1.0, LZ, static_cast<int>(D),
                        Y + cs, 1,
                        clear_or_sum, ZtY, 1);

            aligned_free_64(LZ);
        }
    }

    // ---- Force loop (serial: rff_gradient_elemental uses OMP) ----
    // Energy loop already ran so beta=1.0 always.
    std::vector<std::size_t> cum_atoms(nmol + 1);
    cum_atoms[0] = 0;
    for (std::size_t i = 0; i < nmol; ++i)
        cum_atoms[i + 1] = cum_atoms[i] + static_cast<std::size_t>(sizes[i]);

    for (std::size_t cs = 0; cs < nmol; cs += force_chunk) {
        const std::size_t ce = std::min(cs + force_chunk, nmol);
        const std::size_t nc = ce - cs;

        std::size_t ngrads_chunk = 0;
        for (std::size_t i = cs; i < ce; ++i)
            ngrads_chunk += 3 * static_cast<std::size_t>(sizes[i]);
        if (ngrads_chunk == 0) continue;

        double *G = aligned_alloc_64(D * ngrads_chunk);

        rff_gradient_elemental(
            X  + cs * max_atoms * rep_size,
            dX + cs * max_atoms * rep_size * max_atoms * 3,
            Q  + cs * max_atoms,
            sizes + cs,
            W, b,
            nc, max_atoms, rep_size, nelements, D,
            ngrads_chunk, G);

        kf_dsfrk('N', 'U', 'T',
                 static_cast<blas_int>(D), static_cast<blas_int>(ngrads_chunk),
                 1.0, G, static_cast<blas_int>(ngrads_chunk),
                 1.0, ZtZ_rfp);

        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + 3 * cum_atoms[cs], 1,
                    1.0, ZtY, 1);

        aligned_free_64(G);
    }
}

}  // namespace kf::rff
