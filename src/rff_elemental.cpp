// C++ standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

// Third-party libraries
#include <omp.h>
#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

// Project headers
#include "aligned_alloc64.hpp"
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
    //   2. Gather those atoms into Xsort (total_e, rep_size)
    //   3. Ze = sqrt(2/D) * cos(Xsort @ W[e] + b[e])
    //   4. Scatter-add Ze rows back into LZ per molecule
    for (std::size_t e = 0; e < nelements; ++e) {

        std::vector<std::size_t> mol_start(nmol);
        std::vector<std::size_t> mol_count(nmol);
        std::size_t total_e = 0;

        for (std::size_t i = 0; i < nmol; ++i) {
            mol_start[i] = total_e;
            std::size_t cnt = 0;
            const int natoms_i = sizes[i];
            for (int j = 0; j < natoms_i; ++j) {
                if (Q[i * max_atoms + j] == static_cast<int>(e)) {
                    ++cnt;
                }
            }
            mol_count[i] = cnt;
            total_e += cnt;
        }

        if (total_e == 0) continue;

        // Gather: Xsort (total_e, rep_size) row-major
        double *Xsort = aligned_alloc_64(total_e * rep_size);
        std::size_t idx = 0;

        for (std::size_t i = 0; i < nmol; ++i) {
            const int natoms_i = sizes[i];
            for (int j = 0; j < natoms_i; ++j) {
                if (Q[i * max_atoms + j] == static_cast<int>(e)) {
                    const double *src = X + (i * max_atoms + j) * rep_size;
                    double *dst = Xsort + idx * rep_size;
                    std::memcpy(dst, src, rep_size * sizeof(double));
                    ++idx;
                }
            }
        }

        // Ze = Xsort @ W[e]:  (total_e, D)
        // W layout: (nelements, rep_size, D) row-major; W[e] at W + e*rep_size*D
        double *Ze = aligned_alloc_64(total_e * D);
        const double *We = W + e * rep_size * D;
        const double *be = b + e * D;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(total_e),
                    static_cast<int>(D),
                    static_cast<int>(rep_size),
                    1.0, Xsort, static_cast<int>(rep_size),
                    We, static_cast<int>(D),
                    0.0, Ze, static_cast<int>(D));

        aligned_free_64(Xsort);

        // Apply cos + bias: Ze[i,d] = cos(Ze[i,d] + b[e][d]) * normalization
        for (std::size_t i = 0; i < total_e; ++i) {
            for (std::size_t d = 0; d < D; ++d) {
                Ze[i * D + d] = std::cos(Ze[i * D + d] + be[d]) * normalization;
            }
        }

        // Scatter-add: for each molecule, sum Ze rows into LZ
        #pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nmol; ++i) {
            if (mol_count[i] == 0) continue;
            const std::size_t start = mol_start[i];
            const std::size_t count = mol_count[i];
            double *lz_row = LZ + i * D;
            for (std::size_t k = start; k < start + count; ++k) {
                const double *ze_row = Ze + k * D;
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
    double *LZtLZ, double *LZtY) {

    if (!X || !Q || !sizes || !W || !b || !Y || !LZtLZ || !LZtY)
        throw std::invalid_argument("rff_gramian_elemental: null pointer");
    if (nmol == 0 || D == 0)
        throw std::invalid_argument("rff_gramian_elemental: zero dimension");
    if (chunk_size == 0)
        throw std::invalid_argument("rff_gramian_elemental: zero chunk_size");

    std::memset(LZtLZ, 0, D * D * sizeof(double));
    std::memset(LZtY, 0, D * sizeof(double));

    for (std::size_t chunk_start = 0; chunk_start < nmol; chunk_start += chunk_size) {
        const std::size_t chunk_end = std::min(chunk_start + chunk_size, nmol);
        const std::size_t this_chunk = chunk_end - chunk_start;

        double *LZ = aligned_alloc_64(this_chunk * D);

        rff_features_elemental(
            X + chunk_start * max_atoms * rep_size,
            Q + chunk_start * max_atoms,
            sizes + chunk_start,
            W, b,
            this_chunk, max_atoms, rep_size, nelements, D,
            LZ);

        // LZtLZ += LZ^T @ LZ  (upper triangle via DSYRK)
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    static_cast<int>(D),
                    static_cast<int>(this_chunk),
                    1.0, LZ, static_cast<int>(D),
                    1.0, LZtLZ, static_cast<int>(D));

        // LZtY += LZ^T @ Y_chunk
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    static_cast<int>(this_chunk),
                    static_cast<int>(D),
                    1.0, LZ, static_cast<int>(D),
                    Y + chunk_start, 1,
                    1.0, LZtY, 1);

        aligned_free_64(LZ);
    }

    // Symmetrize: copy upper triangle to lower
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            LZtLZ[j * D + i] = LZtLZ[i * D + j];
        }
    }
}

void rff_gramian_elemental_gradient(
    const double *X, const double *dX,
    const int *Q, const int *sizes,
    const double *W, const double *b,
    const double *Y, const double *F,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t energy_chunk, std::size_t force_chunk,
    double *LZtLZ, double *LZtY) {

    if (!X || !dX || !Q || !sizes || !W || !b || !Y || !F || !LZtLZ || !LZtY)
        throw std::invalid_argument("rff_gramian_elemental_gradient: null pointer");
    if (nmol == 0 || D == 0)
        throw std::invalid_argument("rff_gramian_elemental_gradient: zero dimension");

    std::memset(LZtLZ, 0, D * D * sizeof(double));
    std::memset(LZtY, 0, D * sizeof(double));

    // ---- Energy loop (chunked) ----
    for (std::size_t cs = 0; cs < nmol; cs += energy_chunk) {
        const std::size_t ce = std::min(cs + energy_chunk, nmol);
        const std::size_t nc = ce - cs;

        double *LZ = aligned_alloc_64(nc * D);

        rff_features_elemental(
            X + cs * max_atoms * rep_size,
            Q + cs * max_atoms,
            sizes + cs,
            W, b,
            nc, max_atoms, rep_size, nelements, D,
            LZ);

        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    static_cast<int>(D), static_cast<int>(nc),
                    1.0, LZ, static_cast<int>(D),
                    1.0, LZtLZ, static_cast<int>(D));

        cblas_dgemv(CblasRowMajor, CblasTrans,
                    static_cast<int>(nc), static_cast<int>(D),
                    1.0, LZ, static_cast<int>(D),
                    Y + cs, 1,
                    1.0, LZtY, 1);

        aligned_free_64(LZ);
    }

    // ---- Force loop (chunked) ----
    // Precompute cumulative atom offsets for F indexing
    std::vector<std::size_t> cum_atoms(nmol + 1);
    cum_atoms[0] = 0;
    for (std::size_t i = 0; i < nmol; ++i) {
        cum_atoms[i + 1] = cum_atoms[i] + static_cast<std::size_t>(sizes[i]);
    }

    for (std::size_t cs = 0; cs < nmol; cs += force_chunk) {
        const std::size_t ce = std::min(cs + force_chunk, nmol);
        const std::size_t nc = ce - cs;

        std::size_t ngrads_chunk = 0;
        for (std::size_t i = cs; i < ce; ++i) {
            ngrads_chunk += 3 * static_cast<std::size_t>(sizes[i]);
        }
        if (ngrads_chunk == 0) continue;

        double *G = aligned_alloc_64(D * ngrads_chunk);

        rff_gradient_elemental(
            X + cs * max_atoms * rep_size,
            dX + cs * max_atoms * rep_size * max_atoms * 3,
            Q + cs * max_atoms,
            sizes + cs,
            W, b,
            nc, max_atoms, rep_size, nelements, D,
            ngrads_chunk,
            G);

        // LZtLZ += G @ G^T (upper triangle via DSYRK, NoTrans since G is (D, ngrads))
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    1.0, LZtLZ, static_cast<int>(D));

        // LZtY += G @ F_chunk
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(D), static_cast<int>(ngrads_chunk),
                    1.0, G, static_cast<int>(ngrads_chunk),
                    F + 3 * cum_atoms[cs], 1,
                    1.0, LZtY, 1);

        aligned_free_64(G);
    }

    // Symmetrize
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t j = i + 1; j < D; ++j) {
            LZtLZ[j * D + i] = LZtLZ[i * D + j];
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

    // dZ: (D,) — forward pass scratch for one atom
    // dg: (D, rep_size) row-major — gradient kernel matrix for one atom
    std::vector<double> dZ(D);
    double *dg = aligned_alloc_64(D * rep_size);

    for (std::size_t i = 0; i < nmol; ++i) {
        const int natoms = sizes[i];
        if (natoms == 0) continue;

        const std::size_t ncols = 3 * static_cast<std::size_t>(natoms);

        // dX_atom (rep_size, 3*natoms): reshaped derivatives for atom j
        double *dX_atom = aligned_alloc_64(rep_size * ncols);

        // Strides for dX (nmol, max_atoms, rep_size, max_atoms, 3) row-major
        const std::size_t dX_mol_stride  = max_atoms * rep_size * max_atoms * 3;
        const std::size_t dX_atom_stride = rep_size  * max_atoms * 3;

        for (int j = 0; j < natoms; ++j) {
            const int e = Q[i * max_atoms + j];
            const double *We  = W + static_cast<std::size_t>(e) * rep_size * D;
            const double *be  = b + static_cast<std::size_t>(e) * D;
            const double *Xij = X + (i * max_atoms + j) * rep_size;

            // dZ = b[e] + X[i,j,:] @ W[e]
            std::memcpy(dZ.data(), be, D * sizeof(double));
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        static_cast<int>(rep_size), static_cast<int>(D),
                        1.0, We, static_cast<int>(D),
                        Xij, 1,
                        1.0, dZ.data(), 1);

            // dg[d, r] = sin(dZ[d]) * normalization * We[r, d]
            for (std::size_t d = 0; d < D; ++d) {
                const double factor = std::sin(dZ[d]) * normalization;
                for (std::size_t r = 0; r < rep_size; ++r) {
                    dg[d * rep_size + r] = factor * We[r * D + d];
                }
            }

            // Gather dX_atom[r, k*3+xyz] = dX[i, j, r, k, xyz]
            for (std::size_t r = 0; r < rep_size; ++r) {
                for (int k = 0; k < natoms; ++k) {
                    for (int xyz = 0; xyz < 3; ++xyz) {
                        const std::size_t src_idx =
                            i * dX_mol_stride +
                            static_cast<std::size_t>(j) * dX_atom_stride +
                            r * (max_atoms * 3) +
                            static_cast<std::size_t>(k) * 3 +
                            static_cast<std::size_t>(xyz);
                        dX_atom[r * ncols +
                                static_cast<std::size_t>(k) * 3 +
                                static_cast<std::size_t>(xyz)] = dX[src_idx];
                    }
                }
            }

            // G[:, g_start[i] : g_start[i]+3*natoms] += -1 * dg @ dX_atom
            // dg (D, rep_size) @ dX_atom (rep_size, 3*natoms) -> (D, 3*natoms)
            // G is (D, ngrads) row-major, ldc = ngrads
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(D),
                        static_cast<int>(ncols),
                        static_cast<int>(rep_size),
                        -1.0,
                        dg,      static_cast<int>(rep_size),
                        dX_atom, static_cast<int>(ncols),
                        1.0,
                        G + g_start[i],
                        static_cast<int>(ngrads));
        }

        aligned_free_64(dX_atom);
    }

    aligned_free_64(dg);
}

}  // namespace kf::rff
