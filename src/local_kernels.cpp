// Own header
#include "local_kernels.hpp"

// C++ standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// Third-party libraries
#include <omp.h>

// Project headers
#include "aligned_alloc64.hpp"
#include "blas_config.h"

namespace kf {
namespace fchl19 {

//  #########################
//  # FCHL19 KERNEL HELPERS #
//  #########################

// Distinct labels across both sets, with q1 (nm1,max_atoms1) and q2 (nm2,max_atoms2)
static inline void collect_distinct_labels_T(
    const std::vector<int> &q1, int nm1, int max_atoms1, const std::vector<int> &n1,
    const std::vector<int> &q2, int nm2, int max_atoms2, const std::vector<int> &n2,
    std::vector<int> &labels_out
) {
    std::unordered_map<int, char> seen;
    seen.reserve(128);

    for (int a = 0; a < nm1; ++a) {
        int na = std::min(std::max(n1[a], 0), max_atoms1);
        for (int j = 0; j < na; ++j) {
            int lbl = q1[(std::size_t)a * max_atoms1 + j];
            seen.emplace(lbl, 1);
        }
    }
    for (int b = 0; b < nm2; ++b) {
        int nb = std::min(std::max(n2[b], 0), max_atoms2);
        for (int j = 0; j < nb; ++j) {
            int lbl = q2[(std::size_t)b * max_atoms2 + j];
            seen.emplace(lbl, 1);
        }
    }
    labels_out.clear();
    labels_out.reserve(seen.size());
    for (auto &kv : seen)
        labels_out.push_back(kv.first);
    std::sort(labels_out.begin(), labels_out.end());
}

// Pack for asymmetric case with q1 (nm1,max_atoms1), q2 (nm2,max_atoms2)
struct PackedLabel {
    // Dense blocks
    std::vector<double> A;       // (R x rep_size), rows from set 1
    std::vector<double> B;       // (S x rep_size), rows from set 2
    std::vector<double> row_n2;  // (R) for A rows
    std::vector<double> col_n2;  // (S) for B rows
    int R = 0, S = 0;

    // Per-molecule mapping to row/col indices
    std::vector<std::vector<int>> rows_per_mol1;  // size nm1
    std::vector<std::vector<int>> cols_per_mol2;  // size nm2
};

static inline PackedLabel pack_label_block_T(
    int label, const std::vector<double> &x1, int nm1, int max_atoms1, int rep_size,
    const std::vector<double> &x2, int nm2, int max_atoms2, const std::vector<int> &q1,
    const std::vector<int> &q2, const std::vector<int> &n1, const std::vector<int> &n2
) {
    PackedLabel pk;
    pk.rows_per_mol1.resize(nm1);
    pk.cols_per_mol2.resize(nm2);

    // Count rows/cols
    int R = 0, S = 0;
    for (int a = 0; a < nm1; ++a) {
        const int na = std::min(std::max(n1[a], 0), max_atoms1);
        for (int j = 0; j < na; ++j)
            if (q1[(std::size_t)a * max_atoms1 + j] == label) ++R;
    }
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::min(std::max(n2[b], 0), max_atoms2);
        for (int j = 0; j < nb; ++j)
            if (q2[(std::size_t)b * max_atoms2 + j] == label) ++S;
    }
    pk.R = R;
    pk.S = S;
    if (R == 0 || S == 0) return pk;

    pk.A.resize((std::size_t)R * rep_size);
    pk.B.resize((std::size_t)S * rep_size);
    pk.row_n2.resize(R);
    pk.col_n2.resize(S);

    // Pack A (set 1)
    int ridx = 0;
    for (int a = 0; a < nm1; ++a) {
        const int na = std::min(std::max(n1[a], 0), max_atoms1);
        for (int j = 0; j < na; ++j) {
            if (q1[(std::size_t)a * max_atoms1 + j] != label) continue;

            const std::size_t base1 =
                (std::size_t)a * max_atoms1 * rep_size + (std::size_t)j * rep_size;
            double n2sum = 0.0;
            double *Ai = &pk.A[(std::size_t)ridx * rep_size];
            for (int k = 0; k < rep_size; ++k) {
                const double v = x1[base1 + k];
                Ai[k] = v;
                n2sum += v * v;
            }
            pk.row_n2[ridx] = n2sum;
            pk.rows_per_mol1[a].push_back(ridx);
            ++ridx;
        }
    }

    // Pack B (set 2)
    int cidx = 0;
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::min(std::max(n2[b], 0), max_atoms2);
        for (int j = 0; j < nb; ++j) {
            if (q2[(std::size_t)b * max_atoms2 + j] != label) continue;

            const std::size_t base2 =
                (std::size_t)b * max_atoms2 * rep_size + (std::size_t)j * rep_size;
            double n2sum = 0.0;
            double *Bj = &pk.B[(std::size_t)cidx * rep_size];
            for (int k = 0; k < rep_size; ++k) {
                const double v = x2[base2 + k];
                Bj[k] = v;
                n2sum += v * v;
            }
            pk.col_n2[cidx] = n2sum;
            pk.cols_per_mol2[b].push_back(cidx);
            ++cidx;
        }
    }

    return pk;
}

//  #################################
//  # FCHL19 KERNEL IMPLEMENTATION #
//  #################################

void kernel_gaussian(
    const std::vector<double> &x1,  // (nm1, max_atoms1, rep_size)
    const std::vector<double> &x2,  // (nm2, max_atoms2, rep_size)
    const std::vector<int> &q1,     // (nm1, max_atoms1)
    const std::vector<int> &q2,     // (nm2, max_atoms2)
    const std::vector<int> &n1,     // (nm1)
    const std::vector<int> &n2,     // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size, double sigma,
    double *kernel  // (nm1, nm2), row-major: kernel[a*nm2 + b]
) {
    if (!kernel) throw std::invalid_argument("kernel_out is null");
    if (!(std::isfinite(sigma)) || sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_pack = 0.0, t_dgemm = 0.0, t_scatter = 0.0;
    const double t_func_start = omp_get_wtime();
#endif

    // Zero K (full rectangular matrix)
    std::memset(kernel, 0, sizeof(double) * (size_t)nm1 * nm2);

    const double inv_sigma2 = -1.0 / (2.0 * sigma * sigma);

    // Gather labels shared by the two sets
    std::vector<int> labels;
    collect_distinct_labels_T(q1, nm1, max_atoms1, n1, q2, nm2, max_atoms2, n2, labels);
    if (labels.empty()) return;

    // ---- Tunable tile size ----
    const int B = 8192;

    for (int label : labels) {
#ifdef KERNELFORGE_ENABLE_PROFILING
        double t0 = omp_get_wtime();
#endif
        // Pack rows/cols for this label
        auto pk = pack_label_block_T(
            label,
            x1,
            nm1,
            max_atoms1,
            rep_size,
            x2,
            nm2,
            max_atoms2,
            q1,
            q2,
            n1,
            n2
        );
        const int R = pk.R, S = pk.S;
        if (R == 0 || S == 0) continue;

        const int nblkR = (R + B - 1) / B;
        const int nblkS = (S + B - 1) / B;

        // Bucket molecule rows/cols by block id to avoid inner-range checks
        std::vector<std::vector<std::vector<int>>> bucketsR(
            nm1,
            std::vector<std::vector<int>>(nblkR)
        );
        std::vector<std::vector<std::vector<int>>> bucketsS(
            nm2,
            std::vector<std::vector<int>>(nblkS)
        );
        for (int a = 0; a < nm1; ++a) {
            for (int gi : pk.rows_per_mol1[a])
                bucketsR[a][gi / B].push_back(gi);
        }
        for (int b = 0; b < nm2; ++b) {
            for (int gj : pk.cols_per_mol2[b])
                bucketsS[b][gj / B].push_back(gj);
        }

        // Reusable aligned scratch tile (B×B), written anew per dgemm call
        double *Cblk = aligned_alloc_64((size_t)B * B);
        if (!Cblk) throw std::bad_alloc();
#ifdef KERNELFORGE_ENABLE_PROFILING
        t_pack += omp_get_wtime() - t0;
#endif

        // ---- Tile over (rows, cols) ----
        for (int i0 = 0; i0 < R; i0 += B) {
            const int ib = std::min(B, R - i0);
            const double *Ai0 = &pk.A[(size_t)i0 * rep_size];
            const int bi = i0 / B;

            for (int j0 = 0; j0 < S; j0 += B) {
                const int jb = std::min(B, S - j0);
                const double *Bj0 = &pk.B[(size_t)j0 * rep_size];
                const int bj = j0 / B;

                // Cblk(ib×jb) = -2 * Ai0(ib×rep) * Bj0(jb×rep)^T
#ifdef KERNELFORGE_ENABLE_PROFILING
                t0 = omp_get_wtime();
#endif
                cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    static_cast<blas_int>(ib),
                    static_cast<blas_int>(jb),
                    static_cast<blas_int>(rep_size),
                    -2.0,
                    Ai0,
                    static_cast<blas_int>(rep_size),
                    Bj0,
                    static_cast<blas_int>(rep_size),
                    0.0,
                    Cblk,
                    static_cast<blas_int>(jb)
                );
#ifdef KERNELFORGE_ENABLE_PROFILING
                t_dgemm += omp_get_wtime() - t0;
                t0 = omp_get_wtime();
#endif

                // Accumulate this tile into K
#pragma omp parallel for schedule(guided)
                for (int a = 0; a < nm1; ++a) {
                    const auto &Ia = bucketsR[a][bi];
                    if (Ia.empty()) continue;

                    for (int b = 0; b < nm2; ++b) {
                        const auto &Jb = bucketsS[b][bj];
                        if (Jb.empty()) continue;

                        double kab = 0.0;
                        const int mJ = (int)Jb.size();

                        for (int gi : Ia) {
                            const int li = gi - i0;
                            const double rn = pk.row_n2[gi];
                            const double *__restrict Grow = Cblk + (size_t)li * jb;

                            // 4-way unroll with simd hint on the remainder
                            int t = 0;
                            for (; t + 3 < mJ; t += 4) {
                                const int j0g = Jb[t + 0], j1g = Jb[t + 1];
                                const int j2g = Jb[t + 2], j3g = Jb[t + 3];
                                kab +=
                                    std::exp((rn + pk.col_n2[j0g] + Grow[j0g - j0]) * inv_sigma2) +
                                    std::exp((rn + pk.col_n2[j1g] + Grow[j1g - j0]) * inv_sigma2) +
                                    std::exp((rn + pk.col_n2[j2g] + Grow[j2g - j0]) * inv_sigma2) +
                                    std::exp((rn + pk.col_n2[j3g] + Grow[j3g - j0]) * inv_sigma2);
                            }
                            for (; t < mJ; ++t) {
                                const int jg = Jb[t];
                                kab += std::exp((rn + pk.col_n2[jg] + Grow[jg - j0]) * inv_sigma2);
                            }
                        }

                        // Row-major (nm1, nm2): stride-1 across 'b'
                        kernel[(size_t)a * nm2 + b] += kab;
                    }
                }
#ifdef KERNELFORGE_ENABLE_PROFILING
                t_scatter += omp_get_wtime() - t0;
#endif
            }  // j0
        }  // i0

        std::free(Cblk);
    }  // labels

#ifdef KERNELFORGE_ENABLE_PROFILING
    const double t_total = omp_get_wtime() - t_func_start;
    std::printf(
        "[PROFILE] kernel_gaussian(nm1=%d, nm2=%d, rep=%d):\n"
        "  Pack/bucket:  %8.4fs (%5.1f%%)\n"
        "  DGEMM:        %8.4fs (%5.1f%%)\n"
        "  Exp+scatter:  %8.4fs (%5.1f%%)\n"
        "  Total:        %8.4fs (100.0%%)\n",
        nm1,
        nm2,
        rep_size,
        t_pack,
        100.0 * t_pack / t_total,
        t_dgemm,
        100.0 * t_dgemm / t_total,
        t_scatter,
        100.0 * t_scatter / t_total,
        t_total
    );
#endif
}

//  ###################################
//  # FCHL19 SYMMETRIC KERNEL HELPERS #
//  ###################################

// --- helper: collect labels for single set with q shape (nm, max_atoms)
static inline void collect_distinct_labels_single_T(
    const std::vector<int> &q, int nm, int max_atoms, const std::vector<int> &n,
    std::vector<int> &labels_out
) {
    std::unordered_map<int, char> seen;
    seen.reserve(128);
    for (int a = 0; a < nm; ++a) {
        int na = std::min(std::max(n[a], 0), max_atoms);
        for (int j = 0; j < na; ++j) {
            seen.emplace(q[(std::size_t)a * max_atoms + j], 1);
        }
    }
    labels_out.clear();
    labels_out.reserve(seen.size());
    for (auto &kv : seen)
        labels_out.push_back(kv.first);
    std::sort(labels_out.begin(), labels_out.end());
}

// --- pack for symmetric case (single set), q shape (nm, max_atoms)
struct PackedLabelSym {
    std::vector<double> A;                       // (R x rep_size), row-major
    std::vector<double> row_n2;                  // (R)
    std::vector<std::vector<int>> rows_per_mol;  // size nm
    int R = 0;
};

static inline PackedLabelSym pack_label_block_sym_T(
    int label, const std::vector<double> &x, int nm, int max_atoms, int rep_size,
    const std::vector<int> &q, const std::vector<int> &n
) {
    PackedLabelSym pk;
    pk.rows_per_mol.resize(nm);

    // count rows
    int R = 0;
    for (int a = 0; a < nm; ++a) {
        const int na = std::min(std::max(n[a], 0), max_atoms);
        for (int j = 0; j < na; ++j)
            if (q[(std::size_t)a * max_atoms + j] == label) ++R;
    }
    pk.R = R;
    if (R == 0) return pk;

    pk.A.resize((std::size_t)R * rep_size);
    pk.row_n2.resize(R);

    // pack rows
    int ridx = 0;
    for (int a = 0; a < nm; ++a) {
        const int na = std::min(std::max(n[a], 0), max_atoms);
        for (int j = 0; j < na; ++j) {
            if (q[(std::size_t)a * max_atoms + j] != label) continue;

            const std::size_t base =
                (std::size_t)a * max_atoms * rep_size + (std::size_t)j * rep_size;

            double n2sum = 0.0;
            double *Ai = &pk.A[(std::size_t)ridx * rep_size];
            for (int k = 0; k < rep_size; ++k) {
                const double v = x[base + k];
                Ai[k] = v;
                n2sum += v * v;
            }
            pk.row_n2[ridx] = n2sum;
            pk.rows_per_mol[a].push_back(ridx);
            ++ridx;
        }
    }
    return pk;
}

// assume you already have:
//   double* aligned_alloc_64(size_t n);  // aligned, uninitialized; free with std::free

//  #######################################
//  # FCHL19 KERNEL PACKED IMPLEMENTATION #
//  #######################################

// Map (i,j) with 0-based indices and i <= j to linear 0-based RFP index.
// Matches your Fortran rfp_index (TRANSR='N', UPLO='U').
static inline size_t rfp_index_upper_N(int n, int i, int j) {
    // Preconditions: 0 <= i <= j < n
    // if (i > j) std::swap(i, j);
    const int k = n / 2;                            // floor(n/2)
    const int stride = (n % 2 == 0) ? (n + 1) : n;  // even->n+1, odd->n

    // "top" zone if j >= k  (same as your Fortran)
    // top:   idx = (j-k)*stride + i
    // bottom:idx = i*stride + j + k + 1
    if (j >= k) {
        return (size_t)(j - k) * (size_t)stride + (size_t)i;
    } else {
        return (size_t)i * (size_t)stride + (size_t)(j + k + 1);
    }
}

// Tiled, memory-bounded version that writes **directly to RFP** (TRANSR='N', UPLO='U')
void kernel_gaussian_symm_rfp(
    const std::vector<double> &x,  // (nm, max_atoms, rep_size)
    const std::vector<int> &q,     // (nm, max_atoms)
    const std::vector<int> &n,     // (nm)
    int nm, int max_atoms, int rep_size, double sigma,
    double *arf  // length nt = nm*(nm+1)/2, output RFP (TRANSR='N', UPLO='U')
) {
    if (!arf) throw std::invalid_argument("arf (RFP output) is null");
    if (!(std::isfinite(sigma)) || sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_pack = 0.0, t_dsyrk = 0.0, t_diag_scatter = 0.0, t_dgemm = 0.0,
           t_offdiag_scatter = 0.0;
    const double t_func_start = omp_get_wtime();
#endif

    // Zero the RFP vector
    const size_t nt = (size_t)nm * (nm + 1ull) / 2ull;
    std::memset(arf, 0, nt * sizeof(double));

    const double inv_sigma2 = -1.0 / (2.0 * sigma * sigma);

    // Collect labels
    std::vector<int> labels;
    collect_distinct_labels_single_T(q, nm, max_atoms, n, labels);
    if (labels.empty()) return;

    // Tile size
    const int B = 8192;

    for (int label : labels) {
#ifdef KERNELFORGE_ENABLE_PROFILING
        double t0 = omp_get_wtime();
#endif
        PackedLabelSym pk = pack_label_block_sym_T(label, x, nm, max_atoms, rep_size, q, n);
        const int R = pk.R;
        if (R == 0) continue;

        const int num_blocks = (R + B - 1) / B;

        // Bucket molecule rows by block id to avoid inner checks
        std::vector<std::vector<std::vector<int>>> buckets(
            nm,
            std::vector<std::vector<int>>(num_blocks)
        );
        for (int a = 0; a < nm; ++a) {
            const auto &rows = pk.rows_per_mol[a];
            for (int gi : rows)
                buckets[a][gi / B].push_back(gi);
        }

        // Scratch tile
        double *Cblk = aligned_alloc_64((size_t)B * B);
        if (!Cblk) throw std::bad_alloc();
#ifdef KERNELFORGE_ENABLE_PROFILING
        t_pack += omp_get_wtime() - t0;
#endif

        for (int i0 = 0; i0 < R; i0 += B) {
            const int ib = std::min(B, R - i0);
            const double *Ai0 = &pk.A[(size_t)i0 * rep_size];
            const int bi = i0 / B;

            // ----- Diagonal tile: DSYRK → upper-tri in Cblk (LDC = ib) -----
#ifdef KERNELFORGE_ENABLE_PROFILING
            t0 = omp_get_wtime();
#endif
            cblas_dsyrk(
                CblasRowMajor,
                CblasUpper,
                CblasNoTrans,
                static_cast<blas_int>(ib),
                static_cast<blas_int>(rep_size),
                -2.0,
                Ai0,
                static_cast<blas_int>(rep_size),
                0.0,
                Cblk,
                static_cast<blas_int>(ib)
            );
#ifdef KERNELFORGE_ENABLE_PROFILING
            t_dsyrk += omp_get_wtime() - t0;
            t0 = omp_get_wtime();
#endif

#pragma omp parallel for schedule(guided)
            for (int a = 0; a < nm; ++a) {
                const auto &Ia = buckets[a][bi];
                if (Ia.empty()) continue;

                for (int b = a; b < nm; ++b) {
                    const auto &Ib = buckets[b][bi];
                    if (Ib.empty()) continue;

                    double kab = 0.0;
                    for (int gi : Ia) {
                        const int li = gi - i0;  // [0..ib)
                        const double rn_i = pk.row_n2[gi];
                        // Diagonal tile: need upper-tri index (li <= lj branch).
                        // Branch prevents full unroll; use simd hint on the inner loop.
#pragma omp simd reduction(+ : kab)
                        for (int t = 0; t < (int)Ib.size(); ++t) {
                            const int gj = Ib[t];
                            const int lj = gj - i0;
                            const int r = (li <= lj) ? li : lj;
                            const int c = (li <= lj) ? lj : li;
                            kab += std::exp(
                                (rn_i + pk.row_n2[gj] + Cblk[(size_t)r * ib + c]) * inv_sigma2
                            );
                        }
                    }
                    // Write once to RFP (a<=b)
                    const size_t idx = rfp_index_upper_N(nm, a, b);
                    arf[idx] += kab;
                }
            }
#ifdef KERNELFORGE_ENABLE_PROFILING
            t_diag_scatter += omp_get_wtime() - t0;
#endif

            // ----- Off-diagonal rectangles: DGEMM (ib × jb) -----
            for (int j0 = i0 + B; j0 < R; j0 += B) {
                const int jb = std::min(B, R - j0);
                const double *Aj0 = &pk.A[(size_t)j0 * rep_size];
                const int bj = j0 / B;

#ifdef KERNELFORGE_ENABLE_PROFILING
                t0 = omp_get_wtime();
#endif
                cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    static_cast<blas_int>(ib),
                    static_cast<blas_int>(jb),
                    static_cast<blas_int>(rep_size),
                    -2.0,
                    Ai0,
                    static_cast<blas_int>(rep_size),
                    Aj0,
                    static_cast<blas_int>(rep_size),
                    0.0,
                    Cblk,
                    static_cast<blas_int>(jb)
                );
#ifdef KERNELFORGE_ENABLE_PROFILING
                t_dgemm += omp_get_wtime() - t0;
                t0 = omp_get_wtime();
#endif

#pragma omp parallel for schedule(guided)
                for (int a = 0; a < nm; ++a) {
                    const auto &Ia = buckets[a][bi];
                    if (Ia.empty()) continue;

                    for (int b = a; b < nm; ++b) {
                        const auto &Jb = buckets[b][bj];
                        if (Jb.empty()) continue;

                        double kab = 0.0;
                        const int mJ = (int)Jb.size();

                        for (int gi : Ia) {
                            const int li = gi - i0;
                            const double rn_i = pk.row_n2[gi];
                            const double *__restrict Crow = Cblk + (size_t)li * jb;

                            // 4-way unroll: off-diagonal tile has a plain rectangular
                            // Cblk with no branch, so full unroll is safe.
                            int t = 0;
                            for (; t + 3 < mJ; t += 4) {
                                const int gj0 = Jb[t + 0], gj1 = Jb[t + 1];
                                const int gj2 = Jb[t + 2], gj3 = Jb[t + 3];
                                kab +=
                                    std::exp(
                                        (rn_i + pk.row_n2[gj0] + Crow[gj0 - j0]) * inv_sigma2
                                    ) +
                                    std::exp(
                                        (rn_i + pk.row_n2[gj1] + Crow[gj1 - j0]) * inv_sigma2
                                    ) +
                                    std::exp(
                                        (rn_i + pk.row_n2[gj2] + Crow[gj2 - j0]) * inv_sigma2
                                    ) +
                                    std::exp((rn_i + pk.row_n2[gj3] + Crow[gj3 - j0]) * inv_sigma2);
                            }
                            for (; t < mJ; ++t) {
                                const int gj = Jb[t];
                                kab +=
                                    std::exp((rn_i + pk.row_n2[gj] + Crow[gj - j0]) * inv_sigma2);
                            }
                        }
                        // Self-kernel cross-block pairs counted twice
                        if (a == b) kab *= 2.0;

                        const size_t idx = rfp_index_upper_N(nm, a, b);
                        arf[idx] += kab;
                    }
                }
#ifdef KERNELFORGE_ENABLE_PROFILING
                t_offdiag_scatter += omp_get_wtime() - t0;
#endif
            }  // j0
        }  // i0

        std::free(Cblk);
    }  // labels

#ifdef KERNELFORGE_ENABLE_PROFILING
    const double t_total = omp_get_wtime() - t_func_start;
    std::printf(
        "[PROFILE] kernel_gaussian_symm_rfp(nm=%d, rep=%d):\n"
        "  Pack/bucket:      %8.4fs (%5.1f%%)\n"
        "  DSYRK (diag):    %8.4fs (%5.1f%%)\n"
        "  Diag scatter:    %8.4fs (%5.1f%%)\n"
        "  DGEMM (offdiag): %8.4fs (%5.1f%%)\n"
        "  Offdiag scatter: %8.4fs (%5.1f%%)\n"
        "  Total:           %8.4fs (100.0%%)\n",
        nm,
        rep_size,
        t_pack,
        100.0 * t_pack / t_total,
        t_dsyrk,
        100.0 * t_dsyrk / t_total,
        t_diag_scatter,
        100.0 * t_diag_scatter / t_total,
        t_dgemm,
        100.0 * t_dgemm / t_total,
        t_offdiag_scatter,
        100.0 * t_offdiag_scatter / t_total,
        t_total
    );
#endif
}

//  ###########################################
//  # FCHL19 KERNEL SYMMETRIC IMPLEMENTATION #
//  ###########################################

// Tiled, memory-bounded version: only allocates a B×B scratch tile.
void kernel_gaussian_symm(
    const std::vector<double> &x,  // (nm, max_atoms, rep_size)
    const std::vector<int> &q,     // (nm, max_atoms)
    const std::vector<int> &n,     // (nm)
    int nm, int max_atoms, int rep_size, double sigma,
    double *kernel  // (nm, nm), row-major: kernel[a*nm + b]
) {
    if (!kernel) throw std::invalid_argument("kernel_out is null");
    if (!(std::isfinite(sigma)) || sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_pack = 0.0, t_dsyrk = 0.0, t_diag_scatter = 0.0, t_dgemm = 0.0,
           t_offdiag_scatter = 0.0;
    const double t_func_start = omp_get_wtime();
#endif

// Zero once (full matrix; cheap and simple). If you only ever write the lower
// triangle, you can halve this by zeroing rows up to i inclusive.
// Initialize kernel matrix to zero
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)nm * nm; ++i) {
        kernel[i] = 0.0;
    }

    const double inv_sigma2 = -1.0 / (2.0 * sigma * sigma);

    std::vector<int> labels;
    collect_distinct_labels_single_T(q, nm, max_atoms, n, labels);
    if (labels.empty()) return;

    const int B = 8192;

    for (int label : labels) {
#ifdef KERNELFORGE_ENABLE_PROFILING
        double t0 = omp_get_wtime();
#endif
        PackedLabelSym pk = pack_label_block_sym_T(label, x, nm, max_atoms, rep_size, q, n);
        const int R = pk.R;
        if (R == 0) continue;

        const int num_blocks = (R + B - 1) / B;

        // Bucket molecule rows by block id
        std::vector<std::vector<std::vector<int>>> buckets(
            nm,
            std::vector<std::vector<int>>(num_blocks)
        );
        for (int a = 0; a < nm; ++a) {
            const auto &rows = pk.rows_per_mol[a];
            for (int gi : rows)
                buckets[a][gi / B].push_back(gi);
        }

        // Aligned scratch tile (reused for all tiles of this label)
        double *Cblk = aligned_alloc_64((size_t)B * B);
        if (!Cblk) throw std::bad_alloc();
#ifdef KERNELFORGE_ENABLE_PROFILING
        t_pack += omp_get_wtime() - t0;
#endif

        for (int i0 = 0; i0 < R; i0 += B) {
            const int ib = std::min(B, R - i0);
            const double *Ai0 = &pk.A[(size_t)i0 * rep_size];
            const int bi = i0 / B;

            // Diagonal tile: DSYRK produces upper-tri in Cblk with LDC=ib
#ifdef KERNELFORGE_ENABLE_PROFILING
            t0 = omp_get_wtime();
#endif
            cblas_dsyrk(
                CblasRowMajor,
                CblasUpper,
                CblasNoTrans,
                static_cast<blas_int>(ib),
                static_cast<blas_int>(rep_size),
                -2.0,
                Ai0,
                static_cast<blas_int>(rep_size),
                0.0,
                Cblk,
                static_cast<blas_int>(ib)
            );
#ifdef KERNELFORGE_ENABLE_PROFILING
            t_dsyrk += omp_get_wtime() - t0;
            t0 = omp_get_wtime();
#endif

#pragma omp parallel for schedule(guided)
            for (int a = 0; a < nm; ++a) {
                const auto &Ia = buckets[a][bi];
                if (Ia.empty()) continue;

                for (int b = a; b < nm; ++b) {
                    const auto &Ib = buckets[b][bi];
                    if (Ib.empty()) continue;

                    double kab = 0.0;
                    for (int gi : Ia) {
                        const int li = gi - i0;  // [0..ib)
                        const double rn_i = pk.row_n2[gi];
                        // Diagonal tile: upper-tri index requires branch on li vs lj.
                        // Use simd hint; compiler can still vectorise the exp with SVML.
#pragma omp simd reduction(+ : kab)
                        for (int t = 0; t < (int)Ib.size(); ++t) {
                            const int gj = Ib[t];
                            const int lj = gj - i0;
                            const int r = (li <= lj) ? li : lj;
                            const int c = (li <= lj) ? lj : li;
                            kab += std::exp(
                                (rn_i + pk.row_n2[gj] + Cblk[(size_t)r * ib + c]) * inv_sigma2
                            );
                        }
                    }
                    kernel[(size_t)a * nm + b] += kab;
                    if (b != a) kernel[(size_t)b * nm + a] += kab;
                }
            }
#ifdef KERNELFORGE_ENABLE_PROFILING
            t_diag_scatter += omp_get_wtime() - t0;
#endif

            // Off-diagonal rectangles: DGEMM to Cblk (LDC=jb)
            for (int j0 = i0 + B; j0 < R; j0 += B) {
                const int jb = std::min(B, R - j0);
                const double *Aj0 = &pk.A[(size_t)j0 * rep_size];
                const int bj = j0 / B;

#ifdef KERNELFORGE_ENABLE_PROFILING
                t0 = omp_get_wtime();
#endif
                cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    static_cast<blas_int>(ib),
                    static_cast<blas_int>(jb),
                    static_cast<blas_int>(rep_size),
                    -2.0,
                    Ai0,
                    static_cast<blas_int>(rep_size),
                    Aj0,
                    static_cast<blas_int>(rep_size),
                    0.0,
                    Cblk,
                    static_cast<blas_int>(jb)
                );
#ifdef KERNELFORGE_ENABLE_PROFILING
                t_dgemm += omp_get_wtime() - t0;
                t0 = omp_get_wtime();
#endif

#pragma omp parallel for schedule(guided)
                for (int a = 0; a < nm; ++a) {
                    const auto &Ia = buckets[a][bi];
                    if (Ia.empty()) continue;

                    for (int b = a; b < nm; ++b) {
                        const auto &Jb = buckets[b][bj];
                        if (Jb.empty()) continue;

                        double kab = 0.0;
                        const int mJ = (int)Jb.size();

                        for (int gi : Ia) {
                            const int li = gi - i0;
                            const double rn_i = pk.row_n2[gi];
                            const double *__restrict Crow = Cblk + (size_t)li * jb;

                            // 4-way unroll: rectangular Cblk, no branch needed.
                            int t = 0;
                            for (; t + 3 < mJ; t += 4) {
                                const int gj0 = Jb[t + 0], gj1 = Jb[t + 1];
                                const int gj2 = Jb[t + 2], gj3 = Jb[t + 3];
                                kab +=
                                    std::exp(
                                        (rn_i + pk.row_n2[gj0] + Crow[gj0 - j0]) * inv_sigma2
                                    ) +
                                    std::exp(
                                        (rn_i + pk.row_n2[gj1] + Crow[gj1 - j0]) * inv_sigma2
                                    ) +
                                    std::exp(
                                        (rn_i + pk.row_n2[gj2] + Crow[gj2 - j0]) * inv_sigma2
                                    ) +
                                    std::exp((rn_i + pk.row_n2[gj3] + Crow[gj3 - j0]) * inv_sigma2);
                            }
                            for (; t < mJ; ++t) {
                                const int gj = Jb[t];
                                kab +=
                                    std::exp((rn_i + pk.row_n2[gj] + Crow[gj - j0]) * inv_sigma2);
                            }
                        }
                        // Count cross-block self-pairs twice to match full (i,j)+(j,i)
                        if (a == b) kab *= 2.0;

                        kernel[(size_t)a * nm + b] += kab;
                        if (b != a) kernel[(size_t)b * nm + a] += kab;
                    }
                }
#ifdef KERNELFORGE_ENABLE_PROFILING
                t_offdiag_scatter += omp_get_wtime() - t0;
#endif
            }  // j0
        }  // i0

        std::free(Cblk);
    }  // labels

#ifdef KERNELFORGE_ENABLE_PROFILING
    const double t_total = omp_get_wtime() - t_func_start;
    std::printf(
        "[PROFILE] kernel_gaussian_symm(nm=%d, rep=%d):\n"
        "  Pack/bucket:      %8.4fs (%5.1f%%)\n"
        "  DSYRK (diag):    %8.4fs (%5.1f%%)\n"
        "  Diag scatter:    %8.4fs (%5.1f%%)\n"
        "  DGEMM (offdiag): %8.4fs (%5.1f%%)\n"
        "  Offdiag scatter: %8.4fs (%5.1f%%)\n"
        "  Total:           %8.4fs (100.0%%)\n",
        nm,
        rep_size,
        t_pack,
        100.0 * t_pack / t_total,
        t_dsyrk,
        100.0 * t_dsyrk / t_total,
        t_diag_scatter,
        100.0 * t_diag_scatter / t_total,
        t_dgemm,
        100.0 * t_dgemm / t_total,
        t_offdiag_scatter,
        100.0 * t_offdiag_scatter / t_total,
        t_total
    );
#endif
}

// helper: flat indices (row-major, last dim fastest)
static inline size_t idx_x1(int a, int j1, int k, int nm1, int max_atoms1, int rep) {
    return (static_cast<size_t>(a) * max_atoms1 + j1) * rep + k;
}
static inline size_t idx_x2(int b, int j2, int k, int nm2, int max_atoms2, int rep) {
    return (static_cast<size_t>(b) * max_atoms2 + j2) * rep + k;
}
static inline size_t idx_q1(int a, int j1, int nm1, int max_atoms1) {
    return static_cast<size_t>(a) * max_atoms1 + j1;
}
static inline size_t idx_q2(int b, int j2, int nm2, int max_atoms2) {
    return static_cast<size_t>(b) * max_atoms2 + j2;
}
// dx1: (nm1, max_atoms1, rep, 3*max_atoms1)
static inline size_t base_dx1(int a, int i1, int nm1, int max_atoms1, int rep) {
    return (((size_t)a * max_atoms1 + i1) * rep) * (3 * (size_t)max_atoms1);
}
// dX2: (nm2, max_atoms2, rep, 3*max_atoms2)
static inline size_t base_dx2(int b, int i2, int nm2, int max_atoms2, int rep) {
    return ((static_cast<size_t>(b) * max_atoms2 + i2) * rep) *
           (3 * static_cast<size_t>(max_atoms2));
}

// Symmetric helpers
static inline size_t idx_x(int m, int j, int k, int nm, int max_atoms, int rep) {
    return ((size_t)m * max_atoms + j) * rep + (size_t)k;
}
static inline size_t base_dx(int m, int j, int nm, int max_atoms, int rep) {
    return (((size_t)m * max_atoms + j) * rep) * (3 * (size_t)max_atoms);
}

void kernel_gaussian_jacobian(
    const std::vector<double> &x1,   // (nm1, max_atoms1, rep)
    const std::vector<double> &x2,   // (nm2, max_atoms2, rep)
    const std::vector<double> &dX2,  // (nm2, max_atoms2, rep, 3*max_atoms2)
    const std::vector<int> &q1,      // (nm1, max_atoms1)
    const std::vector<int> &q2,      // (nm2, max_atoms2)
    const std::vector<int> &n1,      // (nm1)
    const std::vector<int> &n2,      // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size, int naq2, double sigma,
    double *kernel_out  // (nm1, naq2)
) {
    // --- validation (unchanged) ---
    if (nm1 <= 0 || nm2 <= 0 || max_atoms1 <= 0 || max_atoms2 <= 0 || rep_size <= 0)
        throw std::invalid_argument("All dims must be positive.");
    if (!std::isfinite(sigma) || sigma <= 0.0)
        throw std::invalid_argument("sigma must be positive and finite.");
    if (!kernel_out) throw std::invalid_argument("kernel_out is null.");

    const size_t x1N = (size_t)nm1 * max_atoms1 * rep_size;
    const size_t x2N = (size_t)nm2 * max_atoms2 * rep_size;
    const size_t dXN = (size_t)nm2 * max_atoms2 * rep_size * (3 * (size_t)max_atoms2);
    const size_t q1N = (size_t)nm1 * max_atoms1;
    const size_t q2N = (size_t)nm2 * max_atoms2;

    if (x1.size() != x1N || x2.size() != x2N) throw std::invalid_argument("x1/x2 size mismatch.");
    if (dX2.size() != dXN) throw std::invalid_argument("dX2 size mismatch.");
    if (q1.size() != q1N || q2.size() != q2N) throw std::invalid_argument("q1/q2 size mismatch.");
    if ((int)n1.size() != nm1 || (int)n2.size() != nm2)
        throw std::invalid_argument("n1/n2 size mismatch.");

    // per-b offsets and ncols (3 * n2[b])
    std::vector<int> offs2(nm2), ncols_b(nm2);
    int acc = 0;
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::max(0, std::min(n2[b], max_atoms2));
        offs2[b] = acc;
        ncols_b[b] = 3 * nb;
        acc += ncols_b[b];
    }
    if (naq2 != acc) throw std::invalid_argument("naq2 != 3*sum(n2)");

    // zero output
    std::fill(kernel_out, kernel_out + (size_t)nm1 * naq2, 0.0);

    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma2 = -1.0 / (sigma * sigma);

    // ------------------------------------------------------------
    // Build per-label list of (a, j1) only for valid j1 < n1[a]
    // ------------------------------------------------------------
    std::unordered_map<int, std::vector<std::pair<int, int>>> lj1;
    lj1.reserve(128);
    for (int a = 0; a < nm1; ++a) {
        const int na = std::max(0, std::min(n1[a], max_atoms1));
        for (int j1 = 0; j1 < na; ++j1) {
            const int lbl = q1[(size_t)a * max_atoms1 + j1];  // q1(a,j1)
            lj1[lbl].emplace_back(a, j1);
        }
    }

    // Tile width: D is (T x rep_size), H is (T x ncols_max).
    // Working set per thread: T*(rep_size + ncols_max)*8 bytes.
    // Target: fit in L2 (512 KB per core). ncols_max = 3*max_atoms2.
    // T=256: 256*(384+27)*8 = 842 KB for ethanol -> fits with prefetch overlap.
    constexpr int BATCH_T = 256;

    // D_scaled: (T x rep_size) row-major — build_D writes D[t*rep+k], sequential per t.
    // H:        (T x ncols_max) row-major — scatter reads H[t*ncols:], sequential per t.
    const int ncols_max = 3 * max_atoms2;

    // ------------------------------------------------------------
    // Parallelize over b: each thread owns a disjoint column block
    // Serialise BLAS inside OMP to prevent oversubscription.
    // ------------------------------------------------------------
    double t_build_D = 0.0, t_dgemm_jac = 0.0, t_scatter = 0.0;
#ifdef KERNELFORGE_ENABLE_PROFILING
    const double t_func_start = omp_get_wtime();
#endif
    const int blas_nt = kf_blas_get_num_threads();
    kf_blas_set_num_threads(1);
#pragma omp parallel default(none) \
    shared(x1,                     \
               x2,                 \
               dX2,                \
               q1,                 \
               q2,                 \
               n1,                 \
               n2,                 \
               nm1,                \
               nm2,                \
               max_atoms1,         \
               max_atoms2,         \
               rep_size,           \
               naq2,               \
               kernel_out,         \
               lj1,                \
               offs2,              \
               ncols_b,            \
               inv_2sigma2,        \
               inv_sigma2,         \
               ncols_max,          \
               t_build_D,          \
               t_dgemm_jac,        \
               t_scatter)
    {
        // Thread-local scratch: D is (T x rep_size), H is (T x ncols_max).
        double *D = aligned_alloc_64((size_t)BATCH_T * rep_size);   // (T x rep_size)
        double *H = aligned_alloc_64((size_t)BATCH_T * ncols_max);  // (T x ncols_max)

#pragma omp for schedule(dynamic)
        for (int b = 0; b < nm2; ++b) {
            const int nb = ncols_b[b] / 3;
            const int ncols = ncols_b[b];
            if (nb == 0) continue;

            const int out_offset = offs2[b];
            const int lda_rowmaj = 3 * max_atoms2;

            for (int j2 = 0; j2 < nb; ++j2) {
                const int label = q2[(size_t)b * max_atoms2 + j2];
                auto it = lj1.find(label);
                if (it == lj1.end() || it->second.empty()) continue;

                const auto &aj1_list = it->second;

                // A = dX2(b, j2, :, :): shape (rep_size, 3*max_atoms2) row-major
                const double *A = &dX2[base_dx2(b, j2, nm2, max_atoms2, rep_size)];
                // x2 row for (b,j2)
                const double *x2_bj2 = &x2[((size_t)b * max_atoms2 + j2) * rep_size];

                // Process (a,j1) pairs in tiles of width BATCH_T
                for (size_t t0 = 0; t0 < aj1_list.size(); t0 += BATCH_T) {
                    const int T = (int)std::min<size_t>(BATCH_T, aj1_list.size() - t0);

                    // 1) Build D (T x rep_size): row t = alpha_t * (x1[a,j1] - x2[b,j2]).
                    //    Writing D[t*rep+k] is sequential for each t — no cache thrash.
#ifdef KERNELFORGE_ENABLE_PROFILING
                    double tp = omp_get_wtime();
#endif
                    for (int t = 0; t < T; ++t) {
                        const int a = aj1_list[t0 + t].first;
                        const int j1 = aj1_list[t0 + t].second;
                        const double *x1_aj1 = &x1[((size_t)a * max_atoms1 + j1) * rep_size];

                        double l2 = 0.0;
                        double *Drow = D + (size_t)t * rep_size;
                        for (int k = 0; k < rep_size; ++k) {
                            const double diff = x1_aj1[k] - x2_bj2[k];
                            Drow[k] = diff;
                            l2 += diff * diff;
                        }
                        const double alpha_t = std::exp(l2 * inv_2sigma2) * inv_sigma2;
                        for (int k = 0; k < rep_size; ++k)
                            Drow[k] *= alpha_t;
                    }
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_build_D += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // 2) H(T x ncols) = D(T x rep) @ A(rep x ncols)
                    //    A is stored as (rep x lda_rowmaj) row-major; we use ncols columns.
                    //    cblas: RowMajor, NoTrans, NoTrans -> H = D * A
                    cblas_dgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        static_cast<blas_int>(T),
                        static_cast<blas_int>(ncols),
                        static_cast<blas_int>(rep_size),
                        1.0,
                        D,
                        static_cast<blas_int>(rep_size),
                        A,
                        static_cast<blas_int>(lda_rowmaj),
                        0.0,
                        H,
                        static_cast<blas_int>(ncols_max)
                    );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_dgemm_jac += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // 3) Scatter-add: H[t, 0:ncols] -> kernel_out[a, out_offset:out_offset+ncols].
                    //    Both reads (H row) and writes (kout) are sequential.
                    for (int t = 0; t < T; ++t) {
                        const int a = aj1_list[t0 + t].first;
                        double *kout = &kernel_out[(size_t)a * naq2 + out_offset];
                        const double *hrow = H + (size_t)t * ncols_max;
                        for (int r = 0; r < ncols; ++r)
                            kout[r] += hrow[r];
                    }
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_scatter += omp_get_wtime() - tp;
#endif
                }  // tiles
            }  // j2
        }  // omp for

        aligned_free_64(D);
        aligned_free_64(H);
    }  // omp parallel
    kf_blas_set_num_threads(blas_nt);

#ifdef KERNELFORGE_ENABLE_PROFILING
    const double t_total = omp_get_wtime() - t_func_start;
    std::printf(
        "[PROFILE] kernel_gaussian_jacobian(nm1=%d, nm2=%d, rep=%d):\n"
        "  Build D (exp+scale):   %8.4fs (%5.1f%%)\n"
        "  DGEMM (H = D @ A):    %8.4fs (%5.1f%%)\n"
        "  Scatter-add to kernel: %8.4fs (%5.1f%%)\n"
        "  Total:                 %8.4fs (100.0%%)\n",
        nm1,
        nm2,
        rep_size,
        t_build_D,
        100.0 * t_build_D / t_total,
        t_dgemm_jac,
        100.0 * t_dgemm_jac / t_total,
        t_scatter,
        100.0 * t_scatter / t_total,
        t_total
    );
#endif
}

// ###################################
// # FCHL19 JACOBIAN KERNEL (TRANSPOSED) #
// ###################################
// kernel_gaussian_jacobian_t: Jacobians on set-1 side (dX1).
// Output shape: (naq1, nm2), where naq1 = 3 * sum(n1).
//
// Mathematical relationship (sign flips because diff d = x1-x2 -> x2-x1 when roles swap):
//   kernel_gaussian_jacobian_t(x1, x2, dX1, ...) ==
//       -kernel_gaussian_jacobian(x2, x1, dX1, ...).T
//
// For each matching-label pair (a,j1) from set-1 and (b,j2) from set-2:
//   d = x1[a,j1] - x2[b,j2]
//   alpha = exp(-|d|^2 / 2sigma^2) * (-1/sigma^2)
//   K_t[naq1_off_a + r, b] += alpha * A1[r, :] @ d   for r in [0, 3*n1[a])
// where A1 = dX1[a, j1, :, :] shape (rep_size, 3*max_atoms1) row-major.
void kernel_gaussian_jacobian_t(
    const std::vector<double> &x1,   // (nm1, max_atoms1, rep)
    const std::vector<double> &x2,   // (nm2, max_atoms2, rep)
    const std::vector<double> &dX1,  // (nm1, max_atoms1, rep, 3*max_atoms1)
    const std::vector<int> &q1,      // (nm1, max_atoms1)
    const std::vector<int> &q2,      // (nm2, max_atoms2)
    const std::vector<int> &n1,      // (nm1)
    const std::vector<int> &n2,      // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq1,  // must equal 3 * sum(n1)
    double sigma,
    double *kernel_out  // (naq1, nm2) row-major
) {
    if (nm1 <= 0 || nm2 <= 0 || max_atoms1 <= 0 || max_atoms2 <= 0 || rep_size <= 0)
        throw std::invalid_argument("All dims must be positive.");
    if (!std::isfinite(sigma) || sigma <= 0.0)
        throw std::invalid_argument("sigma must be positive and finite.");
    if (!kernel_out) throw std::invalid_argument("kernel_out is null.");

    const size_t x1N = (size_t)nm1 * max_atoms1 * rep_size;
    const size_t x2N = (size_t)nm2 * max_atoms2 * rep_size;
    const size_t dXN = (size_t)nm1 * max_atoms1 * rep_size * (3 * (size_t)max_atoms1);
    const size_t q1N = (size_t)nm1 * max_atoms1;
    const size_t q2N = (size_t)nm2 * max_atoms2;

    if (x1.size() != x1N || x2.size() != x2N) throw std::invalid_argument("x1/x2 size mismatch.");
    if (dX1.size() != dXN) throw std::invalid_argument("dX1 size mismatch.");
    if (q1.size() != q1N || q2.size() != q2N) throw std::invalid_argument("q1/q2 size mismatch.");
    if ((int)n1.size() != nm1 || (int)n2.size() != nm2)
        throw std::invalid_argument("n1/n2 size mismatch.");

    // per-a offsets and nrows (3 * n1[a])
    std::vector<int> offs1(nm1), nrows_a(nm1);
    int acc = 0;
    for (int a = 0; a < nm1; ++a) {
        const int na = std::max(0, std::min(n1[a], max_atoms1));
        offs1[a] = acc;
        nrows_a[a] = 3 * na;
        acc += nrows_a[a];
    }
    if (naq1 != acc) throw std::invalid_argument("naq1 != 3*sum(n1)");

    // zero output
    std::fill(kernel_out, kernel_out + (size_t)naq1 * nm2, 0.0);

    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma2 = -1.0 / (sigma * sigma);

    // Build per-label list of (b, j2) only for valid j2 < n2[b]
    std::unordered_map<int, std::vector<std::pair<int, int>>> lj2;
    lj2.reserve(128);
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::max(0, std::min(n2[b], max_atoms2));
        for (int j2 = 0; j2 < nb; ++j2) {
            const int lbl = q2[(size_t)b * max_atoms2 + j2];
            lj2[lbl].emplace_back(b, j2);
        }
    }

    // Tile width: D is (T x rep_size), H is (T x nrows_max).
    // nrows_max = 3*max_atoms1.
    constexpr int BATCH_T = 256;
    const int nrows_max = 3 * max_atoms1;

    // Parallelize over a: each thread owns a disjoint row block of kernel_out.
    // Serialise BLAS inside OMP to prevent oversubscription.
    double t_build_D_jt = 0.0, t_dgemm_jt = 0.0, t_scatter_jt = 0.0;
#ifdef KERNELFORGE_ENABLE_PROFILING
    const double t_func_start = omp_get_wtime();
#endif
    const int blas_nt = kf_blas_get_num_threads();
    kf_blas_set_num_threads(1);
#pragma omp parallel default(none) \
    shared(x1,                     \
               x2,                 \
               dX1,                \
               q1,                 \
               q2,                 \
               n1,                 \
               n2,                 \
               nm1,                \
               nm2,                \
               max_atoms1,         \
               max_atoms2,         \
               rep_size,           \
               naq1,               \
               kernel_out,         \
               lj2,                \
               offs1,              \
               nrows_a,            \
               inv_2sigma2,        \
               inv_sigma2,         \
               nrows_max,          \
               t_build_D_jt,       \
               t_dgemm_jt,         \
               t_scatter_jt)
    {
        // Thread-local scratch: D is (T x rep_size), H is (T x nrows_max).
        double *D = aligned_alloc_64((size_t)BATCH_T * rep_size);
        double *H = aligned_alloc_64((size_t)BATCH_T * nrows_max);

#pragma omp for schedule(dynamic)
        for (int a = 0; a < nm1; ++a) {
            const int nrows = nrows_a[a];
            if (nrows == 0) continue;

            const int out_offset = offs1[a];  // row offset into kernel_out
            const int lda_rowmaj = 3 * max_atoms1;

            for (int j1 = 0; j1 < nrows / 3; ++j1) {
                const int label = q1[(size_t)a * max_atoms1 + j1];
                auto it = lj2.find(label);
                if (it == lj2.end() || it->second.empty()) continue;

                const auto &bj2_list = it->second;

                // A1 = dX1(a, j1, :, :): shape (rep_size, 3*max_atoms1) row-major
                const double *A1 = &dX1[base_dx1(a, j1, nm1, max_atoms1, rep_size)];
                // x1 row for (a,j1)
                const double *x1_aj1 = &x1[((size_t)a * max_atoms1 + j1) * rep_size];

                // Process (b,j2) pairs in tiles of width BATCH_T
                for (size_t t0 = 0; t0 < bj2_list.size(); t0 += BATCH_T) {
                    const int T = (int)std::min<size_t>(BATCH_T, bj2_list.size() - t0);

                    // 1) Build D (T x rep_size): row t = alpha_t * (x1[a,j1] - x2[b,j2]).
#ifdef KERNELFORGE_ENABLE_PROFILING
                    double tp = omp_get_wtime();
#endif
                    for (int t = 0; t < T; ++t) {
                        const int b = bj2_list[t0 + t].first;
                        const int j2 = bj2_list[t0 + t].second;
                        const double *x2_bj2 = &x2[((size_t)b * max_atoms2 + j2) * rep_size];

                        double l2 = 0.0;
                        double *Drow = D + (size_t)t * rep_size;
                        for (int k = 0; k < rep_size; ++k) {
                            const double diff = x1_aj1[k] - x2_bj2[k];
                            Drow[k] = diff;
                            l2 += diff * diff;
                        }
                        const double alpha_t = std::exp(l2 * inv_2sigma2) * inv_sigma2;
                        for (int k = 0; k < rep_size; ++k)
                            Drow[k] *= alpha_t;
                    }
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_build_D_jt += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // 2) H(T x nrows) = D(T x rep) @ A1(rep x nrows)
                    //    A1 is (rep x lda_rowmaj) row-major; we use nrows columns.
                    cblas_dgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        static_cast<blas_int>(T),
                        static_cast<blas_int>(nrows),
                        static_cast<blas_int>(rep_size),
                        1.0,
                        D,
                        static_cast<blas_int>(rep_size),
                        A1,
                        static_cast<blas_int>(lda_rowmaj),
                        0.0,
                        H,
                        static_cast<blas_int>(nrows_max)
                    );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_dgemm_jt += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // 3) Scatter-add H rows into kernel_out.
                    //    kernel_out is (naq1, nm2) row-major.
                    //    H[t, r] contributes to kernel_out[out_offset + r, b].
                    //    We write column b of the output row block — this is a column
                    //    scatter, so we transpose: for each derivative row r, write
                    //    kernel_out[(out_offset+r)*nm2 + b] += H[t*nrows_max + r].
                    for (int t = 0; t < T; ++t) {
                        const int b = bj2_list[t0 + t].first;
                        const double *hrow = H + (size_t)t * nrows_max;
                        for (int r = 0; r < nrows; ++r)
                            kernel_out[(size_t)(out_offset + r) * nm2 + b] += hrow[r];
                    }
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_scatter_jt += omp_get_wtime() - tp;
#endif
                }  // tiles
            }  // j1
        }  // omp for

        aligned_free_64(D);
        aligned_free_64(H);
    }  // omp parallel
    kf_blas_set_num_threads(blas_nt);

#ifdef KERNELFORGE_ENABLE_PROFILING
    const double t_total = omp_get_wtime() - t_func_start;
    std::printf(
        "[PROFILE] kernel_gaussian_jacobian_t(nm1=%d, nm2=%d, rep=%d):\n"
        "  Build D (exp+scale):   %8.4fs (%5.1f%%)\n"
        "  DGEMM (H = D @ A1):   %8.4fs (%5.1f%%)\n"
        "  Scatter-add to kernel: %8.4fs (%5.1f%%)\n"
        "  Total:                 %8.4fs (100.0%%)\n",
        nm1,
        nm2,
        rep_size,
        t_build_D_jt,
        100.0 * t_build_D_jt / t_total,
        t_dgemm_jt,
        100.0 * t_dgemm_jt / t_total,
        t_scatter_jt,
        100.0 * t_scatter_jt / t_total,
        t_total
    );
#endif
}

// #########################
// # FCHL19 HESSIAN KERNEL #
// #########################
void kernel_gaussian_hessian(
    const std::vector<double> &x1,   // (nm1, max_atoms1, rep_size)
    const std::vector<double> &x2,   // (nm2, max_atoms2, rep_size)
    const std::vector<double> &dx1,  // (nm1, max_atoms1, rep_size, 3*max_atoms1)
    const std::vector<double> &dx2,  // (nm2, max_atoms2, rep_size, 3*max_atoms2)
    const std::vector<int> &q1,      // (nm1, max_atoms1)
    const std::vector<int> &q2,      // (nm2, max_atoms2)
    const std::vector<int> &n1,      // (nm1)
    const std::vector<int> &n2,      // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq1,  // must be 3 * sum_a n1[a]
    int naq2,  // must be 3 * sum_b n2[b]
    double sigma,
    double *kernel_out  // (naq1, naq2), row-major => idx = row * naq2 + col
) {
    // ---- validation ----
    if (nm1 <= 0 || nm2 <= 0 || max_atoms1 <= 0 || max_atoms2 <= 0 || rep_size <= 0)
        throw std::invalid_argument("All dims must be positive.");
    if (!std::isfinite(sigma) || sigma <= 0.0)
        throw std::invalid_argument("sigma must be positive and finite.");
    if (!kernel_out) throw std::invalid_argument("kernel_out is null.");

    const size_t x1N = (size_t)nm1 * max_atoms1 * rep_size;
    const size_t x2N = (size_t)nm2 * max_atoms2 * rep_size;
    const size_t dx1N = (size_t)nm1 * max_atoms1 * rep_size * (3 * (size_t)max_atoms1);
    const size_t dx2N = (size_t)nm2 * max_atoms2 * rep_size * (3 * (size_t)max_atoms2);
    const size_t q1N = (size_t)nm1 * max_atoms1;
    const size_t q2N = (size_t)nm2 * max_atoms2;

    if (x1.size() != x1N || x2.size() != x2N) throw std::invalid_argument("x1/x2 size mismatch.");
    if (dx1.size() != dx1N || dx2.size() != dx2N)
        throw std::invalid_argument("dx1/dx2 size mismatch.");
    if (q1.size() != q1N || q2.size() != q2N) throw std::invalid_argument("q1/q2 size mismatch.");
    if ((int)n1.size() != nm1 || (int)n2.size() != nm2)
        throw std::invalid_argument("n1/n2 size mismatch.");

    // column offsets (set-1) and row offsets (set-2)
    std::vector<int> offs1(nm1), offs2(nm2);
    int sum1 = 0;
    for (int a = 0; a < nm1; ++a) {
        offs1[a] = sum1;
        sum1 += 3 * std::max(0, std::min(n1[a], max_atoms1));
    }
    int sum2 = 0;
    for (int b = 0; b < nm2; ++b) {
        offs2[b] = sum2;
        sum2 += 3 * std::max(0, std::min(n2[b], max_atoms2));
    }
    if (naq1 != sum1) throw std::invalid_argument("naq1 != 3*sum(n1)");
    if (naq2 != sum2) throw std::invalid_argument("naq2 != 3*sum(n2)");

    // zero output
    std::fill(kernel_out, kernel_out + (size_t)naq1 * naq2, 0.0);

    // scalars
    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma4 = -1.0 / (sigma * sigma * sigma * sigma);  // (< 0)
    const double sigma2_neg = -(sigma * sigma);                        // (< 0)

    // ----------------------------------------------------------------
    // Precompute, for each molecule a in set-1, a map: label -> list of i1
    // ----------------------------------------------------------------
    std::vector<std::unordered_map<int, std::vector<int>>> lab_to_i1(nm1);
    for (int a = 0; a < nm1; ++a) {
        const int na = std::max(0, std::min(n1[a], max_atoms1));
        auto &m = lab_to_i1[a];
        m.reserve(64);
        for (int i1 = 0; i1 < na; ++i1) {
            m[q1[(size_t)a * max_atoms1 + i1]].push_back(i1);
        }
    }

    // For each (a,b) block, collect ALL matching (i1,i2) pairs, then batch everything
    // in one pass: big D(M x rep), big W(M x ncols_b) = D @ SD2_pack^T, big V(M x ncols_a),
    // big rank-1 update, big S_sum per-i2.
    // For SD2_pack: pack each SD2[i2] transposed into (ncols_b x rep) rows, stacked.
    // Memory per thread: we over-alloc for worst case M = nb*na (Ethanol: 9*9=81 matches per
    // (a,b)).
    const int ncols_b_max = 3 * max_atoms2;
    const int ncols_a_max = 3 * max_atoms1;
    // worst-case M per (a,b) block = na * nb  (if all labels match, which rarely happens)
    // For safety, alloc 512 (well above any realistic Ethanol case).
    constexpr int M_MAX = 512;

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_diff_ssum = 0.0, t_dgemv_W = 0.0, t_dgemv_V = 0.0, t_dger = 0.0, t_dgemm_static = 0.0;
    const double t_func_start = omp_get_wtime();
#endif

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::max(0, std::min(n2[b], max_atoms2));
        if (nb == 0) continue;
        const int ncols_b = 3 * nb;
        const int lda2 = 3 * max_atoms2;
        const int row_off = offs2[b];

        // Thread-local scratch (worst-case M_MAX matching pairs per (a,b) block):
        // D_all:    (M_MAX x rep)       — sqrt(|expd|)-scaled diffs
        // SD2_pack: (M_MAX x rep)       — SD2[i2] row (for GEMM W_all = D_all @ SD2T_pack^T)
        //           BUT SD2 varies per i2 — can't pack into one flat matrix for one GEMM
        //           UNLESS we duplicate rows. We'll do: W_all(M x ncols_b) via loop over unique i2
        //           OR pack SD2_indexed(M x rep) where row m = SD2[i2_m] transposed (ncols_b cols
        //           taken)
        // Actually: W_all(M x ncols_b) where W_all[m] = D_all[m] @ SD2[i2_m]^T
        //   = element-wise: W_all = sum_k D_all[m,k]*SD2[i2_m,k,:]
        // This is NOT a standard GEMM due to varying i2. But: if we pack
        //   SD2_pack(M x rep*ncols_b) — stride not uniform — still not a GEMM.
        //
        // SIMPLEST CORRECT BIG-GEMM: for fixed i2, collect all i1 for that i2, do GEMM.
        // That's what we already do. The real problem is BATCH_T is limited by i1_list.size()
        // which is 1 for single-species Ethanol (each atom has unique label match per molecule).
        //
        // For single-match case (T=1): GEMM(1 x ncols_b x rep) = GEMV. Replace GEMM with GEMV.
        // V(1 x ncols_a): V[0] = -SA[0] @ D[0] = GEMV(ncols_a, rep).
        // W(1 x ncols_b): W[0] =  D[0] @ SD2^T  = GEMV(ncols_b, rep) [D[0] is row vec].
        // rank-1: Kba += W[0]^T @ V[0] = dger(ncols_b, ncols_a, 1, W[0], 1, V[0], 1, Kba, naq1)
        //
        // This avoids GEMM overhead for tiny T. Use GEMV for T=1, GEMM for T>1.
        double *D_row = aligned_alloc_64((size_t)rep_size);                // single diff row
        double *W_row = aligned_alloc_64((size_t)ncols_b_max);             // one W vector
        double *V_row = aligned_alloc_64((size_t)ncols_a_max);             // one V vector
        double *S_sum = aligned_alloc_64((size_t)rep_size * ncols_a_max);  // static sum

        for (int a = 0; a < nm1; ++a) {
            const int na = std::max(0, std::min(n1[a], max_atoms1));
            if (na == 0) continue;
            const int ncols_a = 3 * na;
            const int col_off = offs1[a];
            const int lda1 = 3 * max_atoms1;

            double *Kab = &kernel_out[(size_t)col_off * naq2 + row_off];
            const auto &label_i1 = lab_to_i1[a];

            for (int i2 = 0; i2 < nb; ++i2) {
                const int label = q2[(size_t)b * max_atoms2 + i2];
                auto it = label_i1.find(label);
                if (it == label_i1.end()) continue;
                const auto &i1_list = it->second;
                if (i1_list.empty()) continue;

                const double *SD2 = &dx2[base_dx2(b, i2, nm2, max_atoms2, rep_size)];
                const double *x2_bi2 = &x2[((size_t)b * max_atoms2 + i2) * rep_size];

                // Zero S_sum for this (a, b, i2) triplet
                std::fill(S_sum, S_sum + (size_t)rep_size * ncols_a, 0.0);

                for (int i1 : i1_list) {
                    const double *x1_t = &x1[((size_t)a * max_atoms1 + i1) * rep_size];
                    const double *SD1 = &dx1[base_dx1(a, i1, nm1, max_atoms1, rep_size)];

#ifdef KERNELFORGE_ENABLE_PROFILING
                    double tp = omp_get_wtime();
#endif
                    double l2 = 0.0;
                    for (int k = 0; k < rep_size; ++k) {
                        const double diff = x1_t[k] - x2_bi2[k];
                        D_row[k] = diff;
                        l2 += diff * diff;
                    }
                    const double exp_base = std::exp(l2 * inv_2sigma2);
                    const double expd = inv_sigma4 * exp_base;  // < 0
                    const double expdiag = sigma2_neg * expd;   // > 0
                    const double s = std::sqrt(-expd);          // sqrt(|expd|)

                    // scale D for rank-1
                    for (int k = 0; k < rep_size; ++k)
                        D_row[k] *= s;

                    // accumulate S_sum (rep x ncols_a) += expdiag * SD1
                    // SD1 is (rep x lda1) row-major; only first ncols_a cols used
                    for (int k = 0; k < rep_size; ++k) {
                        double *srow = S_sum + (size_t)k * ncols_a;
                        const double *sdrow = SD1 + (size_t)k * lda1;
                        for (int j = 0; j < ncols_a; ++j)
                            srow[j] += expdiag * sdrow[j];
                    }
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_diff_ssum += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // W = D_row @ SD2^T : shape (ncols_b,)
                    // SD2 is (rep x lda2) row-major
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        SD2,
                        static_cast<blas_int>(lda2),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        W_row,
                        static_cast<blas_int>(1)
                    );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_dgemv_W += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // V = -SD1^T @ D_row : shape (ncols_a,)
                    // SD1 is (rep x lda1) row-major; Trans GEMV: output[j] = sum_k SD1[k,j]*D[k]
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_a),
                        -1.0,
                        SD1,
                        static_cast<blas_int>(lda1),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        V_row,
                        static_cast<blas_int>(1)
                    );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_dgemv_V += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // rank-1: Kab += V ⊗ W  (dger, rows=ncols_a x cols=ncols_b)
                    cblas_dger(
                        CblasRowMajor,
                        static_cast<blas_int>(ncols_a),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        V_row,
                        static_cast<blas_int>(1),
                        W_row,
                        static_cast<blas_int>(1),
                        Kab,
                        static_cast<blas_int>(naq2)
                    );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_dger += omp_get_wtime() - tp;
#endif
                }  // i1

                // ---- static term: Kab += S_sum^T @ SD2  (ncols_a x ncols_b) ----
                // S_sum is (rep x ncols_a) row-major with stride ncols_a; SD2 is (rep x lda2)
#ifdef KERNELFORGE_ENABLE_PROFILING
                double tp_static = omp_get_wtime();
#endif
                cblas_dgemm(
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    static_cast<blas_int>(ncols_a),
                    static_cast<blas_int>(ncols_b),
                    static_cast<blas_int>(rep_size),
                    1.0,
                    S_sum,
                    static_cast<blas_int>(ncols_a),
                    SD2,
                    static_cast<blas_int>(lda2),
                    1.0,
                    Kab,
                    static_cast<blas_int>(naq2)
                );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                t_dgemm_static += omp_get_wtime() - tp_static;
#endif
            }  // i2
        }  // a

        aligned_free_64(D_row);
        aligned_free_64(W_row);
        aligned_free_64(V_row);
        aligned_free_64(S_sum);
    }  // b
    (void)M_MAX;

#ifdef KERNELFORGE_ENABLE_PROFILING
    const double t_wall = omp_get_wtime() - t_func_start;
    const double t_sum = t_diff_ssum + t_dgemv_W + t_dgemv_V + t_dger + t_dgemm_static;
    const double t_denom = (t_sum > 0.0) ? t_sum : 1.0;
    std::printf(
        "[PROFILE] kernel_gaussian_hessian(nm1=%d, nm2=%d, rep=%d):\n"
        "  Diff+exp+S_sum:         %8.4fs (%5.1f%%)\n"
        "  DGEMV W (D@SD2^T):      %8.4fs (%5.1f%%)\n"
        "  DGEMV V (-SD1^T@D):     %8.4fs (%5.1f%%)\n"
        "  DGER rank-1:            %8.4fs (%5.1f%%)\n"
        "  DGEMM static (S^T@SD2): %8.4fs (%5.1f%%)\n"
        "  Sum(thread-time):       %8.4fs (100.0%%)\n"
        "  Wall-clock:             %8.4fs\n",
        nm1,
        nm2,
        rep_size,
        t_diff_ssum,
        100.0 * t_diff_ssum / t_denom,
        t_dgemv_W,
        100.0 * t_dgemv_W / t_denom,
        t_dgemv_V,
        100.0 * t_dgemv_V / t_denom,
        t_dger,
        100.0 * t_dger / t_denom,
        t_dgemm_static,
        100.0 * t_dgemm_static / t_denom,
        t_sum,
        t_wall
    );
#endif
}

void kernel_gaussian_hessian_symm(
    const std::vector<double> &x,   // (nm, max_atoms, rep_size)
    const std::vector<double> &dx,  // (nm, max_atoms, rep_size, 3*max_atoms)
    const std::vector<int> &q,      // (nm, max_atoms)
    const std::vector<int> &n,      // (nm)
    int nm, int max_atoms, int rep_size,
    int naq,  // must be 3 * sum_m n[m]
    double sigma,
    double *kernel_out  // (naq, naq) row-major
) {
    // ---- validation ----
    if (nm <= 0 || max_atoms <= 0 || rep_size <= 0)
        throw std::invalid_argument("dims must be positive");
    if (!std::isfinite(sigma) || sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");
    if (!kernel_out) throw std::invalid_argument("kernel_out is null");

    const size_t xN = (size_t)nm * max_atoms * rep_size;
    const size_t dxN = (size_t)nm * max_atoms * rep_size * (3 * (size_t)max_atoms);
    const size_t qN = (size_t)nm * max_atoms;

    if (x.size() != xN) throw std::invalid_argument("x size mismatch");
    if (dx.size() != dxN) throw std::invalid_argument("dx size mismatch");
    if (q.size() != qN) throw std::invalid_argument("q size mismatch");
    if ((int)n.size() != nm) throw std::invalid_argument("n size mismatch");

    // Offsets in derivative axis (3 * active atoms per molecule)
    std::vector<int> offs(nm);
    int sum = 0;
    for (int m = 0; m < nm; ++m) {
        offs[m] = sum;
        sum += 3 * std::max(0, std::min(n[m], max_atoms));
    }
    if (naq != sum) throw std::invalid_argument("naq != 3*sum(n)");

    // Zero only lower triangle (optional: zero all for simplicity)
    std::fill(kernel_out, kernel_out + (size_t)naq * naq, 0.0);

    // Scalars
    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma4 = -1.0 / (sigma * sigma * sigma * sigma);  // < 0
    const double sigma2_neg = -(sigma * sigma);                        // < 0

    // Label -> indices per molecule
    std::vector<std::unordered_map<int, std::vector<int>>> lab_to_idx(nm);
    for (int a = 0; a < nm; ++a) {
        const int na = std::max(0, std::min(n[a], max_atoms));
        auto &M = lab_to_idx[a];
        M.reserve(64);
        for (int i1 = 0; i1 < na; ++i1) {
            M[q[(size_t)a * max_atoms + i1]].push_back(i1);
        }
    }

    const int ncols_max = 3 * max_atoms;

#ifdef KERNELFORGE_ENABLE_PROFILING
    double t_diff_ssum = 0.0, t_dgemv_W = 0.0, t_dgemv_V = 0.0, t_dger = 0.0, t_dgemm_static = 0.0;
    const double t_func_start = omp_get_wtime();
#endif

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < nm; ++b) {
        const int nb = std::max(0, std::min(n[b], max_atoms));
        if (nb == 0) continue;

        const int ncols_b = 3 * nb;
        const int lda_b = 3 * max_atoms;
        const int row_off = offs[b];

        // Thread-local aligned scratch (per-i1 scalar path):
        // D_row:  (rep_size,)          — sqrt(|expd|)-scaled diff
        // W_row:  (ncols_max,)         — W = SD_b^T @ D_row
        // V_row:  (ncols_max,)         — V = -SD_a^T @ D_row
        // S_sum:  (rep x ncols_max)    — sum_i1 expdiag*SD_a(i1) for static term
        double *D_row = aligned_alloc_64((size_t)rep_size);
        double *W_row = aligned_alloc_64((size_t)ncols_max);
        double *V_row = aligned_alloc_64((size_t)ncols_max);
        double *S_sum = aligned_alloc_64((size_t)rep_size * ncols_max);
        double *xbv = aligned_alloc_64((size_t)rep_size);

        for (int a = 0; a <= b; ++a) {  // lower triangle only
            const int na = std::max(0, std::min(n[a], max_atoms));
            if (na == 0) continue;

            const int ncols_a = 3 * na;
            const int col_off = offs[a];
            const int lda_a = 3 * max_atoms;

            // Destination block
            double *Kba = &kernel_out[(size_t)row_off * naq + col_off];

            // For diagonal block, accumulate into a small temp (ncols_b x ncols_a), then scatter
            // lower
            std::vector<double> Cdiag;
            double *Cdst = Kba;
            if (a == b) {
                Cdiag.assign((size_t)ncols_b * ncols_a, 0.0);
                Cdst = Cdiag.data();  // accumulate full square, scatter lower later
            }

            const auto &lab_a = lab_to_idx[a];
            const auto &lab_b = lab_to_idx[b];

            // loop atoms j2 in molecule b
            for (int j2 = 0; j2 < nb; ++j2) {
                const int lbl = q[(size_t)b * max_atoms + j2];
                auto it_a = lab_a.find(lbl);
                if (it_a == lab_a.end() || it_a->second.empty()) continue;
                const auto &i1_list = it_a->second;

                // SD_b and x_b slice (for j2)
                const double *SD_b = &dx[base_dx(b, j2, nm, max_atoms, rep_size)];
                for (int k = 0; k < rep_size; ++k)
                    xbv[k] = x[idx_x(b, j2, k, nm, max_atoms, rep_size)];

                // Zero S_sum once per (a, j2) pair
                std::fill(S_sum, S_sum + (size_t)rep_size * ncols_a, 0.0);

                for (int i1 : i1_list) {
                    const double *x_ai1 = &x[idx_x(a, i1, 0, nm, max_atoms, rep_size)];
                    const double *SD_a = &dx[base_dx(a, i1, nm, max_atoms, rep_size)];

#ifdef KERNELFORGE_ENABLE_PROFILING
                    double tp = omp_get_wtime();
#endif
                    double l2 = 0.0;
                    for (int k = 0; k < rep_size; ++k) {
                        const double diff = x_ai1[k] - xbv[k];
                        D_row[k] = diff;
                        l2 += diff * diff;
                    }
                    const double eb = std::exp(l2 * inv_2sigma2);
                    const double e1 = inv_sigma4 * eb;       // < 0
                    const double expdiag = sigma2_neg * e1;  // > 0
                    const double s = std::sqrt(-e1);         // sqrt(|expd|)

                    for (int k = 0; k < rep_size; ++k)
                        D_row[k] *= s;

                    // accumulate S_sum (rep x ncols_a) += expdiag * SD_a
                    // SD_a is (rep x lda_a) row-major; only first ncols_a cols used
                    for (int k = 0; k < rep_size; ++k) {
                        double *srow = S_sum + (size_t)k * ncols_a;
                        const double *sdrow = SD_a + (size_t)k * lda_a;
                        for (int j = 0; j < ncols_a; ++j)
                            srow[j] += expdiag * sdrow[j];
                    }
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_diff_ssum += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // W = SD_b^T @ D_row  (ncols_b vector)
                    // SD_b is (rep x lda_b) row-major
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        SD_b,
                        static_cast<blas_int>(lda_b),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        W_row,
                        static_cast<blas_int>(1)
                    );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_dgemv_W += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // V = -SD_a^T @ D_row  (ncols_a vector)
                    // SD_a is (rep x lda_a) row-major; Trans GEMV: output[j] = sum_k SD_a[k,j]*D[k]
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_a),
                        -1.0,
                        SD_a,
                        static_cast<blas_int>(lda_a),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        V_row,
                        static_cast<blas_int>(1)
                    );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_dgemv_V += omp_get_wtime() - tp;
                    tp = omp_get_wtime();
#endif

                    // rank-1: Cdst += W ⊗ V  (dger, lda = ncols_a for diag block, naq otherwise)
                    cblas_dger(
                        CblasRowMajor,
                        static_cast<blas_int>(ncols_b),
                        static_cast<blas_int>(ncols_a),
                        1.0,
                        W_row,
                        static_cast<blas_int>(1),
                        V_row,
                        static_cast<blas_int>(1),
                        Cdst,
                        static_cast<blas_int>(a == b ? ncols_a : naq)
                    );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                    t_dger += omp_get_wtime() - tp;
#endif
                }  // i1

                // ---- static term: Cdst += SD_b^T @ S_sum ----
#ifdef KERNELFORGE_ENABLE_PROFILING
                double tp_static = omp_get_wtime();
#endif
                cblas_dgemm(
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    static_cast<blas_int>(ncols_b),
                    static_cast<blas_int>(ncols_a),
                    static_cast<blas_int>(rep_size),
                    1.0,
                    SD_b,
                    static_cast<blas_int>(lda_b),
                    S_sum,
                    static_cast<blas_int>(ncols_a),
                    1.0,
                    Cdst,
                    static_cast<blas_int>(a == b ? ncols_a : naq)
                );
#ifdef KERNELFORGE_ENABLE_PROFILING
    #pragma omp atomic
                t_dgemm_static += omp_get_wtime() - tp_static;
#endif
            }  // j2

            // Scatter diagonal block's lower triangle only
            if (a == b) {
                for (int r = 0; r < ncols_b; ++r) {
                    double *kout = Kba + (size_t)r * naq;
                    const double *crow = Cdiag.data() + (size_t)r * ncols_a;
                    const int cmax = std::min(r, ncols_a - 1);
                    cblas_daxpy(
                        static_cast<blas_int>(cmax + 1),
                        1.0,
                        crow,
                        static_cast<blas_int>(1),
                        kout,
                        static_cast<blas_int>(1)
                    );
                }
            }
        }  // a

        aligned_free_64(D_row);
        aligned_free_64(W_row);
        aligned_free_64(V_row);
        aligned_free_64(S_sum);
        aligned_free_64(xbv);
    }  // b

#ifdef KERNELFORGE_ENABLE_PROFILING
    const double t_wall = omp_get_wtime() - t_func_start;
    const double t_sum = t_diff_ssum + t_dgemv_W + t_dgemv_V + t_dger + t_dgemm_static;
    const double t_denom = (t_sum > 0.0) ? t_sum : 1.0;
    std::printf(
        "[PROFILE] kernel_gaussian_hessian_symm(nm=%d, rep=%d):\n"
        "  Diff+exp+S_sum:          %8.4fs (%5.1f%%)\n"
        "  DGEMV W (SD_b^T @ D):    %8.4fs (%5.1f%%)\n"
        "  DGEMV V (-SD_a^T @ D):   %8.4fs (%5.1f%%)\n"
        "  DGER rank-1 (W x V):     %8.4fs (%5.1f%%)\n"
        "  DGEMM static (SD_b^T@S): %8.4fs (%5.1f%%)\n"
        "  Sum(thread-time):        %8.4fs (100.0%%)\n"
        "  Wall-clock:              %8.4fs\n",
        nm,
        rep_size,
        t_diff_ssum,
        100.0 * t_diff_ssum / t_denom,
        t_dgemv_W,
        100.0 * t_dgemv_W / t_denom,
        t_dgemv_V,
        100.0 * t_dgemv_V / t_denom,
        t_dger,
        100.0 * t_dger / t_denom,
        t_dgemm_static,
        100.0 * t_dgemm_static / t_denom,
        t_sum,
        t_wall
    );
#endif
}

// Symmetric Hessian kernel in RFP (Rectangular Full Packed) format.
// Output: 1-D array of length naq*(naq+1)/2, TRANSR='N', UPLO='U'.
// rfp_index_upper_N(naq, col, row) with 0 <= col <= row < naq.
//
// This is identical to kernel_gaussian_hessian_symm, but instead of writing
// to a full (naq x naq) square matrix, each (a,b) block is accumulated into
// a small temp buffer, then scattered into the RFP array.
void kernel_gaussian_hessian_symm_rfp(
    const std::vector<double> &x,   // (nm, max_atoms, rep_size)
    const std::vector<double> &dx,  // (nm, max_atoms, rep_size, 3*max_atoms)
    const std::vector<int> &q,      // (nm, max_atoms)
    const std::vector<int> &n,      // (nm)
    int nm, int max_atoms, int rep_size,
    int naq,  // must be 3 * sum_m n[m]
    double sigma,
    double *arf  // length naq*(naq+1)/2, RFP TRANSR='N', UPLO='U'
) {
    // ---- validation ----
    if (nm <= 0 || max_atoms <= 0 || rep_size <= 0)
        throw std::invalid_argument("dims must be positive");
    if (!std::isfinite(sigma) || sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");
    if (!arf) throw std::invalid_argument("arf is null");

    const size_t xN = (size_t)nm * max_atoms * rep_size;
    const size_t dxN = (size_t)nm * max_atoms * rep_size * (3 * (size_t)max_atoms);
    const size_t qN = (size_t)nm * max_atoms;

    if (x.size() != xN) throw std::invalid_argument("x size mismatch");
    if (dx.size() != dxN) throw std::invalid_argument("dx size mismatch");
    if (q.size() != qN) throw std::invalid_argument("q size mismatch");
    if ((int)n.size() != nm) throw std::invalid_argument("n size mismatch");

    // Offsets in derivative axis (3 * active atoms per molecule)
    std::vector<int> offs(nm);
    int sum = 0;
    for (int m = 0; m < nm; ++m) {
        offs[m] = sum;
        sum += 3 * std::max(0, std::min(n[m], max_atoms));
    }
    if (naq != sum) throw std::invalid_argument("naq != 3*sum(n)");

    // Zero the RFP array
    const size_t nt = (size_t)naq * (naq + 1ull) / 2ull;
    std::fill(arf, arf + nt, 0.0);

    // Scalars
    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma4 = -1.0 / (sigma * sigma * sigma * sigma);  // < 0
    const double sigma2_neg = -(sigma * sigma);                        // < 0

    // Label -> indices per molecule
    std::vector<std::unordered_map<int, std::vector<int>>> lab_to_idx(nm);
    for (int a = 0; a < nm; ++a) {
        const int na = std::max(0, std::min(n[a], max_atoms));
        auto &M = lab_to_idx[a];
        M.reserve(64);
        for (int i1 = 0; i1 < na; ++i1) {
            M[q[(size_t)a * max_atoms + i1]].push_back(i1);
        }
    }

    const int ncols_max = 3 * max_atoms;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < nm; ++b) {
        const int nb = std::max(0, std::min(n[b], max_atoms));
        if (nb == 0) continue;

        const int ncols_b = 3 * nb;
        const int lda_b = 3 * max_atoms;
        const int row_off = offs[b];

        // Thread-local scratch
        double *D_row = aligned_alloc_64((size_t)rep_size);
        double *W_row = aligned_alloc_64((size_t)ncols_max);
        double *V_row = aligned_alloc_64((size_t)ncols_max);
        double *S_sum = aligned_alloc_64((size_t)rep_size * ncols_max);
        double *xbv = aligned_alloc_64((size_t)rep_size);

        for (int a = 0; a <= b; ++a) {  // lower triangle only (col <= row)
            const int na = std::max(0, std::min(n[a], max_atoms));
            if (na == 0) continue;

            const int ncols_a = 3 * na;
            const int col_off = offs[a];
            const int lda_a = 3 * max_atoms;

            const auto &lab_a = lab_to_idx[a];
            const auto &lab_b = lab_to_idx[b];

            // Temp block: (ncols_b x ncols_a) — always use a scratch buffer
            // so that dger/dgemm have a contiguous lda, then scatter to RFP
            std::vector<double> Cblk((size_t)ncols_b * ncols_a, 0.0);
            double *Cdst = Cblk.data();
            const int ldc = ncols_a;  // stride for Cblk

            // loop atoms j2 in molecule b
            for (int j2 = 0; j2 < nb; ++j2) {
                const int lbl = q[(size_t)b * max_atoms + j2];
                auto it_a = lab_a.find(lbl);
                if (it_a == lab_a.end() || it_a->second.empty()) continue;
                const auto &i1_list = it_a->second;

                // SD_b and x_b slice (for j2)
                const double *SD_b = &dx[base_dx(b, j2, nm, max_atoms, rep_size)];
                for (int k = 0; k < rep_size; ++k)
                    xbv[k] = x[idx_x(b, j2, k, nm, max_atoms, rep_size)];

                // Zero S_sum once per (a, j2) pair
                std::fill(S_sum, S_sum + (size_t)rep_size * ncols_a, 0.0);

                for (int i1 : i1_list) {
                    const double *x_ai1 = &x[idx_x(a, i1, 0, nm, max_atoms, rep_size)];
                    const double *SD_a = &dx[base_dx(a, i1, nm, max_atoms, rep_size)];

                    double l2 = 0.0;
                    for (int k = 0; k < rep_size; ++k) {
                        const double diff = x_ai1[k] - xbv[k];
                        D_row[k] = diff;
                        l2 += diff * diff;
                    }
                    const double eb = std::exp(l2 * inv_2sigma2);
                    const double e1 = inv_sigma4 * eb;       // < 0
                    const double expdiag = sigma2_neg * e1;  // > 0
                    const double s = std::sqrt(-e1);         // sqrt(|expd|)

                    for (int k = 0; k < rep_size; ++k)
                        D_row[k] *= s;

                    // accumulate S_sum (rep x ncols_a) += expdiag * SD_a
                    for (int k = 0; k < rep_size; ++k) {
                        double *srow = S_sum + (size_t)k * ncols_a;
                        const double *sdrow = SD_a + (size_t)k * lda_a;
                        for (int j = 0; j < ncols_a; ++j)
                            srow[j] += expdiag * sdrow[j];
                    }

                    // W = SD_b^T @ D_row  (ncols_b vector)
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        SD_b,
                        static_cast<blas_int>(lda_b),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        W_row,
                        static_cast<blas_int>(1)
                    );

                    // V = -SD_a^T @ D_row  (ncols_a vector)
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_a),
                        -1.0,
                        SD_a,
                        static_cast<blas_int>(lda_a),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        V_row,
                        static_cast<blas_int>(1)
                    );

                    // rank-1: Cdst += W ⊗ V  (dger)
                    cblas_dger(
                        CblasRowMajor,
                        static_cast<blas_int>(ncols_b),
                        static_cast<blas_int>(ncols_a),
                        1.0,
                        W_row,
                        static_cast<blas_int>(1),
                        V_row,
                        static_cast<blas_int>(1),
                        Cdst,
                        static_cast<blas_int>(ldc)
                    );
                }  // i1

                // ---- static term: Cdst += SD_b^T @ S_sum ----
                cblas_dgemm(
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    static_cast<blas_int>(ncols_b),
                    static_cast<blas_int>(ncols_a),
                    static_cast<blas_int>(rep_size),
                    1.0,
                    SD_b,
                    static_cast<blas_int>(lda_b),
                    S_sum,
                    static_cast<blas_int>(ncols_a),
                    1.0,
                    Cdst,
                    static_cast<blas_int>(ldc)
                );
            }  // j2

            // Scatter Cblk[(r, c)] -> arf[rfp_index_upper_N(naq, col_off+c, row_off+r)]
            // Convention: rfp_index_upper_N(n, col, row) with col <= row.
            // For (a <= b): col = col_off + c, row = row_off + r
            // Off-diagonal (a < b): all entries are valid since col_off <= row_off.
            // Diagonal (a == b): only lower-triangle of Cblk (r >= c) is valid.
            if (a == b) {
                // Diagonal block: scatter lower triangle only (r >= c)
                for (int r = 0; r < ncols_b; ++r) {
                    const double *crow = Cblk.data() + (size_t)r * ncols_a;
                    for (int c = 0; c <= r; ++c) {
                        const size_t idx = rfp_index_upper_N(naq, col_off + c, row_off + r);
#ifdef _OPENMP
    #pragma omp atomic
#endif
                        arf[idx] += crow[c];
                    }
                }
            } else {
                // Off-diagonal block: scatter all ncols_b * ncols_a entries
                for (int r = 0; r < ncols_b; ++r) {
                    const double *crow = Cblk.data() + (size_t)r * ncols_a;
                    for (int c = 0; c < ncols_a; ++c) {
                        const size_t idx = rfp_index_upper_N(naq, col_off + c, row_off + r);
#ifdef _OPENMP
    #pragma omp atomic
#endif
                        arf[idx] += crow[c];
                    }
                }
            }
        }  // a

        aligned_free_64(D_row);
        aligned_free_64(W_row);
        aligned_free_64(V_row);
        aligned_free_64(S_sum);
        aligned_free_64(xbv);
    }  // b
}

// ============================================================================
// Full combined energy+force kernel (asymmetric).
//
// Output layout (row-major, shape (nm1 + naq1) x (nm2 + naq2)):
//   K_full[0:nm1,   0:nm2]   = scalar block      K[a,b] = sum_{i1,i2 label-match} exp(...)
//   K_full[0:nm1,   nm2:]    = jacobian_t block   (nm1, naq2) — derivative on set-2
//   K_full[nm1:,    0:nm2]   = jacobian block     (naq1, nm2) — derivative on set-1
//   K_full[nm1:,    nm2:]    = hessian block      (naq1, naq2)
//
// All four blocks are computed in a single fused pass over matching atom pairs.
//
// Key optimisation over the naive per-pair dgemv approach:
//   For each (a, b, i2) group with M matching i1 atoms:
//   - Accumulate D_ew (rep,) = sum_m (-expdiag_m) * D_raw_m  — one dgemv replaces M jact dgemvs
//   - Jac, S_sum, V, and W remain per-i1 (SD1 varies per i1).
// ============================================================================
void kernel_gaussian_full(
    const std::vector<double> &x1,   // (nm1, max_atoms1, rep_size)
    const std::vector<double> &x2,   // (nm2, max_atoms2, rep_size)
    const std::vector<double> &dx1,  // (nm1, max_atoms1, rep_size, 3*max_atoms1)
    const std::vector<double> &dx2,  // (nm2, max_atoms2, rep_size, 3*max_atoms2)
    const std::vector<int> &q1,      // (nm1, max_atoms1)
    const std::vector<int> &q2,      // (nm2, max_atoms2)
    const std::vector<int> &n1,      // (nm1,)
    const std::vector<int> &n2,      // (nm2,)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq1,  // must equal 3 * sum(n1)
    int naq2,  // must equal 3 * sum(n2)
    double sigma,
    double *kernel_out  // ((nm1+naq1) x (nm2+naq2)), row-major
) {
    // ---- validation ----
    if (nm1 <= 0 || nm2 <= 0 || max_atoms1 <= 0 || max_atoms2 <= 0 || rep_size <= 0)
        throw std::invalid_argument("All dims must be positive.");
    if (!std::isfinite(sigma) || sigma <= 0.0)
        throw std::invalid_argument("sigma must be positive and finite.");
    if (!kernel_out) throw std::invalid_argument("kernel_out is null.");

    const size_t x1N = (size_t)nm1 * max_atoms1 * rep_size;
    const size_t x2N = (size_t)nm2 * max_atoms2 * rep_size;
    const size_t dx1N = (size_t)nm1 * max_atoms1 * rep_size * (3 * (size_t)max_atoms1);
    const size_t dx2N = (size_t)nm2 * max_atoms2 * rep_size * (3 * (size_t)max_atoms2);
    if (x1.size() != x1N || x2.size() != x2N) throw std::invalid_argument("x1/x2 size mismatch.");
    if (dx1.size() != dx1N || dx2.size() != dx2N)
        throw std::invalid_argument("dx1/dx2 size mismatch.");
    if (q1.size() != (size_t)nm1 * max_atoms1 || q2.size() != (size_t)nm2 * max_atoms2)
        throw std::invalid_argument("q1/q2 size mismatch.");
    if ((int)n1.size() != nm1 || (int)n2.size() != nm2)
        throw std::invalid_argument("n1/n2 size mismatch.");

    // Offsets in derivative axes
    std::vector<int> offs1(nm1), offs2(nm2);
    int sum1 = 0;
    for (int a = 0; a < nm1; ++a) {
        offs1[a] = sum1;
        sum1 += 3 * std::max(0, std::min(n1[a], max_atoms1));
    }
    int sum2 = 0;
    for (int b = 0; b < nm2; ++b) {
        offs2[b] = sum2;
        sum2 += 3 * std::max(0, std::min(n2[b], max_atoms2));
    }
    if (naq1 != sum1) throw std::invalid_argument("naq1 != 3*sum(n1)");
    if (naq2 != sum2) throw std::invalid_argument("naq2 != 3*sum(n2)");

    const int full_rows = nm1 + naq1;
    const int full_cols = nm2 + naq2;

    // Zero output
    std::fill(kernel_out, kernel_out + (size_t)full_rows * full_cols, 0.0);

    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma4 = -1.0 / (sigma * sigma * sigma * sigma);  // < 0
    const double sigma2_neg = -(sigma * sigma);                        // < 0

    // Label -> atom-index lists per molecule (set-1)
    std::vector<std::unordered_map<int, std::vector<int>>> lab_to_i1(nm1);
    for (int a = 0; a < nm1; ++a) {
        const int na = std::max(0, std::min(n1[a], max_atoms1));
        auto &m = lab_to_i1[a];
        m.reserve(64);
        for (int i1 = 0; i1 < na; ++i1)
            m[q1[(size_t)a * max_atoms1 + i1]].push_back(i1);
    }

    const int ncols_b_max = 3 * max_atoms2;
    const int ncols_a_max = 3 * max_atoms1;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::max(0, std::min(n2[b], max_atoms2));
        if (nb == 0) continue;
        const int ncols_b = 3 * nb;
        const int lda2 = 3 * max_atoms2;
        const int row2_off = offs2[b];  // offset in naq2 axis

        // Thread-local scratch
        double *D_row = aligned_alloc_64((size_t)rep_size);  // raw diff (reused)
        double *D_ew = aligned_alloc_64((size_t)rep_size);   // (-expdiag)-weighted D sum (for jact)
        double *sD_row = aligned_alloc_64((size_t)rep_size);    // s-scaled D_row (for V/W dgemv)
        double *W_row = aligned_alloc_64((size_t)ncols_b_max);  // W per i1 (for hessian rank-1)
        double *V_row = aligned_alloc_64((size_t)ncols_a_max);
        double *S_sum = aligned_alloc_64((size_t)rep_size * ncols_a_max);

        for (int a = 0; a < nm1; ++a) {
            const int na = std::max(0, std::min(n1[a], max_atoms1));
            if (na == 0) continue;
            const int ncols_a = 3 * na;
            const int lda1 = 3 * max_atoms1;
            const int col1_off = offs1[a];  // offset in naq1 axis

            const auto &label_i1 = lab_to_i1[a];

            double *scalar_ab = &kernel_out[(size_t)a * full_cols + b];
            double *jact_row_a = &kernel_out[(size_t)a * full_cols + nm2 + row2_off];
            double *jac_col_b = &kernel_out[(size_t)(nm1 + col1_off) * full_cols + b];
            double *hess_ab = &kernel_out[(size_t)(nm1 + col1_off) * full_cols + nm2 + row2_off];

            for (int i2 = 0; i2 < nb; ++i2) {
                const int label = q2[(size_t)b * max_atoms2 + i2];
                auto it = label_i1.find(label);
                if (it == label_i1.end()) continue;
                const auto &i1_list = it->second;
                if (i1_list.empty()) continue;

                const double *SD2 = &dx2[base_dx2(b, i2, nm2, max_atoms2, rep_size)];
                const double *x2_bi2 = &x2[((size_t)b * max_atoms2 + i2) * rep_size];

                const int M = (int)i1_list.size();

                // Zero accumulators for this (a, b, i2) group
                std::fill(S_sum, S_sum + (size_t)rep_size * ncols_a, 0.0);
                std::fill(D_ew, D_ew + rep_size, 0.0);

                // ---- Per-i1 scalar/gradient work; accumulate D_ew for jact ----
                for (int m = 0; m < M; ++m) {
                    const int i1 = i1_list[m];
                    const double *x1_ai1 = &x1[((size_t)a * max_atoms1 + i1) * rep_size];
                    const double *SD1 = &dx1[base_dx1(a, i1, nm1, max_atoms1, rep_size)];

                    double l2 = 0.0;
                    for (int k = 0; k < rep_size; ++k) {
                        const double diff = x1_ai1[k] - x2_bi2[k];
                        D_row[k] = diff;
                        l2 += diff * diff;
                    }
                    const double exp_base = std::exp(l2 * inv_2sigma2);
                    const double expd = inv_sigma4 * exp_base;
                    const double expdiag = sigma2_neg * expd;  // > 0
                    const double s = std::sqrt(-expd);

                    // Scalar
                    *scalar_ab += exp_base;

                    // Jacobian (SD1 varies per i1 — must stay per-i1)
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_a),
                        -expdiag,
                        SD1,
                        static_cast<blas_int>(lda1),
                        D_row,
                        static_cast<blas_int>(1),
                        1.0,
                        jac_col_b,
                        static_cast<blas_int>(full_cols)
                    );

// Accumulate D_ew for jact: D_ew += -expdiag * D_row
// (sign: jact coeff is -expdiag * SD2^T @ D_row, factor out SD2^T)
#pragma omp simd
                    for (int k = 0; k < rep_size; ++k)
                        D_ew[k] -= expdiag * D_row[k];

                    // S_sum += expdiag * SD1  (for hessian static term)
                    for (int k = 0; k < rep_size; ++k) {
                        double *srow = S_sum + (size_t)k * ncols_a;
                        const double *sdrow = SD1 + (size_t)k * lda1;
#pragma omp simd
                        for (int j = 0; j < ncols_a; ++j)
                            srow[j] += expdiag * sdrow[j];
                    }

// sD_row = s * D_row  (s-scaled diff, for V and W dgemv)
#pragma omp simd
                    for (int k = 0; k < rep_size; ++k)
                        sD_row[k] = s * D_row[k];

                    // V = -SD1^T @ sD_row  (ncols_a) — per-i1, feeds rank-1 hessian update
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_a),
                        -1.0,
                        SD1,
                        static_cast<blas_int>(lda1),
                        sD_row,
                        static_cast<blas_int>(1),
                        0.0,
                        V_row,
                        static_cast<blas_int>(1)
                    );

                    // W = SD2^T @ sD_row  (per-i1, feeds rank-1 hessian update)
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        SD2,
                        static_cast<blas_int>(lda2),
                        sD_row,
                        static_cast<blas_int>(1),
                        0.0,
                        W_row,
                        static_cast<blas_int>(1)
                    );

                    // Rank-1 hessian: hess_ab += V ⊗ W
                    cblas_dger(
                        CblasRowMajor,
                        static_cast<blas_int>(ncols_a),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        V_row,
                        static_cast<blas_int>(1),
                        W_row,
                        static_cast<blas_int>(1),
                        hess_ab,
                        static_cast<blas_int>(full_cols)
                    );
                }  // i1

                // ---- Jact: one dgemv on accumulated D_ew ----
                // jact_row_a += SD2^T @ D_ew  (D_ew already has -expdiag absorbed)
                cblas_dgemv(
                    CblasRowMajor,
                    CblasTrans,
                    static_cast<blas_int>(rep_size),
                    static_cast<blas_int>(ncols_b),
                    1.0,
                    SD2,
                    static_cast<blas_int>(lda2),
                    D_ew,
                    static_cast<blas_int>(1),
                    1.0,
                    jact_row_a,
                    static_cast<blas_int>(1)
                );

                // ---- Hessian static term: hess_ab += S_sum^T @ SD2 ----
                cblas_dgemm(
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    static_cast<blas_int>(ncols_a),
                    static_cast<blas_int>(ncols_b),
                    static_cast<blas_int>(rep_size),
                    1.0,
                    S_sum,
                    static_cast<blas_int>(ncols_a),
                    SD2,
                    static_cast<blas_int>(lda2),
                    1.0,
                    hess_ab,
                    static_cast<blas_int>(full_cols)
                );

            }  // i2
        }  // a

        aligned_free_64(D_row);
        aligned_free_64(D_ew);
        aligned_free_64(sD_row);
        aligned_free_64(W_row);
        aligned_free_64(V_row);
        aligned_free_64(S_sum);
    }  // b
}

// ============================================================================
// Full combined energy+force kernel (symmetric).
//
// Output layout (row-major, shape (nm + naq) x (nm + naq)):
//   K_full[0:nm,  0:nm]   = scalar block   (full, symmetric)
//   K_full[0:nm,  nm:]    = jacobian_t     (nm, naq)
//   K_full[nm:,   0:nm]   = jacobian       (naq, nm)
//   K_full[nm:,   nm:]    = hessian block  (lower triangle only — symmetric)
//
// Uses the same fused single-pass approach as the asymmetric variant.
// ============================================================================
void kernel_gaussian_full_symm(
    const std::vector<double> &x,   // (nm, max_atoms, rep_size)
    const std::vector<double> &dx,  // (nm, max_atoms, rep_size, 3*max_atoms)
    const std::vector<int> &q,      // (nm, max_atoms)
    const std::vector<int> &n,      // (nm,)
    int nm, int max_atoms, int rep_size,
    int naq,  // must equal 3 * sum(n)
    double sigma,
    double *kernel_out  // ((nm+naq) x (nm+naq)), row-major, full symmetric
) {
    // ---- validation ----
    if (nm <= 0 || max_atoms <= 0 || rep_size <= 0)
        throw std::invalid_argument("dims must be positive");
    if (!std::isfinite(sigma) || sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");
    if (!kernel_out) throw std::invalid_argument("kernel_out is null");

    const size_t xN = (size_t)nm * max_atoms * rep_size;
    const size_t dxN = (size_t)nm * max_atoms * rep_size * (3 * (size_t)max_atoms);
    if (x.size() != xN) throw std::invalid_argument("x size mismatch");
    if (dx.size() != dxN) throw std::invalid_argument("dx size mismatch");
    if (q.size() != (size_t)nm * max_atoms) throw std::invalid_argument("q size mismatch");
    if ((int)n.size() != nm) throw std::invalid_argument("n size mismatch");

    std::vector<int> offs(nm);
    int sum = 0;
    for (int m = 0; m < nm; ++m) {
        offs[m] = sum;
        sum += 3 * std::max(0, std::min(n[m], max_atoms));
    }
    if (naq != sum) throw std::invalid_argument("naq != 3*sum(n)");

    const int BIG = nm + naq;
    std::fill(kernel_out, kernel_out + (size_t)BIG * BIG, 0.0);

    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma4 = -1.0 / (sigma * sigma * sigma * sigma);
    const double sigma2_neg = -(sigma * sigma);

    // Label -> atom-index lists per molecule
    std::vector<std::unordered_map<int, std::vector<int>>> lab_to_idx(nm);
    for (int a = 0; a < nm; ++a) {
        const int na = std::max(0, std::min(n[a], max_atoms));
        auto &M2 = lab_to_idx[a];
        M2.reserve(64);
        for (int i1 = 0; i1 < na; ++i1)
            M2[q[(size_t)a * max_atoms + i1]].push_back(i1);
    }

    const int ncols_max = 3 * max_atoms;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < nm; ++b) {
        const int nb = std::max(0, std::min(n[b], max_atoms));
        if (nb == 0) continue;
        const int ncols_b = 3 * nb;
        const int lda_b = 3 * max_atoms;
        const int row_off = offs[b];  // naq offset for molecule b

        double *D_row = aligned_alloc_64((size_t)rep_size);
        double *D_ew_b =
            aligned_alloc_64((size_t)rep_size);  // +expdiag-weighted D sum for jact (b-side)
        double *V_row = aligned_alloc_64((size_t)ncols_max);
        double *W_row = aligned_alloc_64((size_t)ncols_max);
        double *S_sum = aligned_alloc_64((size_t)rep_size * ncols_max);
        double *xbv = aligned_alloc_64((size_t)rep_size);

        for (int a = 0; a <= b; ++a) {  // lower triangle only
            const int na = std::max(0, std::min(n[a], max_atoms));
            if (na == 0) continue;
            const int ncols_a = 3 * na;
            const int lda_a = 3 * max_atoms;
            const int col_off = offs[a];

            const auto &lab_a = lab_to_idx[a];

            // For diagonal (a==b): use a temp block for hessian to scatter lower-tri only.
            std::vector<double> Hdiag;
            double *Hdst = &kernel_out[(size_t)(nm + col_off) * BIG + (nm + row_off)];
            int Hdst_ld = BIG;
            if (a == b) {
                Hdiag.assign((size_t)ncols_b * ncols_a, 0.0);
                Hdst = Hdiag.data();
                Hdst_ld = ncols_a;
            }

            for (int j2 = 0; j2 < nb; ++j2) {
                const int lbl = q[(size_t)b * max_atoms + j2];
                auto it_a = lab_a.find(lbl);
                if (it_a == lab_a.end() || it_a->second.empty()) continue;
                const auto &i1_list = it_a->second;
                const int M = (int)i1_list.size();

                const double *SD_b = &dx[base_dx(b, j2, nm, max_atoms, rep_size)];
                for (int k = 0; k < rep_size; ++k)
                    xbv[k] = x[idx_x(b, j2, k, nm, max_atoms, rep_size)];

                std::fill(S_sum, S_sum + (size_t)rep_size * ncols_a, 0.0);
                std::fill(D_ew_b, D_ew_b + rep_size, 0.0);

                for (int m = 0; m < M; ++m) {
                    const int i1 = i1_list[m];
                    const double *x_ai1 = &x[idx_x(a, i1, 0, nm, max_atoms, rep_size)];
                    const double *SD_a = &dx[base_dx(a, i1, nm, max_atoms, rep_size)];

                    double l2 = 0.0;
                    for (int k = 0; k < rep_size; ++k) {
                        const double diff = x_ai1[k] - xbv[k];
                        D_row[k] = diff;
                        l2 += diff * diff;
                    }
                    const double eb = std::exp(l2 * inv_2sigma2);
                    const double e1 = inv_sigma4 * eb;
                    const double expdiag = sigma2_neg * e1;  // > 0
                    const double s = std::sqrt(-e1);

                    // Scalar
                    kernel_out[(size_t)a * BIG + b] += eb;
                    if (a != b) kernel_out[(size_t)b * BIG + a] += eb;

                    // Jac a-side: K_jac[nm+col_off+c, b] -= expdiag*(SD_a^T@D_row)[c]
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_a),
                        -expdiag,
                        SD_a,
                        static_cast<blas_int>(lda_a),
                        D_row,
                        static_cast<blas_int>(1),
                        1.0,
                        &kernel_out[(size_t)(nm + col_off) * BIG + b],
                        static_cast<blas_int>(BIG)
                    );

// Accumulate D_ew_b += expdiag * D_row  (for jact b-side: SD_b^T @ D_ew_b)
#pragma omp simd
                    for (int k = 0; k < rep_size; ++k)
                        D_ew_b[k] += expdiag * D_row[k];

                    // S_sum += expdiag * SD_a
                    for (int k = 0; k < rep_size; ++k) {
                        double *srow = S_sum + (size_t)k * ncols_a;
                        const double *sdrow = SD_a + (size_t)k * lda_a;
#pragma omp simd
                        for (int j = 0; j < ncols_a; ++j)
                            srow[j] += expdiag * sdrow[j];
                    }

// Scale D_row by s for hessian rank-1
#pragma omp simd
                    for (int k = 0; k < rep_size; ++k)
                        D_row[k] *= s;

                    // W = SD_b^T @ (s*D_row)
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        SD_b,
                        static_cast<blas_int>(lda_b),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        W_row,
                        static_cast<blas_int>(1)
                    );

                    // V = -SD_a^T @ (s*D_row)
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_a),
                        -1.0,
                        SD_a,
                        static_cast<blas_int>(lda_a),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        V_row,
                        static_cast<blas_int>(1)
                    );

                    // Rank-1 hessian: Hdst += V ⊗ W
                    cblas_dger(
                        CblasRowMajor,
                        static_cast<blas_int>(ncols_a),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        V_row,
                        static_cast<blas_int>(1),
                        W_row,
                        static_cast<blas_int>(1),
                        Hdst,
                        static_cast<blas_int>(Hdst_ld)
                    );
                }  // i1

                // ---- Batched jact: one dgemv per i2 group ----
                // K_jact[a, nm+row_off] += SD_b^T @ D_ew_b
                cblas_dgemv(
                    CblasRowMajor,
                    CblasTrans,
                    static_cast<blas_int>(rep_size),
                    static_cast<blas_int>(ncols_b),
                    1.0,
                    SD_b,
                    static_cast<blas_int>(lda_b),
                    D_ew_b,
                    static_cast<blas_int>(1),
                    1.0,
                    &kernel_out[(size_t)a * BIG + nm + row_off],
                    static_cast<blas_int>(1)
                );

                if (a != b) {
                    // Mirror jac b-side: K_jac[nm+row_off, a] += SD_b^T @ D_ew_b
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        SD_b,
                        static_cast<blas_int>(lda_b),
                        D_ew_b,
                        static_cast<blas_int>(1),
                        1.0,
                        &kernel_out[(size_t)(nm + row_off) * BIG + a],
                        static_cast<blas_int>(BIG)
                    );
                }

                // Hessian static term: Hdst += S_sum^T @ SD_b
                cblas_dgemm(
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    static_cast<blas_int>(ncols_a),
                    static_cast<blas_int>(ncols_b),
                    static_cast<blas_int>(rep_size),
                    1.0,
                    S_sum,
                    static_cast<blas_int>(ncols_a),
                    SD_b,
                    static_cast<blas_int>(lda_b),
                    1.0,
                    Hdst,
                    static_cast<blas_int>(Hdst_ld)
                );
            }  // j2

            // Since K_jact = K_jac^T for the symmetric kernel, copy the jac column into the
            // jact row for the off-diagonal (a!=b) mirror: K_jact[b, nm+col_off+c] =
            // K_jac[nm+col_off+c, b]
            if (a != b) {
                for (int c = 0; c < ncols_a; ++c)
                    kernel_out[(size_t)b * BIG + nm + col_off + c] =
                        kernel_out[(size_t)(nm + col_off + c) * BIG + b];
            }

            // Scatter hessian
            if (a == b) {
                for (int c = 0; c < ncols_a; ++c) {
                    const double *dcol = Hdiag.data() + (size_t)c * ncols_b;
                    for (int r = c; r < ncols_b; ++r)
                        kernel_out[(size_t)(nm + row_off + r) * BIG + (nm + col_off + c)] +=
                            dcol[r];
                }
            } else {
                for (int c = 0; c < ncols_a; ++c)
                    for (int r = 0; r < ncols_b; ++r) {
                        const double val =
                            kernel_out[(size_t)(nm + col_off + c) * BIG + (nm + row_off + r)];
                        kernel_out[(size_t)(nm + row_off + r) * BIG + (nm + col_off + c)] = val;
                    }
            }
        }  // a

        aligned_free_64(D_row);
        aligned_free_64(D_ew_b);
        aligned_free_64(V_row);
        aligned_free_64(W_row);
        aligned_free_64(S_sum);
        aligned_free_64(xbv);
    }  // b
}

// ============================================================================
// Full combined energy+force kernel (symmetric, RFP output).
//
// Output: 1-D array of length BIG*(BIG+1)/2 where BIG = nm + naq,
//         in RFP TRANSR='N', UPLO='U' format.
//
// Block structure in the upper triangle (row <= col):
//   [a, b]               with a <= b: scalar block
//   [a, nm+col_off+c]    for all a < nm, all c: jacobian_t block
//   [nm+col_off+c, nm+row_off+r] with col_off <= row_off (or col_off==row_off, c<=r): hessian block
// ============================================================================
void kernel_gaussian_full_symm_rfp(
    const std::vector<double> &x,   // (nm, max_atoms, rep_size)
    const std::vector<double> &dx,  // (nm, max_atoms, rep_size, 3*max_atoms)
    const std::vector<int> &q,      // (nm, max_atoms)
    const std::vector<int> &n,      // (nm,)
    int nm, int max_atoms, int rep_size,
    int naq,  // must equal 3 * sum(n)
    double sigma,
    double *arf  // length BIG*(BIG+1)/2, RFP TRANSR='N', UPLO='U'
) {
    // ---- validation ----
    if (nm <= 0 || max_atoms <= 0 || rep_size <= 0)
        throw std::invalid_argument("dims must be positive");
    if (!std::isfinite(sigma) || sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");
    if (!arf) throw std::invalid_argument("arf is null");

    const size_t xN = (size_t)nm * max_atoms * rep_size;
    const size_t dxN = (size_t)nm * max_atoms * rep_size * (3 * (size_t)max_atoms);
    if (x.size() != xN) throw std::invalid_argument("x size mismatch");
    if (dx.size() != dxN) throw std::invalid_argument("dx size mismatch");
    if (q.size() != (size_t)nm * max_atoms) throw std::invalid_argument("q size mismatch");
    if ((int)n.size() != nm) throw std::invalid_argument("n size mismatch");

    std::vector<int> offs(nm);
    int sum = 0;
    for (int m = 0; m < nm; ++m) {
        offs[m] = sum;
        sum += 3 * std::max(0, std::min(n[m], max_atoms));
    }
    if (naq != sum) throw std::invalid_argument("naq != 3*sum(n)");

    const int BIG = nm + naq;
    const size_t nt = (size_t)BIG * (BIG + 1ull) / 2ull;
    std::fill(arf, arf + nt, 0.0);

    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma4 = -1.0 / (sigma * sigma * sigma * sigma);
    const double sigma2_neg = -(sigma * sigma);

    std::vector<std::unordered_map<int, std::vector<int>>> lab_to_idx(nm);
    for (int a = 0; a < nm; ++a) {
        const int na = std::max(0, std::min(n[a], max_atoms));
        auto &M2 = lab_to_idx[a];
        M2.reserve(64);
        for (int i1 = 0; i1 < na; ++i1)
            M2[q[(size_t)a * max_atoms + i1]].push_back(i1);
    }

    const int ncols_max = 3 * max_atoms;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < nm; ++b) {
        const int nb = std::max(0, std::min(n[b], max_atoms));
        if (nb == 0) continue;
        const int ncols_b = 3 * nb;
        const int lda_b = 3 * max_atoms;
        const int row_off = offs[b];

        double *D_row = aligned_alloc_64((size_t)rep_size);
        double *D_ew = aligned_alloc_64((size_t)rep_size);       // expdiag-weighted D sum for jact
        double *jact_tmp = aligned_alloc_64((size_t)ncols_max);  // jact dgemv output buffer
        double *W_row = aligned_alloc_64((size_t)ncols_max);
        double *V_row = aligned_alloc_64((size_t)ncols_max);
        double *S_sum = aligned_alloc_64((size_t)rep_size * ncols_max);
        double *xbv = aligned_alloc_64((size_t)rep_size);

        for (int a = 0; a <= b; ++a) {  // upper triangle: b >= a
            const int na = std::max(0, std::min(n[a], max_atoms));
            if (na == 0) continue;
            const int ncols_a = 3 * na;
            const int lda_a = 3 * max_atoms;
            const int col_off = offs[a];

            const auto &lab_a = lab_to_idx[a];

            // Hessian temp block: ncols_a rows (a-side) × ncols_b cols (b-side)
            std::vector<double> Hblk((size_t)ncols_a * ncols_b, 0.0);

            for (int j2 = 0; j2 < nb; ++j2) {
                const int lbl = q[(size_t)b * max_atoms + j2];
                auto it_a = lab_a.find(lbl);
                if (it_a == lab_a.end() || it_a->second.empty()) continue;
                const auto &i1_list = it_a->second;

                const double *SD_b = &dx[base_dx(b, j2, nm, max_atoms, rep_size)];
                for (int k = 0; k < rep_size; ++k)
                    xbv[k] = x[idx_x(b, j2, k, nm, max_atoms, rep_size)];

                std::fill(S_sum, S_sum + (size_t)rep_size * ncols_a, 0.0);
                std::fill(D_ew, D_ew + rep_size, 0.0);

                for (int i1 : i1_list) {
                    const double *x_ai1 = &x[idx_x(a, i1, 0, nm, max_atoms, rep_size)];
                    const double *SD_a = &dx[base_dx(a, i1, nm, max_atoms, rep_size)];

                    double l2 = 0.0;
                    for (int k = 0; k < rep_size; ++k) {
                        const double diff = x_ai1[k] - xbv[k];
                        D_row[k] = diff;
                        l2 += diff * diff;
                    }
                    const double eb = std::exp(l2 * inv_2sigma2);
                    const double e1 = inv_sigma4 * eb;
                    const double expdiag = sigma2_neg * e1;
                    const double s = std::sqrt(-e1);

                    // --- Scalar ---
#ifdef _OPENMP
    #pragma omp atomic
#endif
                    arf[rfp_index_upper_N(BIG, a, b)] += eb;

// Accumulate D_ew += expdiag * D_row  (for jact b-side batched dgemv)
#pragma omp simd
                    for (int k = 0; k < rep_size; ++k)
                        D_ew[k] += expdiag * D_row[k];

                    // Mirror jact a-side: K_jact[b, nm+col_off+c] = -expdiag*(SD_a^T@D_row)[c]
                    // Only for off-diagonal (a != b)
                    if (a != b) {
                        cblas_dgemv(
                            CblasRowMajor,
                            CblasTrans,
                            static_cast<blas_int>(rep_size),
                            static_cast<blas_int>(ncols_a),
                            -expdiag,
                            SD_a,
                            static_cast<blas_int>(lda_a),
                            D_row,
                            static_cast<blas_int>(1),
                            0.0,
                            jact_tmp,
                            static_cast<blas_int>(1)
                        );
                        for (int c = 0; c < ncols_a; ++c) {
                            const size_t idx = rfp_index_upper_N(BIG, b, nm + col_off + c);
#ifdef _OPENMP
    #pragma omp atomic
#endif
                            arf[idx] += jact_tmp[c];
                        }
                    }

                    // S_sum for hessian diagonal term
                    for (int k = 0; k < rep_size; ++k) {
                        double *srow = S_sum + (size_t)k * ncols_a;
                        const double *sdrow = SD_a + (size_t)k * lda_a;
#pragma omp simd
                        for (int j = 0; j < ncols_a; ++j)
                            srow[j] += expdiag * sdrow[j];
                    }

// Scale D_row by s for hessian rank-1
#pragma omp simd
                    for (int k = 0; k < rep_size; ++k)
                        D_row[k] *= s;

                    // W = SD_b^T @ D_row_scaled  (ncols_b)
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        SD_b,
                        static_cast<blas_int>(lda_b),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        W_row,
                        static_cast<blas_int>(1)
                    );

                    // V = -SD_a^T @ D_row_scaled  (ncols_a)
                    cblas_dgemv(
                        CblasRowMajor,
                        CblasTrans,
                        static_cast<blas_int>(rep_size),
                        static_cast<blas_int>(ncols_a),
                        -1.0,
                        SD_a,
                        static_cast<blas_int>(lda_a),
                        D_row,
                        static_cast<blas_int>(1),
                        0.0,
                        V_row,
                        static_cast<blas_int>(1)
                    );

                    // rank-1 hessian into Hblk: V ⊗ W  (ncols_a rows × ncols_b cols)
                    cblas_dger(
                        CblasRowMajor,
                        static_cast<blas_int>(ncols_a),
                        static_cast<blas_int>(ncols_b),
                        1.0,
                        V_row,
                        static_cast<blas_int>(1),
                        W_row,
                        static_cast<blas_int>(1),
                        Hblk.data(),
                        static_cast<blas_int>(ncols_b)
                    );
                }  // i1

                // --- Batched jact: one dgemv on D_ew ---
                // K_jact[a, nm+row_off+r] += SD_b^T @ D_ew (D_ew = sum expdiag*D_row, b-side)
                cblas_dgemv(
                    CblasRowMajor,
                    CblasTrans,
                    static_cast<blas_int>(rep_size),
                    static_cast<blas_int>(ncols_b),
                    1.0,
                    SD_b,
                    static_cast<blas_int>(lda_b),
                    D_ew,
                    static_cast<blas_int>(1),
                    0.0,
                    jact_tmp,
                    static_cast<blas_int>(1)
                );
                for (int r = 0; r < ncols_b; ++r) {
                    const size_t idx = rfp_index_upper_N(BIG, a, nm + row_off + r);
#ifdef _OPENMP
    #pragma omp atomic
#endif
                    arf[idx] += jact_tmp[r];
                }

                // Hessian static term into Hblk: S_sum^T @ SD_b  (ncols_a × ncols_b)
                cblas_dgemm(
                    CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    static_cast<blas_int>(ncols_a),
                    static_cast<blas_int>(ncols_b),
                    static_cast<blas_int>(rep_size),
                    1.0,
                    S_sum,
                    static_cast<blas_int>(ncols_a),
                    SD_b,
                    static_cast<blas_int>(lda_b),
                    1.0,
                    Hblk.data(),
                    static_cast<blas_int>(ncols_b)
                );
            }  // j2

            // Scatter Hblk into RFP.
            // Hblk layout: ncols_a rows (a-side) × ncols_b cols (b-side)
            // Hblk[c, r] = H[nm+col_off+c, nm+row_off+r]
            // For a <= b: col_off <= row_off, so nm+col_off+c <= nm+row_off+r (upper-tri
            // condition).
            if (a == b) {
                // Diagonal a==b: col_off==row_off. Upper-tri means c <= r.
                for (int c = 0; c < ncols_a; ++c) {
                    const double *hrow = Hblk.data() + (size_t)c * ncols_b;
                    for (int r = c; r < ncols_b; ++r) {
                        const size_t idx =
                            rfp_index_upper_N(BIG, nm + col_off + c, nm + row_off + r);
#ifdef _OPENMP
    #pragma omp atomic
#endif
                        arf[idx] += hrow[r];
                    }
                }
            } else {
                // Off-diagonal a < b: col_off < row_off, nm+col_off+c < nm+row_off+r always.
                for (int c = 0; c < ncols_a; ++c) {
                    const double *hrow = Hblk.data() + (size_t)c * ncols_b;
                    for (int r = 0; r < ncols_b; ++r) {
                        const size_t idx =
                            rfp_index_upper_N(BIG, nm + col_off + c, nm + row_off + r);
#ifdef _OPENMP
    #pragma omp atomic
#endif
                        arf[idx] += hrow[r];
                    }
                }
            }
        }  // a

        aligned_free_64(D_row);
        aligned_free_64(D_ew);
        aligned_free_64(jact_tmp);
        aligned_free_64(W_row);
        aligned_free_64(V_row);
        aligned_free_64(S_sum);
        aligned_free_64(xbv);
    }  // b
}

}  // namespace fchl19
}  // namespace kf
