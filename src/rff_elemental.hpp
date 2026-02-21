#pragma once

#include <cstddef>

namespace kf::rff {

// Compute element-stratified Random Fourier Features with per-molecule sum:
//   For each element e in [0, nelements):
//     Gather atoms of element e -> Xsort
//     Z_e = sqrt(2/D) * cos(Xsort @ W[e] + b[e])
//     For each molecule, sum Z_e rows belonging to that molecule into LZ
//
// X:         (nmol, max_atoms, rep_size) row-major - padded representations
// Q:         (nmol, max_atoms) row-major, int32 - element indices (0-indexed,
//            -1 for padding). Element e means W[e], b[e].
// sizes:     (nmol,) int32 - number of real atoms per molecule
// W:         (nelements, rep_size, D) row-major - per-element weight matrices
// b:         (nelements, D) row-major - per-element bias vectors
// LZ:        (nmol, D) row-major output - summed features per molecule
void rff_features_elemental(
    const double *X, const int *Q, const int *sizes,
    const double *W, const double *b,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    double *LZ);

// Compute chunked Gramian for energy-only RFF training:
//   LZtLZ = sum_chunks  LZ_chunk^T @ LZ_chunk   (D, D)
//   LZtY  = sum_chunks  LZ_chunk^T @ Y_chunk     (D,)
//
// Calls rff_features_elemental internally for each chunk, then accumulates
// via DSYRK and DGEMV.
//
// X, Q, sizes, W, b: same as rff_features_elemental
// Y:          (nmol,) target energies
// chunk_size: number of molecules per chunk
// LZtLZ:      (D, D) row-major output — symmetric Gram matrix
// LZtY:       (D,) output — projection vector
void rff_gramian_elemental(
    const double *X, const int *Q, const int *sizes,
    const double *W, const double *b,
    const double *Y,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t chunk_size,
    double *LZtLZ, double *LZtY);

// Compute chunked Gramian for energy+force RFF training.
//   LZtLZ accumulates both energy and force contributions:
//     Energy: LZ^T @ LZ  (from rff_features_elemental, chunked)
//     Forces: G @ G^T    (from rff_gradient_elemental, chunked)
//   LZtY accumulates both:
//     Energy: LZ^T @ Y
//     Forces: G @ F
//
// All inputs same as rff_gramian_elemental plus:
// dX:           (nmol, max_atoms, rep_size, max_atoms, 3) row-major
// F:            (ngrads_total,) force targets, ngrads_total = 3 * sum(sizes)
// energy_chunk: chunk size for energy loop
// force_chunk:  chunk size for force loop
// LZtLZ:        (D, D) row-major output
// LZtY:         (D,) output
void rff_gramian_elemental_gradient(
    const double *X, const double *dX,
    const int *Q, const int *sizes,
    const double *W, const double *b,
    const double *Y, const double *F,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t energy_chunk, std::size_t force_chunk,
    double *LZtLZ, double *LZtY);

// Compute gradient of element-stratified RFF features w.r.t. atomic coords.
//
// For each molecule i, atom j with element e:
//   z_j = X[i,j,:] @ W[e] + b[e]                        (D,)
//   dg[d, r] = -sin(z_j[d]) * sqrt(2/D) * W[e][r, d]   (D, rep_size)
//   G[:, g_start:g_start+3*natoms] += dg @ dX[i,j]
//
// X:       (nmol, max_atoms, rep_size) row-major
// dX:      (nmol, max_atoms, rep_size, max_atoms, 3) row-major
//          dX[i,j,r,k,xyz] = d(repr_r of atom j in mol i) / d(coord xyz of atom k)
// Q:       (nmol, max_atoms) row-major, int32, 0-indexed element indices, -1=pad
// sizes:   (nmol,) int32
// W:       (nelements, rep_size, D) row-major
// b:       (nelements, D) row-major
// ngrads:  3 * sum(sizes)
// G:       (D, ngrads) row-major output
//          G[d, g] = derivative of feature d w.r.t. gradient component g
void rff_gradient_elemental(
    const double *X, const double *dX,
    const int *Q, const int *sizes,
    const double *W, const double *b,
    std::size_t nmol, std::size_t max_atoms, std::size_t rep_size,
    std::size_t nelements, std::size_t D,
    std::size_t ngrads,
    double *G);

}  // namespace kf::rff
