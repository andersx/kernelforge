#pragma once

#include <cstddef>

namespace kf::rff {

// Compute Random Fourier Features (basic):
//   Z[i, d] = sqrt(2/D) * cos(X[i,:] @ W[:,d] + b[d])
//
// X:  (N, rep_size) row-major - input features
// W:  (rep_size, D) row-major - random weight matrix
// b:  (D,) - random bias vector
// Z:  (N, D) row-major output
void rff_features(
    const double *X, const double *W, const double *b, std::size_t N, std::size_t rep_size,
    std::size_t D, double *Z
);

// Gradient of RFF features w.r.t. atomic coordinates:
//   G[i*ncoords + g, d] = d(Z[i,d]) / d(coord_g)
//                       = -sqrt(2/D) * sin(z[d]) * (W^T @ dX[i])_{d,g}
//
// X:            (N, rep_size) row-major - input features
// dX_T:         (rep_size, N_total*ncoords) row-major - feature Jacobians (transposed)
//               Note: N_total >= N (may be larger for chunked calls)
// W:            (rep_size, D) row-major
// b:            (D,)
// ncoords:      number of coordinates per molecule (= 3 * natoms, same for all)
// dX_T_stride:  leading dimension of dX_T (= N_total * ncoords)
// G:            (N*ncoords, D) row-major output (transposed from old layout)
void rff_gradient(
    const double *X, const double *dX_T, const double *W, const double *b, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t dX_T_stride, double *G
);

// Combined energy+force feature matrix (stacked):
//   Z_full[0:N, :]         = rff_features(X, W, b)         (N, D)
//   Z_full[N:N+N*ncoords, :] = rff_gradient(X, dX_T, W, b) (N*ncoords, D)
//
// Z_full satisfies: Z_full^T @ Z_full = Z^T@Z + G@G^T
//
// dX_T:         (rep_size, N*ncoords) row-major - transposed feature Jacobians
// dX_T_stride:  leading dimension of dX_T (= N * ncoords)
// Z_full:       (N + N*ncoords, D) row-major output
void rff_full(
    const double *X, const double *dX_T, const double *W, const double *b, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t dX_T_stride,
    double *Z_full
);

// Chunked symmetric Gramian for energy-only RFF training:
//   ZtZ = Z^T @ Z,  ZtY = Z^T @ Y
//
// X:          (N, rep_size) row-major
// W:          (rep_size, D) row-major
// b:          (D,)
// Y:          (N,) target energies
// chunk_size: molecules per chunk
// ZtZ:        (D, D) output - symmetric Gram matrix
// ZtY:        (D,) output
void rff_gramian_symm(
    const double *X, const double *W, const double *b, const double *Y, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t chunk_size, double *ZtZ, double *ZtY
);

// Chunked symmetric Gramian for force-only RFF training:
//   GtG = G @ G^T,  GtF = G @ F
//
// X:            (N, rep_size) row-major
// dX_T:         (rep_size, N*ncoords) row-major - transposed feature Jacobians
// W:            (rep_size, D) row-major
// b:            (D,)
// F:            (N*ncoords,) target forces
// ncoords:      coordinates per molecule (same for all)
// chunk_size:   molecules per chunk
// dX_T_stride:  leading dimension of dX_T (= N * ncoords)
// GtG:          (D, D) output - symmetric Gram matrix
// GtF:          (D,) output
void rff_gradient_gramian_symm(
    const double *X, const double *dX_T, const double *W, const double *b, const double *F,
    std::size_t N, std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t chunk_size,
    std::size_t dX_T_stride, double *GtG, double *GtF
);

// Chunked symmetric Gramian for energy+force RFF training:
//   ZtZ = Z^T@Z + G@G^T,  ZtY = Z^T@Y + G@F
//
// X:            (N, rep_size) row-major
// dX_T:         (rep_size, N*ncoords) row-major - transposed feature Jacobians
// W:            (rep_size, D) row-major
// b:            (D,)
// Y:            (N,) target energies
// F:            (N*ncoords,) target forces
// ncoords:      coordinates per molecule (same for all)
// energy_chunk: chunk size for energy loop
// force_chunk:  chunk size for force loop
// dX_T_stride:  leading dimension of dX_T (= N * ncoords)
// ZtZ:          (D, D) output
// ZtY:          (D,) output
void rff_full_gramian_symm(
    const double *X, const double *dX_T, const double *W, const double *b, const double *Y,
    const double *F, std::size_t N, std::size_t rep_size, std::size_t D, std::size_t ncoords,
    std::size_t energy_chunk, std::size_t force_chunk, std::size_t dX_T_stride, double *ZtZ,
    double *ZtY
);

// Chunked Gramian for energy-only RFF training, RFP-packed output:
//   ZtZ_rfp = pack_upper(Z^T @ Z),  ZtY = Z^T @ Y
//
// ZtZ_rfp: 1D array of length D*(D+1)/2, TRANSR='N', UPLO='U'
// ZtY:     (D,) output
void rff_gramian_symm_rfp(
    const double *X, const double *W, const double *b, const double *Y, std::size_t N,
    std::size_t rep_size, std::size_t D, std::size_t chunk_size, double *ZtZ_rfp, double *ZtY
);

// Chunked Gramian for force-only RFF training, RFP-packed output:
//   GtG_rfp = pack_upper(G @ G^T),  GtF = G @ F
//
// dX_T:        (rep_size, N*ncoords) row-major - transposed feature Jacobians
// dX_T_stride: leading dimension of dX_T (= N * ncoords)
// GtG_rfp:     1D array of length D*(D+1)/2, TRANSR='N', UPLO='U'
// GtF:         (D,) output
void rff_gradient_gramian_symm_rfp(
    const double *X, const double *dX_T, const double *W, const double *b, const double *F,
    std::size_t N, std::size_t rep_size, std::size_t D, std::size_t ncoords, std::size_t chunk_size,
    std::size_t dX_T_stride, double *GtG_rfp, double *GtF
);

// Chunked Gramian for energy+force RFF training, RFP-packed output:
//   ZtZ_rfp = pack_upper(Z^T@Z + G@G^T),  ZtY = Z^T@Y + G@F
//
// dX_T:        (rep_size, N*ncoords) row-major - transposed feature Jacobians
// dX_T_stride: leading dimension of dX_T (= N * ncoords)
// ZtZ_rfp:     1D array of length D*(D+1)/2, TRANSR='N', UPLO='U'
// ZtY:         (D,) output
void rff_full_gramian_symm_rfp(
    const double *X, const double *dX_T, const double *W, const double *b, const double *Y,
    const double *F, std::size_t N, std::size_t rep_size, std::size_t D, std::size_t ncoords,
    std::size_t energy_chunk, std::size_t force_chunk, std::size_t dX_T_stride, double *ZtZ_rfp,
    double *ZtY
);

}  // namespace kf::rff
