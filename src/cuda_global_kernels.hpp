// cuda_global_kernels.hpp — CUDA Gaussian kernel functions for global descriptors.
//
// Mirrors the CPU interface in global_kernels.hpp but operates entirely on GPU
// memory.  Callers must allocate device buffers; these functions do not manage
// host-device transfers.
//
// Naming follows global_kernels.hpp:
//   kernel_gaussian_full_symm_cu  — symmetric K_full for training
//   kernel_gaussian_full_matvec_cu — contracted inference (J^T·α trick)
//
// Memory layout convention (cuBLAS column-major throughout):
//   A numpy/torch (N, M) C-contiguous array has the same byte layout as a
//   cuBLAS column-major (M, N) matrix.  All d_* device pointers below follow
//   this convention: sizes are stated in the cuBLAS (rows, cols) sense.
#pragma once

namespace kf {

// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm_cu
//
// Build the symmetric energy+force kernel matrix K_full for training.
// Fills both triangles (lower via assemble_row_kernel, upper mirrored).
//
// Parameters (device pointers, cuBLAS col-major):
//   d_X    : (M, N) — descriptor columns
//   d_dXT  : (M, N*D) — Jacobian columns (dX[a,d,:] is column a*D+d)
//   d_K_full: (full, full) output, full = N*(1+D)
//   sigma  : Gaussian length-scale
//   N, M, D: dimensions
// ---------------------------------------------------------------------------
void kernel_gaussian_full_symm_cu(
    const float *d_X,
    const float *d_dXT,
    float       *d_K_full,
    float        sigma,
    int N, int M, int D
);


// ---------------------------------------------------------------------------
// kernel_gaussian_full_matvec_cu
//
// Contracted inference pass: compute E and F predictions without ever
// materialising the full test-train kernel matrix.  Uses the J^T·α trick:
//   alpha_desc_F[m, :] = Σ_d J_train[m,d,:] * alpha_F[m,d]   (precomputed)
//
// Parameters (device pointers, cuBLAS col-major):
//   d_X_q         : (M, N_q)   — query descriptors
//   d_dXT_q       : (M, N_q*D) — query Jacobian columns
//   d_X_t         : (M, N_t)   — training descriptors
//   d_alpha_E     : (N_t,)     — energy dual coefficients
//   d_alpha_desc_F: (M, N_t)   — contracted force weights (= J_t^T @ alpha_F)
//   d_E_pred      : (N_q,)     — energy predictions (output)
//   d_F_pred      : (N_q*D,)   — force predictions (N_q, D) row-major (output)
//   sigma, N_q, N_t, M, D: dimensions
// ---------------------------------------------------------------------------
void kernel_gaussian_full_matvec_cu(
    const float *d_X_q,
    const float *d_dXT_q,
    const float *d_X_t,
    const float *d_alpha_E,
    const float *d_alpha_desc_F,
    float       *d_E_pred,
    float       *d_F_pred,
    float        sigma,
    int N_q, int N_t, int M, int D
);


// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm_rfp_cu
//
// Build the symmetric energy+force kernel matrix K_full for training,
// stored directly in RFP packed format (TRANSR=N, UPLO=L).
// No dense BIG×BIG intermediate — each lower-triangle element is written
// once via rfp_index_lower_N; no mirror step needed.
//
// Parameters (device pointers, cuBLAS col-major):
//   d_X      : (M, N)              — descriptor columns
//   d_dXT    : (M, N*D)            — Jacobian columns
//   d_K_rfp  : (BIG*(BIG+1)/2,)   — output RFP buffer, BIG = N*(1+D)
//   sigma, N, M, D: dimensions
// ---------------------------------------------------------------------------
void kernel_gaussian_full_symm_rfp_cu(
    const float *d_X,
    const float *d_dXT,
    float       *d_K_rfp,
    float        sigma,
    int N, int M, int D
);


// ---------------------------------------------------------------------------
// kernel_gaussian_symm_rfp_cu
//
// Build the energy-only N×N Gaussian kernel matrix in RFP packed format.
// Convention: TRANSR=N, UPLO=L.
//
// Parameters (device pointers):
//   d_X    : (M, N) col-major — descriptor columns
//   d_K_rfp: (N*(N+1)/2,) — output RFP buffer
//   sigma  : Gaussian length-scale
//   N, M   : dimensions
// ---------------------------------------------------------------------------
void kernel_gaussian_symm_rfp_cu(
    const float *d_X,
    float       *d_K_rfp,
    float        sigma,
    int N, int M
);


// ---------------------------------------------------------------------------
// rfp_potrf_cu
//
// Adds l2 to the diagonal of a RFP matrix, then Cholesky-factorises in-place.
// On exit d_K_rfp holds the lower Cholesky factor L in RFP format.
// *info: 0 = success, >0 = matrix not positive definite (leading minor).
// Convention: TRANSR=N, UPLO=L.
// ---------------------------------------------------------------------------
void rfp_potrf_cu(float *d_K_rfp, int N, float l2, int *info);


// ---------------------------------------------------------------------------
// rfp_potrs_cu
//
// Triangular solve with Cholesky factor from rfp_potrf_cu.
// Solves (L * L^T) * X = B in-place, overwriting d_B.
// d_B is (N, nrhs) col-major with leading dimension N.
// Convention: TRANSR=N, UPLO=L.
// ---------------------------------------------------------------------------
void rfp_potrs_cu(const float *d_L_rfp, float *d_B, int N, int nrhs);

}  // namespace kf
