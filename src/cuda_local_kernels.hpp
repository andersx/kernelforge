// cuda_local_kernels.hpp — CUDA Gaussian kernel functions for local (FCHL19) descriptors.
//
// Mirrors the CPU interface in local_kernels.hpp but operates on GPU memory.
// Callers must allocate device buffers; no host-device transfers are performed here.
//
// Memory layout convention (C-contiguous / row-major throughout):
//   X[m, i, k]       at d_X[(m*max_atoms + i)*rep_size + k]
//   dX[m, i, k, c]   at d_dX[((m*max_atoms + i)*rep_size + k)*3*max_atoms + c]
//   Q[m, i]           at d_Q[m*max_atoms + i]
//   N[m]              at d_N[m]
//   K_full[r, c]      at d_K_full[r*BIG + c]       BIG = nm + naq
//   alpha_desc[m,i,k] at d_alpha_desc[(m*max_atoms+i)*rep_size + k]
#pragma once

namespace kf {
namespace fchl19 {

// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm_local_cu
//
// Build the symmetric energy+force training kernel matrix K_full.
// Fills all four blocks: K_EE (nm×nm), K_FE (nm×naq), K_EF (naq×nm), K_FF (naq×naq).
//
// Parameters (device pointers, C-contiguous row-major):
//   d_X       : (nm, max_atoms, rep_size)                   descriptor vectors
//   d_dX      : (nm, max_atoms, rep_size, 3*max_atoms)      Jacobians
//   d_Q       : (nm, max_atoms) int32                       atomic labels
//   d_N       : (nm,) int32                                 active atom counts
//   d_K_full  : (BIG, BIG) float32 output, BIG = nm + naq
//   sigma     : Gaussian length-scale
//   nm, max_atoms, rep_size, naq : dimensions (naq = 3*sum(N))
// ---------------------------------------------------------------------------
void kernel_gaussian_full_symm_local_cu(
    const float *d_X,
    const float *d_dX,
    const int   *d_Q,
    const int   *d_N,
    float       *d_K_full,
    float        sigma,
    int nm, int max_atoms, int rep_size, int naq
);

// ---------------------------------------------------------------------------
// compute_alpha_desc_local_cu
//
// GPU version of kernel_gaussian_local_compute_alpha_desc.
// Computes alpha_desc[b, i2, k] = Σ_{c=0}^{3*N[b]-1} dX[b,i2,k,c] * alpha_F[offs[b]+c]
//
// Parameters:
//   d_dX        : (nm, max_atoms, rep_size, 3*max_atoms)  training Jacobians
//   d_alpha_F   : (naq,)  Cartesian force coefficients from the KRR solve
//   d_N         : (nm,) int32  active atom counts
//   d_alpha_desc: (nm, max_atoms, rep_size) float32 output
//   sigma, nm, max_atoms, rep_size, naq : dimensions
// ---------------------------------------------------------------------------
void compute_alpha_desc_local_cu(
    const float *d_dX,
    const float *d_alpha_F,
    const int   *d_N,
    float       *d_alpha_desc,
    int nm, int max_atoms, int rep_size, int naq
);

// ---------------------------------------------------------------------------
// kernel_gaussian_full_matvec_local_cu
//
// Contracted inference using the J^T·α trick (local version).
// Does not materialise the full test-train kernel matrix.
//
// For each query atom (a, i1) with element z:
//   - Accumulate over training atoms (b, i2) with same element:
//       exp_base  = exp(-||X_q[a,i1]-X_t[b,i2]||²/(2σ²))
//       expdiag   = exp_base/σ²
//       expd      = -exp_base/σ⁴
//       inner_F   = (X_q[a,i1]-X_t[b,i2]) · alpha_desc[b,i2]
//       E[a]     += exp_base * alpha_E[b] + expdiag * inner_F
//       G_acc[k] += expdiag*alpha_desc[b,i2,k]
//                 + expd*inner_F*(X_q[a,i1,k]-X_t[b,i2,k])
//                 + expdiag*alpha_E[b]*X_t[b,i2,k]
//       w_E      += expdiag * alpha_E[b]
//   - Self-correction: G_acc[k] -= w_E * X_q[a,i1,k]
//   - Back-project:    F[offs_q[a]+c] += Σ_k dX_q[a,i1,k,c] * G_acc[k]
//
// Parameters:
//   d_X_q, d_dX_q, d_Q_q, d_N_q : query descriptors / Jacobians / labels / counts
//   d_X_t, d_Q_t, d_N_t          : training descriptors / labels / counts
//   d_alpha_E    : (nm_t,)                      energy dual coefficients
//   d_alpha_desc : (nm_t, max_atoms_t, rep_size) precomputed force weights
//   d_E_pred     : (nm_q,)          output energy predictions
//   d_F_pred     : (naq_q,)         output force predictions (flat Cartesian)
// ---------------------------------------------------------------------------
void kernel_gaussian_full_matvec_local_cu(
    const float *d_X_q,
    const float *d_dX_q,
    const int   *d_Q_q,
    const int   *d_N_q,
    const float *d_X_t,
    const int   *d_Q_t,
    const int   *d_N_t,
    const float *d_alpha_E,
    const float *d_alpha_desc,
    float       *d_E_pred,
    float       *d_F_pred,
    float        sigma,
    int nm_q, int nm_t,
    int max_atoms_q, int max_atoms_t,
    int rep_size, int naq_q
);

}  // namespace fchl19
}  // namespace kf
