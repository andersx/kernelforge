/*
 * cuda_krr.h — Public C API for GPU-accelerated Gaussian KRR (energy + force).
 *
 * This header uses only plain C types (float*, int, float) so it can be
 * included from both CUDA (.cu) and regular C++ (.cpp) translation units
 * without pulling in CUDA headers.
 *
 * Memory layout convention (matches cuBLAS column-major):
 *   A numpy (N, M) C-contiguous array has the same byte layout as a cuBLAS
 *   column-major (M, N) matrix.  So we always pass (N, M) row-major numpy
 *   arrays and let cuBLAS interpret them as (M, N) column-major — no copies.
 *
 *   X    : numpy (N, M)   <-> cuBLAS (M, N)   col-major  (descriptor columns)
 *   dXT  : numpy (N*D, M) <-> cuBLAS (M, N*D) col-major  (Jacobian columns)
 *          equivalently: reshape numpy (N, D, M) to (N*D, M) before passing.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * KrrInferenceState
 *
 * Persistent GPU state kept between predict() calls.
 * Training Jacobians are fully contracted into W_F_bar during
 * cuda_krr_train_ef() and are never needed again at inference.
 * ============================================================ */
typedef struct {
    /* ---- Persistent GPU buffers (allocated once after training) ---- */
    float *d_X_train;      /* (M, Nt)   training descriptors              */
    float *d_W_F_bar;      /* (M, Nt)   J_b^T @ alpha_F_b (contracted J) */
    float *d_W_combined;   /* (M, Nt)   W_F_bar + alpha_E[:,] * X_train  */
    float *d_W_F_self;     /* (Nt,)     W_F_bar[:,b] . X_train[:,b]       */
    float *d_alpha_E;      /* (Nt,)     energy dual coefficients          */
    float *d_norms_tr;     /* (Nt,)     ||X_train[:,b]||^2                */
    float *d_ones;         /* (Nt,)     all-ones for row-sum SGEMV        */

    float sigma;
    float inv_s2;   /* 1 / sigma^2 */
    float sigma2;   /* sigma^2     */
    int   N_train, M, D;

    /* ---- Persistent workspace — grows on demand, freed in cuda_krr_free() ---- */
    int   ws_N_test;
    float *ws_X_test;
    float *ws_dXT_test;
    float *ws_norms_te;
    float *ws_C;
    float *ws_cross_F;
    float *ws_weight_F;
    float *ws_sum_F;
    float *ws_w_E;
    float *ws_E_pred;
    float *ws_G_full;
    float *ws_combined;
    float *ws_F_pred;
} KrrInferenceState;


/* ============================================================
 * cuda_krr_train_ef
 *
 * Full E+F KRR training on GPU.  Builds the symmetric
 * (N*(1+D)) x (N*(1+D)) kernel matrix, regularises with
 * lambda*I, and solves (K + lam*I) alpha = [E, -F] via
 * Cholesky.  Then initialises *st for immediate predict() use.
 *
 * Parameters
 * ----------
 * st      : zeroed KrrInferenceState (caller-owned; not freed here)
 * h_X     : (N, M)   float32 C-contiguous
 * h_dXT   : (N*D, M) float32 C-contiguous  (i.e. dX (N,D,M) reshaped)
 * h_E     : (N,)     float32 physical energies
 * h_F     : (N*D,)   float32 physical forces F = -dE/dR  (negated internally)
 * h_alpha : (N*(1+D),) float32 output — alpha written here on return
 * sigma   : Gaussian kernel length-scale
 * lambda  : L2 regularisation strength
 * N, M, D : n_train, descriptor_dim, 3*n_atoms
 * ============================================================ */
void cuda_krr_train_ef(KrrInferenceState *st,
                       const float *h_X,
                       const float *h_dXT,
                       const float *h_E,
                       const float *h_F,
                       float       *h_alpha,
                       float sigma, float lambda,
                       int N, int M, int D);


/* ============================================================
 * cuda_krr_predict
 *
 * Predict E and F for a batch of N_test molecules.
 * *st must be initialised by cuda_krr_train_ef or cuda_krr_load_state.
 *
 * Parameters
 * ----------
 * h_X_test   : (N_test, M)   float32 C-contiguous
 * h_dXT_test : (N_test*D, M) float32 C-contiguous
 * h_E_pred   : (N_test,)     float32 output
 * h_F_pred   : (N_test*D,)   float32 output  — physical forces F = -dE/dR
 * N_test     : number of test molecules
 * ============================================================ */
void cuda_krr_predict(KrrInferenceState *st,
                      const float *h_X_test,
                      const float *h_dXT_test,
                      float       *h_E_pred,
                      float       *h_F_pred,
                      int          N_test);


/* ============================================================
 * cuda_krr_get_state
 *
 * Copy the persistent GPU inference arrays to host buffers.
 * Used for serialising the model (allows save without storing dX_tr).
 * All output buffers must be pre-allocated to the right size by the caller.
 *
 * h_X_train    : (N, M)  [N*M floats]
 * h_W_F_bar    : (N, M)  [N*M floats]
 * h_W_combined : (N, M)  [N*M floats]
 * h_W_F_self   : (N,)    [N floats]
 * h_alpha_E    : (N,)    [N floats]
 * h_norms_tr   : (N,)    [N floats]
 * ============================================================ */
void cuda_krr_get_state(const KrrInferenceState *st,
                        float *h_X_train,
                        float *h_W_F_bar,
                        float *h_W_combined,
                        float *h_W_F_self,
                        float *h_alpha_E,
                        float *h_norms_tr);


/* ============================================================
 * cuda_krr_load_state
 *
 * Reconstruct the inference state from previously saved arrays.
 * No dX_tr is needed — W_F_bar etc. were precomputed at train time.
 * ============================================================ */
void cuda_krr_load_state(KrrInferenceState *st,
                         const float *h_X_train,
                         const float *h_W_F_bar,
                         const float *h_W_combined,
                         const float *h_W_F_self,
                         const float *h_alpha_E,
                         const float *h_norms_tr,
                         float sigma, int N_train, int M, int D);


/* ============================================================
 * cuda_krr_free
 *
 * Release all GPU memory held by *st.
 * Does NOT free the struct itself (caller owns it).
 * Safe to call on a zero-initialised struct (no-op).
 * ============================================================ */
void cuda_krr_free(KrrInferenceState *st);


/* ============================================================
 * cuda_krr_build_kernel_matrix  [DIAGNOSTIC]
 *
 * Build the full symmetric K_full matrix on GPU and return it
 * to the caller.  Used by the diagnostic comparison script to
 * verify the GPU kernel assembly against the CPU reference.
 *
 * Parameters
 * ----------
 * h_X     : (N, M)   float32 C-contiguous
 * h_dXT   : (N*D, M) float32 C-contiguous
 * h_K_full: (N*(1+D), N*(1+D)) float32 output — fully symmetric
 *           stored column-major (cuBLAS convention), but since the
 *           matrix is symmetric, reshaping as row-major gives K too.
 * sigma   : Gaussian kernel length-scale
 * N, M, D : dimensions
 * ============================================================ */
void cuda_krr_build_kernel_matrix(
    const float *h_X,
    const float *h_dXT,
    float       *h_K_full,
    float        sigma,
    int N, int M, int D);

#ifdef __cplusplus
}
#endif
