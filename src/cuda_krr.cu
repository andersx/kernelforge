/*
 * cuda_krr.cu — GPU-accelerated Gaussian KRR with energy + force training.
 *
 * Kernel:  K(x_a, x_b) = exp(-||x_a - x_b||^2 / (2 sigma^2))
 *
 * Data layout (cuBLAS column-major throughout):
 *   X    : (M, N)   — each column is a molecular descriptor
 *   dXT  : (M, N*D) — column a*D+d is the Jacobian dX_a/d(coord_d)
 *
 * Training (symmetric, lower-triangle only):
 *   K_full (N*(1+D))^2 built via row-batched SGEMM.
 *   Solved with Cholesky: (K_full + lam*I) alpha = [E, -F].
 *
 * Inference (contracted, no K_full_test):
 *   Persistent state: X_train, W_F_bar = J^T alpha_F, W_combined, norms, alpha_E.
 *   Per query: 4 SGEMMs + 1 batched SGEMV.  E = sigma^2*(w_E + sum_F).
 *
 * Public API is in cuda_krr.h — all functions in this file that are not
 * declared there are internal (static or file-scope).
 */

#include "cuda_krr.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


/* ============================================================
 * Error-checking macros
 * ============================================================ */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            abort();                                                             \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error at %s:%d — status %d\n",             \
                    __FILE__, __LINE__, (int)_s);                               \
            abort();                                                             \
        }                                                                       \
    } while (0)

#define CUSOLVER_CHECK(call)                                                    \
    do {                                                                        \
        cusolverStatus_t _s = (call);                                           \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuSOLVER error at %s:%d — status %d\n",           \
                    __FILE__, __LINE__, (int)_s);                               \
            abort();                                                             \
        }                                                                       \
    } while (0)


/* ============================================================
 * Module-level cuBLAS / cuSOLVER handles (lazy initialised).
 * One pair per process — all public API calls share them.
 * ============================================================ */

static cublasHandle_t     s_cublas   = NULL;
static cusolverDnHandle_t s_cusolver = NULL;

static void ensure_handles(void)
{
    if (!s_cublas) {
        if (cublasCreate(&s_cublas) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed\n");
            abort();
        }
    }
    if (!s_cusolver) {
        if (cusolverDnCreate(&s_cusolver) != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "cusolverDnCreate failed\n");
            abort();
        }
    }
}


/* ============================================================
 * CUDA kernels (unchanged from the reference krr_gpu.cu)
 * ============================================================ */

/*
 * compute_sqnorms_kernel
 * norms[i] = ||X[:,i]||^2   where X is (M, N) col-major.
 */
__global__ void compute_sqnorms_kernel(const float *X, float *norms, int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    const float *x = X + (long long)col * M;
    float s = 0.0f;
    for (int i = 0; i < M; i++) s += x[i] * x[i];
    norms[col] = s;
}


/*
 * add_sym_norms_kernel
 * G[a,b] += (norms[a] + norms[b])   symmetric case.
 */
__global__ void add_sym_norms_kernel(float *G, const float *norms, int N)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;
    G[a + (long long)b * N] += norms[a] + norms[b];
}


/*
 * add_asym_norms_kernel
 * G[a,b] += (norms1[a] + norms2[b])   asymmetric case.
 */
__global__ void add_asym_norms_kernel(float *G, const float *norms1,
                                      const float *norms2, int N1, int N2)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N1 || b >= N2) return;
    G[a + (long long)b * N1] += norms1[a] + norms2[b];
}


/*
 * build_C_C4_kernel
 * On entry:  G[a,b] = ||x_a - x_b||^2
 * On exit:   G[a,b] = K[a,b] * inv_s2      (= C)
 *            C4[a,b]= K[a,b] * inv_s2^2    (= C4)
 */
__global__ void build_C_C4_kernel(float *G, float *C4,
                                   float inv_s2, int N1, int N2)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N1 || b >= N2) return;
    long long idx = a + (long long)b * N1;
    float k = expf(-0.5f * inv_s2 * G[idx]);
    G[idx]  = k * inv_s2;
    C4[idx] = k * inv_s2 * inv_s2;
}


/*
 * extract_U_kernel
 * U[g] = V1X2[g, g/D]   (self-projections, molecule a = g/D).
 */
__global__ void extract_U_kernel(const float *V1X2, float *U, int N, int D)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int ND = N * D;
    if (g >= ND) return;
    int a = g / D;
    U[g] = V1X2[g + (long long)a * ND];
}


/*
 * assemble_row_kernel
 *
 * Assembles one block-row a of K_full from C, C4, G_row, V1X2, U1.
 * Called once per row a; grid = a+1 blocks (one per column b=0..a).
 * Each block has D*D threads writing the (1+D)x(1+D) sub-block.
 */
__global__ void assemble_row_kernel(
    float *K_full,
    const float *G_row,
    int G_lda,
    const float *V1X2,
    const float *U1,
    const float *C,
    const float *C4,
    float sigma2,
    int a, int N, int D,
    long long full_rows)
{
    int b   = blockIdx.x;
    int tid = threadIdx.x;
    int DD  = D * D;
    if (tid >= DD) return;

    int d1 = tid / D;
    int d2 = tid % D;
    int ND = N * D;
    int g1 = a * D + d1;
    int g2 = b * D + d2;

    float c  = C [a + (long long)b * N];
    float c4 = C4[a + (long long)b * N];

    float p = V1X2[g1 + (long long)b * ND] - U1[g1];
    float q = V1X2[g2 + (long long)a * ND] - U1[g2];

    float g = G_row[d1 + (long long)(b * D + d2) * G_lda];

    /* K_EE */
    if (tid == 0)
        K_full[a + (long long)b * full_rows] = c * sigma2;

    /* K_FE for (a, b) */
    if (d2 == 0)
        K_full[(N + g1) + (long long)b * full_rows] = c * p;
    /* K_FE mirror for (b, a) — only when a > b */
    if (a > b && d1 == 0)
        K_full[(N + g2) + (long long)a * full_rows] = c * q;

    /* K_FF lower triangle */
    if (a > b || g1 >= g2)
        K_full[(N + g1) + (long long)(N + g2) * full_rows] =
            c * g + c4 * p * q;
}


/*
 * add_diagonal_kernel
 * A[i,i] += val
 */
__global__ void add_diagonal_kernel(float *A, float val, long long N)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) A[i + i * N] += val;
}


/*
 * dot_diagonal_kernel
 * result[a] += A[:,a] . B[:,a]   (accumulate — init to 0 first).
 */
__global__ void dot_diagonal_kernel(float *result, const float *A, const float *B,
                                     int M, int N)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= N) return;
    const float *Acol = A + (long long)a * M;
    const float *Bcol = B + (long long)a * M;
    float s = 0.0f;
    for (int m = 0; m < M; m++) s += Acol[m] * Bcol[m];
    result[a] += s;
}


/*
 * fused_energy_combined_kernel
 * E_pred[a]   = sigma2 * (w_E[a] + sum_F[a])
 * combined[a] = w_E[a] + sum_F[a]
 */
__global__ void fused_energy_combined_kernel(float *E_pred, float *combined,
                                              const float *w_E, const float *sum_F,
                                              float sigma2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float c = w_E[i] + sum_F[i];
    combined[i] = c;
    E_pred[i]   = sigma2 * c;
}


/*
 * subtract_scaled_cols_kernel
 * R[m, a] -= scalar[a] * X[m, a]
 */
__global__ void subtract_scaled_cols_kernel(float *R, const float *X,
                                             const float *scalar, int M, int N)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || a >= N) return;
    R[m + (long long)a * M] -= scalar[a] * X[m + (long long)a * M];
}


/*
 * build_C_kernel
 * In-place: G[a,b] = exp(-0.5 * inv_s2 * G[a,b]) * inv_s2
 */
__global__ void build_C_kernel(float *G, float inv_s2, int N1, int N2)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N1 || b >= N2) return;
    long long idx = a + (long long)b * N1;
    G[idx] = expf(-0.5f * inv_s2 * G[idx]) * inv_s2;
}


/*
 * build_weight_F_kernel
 * weight_F[a,b] = C[a,b] * inv_s2 * (cross_F[a,b] - self_cross_F[b])
 */
__global__ void build_weight_F_kernel(float *weight_F, const float *C,
                                       const float *cross_F,
                                       const float *self_cross_F,
                                       float inv_s2,
                                       int N_test, int N_train)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N_test || b >= N_train) return;
    long long idx = a + (long long)b * N_test;
    weight_F[idx] = C[idx] * inv_s2 * (cross_F[idx] - self_cross_F[b]);
}


/* ============================================================
 * build_kernel_gpu_symmetric
 *
 * Builds the symmetric K_full training matrix (lower triangle).
 * No G_all (ND x ND) intermediate is ever materialised.
 * ============================================================ */

#define BUILD_B_ROWS 64

static void build_kernel_gpu_symmetric(
    const float *d_X,    const float *d_dXT,
    float *d_K_full,
    float sigma, int N, int M, int D)
{
    long long ND        = (long long)N * D;
    long long full_rows = (long long)N * (1 + D);

    float inv_s2 = 1.0f / (sigma * sigma);
    float sigma2 = sigma * sigma;

    float *d_norms, *d_C, *d_C4;
    float *d_V1X2, *d_U1;
    float *d_G_batch;

    CUDA_CHECK(cudaMalloc(&d_norms,   (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,       (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C4,      (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V1X2,    ND * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U1,      ND * sizeof(float)));
    {
        int brows = (N < BUILD_B_ROWS) ? N : BUILD_B_ROWS;
        CUDA_CHECK(cudaMalloc(&d_G_batch, (long long)brows * D * ND * sizeof(float)));
    }

    const float one = 1.0f, neg2 = -2.0f, zero = 0.0f;

    /* Phase 1: squared distances -> C, C4 */
    compute_sqnorms_kernel<<<(N + 255) / 256, 256>>>(d_X, d_norms, M, N);
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, M, &neg2, d_X, M, d_X, M, &zero, d_C, N));
    {
        dim3 blk(16, 16);
        dim3 grd((N + 15) / 16, (N + 15) / 16);
        add_sym_norms_kernel<<<grd, blk>>>(d_C, d_norms, N);
        build_C_C4_kernel<<<grd, blk>>>(d_C, d_C4, inv_s2, N, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Zero K_full */
    CUDA_CHECK(cudaMemset(d_K_full, 0, full_rows * full_rows * sizeof(float)));

    /* Phase 2: V1X2, U1 */
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)ND, N, M,
        &one, d_dXT, M, d_X, M,
        &zero, d_V1X2, (int)ND));
    extract_U_kernel<<<((int)ND + 255) / 256, 256>>>(d_V1X2, d_U1, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 3: row-batched SGEMM + assembly */
    {
        int brows = (N < BUILD_B_ROWS) ? N : BUILD_B_ROWS;
        int DD    = D * D;

        for (int a_start = 0; a_start < N; a_start += brows) {
            int a_end   = a_start + brows;
            if (a_end > N) a_end = N;
            int n_rows  = a_end - a_start;
            int n_cols  = a_end * D;

            CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                n_rows * D, n_cols, M,
                &one,
                d_dXT + (long long)a_start * D * M, M,
                d_dXT, M,
                &zero,
                d_G_batch, n_rows * D));

            for (int a = a_start; a < a_end; a++) {
                assemble_row_kernel<<<a + 1, DD>>>(
                    d_K_full,
                    d_G_batch + (long long)(a - a_start) * D,
                    n_rows * D,
                    d_V1X2, d_U1, d_C, d_C4,
                    sigma2, a, N, D, full_rows);
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_norms);
    cudaFree(d_C);     cudaFree(d_C4);
    cudaFree(d_V1X2);  cudaFree(d_U1);
    cudaFree(d_G_batch);
}


/* ============================================================
 * krr_train_ef_internal
 *
 * Internal: builds K_full, regularises, Cholesky-solves for alpha.
 * Returns alpha on host via h_alpha.
 * ============================================================ */

static void krr_train_ef_internal(
    const float *h_X,
    const float *h_dXT,
    const float *h_E,
    const float *h_F,
    float       *h_alpha,
    float sigma, float lambda,
    int N, int M, int D)
{
    long long ND        = (long long)N * D;
    long long full_size = (long long)N * (1 + D);

    float *d_X, *d_dXT, *d_K_full, *d_rhs, *d_work;
    int   *d_info;
    int    h_info;

    CUDA_CHECK(cudaMalloc(&d_X,      (size_t)M * N  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dXT,    M * ND         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_full, full_size * full_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rhs,    full_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_info,   sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_X,   h_X,   (size_t)M * N  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dXT, h_dXT, M * ND         * sizeof(float), cudaMemcpyHostToDevice));

    build_kernel_gpu_symmetric(d_X, d_dXT, d_K_full, sigma, N, M, D);

    /* Regularise */
    add_diagonal_kernel<<<(full_size + 255) / 256, 256>>>(d_K_full, lambda, full_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Assemble RHS: [E, -F] */
    CUDA_CHECK(cudaMemcpy(d_rhs,     h_E, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs + N, h_F, ND        * sizeof(float), cudaMemcpyHostToDevice));
    {
        const float neg_one = -1.0f;
        CUBLAS_CHECK(cublasSscal(s_cublas, (int)ND, &neg_one, d_rhs + N, 1));
    }

    /* Cholesky factorisation + solve */
    {
        int lwork;
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(s_cusolver,
            CUBLAS_FILL_MODE_LOWER, (int)full_size, d_K_full, (int)full_size, &lwork));
        CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(float)));

        CUSOLVER_CHECK(cusolverDnSpotrf(s_cusolver,
            CUBLAS_FILL_MODE_LOWER, (int)full_size, d_K_full, (int)full_size,
            d_work, lwork, d_info));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0)
            fprintf(stderr, "  [cuda_krr] potrf failed: info=%d\n", h_info);

        CUSOLVER_CHECK(cusolverDnSpotrs(s_cusolver,
            CUBLAS_FILL_MODE_LOWER, (int)full_size, 1,
            d_K_full, (int)full_size, d_rhs, (int)full_size, d_info));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0)
            fprintf(stderr, "  [cuda_krr] potrs failed: info=%d\n", h_info);

        cudaFree(d_work);
    }

    CUDA_CHECK(cudaMemcpy(h_alpha, d_rhs, full_size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_X); cudaFree(d_dXT); cudaFree(d_K_full);
    cudaFree(d_rhs); cudaFree(d_info);
}


/* ============================================================
 * krr_inference_init_gpu
 *
 * Uploads training data to GPU and contracts the Jacobians into W_F_bar.
 * After this call, dXT_train is no longer needed.
 * ============================================================ */

static void krr_inference_init_gpu(KrrInferenceState *st,
                                    const float *h_alpha_full,
                                    const float *h_X_train,
                                    const float *h_dXT_train,
                                    float sigma, int N_train, int M, int D)
{
    st->sigma   = sigma;
    st->inv_s2  = 1.0f / (sigma * sigma);
    st->sigma2  = sigma * sigma;
    st->N_train = N_train;
    st->M       = M;
    st->D       = D;
    st->ws_N_test = 0;
    /* clear workspace pointers */
    memset(&st->ws_N_test, 0,
           sizeof(KrrInferenceState) - offsetof(KrrInferenceState, ws_N_test));

    long long Nt  = N_train;
    long long NtD = (long long)N_train * D;
    const float one = 1.0f, zero = 0.0f;

    CUDA_CHECK(cudaMalloc(&st->d_X_train,    (long long)M * Nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_W_F_bar,    (long long)M * Nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_W_combined, (long long)M * Nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_W_F_self,   Nt               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_alpha_E,    Nt               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_norms_tr,   Nt               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_ones,       Nt               * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(st->d_X_train, h_X_train,    (long long)M * Nt * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->d_alpha_E, h_alpha_full, Nt             * sizeof(float), cudaMemcpyHostToDevice));

    /* ones vector */
    {
        float *h_ones = (float*)malloc(Nt * sizeof(float));
        for (int i = 0; i < N_train; i++) h_ones[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(st->d_ones, h_ones, Nt * sizeof(float), cudaMemcpyHostToDevice));
        free(h_ones);
    }

    compute_sqnorms_kernel<<<(N_train + 255) / 256, 256>>>(st->d_X_train, st->d_norms_tr, M, N_train);

    /* Contract Jacobians: W_F_bar[:,b] = dXT[:,b*D:(b+1)*D] @ alpha_F[b] */
    {
        float *d_alpha_F, *d_dXT_train;
        CUDA_CHECK(cudaMalloc(&d_alpha_F,   NtD * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dXT_train, (long long)M * NtD * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_alpha_F,   h_alpha_full + N_train, NtD * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dXT_train, h_dXT_train,  (long long)M * NtD * sizeof(float), cudaMemcpyHostToDevice));

        CUBLAS_CHECK(cublasSgemvStridedBatched(s_cublas,
            CUBLAS_OP_N, M, D,
            &one,
            d_dXT_train, M, (long long)M * D,
            d_alpha_F,   1, (long long)D,
            &zero,
            st->d_W_F_bar, 1, (long long)M,
            N_train));

        cudaFree(d_alpha_F);
        cudaFree(d_dXT_train);
    }

    /* W_F_self[b] = W_F_bar[:,b] . X_train[:,b] */
    CUDA_CHECK(cudaMemset(st->d_W_F_self, 0, Nt * sizeof(float)));
    dot_diagonal_kernel<<<(N_train + 255) / 256, 256>>>(
        st->d_W_F_self, st->d_W_F_bar, st->d_X_train, M, N_train);

    /* W_combined = alpha_E * X_train (column scaling) + W_F_bar */
    CUBLAS_CHECK(cublasSdgmm(s_cublas, CUBLAS_SIDE_RIGHT,
        M, N_train, st->d_X_train, M, st->d_alpha_E, 1, st->d_W_combined, M));
    CUBLAS_CHECK(cublasSaxpy(s_cublas, (int)((long long)M * Nt),
        &one, st->d_W_F_bar, 1, st->d_W_combined, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
}


/* ============================================================
 * krr_inference_alloc_workspace  (internal)
 * ============================================================ */

static void krr_inference_alloc_workspace(KrrInferenceState *st, int N_test)
{
    if (st->ws_N_test >= N_test) return;

    if (st->ws_N_test > 0) {
        cudaFree(st->ws_X_test);    cudaFree(st->ws_dXT_test);
        cudaFree(st->ws_norms_te);  cudaFree(st->ws_C);
        cudaFree(st->ws_cross_F);   cudaFree(st->ws_weight_F);
        cudaFree(st->ws_sum_F);     cudaFree(st->ws_w_E);
        cudaFree(st->ws_E_pred);    cudaFree(st->ws_G_full);
        cudaFree(st->ws_combined);  cudaFree(st->ws_F_pred);
    }

    int Nt = st->N_train, M = st->M, D = st->D;
    long long NDte = (long long)N_test * D;

    CUDA_CHECK(cudaMalloc(&st->ws_X_test,   (long long)M * N_test  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_dXT_test, (long long)M * NDte    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_norms_te, N_test                  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_C,        (long long)N_test * Nt  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_cross_F,  (long long)N_test * Nt  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_weight_F, (long long)N_test * Nt  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_sum_F,    N_test                   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_w_E,      N_test                   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_E_pred,   N_test                   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_G_full,   (long long)M * N_test   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_combined, N_test                   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_F_pred,   NDte                     * sizeof(float)));

    st->ws_N_test = N_test;
}


/* ============================================================
 * krr_predict_internal
 * ============================================================ */

static void krr_predict_internal(KrrInferenceState *st,
                                  const float *h_X_test,
                                  const float *h_dXT_test,
                                  float       *h_E_pred,
                                  float       *h_F_pred,
                                  int N_test)
{
    int Nt = st->N_train, M = st->M, D = st->D;
    long long NDte = (long long)N_test * D;
    float inv_s2 = st->inv_s2;
    float sigma2 = st->sigma2;
    const float one = 1.0f, zero = 0.0f, neg_one = -1.0f;

    krr_inference_alloc_workspace(st, N_test);

    float *d_X_test   = st->ws_X_test;
    float *d_dXT_test = st->ws_dXT_test;
    float *d_norms_te = st->ws_norms_te;
    float *d_C        = st->ws_C;
    float *d_cross_F  = st->ws_cross_F;
    float *d_weight_F = st->ws_weight_F;
    float *d_sum_F    = st->ws_sum_F;
    float *d_w_E      = st->ws_w_E;
    float *d_E_pred   = st->ws_E_pred;
    float *d_G_full   = st->ws_G_full;
    float *d_combined = st->ws_combined;
    float *d_F_pred   = st->ws_F_pred;

    /* H2D */
    CUDA_CHECK(cudaMemcpy(d_X_test,   h_X_test,  (long long)M * N_test * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dXT_test, h_dXT_test, (long long)M * NDte  * sizeof(float), cudaMemcpyHostToDevice));

    /* Phase 1: C = K(X_te, X_tr) / sigma^2 */
    compute_sqnorms_kernel<<<(N_test + 255) / 256, 256>>>(d_X_test, d_norms_te, M, N_test);
    float neg2 = -2.0f;
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N_test, Nt, M,
        &neg2, d_X_test, M, st->d_X_train, M,
        &zero, d_C, N_test));
    {
        dim3 blk(16, 16);
        dim3 grd((N_test + 15) / 16, (Nt + 15) / 16);
        add_asym_norms_kernel<<<grd, blk>>>(d_C, d_norms_te, st->d_norms_tr, N_test, Nt);
        build_C_kernel<<<grd, blk>>>(d_C, inv_s2, N_test, Nt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 2: cross_F, weight_F, sum_F, w_E */
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N_test, Nt, M,
        &one, d_X_test, M, st->d_W_F_bar, M,
        &zero, d_cross_F, N_test));
    {
        dim3 blk(16, 16);
        dim3 grd((N_test + 15) / 16, (Nt + 15) / 16);
        build_weight_F_kernel<<<grd, blk>>>(d_weight_F, d_C, d_cross_F,
                                             st->d_W_F_self, inv_s2, N_test, Nt);
    }
    CUBLAS_CHECK(cublasSgemv(s_cublas, CUBLAS_OP_N, N_test, Nt,
        &one, d_weight_F, N_test, st->d_ones, 1, &zero, d_sum_F, 1));
    CUBLAS_CHECK(cublasSgemv(s_cublas, CUBLAS_OP_N, N_test, Nt,
        &one, d_C, N_test, st->d_alpha_E, 1, &zero, d_w_E, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 3: E + G_full */
    fused_energy_combined_kernel<<<(N_test + 255) / 256, 256>>>(
        d_E_pred, d_combined, d_w_E, d_sum_F, sigma2, N_test);
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        M, N_test, Nt,
        &one, st->d_W_combined, M, d_C, N_test,
        &zero, d_G_full, M));
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        M, N_test, Nt,
        &one, st->d_X_train, M, d_weight_F, N_test,
        &one, d_G_full, M));
    {
        dim3 blk(16, 16);
        dim3 grd((M + 15) / 16, (N_test + 15) / 16);
        subtract_scaled_cols_kernel<<<grd, blk>>>(d_G_full, d_X_test, d_combined, M, N_test);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 4: F_pred = -J_test @ G_full  (batched SGEMV) */
    CUBLAS_CHECK(cublasSgemvStridedBatched(s_cublas,
        CUBLAS_OP_T, M, D,
        &neg_one,
        d_dXT_test, M, (long long)M * D,
        d_G_full,   1, (long long)M,
        &zero,
        d_F_pred,   1, (long long)D,
        N_test));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* D2H */
    CUDA_CHECK(cudaMemcpy(h_E_pred, d_E_pred, N_test * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_F_pred, d_F_pred, NDte   * sizeof(float), cudaMemcpyDeviceToHost));
}


/* ============================================================
 * Public API
 * ============================================================ */

void cuda_krr_train_ef(KrrInferenceState *st,
                       const float *h_X,
                       const float *h_dXT,
                       const float *h_E,
                       const float *h_F,
                       float       *h_alpha,
                       float sigma, float lambda,
                       int N, int M, int D)
{
    ensure_handles();

    /* 1. Train: solve (K_full + lam*I) alpha = [E, -F] */
    krr_train_ef_internal(h_X, h_dXT, h_E, h_F, h_alpha, sigma, lambda, N, M, D);

    /* 2. Init inference state from alpha + training data */
    krr_inference_init_gpu(st, h_alpha, h_X, h_dXT, sigma, N, M, D);
}


void cuda_krr_predict(KrrInferenceState *st,
                      const float *h_X_test,
                      const float *h_dXT_test,
                      float       *h_E_pred,
                      float       *h_F_pred,
                      int          N_test)
{
    ensure_handles();
    krr_predict_internal(st, h_X_test, h_dXT_test, h_E_pred, h_F_pred, N_test);
}


void cuda_krr_get_state(const KrrInferenceState *st,
                        float *h_X_train,
                        float *h_W_F_bar,
                        float *h_W_combined,
                        float *h_W_F_self,
                        float *h_alpha_E,
                        float *h_norms_tr)
{
    long long Nt = st->N_train;
    long long M  = st->M;
    CUDA_CHECK(cudaMemcpy(h_X_train,    st->d_X_train,    M * Nt * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_W_F_bar,    st->d_W_F_bar,    M * Nt * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_W_combined, st->d_W_combined, M * Nt * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_W_F_self,   st->d_W_F_self,   Nt     * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_alpha_E,    st->d_alpha_E,    Nt     * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_norms_tr,   st->d_norms_tr,   Nt     * sizeof(float), cudaMemcpyDeviceToHost));
}


void cuda_krr_load_state(KrrInferenceState *st,
                         const float *h_X_train,
                         const float *h_W_F_bar,
                         const float *h_W_combined,
                         const float *h_W_F_self,
                         const float *h_alpha_E,
                         const float *h_norms_tr,
                         float sigma, int N_train, int M, int D)
{
    ensure_handles();

    st->sigma   = sigma;
    st->inv_s2  = 1.0f / (sigma * sigma);
    st->sigma2  = sigma * sigma;
    st->N_train = N_train;
    st->M       = M;
    st->D       = D;
    st->ws_N_test = 0;
    /* clear workspace pointers */
    st->ws_X_test = st->ws_dXT_test = st->ws_norms_te = NULL;
    st->ws_C = st->ws_cross_F = st->ws_weight_F = NULL;
    st->ws_sum_F = st->ws_w_E = st->ws_E_pred = NULL;
    st->ws_G_full = st->ws_combined = st->ws_F_pred = NULL;

    long long Nt = N_train;

    CUDA_CHECK(cudaMalloc(&st->d_X_train,    (long long)M * Nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_W_F_bar,    (long long)M * Nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_W_combined, (long long)M * Nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_W_F_self,   Nt               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_alpha_E,    Nt               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_norms_tr,   Nt               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_ones,       Nt               * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(st->d_X_train,    h_X_train,    (long long)M * Nt * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->d_W_F_bar,    h_W_F_bar,    (long long)M * Nt * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->d_W_combined, h_W_combined, (long long)M * Nt * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->d_W_F_self,   h_W_F_self,   Nt               * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->d_alpha_E,    h_alpha_E,    Nt               * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->d_norms_tr,   h_norms_tr,   Nt               * sizeof(float), cudaMemcpyHostToDevice));

    /* Fill ones vector */
    float *h_ones = (float*)malloc(Nt * sizeof(float));
    for (int i = 0; i < N_train; i++) h_ones[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(st->d_ones, h_ones, Nt * sizeof(float), cudaMemcpyHostToDevice));
    free(h_ones);

    CUDA_CHECK(cudaDeviceSynchronize());
}


/* ============================================================
 * mirror_lower_to_upper_kernel  [diagnostic helper]
 *
 * For a col-major (N_full × N_full) matrix A, copies every
 * strictly-lower-triangle entry A[row, col] (row > col) to its
 * symmetric counterpart A[col, row], making A fully symmetric.
 * ============================================================ */
__global__ void mirror_lower_to_upper_kernel(float *A, long long N_full)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long row = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= N_full || row >= N_full || row <= col) return;
    /* row > col: lower triangle → copy to upper */
    A[col + row * N_full] = A[row + col * N_full];
}


/* ============================================================
 * cuda_krr_build_kernel_matrix  [diagnostic]
 *
 * Builds K_full on GPU, mirrors lower→upper, downloads to host.
 * ============================================================ */
void cuda_krr_build_kernel_matrix(
    const float *h_X,
    const float *h_dXT,
    float       *h_K_full,
    float        sigma,
    int N, int M, int D)
{
    ensure_handles();

    long long ND        = (long long)N * D;
    long long full_rows = (long long)N * (1 + D);

    float *d_X, *d_dXT, *d_K_full;
    CUDA_CHECK(cudaMalloc(&d_X,      (long long)M * N  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dXT,    (long long)M * ND * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_full, full_rows * full_rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X,   h_X,   (long long)M * N  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dXT, h_dXT, (long long)M * ND * sizeof(float), cudaMemcpyHostToDevice));

    build_kernel_gpu_symmetric(d_X, d_dXT, d_K_full, sigma, N, M, D);

    /* Mirror lower triangle to upper so the result is fully symmetric */
    {
        dim3 blk(16, 16);
        dim3 grd(((int)full_rows + 15) / 16, ((int)full_rows + 15) / 16);
        mirror_lower_to_upper_kernel<<<grd, blk>>>(d_K_full, full_rows);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_K_full, d_K_full,
                          full_rows * full_rows * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_X);
    cudaFree(d_dXT);
    cudaFree(d_K_full);
}


void cuda_krr_free(KrrInferenceState *st)
{
    if (!st) return;

    /* Persistent buffers */
    if (st->d_X_train)    { cudaFree(st->d_X_train);    st->d_X_train    = NULL; }
    if (st->d_W_F_bar)    { cudaFree(st->d_W_F_bar);    st->d_W_F_bar    = NULL; }
    if (st->d_W_combined) { cudaFree(st->d_W_combined); st->d_W_combined = NULL; }
    if (st->d_W_F_self)   { cudaFree(st->d_W_F_self);   st->d_W_F_self   = NULL; }
    if (st->d_alpha_E)    { cudaFree(st->d_alpha_E);    st->d_alpha_E    = NULL; }
    if (st->d_norms_tr)   { cudaFree(st->d_norms_tr);   st->d_norms_tr   = NULL; }
    if (st->d_ones)       { cudaFree(st->d_ones);       st->d_ones       = NULL; }

    /* Workspace */
    if (st->ws_N_test > 0) {
        cudaFree(st->ws_X_test);    cudaFree(st->ws_dXT_test);
        cudaFree(st->ws_norms_te);  cudaFree(st->ws_C);
        cudaFree(st->ws_cross_F);   cudaFree(st->ws_weight_F);
        cudaFree(st->ws_sum_F);     cudaFree(st->ws_w_E);
        cudaFree(st->ws_E_pred);    cudaFree(st->ws_G_full);
        cudaFree(st->ws_combined);  cudaFree(st->ws_F_pred);
        st->ws_N_test = 0;
    }
}
