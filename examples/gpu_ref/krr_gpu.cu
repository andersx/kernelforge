/*
 * krr_gpu.cu — GPU-accelerated Gaussian KRR with energy + force training
 *
 * Kernel:  K(x_a, x_b) = exp(-||x_a - x_b||² / (2σ²))
 *
 * Data layout (column-major throughout, matching cuBLAS):
 *   X    : (M, N)   — each column is a molecular descriptor
 *   dX_T : (M, N*D) — column a*D+d is Jacobian ∂X_a/∂coord_d
 *
 * Training (symmetric, lower-triangle only):
 *   K_full (N*(1+D))² built via row-batched SGEMM (no G_all intermediate).
 *   Solved with Cholesky: (K_full + λI) α = [E, −F].
 *
 * Inference (contracted, no K_full_test):
 *   Persistent state: X_train, W_F_bar = J^T α_F, W_combined, norms, α_E.
 *   Per query: 4 SGEMMs + 1 batched SGEMV.  E = σ²(w_E + sum_F).
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 krr_gpu.cu -lcublas -lcusolver -o krr_gpu
 *
 * Usage:
 *   ./krr_gpu N_TRAIN N_TEST M D [SIGMA]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


/* ============================================================
 * Error-checking macros
 * ============================================================ */

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t _s = (call);                                           \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d — status %d\n",           \
                    __FILE__, __LINE__, (int)_s);                             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUSOLVER_CHECK(call)                                                  \
    do {                                                                      \
        cusolverStatus_t _s = (call);                                         \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuSOLVER error at %s:%d — status %d\n",         \
                    __FILE__, __LINE__, (int)_s);                             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


/* ============================================================
 * CUDA kernels
 * ============================================================ */

/*
 * compute_sqnorms_kernel
 * norms[i] = ||X[:,i]||²   X is (M, N) column-major.
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
 * G[a,b] += (norms[a] + norms[b])   for symmetric case.
 * G is (N, N) column-major: G[a,b] = G[a + b*N].
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
 * G[a,b] += (norms1[a] + norms2[b])  for asymmetric case.
 * G is (N1, N2) column-major: G[a,b] = G[a + b*N1].
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
 * Converts the squared-distance matrix into the scaled kernel coefficients.
 * On entry:  G[a,b] = ||x_a − x_b||²
 * On exit:   G[a,b] = K[a,b] * inv_s2   (= C)
 *            C4[a,b]= K[a,b] * inv_s2²  (= C4)
 * where K[a,b] = exp(-0.5 * inv_s2 * dist²).
 */
__global__ void build_C_C4_kernel(float *G, float *C4,
                                   float inv_s2, int N1, int N2)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N1 || b >= N2) return;
    long long idx = a + (long long)b * N1;
    float k = expf(-0.5f * inv_s2 * G[idx]);
    G[idx]   = k * inv_s2;
    C4[idx]  = k * inv_s2 * inv_s2;
}


/*
 * extract_U_kernel
 * U[g] = V1X2[g, g/D]   — self-projections (molecule a = g/D).
 * V1X2 is (N*D, N) column-major: V1X2[g, a] = V1X2[g + a*(N*D)].
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
 * Assembles one block-row of K_full from precomputed G_row, V1X2, U1, C, C4.
 * Called once per row a, with grid = (a+1) blocks (one per column b = 0..a).
 * Each block has D*D threads writing the (1+D)×(1+D) sub-block at (a,b).
 *
 *   K_EE[a, b] = σ² · C[a,b]
 *   K_FE[a*D+d1, b] = C[a,b] · (V1X2[a*D+d1, b] − U1[a*D+d1])
 *   K_FE[b*D+d2, a] = C[a,b] · (V1X2[b*D+d2, a] − U1[b*D+d2])  (mirror, a>b)
 *   K_FF[a*D+d1, b*D+d2] = C·G − C4·p·q   (lower triangle)
 *
 * G_row is (D, (a+1)*D) col-major: G_row[d1 + (b*D+d2)*D] = J_a[d1,:] · J_b[d2,:]
 * V1X2 is (ND, N) col-major.
 */
__global__ void assemble_row_kernel(
    float *K_full,
    const float *G_row,    /* points to row a's D rows within G_batch          */
    int G_lda,             /* leading dimension of G matrix                    */
    const float *V1X2,     /* (ND, N) col-major                                */
    const float *U1,       /* (ND,)                                            */
    const float *C,        /* (N, N) col-major                                 */
    const float *C4,       /* (N, N) col-major                                 */
    float sigma2,
    int a, int N, int D,
    long long full_rows)
{
    int b   = blockIdx.x;          /* column block 0..a */
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

    /* p = V1X2[g1, b] − U1[g1]  and  q = V1X2[g2, a] − U1[g2] */
    float p = V1X2[g1 + (long long)b * ND] - U1[g1];
    float q = V1X2[g2 + (long long)a * ND] - U1[g2];

    /* G_ab[d1,d2] from SGEMM output: G_row has leading dimension G_lda */
    float g = G_row[d1 + (long long)(b * D + d2) * G_lda];

    /* K_EE */
    if (tid == 0)
        K_full[a + (long long)b * full_rows] = c * sigma2;

    /* K_FE for pair (a, b): K_FE[g1, b] = C[a,b] · (V1X2[g1,b] - U1[g1]) = c · p */
    if (d2 == 0)
        K_full[(N + g1) + (long long)b * full_rows] = c * p;
    /* K_FE mirror for pair (b, a): K_FE[g2, a] = C[b,a] · (V1X2[g2,a] - U1[g2]) = c · q */
    if (a > b && d1 == 0)
        K_full[(N + g2) + (long long)a * full_rows] = c * q;

    /* K_FF — lower triangle check for diagonal blocks
     * Formula: K_FF = C·G − C4·v1·v2  where v1 = U1−V1X2 = −p, v2 = V1X2−U1 = q
     * So C4·v1·v2 = C4·(−p)·q = −C4·p·q, giving K_FF = C·G + C4·p·q */
    if (a > b || g1 >= g2)
        K_full[(N + g1) + (long long)(N + g2) * full_rows] =
            c * g + c4 * p * q;
}


/*
 * add_diagonal_kernel
 * A[i,i] += val   for a (N, N) column-major matrix.
 */
__global__ void add_diagonal_kernel(float *A, float val, long long N)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) A[i + i * N] += val;
}


/*
 * dot_diagonal_kernel
 * result[a] += A[:,a] · B[:,a]   for each column a.
 * A, B are (M, N) column-major.  Accumulates into result (initialise to 0 first).
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
 * E_pred[a] = sigma2 * (w_E[a] + sum_F[a])
 * combined[a] = w_E[a] + sum_F[a]
 * Replaces 5 cuBLAS calls (2 D2D copies + 2 saxpy + 1 sscal) with 1 kernel.
 */
__global__ void fused_energy_combined_kernel(float *E_pred, float *combined,
                                              const float *w_E, const float *sum_F,
                                              float sigma2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float c = w_E[i] + sum_F[i];
    combined[i] = c;
    E_pred[i] = sigma2 * c;
}


/*
 * subtract_scaled_cols_kernel
 * R[m, a] -= scalar[a] * X[m, a]
 * R and X are (M, N) column-major.
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
 * In-place: G[a,b] = exp(-0.5 · inv_s2 · G[a,b]) · inv_s2
 * Same as build_C_C4_kernel but without writing C4.
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
 * weight_F[a, b] = C[a,b] · inv_s2 · (cross_F[a,b] − self_cross_F[b])
 *
 * Matches reference Phase 5:  weight_F = C·(1/σ²)·(cross_F − self_cross_F)
 * Reads C directly (no separate C4 buffer needed).
 *
 * All matrices are (N_test, N_train) column-major.
 * self_cross_F is (N_train,).
 */
__global__ void build_weight_F_kernel(float *weight_F, const float *C,
                                       const float *cross_F,
                                       const float *self_cross_F,
                                       float inv_s2,
                                       int N_test, int N_train)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;   /* test mol  */
    int b = blockIdx.y * blockDim.y + threadIdx.y;   /* train mol */
    if (a >= N_test || b >= N_train) return;
    long long idx = a + (long long)b * N_test;
    weight_F[idx] = C[idx] * inv_s2 * (cross_F[idx] - self_cross_F[b]);
}


/* ============================================================
 * Timing helpers
 * ============================================================ */

static double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct { cudaEvent_t start, stop; } GpuTimer;

static void timer_start(GpuTimer *t)
{
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
    CUDA_CHECK(cudaEventRecord(t->start, 0));
}

static float timer_stop(GpuTimer *t)
{
    float ms;
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
    return ms * 1e-3f;
}


/* ============================================================
 * build_kernel_gpu_symmetric
 *
 * Builds the symmetric K_full training matrix (lower triangle only).
 *
 *   Phase 1: C, C4 from squared distances (full N×N, needed by K_FE).
 *   Phase 2: V1X2 = dXT^T @ X, U1 = self-projections.
 *   Phase 3: Row-batched SGEMM + assemble_row_kernel.
 *            For each batch of B_ROWS molecules, one SGEMM computes
 *            G_row = J_batch^T @ dXT[:, 0:n_cols], then the assembly
 *            kernel writes K_EE + K_FE + K_FF to K_full.
 *            No G_all (ND×ND) intermediate is ever materialised.
 * ============================================================ */

void build_kernel_gpu_symmetric(cublasHandle_t cublas,
                                const float *d_X,    const float *d_dXT,
                                float *d_K_full,
                                float sigma, int N, int M, int D,
                                float *t_dist, float *t_proj, float *t_fill)
{
    GpuTimer timer;
    long long ND        = (long long)N * D;
    long long full_rows = (long long)N * (1 + D);

    float inv_s2 = 1.0f / (sigma * sigma);
    float sigma2 = sigma * sigma;

    /* Allocate temporaries */
    float *d_norms, *d_C, *d_C4;
    float *d_V1X2, *d_U1;
    /* Batch size for row-batched SGEMM: process B_ROWS rows per SGEMM call.
     * G_batch is (B_ROWS*D, ND) — one large SGEMM replaces B_ROWS small ones. */
    #define BUILD_B_ROWS 64
    float *d_G_batch;

    CUDA_CHECK(cudaMalloc(&d_norms, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,     (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C4,    (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V1X2,  ND * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U1,    ND * sizeof(float)));
    {
        int brows = (N < BUILD_B_ROWS) ? N : BUILD_B_ROWS;
        CUDA_CHECK(cudaMalloc(&d_G_batch, (long long)brows * D * ND * sizeof(float)));
    }

    const float one = 1.0f, neg2 = -2.0f, zero = 0.0f;

    /* ------------------------------------------------------------------ */
    /* Phase 1: Squared distances → C and C4  (full matrix)                */
    /* ------------------------------------------------------------------ */
    timer_start(&timer);

    compute_sqnorms_kernel<<<(N + 255) / 256, 256>>>(d_X, d_norms, M, N);

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, M, &neg2, d_X, M, d_X, M, &zero, d_C, N));

    {
        dim3 blk(16, 16);
        dim3 grd((N + 15) / 16, (N + 15) / 16);
        add_sym_norms_kernel<<<grd, blk>>>(d_C, d_norms, N);
        build_C_C4_kernel<<<grd, blk>>>(d_C, d_C4, inv_s2, N, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    *t_dist = timer_stop(&timer);

    /* Zero K_full — assembly only writes lower triangle elements */
    CUDA_CHECK(cudaMemset(d_K_full, 0, full_rows * full_rows * sizeof(float)));

    /* ------------------------------------------------------------------ */
    /* Phase 2: V1X2 + U1  (needed for p/q lookup in assembly)             */
    /* ------------------------------------------------------------------ */
    timer_start(&timer);

    /* V1X2 = dXT^T @ X  →  (ND, N) */
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)ND, N, M,
        &one, d_dXT, M, d_X, M,
        &zero, d_V1X2, (int)ND));

    /* U1 from V1X2 (valid because X1=X2) */
    extract_U_kernel<<<((int)ND + 255) / 256, 256>>>(d_V1X2, d_U1, N, D);

    CUDA_CHECK(cudaDeviceSynchronize());
    *t_proj = timer_stop(&timer);

    /* ------------------------------------------------------------------ */
    /* Phase 3: Row-batched SGEMM + assembly                               */
    /*   For each row a: one SGEMM gives G_row = J_a^T @ dXT[:, 0:(a+1)*D]*/
    /*   then the assembly kernel writes K_EE + K_FE + K_FF for that row.  */
    /*   No G_all (ND×ND) ever materialised — only D×(a+1)D per row.      */
    /* ------------------------------------------------------------------ */
    timer_start(&timer);

    {
        int brows = (N < BUILD_B_ROWS) ? N : BUILD_B_ROWS;
        int DD = D * D;

        for (int a_start = 0; a_start < N; a_start += brows) {
            int a_end = a_start + brows;
            if (a_end > N) a_end = N;
            int n_rows = a_end - a_start;

            /* One large SGEMM per batch:
             * G_batch (n_rows*D, a_end*D) = dXT[:, a_start*D : a_end*D]^T @ dXT[:, 0 : a_end*D]
             * Only compute columns 0..a_end*D (lower triangle: b ≤ a < a_end). */
            int n_cols = a_end * D;
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                n_rows * D, n_cols, M,
                &one,
                d_dXT + (long long)a_start * D * M, M,
                d_dXT, M,
                &zero,
                d_G_batch, n_rows * D));

            /* Assembly: launch kernel per row within this batch */
            for (int a = a_start; a < a_end; a++) {
                assemble_row_kernel<<<a + 1, DD>>>(
                    d_K_full,
                    d_G_batch + (long long)(a - a_start) * D,  /* row a's slice */
                    n_rows * D,                                 /* G_lda */
                    d_V1X2, d_U1, d_C, d_C4,
                    sigma2, a, N, D, full_rows);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    *t_fill = timer_stop(&timer);

    /* Cleanup */
    cudaFree(d_norms);
    cudaFree(d_C);     cudaFree(d_C4);
    cudaFree(d_V1X2);  cudaFree(d_U1);
    cudaFree(d_G_batch);
}


/* ============================================================
 * krr_train_ef_gpu
 *
 * Full E+F KRR training on GPU.
 * Builds the (N*(1+D)) × (N*(1+D)) symmetric kernel matrix,
 * regularises with lambda*I, and solves via Cholesky for alpha.
 *
 * Training system:  K_full · alpha = [E_train, -F_train]
 *
 * Timing breakdown (seconds):
 *   t_h2d   — host-to-device transfers
 *   t_dist  — distance / C / C4 computation
 *   t_proj  — V1X2, V2X1_T, G_all, U
 *   t_fill  — block fill kernels
 *   t_potrf — Cholesky factorisation
 *   t_potrs — triangular solve
 *   t_d2h   — device-to-host (alpha)
 * ============================================================ */

void krr_train_ef_gpu(cublasHandle_t cublas, cusolverDnHandle_t cusolver,
                      const float *h_X,     /* (M, N)   */
                      const float *h_dXT,   /* (M, N*D) */
                      const float *h_E,     /* (N,)     */
                      const float *h_F,     /* (N*D,)   */
                      float       *h_alpha, /* (N*(1+D),) out */
                      float sigma, float lambda,
                      int N, int M, int D,
                      float *t_h2d,
                      float *t_dist, float *t_proj, float *t_fill,
                      float *t_potrf, float *t_potrs,
                      float *t_d2h)
{
    GpuTimer timer;
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

    /* --- H2D --- */
    timer_start(&timer);
    CUDA_CHECK(cudaMemcpy(d_X,   h_X,   (size_t)M * N  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dXT, h_dXT, M * ND         * sizeof(float), cudaMemcpyHostToDevice));
    *t_h2d = timer_stop(&timer);

    /* --- Build K_full (symmetric optimised: lower triangle only) --- */
    build_kernel_gpu_symmetric(cublas, d_X, d_dXT,
                               d_K_full, sigma, N, M, D,
                               t_dist, t_proj, t_fill);

    /* Add lambda*I to full diagonal */
    add_diagonal_kernel<<<(full_size + 255) / 256, 256>>>(d_K_full, lambda, full_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Assemble RHS: [E_train, -F_train] */
    CUDA_CHECK(cudaMemcpy(d_rhs,       h_E, (size_t)N  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs + N,   h_F, ND         * sizeof(float), cudaMemcpyHostToDevice));
    /* Negate the force block: rhs[N:] = -F */
    {
        const float neg_one = -1.0f;
        CUBLAS_CHECK(cublasSscal(cublas, (int)ND, &neg_one, d_rhs + N, 1));
    }

    /* --- Cholesky factorisation --- */
    {
        int lwork;
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolver,
            CUBLAS_FILL_MODE_LOWER, (int)full_size, d_K_full, (int)full_size, &lwork));
        CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(float)));

        timer_start(&timer);
        CUSOLVER_CHECK(cusolverDnSpotrf(cusolver,
            CUBLAS_FILL_MODE_LOWER, (int)full_size, d_K_full, (int)full_size,
            d_work, lwork, d_info));
        CUDA_CHECK(cudaDeviceSynchronize());
        *t_potrf = timer_stop(&timer);

        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0)
            fprintf(stderr, "  [KRR] potrf failed: info=%d\n", h_info);

        /* --- Cholesky triangular solve --- */
        timer_start(&timer);
        CUSOLVER_CHECK(cusolverDnSpotrs(cusolver,
            CUBLAS_FILL_MODE_LOWER, (int)full_size, 1,
            d_K_full, (int)full_size, d_rhs, (int)full_size, d_info));
        CUDA_CHECK(cudaDeviceSynchronize());
        *t_potrs = timer_stop(&timer);

        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0)
            fprintf(stderr, "  [KRR] potrs failed: info=%d\n", h_info);

        cudaFree(d_work);
    }

    /* --- D2H: alpha = rhs (solution overwrites rhs in potrs) --- */
    timer_start(&timer);
    CUDA_CHECK(cudaMemcpy(h_alpha, d_rhs, full_size * sizeof(float), cudaMemcpyDeviceToHost));
    *t_d2h = timer_stop(&timer);

    cudaFree(d_X); cudaFree(d_dXT); cudaFree(d_K_full);
    cudaFree(d_rhs); cudaFree(d_info);
}



/* ============================================================
 * Persistent inference state — allocated once, kept on GPU.
 * Training Jacobians (M × N_train × D) are fully contracted into
 * W_F_bar (M × N_train), so dXT_train is never needed again.
 * ============================================================ */

typedef struct {
    float *d_X_train;      /* (M, Nt)   training descriptors              */
    float *d_W_F_bar;      /* (M, Nt)   J_b^T @ alpha_F_b (contracted J) */
    float *d_W_combined;   /* (M, Nt)   W_F_bar + alpha_E · X_train      */
    float *d_W_F_self;     /* (Nt,)     W_F_bar[:,b] · X_train[:,b]       */
    float *d_alpha_E;      /* (Nt,)     energy coefficients               */
    float *d_norms_tr;     /* (Nt,)     ||X_train[:,b]||²                 */
    float *d_ones;         /* (Nt,)     all-ones for row-sum SGEMV        */
    float sigma;
    float inv_s2;
    float sigma2;
    int   N_train, M, D;

    /* Persistent workspace — allocated on first predict call, reused.    */
    int   ws_N_test;           /* N_test the workspace is sized for       */
    float *ws_X_test;          /* (M, N_test)                             */
    float *ws_dXT_test;        /* (M, N_test*D)                           */
    float *ws_norms_te;        /* (N_test,)                               */
    float *ws_C;               /* (N_test, N_train)                       */
    float *ws_cross_F;         /* (N_test, N_train)                       */
    float *ws_weight_F;        /* (N_test, N_train)                       */
    float *ws_sum_F;           /* (N_test,)                               */
    float *ws_w_E;             /* (N_test,)                               */
    float *ws_E_pred;          /* (N_test,)                               */
    float *ws_G_full;          /* (M, N_test)                             */
    float *ws_combined;        /* (N_test,)                               */
    float *ws_F_pred;          /* (N_test*D,)                             */
} KrrInferenceState;


/* ============================================================
 * krr_inference_init
 *
 * One-time setup after training.  Uploads training data to GPU,
 * contracts the Jacobian (W_F_bar = J^T @ alpha_F), precomputes
 * W_combined = W_F_bar + alpha_E · X_train, norms, etc.
 *
 * After this call, dXT_train is no longer needed.
 * ============================================================ */

void krr_inference_init(cublasHandle_t cublas, KrrInferenceState *st,
                        const float *h_alpha_full,  /* (Nt*(1+D),)  */
                        const float *h_X_train,     /* (M, Nt)      */
                        const float *h_dXT_train,   /* (M, Nt*D)    */
                        float sigma, int N_train, int M, int D)
{
    st->sigma   = sigma;
    st->inv_s2  = 1.0f / (sigma * sigma);
    st->sigma2  = sigma * sigma;
    st->N_train = N_train;
    st->M       = M;
    st->D       = D;
    st->ws_N_test = 0;     /* workspace allocated lazily on first predict */

    long long Nt  = N_train;
    long long NtD = (long long)N_train * D;
    const float one = 1.0f, zero = 0.0f;

    /* Allocate persistent GPU buffers */
    CUDA_CHECK(cudaMalloc(&st->d_X_train,    (long long)M * Nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_W_F_bar,    (long long)M * Nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_W_combined, (long long)M * Nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_W_F_self,   Nt               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_alpha_E,    Nt               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_norms_tr,   Nt               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_ones,       Nt               * sizeof(float)));

    /* Upload X_train, alpha_E */
    CUDA_CHECK(cudaMemcpy(st->d_X_train, h_X_train,  (long long)M * Nt * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->d_alpha_E, h_alpha_full, Nt * sizeof(float), cudaMemcpyHostToDevice));

    /* ones vector */
    {
        float *h_ones = (float*)malloc(Nt * sizeof(float));
        for (int i = 0; i < N_train; i++) h_ones[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(st->d_ones, h_ones, Nt * sizeof(float), cudaMemcpyHostToDevice));
        free(h_ones);
    }

    /* norms_tr[b] = ||X_train[:,b]||² */
    compute_sqnorms_kernel<<<(N_train + 255) / 256, 256>>>(st->d_X_train, st->d_norms_tr, M, N_train);

    /* --- Contract Jacobians: W_F_bar[:,b] = dXT_train[:,b*D:(b+1)*D] @ alpha_F_b --- */
    {
        float *d_alpha_F, *d_dXT_train;
        CUDA_CHECK(cudaMalloc(&d_alpha_F,   NtD * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dXT_train, (long long)M * NtD * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_alpha_F,   h_alpha_full + N_train, NtD * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dXT_train, h_dXT_train,  (long long)M * NtD * sizeof(float), cudaMemcpyHostToDevice));

        CUBLAS_CHECK(cublasSgemvStridedBatched(cublas,
            CUBLAS_OP_N, M, D,
            &one,
            d_dXT_train, M, (long long)M * D,
            d_alpha_F,   1, (long long)D,
            &zero,
            st->d_W_F_bar, 1, (long long)M,
            N_train));

        cudaFree(d_alpha_F);
        cudaFree(d_dXT_train);
        /* dXT_train is now fully contracted into W_F_bar — never needed again */
    }

    /* W_F_self[b] = W_F_bar[:,b] · X_train[:,b] */
    CUDA_CHECK(cudaMemset(st->d_W_F_self, 0, Nt * sizeof(float)));
    dot_diagonal_kernel<<<(N_train + 255) / 256, 256>>>(
        st->d_W_F_self, st->d_W_F_bar, st->d_X_train, M, N_train);

    /* W_combined = W_F_bar + alpha_E · X_train
     * First: W_combined = alpha_E_Xt via cublasSdgmm */
    CUBLAS_CHECK(cublasSdgmm(cublas, CUBLAS_SIDE_RIGHT,
        M, N_train, st->d_X_train, M, st->d_alpha_E, 1, st->d_W_combined, M));
    /* Then: W_combined += W_F_bar */
    CUBLAS_CHECK(cublasSaxpy(cublas, (int)((long long)M * Nt),
        &one, st->d_W_F_bar, 1, st->d_W_combined, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
}


static void krr_inference_alloc_workspace(KrrInferenceState *st, int N_test)
{
    if (st->ws_N_test >= N_test) return;  /* already big enough */

    /* Free old workspace if resizing */
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

    CUDA_CHECK(cudaMalloc(&st->ws_X_test,   (long long)M * N_test       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_dXT_test, (long long)M * NDte         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_norms_te, N_test                      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_C,        (long long)N_test * Nt      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_cross_F,  (long long)N_test * Nt      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_weight_F, (long long)N_test * Nt      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_sum_F,    N_test                      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_w_E,      N_test                      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_E_pred,   N_test                      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_G_full,   (long long)M * N_test       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_combined, N_test                      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->ws_F_pred,   NDte                        * sizeof(float)));

    st->ws_N_test = N_test;
}


void krr_inference_free(KrrInferenceState *st)
{
    cudaFree(st->d_X_train);    cudaFree(st->d_W_F_bar);
    cudaFree(st->d_W_combined); cudaFree(st->d_W_F_self);
    cudaFree(st->d_alpha_E);    cudaFree(st->d_norms_tr);
    cudaFree(st->d_ones);

    if (st->ws_N_test > 0) {
        cudaFree(st->ws_X_test);    cudaFree(st->ws_dXT_test);
        cudaFree(st->ws_norms_te);  cudaFree(st->ws_C);
        cudaFree(st->ws_cross_F);   cudaFree(st->ws_weight_F);
        cudaFree(st->ws_sum_F);     cudaFree(st->ws_w_E);
        cudaFree(st->ws_E_pred);    cudaFree(st->ws_G_full);
        cudaFree(st->ws_combined);  cudaFree(st->ws_F_pred);
    }
}


/* ============================================================
 * krr_inference_predict
 *
 * Predicts E + F for a batch of test molecules using the persistent
 * inference state.  Only test data (X_test, dXT_test) is uploaded.
 *
 * Key optimisations vs the previous contracted predictor:
 *   - No C4 buffer (weight_F uses C · inv_s2 directly)
 *   - Energy = σ² · (w_E + sum_F) — no SGEMM / dot / SGEMV needed
 *   - G_full built with 2 SGEMMs using precomputed W_combined
 *   - Training data resident on GPU (no per-call H2D)
 *   - No per-call cudaMalloc / cudaFree
 *
 * Timing outputs (seconds):
 *   t_h2d     — test data H2D only
 *   t_C       — squared distances + exp → C
 *   t_cross   — cross_F SGEMM + weight_F kernel + sum_F/w_E SGEMVs
 *   t_gfull   — energy (trivial) + G_full (2 SGEMMs + subtract)
 *   t_bgemv   — batched SGEMV for forces
 *   t_d2h     — results D2H
 * ============================================================ */

void krr_inference_predict(cublasHandle_t cublas,
    KrrInferenceState *st,
    const float *h_X_test,    /* (M, N_test)      */
    const float *h_dXT_test,  /* (M, N_test*D)    */
    float       *h_E_pred,    /* (N_test,)    out */
    float       *h_F_pred,    /* (N_test*D,)  out */
    int N_test,
    float *t_h2d, float *t_C, float *t_cross,
    float *t_gfull, float *t_bgemv, float *t_d2h)
{
    GpuTimer timer;
    int Nt = st->N_train, M = st->M, D = st->D;
    long long NDte = (long long)N_test * D;
    float inv_s2 = st->inv_s2;
    float sigma2 = st->sigma2;
    const float one = 1.0f, zero = 0.0f, neg_one = -1.0f;

    /* Persistent workspace — allocated on first call, reused thereafter */
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

    /* ---- H2D: test data only ---- */
    timer_start(&timer);
    CUDA_CHECK(cudaMemcpy(d_X_test,   h_X_test,  (long long)M * N_test * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dXT_test, h_dXT_test, (long long)M * NDte  * sizeof(float), cudaMemcpyHostToDevice));
    *t_h2d = timer_stop(&timer);

    /* ================================================================
     * Phase 1: C (N_test × N_train) = K/σ²
     *   Uses precomputed norms_tr.  No C4 buffer.
     * ================================================================ */
    timer_start(&timer);

    compute_sqnorms_kernel<<<(N_test + 255) / 256, 256>>>(d_X_test, d_norms_te, M, N_test);

    float neg2 = -2.0f;
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
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
    *t_C = timer_stop(&timer);

    /* ================================================================
     * Phase 2: cross_F + weight_F + sum_F + w_E
     * ================================================================ */
    timer_start(&timer);

    /* cross_F (Nq, Nt) = X_test^T @ W_F_bar */
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N_test, Nt, M,
        &one, d_X_test, M, st->d_W_F_bar, M,
        &zero, d_cross_F, N_test));

    /* weight_F[a,b] = C[a,b] · inv_s2 · (cross_F[a,b] − W_F_self[b]) */
    {
        dim3 blk(16, 16);
        dim3 grd((N_test + 15) / 16, (Nt + 15) / 16);
        build_weight_F_kernel<<<grd, blk>>>(d_weight_F, d_C, d_cross_F,
                                             st->d_W_F_self, inv_s2, N_test, Nt);
    }

    /* sum_F (Nq,) = weight_F @ ones  (row sums) */
    CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N, N_test, Nt,
        &one, d_weight_F, N_test, st->d_ones, 1, &zero, d_sum_F, 1));

    /* w_E (Nq,) = C @ alpha_E */
    CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N, N_test, Nt,
        &one, d_C, N_test, st->d_alpha_E, 1, &zero, d_w_E, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
    *t_cross = timer_stop(&timer);

    /* ================================================================
     * Phase 3: Energy + G_full
     *
     * Energy:  E[a] = σ² · (w_E[a] + sum_F[a])
     *   (Derived from: σ²·w_E + dot(X,G_alpha_F) - C@W_F_self
     *    = σ²·w_E + Σ C·cross_F - Σ C·W_F_self
     *    = σ²·w_E + Σ C·(cross_F - W_F_self)
     *    = σ²·w_E + σ²·sum_F)
     *
     * G_full:
     *   G_full  = W_combined @ C^T   (merges G_alpha_F + alpha_E_Xt @ C^T)
     *   G_full += X_train @ weight_F^T
     *   G_full[:,a] -= (sum_F[a] + w_E[a]) · X_test[:,a]
     * ================================================================ */
    timer_start(&timer);

    /* E_pred + combined in one kernel (replaces 2 D2D copies + 2 saxpy + 1 sscal) */
    fused_energy_combined_kernel<<<(N_test + 255) / 256, 256>>>(
        d_E_pred, d_combined, d_w_E, d_sum_F, sigma2, N_test);

    /* G_full = W_combined (M, Nt) @ C^T (Nt, Nq) */
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        M, N_test, Nt,
        &one, st->d_W_combined, M, d_C, N_test,
        &zero, d_G_full, M));

    /* G_full += X_train (M, Nt) @ weight_F^T (Nt, Nq) */
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        M, N_test, Nt,
        &one, st->d_X_train, M, d_weight_F, N_test,
        &one, d_G_full, M));

    /* G_full[:,a] -= combined[a] · X_test[:,a] */
    {
        dim3 blk(16, 16);
        dim3 grd((M + 15) / 16, (N_test + 15) / 16);
        subtract_scaled_cols_kernel<<<grd, blk>>>(d_G_full, d_X_test, d_combined, M, N_test);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    *t_gfull = timer_stop(&timer);

    /* ================================================================
     * Phase 4: F_pred = −J @ G_full  (batched SGEMV)
     * ================================================================ */
    timer_start(&timer);
    CUBLAS_CHECK(cublasSgemvStridedBatched(cublas,
        CUBLAS_OP_T, M, D,
        &neg_one,
        d_dXT_test, M, (long long)M * D,
        d_G_full,   1, (long long)M,
        &zero,
        d_F_pred,   1, (long long)D,
        N_test));
    CUDA_CHECK(cudaDeviceSynchronize());
    *t_bgemv = timer_stop(&timer);

    /* ---- D2H ---- */
    timer_start(&timer);
    CUDA_CHECK(cudaMemcpy(h_E_pred, d_E_pred, N_test * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_F_pred, d_F_pred, NDte   * sizeof(float), cudaMemcpyDeviceToHost));
    *t_d2h = timer_stop(&timer);

    /* Workspace is persistent — freed in krr_inference_free() */
}


/* ============================================================
 * Accuracy helpers
 * ============================================================ */

static float h_mae(const float *a, const float *b, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fabsf(a[i] - b[i]);
    return s / n;
}


/* ============================================================
 * main — benchmark driver
 * ============================================================ */

int main(int argc, char *argv[])
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s N_TRAIN N_TEST M D [SIGMA]\n", argv[0]);
        return 1;
    }

    int N_train = atoi(argv[1]);
    int N_test  = atoi(argv[2]);
    int M       = atoi(argv[3]);
    int D       = atoi(argv[4]);
    float sigma = (argc >= 6) ? atof(argv[5]) : sqrtf(0.5f * M);
    float lambda = 1.0e-4f;

    long long full_train = (long long)N_train * (1 + D);
    double K_train_GB = (double)full_train * full_train * sizeof(float) / (1 << 30);

    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("=== krr_gpu — Kernel Ridge Regression E+F on GPU (float32) ===\n");
    printf("GPU:        %s (compute %d.%d, %.0f MB)\n",
           prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1 << 20));
    printf("Parameters: N_train=%d  N_test=%d  M=%d  D=%d  sigma=%.4f  lambda=%.1e\n",
           N_train, N_test, M, D, sigma, lambda);
    printf("K_full_train: %lld × %lld = %.2f GB\n\n", full_train, full_train, K_train_GB);

    /* ------------------------------------------------------------------ */
    /* Allocate + fill random host data                                    */
    /* ------------------------------------------------------------------ */
    float *h_X_train   = (float*)malloc((size_t)M * N_train * sizeof(float));
    float *h_dXT_train = (float*)malloc((size_t)M * N_train * D * sizeof(float));
    float *h_X_test    = (float*)malloc((size_t)M * N_test  * sizeof(float));
    float *h_dXT_test  = (float*)malloc((size_t)M * N_test  * D * sizeof(float));
    float *h_alpha_true= (float*)calloc(full_train, sizeof(float));
    float *h_E_train   = (float*)malloc(N_train * sizeof(float));
    float *h_F_train   = (float*)malloc((size_t)N_train * D * sizeof(float));
    float *h_E_test_gt = (float*)malloc(N_test * sizeof(float));
    float *h_F_test_gt = (float*)malloc((size_t)N_test * D * sizeof(float));
    float *h_E_pred    = (float*)malloc(N_test * sizeof(float));
    float *h_F_pred    = (float*)malloc((size_t)N_test * D * sizeof(float));
    float *h_alpha_full= (float*)malloc(full_train * sizeof(float));

    srand(42);
    float inv = 1.0f / RAND_MAX;
    for (int i = 0; i < M * N_train;     i++) h_X_train[i]   = rand() * inv;
    for (int i = 0; i < M * N_train * D; i++) h_dXT_train[i]  = (rand() * inv - 0.5f) * 2.0f;
    for (int i = 0; i < M * N_test;      i++) h_X_test[i]     = rand() * inv;
    for (int i = 0; i < M * N_test * D;  i++) h_dXT_test[i]   = (rand() * inv - 0.5f) * 2.0f;

    srand(1234);
    for (int i = 0; i < N_train; i++)
        h_alpha_true[i] = (rand() * inv - 0.5f) * 2.0f;

    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));

    /* ================================================================== */
    /* Generate ground truth via the contracted predictor                  */
    /*   Uses alpha_true (energy-only) to generate consistent E/F labels   */
    /*   for both train and test sets, without building K_full_test.        */
    /* ================================================================== */
    printf("--- Generating ground truth ---\n");
    {
        KrrInferenceState gt;
        float gt_t[6];
        krr_inference_init(cublas, &gt, h_alpha_true, h_X_train, h_dXT_train,
                           sigma, N_train, M, D);
        krr_inference_predict(cublas, &gt, h_X_train, h_dXT_train,
                              h_E_train, h_F_train, N_train,
                              &gt_t[0], &gt_t[1], &gt_t[2], &gt_t[3], &gt_t[4], &gt_t[5]);
        krr_inference_predict(cublas, &gt, h_X_test, h_dXT_test,
                              h_E_test_gt, h_F_test_gt, N_test,
                              &gt_t[0], &gt_t[1], &gt_t[2], &gt_t[3], &gt_t[4], &gt_t[5]);
        krr_inference_free(&gt);
    }
    printf("  E_train: [%.4f .. %.4f]   F_train: [%.4f .. %.4f]\n",
           h_E_train[0], h_E_train[N_train - 1],
           h_F_train[0], h_F_train[N_train * D - 1]);
    printf("  E_test:  [%.4f .. %.4f]\n\n",
           h_E_test_gt[0], h_E_test_gt[N_test - 1]);

    /* ================================================================== */
    /* Training: build K_full (symmetric) + Cholesky solve                 */
    /* ================================================================== */
    printf("--- Training: K_full (%lld × %lld = %.2f GB) ---\n\n",
           full_train, full_train, K_train_GB);

    float tt_h2d, tt_dist, tt_proj, tt_fill, tt_potrf, tt_potrs, tt_d2h;

    krr_train_ef_gpu(cublas, cusolver,
                     h_X_train, h_dXT_train, h_E_train, h_F_train,
                     h_alpha_full, sigma, lambda, N_train, M, D,
                     &tt_h2d, &tt_dist, &tt_proj, &tt_fill,
                     &tt_potrf, &tt_potrs, &tt_d2h);
    double wall_start = wall_time();
    krr_train_ef_gpu(cublas, cusolver,
                     h_X_train, h_dXT_train, h_E_train, h_F_train,
                     h_alpha_full, sigma, lambda, N_train, M, D,
                     &tt_h2d, &tt_dist, &tt_proj, &tt_fill,
                     &tt_potrf, &tt_potrs, &tt_d2h);
    double wall_train = wall_time() - wall_start;

    float tt_build = tt_dist + tt_proj + tt_fill;
    float tt_solve = tt_potrf + tt_potrs;
    float gpu_train = tt_build + tt_solve;
    printf("  Kernel assembly:\n");
    printf("    Distances + C/C4:          %10.6f s\n", tt_dist);
    printf("    V1X2 + U1:                 %10.6f s\n", tt_proj);
    printf("    Row-batched fill:          %10.6f s\n", tt_fill);
    printf("    Subtotal:                  %10.6f s\n", tt_build);
    printf("  Solver:\n");
    printf("    potrf (Cholesky):          %10.6f s\n", tt_potrf);
    printf("    potrs (solve):             %10.6f s\n", tt_potrs);
    printf("    Subtotal:                  %10.6f s\n", tt_solve);
    printf("  H2D / D2H:                   %10.6f s / %.6f s\n", tt_h2d, tt_d2h);
    printf("  GPU compute total:           %10.6f s\n", gpu_train);
    printf("  Wall time:                   %10.6f s\n\n", (float)wall_train);

    /* ================================================================== */
    /* Initialise inference + predict                                      */
    /* ================================================================== */
    KrrInferenceState inf_state;

    printf("--- Inference init (contract J, precompute W_combined) ---\n");
    wall_start = wall_time();
    krr_inference_init(cublas, &inf_state,
        h_alpha_full, h_X_train, h_dXT_train,
        sigma, N_train, M, D);
    printf("  Wall time: %.5f s\n",  (float)(wall_time() - wall_start));
    printf("  Persistent GPU: %.2f MB\n\n",
           (double)(3LL * M * N_train + 3LL * N_train) * sizeof(float) / (1 << 20));

    printf("--- Prediction (contracted, C only = %.2f MB per query) ---\n\n",
           (double)N_test * N_train * sizeof(float) / (1 << 20));

    float tc_h2d, tc_C, tc_cross, tc_gfull, tc_bgemv, tc_d2h;

    krr_inference_predict(cublas, &inf_state,
        h_X_test, h_dXT_test,
        h_E_pred, h_F_pred, N_test,
        &tc_h2d, &tc_C, &tc_cross, &tc_gfull, &tc_bgemv, &tc_d2h);
    wall_start = wall_time();
    krr_inference_predict(cublas, &inf_state,
        h_X_test, h_dXT_test,
        h_E_pred, h_F_pred, N_test,
        &tc_h2d, &tc_C, &tc_cross, &tc_gfull, &tc_bgemv, &tc_d2h);
    double wall_pred = wall_time() - wall_start;

    float gpu_pred = tc_C + tc_cross + tc_gfull + tc_bgemv;
    printf("  H2D (test only):             %10.6f s\n", tc_h2d);
    printf("  C (dist + exp):              %10.6f s\n", tc_C);
    printf("  cross_F + weight_F + sums:   %10.6f s\n", tc_cross);
    printf("  E + G_full (2 SGEMMs):       %10.6f s\n", tc_gfull);
    printf("  Batched SGEMV (forces):      %10.6f s\n", tc_bgemv);
    printf("  D2H:                         %10.6f s\n", tc_d2h);
    printf("  GPU compute total:           %10.6f s\n", gpu_pred);
    printf("  Wall time:                   %10.6f s\n\n", (float)wall_pred);

    /* ================================================================== */
    /* Results                                                             */
    /* ================================================================== */
    printf("--- Accuracy ---\n");
    printf("  Energy MAE: %.3e\n", h_mae(h_E_pred, h_E_test_gt, N_test));
    printf("  Force  MAE: %.3e\n\n", h_mae(h_F_pred, h_F_test_gt, N_test * D));

    float max_alpha = 0.0f, max_alpha_f = 0.0f;
    for (int i = 0; i < (int)full_train; i++) {
        float a = fabsf(h_alpha_full[i]);
        if (a > max_alpha) max_alpha = a;
    }
    for (int i = N_train; i < (int)full_train; i++) {
        float a = fabsf(h_alpha_full[i]);
        if (a > max_alpha_f) max_alpha_f = a;
    }
    printf("--- Solver diagnostics ---\n");
    printf("  Max |alpha|:       %.3e\n", max_alpha);
    printf("  Max |alpha_force|: %.3e  (should be ~0 for energy-only alpha_true)\n\n",
           max_alpha_f);

    printf("=== Summary ===\n");
    printf("  Training:    %8.4f s  (assembly %.4f s + solver %.4f s)\n",
           gpu_train, tt_build, tt_solve);
    printf("  Inference:   %10.6f s  (%.1f µs / molecule)\n",
           gpu_pred, 1e6f * gpu_pred / N_test);
    printf("  K_full: %.2f GB   C only: %.2f MB\n", K_train_GB,
           (double)N_test * N_train * sizeof(float) / (1 << 20));

    /* ------------------------------------------------------------------ */
    /* Cleanup                                                              */
    /* ------------------------------------------------------------------ */
    krr_inference_free(&inf_state);
    cusolverDnDestroy(cusolver);
    cublasDestroy(cublas);

    free(h_X_train);   free(h_dXT_train);
    free(h_X_test);    free(h_dXT_test);
    free(h_alpha_true); free(h_alpha_full);
    free(h_E_train);   free(h_F_train);
    free(h_E_test_gt); free(h_F_test_gt);
    free(h_E_pred);    free(h_F_pred);

    return 0;
}
