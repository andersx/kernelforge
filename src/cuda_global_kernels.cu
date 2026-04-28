// cuda_global_kernels.cu — GPU Gaussian kernel functions for global descriptors.
//
// Implements kernel_gaussian_full_symm_cu and kernel_gaussian_full_matvec_cu.
//
// Naming and structure mirror global_kernels.cpp.  Memory layout follows the
// cuBLAS column-major convention: a Python/numpy/torch (N, M) C-contiguous
// array is passed as a (M, N) column-major matrix.
//
// Two public functions in namespace kf:
//
//   kernel_gaussian_full_symm_cu
//       Builds the symmetric (N*(1+D))² training kernel matrix (both triangles).
//       Analogue of kf::kernel_gaussian_full_symm on CPU.
//
//   kernel_gaussian_full_matvec_cu
//       Contracted energy+force prediction using the J^T·α trick.
//       Avoids materialising the full test×train kernel matrix.
//       Analogue of kf::kernel_gaussian_full_matvec on CPU.

#include "cuda_global_kernels.hpp"
#include "curfp.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>


// ============================================================================
// Error-checking macros
// ============================================================================

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            abort();                                                        \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t _s = (call);                                         \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d — status %d\n",         \
                    __FILE__, __LINE__, (int)_s);                           \
            abort();                                                        \
        }                                                                   \
    } while (0)

#define CURFP_CHECK(call)                                                   \
    do {                                                                    \
        curfpStatus_t _s = (call);                                          \
        if (_s != CURFP_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuRFP error at %s:%d — status %d\n",          \
                    __FILE__, __LINE__, (int)_s);                           \
            abort();                                                        \
        }                                                                   \
    } while (0)


// ============================================================================
// Module-level cuBLAS handle (lazily initialised, shared across all calls)
// ============================================================================

static cublasHandle_t s_cublas = nullptr;

static void ensure_cublas()
{
    if (!s_cublas) {
        if (cublasCreate(&s_cublas) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed\n");
            abort();
        }
    }
}


// ============================================================================
// Module-level cuRFP handle (lazily initialised, shared across all calls)
// ============================================================================

static curfpHandle_t s_curfp = nullptr;

static void ensure_curfp()
{
    if (!s_curfp) {
        if (curfpCreate(&s_curfp) != CURFP_STATUS_SUCCESS) {
            fprintf(stderr, "curfpCreate failed\n");
            abort();
        }
    }
}


// ============================================================================
// CUDA device kernels (all file-static)
// ============================================================================

// compute_sqnorms_kernel
// norms[i] = ||X[:,i]||²   X is (M, N) col-major.
__global__ static void compute_sqnorms_kernel(
    const float *X, float *norms, int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    const float *x = X + (long long)col * M;
    float s = 0.0f;
    for (int i = 0; i < M; i++) s += x[i] * x[i];
    norms[col] = s;
}


// add_sym_norms_kernel
// G[a,b] += norms[a] + norms[b]   (symmetric, G is (N,N) col-major)
__global__ static void add_sym_norms_kernel(float *G, const float *norms, int N)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;
    G[a + (long long)b * N] += norms[a] + norms[b];
}


// add_asym_norms_kernel
// G[a,b] += norms1[a] + norms2[b]   G is (N1,N2) col-major.
__global__ static void add_asym_norms_kernel(
    float *G, const float *norms1, const float *norms2, int N1, int N2)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N1 || b >= N2) return;
    G[a + (long long)b * N1] += norms1[a] + norms2[b];
}


// build_C_C4_kernel
// On entry:  G[a,b] = ||x_a - x_b||²
// On exit:   G[a,b]  = K[a,b] / sigma²  (= C)
//            C4[a,b] = K[a,b] / sigma⁴  (= C4)
__global__ static void build_C_C4_kernel(
    float *G, float *C4, float inv_s2, int N1, int N2)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N1 || b >= N2) return;
    long long idx = a + (long long)b * N1;
    float k = expf(-0.5f * inv_s2 * G[idx]);
    G[idx]  = k * inv_s2;
    C4[idx] = k * inv_s2 * inv_s2;
}


// build_C_kernel
// In-place: G[a,b] = exp(-0.5 * inv_s2 * G[a,b]) * inv_s2
__global__ static void build_C_kernel(float *G, float inv_s2, int N1, int N2)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N1 || b >= N2) return;
    long long idx = a + (long long)b * N1;
    G[idx] = expf(-0.5f * inv_s2 * G[idx]) * inv_s2;
}


// extract_U_kernel
// U[g] = V1X2[g, g/D]   (diagonal self-projections, g = a*D+d, a = g/D)
// V1X2 is (ND, N) col-major; U[g] = element at row g, col g/D.
__global__ static void extract_U_kernel(
    const float *V1X2, float *U, int N, int D)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int ND = N * D;
    if (g >= ND) return;
    int a = g / D;
    U[g] = V1X2[g + (long long)a * ND];
}


// assemble_row_kernel
// Fills one block-row a of K_full (col-major) from precomputed C, C4, G_row, V1X2, U1.
// Launched with (a+1) blocks × min(D*D, 1024) threads per block.
// Stride loop handles D*D > 1024 (e.g. azobenzene: D=72, D*D=5184).
__global__ static void assemble_row_kernel(
    float       *K_full,
    const float *G_row,       // row slice of G_batch starting at column a_start
    int          G_lda,       // leading dimension of G_batch
    const float *V1X2,        // (ND, N) col-major
    const float *U1,          // (ND,)
    const float *C,           // (N, N) col-major
    const float *C4,          // (N, N) col-major
    float        sigma2,
    int a, int N, int D,
    long long full_rows)
{
    int b   = blockIdx.x;
    int DD  = D * D;
    int ND  = N * D;

    float c  = C [a + (long long)b * N];
    float c4 = C4[a + (long long)b * N];

    /* K_EE[a,b] = K[a,b] */
    if (threadIdx.x == 0)
        K_full[a + (long long)b * full_rows] = c * sigma2;

    for (int tid = (int)threadIdx.x; tid < DD; tid += (int)blockDim.x) {
        int d1 = tid / D;
        int d2 = tid % D;
        int g1 = a * D + d1;
        int g2 = b * D + d2;

        float p = V1X2[g1 + (long long)b * ND] - U1[g1];
        float q = V1X2[g2 + (long long)a * ND] - U1[g2];

        float g = G_row[d1 + (long long)(b * D + d2) * G_lda];

        /* K_FE[N+g1, b] = C[a,b] * p */
        if (d2 == 0)
            K_full[(N + g1) + (long long)b * full_rows] = c * p;
        /* K_FE mirror [N+g2, a] — only when a > b */
        if (a > b && d1 == 0)
            K_full[(N + g2) + (long long)a * full_rows] = c * q;

        /* K_FF lower triangle */
        if (a > b || g1 >= g2)
            K_full[(N + g1) + (long long)(N + g2) * full_rows] = c * g + c4 * p * q;
    }
}


// rfp_index_lower_N
//
// Returns the 1-D index into a TRANSR=N, UPLO=L RFP buffer for element
// A[row, col] where row >= col (lower triangle).  n is the matrix dimension.
//
// Even n (nk = n/2, column stride = n+1):
//   j < nk  →  1 + i + j*(n+1)
//   j >= nk →  (j-nk) + (i-nk)*(n+1)
//
// Odd n (n1 = ceil(n/2), column stride = n):
//   j < n1  →  i + j*n
//   j >= n1 →  n + (j-n1) + (i-n1)*n
__device__ static long long rfp_index_lower_N(int n, int row, int col)
{
    int i = row, j = col;   /* lower triangle: i >= j */
    if (n & 1) {
        int n1 = (n + 1) / 2;
        if (j < n1)
            return (long long)i + (long long)j * n;
        else
            return (long long)n + (j - n1) + (long long)(i - n1) * n;
    } else {
        int nk = n / 2;
        if (j < nk)
            return 1LL + i + (long long)j * (n + 1);
        else
            return (long long)(j - nk) + (long long)(i - nk) * (n + 1);
    }
}


// assemble_rfp_row_kernel
//
// Like assemble_row_kernel but writes elements directly into the RFP
// packed buffer (TRANSR=N, UPLO=L) using rfp_index_lower_N.
// Only lower-triangle elements are stored; no mirror step is needed.
// Launched with (a+1) blocks × min(D*D, 1024) threads per block.
// Stride loop handles D*D > 1024 (e.g. azobenzene: D=72, D*D=5184).
__global__ static void assemble_rfp_row_kernel(
    float       *K_rfp,
    const float *G_row,
    int          G_lda,
    const float *V1X2,
    const float *U1,
    const float *C,
    const float *C4,
    float        sigma2,
    int a, int N, int D,
    int BIG)
{
    int b   = blockIdx.x;
    int DD  = D * D;
    int ND  = N * D;

    float c  = C [a + (long long)b * N];
    float c4 = C4[a + (long long)b * N];

    /* K_EE[a, b]: a >= b always → lower triangle */
    if (threadIdx.x == 0)
        K_rfp[rfp_index_lower_N(BIG, a, b)] = c * sigma2;

    for (int tid = (int)threadIdx.x; tid < DD; tid += (int)blockDim.x) {
        int d1 = tid / D;
        int d2 = tid % D;
        int g1 = a * D + d1;
        int g2 = b * D + d2;

        float p = V1X2[g1 + (long long)b * ND] - U1[g1];
        float q = V1X2[g2 + (long long)a * ND] - U1[g2];

        float g = G_row[d1 + (long long)(b * D + d2) * G_lda];

        /* K_FE[N+g1, b]: always lower triangle (N+g1 >= N > b) */
        if (d2 == 0)
            K_rfp[rfp_index_lower_N(BIG, N + g1, b)] = c * p;

        /* K_FE[N+g2, a]: lower triangle, only when a > b */
        if (a > b && d1 == 0)
            K_rfp[rfp_index_lower_N(BIG, N + g2, a)] = c * q;

        /* K_FF lower triangle: g1 >= g2 */
        if (a > b || g1 >= g2)
            K_rfp[rfp_index_lower_N(BIG, N + g1, N + g2)] = c * g + c4 * p * q;
    }
}


// mirror_lower_to_upper_kernel
// For a (N_full × N_full) col-major matrix A, copies A[row,col] (row > col)
// to A[col,row], making A fully symmetric.
__global__ static void mirror_lower_to_upper_kernel(float *A, long long N_full)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long row = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= N_full || row >= N_full || row <= col) return;
    A[col + row * N_full] = A[row + col * N_full];
}


// row_dot_kernel
// out[m] = A[m,:] · B[m,:]   (A, B are (N, M) row-major)
__global__ static void row_dot_kernel(
    float *out, const float *A, const float *B, int N, int M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    float s = 0.0f;
    for (int k = 0; k < M; k++) s += A[(long long)row * M + k] * B[(long long)row * M + k];
    out[row] = s;
}


// build_W_combined_kernel
// W_combined[m*M + k] = alpha_E[m] * X_t[m*M + k] + alpha_desc_F[m*M + k]
// All three arrays are (N_t, M) row-major.
__global__ static void build_W_combined_kernel(
    float *W_combined, const float *X_t, const float *alpha_E,
    const float *alpha_desc_F, int N_t, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_t * M) return;
    int m = idx / M;
    W_combined[idx] = alpha_E[m] * X_t[idx] + alpha_desc_F[idx];
}


// build_weight_F_kernel
// weight_F[a + b*N_q] = C[a + b*N_q] * inv_s2 * (cross_F[a + b*N_q] - self_cross_F[b])
// All are (N_q, N_t) col-major; self_cross_F is (N_t,).
__global__ static void build_weight_F_kernel(
    float *weight_F, const float *C, const float *cross_F,
    const float *self_cross_F, float inv_s2, int N_q, int N_t)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N_q || b >= N_t) return;
    long long idx = a + (long long)b * N_q;
    weight_F[idx] = C[idx] * inv_s2 * (cross_F[idx] - self_cross_F[b]);
}


// subtract_scaled_cols_kernel
// G_full[m + a*M] -= combined[a] * X_q[m + a*M]
// G_full and X_q are (M, N_q) col-major; combined is (N_q,).
__global__ static void subtract_scaled_cols_kernel(
    float *G_full, const float *X_q, const float *combined, int M, int N_q)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || a >= N_q) return;
    G_full[m + (long long)a * M] -= combined[a] * X_q[m + (long long)a * M];
}


// fused_energy_kernel
// E_pred[a] = sigma2 * (w_E[a] + sum_F[a])
// combined[a] = w_E[a] + sum_F[a]
__global__ static void fused_energy_kernel(
    float *E_pred, float *combined,
    const float *w_E, const float *sum_F, float sigma2, int N_q)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= N_q) return;
    float c = w_E[a] + sum_F[a];
    combined[a] = c;
    E_pred[a]   = sigma2 * c;
}


// ============================================================================
// RFP device kernels (fixed convention: TRANSR=N, UPLO=L)
// ============================================================================

// fill_ones_kernel — fills a float device vector with 1.0f
__global__ static void fill_ones_kernel(float *v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = 1.0f;
}


// apply_gaussian_rfp_kernel
//
// In-place: x → exp(-0.5 * inv_s2 * x)
__global__ static void apply_gaussian_rfp_kernel(
    float *K_rfp, float inv_s2, int nt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nt) return;
    K_rfp[i] = expf(-0.5f * inv_s2 * K_rfp[i]);
}


// add_l2_diag_rfp_kernel
// Adds l2 to the N diagonal elements K_rfp[i,i] in the packed RFP buffer.
// Diagonal index formula (TRANSR=N, UPLO=L):
//   Odd  N: i >= n1  →  (i-n2)*N + (i-n1)
//           else     →  i*N + i
//   Even N: i >= nk  →  (i-nk)*(N+2)
//           else     →  i*(N+2) + 1
__global__ static void add_l2_diag_rfp_kernel(float *K_rfp, float l2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    long arf_idx;
    if (N & 1) {
        int n1 = N - N / 2;
        int n2 = N / 2;
        if (i >= n1)
            arf_idx = (long)(i - n2) * N + (i - n1);
        else
            arf_idx = (long)i * N + i;
    } else {
        int nk = N / 2;
        if (i >= nk)
            arf_idx = (long)(i - nk) * (N + 2);
        else
            arf_idx = (long)i * (N + 2) + 1;
    }
    K_rfp[arf_idx] += l2;
}


// ============================================================================
// build_kernel_gpu_symm_lower
//
// Internal helper: fills the lower triangle of K_full (col-major) using
// batched SGEMM + assemble_row_kernel.  No mirror step — caller does that.
// ============================================================================

#define BUILD_BATCH 64

static void build_kernel_gpu_symm_lower(
    const float *d_X, const float *d_dXT, float *d_K_full,
    float sigma, int N, int M, int D)
{
    long long ND        = (long long)N * D;
    long long full_rows = (long long)N * (1 + D);

    float inv_s2 = 1.0f / (sigma * sigma);
    float sigma2 = sigma * sigma;

    float *d_norms, *d_C, *d_C4, *d_V1X2, *d_U1, *d_G_batch;

    CUDA_CHECK(cudaMalloc(&d_norms,   (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,       (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C4,      (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V1X2,    ND * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U1,      ND * sizeof(float)));
    {
        int brows = (N < BUILD_BATCH) ? N : BUILD_BATCH;
        CUDA_CHECK(cudaMalloc(&d_G_batch, (long long)brows * D * ND * sizeof(float)));
    }

    const float one = 1.0f, neg2 = -2.0f, zero = 0.0f;

    /* Phase 1: squared distances → C, C4 */
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

    /* Phase 2: V1X2 = dXT^T @ X, then U1 from diagonal */
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)ND, N, M,
        &one, d_dXT, M, d_X, M,
        &zero, d_V1X2, (int)ND));
    extract_U_kernel<<<((int)ND + 255) / 256, 256>>>(d_V1X2, d_U1, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 3: row-batched SGEMM + assemble */
    {
        int brows = (N < BUILD_BATCH) ? N : BUILD_BATCH;
        int DD    = D * D;

        for (int a_start = 0; a_start < N; a_start += brows) {
            int a_end  = a_start + brows;
            if (a_end > N) a_end = N;
            int n_rows = a_end - a_start;
            int n_cols = a_end * D;

            CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                n_rows * D, n_cols, M,
                &one,
                d_dXT + (long long)a_start * D * M, M,
                d_dXT, M,
                &zero,
                d_G_batch, n_rows * D));

            for (int a = a_start; a < a_end; a++) {
                assemble_row_kernel<<<a + 1, (DD < 1024 ? DD : 1024)>>>(
                    d_K_full,
                    d_G_batch + (long long)(a - a_start) * D,
                    n_rows * D,
                    d_V1X2, d_U1, d_C, d_C4,
                    sigma2, a, N, D, full_rows);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_norms);
    cudaFree(d_C);   cudaFree(d_C4);
    cudaFree(d_V1X2); cudaFree(d_U1);
    cudaFree(d_G_batch);
}


// ============================================================================
// Public functions in namespace kf
// ============================================================================

namespace kf {

// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm_cu
// ---------------------------------------------------------------------------
void kernel_gaussian_full_symm_cu(
    const float *d_X, const float *d_dXT, float *d_K_full,
    float sigma, int N, int M, int D)
{
    ensure_cublas();

    long long full_rows = (long long)N * (1 + D);

    /* Build lower triangle */
    build_kernel_gpu_symm_lower(d_X, d_dXT, d_K_full, sigma, N, M, D);

    /* Mirror lower → upper so K_full is fully symmetric */
    {
        dim3 blk(16, 16);
        dim3 grd(((int)full_rows + 15) / 16, ((int)full_rows + 15) / 16);
        mirror_lower_to_upper_kernel<<<grd, blk>>>(d_K_full, full_rows);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ---------------------------------------------------------------------------
// kernel_gaussian_full_matvec_cu
//
// Contracted inference:
//   C[a,b]          = K(x_q[a], x_t[b]) / sigma²
//   alpha_desc_F[b] = Σ_d J_t[b,d,:] * alpha_F[b,d]   (precomputed, (M,N_t) col-major)
//   W_combined[b,:] = alpha_E[b] * x_t[b] + alpha_desc_F[b]
//   self_cross_F[b] = x_t[b] · alpha_desc_F[b]
//   cross_F[a,b]    = x_q[a] · alpha_desc_F[b]
//   weight_F[a,b]   = C[a,b] / sigma² * (cross_F[a,b] - self_cross_F[b])
//   G_full[a]       = Σ_b C[a,b]*W_combined[b] + Σ_b weight_F[a,b]*x_t[b]
//                     - (w_E[a]+sum_F[a]) * x_q[a]
//   E_pred[a]       = sigma² * (w_E[a] + sum_F[a])
//   F_pred[a]       = -(dX_q[a]) @ G_full[a]
// ---------------------------------------------------------------------------
void kernel_gaussian_full_matvec_cu(
    const float *d_X_q, const float *d_dXT_q,
    const float *d_X_t, const float *d_alpha_E, const float *d_alpha_desc_F,
    float *d_E_pred, float *d_F_pred,
    float sigma, int N_q, int N_t, int M, int D)
{
    ensure_cublas();

    float inv_s2 = 1.0f / (sigma * sigma);
    float sigma2 = sigma * sigma;
    float neg2   = -2.0f;
    const float one = 1.0f, zero = 0.0f, neg_one = -1.0f;

    /* Allocate all temporaries */
    float *d_norms_q, *d_norms_t, *d_C, *d_cross_F, *d_self_cross_F;
    float *d_weight_F, *d_w_E, *d_sum_F, *d_combined;
    float *d_W_combined, *d_G_full, *d_ones_t;

    CUDA_CHECK(cudaMalloc(&d_norms_q,     (size_t)N_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms_t,     (size_t)N_t * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,           (long long)N_q * N_t * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cross_F,     (long long)N_q * N_t * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_self_cross_F,(size_t)N_t * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight_F,    (long long)N_q * N_t * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_E,         (size_t)N_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_F,       (size_t)N_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_combined,    (size_t)N_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_combined,  (long long)N_t * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G_full,      (long long)M * N_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones_t,      (size_t)N_t * sizeof(float)));

    /* Fill ones vector */
    {
        float *h_ones = (float*)malloc(N_t * sizeof(float));
        for (int i = 0; i < N_t; i++) h_ones[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_ones_t, h_ones, N_t * sizeof(float), cudaMemcpyHostToDevice));
        free(h_ones);
    }

    /* Phase 1: squared norms */
    compute_sqnorms_kernel<<<(N_q + 255) / 256, 256>>>(d_X_q, d_norms_q, M, N_q);
    compute_sqnorms_kernel<<<(N_t + 255) / 256, 256>>>(d_X_t, d_norms_t, M, N_t);

    /* Phase 2: build C[N_q, N_t] = K[a,b]/sigma² (col-major) */
    /* C = -2 * X_q^T @ X_t  (CUBLAS: (N_q,N_t) = X_q^T[N_q,M] @ X_t[M,N_t]) */
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N_q, N_t, M,
        &neg2, d_X_q, M, d_X_t, M,
        &zero, d_C, N_q));
    {
        dim3 blk(16, 16);
        dim3 grd((N_q + 15) / 16, (N_t + 15) / 16);
        add_asym_norms_kernel<<<grd, blk>>>(d_C, d_norms_q, d_norms_t, N_q, N_t);
        build_C_kernel<<<grd, blk>>>(d_C, inv_s2, N_q, N_t);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 3: self_cross_F[b] = x_t[b] · alpha_desc_F[b]
     * alpha_desc_F is (M, N_t) col-major = (N_t, M) row-major.
     * X_t same layout.  row_dot_kernel uses (N_t, M) row-major ordering. */
    /* Reinterpret alpha_desc_F (M, N_t) col-major as (N_t, M) row-major for row_dot */
    /* Element [b, k] in (N_t, M) row-major = alpha_desc_F_colmaj[k, b] = alpha_desc_F[k + b*M] */
    /* But alpha_desc_F is stored as (M, N_t) col-major: element [k, b] at k + b*M = same address */
    /* So the raw pointer layout (N_t*M floats) is identical whether read as (N_t,M) row-major
     * or (M,N_t) col-major — the index formula is the same (k + b*M). ✓ */
    /* X_t is (M, N_t) col-major; read as (N_t, M) row-major: element [b,k] at b*M+k = k+b*M. ✓ */
    row_dot_kernel<<<(N_t + 255) / 256, 256>>>(d_self_cross_F, d_X_t, d_alpha_desc_F, N_t, M);

    /* Phase 4: cross_F[a,b] = x_q[a] · alpha_desc_F[b]
     * cross_F[N_q, N_t] = X_q^T[N_q,M] @ alpha_desc_F[M,N_t]
     * cublasSgemm: (N_q, N_t) = X_q^T @ alpha_desc_F */
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N_q, N_t, M,
        &one, d_X_q, M, d_alpha_desc_F, M,
        &zero, d_cross_F, N_q));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 5: weight_F[a,b] = C[a,b]/sigma² * (cross_F[a,b] - self_cross_F[b]) */
    {
        dim3 blk(16, 16);
        dim3 grd((N_q + 15) / 16, (N_t + 15) / 16);
        build_weight_F_kernel<<<grd, blk>>>(d_weight_F, d_C, d_cross_F, d_self_cross_F,
                                            inv_s2, N_q, N_t);
    }

    /* Phase 6: w_E = C @ alpha_E,  sum_F = weight_F @ ones */
    CUBLAS_CHECK(cublasSgemv(s_cublas, CUBLAS_OP_N, N_q, N_t,
        &one, d_C, N_q, d_alpha_E, 1, &zero, d_w_E, 1));
    CUBLAS_CHECK(cublasSgemv(s_cublas, CUBLAS_OP_N, N_q, N_t,
        &one, d_weight_F, N_q, d_ones_t, 1, &zero, d_sum_F, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 7: E_pred and combined */
    fused_energy_kernel<<<(N_q + 255) / 256, 256>>>(
        d_E_pred, d_combined, d_w_E, d_sum_F, sigma2, N_q);

    /* Phase 8: W_combined[b, k] = alpha_E[b] * X_t[b, k] + alpha_desc_F[b, k]
     * All (N_t, M) row-major layout (identical byte-layout as (M, N_t) col-major pointer) */
    {
        int total = N_t * M;
        build_W_combined_kernel<<<(total + 255) / 256, 256>>>(
            d_W_combined, d_X_t, d_alpha_E, d_alpha_desc_F, N_t, M);
    }

    /* Phase 9: G_full[M, N_q] = W_combined[M, N_t] @ C[N_t, N_q]^T (col-major)
     * G_full[k, a] = Σ_b W_combined[k, b] * C[a, b] ✓ */
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        M, N_q, N_t,
        &one, d_W_combined, M, d_C, N_q,
        &zero, d_G_full, M));

    /* G_full += X_t[M, N_t] @ weight_F[N_q, N_t]^T */
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        M, N_q, N_t,
        &one, d_X_t, M, d_weight_F, N_q,
        &one, d_G_full, M));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* G_full[:,a] -= combined[a] * X_q[:,a] */
    {
        dim3 blk(16, 16);
        dim3 grd((M + 15) / 16, (N_q + 15) / 16);
        subtract_scaled_cols_kernel<<<grd, blk>>>(d_G_full, d_X_q, d_combined, M, N_q);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 10: F_pred[a] = -dX_q[a] @ G_full[a]  (batched SGEMV)
     * dXT_q is (M, N_q*D) col-major → batch a at offset a*D*M, each (M,D) col-major, lda=M
     * G_full is (M, N_q) col-major → column a at a*M, stride M between batches
     * F_pred is (N_q*D) = (N_q, D) row-major → batch a at a*D, stride D */
    CUBLAS_CHECK(cublasSgemvStridedBatched(s_cublas,
        CUBLAS_OP_T, M, D,
        &neg_one,
        d_dXT_q, M, (long long)M * D,
        d_G_full, 1, (long long)M,
        &zero,
        d_F_pred, 1, (long long)D,
        N_q));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Cleanup */
    cudaFree(d_norms_q);    cudaFree(d_norms_t);
    cudaFree(d_C);          cudaFree(d_cross_F);
    cudaFree(d_self_cross_F);
    cudaFree(d_weight_F);   cudaFree(d_w_E);
    cudaFree(d_sum_F);      cudaFree(d_combined);
    cudaFree(d_W_combined); cudaFree(d_G_full);
    cudaFree(d_ones_t);
}


// ---------------------------------------------------------------------------
// kernel_gaussian_symm_rfp_cu
//
// Build the energy-only (N×N) Gaussian kernel matrix in RFP packed format.
//
// Convention fixed to TRANSR=N, UPLO=L throughout.  The caller must pass d_K_rfp pointing to
// N*(N+1)/2 floats of device memory.
//
// Parameters (device pointers, cuBLAS col-major):
//   d_X    : (M, N) col-major — descriptor columns
//   d_K_rfp: (N*(N+1)/2,) — output RFP buffer (col-major, TRANSR=N, UPLO=L)
//   sigma  : Gaussian length-scale
//   N, M   : dimensions
// ---------------------------------------------------------------------------
void kernel_gaussian_symm_rfp_cu(
    const float *d_X, float *d_K_rfp, float sigma, int N, int M)
{
    ensure_curfp();

    float inv_s2 = 1.0f / (sigma * sigma);
    float neg2   = -2.0f;
    float zero   =  0.0f;
    int   nt     = N * (N + 1) / 2;

    /* Allocate squared norms and a ones vector (freed after Phase 4 sync) */
    float *d_norms, *d_ones;
    CUDA_CHECK(cudaMalloc(&d_norms, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones,  (size_t)N * sizeof(float)));

    /* Phase 1: d_norms[i] = ||X[:,i]||² */
    compute_sqnorms_kernel<<<(N + 255) / 256, 256>>>(d_X, d_norms, M, N);

    /* Phase 1b: fill d_ones with 1.0f (can overlap with sqnorms) */
    fill_ones_kernel<<<(N + 255) / 256, 256>>>(d_ones, N);

    /* Phase 2: d_K_rfp = -2 * X^T * X  (in RFP, TRANSR=N, UPLO=L, trans=T) */
    CURFP_CHECK(curfpSsfrk(s_curfp,
        CURFP_OP_N,           /* transr = N */
        CURFP_FILL_MODE_LOWER,/* uplo   = L */
        CURFP_OP_T,           /* trans  = T: C = alpha * A^T * A + beta * C */
        N, M,
        &neg2, d_X, M,        /* alpha=-2, A=(M,N) col-major, lda=M */
        &zero, d_K_rfp));     /* beta=0 */

    /* Phase 3: d_K_rfp[i,j] += norms[i] + norms[j]  (TRANSR=N, UPLO=L)
     * curfpSsfr2 computes C := alpha*x*y^T + alpha*y*x^T + C.
     * With x=d_norms, y=d_ones, alpha=1: C[i,j] += norms[i] + norms[j]. */
    {
        float one = 1.0f;
        CURFP_CHECK(curfpSsfr2(s_curfp,
            CURFP_OP_N,            /* transr = N */
            CURFP_FILL_MODE_LOWER, /* uplo   = L */
            N, &one,
            d_norms, 1,
            d_ones,  1,
            d_K_rfp));
    }

    /* Phase 4: d_K_rfp[i] = exp(-0.5 * inv_s2 * d_K_rfp[i]) */
    apply_gaussian_rfp_kernel<<<(nt + 255) / 256, 256>>>(
        d_K_rfp, inv_s2, nt);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_norms);
    cudaFree(d_ones);
}


// ---------------------------------------------------------------------------
// rfp_potrf_cu
//
// Adds l2 to the diagonal of the RFP matrix (regularisation), then performs
// Cholesky factorisation in-place.
//
// Convention fixed to TRANSR=N, UPLO=L.
// On exit d_K_rfp contains the lower Cholesky factor L in RFP format.
// *info: 0 = success, >0 = leading minor of order *info is not positive definite.
// ---------------------------------------------------------------------------
void rfp_potrf_cu(float *d_K_rfp, int N, float l2, int *info)
{
    ensure_curfp();

    /* Add l2 to diagonal */
    if (l2 != 0.0f)
        add_l2_diag_rfp_kernel<<<(N + 255) / 256, 256>>>(d_K_rfp, l2, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    /* Cholesky factorisation in-place */
    CURFP_CHECK(curfpSpftrf(s_curfp,
        CURFP_OP_N,            /* transr = N */
        CURFP_FILL_MODE_LOWER, /* uplo   = L */
        N, d_K_rfp, info));
}


// ---------------------------------------------------------------------------
// rfp_potrs_cu
//
// Triangular solve using the Cholesky factor produced by rfp_potrf_cu.
// Solves (L * L^T) * X = B in-place, overwriting d_B with the solution.
//
// d_B  : (N, nrhs) col-major device pointer, leading dimension N.
// Convention fixed to TRANSR=N, UPLO=L.
// ---------------------------------------------------------------------------
void rfp_potrs_cu(const float *d_L_rfp, float *d_B, int N, int nrhs)
{
    ensure_curfp();
    CURFP_CHECK(curfpSpftrs(s_curfp,
        CURFP_OP_N,            /* transr = N */
        CURFP_FILL_MODE_LOWER, /* uplo   = L */
        N, nrhs, d_L_rfp, d_B, N));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm_rfp_cu
//
// Build the symmetric energy+force kernel matrix K_full for training,
// stored directly in RFP packed format (TRANSR=N, UPLO=L).
//
// No dense BIG×BIG intermediate is allocated; assemble_rfp_row_kernel
// writes each lower-triangle element exactly once via rfp_index_lower_N.
// No mirror step is needed.
//
// Parameters (device pointers, cuBLAS col-major):
//   d_X      : (M, N)    — descriptor columns
//   d_dXT    : (M, N*D)  — Jacobian columns (dX[a,d,:] is column a*D+d)
//   d_K_rfp  : (BIG*(BIG+1)/2,) — output RFP buffer, BIG = N*(1+D)
//   sigma, N, M, D: dimensions
// ---------------------------------------------------------------------------
void kernel_gaussian_full_symm_rfp_cu(
    const float *d_X, const float *d_dXT, float *d_K_rfp,
    float sigma, int N, int M, int D)
{
    ensure_cublas();

    long long ND  = (long long)N * D;
    int       BIG = N * (1 + D);

    float inv_s2 = 1.0f / (sigma * sigma);
    float sigma2 = sigma * sigma;

    float *d_norms, *d_C, *d_C4, *d_V1X2, *d_U1, *d_G_batch;

    CUDA_CHECK(cudaMalloc(&d_norms,   (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,       (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C4,      (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V1X2,    ND * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U1,      ND * sizeof(float)));
    {
        int brows = (N < BUILD_BATCH) ? N : BUILD_BATCH;
        CUDA_CHECK(cudaMalloc(&d_G_batch, (long long)brows * D * ND * sizeof(float)));
    }

    const float one = 1.0f, neg2 = -2.0f, zero = 0.0f;

    /* Phase 1: squared distances → C, C4 */
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

    /* Phase 2: V1X2 = dXT^T @ X, then U1 from diagonal */
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)ND, N, M,
        &one, d_dXT, M, d_X, M,
        &zero, d_V1X2, (int)ND));
    extract_U_kernel<<<((int)ND + 255) / 256, 256>>>(d_V1X2, d_U1, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 3: row-batched SGEMM + scatter into RFP buffer */
    {
        int brows = (N < BUILD_BATCH) ? N : BUILD_BATCH;
        int DD    = D * D;

        for (int a_start = 0; a_start < N; a_start += brows) {
            int a_end  = a_start + brows;
            if (a_end > N) a_end = N;
            int n_rows = a_end - a_start;
            int n_cols = a_end * D;

            CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                n_rows * D, n_cols, M,
                &one,
                d_dXT + (long long)a_start * D * M, M,
                d_dXT, M,
                &zero,
                d_G_batch, n_rows * D));

            for (int a = a_start; a < a_end; a++) {
                assemble_rfp_row_kernel<<<a + 1, (DD < 1024 ? DD : 1024)>>>(
                    d_K_rfp,
                    d_G_batch + (long long)(a - a_start) * D,
                    n_rows * D,
                    d_V1X2, d_U1, d_C, d_C4,
                    sigma2, a, N, D, BIG);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_norms);
    cudaFree(d_C);   cudaFree(d_C4);
    cudaFree(d_V1X2); cudaFree(d_U1);
    cudaFree(d_G_batch);
}

}  // namespace kf
