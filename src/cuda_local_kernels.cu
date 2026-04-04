// cuda_local_kernels.cu — GPU Gaussian kernel functions for local (FCHL19) descriptors.
//
// Implements three public functions in namespace kf::fchl19:
//
//   kernel_gaussian_full_symm_local_cu
//       Builds the full symmetric (nm+naq)² training kernel matrix.
//       Analogue of kf::fchl19::kernel_gaussian_full_symm on CPU.
//       Algorithm: one CUDA thread block per molecule pair (a,b) with a≤b;
//       threads within the block parallelise over atom pairs (i1,j2);
//       shared-memory accumulators for K_EE, K_FE, K_jact, K_FF.
//
//   compute_alpha_desc_local_cu
//       Computes alpha_desc[b,i2,k] = Σ_c dX[b,i2,k,c]*alpha_F[offs[b]+c]
//       on GPU — avoids a CPU round-trip after the Cholesky solve.
//
//   kernel_gaussian_full_matvec_local_cu
//       Contracted inference using the J^T·α trick (local version).
//       No full test-train kernel matrix is materialised.
//
// Memory layout throughout: C-contiguous (row-major).
//   X[m,i,k]       at d_X[(m*max_atoms+i)*rep + k]
//   dX[m,i,k,c]    at d_dX[((m*max_atoms+i)*rep+k)*3*max_atoms + c]
//   K_full[r,c]     at d_K_full[r*BIG + c]

#include "cuda_local_kernels.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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


// ============================================================================
// Module-level cuBLAS handle
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
// Device kernels
// ============================================================================

// ---------------------------------------------------------------------------
// compute_alpha_desc_kernel
//
// One thread per (b, i2) active atom pair.  Inner loops over k and c are
// serial within each thread — rep_size ≤ ~200 and ncols_b ≤ ~90 in practice.
// ---------------------------------------------------------------------------
__global__ static void compute_alpha_desc_kernel(
    const float *d_dX,         // (nm, max_atoms, rep, lda)  lda = 3*max_atoms
    const float *d_alpha_F,    // (naq,)
    const int   *d_N,          // (nm,)
    const int   *d_offs,       // (nm,) Cartesian prefix-sum offsets
    float       *d_alpha_desc, // (nm, max_atoms, rep) output
    int nm, int max_atoms, int rep_size)
{
    int flat = (int)(blockIdx.x * blockDim.x) + threadIdx.x;
    int b  = flat / max_atoms;
    int i2 = flat % max_atoms;
    if (b >= nm) return;
    int nb = d_N[b];
    if (i2 >= nb) return;

    int lda    = 3 * max_atoms;
    int offs_b = d_offs[b];
    int ncols_b = 3 * nb;

    long long base_dX = ((long long)(b * max_atoms + i2) * rep_size) * lda;
    long long base_out = (long long)(b * max_atoms + i2) * rep_size;

    for (int k = 0; k < rep_size; k++) {
        float s = 0.0f;
        const float *row = d_dX + base_dX + (long long)k * lda;
        const float *aF  = d_alpha_F + offs_b;
        for (int c = 0; c < ncols_b; c++)
            s += row[c] * aF[c];
        d_alpha_desc[base_out + k] = s;
    }
}


// ---------------------------------------------------------------------------
// assemble_full_symm_local_kernel
//
// Grid (nm, nm): block (bx=b, by=a) handles molecule pair (a,b), a ≤ b.
// Threads within the block are spread across atom pairs (i1, j2).
//
// Shared-memory layout (dynamic):
//   [0]              : sh_KEE        (1 float)
//   [1 .. ncM]       : sh_KFE        (ncols_max floats)
//   [1+ncM .. 1+2ncM]: sh_Kjact      (ncols_max floats)
//   [1+2ncM ..]      : sh_KFF        (ncols_max × ncols_max floats)
// where ncM = 3 * max_atoms (worst-case column count).
// ---------------------------------------------------------------------------
__global__ static void assemble_full_symm_local_kernel(
    const float *d_X,      // (nm, max_atoms, rep)
    const float *d_dX,     // (nm, max_atoms, rep, 3*max_atoms)
    const int   *d_Q,      // (nm, max_atoms)
    const int   *d_N,      // (nm,)
    const int   *d_offs,   // (nm,) Cartesian offsets
    float       *d_K_full, // (BIG, BIG) row-major
    float inv_2s2, float inv_s2, float inv_s4,
    int nm, int max_atoms, int rep_size, int BIG)
{
    int b = blockIdx.x;
    int a = blockIdx.y;
    if (a > b || b >= nm || a >= nm) return;

    int na = d_N[a];
    int nb = d_N[b];
    if (na <= 0 || nb <= 0) return;

    int ncols_a   = 3 * na;
    int ncols_b   = 3 * nb;
    int col_off_a = d_offs[a];
    int row_off_b = d_offs[b];
    int lda       = 3 * max_atoms;
    int ncM       = 3 * max_atoms; // shared-memory stride for KFF

    // Shared memory layout
    extern __shared__ float sh[];
    float *sh_KEE   = sh;                          // 1 float
    float *sh_KFE   = sh + 1;                      // ncM floats
    float *sh_Kjact = sh + 1 + ncM;               // ncM floats
    float *sh_KFF   = sh + 1 + 2 * ncM;           // ncM * ncM floats

    // Zero accumulators
    int total_sh = 1 + 2 * ncM + ncM * ncM;
    for (int t = threadIdx.x; t < total_sh; t += blockDim.x)
        sh[t] = 0.0f;
    __syncthreads();

    // Each thread handles one or more atom pairs (i1, j2).
    // No large local arrays — all intermediate results stored in a few scalars
    // and recomputed via extra passes.  This avoids register/local-memory
    // overflow for large rep_size (e.g. FCHL19 rep_size ≈ 312).
    int total_pairs = na * nb;
    for (int pair = (int)threadIdx.x; pair < total_pairs; pair += (int)blockDim.x) {
        int i1 = pair / nb;
        int j2 = pair % nb;

        if (d_Q[a * max_atoms + i1] != d_Q[b * max_atoms + j2]) continue;

        const float *xa  = d_X   + (long long)(a * max_atoms + i1) * rep_size;
        const float *xb  = d_X   + (long long)(b * max_atoms + j2) * rep_size;
        const float *dxa = d_dX  + (long long)(a * max_atoms + i1) * rep_size * lda;
        const float *dxb = d_dX  + (long long)(b * max_atoms + j2) * rep_size * lda;

        // ---- Pass 1: l2, exp scalars, K_EE ----
        float l2 = 0.0f;
        for (int k = 0; k < rep_size; k++) {
            float dk = xa[k] - xb[k];
            l2 += dk * dk;
        }
        float exp_base = expf(l2 * inv_2s2);
        float expdiag  = exp_base * inv_s2;
        float expd     = -(exp_base * inv_s4);

        atomicAdd(sh_KEE, exp_base);

        // ---- Pass 2: K_FE and K_jact ----
        // Iterate c (outer) then k (inner), recomputing d[k] each time.
        // Use double accumulators to reduce float32 rounding errors
        // (rep_size can be ~300, so single-precision sums accumulate ~1e-5 error).
        int nc_max = (ncols_a > ncols_b) ? ncols_a : ncols_b;
        for (int c = 0; c < nc_max; c++) {
            double va = 0.0, vb = 0.0;
            for (int k = 0; k < rep_size; k++) {
                double dk = (double)xa[k] - (double)xb[k];
                if (c < ncols_a) va += (double)dxa[(long long)k * lda + c] * dk;
                if (c < ncols_b) vb += (double)dxb[(long long)k * lda + c] * dk;
            }
            if (c < ncols_a) atomicAdd(sh_KFE  + c, -expdiag * (float)va);
            if (c < ncols_b) atomicAdd(sh_Kjact + c,  expdiag * (float)vb);
        }

        // ---- Pass 3: K_FF (static + rank-1) ----
        // Outer: c1, then inner loop recomputes VA[c1] and iterates c2.
        // Double accumulators for static term (Σ_k dX_a * dX_b, ~300 terms).
        for (int c1 = 0; c1 < ncols_a; c1++) {
            double va = 0.0;
            for (int k = 0; k < rep_size; k++)
                va += (double)dxa[(long long)k * lda + c1]
                    * ((double)xa[k] - (double)xb[k]);

            for (int c2 = 0; c2 < ncols_b; c2++) {
                double vb = 0.0, sff = 0.0;
                for (int k = 0; k < rep_size; k++) {
                    double dxb_kc2 = (double)dxb[(long long)k * lda + c2];
                    double dk      = (double)xa[k] - (double)xb[k];
                    vb  += dxb_kc2 * dk;
                    sff += (double)dxa[(long long)k * lda + c1] * dxb_kc2;
                }
                atomicAdd(sh_KFF + c1 * ncM + c2,
                    expdiag * (float)sff + expd * (float)(va * vb));
            }
        }
    }

    __syncthreads();

    // ---- Write accumulated values to K_full --------------------------------
    //
    // Symmetry strategy:
    //   Off-diagonal (a < b): K_FE writes (nm+c, b) + transpose (b, nm+c).
    //                         K_jact writes (a, nm+c) + transpose (nm+c, a).
    //   Diagonal    (a == b): K_FE writes (nm+c, a) + (a, nm+c) — both from sh_KFE
    //                         to guarantee exact symmetry (sh_FE ≈ sh_Kjact by
    //                         analysis, but may differ by float32 atomicAdd order).
    //                         K_jact is skipped (already covered by K_FE mirror).

    // K_EE[a,b] and K_EE[b,a]
    if (threadIdx.x == 0) {
        float kee = sh_KEE[0];
        d_K_full[(long long)a * BIG + b] = kee;
        if (a != b)
            d_K_full[(long long)b * BIG + a] = kee;
    }

    // K_FE[nm+col_off_a+c1, b]  and ALWAYS its transpose K_full[b, nm+col_off_a+c1].
    // For a==b this makes the (nm+c, a) and (a, nm+c) entries equal (sh_KFE[c]).
    for (int c1 = (int)threadIdx.x; c1 < ncols_a; c1 += (int)blockDim.x) {
        float v = sh_KFE[c1];
        d_K_full[(long long)(nm + col_off_a + c1) * BIG + b] = v;
        d_K_full[(long long)b * BIG + (nm + col_off_a + c1)] = v;  // always (includes a==b)
    }

    // K_jact[a, nm+row_off_b+c2]  and its transpose — OFF-DIAGONAL ONLY.
    // For a==b the entry (a, nm+col_off_a+c) was already written by K_FE mirror above.
    for (int c2 = (int)threadIdx.x; c2 < ncols_b; c2 += (int)blockDim.x) {
        if (a != b) {
            float v = sh_Kjact[c2];
            d_K_full[(long long)a * BIG + (nm + row_off_b + c2)] = v;
            d_K_full[(long long)(nm + row_off_b + c2) * BIG + a] = v;
        }
    }

    // K_FF[nm+col_off_a+c1, nm+row_off_b+c2]  and transpose
    for (int t = (int)threadIdx.x; t < ncols_a * ncols_b; t += (int)blockDim.x) {
        int c1 = t / ncols_b;
        int c2 = t % ncols_b;
        float v = sh_KFF[c1 * ncM + c2];
        d_K_full[(long long)(nm + col_off_a + c1) * BIG + (nm + row_off_b + c2)] = v;
        if (a != b) {
            // Off-diagonal: full transpose (col_off_a region vs row_off_b region)
            d_K_full[(long long)(nm + row_off_b + c2) * BIG + (nm + col_off_a + c1)] = v;
        } else {
            // Diagonal (col_off_a == row_off_b): write symmetric partner using the
            // SAME value sh_KFF[c1*ncM+c2] for both (c1,c2) and (c2,c1) to enforce
            // exact symmetry (they are analytically equal; float32 rounding may differ).
            if (c1 != c2)
                d_K_full[(long long)(nm + col_off_a + c2) * BIG + (nm + col_off_a + c1)] = v;
        }
    }
}


// ---------------------------------------------------------------------------
// local_inference_accumulate_kernel
//
// One thread per active query atom (a, i1).  Each thread iterates over all
// training atoms with the same element label and accumulates:
//   E_partial[(a*max_atoms_q+i1)] — per-query-atom energy contribution
//   G_acc[(a*max_atoms_q+i1)*rep] — descriptor-space force accumulator
//   w_E_arr[(a*max_atoms_q+i1)]   — sum of expdiag*alpha_E for self-correction
//
// Grid: ceil(nm_q * max_atoms_q / BLOCK)
// Block: BLOCK threads
// ---------------------------------------------------------------------------
__global__ static void local_inference_accumulate_kernel(
    const float *d_X_q,       // (nm_q, max_atoms_q, rep)
    const float *d_X_t,       // (nm_t, max_atoms_t, rep)
    const float *d_alpha_E,   // (nm_t,)
    const float *d_alpha_desc,// (nm_t, max_atoms_t, rep)
    const int   *d_Q_q,       // (nm_q, max_atoms_q)
    const int   *d_Q_t,       // (nm_t, max_atoms_t)
    const int   *d_N_q,       // (nm_q,)
    const int   *d_N_t,       // (nm_t,)
    float       *d_E_partial, // (nm_q * max_atoms_q,) per-atom energy
    float       *d_G_acc,     // (nm_q * max_atoms_q, rep) force accumulator
    float       *d_wE_arr,    // (nm_q * max_atoms_q,) w_E for self-correction
    float inv_2s2, float inv_s2, float inv_s4,
    int nm_q, int nm_t, int max_atoms_q, int max_atoms_t, int rep_size)
{
    int flat_q = (int)(blockIdx.x * blockDim.x) + threadIdx.x;
    int a  = flat_q / max_atoms_q;
    int i1 = flat_q % max_atoms_q;
    if (a >= nm_q) return;
    int na = d_N_q[a];
    if (i1 >= na) return;

    int lbl = d_Q_q[a * max_atoms_q + i1];
    const float *xa = d_X_q + (long long)(a * max_atoms_q + i1) * rep_size;

    // Accumulators stored in global memory (G_acc) to avoid register pressure
    // for large rep_size (e.g. FCHL19 rep_size ≈ 312).
    // G_acc is pre-zeroed by the caller.
    float *G_out = d_G_acc + (long long)flat_q * rep_size;

    float E_loc = 0.0f;
    float wE    = 0.0f;

    for (int b = 0; b < nm_t; b++) {
        int nb_t   = d_N_t[b];
        float aE_b = d_alpha_E[b];

        for (int i2 = 0; i2 < nb_t; i2++) {
            if (d_Q_t[b * max_atoms_t + i2] != lbl) continue;

            const float *xb  = d_X_t      + (long long)(b * max_atoms_t + i2) * rep_size;
            const float *adF = d_alpha_desc + (long long)(b * max_atoms_t + i2) * rep_size;

            // Pass 1: l2 and inner_F (iterate k once)
            float l2 = 0.0f, inner_F = 0.0f;
            for (int k = 0; k < rep_size; k++) {
                float dk = xa[k] - xb[k];
                l2      += dk * dk;
                inner_F += dk * adF[k];
            }

            float exp_base   = expf(l2 * inv_2s2);
            float expdiag    = exp_base * inv_s2;
            float expd       = -(exp_base * inv_s4);
            float expd_iF    = expd * inner_F;
            float expdiag_aE = expdiag * aE_b;

            E_loc += exp_base * aE_b + expdiag * inner_F;
            wE    += expdiag_aE;

            // Pass 2: accumulate G_acc[k] (iterate k again, recomputing dk)
            for (int k = 0; k < rep_size; k++) {
                float dk = xa[k] - xb[k];
                G_out[k] += expdiag * adF[k]
                           + expd_iF  * dk
                           + expdiag_aE * xb[k];
            }
        }
    }

    // Self-correction: G_acc[k] -= wE * xa[k]
    for (int k = 0; k < rep_size; k++)
        G_out[k] -= wE * xa[k];

    d_E_partial[flat_q] = E_loc;
    d_wE_arr[flat_q]    = wE;
}


// ---------------------------------------------------------------------------
// local_energy_reduce_kernel
//
// E_pred[a] = Σ_{i1=0}^{N_q[a]-1} E_partial[a*max_atoms_q + i1]
// Grid: nm_q, Block: max_atoms_q (or 1 if small)
// ---------------------------------------------------------------------------
__global__ static void local_energy_reduce_kernel(
    const float *d_E_partial, // (nm_q * max_atoms_q,)
    const int   *d_N_q,       // (nm_q,)
    float       *d_E_pred,    // (nm_q,)
    int nm_q, int max_atoms_q)
{
    int a = blockIdx.x;
    if (a >= nm_q) return;
    int na = d_N_q[a];

    // Shared memory reduction
    extern __shared__ float sh_e[];
    float val = 0.0f;
    for (int i1 = threadIdx.x; i1 < na; i1 += blockDim.x)
        val += d_E_partial[(long long)a * max_atoms_q + i1];
    sh_e[threadIdx.x] = val;
    __syncthreads();

    // Simple tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sh_e[threadIdx.x] += sh_e[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        d_E_pred[a] = sh_e[0];
}


// ---------------------------------------------------------------------------
// local_force_backproject_kernel
//
// F[offs_q[a] + c] += Σ_{i1=0}^{N_q[a]-1}  Σ_k dX_q[a,i1,k,c] * G_acc[a,i1,k]
//
// Grid: nm_q blocks, each handling all atoms of one query molecule.
// Block: ncols_max threads (one per force Cartesian component).
// ---------------------------------------------------------------------------
__global__ static void local_force_backproject_kernel(
    const float *d_dX_q,    // (nm_q, max_atoms_q, rep, lda)  lda=3*max_atoms_q
    const float *d_G_acc,   // (nm_q * max_atoms_q, rep)
    const int   *d_N_q,     // (nm_q,)
    const int   *d_offs_q,  // (nm_q,) Cartesian offsets in F_pred
    float       *d_F_pred,  // (naq_q,)
    int nm_q, int max_atoms_q, int rep_size)
{
    int a = blockIdx.x;
    if (a >= nm_q) return;
    int na  = d_N_q[a];
    int lda = 3 * max_atoms_q;
    int ncols_a = 3 * na;
    int off_a   = d_offs_q[a];

    // Each thread handles one Cartesian component c
    int c = threadIdx.x;
    if (c >= ncols_a) return;

    float sum = 0.0f;
    for (int i1 = 0; i1 < na; i1++) {
        const float *dxa = d_dX_q + (long long)(a * max_atoms_q + i1) * rep_size * lda;
        const float *G   = d_G_acc + (long long)(a * max_atoms_q + i1) * rep_size;
        for (int k = 0; k < rep_size; k++)
            sum += dxa[(long long)k * lda + c] * G[k];
    }
    d_F_pred[off_a + c] = sum;
}


// ============================================================================
// Host-side helper: build Cartesian offset array from N
// ============================================================================
static void build_offsets(const int *d_N, int *d_offs, int nm)
{
    // Download N from device, compute prefix sums on host, upload
    int *h_N   = (int*)malloc(nm * sizeof(int));
    int *h_offs = (int*)malloc(nm * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_N, d_N, nm * sizeof(int), cudaMemcpyDeviceToHost));
    int acc = 0;
    for (int m = 0; m < nm; m++) {
        h_offs[m] = acc;
        int n_m = h_N[m];
        if (n_m < 0) n_m = 0;
        acc += 3 * n_m;
    }
    CUDA_CHECK(cudaMemcpy(d_offs, h_offs, nm * sizeof(int), cudaMemcpyHostToDevice));
    free(h_N);
    free(h_offs);
}


// ============================================================================
// Public functions in namespace kf::fchl19
// ============================================================================

namespace kf {
namespace fchl19 {

// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm_local_cu
// ---------------------------------------------------------------------------
void kernel_gaussian_full_symm_local_cu(
    const float *d_X,
    const float *d_dX,
    const int   *d_Q,
    const int   *d_N,
    float       *d_K_full,
    float        sigma,
    int nm, int max_atoms, int rep_size, int naq)
{
    ensure_cublas();

    int BIG = nm + naq;
    float inv_2s2 = -0.5f / (sigma * sigma);
    float inv_s2  =  1.0f / (sigma * sigma);
    float inv_s4  =  inv_s2 * inv_s2;

    // Build Cartesian offset array on GPU
    int *d_offs;
    CUDA_CHECK(cudaMalloc(&d_offs, nm * sizeof(int)));
    build_offsets(d_N, d_offs, nm);

    // Zero the output matrix
    CUDA_CHECK(cudaMemset(d_K_full, 0, (long long)BIG * BIG * sizeof(float)));

    // Shared memory per block: 1 + 2*ncM + ncM*ncM floats, ncM = 3*max_atoms
    int ncM      = 3 * max_atoms;
    size_t smem  = (size_t)(1 + 2 * ncM + ncM * ncM) * sizeof(float);

    // Check shared memory limit — bail out with a clear message if exceeded
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (smem > prop.sharedMemPerBlock) {
        fprintf(stderr,
            "cuda_local_kernels: max_atoms=%d requires %zu bytes shared memory "
            "but device only provides %zu.  Reduce max_atoms or pad to a smaller value.\n",
            max_atoms, smem, prop.sharedMemPerBlock);
        abort();
    }

    // Number of threads per block: enough to cover max atom pairs, rounded to warp
    int max_pairs = max_atoms * max_atoms;
    int block_sz  = ((max_pairs + 31) / 32) * 32;
    if (block_sz < 32)  block_sz = 32;
    if (block_sz > 256) block_sz = 256;  // cap at 256 for occupancy

    dim3 grid(nm, nm);
    assemble_full_symm_local_kernel<<<grid, block_sz, smem>>>(
        d_X, d_dX, d_Q, d_N, d_offs,
        d_K_full, inv_2s2, inv_s2, inv_s4,
        nm, max_atoms, rep_size, BIG);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_offs);
}


// ---------------------------------------------------------------------------
// compute_alpha_desc_local_cu
// ---------------------------------------------------------------------------
void compute_alpha_desc_local_cu(
    const float *d_dX,
    const float *d_alpha_F,
    const int   *d_N,
    float       *d_alpha_desc,
    int nm, int max_atoms, int rep_size, int /*naq*/)
{
    // Build offset array
    int *d_offs;
    CUDA_CHECK(cudaMalloc(&d_offs, nm * sizeof(int)));
    build_offsets(d_N, d_offs, nm);

    // Zero output
    CUDA_CHECK(cudaMemset(d_alpha_desc,
        0, (long long)nm * max_atoms * rep_size * sizeof(float)));

    int total = nm * max_atoms;
    int block  = 256;
    int grid   = (total + block - 1) / block;

    compute_alpha_desc_kernel<<<grid, block>>>(
        d_dX, d_alpha_F, d_N, d_offs, d_alpha_desc,
        nm, max_atoms, rep_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_offs);
}


// ---------------------------------------------------------------------------
// kernel_gaussian_full_matvec_local_cu
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
    int rep_size, int naq_q)
{
    ensure_cublas();

    float inv_2s2 = -0.5f / (sigma * sigma);
    float inv_s2  =  1.0f / (sigma * sigma);
    float inv_s4  =  inv_s2 * inv_s2;

    // Build query Cartesian offset array
    int *d_offs_q;
    CUDA_CHECK(cudaMalloc(&d_offs_q, nm_q * sizeof(int)));
    build_offsets(d_N_q, d_offs_q, nm_q);

    // Allocate per-atom accumulators
    long long n_q_atoms = (long long)nm_q * max_atoms_q;
    float *d_E_partial, *d_G_acc, *d_wE_arr;
    CUDA_CHECK(cudaMalloc(&d_E_partial, n_q_atoms * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G_acc,     n_q_atoms * rep_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wE_arr,    n_q_atoms * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_E_partial, 0, n_q_atoms * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_G_acc,     0, n_q_atoms * rep_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_wE_arr,    0, n_q_atoms * sizeof(float)));

    // Phase 1: per-query-atom accumulation (E and G_acc)
    {
        int total_q = (int)n_q_atoms;
        int block   = 128;
        int grid    = (total_q + block - 1) / block;
        local_inference_accumulate_kernel<<<grid, block>>>(
            d_X_q, d_X_t, d_alpha_E, d_alpha_desc,
            d_Q_q, d_Q_t, d_N_q, d_N_t,
            d_E_partial, d_G_acc, d_wE_arr,
            inv_2s2, inv_s2, inv_s4,
            nm_q, nm_t, max_atoms_q, max_atoms_t, rep_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Phase 2: reduce E_partial → E_pred (sum over query atoms per molecule)
    {
        int reduce_block = ((max_atoms_q + 31) / 32) * 32;
        if (reduce_block < 32)  reduce_block = 32;
        if (reduce_block > 256) reduce_block = 256;
        size_t smem_r = (size_t)reduce_block * sizeof(float);
        local_energy_reduce_kernel<<<nm_q, reduce_block, smem_r>>>(
            d_E_partial, d_N_q, d_E_pred, nm_q, max_atoms_q);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Phase 3: back-project G_acc → F_pred via dX_q^T @ G_acc per query molecule
    {
        // One block per query molecule; one thread per Cartesian force component
        int ncols_max = 3 * max_atoms_q;
        int bp_block  = ((ncols_max + 31) / 32) * 32;
        if (bp_block < 32)  bp_block = 32;
        if (bp_block > 512) bp_block = 512;
        CUDA_CHECK(cudaMemset(d_F_pred, 0, (long long)naq_q * sizeof(float)));
        local_force_backproject_kernel<<<nm_q, bp_block>>>(
            d_dX_q, d_G_acc, d_N_q, d_offs_q,
            d_F_pred, nm_q, max_atoms_q, rep_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Cleanup
    cudaFree(d_offs_q);
    cudaFree(d_E_partial);
    cudaFree(d_G_acc);
    cudaFree(d_wE_arr);
}

}  // namespace fchl19
}  // namespace kf
