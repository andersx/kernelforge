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
#include <unordered_map>
#include <vector>

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

#define KF_UNUSED_GLOBAL __attribute__((unused))

__global__ static void gather_rows_kernel(
    const float *X_in, float *X_out,
    const int *indices, int R, int rep);

__global__ static void gather_scalars_kernel(
    const float *in, float *out,
    const int *indices, int R);

struct CachedLocalMatvecScratch {
    int *d_offs_q = nullptr;
    size_t offs_q_cap = 0;

    float *d_E_partial = nullptr;
    size_t E_partial_cap = 0;

    float *d_G_acc = nullptr;
    size_t G_acc_cap = 0;

    float *d_wE_arr = nullptr;
    size_t wE_arr_cap = 0;

    float *d_norms_q = nullptr;
    size_t norms_q_cap = 0;

    float *d_C_qt = nullptr;
    size_t C_qt_cap = 0;

    float *d_inner_F = nullptr;
    size_t inner_F_cap = 0;

    float *d_row_expd_iF = nullptr;
    size_t row_expd_iF_cap = 0;

    int *d_query_indices = nullptr;
    size_t query_indices_cap = 0;

    int *d_query_count = nullptr;
    size_t query_count_cap = 0;

    float *d_X_q_elem = nullptr;
    size_t X_q_elem_cap = 0;

    float *d_norms_q_elem = nullptr;
    size_t norms_q_elem_cap = 0;

    float *d_C_elem = nullptr;
    size_t C_elem_cap = 0;

    float *d_inner_F_elem = nullptr;
    size_t inner_F_elem_cap = 0;

    float *d_expd_iF_elem = nullptr;
    size_t expd_iF_elem_cap = 0;

    float *d_G_acc_elem = nullptr;
    size_t G_acc_elem_cap = 0;

    float *d_row_expd_iF_elem = nullptr;
    size_t row_expd_iF_elem_cap = 0;
};

static CachedLocalMatvecScratch s_cached_local_matvec_scratch;

struct CachedLocalTrainingElementGroup {
    int label = 0;
    int count = 0;
    int *d_indices = nullptr;
    float *d_X_t = nullptr;
    float *d_alpha_desc = nullptr;
    float *d_norms_t = nullptr;
    float *d_S_adF = nullptr;
    float *d_alpha_E_t = nullptr;
    float *d_combined_t = nullptr;
};

struct CachedLocalTrainingElementCache {
    const int *d_Q_t = nullptr;
    const int *d_N_t = nullptr;
    const float *d_X_t = nullptr;
    const float *d_alpha_desc = nullptr;
    const float *d_norms_t = nullptr;
    const float *d_S_adF = nullptr;
    const float *d_alpha_E_t = nullptr;
    const float *d_combined_t = nullptr;
    int nm_t = 0;
    int max_atoms_t = 0;
    int rep_size = 0;
    std::vector<CachedLocalTrainingElementGroup> groups;
};

static CachedLocalTrainingElementCache s_cached_local_training_element_cache;

template <typename T>
static void ensure_device_capacity(T **ptr, size_t *cap, size_t count)
{
    if (*cap >= count) return;
    if (*ptr) cudaFree(*ptr);
    CUDA_CHECK(cudaMalloc(ptr, count * sizeof(T)));
    *cap = count;
}

static void ensure_cached_local_matvec_scratch(int nm_q, int max_atoms_q, int rep_size, int nm_t,
                                               int max_atoms_t)
{
    size_t N_q = (size_t)nm_q * (size_t)max_atoms_q;
    size_t N_t = (size_t)nm_t * (size_t)max_atoms_t;
    size_t q_rep = N_q * (size_t)rep_size;
    size_t q_t = N_q * N_t;

    ensure_device_capacity(&s_cached_local_matvec_scratch.d_offs_q,
                           &s_cached_local_matvec_scratch.offs_q_cap,
                           (size_t)nm_q);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_E_partial,
                           &s_cached_local_matvec_scratch.E_partial_cap,
                           N_q);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_G_acc,
                           &s_cached_local_matvec_scratch.G_acc_cap,
                           q_rep);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_wE_arr,
                           &s_cached_local_matvec_scratch.wE_arr_cap,
                           N_q);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_norms_q,
                           &s_cached_local_matvec_scratch.norms_q_cap,
                           N_q);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_C_qt,
                           &s_cached_local_matvec_scratch.C_qt_cap,
                           q_t);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_inner_F,
                           &s_cached_local_matvec_scratch.inner_F_cap,
                           q_t);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_row_expd_iF,
                           &s_cached_local_matvec_scratch.row_expd_iF_cap,
                           N_q);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_query_indices,
                           &s_cached_local_matvec_scratch.query_indices_cap,
                           N_q);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_query_count,
                           &s_cached_local_matvec_scratch.query_count_cap,
                           (size_t)1);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_X_q_elem,
                           &s_cached_local_matvec_scratch.X_q_elem_cap,
                           q_rep);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_norms_q_elem,
                           &s_cached_local_matvec_scratch.norms_q_elem_cap,
                           N_q);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_C_elem,
                           &s_cached_local_matvec_scratch.C_elem_cap,
                           q_t);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_inner_F_elem,
                           &s_cached_local_matvec_scratch.inner_F_elem_cap,
                           q_t);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_expd_iF_elem,
                           &s_cached_local_matvec_scratch.expd_iF_elem_cap,
                           q_t);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_G_acc_elem,
                           &s_cached_local_matvec_scratch.G_acc_elem_cap,
                           q_rep);
    ensure_device_capacity(&s_cached_local_matvec_scratch.d_row_expd_iF_elem,
                           &s_cached_local_matvec_scratch.row_expd_iF_elem_cap,
                           N_q);
}

static void free_cached_local_training_element_cache()
{
    for (CachedLocalTrainingElementGroup &group : s_cached_local_training_element_cache.groups) {
        cudaFree(group.d_indices);
        cudaFree(group.d_X_t);
        cudaFree(group.d_alpha_desc);
        cudaFree(group.d_norms_t);
        cudaFree(group.d_S_adF);
        cudaFree(group.d_alpha_E_t);
        cudaFree(group.d_combined_t);
    }
    s_cached_local_training_element_cache.groups.clear();
    s_cached_local_training_element_cache = CachedLocalTrainingElementCache{};
}

static void ensure_cached_local_training_element_cache(
    const int *d_Q_t,
    const int *d_N_t,
    const float *d_X_t,
    const float *d_alpha_desc,
    const float *d_norms_t,
    const float *d_S_adF,
    const float *d_alpha_E_t,
    const float *d_combined_t,
    int nm_t, int max_atoms_t, int rep_size)
{
    CachedLocalTrainingElementCache &cache = s_cached_local_training_element_cache;
    bool matches =
        cache.d_Q_t == d_Q_t &&
        cache.d_N_t == d_N_t &&
        cache.d_X_t == d_X_t &&
        cache.d_alpha_desc == d_alpha_desc &&
        cache.d_norms_t == d_norms_t &&
        cache.d_S_adF == d_S_adF &&
        cache.d_alpha_E_t == d_alpha_E_t &&
        cache.d_combined_t == d_combined_t &&
        cache.nm_t == nm_t &&
        cache.max_atoms_t == max_atoms_t &&
        cache.rep_size == rep_size;
    if (matches) return;

    free_cached_local_training_element_cache();

    int N_t = nm_t * max_atoms_t;
    int *h_Q = (int*)malloc((size_t)N_t * sizeof(int));
    int *h_N = (int*)malloc((size_t)nm_t * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_Q, d_Q_t, (size_t)N_t * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_N, d_N_t, (size_t)nm_t * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> labels;
    for (int m = 0; m < nm_t; m++) {
        for (int i = 0; i < h_N[m]; i++) {
            int q = h_Q[m * max_atoms_t + i];
            bool found = false;
            for (int label : labels) {
                if (label == q) {
                    found = true;
                    break;
                }
            }
            if (!found) labels.push_back(q);
        }
    }

    for (int label : labels) {
        std::vector<int> atom_indices;
        for (int m = 0; m < nm_t; m++) {
            for (int i = 0; i < h_N[m]; i++) {
                if (h_Q[m * max_atoms_t + i] == label)
                    atom_indices.push_back(m * max_atoms_t + i);
            }
        }
        int R_t = (int)atom_indices.size();
        if (R_t == 0) continue;

        CachedLocalTrainingElementGroup group;
        group.label = label;
        group.count = R_t;
        CUDA_CHECK(cudaMalloc(&group.d_indices, (size_t)R_t * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(group.d_indices, atom_indices.data(), (size_t)R_t * sizeof(int),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&group.d_X_t, (size_t)R_t * rep_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&group.d_alpha_desc, (size_t)R_t * rep_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&group.d_norms_t, (size_t)R_t * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&group.d_S_adF, (size_t)R_t * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&group.d_alpha_E_t, (size_t)R_t * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&group.d_combined_t, (size_t)R_t * rep_size * sizeof(float)));

        gather_rows_kernel<<<(int)(((long long)R_t * rep_size + 255) / 256), 256>>>(
            d_X_t, group.d_X_t, group.d_indices, R_t, rep_size);
        gather_rows_kernel<<<(int)(((long long)R_t * rep_size + 255) / 256), 256>>>(
            d_alpha_desc, group.d_alpha_desc, group.d_indices, R_t, rep_size);
        gather_rows_kernel<<<(int)(((long long)R_t * rep_size + 255) / 256), 256>>>(
            d_combined_t, group.d_combined_t, group.d_indices, R_t, rep_size);
        gather_scalars_kernel<<<(R_t + 255) / 256, 256>>>(
            d_norms_t, group.d_norms_t, group.d_indices, R_t);
        gather_scalars_kernel<<<(R_t + 255) / 256, 256>>>(
            d_S_adF, group.d_S_adF, group.d_indices, R_t);
        gather_scalars_kernel<<<(R_t + 255) / 256, 256>>>(
            d_alpha_E_t, group.d_alpha_E_t, group.d_indices, R_t);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cache.groups.push_back(group);
    }

    free(h_Q);
    free(h_N);

    cache.d_Q_t = d_Q_t;
    cache.d_N_t = d_N_t;
    cache.d_X_t = d_X_t;
    cache.d_alpha_desc = d_alpha_desc;
    cache.d_norms_t = d_norms_t;
    cache.d_S_adF = d_S_adF;
    cache.d_alpha_E_t = d_alpha_E_t;
    cache.d_combined_t = d_combined_t;
    cache.nm_t = nm_t;
    cache.max_atoms_t = max_atoms_t;
    cache.rep_size = rep_size;
}

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


// Maximum number of Cartesian DOFs per molecule supported by the training
// kernel (= 3 * max_atoms_per_molecule).  This bounds the per-thread
// local-memory arrays VA[], VB[], and sff[].  Increase if needed.
#define LOCAL_MAX_NCOLS 96   // covers up to 32 atoms/mol


// ---------------------------------------------------------------------------
// precompute_self_dots_kernel: S[flat_atom, c] = Σ_k dX[atom,k,c] * X[atom,k]
//
// Grid: ceil(N_total / block), Block: 256
// ---------------------------------------------------------------------------
__global__ static void precompute_self_dots_kernel(
    const float *d_X,    // (N_total, rep) row-major
    const float *d_dX,   // (N_total, rep, lda) row-major
    const int   *d_N,    // (nm,) active atom counts
    float       *d_S,    // (N_total, lda) output
    int nm, int max_atoms, int rep_size, int lda)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int mol = flat / max_atoms;
    int atom = flat % max_atoms;
    if (mol >= nm) return;
    if (atom >= d_N[mol]) {
        // Zero inactive atoms
        for (int c = 0; c < lda; c++)
            d_S[flat * lda + c] = 0.0f;
        return;
    }

    const float *x_ptr  = d_X  + (long long)flat * rep_size;
    const float *dx_ptr = d_dX + (long long)flat * rep_size * lda;

    for (int c = 0; c < lda; c++) {
        float s = 0.0f;
        for (int k = 0; k < rep_size; k++)
            s += dx_ptr[k * lda + c] * x_ptr[k];
        d_S[flat * lda + c] = s;
    }
}


__global__ static void build_stage3_pointer_arrays_kernel(
    const float *d_X,
    const float *d_dX,
    float *d_P_ab,
    float *d_P_ba,
    const int *d_pair_a,
    const int *d_pair_b,
    const float **d_Aptr_ab,
    const float **d_Bptr_ab,
    float **d_Cptr_ab,
    const float **d_Aptr_ba,
    const float **d_Bptr_ba,
    float **d_Cptr_ba,
    int rep_size,
    int lda,
    int max_atoms,
    int ncols_max,
    int chunk_start,
    int chunk_sz)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = chunk_sz * max_atoms;
    if (idx >= total) return;

    int local = idx / max_atoms;
    int atom = idx % max_atoms;
    int pair_idx = chunk_start + local;
    int a = d_pair_a[pair_idx];
    int b = d_pair_b[pair_idx];

    d_Aptr_ab[idx] = d_dX + (long long)(a * max_atoms + atom) * rep_size * lda;
    d_Bptr_ab[idx] = d_X + (long long)(b * max_atoms) * rep_size;
    d_Cptr_ab[idx] = d_P_ab + (long long)(local * max_atoms + atom) * ncols_max * max_atoms;

    d_Aptr_ba[idx] = d_dX + (long long)(b * max_atoms + atom) * rep_size * lda;
    d_Bptr_ba[idx] = d_X + (long long)(a * max_atoms) * rep_size;
    d_Cptr_ba[idx] = d_P_ba + (long long)(local * max_atoms + atom) * ncols_max * max_atoms;
}


// ---------------------------------------------------------------------------
// prepare_WA_VB_and_scalars_kernel: prepares WA/VB matrices for rank-1 SGEMM,
// and computes K_EE, K_FE, K_EF from precomputed P_ab, P_ba, S, C_label.
//
// WA[pair, p, c1] = expd * VA[i1,j2,c1] for each atom pair p = i1*nb+j2
// VB[pair, p, c2] = VB[i1,j2,c2]        (zero for non-matching labels)
//
// WA/VB stored col-major per pair for SGEMM: WA(na*nb, ncols_a) at lda=na*nb.
// Element WA[p, c] at c*(na*nb) + p.
//
// Grid: (nm, nm), Block: (128)
// ---------------------------------------------------------------------------
__global__ static void prepare_WA_VB_and_scalars_kernel(
    const float *d_C,       // (N_total, N_total) col-major — C_label
    const float *d_S,       // (N_total, lda) self-dot products
    const float *d_P_ab,    // precomputed cross-terms (host convention)
    const float *d_P_ba,    // precomputed cross-terms (host convention)
    const int   *d_N,
    const int   *d_offs,
    float       *d_K_full,  // (BIG, BIG) — for K_EE, K_FE, K_EF output
    float       *d_WA,      // (chunk_pairs, na*nb, ncols_a) col-major per pair
    float       *d_VB,      // (chunk_pairs, na*nb, ncols_b) col-major per pair
    float sigma2, float inv_s2,
    int nm, int max_atoms, int lda, int N_total, int BIG,
    int na_uniform, int nb_uniform,   // uniform atom counts
    int chunk_start_pair, int chunk_n_pairs)
{
    int local_pair = (int)blockIdx.x;
    if (local_pair < 0 || local_pair >= chunk_n_pairs) return;

    int pair_idx = chunk_start_pair + local_pair;
    int b = (int)(sqrtf(2.0f * (float)pair_idx + 0.25f));
    while ((long long)(b + 1) * (b + 2) / 2 <= pair_idx) ++b;
    while ((long long)b * (b + 1) / 2 > pair_idx) --b;
    int a = pair_idx - b * (b + 1) / 2;
    if (a > b || b >= nm || a >= nm) return;

    int na = d_N[a];
    int nb = d_N[b];
    if (na <= 0 || nb <= 0) return;

    int ncols_a   = 3 * na;
    int ncols_b   = 3 * nb;
    int col_off_a = d_offs[a];
    int row_off_b = d_offs[b];
    int ncM       = 3 * max_atoms;

    // Padded dimensions for buffer addressing (buffers allocated with max_atoms)
    int nap = na_uniform;  // padded atom count (= max_atoms)
    int nbp = nb_uniform;
    int ncols_ap = 3 * nap;
    int ncols_bp = 3 * nbp;
    int nab_p    = nap * nbp;  // padded atom pairs

    // Shared memory for K_EE, K_FE, K_Kjact accumulators
    extern __shared__ float sh[];
    int scalar_acc_sz = 1 + 2 * ncM;

    float *sh_KEE   = sh;
    float *sh_KFE   = sh + 1;
    float *sh_Kjact = sh + 1 + ncM;

    // Zero accumulators
    for (int t = threadIdx.x; t < scalar_acc_sz; t += blockDim.x)
        sh[t] = 0.0f;
    __syncthreads();

    // Buffer bases use PADDED dimensions (matching host allocation)
    long long P_ab_per_pair = (long long)nap * ncols_ap * nbp;
    long long P_ba_per_pair = (long long)nbp * ncols_bp * nap;
    long long P_ab_base = (long long)local_pair * P_ab_per_pair;
    long long P_ba_base = (long long)local_pair * P_ba_per_pair;

    // WA/VB output bases — col-major (nab_p, ncols_ap/bp) per pair
    long long WA_per_pair = (long long)nab_p * ncols_ap;
    long long VB_per_pair_sz = (long long)nab_p * ncols_bp;
    long long WA_base = (long long)local_pair * WA_per_pair;
    long long VB_base = (long long)local_pair * VB_per_pair_sz;

    int flat_a_base = a * max_atoms;
    int flat_b_base = b * max_atoms;

    // Iterate over ALL padded atom pairs (nab_p = nap * nbp).
    // Inactive atoms (i1 >= na or j2 >= nb) get C_qt=0 → zero WA/VB.
    for (int pair = (int)threadIdx.x; pair < nab_p; pair += (int)blockDim.x) {
        int i1 = pair / nbp;
        int j2 = pair % nbp;

        int flat_i = flat_a_base + i1;
        int flat_j = flat_b_base + j2;
        float expdiag = d_C[flat_i + (long long)flat_j * N_total];

        if (expdiag == 0.0f) {
            // Zero WA/VB for non-matching or inactive pairs
            for (int c = 0; c < ncols_ap; c++)
                d_WA[WA_base + (long long)c * nab_p + pair] = 0.0f;
            for (int c = 0; c < ncols_bp; c++)
                d_VB[VB_base + (long long)c * nab_p + pair] = 0.0f;
            continue;
        }

        float exp_base = expdiag * sigma2;
        float expd     = -expdiag * inv_s2;

        // P_ab/P_ba use PADDED strides (ncols_ap, nbp, nap, ncols_bp)
        // P_ba batch i1: output (ncols_bp, nap) col-major at (local*nap + i1)*ncols_bp*nap
        // Element [c, j2] at j2*ncols_bp + c
        long long pba_for_va = P_ba_base + (long long)i1 * ncols_bp * nap + (long long)j2 * ncols_bp;
        // P_ab batch j2: output (ncols_ap, nbp) col-major at (local*nbp + j2)*ncols_ap*nbp
        // Element [c, i1] at i1*ncols_ap + c
        long long pab_for_vb = P_ab_base + (long long)j2 * ncols_ap * nbp + (long long)i1 * ncols_ap;

        // Write WA (col-major: element [pair, c] at c*nab_p + pair)
        for (int c = 0; c < ncols_ap; c++) {
            float va = d_S[flat_i * lda + c] - d_P_ba[pba_for_va + c];
            d_WA[WA_base + (long long)c * nab_p + pair] = expd * va;
        }
        for (int c = 0; c < ncols_bp; c++) {
            float vb = d_P_ab[pab_for_vb + c] - d_S[flat_j * lda + c];
            d_VB[VB_base + (long long)c * nab_p + pair] = vb;
        }

        // K_EE, K_FE, K_EF accumulation (use actual ncols_a/ncols_b for correct active range)
        atomicAdd(sh_KEE, exp_base);
        for (int c = 0; c < ncols_a; c++) {
            float va = d_S[flat_i * lda + c] - d_P_ba[pba_for_va + c];
            atomicAdd(sh_KFE + c, -expdiag * va);
        }
        for (int c = 0; c < ncols_b; c++) {
            float vb = d_P_ab[pab_for_vb + c] - d_S[flat_j * lda + c];
            atomicAdd(sh_Kjact + c, expdiag * vb);
        }
    }

    __syncthreads();

    // Write K_EE
    if (threadIdx.x == 0) {
        float kee = sh_KEE[0];
        d_K_full[(long long)a * BIG + b] = kee;
        if (a != b)
            d_K_full[(long long)b * BIG + a] = kee;
    }

    // Write K_FE
    for (int c1 = (int)threadIdx.x; c1 < ncols_a; c1 += (int)blockDim.x) {
        float v = sh_KFE[c1];
        d_K_full[(long long)(nm + col_off_a + c1) * BIG + b] = v;
        d_K_full[(long long)b * BIG + (nm + col_off_a + c1)] = v;
    }

    // Write K_jact (off-diagonal only)
    for (int c2 = (int)threadIdx.x; c2 < ncols_b; c2 += (int)blockDim.x) {
        if (a != b) {
            float v = sh_Kjact[c2];
            d_K_full[(long long)a * BIG + (nm + row_off_b + c2)] = v;
            d_K_full[(long long)(nm + row_off_b + c2) * BIG + a] = v;
        }
    }
}


// ---------------------------------------------------------------------------
// assemble_from_precomputed_kernel: K_EE, K_FE, K_EF, rank-1 K_FF correction
// using precomputed P_ab, P_ba, S arrays. No inner k-loop needed.
//
// P_ab layout (col-major per batch):
//   P_ab for pair p, atom i1: stored at P_ab_buf[(p*na + i1)*ncols_a*nb]
//   Element P_ab[i1,j2,c1] at buf_offset + j2*ncols_a + c1
//
// P_ba layout (col-major per batch):
//   P_ba for pair p, atom j2: stored at P_ba_buf[(p*nb + j2)*ncols_b*na]
//   Element P_ba[j2,i1,c2] at buf_offset + i1*ncols_b + c2
//
// Grid: (nm, nm), Block: (128)
// ---------------------------------------------------------------------------
__global__ static KF_UNUSED_GLOBAL void assemble_from_precomputed_kernel(
    const float *d_C,       // (N_total, N_total) col-major — C_label
    const float *d_S,       // (N_total, lda) self-dot products
    const float *d_P_ab,    // precomputed cross-terms for VA
    const float *d_P_ba,    // precomputed cross-terms for VB
    const int   *d_N,
    const int   *d_offs,
    float       *d_K_full,
    float sigma2, float inv_s2,
    int nm, int max_atoms, int lda, int N_total, int BIG,
    int chunk_start_pair,   // first pair index in this chunk
    int chunk_n_pairs)      // number of pairs in this chunk
{
    int b = blockIdx.x;
    int a = blockIdx.y;
    if (a > b || b >= nm || a >= nm) return;

    // Compute linear pair index (lower-triangular: row*(row+1)/2 + col where row >= col)
    // Since a <= b in this kernel, b is the "row" and a is the "col"
    int pair_idx = b * (b + 1) / 2 + a;
    int local_pair = pair_idx - chunk_start_pair;
    if (local_pair < 0 || local_pair >= chunk_n_pairs) return;

    int na = d_N[a];
    int nb = d_N[b];
    if (na <= 0 || nb <= 0) return;

    int ncols_a   = 3 * na;
    int ncols_b   = 3 * nb;
    int col_off_a = d_offs[a];
    int row_off_b = d_offs[b];
    int ncM       = 3 * max_atoms;

    // Shared memory for K_EE, K_FE, K_Kjact accumulators
    extern __shared__ float sh[];
    int scalar_acc_sz = 1 + 2 * ncM;

    float *sh_KEE   = sh;
    float *sh_KFE   = sh + 1;
    float *sh_Kjact = sh + 1 + ncM;

    // Zero accumulators
    for (int t = threadIdx.x; t < scalar_acc_sz; t += blockDim.x)
        sh[t] = 0.0f;
    __syncthreads();

    // Pointers into precomputed buffers for this pair
    long long P_ab_base = (long long)local_pair * na * ncols_a * nb;
    long long P_ba_base = (long long)local_pair * nb * ncols_b * na;

    int flat_a_base = a * max_atoms;
    int flat_b_base = b * max_atoms;

    int total_pairs = na * nb;
    for (int pair = (int)threadIdx.x; pair < total_pairs; pair += (int)blockDim.x) {
        int i1 = pair / nb;
        int j2 = pair % nb;

        // Read expdiag from C_label (zero for non-matching labels)
        int flat_i = flat_a_base + i1;
        int flat_j = flat_b_base + j2;
        float expdiag = d_C[flat_i + (long long)flat_j * N_total];
        if (expdiag == 0.0f) continue;

        float exp_base = expdiag * sigma2;
        float expd     = -expdiag * inv_s2;

        // Read VA = S_a - cross_term, VB = cross_term - S_b from precomputed arrays.
        //
        // Convention: host enumerates pairs with host_a >= host_b (lower-triangular),
        // but the kernel has kernel_a <= kernel_b (upper-triangular).
        // So host_a = kernel_b, host_b = kernel_a.
        //
        // P_ab was computed with dX of host_a (= kernel_b) and X of host_b (= kernel_a).
        // P_ba was computed with dX of host_b (= kernel_a) and X of host_a (= kernel_b).
        //
        // VA needs dX[kernel_a, i1]^T @ X[kernel_b, j2] = P_ba(batch=i1, col=j2)
        // VB needs dX[kernel_b, j2]^T @ X[kernel_a, i1] = P_ab(batch=j2, col=i1)
        long long pba_for_va = P_ba_base + (long long)i1 * ncols_b * na + (long long)j2 * ncols_b;
        long long pab_for_vb = P_ab_base + (long long)j2 * ncols_a * nb + (long long)i1 * ncols_a;

        float VA[LOCAL_MAX_NCOLS];
        float VB[LOCAL_MAX_NCOLS];
        for (int c = 0; c < ncols_a; c++)
            VA[c] = d_S[flat_i * lda + c] - d_P_ba[pba_for_va + c];
        for (int c = 0; c < ncols_b; c++)
            VB[c] = d_P_ab[pab_for_vb + c] - d_S[flat_j * lda + c];

        atomicAdd(sh_KEE, exp_base);

        for (int c1 = 0; c1 < ncols_a; c1++)
            atomicAdd(sh_KFE + c1, -expdiag * VA[c1]);

        for (int c2 = 0; c2 < ncols_b; c2++)
            atomicAdd(sh_Kjact + c2, expdiag * VB[c2]);

        // Rank-1 correction to K_FF
        for (int c1 = 0; c1 < ncols_a; c1++) {
            float ev = expd * VA[c1];
            long long row = (long long)(nm + col_off_a + c1) * BIG;
            for (int c2 = 0; c2 < ncols_b; c2++) {
                float correction = ev * VB[c2];
                atomicAdd(&d_K_full[row + (nm + row_off_b + c2)], correction);
                if (a != b) {
                    atomicAdd(&d_K_full[(long long)(nm + row_off_b + c2) * BIG + (nm + col_off_a + c1)], correction);
                }
            }
        }
    }

    __syncthreads();

    // Write K_EE
    if (threadIdx.x == 0) {
        float kee = sh_KEE[0];
        d_K_full[(long long)a * BIG + b] = kee;
        if (a != b)
            d_K_full[(long long)b * BIG + a] = kee;
    }

    // Write K_FE
    for (int c1 = (int)threadIdx.x; c1 < ncols_a; c1 += (int)blockDim.x) {
        float v = sh_KFE[c1];
        d_K_full[(long long)(nm + col_off_a + c1) * BIG + b] = v;
        d_K_full[(long long)b * BIG + (nm + col_off_a + c1)] = v;
    }

    // Write K_jact (off-diagonal only)
    for (int c2 = (int)threadIdx.x; c2 < ncols_b; c2 += (int)blockDim.x) {
        if (a != b) {
            float v = sh_Kjact[c2];
            d_K_full[(long long)a * BIG + (nm + row_off_b + c2)] = v;
            d_K_full[(long long)(nm + row_off_b + c2) * BIG + a] = v;
        }
    }
}


// ============================================================================
// Batched SGEMM kernels for full-symmetric kernel build
// ============================================================================

// ---------------------------------------------------------------------------
// mirror_lower_to_upper_f_kernel: copy lower triangle to upper for a col-major
// N×N float matrix.  A[col, row] = A[row, col] for row > col.
// ---------------------------------------------------------------------------
__global__ static KF_UNUSED_GLOBAL void mirror_lower_to_upper_f_kernel(float *A, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= N || row >= N || row <= col) return;
    // row > col: lower triangle element → copy to upper
    A[col + (long long)row * N] = A[row + (long long)col * N];
}


// ---------------------------------------------------------------------------
// compute_sqnorms_kernel: norms[i] = ||X[i,:]||²
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// build_C_label_kernel: in-place on G (col-major, N×N)
//   G[i,j] = exp(-||x_i-x_j||²/(2σ²)) / σ²    if Q[i]==Q[j]
//          = 0                                  otherwise
// G already contains -2*X^T*X; add norms then exp.
// ---------------------------------------------------------------------------
__global__ static void build_C_label_kernel(
    float *G, const float *norms,
    const int *Q, const int *N_arr,
    float inv_s2, int N_total, int max_atoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N_total || j >= N_total) return;

    // Check both atoms are active
    int mol_i = i / max_atoms, atom_i = i % max_atoms;
    int mol_j = j / max_atoms, atom_j = j % max_atoms;
    int na_i = N_arr[mol_i], na_j = N_arr[mol_j];

    long long idx = i + (long long)j * N_total;

    if (atom_i >= na_i || atom_j >= na_j || Q[i] != Q[j]) {
        G[idx] = 0.0f;
        return;
    }

    float dist2 = G[idx] + norms[i] + norms[j];

    G[idx] = expf(-0.5f * inv_s2 * dist2) * inv_s2;
}

// build_C_label_lower_kernel: same as build_C_label_kernel but only processes
// the lower triangle (i >= j).  Used after cublasSsyrk which only fills lower.
__global__ static KF_UNUSED_GLOBAL void build_C_label_lower_kernel(
    float *G, const float *norms,
    const int *Q, const int *N_arr,
    float inv_s2, int N_total, int max_atoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N_total || j >= N_total || i < j) return;  // lower triangle only

    int mol_i = i / max_atoms, atom_i = i % max_atoms;
    int mol_j = j / max_atoms, atom_j = j % max_atoms;
    int na_i = N_arr[mol_i], na_j = N_arr[mol_j];

    long long idx = i + (long long)j * N_total;

    if (atom_i >= na_i || atom_j >= na_j || Q[i] != Q[j]) {
        G[idx] = 0.0f;
        return;
    }

    float dist2 = G[idx] + norms[i] + norms[j];
    G[idx] = expf(-0.5f * inv_s2 * dist2) * inv_s2;
}

// ---------------------------------------------------------------------------
// scatter_K_FF_kernel: write K_FF blocks from per-pair lda×lda buffers
// to the global K_full matrix, including symmetric transpose.
//
// Grid: (n_mol_pairs,), Block: (256)
// d_K_FF_pairs: (n_pairs, ncols_max, ncols_max) row-major per pair
// ---------------------------------------------------------------------------
__global__ static void scatter_K_FF_kernel(
    const float *d_K_FF_pairs,   // (n_pairs, ncols_max, ncols_max) row-major
    const int   *d_N,
    const int   *d_offs,
    float       *d_K_full,
    int nm, int max_atoms, int ncols_max, int BIG)
{
    int pair_idx = blockIdx.x;

    // Decode lower-triangular index (a >= b)
    int a = (int)(sqrtf(2.0f * (float)pair_idx + 0.25f));
    while ((long long)(a + 1) * (a + 2) / 2 <= pair_idx) ++a;
    while ((long long)a       * (a + 1) / 2 >  pair_idx) --a;
    int b = pair_idx - a * (a + 1) / 2;
    if (a >= nm) return;

    int na = d_N[a], nb = d_N[b];
    int ncols_a = 3 * na, ncols_b = 3 * nb;
    int col_off_a = d_offs[a], row_off_b = d_offs[b];

    const float *kff = d_K_FF_pairs + (long long)pair_idx * ncols_max * ncols_max;

    for (int t = threadIdx.x; t < ncols_a * ncols_b; t += blockDim.x) {
        int c1 = t / ncols_b;
        int c2 = t % ncols_b;
        float v = kff[c1 * ncols_max + c2];
        d_K_full[(long long)(nm + col_off_a + c1) * BIG + (nm + row_off_b + c2)] = v;
        if (a != b) {
            d_K_full[(long long)(nm + row_off_b + c2) * BIG + (nm + col_off_a + c1)] = v;
        } else {
            if (c1 != c2)
                d_K_full[(long long)(nm + col_off_a + c2) * BIG + (nm + col_off_a + c1)] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// assemble_scalar_kernel: computes K_EE, K_FE, K_EF, and rank-1 correction
// to K_FF (the expd * VA * VB term) per molecule pair.
//
// Reads exp scalars directly from the precomputed C_label matrix (Stage 1)
// instead of recomputing l2 + expf(). C_label[i,j] = exp(-l2/(2σ²))/σ²
// for matching labels, 0 otherwise.
//
// Also adds the rank-1 correction to the precomputed K_FF (from SGEMM).
//
// Grid: (nm, nm), Block: (128)
// ---------------------------------------------------------------------------
__global__ static KF_UNUSED_GLOBAL void assemble_scalar_kernel(
    const float *d_X,       // (nm, max_atoms, rep)
    const float *d_dX,      // (nm, max_atoms, rep, 3*max_atoms)
    const float *d_C,       // (N_total, N_total) col-major — C_label from Stage 1
    const int   *d_N,       // (nm,)
    const int   *d_offs,    // (nm,) Cartesian offsets
    float       *d_K_full,  // (BIG, BIG) row-major — K_FF already has static part
    float sigma2,           // σ²
    float inv_s2,           // 1/σ²
    int nm, int max_atoms, int rep_size, int N_total, int BIG)
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
    int ncM       = 3 * max_atoms;

    // Shared memory layout:
    //   sh_KEE   (1 float)
    //   sh_KFE   (ncM floats)
    //   sh_Kjact (ncM floats)
    //   sh_Xa    (na * rep_size)  — cached X for molecule a
    //   sh_Xb    (nb * rep_size)  — cached X for molecule b
    extern __shared__ float sh[];
    int scalar_acc_sz = 1 + 2 * ncM;

    float *sh_KEE   = sh;
    float *sh_KFE   = sh + 1;
    float *sh_Kjact = sh + 1 + ncM;
    float *sh_Xa    = sh + scalar_acc_sz;
    float *sh_Xb    = sh + scalar_acc_sz + na * rep_size;

    // Zero accumulators
    for (int t = threadIdx.x; t < scalar_acc_sz; t += blockDim.x)
        sh[t] = 0.0f;

    // Load X into shared memory
    for (int t = threadIdx.x; t < na * rep_size; t += blockDim.x)
        sh_Xa[t] = d_X[(long long)(a * max_atoms + t / rep_size) * rep_size + t % rep_size];
    for (int t = threadIdx.x; t < nb * rep_size; t += blockDim.x)
        sh_Xb[t] = d_X[(long long)(b * max_atoms + t / rep_size) * rep_size + t % rep_size];

    __syncthreads();

    // Per-atom-pair: compute VA, VB; read scalars from C_label
    int total_pairs = na * nb;
    for (int pair = (int)threadIdx.x; pair < total_pairs; pair += (int)blockDim.x) {
        int i1 = pair / nb;
        int j2 = pair % nb;

        // Read expdiag directly from C_label (zero for non-matching labels)
        int flat_i = a * max_atoms + i1;
        int flat_j = b * max_atoms + j2;
        float expdiag = d_C[flat_i + (long long)flat_j * N_total];
        if (expdiag == 0.0f) continue;

        float exp_base = expdiag * sigma2;
        float expd     = -expdiag * inv_s2;

        const float *dxa = d_dX + (long long)(a * max_atoms + i1) * rep_size * lda;
        const float *dxb = d_dX + (long long)(b * max_atoms + j2) * rep_size * lda;

        float VA[LOCAL_MAX_NCOLS];
        float VB[LOCAL_MAX_NCOLS];
        for (int c = 0; c < ncols_a; c++) VA[c] = 0.0f;
        for (int c = 0; c < ncols_b; c++) VB[c] = 0.0f;

        const float *sh_xa = sh_Xa + i1 * rep_size;
        const float *sh_xb = sh_Xb + j2 * rep_size;

        for (int k = 0; k < rep_size; k++) {
            float dk = sh_xa[k] - sh_xb[k];

            const float *dxa_k = dxa + (long long)k * lda;
            const float *dxb_k = dxb + (long long)k * lda;

            for (int c1 = 0; c1 < ncols_a; c1++)
                VA[c1] += dxa_k[c1] * dk;
            for (int c2 = 0; c2 < ncols_b; c2++)
                VB[c2] += dxb_k[c2] * dk;
        }

        atomicAdd(sh_KEE, exp_base);

        for (int c1 = 0; c1 < ncols_a; c1++)
            atomicAdd(sh_KFE + c1, -expdiag * VA[c1]);

        for (int c2 = 0; c2 < ncols_b; c2++)
            atomicAdd(sh_Kjact + c2, expdiag * VB[c2]);

        // Rank-1 correction to K_FF (add expd * VA[c1] * VB[c2])
        // For a==b: only write (c1,c2) once — the symmetric (c2,c1) entry will
        // be naturally written when the loop visits pair (c2,c1).
        for (int c1 = 0; c1 < ncols_a; c1++) {
            float ev = expd * VA[c1];
            long long row = (long long)(nm + col_off_a + c1) * BIG;
            for (int c2 = 0; c2 < ncols_b; c2++) {
                float correction = ev * VB[c2];
                atomicAdd(&d_K_full[row + (nm + row_off_b + c2)], correction);
                if (a != b) {
                    atomicAdd(&d_K_full[(long long)(nm + row_off_b + c2) * BIG + (nm + col_off_a + c1)], correction);
                }
            }
        }
    }

    __syncthreads();

    // Write K_EE
    if (threadIdx.x == 0) {
        float kee = sh_KEE[0];
        d_K_full[(long long)a * BIG + b] = kee;
        if (a != b)
            d_K_full[(long long)b * BIG + a] = kee;
    }

    // Write K_FE
    for (int c1 = (int)threadIdx.x; c1 < ncols_a; c1 += (int)blockDim.x) {
        float v = sh_KFE[c1];
        d_K_full[(long long)(nm + col_off_a + c1) * BIG + b] = v;
        d_K_full[(long long)b * BIG + (nm + col_off_a + c1)] = v;
    }

    // Write K_jact (off-diagonal only)
    for (int c2 = (int)threadIdx.x; c2 < ncols_b; c2 += (int)blockDim.x) {
        if (a != b) {
            float v = sh_Kjact[c2];
            d_K_full[(long long)a * BIG + (nm + row_off_b + c2)] = v;
            d_K_full[(long long)(nm + row_off_b + c2) * BIG + a] = v;
        }
    }
}

// (assemble_full_symm_local_kernel removed — replaced by batched SGEMM + assemble_scalar_kernel)


// ---------------------------------------------------------------------------
// build_KEE_from_C_label_kernel: reduce C_label (N×N) to K_EE (nm×nm) by
// summing max_atoms×max_atoms blocks, scaling by sigma².
//
// K_EE[a,b] = sigma2 * Σ_{i1=0}^{N[a]-1} Σ_{j2=0}^{N[b]-1} C_label[a*max+i1, b*max+j2]
//
// Reads ONLY the lower triangle of C (row >= col in col-major), swapping
// indices when needed.  This allows C to come from cublasSsyrk.
//
// Grid: (nm, nm), Block: (256)
// ---------------------------------------------------------------------------
__global__ static KF_UNUSED_GLOBAL void build_KEE_from_C_label_kernel(
    const float *d_C,      // (N_total, N_total) col-major — lower triangle valid
    const int   *d_N,      // (nm,) active atom counts
    float       *d_KEE,    // (nm, nm) col-major output
    float sigma2,
    int nm, int max_atoms, int N_total)
{
    int a = blockIdx.x;
    int b = blockIdx.y;
    if (a >= nm || b >= nm) return;

    int na = d_N[a];
    int nb = d_N[b];

    float local_sum = 0.0f;
    int total = na * nb;
    for (int t = (int)threadIdx.x; t < total; t += (int)blockDim.x) {
        int i1 = t / nb;
        int j2 = t % nb;
        int flat_i = a * max_atoms + i1;
        int flat_j = b * max_atoms + j2;
        local_sum += d_C[flat_i + (long long)flat_j * N_total];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);

    // Block reduction via shared memory (first lane of each warp)
    __shared__ float sh[8];  // up to 256/32 = 8 warps
    int warp_id = (int)threadIdx.x / 32;
    int lane = (int)threadIdx.x & 31;
    if (lane == 0) sh[warp_id] = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float s = 0.0f;
        int n_warps = ((int)blockDim.x + 31) / 32;
        for (int w = 0; w < n_warps; w++) s += sh[w];
        d_KEE[a + (long long)b * nm] = s * sigma2;
    }
}


// ---------------------------------------------------------------------------
// build_KEE_rect_kernel: reduce rectangular C_qt (N_q × N_t) to K_EE (nm_q × nm_t)
// by summing atom-pair blocks, scaling by sigma².
//
// K_EE[a,b] = sigma2 * Σ_{i1,j2} C_qt[a*max_q+i1, b*max_t+j2]
//
// Grid: (nm_q, nm_t), Block: (256)
// ---------------------------------------------------------------------------
__global__ static void build_KEE_rect_kernel(
    const float *d_C,      // (N_q, N_t) col-major
    const int   *d_N_q,
    const int   *d_N_t,
    float       *d_KEE,    // (nm_q, nm_t) col-major output
    float sigma2,
    int nm_q, int nm_t, int max_atoms_q, int max_atoms_t, int N_q)
{
    int a = blockIdx.x;
    int b = blockIdx.y;
    if (a >= nm_q || b >= nm_t) return;

    int na = d_N_q[a];
    int nb = d_N_t[b];

    float local_sum = 0.0f;
    int total = na * nb;
    for (int t = (int)threadIdx.x; t < total; t += (int)blockDim.x) {
        int i1 = t / nb;
        int j2 = t % nb;
        int flat_i = a * max_atoms_q + i1;
        int flat_j = b * max_atoms_t + j2;
        local_sum += d_C[flat_i + (long long)flat_j * N_q];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float sh[8];
    int warp_id = (int)threadIdx.x / 32;
    int lane = (int)threadIdx.x & 31;
    if (lane == 0) sh[warp_id] = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float s = 0.0f;
        int n_warps = ((int)blockDim.x + 31) / 32;
        for (int w = 0; w < n_warps; w++) s += sh[w];
        d_KEE[(long long)a * nm_t + b] = s * sigma2;  // row-major for PyTorch
    }
}


// ---------------------------------------------------------------------------
// gather_rows_kernel: X_out[r, :] = X_in[indices[r], :]
// Grid: ceil(R * rep / 256)
// ---------------------------------------------------------------------------
__global__ static void gather_rows_kernel(
    const float *X_in, float *X_out,
    const int *indices, int R, int rep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * rep) return;
    int r = idx / rep;
    int k = idx % rep;
    X_out[idx] = X_in[(long long)indices[r] * rep + k];
}

// gather_scalars_kernel: out[r] = in[indices[r]]
__global__ static void gather_scalars_kernel(
    const float *in, float *out,
    const int *indices, int R)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    out[r] = in[indices[r]];
}

// ---------------------------------------------------------------------------
// apply_norms_exp_kernel: in-place on G (col-major, ni × nj)
// G[i,j] = exp(-0.5 * inv_s2 * (G[i,j] + norms_i[i] + norms_j[j])) * inv_s2
// No label check — used for per-element C tiles where all atoms match.
// ---------------------------------------------------------------------------
__global__ static void apply_norms_exp_kernel(
    float *G, const float *norms_i, const float *norms_j,
    float inv_s2, int ni, int nj, int lower_only)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ni || j >= nj) return;
    long long idx = i + (long long)j * ni;
    if (lower_only && i < j) {
        G[idx] = 0.0f;
        return;
    }
    float dist2 = G[idx] + norms_i[i] + norms_j[j];
    G[idx] = expf(-0.5f * inv_s2 * dist2) * inv_s2;
}

__global__ static void gather_active_query_indices_kernel(
    const int *d_Q_q,
    const int *d_N_q,
    int *d_indices,
    int *d_count,
    int label,
    int nm_q,
    int max_atoms_q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_q = nm_q * max_atoms_q;
    if (idx >= N_q) return;

    int mol = idx / max_atoms_q;
    int atom = idx % max_atoms_q;
    if (atom >= d_N_q[mol]) return;
    if (d_Q_q[idx] != label) return;

    int out = atomicAdd(d_count, 1);
    d_indices[out] = idx;
}

__global__ static void scatter_rect_tile_kernel(
    const float *src,
    float *dst,
    const int *row_indices,
    const int *col_indices,
    int row_count,
    int col_count,
    int ld_src,
    int ld_dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= row_count || j >= col_count) return;

    int row = row_indices[i];
    int col = col_indices[j];
    dst[row + (long long)col * ld_dst] = src[i + (long long)j * ld_src];
}

__global__ static void scatter_g_acc_tile_kernel(
    const float *src,
    float *dst,
    const int *row_indices,
    int row_count,
    int rep_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= row_count * rep_size) return;

    int i = idx / rep_size;
    int k = idx % rep_size;
    int row = row_indices[i];
    dst[(long long)row * rep_size + k] += src[idx];
}

__global__ static void scatter_row_sums_kernel(
    const float *src,
    float *dst,
    const int *row_indices,
    int row_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count) return;
    dst[row_indices[i]] += src[i];
}


// ---------------------------------------------------------------------------
// scatter_C_to_KEE_kernel: atomically add C_elem values to K_EE.
// C_elem[i,j] (col-major, ni×nj) contributes to K_EE[mol_i[i], mol_j[j]].
// K_EE is row-major (nm × nm).
// ---------------------------------------------------------------------------
__global__ static void scatter_C_to_KEE_kernel(
    const float *d_C,          // (ni, nj) col-major
    const int   *d_mol_i,      // (ni,) molecule index for each row
    const int   *d_mol_j,      // (nj,) molecule index for each col
    float       *d_KEE,        // (nm, nm) row-major output
    float sigma2,
    int ni, int nj, int nm,
    int do_mirror)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ni || j >= nj) return;

    float val = d_C[i + (long long)j * ni] * sigma2;
    if (val == 0.0f) return;
    int ma = d_mol_i[i];
    int mb = d_mol_j[j];
    atomicAdd(&d_KEE[(long long)ma * nm + mb], val);
    if (do_mirror)
        atomicAdd(&d_KEE[(long long)mb * nm + ma], val);
}


// ---------------------------------------------------------------------------
// reduce_C_tile_to_KEE_kernel: reduce a C_tile (Na × Nb col-major) into the
// full K_EE matrix at position (ta, tb), with proper stride = nm.
// Also mirrors to (tb, ta) if do_mirror is set.
//
// Grid: (na_mols, nb_mols), Block: (256)
// ---------------------------------------------------------------------------
__global__ static KF_UNUSED_GLOBAL void reduce_C_tile_to_KEE_kernel(
    const float *d_C,         // (Na, Nb) col-major tile
    const int   *d_N_a,       // (na_mols,) — N array offset to tile_a start
    const int   *d_N_b,       // (nb_mols,) — N array offset to tile_b start
    float       *d_KEE,       // (nm, nm) row-major — full output matrix
    float sigma2,
    int na_mols, int nb_mols,
    int max_atoms_a, int max_atoms_b,
    int Na,                   // = na_mols * max_atoms_a (leading dim of C_tile)
    int ta, int tb,           // molecule offsets into full K_EE
    int nm,                   // full K_EE dimension (stride)
    int do_mirror)            // 1 if ta != tb, 0 otherwise
{
    int a = blockIdx.x;
    int b = blockIdx.y;
    if (a >= na_mols || b >= nb_mols) return;

    int na = d_N_a[a];
    int nb = d_N_b[b];

    float local_sum = 0.0f;
    int total = na * nb;
    for (int t = (int)threadIdx.x; t < total; t += (int)blockDim.x) {
        int i1 = t / nb;
        int j2 = t % nb;
        int flat_i = a * max_atoms_a + i1;
        int flat_j = b * max_atoms_b + j2;
        local_sum += d_C[flat_i + (long long)flat_j * Na];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float sh[8];
    int warp_id = (int)threadIdx.x / 32;
    int lane = (int)threadIdx.x & 31;
    if (lane == 0) sh[warp_id] = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float s = 0.0f;
        int n_warps = ((int)blockDim.x + 31) / 32;
        for (int w = 0; w < n_warps; w++) s += sh[w];
        float val = s * sigma2;
        d_KEE[(long long)(ta + a) * nm + (tb + b)] = val;
        if (do_mirror)
            d_KEE[(long long)(tb + b) * nm + (ta + a)] = val;
    }
}


// ---------------------------------------------------------------------------
// build_C_qt_kernel: build rectangular query×train Gaussian kernel with label
// screening.  G already contains -2*X_q^T*X_t from cuBLAS SGEMM.
// G is col-major (N_q, N_t): G[i + j*N_q].
// ---------------------------------------------------------------------------
__global__ static void build_C_qt_kernel(
    float *G, const float *norms_q, const float *norms_t,
    const int *Q_q, const int *Q_t,
    const int *N_q_arr, const int *N_t_arr,
    float inv_s2, int N_q, int N_t, int max_atoms_q, int max_atoms_t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N_q || j >= N_t) return;

    int mol_i = i / max_atoms_q, atom_i = i % max_atoms_q;
    int mol_j = j / max_atoms_t, atom_j = j % max_atoms_t;

    long long idx = i + (long long)j * N_q;

    if (atom_i >= N_q_arr[mol_i] || atom_j >= N_t_arr[mol_j] || Q_q[i] != Q_t[j]) {
        G[idx] = 0.0f;
        return;
    }

    float dist2 = G[idx] + norms_q[i] + norms_t[j];
    G[idx] = expf(-0.5f * inv_s2 * dist2) * inv_s2;
}


// ---------------------------------------------------------------------------
// inference_scalars_kernel: compute E_partial, wE, row_sum_expd_iF, and
// build the expd_iF matrix from C_qt and inner_F.
//
// Also computes inner_F in-place: inner_F_raw[q,t] -= S_adF[t]
// (S_adF is the self-dot x_t · adF[t], subtracted from X_q @ adF^T)
//
// Grid: ceil(N_q * N_t / block)
// ---------------------------------------------------------------------------
__global__ static void inference_build_expd_iF_kernel(
    const float *d_C_qt,      // (N_q, N_t) col-major — C_qt[i,j] = expdiag
    float       *d_inner_F,   // (N_q, N_t) col-major — on entry: X_q @ adF^T, on exit: corrected
    const float *d_S_adF,     // (N_t,) self-dots x_t · adF[t]
    float       *d_expd_iF,   // (N_q, N_t) col-major output — expd * inner_F
    float       *d_row_expd_iF, // (N_q,) output accumulator
    float inv_s2,
    int N_q, int N_t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_q * N_t) return;

    int q = idx % N_q;
    int j = idx / N_q;  // train atom (col-major)

    // Correct inner_F: subtract S_adF[t]
    float c_qt = d_C_qt[idx];
    float iF_raw = d_inner_F[idx] - d_S_adF[j];
    d_inner_F[idx] = iF_raw;

    // expd_iF = -(C_qt / σ²) * inner_F
    float expd_iF = -c_qt * inv_s2 * iF_raw;
    d_expd_iF[idx] = expd_iF;
    atomicAdd(&d_row_expd_iF[q], expd_iF);
}


// ---------------------------------------------------------------------------
// inference_E_and_diag_kernel: compute E_partial[q] and wE[q] from
// C_qt, inner_F, alpha_E.
//
// E_partial[q] = Σ_t C_qt[q,t] * σ² * alpha_E_expand[t]  +  Σ_t C_qt[q,t] * inner_F[q,t]
// wE[q]        = Σ_t C_qt[q,t] * alpha_E_expand[t]
// row_expd_iF[q] = Σ_t expd_iF[q,t]
//
// One thread per query atom.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// inference_E_and_diag_kernel: tile over both query atoms and training atoms.
//
// The previous version launched only ceil(N_q / 32) blocks, which collapses to
// a single block for the common single-query inference case (e.g. aspirin with
// N_q ~= 21). This tiled version launches a 2-D grid:
//   blockIdx.x -> query-atom tile
//   blockIdx.y -> training-atom tile
//
// Each block reduces one (query-tile, training-tile) slice into registers and
// shared memory, then atomically accumulates the partial sums into d_E_partial
// and d_wE. row_sum_expd_iF is accumulated earlier in
// inference_build_expd_iF_kernel.
// ---------------------------------------------------------------------------
#define TILE_Q 32
#define TILE_T 256
__global__ static void inference_E_and_diag_kernel(
    const float *d_C_qt,       // (N_q, N_t) col-major
    const float *d_inner_F,    // (N_q, N_t) col-major (corrected)
    const float *d_alpha_E_t,  // (N_t,) atom-expanded alpha_E
    float       *d_E_partial,  // (N_q,) output
    float       *d_wE,         // (N_q,) output
    float       *d_row_expd_iF,// (N_q,) output — row sum of expd_iF
    float sigma2,
    int N_q, int N_t)
{
    (void)d_row_expd_iF;

    int tq = threadIdx.x;  // 0..TILE_Q-1
    int q  = blockIdx.x * TILE_Q + tq;
    bool valid_q = (q < N_q);

    int t0 = blockIdx.y * TILE_T;
    int t1 = t0 + TILE_T;
    if (t1 > N_t) t1 = N_t;

    float E_loc     = 0.0f;
    float wE_loc    = 0.0f;
    if (valid_q) {
        for (int t = t0; t < t1; t++) {
            long long idx = q + (long long)t * N_q;
            float c = d_C_qt[idx];
            if (c != 0.0f) {
                float aE  = d_alpha_E_t[t];
                E_loc   += c * (sigma2 * aE + d_inner_F[idx]);
                wE_loc  += c * aE;
            }
        }
    }

    if (valid_q) {
        atomicAdd(&d_E_partial[q], E_loc);
        atomicAdd(&d_wE[q], wE_loc);
    }
}
#undef TILE_Q
#undef TILE_T


// ---------------------------------------------------------------------------
// inference_G_diag_correction_kernel: apply diagonal corrections to G_acc.
// G_acc[q,k] += (row_expd_iF[q] - wE[q]) * X_q[q,k]
//
// One thread per (q, k) element.
// ---------------------------------------------------------------------------
__global__ static void inference_G_diag_correction_kernel(
    float       *d_G_acc,           // (N_q, rep) row-major — modified in-place
    const float *d_X_q,             // (N_q, rep) row-major
    const float *d_row_expd_iF,     // (N_q,)
    const float *d_wE,              // (N_q,)
    int N_q, int rep_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_q * rep_size) return;

    int q = idx / rep_size;
    int k = idx % rep_size;

    float scale = d_row_expd_iF[q] - d_wE[q];
    d_G_acc[idx] += scale * d_X_q[(long long)q * rep_size + k];
}


// ---------------------------------------------------------------------------
// precompute_combined_train_kernel: build combined = adF + alpha_E_expand * X_t
// combined[t, k] = alpha_desc[t, k] + alpha_E[mol_of_t] * X_t[t, k]
// (N_t, rep) row-major
// ---------------------------------------------------------------------------
__global__ static void precompute_combined_train_kernel(
    const float *d_alpha_desc,  // (N_t, rep) row-major (flat view of (nm_t, max_atoms_t, rep))
    const float *d_X_t,         // (N_t, rep) row-major
    const float *d_alpha_E_t,   // (N_t,) atom-expanded alpha_E
    const int   *d_N_t,         // (nm_t,)
    float       *d_combined,    // (N_t, rep) row-major output
    int N_t, int max_atoms_t, int rep_size, int nm_t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_t * rep_size) return;

    int t = idx / rep_size;
    int mol = t / max_atoms_t;
    int atom = t % max_atoms_t;

    if (mol >= nm_t || atom >= d_N_t[mol]) {
        d_combined[idx] = 0.0f;
        return;
    }

    d_combined[idx] = d_alpha_desc[idx] + d_alpha_E_t[t] * d_X_t[idx];
}


// ---------------------------------------------------------------------------
// precompute_alpha_E_expand_kernel: alpha_E_t[t] = alpha_E[mol(t)]
// ---------------------------------------------------------------------------
__global__ static void precompute_alpha_E_expand_kernel(
    const float *d_alpha_E,
    const int   *d_N_t,
    float       *d_alpha_E_t,
    int N_t, int max_atoms_t, int nm_t)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N_t) return;

    int mol = t / max_atoms_t;
    int atom = t % max_atoms_t;
    if (mol >= nm_t || atom >= d_N_t[mol]) {
        d_alpha_E_t[t] = 0.0f;
        return;
    }
    d_alpha_E_t[t] = d_alpha_E[mol];
}


// ---------------------------------------------------------------------------
// precompute_S_adF_kernel: S_adF[t] = Σ_k X_t[t,k] * alpha_desc[t,k]
// ---------------------------------------------------------------------------
__global__ static void precompute_S_adF_kernel(
    const float *d_X_t,
    const float *d_alpha_desc,
    const int   *d_N_t,
    float       *d_S_adF,
    int N_t, int max_atoms_t, int rep_size, int nm_t)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N_t) return;

    int mol = t / max_atoms_t;
    int atom = t % max_atoms_t;
    if (mol >= nm_t || atom >= d_N_t[mol]) {
        d_S_adF[t] = 0.0f;
        return;
    }

    float s = 0.0f;
    const float *xt = d_X_t + (long long)t * rep_size;
    const float *ad = d_alpha_desc + (long long)t * rep_size;
    for (int k = 0; k < rep_size; k++)
        s += xt[k] * ad[k];
    d_S_adF[t] = s;
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
__global__ static KF_UNUSED_GLOBAL void local_inference_accumulate_kernel(
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
// Grid: (ncols_max, nm_q), with one block per (molecule, Cartesian component).
// Block: reduction threads over the flattened (atom, descriptor) space.
// ---------------------------------------------------------------------------
__global__ static void local_force_backproject_kernel(
    const float *d_dX_q,    // (nm_q, max_atoms_q, rep, lda)  lda=3*max_atoms_q
    const float *d_G_acc,   // (nm_q * max_atoms_q, rep)
    const int   *d_N_q,     // (nm_q,)
    const int   *d_offs_q,  // (nm_q,) Cartesian offsets in F_pred
    float       *d_F_pred,  // (naq_q,)
    int nm_q, int max_atoms_q, int rep_size)
{
    int a = blockIdx.y;
    if (a >= nm_q) return;
    int na  = d_N_q[a];
    int lda = 3 * max_atoms_q;
    int ncols_a = 3 * na;
    int off_a   = d_offs_q[a];

    int c = blockIdx.x;
    if (c >= ncols_a) return;

    long long nterms = (long long)na * rep_size;
    float sum = 0.0f;
    for (long long idx = threadIdx.x; idx < nterms; idx += blockDim.x) {
        int i1 = (int)(idx / rep_size);
        int k  = (int)(idx - (long long)i1 * rep_size);
        const float *dxa = d_dX_q + (long long)(a * max_atoms_q + i1) * rep_size * lda;
        const float *G   = d_G_acc + (long long)(a * max_atoms_q + i1) * rep_size;
        sum += dxa[(long long)k * lda + c] * G[k];
    }

    __shared__ float sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_F_pred[off_a + c] = sh[0];
}


// ============================================================================
// Host-side helper: build Cartesian offset array from N
// ============================================================================
static void build_offsets_from_host(const int *h_N, int *d_offs, int nm)
{
    int *h_offs = (int*)malloc(nm * sizeof(int));
    int acc = 0;
    for (int m = 0; m < nm; m++) {
        h_offs[m] = acc;
        int n_m = h_N[m];
        if (n_m < 0) n_m = 0;
        acc += 3 * n_m;
    }
    CUDA_CHECK(cudaMemcpy(d_offs, h_offs, nm * sizeof(int), cudaMemcpyHostToDevice));
    free(h_offs);
}

static void build_offsets(const int *d_N, int *d_offs, int nm)
{
    int *h_N = (int*)malloc(nm * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_N, d_N, nm * sizeof(int), cudaMemcpyDeviceToHost));
    build_offsets_from_host(h_N, d_offs, nm);
    free(h_N);
}

static void build_pair_indices(std::vector<int> &pair_a, std::vector<int> &pair_b, int nm)
{
    long long n_pairs = (long long)nm * (nm + 1) / 2;
    pair_a.resize((size_t)n_pairs);
    pair_b.resize((size_t)n_pairs);

    int pair_idx = 0;
    for (int a = 0; a < nm; a++) {
        for (int b = 0; b <= a; b++, pair_idx++) {
            pair_a[(size_t)pair_idx] = a;
            pair_b[(size_t)pair_idx] = b;
        }
    }
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
    int N   = nm * max_atoms;   // total atoms (padded)
    float inv_2s2 = -0.5f / (sigma * sigma);
    float inv_s2  =  1.0f / (sigma * sigma);
    float inv_s4  =  inv_s2 * inv_s2;
    int ncols_max = 3 * max_atoms;
    const float neg2 = -2.0f, zero_f = 0.0f, one_f = 1.0f;

    // Download N for host-side pair iteration and offset construction.
    int *h_N = (int*)malloc(nm * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_N, d_N, nm * sizeof(int), cudaMemcpyDeviceToHost));

    // Build Cartesian offset array
    int *d_offs;
    CUDA_CHECK(cudaMalloc(&d_offs, nm * sizeof(int)));
    build_offsets_from_host(h_N, d_offs, nm);

    // Zero the output matrix
    CUDA_CHECK(cudaMemset(d_K_full, 0, (long long)BIG * BIG * sizeof(float)));

    // CUDA events for per-stage timing
    cudaEvent_t ev_start, ev_s1, ev_s2, ev_s3, ev_s4;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_s1));
    CUDA_CHECK(cudaEventCreate(&ev_s2));
    CUDA_CHECK(cudaEventCreate(&ev_s3));
    CUDA_CHECK(cudaEventCreate(&ev_s4));
    CUDA_CHECK(cudaEventRecord(ev_start));

    // ================================================================
    // Stage 1: Build C_label matrix (N × N) — Gaussian kernel values
    //          with label screening. Uses cuBLAS SGEMM for distances.
    // ================================================================
    //
    // d_X is (nm, max_atoms, rep_size) row-major = (N, rep_size) row-major.
    // cuBLAS sees column-major, so d_X is (rep_size, N) col-major.
    // We want C = X @ X^T in row-major = X^T @ X in col-major.
    // C[i,j] = -2 * X[i,:] · X[j,:] (will add norms + exp later)
    //
    // Col-major: C(N,N) = X^T(N,rep) @ X(rep,N)
    // cublasSgemm(T, N, N, N, rep, -2, X, rep, X, rep, 0, C, N)

    float *d_C, *d_norms;
    CUDA_CHECK(cudaMalloc(&d_C,     (long long)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms, (long long)N * sizeof(float)));

    // Squared norms of each atom's descriptor
    compute_sqnorms_kernel<<<(N + 255) / 256, 256>>>(d_X, d_norms, rep_size, N);

    // C = -2 * X^T @ X  (col-major; d_X is rep_size × N col-major)
    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, rep_size,
        &neg2, d_X, rep_size, d_X, rep_size,
        &zero_f, d_C, N));

    // In-place: C[i,j] = exp(-0.5*inv_s2*(C[i,j]+n[i]+n[j])) * inv_s2
    // with label screening (zero for non-matching labels or inactive atoms)
    {
        dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
        build_C_label_kernel<<<grd, blk>>>(d_C, d_norms, d_Q, d_N, inv_s2, N, max_atoms);
    }
    CUDA_CHECK(cudaEventRecord(ev_s1));

    cudaFree(d_norms);

    // ================================================================
    // Stage 2: K_FF static part via batched SGEMM
    //
    // For each pair (a, b) with a >= b:
    //   GEMM 1: W = C_ab (na,nb) @ dX_b_flat (nb, rep*lda) → (na, rep*lda)
    //   GEMM 2: K_FF_static = dX_a_flat^T (lda, na*rep) @ W (na*rep, lda) → (lda, lda)
    //
    // Use cublasSgemmBatched to launch all pairs at once.
    // ================================================================

    long long n_pairs = (long long)nm * (nm + 1) / 2;
    int lda = 3 * max_atoms;
    int rl = rep_size * lda;  // rep*lda

    std::vector<int> pair_a;
    std::vector<int> pair_b;
    build_pair_indices(pair_a, pair_b, nm);

    int *d_pair_a = nullptr;
    int *d_pair_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pair_a, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_b, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_a, pair_a.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pair_b, pair_b.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate K_FF accumulator per pair (ncols_max × ncols_max per pair)
    float *d_K_FF_pairs;
    CUDA_CHECK(cudaMalloc(&d_K_FF_pairs,
        n_pairs * (long long)ncols_max * ncols_max * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_K_FF_pairs, 0,
        n_pairs * (long long)ncols_max * ncols_max * sizeof(float)));

    // Workspace W for per-pair intermediate.
    long long W_per_pair = (long long)max_atoms * rl;
    long long max_batch_mem = 2LL * 1024 * 1024 * 1024 / sizeof(float);  // 2 GB in floats
    int max_batch = (int)std::min((long long)n_pairs, max_batch_mem / W_per_pair);
    if (max_batch < 1) max_batch = 1;

    float *d_W;
    CUDA_CHECK(cudaMalloc(&d_W, (long long)max_batch * W_per_pair * sizeof(float)));

    // Build pointer arrays on host for cublasSgemmBatched
    // GEMM 1: W_cm[p] = dX_b_flat_cm[p] @ C_ab_cm^T[p]
    //   (rl, na[p]) = (rl, nb[p]) @ (nb[p], na[p])
    //   cublasSgemm(N, T, rl, na, nb, 1, dX_b, rl, C_ab, N, 0, W, rl)
    //
    // GEMM 2: K_FF_cm[p] = W_cm_reshaped[p] @ dX_a_cm_reshaped^T[p]
    //   (ncols_b, ncols_a) = (lda, K2) @ (K2, lda) ... with only ncols_b/ncols_a used
    //   cublasSgemm(N, T, ncols_b, ncols_a, K2, 1, W, lda, dX_a, lda, 0, KFF, ncols_max)

    // Check if all molecules have the same atom count (uniform case -> strided batched)
    bool uniform = true;
    int na0 = h_N[0];
    for (int m = 1; m < nm; m++) {
        if (h_N[m] != na0) { uniform = false; break; }
    }

    if (uniform && na0 > 0) {
        // All molecules have the same atom count -> use cublasSgemmBatched
        int na = na0, nb = na0;
        int ncols_a = 3 * na, ncols_b = 3 * nb;
        int K2 = na * rep_size;

        // Pre-allocate device pointer arrays once (reused across chunks)
        const float **d_Aptr1, **d_Bptr1;
        float **d_Cptr1;
        const float **d_Aptr2, **d_Bptr2;
        float **d_Cptr2;
        CUDA_CHECK(cudaMalloc(&d_Aptr1, max_batch * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Bptr1, max_batch * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Cptr1, max_batch * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Aptr2, max_batch * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Bptr2, max_batch * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Cptr2, max_batch * sizeof(float*)));

        // Process pairs in chunks of max_batch to limit W memory
        for (long long chunk_start = 0; chunk_start < n_pairs; chunk_start += max_batch) {
            int chunk_sz = (int)std::min((long long)max_batch, n_pairs - chunk_start);

            // Build pointer arrays for this chunk
            std::vector<const float*> h_A1(chunk_sz), h_B1(chunk_sz);
            std::vector<float*> h_C1(chunk_sz);
            std::vector<const float*> h_A2(chunk_sz), h_B2(chunk_sz);
            std::vector<float*> h_C2(chunk_sz);

            for (int cidx = 0; cidx < chunk_sz; ++cidx) {
                int pidx = (int)chunk_start + cidx;
                int a = pair_a[(size_t)pidx];
                int b = pair_b[(size_t)pidx];

                h_A1[cidx] = d_dX + (long long)(b * max_atoms) * rl;
                h_B1[cidx] = d_C + (long long)a * max_atoms + (long long)b * max_atoms * N;
                h_C1[cidx] = d_W + (long long)cidx * W_per_pair;

                h_A2[cidx] = d_W + (long long)cidx * W_per_pair;
                h_B2[cidx] = d_dX + (long long)(a * max_atoms) * rl;
                h_C2[cidx] = d_K_FF_pairs + (long long)pidx * ncols_max * ncols_max;
            }

            CUDA_CHECK(cudaMemcpy(d_Aptr1, h_A1.data(), chunk_sz * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Bptr1, h_B1.data(), chunk_sz * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Cptr1, h_C1.data(), chunk_sz * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Aptr2, h_A2.data(), chunk_sz * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Bptr2, h_B2.data(), chunk_sz * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Cptr2, h_C2.data(), chunk_sz * sizeof(float*), cudaMemcpyHostToDevice));

            CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                rl, na, nb,
                &one_f, d_Aptr1, rl, d_Bptr1, N,
                &zero_f, d_Cptr1, rl,
                chunk_sz));

            CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                ncols_b, ncols_a, K2,
                &one_f, d_Aptr2, lda, d_Bptr2, lda,
                &zero_f, d_Cptr2, ncols_max,
                chunk_sz));

        }

        cudaFree(d_Aptr1); cudaFree(d_Bptr1); cudaFree(d_Cptr1);
        cudaFree(d_Aptr2); cudaFree(d_Bptr2); cudaFree(d_Cptr2);

    } else {
        // Non-uniform atom counts -> batch pairs by (na, nb) so we still use
        // cuBLAS batched GEMM instead of issuing two GEMMs per pair.
        struct PairGroup {
            int na = 0;
            int nb = 0;
            std::vector<int> pair_indices;
        };

        std::unordered_map<long long, PairGroup> groups;
        groups.reserve((size_t)max_atoms * max_atoms);

        int pair_idx = 0;
        for (int a = 0; a < nm; a++) {
            int na = h_N[a];
            for (int b = 0; b <= a; b++, pair_idx++) {
                int nb = h_N[b];
                if (na <= 0 || nb <= 0) continue;
                long long key = ((long long)na << 32) | (unsigned int)nb;
                auto &group = groups[key];
                if (group.pair_indices.empty()) {
                    group.na = na;
                    group.nb = nb;
                }
                group.pair_indices.push_back(pair_idx);
            }
        }

        int max_group_pairs = 0;
        long long max_group_W = 0;
        for (const auto &kv : groups) {
            const PairGroup &group = kv.second;
            int group_pairs = (int)group.pair_indices.size();
            if (group_pairs > max_group_pairs) max_group_pairs = group_pairs;
            long long group_W = (long long)group.na * rl;
            if (group_W > max_group_W) max_group_W = group_W;
        }

        float *d_W_nu = nullptr;
        const float **d_Aptr1 = nullptr, **d_Bptr1 = nullptr;
        float **d_Cptr1 = nullptr;
        const float **d_Aptr2 = nullptr, **d_Bptr2 = nullptr;
        float **d_Cptr2 = nullptr;

        if (max_group_pairs > 0) {
            CUDA_CHECK(cudaMalloc(&d_W_nu, (long long)max_group_pairs * max_group_W * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_Aptr1, max_group_pairs * sizeof(float*)));
            CUDA_CHECK(cudaMalloc(&d_Bptr1, max_group_pairs * sizeof(float*)));
            CUDA_CHECK(cudaMalloc(&d_Cptr1, max_group_pairs * sizeof(float*)));
            CUDA_CHECK(cudaMalloc(&d_Aptr2, max_group_pairs * sizeof(float*)));
            CUDA_CHECK(cudaMalloc(&d_Bptr2, max_group_pairs * sizeof(float*)));
            CUDA_CHECK(cudaMalloc(&d_Cptr2, max_group_pairs * sizeof(float*)));
        }

        for (const auto &kv : groups) {
            const PairGroup &group = kv.second;
            int na = group.na;
            int nb = group.nb;
            int ncols_a = 3 * na;
            int ncols_b = 3 * nb;
            int K2 = na * rep_size;
            int group_pairs = (int)group.pair_indices.size();
            long long W_group_stride = (long long)na * rl;

            std::vector<const float*> h_A1(group_pairs), h_B1(group_pairs);
            std::vector<float*> h_C1(group_pairs);
            std::vector<const float*> h_A2(group_pairs), h_B2(group_pairs);
            std::vector<float*> h_C2(group_pairs);

            for (int idx = 0; idx < group_pairs; ++idx) {
                int p = group.pair_indices[idx];
                int a = pair_a[(size_t)p];
                int b = pair_b[(size_t)p];

                h_A1[idx] = d_dX + (long long)(b * max_atoms) * rl;
                h_B1[idx] = d_C + (long long)a * max_atoms + (long long)b * max_atoms * N;
                h_C1[idx] = d_W_nu + (long long)idx * W_group_stride;

                h_A2[idx] = d_W_nu + (long long)idx * W_group_stride;
                h_B2[idx] = d_dX + (long long)(a * max_atoms) * rl;
                h_C2[idx] = d_K_FF_pairs + (long long)p * ncols_max * ncols_max;
            }

            CUDA_CHECK(cudaMemcpy(d_Aptr1, h_A1.data(), group_pairs * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Bptr1, h_B1.data(), group_pairs * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Cptr1, h_C1.data(), group_pairs * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Aptr2, h_A2.data(), group_pairs * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Bptr2, h_B2.data(), group_pairs * sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Cptr2, h_C2.data(), group_pairs * sizeof(float*), cudaMemcpyHostToDevice));

            CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                rl, na, nb,
                &one_f, d_Aptr1, rl, d_Bptr1, N,
                &zero_f, d_Cptr1, rl,
                group_pairs));

            CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                ncols_b, ncols_a, K2,
                &one_f, d_Aptr2, lda, d_Bptr2, lda,
                &zero_f, d_Cptr2, ncols_max,
                group_pairs));
        }
        if (max_group_pairs > 0) {
            cudaFree(d_W_nu);
            cudaFree(d_Aptr1); cudaFree(d_Bptr1); cudaFree(d_Cptr1);
            cudaFree(d_Aptr2); cudaFree(d_Bptr2); cudaFree(d_Cptr2);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev_s2));

    // ================================================================
    // Stage 3: VA/VB via SGEMM, rank-1 correction via SGEMM,
    //          K_EE/K_FE/K_EF assembly.
    //
    // Flow per chunk:
    //   a) P_ab, P_ba cross-terms via batched SGEMM
    //   b) Prepare WA/VB matrices + K_EE/K_FE/K_EF (custom kernel)
    //   c) Rank-1 SGEMM: K_FF_pairs += WA^T @ VB (adds to K_FF_static)
    // ================================================================
    {
        float sigma2 = sigma * sigma;
        int ncM = 3 * max_atoms;
        int nab = max_atoms * max_atoms;  // padded atom pairs per molecule pair

        // 4a. Precompute self-dot products S for all atoms
        float *d_S;
        CUDA_CHECK(cudaMalloc(&d_S, (long long)N * lda * sizeof(float)));
        precompute_self_dots_kernel<<<(N + 255) / 256, 256>>>(
            d_X, d_dX, d_N, d_S, nm, max_atoms, rep_size, lda);

        // Memory per pair (padded to max_atoms): P_ab + P_ba + WA + VB
        long long P_ab_per_pair = (long long)max_atoms * ncols_max * max_atoms;
        long long P_ba_per_pair = (long long)max_atoms * ncols_max * max_atoms;
        long long WA_per_pair = (long long)nab * ncols_max;
        long long VB_per_pair = (long long)nab * ncols_max;
        long long mem_per_pair = (P_ab_per_pair + P_ba_per_pair + WA_per_pair + VB_per_pair) * sizeof(float);

        long long max_chunk_mem = 2LL * 1024 * 1024 * 1024;  // 2 GB
        int max_chunk_pairs = (int)std::min((long long)n_pairs,
                                             max_chunk_mem / std::max(mem_per_pair, 1LL));
        if (max_chunk_pairs < 1) max_chunk_pairs = 1;

        float *d_P_ab, *d_P_ba, *d_WA, *d_VB_buf;
        CUDA_CHECK(cudaMalloc(&d_P_ab,   (long long)max_chunk_pairs * P_ab_per_pair * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_P_ba,   (long long)max_chunk_pairs * P_ba_per_pair * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_WA,     (long long)max_chunk_pairs * WA_per_pair * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_VB_buf, (long long)max_chunk_pairs * VB_per_pair * sizeof(float)));

        int scalar_acc_sz = 1 + 2 * ncM;
        size_t smem_precomp = (size_t)scalar_acc_sz * sizeof(float);

        int max_atom_pairs = max_atoms * max_atoms;
        int block_sz = ((max_atom_pairs + 31) / 32) * 32;
        if (block_sz < 32)  block_sz = 32;
        if (block_sz > 128) block_sz = 128;

        // Pre-allocate device pointer arrays once (reused across chunks)
        int max_batch_pab = max_chunk_pairs * max_atoms;
        const float **d_Aptr_ab, **d_Bptr_ab;
        float **d_Cptr_ab;
        const float **d_Aptr_ba, **d_Bptr_ba;
        float **d_Cptr_ba;

        CUDA_CHECK(cudaMalloc(&d_Aptr_ab, max_batch_pab * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Bptr_ab, max_batch_pab * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Cptr_ab, max_batch_pab * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Aptr_ba, max_batch_pab * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Bptr_ba, max_batch_pab * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_Cptr_ba, max_batch_pab * sizeof(float*)));

        for (long long chunk_start = 0; chunk_start < n_pairs; chunk_start += max_chunk_pairs) {
            int chunk_sz = (int)std::min((long long)max_chunk_pairs, n_pairs - chunk_start);

            int batch_pab = chunk_sz * max_atoms;
            int batch_pba = chunk_sz * max_atoms;

            build_stage3_pointer_arrays_kernel<<<(batch_pab + 255) / 256, 256>>>(
                d_X, d_dX, d_P_ab, d_P_ba,
                d_pair_a, d_pair_b,
                d_Aptr_ab, d_Bptr_ab, d_Cptr_ab,
                d_Aptr_ba, d_Bptr_ba, d_Cptr_ba,
                rep_size, lda, max_atoms, ncols_max,
                (int)chunk_start, chunk_sz);

            // P_ab GEMMs (padded to max_atoms/ncols_max)
            CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                ncols_max, max_atoms, rep_size,
                &one_f, d_Aptr_ab, lda, d_Bptr_ab, rep_size,
                &zero_f, d_Cptr_ab, ncols_max,
                batch_pab));

            // P_ba GEMMs
            CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                ncols_max, max_atoms, rep_size,
                &one_f, d_Aptr_ba, lda, d_Bptr_ba, rep_size,
                &zero_f, d_Cptr_ba, ncols_max,
                batch_pba));

            // Prepare WA/VB + K_EE/K_FE/K_EF
            {
                prepare_WA_VB_and_scalars_kernel<<<chunk_sz, block_sz, smem_precomp>>>(
                    d_C, d_S, d_P_ab, d_P_ba,
                    d_N, d_offs, d_K_full,
                    d_WA, d_VB_buf,
                    sigma2, inv_s2,
                    nm, max_atoms, lda, N, BIG,
                    max_atoms, max_atoms,
                    (int)chunk_start, chunk_sz);
            }

            // Rank-1 SGEMM: K_FF_pairs += WA^T @ VB
            {
                CUBLAS_CHECK(cublasSgemmStridedBatched(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    ncols_max, ncols_max, nab,
                    &one_f,
                    d_WA, nab, WA_per_pair,
                    d_VB_buf, nab, VB_per_pair,
                    &one_f,
                    d_K_FF_pairs + (long long)chunk_start * ncols_max * ncols_max,
                    ncols_max, (long long)ncols_max * ncols_max,
                    chunk_sz));

            }
        }

        cudaFree(d_Aptr_ab); cudaFree(d_Bptr_ab); cudaFree(d_Cptr_ab);
        cudaFree(d_Aptr_ba); cudaFree(d_Bptr_ba); cudaFree(d_Cptr_ba);

        cudaFree(d_S);
        cudaFree(d_P_ab);
        cudaFree(d_P_ba);
        cudaFree(d_WA);
        cudaFree(d_VB_buf);
    }
    CUDA_CHECK(cudaEventRecord(ev_s3));

    // ================================================================
    // Stage 4: scatter K_FF (static + rank-1) into K_full.
    // ================================================================
    // Scatter K_FF (static + rank-1) into K_full
    scatter_K_FF_kernel<<<(int)n_pairs, 256>>>(
        d_K_FF_pairs, d_N, d_offs, d_K_full,
        nm, max_atoms, ncols_max, BIG);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_s4));
    CUDA_CHECK(cudaEventSynchronize(ev_s4));

    // Print per-stage timing
    {
        float t1, t2, t3, t4;
        cudaEventElapsedTime(&t1, ev_start, ev_s1);
        cudaEventElapsedTime(&t2, ev_s1, ev_s2);
        cudaEventElapsedTime(&t3, ev_s2, ev_s3);
        cudaEventElapsedTime(&t4, ev_s3, ev_s4);
        fprintf(stderr,
            "  [kernel_symm stages]  S1=%.1f ms  S2=%.1f ms  S3=%.1f ms  S4=%.1f ms  total=%.1f ms\n",
            t1, t2, t3, t4, t1+t2+t3+t4);
    }
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_s1);
    cudaEventDestroy(ev_s2);
    cudaEventDestroy(ev_s3);
    cudaEventDestroy(ev_s4);

    // Cleanup
    cudaFree(d_offs);
    cudaFree(d_C);
    cudaFree(d_W);
    cudaFree(d_K_FF_pairs);
    cudaFree(d_pair_a);
    cudaFree(d_pair_b);
    free(h_N);
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
    CUDA_CHECK(cudaMemset(d_G_acc, 0, n_q_atoms * rep_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_wE_arr, 0, n_q_atoms * sizeof(float)));

    cudaEvent_t ev_mv0, ev_mv1, ev_mv2, ev_mv3;
    CUDA_CHECK(cudaEventCreate(&ev_mv0));
    CUDA_CHECK(cudaEventCreate(&ev_mv1));
    CUDA_CHECK(cudaEventCreate(&ev_mv2));
    CUDA_CHECK(cudaEventCreate(&ev_mv3));
    CUDA_CHECK(cudaEventRecord(ev_mv0));

    // Phase 1: SGEMM-based accumulation of E_partial, G_acc, wE
    //
    // Decompose into large matrix multiplies:
    //   C_qt(Nq, Nt) — query-train Gaussian kernel with label screening
    //   inner_F(Nq, Nt) = X_q @ adF^T - broadcast(S_adF)
    //   expd_iF = -(C_qt/σ²) ⊙ inner_F
    //   G_acc = C_qt @ combined_t  -  expd_iF @ X_t  +  diag(row_expd_iF - wE) * X_q
    //   E_partial = σ² * C_qt @ alpha_E_expand + row_sum(C_qt ⊙ inner_F)
    {
        int N_q = nm_q * max_atoms_q;
        int N_t = nm_t * max_atoms_t;
        const float neg2 = -2.0f, zero_f = 0.0f, one_f = 1.0f, neg1 = -1.0f;

        // 1a. Squared norms for query and training atoms
        float *d_norms_q, *d_norms_t;
        CUDA_CHECK(cudaMalloc(&d_norms_q, N_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_norms_t, N_t * sizeof(float)));
        compute_sqnorms_kernel<<<(N_q + 255) / 256, 256>>>(d_X_q, d_norms_q, rep_size, N_q);
        compute_sqnorms_kernel<<<(N_t + 255) / 256, 256>>>(d_X_t, d_norms_t, rep_size, N_t);

        // 1b. C_qt = -2 * X_q^T @ X_t  (col-major: (N_q, N_t))
        float *d_C_qt;
        CUDA_CHECK(cudaMalloc(&d_C_qt, (long long)N_q * N_t * sizeof(float)));
        CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            N_q, N_t, rep_size,
            &neg2, d_X_q, rep_size, d_X_t, rep_size,
            &zero_f, d_C_qt, N_q));

        // 1c. Apply norms + exp + label screening
        {
            dim3 blk(16, 16), grd((N_q + 15) / 16, (N_t + 15) / 16);
            build_C_qt_kernel<<<grd, blk>>>(
                d_C_qt, d_norms_q, d_norms_t,
                d_Q_q, d_Q_t, d_N_q, d_N_t,
                inv_s2, N_q, N_t, max_atoms_q, max_atoms_t);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_norms_q);
        cudaFree(d_norms_t);

        // 2. Precompute S_adF[t] = Σ_k X_t[t,k] * alpha_desc[t,k]
        float *d_S_adF;
        CUDA_CHECK(cudaMalloc(&d_S_adF, N_t * sizeof(float)));
        precompute_S_adF_kernel<<<(N_t + 255) / 256, 256>>>(
            d_X_t, d_alpha_desc, d_N_t, d_S_adF,
            N_t, max_atoms_t, rep_size, nm_t);

        // 3. inner_F(Nq, Nt) = X_q @ adF^T
        float *d_inner_F;
        CUDA_CHECK(cudaMalloc(&d_inner_F, (long long)N_q * N_t * sizeof(float)));
        CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            N_q, N_t, rep_size,
            &one_f, d_X_q, rep_size, d_alpha_desc, rep_size,
            &zero_f, d_inner_F, N_q));

        // 4. Allocate row-sum buffer and build expd_iF = -(C_qt/σ²) *
        //    (inner_F - S_adF broadcast). Also corrects inner_F in-place.
        float *d_row_expd_iF;
        CUDA_CHECK(cudaMalloc(&d_row_expd_iF, N_q * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_row_expd_iF, 0, N_q * sizeof(float)));

        float *d_expd_iF;
        CUDA_CHECK(cudaMalloc(&d_expd_iF, (long long)N_q * N_t * sizeof(float)));
        {
            long long total = (long long)N_q * N_t;
            int blk = 256;
            int grd = (int)((total + blk - 1) / blk);
            inference_build_expd_iF_kernel<<<grd, blk>>>(
                d_C_qt, d_inner_F, d_S_adF, d_expd_iF, d_row_expd_iF,
                inv_s2, N_q, N_t);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_S_adF);

        // 5. Precompute alpha_E_t[t] = alpha_E[mol(t)] and
        //    combined_t[t,k] = alpha_desc[t,k] + alpha_E_t[t] * X_t[t,k]
        float *d_alpha_E_t;
        CUDA_CHECK(cudaMalloc(&d_alpha_E_t, N_t * sizeof(float)));
        precompute_alpha_E_expand_kernel<<<(N_t + 255) / 256, 256>>>(
            d_alpha_E, d_N_t, d_alpha_E_t,
            N_t, max_atoms_t, nm_t);

        float *d_combined_t;
        CUDA_CHECK(cudaMalloc(&d_combined_t, (long long)N_t * rep_size * sizeof(float)));
        {
            long long total = (long long)N_t * rep_size;
            precompute_combined_train_kernel<<<(int)((total + 255) / 256), 256>>>(
                d_alpha_desc, d_X_t, d_alpha_E_t, d_N_t, d_combined_t,
                N_t, max_atoms_t, rep_size, nm_t);
        }

        // 6. G_acc = C_qt @ combined_t  (col-major NT)
        CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            rep_size, N_q, N_t,
            &one_f, d_combined_t, rep_size, d_C_qt, N_q,
            &zero_f, d_G_acc, rep_size));
        cudaFree(d_combined_t);

        // 7. G_acc -= expd_iF @ X_t  (col-major NT, beta=1 accumulate)
        CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            rep_size, N_q, N_t,
            &neg1, d_X_t, rep_size, d_expd_iF, N_q,
            &one_f, d_G_acc, rep_size));

        CUDA_CHECK(cudaDeviceSynchronize());

        // 8. E_partial and wE (per query atom). row_sum_expd_iF already built.
        {
            dim3 blk_e(32, 1);
            dim3 grd_e((N_q + 31) / 32, (N_t + 255) / 256);
            inference_E_and_diag_kernel<<<grd_e, blk_e>>>(
                d_C_qt, d_inner_F, d_alpha_E_t,
                d_E_partial, d_wE_arr, d_row_expd_iF,
                sigma * sigma,
                N_q, N_t);
        }

        // 9. Diagonal correction: G_acc[q,k] += (row_expd_iF[q] - wE[q]) * X_q[q,k]
        {
            long long total = (long long)N_q * rep_size;
            inference_G_diag_correction_kernel<<<(int)((total + 255) / 256), 256>>>(
                d_G_acc, d_X_q, d_row_expd_iF, d_wE_arr,
                N_q, rep_size);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_C_qt);
        cudaFree(d_inner_F);
        cudaFree(d_expd_iF);
        cudaFree(d_alpha_E_t);
        cudaFree(d_row_expd_iF);
    }
    CUDA_CHECK(cudaEventRecord(ev_mv1));

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
    CUDA_CHECK(cudaEventRecord(ev_mv2));

    // Phase 3: back-project G_acc → F_pred via dX_q^T @ G_acc per query molecule
    {
        int ncols_max = 3 * max_atoms_q;
        int bp_block  = 256;
        dim3 bp_grid(ncols_max, nm_q);
        local_force_backproject_kernel<<<bp_grid, bp_block>>>(
            d_dX_q, d_G_acc, d_N_q, d_offs_q,
            d_F_pred, nm_q, max_atoms_q, rep_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaEventRecord(ev_mv3));
    CUDA_CHECK(cudaEventSynchronize(ev_mv3));

    {
        float t1, t2, t3;
        cudaEventElapsedTime(&t1, ev_mv0, ev_mv1);
        cudaEventElapsedTime(&t2, ev_mv1, ev_mv2);
        cudaEventElapsedTime(&t3, ev_mv2, ev_mv3);
        fprintf(stderr,
            "  [matvec phases]  P1_accum=%.1f ms  P2_reduce=%.1f ms  P3_backproj=%.1f ms  total=%.1f ms\n",
            t1, t2, t3, t1+t2+t3);
    }
    cudaEventDestroy(ev_mv0);
    cudaEventDestroy(ev_mv1);
    cudaEventDestroy(ev_mv2);
    cudaEventDestroy(ev_mv3);

    // Cleanup
    cudaFree(d_offs_q);
    cudaFree(d_E_partial);
    cudaFree(d_G_acc);
    cudaFree(d_wE_arr);
}

// ===========================================================================
// kernel_gaussian_precompute_train_local_cu
// ===========================================================================
//
// Precomputes the three training-side constants that are fixed across all
// calls to kernel_gaussian_full_matvec_cached_local_cu.  Call once after
// fitting; pass the results into the cached matvec at every MD step.
//
//   d_norms_t    (N_t,)         ||X_t[t]||²
//   d_S_adF      (N_t,)         X_t[t] · alpha_desc[t]
//   d_alpha_E_t  (N_t,)         alpha_E expanded to atom rows
//   d_combined_t (N_t, rep)     alpha_desc[t] + alpha_E_t[t] * X_t[t]
//
// All three output buffers must be pre-allocated by the caller.
// ---------------------------------------------------------------------------
void kernel_gaussian_precompute_train_local_cu(
    const float *d_X_t,
    const float *d_alpha_desc,
    const float *d_alpha_E,
    const int   *d_N_t,
    float       *d_norms_t,
    float       *d_S_adF,
    float       *d_alpha_E_t,
    float       *d_combined_t,
    int nm_t, int max_atoms_t, int rep_size)
{
    int N_t = nm_t * max_atoms_t;

    compute_sqnorms_kernel<<<(N_t + 255) / 256, 256>>>(
        d_X_t, d_norms_t, rep_size, N_t);

    precompute_S_adF_kernel<<<(N_t + 255) / 256, 256>>>(
        d_X_t, d_alpha_desc, d_N_t, d_S_adF,
        N_t, max_atoms_t, rep_size, nm_t);

    precompute_alpha_E_expand_kernel<<<(N_t + 255) / 256, 256>>>(
        d_alpha_E, d_N_t, d_alpha_E_t,
        N_t, max_atoms_t, nm_t);

    {
        long long total = (long long)N_t * rep_size;
        precompute_combined_train_kernel<<<(int)((total + 255) / 256), 256>>>(
            d_alpha_desc, d_X_t, d_alpha_E_t, d_N_t, d_combined_t,
            N_t, max_atoms_t, rep_size, nm_t);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}


// ===========================================================================
// kernel_gaussian_full_matvec_cached_local_cu
// ===========================================================================
//
// Variant of kernel_gaussian_full_matvec_local_cu for repeated inference
// (e.g. MD simulation) where the three training-side constants (d_norms_t,
// d_S_adF, d_combined_t) are precomputed once and reused across calls.
//
// Changes vs the uncached version:
//   * cudaMalloc/cudaFree for d_norms_t, d_S_adF, d_combined_t removed.
//   * Three device-kernel launches to compute those terms removed.
//   * Timing fprintf / cudaEvent instrumentation removed.
//
// Caller is responsible for ensuring precomputed buffers remain valid and
// match the current d_X_t, d_alpha_desc, d_alpha_E, d_N_t.
// ---------------------------------------------------------------------------
void kernel_gaussian_full_matvec_cached_local_cu(
    const float *d_X_q,
    const float *d_dX_q,
    const int   *d_Q_q,
    const int   *d_N_q,
    const float *d_X_t,
    const int   *d_Q_t,
    const int   *d_N_t,
    const float *d_alpha_E,
    const float *d_alpha_desc,
    const float *d_norms_t,
    const float *d_S_adF,
    const float *d_alpha_E_t,
    const float *d_combined_t,
    float       *d_E_pred,
    float       *d_F_pred,
    float        sigma,
    int nm_q, int nm_t,
    int max_atoms_q, int max_atoms_t,
    int rep_size, int naq_q)
{
    ensure_cublas();

    float inv_s2 = 1.0f / (sigma * sigma);
    ensure_cached_local_matvec_scratch(nm_q, max_atoms_q, rep_size, nm_t, max_atoms_t);
    ensure_cached_local_training_element_cache(
        d_Q_t, d_N_t, d_X_t, d_alpha_desc, d_norms_t, d_S_adF, d_alpha_E_t, d_combined_t,
        nm_t, max_atoms_t, rep_size);

    // Build query Cartesian offset array
    int *d_offs_q = s_cached_local_matvec_scratch.d_offs_q;
    build_offsets(d_N_q, d_offs_q, nm_q);

    // Reuse per-call scratch buffers to avoid allocator overhead in MD inference.
    long long n_q_atoms = (long long)nm_q * max_atoms_q;
    float *d_E_partial = s_cached_local_matvec_scratch.d_E_partial;
    float *d_G_acc     = s_cached_local_matvec_scratch.d_G_acc;
    float *d_wE_arr    = s_cached_local_matvec_scratch.d_wE_arr;
    CUDA_CHECK(cudaMemset(d_E_partial, 0, n_q_atoms * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_G_acc, 0, n_q_atoms * rep_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_wE_arr, 0, n_q_atoms * sizeof(float)));

    // Phase 1: SGEMM-based accumulation of E_partial, G_acc, wE
    {
        int N_q = nm_q * max_atoms_q;
        const float neg2 = -2.0f, zero_f = 0.0f, one_f = 1.0f, neg1 = -1.0f;
        CachedLocalTrainingElementCache &elem_cache = s_cached_local_training_element_cache;
        int *d_query_indices = s_cached_local_matvec_scratch.d_query_indices;
        int *d_query_count = s_cached_local_matvec_scratch.d_query_count;

        // 1a. Query squared norms only (training norms are precomputed)
        float *d_norms_q = s_cached_local_matvec_scratch.d_norms_q;
        compute_sqnorms_kernel<<<(N_q + 255) / 256, 256>>>(
            d_X_q, d_norms_q, rep_size, N_q);

        // 1b. Element-blocked SGEMMs to avoid cross-element work.
        float *d_C_qt = s_cached_local_matvec_scratch.d_C_qt;
        float *d_inner_F = s_cached_local_matvec_scratch.d_inner_F;
        float *d_row_expd_iF = s_cached_local_matvec_scratch.d_row_expd_iF;
        CUDA_CHECK(cudaMemset(d_C_qt, 0, (long long)N_q * (long long)(nm_t * max_atoms_t) * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_inner_F, 0, (long long)N_q * (long long)(nm_t * max_atoms_t) * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_row_expd_iF, 0, N_q * sizeof(float)));

        float *d_X_q_elem = s_cached_local_matvec_scratch.d_X_q_elem;
        float *d_norms_q_elem = s_cached_local_matvec_scratch.d_norms_q_elem;
        float *d_C_elem = s_cached_local_matvec_scratch.d_C_elem;
        float *d_inner_F_elem = s_cached_local_matvec_scratch.d_inner_F_elem;
        float *d_expd_iF_elem = s_cached_local_matvec_scratch.d_expd_iF_elem;
        float *d_G_acc_elem = s_cached_local_matvec_scratch.d_G_acc_elem;
        float *d_row_expd_iF_elem = s_cached_local_matvec_scratch.d_row_expd_iF_elem;

        for (const CachedLocalTrainingElementGroup &group : elem_cache.groups) {
            CUDA_CHECK(cudaMemset(d_query_count, 0, sizeof(int)));
            gather_active_query_indices_kernel<<<(N_q + 255) / 256, 256>>>(
                d_Q_q, d_N_q, d_query_indices, d_query_count,
                group.label, nm_q, max_atoms_q);

            int query_count = 0;
            CUDA_CHECK(cudaMemcpy(&query_count, d_query_count, sizeof(int), cudaMemcpyDeviceToHost));
            if (query_count == 0) continue;

            CUDA_CHECK(cudaMemset(d_G_acc_elem, 0, (size_t)query_count * rep_size * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_row_expd_iF_elem, 0, (size_t)query_count * sizeof(float)));

            gather_rows_kernel<<<(int)(((long long)query_count * rep_size + 255) / 256), 256>>>(
                d_X_q, d_X_q_elem, d_query_indices, query_count, rep_size);
            gather_scalars_kernel<<<(query_count + 255) / 256, 256>>>(
                d_norms_q, d_norms_q_elem, d_query_indices, query_count);

            CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                query_count, group.count, rep_size,
                &neg2, d_X_q_elem, rep_size, group.d_X_t, rep_size,
                &zero_f, d_C_elem, query_count));

            {
                dim3 blk(16, 16), grd((query_count + 15) / 16, (group.count + 15) / 16);
                apply_norms_exp_kernel<<<grd, blk>>>(
                    d_C_elem, d_norms_q_elem, group.d_norms_t,
                    inv_s2, query_count, group.count, 0);
            }

            CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                query_count, group.count, rep_size,
                &one_f, d_X_q_elem, rep_size, group.d_alpha_desc, rep_size,
                &zero_f, d_inner_F_elem, query_count));

            {
                long long total = (long long)query_count * group.count;
                int blk = 256;
                int grd = (int)((total + blk - 1) / blk);
                inference_build_expd_iF_kernel<<<grd, blk>>>(
                    d_C_elem, d_inner_F_elem, group.d_S_adF, d_expd_iF_elem, d_row_expd_iF_elem,
                    inv_s2, query_count, group.count);
            }

            CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                rep_size, query_count, group.count,
                &one_f, group.d_combined_t, rep_size, d_C_elem, query_count,
                &zero_f, d_G_acc_elem, rep_size));

            CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                rep_size, query_count, group.count,
                &neg1, group.d_X_t, rep_size, d_expd_iF_elem, query_count,
                &one_f, d_G_acc_elem, rep_size));

            {
                dim3 blk(16, 16), grd((query_count + 15) / 16, (group.count + 15) / 16);
                scatter_rect_tile_kernel<<<grd, blk>>>(
                    d_C_elem, d_C_qt, d_query_indices, group.d_indices,
                    query_count, group.count, query_count, N_q);
                scatter_rect_tile_kernel<<<grd, blk>>>(
                    d_inner_F_elem, d_inner_F, d_query_indices, group.d_indices,
                    query_count, group.count, query_count, N_q);
            }

            {
                long long total = (long long)query_count * rep_size;
                scatter_g_acc_tile_kernel<<<(int)((total + 255) / 256), 256>>>(
                    d_G_acc_elem, d_G_acc, d_query_indices, query_count, rep_size);
            }

            scatter_row_sums_kernel<<<(query_count + 255) / 256, 256>>>(
                d_row_expd_iF_elem, d_row_expd_iF, d_query_indices, query_count);
        }

        // 4. G_acc = C_qt @ combined_t   (uses precomputed d_combined_t)
        // 5. E_partial and wE. row_sum_expd_iF already built.
        {
            dim3 blk_e(32, 1);
            dim3 grd_e((N_q + 31) / 32, ((nm_t * max_atoms_t) + 255) / 256);
            inference_E_and_diag_kernel<<<grd_e, blk_e>>>(
                d_C_qt, d_inner_F, d_alpha_E_t,
                d_E_partial, d_wE_arr, d_row_expd_iF,
                sigma * sigma,
                N_q, nm_t * max_atoms_t);
        }

        // 6. Diagonal correction: G_acc[q,k] += (row_expd_iF[q] - wE[q]) * X_q[q,k]
        {
            long long total = (long long)N_q * rep_size;
            inference_G_diag_correction_kernel<<<(int)((total + 255) / 256), 256>>>(
                d_G_acc, d_X_q, d_row_expd_iF, d_wE_arr,
                N_q, rep_size);
        }
    }

    // Phase 2: reduce E_partial → E_pred (sum over query atoms per molecule)
    {
        int reduce_block = ((max_atoms_q + 31) / 32) * 32;
        if (reduce_block < 32)  reduce_block = 32;
        if (reduce_block > 256) reduce_block = 256;
        size_t smem_r = (size_t)reduce_block * sizeof(float);
        local_energy_reduce_kernel<<<nm_q, reduce_block, smem_r>>>(
            d_E_partial, d_N_q, d_E_pred, nm_q, max_atoms_q);
    }

    // Phase 3: back-project G_acc → F_pred via dX_q^T @ G_acc per query molecule
    {
        int ncols_max = 3 * max_atoms_q;
        int bp_block  = 256;
        dim3 bp_grid(ncols_max, nm_q);
        local_force_backproject_kernel<<<bp_grid, bp_block>>>(
            d_dX_q, d_G_acc, d_N_q, d_offs_q,
            d_F_pred, nm_q, max_atoms_q, rep_size);
    }
}


// ---------------------------------------------------------------------------
// kernel_gaussian_symm_local_cu — energy-only K_EE (nm × nm)
//
// Builds the symmetric energy kernel using C_label (Stage 1) + block-sum.
// Much cheaper than kernel_gaussian_full_symm which builds the full (nm+naq)².
// ---------------------------------------------------------------------------
void kernel_gaussian_symm_local_cu(
    const float *d_X,
    const int   *d_Q,
    const int   *d_N,
    float       *d_KEE,
    float        sigma,
    int nm, int max_atoms, int rep_size)
{
    ensure_cublas();

    int N = nm * max_atoms;
    float inv_s2 = 1.0f / (sigma * sigma);
    float sigma2 = sigma * sigma;
    const float neg2 = -2.0f, zero_f = 0.0f;

    // Zero K_EE
    CUDA_CHECK(cudaMemset(d_KEE, 0, (long long)nm * nm * sizeof(float)));

    // Precompute all squared norms
    float *d_norms;
    CUDA_CHECK(cudaMalloc(&d_norms, N * sizeof(float)));
    compute_sqnorms_kernel<<<(N + 255) / 256, 256>>>(d_X, d_norms, rep_size, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download Q and N to host for element screening
    int *h_Q = (int*)malloc(N * sizeof(int));
    int *h_N = (int*)malloc(nm * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_Q, d_Q, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_N, d_N, nm * sizeof(int), cudaMemcpyDeviceToHost));

    // Collect distinct element labels
    std::vector<int> labels;
    for (int m = 0; m < nm; m++)
        for (int i = 0; i < h_N[m]; i++) {
            int q = h_Q[m * max_atoms + i];
            bool found = false;
            for (int l : labels) if (l == q) { found = true; break; }
            if (!found) labels.push_back(q);
        }

    // For each element: gather flat atom indices, build per-element C, reduce to K_EE.
    // C_elem is (R × R) where R = number of atoms of this element.
    // This avoids computing cross-element inner products entirely.
    for (int label : labels) {
        // Gather flat indices of atoms with this label
        std::vector<int> atom_indices;        // flat atom index in d_X
        std::vector<int> mol_of_atom;         // which molecule this atom belongs to
        for (int m = 0; m < nm; m++)
            for (int i = 0; i < h_N[m]; i++)
                if (h_Q[m * max_atoms + i] == label) {
                    atom_indices.push_back(m * max_atoms + i);
                    mol_of_atom.push_back(m);
                }
        int R = (int)atom_indices.size();
        if (R == 0) continue;

        // Upload atom indices to GPU and gather X rows + norms via GPU kernels
        int *d_atom_indices;
        CUDA_CHECK(cudaMalloc(&d_atom_indices, R * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_atom_indices, atom_indices.data(), R * sizeof(int), cudaMemcpyHostToDevice));

        float *d_X_elem;
        CUDA_CHECK(cudaMalloc(&d_X_elem, (long long)R * rep_size * sizeof(float)));
        gather_rows_kernel<<<((long long)R * rep_size + 255) / 256, 256>>>(
            d_X, d_X_elem, d_atom_indices, R, rep_size);

        float *d_norms_elem;
        CUDA_CHECK(cudaMalloc(&d_norms_elem, R * sizeof(float)));
        gather_scalars_kernel<<<(R + 255) / 256, 256>>>(
            d_norms, d_norms_elem, d_atom_indices, R);

        cudaFree(d_atom_indices);

        // Tile the R×R SGEMM to limit memory
        long long max_tile_mem = 1024LL * 1024 * 1024 / sizeof(float);
        int tile_R = R;
        while ((long long)tile_R * tile_R > max_tile_mem && tile_R > 1)
            tile_R /= 2;

        float *d_C_elem;
        CUDA_CHECK(cudaMalloc(&d_C_elem, (long long)tile_R * tile_R * sizeof(float)));

        // Pre-allocate mol index buffer for scatter (2 * tile_R ints)
        int *d_mol_elem;
        CUDA_CHECK(cudaMalloc(&d_mol_elem, 2 * tile_R * sizeof(int)));

        for (int ri = 0; ri < R; ri += tile_R) {
            int ni = std::min(tile_R, R - ri);
            for (int rj = 0; rj <= ri; rj += tile_R) {
                int nj = std::min(tile_R, R - rj);
                if (rj + nj > ri + ni) nj = ri + ni - rj;

                CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    ni, nj, rep_size,
                    &neg2,
                    d_X_elem + (long long)ri * rep_size, rep_size,
                    d_X_elem + (long long)rj * rep_size, rep_size,
                    &zero_f, d_C_elem, ni));

                // Apply norms + exp (no label check needed — all same element)
                // Apply norms and exponentiate in-place:
                // C[i,j] = exp(-0.5 * inv_s2 * (C[i,j] + norms[ri+i] + norms[rj+j])) * inv_s2
                // No label check needed since all atoms have the same element.
                {
                    dim3 blk(16, 16), grd((ni + 15) / 16, (nj + 15) / 16);
                    apply_norms_exp_kernel<<<grd, blk>>>(
                        d_C_elem, d_norms_elem + ri, d_norms_elem + rj,
                        inv_s2, ni, nj, 0);
                }

                // Reduce to K_EE: for each (i_local, j_local) in the tile,
                // mol_a = mol_of_atom[ri + i_local], mol_b = mol_of_atom[rj + j_local].
                // K_EE[mol_a, mol_b] += sigma2 * C_elem[i_local, j_local]
                // Need mol_of_atom on GPU.
                // Upload mol_of_atom indices for this tile.
                {
                    CUDA_CHECK(cudaMemcpy(d_mol_elem, mol_of_atom.data() + ri, ni * sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_mol_elem + tile_R, mol_of_atom.data() + rj, nj * sizeof(int), cudaMemcpyHostToDevice));

                    // Off-diagonal tiles represent only one half of the symmetric
                    // atom-pair space, so mirror them into K_EE. Diagonal tiles are
                    // computed as full dense blocks here, so mirroring would double-count.
                    dim3 blk2(16, 16), grd2((ni + 15) / 16, (nj + 15) / 16);
                    scatter_C_to_KEE_kernel<<<grd2, blk2>>>(
                        d_C_elem, d_mol_elem, d_mol_elem + tile_R,
                        d_KEE, sigma2,
                        ni, nj, nm, ri != rj ? 1 : 0);

                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }
        }

        cudaFree(d_X_elem);
        cudaFree(d_norms_elem);
        cudaFree(d_C_elem);
        cudaFree(d_mol_elem);
    }

    free(h_Q);
    free(h_N);
    cudaFree(d_norms);
}

// ---------------------------------------------------------------------------
// rfp_index_lower_N_local — device helper for RFP packed format (TRANSR=N, UPLO=L)
//
// Returns the flat RFP index for element (row, col) with row >= col.
//
// Even n (nk = n/2, column stride = n+1):
//   j < nk  →  1 + i + j*(n+1)
//   j >= nk →  (j-nk) + (i-nk)*(n+1)
//
// Odd n (n1 = ceil(n/2), column stride = n):
//   j < n1  →  i + j*n
//   j >= n1 →  n + (j-n1) + (i-n1)*n
// ---------------------------------------------------------------------------
__device__ static long long rfp_index_lower_N_local(int n, int row, int col)
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


// ---------------------------------------------------------------------------
// scatter_C_to_KEE_rfp_kernel — scatter C_elem tile directly into RFP buffer.
//
// C_elem (ni×nj col-major) holds atom-pair kernel values for one element tile.
// mol_i[i] / mol_j[j] give the molecule indices for each row/col atom.
//
// Off-diagonal tiles (diagonal_tile=0, ri!=rj):
//   Each atom pair (i,j) is unique — the transposed tile (rj,ri) is never
//   computed (loop rj<=ri).  Scatter to lower-triangle slot max/min.
//
// Diagonal tiles (diagonal_tile=1, ri==rj):
//   Both C_elem[i,j] and C_elem[j,i] are present and equal.  For a
//   different-molecule pair (mol_a != mol_b) both map to the same RFP slot,
//   so we guard on mol_a >= mol_b to contribute exactly once.
//   For same-molecule pairs (mol_a == mol_b) all (i,j) atom pairs are
//   distinct contributions and must all accumulate, so no guard is applied.
// ---------------------------------------------------------------------------
__global__ static void scatter_C_to_KEE_rfp_kernel(
    const float *d_C,          // (ni, nj) col-major
    const int   *d_mol_i,      // (ni,) molecule index for each row atom
    const int   *d_mol_j,      // (nj,) molecule index for each col atom
    float       *d_K_rfp,      // nm*(nm+1)/2 RFP output (TRANSR=N, UPLO=L)
    float        sigma2,
    int ni, int nj, int nm,
    int diagonal_tile)         // 1 if ri==rj, 0 for off-diagonal tiles
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ni || j >= nj) return;

    float val = d_C[i + (long long)j * ni] * sigma2;
    if (val == 0.0f) return;

    int mol_a = d_mol_i[i];
    int mol_b = d_mol_j[j];

    /* For diagonal tiles, skip upper-triangle mol pairs to avoid double-count */
    if (diagonal_tile && mol_a < mol_b) return;

    int row = mol_a >= mol_b ? mol_a : mol_b;
    int col = mol_a >= mol_b ? mol_b : mol_a;

    atomicAdd(&d_K_rfp[rfp_index_lower_N_local(nm, row, col)], val);
}


// ---------------------------------------------------------------------------
// kernel_gaussian_symm_rfp_local_cu — energy-only K_EE in RFP packed format.
//
// Mirrors kernel_gaussian_symm_local_cu but writes directly into the RFP
// buffer (TRANSR=N, UPLO=L) without building a dense nm×nm intermediate.
//
// Memory saving vs. pack-after-dense: peak extra allocation is nm*(nm+1)/2
// floats (the output itself) rather than nm×nm.  The per-tile C_elem buffer
// is identical in both approaches.
//
// Convention: TRANSR=N, UPLO=L  →  unpack in Python with uplo='U', transr='N'.
// d_K_rfp must point to nm*(nm+1)/2 floats.
// ---------------------------------------------------------------------------
void kernel_gaussian_symm_rfp_local_cu(
    const float *d_X,
    const int   *d_Q,
    const int   *d_N,
    float       *d_K_rfp,
    float        sigma,
    int nm, int max_atoms, int rep_size)
{
    ensure_cublas();

    int N = nm * max_atoms;
    int nt = nm * (nm + 1) / 2;
    float inv_s2 = 1.0f / (sigma * sigma);
    float sigma2 = sigma * sigma;
    const float neg2 = -2.0f, zero_f = 0.0f;

    /* Zero the RFP output */
    CUDA_CHECK(cudaMemset(d_K_rfp, 0, (long long)nt * sizeof(float)));

    /* Precompute all squared norms */
    float *d_norms;
    CUDA_CHECK(cudaMalloc(&d_norms, N * sizeof(float)));
    compute_sqnorms_kernel<<<(N + 255) / 256, 256>>>(d_X, d_norms, rep_size, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Download Q and N to host for element screening */
    int *h_Q = (int*)malloc(N * sizeof(int));
    int *h_N = (int*)malloc(nm * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_Q, d_Q, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_N, d_N, nm * sizeof(int), cudaMemcpyDeviceToHost));

    /* Collect distinct element labels */
    std::vector<int> labels;
    for (int m = 0; m < nm; m++)
        for (int i = 0; i < h_N[m]; i++) {
            int q = h_Q[m * max_atoms + i];
            bool found = false;
            for (int l : labels) if (l == q) { found = true; break; }
            if (!found) labels.push_back(q);
        }

    for (int label : labels) {
        /* Gather flat atom indices and molecule indices for this element */
        std::vector<int> atom_indices;
        std::vector<int> mol_of_atom;
        for (int m = 0; m < nm; m++)
            for (int i = 0; i < h_N[m]; i++)
                if (h_Q[m * max_atoms + i] == label) {
                    atom_indices.push_back(m * max_atoms + i);
                    mol_of_atom.push_back(m);
                }
        int R = (int)atom_indices.size();
        if (R == 0) continue;

        /* Upload atom indices; gather X rows and norms on GPU */
        int *d_atom_indices;
        CUDA_CHECK(cudaMalloc(&d_atom_indices, R * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_atom_indices, atom_indices.data(), R * sizeof(int), cudaMemcpyHostToDevice));

        float *d_X_elem;
        CUDA_CHECK(cudaMalloc(&d_X_elem, (long long)R * rep_size * sizeof(float)));
        gather_rows_kernel<<<((long long)R * rep_size + 255) / 256, 256>>>(
            d_X, d_X_elem, d_atom_indices, R, rep_size);

        float *d_norms_elem;
        CUDA_CHECK(cudaMalloc(&d_norms_elem, R * sizeof(float)));
        gather_scalars_kernel<<<(R + 255) / 256, 256>>>(
            d_norms, d_norms_elem, d_atom_indices, R);

        cudaFree(d_atom_indices);

        /* Tile the lower-triangle R×R SGEMM to limit memory */
        long long max_tile_mem = 1024LL * 1024 * 1024 / sizeof(float);
        int tile_R = R;
        while ((long long)tile_R * tile_R > max_tile_mem && tile_R > 1)
            tile_R /= 2;

        float *d_C_elem;
        CUDA_CHECK(cudaMalloc(&d_C_elem, (long long)tile_R * tile_R * sizeof(float)));

        int *d_mol_elem;
        CUDA_CHECK(cudaMalloc(&d_mol_elem, 2 * tile_R * sizeof(int)));

        for (int ri = 0; ri < R; ri += tile_R) {
            int ni = std::min(tile_R, R - ri);
            for (int rj = 0; rj <= ri; rj += tile_R) {
                int nj = std::min(tile_R, R - rj);
                if (rj + nj > ri + ni) nj = ri + ni - rj;

                /* SGEMM: C_elem = -2 * X[ri-group]^T * X[rj-group] (col-major, ni×nj) */
                CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    ni, nj, rep_size,
                    &neg2,
                    d_X_elem + (long long)ri * rep_size, rep_size,
                    d_X_elem + (long long)rj * rep_size, rep_size,
                    &zero_f, d_C_elem, ni));

                /* Apply squared norms + exp: C_elem[i,j] = exp(-0.5*inv_s2*(C+n_i+n_j))*inv_s2 */
                {
                    dim3 blk(16, 16), grd((ni + 15) / 16, (nj + 15) / 16);
                    apply_norms_exp_kernel<<<grd, blk>>>(
                        d_C_elem, d_norms_elem + ri, d_norms_elem + rj,
                        inv_s2, ni, nj, 0);
                }

                /* Scatter directly to RFP.
                 * Diagonal tiles: guard mol_a >= mol_b to avoid double-counting
                 *   different-molecule pairs (C_elem[i,j] and C_elem[j,i] are
                 *   equal and both present in the full-square tile).
                 * Off-diagonal tiles: always scatter; the transposed tile (rj,ri)
                 *   is never computed (loop rj<=ri), so no double-counting. */
                {
                    CUDA_CHECK(cudaMemcpy(d_mol_elem,          mol_of_atom.data() + ri, ni * sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_mol_elem + tile_R, mol_of_atom.data() + rj, nj * sizeof(int), cudaMemcpyHostToDevice));

                    dim3 blk2(16, 16), grd2((ni + 15) / 16, (nj + 15) / 16);
                    scatter_C_to_KEE_rfp_kernel<<<grd2, blk2>>>(
                        d_C_elem, d_mol_elem, d_mol_elem + tile_R,
                        d_K_rfp, sigma2,
                        ni, nj, nm, ri == rj ? 1 : 0);

                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }
        }

        cudaFree(d_X_elem);
        cudaFree(d_norms_elem);
        cudaFree(d_C_elem);
        cudaFree(d_mol_elem);
    }

    free(h_Q);
    free(h_N);
    cudaFree(d_norms);
}


// ---------------------------------------------------------------------------
// kernel_gaussian_rect_local_cu — rectangular energy-only K_EE (nm_q × nm_t)
// ---------------------------------------------------------------------------
void kernel_gaussian_rect_local_cu(
    const float *d_X_q,
    const int   *d_Q_q,
    const int   *d_N_q,
    const float *d_X_t,
    const int   *d_Q_t,
    const int   *d_N_t,
    float       *d_KEE,
    float        sigma,
    int nm_q, int nm_t,
    int max_atoms_q, int max_atoms_t,
    int rep_size)
{
    ensure_cublas();

    int N_q = nm_q * max_atoms_q;
    int N_t = nm_t * max_atoms_t;
    float inv_s2 = 1.0f / (sigma * sigma);
    float sigma2 = sigma * sigma;
    const float neg2 = -2.0f, zero_f = 0.0f;

    float *d_C, *d_norms_q, *d_norms_t;
    CUDA_CHECK(cudaMalloc(&d_C,       (long long)N_q * N_t * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms_q, N_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms_t, N_t * sizeof(float)));

    compute_sqnorms_kernel<<<(N_q + 255) / 256, 256>>>(d_X_q, d_norms_q, rep_size, N_q);
    compute_sqnorms_kernel<<<(N_t + 255) / 256, 256>>>(d_X_t, d_norms_t, rep_size, N_t);

    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N_q, N_t, rep_size,
        &neg2, d_X_q, rep_size, d_X_t, rep_size,
        &zero_f, d_C, N_q));

    {
        dim3 blk(16, 16), grd((N_q + 15) / 16, (N_t + 15) / 16);
        build_C_qt_kernel<<<grd, blk>>>(
            d_C, d_norms_q, d_norms_t,
            d_Q_q, d_Q_t, d_N_q, d_N_t,
            inv_s2, N_q, N_t, max_atoms_q, max_atoms_t);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_norms_q);
    cudaFree(d_norms_t);

    // Reduce C_qt to K_EE
    {
        dim3 grid(nm_q, nm_t);
        build_KEE_rect_kernel<<<grid, 256>>>(
            d_C, d_N_q, d_N_t, d_KEE, sigma2,
            nm_q, nm_t, max_atoms_q, max_atoms_t, N_q);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_C);
}

// ===========================================================================
// kernel_gaussian_full_symm_rfp_local_cu
// ===========================================================================
//
// RFP-output variant of kernel_gaussian_full_symm_local_cu.
//
// Builds the symmetric energy+force training kernel matrix K_full of shape
// (BIG, BIG) where BIG = nm + naq, stored directly in RFP packed format
// (TRANSR=N, UPLO=L) — *without* materialising the dense BIG×BIG buffer.
//
// Stages 1, 2, 3 (build C_label, build per-pair K_FF static via batched
// SGEMM, build P_ab/P_ba/S/WA/VB and rank-1 SGEMM correction) are identical
// to the dense version.  The differences are:
//
//   * The K_EE / K_FE / K_EF writes inside prepare_WA_VB_and_scalars are
//     replaced with atomicAdd into d_K_rfp at indices computed via
//     rfp_index_lower_N_local(BIG, row, col).  Since each pair (a, b) with
//     a >= b contributes to disjoint global slots there are no actual
//     conflicts; we still use atomicAdd defensively (single thread writes
//     after a __syncthreads()).
//   * scatter_K_FF_rfp_kernel writes per-pair K_FF blocks directly into
//     RFP slots with proper diagonal-pair lower-triangle masking.
//
// Convention: TRANSR='N', UPLO='L'.  Unpack with kernelmath.rfp_to_full(
//   K_rfp, BIG, uplo='U', transr='N').
//
// Memory: peak intermediate K_FF_pairs is unchanged (n_pairs*ncols_max^2);
// the BIG×BIG output is replaced with BIG*(BIG+1)/2 floats.
// ===========================================================================

// ---------------------------------------------------------------------------
// prepare_WA_VB_and_scalars_rfp_kernel
//
// Identical to prepare_WA_VB_and_scalars_kernel except the final K_EE/K_FE/
// K_EF writes go into the RFP-packed buffer d_K_rfp.
// ---------------------------------------------------------------------------
__global__ static void prepare_WA_VB_and_scalars_rfp_kernel(
    const float *d_C,
    const float *d_S,
    const float *d_P_ab,
    const float *d_P_ba,
    const int   *d_N,
    const int   *d_offs,
    float       *d_K_rfp,    // BIG*(BIG+1)/2 RFP buffer (TRANSR=N, UPLO=L)
    float       *d_WA,
    float       *d_VB,
    float sigma2, float inv_s2,
    int nm, int max_atoms, int lda, int N_total, int BIG,
    int na_uniform, int nb_uniform,
    int chunk_start_pair, int chunk_n_pairs)
{
    int local_pair = (int)blockIdx.x;
    if (local_pair < 0 || local_pair >= chunk_n_pairs) return;

    int pair_idx = chunk_start_pair + local_pair;
    int b = (int)(sqrtf(2.0f * (float)pair_idx + 0.25f));
    while ((long long)(b + 1) * (b + 2) / 2 <= pair_idx) ++b;
    while ((long long)b * (b + 1) / 2 > pair_idx) --b;
    int a = pair_idx - b * (b + 1) / 2;
    if (a > b || b >= nm || a >= nm) return;

    int na = d_N[a];
    int nb = d_N[b];
    if (na <= 0 || nb <= 0) return;

    int ncols_a   = 3 * na;
    int ncols_b   = 3 * nb;
    int col_off_a = d_offs[a];
    int row_off_b = d_offs[b];
    int ncM       = 3 * max_atoms;

    int nap = na_uniform;
    int nbp = nb_uniform;
    int ncols_ap = 3 * nap;
    int ncols_bp = 3 * nbp;
    int nab_p    = nap * nbp;

    extern __shared__ float sh[];
    int scalar_acc_sz = 1 + 2 * ncM;

    float *sh_KEE   = sh;
    float *sh_KFE   = sh + 1;
    float *sh_Kjact = sh + 1 + ncM;

    for (int t = threadIdx.x; t < scalar_acc_sz; t += blockDim.x)
        sh[t] = 0.0f;
    __syncthreads();

    long long P_ab_per_pair = (long long)nap * ncols_ap * nbp;
    long long P_ba_per_pair = (long long)nbp * ncols_bp * nap;
    long long P_ab_base = (long long)local_pair * P_ab_per_pair;
    long long P_ba_base = (long long)local_pair * P_ba_per_pair;

    long long WA_per_pair = (long long)nab_p * ncols_ap;
    long long VB_per_pair_sz = (long long)nab_p * ncols_bp;
    long long WA_base = (long long)local_pair * WA_per_pair;
    long long VB_base = (long long)local_pair * VB_per_pair_sz;

    int flat_a_base = a * max_atoms;
    int flat_b_base = b * max_atoms;

    for (int pair = (int)threadIdx.x; pair < nab_p; pair += (int)blockDim.x) {
        int i1 = pair / nbp;
        int j2 = pair % nbp;

        int flat_i = flat_a_base + i1;
        int flat_j = flat_b_base + j2;
        float expdiag = d_C[flat_i + (long long)flat_j * N_total];

        if (expdiag == 0.0f) {
            for (int c = 0; c < ncols_ap; c++)
                d_WA[WA_base + (long long)c * nab_p + pair] = 0.0f;
            for (int c = 0; c < ncols_bp; c++)
                d_VB[VB_base + (long long)c * nab_p + pair] = 0.0f;
            continue;
        }

        float exp_base = expdiag * sigma2;
        float expd     = -expdiag * inv_s2;

        long long pba_for_va = P_ba_base + (long long)i1 * ncols_bp * nap + (long long)j2 * ncols_bp;
        long long pab_for_vb = P_ab_base + (long long)j2 * ncols_ap * nbp + (long long)i1 * ncols_ap;

        for (int c = 0; c < ncols_ap; c++) {
            float va = d_S[flat_i * lda + c] - d_P_ba[pba_for_va + c];
            d_WA[WA_base + (long long)c * nab_p + pair] = expd * va;
        }
        for (int c = 0; c < ncols_bp; c++) {
            float vb = d_P_ab[pab_for_vb + c] - d_S[flat_j * lda + c];
            d_VB[VB_base + (long long)c * nab_p + pair] = vb;
        }

        atomicAdd(sh_KEE, exp_base);
        for (int c = 0; c < ncols_a; c++) {
            float va = d_S[flat_i * lda + c] - d_P_ba[pba_for_va + c];
            atomicAdd(sh_KFE + c, -expdiag * va);
        }
        for (int c = 0; c < ncols_b; c++) {
            float vb = d_P_ab[pab_for_vb + c] - d_S[flat_j * lda + c];
            atomicAdd(sh_Kjact + c, expdiag * vb);
        }
    }

    __syncthreads();

    /* RFP writes — pair convention here is a <= b:
     *   K_EE:  dense writes (a,b) and mirror (b,a); lower-triangle slot is (b,a).
     *   K_FE:  global (nm+col_off_a+c1, b); since nm > b, row > col → lower.
     *   K_EF:  off-diagonal only (a != b); dense writes (a, nm+row_off_b+c2);
     *          col > row, so lower-triangle slot is (nm+row_off_b+c2, a).
     */
    if (threadIdx.x == 0) {
        float kee = sh_KEE[0];
        atomicAdd(&d_K_rfp[rfp_index_lower_N_local(BIG, b, a)], kee);
    }

    for (int c1 = (int)threadIdx.x; c1 < ncols_a; c1 += (int)blockDim.x) {
        float v = sh_KFE[c1];
        int row = nm + col_off_a + c1;
        int col = b;
        atomicAdd(&d_K_rfp[rfp_index_lower_N_local(BIG, row, col)], v);
    }

    if (a != b) {
        for (int c2 = (int)threadIdx.x; c2 < ncols_b; c2 += (int)blockDim.x) {
            float v = sh_Kjact[c2];
            int row = nm + row_off_b + c2;   /* > a */
            int col = a;
            atomicAdd(&d_K_rfp[rfp_index_lower_N_local(BIG, row, col)], v);
        }
    }
}


// ---------------------------------------------------------------------------
// scatter_K_FF_rfp_kernel
//
// Per-pair K_FF blocks into the RFP buffer.  Mirrors scatter_K_FF_kernel.
//
// Pair convention: a >= b, so col_off_a >= row_off_b.
//
// Off-diagonal pairs (a != b):
//   row = nm + col_off_a + c1, col = nm + row_off_b + c2.
//   col_off_a > row_off_b → row > col strictly → lower triangle, write once.
//
// Diagonal pairs (a == b):
//   row = nm + col_off_a + c1, col = nm + col_off_a + c2.
//   Lower triangle iff c1 >= c2; write once for that half (the dense version
//   wrote both (c1,c2) and the mirrored (c2,c1) — both map to the same RFP
//   slot here, so we skip c1 < c2).
// ---------------------------------------------------------------------------
__global__ static void scatter_K_FF_rfp_kernel(
    const float *d_K_FF_pairs,
    const int   *d_N,
    const int   *d_offs,
    float       *d_K_rfp,
    int nm, int max_atoms, int ncols_max, int BIG,
    int pair_offset)
{
    int pair_local = blockIdx.x;
    int pair_idx   = pair_local + pair_offset;

    int a = (int)(sqrtf(2.0f * (float)pair_idx + 0.25f));
    while ((long long)(a + 1) * (a + 2) / 2 <= pair_idx) ++a;
    while ((long long)a       * (a + 1) / 2 >  pair_idx) --a;
    int b = pair_idx - a * (a + 1) / 2;
    if (a >= nm) return;

    int na = d_N[a], nb = d_N[b];
    int ncols_a = 3 * na, ncols_b = 3 * nb;
    int col_off_a = d_offs[a], row_off_b = d_offs[b];

    /* d_K_FF_pairs is indexed by *local* (chunk-relative) pair index. */
    const float *kff = d_K_FF_pairs + (long long)pair_local * ncols_max * ncols_max;

    for (int t = threadIdx.x; t < ncols_a * ncols_b; t += blockDim.x) {
        int c1 = t / ncols_b;
        int c2 = t % ncols_b;
        float v = kff[c1 * ncols_max + c2];
        int row = nm + col_off_a + c1;
        int col = nm + row_off_b + c2;
        if (a == b) {
            if (c1 < c2) continue;
        }
        atomicAdd(&d_K_rfp[rfp_index_lower_N_local(BIG, row, col)], v);
    }
}


// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm_rfp_local_cu — host driver
//
// Identical control flow to kernel_gaussian_full_symm_local_cu but writes
// directly into RFP-packed K_rfp instead of the dense BIG×BIG K_full.
// ---------------------------------------------------------------------------
void kernel_gaussian_full_symm_rfp_local_cu(
    const float *d_X,
    const float *d_dX,
    const int   *d_Q,   // currently unused (label screening lives in C_label)
    const int   *d_N,
    float       *d_K_rfp,
    float        sigma,
    int nm, int max_atoms, int rep_size, int naq)
{
    (void)d_Q;
    ensure_cublas();

    int BIG = nm + naq;
    long long nt = (long long)BIG * (BIG + 1) / 2;
    int N   = nm * max_atoms;
    float inv_s2  =  1.0f / (sigma * sigma);
    int ncols_max = 3 * max_atoms;
    const float neg2 = -2.0f, zero_f = 0.0f, one_f = 1.0f;

    int *h_N = (int*)malloc(nm * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_N, d_N, nm * sizeof(int), cudaMemcpyDeviceToHost));

    int *d_offs;
    CUDA_CHECK(cudaMalloc(&d_offs, nm * sizeof(int)));
    build_offsets_from_host(h_N, d_offs, nm);

    /* Zero the RFP output */
    CUDA_CHECK(cudaMemset(d_K_rfp, 0, (size_t)nt * sizeof(float)));

    cudaEvent_t ev_start, ev_s1, ev_s2_total, ev_s3_total, ev_s4_total;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_s1));
    CUDA_CHECK(cudaEventCreate(&ev_s2_total));
    CUDA_CHECK(cudaEventCreate(&ev_s3_total));
    CUDA_CHECK(cudaEventCreate(&ev_s4_total));
    CUDA_CHECK(cudaEventRecord(ev_start));

    /* ===== Stage 1: build C_label (NxN) ===== */
    float *d_C, *d_norms;
    CUDA_CHECK(cudaMalloc(&d_C,     (long long)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms, (long long)N * sizeof(float)));

    compute_sqnorms_kernel<<<(N + 255) / 256, 256>>>(d_X, d_norms, rep_size, N);

    CUBLAS_CHECK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, rep_size,
        &neg2, d_X, rep_size, d_X, rep_size,
        &zero_f, d_C, N));
    {
        dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
        build_C_label_kernel<<<grd, blk>>>(d_C, d_norms, d_Q, d_N, inv_s2, N, max_atoms);
    }
    CUDA_CHECK(cudaEventRecord(ev_s1));
    cudaFree(d_norms);

    /* ===== Pair indices, persistent ===== */
    long long n_pairs = (long long)nm * (nm + 1) / 2;
    int lda = 3 * max_atoms;
    int rl = rep_size * lda;

    std::vector<int> pair_a;
    std::vector<int> pair_b;
    build_pair_indices(pair_a, pair_b, nm);

    int *d_pair_a = nullptr;
    int *d_pair_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pair_a, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_b, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_a, pair_a.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pair_b, pair_b.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    bool uniform = true;
    int na0 = h_N[0];
    for (int m = 1; m < nm; m++)
        if (h_N[m] != na0) { uniform = false; break; }

    /* ===== Outer chunk loop: K_FF chunk + Stage 2 + Stage 3 + scatter ===== */
    /*
     * Memory budget per pair (in chunk):
     *   K_FF      : ncols_max^2
     *   P_ab/P_ba : 2 * max_atoms * ncols_max * max_atoms
     *   WA/VB     : 2 * (max_atoms*max_atoms) * ncols_max
     * Total floats/pair = ncols_max^2 + 4 * max_atoms^2 * ncols_max
     *
     * Chunk cap (bytes) controlled by env var KF_LOCAL_RFP_CHUNK_GB
     * (default 1 GiB).  Must hold ALL of the above per pair.
     */
    long long nab = (long long)max_atoms * max_atoms;
    long long K_FF_per_pair = (long long)ncols_max * ncols_max;
    long long P_ab_per_pair = (long long)max_atoms * ncols_max * max_atoms;
    long long P_ba_per_pair = P_ab_per_pair;
    long long WA_per_pair   = nab * ncols_max;
    long long VB_per_pair   = WA_per_pair;
    long long bytes_per_pair = (K_FF_per_pair + P_ab_per_pair + P_ba_per_pair
                                + WA_per_pair + VB_per_pair) * sizeof(float);

    long long chunk_cap_bytes = (long long)(1.0 * 1024.0 * 1024.0 * 1024.0);  /* 1 GiB default */
    if (const char *env = std::getenv("KF_LOCAL_RFP_CHUNK_GB")) {
        double gb = atof(env);
        if (gb > 0.0)
            chunk_cap_bytes = (long long)(gb * 1024.0 * 1024.0 * 1024.0);
    }
    int max_chunk_pairs = (int)std::min((long long)n_pairs,
                                          chunk_cap_bytes / std::max(bytes_per_pair, 1LL));
    if (max_chunk_pairs < 1) max_chunk_pairs = 1;

    /* Persistent chunk-sized buffers (allocated once, reused per chunk) */
    float *d_K_FF_chunk;
    CUDA_CHECK(cudaMalloc(&d_K_FF_chunk,
        (long long)max_chunk_pairs * K_FF_per_pair * sizeof(float)));

    float *d_P_ab, *d_P_ba, *d_WA, *d_VB_buf;
    CUDA_CHECK(cudaMalloc(&d_P_ab,   (long long)max_chunk_pairs * P_ab_per_pair * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P_ba,   (long long)max_chunk_pairs * P_ba_per_pair * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WA,     (long long)max_chunk_pairs * WA_per_pair   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_VB_buf, (long long)max_chunk_pairs * VB_per_pair   * sizeof(float)));

    /* Stage 2 workspace (cublas batched gemm intermediate W).
     * Cap controlled by env var KF_LOCAL_RFP_W_GB (default 0.5 GiB).
     * Lower values just mean more inner-batched SGEMM calls (negligible perf
     * impact); higher values trade VRAM for slightly larger batches. */
    long long W_per_pair = (long long)max_atoms * rl;
    long long W_cap_bytes = (long long)(0.5 * 1024.0 * 1024.0 * 1024.0);
    if (const char *envW = std::getenv("KF_LOCAL_RFP_W_GB")) {
        double gb = atof(envW);
        if (gb > 0.0)
            W_cap_bytes = (long long)(gb * 1024.0 * 1024.0 * 1024.0);
    }
    long long max_W_mem  = W_cap_bytes / sizeof(float);
    int max_batch = (int)std::min((long long)max_chunk_pairs,
                                    max_W_mem / std::max(W_per_pair, 1LL));
    if (max_batch < 1) max_batch = 1;
    float *d_W;
    CUDA_CHECK(cudaMalloc(&d_W, (long long)max_batch * W_per_pair * sizeof(float)));

    /* Stage 3 pointer arrays (sized for max chunk) */
    float *d_S;
    CUDA_CHECK(cudaMalloc(&d_S, (long long)N * lda * sizeof(float)));
    precompute_self_dots_kernel<<<(N + 255) / 256, 256>>>(
        d_X, d_dX, d_N, d_S, nm, max_atoms, rep_size, lda);

    int max_batch_pab = max_chunk_pairs * max_atoms;
    const float **d_Aptr_ab, **d_Bptr_ab;
    float **d_Cptr_ab;
    const float **d_Aptr_ba, **d_Bptr_ba;
    float **d_Cptr_ba;
    CUDA_CHECK(cudaMalloc(&d_Aptr_ab, max_batch_pab * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Bptr_ab, max_batch_pab * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Cptr_ab, max_batch_pab * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Aptr_ba, max_batch_pab * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Bptr_ba, max_batch_pab * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Cptr_ba, max_batch_pab * sizeof(float*)));

    /* Stage 2 batched-pointer arrays (sized for max batch) */
    const float **d_Aptr1, **d_Bptr1;
    float **d_Cptr1;
    const float **d_Aptr2, **d_Bptr2;
    float **d_Cptr2;
    CUDA_CHECK(cudaMalloc(&d_Aptr1, max_batch * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Bptr1, max_batch * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Cptr1, max_batch * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Aptr2, max_batch * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Bptr2, max_batch * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_Cptr2, max_batch * sizeof(float*)));

    int scalar_acc_sz = 1 + 2 * (3 * max_atoms);
    size_t smem_precomp = (size_t)scalar_acc_sz * sizeof(float);

    int max_atom_pairs = max_atoms * max_atoms;
    int block_sz = ((max_atom_pairs + 31) / 32) * 32;
    if (block_sz < 32)  block_sz = 32;
    if (block_sz > 128) block_sz = 128;

    float sigma2 = sigma * sigma;

    /* For non-uniform: pre-group pair indices by (na, nb) */
    struct PairGroup {
        int na = 0;
        int nb = 0;
        std::vector<int> pair_indices;
    };
    std::unordered_map<long long, PairGroup> groups;
    if (!uniform) {
        groups.reserve((size_t)max_atoms * max_atoms);
        int pidx = 0;
        for (int a = 0; a < nm; a++) {
            int na = h_N[a];
            for (int bb = 0; bb <= a; bb++, pidx++) {
                int nb = h_N[bb];
                if (na <= 0 || nb <= 0) continue;
                long long key = ((long long)na << 32) | (unsigned int)nb;
                auto &group = groups[key];
                if (group.pair_indices.empty()) {
                    group.na = na;
                    group.nb = nb;
                }
                group.pair_indices.push_back(pidx);
            }
        }
    }

    cudaEventRecord(ev_s2_total);  /* placeholder; we'll re-record after Stage 2 of last chunk */

    float t_s2_acc = 0.0f, t_s3_acc = 0.0f, t_s4_acc = 0.0f;
    cudaEvent_t e_a, e_b, e_c, e_d;
    cudaEventCreate(&e_a); cudaEventCreate(&e_b);
    cudaEventCreate(&e_c); cudaEventCreate(&e_d);

    for (long long chunk_start = 0; chunk_start < n_pairs; chunk_start += max_chunk_pairs) {
        int chunk_sz = (int)std::min((long long)max_chunk_pairs, n_pairs - chunk_start);

        /* Zero this chunk's K_FF */
        CUDA_CHECK(cudaMemsetAsync(d_K_FF_chunk, 0,
            (long long)chunk_sz * K_FF_per_pair * sizeof(float)));

        cudaEventRecord(e_a);

        /* ---------- Stage 2: K_FF static for this chunk ---------- */
        if (uniform && na0 > 0) {
            int na = na0, nb = na0;
            int ncols_a = 3 * na, ncols_b = 3 * nb;
            int K2 = na * rep_size;

            /* Inner batching: still cap by max_batch (W workspace size) */
            for (long long inner_start = 0; inner_start < chunk_sz; inner_start += max_batch) {
                int inner_sz = (int)std::min((long long)max_batch, (long long)chunk_sz - inner_start);

                std::vector<const float*> h_A1(inner_sz), h_B1(inner_sz);
                std::vector<float*> h_C1(inner_sz);
                std::vector<const float*> h_A2(inner_sz), h_B2(inner_sz);
                std::vector<float*> h_C2(inner_sz);

                for (int cidx = 0; cidx < inner_sz; ++cidx) {
                    long long local_idx  = inner_start + cidx;       /* index within chunk */
                    long long global_idx = chunk_start + local_idx;  /* index in pair_a/pair_b */
                    int aa = pair_a[(size_t)global_idx];
                    int bb = pair_b[(size_t)global_idx];

                    h_A1[cidx] = d_dX + (long long)(bb * max_atoms) * rl;
                    h_B1[cidx] = d_C + (long long)aa * max_atoms + (long long)bb * max_atoms * N;
                    h_C1[cidx] = d_W + (long long)cidx * W_per_pair;

                    h_A2[cidx] = d_W + (long long)cidx * W_per_pair;
                    h_B2[cidx] = d_dX + (long long)(aa * max_atoms) * rl;
                    h_C2[cidx] = d_K_FF_chunk + local_idx * K_FF_per_pair;
                }

                CUDA_CHECK(cudaMemcpy(d_Aptr1, h_A1.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_Bptr1, h_B1.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_Cptr1, h_C1.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_Aptr2, h_A2.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_Bptr2, h_B2.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_Cptr2, h_C2.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));

                CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    rl, na, nb,
                    &one_f, d_Aptr1, rl, d_Bptr1, N,
                    &zero_f, d_Cptr1, rl,
                    inner_sz));

                CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    ncols_b, ncols_a, K2,
                    &one_f, d_Aptr2, lda, d_Bptr2, lda,
                    &zero_f, d_Cptr2, ncols_max,
                    inner_sz));
            }
        } else {
            /* Non-uniform: for each (na,nb) group, filter to current chunk range. */
            long long chunk_end = chunk_start + chunk_sz;  /* exclusive */
            for (const auto &kv : groups) {
                const PairGroup &group = kv.second;
                int na = group.na;
                int nb = group.nb;
                int ncols_a = 3 * na;
                int ncols_b = 3 * nb;
                int K2 = na * rep_size;

                /* Collect indices in [chunk_start, chunk_end) */
                std::vector<int> sel;
                sel.reserve(group.pair_indices.size());
                for (int p : group.pair_indices) {
                    if ((long long)p >= chunk_start && (long long)p < chunk_end)
                        sel.push_back(p);
                }
                if (sel.empty()) continue;

                /* Inner batching cap by max_batch */
                long long W_group_stride = (long long)na * rl;
                int sel_sz = (int)sel.size();
                for (int inner_start = 0; inner_start < sel_sz; inner_start += max_batch) {
                    int inner_sz = std::min(max_batch, sel_sz - inner_start);

                    std::vector<const float*> h_A1(inner_sz), h_B1(inner_sz);
                    std::vector<float*> h_C1(inner_sz);
                    std::vector<const float*> h_A2(inner_sz), h_B2(inner_sz);
                    std::vector<float*> h_C2(inner_sz);

                    for (int idx = 0; idx < inner_sz; ++idx) {
                        int p = sel[(size_t)(inner_start + idx)];
                        long long local_idx = (long long)p - chunk_start;
                        int aa = pair_a[(size_t)p];
                        int bb = pair_b[(size_t)p];

                        h_A1[idx] = d_dX + (long long)(bb * max_atoms) * rl;
                        h_B1[idx] = d_C + (long long)aa * max_atoms + (long long)bb * max_atoms * N;
                        h_C1[idx] = d_W + (long long)idx * W_group_stride;

                        h_A2[idx] = d_W + (long long)idx * W_group_stride;
                        h_B2[idx] = d_dX + (long long)(aa * max_atoms) * rl;
                        h_C2[idx] = d_K_FF_chunk + local_idx * K_FF_per_pair;
                    }

                    CUDA_CHECK(cudaMemcpy(d_Aptr1, h_A1.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_Bptr1, h_B1.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_Cptr1, h_C1.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_Aptr2, h_A2.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_Bptr2, h_B2.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_Cptr2, h_C2.data(), inner_sz * sizeof(float*), cudaMemcpyHostToDevice));

                    CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                        rl, na, nb,
                        &one_f, d_Aptr1, rl, d_Bptr1, N,
                        &zero_f, d_Cptr1, rl,
                        inner_sz));

                    CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                        ncols_b, ncols_a, K2,
                        &one_f, d_Aptr2, lda, d_Bptr2, lda,
                        &zero_f, d_Cptr2, ncols_max,
                        inner_sz));
                }
            }
        }

        cudaEventRecord(e_b);

        /* ---------- Stage 3: P_ab/P_ba; K_EE/K_FE/K_EF -> RFP; rank-1 SGEMM ---------- */
        int batch_pab = chunk_sz * max_atoms;

        build_stage3_pointer_arrays_kernel<<<(batch_pab + 255) / 256, 256>>>(
            d_X, d_dX, d_P_ab, d_P_ba,
            d_pair_a, d_pair_b,
            d_Aptr_ab, d_Bptr_ab, d_Cptr_ab,
            d_Aptr_ba, d_Bptr_ba, d_Cptr_ba,
            rep_size, lda, max_atoms, ncols_max,
            (int)chunk_start, chunk_sz);

        CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            ncols_max, max_atoms, rep_size,
            &one_f, d_Aptr_ab, lda, d_Bptr_ab, rep_size,
            &zero_f, d_Cptr_ab, ncols_max,
            batch_pab));

        CUBLAS_CHECK(cublasSgemmBatched(s_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            ncols_max, max_atoms, rep_size,
            &one_f, d_Aptr_ba, lda, d_Bptr_ba, rep_size,
            &zero_f, d_Cptr_ba, ncols_max,
            batch_pab));

        prepare_WA_VB_and_scalars_rfp_kernel<<<chunk_sz, block_sz, smem_precomp>>>(
            d_C, d_S, d_P_ab, d_P_ba,
            d_N, d_offs, d_K_rfp,
            d_WA, d_VB_buf,
            sigma2, inv_s2,
            nm, max_atoms, lda, N, BIG,
            max_atoms, max_atoms,
            (int)chunk_start, chunk_sz);

        /* Rank-1 SGEMM: K_FF_chunk += WA^T * VB.  Note: writes into the chunk
           buffer starting at offset 0 (not chunk_start * ...). */
        CUBLAS_CHECK(cublasSgemmStridedBatched(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            ncols_max, ncols_max, (int)nab,
            &one_f,
            d_WA, (int)nab, WA_per_pair,
            d_VB_buf, (int)nab, VB_per_pair,
            &one_f,
            d_K_FF_chunk,
            ncols_max, K_FF_per_pair,
            chunk_sz));

        cudaEventRecord(e_c);

        /* ---------- Stage 4: scatter K_FF chunk into RFP ---------- */
        scatter_K_FF_rfp_kernel<<<chunk_sz, 256>>>(
            d_K_FF_chunk, d_N, d_offs, d_K_rfp,
            nm, max_atoms, ncols_max, BIG,
            (int)chunk_start);

        cudaEventRecord(e_d);
        cudaEventSynchronize(e_d);

        float t_ab, t_bc, t_cd;
        cudaEventElapsedTime(&t_ab, e_a, e_b);
        cudaEventElapsedTime(&t_bc, e_b, e_c);
        cudaEventElapsedTime(&t_cd, e_c, e_d);
        t_s2_acc += t_ab;
        t_s3_acc += t_bc;
        t_s4_acc += t_cd;
    }

    cudaEventRecord(ev_s4_total);
    CUDA_CHECK(cudaEventSynchronize(ev_s4_total));

    {
        float t1;
        cudaEventElapsedTime(&t1, ev_start, ev_s1);
        fprintf(stderr,
            "  [kernel_full_symm_rfp stages]  S1=%.1f ms  S2=%.1f ms  S3=%.1f ms  S4=%.1f ms  total=%.1f ms\n",
            t1, t_s2_acc, t_s3_acc, t_s4_acc, t1 + t_s2_acc + t_s3_acc + t_s4_acc);
        fprintf(stderr,
            "  [kernel_full_symm_rfp chunk]   pairs=%lld  chunk_pairs=%d  n_chunks=%lld  K_FF_chunk=%.2f MB\n",
            n_pairs, max_chunk_pairs,
            (n_pairs + max_chunk_pairs - 1) / max_chunk_pairs,
            (double)max_chunk_pairs * K_FF_per_pair * sizeof(float) / (1024.0 * 1024.0));
    }

    cudaEventDestroy(e_a); cudaEventDestroy(e_b);
    cudaEventDestroy(e_c); cudaEventDestroy(e_d);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_s1);
    cudaEventDestroy(ev_s2_total);
    cudaEventDestroy(ev_s3_total);
    cudaEventDestroy(ev_s4_total);

    cudaFree(d_Aptr_ab); cudaFree(d_Bptr_ab); cudaFree(d_Cptr_ab);
    cudaFree(d_Aptr_ba); cudaFree(d_Bptr_ba); cudaFree(d_Cptr_ba);
    cudaFree(d_Aptr1); cudaFree(d_Bptr1); cudaFree(d_Cptr1);
    cudaFree(d_Aptr2); cudaFree(d_Bptr2); cudaFree(d_Cptr2);
    cudaFree(d_S);
    cudaFree(d_P_ab);
    cudaFree(d_P_ba);
    cudaFree(d_WA);
    cudaFree(d_VB_buf);
    cudaFree(d_K_FF_chunk);
    cudaFree(d_W);
    cudaFree(d_offs);
    cudaFree(d_C);
    cudaFree(d_pair_a);
    cudaFree(d_pair_b);
    free(h_N);
}


}  // namespace fchl19
}  // namespace kf
