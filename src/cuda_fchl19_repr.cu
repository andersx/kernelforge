// cuda_fchl19_repr.cu — GPU-accelerated FCHL19 forward + gradient (FP32).
//
// Public functions (namespace kf::fchl19):
//
//   generate_fchl_acsf_cuda(coords, Q, N, nelements, ...)
//       Batched FCHL19 ACSF representation for nm padded molecules.
//       Returns (nm, max_atoms, rep_size) float32 CUDA tensor.
//
//   generate_fchl_acsf_and_gradients_cuda(coords, Q, N, nelements, ...)
//       Returns (rep, grad) where grad has shape
//       (nm, max_atoms, rep_size, max_atoms, 3).
//
// Parallelism:
//   Grid  = (nm * max_atoms,) blocks; each block owns one center atom (m, i).
//   Block = dim3(BLOCK_X=16, BLOCK_Y=8) = 128 threads.
//   Two-body  : threads stripe over neighbors j in [0, natoms) via tid.
//   Three-body: j strides threadIdx.x, k = j+1+threadIdx.y strides threadIdx.y;
//               no serial triplet builder, no full distance matrix.
//
// Shared memory layout per block (dynamic):
//   sh_coords  [max_atoms * 3]   float32
//   sh_Z       [max_atoms]       int32
//   sh_rep     [rep_size]        float32   (accumulator, atomicAdd into shmem)
//
// For rmd17 defaults (max_atoms=24, rep_size=312):
//   total shared memory ≈ 1.6 KB — well within the 48 KB limit.

#include "cuda_fchl19_repr.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            abort();                                                    \
        }                                                               \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define BLOCK_X      16
#define BLOCK_Y      8
#define BLOCK_SZ     (BLOCK_X * BLOCK_Y)
#define MAX_NABASIS  16        // compile-time upper bound for angular register array
#define SQRT_2PI_F   2.5066282746310002f
#define PI_F         3.14159265358979323846f
#define EPS_F        1e-10f

// ---------------------------------------------------------------------------
// Device helper: Euclidean distance between atoms a and b from shared coords
// ---------------------------------------------------------------------------
__device__ static inline float dist3_dev(const float * __restrict__ sh_coords, int a, int b)
{
    const float dx = sh_coords[b * 3 + 0] - sh_coords[a * 3 + 0];
    const float dy = sh_coords[b * 3 + 1] - sh_coords[a * 3 + 1];
    const float dz = sh_coords[b * 3 + 2] - sh_coords[a * 3 + 2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// Per-device cached sharedMemPerBlock limit.
//
// cudaGetDeviceProperties is ~800 us per call on this host; calling it on
// every kernel launch dominates host overhead for small molecules.
// Cache the result per device id (fixed for the lifetime of the process).
// ---------------------------------------------------------------------------
static int shared_mem_per_block_cached(int device)
{
    static std::mutex mtx;
    static std::unordered_map<int, int> cache;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(device);
        if (it != cache.end()) return it->second;
    }
    int value = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &value, cudaDevAttrMaxSharedMemoryPerBlock, device));
    {
        std::lock_guard<std::mutex> lock(mtx);
        cache[device] = value;
    }
    return value;
}

// ---------------------------------------------------------------------------
//   p, q in [0, nelements); result in [0, nelements*(nelements+1)/2)
// ---------------------------------------------------------------------------
__device__ static inline int pair_index_dev(int nelements, int p, int q)
{
    if (p > q) { int tmp = p; p = q; q = tmp; }
    return p * nelements - p * (p + 1) / 2 + q;
}

// ---------------------------------------------------------------------------
// FCHL19 forward kernel
// ---------------------------------------------------------------------------

__global__ static void fchl19_forward_kernel(
    const float * __restrict__ coords,      // (nm, max_atoms, 3)
    const int   * __restrict__ Q,           // (nm, max_atoms)  element indices
    const int   * __restrict__ N,           // (nm,)            active atom counts
    float       *              out,         // (nm, max_atoms, rep_size)
    const float * __restrict__ d_Rs2,       // (nbasis2,)
    const float * __restrict__ d_log_Rs2,   // (nbasis2,)
    const float * __restrict__ d_inv_Rs2,   // (nbasis2,)
    const float * __restrict__ d_Rs3,       // (nbasis3,)
    const float * __restrict__ d_ang_w,     // (n_harm,)
    const int   * __restrict__ d_ang_o,     // (n_harm,)
    int nm, int max_atoms, int rep_size,
    int nelements, int nbasis2, int nbasis3, int nabasis, int n_harm,
    float eta2, float eta3,
    float rcut, float acut,
    float two_body_decay, float three_body_decay, float three_body_weight
) {
    const int bid = blockIdx.x;
    const int m   = bid / max_atoms;
    const int i   = bid % max_atoms;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (m >= nm) return;

    const int natoms = N[m];

    // Inactive atom slot: zero the output and return.
    if (i >= natoms) {
        float *dst = out + ((long long)m * max_atoms + i) * rep_size;
        for (int r = tid; r < rep_size; r += BLOCK_SZ)
            dst[r] = 0.0f;
        return;
    }

    // -----------------------------------------------------------------------
    // Dynamic shared memory layout
    // -----------------------------------------------------------------------
    extern __shared__ char smem[];

    const int off_coords = 0;
    const int off_Z      = off_coords + max_atoms * 3 * (int)sizeof(float);
    const int off_rep    = off_Z      + max_atoms     * (int)sizeof(int);

    float *sh_coords = (float *)(smem + off_coords);
    int   *sh_Z      = (int   *)(smem + off_Z);
    float *sh_rep    = (float *)(smem + off_rep);

    // -----------------------------------------------------------------------
    // Load coordinates, element indices, zero accumulator
    // -----------------------------------------------------------------------
    const float *mol_coords = coords + (long long)m * max_atoms * 3;
    const int   *mol_Q      = Q      + m * max_atoms;

    for (int a = tid; a < natoms * 3; a += BLOCK_SZ)
        sh_coords[a] = mol_coords[a];
    for (int a = tid; a < natoms; a += BLOCK_SZ)
        sh_Z[a] = mol_Q[a];
    for (int r = tid; r < rep_size; r += BLOCK_SZ)
        sh_rep[r] = 0.0f;

    __syncthreads();

    // -----------------------------------------------------------------------
    // Two-body contributions — each thread handles a subset of neighbors
    // -----------------------------------------------------------------------
    const float inv_rcut    = (rcut > 0.0f) ? (1.0f / rcut) : 0.0f;
    const float pi_inv_rcut = PI_F * inv_rcut;

    for (int j = tid; j < natoms; j += BLOCK_SZ) {
        if (j == i) continue;
        const float rij = dist3_dev(sh_coords, i, j);
        if (rij >= rcut) continue;

        const int elem_j = sh_Z[j];

        const float rij2    = rij * rij;
        const float t       = eta2 / fmaxf(rij2, EPS_F);
        const float log1pt  = log1pf(t);
        const float sigma   = sqrtf(fmaxf(log1pt, 0.0f));
        if (sigma < EPS_F) continue;

        const float mu         = logf(rij) - 0.5f * log1pt;
        const float decay_ij   = 0.5f * (cosf(pi_inv_rcut * rij) + 1.0f);
        const float inv_pref   = decay_ij /
                                 (sigma * SQRT_2PI_F * powf(rij, two_body_decay));
        const float inv_sig_sq = 1.0f / (sigma * sigma);

        const int base_j = elem_j * nbasis2;
        for (int k = 0; k < nbasis2; ++k) {
            const float dlog = d_log_Rs2[k] - mu;
            const float g    = expf(-0.5f * dlog * dlog * inv_sig_sq);
            const float val  = inv_pref * g * d_inv_Rs2[k];
            atomicAdd(&sh_rep[base_j + k], val);
        }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // Three-body contributions
    // threadIdx.x strides over j, threadIdx.y strides over k (k > j).
    // -----------------------------------------------------------------------
    const float inv_acut    = (acut > 0.0f) ? (1.0f / acut) : 0.0f;
    const float pi_inv_acut = PI_F * inv_acut;
    const int   three_offset = nelements * nbasis2;

    for (int j = threadIdx.x; j < natoms; j += blockDim.x) {
        if (j == i) continue;
        const float rij = dist3_dev(sh_coords, i, j);
        if (rij >= acut) continue;
        for (int k = j + 1 + threadIdx.y; k < natoms; k += blockDim.y) {
            if (k == i) continue;
            const float rik = dist3_dev(sh_coords, i, k);
            if (rik >= acut) continue;
            const float rjk = dist3_dev(sh_coords, j, k);

            const float Bx = sh_coords[i * 3 + 0], By = sh_coords[i * 3 + 1], Bz = sh_coords[i * 3 + 2];
            const float Ax = sh_coords[j * 3 + 0], Ay = sh_coords[j * 3 + 1], Az = sh_coords[j * 3 + 2];
            const float Cx = sh_coords[k * 3 + 0], Cy = sh_coords[k * 3 + 1], Cz = sh_coords[k * 3 + 2];

            const float inv_rij = 1.0f / fmaxf(rij, EPS_F);
            const float inv_rik = 1.0f / fmaxf(rik, EPS_F);
            const float inv_rjk = 1.0f / fmaxf(rjk, EPS_F);

            const float eij0 = (Ax - Bx) * inv_rij, eij1 = (Ay - By) * inv_rij, eij2 = (Az - Bz) * inv_rij;
            const float eik0 = (Cx - Bx) * inv_rik, eik1 = (Cy - By) * inv_rik, eik2 = (Cz - Bz) * inv_rik;
            const float ejk0 = (Cx - Ax) * inv_rjk, ejk1 = (Cy - Ay) * inv_rjk, ejk2 = (Cz - Az) * inv_rjk;

            float cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
            cos_i = fmaxf(-1.0f, fminf(1.0f, cos_i));
            const float cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
            const float cos_k =   eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

            const float atm_denom =
                powf(fmaxf(rij * rik * rjk, EPS_F), three_body_decay);
            const float ksi3 =
                (1.0f + 3.0f * cos_i * cos_j * cos_k) * (three_body_weight / atm_denom);

            const float decay_ij   = 0.5f * (cosf(pi_inv_acut * rij) + 1.0f);
            const float decay_ik   = 0.5f * (cosf(pi_inv_acut * rik) + 1.0f);
            const float decay_prod = decay_ij * decay_ik;

            const float rbar     = 0.5f * (rij + rik);
            const int   elem_j   = sh_Z[j];
            const int   elem_k   = sh_Z[k];
            const int   pair_idx = pair_index_dev(nelements, elem_j, elem_k);
            const int   base     = three_offset + pair_idx * (nbasis3 * nabasis);

            const float sin_i = sqrtf(fmaxf(0.0f, 1.0f - cos_i * cos_i));

            float angular[MAX_NABASIS];
            angular[0] = d_ang_w[0] * cos_i;
            if (nabasis > 1) angular[1] = d_ang_w[0] * sin_i;

            if (n_harm > 1) {
                const float two_cos = 2.0f * cos_i;
                float cn_2 = 1.0f, sn_2 = 0.0f;
                float cn_1 = cos_i, sn_1 = sin_i;
                int harm_stored = 1;
                const int max_o = d_ang_o[n_harm - 1];
                for (int n = 2; n <= max_o; ++n) {
                    const float cn = two_cos * cn_1 - cn_2;
                    const float sn = two_cos * sn_1 - sn_2;
                    cn_2 = cn_1; sn_2 = sn_1;
                    cn_1 = cn;   sn_1 = sn;
                    if (n == d_ang_o[harm_stored]) {
                        angular[2 * harm_stored]     = d_ang_w[harm_stored] * cn;
                        angular[2 * harm_stored + 1] = d_ang_w[harm_stored] * sn;
                        if (++harm_stored >= n_harm) break;
                    }
                }
            }

            for (int l = 0; l < nbasis3; ++l) {
                const float dr     = rbar - d_Rs3[l];
                const float radial = expf(-eta3 * dr * dr) * decay_prod;
                const float scale  = radial * ksi3;
                const int   z0     = base + l * nabasis;
                for (int a = 0; a < nabasis; ++a)
                    atomicAdd(&sh_rep[z0 + a], angular[a] * scale);
            }
        }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // Write shared accumulator to global output
    // -----------------------------------------------------------------------
    float *dst = out + ((long long)m * max_atoms + i) * rep_size;
    for (int r = tid; r < rep_size; r += BLOCK_SZ)
        dst[r] = sh_rep[r];
}
__global__ static void fchl19_grad_kernel(
    const float * __restrict__ coords,
    const int   * __restrict__ Q,
    const int   * __restrict__ N,
    float       *              rep,
    float       *              grad,
    const float * __restrict__ d_Rs2,
    const float * __restrict__ d_log_Rs2,
    const float * __restrict__ d_inv_Rs2,
    const float * __restrict__ d_Rs3,
    int nm, int max_atoms, int rep_size,
    int nelements, int nbasis2, int nbasis3, int nabasis,
    float eta2, float eta3,
    float zeta,
    float rcut, float acut,
    float two_body_decay, float three_body_decay, float three_body_weight
) {
    const int bid = blockIdx.x;
    const int m = bid / max_atoms;
    const int i = bid % max_atoms;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (m >= nm) return;
    const int natoms = N[m];
    if (i >= natoms) return;

    extern __shared__ char smem[];

    const int off_coords = 0;
    const int off_Z = off_coords + max_atoms * 3 * (int)sizeof(float);
    const int off_rep = off_Z + max_atoms * (int)sizeof(int);

    float *sh_coords = (float *)(smem + off_coords);
    int   *sh_Z      = (int   *)(smem + off_Z);
    float *sh_rep    = (float *)(smem + off_rep);

    const float *mol_coords = coords + (long long)m * max_atoms * 3;
    const int   *mol_Q      = Q + m * max_atoms;

    for (int a = tid; a < natoms * 3; a += BLOCK_SZ) sh_coords[a] = mol_coords[a];
    for (int a = tid; a < natoms; a += BLOCK_SZ) sh_Z[a] = mol_Q[a];
    for (int r = tid; r < rep_size; r += BLOCK_SZ) sh_rep[r] = 0.0f;

    __syncthreads();

    const float inv_rcut = (rcut > 0.0f) ? (1.0f / rcut) : 0.0f;
    const float pi_inv_rcut = PI_F * inv_rcut;
    const long long rep_base = ((long long)m * max_atoms + i) * rep_size;
    const long long grad_base = (((long long)m * max_atoms + i) * rep_size) * max_atoms * 3;

    // -----------------------------------------------------------------------
    // Two-body contributions — each thread handles a subset of neighbors
    // -----------------------------------------------------------------------
    for (int j = tid; j < natoms; j += BLOCK_SZ) {
        if (j == i) continue;
        const float rij = dist3_dev(sh_coords, i, j);
        if (rij > rcut) continue;

        const int elem_j = sh_Z[j];
        const float rij2 = rij * rij;
        const float invr2 = 1.0f / fmaxf(rij2, EPS_F);
        const float s2 = log1pf(eta2 * invr2);
        const float sigma = sqrtf(fmaxf(s2, 0.0f));
        if (sigma < EPS_F) continue;

        const float mu = logf(rij) - 0.5f * s2;
        const float decay_ij = 0.5f * (cosf(pi_inv_rcut * rij) + 1.0f);
        const float scaling = powf(rij, -two_body_decay);
        const float inv_pref_common = 1.0f / (sigma * SQRT_2PI_F);
        const float exp_s2 = expf(s2);
        const float sqrt_exp_s2 = sqrtf(exp_s2);
        const int feat_base = elem_j * nbasis2;

        for (int k = 0; k < nbasis2; ++k) {
            const float L = d_log_Rs2[k] - mu;
            const float g = expf(-0.5f * L * L / s2);
            const float exp_ln = g * 1.4142135623730951f;
            const float radial_base = inv_pref_common * d_inv_Rs2[k] * g;
            const float radial = radial_base * scaling * decay_ij;
            atomicAdd(&sh_rep[feat_base + k], radial);

            for (int t = 0; t < 3; ++t) {
                const float dx = sh_coords[j * 3 + t] - sh_coords[i * 3 + t];
                const float dscal = two_body_decay * dx * powf(rij, -(two_body_decay + 2.0f));
                const float ddecay = dx * 0.5f * PI_F * sinf(pi_inv_rcut * rij) * inv_rcut /
                                     fmaxf(rij, EPS_F);
                const float term1 = L *
                                    (-dx * (rij * rij * exp_s2 + eta2) /
                                     powf(rij * sqrt_exp_s2, 3.0f)) *
                                    (sqrt_exp_s2 / (s2 * rij));
                const float term2 = L * L * eta2 * dx / (s2 * s2 * powf(rij, 4.0f) * exp_s2);
                const float A =
                    (term1 + term2) *
                        (exp_ln * d_inv_Rs2[k] / (sigma * 1.7724538509055160f * 2.0f)) -
                    (exp_ln * eta2 * dx * d_inv_Rs2[k]) /
                        (s2 * 1.7724538509055160f * sigma * powf(rij, 4.0f) * exp_s2 * 2.0f);
                const float part =
                    A * scaling * decay_ij + radial_base * dscal * decay_ij +
                    radial_base * scaling * ddecay;
                const long long f = feat_base + k;
                atomicAdd(&grad[grad_base + (f * max_atoms + i) * 3 + t], part);
                atomicAdd(&grad[grad_base + (f * max_atoms + j) * 3 + t], -part);
            }
        }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // Three-body contributions
    // threadIdx.x strides over j, threadIdx.y strides over k (k > j).
    // -----------------------------------------------------------------------
    const float inv_acut = (acut > 0.0f) ? (1.0f / acut) : 0.0f;
    const float pi_inv_acut = PI_F * inv_acut;
    const int three_offset = nelements * nbasis2;
    const float ang_w_pre = expf(-0.5f * zeta * zeta) * 2.0f;
    const float tbd_over_w_pre = (three_body_weight != 0.0f) ? (three_body_decay / three_body_weight) : 0.0f;

    for (int j = threadIdx.x; j < natoms; j += blockDim.x) {
        if (j == i) continue;
        const float rij = dist3_dev(sh_coords, i, j);
        if (rij > acut) continue;
        for (int k = j + 1 + threadIdx.y; k < natoms; k += blockDim.y) {
            if (k == i) continue;
            const float rik = dist3_dev(sh_coords, i, k);
            if (rik > acut) continue;
            const float rjk = dist3_dev(sh_coords, j, k);
            const float rij2 = fmaxf(rij * rij, EPS_F);
            const float rik2 = fmaxf(rik * rik, EPS_F);
            const float invrij = 1.0f / fmaxf(rij, EPS_F);
            const float invrik = 1.0f / fmaxf(rik, EPS_F);
            const float invrjk = 1.0f / fmaxf(rjk, EPS_F);
            const float invrij2 = 1.0f / rij2;
            const float invrik2 = 1.0f / rik2;
            const float invrjk2 = invrjk * invrjk;

            const float Bx = sh_coords[i * 3 + 0], By = sh_coords[i * 3 + 1], Bz = sh_coords[i * 3 + 2];
            const float Ax = sh_coords[j * 3 + 0], Ay = sh_coords[j * 3 + 1], Az = sh_coords[j * 3 + 2];
            const float Cx = sh_coords[k * 3 + 0], Cy = sh_coords[k * 3 + 1], Cz = sh_coords[k * 3 + 2];

            const float eij0 = (Ax - Bx) * invrij, eij1 = (Ay - By) * invrij, eij2 = (Az - Bz) * invrij;
            const float eik0 = (Cx - Bx) * invrik, eik1 = (Cy - By) * invrik, eik2 = (Cz - Bz) * invrik;
            const float ejk0 = (Cx - Ax) * invrjk, ejk1 = (Cy - Ay) * invrjk, ejk2 = (Cz - Az) * invrjk;

            float cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
            cos_i = fmaxf(-1.0f, fminf(1.0f, cos_i));
            const float cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
            const float cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;
            const float dot = (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);

            const float sin_i = sqrtf(fmaxf(0.0f, 1.0f - cos_i * cos_i));
            const float angular0 = ang_w_pre * cos_i;
            const float angular1 = ang_w_pre * sin_i;
            const float d_angular0 = ang_w_pre * sin_i;
            const float d_angular1 = -ang_w_pre * cos_i;

            const float denom = sqrtf(fmaxf(1e-10f, rij2 * rik2 - dot * dot));
            const float inv_denom = 1.0f / denom;
            const float d_ang_d_j0 = Cx - Bx + dot * ((Bx - Ax) * invrij2);
            const float d_ang_d_j1 = Cy - By + dot * ((By - Ay) * invrij2);
            const float d_ang_d_j2 = Cz - Bz + dot * ((Bz - Az) * invrij2);
            const float d_ang_d_k0 = Ax - Bx + dot * ((Bx - Cx) * invrik2);
            const float d_ang_d_k1 = Ay - By + dot * ((By - Cy) * invrik2);
            const float d_ang_d_k2 = Az - Bz + dot * ((Bz - Cz) * invrik2);
            const float dai0 = -(d_ang_d_j0 + d_ang_d_k0) * inv_denom;
            const float dai1 = -(d_ang_d_j1 + d_ang_d_k1) * inv_denom;
            const float dai2 = -(d_ang_d_j2 + d_ang_d_k2) * inv_denom;
            const float daj0 = d_ang_d_j0 * inv_denom, daj1 = d_ang_d_j1 * inv_denom, daj2 = d_ang_d_j2 * inv_denom;
            const float dak0 = d_ang_d_k0 * inv_denom, dak1 = d_ang_d_k1 * inv_denom, dak2 = d_ang_d_k2 * inv_denom;

            const float decay_ij = 0.5f * (cosf(pi_inv_acut * rij) + 1.0f);
            const float decay_ik = 0.5f * (cosf(pi_inv_acut * rik) + 1.0f);
            const float decay_prod = decay_ij * decay_ik;
            const float s_ij = -PI_F * sinf(pi_inv_acut * rij) * 0.5f * invrij * inv_acut;
            const float s_ik = -PI_F * sinf(pi_inv_acut * rik) * 0.5f * invrik * inv_acut;
            const float d_ijd0 = s_ij * (Bx - Ax), d_ijd1 = s_ij * (By - Ay), d_ijd2 = s_ij * (Bz - Az);
            const float d_ikd0 = s_ik * (Bx - Cx), d_ikd1 = s_ik * (By - Cy), d_ikd2 = s_ik * (Bz - Cz);

            const float invr_atm = powf(invrij * invrjk * invrik, three_body_decay);
            const float atm = (1.0f + 3.0f * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;
            const float atm_i = (3.0f * cos_j * cos_k) * invr_atm * invrij * invrik;
            const float atm_j = (3.0f * cos_k * cos_i) * invr_atm * invrij * invrjk;
            const float atm_k = (3.0f * cos_i * cos_j) * invr_atm * invrjk * invrik;
            const float vi = dot;
            const float vj = (Cx - Ax) * (Bx - Ax) + (Cy - Ay) * (By - Ay) + (Cz - Az) * (Bz - Az);
            const float vk = (Bx - Cx) * (Ax - Cx) + (By - Cy) * (Ay - Cy) + (Bz - Cz) * (Az - Cz);

            const float d_atm_ii0 = 2 * Bx - Ax - Cx - vi * ((Bx - Ax) * invrij2 + (Bx - Cx) * invrik2);
            const float d_atm_ii1 = 2 * By - Ay - Cy - vi * ((By - Ay) * invrij2 + (By - Cy) * invrik2);
            const float d_atm_ii2 = 2 * Bz - Az - Cz - vi * ((Bz - Az) * invrij2 + (Bz - Cz) * invrik2);
            const float d_atm_ij0 = Cx - Ax - vj * (Bx - Ax) * invrij2;
            const float d_atm_ij1 = Cy - Ay - vj * (By - Ay) * invrij2;
            const float d_atm_ij2 = Cz - Az - vj * (Bz - Az) * invrij2;
            const float d_atm_ik0 = Ax - Cx - vk * (Bx - Cx) * invrik2;
            const float d_atm_ik1 = Ay - Cy - vk * (By - Cy) * invrik2;
            const float d_atm_ik2 = Az - Cz - vk * (Bz - Cz) * invrik2;
            const float d_atm_ji0 = Cx - Bx - vi * (Ax - Bx) * invrij2;
            const float d_atm_ji1 = Cy - By - vi * (Ay - By) * invrij2;
            const float d_atm_ji2 = Cz - Bz - vi * (Az - Bz) * invrij2;
            const float d_atm_jj0 = 2 * Ax - Bx - Cx - vj * ((Ax - Bx) * invrij2 + (Ax - Cx) * invrjk2);
            const float d_atm_jj1 = 2 * Ay - By - Cy - vj * ((Ay - By) * invrij2 + (Ay - Cy) * invrjk2);
            const float d_atm_jj2 = 2 * Az - Bz - Cz - vj * ((Az - Bz) * invrij2 + (Az - Cz) * invrjk2);
            const float d_atm_jk0 = Bx - Cx - vk * (Ax - Cx) * invrjk2;
            const float d_atm_jk1 = By - Cy - vk * (Ay - Cy) * invrjk2;
            const float d_atm_jk2 = Bz - Cz - vk * (Az - Cz) * invrjk2;
            const float d_atm_ki0 = Ax - Bx - vi * (Cx - Bx) * invrik2;
            const float d_atm_ki1 = Ay - By - vi * (Cy - By) * invrik2;
            const float d_atm_ki2 = Az - Bz - vi * (Cz - Bz) * invrik2;
            const float d_atm_kj0 = Bx - Ax - vj * (Cx - Ax) * invrjk2;
            const float d_atm_kj1 = By - Ay - vj * (Cy - Ay) * invrjk2;
            const float d_atm_kj2 = Bz - Az - vj * (Cz - Az) * invrjk2;
            const float d_atm_kk0 = 2 * Cx - Ax - Bx - vk * ((Cx - Ax) * invrjk2 + (Cx - Bx) * invrik2);
            const float d_atm_kk1 = 2 * Cy - Ay - By - vk * ((Cy - Ay) * invrjk2 + (Cy - By) * invrik2);
            const float d_atm_kk2 = 2 * Cz - Az - Bz - vk * ((Cz - Az) * invrjk2 + (Cz - Bz) * invrik2);

            const float atm_tbd = atm * tbd_over_w_pre;
            const float d_extra_i0 = ((Ax - Bx) * invrij2 + (Cx - Bx) * invrik2) * atm_tbd;
            const float d_extra_i1 = ((Ay - By) * invrij2 + (Cy - By) * invrik2) * atm_tbd;
            const float d_extra_i2 = ((Az - Bz) * invrij2 + (Cz - Bz) * invrik2) * atm_tbd;
            const float d_extra_j0 = ((Bx - Ax) * invrij2 + (Cx - Ax) * invrjk2) * atm_tbd;
            const float d_extra_j1 = ((By - Ay) * invrij2 + (Cy - Ay) * invrjk2) * atm_tbd;
            const float d_extra_j2 = ((Bz - Az) * invrij2 + (Cz - Az) * invrjk2) * atm_tbd;
            const float d_extra_k0 = ((Ax - Cx) * invrjk2 + (Bx - Cx) * invrik2) * atm_tbd;
            const float d_extra_k1 = ((Ay - Cy) * invrjk2 + (By - Cy) * invrik2) * atm_tbd;
            const float d_extra_k2 = ((Az - Cz) * invrjk2 + (Bz - Cz) * invrik2) * atm_tbd;

            const float atmi0 = (atm_i * d_atm_ii0 + atm_j * d_atm_ij0 + atm_k * d_atm_ik0 + d_extra_i0) * three_body_weight;
            const float atmi1 = (atm_i * d_atm_ii1 + atm_j * d_atm_ij1 + atm_k * d_atm_ik1 + d_extra_i1) * three_body_weight;
            const float atmi2 = (atm_i * d_atm_ii2 + atm_j * d_atm_ij2 + atm_k * d_atm_ik2 + d_extra_i2) * three_body_weight;
            const float atmj0 = (atm_i * d_atm_ji0 + atm_j * d_atm_jj0 + atm_k * d_atm_jk0 + d_extra_j0) * three_body_weight;
            const float atmj1 = (atm_i * d_atm_ji1 + atm_j * d_atm_jj1 + atm_k * d_atm_jk1 + d_extra_j1) * three_body_weight;
            const float atmj2 = (atm_i * d_atm_ji2 + atm_j * d_atm_jj2 + atm_k * d_atm_jk2 + d_extra_j2) * three_body_weight;
            const float atmk0 = (atm_i * d_atm_ki0 + atm_j * d_atm_kj0 + atm_k * d_atm_kk0 + d_extra_k0) * three_body_weight;
            const float atmk1 = (atm_i * d_atm_ki1 + atm_j * d_atm_kj1 + atm_k * d_atm_kk1 + d_extra_k1) * three_body_weight;
            const float atmk2 = (atm_i * d_atm_ki2 + atm_j * d_atm_kj2 + atm_k * d_atm_kk2 + d_extra_k2) * three_body_weight;

            const float dec_i0 = d_ijd0 * decay_ik + decay_ij * d_ikd0;
            const float dec_i1 = d_ijd1 * decay_ik + decay_ij * d_ikd1;
            const float dec_i2 = d_ijd2 * decay_ik + decay_ij * d_ikd2;
            const float dec_j0 = -d_ijd0 * decay_ik, dec_j1 = -d_ijd1 * decay_ik, dec_j2 = -d_ijd2 * decay_ik;
            const float dec_k0 = -decay_ij * d_ikd0, dec_k1 = -decay_ij * d_ikd1, dec_k2 = -decay_ij * d_ikd2;

            const float BmA0 = (Bx - Ax) * invrij, BmA1 = (By - Ay) * invrij, BmA2 = (Bz - Az) * invrij;
            const float BmC0 = (Bx - Cx) * invrik, BmC1 = (By - Cy) * invrik, BmC2 = (Bz - Cz) * invrik;
            const int elem_j = sh_Z[j], elem_k = sh_Z[k];
            const int pair_idx = pair_index_dev(nelements, elem_j, elem_k);
            const int base = three_offset + pair_idx * (nbasis3 * nabasis);

            for (int l = 0; l < nbasis3; ++l) {
                const float rbar = 0.5f * (rij + rik) - d_Rs3[l];
                const float rad_l = expf(-eta3 * rbar * rbar);
                const float d_rad_l = rad_l * eta3 * rbar;
                const float scale_val = rad_l * atm * decay_prod;
                const float scale_ang = decay_prod * rad_l;
                const float dri0 = d_rad_l * (-(BmA0 + BmC0));
                const float dri1 = d_rad_l * (-(BmA1 + BmC1));
                const float dri2 = d_rad_l * (-(BmA2 + BmC2));
                const float drj0 = d_rad_l * BmA0, drj1 = d_rad_l * BmA1, drj2 = d_rad_l * BmA2;
                const float drk0 = d_rad_l * BmC0, drk1 = d_rad_l * BmC1, drk2 = d_rad_l * BmC2;

                for (int aidx = 0; aidx < nabasis; ++aidx) {
                    const float ang = (aidx == 0) ? angular0 : angular1;
                    const float dang = (aidx == 0) ? d_angular0 : d_angular1;
                    const long long f = base + l * nabasis + aidx;
                    atomicAdd(&sh_rep[f], ang * scale_val);

                    const float gi0 = dang * dai0 * scale_ang * atm + ang * dri0 * atm * decay_prod + ang * rad_l * atmi0 * decay_prod + ang * rad_l * dec_i0 * atm;
                    const float gi1 = dang * dai1 * scale_ang * atm + ang * dri1 * atm * decay_prod + ang * rad_l * atmi1 * decay_prod + ang * rad_l * dec_i1 * atm;
                    const float gi2 = dang * dai2 * scale_ang * atm + ang * dri2 * atm * decay_prod + ang * rad_l * atmi2 * decay_prod + ang * rad_l * dec_i2 * atm;
                    const float gj0 = dang * daj0 * scale_ang * atm + ang * drj0 * atm * decay_prod + ang * rad_l * atmj0 * decay_prod + ang * rad_l * dec_j0 * atm;
                    const float gj1 = dang * daj1 * scale_ang * atm + ang * drj1 * atm * decay_prod + ang * rad_l * atmj1 * decay_prod + ang * rad_l * dec_j1 * atm;
                    const float gj2 = dang * daj2 * scale_ang * atm + ang * drj2 * atm * decay_prod + ang * rad_l * atmj2 * decay_prod + ang * rad_l * dec_j2 * atm;
                    const float gk0 = dang * dak0 * scale_ang * atm + ang * drk0 * atm * decay_prod + ang * rad_l * atmk0 * decay_prod + ang * rad_l * dec_k0 * atm;
                    const float gk1 = dang * dak1 * scale_ang * atm + ang * drk1 * atm * decay_prod + ang * rad_l * atmk1 * decay_prod + ang * rad_l * dec_k1 * atm;
                    const float gk2 = dang * dak2 * scale_ang * atm + ang * drk2 * atm * decay_prod + ang * rad_l * atmk2 * decay_prod + ang * rad_l * dec_k2 * atm;
                    atomicAdd(&grad[grad_base + (f * max_atoms + i) * 3 + 0], gi0);
                    atomicAdd(&grad[grad_base + (f * max_atoms + i) * 3 + 1], gi1);
                    atomicAdd(&grad[grad_base + (f * max_atoms + i) * 3 + 2], gi2);
                    atomicAdd(&grad[grad_base + (f * max_atoms + j) * 3 + 0], gj0);
                    atomicAdd(&grad[grad_base + (f * max_atoms + j) * 3 + 1], gj1);
                    atomicAdd(&grad[grad_base + (f * max_atoms + j) * 3 + 2], gj2);
                    atomicAdd(&grad[grad_base + (f * max_atoms + k) * 3 + 0], gk0);
                    atomicAdd(&grad[grad_base + (f * max_atoms + k) * 3 + 1], gk1);
                    atomicAdd(&grad[grad_base + (f * max_atoms + k) * 3 + 2], gk2);
                }
            }
        }
    }

    __syncthreads();

    float *dst = rep + rep_base;
    for (int r = tid; r < rep_size; r += BLOCK_SZ) dst[r] = sh_rep[r];
}


// ===========================================================================
// Host driver
// ===========================================================================

namespace kf {
namespace fchl19 {

torch::Tensor generate_fchl_acsf_cuda(
    const torch::Tensor &coords,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    int nelements,
    int nRs2,
    int nRs3,
    int nFourier,
    float eta2,
    float eta3,
    float zeta,
    float rcut,
    float acut,
    float two_body_decay,
    float three_body_decay,
    float three_body_weight_norm
) {
    TORCH_CHECK(coords.is_cuda() && coords.scalar_type() == torch::kFloat32,
                "coords must be float32 CUDA");
    TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kInt32,
                "Q must be int32 CUDA");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(N.is_cuda() && N.scalar_type() == torch::kInt32,
                "N must be int32 CUDA");
    TORCH_CHECK(N.is_contiguous(), "N must be contiguous");

    TORCH_CHECK(coords.dim() == 3, "coords must be 3-D (nm, max_atoms, 3)");
    TORCH_CHECK(Q.dim()      == 2, "Q must be 2-D (nm, max_atoms)");
    TORCH_CHECK(N.dim()      == 1, "N must be 1-D (nm,)");
    TORCH_CHECK(coords.size(2) == 3, "coords.size(2) must be 3");
    TORCH_CHECK(nelements >= 1,   "nelements must be >= 1");
    TORCH_CHECK(nRs2 >= 1,        "nRs2 must be >= 1");
    TORCH_CHECK(nRs3 >= 1,        "nRs3 must be >= 1");
    TORCH_CHECK(nFourier >= 1,    "nFourier must be >= 1");
    TORCH_CHECK(nFourier * 2 <= MAX_NABASIS,
                "nFourier*2 must be <= ", MAX_NABASIS, " (increase MAX_NABASIS to support larger values)");

    const int nm        = (int)coords.size(0);
    const int max_atoms = (int)coords.size(1);

    TORCH_CHECK(Q.size(0) == nm,         "Q.size(0) must equal nm");
    TORCH_CHECK(Q.size(1) == max_atoms,  "Q.size(1) must equal max_atoms");
    TORCH_CHECK(N.size(0) == nm,         "N.size(0) must equal nm");

    // -----------------------------------------------------------------------
    // Build basis arrays on CPU, upload to GPU
    // -----------------------------------------------------------------------
    const int n_harm = nFourier;
    const int nabasis = 2 * nFourier;
    const int nbasis2 = nRs2;
    const int nbasis3 = nRs3;

    std::vector<float> h_Rs2(nbasis2), h_log_Rs2(nbasis2), h_inv_Rs2(nbasis2);
    for (int i = 1; i <= nbasis2; ++i) {
        h_Rs2[i - 1]     = rcut * (float)i / (float)nbasis2;
        h_log_Rs2[i - 1] = std::log(h_Rs2[i - 1]);
        h_inv_Rs2[i - 1] = 1.0f / h_Rs2[i - 1];
    }

    std::vector<float> h_Rs3(nbasis3);
    for (int i = 1; i <= nbasis3; ++i)
        h_Rs3[i - 1] = acut * (float)i / (float)nbasis3;

    std::vector<float> h_ang_w(n_harm);
    std::vector<int>   h_ang_o(n_harm);
    for (int l = 0; l < n_harm; ++l) {
        const int   o = 2 * l + 1;  // 1, 3, 5, ...
        h_ang_o[l] = o;
        const float tv = zeta * (float)o;
        h_ang_w[l] = 2.0f * std::exp(-0.5f * tv * tv);
    }

    // Upload (from_blob → .to(device) copies to CUDA)
    auto d_Rs2 = torch::from_blob(h_Rs2.data(), {nbasis2}, torch::kFloat32)
                     .to(coords.device())
                     .contiguous();
    auto d_log_Rs2 = torch::from_blob(h_log_Rs2.data(), {nbasis2}, torch::kFloat32)
                         .to(coords.device())
                         .contiguous();
    auto d_inv_Rs2 = torch::from_blob(h_inv_Rs2.data(), {nbasis2}, torch::kFloat32)
                         .to(coords.device())
                         .contiguous();
    auto d_Rs3 = torch::from_blob(h_Rs3.data(), {nbasis3}, torch::kFloat32)
                     .to(coords.device())
                     .contiguous();
    auto d_ang_w = torch::from_blob(h_ang_w.data(), {n_harm}, torch::kFloat32)
                       .to(coords.device())
                       .contiguous();
    auto d_ang_o = torch::from_blob(h_ang_o.data(), {n_harm}, torch::kInt32)
                       .to(coords.device())
                       .contiguous();

    // -----------------------------------------------------------------------
    // Compute rep_size and output tensor
    // -----------------------------------------------------------------------
    const int n_pairs_sym = nelements * (nelements + 1) / 2;
    const int rep_size    = nelements * nbasis2 + n_pairs_sym * nbasis3 * nabasis;

    auto out = torch::empty(
        {nm, max_atoms, rep_size},
        coords.options()  // float32, same device
    );

    // -----------------------------------------------------------------------
    // Compute shared memory size and sanity-check against device limit
    // -----------------------------------------------------------------------
    const int smem_bytes =
        max_atoms * 3 * (int)sizeof(float) +  // sh_coords
        max_atoms     * (int)sizeof(int)   +  // sh_Z
        rep_size      * (int)sizeof(float);   // sh_rep

    const int dev = (int)coords.get_device();
    TORCH_CHECK(
        smem_bytes <= shared_mem_per_block_cached(dev),
        "Shared memory required (", smem_bytes, " bytes) exceeds device limit (",
        shared_mem_per_block_cached(dev), " bytes). "
        "Reduce max_atoms or rep_size."
    );

    // -----------------------------------------------------------------------
    // Launch kernel
    // -----------------------------------------------------------------------
    const int grid = nm * max_atoms;
    const dim3 block(BLOCK_X, BLOCK_Y);

    fchl19_forward_kernel<<<grid, block, smem_bytes>>>(
        coords.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        out.data_ptr<float>(),
        d_Rs2.data_ptr<float>(),
        d_log_Rs2.data_ptr<float>(),
        d_inv_Rs2.data_ptr<float>(),
        d_Rs3.data_ptr<float>(),
        d_ang_w.data_ptr<float>(),
        d_ang_o.data_ptr<int>(),
        nm, max_atoms, rep_size,
        nelements, nbasis2, nbasis3, nabasis, n_harm,
        eta2, eta3,
        rcut, acut,
        two_body_decay, three_body_decay, three_body_weight_norm
    );

    CUDA_CHECK(cudaGetLastError());

    return out;
}

std::tuple<torch::Tensor, torch::Tensor> generate_fchl_acsf_and_gradients_cuda(
    const torch::Tensor &coords,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    int nelements,
    int nRs2,
    int nRs3,
    int nFourier,
    float eta2,
    float eta3,
    float zeta,
    float rcut,
    float acut,
    float two_body_decay,
    float three_body_decay,
    float three_body_weight_norm
) {
    TORCH_CHECK(coords.is_cuda() && coords.scalar_type() == torch::kFloat32,
                "coords must be float32 CUDA");
    TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kInt32,
                "Q must be int32 CUDA");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(N.is_cuda() && N.scalar_type() == torch::kInt32,
                "N must be int32 CUDA");
    TORCH_CHECK(N.is_contiguous(), "N must be contiguous");
    TORCH_CHECK(coords.dim() == 3, "coords must be 3-D (nm, max_atoms, 3)");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2-D (nm, max_atoms)");
    TORCH_CHECK(N.dim() == 1, "N must be 1-D (nm,)");
    TORCH_CHECK(coords.size(2) == 3, "coords.size(2) must be 3");
    TORCH_CHECK(nelements >= 1, "nelements must be >= 1");
    TORCH_CHECK(nRs2 >= 1, "nRs2 must be >= 1");
    TORCH_CHECK(nRs3 >= 1, "nRs3 must be >= 1");
    TORCH_CHECK(nFourier == 1, "gradient currently supports nFourier == 1");

    const int nm = (int)coords.size(0);
    const int max_atoms = (int)coords.size(1);
    TORCH_CHECK(Q.size(0) == nm, "Q.size(0) must equal nm");
    TORCH_CHECK(Q.size(1) == max_atoms, "Q.size(1) must equal max_atoms");
    TORCH_CHECK(N.size(0) == nm, "N.size(0) must equal nm");

    const int nbasis2 = nRs2;
    const int nbasis3 = nRs3;
    const int nabasis = 2 * nFourier;
    const int n_pairs_sym = nelements * (nelements + 1) / 2;
    const int rep_size = nelements * nbasis2 + n_pairs_sym * nbasis3 * nabasis;

    std::vector<float> h_Rs2(nbasis2), h_log_Rs2(nbasis2), h_inv_Rs2(nbasis2);
    for (int i = 1; i <= nbasis2; ++i) {
        h_Rs2[i - 1] = rcut * (float)i / (float)nbasis2;
        h_log_Rs2[i - 1] = std::log(h_Rs2[i - 1]);
        h_inv_Rs2[i - 1] = 1.0f / h_Rs2[i - 1];
    }

    std::vector<float> h_Rs3(nbasis3);
    for (int i = 1; i <= nbasis3; ++i) h_Rs3[i - 1] = acut * (float)i / (float)nbasis3;

    auto d_Rs2 = torch::from_blob(h_Rs2.data(), {nbasis2}, torch::kFloat32)
                     .to(coords.device())
                     .contiguous();
    auto d_log_Rs2 = torch::from_blob(h_log_Rs2.data(), {nbasis2}, torch::kFloat32)
                         .to(coords.device())
                         .contiguous();
    auto d_inv_Rs2 = torch::from_blob(h_inv_Rs2.data(), {nbasis2}, torch::kFloat32)
                         .to(coords.device())
                         .contiguous();
    auto d_Rs3 = torch::from_blob(h_Rs3.data(), {nbasis3}, torch::kFloat32)
                     .to(coords.device())
                     .contiguous();

    auto rep = torch::zeros({nm, max_atoms, rep_size}, coords.options());
    auto grad = torch::zeros({nm, max_atoms, rep_size, max_atoms, 3}, coords.options());

    const int smem_bytes =
        max_atoms * 3 * (int)sizeof(float) +  // sh_coords
        max_atoms     * (int)sizeof(int)   +  // sh_Z
        rep_size      * (int)sizeof(float);   // sh_rep

    const int dev = (int)coords.get_device();
    TORCH_CHECK(
        smem_bytes <= shared_mem_per_block_cached(dev),
        "Shared memory required (", smem_bytes, " bytes) exceeds device limit (",
        shared_mem_per_block_cached(dev), " bytes). Reduce max_atoms or rep_size."
    );

    const int grid = nm * max_atoms;
    const dim3 block(BLOCK_X, BLOCK_Y);
    fchl19_grad_kernel<<<grid, block, smem_bytes>>>(
        coords.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        rep.data_ptr<float>(),
        grad.data_ptr<float>(),
        d_Rs2.data_ptr<float>(),
        d_log_Rs2.data_ptr<float>(),
        d_inv_Rs2.data_ptr<float>(),
        d_Rs3.data_ptr<float>(),
        nm, max_atoms, rep_size,
        nelements, nbasis2, nbasis3, nabasis,
        eta2, eta3, zeta,
        rcut, acut,
        two_body_decay, three_body_decay, three_body_weight_norm
    );
    CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(rep, grad);
}

}  // namespace fchl19
}  // namespace kf
