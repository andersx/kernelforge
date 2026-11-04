// Coords-only FCHL18 Gaussian Hessian kernel (FP32, no cutoff), molecules only.
//
//   H[(a,amu),(b,bnu)] = d^2 K[a,b] / dR_a[amu] dR_b[bnu],
//   K[a,b] = sum_{i in a, j in b} exp(-(s_ii + s_jj - 2 s_ij) * inv),  inv = 1 / (2 sigma^2)
//
// Pipeline (see kernel_gaussian_hessian_fp32):
//   1. element map                (mark_z_kernel)
//   2. CSR offsets + allocation
//   3. precompute_kernel          Neigh/Fock blob + per-centre gradient slab (gblob)
//   4. alchemy: mix Fock -> blob_e (effective Fock of the "B" side of a scalar)
//   5. self_kernel                s_ii and d(s_ii)/dR (= 2 * d s(A,A)/dR_A)
//   6. hess_hot_kernel            one block per (molecule pair, centre i, chunk of centres j)
//
// hess_hot_kernel: all threads of a block cooperate on one centre pair (i,j) at a time (like
// the baseline cuda_fchl18_hessian.cu hot kernel); ds/dR lives in shared memory and the
// coefficient-weighted Hij of all j of the block is accumulated in a shared tile that is
// flushed once with atomicAdd into the global H (blocks of one mol pair share the same H block).

#include "cuda_fchl18_direct_hess.hpp"

#include "cuda_fchl18_direct_hess_grads.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>

namespace kf {
namespace fchl18 {
namespace direct {
namespace {

using hess::Fock;
using hess::Neigh;

struct Params {
    float two_body_power;
    float three_body_power;
    int fourier_order;
    int pmax;
    int use_atm;
    int z_to_idx[256];
};

// Full Z-indexed pd in constant mem (118^2 * 4 ~ 55 KiB < 64 KiB limit).
__constant__ float c_alchemy_pd[kAlchemyEmaxDefault * kAlchemyEmaxDefault];

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

// Forward blob, per centre: Neigh[na] | Fock[na * pmax * order]   (same as direct.cu)
__host__ __device__ inline size_t centre_stride_bytes(int na, int pmax, int order) {
    const size_t raw = static_cast<size_t>(na) * sizeof(Neigh) +
                       static_cast<size_t>(na) * static_cast<size_t>(pmax) * order * sizeof(Fock);
    return (raw + 15u) & ~size_t(15);
}

__device__ inline void centre_views_base(
    char *base, int i, int na, int pmax, int order, Neigh **neigh, Fock **fock
) {
    char *p = base + static_cast<long long>(i) * centre_stride_bytes(na, pmax, order);
    *neigh = reinterpret_cast<Neigh *>(p);
    *fock = reinterpret_cast<Fock *>(*neigh + na);
}

__device__ inline void centre_views(
    char *blob, const long long *mol_byte_off, int a, int i, int na, int pmax, int order,
    Neigh **neigh, Fock **fock
) {
    centre_views_base(blob + mol_byte_off[a], i, na, pmax, order, neigh, fock);
}

// Read-only views (hot kernel).
__device__ inline void centre_views(
    const char *blob, const long long *mol_byte_off, int a, int i, int na, int pmax, int order,
    const Neigh **neigh, const Fock **fock
) {
    const char *p = blob + mol_byte_off[a] +
                    static_cast<long long>(i) * centre_stride_bytes(na, pmax, order);
    *neigh = reinterpret_cast<const Neigh *>(p);
    *fock = reinterpret_cast<const Fock *>(*neigh + na);
}

__device__ inline const Fock *centre_fock_ptr(
    const char *blob, const long long *mol_byte_off, int a, int i, int na, int pmax, int order
) {
    const char *p =
        blob + mol_byte_off[a] + static_cast<long long>(i) * centre_stride_bytes(na, pmax, order);
    return reinterpret_cast<const Fock *>(reinterpret_cast<const Neigh *>(p) + na);
}

__device__ inline Fock *centre_fock_ptr(
    char *blob, const long long *mol_byte_off, int a, int i, int na, int pmax, int order
) {
    char *p =
        blob + mol_byte_off[a] + static_cast<long long>(i) * centre_stride_bytes(na, pmax, order);
    return reinterpret_cast<Fock *>(reinterpret_cast<Neigh *>(p) + na);
}

// Gradient slab, per centre (na3 = 3 na, F = na * pmax * order):
//   int    nbr_atom_idx[na] | pad to 16 B |
//   float4 rad[na]   ({ux,uy,uz,dksi_dr}, see grads.cuh) |
//   float  dcosp[F*na3] | dsinp[F*na3]
__host__ __device__ inline size_t grad_centre_stride_bytes(int na, int pmax, int order) {
    const size_t head = (static_cast<size_t>(na) * sizeof(int) + 15u) & ~size_t(15);
    const size_t na3 = static_cast<size_t>(3) * na;
    const size_t F = static_cast<size_t>(na) * pmax * order;
    const size_t floats = static_cast<size_t>(hess::hess_grad_slot_floats(na)) + 2 * F * na3;
    return (head + floats * sizeof(float) + 15u) & ~size_t(15);
}

struct GradViews {
    int *nbr;
    float4 *rad;
    float *dcosp;
    float *dsinp;
};

__device__ inline GradViews grad_views(
    char *gblob, const long long *mol_gbyte_off, int a, int i, int na, int pmax, int order
) {
    char *p = gblob + mol_gbyte_off[a] +
              static_cast<long long>(i) * grad_centre_stride_bytes(na, pmax, order);
    const size_t head = (static_cast<size_t>(na) * sizeof(int) + 15u) & ~size_t(15);
    const long long na3 = 3LL * na;
    const long long F = static_cast<long long>(na) * pmax * order;
    GradViews g;
    g.nbr = reinterpret_cast<int *>(p);
    g.rad = reinterpret_cast<float4 *>(p + head);
    g.dcosp = reinterpret_cast<float *>(g.rad + na);
    g.dsinp = g.dcosp + F * na3;
    return g;
}

struct GradViewsC {
    const int *nbr;
    const float4 *rad;
    const float *dcosp;
    const float *dsinp;
};

__device__ inline GradViewsC grad_views(
    const char *gblob, const long long *mol_gbyte_off, int a, int i, int na, int pmax, int order
) {
    const GradViews g = grad_views(
        const_cast<char *>(gblob), mol_gbyte_off, a, i, na, pmax, order
    );
    return GradViewsC{g.nbr, g.rad, g.dcosp, g.dsinp};
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void mark_z_kernel(const int *Z, const int *N, int nm, int max_atoms, int *z_present) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nm * max_atoms) {
        return;
    }
    const int a = flat / max_atoms;
    const int i = flat - a * max_atoms;
    if (i >= N[a]) {
        return;
    }
    const int z = Z[a * max_atoms + i];
    if (z > 0 && z < 256) {
        atomicOr(z_present + z, 1);
    }
}

template <bool UseAtm>
__global__ void precompute_kernel(
    const float *coords,
    const int *Z,
    const int *n,
    const long long *mol_byte_off,
    const long long *mol_gbyte_off,
    int *nn,
    char *blob,
    char *gblob,
    int nm,
    int max_atoms,
    Params params
) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nm * max_atoms) {
        return;
    }
    const int a = flat / max_atoms;
    const int i = flat - a * max_atoms;
    if (i >= n[a]) {
        nn[flat] = 0;
        return;
    }
    const int na = n[a];
    Neigh *neigh = nullptr;
    Fock *fock = nullptr;
    centre_views(blob, mol_byte_off, a, i, na, params.pmax, params.fourier_order, &neigh, &fock);
    const GradViews g =
        grad_views(gblob, mol_gbyte_off, a, i, na, params.pmax, params.fourier_order);
    const float *coords_mol = coords + static_cast<long long>(a) * max_atoms * 3;
    const int *Z_mol = Z + static_cast<long long>(a) * max_atoms;

    // Caller cudaMemset's the whole gblob; skip per-thread zero loops.
    hess::precompute_atom_with_grad<UseAtm>(
        coords_mol, Z_mol, i, na, params.two_body_power, params.three_body_power,
        params.fourier_order, params.pmax, params.z_to_idx, nn + flat, neigh, fock, g.nbr, g.rad,
        g.dcosp, g.dsinp, /*zero_outputs=*/false
    );
}

// Mix element channels of Fock float2 slabs: F_eff[p1] = sum_p2 PDc[p1,p2] F[p2].
// Only the Fock region of blob_out is written.
__global__ void alchemy_mix_fock_blob_kernel(
    const int *n,
    const int *nn,
    const long long *mol_byte_off,
    const char *blob_in,
    char *blob_out,
    const float *pdc,
    int nm,
    int max_atoms,
    int pmax,
    int order
) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nm * max_atoms) {
        return;
    }
    const int a = flat / max_atoms;
    const int i = flat - a * max_atoms;
    if (i >= n[a]) {
        return;
    }
    const int na = n[a];
    const int n_neigh = nn[flat];
    const Fock *fi = centre_fock_ptr(blob_in, mol_byte_off, a, i, na, pmax, order);
    Fock *fo = centre_fock_ptr(blob_out, mol_byte_off, a, i, na, pmax, order);
    for (int s = 0; s < n_neigh; ++s) {
        for (int m = 0; m < order; ++m) {
            float2 v[kMaxElements];
            for (int p = 0; p < pmax; ++p) {
                v[p] = fi[static_cast<long long>(s) * pmax * order +
                          static_cast<long long>(p) * order + m];
            }
            for (int p1 = 0; p1 < pmax; ++p1) {
                float2 acc = make_float2(0.0f, 0.0f);
                for (int p2 = 0; p2 < pmax; ++p2) {
                    const float w = pdc[p1 * pmax + p2];
                    acc.x += w * v[p2].x;
                    acc.y += w * v[p2].y;
                }
                fo[static_cast<long long>(s) * pmax * order +
                   static_cast<long long>(p1) * order + m] = acc;
            }
        }
    }
}

// s_ii and d(s_ii)/dR (dss = 2 * d s(A,A)/dR_A, like CPU HessMolSide).
template <bool Alch>
__global__ void self_kernel(
    const int *n,
    const int *nn,
    const long long *mol_byte_off,
    const long long *mol_gbyte_off,
    const long long *dss_off,
    char *blob,
    char *blob_e,
    char *gblob,
    float *ss,
    float *dss,
    int nm,
    int max_atoms,
    hess::ScalarParams sp
) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nm * max_atoms) {
        return;
    }
    const int a = flat / max_atoms;
    const int i = flat - a * max_atoms;
    if (i >= n[a]) {
        return;
    }
    const int na = n[a];
    const int na3 = 3 * na;
    const int n_neigh = nn[flat];
    Neigh *ni = nullptr;
    Fock *fi = nullptr;
    centre_views(blob, mol_byte_off, a, i, na, sp.pmax, sp.fourier_order, &ni, &fi);
    const GradViews g =
        grad_views(gblob, mol_gbyte_off, a, i, na, sp.pmax, sp.fourier_order);

    float *out = dss + dss_off[a] + static_cast<long long>(i) * na3;
    float s;
    if constexpr (Alch) {
        const Fock *fi_e =
            centre_fock_ptr(blob_e, mol_byte_off, a, i, na, sp.pmax, sp.fourier_order);
        s = hess::scalar_alchemy_and_grad(
            ni, n_neigh, fi, g.rad, g.nbr, g.dcosp, g.dsinp, ni, n_neigh, fi_e, sp,
            c_alchemy_pd, na3, out
        );
    } else {
        s = hess::scalar_noalchemy_and_grad(
            ni, n_neigh, fi, g.rad, g.nbr, g.dcosp, g.dsinp, ni, n_neigh, fi, sp, na3, out
        );
    }
    ss[flat] = s;
    for (int q = 0; q < na3; ++q) {
        out[q] *= 2.0f;
    }
}

// Square case (both sides identical): hess_hot_kernel only computes the molecule blocks (a, b)
// with b <= a; block k of this kernel mirrors lower block (a, b), a > b, onto (b, a).
__global__ void mirror_lower_kernel(const int *n, const int *atom_base, float *H, long long ld) {
    const long long k = blockIdx.x;
    long long a = static_cast<long long>((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(k))) * 0.5);
    while (a * (a - 1) / 2 > k) {
        --a;
    }
    while ((a + 1) * a / 2 <= k) {
        ++a;
    }
    const long long b = k - a * (a - 1) / 2;
    const int na3 = 3 * n[a];
    const int nb3 = 3 * n[b];
    const long long ra = 3LL * atom_base[a];
    const long long rb = 3LL * atom_base[b];
    for (int e = threadIdx.x; e < na3 * nb3; e += blockDim.x) {
        const int amu = e / nb3;
        const int bnu = e - amu * nb3;
        H[(rb + bnu) * ld + ra + amu] = H[(ra + amu) * ld + rb + bnu];
    }
}

// Parameters of the cooperative Hessian kernel (all device-uniform scalars).
struct HotParams {
    float inv_width;        // -1 / (4 w^2)
    float maxgausdist2;     // (8 w)^2
    float dscale;           // true_distance_scale
    float ascale;           // true_angular_scale
    float spf[hess::kHessMaxOrder];  // per-mode angular prefactor
    int order;
    int pmax;
    int emax;
};

// One block per (a, b, i, j-chunk): centre i of molecule a against a chunk of centres j of
// molecule b (the whole molecule b when jchunk >= nb).  All threads cooperate on one (i, j)
// at a time; the coefficient-weighted Hij of every j in the chunk is accumulated into a
// block-local shared-memory tile and flushed once to the global H with atomicAdd (blocks of
// one molecule pair share the same H block).
//
// Per neighbour pair (p on A, q on B), with G = pd exp(-(rp-rq)^2/(4w^2)), dGp = dG/drp,
// dGq = -dGp, d2G = d2G/drp drq, and
//   V     = ksi_p ksi_q ds + as sum_t spf_t <fA_p[t], fBe_q[t]>               (fBe = PDc-mixed)
//   dVA   = dksiA_p ksi_q ds + as sum_t spf_t <dFA_p[t,amu], fBe_q[t]>
//   dVB   = dksiB_q ksi_p ds + as sum_t spf_t <fAe_p[t], dFB_q[t,bnu]>
//   d2V   = dksiA dksiB ds + as sum_t spf_t <dFA_p[t,amu], (PDc dFB_q)[t,bnu]>
// the Hessian contribution is
//   H[amu,bnu] += sum_{p,q} dGp driA dVB + dGq driB dVA + d2G driA driB V + G d2V.
//
// The sum over q is factored for fixed p (A-side objects depend on p only):
//   H[amu,bnu] += driA_p[amu]    * X[bnu]
//               + dksiA_p[amu]   * Y[bnu]
//               + sum_t dFA_p[t,amu] . Z[t,bnu]
//   X[bnu]   = sum_q dGp dVB_q[bnu] + d2G driB_q[bnu] V_q                  (terms 1 + 3)
//   Y[bnu]   = ds * sum_q (dGq driB_q[bnu] ksi_q + G dksiB_q[bnu])         (terms 2 + 4)
//   Z[t,bnu] = as spf_t * PDc sum_q (G dFB_q[t,bnu] + dGq driB_q[bnu] fB_q[t])  (terms 2 + 4)
// (X/Y/Z are the sums of the factored vectors Sgp_dksi, Sd2_dri_ksi, ... / M_dGp, W_d2, ... that the
// individual terms would use; the dFB-dot-fA parts are folded into X directly.)  Likewise
//   dsA[amu] += driA S_dGpV + dksiA ds S_Gksi + as sum_t spf_t dFA[t,amu] . Fsum[t],
//   Fsum[t] = sum_q G_q fBe_q[t].
// Cost per p: O(nnB nb3 FB) build + O(na3 nb3 FB) apply (previously O(nnB na3 nb3 FB)).
//
// dri / dksi are never read densely: slot p (or q) has support on 6 DOFs only (neighbour atom
// +u, centre atom -u; dksi = dksi_dr * dri), so the radial terms are selects on the owning atom.
//
// The Gaussian coefficient of (i, j) needs s_ij, so a cheap pre-pass (thread-parallel over p,
// no barriers inside) computes s_ij first; the Hij terms are then accumulated already scaled
// by coeff_hess, and the rank-1 term coeff_g gA gB is added when ds is complete.
//
// Dynamic shared layout (floats):
//   s_rad[max_nb] (float4) | s_nbq[max_nb] (int) |
//   s_Z[2 FB max_nb3] | s_Fsum[2 FB] | s_dsA[max_na3] | s_dsB[max_nb3] | s_X[max_nb3] |
//   s_Y[max_nb3] | s_G,s_dGp,s_d2G,s_V,s_ksi [5 * max_nb] | s_H[h_cap] | s_sij[1]
// h_cap = max_na3*max_nb3 when the accumulated Hij tile fits in shared memory.  Otherwise the
// host passes a private global tile Hscratch[blk * max_na3 * max_nb3] (plain adds, one atomic
// flush at the end).  Only if that pool is unavailable do we fall back to per-entry atomics.
//
// Alchemy: V uses raw F_i[p] and element-mixed F_j_e[q]; dVB uses mixed F_i_e[p] against raw dFB;
// Z mixes the channel index with PDc after the q-sum (raw dFB and raw fB are mixed together).
// Order (= number of retained Fourier modes, see order_eff on the host): 1 specialized;
// 0 = runtime hp.order.
template <bool Alch, int Order>
__global__ void __launch_bounds__(128, 4) hess_hot_kernel(
    const int *n1,
    const int *n2,
    const int *nn1,
    const int *nn2,
    const long long *mol_byte_off1,
    const long long *mol_byte_off2,
    const long long *mol_gbyte_off1,
    const long long *mol_gbyte_off2,
    const long long *dss_off1,
    const long long *dss_off2,
    const int *atom_base1,
    const int *atom_base2,
    const char *__restrict__ blob1,
    const char *__restrict__ blob2,
    const char *__restrict__ blob1_e,
    const char *__restrict__ blob2_e,
    const char *__restrict__ gblob1,
    const char *__restrict__ gblob2,
    const float *__restrict__ ss1,
    const float *__restrict__ ss2,
    const float *__restrict__ dss1,
    const float *__restrict__ dss2,
    float *__restrict__ H,
    long long ld_H,  // D_B
    int nm2,
    int max1,
    int max2,
    float inv,       // 1 / (2 sigma^2)
    HotParams hp,
    const float *pdc,
    const int *pair_offset,
    int n_mol_pairs,
    int n_hot_blocks,
    int max_na3,
    int max_nb3,
    int h_cap,
    int jchunk,
    float *__restrict__ Hscratch  // n_hot_blocks * max_na3 * max_nb3, or nullptr
) {
    extern __shared__ float4 hot_smem4[];
    float *hot_smem = reinterpret_cast<float *>(hot_smem4);
    __shared__ float s_pdc[hess::kHessMaxPmax * hess::kHessMaxPmax];

    const int blk = static_cast<int>(blockIdx.x);
    if (blk >= n_hot_blocks) {
        return;
    }

    int lo = 0;
    int hi = n_mol_pairs;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (pair_offset[mid] <= blk) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int mol_pair = lo;
    const int local = blk - pair_offset[mol_pair];
    const int a = mol_pair / nm2;
    const int b = mol_pair - a * nm2;
    const int na = n1[a];
    const int nb = n2[b];
    const int n_chunks = (nb + jchunk - 1) / jchunk;
    const int i = local / n_chunks;
    const int j_lo = (local - i * n_chunks) * jchunk;
    const int j_hi = min(nb, j_lo + jchunk);

    const int na3 = 3 * na;
    const int nb3 = 3 * nb;
    constexpr int order_c = Order;
    const int order = (order_c > 0) ? order_c : hp.order;
    const int pmax = hp.pmax;
    const int FB = pmax * order;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);
    const int n_entries = na3 * nb3;
    const bool use_sh_H = (h_cap > 0 && n_entries <= h_cap);
    float *g_H = nullptr;
    if (!use_sh_H && Hscratch != nullptr) {
        g_H = Hscratch + static_cast<long long>(blk) * max_na3 * max_nb3;
        for (int e = tid; e < n_entries; e += nthr) {
            g_H[e] = 0.0f;
        }
        // Zero before any j writes; the first j loop's rad load does not need this barrier,
        // but the flush at the end does — sync once here so later j iterations see a clean tile.
        __syncthreads();
    }
    const float s0 = hp.spf[0];
    const float s1 = hp.spf[1];
    const float dscale = hp.dscale;
    const float ascale = hp.ascale;
    const float inv_width = hp.inv_width;
    const float maxgausdist2 = hp.maxgausdist2;

    const int max_nb = max_nb3 / 3;
    float4 *s_rad = hot_smem4;
    int *s_nbq = reinterpret_cast<int *>(hot_smem + 4 * max_nb);
    float2 *s_Z = reinterpret_cast<float2 *>(hot_smem + ((5 * max_nb + 1) & ~1));  // 8 B aligned
    float2 *s_Fsum = s_Z + static_cast<long long>(FB) * max_nb3;
    float *s_dsA = reinterpret_cast<float *>(s_Fsum + FB);
    float *s_dsB = s_dsA + max_na3;
    float *s_X = s_dsB + max_nb3;
    float *s_Y = s_X + max_nb3;
    float *s_G = s_Y + max_nb3;
    float *s_dGp = s_G + max_nb;
    float *s_d2G = s_dGp + max_nb;
    float *s_V = s_d2G + max_nb;
    float *s_ksi = s_V + max_nb;
    float *s_H = s_ksi + max_nb;
    float *s_sij = s_H + h_cap;

    float *Hblk = H + 3LL * atom_base1[a] * ld_H + 3LL * atom_base2[b];

    const int flat_i = a * max1 + i;
    const int nnA = nn1[flat_i];
    const Neigh *ni = nullptr;
    const Fock *fi = nullptr;
    centre_views(blob1, mol_byte_off1, a, i, na, pmax, order, &ni, &fi);

    const Fock *fi_e = fi;
    if constexpr (Alch) {
        fi_e = centre_fock_ptr(blob1_e, mol_byte_off1, a, i, na, pmax, order);
        if (tid < pmax * pmax) {
            s_pdc[tid] = pdc[tid];
        }
    }
    const GradViewsC gi = grad_views(gblob1, mol_gbyte_off1, a, i, na, pmax, order);
    const float *dssA = dss1 + dss_off1[a] + static_cast<long long>(i) * na3;

    // Per-j tiles are zeroed inside the j loop (fused s_ij defers coeff scale until
    // after the p-loop, so each j owns the tile then flushes).

    // Per-mode angular prefactor of channel t = pp * order + m.
    auto spf_of = [&](int t) -> float {
        if constexpr (order_c == 1) {
            return s0;
        } else if constexpr (order_c == 2) {
            return (t & 1) ? s1 : s0;
        } else {
            return hp.spf[t % order];
        }
    };

    auto ang_val = [&](const Fock *fa, const Fock *fb) -> float {
        float ang = 0.0f;
        for (int t = 0; t < FB; ++t) {
            const float2 x = fa[t];
            const float2 y = fb[t];
            ang += (x.x * y.x + x.y * y.y) * spf_of(t);
        }
        return ang;
    };

    for (int j = j_lo; j < j_hi; ++j) {
        const int flat_j = b * max2 + j;
        const int nnB = nn2[flat_j];
        const Neigh *nj = nullptr;
        const Fock *fj = nullptr;
        centre_views(blob2, mol_byte_off2, b, j, nb, pmax, order, &nj, &fj);
        const Fock *fj_e = fj;
        if constexpr (Alch) {
            fj_e = centre_fock_ptr(blob2_e, mol_byte_off2, b, j, nb, pmax, order);
        }
        const GradViewsC gj = grad_views(gblob2, mol_gbyte_off2, b, j, nb, pmax, order);
        const float *dssB = dss2 + dss_off2[b] + static_cast<long long>(j) * nb3;

        const bool use_tile = use_sh_H || (g_H != nullptr);
        float *tile = use_sh_H ? s_H : g_H;

        for (int q = tid; q < nnB; q += nthr) {
            s_rad[q] = gj.rad[q];
            s_nbq[q] = gj.nbr[q];
        }
        for (int e = tid; e < na3; e += nthr) {
            s_dsA[e] = 0.0f;
        }
        for (int e = tid; e < nb3; e += nthr) {
            s_dsB[e] = 0.0f;
        }
        if (use_tile) {
            for (int e = tid; e < n_entries; e += nthr) {
                tile[e] = 0.0f;
            }
        }

        float pd_central = 1.0f;
        bool pair_on;
        if constexpr (Alch) {
            pd_central =
                alch_pd(static_cast<const float *>(c_alchemy_pd), hp.emax, ni[0].z, nj[0].z);
            pair_on = (pd_central != 0.0f);
        } else {
            pair_on = (ni[0].z == nj[0].z);
        }
        if (tid == 0) {
            s_sij[0] = pair_on ? pd_central : 0.0f;
        }
        __syncthreads();

        // Atomic fallback (no private tile): still need kij before Stage C, so keep a
        // lightweight s_ij pre-pass.  Tile path folds s_ij into Stage A and defers scale.
        float coeff = 0.0f, coeff_g = 0.0f, coeff_hess = 0.0f;
        if (!use_tile) {
            if (pair_on) {
                float part = 0.0f;
                for (int p = 1 + tid; p < nnA; p += nthr) {
                    const float rp = ni[p].rij;
                    const float ksi_p = ni[p].ksi;
                    const int zp = ni[p].z;
                    const Fock *fip = fi + static_cast<long long>(p) * FB;
                    for (int q = 1; q < nnB; ++q) {
                        float pd_pq = pd_central;
                        bool ok = true;
                        if constexpr (Alch) {
                            pd_pq *= alch_pd(
                                static_cast<const float *>(c_alchemy_pd), hp.emax, zp, nj[q].z
                            );
                            ok = (pd_pq != 0.0f);
                        } else {
                            ok = (zp == nj[q].z);
                        }
                        if (!ok) {
                            continue;
                        }
                        const float dr = rp - nj[q].rij;
                        const float r2 = dr * dr;
                        if (r2 < maxgausdist2) {
                            const float G = expf(r2 * inv_width) * pd_pq;
                            const float V =
                                ksi_p * nj[q].ksi * dscale +
                                ang_val(fip, fj_e + static_cast<long long>(q) * FB) * ascale;
                            part += G * V;
                        }
                    }
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    part += __shfl_down_sync(0xffffffffu, part, off);
                }
                if ((tid & 31) == 0 && part != 0.0f) {
                    atomicAdd(s_sij, part);
                }
            }
            __syncthreads();
            const float kij = expf(-(ss1[flat_i] + ss2[flat_j] - 2.0f * s_sij[0]) * inv);
            coeff = kij * inv;
            coeff_g = coeff * inv;
            coeff_hess = 2.0f * coeff;
        }

        if (pair_on) {
            for (int p = 1; p < nnA; ++p) {
                const float rp = ni[p].rij;
                const float ksi_p = ni[p].ksi;
                const int zp = ni[p].z;
                // A-side radial gradients of slot p: support on 6 DOFs (nbr atom +u, centre -u).
                const float4 rdA = gi.rad[p];
                const int pat = gi.nbr[p];
                const Fock *fip = fi + static_cast<long long>(p) * FB;
                const Fock *fip_e = fi_e + static_cast<long long>(p) * FB;

                // ---- Stage A: per-q scalars (one thread per q) ----
                for (int q = 1 + tid; q < nnB; q += nthr) {
                    float G = 0.0f, dGp = 0.0f, d2G = 0.0f, V = 0.0f, ksi_q = 0.0f;
                    float pd_pq = pd_central;
                    bool ok = true;
                    if constexpr (Alch) {
                        pd_pq *= alch_pd(
                            static_cast<const float *>(c_alchemy_pd), hp.emax, zp, nj[q].z
                        );
                        ok = (pd_pq != 0.0f);
                    } else {
                        ok = (zp == nj[q].z);
                    }
                    if (ok) {
                        const float dr = rp - nj[q].rij;
                        const float r2 = dr * dr;
                        if (r2 < maxgausdist2) {
                            G = expf(r2 * inv_width) * pd_pq;
                            dGp = G * 2.0f * dr * inv_width;
                            d2G = G * 2.0f * inv_width * (-2.0f * inv_width * r2 - 1.0f);
                            ksi_q = nj[q].ksi;
                            V = ksi_p * ksi_q * dscale +
                                ang_val(fip, fj_e + static_cast<long long>(q) * FB) * ascale;
                        }
                    }
                    s_G[q] = G;
                    s_dGp[q] = dGp;
                    s_d2G[q] = d2G;
                    s_V[q] = V;
                    s_ksi[q] = ksi_q;
                }
                __syncthreads();

                // Tile path: fold s_ij = pd + sum G V into Stage A (no separate pre-pass).
                if (use_tile) {
                    float part = 0.0f;
                    for (int q = 1 + tid; q < nnB; q += nthr) {
                        part += s_G[q] * s_V[q];
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        part += __shfl_down_sync(0xffffffffu, part, off);
                    }
                    if ((tid & 31) == 0 && part != 0.0f) {
                        atomicAdd(s_sij, part);
                    }
                }

                // Redundant per-thread scalar reductions over q (cheap; avoids another barrier).
                float S_dGpV = 0.0f, S_Gksi = 0.0f;
                int n_valid = 0;
                for (int q = 1; q < nnB; ++q) {
                    const float G = s_G[q];
                    if (G == 0.0f) {
                        continue;
                    }
                    ++n_valid;
                    S_dGpV += s_dGp[q] * s_V[q];
                    S_Gksi += G * s_ksi[q];
                }
                if (n_valid == 0) {
                    __syncthreads();  // keep s_G.. reads of this p ordered before next p's writes
                    continue;
                }

                // ---- Stage B: Fsum[t] (t-parallel) and X, Y, Z, dsB (bnu-parallel) ----
                for (int t = tid; t < FB; t += nthr) {
                    float2 acc = make_float2(0.0f, 0.0f);
                    for (int q = 1; q < nnB; ++q) {
                        const float G = s_G[q];
                        if (G == 0.0f) {
                            continue;
                        }
                        const float2 f = fj_e[static_cast<long long>(q) * FB + t];
                        acc.x += G * f.x;
                        acc.y += G * f.y;
                    }
                    s_Fsum[t] = acc;
                }
                for (int bnu = tid; bnu < nb3; bnu += nthr) {
                    float x = 0.0f, y = 0.0f, dsb = 0.0f;
                    const int bat = bnu / 3;
                    const int bmu = bnu - 3 * bat;
                    for (int t = 0; t < FB; ++t) {
                        s_Z[static_cast<long long>(t) * max_nb3 + bnu] = make_float2(0.0f, 0.0f);
                    }
                    for (int q = 1; q < nnB; ++q) {
                        const float G = s_G[q];
                        if (G == 0.0f) {
                            continue;
                        }
                        const float dGp = s_dGp[q];
                        const float d2G = s_d2G[q];
                        const float V = s_V[q];
                        // dri/dksi of slot q on DOF bnu: nonzero only on nbr atom (+u) / centre (-u)
                        const float4 rd = s_rad[q];
                        const float um = (bmu == 0) ? rd.x : ((bmu == 1) ? rd.y : rd.z);
                        const float sgn = (bat == s_nbq[q]) ? 1.0f : ((bat == j) ? -1.0f : 0.0f);
                        const float dri = sgn * um;
                        const float dksi = dri * rd.w;
                        const float c = -dGp * dri;  // dGq * driB
                        float d = 0.0f;
                        const Fock *fjq = fj + static_cast<long long>(q) * FB;
                        for (int t = 0; t < FB; ++t) {
                            const long long ib = (static_cast<long long>(q) * FB + t) * nb3 + bnu;
                            const float dc = gj.dcosp[ib];
                            const float dsn = gj.dsinp[ib];
                            const float2 fa = fip_e[t];
                            d += (fa.x * dc + fa.y * dsn) * spf_of(t);
                            const float2 fq = fjq[t];
                            float2 *z = s_Z + static_cast<long long>(t) * max_nb3 + bnu;
                            float2 zv = *z;
                            zv.x += G * dc + c * fq.x;
                            zv.y += G * dsn + c * fq.y;
                            *z = zv;
                        }
                        const float dVB = dksi * ksi_p * dscale + d * ascale;
                        x += dGp * dVB + d2G * dri * V;
                        y += c * s_ksi[q] + G * dksi;
                        dsb += c * V + G * dVB;
                    }
                    s_dsB[bnu] += dsb;
                    s_X[bnu] = x;
                    s_Y[bnu] = y * dscale;
                    if constexpr (Alch) {
                        for (int m = 0; m < order; ++m) {
                            float2 v[hess::kHessMaxPmax];
                            for (int p2 = 0; p2 < pmax; ++p2) {
                                v[p2] = s_Z[static_cast<long long>(p2 * order + m) * max_nb3 + bnu];
                            }
                            for (int p1 = 0; p1 < pmax; ++p1) {
                                float2 acc = make_float2(0.0f, 0.0f);
                                for (int p2 = 0; p2 < pmax; ++p2) {
                                    const float w = s_pdc[p1 * pmax + p2];
                                    acc.x += w * v[p2].x;
                                    acc.y += w * v[p2].y;
                                }
                                s_Z[static_cast<long long>(p1 * order + m) * max_nb3 + bnu] = acc;
                            }
                        }
                    }
                    for (int t = 0; t < FB; ++t) {
                        float2 *z = s_Z + static_cast<long long>(t) * max_nb3 + bnu;
                        const float sc = ascale * spf_of(t);
                        float2 zv = *z;
                        zv.x *= sc;
                        zv.y *= sc;
                        *z = zv;
                    }
                }
                __syncthreads();

                // ---- Stage C: apply (A-side owners) ----
                for (int amu = tid; amu < na3; amu += nthr) {
                    float acc = 0.0f;
                    {
                        const int aat = amu / 3;
                        const float sg = (aat == pat) ? 1.0f : ((aat == i) ? -1.0f : 0.0f);
                        if (sg != 0.0f) {
                            const int amo = amu - 3 * aat;
                            const float um = (amo == 0) ? rdA.x : ((amo == 1) ? rdA.y : rdA.z);
                            acc = sg * um * (S_dGpV + rdA.w * dscale * S_Gksi);
                        }
                    }
                    float ang = 0.0f;
                    for (int t = 0; t < FB; ++t) {
                        const long long ia = (static_cast<long long>(p) * FB + t) * na3 + amu;
                        const float2 fs = s_Fsum[t];
                        ang += (gi.dcosp[ia] * fs.x + gi.dsinp[ia] * fs.y) * spf_of(t);
                    }
                    s_dsA[amu] += acc + ang * ascale;
                }
                for (int flat = tid; flat < n_entries; flat += nthr) {
                    const int amu = flat / nb3;
                    const int bnu = flat - amu * nb3;
                    float h = 0.0f;
                    {
                        const int aat = amu / 3;
                        const float sg = (aat == pat) ? 1.0f : ((aat == i) ? -1.0f : 0.0f);
                        if (sg != 0.0f) {
                            const int amo = amu - 3 * aat;
                            const float um = (amo == 0) ? rdA.x : ((amo == 1) ? rdA.y : rdA.z);
                            h = sg * um * (s_X[bnu] + rdA.w * s_Y[bnu]);
                        }
                    }
                    for (int t = 0; t < FB; ++t) {
                        const long long ia = (static_cast<long long>(p) * FB + t) * na3 + amu;
                        const float2 z = s_Z[static_cast<long long>(t) * max_nb3 + bnu];
                        h += gi.dcosp[ia] * z.x + gi.dsinp[ia] * z.y;
                    }
                    if (use_tile) {
                        // Unscaled; multiply by coeff_hess after s_ij / kij is known.
                        tile[flat] += h;
                    } else {
                        atomicAdd(
                            Hblk + static_cast<long long>(amu) * ld_H + bnu, coeff_hess * h
                        );
                    }
                }
                // No trailing barrier: the next p's stage-A barrier orders this apply's smem reads
                // (s_X/s_Y/s_Z/s_Fsum) before stage B of the next p overwrites them.  Stage A of
                // p+1 only writes s_G.. which stage C does not read.
            }
        }
        __syncthreads();  // s_dsA / s_dsB / s_sij / tile complete

        if (use_tile) {
            const float kij = expf(-(ss1[flat_i] + ss2[flat_j] - 2.0f * s_sij[0]) * inv);
            coeff = kij * inv;
            coeff_g = coeff * inv;
            coeff_hess = 2.0f * coeff;
        }

        // Scale tile + rank-1, then flush this j (tile path).  Atomic path already wrote
        // scaled Stage C; only rank-1 remains.
        for (int flat = tid; flat < n_entries; flat += nthr) {
            const int amu = flat / nb3;
            const int bnu = flat - amu * nb3;
            const float gA = -dssA[amu] + 2.0f * s_dsA[amu];
            const float gB = -dssB[bnu] + 2.0f * s_dsB[bnu];
            const float r1 = coeff_g * gA * gB;
            if (use_tile) {
                const float wh = coeff_hess * tile[flat] + r1;
                atomicAdd(Hblk + static_cast<long long>(amu) * ld_H + bnu, wh);
            } else {
                atomicAdd(Hblk + static_cast<long long>(amu) * ld_H + bnu, r1);
            }
        }
        __syncthreads();  // before the next j resets s_dsA / s_dsB / s_rad / tile
    }
}

#include "cuda_fchl18_direct_hess_blas.cuh"

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct SideKey {
    const float *coords = nullptr;
    const int *Z = nullptr;
    const int *N = nullptr;
    int nm = 0;
    int max_atoms = 0;
    float sigma = 0;
    float two_body_scaling = 0;
    float two_body_width = 0;
    float two_body_power = 0;
    float three_body_scaling = 0;
    float three_body_width = 0;
    float three_body_power = 0;
    int fourier_order = 0;
    int use_atm = 0;
    int use_alchemy = 0;
    float alchemy_period_width = 0;
    float alchemy_group_width = 0;
    int pmax = 0;
    int z_to_idx[256]{};
    bool valid = false;
};

struct SideBuf {
    int *nn = nullptr;
    char *blob = nullptr;
    char *blob_e = nullptr;
    char *gblob = nullptr;
    float *ss = nullptr;
    float *dss = nullptr;
    long long *mol_byte_off = nullptr;   // nm+1
    long long *mol_gbyte_off = nullptr;  // nm+1
    long long *dss_off = nullptr;        // nm+1 (floats)
    int *atom_base = nullptr;            // nm+1
    size_t nn_cap = 0, blob_cap = 0, blob_e_cap = 0, gblob_cap = 0, ss_cap = 0, dss_cap = 0;
    size_t mol_byte_off_cap = 0, mol_gbyte_off_cap = 0, dss_off_cap = 0, atom_base_cap = 0;
    int max_n = 0;  // largest actual N on this side (host, last call)
    SideKey key{};
};

// Compact molecule-pair -> block map for hess_hot_kernel (int[nm1*nm2+1]), exclusive prefix of
// n1[a] * ceil(n2[b] / jchunk).
int *d_pair_offset = nullptr;
size_t d_pair_offset_cap = 0;
// Private per-block Hij tiles when the shared-memory H tile does not fit (large molecules).
float *d_Hscratch = nullptr;
size_t d_Hscratch_cap = 0;
// cuBLAS pack / tile buffers (large-N path).
hess_blas::Pair *d_blas_pairs = nullptr;
size_t d_blas_pairs_cap = 0;
hess_blas::Centre *d_blas_centres = nullptr;
size_t d_blas_centres_cap = 0;
hess_blas::Target *d_blas_targets = nullptr;
size_t d_blas_targets_cap = 0;
float *d_blas_Dri = nullptr, *d_blas_Dksi = nullptr, *d_blas_X = nullptr, *d_blas_Y = nullptr;
float *d_blas_Afock = nullptr, *d_blas_Z = nullptr, *d_blas_Htile = nullptr;
float *d_blas_gA = nullptr, *d_blas_gB = nullptr, *d_blas_ch = nullptr, *d_blas_cg = nullptr;
size_t d_blas_Dri_cap = 0, d_blas_Dksi_cap = 0, d_blas_X_cap = 0, d_blas_Y_cap = 0;
size_t d_blas_Afock_cap = 0, d_blas_Z_cap = 0, d_blas_Htile_cap = 0;
size_t d_blas_gA_cap = 0, d_blas_gB_cap = 0, d_blas_ch_cap = 0, d_blas_cg_cap = 0;

__host__ inline bool side_key_match(
    const SideKey &k,
    const float *coords,
    const int *Z,
    const int *N,
    int nm,
    int max_atoms,
    float sigma,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    float alchemy_period_width,
    float alchemy_group_width
) {
    return k.valid && k.coords == coords && k.Z == Z && k.N == N && k.nm == nm &&
           k.max_atoms == max_atoms && k.sigma == sigma && k.two_body_scaling == two_body_scaling &&
           k.two_body_width == two_body_width && k.two_body_power == two_body_power &&
           k.three_body_scaling == three_body_scaling && k.three_body_width == three_body_width &&
           k.three_body_power == three_body_power && k.fourier_order == fourier_order &&
           k.use_atm == (use_atm ? 1 : 0) && k.use_alchemy == (use_alchemy ? 1 : 0) &&
           k.alchemy_period_width == alchemy_period_width &&
           k.alchemy_group_width == alchemy_group_width;
}

__host__ inline void side_key_store(
    SideKey &k,
    const float *coords,
    const int *Z,
    const int *N,
    int nm,
    int max_atoms,
    float sigma,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    float alchemy_period_width,
    float alchemy_group_width,
    int pmax,
    const int *z_to_idx
) {
    k.coords = coords;
    k.Z = Z;
    k.N = N;
    k.nm = nm;
    k.max_atoms = max_atoms;
    k.sigma = sigma;
    k.two_body_scaling = two_body_scaling;
    k.two_body_width = two_body_width;
    k.two_body_power = two_body_power;
    k.three_body_scaling = three_body_scaling;
    k.three_body_width = three_body_width;
    k.three_body_power = three_body_power;
    k.fourier_order = fourier_order;
    k.use_atm = use_atm ? 1 : 0;
    k.use_alchemy = use_alchemy ? 1 : 0;
    k.alchemy_period_width = alchemy_period_width;
    k.alchemy_group_width = alchemy_group_width;
    k.pmax = pmax;
    std::memcpy(k.z_to_idx, z_to_idx, sizeof(k.z_to_idx));
    k.valid = true;
}

// Host-side build + upload of all per-side offsets and buffer allocation.
void prepare_side(SideBuf &side, const int *d_N, int nm, int max_atoms, int pmax, int order) {
    std::vector<int> h_N(static_cast<size_t>(nm));
    CUDA_CHECK(cudaMemcpy(
        h_N.data(), d_N, static_cast<size_t>(nm) * sizeof(int), cudaMemcpyDeviceToHost
    ));
    std::vector<long long> boff(static_cast<size_t>(nm) + 1, 0);
    std::vector<long long> goff(static_cast<size_t>(nm) + 1, 0);
    std::vector<long long> doff(static_cast<size_t>(nm) + 1, 0);
    std::vector<int> abase(static_cast<size_t>(nm) + 1, 0);
    side.max_n = 0;
    for (int a = 0; a < nm; ++a) {
        const int na = h_N[static_cast<size_t>(a)];
        if (na < 0 || na > max_atoms) {
            throw std::invalid_argument("cuda_fchl18_direct_hess: N out of range [0, max_atoms]");
        }
        side.max_n = std::max(side.max_n, na);
        const size_t u = static_cast<size_t>(a);
        boff[u + 1] = boff[u] + static_cast<long long>(
                                    static_cast<size_t>(na) * centre_stride_bytes(na, pmax, order)
                                );
        goff[u + 1] = goff[u] + static_cast<long long>(
                                    static_cast<size_t>(na) *
                                    grad_centre_stride_bytes(na, pmax, order)
                                );
        doff[u + 1] = doff[u] + static_cast<long long>(na) * 3 * na;
        abase[u + 1] = abase[u] + na;
    }
    const size_t nbytes = static_cast<size_t>(boff[static_cast<size_t>(nm)]);
    const size_t gbytes = static_cast<size_t>(goff[static_cast<size_t>(nm)]);
    const size_t ndss = static_cast<size_t>(doff[static_cast<size_t>(nm)]);

    {
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        if (gbytes > free_b) {
            throw std::runtime_error(
                "cuda_fchl18_direct_hess: gradient slab (" + std::to_string(gbytes >> 20) +
                " MiB) does not fit in free GPU memory (" + std::to_string(free_b >> 20) +
                " MiB)"
            );
        }
    }

    const size_t n_flat = static_cast<size_t>(nm) * max_atoms;
    ensure_capacity(&side.nn, &side.nn_cap, n_flat);
    ensure_capacity(&side.ss, &side.ss_cap, n_flat);
    ensure_capacity(&side.blob, &side.blob_cap, std::max<size_t>(nbytes, 16));
    ensure_capacity(&side.gblob, &side.gblob_cap, std::max<size_t>(gbytes, 16));
    ensure_capacity(&side.dss, &side.dss_cap, std::max<size_t>(ndss, 1));
    ensure_capacity(&side.mol_byte_off, &side.mol_byte_off_cap, static_cast<size_t>(nm) + 1);
    ensure_capacity(&side.mol_gbyte_off, &side.mol_gbyte_off_cap, static_cast<size_t>(nm) + 1);
    ensure_capacity(&side.dss_off, &side.dss_off_cap, static_cast<size_t>(nm) + 1);
    ensure_capacity(&side.atom_base, &side.atom_base_cap, static_cast<size_t>(nm) + 1);
    const size_t o = (static_cast<size_t>(nm) + 1);
    CUDA_CHECK(cudaMemcpy(side.mol_byte_off, boff.data(), o * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(side.mol_gbyte_off, goff.data(), o * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(side.dss_off, doff.data(), o * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(side.atom_base, abase.data(), o * sizeof(int), cudaMemcpyHostToDevice));
}

}  // namespace

void kernel_gaussian_hessian_fp32(
    const float *d_coords1,
    const int *d_Z1,
    const int *d_N1,
    const float *d_coords2,
    const int *d_Z2,
    const int *d_N2,
    float *d_H_out,
    float sigma,
    int nm1,
    int nm2,
    int max_atoms1,
    int max_atoms2,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    float alchemy_period_width,
    float alchemy_group_width
) {
    if (nm1 <= 0 || nm2 <= 0) {
        return;
    }
    if (fourier_order < 1 || fourier_order > hess::kHessMaxOrder) {
        throw std::invalid_argument(
            "cuda_fchl18_direct_hess: fourier_order must be in [1, " +
            std::to_string(hess::kHessMaxOrder) + "]"
        );
    }
    // Even n = m + 1 Fourier modes vanish identically (cos(n th) - cos(n (th + pi)) = 0), so
    // only the odd-n modes m = 0, 2, 4, ... are stored/used: order_eff of them, stored at
    // index mi (m = 2 mi) everywhere downstream (blob layout, FB, prefactors).
    const int order_eff = (fourier_order + 1) / 2;
    if (max_atoms1 > hess::kHessMaxAtoms || max_atoms2 > hess::kHessMaxAtoms) {
        throw std::invalid_argument(
            "cuda_fchl18_direct_hess: max_atoms > " + std::to_string(hess::kHessMaxAtoms) +
            " not supported"
        );
    }

    static SideBuf side1, side2;
    static int *d_z_present = nullptr;
    static float *d_s_prefactor = nullptr;
    if (!d_z_present) {
        CUDA_CHECK(cudaMalloc(&d_z_present, 256 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_s_prefactor, kMaxFourierOrder * sizeof(float)));
    }

    const bool same = (d_coords1 == d_coords2 && d_Z1 == d_Z2 && d_N1 == d_N2 && nm1 == nm2 &&
                       max_atoms1 == max_atoms2);

    bool hit1 = side_key_match(
        side1.key, d_coords1, d_Z1, d_N1, nm1, max_atoms1, sigma, two_body_scaling, two_body_width,
        two_body_power, three_body_scaling, three_body_width, three_body_power, fourier_order,
        use_atm, use_alchemy, alchemy_period_width, alchemy_group_width
    );
    bool hit2 =
        same ? hit1
             : side_key_match(
                   side2.key, d_coords2, d_Z2, d_N2, nm2, max_atoms2, sigma, two_body_scaling,
                   two_body_width, two_body_power, three_body_scaling, three_body_width,
                   three_body_power, fourier_order, use_atm, use_alchemy, alchemy_period_width,
                   alchemy_group_width
               );

    const int block = 256;
    const size_t n_flat1 = static_cast<size_t>(nm1) * max_atoms1;
    const size_t n_flat2 = static_cast<size_t>(nm2) * max_atoms2;
    const int grid1 = static_cast<int>((n_flat1 + block - 1) / block);
    const int grid2 = static_cast<int>((n_flat2 + block - 1) / block);

    Params params{};
    params.two_body_power = two_body_power;
    params.three_body_power = three_body_power;
    params.fourier_order = order_eff;
    params.use_atm = use_atm ? 1 : 0;

    bool map_ok = false;
    if (hit1 && hit2) {
        if (same || (side1.key.pmax == side2.key.pmax &&
                     std::memcmp(side1.key.z_to_idx, side2.key.z_to_idx, sizeof(side1.key.z_to_idx)) ==
                         0)) {
            params.pmax = side1.key.pmax;
            std::memcpy(params.z_to_idx, side1.key.z_to_idx, sizeof(params.z_to_idx));
            map_ok = true;
        }
    }
    if (!map_ok) {
        hit1 = false;
        hit2 = false;
        CUDA_CHECK(cudaMemset(d_z_present, 0, 256 * sizeof(int)));
        mark_z_kernel<<<grid1, block>>>(d_Z1, d_N1, nm1, max_atoms1, d_z_present);
        CUDA_CHECK(cudaGetLastError());
        if (!same) {
            mark_z_kernel<<<grid2, block>>>(d_Z2, d_N2, nm2, max_atoms2, d_z_present);
            CUDA_CHECK(cudaGetLastError());
        }
        int h_z_present[256];
        CUDA_CHECK(cudaMemcpy(h_z_present, d_z_present, 256 * sizeof(int), cudaMemcpyDeviceToHost));
        params.pmax = build_element_map_from_flags(h_z_present, params.z_to_idx);
    }

    if (!hit1) {
        prepare_side(side1, d_N1, nm1, max_atoms1, params.pmax, order_eff);
    }
    if (!same && !hit2) {
        prepare_side(side2, d_N2, nm2, max_atoms2, params.pmax, order_eff);
    }
    SideBuf &sb2 = same ? side1 : side2;

    std::vector<int> h_ab(2);
    CUDA_CHECK(cudaMemcpy(&h_ab[0], side1.atom_base + nm1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_ab[1], sb2.atom_base + nm2, sizeof(int), cudaMemcpyDeviceToHost));
    const long long D_A = 3LL * h_ab[0];
    const long long D_B = 3LL * h_ab[1];
    CUDA_CHECK(cudaMemset(d_H_out, 0, static_cast<size_t>(D_A) * static_cast<size_t>(D_B) *
                                          sizeof(float)));
    if (params.pmax == 0) {
        return;
    }
    if (params.pmax > hess::kHessMaxPmax) {
        throw std::invalid_argument(
            "cuda_fchl18_direct_hess: more than " + std::to_string(hess::kHessMaxPmax) +
            " distinct elements not supported"
        );
    }

    // Shared-memory budget (floats), see hess_hot_kernel:
    //   rad+nbq[5 max_nb] | Z[2 FB max_nb3] | Fsum[2 FB] | dsA | dsB | X | Y | 5 * max_nb | H | sij.
    // If the full Hij tile does not fit, h_cap = 0 and the kernel uses its atomic fallback.
    const int max_na3 = 3 * side1.max_n;
    const int max_nb3 = 3 * sb2.max_n;
    const int FBmax = params.pmax * order_eff;
    const size_t fixed_floats =
        2 * static_cast<size_t>(FBmax) * max_nb3 + 2 * static_cast<size_t>(FBmax) +
        static_cast<size_t>(max_na3) + 3 * static_cast<size_t>(max_nb3) +
        10 * static_cast<size_t>(sb2.max_n) + 2;  // +1 pad so s_Z stays 8 B aligned
    int h_cap = max_na3 * max_nb3;
    size_t smem_bytes = (fixed_floats + static_cast<size_t>(h_cap)) * sizeof(float);
    {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        int smem_optin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        if (smem_bytes > static_cast<size_t>(smem_optin)) {
            h_cap = 0;
            smem_bytes = fixed_floats * sizeof(float);
            if (smem_bytes > static_cast<size_t>(smem_optin)) {
                throw std::runtime_error(
                    "cuda_fchl18_direct_hess: factored Hessian buffers exceed shared memory"
                );
            }
        }
    }

    // Scalar / Hessian parameters.
    float h_s_prefactor[kMaxFourierOrder]{};
    const double ang_norm2 = get_angular_norm2(static_cast<double>(three_body_width));
    const double g1 = std::sqrt(2.0 * 3.14159265358979323846) / ang_norm2;
    for (int mi = 0; mi < order_eff; ++mi) {
        const double mf = static_cast<double>(2 * mi + 1);  // retained mode n = m + 1, m = 2 mi
        const double tw = static_cast<double>(three_body_width);
        h_s_prefactor[mi] = static_cast<float>(g1 * std::exp(-(tw * mf) * (tw * mf) / 2.0));
    }
    CUDA_CHECK(cudaMemcpy(
        d_s_prefactor, h_s_prefactor, kMaxFourierOrder * sizeof(float), cudaMemcpyHostToDevice
    ));

    const float true_distance_scale = two_body_scaling / 16.0f;
    const float true_angular_scale = three_body_scaling / static_cast<float>(std::sqrt(8.0));
    const int emax = use_alchemy ? kAlchemyEmaxDefault : 0;

    hess::ScalarParams sp{};
    sp.two_body_width = two_body_width;
    sp.true_distance_scale = true_distance_scale;
    sp.true_angular_scale = true_angular_scale;
    sp.fourier_order = order_eff;
    sp.pmax = params.pmax;
    sp.emax = emax;
    sp.s_prefactor = d_s_prefactor;

    HotParams hp{};
    hp.inv_width = -1.0f / (4.0f * two_body_width * two_body_width);
    hp.maxgausdist2 = (8.0f * two_body_width) * (8.0f * two_body_width);
    hp.dscale = true_distance_scale;
    hp.ascale = true_angular_scale;
    for (int m = 0; m < hess::kHessMaxOrder; ++m) {
        hp.spf[m] = (m < order_eff) ? h_s_prefactor[m] : 0.0f;
    }
    hp.order = order_eff;
    hp.pmax = params.pmax;
    hp.emax = emax;

    const float inv = 0.5f / (sigma * sigma);

    const float *d_pdc = nullptr;
    if (use_alchemy) {
        const float *d_pd = alchemy_upload_pd<float>(
            static_cast<double>(alchemy_period_width), static_cast<double>(alchemy_group_width)
        );
        CUDA_CHECK(cudaMemcpyToSymbol(
            c_alchemy_pd, d_pd,
            static_cast<size_t>(kAlchemyEmaxDefault) * kAlchemyEmaxDefault * sizeof(float), 0,
            cudaMemcpyDeviceToDevice
        ));
        d_pdc = alchemy_upload_pdc<float>(params.z_to_idx, params.pmax);
    }

    // 3-5. per-side precompute, alchemy mix, self scalars + gradients
    auto build_side = [&](const float *coords, const int *Z, const int *N, SideBuf &side, int nm,
                          int max_atoms, int grid) {
        if (side.gblob_cap > 0) {
            CUDA_CHECK(cudaMemset(side.gblob, 0, side.gblob_cap));
        }
        if (use_atm) {
            precompute_kernel<true><<<grid, block>>>(
                coords, Z, N, side.mol_byte_off, side.mol_gbyte_off, side.nn, side.blob,
                side.gblob, nm, max_atoms, params
            );
        } else {
            precompute_kernel<false><<<grid, block>>>(
                coords, Z, N, side.mol_byte_off, side.mol_gbyte_off, side.nn, side.blob,
                side.gblob, nm, max_atoms, params
            );
        }
        CUDA_CHECK(cudaGetLastError());
        if (use_alchemy) {
            alchemy_mix_fock_blob_kernel<<<grid, block>>>(
                N, side.nn, side.mol_byte_off, side.blob, side.blob_e, d_pdc, nm, max_atoms,
                params.pmax, order_eff
            );
            CUDA_CHECK(cudaGetLastError());
            self_kernel<true><<<grid, block>>>(
                N, side.nn, side.mol_byte_off, side.mol_gbyte_off, side.dss_off, side.blob,
                side.blob_e, side.gblob, side.ss, side.dss, nm, max_atoms, sp
            );
        } else {
            self_kernel<false><<<grid, block>>>(
                N, side.nn, side.mol_byte_off, side.mol_gbyte_off, side.dss_off, side.blob,
                nullptr, side.gblob, side.ss, side.dss, nm, max_atoms, sp
            );
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // blob_e needs capacity before the mix kernel; sizes follow blob.
    if (use_alchemy) {
        if (!hit1) {
            ensure_capacity(&side1.blob_e, &side1.blob_e_cap, side1.blob_cap);
        }
        if (!same && !hit2) {
            ensure_capacity(&side2.blob_e, &side2.blob_e_cap, side2.blob_cap);
        }
    }
    if (!hit1) {
        build_side(d_coords1, d_Z1, d_N1, side1, nm1, max_atoms1, grid1);
        side_key_store(
            side1.key, d_coords1, d_Z1, d_N1, nm1, max_atoms1, sigma, two_body_scaling,
            two_body_width, two_body_power, three_body_scaling, three_body_width, three_body_power,
            fourier_order, use_atm, use_alchemy, alchemy_period_width, alchemy_group_width,
            params.pmax, params.z_to_idx
        );
    }
    if (!same && !hit2) {
        build_side(d_coords2, d_Z2, d_N2, side2, nm2, max_atoms2, grid2);
        side_key_store(
            side2.key, d_coords2, d_Z2, d_N2, nm2, max_atoms2, sigma, two_body_scaling,
            two_body_width, two_body_power, three_body_scaling, three_body_width, three_body_power,
            fourier_order, use_atm, use_alchemy, alchemy_period_width, alchemy_group_width,
            params.pmax, params.z_to_idx
        );
    }

    // 6. Hessian blocks.
    const int n_pairs = nm1 * nm2;
    char *be1 = use_alchemy ? side1.blob_e : nullptr;
    char *be2 = use_alchemy ? sb2.blob_e : nullptr;

    std::vector<int> h_n1(static_cast<size_t>(nm1));
    std::vector<int> h_n2(static_cast<size_t>(nm2));
    CUDA_CHECK(cudaMemcpy(h_n1.data(), d_N1, h_n1.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_n2.data(), d_N2, h_n2.size() * sizeof(int), cudaMemcpyDeviceToHost));

    const int max_n_both = std::max(side1.max_n, sb2.max_n);
    const int n_ref1 = h_n1.empty() ? 0 : h_n1[0];
    const int n_ref2 = h_n2.empty() ? 0 : h_n2[0];
    bool uniform = (n_ref1 > 0 && n_ref2 > 0);
    for (int a = 0; uniform && a < nm1; ++a) {
        uniform = uniform && (h_n1[static_cast<size_t>(a)] == n_ref1);
    }
    for (int b = 0; uniform && b < nm2; ++b) {
        uniform = uniform && (h_n2[static_cast<size_t>(b)] == n_ref2);
    }
    // Centre-wide GEMM (Dri @ [X_j…]): opt-in via KF_HESS_WIDE=1.  Benchmarks show
    // pack still dominates for ≤16-atom mols (ethanol 100×100 ~1.5× slower than
    // in-kernel), so it is not selected automatically.  Larger mols use fused /
    // parallel-ij BLAS when max_n >= kBlasAtomThresh; else in-kernel.
    const int n_targ_all = uniform ? (nm2 * n_ref2) : 0;
    const int wide_n_all = n_targ_all * max_nb3;
    bool use_wide = false;
    if (const char *env = std::getenv("KF_HESS_WIDE"); env && env[0] == '1') {
        use_wide = uniform && max_n_both <= hess_blas::kWideAtomMax && max_n_both >= 1 &&
                   wide_n_all >= hess_blas::kWideMinN;
    }
    const bool use_blas = use_wide || (max_n_both >= hess_blas::kBlasAtomThresh);
    const int same_i = same ? 1 : 0;

    if (use_wide) {
        std::vector<hess_blas::Centre> h_centres;
        h_centres.reserve(static_cast<size_t>(nm1) * n_ref1);
        for (int a = 0; a < nm1; ++a) {
            for (int i = 0; i < n_ref1; ++i) {
                h_centres.push_back(hess_blas::Centre{a, i});
            }
        }
        std::vector<hess_blas::Target> h_targets(static_cast<size_t>(n_targ_all));
        for (int b = 0; b < nm2; ++b) {
            for (int j = 0; j < n_ref2; ++j) {
                h_targets[static_cast<size_t>(b * n_ref2 + j)] = hess_blas::Target{b, j};
            }
        }

        const int n_centres = static_cast<int>(h_centres.size());
        const int P = max_n_both;
        const int FB = FBmax;
        const int K1 = P * 2 * FB;
        const size_t fixed_pack_smem = fixed_floats * sizeof(float);
        {
            int dev = 0, smem_optin = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            CUDA_CHECK(
                cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)
            );
            if (fixed_pack_smem > static_cast<size_t>(smem_optin)) {
                throw std::runtime_error(
                    "cuda_fchl18_direct_hess: wide pack buffers exceed shared memory"
                );
            }
            if (fixed_pack_smem > 48 * 1024) {
                auto set_smem = [&](auto *fn) {
                    CUDA_CHECK(cudaFuncSetAttribute(
                        fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(fixed_pack_smem)
                    ));
                };
                if (use_alchemy) {
                    if (order_eff == 1) {
                        set_smem(hess_blas::blas_pack_centre_b_kernel<true, 1>);
                    } else {
                        set_smem(hess_blas::blas_pack_centre_b_kernel<true, 0>);
                    }
                } else if (order_eff == 1) {
                    set_smem(hess_blas::blas_pack_centre_b_kernel<false, 1>);
                } else {
                    set_smem(hess_blas::blas_pack_centre_b_kernel<false, 0>);
                }
            }
        }

        // Target tiles keep GEMM n large while bounding workspace (esp. 1K ethanol).
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const size_t budget = free_b / 2;
        auto bytes_per_centre = [&](int nt) -> size_t {
            const int wn = max_nb3 * nt;
            return (2ull * max_na3 * P + 2ull * P * wn + 1ull * max_na3 * K1 + 1ull * K1 * wn +
                    1ull * max_na3 * wn + 1ull * max_na3 * nt + 1ull * max_nb3 * nt) *
                   sizeof(float);
        };
        int t_batch = n_targ_all;
        int c_batch = 1;
        {
            // Prefer wide_n around 8k–16k columns; shrink if needed to fit ≥1 centre.
            const int prefer_n = 8192;
            t_batch = std::min(n_targ_all, std::max(1, prefer_n / std::max(max_nb3, 1)));
            while (t_batch > 1 && bytes_per_centre(t_batch) > budget) {
                t_batch = std::max(1, t_batch / 2);
            }
            if (bytes_per_centre(t_batch) > budget) {
                throw std::runtime_error(
                    "cuda_fchl18_direct_hess: insufficient GPU memory for centre-wide BLAS"
                );
            }
            c_batch = static_cast<int>(std::max<size_t>(
                1, std::min<size_t>(
                       budget / bytes_per_centre(t_batch),
                       static_cast<size_t>(std::min(n_centres, hess_blas::kBlasBatchDefault))
                   )
            ));
        }
        const int wide_n = max_nb3 * t_batch;

        ensure_capacity(&d_blas_centres, &d_blas_centres_cap, static_cast<size_t>(c_batch));
        ensure_capacity(&d_blas_targets, &d_blas_targets_cap, static_cast<size_t>(t_batch));
        ensure_capacity(&d_blas_Dri, &d_blas_Dri_cap, static_cast<size_t>(c_batch) * max_na3 * P);
        ensure_capacity(
            &d_blas_Dksi, &d_blas_Dksi_cap, static_cast<size_t>(c_batch) * max_na3 * P
        );
        ensure_capacity(&d_blas_X, &d_blas_X_cap, static_cast<size_t>(c_batch) * P * wide_n);
        ensure_capacity(&d_blas_Y, &d_blas_Y_cap, static_cast<size_t>(c_batch) * P * wide_n);
        ensure_capacity(
            &d_blas_Afock, &d_blas_Afock_cap, static_cast<size_t>(c_batch) * max_na3 * K1
        );
        ensure_capacity(&d_blas_Z, &d_blas_Z_cap, static_cast<size_t>(c_batch) * K1 * wide_n);
        ensure_capacity(
            &d_blas_Htile, &d_blas_Htile_cap, static_cast<size_t>(c_batch) * max_na3 * wide_n
        );
        ensure_capacity(
            &d_blas_gA, &d_blas_gA_cap, static_cast<size_t>(c_batch) * max_na3 * t_batch
        );
        ensure_capacity(
            &d_blas_gB, &d_blas_gB_cap, static_cast<size_t>(c_batch) * max_nb3 * t_batch
        );

#define KF_LAUNCH_WIDE_A(AL, ORD)                                                                 \
        do {                                                                                      \
            hess_blas::blas_pack_centre_a_kernel<AL, ORD><<<B, 128>>>(                            \
                d_blas_centres, B, d_N1, side1.nn, side1.mol_byte_off, side1.mol_gbyte_off,       \
                side1.gblob, max_atoms1, hp, max_na3, P, FB, d_blas_Dri, d_blas_Dksi,             \
                d_blas_Afock                                                                      \
            );                                                                                    \
        } while (0)
#define KF_LAUNCH_WIDE_B(AL, ORD)                                                                 \
        do {                                                                                      \
            hess_blas::blas_pack_centre_b_kernel<AL, ORD><<<B * T, 128, fixed_pack_smem>>>(       \
                d_blas_centres, B, d_blas_targets, T, same_i, d_N1, d_N2, side1.nn, sb2.nn,       \
                side1.mol_byte_off, sb2.mol_byte_off, side1.mol_gbyte_off, sb2.mol_gbyte_off,     \
                side1.dss_off, sb2.dss_off, side1.blob, sb2.blob, be1, be2, side1.gblob,          \
                sb2.gblob, side1.ss, sb2.ss, side1.dss, sb2.dss, max_atoms1, max_atoms2, inv, hp, \
                d_pdc, max_na3, max_nb3, P, FB, d_blas_X, d_blas_Y, d_blas_Z, d_blas_gA,          \
                d_blas_gB                                                                         \
            );                                                                                    \
        } while (0)

        for (int off = 0; off < n_centres; off += c_batch) {
            const int B = std::min(c_batch, n_centres - off);
            CUDA_CHECK(cudaMemcpy(
                d_blas_centres, h_centres.data() + off,
                static_cast<size_t>(B) * sizeof(hess_blas::Centre), cudaMemcpyHostToDevice
            ));
            if (use_alchemy) {
                if (order_eff == 1) {
                    KF_LAUNCH_WIDE_A(true, 1);
                } else {
                    KF_LAUNCH_WIDE_A(true, 0);
                }
            } else if (order_eff == 1) {
                KF_LAUNCH_WIDE_A(false, 1);
            } else {
                KF_LAUNCH_WIDE_A(false, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            for (int toff = 0; toff < n_targ_all; toff += t_batch) {
                const int T = std::min(t_batch, n_targ_all - toff);
                const int wn = max_nb3 * T;
                CUDA_CHECK(cudaMemcpy(
                    d_blas_targets, h_targets.data() + toff,
                    static_cast<size_t>(T) * sizeof(hess_blas::Target), cudaMemcpyHostToDevice
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_X, 0, static_cast<size_t>(B) * P * wn * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_Y, 0, static_cast<size_t>(B) * P * wn * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_Z, 0, static_cast<size_t>(B) * K1 * wn * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_gA, 0, static_cast<size_t>(B) * max_na3 * T * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_gB, 0, static_cast<size_t>(B) * max_nb3 * T * sizeof(float)
                ));
                if (use_alchemy) {
                    if (order_eff == 1) {
                        KF_LAUNCH_WIDE_B(true, 1);
                    } else {
                        KF_LAUNCH_WIDE_B(true, 0);
                    }
                } else if (order_eff == 1) {
                    KF_LAUNCH_WIDE_B(false, 1);
                } else {
                    KF_LAUNCH_WIDE_B(false, 0);
                }
                CUDA_CHECK(cudaGetLastError());
                hess_blas::apply_centre_wide(
                    B, T, max_na3, max_nb3, P, K1, d_blas_Dri, d_blas_Dksi, d_blas_X, d_blas_Y,
                    d_blas_Afock, d_blas_Z, d_blas_Htile, d_blas_gA, d_blas_gB, d_blas_centres,
                    d_blas_targets, d_N1, d_N2, side1.atom_base, sb2.atom_base, d_H_out, D_B,
                    same_i
                );
            }
        }
#undef KF_LAUNCH_WIDE_A
#undef KF_LAUNCH_WIDE_B

        if (same && nm1 > 1) {
            const long long n_lower = static_cast<long long>(nm1) * (nm1 - 1) / 2;
            mirror_lower_kernel<<<static_cast<unsigned int>(n_lower), 256>>>(
                d_N1, side1.atom_base, d_H_out, D_B
            );
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (use_blas) {
        // Fixed-size sides → fuse all (i,j) of a mol pair into one large-K GEMM, batch mol pairs.
        const int n_ij_fuse = uniform ? (n_ref1 * n_ref2) : 0;

        // Bulk mol-pack only when n_ij is small (serial (i,j) in one block is fine).
        // Large n_ij (big single mols): parallel one-block-per-(i,j) into fused columns.
        const int bulk_mol = (n_ij_fuse > 0 && n_ij_fuse <= 128) ? 1 : 0;

        std::vector<hess_blas::Pair> h_mol_pairs;
        std::vector<hess_blas::Pair> h_ij_pairs;
        if (uniform) {
            for (int a = 0; a < nm1; ++a) {
                for (int b = 0; b < nm2; ++b) {
                    if (same && b > a) {
                        continue;
                    }
                    h_mol_pairs.push_back(hess_blas::Pair{a, b, 0, 0});
                    for (int i = 0; i < n_ref1; ++i) {
                        for (int j = 0; j < n_ref2; ++j) {
                            h_ij_pairs.push_back(hess_blas::Pair{a, b, i, j});
                        }
                    }
                }
            }
        } else {
            for (int a = 0; a < nm1; ++a) {
                for (int b = 0; b < nm2; ++b) {
                    if (same && b > a) {
                        continue;
                    }
                    for (int i = 0; i < h_n1[static_cast<size_t>(a)]; ++i) {
                        for (int j = 0; j < h_n2[static_cast<size_t>(b)]; ++j) {
                            h_ij_pairs.push_back(hess_blas::Pair{a, b, i, j});
                        }
                    }
                }
            }
        }
        const int n_mol_total =
            uniform ? static_cast<int>(h_mol_pairs.size()) : static_cast<int>(h_ij_pairs.size());
        if (n_mol_total == 0) {
            return;
        }
        const std::vector<hess_blas::Pair> &h_pack_pairs =
            (n_ij_fuse > 0 && bulk_mol) ? h_mol_pairs : h_ij_pairs;

        const int P = max_n_both;
        const int FB = FBmax;
        const int K1 = P * 2 * FB;
        const int k_rad = (n_ij_fuse > 0) ? n_ij_fuse * P : P;
        const int k_fock = (n_ij_fuse > 0) ? n_ij_fuse * K1 : K1;
        const size_t fixed_pack_smem = fixed_floats * sizeof(float);
        {
            int dev = 0, smem_optin = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            CUDA_CHECK(
                cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)
            );
            if (fixed_pack_smem > static_cast<size_t>(smem_optin)) {
                throw std::runtime_error(
                    "cuda_fchl18_direct_hess: BLAS pack buffers exceed shared memory"
                );
            }
        }

        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const size_t per_mol =
            (2ull * max_na3 * k_rad + 2ull * k_rad * max_nb3 + 1ull * max_na3 * k_fock +
             1ull * k_fock * max_nb3 + 1ull * max_na3 * max_nb3 +
             1ull * max_na3 * std::max(n_ij_fuse, 1) + 1ull * max_nb3 * std::max(n_ij_fuse, 1) +
             2ull) *
            sizeof(float);
        int mol_batch_cap = hess_blas::kBlasBatchDefault;
        {
            const size_t budget = free_b / 2;
            mol_batch_cap = static_cast<int>(std::max<size_t>(
                1, std::min<size_t>(budget / std::max<size_t>(per_mol, 1),
                                    static_cast<size_t>(n_mol_total))
            ));
            mol_batch_cap = std::min(mol_batch_cap, hess_blas::kBlasBatchDefault);
            int n_sm = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0));
            mol_batch_cap = std::max(mol_batch_cap, std::min(n_mol_total, n_sm * 8));
            mol_batch_cap = static_cast<int>(std::min<size_t>(
                static_cast<size_t>(mol_batch_cap), budget / std::max<size_t>(per_mol, 1)
            ));
            mol_batch_cap = std::max(1, mol_batch_cap);
        }
        const int pack_slots_per_mol = (n_ij_fuse > 0 && !bulk_mol) ? n_ij_fuse : 1;
        ensure_capacity(
            &d_blas_pairs, &d_blas_pairs_cap,
            static_cast<size_t>(mol_batch_cap) * static_cast<size_t>(pack_slots_per_mol)
        );
        ensure_capacity(
            &d_blas_Dri, &d_blas_Dri_cap, static_cast<size_t>(mol_batch_cap) * max_na3 * k_rad
        );
        ensure_capacity(
            &d_blas_Dksi, &d_blas_Dksi_cap, static_cast<size_t>(mol_batch_cap) * max_na3 * k_rad
        );
        ensure_capacity(
            &d_blas_X, &d_blas_X_cap, static_cast<size_t>(mol_batch_cap) * k_rad * max_nb3
        );
        ensure_capacity(
            &d_blas_Y, &d_blas_Y_cap, static_cast<size_t>(mol_batch_cap) * k_rad * max_nb3
        );
        ensure_capacity(
            &d_blas_Afock, &d_blas_Afock_cap,
            static_cast<size_t>(mol_batch_cap) * max_na3 * k_fock
        );
        ensure_capacity(
            &d_blas_Z, &d_blas_Z_cap, static_cast<size_t>(mol_batch_cap) * k_fock * max_nb3
        );
        ensure_capacity(
            &d_blas_Htile, &d_blas_Htile_cap,
            static_cast<size_t>(mol_batch_cap) * max_na3 * max_nb3
        );
        const int g_cols = std::max(n_ij_fuse, 1);
        ensure_capacity(
            &d_blas_gA, &d_blas_gA_cap, static_cast<size_t>(mol_batch_cap) * max_na3 * g_cols
        );
        ensure_capacity(
            &d_blas_gB, &d_blas_gB_cap, static_cast<size_t>(mol_batch_cap) * max_nb3 * g_cols
        );
        ensure_capacity(&d_blas_ch, &d_blas_ch_cap, static_cast<size_t>(mol_batch_cap));
        ensure_capacity(&d_blas_cg, &d_blas_cg_cap, static_cast<size_t>(mol_batch_cap));

        if (fixed_pack_smem > 48 * 1024) {
            auto set_smem = [&](auto *fn) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(fixed_pack_smem)
                ));
            };
            if (use_alchemy) {
                if (order_eff == 1) {
                    set_smem(hess_blas::blas_pack_kernel<true, 1>);
                } else {
                    set_smem(hess_blas::blas_pack_kernel<true, 0>);
                }
            } else if (order_eff == 1) {
                set_smem(hess_blas::blas_pack_kernel<false, 1>);
            } else {
                set_smem(hess_blas::blas_pack_kernel<false, 0>);
            }
        }

#define KF_LAUNCH_BLAS_PACK(AL, ORD)                                                              \
        do {                                                                                      \
            hess_blas::blas_pack_kernel<AL, ORD><<<n_pack, 128, fixed_pack_smem>>>(               \
                d_blas_pairs, n_pack, d_N1, d_N2, side1.nn, sb2.nn, side1.mol_byte_off,           \
                sb2.mol_byte_off, side1.mol_gbyte_off, sb2.mol_gbyte_off, side1.dss_off,          \
                sb2.dss_off, side1.blob, sb2.blob, be1, be2, side1.gblob, sb2.gblob, side1.ss,    \
                sb2.ss, side1.dss, sb2.dss, max_atoms1, max_atoms2, inv, hp, d_pdc, max_na3,      \
                max_nb3, P, FB, n_ij_fuse, bulk_mol, d_blas_Dri, d_blas_Dksi, d_blas_X,           \
                d_blas_Y, d_blas_Afock, d_blas_Z, d_blas_gA, d_blas_gB, d_blas_ch, d_blas_cg      \
            );                                                                                    \
        } while (0)

        for (int mol_off = 0; mol_off < n_mol_total; mol_off += mol_batch_cap) {
            const int B_mol = std::min(mol_batch_cap, n_mol_total - mol_off);
            const int n_pack = B_mol * pack_slots_per_mol;
            const int pack_off =
                (n_ij_fuse > 0 && !bulk_mol) ? mol_off * n_ij_fuse : mol_off;

            if (n_ij_fuse > 0) {
                CUDA_CHECK(cudaMemset(
                    d_blas_Dri, 0, static_cast<size_t>(B_mol) * max_na3 * k_rad * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_Dksi, 0, static_cast<size_t>(B_mol) * max_na3 * k_rad * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_X, 0, static_cast<size_t>(B_mol) * k_rad * max_nb3 * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_Y, 0, static_cast<size_t>(B_mol) * k_rad * max_nb3 * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_Afock, 0, static_cast<size_t>(B_mol) * max_na3 * k_fock * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_Z, 0, static_cast<size_t>(B_mol) * k_fock * max_nb3 * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_gA, 0, static_cast<size_t>(B_mol) * max_na3 * n_ij_fuse * sizeof(float)
                ));
                CUDA_CHECK(cudaMemset(
                    d_blas_gB, 0, static_cast<size_t>(B_mol) * max_nb3 * n_ij_fuse * sizeof(float)
                ));
            }

            CUDA_CHECK(cudaMemcpy(
                d_blas_pairs, h_pack_pairs.data() + pack_off,
                static_cast<size_t>(n_pack) * sizeof(hess_blas::Pair), cudaMemcpyHostToDevice
            ));
            if (use_alchemy) {
                if (order_eff == 1) {
                    KF_LAUNCH_BLAS_PACK(true, 1);
                } else {
                    KF_LAUNCH_BLAS_PACK(true, 0);
                }
            } else if (order_eff == 1) {
                KF_LAUNCH_BLAS_PACK(false, 1);
            } else {
                KF_LAUNCH_BLAS_PACK(false, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            // Scatter needs mol-pair list when fused.
            if (n_ij_fuse > 0) {
                CUDA_CHECK(cudaMemcpy(
                    d_blas_pairs, h_mol_pairs.data() + mol_off,
                    static_cast<size_t>(B_mol) * sizeof(hess_blas::Pair), cudaMemcpyHostToDevice
                ));
            }
            hess_blas::apply_batch(
                B_mol, max_na3, max_nb3, P, K1, n_ij_fuse, d_blas_Dri, d_blas_Dksi, d_blas_X,
                d_blas_Y, d_blas_Afock, d_blas_Z, d_blas_Htile, d_blas_gA, d_blas_gB, d_blas_ch,
                d_blas_cg, d_blas_pairs, d_N1, d_N2, side1.atom_base, sb2.atom_base, d_H_out, D_B
            );
        }
#undef KF_LAUNCH_BLAS_PACK
    } else {
        auto count_hot_blocks = [&](int jc) -> long long {
            long long blocks = 0;
            for (int a = 0; a < nm1; ++a) {
                for (int b = 0; b < (same ? a + 1 : nm2); ++b) {
                    blocks += static_cast<long long>(h_n1[static_cast<size_t>(a)]) *
                              ((h_n2[static_cast<size_t>(b)] + jc - 1) / jc);
                }
            }
            return blocks;
        };

        int jchunk = 1;
        {
            int dev = 0, n_sm = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            CUDA_CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev));
            // When Hij fits in shared memory, amortize A-side centre loads over more j
            // per block (tile is flushed per j after fused s_ij).  Keep enough blocks
            // for occupancy; fall back to smaller chunks when the grid would be thin.
            if (h_cap > 0) {
                for (const int jc : {8, 4, 2}) {
                    if (count_hot_blocks(jc) >= 64LL * n_sm) {
                        jchunk = jc;
                        break;
                    }
                }
            } else {
                for (const int jc : {4, 2}) {
                    if (count_hot_blocks(jc) >= 128LL * n_sm) {
                        jchunk = jc;
                        break;
                    }
                }
            }
        }
        std::vector<int> h_pair_offset(static_cast<size_t>(n_pairs) + 1, 0);
        {
            long long acc = 0;
            for (int a = 0; a < nm1; ++a) {
                for (int b = 0; b < nm2; ++b) {
                    h_pair_offset[static_cast<size_t>(a) * nm2 + b] = static_cast<int>(acc);
                    if (same && b > a) {
                        continue;
                    }
                    acc += static_cast<long long>(h_n1[static_cast<size_t>(a)]) *
                           ((h_n2[static_cast<size_t>(b)] + jchunk - 1) / jchunk);
                    if (acc > 0x7fffffffLL) {
                        throw std::invalid_argument(
                            "cuda_fchl18_direct_hess: too many centre pairs for one launch"
                        );
                    }
                }
            }
            h_pair_offset[static_cast<size_t>(n_pairs)] = static_cast<int>(acc);
        }
        const int n_hot_blocks = h_pair_offset[static_cast<size_t>(n_pairs)];
        if (n_hot_blocks == 0) {
            return;
        }
        ensure_capacity(&d_pair_offset, &d_pair_offset_cap, static_cast<size_t>(n_pairs) + 1);
        CUDA_CHECK(cudaMemcpy(
            d_pair_offset, h_pair_offset.data(), h_pair_offset.size() * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        float *Hscratch = nullptr;
        if (h_cap == 0) {
            const size_t tile = static_cast<size_t>(max_na3) * static_cast<size_t>(max_nb3);
            const size_t need = tile * static_cast<size_t>(n_hot_blocks);
            size_t free_b = 0, total_b = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            if (need * sizeof(float) > free_b) {
                throw std::runtime_error(
                    "cuda_fchl18_direct_hess: private H tiles (" +
                    std::to_string((need * sizeof(float)) >> 20) +
                    " MiB) do not fit in free GPU memory (" + std::to_string(free_b >> 20) +
                    " MiB)"
                );
            }
            ensure_capacity(&d_Hscratch, &d_Hscratch_cap, need);
            Hscratch = d_Hscratch;
        }

        const int hess_block = 128;
#define KF_LAUNCH_HESS(AL, ORD)                                                                   \
        do {                                                                                      \
            if (smem_bytes > 48 * 1024) {                                                         \
                CUDA_CHECK(cudaFuncSetAttribute(                                                  \
                    hess_hot_kernel<AL, ORD>, cudaFuncAttributeMaxDynamicSharedMemorySize,        \
                    static_cast<int>(smem_bytes)                                                  \
                ));                                                                               \
            }                                                                                     \
            hess_hot_kernel<AL, ORD><<<n_hot_blocks, hess_block, smem_bytes>>>(                   \
                d_N1, d_N2, side1.nn, sb2.nn, side1.mol_byte_off, sb2.mol_byte_off,               \
                side1.mol_gbyte_off, sb2.mol_gbyte_off, side1.dss_off, sb2.dss_off,               \
                side1.atom_base, sb2.atom_base, side1.blob, sb2.blob, be1, be2, side1.gblob,      \
                sb2.gblob, side1.ss, sb2.ss, side1.dss, sb2.dss, d_H_out, D_B, nm2, max_atoms1,   \
                max_atoms2, inv, hp, d_pdc, d_pair_offset, n_pairs, n_hot_blocks, max_na3,        \
                max_nb3, h_cap, jchunk, Hscratch                                                  \
            );                                                                                    \
        } while (0)

        if (use_alchemy) {
            if (order_eff == 1) {
                KF_LAUNCH_HESS(true, 1);
            } else {
                KF_LAUNCH_HESS(true, 0);
            }
        } else if (order_eff == 1) {
            KF_LAUNCH_HESS(false, 1);
        } else {
            KF_LAUNCH_HESS(false, 0);
        }
#undef KF_LAUNCH_HESS
        CUDA_CHECK(cudaGetLastError());
    }

    if (same && nm1 > 1) {
        const long long n_lower = static_cast<long long>(nm1) * (nm1 - 1) / 2;
        mirror_lower_kernel<<<static_cast<unsigned int>(n_lower), 256>>>(
            d_N1, side1.atom_base, d_H_out, D_B
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace direct
}  // namespace fchl18
}  // namespace kf
