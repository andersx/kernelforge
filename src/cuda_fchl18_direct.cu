// Coords-only FCHL18 rectangular Gaussian kernel (FP32, no cutoff).
// Layout: CSR-style mol blocks — each mol a stores dense N[a]×N[a] neighbour
// tables (not padded to max_atoms×max_atoms).

#include "cuda_fchl18_direct.hpp"

#include "cuda_fchl18_common.cuh"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace kf {
namespace fchl18 {
namespace direct {
namespace {

struct Params {
    float two_body_power;
    float three_body_power;
    int fourier_order;
    int pmax;
    int use_atm;
    int z_to_idx[256];
};

struct HotParams {
    float inv_sigma2;
    float two_body_width;
    float true_distance_scale;
    float true_angular_scale;
    float s0;
    float s1;
    int fourier_order;
    int pmax;
    int emax = 0;  // alchemy: pd lives in c_alchemy_pd
};

// Full Z-indexed pd in constant mem (118²×4 ≈ 55 KiB < 64 KiB limit).
__constant__ float c_alchemy_pd[kAlchemyEmaxDefault * kAlchemyEmaxDefault];

// One neighbour slot: rij + ksi + Z packed for fewer loads in the scalar.
struct alignas(16) Neigh {
    float rij;
    float ksi;
    int z;
    int pad;
};

// Fourier (cos,sin) interleaved — one float2 load per (p,m) in angular dots.
using Fock = float2;

__host__ __device__ inline size_t centre_stride_bytes(int na, int pmax, int order) {
    const size_t raw = static_cast<size_t>(na) * sizeof(Neigh) +
                       static_cast<size_t>(na) * static_cast<size_t>(pmax) * order * sizeof(Fock);
    return (raw + 15u) & ~size_t(15);  // keep Neigh alignas(16)
}

__device__ inline void centre_views_base(
    const char *base, int i, int na, int pmax, int order, const Neigh **neigh, const Fock **fock
) {
    const char *p = base + static_cast<long long>(i) * centre_stride_bytes(na, pmax, order);
    *neigh = reinterpret_cast<const Neigh *>(p);
    *fock = reinterpret_cast<const Fock *>(*neigh + na);
}

__device__ inline void centre_views_base(
    char *base, int i, int na, int pmax, int order, Neigh **neigh, Fock **fock
) {
    char *p = base + static_cast<long long>(i) * centre_stride_bytes(na, pmax, order);
    *neigh = reinterpret_cast<Neigh *>(p);
    *fock = reinterpret_cast<Fock *>(*neigh + na);
}

__device__ inline void centre_views(
    const char *blob, const long long *mol_byte_off, int a, int i, int na, int pmax, int order,
    const Neigh **neigh, const Fock **fock
) {
    centre_views_base(blob + mol_byte_off[a], i, na, pmax, order, neigh, fock);
}

__device__ inline void centre_views(
    char *blob, const long long *mol_byte_off, int a, int i, int na, int pmax, int order,
    Neigh **neigh, Fock **fock
) {
    centre_views_base(blob + mol_byte_off[a], i, na, pmax, order, neigh, fock);
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

// n_slots = na (dense); Fourier stride = pmax*order*n_slots.
template <bool UseAtm>
__device__ void precompute_atom(
    const float *coords_mol,
    const int *Z_mol,
    int i,
    int na,
    float two_body_power,
    float three_body_power,
    int order,
    int pmax,
    const int *z_to_idx,
    int *nn_out,
    Neigh *neigh,
    Fock *fock
) {
    const int n_slots = na;
    for (int k = 0; k < n_slots; ++k) {
        neigh[k] = Neigh{0.0f, 0.0f, 0, 0};
    }
    const long long fsz = static_cast<long long>(pmax) * order * n_slots;
    for (long long t = 0; t < fsz; ++t) {
        fock[t] = make_float2(0.0f, 0.0f);
    }

    const float *ra = coords_mol + i * 3;
    const int zi = Z_mol[i];
    neigh[0] = Neigh{0.0f, 0.0f, zi, 0};

    float dxs[64];
    float dys[64];
    float dzs[64];
    const int use_local = (na <= 64);

    int slot = 1;
    for (int j = 0; j < na; ++j) {
        if (j == i) {
            continue;
        }
        const float *rj = coords_mol + j * 3;
        const float dx = rj[0] - ra[0];
        const float dy = rj[1] - ra[1];
        const float dz = rj[2] - ra[2];
        const float r = sqrtf(dx * dx + dy * dy + dz * dz);
        float k = 0.0f;
        if (r >= Math<float>::eps()) {
            k = 1.0f / fast_pow(r, two_body_power);
        }
        neigh[slot] = Neigh{r, k, Z_mol[j], 0};
        if (use_local) {
            dxs[slot] = dx;
            dys[slot] = dy;
            dzs[slot] = dz;
        }
        ++slot;
    }
    const int n_neigh = slot;
    *nn_out = n_neigh;

    for (int k = 2; k < n_neigh; ++k) {
        const Neigh saved = neigh[k];
        float saved_dx = 0, saved_dy = 0, saved_dz = 0;
        if (use_local) {
            saved_dx = dxs[k];
            saved_dy = dys[k];
            saved_dz = dzs[k];
        }
        int insert_at = k;
        while (insert_at > 1 && neigh[insert_at - 1].z > saved.z) {
            neigh[insert_at] = neigh[insert_at - 1];
            if (use_local) {
                dxs[insert_at] = dxs[insert_at - 1];
                dys[insert_at] = dys[insert_at - 1];
                dzs[insert_at] = dzs[insert_at - 1];
            }
            --insert_at;
        }
        neigh[insert_at] = saved;
        if (use_local) {
            dxs[insert_at] = saved_dx;
            dys[insert_at] = saved_dy;
            dzs[insert_at] = saved_dz;
        }
    }

    for (int j = 1; j < n_neigh; ++j) {
        if (!use_local) {
            continue;
        }
        const float dxj = dxs[j];
        const float dyj = dys[j];
        const float dzj = dzs[j];
        const float dj = neigh[j].rij;
        if (dj < Math<float>::eps()) {
            continue;
        }
        const float inv_dj = 1.0f / dj;
        const float uxj = dxj * inv_dj;
        const float uyj = dyj * inv_dj;
        const float uzj = dzj * inv_dj;

        for (int k = j + 1; k < n_neigh; ++k) {
            const float dxk = dxs[k];
            const float dyk = dys[k];
            const float dzk = dzs[k];
            const float dk = neigh[k].rij;
            if (dk < Math<float>::eps()) {
                continue;
            }
            const float dxjk = dxj - dxk;
            const float dyjk = dyj - dyk;
            const float dzjk = dzj - dzk;
            const float di2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
            const float di = sqrtf(di2);
            if (di < Math<float>::eps()) {
                continue;
            }

            const float inv_dk = 1.0f / dk;
            const float uxk = dxk * inv_dk;
            const float uyk = dyk * inv_dk;
            const float uzk = dzk * inv_dk;
            const float cos_i = clamp11(uxj * uxk + uyj * uyk + uzj * uzk);

            float atm = 1.0f;
            if constexpr (UseAtm) {
                const float inv_di = 1.0f / di;
                const float ukj_x = -dxjk * inv_di;
                const float ukj_y = -dyjk * inv_di;
                const float ukj_z = -dzjk * inv_di;
                const float cos_j = clamp11(ukj_x * (-uxj) + ukj_y * (-uyj) + ukj_z * (-uzj));
                const float ujk_x = dxjk * inv_di;
                const float ujk_y = dyjk * inv_di;
                const float ujk_z = dzjk * inv_di;
                const float cos_k = clamp11(ujk_x * (-uxk) + ujk_y * (-uyk) + ujk_z * (-uzk));
                atm = 1.0f + 3.0f * cos_i * cos_j * cos_k;
            }
            if (atm == 0.0f) {
                continue;
            }

            const int zj = neigh[j].z;
            const int zk = neigh[k].z;
            if (zk <= 0 || zk >= 256 || zj <= 0 || zj >= 256) {
                continue;
            }
            const float r0_ij = pair_bond_r0(zi, zj);
            const float r0_ik = pair_bond_r0(zi, zk);
            const float r0_jk = pair_bond_r0(zj, zk);
            const auto radial = three_body_radial_weight(
                kTbWeightProduct, dj, dk, di, r0_ij, r0_ik, r0_jk, three_body_power
            );
            const float ksi3 = atm * radial.w;
            if (ksi3 == 0.0f) {
                continue;
            }

            const float theta = acosf(cos_i);
            const int pj = z_to_idx[zk];
            const int pk = z_to_idx[zj];
            if (pj < 0 || pk < 0) {
                continue;
            }

            for (int m = 0; m < order; ++m) {
                const float mf = static_cast<float>(m + 1);
                float sin_mth, cos_mth, sin_mthpi, cos_mthpi;
                sincosf(mf * theta, &sin_mth, &cos_mth);
                sincosf(mf * (theta + Math<float>::pi()), &sin_mthpi, &cos_mthpi);
                const float cos_m = (cos_mth - cos_mthpi) * ksi3;
                const float sin_m = (sin_mth - sin_mthpi) * ksi3;
                const long long idx_j =
                    static_cast<long long>(j) * pmax * order + static_cast<long long>(pj) * order + m;
                const long long idx_k =
                    static_cast<long long>(k) * pmax * order + static_cast<long long>(pk) * order + m;
                fock[idx_j].x += cos_m;
                fock[idx_j].y += sin_m;
                fock[idx_k].x += cos_m;
                fock[idx_k].y += sin_m;
            }
        }
    }
}

template <bool UseAtm>
__global__ void precompute_kernel(
    const float *coords,
    const int *Z,
    const int *n,
    const long long *mol_byte_off,
    int *nn,
    char *blob,
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
    centre_views(
        blob, mol_byte_off, a, i, na, params.pmax, params.fourier_order, &neigh, &fock
    );
    const float *coords_mol = coords + static_cast<long long>(a) * max_atoms * 3;
    const int *Z_mol = Z + static_cast<long long>(a) * max_atoms;

    precompute_atom<UseAtm>(
        coords_mol,
        Z_mol,
        i,
        na,
        params.two_body_power,
        params.three_body_power,
        params.fourier_order,
        params.pmax,
        params.z_to_idx,
        nn + flat,
        neigh,
        fock
    );
}

template <int PMax>
__device__ float angular_dot_order1_t(const Fock *f1, const Fock *f2, float s0) {
    float ang0 = 0.0f;
#pragma unroll
    for (int p = 0; p < PMax; ++p) {
        ang0 += f1[p].x * f2[p].x + f1[p].y * f2[p].y;
    }
    return ang0 * s0;
}

__device__ float angular_dot_order1(const Fock *f1, const Fock *f2, int pmax, float s0) {
    float ang0 = 0.0f;
    for (int p = 0; p < pmax; ++p) {
        ang0 += f1[p].x * f2[p].x + f1[p].y * f2[p].y;
    }
    return ang0 * s0;
}

__device__ float angular_dot_order2(const Fock *f1, const Fock *f2, int pmax, float s0, float s1) {
    float ang0 = 0.0f;
    float ang1 = 0.0f;
    for (int p = 0; p < pmax; ++p) {
        const int o = p * 2;
        ang0 += f1[o].x * f2[o].x + f1[o].y * f2[o].y;
        ang1 += f1[o + 1].x * f2[o + 1].x + f1[o + 1].y * f2[o + 1].y;
    }
    return ang0 * s0 + ang1 * s1;
}

__device__ float angular_dot_generic(
    const Fock *f1, const Fock *f2, int i, int j, int order, int pmax, const float *s_prefactor
) {
    const long long base1 = static_cast<long long>(i) * pmax * order;
    const long long base2 = static_cast<long long>(j) * pmax * order;
    float angular = 0.0f;
    for (int m = 0; m < order; ++m) {
        float ang_m = 0.0f;
        for (int p = 0; p < pmax; ++p) {
            const long long o1 = base1 + static_cast<long long>(p) * order + m;
            const long long o2 = base2 + static_cast<long long>(p) * order + m;
            ang_m += f1[o1].x * f2[o2].x + f1[o1].y * f2[o2].y;
        }
        angular += ang_m * s_prefactor[m];
    }
    return angular;
}

template <int FourierOrder, int PMax>
__device__ float scalar_noalchemy(
    const Neigh *n1,
    int nn1,
    const Fock *f1,
    const Neigh *n2,
    int nn2,
    const Fock *f2,
    const HotParams &p,
    const float *s_prefactor
) {
    if (n1[0].z != n2[0].z) {
        return 0.0f;
    }

    const float inv_width = -1.0f / (4.0f * p.two_body_width * p.two_body_width);
    const float maxgausdist2 = (8.0f * p.two_body_width) * (8.0f * p.two_body_width);
    constexpr int order_c = FourierOrder;
    const int order = (order_c > 0) ? order_c : p.fourier_order;
    const int pmax = (PMax > 0) ? PMax : p.pmax;
    const int fblock = pmax * order;

    float aadist = 1.0f;
    int i = 1;
    int j = 1;
    while (i < nn1 && j < nn2) {
        const int zi = n1[i].z;
        const int zj = n2[j].z;
        if (zi < zj) {
            ++i;
            continue;
        }
        if (zj < zi) {
            ++j;
            continue;
        }
        int i_end = i + 1;
        while (i_end < nn1 && n1[i_end].z == zi) {
            ++i_end;
        }
        int j_end = j + 1;
        while (j_end < nn2 && n2[j_end].z == zi) {
            ++j_end;
        }
        for (int ii = i; ii < i_end; ++ii) {
            const float ri = n1[ii].rij;
            const float ksi_i = n1[ii].ksi;
            const Fock *f1_i = f1 + static_cast<long long>(ii) * fblock;
            for (int jj = j; jj < j_end; ++jj) {
                const float rj = n2[jj].rij;
                const float r2 = (rj - ri) * (rj - ri);
                if (r2 >= maxgausdist2) {
                    continue;
                }
                const float d = expf(r2 * inv_width);
                float angular;
                if constexpr (order_c == 1 && PMax > 0) {
                    angular = angular_dot_order1_t<PMax>(
                        f1_i, f2 + static_cast<long long>(jj) * fblock, p.s0
                    );
                } else if constexpr (order_c == 1) {
                    angular = angular_dot_order1(
                        f1_i, f2 + static_cast<long long>(jj) * fblock, pmax, p.s0
                    );
                } else if constexpr (order_c == 2) {
                    angular = angular_dot_order2(
                        f1_i, f2 + static_cast<long long>(jj) * fblock, pmax, p.s0, p.s1
                    );
                } else {
                    angular = angular_dot_generic(f1, f2, ii, jj, order, pmax, s_prefactor);
                }
                aadist +=
                    d * (ksi_i * n2[jj].ksi * p.true_distance_scale + angular * p.true_angular_scale);
            }
        }
        i = i_end;
        j = j_end;
    }
    return aadist;
}

// Alchemy scalar: full neighbour double-loop with pd weights; f2 is element-mixed Fock.
template <int FourierOrder, int PMax>
__device__ float scalar_alchemy(
    const Neigh *n1,
    int nn1,
    const Fock *f1,
    const Neigh *n2,
    int nn2,
    const Fock *f2_eff,
    const HotParams &p,
    const float *s_prefactor
) {
    const float pd_central = alch_pd(c_alchemy_pd, p.emax, n1[0].z, n2[0].z);
    if (pd_central == 0.0f) {
        return 0.0f;
    }

    const float inv_width = -1.0f / (4.0f * p.two_body_width * p.two_body_width);
    const float maxgausdist2 = (8.0f * p.two_body_width) * (8.0f * p.two_body_width);
    constexpr int order_c = FourierOrder;
    const int order = (order_c > 0) ? order_c : p.fourier_order;
    const int pmax = (PMax > 0) ? PMax : p.pmax;
    const int fblock = pmax * order;

    float aadist = 1.0f;
    for (int ii = 1; ii < nn1; ++ii) {
        const int zi = n1[ii].z;
        const float ri = n1[ii].rij;
        const float ksi_i = n1[ii].ksi;
        const Fock *f1_i = f1 + static_cast<long long>(ii) * fblock;
        for (int jj = 1; jj < nn2; ++jj) {
            const float rj = n2[jj].rij;
            const float r2 = (rj - ri) * (rj - ri);
            if (r2 >= maxgausdist2) {
                continue;
            }
            const float pd_ij = alch_pd(c_alchemy_pd, p.emax, zi, n2[jj].z);
            if (pd_ij == 0.0f) {
                continue;
            }
            const float d = expf(r2 * inv_width) * pd_ij;
            float angular;
            if constexpr (order_c == 1 && PMax > 0) {
                angular = angular_dot_order1_t<PMax>(
                    f1_i, f2_eff + static_cast<long long>(jj) * fblock, p.s0
                );
            } else if constexpr (order_c == 1) {
                angular = angular_dot_order1(
                    f1_i, f2_eff + static_cast<long long>(jj) * fblock, pmax, p.s0
                );
            } else if constexpr (order_c == 2) {
                angular = angular_dot_order2(
                    f1_i, f2_eff + static_cast<long long>(jj) * fblock, pmax, p.s0, p.s1
                );
            } else {
                angular = angular_dot_generic(f1, f2_eff, ii, jj, order, pmax, s_prefactor);
            }
            aadist +=
                d * (ksi_i * n2[jj].ksi * p.true_distance_scale + angular * p.true_angular_scale);
        }
    }
    return aadist * pd_central;
}

// Mix element channels of Fock float2 slabs: F_eff[p1] = Σ_p2 PDc[p1,p2] F[p2].
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
    const Neigh *ni = nullptr;
    const Fock *fi = nullptr;
    centre_views(blob_in, mol_byte_off, a, i, na, pmax, order, &ni, &fi);
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

template <int FourierOrder, int PMax, bool Alch>
__global__ void compute_self_scalars_kernel(
    const int *n,
    const int *nn,
    const long long *mol_byte_off,
    char *blob,
    char *blob_e,
    float *ss,
    int nm,
    int max_atoms,
    HotParams hot,
    const float *__restrict__ s_prefactor
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
    const int order = (FourierOrder > 0) ? FourierOrder : hot.fourier_order;
    const int na = n[a];
    const int n_neigh = nn[flat];
    Neigh *ni = nullptr;
    Fock *fi = nullptr;
    centre_views(blob, mol_byte_off, a, i, na, hot.pmax, order, &ni, &fi);
    if constexpr (Alch) {
        const Fock *fi_e = centre_fock_ptr(blob_e, mol_byte_off, a, i, na, hot.pmax, order);
        ss[flat] = scalar_alchemy<FourierOrder, PMax>(
            ni, n_neigh, fi, ni, n_neigh, fi_e, hot, s_prefactor
        );
    } else {
        ss[flat] =
            scalar_noalchemy<FourierOrder, PMax>(ni, n_neigh, fi, ni, n_neigh, fi, hot, s_prefactor);
    }
}

template <int FourierOrder, int PMax, bool Alch>
__global__ void __launch_bounds__(128, 8) rect_kernel(
    const int *n1,
    const int *n2,
    const int *nn1,
    const int *nn2,
    const long long *mol_byte_off1,
    const long long *mol_byte_off2,
    char *blob1,
    char *blob2,
    char *blob1_e,
    char *blob2_e,
    const float *ss1,
    const float *ss2,
    float *K,
    int nm1,
    int nm2,
    int max1,
    int max2,
    HotParams hot,
    const float *__restrict__ s_prefactor
) {
    const int pair = static_cast<int>(blockIdx.x);
    if (pair >= nm1 * nm2) {
        return;
    }
    const int a = pair / nm2;
    const int b = pair - a * nm2;
    const int na = n1[a];
    const int nb = n2[b];
    if (na <= 0 || nb <= 0) {
        if (threadIdx.x == 0) {
            K[static_cast<long long>(a) * nm2 + b] = 0.0f;
        }
        return;
    }

    const int order = (FourierOrder > 0) ? FourierOrder : hot.fourier_order;
    const int n_atom_pairs = na * nb;

    float kab = 0.0f;
    for (int p = static_cast<int>(threadIdx.x); p < n_atom_pairs; p += static_cast<int>(blockDim.x)) {
        const int i = p / nb;
        const int j = p - i * nb;
        const int flat_i = a * max1 + i;
        const int flat_j = b * max2 + j;
        Neigh *ni = nullptr;
        Fock *fi = nullptr;
        Neigh *nj = nullptr;
        Fock *fj = nullptr;
        centre_views(blob1, mol_byte_off1, a, i, na, hot.pmax, order, &ni, &fi);
        centre_views(blob2, mol_byte_off2, b, j, nb, hot.pmax, order, &nj, &fj);
        float s12;
        if constexpr (Alch) {
            // Reject before Fock loads when central pd is zero.
            if (alch_pd(c_alchemy_pd, hot.emax, ni[0].z, nj[0].z) == 0.0f) {
                s12 = 0.0f;
            } else {
                const Fock *fj_e =
                    centre_fock_ptr(blob2_e, mol_byte_off2, b, j, nb, hot.pmax, order);
                s12 = scalar_alchemy<FourierOrder, PMax>(
                    ni, nn1[flat_i], fi, nj, nn2[flat_j], fj_e, hot, s_prefactor
                );
            }
        } else {
            s12 = scalar_noalchemy<FourierOrder, PMax>(
                ni, nn1[flat_i], fi, nj, nn2[flat_j], fj, hot, s_prefactor
            );
        }
        kab += expf((ss1[flat_i] + ss2[flat_j] - 2.0f * s12) * hot.inv_sigma2);
    }
    kab = block_reduce_sum(kab);
    if (threadIdx.x == 0) {
        K[static_cast<long long>(a) * nm2 + b] = kab;
    }
}

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
    char *blob = nullptr;    // per-centre Neigh[na]|Fock[frow] slabs (raw)
    char *blob_e = nullptr;  // alchemy-mixed Fock twin (same layout)
    float *ss = nullptr;
    long long *mol_byte_off = nullptr;  // nm+1
    size_t nn_cap = 0, blob_cap = 0, blob_e_cap = 0, ss_cap = 0, mol_byte_off_cap = 0;
    SideKey key{};
};

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

__host__ inline void build_mol_byte_offsets(
    const int *h_N, int nm, int pmax, int fourier_order, std::vector<long long> *mol_byte_off,
    size_t *nbytes
) {
    mol_byte_off->assign(static_cast<size_t>(nm) + 1, 0);
    for (int a = 0; a < nm; ++a) {
        const int na = h_N[a];
        const size_t mol_bytes =
            static_cast<size_t>(na) * centre_stride_bytes(na, pmax, fourier_order);
        (*mol_byte_off)[static_cast<size_t>(a) + 1] =
            (*mol_byte_off)[static_cast<size_t>(a)] + static_cast<long long>(mol_bytes);
    }
    *nbytes = static_cast<size_t>((*mol_byte_off)[static_cast<size_t>(nm)]);
}

}  // namespace

void kernel_gaussian_rect_fp32(
    const float *d_coords1,
    const int *d_Z1,
    const int *d_N1,
    const float *d_coords2,
    const int *d_Z2,
    const int *d_N2,
    float *d_K_out,
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
    if (fourier_order < 1 || fourier_order > kMaxFourierOrder) {
        throw std::invalid_argument("fourier_order out of range");
    }
    if (max_atoms1 > 64 || max_atoms2 > 64) {
        throw std::invalid_argument("cuda_fchl18_direct: max_atoms > 64 not supported");
    }

    static SideBuf side1, side2;
    static int *d_z_present = nullptr;
    static float *d_s_prefactor = nullptr;
    if (!d_z_present) {
        CUDA_CHECK(cudaMalloc(&d_z_present, 256 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_s_prefactor, kMaxFourierOrder * sizeof(float)));
    }

    const bool same = (d_coords1 == d_coords2 && d_Z1 == d_Z2 && d_N1 == d_N2 &&
                       nm1 == nm2 && max_atoms1 == max_atoms2);

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

    Params params{};
    params.two_body_power = two_body_power;
    params.three_body_power = three_body_power;
    params.fourier_order = fourier_order;
    params.use_atm = use_atm ? 1 : 0;

    HotParams hot{};
    hot.inv_sigma2 = -0.5f / (sigma * sigma);
    hot.two_body_width = two_body_width;
    hot.true_distance_scale = two_body_scaling / 16.0f;
    hot.true_angular_scale = three_body_scaling / static_cast<float>(std::sqrt(8.0));
    hot.fourier_order = fourier_order;

    const size_t n_flat1 = static_cast<size_t>(nm1) * max_atoms1;
    ensure_capacity(&side1.nn, &side1.nn_cap, n_flat1);
    ensure_capacity(&side1.ss, &side1.ss_cap, n_flat1);
    ensure_capacity(&side1.mol_byte_off, &side1.mol_byte_off_cap, static_cast<size_t>(nm1) + 1);

    const int block = 256;
    int grid1 = static_cast<int>((n_flat1 + block - 1) / block);

    int *nn2 = side1.nn;
    char *blob2 = side1.blob;
    char *blob2_e = side1.blob_e;
    float *ss2 = side1.ss;
    long long *mol_byte_off2 = side1.mol_byte_off;
    int max2 = max_atoms1;

    if (!same) {
        const size_t n_flat2 = static_cast<size_t>(nm2) * max_atoms2;
        ensure_capacity(&side2.nn, &side2.nn_cap, n_flat2);
        ensure_capacity(&side2.ss, &side2.ss_cap, n_flat2);
        ensure_capacity(&side2.mol_byte_off, &side2.mol_byte_off_cap, static_cast<size_t>(nm2) + 1);
        nn2 = side2.nn;
        ss2 = side2.ss;
        mol_byte_off2 = side2.mol_byte_off;
        max2 = max_atoms2;
    }

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
            const size_t n_flat2 = static_cast<size_t>(nm2) * max_atoms2;
            int grid2 = static_cast<int>((n_flat2 + block - 1) / block);
            mark_z_kernel<<<grid2, block>>>(d_Z2, d_N2, nm2, max_atoms2, d_z_present);
            CUDA_CHECK(cudaGetLastError());
        }
        int h_z_present[256];
        CUDA_CHECK(cudaMemcpy(h_z_present, d_z_present, 256 * sizeof(int), cudaMemcpyDeviceToHost));
        params.pmax = build_element_map_from_flags(h_z_present, params.z_to_idx);
    }

    if (params.pmax == 0) {
        CUDA_CHECK(cudaMemset(d_K_out, 0, static_cast<size_t>(nm1) * nm2 * sizeof(float)));
        return;
    }
    if (params.pmax > kMaxElements) {
        throw std::invalid_argument("too many distinct elements");
    }

    float h_s_prefactor[kMaxFourierOrder]{};
    const double ang_norm2 = get_angular_norm2(static_cast<double>(three_body_width));
    const double g1 = std::sqrt(2.0 * 3.14159265358979323846) / ang_norm2;
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        const double tw = static_cast<double>(three_body_width);
        h_s_prefactor[m] = static_cast<float>(g1 * std::exp(-(tw * mf) * (tw * mf) / 2.0));
    }
    hot.s0 = h_s_prefactor[0];
    hot.s1 = (fourier_order > 1) ? h_s_prefactor[1] : 0.0f;
    hot.pmax = params.pmax;
    if (use_alchemy) {
        const float *d_pd = alchemy_upload_pd<float>(
            static_cast<double>(alchemy_period_width), static_cast<double>(alchemy_group_width)
        );
        hot.emax = kAlchemyEmaxDefault;
        CUDA_CHECK(cudaMemcpyToSymbol(
            c_alchemy_pd, d_pd,
            static_cast<size_t>(kAlchemyEmaxDefault) * kAlchemyEmaxDefault * sizeof(float), 0,
            cudaMemcpyDeviceToDevice
        ));
    } else {
        hot.emax = 0;
    }
    CUDA_CHECK(cudaMemcpy(
        d_s_prefactor, h_s_prefactor, kMaxFourierOrder * sizeof(float), cudaMemcpyHostToDevice
    ));

    const float *d_pdc = nullptr;
    if (use_alchemy) {
        d_pdc = alchemy_upload_pdc<float>(params.z_to_idx, params.pmax);
    }

    auto fill_side_offsets = [&](SideBuf &side, const int *d_N, int nm, bool hit) {
        if (hit) {
            return;
        }
        std::vector<int> h_N(static_cast<size_t>(nm));
        CUDA_CHECK(cudaMemcpy(h_N.data(), d_N, static_cast<size_t>(nm) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        std::vector<long long> h_boff;
        size_t nbytes = 0;
        build_mol_byte_offsets(h_N.data(), nm, params.pmax, fourier_order, &h_boff, &nbytes);
        ensure_capacity(&side.blob, &side.blob_cap, nbytes);
        if (use_alchemy) {
            ensure_capacity(&side.blob_e, &side.blob_e_cap, nbytes);
        }
        CUDA_CHECK(cudaMemcpy(
            side.mol_byte_off, h_boff.data(), (static_cast<size_t>(nm) + 1) * sizeof(long long),
            cudaMemcpyHostToDevice
        ));
    };

    fill_side_offsets(side1, d_N1, nm1, hit1);
    if (!same) {
        fill_side_offsets(side2, d_N2, nm2, hit2);
        blob2 = side2.blob;
        blob2_e = side2.blob_e;
    } else {
        blob2 = side1.blob;
        blob2_e = side1.blob_e;
    }

    auto launch_precompute = [&](const float *coords, const int *Z, const int *N, SideBuf &side,
                                 int nm, int max_atoms, int grid) {
        if (use_atm) {
            precompute_kernel<true><<<grid, block>>>(
                coords, Z, N, side.mol_byte_off, side.nn, side.blob, nm, max_atoms, params
            );
        } else {
            precompute_kernel<false><<<grid, block>>>(
                coords, Z, N, side.mol_byte_off, side.nn, side.blob, nm, max_atoms, params
            );
        }
    };

    auto launch_mix = [&](const int *N, SideBuf &side, int nm, int max_atoms, int grid) {
        alchemy_mix_fock_blob_kernel<<<grid, block>>>(
            N, side.nn, side.mol_byte_off, side.blob, side.blob_e, d_pdc, nm, max_atoms,
            params.pmax, fourier_order
        );
    };

    auto launch_self = [&](const int *N, SideBuf &side, float *ss, int nm, int max_atoms, int grid) {
        const int pm = params.pmax;
        char *be = use_alchemy ? side.blob_e : nullptr;
#define KF_LAUNCH_SELF(FO, PM, AL)                                                                 \
    compute_self_scalars_kernel<FO, PM, AL><<<grid, block>>>(                                      \
        N, side.nn, side.mol_byte_off, side.blob, be, ss, nm, max_atoms, hot, d_s_prefactor          \
    )
        if (use_alchemy) {
            if (fourier_order == 1 && pm == 5) {
                KF_LAUNCH_SELF(1, 5, true);
            } else if (fourier_order == 1 && pm == 4) {
                KF_LAUNCH_SELF(1, 4, true);
            } else if (fourier_order == 1) {
                KF_LAUNCH_SELF(1, 0, true);
            } else if (fourier_order == 2) {
                KF_LAUNCH_SELF(2, 0, true);
            } else {
                KF_LAUNCH_SELF(0, 0, true);
            }
        } else if (fourier_order == 1 && pm == 5) {
            KF_LAUNCH_SELF(1, 5, false);
        } else if (fourier_order == 1 && pm == 4) {
            KF_LAUNCH_SELF(1, 4, false);
        } else if (fourier_order == 1) {
            KF_LAUNCH_SELF(1, 0, false);
        } else if (fourier_order == 2) {
            KF_LAUNCH_SELF(2, 0, false);
        } else {
            KF_LAUNCH_SELF(0, 0, false);
        }
#undef KF_LAUNCH_SELF
    };

    if (!hit1) {
        launch_precompute(d_coords1, d_Z1, d_N1, side1, nm1, max_atoms1, grid1);
        CUDA_CHECK(cudaGetLastError());
        if (use_alchemy) {
            launch_mix(d_N1, side1, nm1, max_atoms1, grid1);
            CUDA_CHECK(cudaGetLastError());
        }
        launch_self(d_N1, side1, side1.ss, nm1, max_atoms1, grid1);
        CUDA_CHECK(cudaGetLastError());
        side_key_store(
            side1.key, d_coords1, d_Z1, d_N1, nm1, max_atoms1, sigma, two_body_scaling,
            two_body_width, two_body_power, three_body_scaling, three_body_width, three_body_power,
            fourier_order, use_atm, use_alchemy, alchemy_period_width, alchemy_group_width,
            params.pmax, params.z_to_idx
        );
    }

    if (!same) {
        if (!hit2) {
            const size_t n_flat2 = static_cast<size_t>(nm2) * max_atoms2;
            int grid2 = static_cast<int>((n_flat2 + block - 1) / block);
            launch_precompute(d_coords2, d_Z2, d_N2, side2, nm2, max_atoms2, grid2);
            CUDA_CHECK(cudaGetLastError());
            if (use_alchemy) {
                launch_mix(d_N2, side2, nm2, max_atoms2, grid2);
                CUDA_CHECK(cudaGetLastError());
            }
            launch_self(d_N2, side2, ss2, nm2, max2, grid2);
            CUDA_CHECK(cudaGetLastError());
            side_key_store(
                side2.key, d_coords2, d_Z2, d_N2, nm2, max_atoms2, sigma, two_body_scaling,
                two_body_width, two_body_power, three_body_scaling, three_body_width,
                three_body_power, fourier_order, use_atm, use_alchemy, alchemy_period_width,
                alchemy_group_width, params.pmax, params.z_to_idx
            );
        }
        blob2_e = side2.blob_e;
    } else {
        ss2 = side1.ss;
        blob2_e = side1.blob_e;
    }

    const int n_pairs = nm1 * nm2;
    const int pm = params.pmax;
    char *be1 = use_alchemy ? side1.blob_e : nullptr;
    char *be2 = use_alchemy ? blob2_e : nullptr;
#define KF_LAUNCH_RECT(FO, PM, AL)                                                                 \
    rect_kernel<FO, PM, AL><<<n_pairs, 128>>>(                                                     \
        d_N1, d_N2, side1.nn, nn2, side1.mol_byte_off, mol_byte_off2, side1.blob, blob2, be1, be2, \
        side1.ss, ss2, d_K_out, nm1, nm2, max_atoms1, max2, hot, d_s_prefactor                        \
    )
    if (use_alchemy) {
        if (fourier_order == 1 && pm == 5) {
            KF_LAUNCH_RECT(1, 5, true);
        } else if (fourier_order == 1 && pm == 4) {
            KF_LAUNCH_RECT(1, 4, true);
        } else if (fourier_order == 1) {
            KF_LAUNCH_RECT(1, 0, true);
        } else if (fourier_order == 2) {
            KF_LAUNCH_RECT(2, 0, true);
        } else {
            KF_LAUNCH_RECT(0, 0, true);
        }
    } else if (fourier_order == 1 && pm == 5) {
        KF_LAUNCH_RECT(1, 5, false);
    } else if (fourier_order == 1 && pm == 4) {
        KF_LAUNCH_RECT(1, 4, false);
    } else if (fourier_order == 1) {
        KF_LAUNCH_RECT(1, 0, false);
    } else if (fourier_order == 2) {
        KF_LAUNCH_RECT(2, 0, false);
    } else {
        KF_LAUNCH_RECT(0, 0, false);
    }
#undef KF_LAUNCH_RECT
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace direct
}  // namespace fchl18
}  // namespace kf
