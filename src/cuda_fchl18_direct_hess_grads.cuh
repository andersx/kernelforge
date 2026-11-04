#pragma once

// Device helpers for the coords-only FCHL18 Hessian (FP32, no cutoff: cut = 1, fcp = 0).
//
// Mirrors the layouts of cuda_fchl18_direct.cu:
//   Neigh{rij, ksi, z, pad}, Fock = float2 (cos, sin) with per-slot layout
//   [slot * pmax * order + p * order + m], Z-sorted neighbours, slot 0 = centre.
//
// Contents
//   1. precompute_atom_with_grad<UseAtm>     forward (Neigh/Fock) + all first derivatives
//   2. scalar_noalchemy_and_grad / scalar_alchemy_and_grad   s and ds/dR_A
//
// Gradient layouts (na3 = 3 * na, atom-major then xyz, "amu" = atom*3 + mu):
//   rad[slot] = {ux, uy, uz, dksi_dr}            compact two-body gradients (float4):
//       u = (R_nbr - R_centre)/r,  dksi_dr = d ksi / d r
//       d r_slot/dR[nbr] = +u, d r_slot/dR[centre] = -u,
//       d ksi_slot/dR[nbr] = +dksi_dr u, d ksi_slot/dR[centre] = -dksi_dr u,  zero elsewhere
//       (slot 0 and r < eps slots are all zero)
//   dcosp/dsinp [(slot*pmax*order + p*order + m) * na3 + amu]
//                                                d Fock[slot,p,m].{x,y} / dR
//   nbr_atom_idx[slot]                           molecule atom index (slot 0 = centre)
//
// Assumptions
//   * Three-body Fourier terms are filled for any na (displacement vectors are recomputed from
//     coords via nbr_atom_idx; no fixed-size dxs/dys/dzs stack).
//   * Three-body radial weight = three_body_radial_weight(kTbWeightProduct, ...) like direct.cu
//     Fourier modes n = m + 1 with m = 0..order_full-1; the even-n modes are identically zero
//     (cos(n th) - cos(n (th + pi)) = 0), so `order` below is the number of retained odd-n
//     modes ((order_full + 1) / 2) and slot mi holds n = 2 mi + 1. Gradients use the same
//     cos(n th) - cos(n (th + pi)) factors as the forward.
//   * Alchemy: the pd table (Z-indexed emax x emax) is passed as a pointer (e.g. the address
//     of the module's __constant__ c_alchemy_pd); no symbol is defined here.

#include "cuda_fchl18_common.cuh"

namespace kf {
namespace fchl18 {
namespace direct {
namespace hess {

// One neighbour slot (same as direct.cu).
struct alignas(16) Neigh {
    float rij;
    float ksi;
    int z;
    int pad;
};

// Fourier (cos, sin) interleaved.
using Fock = float2;

// Soft host-side cap (gblob is O(N^3) floats; real limit is GPU memory).
constexpr int kHessMaxAtoms = 1024;
constexpr int kHessMaxOrder = 2;            // cross-Hessian channel table limit
constexpr int kHessMaxPmax = 8;             // cross-Hessian channel table limit
constexpr int kHessMaxChannels = 1 + 2 * kHessMaxOrder * kHessMaxPmax;  // 33

// ---------------------------------------------------------------------------
// 1. Precompute (forward + gradients) for centre atom i
// ---------------------------------------------------------------------------

// Sizes (floats) of the gradient output buffers for a centre atom of a molecule with na atoms:
//   rad             : na * 4   (float4 {ux, uy, uz, dksi_dr} per slot)
//   dcosp, dsinp    : na * pmax * order * 3 * na   (dense, per output)
__host__ __device__ inline long long hess_grad_slot_floats(int na) {
    return static_cast<long long>(na) * 4;
}
__host__ __device__ inline long long hess_grad_fourier_floats(int na, int pmax, int order) {
    return static_cast<long long>(na) * pmax * order * 3 * na;
}

// n_slots = na (dense). All output pointers are per-centre buffers sized as above
// (nbr_atom_idx: na ints). If zero_outputs is false the caller guarantees gradient buffers
// (dcosp, dsinp) are already zero (e.g. cudaMemset) and only the entries written by this
// function are touched; neigh/fock/nbr_atom_idx/rad are always (re)initialised.
template <bool UseAtm>
__device__ void precompute_atom_with_grad(
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
    Fock *fock,
    int *nbr_atom_idx,
    float4 *rad,
    float *dcosp,
    float *dsinp,
    bool zero_outputs = true
) {
    const int n_slots = na;
    const int na3 = 3 * na;
    const long long fblock = static_cast<long long>(pmax) * order;
    const long long fsz = fblock * n_slots;

    for (int k = 0; k < n_slots; ++k) {
        neigh[k] = Neigh{0.0f, 0.0f, 0, 0};
        nbr_atom_idx[k] = -1;
        rad[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (long long t = 0; t < fsz; ++t) {
        fock[t] = make_float2(0.0f, 0.0f);
    }
    if (zero_outputs) {
        const long long gf = fsz * na3;
        for (long long t = 0; t < gf; ++t) {
            dcosp[t] = 0.0f;
            dsinp[t] = 0.0f;
        }
    }

    const float *ra = coords_mol + i * 3;
    const int zi = Z_mol[i];
    neigh[0] = Neigh{0.0f, 0.0f, zi, 0};
    nbr_atom_idx[0] = i;

    // --- neighbours (unsorted) -------------------------------------------------------
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
        nbr_atom_idx[slot] = j;
        ++slot;
    }
    const int n_neigh = slot;
    *nn_out = n_neigh;

    // --- stable insertion sort by Z (identical to direct.cu), carrying nbr idx ---------
    for (int k = 2; k < n_neigh; ++k) {
        const Neigh saved = neigh[k];
        const int saved_idx = nbr_atom_idx[k];
        int insert_at = k;
        while (insert_at > 1 && neigh[insert_at - 1].z > saved.z) {
            neigh[insert_at] = neigh[insert_at - 1];
            nbr_atom_idx[insert_at] = nbr_atom_idx[insert_at - 1];
            --insert_at;
        }
        neigh[insert_at] = saved;
        nbr_atom_idx[insert_at] = saved_idx;
    }

    // --- two-body gradients: compact rad = {u, dksi_dr} (cut = 1, fcp = 0) --------------
    //   ksi = r^-p,  d ksi / d r = -p r^(-p-1) = -p ksi / r
    //   d r / d R[nbr] = +u,  d r / d R[centre] = -u,  u = (R_nbr - R_centre) / r
    for (int s = 1; s < n_neigh; ++s) {
        const float r = neigh[s].rij;
        if (r < Math<float>::eps()) {
            continue;
        }
        const float *rn = coords_mol + nbr_atom_idx[s] * 3;
        const float inv_r = 1.0f / r;
        const float ux = (rn[0] - ra[0]) * inv_r;
        const float uy = (rn[1] - ra[1]) * inv_r;
        const float uz = (rn[2] - ra[2]) * inv_r;
        const float dksi_dr = -two_body_power * neigh[s].ksi * inv_r;
        rad[s] = make_float4(ux, uy, uz, dksi_dr);
    }

    // --- three-body Fourier terms and their gradients ----------------------------------
    for (int j = 1; j < n_neigh; ++j) {
        const int j_atom = nbr_atom_idx[j];
        const float *rj = coords_mol + j_atom * 3;
        const float dxj = rj[0] - ra[0];
        const float dyj = rj[1] - ra[1];
        const float dzj = rj[2] - ra[2];
        const float dj = neigh[j].rij;
        if (dj < Math<float>::eps()) {
            continue;
        }
        const float inv_dj = 1.0f / dj;
        const float ujx = dxj * inv_dj;
        const float ujy = dyj * inv_dj;
        const float ujz = dzj * inv_dj;

        for (int k = j + 1; k < n_neigh; ++k) {
            const int k_atom_pre = nbr_atom_idx[k];
            const float *rk = coords_mol + k_atom_pre * 3;
            const float dxk = rk[0] - ra[0];
            const float dyk = rk[1] - ra[1];
            const float dzk = rk[2] - ra[2];
            const float dk = neigh[k].rij;
            if (dk < Math<float>::eps()) {
                continue;
            }
            const float dxjk = dxj - dxk;
            const float dyjk = dyj - dyk;
            const float dzjk = dzj - dzk;
            const float di = sqrtf(dxjk * dxjk + dyjk * dyjk + dzjk * dzjk);
            if (di < Math<float>::eps()) {
                continue;
            }

            const float inv_dk = 1.0f / dk;
            const float ukx = dxk * inv_dk;
            const float uky = dyk * inv_dk;
            const float ukz = dzk * inv_dk;
            const float cos_i = clamp11(ujx * ukx + ujy * uky + ujz * ukz);

            const float inv_di = 1.0f / di;
            // e = (R_j - R_k) / di  (unit vector k -> j);  ukj = -e (unit vector j -> k)
            const float ex = dxjk * inv_di;
            const float ey = dyjk * inv_di;
            const float ez = dzjk * inv_di;

            float atm = 1.0f;
            float cos_jj = 0.0f;
            float cos_kk = 0.0f;
            float datm_i = 0.0f;
            float datm_j = 0.0f;
            float datm_k = 0.0f;
            if constexpr (UseAtm) {
                cos_jj = clamp11(-ex * (-ujx) + -ey * (-ujy) + -ez * (-ujz));
                cos_kk = clamp11(ex * (-ukx) + ey * (-uky) + ez * (-ukz));
                atm = 1.0f + 3.0f * cos_i * cos_jj * cos_kk;
                datm_i = 3.0f * cos_jj * cos_kk;
                datm_j = 3.0f * cos_i * cos_kk;
                datm_k = 3.0f * cos_i * cos_jj;
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

            const int pj = z_to_idx[zk];
            const int pk = z_to_idx[zj];
            if (pj < 0 || pk < 0) {
                continue;
            }
            const int k_atom = k_atom_pre;
            const int atoms[3] = {i, j_atom, k_atom};

            const float uj[3] = {ujx, ujy, ujz};
            const float uk[3] = {ukx, uky, ukz};
            const float e[3] = {ex, ey, ez};
            const float ukj[3] = {-ex, -ey, -ez};

            // d cos_i / dR  [la: 0 centre, 1 j_atom, 2 k_atom][mu]
            float dcos_i[3][3];
            for (int mu = 0; mu < 3; ++mu) {
                dcos_i[1][mu] = (uk[mu] - cos_i * uj[mu]) * inv_dj;
                dcos_i[2][mu] = (uj[mu] - cos_i * uk[mu]) * inv_dk;
                dcos_i[0][mu] = -dcos_i[1][mu] - dcos_i[2][mu];
            }

            // d ksi3 / dR = atm*w*(dlog_dj d(dj) + dlog_dk d(dk) + dlog_di d(di))
            //             + w * (datm_i dcos_i + datm_j dcos_j + datm_k dcos_k)
            //   d(dj)/dR: centre -uj, j +uj;  d(dk)/dR: centre -uk, k +uk;
            //   d(di)/dR: j +e, k -e.
            float gksi3[3][3];
            const float kw = ksi3;  // atm * w
            for (int mu = 0; mu < 3; ++mu) {
                gksi3[0][mu] = kw * (radial.dlog_dj * (-uj[mu]) + radial.dlog_dk * (-uk[mu]));
                gksi3[1][mu] = kw * (radial.dlog_dj * uj[mu] + radial.dlog_di * e[mu]);
                gksi3[2][mu] = kw * (radial.dlog_dk * uk[mu] + radial.dlog_di * (-e[mu]));
            }
            if constexpr (UseAtm) {
                const float w = radial.w;
                for (int mu = 0; mu < 3; ++mu) {
                    // d cos_j / dR
                    const float cj0 = inv_dj * (ukj[mu] + cos_jj * uj[mu]);
                    const float cj1 = inv_di * (uj[mu] + cos_jj * ukj[mu]) - cj0;
                    const float cj2 = -inv_di * (uj[mu] + cos_jj * ukj[mu]);
                    // d cos_k / dR  (ujk = e)
                    const float ck0 = inv_dk * (e[mu] + cos_kk * uk[mu]);
                    const float ck1 = -inv_di * (uk[mu] + cos_kk * e[mu]);
                    const float ck2 = inv_di * (uk[mu] + cos_kk * e[mu]) - ck0;
                    gksi3[0][mu] += w * (datm_i * dcos_i[0][mu] + datm_j * cj0 + datm_k * ck0);
                    gksi3[1][mu] += w * (datm_i * dcos_i[1][mu] + datm_j * cj1 + datm_k * ck1);
                    gksi3[2][mu] += w * (datm_i * dcos_i[2][mu] + datm_j * cj2 + datm_k * ck2);
                }
            }

            const float theta = acosf(cos_i);
            const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_i * cos_i));
            // d theta / dR * ksi3
            float dth_ksi[3][3];
            for (int la = 0; la < 3; ++la) {
                for (int mu = 0; mu < 3; ++mu) {
                    dth_ksi[la][mu] =
                        (sin_theta > 1e-6f) ? (-1.0f / sin_theta) * dcos_i[la][mu] * ksi3 : 0.0f;
                }
            }

            for (int mi = 0; mi < order; ++mi) {
                // Only odd n = m + 1 survive (even n: cos(n th) - cos(n (th + pi)) == 0 and so
                // does its derivative), so slot mi stores mode m = 2 mi, n = 2 mi + 1.
                const int m = 2 * mi;
                const float mf = static_cast<float>(m + 1);
                float sin_mth, cos_mth, sin_mthpi, cos_mthpi;
                sincosf(mf * theta, &sin_mth, &cos_mth);
                sincosf(mf * (theta + Math<float>::pi()), &sin_mthpi, &cos_mthpi);
                const float cos_mfac = cos_mth - cos_mthpi;
                const float sin_mfac = sin_mth - sin_mthpi;
                const float dcos_mfac_dth = -mf * sin_mth + mf * sin_mthpi;
                const float dsin_mfac_dth = mf * cos_mth - mf * cos_mthpi;

                const long long fidx_j =
                    static_cast<long long>(j) * fblock + static_cast<long long>(pj) * order + mi;
                const long long fidx_k =
                    static_cast<long long>(k) * fblock + static_cast<long long>(pk) * order + mi;

                const float cos_m = cos_mfac * ksi3;
                const float sin_m = sin_mfac * ksi3;
                fock[fidx_j].x += cos_m;
                fock[fidx_j].y += sin_m;
                fock[fidx_k].x += cos_m;
                fock[fidx_k].y += sin_m;

                for (int la = 0; la < 3; ++la) {
                    for (int mu = 0; mu < 3; ++mu) {
                        const float dc =
                            dcos_mfac_dth * dth_ksi[la][mu] + cos_mfac * gksi3[la][mu];
                        const float ds =
                            dsin_mfac_dth * dth_ksi[la][mu] + sin_mfac * gksi3[la][mu];
                        const int amu = atoms[la] * 3 + mu;
                        const long long oj = fidx_j * na3 + amu;
                        const long long ok = fidx_k * na3 + amu;
                        dcosp[oj] += dc;
                        dsinp[oj] += ds;
                        dcosp[ok] += dc;
                        dsinp[ok] += ds;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. s and ds/dR_A
// ---------------------------------------------------------------------------

struct ScalarParams {
    float two_body_width;
    float true_distance_scale;
    float true_angular_scale;
    int fourier_order;
    int pmax;
    int emax;                    // alchemy only
    const float *s_prefactor;    // [order] per-mode angular prefactor (device-readable)
};

namespace detail {

struct PairCtx {
    const Neigh *nA;
    const Fock *fA;
    const float4 *radA;
    const int *nbrA;  // nbrA[0] = centre atom
    const float *dcospA;
    const float *dsinpA;
    const Neigh *nB;
    const Fock *fB;
    int na3;
    int order;
    int pmax;
    int fblock;
    float inv_width;
    float dscale;
    float ascale;
    const float *spf;
};

// One (ii, jj) neighbour pair: adds G*W to s and its A-gradient to ds. G includes pd_ij.
// fB is the (possibly element-mixed) Fock of side B.
__device__ inline void accumulate_pair(
    const PairCtx &c, int ii, int jj, float pd_ij, float *aadist, float *ds_dR
) {
    const float ri = c.nA[ii].rij;
    const float rj = c.nB[jj].rij;
    const float dr = ri - rj;
    const float r2 = dr * dr;
    const float G = expf(r2 * c.inv_width) * pd_ij;
    const float dG_dri = G * 2.0f * dr * c.inv_width;

    const Fock *f1 = c.fA + static_cast<long long>(ii) * c.fblock;
    const Fock *f2 = c.fB + static_cast<long long>(jj) * c.fblock;

    float angular = 0.0f;
    for (int p = 0; p < c.pmax; ++p) {
        for (int m = 0; m < c.order; ++m) {
            const int t = p * c.order + m;
            angular += (f1[t].x * f2[t].x + f1[t].y * f2[t].y) * c.spf[m];
        }
    }
    const float ksi_j = c.nB[jj].ksi;
    const float W = c.nA[ii].ksi * ksi_j * c.dscale + angular * c.ascale;
    *aadist += G * W;

    // Radial terms are supported on 6 DOFs only (neighbour atom +u, centre atom -u).
    {
        const float4 rd = c.radA[ii];
        const float cu = dG_dri * W + G * c.dscale * rd.w * ksi_j;
        const int bn = c.nbrA[ii] * 3;
        const int bc = c.nbrA[0] * 3;
        ds_dR[bn + 0] += cu * rd.x;
        ds_dR[bn + 1] += cu * rd.y;
        ds_dR[bn + 2] += cu * rd.z;
        ds_dR[bc + 0] -= cu * rd.x;
        ds_dR[bc + 1] -= cu * rd.y;
        ds_dR[bc + 2] -= cu * rd.z;
    }

    const float G_ang = G * c.ascale;
    for (int p = 0; p < c.pmax; ++p) {
        for (int m = 0; m < c.order; ++m) {
            const int t = p * c.order + m;
            const float c2 = f2[t].x;
            const float s2 = f2[t].y;
            if (c2 == 0.0f && s2 == 0.0f) {
                continue;
            }
            const float coef = G_ang * c.spf[m];
            const long long fi =
                (static_cast<long long>(ii) * c.fblock + t) * static_cast<long long>(c.na3);
            const float *rc = c.dcospA + fi;
            const float *rs = c.dsinpA + fi;
            for (int amu = 0; amu < c.na3; ++amu) {
                ds_dR[amu] += coef * (rc[amu] * c2 + rs[amu] * s2);
            }
        }
    }
}

}  // namespace detail

// No alchemy: centres must share Z; only same-Z neighbour pairs contribute (merge over
// Z-sorted slots). Returns s; ds_dR_A[na3A] is overwritten with ds/dR_A.
__device__ inline float scalar_noalchemy_and_grad(
    const Neigh *nA,
    int nnA,
    const Fock *fA,
    const float4 *radA,
    const int *nbrA,
    const float *dcospA,
    const float *dsinpA,
    const Neigh *nB,
    int nnB,
    const Fock *fB,
    const ScalarParams &p,
    int na3A,
    float *ds_dR
) {
    for (int a = 0; a < na3A; ++a) {
        ds_dR[a] = 0.0f;
    }
    if (nA[0].z != nB[0].z) {
        return 0.0f;
    }

    detail::PairCtx c;
    c.nA = nA;
    c.fA = fA;
    c.radA = radA;
    c.nbrA = nbrA;
    c.dcospA = dcospA;
    c.dsinpA = dsinpA;
    c.nB = nB;
    c.fB = fB;
    c.na3 = na3A;
    c.order = p.fourier_order;
    c.pmax = p.pmax;
    c.fblock = p.pmax * p.fourier_order;
    c.inv_width = -1.0f / (4.0f * p.two_body_width * p.two_body_width);
    c.dscale = p.true_distance_scale;
    c.ascale = p.true_angular_scale;
    c.spf = p.s_prefactor;
    const float maxgausdist2 = (8.0f * p.two_body_width) * (8.0f * p.two_body_width);

    float aadist = 1.0f;
    int i = 1;
    int j = 1;
    while (i < nnA && j < nnB) {
        const int zi = nA[i].z;
        const int zj = nB[j].z;
        if (zi < zj) {
            ++i;
            continue;
        }
        if (zj < zi) {
            ++j;
            continue;
        }
        int i_end = i + 1;
        while (i_end < nnA && nA[i_end].z == zi) {
            ++i_end;
        }
        int j_end = j + 1;
        while (j_end < nnB && nB[j_end].z == zi) {
            ++j_end;
        }
        for (int ii = i; ii < i_end; ++ii) {
            const float ri = nA[ii].rij;
            for (int jj = j; jj < j_end; ++jj) {
                const float d = nB[jj].rij - ri;
                if (d * d >= maxgausdist2) {
                    continue;
                }
                detail::accumulate_pair(c, ii, jj, 1.0f, &aadist, ds_dR);
            }
        }
        i = i_end;
        j = j_end;
    }
    return aadist;
}

// Alchemy: full neighbour double loop with pd(Zi, Zj) weights and pd(Zc, Zc') on the centre.
// fB must be the element-mixed ("effective") Fock of side B:
//   F_eff[p1] = sum_p2 PDc[p1, p2] F[p2]   (same convention as direct.cu).
// pd: Z-indexed emax x emax table (e.g. address of __constant__ c_alchemy_pd).
__device__ inline float scalar_alchemy_and_grad(
    const Neigh *nA,
    int nnA,
    const Fock *fA,
    const float4 *radA,
    const int *nbrA,
    const float *dcospA,
    const float *dsinpA,
    const Neigh *nB,
    int nnB,
    const Fock *fB_eff,
    const ScalarParams &p,
    const float *pd,
    int na3A,
    float *ds_dR
) {
    for (int a = 0; a < na3A; ++a) {
        ds_dR[a] = 0.0f;
    }
    const float pd_central = alch_pd(pd, p.emax, nA[0].z, nB[0].z);
    if (pd_central == 0.0f) {
        return 0.0f;
    }

    detail::PairCtx c;
    c.nA = nA;
    c.fA = fA;
    c.radA = radA;
    c.nbrA = nbrA;
    c.dcospA = dcospA;
    c.dsinpA = dsinpA;
    c.nB = nB;
    c.fB = fB_eff;
    c.na3 = na3A;
    c.order = p.fourier_order;
    c.pmax = p.pmax;
    c.fblock = p.pmax * p.fourier_order;
    c.inv_width = -1.0f / (4.0f * p.two_body_width * p.two_body_width);
    c.dscale = p.true_distance_scale;
    c.ascale = p.true_angular_scale;
    c.spf = p.s_prefactor;
    const float maxgausdist2 = (8.0f * p.two_body_width) * (8.0f * p.two_body_width);

    float aadist = 1.0f;
    for (int ii = 1; ii < nnA; ++ii) {
        const int zi = nA[ii].z;
        const float ri = nA[ii].rij;
        for (int jj = 1; jj < nnB; ++jj) {
            const float d = nB[jj].rij - ri;
            if (d * d >= maxgausdist2) {
                continue;
            }
            const float pd_ij = alch_pd(pd, p.emax, zi, nB[jj].z);
            if (pd_ij == 0.0f) {
                continue;
            }
            detail::accumulate_pair(c, ii, jj, pd_ij, &aadist, ds_dR);
        }
    }
    if (pd_central != 1.0f) {
        aadist *= pd_central;
        for (int a = 0; a < na3A; ++a) {
            ds_dR[a] *= pd_central;
        }
    }
    return aadist;
}

}  // namespace hess
}  // namespace direct
}  // namespace fchl18
}  // namespace kf
