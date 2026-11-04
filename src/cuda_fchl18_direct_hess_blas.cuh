#pragma once

// Hessian apply via cuBLAS.
//
// Per centre pair (i, j) the pack kernel builds factored tiles.  Two layouts:
//
//   n_ij_fuse == 0  (single centre-pair slots):
//     Dri,Dksi (na3 x P), X,Y (P x nb3), Afock (na3 x K1), Z (K1 x nb3), K1=P*2*FB
//     H += ch * (Dri X + ...) + cg * gA gB^T
//
//   n_ij_fuse == na*nb  (fixed-size mols; slots grouped by molecule pair):
//     Concatenate all (i,j) columns so one mol pair is one GEMM with
//       K_rad = n_ij*P,  K_fock = n_ij*K1
//     coeff_hess is baked into X/Y/Z; coeff_g into gB columns.
//     Then batch over molecule pairs — the right shape for 100–1K train sets.

#include <cublas_v2.h>
#include <cstdio>

#define KF_HESS_CUBLAS_CHECK(call)                                                        \
    do {                                                                                  \
        cublasStatus_t _s = (call);                                                       \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                                \
            fprintf(stderr, "cuBLAS error at %s:%d — status %d\n", __FILE__, __LINE__,    \
                    (int)_s);                                                             \
            throw std::runtime_error("cuda_fchl18_direct_hess: cuBLAS call failed");      \
        }                                                                                 \
    } while (0)

// Included inside the translation unit's anonymous namespace (sees HotParams, Fock, ...).
namespace hess_blas {

// BLAS path when max atoms >= this.  Below: in-kernel (still fastest for ~9-atom
// rMD17).  Bulk mol-pair pack is used automatically when n_ij <= 128.
constexpr int kBlasAtomThresh = 32;
// Max molecule-pairs (fused) or centre-pairs (unfused) per pack/GEMM wave.
constexpr int kBlasBatchDefault = 2048;

inline cublasHandle_t &handle() {
    static cublasHandle_t h = nullptr;
    return h;
}

inline void ensure_cublas() {
    if (!handle()) {
        if (cublasCreate(&handle()) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuda_fchl18_direct_hess: cublasCreate failed");
        }
    }
}

// Flat centre-pair list entry.
struct Pair {
    int a, b, i, j;
};

// ---------------------------------------------------------------------------
// Scatter Htile (col-major, lda=max_na3) into global H.  Optional rank-1
// (unfused): val = ch*Htile + cg*gA*gB.  Fused path bakes coeffs into Htile.
// ---------------------------------------------------------------------------
__global__ void blas_scatter_kernel(
    const float *__restrict__ Htile,
    const float *__restrict__ gA,  // null if fused
    const float *__restrict__ gB,
    const float *__restrict__ ch,
    const float *__restrict__ cg,
    const Pair *__restrict__ pairs,
    const int *n1,
    const int *n2,
    const int *atom_base1,
    const int *atom_base2,
    float *__restrict__ H,
    long long ld_H,
    int batch,
    int max_na3,
    int max_nb3,
    int fused  // 1: H += Htile only (pairs are molecule pairs)
) {
    const int s = static_cast<int>(blockIdx.x);
    if (s >= batch) {
        return;
    }
    const Pair pr = pairs[s];
    const int na3 = 3 * n1[pr.a];
    const int nb3 = 3 * n2[pr.b];
    float *Hblk = H + 3LL * atom_base1[pr.a] * ld_H + 3LL * atom_base2[pr.b];
    const float *tile = Htile + static_cast<long long>(s) * max_na3 * max_nb3;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);
    if (fused) {
        for (int flat = tid; flat < na3 * nb3; flat += nthr) {
            const int amu = flat / nb3;
            const int bnu = flat - amu * nb3;
            atomicAdd(
                Hblk + static_cast<long long>(amu) * ld_H + bnu,
                tile[amu + static_cast<long long>(bnu) * max_na3]
            );
        }
        return;
    }
    const float *ga = gA + static_cast<long long>(s) * max_na3;
    const float *gb = gB + static_cast<long long>(s) * max_nb3;
    const float c_h = ch[s];
    const float c_g = cg[s];
    for (int flat = tid; flat < na3 * nb3; flat += nthr) {
        const int amu = flat / nb3;
        const int bnu = flat - amu * nb3;
        const float ht = tile[amu + static_cast<long long>(bnu) * max_na3];
        atomicAdd(
            Hblk + static_cast<long long>(amu) * ld_H + bnu, c_h * ht + c_g * ga[amu] * gb[bnu]
        );
    }
}

// ---------------------------------------------------------------------------
// Pack kernel: one block per batch slot (one centre pair).
// Dynamic smem layout matches hess_hot_kernel fixed part (no H tile).
// ---------------------------------------------------------------------------
template <bool Alch, int Order>
__global__ void __launch_bounds__(128, 4) blas_pack_kernel(
    const Pair *__restrict__ pairs,
    int batch,
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
    int max1,
    int max2,
    float inv,
    HotParams hp,
    const float *pdc,
    int max_na3,
    int max_nb3,
    int P,           // = max_n pad for neighbour index
    int FB,          // pmax * order
    int n_ij_fuse,   // 0 = unfused; else na*nb fused columns per mol pair
    int bulk_mol,    // 1: pairs[s]=(a,b), loop all (i,j); 0: pairs[s]=(a,b,i,j)
    // outputs — see file header.  Host zero-fills fused buffers before launch.
    float *__restrict__ out_Dri,
    float *__restrict__ out_Dksi,
    float *__restrict__ out_X,
    float *__restrict__ out_Y,
    float *__restrict__ out_Afock,
    float *__restrict__ out_Z,
    float *__restrict__ out_gA,
    float *__restrict__ out_gB,
    float *__restrict__ out_ch,
    float *__restrict__ out_cg
) {
    extern __shared__ float4 hot_smem4[];
    float *hot_smem = reinterpret_cast<float *>(hot_smem4);
    __shared__ float s_pdc[hess::kHessMaxPmax * hess::kHessMaxPmax];
    __shared__ float s_sij[1];

    const int s = static_cast<int>(blockIdx.x);
    if (s >= batch) {
        return;
    }
    // bulk_mol: one block / mol pair, serial (i,j) — good when n_ij is small (rMD17).
    // fused !bulk: one block / (i,j), writes into fused mol columns — needs many blocks.
    // unfused: one block / (i,j), own slot.
    const Pair pr = pairs[s];
    const int a = pr.a, b = pr.b;
    const int na = n1[a], nb = n2[b];
    const int na3 = 3 * na, nb3 = 3 * nb;
    constexpr int order_c = Order;
    const int order = (order_c > 0) ? order_c : hp.order;
    const int pmax = hp.pmax;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);
    const float dscale = hp.dscale;
    const float ascale = hp.ascale;
    const float inv_width = hp.inv_width;
    const float maxgausdist2 = hp.maxgausdist2;
    const float s0 = hp.spf[0];
    const float s1 = hp.spf[1];
    const int K = 2 * FB;
    const int K1 = P * K;  // fock cols per centre pair
    const bool fuse = (n_ij_fuse > 0);
    const int n_ij = fuse ? n_ij_fuse : 1;
    const int mol = (fuse && !bulk_mol) ? (s / n_ij) : s;
    const int k_rad = n_ij * P;
    const int k_fock = n_ij * K1;
    const int max_nb = max_nb3 / 3;
    const int i_lo = bulk_mol ? 0 : pr.i;
    const int i_hi = bulk_mol ? na : (pr.i + 1);
    const int j_lo = bulk_mol ? 0 : pr.j;
    const int j_hi = bulk_mol ? nb : (pr.j + 1);

    float4 *s_rad = hot_smem4;
    int *s_nbq = reinterpret_cast<int *>(hot_smem + 4 * max_nb);
    float2 *s_Z = reinterpret_cast<float2 *>(hot_smem + ((5 * max_nb + 1) & ~1));
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

    // Unfused: zero this centre-pair slot.  Fused: host cudaMemset's the mol-pair buffers.
    if (!fuse) {
        float *Dri0 = out_Dri + static_cast<long long>(s) * max_na3 * P;
        float *Dksi0 = out_Dksi + static_cast<long long>(s) * max_na3 * P;
        float *X0 = out_X + static_cast<long long>(s) * P * max_nb3;
        float *Y0 = out_Y + static_cast<long long>(s) * P * max_nb3;
        float *Af0 = out_Afock + static_cast<long long>(s) * max_na3 * K1;
        float *Z0 = out_Z + static_cast<long long>(s) * K1 * max_nb3;
        for (int e = tid; e < max_na3 * P; e += nthr) {
            Dri0[e] = 0.0f;
            Dksi0[e] = 0.0f;
        }
        for (int e = tid; e < P * max_nb3; e += nthr) {
            X0[e] = 0.0f;
            Y0[e] = 0.0f;
        }
        for (int e = tid; e < max_na3 * K1; e += nthr) {
            Af0[e] = 0.0f;
        }
        for (int e = tid; e < K1 * max_nb3; e += nthr) {
            Z0[e] = 0.0f;
        }
        for (int e = tid; e < max_na3; e += nthr) {
            out_gA[static_cast<long long>(s) * max_na3 + e] = 0.0f;
        }
        for (int e = tid; e < max_nb3; e += nthr) {
            out_gB[static_cast<long long>(s) * max_nb3 + e] = 0.0f;
        }
    }
    if constexpr (Alch) {
        if (tid < pmax * pmax) {
            s_pdc[tid] = pdc[tid];
        }
        __syncthreads();
    }

    // Bulk: one block walks all centre pairs of this molecule pair (fused), or just one (i,j).
    for (int i = i_lo; i < i_hi; ++i) {
    for (int j = j_lo; j < j_hi; ++j) {
    const int ij = fuse ? (bulk_mol ? (i * nb + j) : (s - mol * n_ij)) : 0;

    for (int e = tid; e < max_na3; e += nthr) {
        s_dsA[e] = 0.0f;
    }
    for (int e = tid; e < max_nb3; e += nthr) {
        s_dsB[e] = 0.0f;
    }
    __syncthreads();

    const int flat_i = a * max1 + i;
    const int flat_j = b * max2 + j;
    const int nnA = nn1[flat_i];
    const int nnB = nn2[flat_j];
    const Neigh *ni = nullptr;
    const Fock *fi = nullptr;
    centre_views(blob1, mol_byte_off1, a, i, na, pmax, order, &ni, &fi);
    const Neigh *nj = nullptr;
    const Fock *fj = nullptr;
    centre_views(blob2, mol_byte_off2, b, j, nb, pmax, order, &nj, &fj);
    const Fock *fi_e = fi;
    const Fock *fj_e = fj;
    if constexpr (Alch) {
        fi_e = centre_fock_ptr(blob1_e, mol_byte_off1, a, i, na, pmax, order);
        fj_e = centre_fock_ptr(blob2_e, mol_byte_off2, b, j, nb, pmax, order);
    }
    const GradViewsC gi = grad_views(gblob1, mol_gbyte_off1, a, i, na, pmax, order);
    const GradViewsC gj = grad_views(gblob2, mol_gbyte_off2, b, j, nb, pmax, order);
    const float *dssA = dss1 + dss_off1[a] + static_cast<long long>(i) * na3;
    const float *dssB = dss2 + dss_off2[b] + static_cast<long long>(j) * nb3;

    for (int q = tid; q < nnB; q += nthr) {
        s_rad[q] = gj.rad[q];
        s_nbq[q] = gj.nbr[q];
    }

    float pd_central = 1.0f;
    bool pair_on;
    if constexpr (Alch) {
        pd_central = alch_pd(static_cast<const float *>(c_alchemy_pd), hp.emax, ni[0].z, nj[0].z);
        pair_on = (pd_central != 0.0f);
    } else {
        pair_on = (ni[0].z == nj[0].z);
    }
    if (tid == 0) {
        s_sij[0] = pair_on ? pd_central : 0.0f;
    }
    __syncthreads();

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
                    pd_pq *= alch_pd(static_cast<const float *>(c_alchemy_pd), hp.emax, zp, nj[q].z);
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
                    const float V = ksi_p * nj[q].ksi * dscale +
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
    const float coeff = kij * inv;
    const float coeff_hess = 2.0f * coeff;
    const float coeff_g = coeff * inv;
    if (!fuse && tid == 0) {
        out_ch[s] = coeff_hess;
        out_cg[s] = coeff_g;
    }

    float *Dri = out_Dri + static_cast<long long>(mol) * max_na3 * k_rad;
    float *Dksi = out_Dksi + static_cast<long long>(mol) * max_na3 * k_rad;
    float *X = out_X + static_cast<long long>(mol) * k_rad * max_nb3;
    float *Y = out_Y + static_cast<long long>(mol) * k_rad * max_nb3;
    float *Af = out_Afock + static_cast<long long>(mol) * max_na3 * k_fock;
    float *Zout = out_Z + static_cast<long long>(mol) * k_fock * max_nb3;
    const int col0_ij = ij * P;       // radial column base for this (i,j)
    const int fcol0_ij = ij * K1;     // fock column base

    if (pair_on) {
        for (int p = 1; p < nnA; ++p) {
            const float rp = ni[p].rij;
            const float ksi_p = ni[p].ksi;
            const int zp = ni[p].z;
            const float4 rdA = gi.rad[p];
            const int pat = gi.nbr[p];
            const Fock *fip = fi + static_cast<long long>(p) * FB;
            const Fock *fip_e = fi_e + static_cast<long long>(p) * FB;

            for (int q = 1 + tid; q < nnB; q += nthr) {
                float G = 0.0f, dGp = 0.0f, d2G = 0.0f, V = 0.0f, ksi_q = 0.0f;
                float pd_pq = pd_central;
                bool ok = true;
                if constexpr (Alch) {
                    pd_pq *= alch_pd(static_cast<const float *>(c_alchemy_pd), hp.emax, zp, nj[q].z);
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
                __syncthreads();
                continue;
            }

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
                    const float4 rd = s_rad[q];
                    const float um = (bmu == 0) ? rd.x : ((bmu == 1) ? rd.y : rd.z);
                    const float sgn = (bat == s_nbq[q]) ? 1.0f : ((bat == j) ? -1.0f : 0.0f);
                    const float dri = sgn * um;
                    const float dksi = dri * rd.w;
                    const float c = -dGp * dri;
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

            // dsA (same as hot Stage C head)
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

            // Pack column p (fused: offset by ij * P / ij * K1). Bake coeff_hess into X/Y/Z.
            const int col = col0_ij + p;
            if (tid == 0) {
                for (int mu = 0; mu < 3; ++mu) {
                    const float um = (mu == 0) ? rdA.x : ((mu == 1) ? rdA.y : rdA.z);
                    const int a_nbr = 3 * pat + mu;
                    const int a_ctr = 3 * i + mu;
                    Dri[a_nbr + col * max_na3] = um;
                    Dri[a_ctr + col * max_na3] = -um;
                    Dksi[a_nbr + col * max_na3] = um * rdA.w;
                    Dksi[a_ctr + col * max_na3] = -um * rdA.w;
                }
            }
            const float xscale = fuse ? coeff_hess : 1.0f;
            for (int bnu = tid; bnu < nb3; bnu += nthr) {
                X[col + static_cast<long long>(bnu) * k_rad] = s_X[bnu] * xscale;
                Y[col + static_cast<long long>(bnu) * k_rad] = s_Y[bnu] * xscale;
                for (int t = 0; t < FB; ++t) {
                    const float2 z = s_Z[static_cast<long long>(t) * max_nb3 + bnu];
                    const int row0 = fcol0_ij + p * K + 2 * t;
                    Zout[row0 + static_cast<long long>(bnu) * k_fock] = z.x * xscale;
                    Zout[row0 + 1 + static_cast<long long>(bnu) * k_fock] = z.y * xscale;
                }
            }
            for (int amu = tid; amu < na3; amu += nthr) {
                for (int t = 0; t < FB; ++t) {
                    const long long ia = (static_cast<long long>(p) * FB + t) * na3 + amu;
                    const int colf = fcol0_ij + p * K + 2 * t;
                    Af[amu + static_cast<long long>(colf) * max_na3] = gi.dcosp[ia];
                    Af[amu + static_cast<long long>(colf + 1) * max_na3] = gi.dsinp[ia];
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    // gA / gB: unfused one vector; fused = column ij of (na3 x n_ij) / (nb3 x n_ij), cg in gB.
    if (fuse) {
        for (int amu = tid; amu < na3; amu += nthr) {
            out_gA[amu + static_cast<long long>(ij) * max_na3 +
                   static_cast<long long>(mol) * max_na3 * n_ij] =
                -dssA[amu] + 2.0f * s_dsA[amu];
        }
        for (int bnu = tid; bnu < nb3; bnu += nthr) {
            out_gB[bnu + static_cast<long long>(ij) * max_nb3 +
                   static_cast<long long>(mol) * max_nb3 * n_ij] =
                (-dssB[bnu] + 2.0f * s_dsB[bnu]) * coeff_g;
        }
    } else {
        for (int amu = tid; amu < na3; amu += nthr) {
            out_gA[static_cast<long long>(s) * max_na3 + amu] = -dssA[amu] + 2.0f * s_dsA[amu];
        }
        for (int bnu = tid; bnu < nb3; bnu += nthr) {
            out_gB[static_cast<long long>(s) * max_nb3 + bnu] = -dssB[bnu] + 2.0f * s_dsB[bnu];
        }
    }
    __syncthreads();  // before next (i,j) reuses s_dsA / s_rad / s_Z
    }  // j
    }  // i
}

// Host: strided-batched GEMMs into Htile then scatter.
// n_ij_fuse==0: batch = centre pairs, k = P / K1.
// n_ij_fuse>0:  batch = molecule pairs, k = n_ij*P / n_ij*K1; gA/gB are (na3 x n_ij).
inline void apply_batch(
    int batch,
    int max_na3,
    int max_nb3,
    int P,
    int K1,
    int n_ij_fuse,
    float *Dri,
    float *Dksi,
    float *X,
    float *Y,
    float *Afock,
    float *Z,
    float *Htile,
    float *gA,
    float *gB,
    const float *ch,
    const float *cg,
    const Pair *pairs,
    const int *n1,
    const int *n2,
    const int *atom_base1,
    const int *atom_base2,
    float *H,
    long long ld_H
) {
    ensure_cublas();
    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;
    const int n_ij = (n_ij_fuse > 0) ? n_ij_fuse : 1;
    const int k_rad = n_ij * P;
    const int k_fock = n_ij * K1;
    const long long stride_Dri = static_cast<long long>(max_na3) * k_rad;
    const long long stride_X = static_cast<long long>(k_rad) * max_nb3;
    const long long stride_H = static_cast<long long>(max_na3) * max_nb3;
    const long long stride_Af = static_cast<long long>(max_na3) * k_fock;
    const long long stride_Z = static_cast<long long>(k_fock) * max_nb3;

    KF_HESS_CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle(), CUBLAS_OP_N, CUBLAS_OP_N, max_na3, max_nb3, k_rad, &alpha, Dri, max_na3,
        stride_Dri, X, k_rad, stride_X, &beta0, Htile, max_na3, stride_H, batch
    ));
    KF_HESS_CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle(), CUBLAS_OP_N, CUBLAS_OP_N, max_na3, max_nb3, k_rad, &alpha, Dksi, max_na3,
        stride_Dri, Y, k_rad, stride_X, &beta1, Htile, max_na3, stride_H, batch
    ));
    if (k_fock > 0) {
        KF_HESS_CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle(), CUBLAS_OP_N, CUBLAS_OP_N, max_na3, max_nb3, k_fock, &alpha, Afock, max_na3,
            stride_Af, Z, k_fock, stride_Z, &beta1, Htile, max_na3, stride_H, batch
        ));
    }

    if (n_ij_fuse > 0) {
        // Htile += GA @ GB^T  (GA: na3 x n_ij, GB: nb3 x n_ij, coeffs baked into GB)
        const long long stride_GA = static_cast<long long>(max_na3) * n_ij;
        const long long stride_GB = static_cast<long long>(max_nb3) * n_ij;
        KF_HESS_CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle(), CUBLAS_OP_N, CUBLAS_OP_T, max_na3, max_nb3, n_ij, &alpha, gA, max_na3,
            stride_GA, gB, max_nb3, stride_GB, &beta1, Htile, max_na3, stride_H, batch
        ));
        blas_scatter_kernel<<<batch, 128>>>(
            Htile, nullptr, nullptr, nullptr, nullptr, pairs, n1, n2, atom_base1, atom_base2, H,
            ld_H, batch, max_na3, max_nb3, /*fused=*/1
        );
    } else {
        blas_scatter_kernel<<<batch, 128>>>(
            Htile, gA, gB, ch, cg, pairs, n1, n2, atom_base1, atom_base2, H, ld_H, batch, max_na3,
            max_nb3, /*fused=*/0
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Small-molecule wide GEMM: group by A-centre (a,i).
// Dri/Dksi/Afock depend only on (a,i); X/Y/Z vary over targets (b,j).
// One GEMM: Dri @ [X_0|X_1|…] with n = nb3 * n_targ (≫ nb3) so cuBLAS pays off.
constexpr int kWideAtomMax = 32;  // KF_HESS_WIDE=1: centre-wide when max_n <= this
// Prefer GEMM n (= max_nb3 * n_targ_batch) at least this large.
constexpr int kWideMinN = 512;

struct Centre {
    int a, i;
};
struct Target {
    int b, j;
};

// A-side once per centre: Dri / Dksi / Afock (independent of targets).
template <bool Alch, int Order>
__global__ void __launch_bounds__(128, 4) blas_pack_centre_a_kernel(
    const Centre *__restrict__ centres,
    int batch_c,
    const int *n1,
    const int *nn1,
    const long long *mol_byte_off1,
    const long long *mol_gbyte_off1,
    const char *__restrict__ gblob1,
    int max1,
    HotParams hp,
    int max_na3,
    int P,
    int FB,
    float *__restrict__ out_Dri,
    float *__restrict__ out_Dksi,
    float *__restrict__ out_Afock
) {
    const int c = static_cast<int>(blockIdx.x);
    if (c >= batch_c) {
        return;
    }
    const Centre ctr = centres[c];
    const int a = ctr.a, i = ctr.i;
    const int na = n1[a];
    const int na3 = 3 * na;
    constexpr int order_c = Order;
    const int order = (order_c > 0) ? order_c : hp.order;
    const int pmax = hp.pmax;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);
    const int K = 2 * FB, K1 = P * K;
    const int flat_i = a * max1 + i;
    const int nnA = nn1[flat_i];
    const GradViewsC gi = grad_views(gblob1, mol_gbyte_off1, a, i, na, pmax, order);

    float *Dri = out_Dri + static_cast<long long>(c) * max_na3 * P;
    float *Dksi = out_Dksi + static_cast<long long>(c) * max_na3 * P;
    float *Af = out_Afock + static_cast<long long>(c) * max_na3 * K1;

    for (int e = tid; e < max_na3 * P; e += nthr) {
        Dri[e] = 0.0f;
        Dksi[e] = 0.0f;
    }
    for (int e = tid; e < max_na3 * K1; e += nthr) {
        Af[e] = 0.0f;
    }
    __syncthreads();
    for (int p = 1; p < nnA; ++p) {
        const float4 rdA = gi.rad[p];
        const int pat = gi.nbr[p];
        if (tid == 0) {
            for (int mu = 0; mu < 3; ++mu) {
                const float um = (mu == 0) ? rdA.x : ((mu == 1) ? rdA.y : rdA.z);
                Dri[(3 * pat + mu) + p * max_na3] = um;
                Dri[(3 * i + mu) + p * max_na3] = -um;
                Dksi[(3 * pat + mu) + p * max_na3] = um * rdA.w;
                Dksi[(3 * i + mu) + p * max_na3] = -um * rdA.w;
            }
        }
        for (int amu = tid; amu < na3; amu += nthr) {
            for (int t = 0; t < FB; ++t) {
                const long long ia = (static_cast<long long>(p) * FB + t) * na3 + amu;
                const int colf = p * K + 2 * t;
                Af[amu + static_cast<long long>(colf) * max_na3] = gi.dcosp[ia];
                Af[amu + static_cast<long long>(colf + 1) * max_na3] = gi.dsinp[ia];
            }
        }
        __syncthreads();
    }
    (void)mol_byte_off1;
    (void)Alch;
}

// One block per (centre, target): fill wide X/Y/Z columns + gA/gB for that pair.
template <bool Alch, int Order>
__global__ void __launch_bounds__(128, 4) blas_pack_centre_b_kernel(
    const Centre *__restrict__ centres,
    int batch_c,
    const Target *__restrict__ targets,
    int n_targ,
    int same_mols,
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
    int max1,
    int max2,
    float inv,
    HotParams hp,
    const float *pdc,
    int max_na3,
    int max_nb3,
    int P,
    int FB,
    float *__restrict__ out_X,
    float *__restrict__ out_Y,
    float *__restrict__ out_Z,
    float *__restrict__ out_gA,
    float *__restrict__ out_gB
) {
    extern __shared__ float4 hot_smem4[];
    float *hot_smem = reinterpret_cast<float *>(hot_smem4);
    __shared__ float s_pdc[hess::kHessMaxPmax * hess::kHessMaxPmax];
    __shared__ float s_sij[1];

    const int flat = static_cast<int>(blockIdx.x);
    const int n_blocks = batch_c * n_targ;
    if (flat >= n_blocks) {
        return;
    }
    const int c = flat / n_targ;
    const int t_ix = flat - c * n_targ;
    const Centre ctr = centres[c];
    const Target tg = targets[t_ix];
    const int a = ctr.a, i = ctr.i;
    if (tg.b < 0 || (same_mols && tg.b > a)) {
        return;
    }
    const int b = tg.b, j = tg.j;
    const int na = n1[a], nb = n2[b];
    const int na3 = 3 * na, nb3 = 3 * nb;
    constexpr int order_c = Order;
    const int order = (order_c > 0) ? order_c : hp.order;
    const int pmax = hp.pmax;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);
    const float dscale = hp.dscale, ascale = hp.ascale;
    const float inv_width = hp.inv_width, maxgausdist2 = hp.maxgausdist2;
    const float s0 = hp.spf[0], s1 = hp.spf[1];
    const int K = 2 * FB, K1 = P * K;
    const int wide_n = max_nb3 * n_targ;
    const int max_nb = max_nb3 / 3;

    float4 *s_rad = hot_smem4;
    int *s_nbq = reinterpret_cast<int *>(hot_smem + 4 * max_nb);
    float2 *s_Z = reinterpret_cast<float2 *>(hot_smem + ((5 * max_nb + 1) & ~1));
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

    auto spf_of = [&](int t) -> float {
        if constexpr (order_c == 1) {
            return s0;
        } else if constexpr (order_c == 2) {
            return (t & 1) ? s1 : s0;
        }
        return hp.spf[t % order];
    };
    auto ang_val = [&](const Fock *fa, const Fock *fb) -> float {
        float ang = 0.0f;
        for (int t = 0; t < FB; ++t) {
            ang += (fa[t].x * fb[t].x + fa[t].y * fb[t].y) * spf_of(t);
        }
        return ang;
    };

    if constexpr (Alch) {
        if (tid < pmax * pmax) {
            s_pdc[tid] = pdc[tid];
        }
        __syncthreads();
    }

    const int flat_i = a * max1 + i;
    const int flat_j = b * max2 + j;
    const int nnA = nn1[flat_i];
    const int nnB = nn2[flat_j];
    const Neigh *ni = nullptr;
    const Fock *fi = nullptr;
    centre_views(blob1, mol_byte_off1, a, i, na, pmax, order, &ni, &fi);
    const Neigh *nj = nullptr;
    const Fock *fj = nullptr;
    centre_views(blob2, mol_byte_off2, b, j, nb, pmax, order, &nj, &fj);
    const Fock *fi_e = fi;
    const Fock *fj_e = fj;
    if constexpr (Alch) {
        fi_e = centre_fock_ptr(blob1_e, mol_byte_off1, a, i, na, pmax, order);
        fj_e = centre_fock_ptr(blob2_e, mol_byte_off2, b, j, nb, pmax, order);
    }
    const GradViewsC gi = grad_views(gblob1, mol_gbyte_off1, a, i, na, pmax, order);
    const GradViewsC gj = grad_views(gblob2, mol_gbyte_off2, b, j, nb, pmax, order);
    const float *dssA = dss1 + dss_off1[a] + static_cast<long long>(i) * na3;
    const float *dssB = dss2 + dss_off2[b] + static_cast<long long>(j) * nb3;

    float *Xw = out_X + static_cast<long long>(c) * P * wide_n;
    float *Yw = out_Y + static_cast<long long>(c) * P * wide_n;
    float *Zw = out_Z + static_cast<long long>(c) * K1 * wide_n;

    for (int e = tid; e < na3; e += nthr) {
        s_dsA[e] = 0.0f;
    }
    for (int e = tid; e < nb3; e += nthr) {
        s_dsB[e] = 0.0f;
    }
    for (int q = tid; q < nnB; q += nthr) {
        s_rad[q] = gj.rad[q];
        s_nbq[q] = gj.nbr[q];
    }

    float pd_central = 1.0f;
    bool pair_on;
    if constexpr (Alch) {
        pd_central = alch_pd(static_cast<const float *>(c_alchemy_pd), hp.emax, ni[0].z, nj[0].z);
        pair_on = (pd_central != 0.0f);
    } else {
        pair_on = (ni[0].z == nj[0].z);
    }
    if (tid == 0) {
        s_sij[0] = pair_on ? pd_central : 0.0f;
    }
    __syncthreads();

    if (pair_on) {
        float part = 0.0f;
        for (int p = 1 + tid; p < nnA; p += nthr) {
            const float rp = ni[p].rij, ksi_p = ni[p].ksi;
            const int zp = ni[p].z;
            const Fock *fip = fi + static_cast<long long>(p) * FB;
            for (int q = 1; q < nnB; ++q) {
                float pd_pq = pd_central;
                bool ok = true;
                if constexpr (Alch) {
                    pd_pq *=
                        alch_pd(static_cast<const float *>(c_alchemy_pd), hp.emax, zp, nj[q].z);
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
                    part += G * (ksi_p * nj[q].ksi * dscale +
                                 ang_val(fip, fj_e + static_cast<long long>(q) * FB) * ascale);
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
    const float coeff = kij * inv;
    const float coeff_hess = 2.0f * coeff;
    const float coeff_g = coeff * inv;

    if (pair_on) {
        for (int p = 1; p < nnA; ++p) {
            const float rp = ni[p].rij, ksi_p = ni[p].ksi;
            const int zp = ni[p].z;
            const float4 rdA = gi.rad[p];
            const int pat = gi.nbr[p];
            const Fock *fip = fi + static_cast<long long>(p) * FB;
            const Fock *fip_e = fi_e + static_cast<long long>(p) * FB;

            for (int q = 1 + tid; q < nnB; q += nthr) {
                float G = 0, dGp = 0, d2G = 0, V = 0, ksi_q = 0;
                float pd_pq = pd_central;
                bool ok = true;
                if constexpr (Alch) {
                    pd_pq *=
                        alch_pd(static_cast<const float *>(c_alchemy_pd), hp.emax, zp, nj[q].z);
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

            float S_dGpV = 0, S_Gksi = 0;
            int n_valid = 0;
            for (int q = 1; q < nnB; ++q) {
                if (s_G[q] == 0) {
                    continue;
                }
                ++n_valid;
                S_dGpV += s_dGp[q] * s_V[q];
                S_Gksi += s_G[q] * s_ksi[q];
            }
            if (n_valid == 0) {
                __syncthreads();
                continue;
            }

            for (int tt = tid; tt < FB; tt += nthr) {
                float2 acc = make_float2(0, 0);
                for (int q = 1; q < nnB; ++q) {
                    if (s_G[q] == 0) {
                        continue;
                    }
                    const float2 f = fj_e[static_cast<long long>(q) * FB + tt];
                    acc.x += s_G[q] * f.x;
                    acc.y += s_G[q] * f.y;
                }
                s_Fsum[tt] = acc;
            }
            for (int bnu = tid; bnu < nb3; bnu += nthr) {
                float x = 0, y = 0, dsb = 0;
                const int bat = bnu / 3, bmu = bnu - 3 * bat;
                for (int tt = 0; tt < FB; ++tt) {
                    s_Z[static_cast<long long>(tt) * max_nb3 + bnu] = make_float2(0, 0);
                }
                for (int q = 1; q < nnB; ++q) {
                    const float G = s_G[q];
                    if (G == 0) {
                        continue;
                    }
                    const float4 rd = s_rad[q];
                    const float um = (bmu == 0) ? rd.x : ((bmu == 1) ? rd.y : rd.z);
                    const float sgn = (bat == s_nbq[q]) ? 1.0f : ((bat == j) ? -1.0f : 0.0f);
                    const float dri = sgn * um;
                    const float dksi = dri * rd.w;
                    const float cq = -s_dGp[q] * dri;
                    float d = 0;
                    const Fock *fjq = fj + static_cast<long long>(q) * FB;
                    for (int tt = 0; tt < FB; ++tt) {
                        const long long ib = (static_cast<long long>(q) * FB + tt) * nb3 + bnu;
                        const float dc = gj.dcosp[ib], dsn = gj.dsinp[ib];
                        const float2 fa = fip_e[tt];
                        d += (fa.x * dc + fa.y * dsn) * spf_of(tt);
                        float2 *z = s_Z + static_cast<long long>(tt) * max_nb3 + bnu;
                        float2 zv = *z;
                        zv.x += G * dc + cq * fjq[tt].x;
                        zv.y += G * dsn + cq * fjq[tt].y;
                        *z = zv;
                    }
                    const float dVB = dksi * ksi_p * dscale + d * ascale;
                    x += s_dGp[q] * dVB + s_d2G[q] * dri * s_V[q];
                    y += cq * s_ksi[q] + G * dksi;
                    dsb += cq * s_V[q] + G * dVB;
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
                            float2 acc = make_float2(0, 0);
                            for (int p2 = 0; p2 < pmax; ++p2) {
                                const float w = s_pdc[p1 * pmax + p2];
                                acc.x += w * v[p2].x;
                                acc.y += w * v[p2].y;
                            }
                            s_Z[static_cast<long long>(p1 * order + m) * max_nb3 + bnu] = acc;
                        }
                    }
                }
                for (int tt = 0; tt < FB; ++tt) {
                    float2 *z = s_Z + static_cast<long long>(tt) * max_nb3 + bnu;
                    const float sc = ascale * spf_of(tt);
                    z->x *= sc;
                    z->y *= sc;
                }
            }
            __syncthreads();

            for (int amu = tid; amu < na3; amu += nthr) {
                float acc = 0;
                {
                    const int aat = amu / 3;
                    const float sg = (aat == pat) ? 1.0f : ((aat == i) ? -1.0f : 0.0f);
                    if (sg != 0) {
                        const int amo = amu - 3 * aat;
                        const float um = (amo == 0) ? rdA.x : ((amo == 1) ? rdA.y : rdA.z);
                        acc = sg * um * (S_dGpV + rdA.w * dscale * S_Gksi);
                    }
                }
                float ang = 0;
                for (int tt = 0; tt < FB; ++tt) {
                    const long long ia = (static_cast<long long>(p) * FB + tt) * na3 + amu;
                    ang += (gi.dcosp[ia] * s_Fsum[tt].x + gi.dsinp[ia] * s_Fsum[tt].y) *
                           spf_of(tt);
                }
                s_dsA[amu] += acc + ang * ascale;
            }

            for (int bnu = tid; bnu < nb3; bnu += nthr) {
                const int colw = t_ix * max_nb3 + bnu;
                Xw[p + static_cast<long long>(colw) * P] = s_X[bnu] * coeff_hess;
                Yw[p + static_cast<long long>(colw) * P] = s_Y[bnu] * coeff_hess;
                for (int tt = 0; tt < FB; ++tt) {
                    const float2 z = s_Z[static_cast<long long>(tt) * max_nb3 + bnu];
                    const int row0 = p * K + 2 * tt;
                    Zw[row0 + static_cast<long long>(colw) * K1] = z.x * coeff_hess;
                    Zw[row0 + 1 + static_cast<long long>(colw) * K1] = z.y * coeff_hess;
                }
            }
            __syncthreads();
        }
    }

    for (int amu = tid; amu < na3; amu += nthr) {
        out_gA[amu + static_cast<long long>(t_ix) * max_na3 +
               static_cast<long long>(c) * max_na3 * n_targ] =
            -dssA[amu] + 2.0f * s_dsA[amu];
    }
    for (int bnu = tid; bnu < nb3; bnu += nthr) {
        out_gB[bnu + static_cast<long long>(t_ix) * max_nb3 +
               static_cast<long long>(c) * max_nb3 * n_targ] =
            (-dssB[bnu] + 2.0f * s_dsB[bnu]) * coeff_g;
    }
}


__global__ void wide_scatter_kernel(
    const float *__restrict__ Hwide,  // batch_c * max_na3 * (max_nb3 * n_targ)
    const float *__restrict__ gA,     // batch_c * max_na3 * n_targ
    const float *__restrict__ gB,     // batch_c * max_nb3 * n_targ
    const Centre *__restrict__ centres,
    const Target *__restrict__ targets,
    const int *n1,
    const int *n2,
    const int *atom_base1,
    const int *atom_base2,
    float *__restrict__ H,
    long long ld_H,
    int batch_c,
    int n_targ,
    int max_na3,
    int max_nb3,
    int same_mols
) {
    const int c = static_cast<int>(blockIdx.x);
    if (c >= batch_c) {
        return;
    }
    const Centre ctr = centres[c];
    const int na3 = 3 * n1[ctr.a];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);
    const int wide_n = max_nb3 * n_targ;
    const float *tile = Hwide + static_cast<long long>(c) * max_na3 * wide_n;
    const float *ga = gA + static_cast<long long>(c) * max_na3 * n_targ;
    const float *gb = gB + static_cast<long long>(c) * max_nb3 * n_targ;

    for (int t = 0; t < n_targ; ++t) {
        const Target tg = targets[t];
        if (tg.b < 0) {
            continue;  // padded / inactive
        }
        if (same_mols && tg.b > ctr.a) {
            continue;
        }
        const int nb3 = 3 * n2[tg.b];
        float *Hblk = H + 3LL * atom_base1[ctr.a] * ld_H + 3LL * atom_base2[tg.b];
        for (int flat = tid; flat < na3 * nb3; flat += nthr) {
            const int amu = flat / nb3;
            const int bnu = flat - amu * nb3;
            const float ht =
                tile[amu + static_cast<long long>(t * max_nb3 + bnu) * max_na3];
            const float r1 = ga[amu + static_cast<long long>(t) * max_na3] *
                             gb[bnu + static_cast<long long>(t) * max_nb3];
            atomicAdd(Hblk + static_cast<long long>(amu) * ld_H + bnu, ht + r1);
        }
        __syncthreads();
    }
}

inline void apply_centre_wide(
    int batch_c,
    int n_targ,
    int max_na3,
    int max_nb3,
    int P,
    int K1,
    float *Dri,
    float *Dksi,
    float *X,
    float *Y,
    float *Afock,
    float *Z,
    float *Hwide,
    float *gA,
    float *gB,
    const Centre *centres,
    const Target *targets,
    const int *n1,
    const int *n2,
    const int *atom_base1,
    const int *atom_base2,
    float *H,
    long long ld_H,
    int same_mols
) {
    ensure_cublas();
    const float alpha = 1.0f, beta0 = 0.0f, beta1 = 1.0f;
    const int wide_n = max_nb3 * n_targ;
    const long long stride_Dri = static_cast<long long>(max_na3) * P;
    const long long stride_X = static_cast<long long>(P) * wide_n;
    const long long stride_H = static_cast<long long>(max_na3) * wide_n;
    const long long stride_Af = static_cast<long long>(max_na3) * K1;
    const long long stride_Z = static_cast<long long>(K1) * wide_n;

    // Hwide = Dri @ X_wide   (n = nb3*n_targ)
    KF_HESS_CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle(), CUBLAS_OP_N, CUBLAS_OP_N, max_na3, wide_n, P, &alpha, Dri, max_na3, stride_Dri,
        X, P, stride_X, &beta0, Hwide, max_na3, stride_H, batch_c
    ));
    KF_HESS_CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle(), CUBLAS_OP_N, CUBLAS_OP_N, max_na3, wide_n, P, &alpha, Dksi, max_na3, stride_Dri,
        Y, P, stride_X, &beta1, Hwide, max_na3, stride_H, batch_c
    ));
    if (K1 > 0) {
        KF_HESS_CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle(), CUBLAS_OP_N, CUBLAS_OP_N, max_na3, wide_n, K1, &alpha, Afock, max_na3,
            stride_Af, Z, K1, stride_Z, &beta1, Hwide, max_na3, stride_H, batch_c
        ));
    }

    wide_scatter_kernel<<<batch_c, 128>>>(
        Hwide, gA, gB, centres, targets, n1, n2, atom_base1, atom_base2, H, ld_H, batch_c, n_targ,
        max_na3, max_nb3, same_mols
    );
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace hess_blas

