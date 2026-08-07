// CUDA FCHL18 Hessian kernel — H[row, col] = d²K(A_a, B_b) / dR_A dR_B
//
// Port of kf::fchl18::kernel_gaussian_hessian_rect (src/fchl18_kernel.cpp).
// Both sides take a precomputed representation (like the CUDA Jacobian's query
// side) plus padded coords/z, which are needed to map neighbour slots back to
// atom indices for the chain rule.
//
// Layout notes
// ------------
// Fourier terms use the CPU layout [p][m][neigh]; gradients hang off that flat
// index as [fidx * na3 + alpha*3 + mu] with na3 = 3 * n[mol]. Buffers are
// strided by 3 * max_size so one allocation covers every molecule.
//
// The device helpers below are deliberate copies of their cuda_fchl18_jacobian.cu
// counterparts (which are file-local there); the two translation units stay
// independent.

#include "cuda_fchl18_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_fchl18_common.cuh"

namespace kf {
namespace fchl18 {

namespace {

constexpr int kHessPrecomputeBlockSize = 128;
constexpr int kHessSelfBlockSize = 128;
constexpr int kHessBlockSize = 128;
// Shared scratch for per-atom-pair gradients / dV vectors.
constexpr int kHessMaxNa3 = 384;  // up to 128 atoms per molecule
// Cap on shared Hij tile (float/double elements). Above this, Hij is
// accumulated with global atomics after coeff is known (two-pass).
constexpr int kHessSharedHMax = 4096;

// Flat index into the CPU-layout Fourier arrays: [p][m][neigh].
__device__ inline long long hess_fourier_idx(int p, int m, int neigh, int order, int max_size) {
    return (static_cast<long long>(p) * order + m) * max_size + neigh;
}

template <typename T>
struct HessParams {
    T two_body_width;
    T two_body_power;
    T three_body_power;
    T cut_start;
    T cut_distance;
    T true_distance_scale;
    T true_angular_scale;
    T inv_sigma2;  // +0.5 / sigma^2, matching the CPU convention
    int fourier_order;
    int pmax;
    int use_atm;
};

// ---------------------------------------------------------------------------
// Per-atom precompute (values + gradients), needed on both sides
// ---------------------------------------------------------------------------

// Map neighbour slot k of atom `centre` back to its atom index by matching the
// stored displacement and nuclear charge. The CPU uses an exact 1e-10 window; a
// nearest-match search behaves identically in fp64 and stays robust in fp32.
template <typename T>
__device__ void hess_build_nbr_atom_idx(
    const T *atom_chan,
    int max_size,
    int n_neigh,
    const T *coords,
    const int *z_mol,
    int centre_idx,
    int n_atoms,
    int *nbr_out
) {
    for (int k = 0; k < max_size; ++k) {
        nbr_out[k] = -1;
    }
    nbr_out[0] = centre_idx;

    const T *z_chan = atom_chan + max_size;
    const T *xc = atom_chan + 2 * max_size;
    const T *yc = atom_chan + 3 * max_size;
    const T *zc = atom_chan + 4 * max_size;

    const T cx = coords[centre_idx * 3 + 0];
    const T cy = coords[centre_idx * 3 + 1];
    const T cz = coords[centre_idx * 3 + 2];

    for (int k = 1; k < n_neigh; ++k) {
        const T dx = xc[k];
        const T dy = yc[k];
        const T dz = zc[k];
        const int zval = static_cast<int>(z_chan[k]);

        int best = -1;
        T best_d2 = Math<T>::zero();
        for (int j = 0; j < n_atoms; ++j) {
            if (z_mol[j] != zval) {
                continue;
            }
            const T ex = coords[j * 3 + 0] - cx - dx;
            const T ey = coords[j * 3 + 1] - cy - dy;
            const T ez = coords[j * 3 + 2] - cz - dz;
            const T d2 = ex * ex + ey * ey + ez * ez;
            if (best < 0 || d2 < best_d2) {
                best_d2 = d2;
                best = j;
            }
        }
        nbr_out[k] = best;
    }
}

// ksi[k] = f_cut(r_k) / r_k^power, plus d(ksi)/dR and d(r_k)/dR.
// dksi/dri must be zero-initialised by the caller.
template <typename T>
__device__ void hess_compute_ksi_and_grad(
    const T *atom_chan,
    int max_size,
    int n_neigh,
    T power,
    T cut_start,
    T cut_distance,
    int centre_atom_idx,
    int na3,
    const int *nbr_atom_idx,
    T *ksi,
    T *dksi,
    T *dri
) {
    for (int k = 0; k < max_size; ++k) {
        ksi[k] = Math<T>::zero();
    }

    const T *dist_chan = atom_chan;
    const T *xc = atom_chan + 2 * max_size;
    const T *yc = atom_chan + 3 * max_size;
    const T *zc = atom_chan + 4 * max_size;

    for (int k = 1; k < n_neigh; ++k) {
        const T r = dist_chan[k];
        if (r < Math<T>::eps()) {
            continue;
        }
        const T fc = cut_function(r, cut_start, cut_distance);
        const T rp = fast_pow(r, power);
        ksi[k] = fc / rp;

        const T fcp = cut_function_deriv(r, cut_start, cut_distance);
        const T dksi_dr = (fcp - fc * power / r) / rp;

        const int k_atom = nbr_atom_idx[k];
        if (k_atom < 0) {
            continue;
        }

        const T dxk = xc[k];
        const T dyk = yc[k];
        const T dzk = zc[k];
        const T inv_r = Math<T>::one() / r;

        const long long base_k = static_cast<long long>(k) * na3 + k_atom * 3;
        const long long base_c = static_cast<long long>(k) * na3 + centre_atom_idx * 3;

        dksi[base_k + 0] += dksi_dr * dxk * inv_r;
        dksi[base_k + 1] += dksi_dr * dyk * inv_r;
        dksi[base_k + 2] += dksi_dr * dzk * inv_r;
        dksi[base_c + 0] -= dksi_dr * dxk * inv_r;
        dksi[base_c + 1] -= dksi_dr * dyk * inv_r;
        dksi[base_c + 2] -= dksi_dr * dzk * inv_r;

        dri[base_k + 0] = dxk * inv_r;
        dri[base_k + 1] = dyk * inv_r;
        dri[base_k + 2] = dzk * inv_r;
        dri[base_c + 0] = -dxk * inv_r;
        dri[base_c + 1] = -dyk * inv_r;
        dri[base_c + 2] = -dzk * inv_r;
    }
}

// Three-body Fourier terms and their gradients for one atom.
// cosp/sinp/dcosp/dsinp must be zero-initialised by the caller.
template <typename T>
__device__ void hess_compute_threebody_fourier_and_grad(
    const T *atom_chan,
    int max_size,
    int n_neigh,
    T three_body_power,
    T cut_start,
    T cut_distance,
    int order,
    const int *z_to_idx,
    bool use_atm,
    int centre_atom_idx,
    int na3,
    const int *nbr_atom_idx,
    T *cosp,
    T *sinp,
    T *dcosp,
    T *dsinp
) {
    const T zero = Math<T>::zero();
    const T one = Math<T>::one();

    const T *dist_chan = atom_chan;
    const T *z_chan = atom_chan + max_size;
    const T *xc = atom_chan + 2 * max_size;
    const T *yc = atom_chan + 3 * max_size;
    const T *zc = atom_chan + 4 * max_size;

    for (int j = 1; j < n_neigh; ++j) {
        const T dj = dist_chan[j];
        if (dj < Math<T>::eps()) {
            continue;
        }
        const T cutj = cut_function(dj, cut_start, cut_distance);
        if (cutj == zero) {
            continue;
        }
        const T inv_dj = one / dj;
        const T ui_j[3] = {xc[j] * inv_dj, yc[j] * inv_dj, zc[j] * inv_dj};

        for (int k = j + 1; k < n_neigh; ++k) {
            const T dk = dist_chan[k];
            if (dk < Math<T>::eps()) {
                continue;
            }
            const T cutk = cut_function(dk, cut_start, cut_distance);
            if (cutk == zero) {
                continue;
            }

            const T dxjk = xc[j] - xc[k];
            const T dyjk = yc[j] - yc[k];
            const T dzjk = zc[j] - zc[k];
            const T di2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
            const T di = Math<T>::sqrt_(di2);
            if (di < Math<T>::eps()) {
                continue;
            }
            const T inv_di = one / di;

            const T cut_jk = cut_function(di, cut_start, cut_distance);
            if (cut_jk == zero) {
                continue;
            }

            const T inv_dk = one / dk;
            const T ui_k[3] = {xc[k] * inv_dk, yc[k] * inv_dk, zc[k] * inv_dk};

            const T cos_i = clamp11(ui_j[0] * ui_k[0] + ui_j[1] * ui_k[1] + ui_j[2] * ui_k[2]);
            const T theta = Math<T>::acos_(cos_i);
            const T sin_theta = Math<T>::sin_(theta);

            // Unit vectors along the j-k edge, in both directions.
            const T ukj[3] = {-dxjk * inv_di, -dyjk * inv_di, -dzjk * inv_di};
            const T ujk[3] = {dxjk * inv_di, dyjk * inv_di, dzjk * inv_di};

            T atm = one;
            T datm_dcos_i = zero;
            T datm_dcos_j = zero;
            T datm_dcos_k = zero;
            T cos_j = zero;
            T cos_k_val = zero;
            if (use_atm) {
                cos_j = clamp11(ukj[0] * (-ui_j[0]) + ukj[1] * (-ui_j[1]) + ukj[2] * (-ui_j[2]));
                cos_k_val =
                    clamp11(ujk[0] * (-ui_k[0]) + ujk[1] * (-ui_k[1]) + ujk[2] * (-ui_k[2]));
                atm = one + T(3) * cos_i * cos_j * cos_k_val;
                datm_dcos_i = T(3) * cos_j * cos_k_val;
                datm_dcos_j = T(3) * cos_i * cos_k_val;
                datm_dcos_k = T(3) * cos_i * cos_j;
            }
            if (atm == zero) {
                continue;
            }

            const T dijk = dj * dk * di;
            const T dijk_p = fast_pow(dijk, three_body_power);
            const T cut_prod = cutj * cutk * cut_jk;
            const T ksi3 = cut_prod * atm / dijk_p;
            if (ksi3 == zero) {
                continue;
            }

            const int j_atom = nbr_atom_idx[j];
            const int k_atom = nbr_atom_idx[k];
            if (j_atom < 0 || k_atom < 0) {
                continue;
            }

            const T fcp_j = cut_function_deriv(dj, cut_start, cut_distance);
            const T fcp_k = cut_function_deriv(dk, cut_start, cut_distance);
            const T fcp_jk = cut_function_deriv(di, cut_start, cut_distance);

            // Local atom order: 0 = centre, 1 = j_atom, 2 = k_atom.
            T dcos_i_dR[3][3];
            for (int mu = 0; mu < 3; ++mu) {
                dcos_i_dR[1][mu] = (ui_k[mu] - cos_i * ui_j[mu]) * inv_dj;
                dcos_i_dR[2][mu] = (ui_j[mu] - cos_i * ui_k[mu]) * inv_dk;
                dcos_i_dR[0][mu] = -dcos_i_dR[1][mu] - dcos_i_dR[2][mu];
            }

            T dcos_j_dR[3][3] = {};
            T dcos_k_dR[3][3] = {};
            if (use_atm) {
                for (int mu = 0; mu < 3; ++mu) {
                    dcos_j_dR[0][mu] = inv_dj * (ukj[mu] + cos_j * ui_j[mu]);
                    dcos_j_dR[1][mu] = inv_di * (ui_j[mu] + cos_j * ukj[mu]) -
                                       inv_dj * (ukj[mu] + cos_j * ui_j[mu]);
                    dcos_j_dR[2][mu] = -inv_di * (ui_j[mu] + cos_j * ukj[mu]);

                    dcos_k_dR[0][mu] = inv_dk * (ujk[mu] + cos_k_val * ui_k[mu]);
                    dcos_k_dR[1][mu] = -inv_di * (ui_k[mu] + cos_k_val * ujk[mu]);
                    dcos_k_dR[2][mu] = inv_di * (ui_k[mu] + cos_k_val * ujk[mu]) -
                                       inv_dk * (ujk[mu] + cos_k_val * ui_k[mu]);
                }
            }

            const T fcp_j_over_p = (cutj > zero) ? (fcp_j * cutk * cut_jk) / dijk_p : zero;
            const T fcp_k_over_p = (cutk > zero) ? (fcp_k * cutj * cut_jk) / dijk_p : zero;
            const T fcp_jk_over_p = (cut_jk > zero) ? (fcp_jk * cutj * cutk) / dijk_p : zero;
            const T beta3_ksi3_inv = three_body_power / dijk_p * cut_prod;

            T gksi3[3][3] = {};
            for (int mu = 0; mu < 3; ++mu) {
                gksi3[0][mu] += atm * (-ui_j[mu] * fcp_j_over_p + -ui_k[mu] * fcp_k_over_p);
                gksi3[1][mu] += atm * (ui_j[mu] * fcp_j_over_p + (-ukj[mu]) * fcp_jk_over_p);
                gksi3[2][mu] += atm * (ui_k[mu] * fcp_k_over_p + (ukj[mu]) * fcp_jk_over_p);

                gksi3[0][mu] -= atm * beta3_ksi3_inv * (-ui_j[mu] * inv_dj + -ui_k[mu] * inv_dk);
                gksi3[1][mu] -= atm * beta3_ksi3_inv * (ui_j[mu] * inv_dj + (-ukj[mu]) * inv_di);
                gksi3[2][mu] -= atm * beta3_ksi3_inv * (ui_k[mu] * inv_dk + (ukj[mu]) * inv_di);

                if (use_atm) {
                    const T scale = cut_prod / dijk_p;
                    for (int la = 0; la < 3; ++la) {
                        gksi3[la][mu] += scale * (datm_dcos_i * dcos_i_dR[la][mu] +
                                                  datm_dcos_j * dcos_j_dR[la][mu] +
                                                  datm_dcos_k * dcos_k_dR[la][mu]);
                    }
                }
            }

            const int zk = static_cast<int>(z_chan[k]);
            const int zj = static_cast<int>(z_chan[j]);
            if (zk <= 0 || zk >= 256 || zj <= 0 || zj >= 256) {
                continue;
            }
            const int pj = z_to_idx[zk];
            const int pk = z_to_idx[zj];
            if (pj < 0 || pk < 0) {
                continue;
            }

            const int global_atoms[3] = {centre_atom_idx, j_atom, k_atom};

            // Only odd n = m+1 contribute: cos(n*theta) - cos(n*(theta+pi)) is
            // 2*cos(n*theta) for odd n and identically zero for even n.
            for (int m = 0; m < order; m += 2) {
                const T mf = static_cast<T>(m + 1);
                const T mth = mf * theta;
                T c, s;
                Math<T>::sincos_(mth, &s, &c);

                const T cos_mfac = T(2) * c;
                const T sin_mfac = T(2) * s;
                const T cos_m = cos_mfac * ksi3;
                const T sin_m = sin_mfac * ksi3;
                const T dcos_mfac_dtheta = T(-2) * mf * s;
                const T dsin_mfac_dtheta = T(2) * mf * c;

                const long long fidx_j = hess_fourier_idx(pj, m, j, order, max_size);
                const long long fidx_k = hess_fourier_idx(pk, m, k, order, max_size);

                cosp[fidx_j] += cos_m;
                sinp[fidx_j] += sin_m;
                cosp[fidx_k] += cos_m;
                sinp[fidx_k] += sin_m;

                for (int la = 0; la < 3; ++la) {
                    const int alpha = global_atoms[la];
                    for (int mu = 0; mu < 3; ++mu) {
                        T dtheta_dR = zero;
                        if (sin_theta > T(1e-10)) {
                            dtheta_dR = (-one / sin_theta) * dcos_i_dR[la][mu];
                        }
                        const T dcos_m =
                            dcos_mfac_dtheta * dtheta_dR * ksi3 + cos_mfac * gksi3[la][mu];
                        const T dsin_m =
                            dsin_mfac_dtheta * dtheta_dR * ksi3 + sin_mfac * gksi3[la][mu];

                        const long long oj = fidx_j * na3 + alpha * 3 + mu;
                        const long long ok = fidx_k * na3 + alpha * 3 + mu;
                        dcosp[oj] += dcos_m;
                        dsinp[oj] += dsin_m;
                        dcosp[ok] += dcos_m;
                        dsinp[ok] += dsin_m;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar product s(i, j) and its gradient w.r.t. one coordinate component of
// the first side. Pass amu < 0 (or ds_out == nullptr) for a value-only call.
// ---------------------------------------------------------------------------
template <typename T>
__device__ T hess_scalar_and_grad_amu(
    const T *x1_chan,
    int max_size1,
    int n_neigh1,
    const T *ksi1,
    const T *dksi1,
    const T *dri1,
    const T *cos1,
    const T *sin1,
    const T *dcos1,
    const T *dsin1,
    const T *x2_chan,
    int max_size2,
    int n_neigh2,
    const T *ksi2,
    const T *cos2,
    const T *sin2,
    const HessParams<T> &params,
    const T *s_prefactor,
    int na3,
    int amu,
    T *ds_out
) {
    const bool want_grad = (amu >= 0 && amu < na3 && ds_out != nullptr);
    if (want_grad) {
        *ds_out = Math<T>::zero();
    }

    const int Z1 = static_cast<int>(x1_chan[max_size1]);
    const int Z2 = static_cast<int>(x2_chan[max_size2]);
    if (Z1 != Z2) {
        return Math<T>::zero();
    }

    const T d_width = params.two_body_width;
    const T inv_width = -Math<T>::one() / (T(4) * d_width * d_width);
    const T maxgausdist2 = (T(8) * d_width) * (T(8) * d_width);
    const T dist_scale = params.true_distance_scale;
    const T ang_scale = params.true_angular_scale;
    const int order = params.fourier_order;
    const int pmax = params.pmax;

    T aadist = Math<T>::one();
    T ds = Math<T>::zero();

    for (int i = 1; i < n_neigh1; ++i) {
        const int Zi = static_cast<int>(x1_chan[max_size1 + i]);
        const T ri = x1_chan[i];
        const long long i_na3 = static_cast<long long>(i) * na3;

        for (int j = 1; j < n_neigh2; ++j) {
            if (Zi != static_cast<int>(x2_chan[max_size2 + j])) {
                continue;
            }

            const T dr = ri - x2_chan[j];
            const T r2 = dr * dr;
            if (r2 >= maxgausdist2) {
                continue;
            }

            const T G = Math<T>::exp_(r2 * inv_width);
            const T dG_dri = G * T(2) * dr * inv_width;

            T angular = Math<T>::zero();
            T d_angular = Math<T>::zero();
            for (int m = 0; m < order; m += 2) {
                const T spf = s_prefactor[m];
                T ang_m = Math<T>::zero();
                T dang_m = Math<T>::zero();
                for (int p = 0; p < pmax; ++p) {
                    const long long i1 = hess_fourier_idx(p, m, i, order, max_size1);
                    const long long i2 = hess_fourier_idx(p, m, j, order, max_size2);
                    const T c2 = cos2[i2];
                    const T s2 = sin2[i2];
                    ang_m += cos1[i1] * c2 + sin1[i1] * s2;
                    if (want_grad) {
                        dang_m += dcos1[i1 * na3 + amu] * c2 + dsin1[i1 * na3 + amu] * s2;
                    }
                }
                angular += ang_m * spf;
                d_angular += dang_m * spf;
            }

            const T ksi2_j = ksi2[j];
            const T W = ksi1[i] * ksi2_j * dist_scale + angular * ang_scale;
            aadist += G * W;

            if (want_grad) {
                ds += dG_dri * dri1[i_na3 + amu] * W +
                      G * dist_scale * dksi1[i_na3 + amu] * ksi2_j + G * ang_scale * d_angular;
            }
        }
    }

    if (want_grad) {
        *ds_out = ds;
    }
    return aadist;
}

// ---------------------------------------------------------------------------
// Hot path: one block per (molecule-pair, atom-i, atom-j).
//
// For each matching neighbour pair (p, q) the block:
//   1. Computes G / V / angular once
//   2. Fills dV/dR_A and dV/dR_B vectors in shared memory
//   3. Accumulates scalar-product gradients dsij/dR into shared
//   4. Accumulates Hij via outer products (no per-(amu,bnu) neighbour rescan)
// Then applies the Gaussian kernel chain rule and atomicAdds into hess_out.
//
// Dynamic shared layout (sized on the host from max_size1/2):
//   [dsA | dsB | dVA | dVB | H(optional) | sij]
// launch_bounds(_, 8) targets >=8 blocks/SM to raise occupancy (NCU: 128 regs).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void __launch_bounds__(kHessBlockSize, 8) hess_hot_kernel(
    const T *x1,
    const T *x2,
    const int *n1,
    const int *n2,
    const int *nn1,
    const int *nn2,
    const int *row_offset,
    const int *col_offset,
    const T *ksi1,
    const T *dksi1,
    const T *dri1,
    const T *cosp1,
    const T *sinp1,
    const T *dcosp1,
    const T *dsinp1,
    const T *ss1,
    const T *dss1,
    const T *ksi2,
    const T *dksi2,
    const T *dri2,
    const T *cosp2,
    const T *sinp2,
    const T *dcosp2,
    const T *dsinp2,
    const T *ss2,
    const T *dss2,
    T *hess_out,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    long long d_b_cols,
    int lower_triangle,
    HessParams<T> params,
    const T *s_prefactor
) {
    extern __shared__ unsigned char hess_smem[];
    const int strideA = 3 * max_size1;
    const int strideB = 3 * max_size2;
    const int h_cap = (strideA * strideB <= kHessSharedHMax) ? (strideA * strideB) : 0;
    T *s_dsA = reinterpret_cast<T *>(hess_smem);
    T *s_dsB = s_dsA + strideA;
    T *s_dVA = s_dsB + strideB;
    T *s_dVB = s_dVA + strideA;
    T *s_H = s_dVB + strideB;
    T *s_sij = s_H + h_cap;

    const int n_mol_pairs =
        lower_triangle ? (nm1 * (nm1 + 1) / 2) : (nm1 * nm2);
    const int n_ij_slots = max_size1 * max_size2;
    const int flat_block = static_cast<int>(blockIdx.x);
    if (flat_block >= n_mol_pairs * n_ij_slots) {
        return;
    }

    const int mol_pair = flat_block / n_ij_slots;
    const int ij = flat_block - mol_pair * n_ij_slots;
    int a;
    int b;
    if (lower_triangle) {
        // mol_pair indexes lower triangle: a*(a+1)/2 + b with 0 <= b <= a.
        const double s = sqrt(1.0 + 8.0 * static_cast<double>(mol_pair));
        a = static_cast<int>((s - 1.0) * 0.5);
        while (a * (a + 1) / 2 > mol_pair) {
            --a;
        }
        while ((a + 1) * (a + 2) / 2 <= mol_pair) {
            ++a;
        }
        b = mol_pair - a * (a + 1) / 2;
    } else {
        a = mol_pair / nm2;
        b = mol_pair - a * nm2;
    }
    const int i = ij / max_size2;
    const int j = ij - i * max_size2;

    const int na = n1[a];
    const int nb = n2[b];
    if (i >= na || j >= nb) {
        return;
    }

    const int na3A = na * 3;
    const int na3B = nb * 3;
    if (na3A > kHessMaxNa3 || na3B > kHessMaxNa3) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int n_entries = na3A * na3B;
    const bool use_sh_H = (h_cap > 0) && (n_entries <= h_cap);

    const long long na3_max1 = 3LL * max_size1;
    const long long na3_max2 = 3LL * max_size2;
    const long long fstride1 = fourier_atom_stride(params.pmax, params.fourier_order, max_size1);
    const long long fstride2 = fourier_atom_stride(params.pmax, params.fourier_order, max_size2);
    const long long row0 = row_offset[a];
    const long long col0 = col_offset[b];

    const int fi = a * max_size1 + i;
    const int fj = b * max_size2 + j;
    const T *x1_chan = atom_slice(x1, max_size1, a, i);
    const T *x2_chan = atom_slice(x2, max_size2, b, j);
    const int n_neigh_i = nn1[fi];
    const int n_neigh_j = nn2[fj];
    const int Zi = static_cast<int>(x1_chan[max_size1]);
    const int Zj = static_cast<int>(x2_chan[max_size2]);
    const T sii = ss1[fi];
    const T sjj = ss2[fj];
    const T *dsii = dss1 + static_cast<long long>(fi) * na3_max1;
    const T *dsjj = dss2 + static_cast<long long>(fj) * na3_max2;
    const T *ksi_i = ksi1 + static_cast<long long>(fi) * max_size1;
    const T *dksi_i = dksi1 + static_cast<long long>(fi) * max_size1 * na3_max1;
    const T *dri_i = dri1 + static_cast<long long>(fi) * max_size1 * na3_max1;
    const T *cos_i = cosp1 + static_cast<long long>(fi) * fstride1;
    const T *sin_i = sinp1 + static_cast<long long>(fi) * fstride1;
    const T *dcos_i = dcosp1 + static_cast<long long>(fi) * fstride1 * na3_max1;
    const T *dsin_i = dsinp1 + static_cast<long long>(fi) * fstride1 * na3_max1;
    const T *ksi_j = ksi2 + static_cast<long long>(fj) * max_size2;
    const T *dksi_j = dksi2 + static_cast<long long>(fj) * max_size2 * na3_max2;
    const T *dri_j = dri2 + static_cast<long long>(fj) * max_size2 * na3_max2;
    const T *cos_j = cosp2 + static_cast<long long>(fj) * fstride2;
    const T *sin_j = sinp2 + static_cast<long long>(fj) * fstride2;
    const T *dcos_j = dcosp2 + static_cast<long long>(fj) * fstride2 * na3_max2;
    const T *dsin_j = dsinp2 + static_cast<long long>(fj) * fstride2 * na3_max2;

    for (int amu = tid; amu < na3A; amu += nthreads) {
        s_dsA[amu] = Math<T>::zero();
    }
    for (int bnu = tid; bnu < na3B; bnu += nthreads) {
        s_dsB[bnu] = Math<T>::zero();
    }
    if (use_sh_H) {
        for (int flat = tid; flat < n_entries; flat += nthreads) {
            s_H[flat] = Math<T>::zero();
        }
    }
    if (tid == 0) {
        s_sij[0] = (Zi == Zj) ? Math<T>::one() : Math<T>::zero();
    }
    __syncthreads();

    const T d_width = params.two_body_width;
    const T inv_width = -Math<T>::one() / (T(4) * d_width * d_width);
    const T maxgausdist2 = (T(8) * d_width) * (T(8) * d_width);
    const T dist_scale = params.true_distance_scale;
    const T ang_scale = params.true_angular_scale;
    const int order = params.fourier_order;
    const int pmax = params.pmax;
    const bool order1 = (order == 1);
    const T s0 = s_prefactor[0];

    if (Zi == Zj) {
        for (int p = 1; p < n_neigh_i; ++p) {
            const int Zp = static_cast<int>(x1_chan[max_size1 + p]);
            const T rp = x1_chan[p];
            const long long p_na3A = static_cast<long long>(p) * na3A;
            const T ksi_p = ksi_i[p];

            for (int q = 1; q < n_neigh_j; ++q) {
                if (Zp != static_cast<int>(x2_chan[max_size2 + q])) {
                    continue;
                }
                const T rq = x2_chan[q];
                const T dr = rp - rq;
                const T r2 = dr * dr;
                if (r2 >= maxgausdist2) {
                    continue;
                }

                const T G = Math<T>::exp_(r2 * inv_width);
                const T dG_drp = G * T(2) * dr * inv_width;
                const T dG_drq = -dG_drp;
                const T d2G_drpq =
                    G * T(2) * inv_width * (T(-2) * inv_width * r2 - Math<T>::one());
                const long long q_na3B = static_cast<long long>(q) * na3B;
                const T ksi_q = ksi_j[q];

                T angular = Math<T>::zero();
                if (order1) {
                    for (int pp = 0; pp < pmax; ++pp) {
                        const long long fiA = hess_fourier_idx(pp, 0, p, order, max_size1);
                        const long long fiB = hess_fourier_idx(pp, 0, q, order, max_size2);
                        angular += cos_i[fiA] * cos_j[fiB] + sin_i[fiA] * sin_j[fiB];
                    }
                    angular *= s0;
                } else {
                    for (int m = 0; m < order; m += 2) {
                        const T spf = s_prefactor[m];
                        T ang_m = Math<T>::zero();
                        for (int pp = 0; pp < pmax; ++pp) {
                            const long long fiA = hess_fourier_idx(pp, m, p, order, max_size1);
                            const long long fiB = hess_fourier_idx(pp, m, q, order, max_size2);
                            ang_m += cos_i[fiA] * cos_j[fiB] + sin_i[fiA] * sin_j[fiB];
                        }
                        angular += ang_m * spf;
                    }
                }
                const T V = ksi_p * ksi_q * dist_scale + angular * ang_scale;
                if (tid == 0) {
                    s_sij[0] += G * V;
                }

                for (int amu = tid; amu < na3A; amu += nthreads) {
                    T dang_A = Math<T>::zero();
                    if (order1) {
                        for (int pp = 0; pp < pmax; ++pp) {
                            const long long fiA = hess_fourier_idx(pp, 0, p, order, max_size1);
                            const long long fiB = hess_fourier_idx(pp, 0, q, order, max_size2);
                            dang_A += dcos_i[fiA * na3A + amu] * cos_j[fiB] +
                                      dsin_i[fiA * na3A + amu] * sin_j[fiB];
                        }
                        dang_A *= s0;
                    } else {
                        for (int m = 0; m < order; m += 2) {
                            const T spf = s_prefactor[m];
                            T dam = Math<T>::zero();
                            for (int pp = 0; pp < pmax; ++pp) {
                                const long long fiA = hess_fourier_idx(pp, m, p, order, max_size1);
                                const long long fiB = hess_fourier_idx(pp, m, q, order, max_size2);
                                dam += dcos_i[fiA * na3A + amu] * cos_j[fiB] +
                                       dsin_i[fiA * na3A + amu] * sin_j[fiB];
                            }
                            dang_A += dam * spf;
                        }
                    }
                    const T dksi_A = dksi_i[p_na3A + amu];
                    const T dri_A = dri_i[p_na3A + amu];
                    const T dV_A = dksi_A * ksi_q * dist_scale + dang_A * ang_scale;
                    s_dVA[amu] = dV_A;
                    s_dsA[amu] += dG_drp * dri_A * V + G * dV_A;
                }

                for (int bnu = tid; bnu < na3B; bnu += nthreads) {
                    T dang_B = Math<T>::zero();
                    if (order1) {
                        for (int pp = 0; pp < pmax; ++pp) {
                            const long long fiA = hess_fourier_idx(pp, 0, p, order, max_size1);
                            const long long fiB = hess_fourier_idx(pp, 0, q, order, max_size2);
                            dang_B += dcos_j[fiB * na3B + bnu] * cos_i[fiA] +
                                      dsin_j[fiB * na3B + bnu] * sin_i[fiA];
                        }
                        dang_B *= s0;
                    } else {
                        for (int m = 0; m < order; m += 2) {
                            const T spf = s_prefactor[m];
                            T dam = Math<T>::zero();
                            for (int pp = 0; pp < pmax; ++pp) {
                                const long long fiA = hess_fourier_idx(pp, m, p, order, max_size1);
                                const long long fiB = hess_fourier_idx(pp, m, q, order, max_size2);
                                dam += dcos_j[fiB * na3B + bnu] * cos_i[fiA] +
                                       dsin_j[fiB * na3B + bnu] * sin_i[fiA];
                            }
                            dang_B += dam * spf;
                        }
                    }
                    const T dksi_B = dksi_j[q_na3B + bnu];
                    const T dri_B = dri_j[q_na3B + bnu];
                    const T dV_B = dksi_B * ksi_p * dist_scale + dang_B * ang_scale;
                    s_dVB[bnu] = dV_B;
                    s_dsB[bnu] += dG_drq * dri_B * V + G * dV_B;
                }
                __syncthreads();

                if (use_sh_H) {
                    for (int flat = tid; flat < n_entries; flat += nthreads) {
                        const int amu = flat / na3B;
                        const int bnu = flat - amu * na3B;
                        const T dri_A = dri_i[p_na3A + amu];
                        const T dri_B = dri_j[q_na3B + bnu];
                        const T dksi_A = dksi_i[p_na3A + amu];
                        const T dksi_B = dksi_j[q_na3B + bnu];

                        T dang_AB = Math<T>::zero();
                        if (order1) {
                            for (int pp = 0; pp < pmax; ++pp) {
                                const long long fiA = hess_fourier_idx(pp, 0, p, order, max_size1);
                                const long long fiB = hess_fourier_idx(pp, 0, q, order, max_size2);
                                dang_AB += dcos_i[fiA * na3A + amu] * dcos_j[fiB * na3B + bnu] +
                                           dsin_i[fiA * na3A + amu] * dsin_j[fiB * na3B + bnu];
                            }
                            dang_AB *= s0;
                        } else {
                            for (int m = 0; m < order; m += 2) {
                                const T spf = s_prefactor[m];
                                T dam = Math<T>::zero();
                                for (int pp = 0; pp < pmax; ++pp) {
                                    const long long fiA =
                                        hess_fourier_idx(pp, m, p, order, max_size1);
                                    const long long fiB =
                                        hess_fourier_idx(pp, m, q, order, max_size2);
                                    dam += dcos_i[fiA * na3A + amu] * dcos_j[fiB * na3B + bnu] +
                                           dsin_i[fiA * na3A + amu] * dsin_j[fiB * na3B + bnu];
                                }
                                dang_AB += dam * spf;
                            }
                        }
                        const T d2V = dksi_A * dksi_B * dist_scale + dang_AB * ang_scale;
                        s_H[flat] += dG_drp * dri_A * s_dVB[bnu] + dG_drq * dri_B * s_dVA[amu] +
                                     d2G_drpq * dri_A * dri_B * V + G * d2V;
                    }
                    __syncthreads();
                }
            }
        }
    }
    __syncthreads();

    const T kij = Math<T>::exp_(-(sii + sjj - T(2) * s_sij[0]) * params.inv_sigma2);
    const T coeff = kij * params.inv_sigma2;
    const T coeff_hess = coeff * T(2);
    const T coeff_g = coeff * params.inv_sigma2;

    if (use_sh_H) {
        for (int flat = tid; flat < n_entries; flat += nthreads) {
            const int amu = flat / na3B;
            const int bnu = flat - amu * na3B;
            const T gA = -dsii[amu] + T(2) * s_dsA[amu];
            const T gB = -dsjj[bnu] + T(2) * s_dsB[bnu];
            atomicAdd(
                hess_out + (row0 + amu) * d_b_cols + (col0 + bnu),
                coeff_g * gA * gB + coeff_hess * s_H[flat]
            );
        }
        return;
    }

    for (int flat = tid; flat < n_entries; flat += nthreads) {
        const int amu = flat / na3B;
        const int bnu = flat - amu * na3B;
        const T gA = -dsii[amu] + T(2) * s_dsA[amu];
        const T gB = -dsjj[bnu] + T(2) * s_dsB[bnu];
        atomicAdd(hess_out + (row0 + amu) * d_b_cols + (col0 + bnu), coeff_g * gA * gB);
    }
    __syncthreads();

    if (Zi != Zj) {
        return;
    }

    // Large-molecule Hij fallback (rare for typical Hessian mols).
    for (int p = 1; p < n_neigh_i; ++p) {
        const int Zp = static_cast<int>(x1_chan[max_size1 + p]);
        const T rp = x1_chan[p];
        const long long p_na3A = static_cast<long long>(p) * na3A;
        const T ksi_p = ksi_i[p];

        for (int q = 1; q < n_neigh_j; ++q) {
            if (Zp != static_cast<int>(x2_chan[max_size2 + q])) {
                continue;
            }
            const T rq = x2_chan[q];
            const T dr = rp - rq;
            const T r2 = dr * dr;
            if (r2 >= maxgausdist2) {
                continue;
            }

            const T G = Math<T>::exp_(r2 * inv_width);
            const T dG_drp = G * T(2) * dr * inv_width;
            const T dG_drq = -dG_drp;
            const T d2G_drpq =
                G * T(2) * inv_width * (T(-2) * inv_width * r2 - Math<T>::one());
            const long long q_na3B = static_cast<long long>(q) * na3B;
            const T ksi_q = ksi_j[q];

            T angular = Math<T>::zero();
            if (order1) {
                for (int pp = 0; pp < pmax; ++pp) {
                    const long long fiA = hess_fourier_idx(pp, 0, p, order, max_size1);
                    const long long fiB = hess_fourier_idx(pp, 0, q, order, max_size2);
                    angular += cos_i[fiA] * cos_j[fiB] + sin_i[fiA] * sin_j[fiB];
                }
                angular *= s0;
            } else {
                for (int m = 0; m < order; m += 2) {
                    const T spf = s_prefactor[m];
                    T ang_m = Math<T>::zero();
                    for (int pp = 0; pp < pmax; ++pp) {
                        const long long fiA = hess_fourier_idx(pp, m, p, order, max_size1);
                        const long long fiB = hess_fourier_idx(pp, m, q, order, max_size2);
                        ang_m += cos_i[fiA] * cos_j[fiB] + sin_i[fiA] * sin_j[fiB];
                    }
                    angular += ang_m * spf;
                }
            }
            const T V = ksi_p * ksi_q * dist_scale + angular * ang_scale;

            for (int amu = tid; amu < na3A; amu += nthreads) {
                T dang_A = Math<T>::zero();
                if (order1) {
                    for (int pp = 0; pp < pmax; ++pp) {
                        const long long fiA = hess_fourier_idx(pp, 0, p, order, max_size1);
                        const long long fiB = hess_fourier_idx(pp, 0, q, order, max_size2);
                        dang_A += dcos_i[fiA * na3A + amu] * cos_j[fiB] +
                                  dsin_i[fiA * na3A + amu] * sin_j[fiB];
                    }
                    dang_A *= s0;
                } else {
                    for (int m = 0; m < order; m += 2) {
                        const T spf = s_prefactor[m];
                        T dam = Math<T>::zero();
                        for (int pp = 0; pp < pmax; ++pp) {
                            const long long fiA = hess_fourier_idx(pp, m, p, order, max_size1);
                            const long long fiB = hess_fourier_idx(pp, m, q, order, max_size2);
                            dam += dcos_i[fiA * na3A + amu] * cos_j[fiB] +
                                   dsin_i[fiA * na3A + amu] * sin_j[fiB];
                        }
                        dang_A += dam * spf;
                    }
                }
                s_dVA[amu] = dksi_i[p_na3A + amu] * ksi_q * dist_scale + dang_A * ang_scale;
            }
            for (int bnu = tid; bnu < na3B; bnu += nthreads) {
                T dang_B = Math<T>::zero();
                if (order1) {
                    for (int pp = 0; pp < pmax; ++pp) {
                        const long long fiA = hess_fourier_idx(pp, 0, p, order, max_size1);
                        const long long fiB = hess_fourier_idx(pp, 0, q, order, max_size2);
                        dang_B += dcos_j[fiB * na3B + bnu] * cos_i[fiA] +
                                  dsin_j[fiB * na3B + bnu] * sin_i[fiA];
                    }
                    dang_B *= s0;
                } else {
                    for (int m = 0; m < order; m += 2) {
                        const T spf = s_prefactor[m];
                        T dam = Math<T>::zero();
                        for (int pp = 0; pp < pmax; ++pp) {
                            const long long fiA = hess_fourier_idx(pp, m, p, order, max_size1);
                            const long long fiB = hess_fourier_idx(pp, m, q, order, max_size2);
                            dam += dcos_j[fiB * na3B + bnu] * cos_i[fiA] +
                                   dsin_j[fiB * na3B + bnu] * sin_i[fiA];
                        }
                        dang_B += dam * spf;
                    }
                }
                s_dVB[bnu] = dksi_j[q_na3B + bnu] * ksi_p * dist_scale + dang_B * ang_scale;
            }
            __syncthreads();

            for (int flat = tid; flat < n_entries; flat += nthreads) {
                const int amu = flat / na3B;
                const int bnu = flat - amu * na3B;
                const T dri_A = dri_i[p_na3A + amu];
                const T dri_B = dri_j[q_na3B + bnu];
                const T dksi_A = dksi_i[p_na3A + amu];
                const T dksi_B = dksi_j[q_na3B + bnu];
                T dang_AB = Math<T>::zero();
                if (order1) {
                    for (int pp = 0; pp < pmax; ++pp) {
                        const long long fiA = hess_fourier_idx(pp, 0, p, order, max_size1);
                        const long long fiB = hess_fourier_idx(pp, 0, q, order, max_size2);
                        dang_AB += dcos_i[fiA * na3A + amu] * dcos_j[fiB * na3B + bnu] +
                                   dsin_i[fiA * na3A + amu] * dsin_j[fiB * na3B + bnu];
                    }
                    dang_AB *= s0;
                } else {
                    for (int m = 0; m < order; m += 2) {
                        const T spf = s_prefactor[m];
                        T dam = Math<T>::zero();
                        for (int pp = 0; pp < pmax; ++pp) {
                            const long long fiA = hess_fourier_idx(pp, m, p, order, max_size1);
                            const long long fiB = hess_fourier_idx(pp, m, q, order, max_size2);
                            dam += dcos_i[fiA * na3A + amu] * dcos_j[fiB * na3B + bnu] +
                                   dsin_i[fiA * na3A + amu] * dsin_j[fiB * na3B + bnu];
                        }
                        dang_AB += dam * spf;
                    }
                }
                const T d2V = dksi_A * dksi_B * dist_scale + dang_AB * ang_scale;
                const T hij = dG_drp * dri_A * s_dVB[bnu] + dG_drq * dri_B * s_dVA[amu] +
                              d2G_drpq * dri_A * dri_B * V + G * d2V;
                atomicAdd(
                    hess_out + (row0 + amu) * d_b_cols + (col0 + bnu), coeff_hess * hij
                );
            }
            __syncthreads();
        }
    }
}

// After lower-triangle-only hot path: mirror off-diagonal blocks and
// symmetrise diagonal blocks so H is exactly symmetric (CPU hessian_symm_blocks).
template <typename T>
__global__ void hess_symm_finalize_kernel(
    T *H,
    const int *offset,
    const int *n,
    int nm,
    long long D
) {
    const int n_pairs = nm * (nm + 1) / 2;
    const int pair = static_cast<int>(blockIdx.x);
    if (pair >= n_pairs) {
        return;
    }

    const double s = sqrt(1.0 + 8.0 * static_cast<double>(pair));
    int a = static_cast<int>((s - 1.0) * 0.5);
    while (a * (a + 1) / 2 > pair) {
        --a;
    }
    while ((a + 1) * (a + 2) / 2 <= pair) {
        ++a;
    }
    const int b = pair - a * (a + 1) / 2;

    const int r0 = offset[a];
    const int c0 = offset[b];
    const int na3 = n[a] * 3;
    const int nb3 = n[b] * 3;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    if (a == b) {
        for (int flat = tid; flat < na3 * na3; flat += nthreads) {
            const int amu = flat / na3;
            const int bnu = flat - amu * na3;
            if (amu > bnu) {
                continue;
            }
            const long long i0 = (static_cast<long long>(r0 + amu) * D) + (c0 + bnu);
            const long long i1 = (static_cast<long long>(r0 + bnu) * D) + (c0 + amu);
            const T v = T(0.5) * (H[i0] + H[i1]);
            H[i0] = v;
            H[i1] = v;
        }
    } else {
        for (int flat = tid; flat < na3 * nb3; flat += nthreads) {
            const int amu = flat / nb3;
            const int bnu = flat - amu * nb3;
            const long long i_ab = (static_cast<long long>(r0 + amu) * D) + (c0 + bnu);
            const long long i_ba = (static_cast<long long>(c0 + bnu) * D) + (r0 + amu);
            H[i_ba] = H[i_ab];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels (precompute / self-scalar)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void hess_mark_elements_kernel(
    const T *x, const int *n, const int *nn, int nm, int max_size, int *z_present
) {
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= nm * max_size) {
        return;
    }
    const int a = flat_idx / max_size;
    const int i = flat_idx % max_size;
    if (i >= n[a]) {
        return;
    }
    const T *z_chan = atom_slice(x, max_size, a, i) + max_size;
    const int n_neigh = nn[flat_idx];
    for (int k = 0; k < n_neigh; ++k) {
        const int z = static_cast<int>(z_chan[k]);
        if (z > 0 && z < 256) {
            atomicOr(z_present + z, 1);
        }
    }
}

template <typename T>
__global__ void hess_precompute_kernel(
    const T *x,
    const int *n,
    const int *nn,
    const T *coords,
    const int *z,
    int *nbr_all,
    T *ksi_all,
    T *dksi_all,
    T *dri_all,
    T *cosp_all,
    T *sinp_all,
    T *dcosp_all,
    T *dsinp_all,
    int nm,
    int max_size,
    HessParams<T> params,
    const int *z_to_idx
) {
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= nm * max_size) {
        return;
    }
    const int a = flat_idx / max_size;
    const int i = flat_idx % max_size;
    const int na = n[a];
    if (i >= na) {
        return;
    }

    const int na3 = na * 3;
    const long long na3_max = 3LL * max_size;
    const long long fstride = fourier_atom_stride(params.pmax, params.fourier_order, max_size);

    const T *atom_chan = atom_slice(x, max_size, a, i);
    const int n_neigh = nn[flat_idx];

    int *nbr = nbr_all + static_cast<long long>(flat_idx) * max_size;
    T *ksi = ksi_all + static_cast<long long>(flat_idx) * max_size;
    T *dksi = dksi_all + static_cast<long long>(flat_idx) * max_size * na3_max;
    T *dri = dri_all + static_cast<long long>(flat_idx) * max_size * na3_max;
    T *cosp = cosp_all + static_cast<long long>(flat_idx) * fstride;
    T *sinp = sinp_all + static_cast<long long>(flat_idx) * fstride;
    T *dcosp = dcosp_all + static_cast<long long>(flat_idx) * fstride * na3_max;
    T *dsinp = dsinp_all + static_cast<long long>(flat_idx) * fstride * na3_max;

    hess_build_nbr_atom_idx(
        atom_chan,
        max_size,
        n_neigh,
        coords + static_cast<long long>(a) * max_size * 3,
        z + static_cast<long long>(a) * max_size,
        i,
        na,
        nbr
    );

    hess_compute_ksi_and_grad(
        atom_chan,
        max_size,
        n_neigh,
        params.two_body_power,
        params.cut_start,
        params.cut_distance,
        i,
        na3,
        nbr,
        ksi,
        dksi,
        dri
    );

    hess_compute_threebody_fourier_and_grad(
        atom_chan,
        max_size,
        n_neigh,
        params.three_body_power,
        params.cut_start,
        params.cut_distance,
        params.fourier_order,
        z_to_idx,
        params.use_atm != 0,
        i,
        na3,
        nbr,
        cosp,
        sinp,
        dcosp,
        dsinp
    );
}

// One block per atom (a, i); threads split the na3 gradient components.
// dss = 2 * d(s_ii)/dR, matching the CPU factor.
template <typename T>
__global__ void hess_self_scalar_kernel(
    const T *x,
    const int *n,
    const int *nn,
    const T *ksi_all,
    const T *dksi_all,
    const T *dri_all,
    const T *cosp_all,
    const T *sinp_all,
    const T *dcosp_all,
    const T *dsinp_all,
    T *ss_out,
    T *dss_out,
    int nm,
    int max_size,
    HessParams<T> params,
    const T *s_prefactor
) {
    const int flat_idx = static_cast<int>(blockIdx.x);
    if (flat_idx >= nm * max_size) {
        return;
    }
    const int a = flat_idx / max_size;
    const int i = flat_idx % max_size;
    const int na = n[a];
    if (i >= na) {
        if (threadIdx.x == 0) {
            ss_out[flat_idx] = Math<T>::zero();
        }
        return;
    }

    const int na3 = na * 3;
    const long long na3_max = 3LL * max_size;
    const long long fstride = fourier_atom_stride(params.pmax, params.fourier_order, max_size);

    const T *x_chan = atom_slice(x, max_size, a, i);
    const int n_neigh = nn[flat_idx];
    const T *ksi = ksi_all + static_cast<long long>(flat_idx) * max_size;
    const T *dksi = dksi_all + static_cast<long long>(flat_idx) * max_size * na3_max;
    const T *dri = dri_all + static_cast<long long>(flat_idx) * max_size * na3_max;
    const T *cosp = cosp_all + static_cast<long long>(flat_idx) * fstride;
    const T *sinp = sinp_all + static_cast<long long>(flat_idx) * fstride;
    const T *dcosp = dcosp_all + static_cast<long long>(flat_idx) * fstride * na3_max;
    const T *dsinp = dsinp_all + static_cast<long long>(flat_idx) * fstride * na3_max;
    T *dss = dss_out + static_cast<long long>(flat_idx) * na3_max;

    T ss = Math<T>::zero();
    for (int amu = static_cast<int>(threadIdx.x); amu < na3;
         amu += static_cast<int>(blockDim.x)) {
        T ds = Math<T>::zero();
        ss = hess_scalar_and_grad_amu(
            x_chan,
            max_size,
            n_neigh,
            ksi,
            dksi,
            dri,
            cosp,
            sinp,
            dcosp,
            dsinp,
            x_chan,
            max_size,
            n_neigh,
            ksi,
            cosp,
            sinp,
            params,
            s_prefactor,
            na3,
            amu,
            &ds
        );
        dss[amu] = T(2) * ds;
    }

    // na3 >= 3, so thread 0 always ran at least one iteration.
    if (threadIdx.x == 0) {
        ss_out[flat_idx] = ss;
    }
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

template <typename T>
struct HessSideBuffers {
    T *ksi = nullptr;
    T *dksi = nullptr;
    T *dri = nullptr;
    T *cosp = nullptr;
    T *sinp = nullptr;
    T *dcosp = nullptr;
    T *dsinp = nullptr;
    T *ss = nullptr;
    T *dss = nullptr;
    int *nbr = nullptr;
    int *offset = nullptr;
    size_t ksi_cap = 0;
    size_t dksi_cap = 0;
    size_t dri_cap = 0;
    size_t cosp_cap = 0;
    size_t sinp_cap = 0;
    size_t dcosp_cap = 0;
    size_t dsinp_cap = 0;
    size_t ss_cap = 0;
    size_t dss_cap = 0;
    size_t nbr_cap = 0;
    size_t offset_cap = 0;
};

template <typename T>
struct HessWorkspace {
    HessSideBuffers<T> a;
    HessSideBuffers<T> b;
    T *s_prefactor = nullptr;
    size_t s_prefactor_cap = 0;
    int *z_present = nullptr;
    int *z_to_idx = nullptr;
};

template <typename T>
HessWorkspace<T> &hess_workspace() {
    static HessWorkspace<T> ws;
    return ws;
}

// Allocate, zero and fill one side's precompute buffers.
template <typename T>
void hess_prepare_side(
    HessSideBuffers<T> &buf,
    const T *d_x,
    const int *d_n,
    const int *d_nn,
    const T *d_coords,
    const int *d_z,
    int nm,
    int max_size,
    int pmax,
    int fourier_order,
    const HessParams<T> &params,
    const int *d_z_to_idx,
    const T *d_s_prefactor
) {
    const size_t n_atoms = static_cast<size_t>(nm) * max_size;
    const size_t na3_max = static_cast<size_t>(max_size) * 3;
    const size_t fstride =
        static_cast<size_t>(pmax) * fourier_order * static_cast<size_t>(max_size);

    const size_t ksi_need = n_atoms * static_cast<size_t>(max_size);
    const size_t dksi_need = ksi_need * na3_max;
    const size_t cosp_need = n_atoms * fstride;
    const size_t dcosp_need = cosp_need * na3_max;
    const size_t dss_need = n_atoms * na3_max;

    ensure_capacity(&buf.ksi, &buf.ksi_cap, ksi_need);
    ensure_capacity(&buf.dksi, &buf.dksi_cap, dksi_need);
    ensure_capacity(&buf.dri, &buf.dri_cap, dksi_need);
    ensure_capacity(&buf.cosp, &buf.cosp_cap, cosp_need);
    ensure_capacity(&buf.sinp, &buf.sinp_cap, cosp_need);
    ensure_capacity(&buf.dcosp, &buf.dcosp_cap, dcosp_need);
    ensure_capacity(&buf.dsinp, &buf.dsinp_cap, dcosp_need);
    ensure_capacity(&buf.ss, &buf.ss_cap, n_atoms);
    ensure_capacity(&buf.dss, &buf.dss_cap, dss_need);
    ensure_capacity(&buf.nbr, &buf.nbr_cap, ksi_need);

    // The precompute kernel accumulates, and padding atoms are never written.
    CUDA_CHECK(cudaMemset(buf.dksi, 0, dksi_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(buf.dri, 0, dksi_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(buf.cosp, 0, cosp_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(buf.sinp, 0, cosp_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(buf.dcosp, 0, dcosp_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(buf.dsinp, 0, dcosp_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(buf.dss, 0, dss_need * sizeof(T)));

    const int grid =
        static_cast<int>((n_atoms + kHessPrecomputeBlockSize - 1) / kHessPrecomputeBlockSize);

    hess_precompute_kernel<<<grid, kHessPrecomputeBlockSize>>>(
        d_x,
        d_n,
        d_nn,
        d_coords,
        d_z,
        buf.nbr,
        buf.ksi,
        buf.dksi,
        buf.dri,
        buf.cosp,
        buf.sinp,
        buf.dcosp,
        buf.dsinp,
        nm,
        max_size,
        params,
        d_z_to_idx
    );
    CUDA_CHECK(cudaGetLastError());

    hess_self_scalar_kernel<<<static_cast<int>(n_atoms), kHessSelfBlockSize>>>(
        d_x,
        d_n,
        d_nn,
        buf.ksi,
        buf.dksi,
        buf.dri,
        buf.cosp,
        buf.sinp,
        buf.dcosp,
        buf.dsinp,
        buf.ss,
        buf.dss,
        nm,
        max_size,
        params,
        d_s_prefactor
    );
    CUDA_CHECK(cudaGetLastError());
}

// Upload the per-molecule row/column offsets and validate the output extent.
template <typename T>
void hess_upload_offsets(
    HessSideBuffers<T> &buf, const int *d_n, int nm, int expected_dim, const char *what
) {
    std::vector<int> h_n(static_cast<size_t>(nm));
    CUDA_CHECK(cudaMemcpy(
        h_n.data(), d_n, static_cast<size_t>(nm) * sizeof(int), cudaMemcpyDeviceToHost
    ));
    std::vector<int> h_offset(static_cast<size_t>(nm) + 1, 0);
    for (int a = 0; a < nm; ++a) {
        h_offset[static_cast<size_t>(a) + 1] =
            h_offset[static_cast<size_t>(a)] + h_n[static_cast<size_t>(a)] * 3;
    }
    if (h_offset[static_cast<size_t>(nm)] != expected_dim) {
        throw std::invalid_argument(what);
    }
    ensure_capacity(&buf.offset, &buf.offset_cap, static_cast<size_t>(nm) + 1);
    CUDA_CHECK(cudaMemcpy(
        buf.offset,
        h_offset.data(),
        (static_cast<size_t>(nm) + 1) * sizeof(int),
        cudaMemcpyHostToDevice
    ));
}

template <typename T>
void kernel_gaussian_hessian_cu_impl(
    const T *d_x1,
    const T *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const T *d_coords1,
    const int *d_z1,
    const T *d_coords2,
    const int *d_z2,
    T *d_hess_out,
    T sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    T two_body_scaling,
    T two_body_width,
    T two_body_power,
    T three_body_scaling,
    T three_body_width,
    T three_body_power,
    T cut_start,
    T cut_distance,
    int fourier_order,
    bool use_atm,
    bool lower_triangle
) {
    if (fourier_order <= 0 || fourier_order > kMaxFourierOrder) {
        throw std::invalid_argument("fourier_order must be in [1, 16] for cuda_fchl18_kernel");
    }
    if (nm1 <= 0 || nm2 <= 0 || d_a_rows <= 0 || d_b_cols <= 0) {
        return;
    }
    if (lower_triangle && (nm1 != nm2 || d_a_rows != d_b_cols || max_size1 != max_size2)) {
        throw std::invalid_argument(
            "hessian symm requires identical A/B molecule sets and dimensions"
        );
    }
    if (3 * max_size1 > kHessMaxNa3 || 3 * max_size2 > kHessMaxNa3) {
        throw std::invalid_argument(
            "cuda_fchl18_kernel hessian supports at most 128 atoms per molecule"
        );
    }

    HessWorkspace<T> &ws = hess_workspace<T>();
    if (ws.z_present == nullptr) {
        CUDA_CHECK(cudaMalloc(&ws.z_present, 256 * sizeof(int)));
    }
    if (ws.z_to_idx == nullptr) {
        CUDA_CHECK(cudaMalloc(&ws.z_to_idx, 256 * sizeof(int)));
    }
    CUDA_CHECK(cudaMemset(ws.z_present, 0, 256 * sizeof(int)));

    const size_t n_atoms1 = static_cast<size_t>(nm1) * max_size1;
    const size_t n_atoms2 = static_cast<size_t>(nm2) * max_size2;
    const int grid1 =
        static_cast<int>((n_atoms1 + kHessPrecomputeBlockSize - 1) / kHessPrecomputeBlockSize);
    const int grid2 =
        static_cast<int>((n_atoms2 + kHessPrecomputeBlockSize - 1) / kHessPrecomputeBlockSize);

    hess_mark_elements_kernel<<<grid1, kHessPrecomputeBlockSize>>>(
        d_x1, d_n1, d_nn1, nm1, max_size1, ws.z_present
    );
    CUDA_CHECK(cudaGetLastError());
    hess_mark_elements_kernel<<<grid2, kHessPrecomputeBlockSize>>>(
        d_x2, d_n2, d_nn2, nm2, max_size2, ws.z_present
    );
    CUDA_CHECK(cudaGetLastError());

    int h_z_present[256];
    CUDA_CHECK(cudaMemcpy(h_z_present, ws.z_present, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    int h_z_to_idx[256];
    const int pmax = build_element_map_from_flags(h_z_present, h_z_to_idx);

    const size_t out_n = static_cast<size_t>(d_a_rows) * static_cast<size_t>(d_b_cols);
    CUDA_CHECK(cudaMemset(d_hess_out, 0, out_n * sizeof(T)));
    if (pmax == 0) {
        return;
    }
    if (pmax > kMaxElements) {
        throw std::invalid_argument("too many distinct elements for cuda_fchl18_kernel");
    }
    CUDA_CHECK(cudaMemcpy(ws.z_to_idx, h_z_to_idx, 256 * sizeof(int), cudaMemcpyHostToDevice));

    HessParams<T> params{};
    params.two_body_width = two_body_width;
    params.two_body_power = two_body_power;
    params.three_body_power = three_body_power;
    params.cut_start = cut_start;
    params.cut_distance = cut_distance;
    params.true_distance_scale = two_body_scaling / static_cast<T>(16);
    params.true_angular_scale = three_body_scaling / static_cast<T>(std::sqrt(8.0));
    params.inv_sigma2 = static_cast<T>(0.5) / (sigma * sigma);
    params.fourier_order = fourier_order;
    params.pmax = pmax;
    params.use_atm = use_atm ? 1 : 0;

    T h_s_prefactor[kMaxFourierOrder] = {};
    const double ang_norm2 = get_angular_norm2(static_cast<double>(three_body_width));
    const double g1 = std::sqrt(2.0 * 3.14159265358979323846) / ang_norm2;
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        const double tw = static_cast<double>(three_body_width);
        h_s_prefactor[m] = static_cast<T>(g1 * std::exp(-(tw * mf) * (tw * mf) / 2.0));
    }
    ensure_capacity(&ws.s_prefactor, &ws.s_prefactor_cap, static_cast<size_t>(kMaxFourierOrder));
    CUDA_CHECK(cudaMemcpy(
        ws.s_prefactor,
        h_s_prefactor,
        static_cast<size_t>(kMaxFourierOrder) * sizeof(T),
        cudaMemcpyHostToDevice
    ));

    hess_upload_offsets(ws.a, d_n1, nm1, d_a_rows, "hessian output rows must equal 3 * sum(n1)");
    hess_upload_offsets(ws.b, d_n2, nm2, d_b_cols, "hessian output cols must equal 3 * sum(n2)");

    hess_prepare_side(
        ws.a,
        d_x1,
        d_n1,
        d_nn1,
        d_coords1,
        d_z1,
        nm1,
        max_size1,
        pmax,
        fourier_order,
        params,
        ws.z_to_idx,
        ws.s_prefactor
    );
    hess_prepare_side(
        ws.b,
        d_x2,
        d_n2,
        d_nn2,
        d_coords2,
        d_z2,
        nm2,
        max_size2,
        pmax,
        fourier_order,
        params,
        ws.z_to_idx,
        ws.s_prefactor
    );

    const int n_ij_slots = max_size1 * max_size2;
    const int n_mol_pairs =
        lower_triangle ? (nm1 * (nm1 + 1) / 2) : (nm1 * nm2);
    const int n_hot_blocks = n_mol_pairs * n_ij_slots;
    const int strideA = 3 * max_size1;
    const int strideB = 3 * max_size2;
    const int h_cap =
        (strideA * strideB <= kHessSharedHMax) ? (strideA * strideB) : 0;
    const size_t hot_shmem =
        sizeof(T) * (static_cast<size_t>(2 * strideA + 2 * strideB + h_cap + 1));
    hess_hot_kernel<<<n_hot_blocks, kHessBlockSize, hot_shmem>>>(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        ws.a.offset,
        ws.b.offset,
        ws.a.ksi,
        ws.a.dksi,
        ws.a.dri,
        ws.a.cosp,
        ws.a.sinp,
        ws.a.dcosp,
        ws.a.dsinp,
        ws.a.ss,
        ws.a.dss,
        ws.b.ksi,
        ws.b.dksi,
        ws.b.dri,
        ws.b.cosp,
        ws.b.sinp,
        ws.b.dcosp,
        ws.b.dsinp,
        ws.b.ss,
        ws.b.dss,
        d_hess_out,
        nm1,
        nm2,
        max_size1,
        max_size2,
        static_cast<long long>(d_b_cols),
        lower_triangle ? 1 : 0,
        params,
        ws.s_prefactor
    );
    CUDA_CHECK(cudaGetLastError());

    if (lower_triangle) {
        hess_symm_finalize_kernel<<<n_mol_pairs, kHessBlockSize>>>(
            d_hess_out,
            ws.a.offset,
            d_n1,
            nm1,
            static_cast<long long>(d_a_rows)
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace

void kernel_gaussian_hessian_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const float *d_coords1,
    const int *d_z1,
    const float *d_coords2,
    const int *d_z2,
    float *d_hess_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_hessian_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_coords1,
        d_z1,
        d_coords2,
        d_z2,
        d_hess_out,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
        d_a_rows,
        d_b_cols,
        two_body_scaling,
        two_body_width,
        two_body_power,
        three_body_scaling,
        three_body_width,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm,
        /*lower_triangle=*/false
    );
}

void kernel_gaussian_hessian_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const double *d_coords1,
    const int *d_z1,
    const double *d_coords2,
    const int *d_z2,
    double *d_hess_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_hessian_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_coords1,
        d_z1,
        d_coords2,
        d_z2,
        d_hess_out,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
        d_a_rows,
        d_b_cols,
        two_body_scaling,
        two_body_width,
        two_body_power,
        three_body_scaling,
        three_body_width,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm,
        /*lower_triangle=*/false
    );
}

namespace {

constexpr int kHessRfpPackBlock = 256;

// Device RFP index — matches kf::rfp_index_upper_N (TRANSR='N', UPLO='U').
__device__ inline long long hess_rfp_index_upper_N(int n, int i, int j) {
    const int k = n / 2;
    const int stride = (n % 2 == 0) ? (n + 1) : n;
    if (j >= k) {
        return static_cast<long long>(j - k) * stride + i;
    }
    return static_cast<long long>(i) * stride + (j + k + 1);
}

template <typename T>
__global__ void hess_pack_rfp_kernel(const T *H, T *rfp, int D) {
    const long long n = static_cast<long long>(D) * (D + 1) / 2;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    const double s = sqrt(1.0 + 8.0 * static_cast<double>(tid));
    int row = static_cast<int>((s - 1.0) * 0.5);
    while (row * (row + 1) / 2 > static_cast<int>(tid)) {
        --row;
    }
    while ((row + 1) * (row + 2) / 2 <= static_cast<int>(tid)) {
        ++row;
    }
    const int col = static_cast<int>(tid) - row * (row + 1) / 2;
    rfp[hess_rfp_index_upper_N(D, col, row)] = H[static_cast<long long>(row) * D + col];
}

}  // namespace

template <typename T>
static void kernel_gaussian_hessian_symm_rfp_cu_impl(
    const T *d_x,
    const int *d_n,
    const int *d_nn,
    const T *d_coords,
    const int *d_z,
    T *d_rfp_out,
    T sigma,
    int nm,
    int max_size,
    int d_rows,
    T two_body_scaling,
    T two_body_width,
    T two_body_power,
    T three_body_scaling,
    T three_body_width,
    T three_body_power,
    T cut_start,
    T cut_distance,
    int fourier_order,
    bool use_atm
) {
    if (nm <= 0 || d_rows <= 0) {
        throw std::invalid_argument("kernel_gaussian_hessian_symm_rfp: empty molecule set");
    }

    T *d_hess = nullptr;
    const size_t hess_n = static_cast<size_t>(d_rows) * static_cast<size_t>(d_rows);
    CUDA_CHECK(cudaMalloc(&d_hess, hess_n * sizeof(T)));
    kernel_gaussian_hessian_symm_cu(
        d_x,
        d_n,
        d_nn,
        d_coords,
        d_z,
        d_hess,
        sigma,
        nm,
        max_size,
        d_rows,
        two_body_scaling,
        two_body_width,
        two_body_power,
        three_body_scaling,
        three_body_width,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm
    );

    const long long rfp_n = static_cast<long long>(d_rows) * (d_rows + 1) / 2;
    const int grid = static_cast<int>((rfp_n + kHessRfpPackBlock - 1) / kHessRfpPackBlock);
    hess_pack_rfp_kernel<<<grid, kHessRfpPackBlock>>>(d_hess, d_rfp_out, d_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_hess));
}

void kernel_gaussian_hessian_symm_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    const float *d_coords,
    const int *d_z,
    float *d_hess_out,
    float sigma,
    int nm,
    int max_size,
    int d_rows,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_hessian_cu_impl(
        d_x,
        d_x,
        d_n,
        d_n,
        d_nn,
        d_nn,
        d_coords,
        d_z,
        d_coords,
        d_z,
        d_hess_out,
        sigma,
        nm,
        nm,
        max_size,
        max_size,
        d_rows,
        d_rows,
        two_body_scaling,
        two_body_width,
        two_body_power,
        three_body_scaling,
        three_body_width,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm,
        /*lower_triangle=*/true
    );
}

void kernel_gaussian_hessian_symm_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    const double *d_coords,
    const int *d_z,
    double *d_hess_out,
    double sigma,
    int nm,
    int max_size,
    int d_rows,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_hessian_cu_impl(
        d_x,
        d_x,
        d_n,
        d_n,
        d_nn,
        d_nn,
        d_coords,
        d_z,
        d_coords,
        d_z,
        d_hess_out,
        sigma,
        nm,
        nm,
        max_size,
        max_size,
        d_rows,
        d_rows,
        two_body_scaling,
        two_body_width,
        two_body_power,
        three_body_scaling,
        three_body_width,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm,
        /*lower_triangle=*/true
    );
}

void kernel_gaussian_hessian_symm_rfp_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    const float *d_coords,
    const int *d_z,
    float *d_rfp_out,
    float sigma,
    int nm,
    int max_size,
    int d_rows,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_hessian_symm_rfp_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_coords,
        d_z,
        d_rfp_out,
        sigma,
        nm,
        max_size,
        d_rows,
        two_body_scaling,
        two_body_width,
        two_body_power,
        three_body_scaling,
        three_body_width,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm
    );
}

void kernel_gaussian_hessian_symm_rfp_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    const double *d_coords,
    const int *d_z,
    double *d_rfp_out,
    double sigma,
    int nm,
    int max_size,
    int d_rows,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_hessian_symm_rfp_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_coords,
        d_z,
        d_rfp_out,
        sigma,
        nm,
        max_size,
        d_rows,
        two_body_scaling,
        two_body_width,
        two_body_power,
        three_body_scaling,
        three_body_width,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm
    );
}

}  // namespace fchl18
}  // namespace kf
