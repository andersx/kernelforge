// CUDA FCHL18 Jacobian kernel — J[row, b] = dK(A_a, B_b) / dR_{A_a}[alpha, mu]
//
// Port of kf::fchl18::kernel_gaussian_jacobian (src/fchl18_kernel.cpp). The CPU
// entry point takes raw coordinates for the query molecules and generates their
// representation internally; this one takes a precomputed representation x1
// (like the CUDA scalar kernel) plus coords1/z1, which are needed to map
// neighbour slots back to atom indices for the chain rule.
//
// Layout notes
// ------------
// Fourier terms use the CPU layout  [p][m][neigh]  (not the atom-major layout of
// cuda_fchl18_kernel.cu) so the gradient port stays line-for-line comparable with
// the reference. Gradients hang off that flat index:  [fidx * na3 + alpha*3 + mu]
// with na3 = 3 * n1[a]; buffers are strided by 3 * max_size1 so a single
// allocation covers every molecule.
//
// The query side is deliberately not Z-sorted (the CPU gradient path is not
// either): the scalar product uses nested Zi == Zj loops rather than the sorted
// merge, so both sides may be in arbitrary neighbour order.

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

constexpr int kJacBlockSize = 128;
constexpr int kPrecomputeBlockSize = 128;
// Shared reduction + register-tile caps for the fp32 hot kernel.
constexpr int kHotBlockSize = 128;
constexpr int kMaxNa3Local = 64;  // up to 21 atoms with register tiles (8*8)
constexpr int kAmuTile = 8;
constexpr int kNumTiles = kMaxNa3Local / kAmuTile;  // 8


// Flat index into the CPU-layout Fourier arrays: [p][m][neigh].
__device__ inline long long jac_fourier_idx(int p, int m, int neigh, int order, int max_size) {
    return (static_cast<long long>(p) * order + m) * max_size + neigh;
}

template <typename T>
struct JacParams {
    T two_body_width;
    T two_body_power;
    T three_body_power;
    T cut_start;
    T cut_distance;
    T true_distance_scale;
    T true_angular_scale;
    T inv_sigma2;  // +0.5 / sigma^2, matching the CPU GradBSide convention
    int fourier_order;
    int pmax;
    int use_atm;
};

// ---------------------------------------------------------------------------
// Query-side per-atom precompute (values + gradients)
// ---------------------------------------------------------------------------

// Map neighbour slot k of atom `centre` back to its atom index in molecule A by
// matching the stored displacement and nuclear charge. The CPU uses an exact
// 1e-10 window; a nearest-match search behaves identically in fp64 and stays
// robust in fp32.
template <typename T>
__device__ void build_nbr_atom_idx_device(
    const T *atom_chan,
    int max_size,
    int n_neigh,
    const T *coords_A,
    const int *z_A,
    int centre_idx,
    int n_atoms_A,
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

    const T cx = coords_A[centre_idx * 3 + 0];
    const T cy = coords_A[centre_idx * 3 + 1];
    const T cz = coords_A[centre_idx * 3 + 2];

    for (int k = 1; k < n_neigh; ++k) {
        const T dx = xc[k];
        const T dy = yc[k];
        const T dz = zc[k];
        const int zval = static_cast<int>(z_chan[k]);

        int best = -1;
        T best_d2 = Math<T>::zero();
        for (int j = 0; j < n_atoms_A; ++j) {
            if (z_A[j] != zval) {
                continue;
            }
            const T ex = coords_A[j * 3 + 0] - cx - dx;
            const T ey = coords_A[j * 3 + 1] - cy - dy;
            const T ez = coords_A[j * 3 + 2] - cz - dz;
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
__device__ void compute_ksi_and_grad_device(
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

// Three-body Fourier terms and their gradients for one query atom.
// cosp/sinp/dcosp/dsinp must be zero-initialised by the caller.
template <typename T>
__device__ void compute_threebody_fourier_and_grad_device(
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

            const T cos_i =
                clamp11(ui_j[0] * ui_k[0] + ui_j[1] * ui_k[1] + ui_j[2] * ui_k[2]);
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

                gksi3[0][mu] -=
                    atm * beta3_ksi3_inv * (-ui_j[mu] * inv_dj + -ui_k[mu] * inv_dk);
                gksi3[1][mu] -=
                    atm * beta3_ksi3_inv * (ui_j[mu] * inv_dj + (-ukj[mu]) * inv_di);
                gksi3[2][mu] -=
                    atm * beta3_ksi3_inv * (ui_k[mu] * inv_dk + (ukj[mu]) * inv_di);

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

                const long long fidx_j = jac_fourier_idx(pj, m, j, order, max_size);
                const long long fidx_k = jac_fourier_idx(pk, m, k, order, max_size);

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
// Training-side per-atom precompute (values only)
// ---------------------------------------------------------------------------

template <typename T>
__device__ void compute_ksi_value_device(
    const T *atom_chan, int max_size, int n_neigh, T power, T cut_start, T cut_distance, T *ksi
) {
    for (int k = 0; k < max_size; ++k) {
        ksi[k] = Math<T>::zero();
    }
    for (int k = 1; k < n_neigh; ++k) {
        const T r = atom_chan[k];
        if (r < Math<T>::eps()) {
            continue;
        }
        ksi[k] = cut_function(r, cut_start, cut_distance) / fast_pow(r, power);
    }
}

template <typename T>
__device__ void compute_threebody_fourier_value_device(
    const T *atom_chan,
    int max_size,
    int n_neigh,
    T three_body_power,
    T cut_start,
    T cut_distance,
    int order,
    const int *z_to_idx,
    bool use_atm,
    T *cosp,
    T *sinp
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
        const T uxj = xc[j] * inv_dj;
        const T uyj = yc[j] * inv_dj;
        const T uzj = zc[j] * inv_dj;

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
            const T di = Math<T>::sqrt_(dxjk * dxjk + dyjk * dyjk + dzjk * dzjk);
            if (di < Math<T>::eps()) {
                continue;
            }
            const T cut_jk = cut_function(di, cut_start, cut_distance);
            if (cut_jk == zero) {
                continue;
            }

            const T inv_dk = one / dk;
            const T uxk = xc[k] * inv_dk;
            const T uyk = yc[k] * inv_dk;
            const T uzk = zc[k] * inv_dk;

            const T cos_i = clamp11(uxj * uxk + uyj * uyk + uzj * uzk);

            T atm = one;
            if (use_atm) {
                const T inv_di = one / di;
                const T cos_j = clamp11(
                    (-dxjk * inv_di) * (-uxj) + (-dyjk * inv_di) * (-uyj) +
                    (-dzjk * inv_di) * (-uzj)
                );
                const T cos_k = clamp11(
                    (dxjk * inv_di) * (-uxk) + (dyjk * inv_di) * (-uyk) +
                    (dzjk * inv_di) * (-uzk)
                );
                atm = one + T(3) * cos_i * cos_j * cos_k;
            }
            if (atm == zero) {
                continue;
            }

            const T dijk = di * dj * dk;
            const T ksi3 = cutj * cutk * cut_jk * atm / fast_pow(dijk, three_body_power);
            if (ksi3 == zero) {
                continue;
            }

            const T theta = Math<T>::acos_(cos_i);

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

            for (int m = 0; m < order; m += 2) {
                const T mf = static_cast<T>(m + 1);
                T c, s;
                Math<T>::sincos_(mf * theta, &s, &c);
                const T cos_m = T(2) * c * ksi3;
                const T sin_m = T(2) * s * ksi3;

                const long long idx_j = jac_fourier_idx(pj, m, j, order, max_size);
                const long long idx_k = jac_fourier_idx(pk, m, k, order, max_size);
                cosp[idx_j] += cos_m;
                sinp[idx_j] += sin_m;
                cosp[idx_k] += cos_m;
                sinp[idx_k] += sin_m;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Value-only scalar product (no gradient buffer — used by the fp32 hot path).
// ---------------------------------------------------------------------------
template <typename T>
__device__ T scalar_noalchemy_value(
    const T *x1_chan,
    int max_size1,
    int n_neigh1,
    const T *ksi1,
    const T *cos1,
    const T *sin1,
    const T *x2_chan,
    int max_size2,
    int n_neigh2,
    const T *ksi2,
    const T *cos2,
    const T *sin2,
    const JacParams<T> &params,
    const T *s_prefactor
) {
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
    const bool order2 = (order == 1 || order == 2);
    const T s0 = s_prefactor[0];

    T aadist = Math<T>::one();
    for (int i = 1; i < n_neigh1; ++i) {
        const int Zi = static_cast<int>(x1_chan[max_size1 + i]);
        const T ri = x1_chan[i];
        const T ksi1_i = ksi1[i];
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
            T angular = Math<T>::zero();
            if (order2) {
                for (int p = 0; p < pmax; ++p) {
                    const long long i1 = jac_fourier_idx(p, 0, i, order, max_size1);
                    const long long i2 = jac_fourier_idx(p, 0, j, order, max_size2);
                    angular += (cos1[i1] * cos2[i2] + sin1[i1] * sin2[i2]) * s0;
                }
            } else {
                for (int m = 0; m < order; m += 2) {
                    const T spf = s_prefactor[m];
                    T ang_m = Math<T>::zero();
                    for (int p = 0; p < pmax; ++p) {
                        const long long i1 = jac_fourier_idx(p, m, i, order, max_size1);
                        const long long i2 = jac_fourier_idx(p, m, j, order, max_size2);
                        ang_m += cos1[i1] * cos2[i2] + sin1[i1] * sin2[i2];
                    }
                    angular += ang_m * spf;
                }
            }
            aadist += G * (ksi1_i * ksi2[j] * dist_scale + angular * ang_scale);
        }
    }
    return aadist;
}

// One neighbor pass: accumulate coeff * (-dsii[amu] + 2 * dsij[amu]) into
// smem_grad via atomicAdd. No per-thread dsij buffer → no local-memory spill.
template <typename T>
__device__ void scalar_noalchemy_grad_atomic(
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
    const JacParams<T> &params,
    const T *s_prefactor,
    int na3,
    T coeff,
    const T *dsii,
    T *smem_grad
) {
    const int Z1 = static_cast<int>(x1_chan[max_size1]);
    const int Z2 = static_cast<int>(x2_chan[max_size2]);

    // Even on Z mismatch, -dsii still contributes (matches CPU / amu path).
    const T neg_coeff = -coeff;
    for (int amu = 0; amu < na3; ++amu) {
        atomicAdd(smem_grad + amu, neg_coeff * dsii[amu]);
    }
    if (Z1 != Z2) {
        return;
    }

    const T d_width = params.two_body_width;
    const T inv_width = -Math<T>::one() / (T(4) * d_width * d_width);
    const T maxgausdist2 = (T(8) * d_width) * (T(8) * d_width);
    const T dist_scale = params.true_distance_scale;
    const T ang_scale = params.true_angular_scale;
    const int order = params.fourier_order;
    const int pmax = params.pmax;
    const bool order2 = (order == 1 || order == 2);
    const T s0 = s_prefactor[0];
    const T two_coeff = T(2) * coeff;

    for (int i = 1; i < n_neigh1; ++i) {
        const int Zi = static_cast<int>(x1_chan[max_size1 + i]);
        const T ri = x1_chan[i];
        const long long i_na3 = static_cast<long long>(i) * na3;
        const T ksi1_i = ksi1[i];

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
            if (order2) {
                for (int p = 0; p < pmax; ++p) {
                    const long long i1 = jac_fourier_idx(p, 0, i, order, max_size1);
                    const long long i2 = jac_fourier_idx(p, 0, j, order, max_size2);
                    angular += (cos1[i1] * cos2[i2] + sin1[i1] * sin2[i2]) * s0;
                }
            } else {
                for (int m = 0; m < order; m += 2) {
                    const T spf = s_prefactor[m];
                    T ang_m = Math<T>::zero();
                    for (int p = 0; p < pmax; ++p) {
                        const long long i1 = jac_fourier_idx(p, m, i, order, max_size1);
                        const long long i2 = jac_fourier_idx(p, m, j, order, max_size2);
                        ang_m += cos1[i1] * cos2[i2] + sin1[i1] * sin2[i2];
                    }
                    angular += ang_m * spf;
                }
            }

            const T ksi2_j = ksi2[j];
            const T W = ksi1_i * ksi2_j * dist_scale + angular * ang_scale;
            const T G_dist = G * dist_scale;
            const T G_ang = G * ang_scale;

            for (int amu = 0; amu < na3; ++amu) {
                const T dsij =
                    dG_dri * dri1[i_na3 + amu] * W + G_dist * dksi1[i_na3 + amu] * ksi2_j;
                atomicAdd(smem_grad + amu, two_coeff * dsij);
            }

            if (order2) {
                for (int p = 0; p < pmax; ++p) {
                    const long long i1 = jac_fourier_idx(p, 0, i, order, max_size1);
                    const long long i2 = jac_fourier_idx(p, 0, j, order, max_size2);
                    const T c2 = cos2[i2];
                    const T s2 = sin2[i2];
                    const long long gbase = i1 * na3;
                    const T scale = two_coeff * G_ang * s0;
                    for (int amu = 0; amu < na3; ++amu) {
                        atomicAdd(
                            smem_grad + amu,
                            scale * (dcos1[gbase + amu] * c2 + dsin1[gbase + amu] * s2)
                        );
                    }
                }
            } else {
                for (int m = 0; m < order; m += 2) {
                    const T spf = s_prefactor[m];
                    for (int p = 0; p < pmax; ++p) {
                        const long long i1 = jac_fourier_idx(p, m, i, order, max_size1);
                        const long long i2 = jac_fourier_idx(p, m, j, order, max_size2);
                        const T c2 = cos2[i2];
                        const T s2 = sin2[i2];
                        const long long gbase = i1 * na3;
                        const T scale = two_coeff * G_ang * spf;
                        for (int amu = 0; amu < na3; ++amu) {
                            atomicAdd(
                                smem_grad + amu,
                                scale * (dcos1[gbase + amu] * c2 + dsin1[gbase + amu] * s2)
                            );
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar product (+ optional full gradient w.r.t. all A coordinates).
//
// ds_dR may be null for a value-only evaluation. When non-null it must hold
// na3 elements and is zero-initialized by this function.
// ---------------------------------------------------------------------------
template <typename T>
__device__ T scalar_noalchemy_and_grad_full(
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
    const JacParams<T> &params,
    const T *s_prefactor,
    int na3,
    T *ds_dR
) {
    const bool want_grad = (ds_dR != nullptr);
    if (want_grad) {
        for (int a = 0; a < na3; ++a) {
            ds_dR[a] = Math<T>::zero();
        }
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
    const bool order2 = (order == 1 || order == 2);
    const T s0 = s_prefactor[0];

    T aadist = Math<T>::one();

    for (int i = 1; i < n_neigh1; ++i) {
        const int Zi = static_cast<int>(x1_chan[max_size1 + i]);
        const T ri = x1_chan[i];
        const long long i_na3 = static_cast<long long>(i) * na3;
        const T ksi1_i = ksi1[i];

        for (int j = 1; j < n_neigh2; ++j) {
            const int Zj = static_cast<int>(x2_chan[max_size2 + j]);
            if (Zi != Zj) {
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
            if (order2) {
                for (int p = 0; p < pmax; ++p) {
                    const long long i1 = jac_fourier_idx(p, 0, i, order, max_size1);
                    const long long i2 = jac_fourier_idx(p, 0, j, order, max_size2);
                    angular += (cos1[i1] * cos2[i2] + sin1[i1] * sin2[i2]) * s0;
                }
            } else {
                for (int m = 0; m < order; m += 2) {
                    const T spf = s_prefactor[m];
                    T ang_m = Math<T>::zero();
                    for (int p = 0; p < pmax; ++p) {
                        const long long i1 = jac_fourier_idx(p, m, i, order, max_size1);
                        const long long i2 = jac_fourier_idx(p, m, j, order, max_size2);
                        ang_m += cos1[i1] * cos2[i2] + sin1[i1] * sin2[i2];
                    }
                    angular += ang_m * spf;
                }
            }

            const T ksi2_j = ksi2[j];
            const T W = ksi1_i * ksi2_j * dist_scale + angular * ang_scale;
            aadist += G * W;

            if (!want_grad) {
                continue;
            }

            const T G_dist = G * dist_scale;
            const T G_ang = G * ang_scale;

            for (int amu = 0; amu < na3; ++amu) {
                const T dG_dR = dG_dri * dri1[i_na3 + amu];
                const T dW_ksi = dksi1[i_na3 + amu] * ksi2_j;
                ds_dR[amu] += dG_dR * W + G_dist * dW_ksi;
            }

            if (order2) {
                for (int p = 0; p < pmax; ++p) {
                    const long long i1 = jac_fourier_idx(p, 0, i, order, max_size1);
                    const long long i2 = jac_fourier_idx(p, 0, j, order, max_size2);
                    const T c2 = cos2[i2];
                    const T s2 = sin2[i2];
                    const long long gbase = i1 * na3;
                    for (int amu = 0; amu < na3; ++amu) {
                        ds_dR[amu] +=
                            G_ang * s0 * (dcos1[gbase + amu] * c2 + dsin1[gbase + amu] * s2);
                    }
                }
            } else {
                for (int m = 0; m < order; m += 2) {
                    const T spf = s_prefactor[m];
                    for (int p = 0; p < pmax; ++p) {
                        const long long i1 = jac_fourier_idx(p, m, i, order, max_size1);
                        const long long i2 = jac_fourier_idx(p, m, j, order, max_size2);
                        const T c2 = cos2[i2];
                        const T s2 = sin2[i2];
                        const long long gbase = i1 * na3;
                        for (int amu = 0; amu < na3; ++amu) {
                            ds_dR[amu] += G_ang * spf *
                                          (dcos1[gbase + amu] * c2 + dsin1[gbase + amu] * s2);
                        }
                    }
                }
            }
        }
    }

    return aadist;
}

// Single-component variant for the A-side self-scalar kernel (threads split
// over amu). The hot pair kernel uses scalar_noalchemy_and_grad_full instead.
template <typename T>
__device__ T scalar_noalchemy_and_grad_amu(
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
    const JacParams<T> &params,
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
    const bool order2 = (order == 1 || order == 2);
    const T s0 = s_prefactor[0];

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
            if (order2) {
                for (int p = 0; p < pmax; ++p) {
                    const long long i1 = jac_fourier_idx(p, 0, i, order, max_size1);
                    const long long i2 = jac_fourier_idx(p, 0, j, order, max_size2);
                    const T c2 = cos2[i2];
                    const T s2 = sin2[i2];
                    angular += (cos1[i1] * c2 + sin1[i1] * s2) * s0;
                    if (want_grad) {
                        d_angular += (dcos1[i1 * na3 + amu] * c2 + dsin1[i1 * na3 + amu] * s2) * s0;
                    }
                }
            } else {
                for (int m = 0; m < order; m += 2) {
                    const T spf = s_prefactor[m];
                    T ang_m = Math<T>::zero();
                    T dang_m = Math<T>::zero();
                    for (int p = 0; p < pmax; ++p) {
                        const long long i1 = jac_fourier_idx(p, m, i, order, max_size1);
                        const long long i2 = jac_fourier_idx(p, m, j, order, max_size2);
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
// Kernels
// ---------------------------------------------------------------------------

template <typename T>
__global__ void jac_mark_elements_kernel(
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
__global__ void jac_precompute_b_kernel(
    const T *x2,
    const int *n2,
    const int *nn2,
    T *ksi_all,
    T *cosp_all,
    T *sinp_all,
    int nm2,
    int max_size2,
    JacParams<T> params,
    const int *z_to_idx
) {
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= nm2 * max_size2) {
        return;
    }
    const int b = flat_idx / max_size2;
    const int j = flat_idx % max_size2;
    if (j >= n2[b]) {
        return;
    }

    const long long fstride =
        fourier_atom_stride(params.pmax, params.fourier_order, max_size2);
    const T *atom_chan = atom_slice(x2, max_size2, b, j);
    const int n_neigh = nn2[flat_idx];

    compute_ksi_value_device(
        atom_chan,
        max_size2,
        n_neigh,
        params.two_body_power,
        params.cut_start,
        params.cut_distance,
        ksi_all + static_cast<long long>(flat_idx) * max_size2
    );
    compute_threebody_fourier_value_device(
        atom_chan,
        max_size2,
        n_neigh,
        params.three_body_power,
        params.cut_start,
        params.cut_distance,
        params.fourier_order,
        z_to_idx,
        params.use_atm != 0,
        cosp_all + static_cast<long long>(flat_idx) * fstride,
        sinp_all + static_cast<long long>(flat_idx) * fstride
    );
}

template <typename T>
__global__ void jac_precompute_a_kernel(
    const T *x1,
    const int *n1,
    const int *nn1,
    const T *coords1,
    const int *z1,
    int *nbr_all,
    T *ksi_all,
    T *dksi_all,
    T *dri_all,
    T *cosp_all,
    T *sinp_all,
    T *dcosp_all,
    T *dsinp_all,
    int nm1,
    int max_size1,
    JacParams<T> params,
    const int *z_to_idx
) {
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= nm1 * max_size1) {
        return;
    }
    const int a = flat_idx / max_size1;
    const int i = flat_idx % max_size1;
    const int na = n1[a];
    if (i >= na) {
        return;
    }

    const int na3 = na * 3;
    const long long na3_max = 3LL * max_size1;
    const long long fstride =
        fourier_atom_stride(params.pmax, params.fourier_order, max_size1);

    const T *atom_chan = atom_slice(x1, max_size1, a, i);
    const int n_neigh = nn1[flat_idx];

    int *nbr = nbr_all + static_cast<long long>(flat_idx) * max_size1;
    T *ksi = ksi_all + static_cast<long long>(flat_idx) * max_size1;
    T *dksi = dksi_all + static_cast<long long>(flat_idx) * max_size1 * na3_max;
    T *dri = dri_all + static_cast<long long>(flat_idx) * max_size1 * na3_max;
    T *cosp = cosp_all + static_cast<long long>(flat_idx) * fstride;
    T *sinp = sinp_all + static_cast<long long>(flat_idx) * fstride;
    T *dcosp = dcosp_all + static_cast<long long>(flat_idx) * fstride * na3_max;
    T *dsinp = dsinp_all + static_cast<long long>(flat_idx) * fstride * na3_max;

    build_nbr_atom_idx_device(
        atom_chan,
        max_size1,
        n_neigh,
        coords1 + static_cast<long long>(a) * max_size1 * 3,
        z1 + static_cast<long long>(a) * max_size1,
        i,
        na,
        nbr
    );

    compute_ksi_and_grad_device(
        atom_chan,
        max_size1,
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

    compute_threebody_fourier_and_grad_device(
        atom_chan,
        max_size1,
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

template <typename T>
__global__ void jac_self_scalar_b_kernel(
    const T *x2,
    const int *n2,
    const int *nn2,
    const T *ksi_all,
    const T *cosp_all,
    const T *sinp_all,
    T *ss_out,
    int nm2,
    int max_size2,
    JacParams<T> params,
    const T *s_prefactor
) {
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= nm2 * max_size2) {
        return;
    }
    const int b = flat_idx / max_size2;
    const int j = flat_idx % max_size2;
    if (j >= n2[b]) {
        ss_out[flat_idx] = Math<T>::zero();
        return;
    }

    const long long fstride =
        fourier_atom_stride(params.pmax, params.fourier_order, max_size2);
    const T *x_chan = atom_slice(x2, max_size2, b, j);
    const int n_neigh = nn2[flat_idx];
    const T *ksi = ksi_all + static_cast<long long>(flat_idx) * max_size2;
    const T *cosp = cosp_all + static_cast<long long>(flat_idx) * fstride;
    const T *sinp = sinp_all + static_cast<long long>(flat_idx) * fstride;

    T dummy = Math<T>::zero();
    const T *no_grad = nullptr;
    ss_out[flat_idx] = scalar_noalchemy_and_grad_amu(
        x_chan,
        max_size2,
        n_neigh,
        ksi,
        no_grad,
        no_grad,
        cosp,
        sinp,
        no_grad,
        no_grad,
        x_chan,
        max_size2,
        n_neigh,
        ksi,
        cosp,
        sinp,
        params,
        s_prefactor,
        0,
        -1,
        &dummy
    );
}

// One block per query atom (a, i); threads split the na3 gradient components.
// dss = 2 * d(s_ii)/dR, matching the CPU factor.
template <typename T>
__global__ void jac_self_scalar_a_kernel(
    const T *x1,
    const int *n1,
    const int *nn1,
    const T *ksi_all,
    const T *dksi_all,
    const T *dri_all,
    const T *cosp_all,
    const T *sinp_all,
    const T *dcosp_all,
    const T *dsinp_all,
    T *ss_out,
    T *dss_out,
    int nm1,
    int max_size1,
    JacParams<T> params,
    const T *s_prefactor
) {
    const int flat_idx = static_cast<int>(blockIdx.x);
    if (flat_idx >= nm1 * max_size1) {
        return;
    }
    const int a = flat_idx / max_size1;
    const int i = flat_idx % max_size1;
    const int na = n1[a];
    if (i >= na) {
        if (threadIdx.x == 0) {
            ss_out[flat_idx] = Math<T>::zero();
        }
        return;
    }

    const int na3 = na * 3;
    const long long na3_max = 3LL * max_size1;
    const long long fstride =
        fourier_atom_stride(params.pmax, params.fourier_order, max_size1);

    const T *x_chan = atom_slice(x1, max_size1, a, i);
    const int n_neigh = nn1[flat_idx];
    const T *ksi = ksi_all + static_cast<long long>(flat_idx) * max_size1;
    const T *dksi = dksi_all + static_cast<long long>(flat_idx) * max_size1 * na3_max;
    const T *dri = dri_all + static_cast<long long>(flat_idx) * max_size1 * na3_max;
    const T *cosp = cosp_all + static_cast<long long>(flat_idx) * fstride;
    const T *sinp = sinp_all + static_cast<long long>(flat_idx) * fstride;
    const T *dcosp = dcosp_all + static_cast<long long>(flat_idx) * fstride * na3_max;
    const T *dsinp = dsinp_all + static_cast<long long>(flat_idx) * fstride * na3_max;
    T *dss = dss_out + static_cast<long long>(flat_idx) * na3_max;

    T ss = Math<T>::zero();
    for (int amu = static_cast<int>(threadIdx.x); amu < na3;
         amu += static_cast<int>(blockDim.x)) {
        T ds = Math<T>::zero();
        ss = scalar_noalchemy_and_grad_amu(
            x_chan,
            max_size1,
            n_neigh,
            ksi,
            dksi,
            dri,
            cosp,
            sinp,
            dcosp,
            dsinp,
            x_chan,
            max_size1,
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

// B-side atom-minor packs: layout [b][neigh][atom] so consecutive j in a warp
// coalesce on neighbour reads (the representation atom stride is 5*max_size).
__device__ inline long long jac_pack_bj(int b, int neigh, int atom, int max_size) {
    return (static_cast<long long>(b) * max_size + neigh) * max_size + atom;
}

__device__ inline long long jac_pack_fourier(
    int b, int p, int m, int neigh, int atom, int pmax, int order, int max_size
) {
    return ((((static_cast<long long>(b) * pmax + p) * order + m) * max_size + neigh) *
            max_size) +
           atom;
}


// Hot path: one block per (A,B) molecule pair.
//
// fp32: one neighbour pass fills compile-time register tiles ds0..ds7 (no local
// memory). Threads own flat (i,j) atom pairs; contributions reduce into shared
// memory via atomicAdd. B-side neighbour fields are read from atom-minor packed
// buffers (see jac_pack_b_kernel) so consecutive j in a warp coalesce.
// fp64 / na3>64: amu-strided fallback.
template <typename T>
__device__ inline void jac_accum_tile8(
    T ds[kAmuTile],
    int amu_base,
    int na3,
    T dG_dri,
    T W,
    T G_dist,
    T ksi2_j,
    const T *dri1_row,
    const T *dksi1_row
) {
#pragma unroll
    for (int t = 0; t < kAmuTile; ++t) {
        const int amu = amu_base + t;
        if (amu < na3) {
            ds[t] += dG_dri * dri1_row[amu] * W + G_dist * dksi1_row[amu] * ksi2_j;
        }
    }
}

template <typename T>
__device__ inline void jac_accum_ang_tile8(
    T ds[kAmuTile],
    int amu_base,
    int na3,
    T scale,
    const T *dcos_row,
    const T *dsin_row,
    T c2,
    T s2
) {
#pragma unroll
    for (int t = 0; t < kAmuTile; ++t) {
        const int amu = amu_base + t;
        if (amu < na3) {
            ds[t] += scale * (dcos_row[amu] * c2 + dsin_row[amu] * s2);
        }
    }
}


// JacStoreLayout: 0 = RowsAreCoords J[d_a_rows, nm2]; 1 = ColsAreCoords J[nm2, d_a_rows].
template <typename T>
__global__ void __launch_bounds__(kHotBlockSize, 4) jac_hot_kernel(
    const T *x1,
    const T *x2,
    const int *n1,
    const int *n2,
    const int *nn1,
    const int *nn2,
    const int *row_offset,
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
    const T *cosp2,
    const T *sinp2,
    const T *ss2,
    const T *pack_dist2,
    const T *pack_z2,
    const T *pack_ksi2,
    const T *pack_cos2,
    const T *pack_sin2,
    T *jac_out,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int cols_are_coords,
    JacParams<T> params,
    const T *s_prefactor
) {
    __shared__ T smem_grad[kMaxNa3Local];

    const int pair = static_cast<int>(blockIdx.x);
    if (pair >= nm1 * nm2) {
        return;
    }
    const int a = pair / nm2;
    const int b = pair - a * nm2;

    const int na = n1[a];
    const int nb = n2[b];
    if (na <= 0) {
        return;
    }

    const int na3 = na * 3;
    const int tid = static_cast<int>(threadIdx.x);
    const long long na3_max = 3LL * max_size1;
    const long long fstride1 =
        fourier_atom_stride(params.pmax, params.fourier_order, max_size1);
    const long long fstride2 =
        fourier_atom_stride(params.pmax, params.fourier_order, max_size2);
    const long long row0 = row_offset[a];

    const bool use_tiles = (sizeof(T) == 4) && (na3 <= kMaxNa3Local);

    if (!use_tiles) {
        for (int amu = tid; amu < na3; amu += static_cast<int>(blockDim.x)) {
            T acc = Math<T>::zero();
            for (int i = 0; i < na; ++i) {
                const int fi = a * max_size1 + i;
                const T *x1_chan = atom_slice(x1, max_size1, a, i);
                const int n_neigh_i = nn1[fi];
                const T sii = ss1[fi];
                const T dsii = dss1[static_cast<long long>(fi) * na3_max + amu];
                for (int j = 0; j < nb; ++j) {
                    const int fj = b * max_size2 + j;
                    T dsij = Math<T>::zero();
                    const T sij = scalar_noalchemy_and_grad_amu(
                        x1_chan,
                        max_size1,
                        n_neigh_i,
                        ksi1 + static_cast<long long>(fi) * max_size1,
                        dksi1 + static_cast<long long>(fi) * max_size1 * na3_max,
                        dri1 + static_cast<long long>(fi) * max_size1 * na3_max,
                        cosp1 + static_cast<long long>(fi) * fstride1,
                        sinp1 + static_cast<long long>(fi) * fstride1,
                        dcosp1 + static_cast<long long>(fi) * fstride1 * na3_max,
                        dsinp1 + static_cast<long long>(fi) * fstride1 * na3_max,
                        atom_slice(x2, max_size2, b, j),
                        max_size2,
                        nn2[fj],
                        ksi2 + static_cast<long long>(fj) * max_size2,
                        cosp2 + static_cast<long long>(fj) * fstride2,
                        sinp2 + static_cast<long long>(fj) * fstride2,
                        params,
                        s_prefactor,
                        na3,
                        amu,
                        &dsij
                    );
                    const T kij =
                        Math<T>::exp_(-(sii + ss2[fj] - T(2) * sij) * params.inv_sigma2);
                    acc += kij * params.inv_sigma2 * (-dsii + T(2) * dsij);
                }
            }
            if (cols_are_coords) {
                jac_out[static_cast<long long>(b) * d_a_rows + (row0 + amu)] = acc;
            } else {
                jac_out[(row0 + amu) * nm2 + b] = acc;
            }
        }
        return;
    }

    for (int amu = tid; amu < na3; amu += static_cast<int>(blockDim.x)) {
        smem_grad[amu] = Math<T>::zero();
    }
    __syncthreads();

    const int order = params.fourier_order;
    const int pmax = params.pmax;
    const bool order2 = (order == 1 || order == 2);
    const T s0 = s_prefactor[0];
    const T d_width = params.two_body_width;
    const T inv_width = -Math<T>::one() / (T(4) * d_width * d_width);
    const T maxgausdist2 = (T(8) * d_width) * (T(8) * d_width);
    const T dist_scale = params.true_distance_scale;
    const T ang_scale = params.true_angular_scale;

    const int n_atom_pairs = na * nb;
    for (int p = tid; p < n_atom_pairs; p += static_cast<int>(blockDim.x)) {
        const int i = p / nb;
        const int j = p - i * nb;

        const int fi = a * max_size1 + i;
        const int fj = b * max_size2 + j;
        const T *x1_chan = atom_slice(x1, max_size1, a, i);
        const int n_neigh_i = nn1[fi];
        const int n_neigh_j = nn2[fj];
        const T sii = ss1[fi];
        const T sjj = ss2[fj];
        const T *dsii = dss1 + static_cast<long long>(fi) * na3_max;
        const T *ksi_i = ksi1 + static_cast<long long>(fi) * max_size1;
        const T *dksi_i = dksi1 + static_cast<long long>(fi) * max_size1 * na3_max;
        const T *dri_i = dri1 + static_cast<long long>(fi) * max_size1 * na3_max;
        const T *cos_i = cosp1 + static_cast<long long>(fi) * fstride1;
        const T *sin_i = sinp1 + static_cast<long long>(fi) * fstride1;
        const T *dcos_i = dcosp1 + static_cast<long long>(fi) * fstride1 * na3_max;
        const T *dsin_i = dsinp1 + static_cast<long long>(fi) * fstride1 * na3_max;

        T ds0[kAmuTile], ds1[kAmuTile], ds2[kAmuTile], ds3[kAmuTile];
        T ds4[kAmuTile], ds5[kAmuTile], ds6[kAmuTile], ds7[kAmuTile];
#pragma unroll
        for (int t = 0; t < kAmuTile; ++t) {
            ds0[t] = ds1[t] = ds2[t] = ds3[t] = Math<T>::zero();
            ds4[t] = ds5[t] = ds6[t] = ds7[t] = Math<T>::zero();
        }

        T aadist = Math<T>::zero();
        const int Z1 = static_cast<int>(x1_chan[max_size1]);
        const int Z2 = static_cast<int>(pack_z2[jac_pack_bj(b, 0, j, max_size2)]);
        if (Z1 == Z2) {
            aadist = Math<T>::one();
            for (int ii = 1; ii < n_neigh_i; ++ii) {
                const int Zi = static_cast<int>(x1_chan[max_size1 + ii]);
                const T ri = x1_chan[ii];
                const long long i_na3 = static_cast<long long>(ii) * na3;
                const T ksi1_i = ksi_i[ii];
                const T *dri_row = dri_i + i_na3;
                const T *dksi_row = dksi_i + i_na3;

                for (int jj = 1; jj < n_neigh_j; ++jj) {
                    const long long bj = jac_pack_bj(b, jj, j, max_size2);
                    if (Zi != static_cast<int>(pack_z2[bj])) {
                        continue;
                    }
                    const T dr = ri - pack_dist2[bj];
                    const T r2 = dr * dr;
                    if (r2 >= maxgausdist2) {
                        continue;
                    }
                    const T G = Math<T>::exp_(r2 * inv_width);
                    const T dG_dri = G * T(2) * dr * inv_width;

                    T angular = Math<T>::zero();
                    if (order2) {
                        for (int pp = 0; pp < pmax; ++pp) {
                            const long long i1 = jac_fourier_idx(pp, 0, ii, order, max_size1);
                            const long long i2 = jac_pack_fourier(
                                b, pp, 0, jj, j, pmax, order, max_size2
                            );
                            angular +=
                                (cos_i[i1] * pack_cos2[i2] + sin_i[i1] * pack_sin2[i2]) * s0;
                        }
                    } else {
                        for (int m = 0; m < order; m += 2) {
                            const T spf = s_prefactor[m];
                            T ang_m = Math<T>::zero();
                            for (int pp = 0; pp < pmax; ++pp) {
                                const long long i1 =
                                    jac_fourier_idx(pp, m, ii, order, max_size1);
                                const long long i2 = jac_pack_fourier(
                                    b, pp, m, jj, j, pmax, order, max_size2
                                );
                                ang_m += cos_i[i1] * pack_cos2[i2] + sin_i[i1] * pack_sin2[i2];
                            }
                            angular += ang_m * spf;
                        }
                    }

                    const T ksi2_j = pack_ksi2[bj];
                    const T W = ksi1_i * ksi2_j * dist_scale + angular * ang_scale;
                    aadist += G * W;
                    const T G_dist = G * dist_scale;
                    const T G_ang = G * ang_scale;

                    jac_accum_tile8(ds0, 0, na3, dG_dri, W, G_dist, ksi2_j, dri_row, dksi_row);
                    jac_accum_tile8(ds1, 8, na3, dG_dri, W, G_dist, ksi2_j, dri_row, dksi_row);
                    jac_accum_tile8(ds2, 16, na3, dG_dri, W, G_dist, ksi2_j, dri_row, dksi_row);
                    jac_accum_tile8(ds3, 24, na3, dG_dri, W, G_dist, ksi2_j, dri_row, dksi_row);
                    jac_accum_tile8(ds4, 32, na3, dG_dri, W, G_dist, ksi2_j, dri_row, dksi_row);
                    jac_accum_tile8(ds5, 40, na3, dG_dri, W, G_dist, ksi2_j, dri_row, dksi_row);
                    jac_accum_tile8(ds6, 48, na3, dG_dri, W, G_dist, ksi2_j, dri_row, dksi_row);
                    jac_accum_tile8(ds7, 56, na3, dG_dri, W, G_dist, ksi2_j, dri_row, dksi_row);

                    if (order2) {
                        for (int pp = 0; pp < pmax; ++pp) {
                            const long long i1 = jac_fourier_idx(pp, 0, ii, order, max_size1);
                            const long long i2 = jac_pack_fourier(
                                b, pp, 0, jj, j, pmax, order, max_size2
                            );
                            const T c2 = pack_cos2[i2];
                            const T s2 = pack_sin2[i2];
                            const T *dcos_row = dcos_i + i1 * na3;
                            const T *dsin_row = dsin_i + i1 * na3;
                            const T scale = G_ang * s0;
                            jac_accum_ang_tile8(ds0, 0, na3, scale, dcos_row, dsin_row, c2, s2);
                            jac_accum_ang_tile8(ds1, 8, na3, scale, dcos_row, dsin_row, c2, s2);
                            jac_accum_ang_tile8(ds2, 16, na3, scale, dcos_row, dsin_row, c2, s2);
                            jac_accum_ang_tile8(ds3, 24, na3, scale, dcos_row, dsin_row, c2, s2);
                            jac_accum_ang_tile8(ds4, 32, na3, scale, dcos_row, dsin_row, c2, s2);
                            jac_accum_ang_tile8(ds5, 40, na3, scale, dcos_row, dsin_row, c2, s2);
                            jac_accum_ang_tile8(ds6, 48, na3, scale, dcos_row, dsin_row, c2, s2);
                            jac_accum_ang_tile8(ds7, 56, na3, scale, dcos_row, dsin_row, c2, s2);
                        }
                    } else {
                        for (int m = 0; m < order; m += 2) {
                            const T spf = s_prefactor[m];
                            for (int pp = 0; pp < pmax; ++pp) {
                                const long long i1 =
                                    jac_fourier_idx(pp, m, ii, order, max_size1);
                                const long long i2 = jac_pack_fourier(
                                    b, pp, m, jj, j, pmax, order, max_size2
                                );
                                const T c2 = pack_cos2[i2];
                                const T s2 = pack_sin2[i2];
                                const T *dcos_row = dcos_i + i1 * na3;
                                const T *dsin_row = dsin_i + i1 * na3;
                                const T scale = G_ang * spf;
                                jac_accum_ang_tile8(ds0, 0, na3, scale, dcos_row, dsin_row, c2, s2);
                                jac_accum_ang_tile8(ds1, 8, na3, scale, dcos_row, dsin_row, c2, s2);
                                jac_accum_ang_tile8(ds2, 16, na3, scale, dcos_row, dsin_row, c2, s2);
                                jac_accum_ang_tile8(ds3, 24, na3, scale, dcos_row, dsin_row, c2, s2);
                                jac_accum_ang_tile8(ds4, 32, na3, scale, dcos_row, dsin_row, c2, s2);
                                jac_accum_ang_tile8(ds5, 40, na3, scale, dcos_row, dsin_row, c2, s2);
                                jac_accum_ang_tile8(ds6, 48, na3, scale, dcos_row, dsin_row, c2, s2);
                                jac_accum_ang_tile8(ds7, 56, na3, scale, dcos_row, dsin_row, c2, s2);
                            }
                        }
                    }

                }
            }
        }

        const T kij = Math<T>::exp_(-(sii + sjj - T(2) * aadist) * params.inv_sigma2);
        const T coeff = kij * params.inv_sigma2;

#pragma unroll
        for (int t = 0; t < kAmuTile; ++t) {
            if (t < na3) {
                atomicAdd(smem_grad + t, coeff * (-dsii[t] + T(2) * ds0[t]));
            }
            if (8 + t < na3) {
                atomicAdd(smem_grad + 8 + t, coeff * (-dsii[8 + t] + T(2) * ds1[t]));
            }
            if (16 + t < na3) {
                atomicAdd(smem_grad + 16 + t, coeff * (-dsii[16 + t] + T(2) * ds2[t]));
            }
            if (24 + t < na3) {
                atomicAdd(smem_grad + 24 + t, coeff * (-dsii[24 + t] + T(2) * ds3[t]));
            }
            if (32 + t < na3) {
                atomicAdd(smem_grad + 32 + t, coeff * (-dsii[32 + t] + T(2) * ds4[t]));
            }
            if (40 + t < na3) {
                atomicAdd(smem_grad + 40 + t, coeff * (-dsii[40 + t] + T(2) * ds5[t]));
            }
            if (48 + t < na3) {
                atomicAdd(smem_grad + 48 + t, coeff * (-dsii[48 + t] + T(2) * ds6[t]));
            }
            if (56 + t < na3) {
                atomicAdd(smem_grad + 56 + t, coeff * (-dsii[56 + t] + T(2) * ds7[t]));
            }
        }
    }
    __syncthreads();

    for (int amu = tid; amu < na3; amu += static_cast<int>(blockDim.x)) {
        if (cols_are_coords) {
            jac_out[static_cast<long long>(b) * d_a_rows + (row0 + amu)] = smem_grad[amu];
        } else {
            jac_out[(row0 + amu) * nm2 + b] = smem_grad[amu];
        }
    }
}

template <typename T>
__global__ void jac_pack_b_kernel(
    const T *x2,
    const int *n2,
    const T *ksi2,
    const T *cosp2,
    const T *sinp2,
    T *pack_dist,
    T *pack_z,
    T *pack_ksi,
    T *pack_cos,
    T *pack_sin,
    int nm2,
    int max_size2,
    int pmax,
    int order
) {
    const int flat = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nm2 * max_size2) {
        return;
    }
    const int b = flat / max_size2;
    const int j = flat - b * max_size2;
    if (j >= n2[b]) {
        return;
    }

    const T *x2_chan = atom_slice(x2, max_size2, b, j);
    const T *ksi = ksi2 + static_cast<long long>(flat) * max_size2;
    const long long fstride = fourier_atom_stride(pmax, order, max_size2);
    const T *cosp = cosp2 + static_cast<long long>(flat) * fstride;
    const T *sinp = sinp2 + static_cast<long long>(flat) * fstride;

    for (int jj = 0; jj < max_size2; ++jj) {
        const long long dst = jac_pack_bj(b, jj, j, max_size2);
        pack_dist[dst] = x2_chan[jj];
        pack_z[dst] = x2_chan[max_size2 + jj];
        pack_ksi[dst] = ksi[jj];
    }
    for (int p = 0; p < pmax; ++p) {
        for (int m = 0; m < order; ++m) {
            for (int jj = 0; jj < max_size2; ++jj) {
                const long long src = jac_fourier_idx(p, m, jj, order, max_size2);
                const long long dst =
                    jac_pack_fourier(b, p, m, jj, j, pmax, order, max_size2);
                pack_cos[dst] = cosp[src];
                pack_sin[dst] = sinp[src];
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

template <typename T>
struct JacWorkspace {
    T *ksi1 = nullptr;
    T *dksi1 = nullptr;
    T *dri1 = nullptr;
    T *cosp1 = nullptr;
    T *sinp1 = nullptr;
    T *dcosp1 = nullptr;
    T *dsinp1 = nullptr;
    T *ss1 = nullptr;
    T *dss1 = nullptr;
    T *ksi2 = nullptr;
    T *cosp2 = nullptr;
    T *sinp2 = nullptr;
    T *ss2 = nullptr;
    T *pack_dist2 = nullptr;
    T *pack_z2 = nullptr;
    T *pack_ksi2 = nullptr;
    T *pack_cos2 = nullptr;
    T *pack_sin2 = nullptr;
    T *s_prefactor = nullptr;
    int *nbr1 = nullptr;
    int *row_offset = nullptr;
    int *z_present = nullptr;
    int *z_to_idx = nullptr;
    size_t ksi1_cap = 0;
    size_t dksi1_cap = 0;
    size_t dri1_cap = 0;
    size_t cosp1_cap = 0;
    size_t sinp1_cap = 0;
    size_t dcosp1_cap = 0;
    size_t dsinp1_cap = 0;
    size_t ss1_cap = 0;
    size_t dss1_cap = 0;
    size_t ksi2_cap = 0;
    size_t cosp2_cap = 0;
    size_t sinp2_cap = 0;
    size_t ss2_cap = 0;
    size_t pack_dist2_cap = 0;
    size_t pack_z2_cap = 0;
    size_t pack_ksi2_cap = 0;
    size_t pack_cos2_cap = 0;
    size_t pack_sin2_cap = 0;
    size_t s_prefactor_cap = 0;
    size_t nbr1_cap = 0;
    size_t row_offset_cap = 0;
};

template <typename T>
JacWorkspace<T> &jac_workspace() {
    static JacWorkspace<T> ws;
    return ws;
}

template <typename T>
void kernel_gaussian_jacobian_cu_impl(
    const T *d_x1,
    const T *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const T *d_coords1,
    const int *d_z1,
    T *d_jac_out,
    T sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
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
    bool cols_are_coords
) {
    if (fourier_order <= 0 || fourier_order > kMaxFourierOrder) {
        throw std::invalid_argument("fourier_order must be in [1, 16] for cuda_fchl18_kernel");
    }
    if (nm1 <= 0 || nm2 <= 0 || d_a_rows <= 0) {
        return;
    }

    JacWorkspace<T> &ws = jac_workspace<T>();
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
        static_cast<int>((n_atoms1 + kPrecomputeBlockSize - 1) / kPrecomputeBlockSize);
    const int grid2 =
        static_cast<int>((n_atoms2 + kPrecomputeBlockSize - 1) / kPrecomputeBlockSize);

    jac_mark_elements_kernel<<<grid1, kPrecomputeBlockSize>>>(
        d_x1, d_n1, d_nn1, nm1, max_size1, ws.z_present
    );
    CUDA_CHECK(cudaGetLastError());
    jac_mark_elements_kernel<<<grid2, kPrecomputeBlockSize>>>(
        d_x2, d_n2, d_nn2, nm2, max_size2, ws.z_present
    );
    CUDA_CHECK(cudaGetLastError());

    int h_z_present[256];
    CUDA_CHECK(cudaMemcpy(h_z_present, ws.z_present, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    int h_z_to_idx[256];
    const int pmax = build_element_map_from_flags(h_z_present, h_z_to_idx);

    const size_t out_n = static_cast<size_t>(d_a_rows) * nm2;
    CUDA_CHECK(cudaMemset(d_jac_out, 0, out_n * sizeof(T)));
    if (pmax == 0) {
        return;
    }
    if (pmax > kMaxElements) {
        throw std::invalid_argument("too many distinct elements for cuda_fchl18_kernel");
    }
    CUDA_CHECK(
        cudaMemcpy(ws.z_to_idx, h_z_to_idx, 256 * sizeof(int), cudaMemcpyHostToDevice)
    );

    JacParams<T> params{};
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

    // Row offsets: molecule a owns rows [offset[a], offset[a] + 3*n1[a]).
    std::vector<int> h_n1(static_cast<size_t>(nm1));
    CUDA_CHECK(
        cudaMemcpy(h_n1.data(), d_n1, static_cast<size_t>(nm1) * sizeof(int),
                   cudaMemcpyDeviceToHost)
    );
    std::vector<int> h_row_offset(static_cast<size_t>(nm1) + 1, 0);
    for (int a = 0; a < nm1; ++a) {
        h_row_offset[static_cast<size_t>(a) + 1] =
            h_row_offset[static_cast<size_t>(a)] + h_n1[static_cast<size_t>(a)] * 3;
    }
    if (h_row_offset[static_cast<size_t>(nm1)] != d_a_rows) {
        throw std::invalid_argument("jacobian output rows must equal 3 * sum(n1)");
    }
    ensure_capacity(&ws.row_offset, &ws.row_offset_cap, static_cast<size_t>(nm1) + 1);
    CUDA_CHECK(cudaMemcpy(
        ws.row_offset,
        h_row_offset.data(),
        (static_cast<size_t>(nm1) + 1) * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    const size_t na3_max = static_cast<size_t>(max_size1) * 3;
    const size_t fstride1 =
        static_cast<size_t>(pmax) * fourier_order * static_cast<size_t>(max_size1);
    const size_t fstride2 =
        static_cast<size_t>(pmax) * fourier_order * static_cast<size_t>(max_size2);

    const size_t ksi1_need = n_atoms1 * static_cast<size_t>(max_size1);
    const size_t dksi1_need = ksi1_need * na3_max;
    const size_t cosp1_need = n_atoms1 * fstride1;
    const size_t dcosp1_need = cosp1_need * na3_max;
    const size_t dss1_need = n_atoms1 * na3_max;
    const size_t ksi2_need = n_atoms2 * static_cast<size_t>(max_size2);
    const size_t cosp2_need = n_atoms2 * fstride2;

    ensure_capacity(&ws.ksi1, &ws.ksi1_cap, ksi1_need);
    ensure_capacity(&ws.dksi1, &ws.dksi1_cap, dksi1_need);
    ensure_capacity(&ws.dri1, &ws.dri1_cap, dksi1_need);
    ensure_capacity(&ws.cosp1, &ws.cosp1_cap, cosp1_need);
    ensure_capacity(&ws.sinp1, &ws.sinp1_cap, cosp1_need);
    ensure_capacity(&ws.dcosp1, &ws.dcosp1_cap, dcosp1_need);
    ensure_capacity(&ws.dsinp1, &ws.dsinp1_cap, dcosp1_need);
    ensure_capacity(&ws.ss1, &ws.ss1_cap, n_atoms1);
    ensure_capacity(&ws.dss1, &ws.dss1_cap, dss1_need);
    ensure_capacity(&ws.nbr1, &ws.nbr1_cap, ksi1_need);
    ensure_capacity(&ws.ksi2, &ws.ksi2_cap, ksi2_need);
    ensure_capacity(&ws.cosp2, &ws.cosp2_cap, cosp2_need);
    ensure_capacity(&ws.sinp2, &ws.sinp2_cap, cosp2_need);
    ensure_capacity(&ws.ss2, &ws.ss2_cap, n_atoms2);

    const bool pack_b = (sizeof(T) == 4);
    const size_t pack_bj_need =
        static_cast<size_t>(nm2) * static_cast<size_t>(max_size2) * static_cast<size_t>(max_size2);
    const size_t pack_f_need = pack_bj_need * static_cast<size_t>(pmax) * fourier_order;
    if (pack_b) {
        ensure_capacity(&ws.pack_dist2, &ws.pack_dist2_cap, pack_bj_need);
        ensure_capacity(&ws.pack_z2, &ws.pack_z2_cap, pack_bj_need);
        ensure_capacity(&ws.pack_ksi2, &ws.pack_ksi2_cap, pack_bj_need);
        ensure_capacity(&ws.pack_cos2, &ws.pack_cos2_cap, pack_f_need);
        ensure_capacity(&ws.pack_sin2, &ws.pack_sin2_cap, pack_f_need);
    }

    // The precompute kernels accumulate, and padding atoms are never written.
    CUDA_CHECK(cudaMemset(ws.dksi1, 0, dksi1_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(ws.dri1, 0, dksi1_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(ws.cosp1, 0, cosp1_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(ws.sinp1, 0, cosp1_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(ws.dcosp1, 0, dcosp1_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(ws.dsinp1, 0, dcosp1_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(ws.dss1, 0, dss1_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(ws.cosp2, 0, cosp2_need * sizeof(T)));
    CUDA_CHECK(cudaMemset(ws.sinp2, 0, cosp2_need * sizeof(T)));

    jac_precompute_b_kernel<<<grid2, kPrecomputeBlockSize>>>(
        d_x2, d_n2, d_nn2, ws.ksi2, ws.cosp2, ws.sinp2, nm2, max_size2, params, ws.z_to_idx
    );
    CUDA_CHECK(cudaGetLastError());

    jac_precompute_a_kernel<<<grid1, kPrecomputeBlockSize>>>(
        d_x1,
        d_n1,
        d_nn1,
        d_coords1,
        d_z1,
        ws.nbr1,
        ws.ksi1,
        ws.dksi1,
        ws.dri1,
        ws.cosp1,
        ws.sinp1,
        ws.dcosp1,
        ws.dsinp1,
        nm1,
        max_size1,
        params,
        ws.z_to_idx
    );
    CUDA_CHECK(cudaGetLastError());

    jac_self_scalar_b_kernel<<<grid2, kPrecomputeBlockSize>>>(
        d_x2,
        d_n2,
        d_nn2,
        ws.ksi2,
        ws.cosp2,
        ws.sinp2,
        ws.ss2,
        nm2,
        max_size2,
        params,
        ws.s_prefactor
    );
    CUDA_CHECK(cudaGetLastError());

    jac_self_scalar_a_kernel<<<static_cast<int>(n_atoms1), kJacBlockSize>>>(
        d_x1,
        d_n1,
        d_nn1,
        ws.ksi1,
        ws.dksi1,
        ws.dri1,
        ws.cosp1,
        ws.sinp1,
        ws.dcosp1,
        ws.dsinp1,
        ws.ss1,
        ws.dss1,
        nm1,
        max_size1,
        params,
        ws.s_prefactor
    );
    CUDA_CHECK(cudaGetLastError());

    if (pack_b) {
        jac_pack_b_kernel<<<grid2, kPrecomputeBlockSize>>>(
            d_x2,
            d_n2,
            ws.ksi2,
            ws.cosp2,
            ws.sinp2,
            ws.pack_dist2,
            ws.pack_z2,
            ws.pack_ksi2,
            ws.pack_cos2,
            ws.pack_sin2,
            nm2,
            max_size2,
            pmax,
            fourier_order
        );
        CUDA_CHECK(cudaGetLastError());
    }


    jac_hot_kernel<<<nm1 * nm2, kHotBlockSize>>>(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        ws.row_offset,
        ws.ksi1,
        ws.dksi1,
        ws.dri1,
        ws.cosp1,
        ws.sinp1,
        ws.dcosp1,
        ws.dsinp1,
        ws.ss1,
        ws.dss1,
        ws.ksi2,
        ws.cosp2,
        ws.sinp2,
        ws.ss2,
        pack_b ? ws.pack_dist2 : nullptr,
        pack_b ? ws.pack_z2 : nullptr,
        pack_b ? ws.pack_ksi2 : nullptr,
        pack_b ? ws.pack_cos2 : nullptr,
        pack_b ? ws.pack_sin2 : nullptr,
        d_jac_out,
        nm1,
        nm2,
        max_size1,
        max_size2,
        d_a_rows,
        cols_are_coords ? 1 : 0,
        params,
        ws.s_prefactor
    );
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void kernel_gaussian_jacobian_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const float *d_coords1,
    const int *d_z1,
    float *d_jac_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
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
    kernel_gaussian_jacobian_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_coords1,
        d_z1,
        d_jac_out,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
        d_a_rows,
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
        /*cols_are_coords=*/false
    );
}

void kernel_gaussian_jacobian_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const double *d_coords1,
    const int *d_z1,
    double *d_jac_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
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
    kernel_gaussian_jacobian_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_coords1,
        d_z1,
        d_jac_out,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
        d_a_rows,
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
        /*cols_are_coords=*/false
    );
}

void kernel_gaussian_jacobian_t_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const float *d_coords1,
    const int *d_z1,
    float *d_jac_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
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
    kernel_gaussian_jacobian_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_coords1,
        d_z1,
        d_jac_out,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
        d_a_rows,
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
        /*cols_are_coords=*/true
    );
}

void kernel_gaussian_jacobian_t_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const double *d_coords1,
    const int *d_z1,
    double *d_jac_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
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
    kernel_gaussian_jacobian_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_coords1,
        d_z1,
        d_jac_out,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
        d_a_rows,
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
        /*cols_are_coords=*/true
    );
}

}  // namespace fchl18
}  // namespace kf
