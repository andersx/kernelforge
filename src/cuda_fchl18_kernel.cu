#include "cuda_fchl18_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

#include "cuda_fchl18_common.cuh"

namespace kf {
namespace fchl18 {

namespace {

int &fchl18_three_body_weight_mode_storage() {
    static int mode = kTbWeightProduct;
    return mode;
}

void set_fchl18_three_body_weight_mode_impl(int mode) {
    if (mode < kTbWeightProduct || mode > kTbWeightBondNormExpSum) {
        mode = kTbWeightProduct;
    }
    fchl18_three_body_weight_mode_storage() = mode;
}

int get_fchl18_three_body_weight_mode_impl() { return fchl18_three_body_weight_mode_storage(); }

}  // namespace

void set_fchl18_three_body_weight_mode(int mode) { set_fchl18_three_body_weight_mode_impl(mode); }

int get_fchl18_three_body_weight_mode() { return get_fchl18_three_body_weight_mode_impl(); }

namespace {

__global__ void fill_coord_offsets_kernel(
    const int *n, int *offset, int nm, int expected_dim, int *ok
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    offset[0] = 0;
    for (int a = 0; a < nm; ++a) {
        offset[a + 1] = offset[a] + n[a] * 3;
    }
    if (ok != nullptr) {
        *ok = (expected_dim < 0 || offset[nm] == expected_dim) ? 1 : 0;
    }
}

}  // namespace

void fill_coord_offsets_device(
    const int *d_n, int **d_offset, size_t *offset_cap, int nm, int expected_dim, const char *what
) {
    if (nm == 1 && expected_dim >= 0) {
        int h_offset[2] = {0, expected_dim};
        ensure_capacity(d_offset, offset_cap, 2);
        CUDA_CHECK(cudaMemcpy(*d_offset, h_offset, 2 * sizeof(int), cudaMemcpyHostToDevice));
        return;
    }
    ensure_capacity(d_offset, offset_cap, static_cast<size_t>(nm) + 1);
    if (expected_dim < 0) {
        fill_coord_offsets_kernel<<<1, 1>>>(d_n, *d_offset, nm, -1, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    int *d_ok = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ok, sizeof(int)));
    fill_coord_offsets_kernel<<<1, 1>>>(d_n, *d_offset, nm, expected_dim, d_ok);
    CUDA_CHECK(cudaGetLastError());
    int h_ok = 0;
    CUDA_CHECK(cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_ok));
    if (!h_ok) {
        throw std::invalid_argument(what);
    }
}

namespace {

constexpr int kPairBlockSize = 128;

// Slim params for hot scalar/rect kernels — no z_to_idx[256] by value
// (that alone is ~1KB and blew register pressure to ~106 regs/thread).
template <typename T>
struct ScalarParams {
    T inv_sigma2;
    T two_body_width;
    T true_distance_scale;
    T true_angular_scale;
    int fourier_order;
    int pmax;
    T s_prefactor[kMaxFourierOrder];
};

// Flat hot params without arrays (avoids local-memory spills from dynamic indexing).
template <typename T>
struct HotParams {
    T inv_sigma2;
    T two_body_width;
    T true_distance_scale;
    T true_angular_scale;
    T s0;
    T s1;
    int fourier_order;
    int pmax;
    // Periodic-table alchemy: Z-indexed pd table (emax x emax); nullptr when off.
    const T *pd;
    int emax;
};

template <typename T>
__host__ __device__ inline HotParams<T> to_hot(const ScalarParams<T> &s) {
    HotParams<T> h{};
    h.inv_sigma2 = s.inv_sigma2;
    h.two_body_width = s.two_body_width;
    h.true_distance_scale = s.true_distance_scale;
    h.true_angular_scale = s.true_angular_scale;
    h.s0 = s.s_prefactor[0];
    h.s1 = s.fourier_order >= 2 ? s.s_prefactor[1] : T(0);
    h.fourier_order = s.fourier_order;
    h.pmax = s.pmax;
    h.pd = nullptr;
    h.emax = 0;
    return h;
}

template <typename T>
struct KernelParams {
    T sigma;
    T inv_sigma2;
    T two_body_scaling;
    T two_body_width;
    T two_body_power;
    T three_body_scaling;
    T three_body_width;
    T three_body_power;
    T cut_start;
    T cut_distance;
    T true_distance_scale;
    T true_angular_scale;
    int fourier_order;
    int pmax;
    int use_atm;
    int three_body_weight_mode;
    T s_prefactor[kMaxFourierOrder];
    int z_to_idx[256];

    __host__ __device__ ScalarParams<T> to_scalar() const {
        ScalarParams<T> s{};
        s.inv_sigma2 = inv_sigma2;
        s.two_body_width = two_body_width;
        s.true_distance_scale = true_distance_scale;
        s.true_angular_scale = true_angular_scale;
        s.fourier_order = fourier_order;
        s.pmax = pmax;
        for (int m = 0; m < kMaxFourierOrder; ++m) {
            s.s_prefactor[m] = s_prefactor[m];
        }
        return s;
    }
};

template <typename T>
struct Workspace {
    T *x1_sorted = nullptr;
    T *x2_sorted = nullptr;
    T *ss1 = nullptr;
    T *ss2 = nullptr;
    T *ksi1 = nullptr;
    T *ksi2 = nullptr;
    T *cosp1 = nullptr;
    T *sinp1 = nullptr;
    T *cosp2 = nullptr;
    T *sinp2 = nullptr;
    // Alchemy: element-channel-mixed ("effective") Fourier buffers.
    T *cosp1e = nullptr;
    T *sinp1e = nullptr;
    T *cosp2e = nullptr;
    T *sinp2e = nullptr;
    T *s_prefactor = nullptr;
    int *z_present = nullptr;
    size_t x1_sorted_cap = 0;
    size_t x2_sorted_cap = 0;
    size_t ss1_cap = 0;
    size_t ss2_cap = 0;
    size_t ksi1_cap = 0;
    size_t ksi2_cap = 0;
    size_t cosp1_cap = 0;
    size_t sinp1_cap = 0;
    size_t cosp2_cap = 0;
    size_t sinp2_cap = 0;
    size_t cosp1e_cap = 0;
    size_t sinp1e_cap = 0;
    size_t cosp2e_cap = 0;
    size_t sinp2e_cap = 0;
    size_t s_prefactor_cap = 0;

    // Durable train-side precompute for repeated energy matvecs.
    T *train_x_sorted = nullptr;
    T *train_ss = nullptr;
    T *train_ksi = nullptr;
    T *train_cosp = nullptr;
    T *train_sinp = nullptr;
    T *train_cospe = nullptr;
    T *train_sinpe = nullptr;
    size_t train_x_sorted_cap = 0;
    size_t train_ss_cap = 0;
    size_t train_ksi_cap = 0;
    size_t train_cosp_cap = 0;
    size_t train_sinp_cap = 0;
    size_t train_cospe_cap = 0;
    size_t train_sinpe_cap = 0;
    bool train_valid = false;
    const T *train_x_ptr = nullptr;
    const int *train_n_ptr = nullptr;
    const int *train_nn_ptr = nullptr;
    int train_nm = 0;
    int train_max_size = 0;
    T train_sigma{};
    T train_two_body_scaling{};
    T train_two_body_width{};
    T train_two_body_power{};
    T train_three_body_scaling{};
    T train_three_body_width{};
    T train_three_body_power{};
    T train_cut_start{};
    T train_cut_distance{};
    int train_fourier_order = 0;
    bool train_use_atm = false;
    bool train_use_alchemy = false;
    double train_alchemy_period_width = 0.0;
    double train_alchemy_group_width = 0.0;
    int train_tb_weight_mode = 0;
    int train_pmax = 0;
    int train_z_to_idx[256]{};
};

template <typename T>
Workspace<T> &workspace() {
    static Workspace<T> ws;
    return ws;
}

// Atom-major Fourier layout: [neigh][p][m]
// Makes the (p,m) angular loop contiguous for a fixed neighbour.
__device__ inline long long fourier_idx(int p, int m, int neigh, int order, int pmax) {
    return static_cast<long long>(neigh) * pmax * order +
           static_cast<long long>(p) * order + m;
}

template <typename T>
__device__ void compute_ksi_device(
    const T *atom_chan,
    int max_size,
    int n_neigh,
    T power,
    T cut_start,
    T cut_distance,
    T *ksi_out
) {
    for (int k = 0; k < max_size; ++k) {
        ksi_out[k] = Math<T>::zero();
    }
    for (int k = 1; k < n_neigh; ++k) {
        const T r = atom_chan[k];
        if (r < Math<T>::eps()) {
            continue;
        }
        ksi_out[k] = cut_function(r, cut_start, cut_distance) / fast_pow(r, power);
    }
}

template <typename T, bool UseAtm>
__device__ void compute_threebody_fourier_device(
    const T *atom_chan,
    int max_size,
    int n_neigh,
    T three_body_power,
    T cut_start,
    T cut_distance,
    int order,
    int pmax,
    const int *z_to_idx,
    int three_body_weight_mode,
    T *cosp,
    T *sinp
) {
    const long long sz = fourier_atom_stride(pmax, order, max_size);
    for (long long t = 0; t < sz; ++t) {
        cosp[t] = Math<T>::zero();
        sinp[t] = Math<T>::zero();
    }

    const T *dist_chan = atom_chan;
    const T *z_chan = atom_chan + max_size;
    const T *xc = atom_chan + 2 * max_size;
    const T *yc = atom_chan + 3 * max_size;
    const T *zc = atom_chan + 4 * max_size;
    const int zi = static_cast<int>(z_chan[0]);

    for (int j = 1; j < n_neigh; ++j) {
        const T dj = dist_chan[j];
        if (dj < Math<T>::eps()) {
            continue;
        }
        const T cutj = cut_function(dj, cut_start, cut_distance);
        if (cutj == Math<T>::zero()) {
            continue;
        }
        const T inv_dj = Math<T>::one() / dj;
        const T uxj = xc[j] * inv_dj;
        const T uyj = yc[j] * inv_dj;
        const T uzj = zc[j] * inv_dj;

        for (int k = j + 1; k < n_neigh; ++k) {
            const T dk = dist_chan[k];
            if (dk < Math<T>::eps()) {
                continue;
            }
            const T cutk = cut_function(dk, cut_start, cut_distance);
            if (cutk == Math<T>::zero()) {
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

            const T cut_jk = cut_function(di, cut_start, cut_distance);
            if (cut_jk == Math<T>::zero()) {
                continue;
            }

            const T inv_dk = Math<T>::one() / dk;
            const T uxk = xc[k] * inv_dk;
            const T uyk = yc[k] * inv_dk;
            const T uzk = zc[k] * inv_dk;

            const T cos_i = clamp11(uxj * uxk + uyj * uyk + uzj * uzk);

            T atm = Math<T>::one();
            if constexpr (UseAtm) {
                const T inv_di = Math<T>::one() / di;
                const T ukj_x = -dxjk * inv_di;
                const T ukj_y = -dyjk * inv_di;
                const T ukj_z = -dzjk * inv_di;
                const T cos_j = clamp11(ukj_x * (-uxj) + ukj_y * (-uyj) + ukj_z * (-uzj));

                const T ujk_x = dxjk * inv_di;
                const T ujk_y = dyjk * inv_di;
                const T ujk_z = dzjk * inv_di;
                const T cos_k = clamp11(ujk_x * (-uxk) + ujk_y * (-uyk) + ujk_z * (-uzk));

                atm = Math<T>::one() + T(3) * cos_i * cos_j * cos_k;
            }
            if (atm == Math<T>::zero()) {
                continue;
            }

            const int zk = static_cast<int>(z_chan[k]);
            const int zj = static_cast<int>(z_chan[j]);
            if (zk <= 0 || zk >= 256 || zj <= 0 || zj >= 256) {
                continue;
            }
            const T r0_ij = static_cast<T>(pair_bond_r0(zi, zj));
            const T r0_ik = static_cast<T>(pair_bond_r0(zi, zk));
            const T r0_jk = static_cast<T>(pair_bond_r0(zj, zk));
            const auto radial = three_body_radial_weight(
                three_body_weight_mode, dj, dk, di, r0_ij, r0_ik, r0_jk, three_body_power
            );
            const T ksi3 = cutj * cutk * cut_jk * atm * radial.w;
            if (ksi3 == Math<T>::zero()) {
                continue;
            }

            const T theta = Math<T>::acos_(cos_i);

            const int pj = z_to_idx[zk];
            const int pk = z_to_idx[zj];
            if (pj < 0 || pk < 0) {
                continue;
            }

            for (int m = 0; m < order; ++m) {
                const T mf = static_cast<T>(m + 1);
                const T mth = mf * theta;
                const T mthpi = mf * (theta + Math<T>::pi());
                T cos_mth, sin_mth, cos_mthpi, sin_mthpi;
                Math<T>::sincos_(mth, &sin_mth, &cos_mth);
                Math<T>::sincos_(mthpi, &sin_mthpi, &cos_mthpi);
                const T cos_m = (cos_mth - cos_mthpi) * ksi3;
                const T sin_m = (sin_mth - sin_mthpi) * ksi3;

                const long long idx_j = fourier_idx(pj, m, j, order, pmax);
                const long long idx_k = fourier_idx(pk, m, k, order, pmax);
                cosp[idx_j] += cos_m;
                sinp[idx_j] += sin_m;
                cosp[idx_k] += cos_m;
                sinp[idx_k] += sin_m;
            }
        }
    }
}

// Atom-major Fourier: neighbor i's (p,m) block is contiguous of length pmax*order.
template <typename T>
__device__ inline T angular_dot_generic(
    const T *cos1,
    const T *sin1,
    const T *cos2,
    const T *sin2,
    int i,
    int j,
    int order,
    int pmax,
    const T *s_prefactor
) {
    const long long base1 = static_cast<long long>(i) * pmax * order;
    const long long base2 = static_cast<long long>(j) * pmax * order;
    T angular = Math<T>::zero();
#pragma unroll
    for (int m = 0; m < kMaxFourierOrder; ++m) {
        if (m >= order) {
            break;
        }
        T ang_m = Math<T>::zero();
        for (int p = 0; p < pmax; ++p) {
            const long long o1 = base1 + static_cast<long long>(p) * order + m;
            const long long o2 = base2 + static_cast<long long>(p) * order + m;
            ang_m += cos1[o1] * cos2[o2] + sin1[o1] * sin2[o2];
        }
        angular += ang_m * s_prefactor[m];
    }
    return angular;
}

// Hot path for fourier_order=1 (QML / kernelbench default): contiguous p stride.
template <typename T>
__device__ inline T angular_dot_order1(
    const T *cos1, const T *sin1, const T *cos2, const T *sin2, int pmax, T s0
) {
    T ang0 = Math<T>::zero();
#pragma unroll 4
    for (int p = 0; p < pmax; ++p) {
        ang0 += cos1[p] * cos2[p] + sin1[p] * sin2[p];
    }
    return ang0 * s0;
}

// Hot path for fourier_order=2: fully unrolled m, contiguous p stride.
// cos1/sin1 already point at neighbor-i's Fourier block; cos2/sin2 at neighbor-j.
template <typename T>
__device__ inline T angular_dot_order2(
    const T *cos1,
    const T *sin1,
    const T *cos2,
    const T *sin2,
    int pmax,
    T s0,
    T s1
) {
    T ang0 = Math<T>::zero();
    T ang1 = Math<T>::zero();
#pragma unroll 4
    for (int p = 0; p < pmax; ++p) {
        const int o = p * 2;
        ang0 += cos1[o] * cos2[o] + sin1[o] * sin2[o];
        ang1 += cos1[o + 1] * cos2[o + 1] + sin1[o + 1] * sin2[o + 1];
    }
    return ang0 * s0 + ang1 * s1;
}

template <typename T>
__device__ T scalar_noalchemy_precomputed(
    const T *x1_chan,
    int max_size1,
    int n1,
    const T *ksi1,
    const T *cos1,
    const T *sin1,
    const T *x2_chan,
    int max_size2,
    int n2,
    const T *ksi2,
    const T *cos2,
    const T *sin2,
    const HotParams<T> &params,
    const T *s_prefactor_full
) {
    const int z1_center = static_cast<int>(x1_chan[1 * max_size1 + 0]);
    const int z2_center = static_cast<int>(x2_chan[1 * max_size2 + 0]);
    if (z1_center != z2_center) {
        return Math<T>::zero();
    }

    const T inv_width = -Math<T>::one() / (T(4) * params.two_body_width * params.two_body_width);
    const T maxgausdist2 = (T(8) * params.two_body_width) * (T(8) * params.two_body_width);
    const int order = params.fourier_order;
    const int pmax = params.pmax;
    const T *z1 = x1_chan + max_size1;
    const T *z2 = x2_chan + max_size2;
    const T dist_scale = params.true_distance_scale;
    const T ang_scale = params.true_angular_scale;
    const T s0 = params.s0;
    const T s1 = params.s1;
    const bool order1 = (order == 1);
    const bool order2 = (order == 2);
    const int fblock = pmax * order;

    T aadist = Math<T>::one();

    // Neighbors are Z-sorted (center fixed at slot 0). Merge equal-Z runs.
    int i = 1;
    int j = 1;
    while (i < n1 && j < n2) {
        const int zi = static_cast<int>(z1[i]);
        const int zj = static_cast<int>(z2[j]);
        if (zi < zj) {
            ++i;
            continue;
        }
        if (zj < zi) {
            ++j;
            continue;
        }

        int i_end = i + 1;
        while (i_end < n1 && static_cast<int>(z1[i_end]) == zi) {
            ++i_end;
        }
        int j_end = j + 1;
        while (j_end < n2 && static_cast<int>(z2[j_end]) == zi) {
            ++j_end;
        }

        for (int ii = i; ii < i_end; ++ii) {
            const T ri = x1_chan[ii];
            const T ksi_i = ksi1[ii];
            const T *cos1_i = cos1 + static_cast<long long>(ii) * fblock;
            const T *sin1_i = sin1 + static_cast<long long>(ii) * fblock;
            for (int jj = j; jj < j_end; ++jj) {
                const T rj = x2_chan[jj];
                const T r2 = (rj - ri) * (rj - ri);
                if (r2 >= maxgausdist2) {
                    continue;
                }

                const T d = Math<T>::exp_(r2 * inv_width);
                T angular;
                if (order1) {
                    const T *cos2_j = cos2 + static_cast<long long>(jj) * fblock;
                    const T *sin2_j = sin2 + static_cast<long long>(jj) * fblock;
                    angular = angular_dot_order1(cos1_i, sin1_i, cos2_j, sin2_j, pmax, s0);
                } else if (order2) {
                    const T *cos2_j = cos2 + static_cast<long long>(jj) * fblock;
                    const T *sin2_j = sin2 + static_cast<long long>(jj) * fblock;
                    angular = angular_dot_order2(cos1_i, sin1_i, cos2_j, sin2_j, pmax, s0, s1);
                } else {
                    angular = angular_dot_generic(
                        cos1, sin1, cos2, sin2, ii, jj, order, pmax, s_prefactor_full
                    );
                }
                aadist += d * (ksi_i * ksi2[jj] * dist_scale + angular * ang_scale);
            }
        }

        i = i_end;
        j = j_end;
    }

    return aadist;
}

// Periodic-table alchemy scalar product (CPU scalar_alchemy). Neighbours are in
// representation order (no Z sort); every (i, j) pair contributes with weight
// pd(Zi, Zj); the whole scalar is scaled by pd(Zc1, Zc2). cos2/sin2 are the
// element-channel-mixed ("effective") side-2 Fourier buffers, so the angular term
// is the plain per-element dot product against side-1's raw buffers.
template <typename T>
__device__ T scalar_alchemy_precomputed(
    const T *x1_chan,
    int max_size1,
    int n1,
    const T *ksi1,
    const T *cos1,
    const T *sin1,
    const T *x2_chan,
    int max_size2,
    int n2,
    const T *ksi2,
    const T *cos2_eff,
    const T *sin2_eff,
    const HotParams<T> &params,
    const T *s_prefactor_full
) {
    const T *pd = params.pd;
    const int emax = params.emax;
    const int z1_center = static_cast<int>(x1_chan[1 * max_size1 + 0]);
    const int z2_center = static_cast<int>(x2_chan[1 * max_size2 + 0]);
    const T pd_central = alch_pd(pd, emax, z1_center, z2_center);
    if (pd_central == Math<T>::zero()) {
        return Math<T>::zero();
    }

    const T inv_width = -Math<T>::one() / (T(4) * params.two_body_width * params.two_body_width);
    const T maxgausdist2 = (T(8) * params.two_body_width) * (T(8) * params.two_body_width);
    const int order = params.fourier_order;
    const int pmax = params.pmax;
    const T *z1 = x1_chan + max_size1;
    const T *z2 = x2_chan + max_size2;
    const T dist_scale = params.true_distance_scale;
    const T ang_scale = params.true_angular_scale;
    const T s0 = params.s0;
    const T s1 = params.s1;
    const bool order1 = (order == 1);
    const bool order2 = (order == 2);
    const int fblock = pmax * order;

    T aadist = Math<T>::one();
    for (int ii = 1; ii < n1; ++ii) {
        const int zi = static_cast<int>(z1[ii]);
        const T ri = x1_chan[ii];
        const T ksi_i = ksi1[ii];
        const T *cos1_i = cos1 + static_cast<long long>(ii) * fblock;
        const T *sin1_i = sin1 + static_cast<long long>(ii) * fblock;
        for (int jj = 1; jj < n2; ++jj) {
            const T rj = x2_chan[jj];
            const T r2 = (rj - ri) * (rj - ri);
            if (r2 >= maxgausdist2) {
                continue;
            }
            const T pd_ij = alch_pd(pd, emax, zi, static_cast<int>(z2[jj]));
            if (pd_ij == Math<T>::zero()) {
                continue;
            }

            const T d = Math<T>::exp_(r2 * inv_width) * pd_ij;
            const T *cos2_j = cos2_eff + static_cast<long long>(jj) * fblock;
            const T *sin2_j = sin2_eff + static_cast<long long>(jj) * fblock;
            T angular;
            if (order1) {
                angular = angular_dot_order1(cos1_i, sin1_i, cos2_j, sin2_j, pmax, s0);
            } else if (order2) {
                angular = angular_dot_order2(cos1_i, sin1_i, cos2_j, sin2_j, pmax, s0, s1);
            } else {
                angular = angular_dot_generic(
                    cos1, sin1, cos2_eff, sin2_eff, ii, jj, order, pmax, s_prefactor_full
                );
            }
            aadist += d * (ksi_i * ksi2[jj] * dist_scale + angular * ang_scale);
        }
    }
    return aadist * pd_central;
}

template <typename T, bool Alch>
__device__ inline T scalar_dispatch_precomputed(
    const T *x1_chan,
    int max_size1,
    int n1,
    const T *ksi1,
    const T *cos1,
    const T *sin1,
    const T *x2_chan,
    int max_size2,
    int n2,
    const T *ksi2,
    const T *cos2,
    const T *sin2,
    const HotParams<T> &params,
    const T *s_prefactor_full
) {
    if constexpr (Alch) {
        return scalar_alchemy_precomputed(
            x1_chan, max_size1, n1, ksi1, cos1, sin1, x2_chan, max_size2, n2, ksi2, cos2, sin2,
            params, s_prefactor_full
        );
    } else {
        return scalar_noalchemy_precomputed(
            x1_chan, max_size1, n1, ksi1, cos1, sin1, x2_chan, max_size2, n2, ksi2, cos2, sin2,
            params, s_prefactor_full
        );
    }
}

template <typename T>
__global__ void mark_elements_kernel(
    const T *x, const int *n, const int *nn, int nm, int max_size, int *z_present
) {
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nm * max_size;
    if (flat_idx >= total) {
        return;
    }

    const int a = flat_idx / max_size;
    const int i = flat_idx % max_size;
    if (i >= n[a]) {
        return;
    }

    const T *atom_chan = atom_slice(x, max_size, a, i);
    const T *z_chan = atom_chan + max_size;
    const int n_neigh = nn[flat_idx];
    for (int k = 0; k < n_neigh; ++k) {
        const int z = static_cast<int>(z_chan[k]);
        if (z > 0 && z < 256) {
            atomicOr(z_present + z, 1);
        }
    }
}

// Keep the central atom fixed and group neighbor species to make the scalar
// kernel's zi == zj branch more coherent across a warp.
template <typename T>
__global__ void sort_neighbors_by_z_kernel(
    const T *x_in, const int *n, const int *nn, T *x_out, int nm, int max_size
) {
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nm * max_size;
    if (flat_idx >= total) {
        return;
    }

    const int a = flat_idx / max_size;
    const int i = flat_idx % max_size;
    // Padding atoms past n[a] are never read by later kernels — skip the copy.
    if (i >= n[a]) {
        return;
    }

    const T *src = atom_slice(x_in, max_size, a, i);
    T *dst = x_out + static_cast<long long>(flat_idx) * 5 * max_size;
    const int n_neigh = nn[flat_idx];
    // Only copy occupied neighbor slots (plus leave center at k=0).
    for (int channel = 0; channel < 5; ++channel) {
        for (int k = 0; k < n_neigh; ++k) {
            dst[channel * max_size + k] = src[channel * max_size + k];
        }
    }

    T *z_chan = dst + max_size;
    for (int k = 2; k < n_neigh; ++k) {
        // Save all channels of key k before shifting (dst[...,k] is overwritten).
        T saved[5];
        for (int channel = 0; channel < 5; ++channel) {
            saved[channel] = dst[channel * max_size + k];
        }
        const T z = saved[1];
        int insert_at = k;
        while (insert_at > 1 && z_chan[insert_at - 1] > z) {
            for (int channel = 0; channel < 5; ++channel) {
                dst[channel * max_size + insert_at] =
                    dst[channel * max_size + insert_at - 1];
            }
            --insert_at;
        }
        for (int channel = 0; channel < 5; ++channel) {
            dst[channel * max_size + insert_at] = saved[channel];
        }
    }
}

template <typename T>
__global__ void precompute_atom_data_kernel(
    const T *x,
    const int *n,
    const int *nn,
    T *ksi,
    T *cosp,
    T *sinp,
    int nm,
    int max_size,
    KernelParams<T> params
) {
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nm * max_size;
    if (flat_idx >= total) {
        return;
    }

    const int a = flat_idx / max_size;
    const int i = flat_idx % max_size;
    // Padding atoms are never consumed by pair kernels — skip zero-fills.
    if (i >= n[a]) {
        return;
    }

    const long long fstride = fourier_atom_stride(params.pmax, params.fourier_order, max_size);
    T *ksi_out = ksi + static_cast<long long>(flat_idx) * max_size;
    T *cosp_out = cosp + static_cast<long long>(flat_idx) * fstride;
    T *sinp_out = sinp + static_cast<long long>(flat_idx) * fstride;

    const T *atom_chan = atom_slice(x, max_size, a, i);
    const int n_neigh = nn[flat_idx];

    compute_ksi_device(
        atom_chan,
        max_size,
        n_neigh,
        params.two_body_power,
        params.cut_start,
        params.cut_distance,
        ksi_out
    );
    if (params.use_atm != 0) {
        compute_threebody_fourier_device<T, true>(
            atom_chan,
            max_size,
            n_neigh,
            params.three_body_power,
            params.cut_start,
            params.cut_distance,
            params.fourier_order,
            params.pmax,
            params.z_to_idx,
            params.three_body_weight_mode,
            cosp_out,
            sinp_out
        );
    } else {
        compute_threebody_fourier_device<T, false>(
            atom_chan,
            max_size,
            n_neigh,
            params.three_body_power,
            params.cut_start,
            params.cut_distance,
            params.fourier_order,
            params.pmax,
            params.z_to_idx,
            params.three_body_weight_mode,
            cosp_out,
            sinp_out
        );
    }
}

template <typename T, bool Alch>
__global__ void compute_self_scalars_precomputed_kernel(
    const T *x,
    const int *n,
    const int *nn,
    const T *ksi,
    const T *cosp,
    const T *sinp,
    const T *cospe,
    const T *sinpe,
    T *ss,
    int nm,
    int max_size,
    HotParams<T> params,
    const T *s_prefactor_full
) {
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nm * max_size;
    if (flat_idx >= total) {
        return;
    }

    const int a = flat_idx / max_size;
    const int i = flat_idx % max_size;
    if (i >= n[a]) {
        ss[flat_idx] = Math<T>::zero();
        return;
    }

    const T *x_chan = atom_slice(x, max_size, a, i);
    const int n_neigh = nn[flat_idx];
    const long long fstride = fourier_atom_stride(params.pmax, params.fourier_order, max_size);
    const T *ksi_i = ksi + static_cast<long long>(flat_idx) * max_size;
    const T *cos_i = cosp + static_cast<long long>(flat_idx) * fstride;
    const T *sin_i = sinp + static_cast<long long>(flat_idx) * fstride;
    const T *cose_i = cospe + static_cast<long long>(flat_idx) * fstride;
    const T *sine_i = sinpe + static_cast<long long>(flat_idx) * fstride;

    ss[flat_idx] = scalar_dispatch_precomputed<T, Alch>(
        x_chan,
        max_size,
        n_neigh,
        ksi_i,
        cos_i,
        sin_i,
        x_chan,
        max_size,
        n_neigh,
        ksi_i,
        cose_i,
        sine_i,
        params,
        s_prefactor_full
    );
}

__device__ inline void upper_pair_from_index(int p, int nm, int *a_out, int *b_out) {
    // Map flat p in [0, nm*(nm+1)/2) to (a, b) with 0 <= a <= b < nm.
    // cum(a) = number of pairs with first index < a = a*nm - a*(a-1)/2.
    int lo = 0;
    int hi = nm;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        const long long cum =
            static_cast<long long>(mid) * nm - static_cast<long long>(mid) * (mid - 1) / 2;
        if (cum <= p) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int a = lo;
    const long long cum_a =
        static_cast<long long>(a) * nm - static_cast<long long>(a) * (a - 1) / 2;
    *a_out = a;
    *b_out = a + static_cast<int>(p - cum_a);
}

// Device RFP index — matches kf::rfp_index_upper_N (TRANSR='N', UPLO='U').
// Precondition: 0 <= i <= j < n
__device__ inline long long rfp_index_upper_N_dev(int n, int i, int j) {
    const int k = n / 2;
    const int stride = (n % 2 == 0) ? (n + 1) : n;
    if (j >= k) {
        return static_cast<long long>(j - k) * stride + i;
    }
    return static_cast<long long>(i) * stride + (j + k + 1);
}

// One block per molecule pair (a,b). Threads parallelize over atom pairs (i,j)
// to raise occupancy and reduce warp divergence vs one-thread-per-mol-pair.
// Target >=8 warps/SM: minBlocksPerSM=8 forces fewer regs if possible.
template <typename T, bool Alch>
__global__ void __launch_bounds__(128, 8) kernel_gaussian_rect_precomputed_kernel(
    const T *x1,
    const T *x2,
    const int *n1,
    const int *n2,
    const int *nn1,
    const int *nn2,
    const T *ss1,
    const T *ss2,
    const T *ksi1,
    const T *ksi2,
    const T *cosp1,
    const T *sinp1,
    const T *cosp2,
    const T *sinp2,
    const T *cosp2e,
    const T *sinp2e,
    T *kernel_out,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    HotParams<T> params,
    const T *s_prefactor_full
) {
    const int pair = static_cast<int>(blockIdx.x);
    const int n_pairs = nm1 * nm2;
    if (pair >= n_pairs) {
        return;
    }

    const int a = pair / nm2;
    const int b = pair - a * nm2;
    const int na = n1[a];
    const int nb = n2[b];
    if (na <= 0 || nb <= 0) {
        if (threadIdx.x == 0) {
            kernel_out[static_cast<long long>(a) * nm2 + b] = Math<T>::zero();
        }
        return;
    }

    const long long fstride1 = fourier_atom_stride(params.pmax, params.fourier_order, max_size1);
    const long long fstride2 = fourier_atom_stride(params.pmax, params.fourier_order, max_size2);
    const int n_atom_pairs = na * nb;

    T kab = Math<T>::zero();
    for (int p = static_cast<int>(threadIdx.x); p < n_atom_pairs;
         p += static_cast<int>(blockDim.x)) {
        const int i = p / nb;
        const int j = p - i * nb;

        const int flat_i = a * max_size1 + i;
        const int flat_j = b * max_size2 + j;
        const T *x1_chan = atom_slice(x1, max_size1, a, i);
        const T *x2_chan = atom_slice(x2, max_size2, b, j);
        const int n_neigh_i = nn1[flat_i];
        const int n_neigh_j = nn2[flat_j];
        const T sii = ss1[flat_i];
        const T sjj = ss2[flat_j];
        const T *ksi_i = ksi1 + static_cast<long long>(flat_i) * max_size1;
        const T *ksi_j = ksi2 + static_cast<long long>(flat_j) * max_size2;
        const T *cos_i = cosp1 + static_cast<long long>(flat_i) * fstride1;
        const T *sin_i = sinp1 + static_cast<long long>(flat_i) * fstride1;
        const T *cos_j = (Alch ? cosp2e : cosp2) + static_cast<long long>(flat_j) * fstride2;
        const T *sin_j = (Alch ? sinp2e : sinp2) + static_cast<long long>(flat_j) * fstride2;

        const T s12 = scalar_dispatch_precomputed<T, Alch>(
            x1_chan,
            max_size1,
            n_neigh_i,
            ksi_i,
            cos_i,
            sin_i,
            x2_chan,
            max_size2,
            n_neigh_j,
            ksi_j,
            cos_j,
            sin_j,
            params,
            s_prefactor_full
        );
        kab += Math<T>::exp_((sii + sjj - T(2) * s12) * params.inv_sigma2);
    }

    kab = block_reduce_sum(kab);
    if (threadIdx.x == 0) {
        kernel_out[static_cast<long long>(a) * nm2 + b] = kab;
    }
}

// Energy-only matvec: one block per molecule pair (a,b) — same launch as dense
// rect kernel — then atomicAdd(E[a], kab * alpha[b]). Avoids forming K.
template <typename T, bool Alch>
__global__ void __launch_bounds__(128, 8) kernel_gaussian_rect_matvec_precomputed_kernel(
    const T *x1,
    const T *x2,
    const int *n1,
    const int *n2,
    const int *nn1,
    const int *nn2,
    const T *ss1,
    const T *ss2,
    const T *ksi1,
    const T *ksi2,
    const T *cosp1,
    const T *sinp1,
    const T *cosp2,
    const T *sinp2,
    const T *cosp2e,
    const T *sinp2e,
    const T *alpha,
    T *E_out,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    HotParams<T> params,
    const T *s_prefactor_full
) {
    const int pair = static_cast<int>(blockIdx.x);
    const int n_pairs = nm1 * nm2;
    if (pair >= n_pairs) {
        return;
    }

    const int a = pair / nm2;
    const int b = pair - a * nm2;
    const int na = n1[a];
    const int nb = n2[b];
    if (na <= 0 || nb <= 0) {
        return;
    }

    const long long fstride1 = fourier_atom_stride(params.pmax, params.fourier_order, max_size1);
    const long long fstride2 = fourier_atom_stride(params.pmax, params.fourier_order, max_size2);
    const int n_atom_pairs = na * nb;

    T kab = Math<T>::zero();
    for (int p = static_cast<int>(threadIdx.x); p < n_atom_pairs;
         p += static_cast<int>(blockDim.x)) {
        const int i = p / nb;
        const int j = p - i * nb;

        const int flat_i = a * max_size1 + i;
        const int flat_j = b * max_size2 + j;
        const T *x1_chan = atom_slice(x1, max_size1, a, i);
        const T *x2_chan = atom_slice(x2, max_size2, b, j);
        const int n_neigh_i = nn1[flat_i];
        const int n_neigh_j = nn2[flat_j];
        const T sii = ss1[flat_i];
        const T sjj = ss2[flat_j];
        const T *ksi_i = ksi1 + static_cast<long long>(flat_i) * max_size1;
        const T *ksi_j = ksi2 + static_cast<long long>(flat_j) * max_size2;
        const T *cos_i = cosp1 + static_cast<long long>(flat_i) * fstride1;
        const T *sin_i = sinp1 + static_cast<long long>(flat_i) * fstride1;
        const T *cos_j = (Alch ? cosp2e : cosp2) + static_cast<long long>(flat_j) * fstride2;
        const T *sin_j = (Alch ? sinp2e : sinp2) + static_cast<long long>(flat_j) * fstride2;

        const T s12 = scalar_dispatch_precomputed<T, Alch>(
            x1_chan,
            max_size1,
            n_neigh_i,
            ksi_i,
            cos_i,
            sin_i,
            x2_chan,
            max_size2,
            n_neigh_j,
            ksi_j,
            cos_j,
            sin_j,
            params,
            s_prefactor_full
        );
        kab += Math<T>::exp_((sii + sjj - T(2) * s12) * params.inv_sigma2);
    }

    kab = block_reduce_sum(kab);
    if (threadIdx.x == 0) {
        atomicAdd(E_out + a, kab * alpha[b]);
    }
}

// Symmetric training kernel: one block per upper-triangle pair (a <= b), writes both
// K[a,b] and K[b,a]. Same precomputed buffers on both sides.
template <typename T, bool Alch>
__global__ void __launch_bounds__(128, 8) kernel_gaussian_symm_precomputed_kernel(
    const T *x,
    const int *n,
    const int *nn,
    const T *ss,
    const T *ksi,
    const T *cosp,
    const T *sinp,
    const T *cospe,
    const T *sinpe,
    T *kernel_out,
    int nm,
    int max_size,
    HotParams<T> params,
    const T *s_prefactor_full
) {
    const long long n_upper = static_cast<long long>(nm) * (nm + 1) / 2;
    const int pair = static_cast<int>(blockIdx.x);
    if (pair >= n_upper) {
        return;
    }

    int a = 0;
    int b = 0;
    upper_pair_from_index(pair, nm, &a, &b);

    const int na = n[a];
    const int nb = n[b];
    if (na <= 0 || nb <= 0) {
        if (threadIdx.x == 0) {
            kernel_out[static_cast<long long>(a) * nm + b] = Math<T>::zero();
            kernel_out[static_cast<long long>(b) * nm + a] = Math<T>::zero();
        }
        return;
    }

    const long long fstride = fourier_atom_stride(params.pmax, params.fourier_order, max_size);
    const int n_atom_pairs = na * nb;

    T kab = Math<T>::zero();
    for (int p = static_cast<int>(threadIdx.x); p < n_atom_pairs;
         p += static_cast<int>(blockDim.x)) {
        const int i = p / nb;
        const int j = p - i * nb;

        const int flat_i = a * max_size + i;
        const int flat_j = b * max_size + j;
        const T *x1_chan = atom_slice(x, max_size, a, i);
        const T *x2_chan = atom_slice(x, max_size, b, j);
        const int n_neigh_i = nn[flat_i];
        const int n_neigh_j = nn[flat_j];
        const T sii = ss[flat_i];
        const T sjj = ss[flat_j];
        const T *ksi_i = ksi + static_cast<long long>(flat_i) * max_size;
        const T *ksi_j = ksi + static_cast<long long>(flat_j) * max_size;
        const T *cos_i = cosp + static_cast<long long>(flat_i) * fstride;
        const T *sin_i = sinp + static_cast<long long>(flat_i) * fstride;
        const T *cos_j = (Alch ? cospe : cosp) + static_cast<long long>(flat_j) * fstride;
        const T *sin_j = (Alch ? sinpe : sinp) + static_cast<long long>(flat_j) * fstride;

        const T s12 = scalar_dispatch_precomputed<T, Alch>(
            x1_chan,
            max_size,
            n_neigh_i,
            ksi_i,
            cos_i,
            sin_i,
            x2_chan,
            max_size,
            n_neigh_j,
            ksi_j,
            cos_j,
            sin_j,
            params,
            s_prefactor_full
        );
        kab += Math<T>::exp_((sii + sjj - T(2) * s12) * params.inv_sigma2);
    }

    kab = block_reduce_sum(kab);
    if (threadIdx.x == 0) {
        kernel_out[static_cast<long long>(a) * nm + b] = kab;
        kernel_out[static_cast<long long>(b) * nm + a] = kab;
    }
}

// Symmetric training kernel writing RFP directly (TRANSR='N', UPLO='U').
// One block per upper-triangle pair (a <= b); stores at rfp_index_upper_N(nm, a, b).
template <typename T, bool Alch>
__global__ void __launch_bounds__(128, 8) kernel_gaussian_symm_rfp_precomputed_kernel(
    const T *x,
    const int *n,
    const int *nn,
    const T *ss,
    const T *ksi,
    const T *cosp,
    const T *sinp,
    const T *cospe,
    const T *sinpe,
    T *kernel_rfp_out,
    int nm,
    int max_size,
    HotParams<T> params,
    const T *s_prefactor_full
) {
    const long long n_upper = static_cast<long long>(nm) * (nm + 1) / 2;
    const int pair = static_cast<int>(blockIdx.x);
    if (pair >= n_upper) {
        return;
    }

    int a = 0;
    int b = 0;
    upper_pair_from_index(pair, nm, &a, &b);

    const int na = n[a];
    const int nb = n[b];
    if (na <= 0 || nb <= 0) {
        if (threadIdx.x == 0) {
            kernel_rfp_out[rfp_index_upper_N_dev(nm, a, b)] = Math<T>::zero();
        }
        return;
    }

    const long long fstride = fourier_atom_stride(params.pmax, params.fourier_order, max_size);
    const int n_atom_pairs = na * nb;

    T kab = Math<T>::zero();
    for (int p = static_cast<int>(threadIdx.x); p < n_atom_pairs;
         p += static_cast<int>(blockDim.x)) {
        const int i = p / nb;
        const int j = p - i * nb;

        const int flat_i = a * max_size + i;
        const int flat_j = b * max_size + j;
        const T *x1_chan = atom_slice(x, max_size, a, i);
        const T *x2_chan = atom_slice(x, max_size, b, j);
        const int n_neigh_i = nn[flat_i];
        const int n_neigh_j = nn[flat_j];
        const T sii = ss[flat_i];
        const T sjj = ss[flat_j];
        const T *ksi_i = ksi + static_cast<long long>(flat_i) * max_size;
        const T *ksi_j = ksi + static_cast<long long>(flat_j) * max_size;
        const T *cos_i = cosp + static_cast<long long>(flat_i) * fstride;
        const T *sin_i = sinp + static_cast<long long>(flat_i) * fstride;
        const T *cos_j = (Alch ? cospe : cosp) + static_cast<long long>(flat_j) * fstride;
        const T *sin_j = (Alch ? sinpe : sinp) + static_cast<long long>(flat_j) * fstride;

        const T s12 = scalar_dispatch_precomputed<T, Alch>(
            x1_chan,
            max_size,
            n_neigh_i,
            ksi_i,
            cos_i,
            sin_i,
            x2_chan,
            max_size,
            n_neigh_j,
            ksi_j,
            cos_j,
            sin_j,
            params,
            s_prefactor_full
        );
        kab += Math<T>::exp_((sii + sjj - T(2) * s12) * params.inv_sigma2);
    }

    kab = block_reduce_sum(kab);
    if (threadIdx.x == 0) {
        kernel_rfp_out[rfp_index_upper_N_dev(nm, a, b)] = kab;
    }
}

// Fill the periodic-table alchemy device tables and the element-mixed ("effective")
// Fourier buffers for one atom set. cosp/sinp are the raw atom-major buffers.
template <typename T>
void alchemy_mix_atom_major(
    const T *cosp,
    const T *sinp,
    T *cosp_e,
    T *sinp_e,
    const T *pdc,
    const int *d_n,
    int nm,
    int max_size,
    const KernelParams<T> &params
) {
    const long long fstride = fourier_atom_stride(params.pmax, params.fourier_order, max_size);
    alchemy_mix_fourier(
        cosp,
        cosp_e,
        pdc,
        d_n,
        nm,
        max_size,
        params.pmax,
        fstride,
        max_size,
        static_cast<long long>(params.pmax) * params.fourier_order,
        params.fourier_order,
        false,
        false,
        params.fourier_order
    );
    alchemy_mix_fourier(
        sinp,
        sinp_e,
        pdc,
        d_n,
        nm,
        max_size,
        params.pmax,
        fstride,
        max_size,
        static_cast<long long>(params.pmax) * params.fourier_order,
        params.fourier_order,
        false,
        false,
        params.fourier_order
    );
}

enum class RectOutMode { Dense, Matvec };

template <typename T>
void kernel_gaussian_rect_cu_impl(
    const T *d_x1,
    const T *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    T *d_out,
    const T *d_alpha,
    RectOutMode mode,
    T sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
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
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    if (fourier_order <= 0 || fourier_order > kMaxFourierOrder) {
        throw std::invalid_argument("fourier_order must be in [1, 16] for cuda_fchl18_kernel");
    }

    Workspace<T> &ws = workspace<T>();
    if (ws.z_present == nullptr) {
        CUDA_CHECK(cudaMalloc(&ws.z_present, 256 * sizeof(int)));
    }
    CUDA_CHECK(cudaMemset(ws.z_present, 0, 256 * sizeof(int)));

    // Dense path mutates shared side-2 buffers — drop durable train cache.
    if (mode == RectOutMode::Dense) {
        ws.train_valid = false;
    }

    // Identical query/train tensors (common in benches): share sorted + precomputed side.
    const bool same_side = (d_x1 == d_x2 && d_n1 == d_n2 && d_nn1 == d_nn2 && nm1 == nm2 &&
                            max_size1 == max_size2);

    const int tb_mode = get_fchl18_three_body_weight_mode();
    const bool want_train_cache = (mode == RectOutMode::Matvec && !same_side);
    bool train_hit =
        want_train_cache && ws.train_valid && ws.train_x_ptr == d_x2 &&
        ws.train_n_ptr == d_n2 && ws.train_nn_ptr == d_nn2 && ws.train_nm == nm2 &&
        ws.train_max_size == max_size2 && ws.train_sigma == sigma &&
        ws.train_two_body_scaling == two_body_scaling &&
        ws.train_two_body_width == two_body_width &&
        ws.train_two_body_power == two_body_power &&
        ws.train_three_body_scaling == three_body_scaling &&
        ws.train_three_body_width == three_body_width &&
        ws.train_three_body_power == three_body_power &&
        ws.train_cut_start == cut_start && ws.train_cut_distance == cut_distance &&
        ws.train_fourier_order == fourier_order && ws.train_use_atm == use_atm &&
        ws.train_use_alchemy == use_alchemy &&
        ws.train_alchemy_period_width == alchemy_period_width &&
        ws.train_alchemy_group_width == alchemy_group_width &&
        ws.train_tb_weight_mode == tb_mode;

    const int block_ss = 256;
    const size_t ss1_need = static_cast<size_t>(nm1) * max_size1;
    const size_t ss2_need = static_cast<size_t>(nm2) * max_size2;
    const size_t x1_need = ss1_need * 5 * static_cast<size_t>(max_size1);
    const size_t x2_need = ss2_need * 5 * static_cast<size_t>(max_size2);
    const int grid_ss1 = static_cast<int>((ss1_need + block_ss - 1) / block_ss);
    const int grid_ss2 = static_cast<int>((ss2_need + block_ss - 1) / block_ss);

    // No-alchemy: Z-sort neighbours (equal-Z merge in the scalar kernel).
    // Alchemy: every neighbour pair contributes, so use the representation as-is.
    const T *x1_use = d_x1;
    const T *x2_use = d_x2;
    if (!use_alchemy) {
        ensure_capacity(&ws.x1_sorted, &ws.x1_sorted_cap, x1_need);
        sort_neighbors_by_z_kernel<<<grid_ss1, block_ss>>>(
            d_x1, d_n1, d_nn1, ws.x1_sorted, nm1, max_size1
        );
        CUDA_CHECK(cudaGetLastError());
        x1_use = ws.x1_sorted;
        x2_use = ws.x1_sorted;
        if (!same_side) {
            if (train_hit) {
                x2_use = ws.train_x_sorted;
            } else if (want_train_cache) {
                ensure_capacity(&ws.train_x_sorted, &ws.train_x_sorted_cap, x2_need);
                sort_neighbors_by_z_kernel<<<grid_ss2, block_ss>>>(
                    d_x2, d_n2, d_nn2, ws.train_x_sorted, nm2, max_size2
                );
                CUDA_CHECK(cudaGetLastError());
                x2_use = ws.train_x_sorted;
            } else {
                ensure_capacity(&ws.x2_sorted, &ws.x2_sorted_cap, x2_need);
                sort_neighbors_by_z_kernel<<<grid_ss2, block_ss>>>(
                    d_x2, d_n2, d_nn2, ws.x2_sorted, nm2, max_size2
                );
                CUDA_CHECK(cudaGetLastError());
                x2_use = ws.x2_sorted;
            }
        }
    } else if (same_side) {
        x2_use = x1_use;
    }

    mark_elements_kernel<<<grid_ss1, block_ss>>>(
        x1_use, d_n1, d_nn1, nm1, max_size1, ws.z_present
    );
    CUDA_CHECK(cudaGetLastError());
    if (!same_side) {
        mark_elements_kernel<<<grid_ss2, block_ss>>>(
            x2_use, d_n2, d_nn2, nm2, max_size2, ws.z_present
        );
        CUDA_CHECK(cudaGetLastError());
    }

    int h_z_present[256];
    CUDA_CHECK(cudaMemcpy(h_z_present, ws.z_present, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    KernelParams<T> params{};
    params.sigma = sigma;
    params.inv_sigma2 = static_cast<T>(-0.5) / (sigma * sigma);
    params.two_body_scaling = two_body_scaling;
    params.two_body_width = two_body_width;
    params.two_body_power = two_body_power;
    params.three_body_scaling = three_body_scaling;
    params.three_body_width = three_body_width;
    params.three_body_power = three_body_power;
    params.three_body_weight_mode = tb_mode;
    params.cut_start = cut_start;
    params.cut_distance = cut_distance;
    params.true_distance_scale = two_body_scaling / static_cast<T>(16);
    params.true_angular_scale = three_body_scaling / static_cast<T>(std::sqrt(8.0));
    params.fourier_order = fourier_order;
    params.use_atm = use_atm ? 1 : 0;
    params.pmax = build_element_map_from_flags(h_z_present, params.z_to_idx);

    // Train Fourier channels depend on the combined element map — invalidate if it shifted.
    if (train_hit &&
        (ws.train_pmax != params.pmax ||
         std::memcmp(ws.train_z_to_idx, params.z_to_idx, sizeof(ws.train_z_to_idx)) != 0)) {
        train_hit = false;
    }

    if (params.pmax == 0) {
        if (mode == RectOutMode::Dense) {
            CUDA_CHECK(cudaMemset(d_out, 0, static_cast<size_t>(nm1) * nm2 * sizeof(T)));
        } else {
            CUDA_CHECK(cudaMemset(d_out, 0, static_cast<size_t>(nm1) * sizeof(T)));
        }
        return;
    }
    if (params.pmax > kMaxElements) {
        throw std::invalid_argument("too many distinct elements for cuda_fchl18_kernel");
    }

    const double ang_norm2 = get_angular_norm2(static_cast<double>(three_body_width));
    const double g1 = std::sqrt(2.0 * 3.14159265358979323846) / ang_norm2;
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        const double tw = static_cast<double>(three_body_width);
        params.s_prefactor[m] = static_cast<T>(g1 * std::exp(-(tw * mf) * (tw * mf) / 2.0));
    }

    const size_t ksi1_need = ss1_need * static_cast<size_t>(max_size1);
    const size_t ksi2_need = ss2_need * static_cast<size_t>(max_size2);
    const size_t f1_need =
        ss1_need * static_cast<size_t>(params.pmax) * fourier_order * max_size1;
    const size_t f2_need =
        ss2_need * static_cast<size_t>(params.pmax) * fourier_order * max_size2;

    ensure_capacity(&ws.ss1, &ws.ss1_cap, ss1_need);
    ensure_capacity(&ws.ksi1, &ws.ksi1_cap, ksi1_need);
    ensure_capacity(&ws.cosp1, &ws.cosp1_cap, f1_need);
    ensure_capacity(&ws.sinp1, &ws.sinp1_cap, f1_need);

    T *ss2 = ws.ss1;
    T *ksi2 = ws.ksi1;
    T *cosp2 = ws.cosp1;
    T *sinp2 = ws.sinp1;
    if (!same_side) {
        if (want_train_cache) {
            ensure_capacity(&ws.train_ss, &ws.train_ss_cap, ss2_need);
            ensure_capacity(&ws.train_ksi, &ws.train_ksi_cap, ksi2_need);
            ensure_capacity(&ws.train_cosp, &ws.train_cosp_cap, f2_need);
            ensure_capacity(&ws.train_sinp, &ws.train_sinp_cap, f2_need);
            ss2 = ws.train_ss;
            ksi2 = ws.train_ksi;
            cosp2 = ws.train_cosp;
            sinp2 = ws.train_sinp;
        } else {
            ensure_capacity(&ws.ss2, &ws.ss2_cap, ss2_need);
            ensure_capacity(&ws.ksi2, &ws.ksi2_cap, ksi2_need);
            ensure_capacity(&ws.cosp2, &ws.cosp2_cap, f2_need);
            ensure_capacity(&ws.sinp2, &ws.sinp2_cap, f2_need);
            ss2 = ws.ss2;
            ksi2 = ws.ksi2;
            cosp2 = ws.cosp2;
            sinp2 = ws.sinp2;
        }
    }

    precompute_atom_data_kernel<<<grid_ss1, block_ss>>>(
        x1_use, d_n1, d_nn1, ws.ksi1, ws.cosp1, ws.sinp1, nm1, max_size1, params
    );
    CUDA_CHECK(cudaGetLastError());
    if (!same_side && !train_hit) {
        precompute_atom_data_kernel<<<grid_ss2, block_ss>>>(
            x2_use, d_n2, d_nn2, ksi2, cosp2, sinp2, nm2, max_size2, params
        );
        CUDA_CHECK(cudaGetLastError());
    }

    const ScalarParams<T> scalar_params = params.to_scalar();
    HotParams<T> hot_params = to_hot(scalar_params);

    // Alchemy: element-channel-mixed Fourier buffers (self scalar + side-2 of pair kernel).
    T *cosp1e = ws.cosp1;
    T *sinp1e = ws.sinp1;
    T *cosp2e = cosp2;
    T *sinp2e = sinp2;
    if (use_alchemy) {
        hot_params.pd = alchemy_upload_pd<T>(alchemy_period_width, alchemy_group_width);
        hot_params.emax = kf::fchl18::kAlchemyEmaxDefault;
        const T *pdc = alchemy_upload_pdc<T>(params.z_to_idx, params.pmax);
        ensure_capacity(&ws.cosp1e, &ws.cosp1e_cap, f1_need);
        ensure_capacity(&ws.sinp1e, &ws.sinp1e_cap, f1_need);
        alchemy_mix_atom_major(
            ws.cosp1, ws.sinp1, ws.cosp1e, ws.sinp1e, pdc, d_n1, nm1, max_size1, params
        );
        cosp1e = ws.cosp1e;
        sinp1e = ws.sinp1e;
        cosp2e = cosp1e;
        sinp2e = sinp1e;
        if (!same_side) {
            if (want_train_cache) {
                ensure_capacity(&ws.train_cospe, &ws.train_cospe_cap, f2_need);
                ensure_capacity(&ws.train_sinpe, &ws.train_sinpe_cap, f2_need);
                cosp2e = ws.train_cospe;
                sinp2e = ws.train_sinpe;
            } else {
                ensure_capacity(&ws.cosp2e, &ws.cosp2e_cap, f2_need);
                ensure_capacity(&ws.sinp2e, &ws.sinp2e_cap, f2_need);
                cosp2e = ws.cosp2e;
                sinp2e = ws.sinp2e;
            }
            if (!train_hit) {
                alchemy_mix_atom_major(
                    cosp2, sinp2, cosp2e, sinp2e, pdc, d_n2, nm2, max_size2, params
                );
            }
        }
    }

    ensure_capacity(&ws.s_prefactor, &ws.s_prefactor_cap, static_cast<size_t>(kMaxFourierOrder));
    CUDA_CHECK(cudaMemcpy(
        ws.s_prefactor,
        scalar_params.s_prefactor,
        static_cast<size_t>(kMaxFourierOrder) * sizeof(T),
        cudaMemcpyHostToDevice
    ));

    const long long n_pairs = static_cast<long long>(nm1) * nm2;
    if (mode == RectOutMode::Matvec) {
        CUDA_CHECK(cudaMemset(d_out, 0, static_cast<size_t>(nm1) * sizeof(T)));
    }
    auto run = [&](auto alch_tag) {
        constexpr bool Alch = decltype(alch_tag)::value;
        compute_self_scalars_precomputed_kernel<T, Alch><<<grid_ss1, block_ss>>>(
            x1_use,
            d_n1,
            d_nn1,
            ws.ksi1,
            ws.cosp1,
            ws.sinp1,
            cosp1e,
            sinp1e,
            ws.ss1,
            nm1,
            max_size1,
            hot_params,
            ws.s_prefactor
        );
        CUDA_CHECK(cudaGetLastError());
        if (!same_side && !train_hit) {
            compute_self_scalars_precomputed_kernel<T, Alch><<<grid_ss2, block_ss>>>(
                x2_use,
                d_n2,
                d_nn2,
                ksi2,
                cosp2,
                sinp2,
                cosp2e,
                sinp2e,
                ss2,
                nm2,
                max_size2,
                hot_params,
                ws.s_prefactor
            );
            CUDA_CHECK(cudaGetLastError());
        }

        if (mode == RectOutMode::Matvec) {
            kernel_gaussian_rect_matvec_precomputed_kernel<T, Alch>
                <<<static_cast<int>(n_pairs), kPairBlockSize>>>(
                    x1_use,
                    x2_use,
                    d_n1,
                    d_n2,
                    d_nn1,
                    d_nn2,
                    ws.ss1,
                    ss2,
                    ws.ksi1,
                    ksi2,
                    ws.cosp1,
                    ws.sinp1,
                    cosp2,
                    sinp2,
                    cosp2e,
                    sinp2e,
                    d_alpha,
                    d_out,
                    nm1,
                    nm2,
                    max_size1,
                    max_size2,
                    hot_params,
                    ws.s_prefactor
                );
        } else {
            kernel_gaussian_rect_precomputed_kernel<T, Alch>
                <<<static_cast<int>(n_pairs), kPairBlockSize>>>(
                    x1_use,
                    x2_use,
                    d_n1,
                    d_n2,
                    d_nn1,
                    d_nn2,
                    ws.ss1,
                    ss2,
                    ws.ksi1,
                    ksi2,
                    ws.cosp1,
                    ws.sinp1,
                    cosp2,
                    sinp2,
                    cosp2e,
                    sinp2e,
                    d_out,
                    nm1,
                    nm2,
                    max_size1,
                    max_size2,
                    hot_params,
                    ws.s_prefactor
                );
        }
        CUDA_CHECK(cudaGetLastError());
    };
    if (use_alchemy) {
        run(std::true_type{});
    } else {
        run(std::false_type{});
    }

    if (want_train_cache && !train_hit) {
        ws.train_valid = true;
        ws.train_x_ptr = d_x2;
        ws.train_n_ptr = d_n2;
        ws.train_nn_ptr = d_nn2;
        ws.train_nm = nm2;
        ws.train_max_size = max_size2;
        ws.train_sigma = sigma;
        ws.train_two_body_scaling = two_body_scaling;
        ws.train_two_body_width = two_body_width;
        ws.train_two_body_power = two_body_power;
        ws.train_three_body_scaling = three_body_scaling;
        ws.train_three_body_width = three_body_width;
        ws.train_three_body_power = three_body_power;
        ws.train_cut_start = cut_start;
        ws.train_cut_distance = cut_distance;
        ws.train_fourier_order = fourier_order;
        ws.train_use_atm = use_atm;
        ws.train_use_alchemy = use_alchemy;
        ws.train_alchemy_period_width = alchemy_period_width;
        ws.train_alchemy_group_width = alchemy_group_width;
        ws.train_tb_weight_mode = tb_mode;
        ws.train_pmax = params.pmax;
        std::memcpy(ws.train_z_to_idx, params.z_to_idx, sizeof(ws.train_z_to_idx));
    }
}

enum class SymmOutMode { Dense, Rfp };

template <typename T>
void kernel_gaussian_symm_cu_impl(
    const T *d_x,
    const int *d_n,
    const int *d_nn,
    T *d_kernel_out,
    T sigma,
    int nm,
    int max_size,
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
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width,
    SymmOutMode out_mode
) {
    if (fourier_order <= 0 || fourier_order > kMaxFourierOrder) {
        throw std::invalid_argument("fourier_order must be in [1, 16] for cuda_fchl18_kernel");
    }

    const long long n_upper = static_cast<long long>(nm) * (nm + 1) / 2;
    if (n_upper > static_cast<long long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("too many molecule pairs for cuda_fchl18_kernel_symm");
    }

    Workspace<T> &ws = workspace<T>();
    if (ws.z_present == nullptr) {
        CUDA_CHECK(cudaMalloc(&ws.z_present, 256 * sizeof(int)));
    }
    CUDA_CHECK(cudaMemset(ws.z_present, 0, 256 * sizeof(int)));

    const int block_ss = 256;
    const size_t ss_need = static_cast<size_t>(nm) * max_size;
    const size_t x_need = ss_need * 5 * static_cast<size_t>(max_size);
    const int grid_ss = static_cast<int>((ss_need + block_ss - 1) / block_ss);

    const T *x_use = d_x;
    if (!use_alchemy) {
        ensure_capacity(&ws.x1_sorted, &ws.x1_sorted_cap, x_need);
        sort_neighbors_by_z_kernel<<<grid_ss, block_ss>>>(
            d_x, d_n, d_nn, ws.x1_sorted, nm, max_size
        );
        CUDA_CHECK(cudaGetLastError());
        x_use = ws.x1_sorted;
    }

    mark_elements_kernel<<<grid_ss, block_ss>>>(x_use, d_n, d_nn, nm, max_size, ws.z_present);
    CUDA_CHECK(cudaGetLastError());

    int h_z_present[256];
    CUDA_CHECK(cudaMemcpy(h_z_present, ws.z_present, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    KernelParams<T> params{};
    params.sigma = sigma;
    params.inv_sigma2 = static_cast<T>(-0.5) / (sigma * sigma);
    params.two_body_scaling = two_body_scaling;
    params.two_body_width = two_body_width;
    params.two_body_power = two_body_power;
    params.three_body_scaling = three_body_scaling;
    params.three_body_width = three_body_width;
    params.three_body_power = three_body_power;
    params.three_body_weight_mode = get_fchl18_three_body_weight_mode();
    params.cut_start = cut_start;
    params.cut_distance = cut_distance;
    params.true_distance_scale = two_body_scaling / static_cast<T>(16);
    params.true_angular_scale = three_body_scaling / static_cast<T>(std::sqrt(8.0));
    params.fourier_order = fourier_order;
    params.use_atm = use_atm ? 1 : 0;
    params.pmax = build_element_map_from_flags(h_z_present, params.z_to_idx);

    if (params.pmax == 0) {
        const size_t n_out =
            (out_mode == SymmOutMode::Rfp) ? static_cast<size_t>(n_upper)
                                           : static_cast<size_t>(nm) * static_cast<size_t>(nm);
        CUDA_CHECK(cudaMemset(d_kernel_out, 0, n_out * sizeof(T)));
        return;
    }
    if (params.pmax > kMaxElements) {
        throw std::invalid_argument("too many distinct elements for cuda_fchl18_kernel");
    }

    const double ang_norm2 = get_angular_norm2(static_cast<double>(three_body_width));
    const double g1 = std::sqrt(2.0 * 3.14159265358979323846) / ang_norm2;
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        const double tw = static_cast<double>(three_body_width);
        params.s_prefactor[m] = static_cast<T>(g1 * std::exp(-(tw * mf) * (tw * mf) / 2.0));
    }

    const size_t ksi_need = ss_need * static_cast<size_t>(max_size);
    const size_t f_need =
        ss_need * static_cast<size_t>(params.pmax) * fourier_order * max_size;

    ensure_capacity(&ws.ss1, &ws.ss1_cap, ss_need);
    ensure_capacity(&ws.ksi1, &ws.ksi1_cap, ksi_need);
    ensure_capacity(&ws.cosp1, &ws.cosp1_cap, f_need);
    ensure_capacity(&ws.sinp1, &ws.sinp1_cap, f_need);

    precompute_atom_data_kernel<<<grid_ss, block_ss>>>(
        x_use, d_n, d_nn, ws.ksi1, ws.cosp1, ws.sinp1, nm, max_size, params
    );
    CUDA_CHECK(cudaGetLastError());

    const ScalarParams<T> scalar_params = params.to_scalar();
    HotParams<T> hot_params = to_hot(scalar_params);

    T *cospe = ws.cosp1;
    T *sinpe = ws.sinp1;
    if (use_alchemy) {
        hot_params.pd = alchemy_upload_pd<T>(alchemy_period_width, alchemy_group_width);
        hot_params.emax = kf::fchl18::kAlchemyEmaxDefault;
        const T *pdc = alchemy_upload_pdc<T>(params.z_to_idx, params.pmax);
        ensure_capacity(&ws.cosp1e, &ws.cosp1e_cap, f_need);
        ensure_capacity(&ws.sinp1e, &ws.sinp1e_cap, f_need);
        alchemy_mix_atom_major(
            ws.cosp1, ws.sinp1, ws.cosp1e, ws.sinp1e, pdc, d_n, nm, max_size, params
        );
        cospe = ws.cosp1e;
        sinpe = ws.sinp1e;
    }

    ensure_capacity(&ws.s_prefactor, &ws.s_prefactor_cap, static_cast<size_t>(kMaxFourierOrder));
    CUDA_CHECK(cudaMemcpy(
        ws.s_prefactor,
        scalar_params.s_prefactor,
        static_cast<size_t>(kMaxFourierOrder) * sizeof(T),
        cudaMemcpyHostToDevice
    ));

    auto run = [&](auto alch_tag) {
        constexpr bool Alch = decltype(alch_tag)::value;
        compute_self_scalars_precomputed_kernel<T, Alch><<<grid_ss, block_ss>>>(
            x_use,
            d_n,
            d_nn,
            ws.ksi1,
            ws.cosp1,
            ws.sinp1,
            cospe,
            sinpe,
            ws.ss1,
            nm,
            max_size,
            hot_params,
            ws.s_prefactor
        );
        CUDA_CHECK(cudaGetLastError());

        if (out_mode == SymmOutMode::Rfp) {
            kernel_gaussian_symm_rfp_precomputed_kernel<T, Alch>
                <<<static_cast<int>(n_upper), kPairBlockSize>>>(
                    x_use,
                    d_n,
                    d_nn,
                    ws.ss1,
                    ws.ksi1,
                    ws.cosp1,
                    ws.sinp1,
                    cospe,
                    sinpe,
                    d_kernel_out,
                    nm,
                    max_size,
                    hot_params,
                    ws.s_prefactor
                );
        } else {
            kernel_gaussian_symm_precomputed_kernel<T, Alch>
                <<<static_cast<int>(n_upper), kPairBlockSize>>>(
                    x_use,
                    d_n,
                    d_nn,
                    ws.ss1,
                    ws.ksi1,
                    ws.cosp1,
                    ws.sinp1,
                    cospe,
                    sinpe,
                    d_kernel_out,
                    nm,
                    max_size,
                    hot_params,
                    ws.s_prefactor
                );
        }
        CUDA_CHECK(cudaGetLastError());
    };
    if (use_alchemy) {
        run(std::true_type{});
    } else {
        run(std::false_type{});
    }
}

}  // namespace

void kernel_gaussian_rect_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    float *d_kernel_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    kernel_gaussian_rect_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_kernel_out,
        static_cast<const float *>(nullptr),
        RectOutMode::Dense,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
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
        use_alchemy,
        alchemy_period_width,
        alchemy_group_width
    );
}

void kernel_gaussian_rect_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    double *d_kernel_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    kernel_gaussian_rect_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_kernel_out,
        static_cast<const double *>(nullptr),
        RectOutMode::Dense,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
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
        use_alchemy,
        alchemy_period_width,
        alchemy_group_width
    );
}

void kernel_gaussian_rect_matvec_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const float *d_alpha,
    float *d_E_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    kernel_gaussian_rect_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_E_out,
        d_alpha,
        RectOutMode::Matvec,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
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
        use_alchemy,
        alchemy_period_width,
        alchemy_group_width
    );
}

void kernel_gaussian_rect_matvec_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const double *d_alpha,
    double *d_E_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    kernel_gaussian_rect_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_E_out,
        d_alpha,
        RectOutMode::Matvec,
        sigma,
        nm1,
        nm2,
        max_size1,
        max_size2,
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
        use_alchemy,
        alchemy_period_width,
        alchemy_group_width
    );
}

void kernel_gaussian_symm_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    float *d_kernel_out,
    float sigma,
    int nm,
    int max_size,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    kernel_gaussian_symm_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_kernel_out,
        sigma,
        nm,
        max_size,
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
        use_alchemy,
        alchemy_period_width,
        alchemy_group_width,
        SymmOutMode::Dense
    );
}

void kernel_gaussian_symm_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    double *d_kernel_out,
    double sigma,
    int nm,
    int max_size,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    kernel_gaussian_symm_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_kernel_out,
        sigma,
        nm,
        max_size,
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
        use_alchemy,
        alchemy_period_width,
        alchemy_group_width,
        SymmOutMode::Dense
    );
}

void kernel_gaussian_symm_rfp_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    float *d_kernel_rfp_out,
    float sigma,
    int nm,
    int max_size,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    kernel_gaussian_symm_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_kernel_rfp_out,
        sigma,
        nm,
        max_size,
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
        use_alchemy,
        alchemy_period_width,
        alchemy_group_width,
        SymmOutMode::Rfp
    );
}

void kernel_gaussian_symm_rfp_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    double *d_kernel_rfp_out,
    double sigma,
    int nm,
    int max_size,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    kernel_gaussian_symm_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_kernel_rfp_out,
        sigma,
        nm,
        max_size,
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
        use_alchemy,
        alchemy_period_width,
        alchemy_group_width,
        SymmOutMode::Rfp
    );
}

}  // namespace fchl18
}  // namespace kf
