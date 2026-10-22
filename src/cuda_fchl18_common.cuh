#pragma once

// Shared device/host helpers for the CUDA FCHL18 kernels.
//
// Included by cuda_fchl18_kernel.cu (scalar kernels) and
// cuda_fchl18_jacobian.cu (dK/dR_A). Only small inline/template helpers live
// here; __global__ entry points stay in their translation unit.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "fchl18_alchemy.hpp"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t _e = (call);                                                          \
        if (_e != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(_e));                                              \
            std::abort();                                                                 \
        }                                                                                 \
    } while (0)
#endif

namespace kf {
namespace fchl18 {

constexpr int kMaxFourierOrder = 16;
constexpr int kMaxElements = 32;

template <typename T>
struct Math;

template <>
struct Math<float> {
    static __device__ inline float pi() { return 3.14159265358979323846f; }
    static __device__ inline float zero() { return 0.0f; }
    static __device__ inline float one() { return 1.0f; }
    static __device__ inline float eps() { return 1e-14f; }
    static __device__ inline float sqrt_(float x) { return sqrtf(x); }
    static __device__ inline float pow_(float x, float p) { return powf(x, p); }
    static __device__ inline float acos_(float x) { return acosf(x); }
    static __device__ inline float cos_(float x) { return cosf(x); }
    static __device__ inline float sin_(float x) { return sinf(x); }
    static __device__ inline float exp_(float x) { return expf(x); }
    static __device__ inline float abs_(float x) { return fabsf(x); }
    static __device__ inline void sincos_(float x, float *s, float *c) { sincosf(x, s, c); }
};

template <>
struct Math<double> {
    static __device__ inline double pi() { return 3.14159265358979323846; }
    static __device__ inline double zero() { return 0.0; }
    static __device__ inline double one() { return 1.0; }
    static __device__ inline double eps() { return 1e-14; }
    static __device__ inline double sqrt_(double x) { return sqrt(x); }
    static __device__ inline double pow_(double x, double p) { return pow(x, p); }
    static __device__ inline double acos_(double x) { return acos(x); }
    static __device__ inline double cos_(double x) { return cos(x); }
    static __device__ inline double sin_(double x) { return sin(x); }
    static __device__ inline double exp_(double x) { return exp(x); }
    static __device__ inline double abs_(double x) { return fabs(x); }
    static __device__ inline void sincos_(double x, double *s, double *c) { sincos(x, s, c); }
};

template <typename T>
__device__ inline T ipow(T x, int n) {
    T r = Math<T>::one();
    for (; n > 0; --n) {
        r *= x;
    }
    return r;
}

template <typename T>
__device__ inline T fast_pow(T x, T p) {
    const int ip = static_cast<int>(p);
    if (static_cast<T>(ip) == p && ip >= 0 && ip <= 16) {
        return ipow(x, ip);
    }
    return Math<T>::pow_(x, p);
}

template <typename T>
__device__ inline T clamp11(T x) {
    const T one = Math<T>::one();
    return x < -one ? -one : (x > one ? one : x);
}

template <typename T>
__device__ inline T cut_function(T r, T cut_start, T cut_distance) {
    const T ru = cut_distance;
    const T rl = cut_start * cut_distance;
    if (r >= ru) {
        return Math<T>::zero();
    }
    if (r <= rl) {
        return Math<T>::one();
    }
    const T x = (ru - r) / (ru - rl);
    return T(10) * x * x * x - T(15) * x * x * x * x + T(6) * x * x * x * x * x;
}

// Three-body radial weight modes (ATM is independent and usually off for EF).
// 0 = legacy product power:          (r_ij r_ik r_jk)^(-p)
// 1 = bond-normalized product power: (r'_ij r'_ik r'_jk)^(-p), r'=r/r0
// 2 = bond-normalized exp-sum:       exp(-α (r'_ij + r'_ik + r'_jk))
constexpr int kTbWeightProduct = 0;
constexpr int kTbWeightBondNormProduct = 1;
constexpr int kTbWeightBondNormExpSum = 2;

// Host-side mode used when packing KernelParams (defined in cuda_fchl18_kernel.cu).
void set_fchl18_three_body_weight_mode(int mode);
int get_fchl18_three_body_weight_mode();

// Cordero et al. single-bond covalent radii (Å). Unknown Z falls back to 1.5 Å.
__host__ __device__ inline float covalent_radius_angstrom(int z) {
    // clang-format off
    constexpr float kRad[37] = {
        0.00f,
        0.31f, 0.28f, 1.28f, 0.96f, 0.84f, 0.76f, 0.71f, 0.66f, 0.57f, 0.58f,  // 1-10
        1.66f, 1.41f, 1.21f, 1.11f, 1.07f, 1.05f, 1.02f, 1.06f, 2.03f, 1.76f,  // 11-20
        1.70f, 1.60f, 1.53f, 1.39f, 1.39f, 1.32f, 1.26f, 1.24f, 1.32f, 1.22f,  // 21-30
        1.22f, 1.20f, 1.19f, 1.20f, 1.20f, 1.16f                                // 31-36
    };
    // clang-format on
    if (z > 0 && z <= 36) {
        return kRad[z];
    }
    return 1.50f;
}

__host__ __device__ inline float pair_bond_r0(int za, int zb) {
    return covalent_radius_angstrom(za) + covalent_radius_angstrom(zb);
}

// Radial three-body weight and d(log w)/dr for the three triangle edges.
// Edges: dj = r_ij (centre-j), dk = r_ik (centre-k), di = r_jk (j-k).
template <typename T>
struct ThreeBodyRadial {
    T w;
    T dlog_dj;
    T dlog_dk;
    T dlog_di;
};

template <typename T>
__device__ inline ThreeBodyRadial<T> three_body_radial_weight(
    int mode,
    T dj,
    T dk,
    T di,
    T r0_ij,
    T r0_ik,
    T r0_jk,
    T power_or_alpha
) {
    ThreeBodyRadial<T> out;
    const T one = Math<T>::one();
    const T eps = Math<T>::eps();
    if (dj < eps || dk < eps || di < eps) {
        out.w = Math<T>::zero();
        out.dlog_dj = out.dlog_dk = out.dlog_di = Math<T>::zero();
        return out;
    }
    if (mode == kTbWeightBondNormExpSum) {
        const T a = power_or_alpha;
        const T s = dj / r0_ij + dk / r0_ik + di / r0_jk;
        out.w = Math<T>::exp_(-a * s);
        out.dlog_dj = -a / r0_ij;
        out.dlog_dk = -a / r0_ik;
        out.dlog_di = -a / r0_jk;
        return out;
    }
    // Product power: w = scale / (dj*dk*di)^p with scale=1 (legacy) or (r0ij r0ik r0jk)^p.
    const T p = power_or_alpha;
    const T dijk = dj * dk * di;
    T scale = one;
    if (mode == kTbWeightBondNormProduct) {
        scale = fast_pow(r0_ij * r0_ik * r0_jk, p);
    }
    const T dijk_p = fast_pow(dijk, p);
    out.w = scale / dijk_p;
    out.dlog_dj = -p / dj;
    out.dlog_dk = -p / dk;
    out.dlog_di = -p / di;
    return out;
}

// d(cut_function)/dr — mirrors kf::fchl18::cut_function_deriv on the CPU side.
template <typename T>
__device__ inline T cut_function_deriv(T r, T cut_start, T cut_distance) {
    const T ru = cut_distance;
    const T rl = cut_start * cut_distance;
    if (r >= ru || r <= rl) {
        return Math<T>::zero();
    }
    const T x = (ru - r) / (ru - rl);
    const T dxdr = -Math<T>::one() / (ru - rl);
    const T dfdx = T(30) * x * x - T(60) * x * x * x + T(30) * x * x * x * x;
    return dfdx * dxdr;
}

template <typename T>
__device__ inline const T *atom_slice(const T *x, int max_size, int mol_idx, int atom_idx) {
    const long long atom_stride = 5LL * max_size;
    const long long mol_stride = static_cast<long long>(max_size) * atom_stride;
    return x + mol_idx * mol_stride + atom_idx * atom_stride;
}

__host__ __device__ inline long long fourier_atom_stride(int pmax, int order, int max_size) {
    return static_cast<long long>(pmax) * order * max_size;
}

template <typename T>
void ensure_capacity(T **ptr, size_t *cap, size_t need) {
    if (need <= *cap) {
        return;
    }
    if (*ptr != nullptr) {
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = nullptr;
    }
    CUDA_CHECK(cudaMalloc(ptr, need * sizeof(T)));
    *cap = need;
}

inline double get_angular_norm2(double t_width) {
    // Cache by width — the 20k-term sum is identical across FCHL18 calls with fixed hypers.
    struct Cache {
        double width = -1.0;
        double value = 0.0;
    };
    static Cache cache;
    if (cache.width == t_width) {
        return cache.value;
    }
    constexpr int limit = 10000;
    constexpr double pi = 3.14159265358979323846;
    double ang_norm2 = 0.0;
    for (int n = -limit; n <= limit; ++n) {
        const double tn = t_width * n;
        ang_norm2 += std::exp(-(tn * tn)) * (2.0 - 2.0 * std::cos(n * pi));
    }
    cache.width = t_width;
    cache.value = std::sqrt(ang_norm2 * pi) * 2.0;
    return cache.value;
}

inline int build_element_map_from_flags(const int *z_present, int *z_to_idx) {
    for (int z = 0; z < 256; ++z) {
        z_to_idx[z] = -1;
    }
    int pmax = 0;
    for (int z = 1; z < 256; ++z) {
        if (z_present[z]) {
            z_to_idx[z] = pmax++;
        }
    }
    return pmax;
}

// Host helper (defined in cuda_fchl18_kernel.cu): device exclusive scan
// offset[0]=0, offset[a+1]=offset[a]+3*n[a].
// If expected_dim >= 0, throws if offset[nm] != expected_dim.
void fill_coord_offsets_device(
    const int *d_n, int **d_offset, size_t *offset_cap, int nm, int expected_dim, const char *what
);

// Device-buffer fingerprint for train-side workspace caches.
// Samples many evenly spaced words so ordinary geometry / representation edits are
// visible even when the first, middle, and last slots are padding zeros / sentinels
// (e.g. small molecules padded to max_size).
template <typename T>
inline unsigned long long buffer_fingerprint(const T *d_ptr, size_t n_elem) {
    // Full-buffer checksum (chunked D2H). Sparse edge samples miss whole molecules
    // in padded multi-mol representations; callers only fingerprint on pointer hits.
    if (d_ptr == nullptr || n_elem == 0) {
        return 0;
    }
    static_assert(sizeof(T) <= sizeof(unsigned long long), "fingerprint word");
    unsigned long long fp = static_cast<unsigned long long>(n_elem);
    constexpr size_t kChunk = 4096;
    std::vector<T> host(std::min(n_elem, kChunk));
    for (size_t off = 0; off < n_elem; off += kChunk) {
        const size_t n = std::min(kChunk, n_elem - off);
        host.resize(n);
        CUDA_CHECK(cudaMemcpy(host.data(), d_ptr + off, n * sizeof(T), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            unsigned long long w = 0;
            std::memcpy(&w, &host[i], sizeof(T));
            fp ^= w + 0x9e3779b97f4a7c15ULL + (fp << 6) + (fp >> 2);
        }
    }
    return fp;
}

// ---------------------------------------------------------------------------
// Periodic-table alchemy (qml / CPU parity, see fchl18_alchemy.hpp)
//
// The scalar product with alchemy is
//   s_ij = pd(Zc_i, Zc_j) * [ 1 + sum_{p,q} pd(Z_p, Z_q) G(r_p - r_q)
//                                 * ( ksi_p ksi_q d_scale + a_scale * sum_m s_m
//                                     sum_{e1,e2} pd(Z_e1, Z_e2) (c_p[e1] c_q[e2] + s_p[e1] s_q[e2]) ) ]
// The element-channel mixing is linear and acts on one side only:
//   F_eff[e1] = sum_e2 PD[e1, e2] F[e2]    (PD symmetric, compact pmax x pmax)
// so the kernels keep the plain per-element dot product and are handed
// "effective" (PD-mixed) Fourier / dFourier buffers for the second side.
// What remains in-kernel is the neighbour weight pd(Zp, Zq) and pd(Zc, Zc').
// ---------------------------------------------------------------------------

// pd lookup by nuclear charge (0 outside 1..emax); mirrors kf::fchl18::pd_lookup.
template <typename T>
__device__ inline T alch_pd(const T *pd, int emax, int za, int zb) {
    if (za < 1 || zb < 1 || za > emax || zb > emax) {
        return Math<T>::zero();
    }
    return pd[static_cast<long long>(za - 1) * emax + (zb - 1)];
}

template <typename T>
struct AlchemyTables {
    T *d_pd = nullptr;   // emax x emax, Z-indexed
    T *d_pdc = nullptr;  // pmax x pmax compact (element-channel mixing)
    size_t pd_cap = 0;
    size_t pdc_cap = 0;
    double period_w = -1.0;
    double group_w = -1.0;
    std::vector<double> h_pd;  // host copy (double), emax x emax
};

template <typename T>
inline AlchemyTables<T> &alchemy_tables() {
    static AlchemyTables<T> t;
    return t;
}

// Ensure the device Z-indexed pd table matches the widths; returns the device pointer.
template <typename T>
inline const T *alchemy_upload_pd(double period_w, double group_w) {
    AlchemyTables<T> &t = alchemy_tables<T>();
    const int emax = kf::fchl18::kAlchemyEmaxDefault;
    if (t.d_pd == nullptr || t.period_w != period_w || t.group_w != group_w) {
        t.h_pd = kf::fchl18::build_periodic_table_pd(emax, period_w, group_w);
        std::vector<T> tmp(t.h_pd.size());
        for (size_t k = 0; k < tmp.size(); ++k) {
            tmp[k] = static_cast<T>(t.h_pd[k]);
        }
        ensure_capacity(&t.d_pd, &t.pd_cap, tmp.size());
        CUDA_CHECK(cudaMemcpy(t.d_pd, tmp.data(), tmp.size() * sizeof(T), cudaMemcpyHostToDevice));
        t.period_w = period_w;
        t.group_w = group_w;
    }
    return t.d_pd;
}

// Compact element-channel matrix PDc[p1*pmax+p2] = pd(Z(p1), Z(p2)); requires a prior
// alchemy_upload_pd for the same widths. Returns the device pointer.
template <typename T>
inline const T *alchemy_upload_pdc(const int *z_to_idx, int pmax) {
    AlchemyTables<T> &t = alchemy_tables<T>();
    const int emax = kf::fchl18::kAlchemyEmaxDefault;
    std::vector<int> idx_to_z(static_cast<size_t>(pmax), 0);
    for (int z = 1; z < 256; ++z) {
        const int p = z_to_idx[z];
        if (p >= 0 && p < pmax) {
            idx_to_z[static_cast<size_t>(p)] = z;
        }
    }
    std::vector<T> pdc(static_cast<size_t>(pmax) * pmax);
    for (int p1 = 0; p1 < pmax; ++p1) {
        for (int p2 = 0; p2 < pmax; ++p2) {
            pdc[static_cast<size_t>(p1) * pmax + p2] = static_cast<T>(kf::fchl18::pd_lookup(
                t.h_pd.data(), emax, idx_to_z[static_cast<size_t>(p1)],
                idx_to_z[static_cast<size_t>(p2)]
            ));
        }
    }
    ensure_capacity(&t.d_pdc, &t.pdc_cap, pdc.size());
    CUDA_CHECK(cudaMemcpy(t.d_pdc, pdc.data(), pdc.size() * sizeof(T), cudaMemcpyHostToDevice));
    return t.d_pdc;
}

// Mix the element-channel index p of per-atom Fourier-like buffers:
//   out[a, i][..p1..] = sum_p2 PDc[p1, p2] * in[a, i][..p2..]
// Per valid atom (flat = a*max_size + i, i < n[a]) the buffer starts at flat*atom_stride and
// is addressed as  o*outer_stride + c + p*pstride,  o < n_outer,  c < n_inner.
// n_inner = n_inner_base * (scale_by_na3 ? 3*n[a] : 1); pstride = pstride_is_inner ? n_inner
// : pstride_fixed.  Covers both the CPU layout [p][m][neigh][(na3)] (n_outer = 1) and the
// atom-major layout [neigh][p][m] (n_outer = max_size).
template <typename T>
__global__ void alchemy_mix_fourier_kernel(
    const T *in,
    T *out,
    const T *pdc,
    const int *n,
    int nm,
    int max_size,
    int pmax,
    long long atom_stride,
    int n_outer,
    long long outer_stride,
    long long n_inner_base,
    int scale_by_na3,
    int pstride_is_inner,
    long long pstride_fixed,
    long long n_inner_cap
) {
    const long long per_atom = static_cast<long long>(n_outer) * n_inner_cap;
    const long long total = static_cast<long long>(nm) * max_size * per_atom;
    for (long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long flat = t / per_atom;
        const long long rem = t - flat * per_atom;
        const int o = static_cast<int>(rem / n_inner_cap);
        const long long c = rem - static_cast<long long>(o) * n_inner_cap;
        const int a = static_cast<int>(flat / max_size);
        const int i = static_cast<int>(flat - static_cast<long long>(a) * max_size);
        if (i >= n[a]) {
            continue;
        }
        const long long n_inner = n_inner_base * (scale_by_na3 ? 3LL * n[a] : 1LL);
        if (c >= n_inner) {
            continue;
        }
        const long long pstride = pstride_is_inner ? n_inner : pstride_fixed;
        const long long base = flat * atom_stride + o * outer_stride + c;
        T v[kMaxElements];
        for (int p = 0; p < pmax; ++p) {
            v[p] = in[base + p * pstride];
        }
        for (int p1 = 0; p1 < pmax; ++p1) {
            T acc = Math<T>::zero();
            for (int p2 = 0; p2 < pmax; ++p2) {
                acc += pdc[p1 * pmax + p2] * v[p2];
            }
            out[base + p1 * pstride] = acc;
        }
    }
}

// Launch helper for alchemy_mix_fourier_kernel.
template <typename T>
inline void alchemy_mix_fourier(
    const T *in,
    T *out,
    const T *pdc,
    const int *n,
    int nm,
    int max_size,
    int pmax,
    long long atom_stride,
    int n_outer,
    long long outer_stride,
    long long n_inner_base,
    bool scale_by_na3,
    bool pstride_is_inner,
    long long pstride_fixed
) {
    // Capacity of n_inner over atoms: na3 <= 3*max_size.
    const long long n_inner_cap = n_inner_base * (scale_by_na3 ? 3LL * max_size : 1LL);
    const long long total =
        static_cast<long long>(nm) * max_size * static_cast<long long>(n_outer) * n_inner_cap;
    if (total <= 0) {
        return;
    }
    const int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 65535LL * 16) {
        grid = 65535LL * 16;
    }
    alchemy_mix_fourier_kernel<<<static_cast<unsigned>(grid), block>>>(
        in,
        out,
        pdc,
        n,
        nm,
        max_size,
        pmax,
        atom_stride,
        n_outer,
        outer_stride,
        n_inner_base,
        scale_by_na3 ? 1 : 0,
        pstride_is_inner ? 1 : 0,
        pstride_fixed,
        n_inner_cap
    );
    CUDA_CHECK(cudaGetLastError());
}

// Warp-shuffle block reduction (blockDim.x must be a multiple of 32, <= 1024).
template <typename T>
__device__ inline T block_reduce_sum(T val) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    __shared__ T warp_sums[32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wid = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        warp_sums[wid] = val;
    }
    __syncthreads();
    const int nwarps = static_cast<int>(blockDim.x) >> 5;
    val = (static_cast<int>(threadIdx.x) < nwarps) ? warp_sums[lane] : Math<T>::zero();
    if (wid == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
    }
    return val;
}

}  // namespace fchl18
}  // namespace kf
