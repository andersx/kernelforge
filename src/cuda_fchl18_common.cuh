#pragma once

// Shared device/host helpers for the CUDA FCHL18 kernels.
//
// Included by cuda_fchl18_kernel.cu (scalar kernels) and
// cuda_fchl18_jacobian.cu (dK/dR_A). Only small inline/template helpers live
// here; __global__ entry points stay in their translation unit.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

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
    constexpr int limit = 10000;
    constexpr double pi = 3.14159265358979323846;
    double ang_norm2 = 0.0;
    for (int n = -limit; n <= limit; ++n) {
        const double tn = t_width * n;
        ang_norm2 += std::exp(-(tn * tn)) * (2.0 - 2.0 * std::cos(n * pi));
    }
    return std::sqrt(ang_norm2 * pi) * 2.0;
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

}  // namespace fchl18
}  // namespace kf
