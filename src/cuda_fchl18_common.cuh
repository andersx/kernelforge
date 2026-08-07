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
