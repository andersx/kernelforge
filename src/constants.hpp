#pragma once
#include <cstddef>

namespace kf {

// ---- Memory alignment ----
// SIMD/cache alignment for optimal performance
constexpr std::size_t ALIGNMENT_BYTES = 64;

// ---- BLAS/kernel tuning ----
// Default tile size for blocked matrix operations
constexpr std::size_t DEFAULT_TILE_SIZE = 64;

// ---- Numerical constants ----
// Small epsilon for numerical stability
constexpr double EPS = 1e-12;

// Mathematical constants
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double SQRT_2PI = 2.5066282746310005;  // sqrt(2*pi)

}  // namespace kf
