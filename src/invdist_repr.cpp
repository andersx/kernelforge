// invdist_repr.cpp
#include "invdist_repr.hpp"

#include <algorithm>
#include <cmath>

namespace kf {
namespace invdist {

std::size_t num_pairs(std::size_t N) noexcept {
    return N * (N - 1) / 2;
}

std::size_t pair_to_index(std::size_t i, std::size_t j, std::size_t N) noexcept {
    // assumes 0 <= i < j < N
    // p = i*(N-1) - i*(i+1)/2 + (j - i - 1)
    return i * (N - 1) - (i * (i + 1)) / 2 + (j - i - 1);
}

void inverse_distance_upper(const double *R_flat, std::size_t N, double eps, double *x) {
    const double eps2 = eps * eps;
    std::size_t p = 0;

    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t i0 = 3 * i;
        const double xi = R_flat[i0 + 0];
        const double yi = R_flat[i0 + 1];
        const double zi = R_flat[i0 + 2];

        for (std::size_t j = i + 1; j < N; ++j, ++p) {
            const std::size_t j0 = 3 * j;
            const double Dx = xi - R_flat[j0 + 0];
            const double Dy = yi - R_flat[j0 + 1];
            const double Dz = zi - R_flat[j0 + 2];

            double r2 = Dx * Dx + Dy * Dy + Dz * Dz;
            if (r2 < eps2)
                r2 = eps2;

            x[p] = 1.0 / std::sqrt(r2);
        }
    }
}

void inverse_distance_upper_and_jacobian(const double *R_flat, std::size_t N, double eps, double *x,
                                         double *J) {
    const std::size_t M = num_pairs(N);
    const std::size_t D = 3 * N;

    // Zero J (only two atoms per row are nonzero)
    std::fill(J, J + M * D, 0.0);

    const double eps2 = eps * eps;
    std::size_t p = 0;

    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t i0 = 3 * i;
        const double xi = R_flat[i0 + 0];
        const double yi = R_flat[i0 + 1];
        const double zi = R_flat[i0 + 2];

        for (std::size_t j = i + 1; j < N; ++j, ++p) {
            const std::size_t j0 = 3 * j;
            const double Dx = xi - R_flat[j0 + 0];
            const double Dy = yi - R_flat[j0 + 1];
            const double Dz = zi - R_flat[j0 + 2];

            double r2 = Dx * Dx + Dy * Dy + Dz * Dz;
            if (r2 < eps2)
                r2 = eps2;

            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r / r2;  // 1/r^3

            x[p] = inv_r;

            const double gx = Dx * inv_r3;
            const double gy = Dy * inv_r3;
            const double gz = Dz * inv_r3;

            // Columns for atom i and j in flattened (x1,y1,z1, x2,y2,z2, ...)
            J[p * D + (i0 + 0)] = -gx;
            J[p * D + (i0 + 1)] = -gy;
            J[p * D + (i0 + 2)] = -gz;

            J[p * D + (j0 + 0)] = +gx;
            J[p * D + (j0 + 1)] = +gy;
            J[p * D + (j0 + 2)] = +gz;
        }
    }
}

}  // namespace invdist
}  // namespace kf
