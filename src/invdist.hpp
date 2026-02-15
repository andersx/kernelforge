// invdist.hpp
#pragma once
#include <cstddef>  // size_t

namespace invdist {

// Number of i<j pairs for N atoms
std::size_t num_pairs(std::size_t N) noexcept;

// Strict upper-triangle (row-major) mapping: (i<j) -> row index p
std::size_t pair_to_index(std::size_t i, std::size_t j, std::size_t N) noexcept;

/**
 * Compute strict upper-triangle inverse distances into x (length M=N(N-1)/2).
 * R_flat: (3N) flattened coords (x1,y1,z1, x2,y2,z2, ..., xN,yN,zN)
 * eps: small distance floor (used as max(eps^2, r^2))
 */
void inverse_distance_upper(const double *R_flat, std::size_t N, double eps, double *x);

/**
 * Same as above, plus dense Jacobian J (M x 3N).
 * Row p (pair i<j) has nonzeros only in columns for atoms i and j:
 *   d(1/r_ij)/dr_i = -(r_i - r_j)/r^3
 *   d(1/r_ij)/dr_j = +(r_i - r_j)/r^3
 */
void inverse_distance_upper_and_jacobian(const double *R_flat, std::size_t N, double eps, double *x,
                                         double *J);

}  // namespace invdist
