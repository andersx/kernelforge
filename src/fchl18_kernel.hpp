#pragma once
#include <cstddef>
#include <vector>

namespace kf {
namespace fchl18 {

// FCHL18 Gaussian molecular kernel (asymmetric).
//
// x1, x2: representations with layout (nm, max_size, 5, max_size), row-major.
//   Produced by generate_fchl18 for each molecule, then stacked.
// n1, n2: (nm1) / (nm2) number of real atoms per molecule.
// nn1, nn2: (nm1 * max_size1) / (nm2 * max_size2) neighbour counts per atom.
//
// K[a, b] = sum_{i in mol a, j in mol b: Z_i == Z_j}
//               exp( -(s_ii + s_jj - 2*s_ij) / sigma^2 )
//
// where s_ij is the FCHL18 scalar product between atoms i and j.
void kernel_gaussian(
    const std::vector<double> &x1,  // (nm1, max_size1, 5, max_size1)
    const std::vector<double> &x2,  // (nm2, max_size2, 5, max_size2)
    const std::vector<int>    &n1,  // (nm1)
    const std::vector<int>    &n2,  // (nm2)
    const std::vector<int>    &nn1, // (nm1 * max_size1)
    const std::vector<int>    &nn2, // (nm2 * max_size2)
    int nm1, int nm2, int max_size1, int max_size2,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int    fourier_order,
    double *kernel_out  // (nm1, nm2) row-major OUT
);

// Symmetric variant: K[a, b] = K[b, a].
void kernel_gaussian_symm(
    const std::vector<double> &x,   // (nm, max_size, 5, max_size)
    const std::vector<int>    &n,   // (nm)
    const std::vector<int>    &nn,  // (nm * max_size)
    int nm, int max_size,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int    fourier_order,
    double *kernel_out  // (nm, nm) row-major OUT
);

}  // namespace fchl18
}  // namespace kf
