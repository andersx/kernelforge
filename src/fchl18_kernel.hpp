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
//
// use_atm: if false, the Axilrod-Teller-Muto factor (1 + 3*cos_i*cos_j*cos_k)
//          is replaced with 1.0 in the three-body weight ksi3.
void kernel_gaussian(
    const std::vector<double> &x1,  // (nm1, max_size1, 5, max_size1)
    const std::vector<double> &x2,  // (nm2, max_size2, 5, max_size2)
    const std::vector<int> &n1,     // (nm1)
    const std::vector<int> &n2,     // (nm2)
    const std::vector<int> &nn1,    // (nm1 * max_size1)
    const std::vector<int> &nn2,    // (nm2 * max_size2)
    int nm1, int nm2, int max_size1, int max_size2, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order,
    bool use_atm,       // if false: replace ATM factor with 1.0
    double *kernel_out  // (nm1, nm2) row-major OUT
);

// Symmetric variant: K[a, b] = K[b, a].
void kernel_gaussian_symm(
    const std::vector<double> &x,  // (nm, max_size, 5, max_size)
    const std::vector<int> &n,     // (nm)
    const std::vector<int> &nn,    // (nm * max_size)
    int nm, int max_size, double sigma, double two_body_scaling, double two_body_width,
    double two_body_power, double three_body_scaling, double three_body_width,
    double three_body_power, double cut_start, double cut_distance, int fourier_order,
    bool use_atm,       // if false: replace ATM factor with 1.0
    double *kernel_out  // (nm, nm) row-major OUT
);

// Gradient of the FCHL18 Gaussian kernel w.r.t. Cartesian coordinates of
// molecule A (the query molecule).
//
// G[alpha, mu, b] = dK[A,b] / dR_A[alpha, mu]
//
// where alpha indexes atoms of A, mu in {0,1,2} = {x,y,z}, and b indexes
// training molecules in the second set.
//
// coords_A : (n_atoms_A * 3) row-major Cartesian coordinates of molecule A.
// z_A      : (n_atoms_A) nuclear charges of molecule A.
// x2,n2,nn2: pre-computed representations for training set B (same format as
//             the inputs to kernel_gaussian).
// grad_out : (n_atoms_A, 3, nm2) row-major output.
//            grad_out[alpha * 3 * nm2 + mu * nm2 + b] = dK[A,b]/dR_A[alpha,mu]
void kernel_gaussian_gradient(
    const std::vector<double> &coords_A,  // (n_atoms_A * 3)
    const std::vector<int> &z_A,          // (n_atoms_A)
    const std::vector<double> &x2,        // (nm2, max_size2, 5, max_size2)
    const std::vector<int> &n2,           // (nm2)
    const std::vector<int> &nn2,          // (nm2 * max_size2)
    int n_atoms_A, int nm2, int max_size2, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm,
    double *grad_out  // (n_atoms_A, 3, nm2) row-major OUT
);

// Mixed second derivative of the FCHL18 Gaussian kernel w.r.t. coordinates of
// both molecule A and molecule B (single pair).
//
// H[α*3+μ, β*3+ν] = d²K[A,B] / dR_A[α,μ] dR_B[β,ν]
//
// hess_out: (n_atoms_A*3, n_atoms_B*3) row-major output.
void kernel_gaussian_hessian(
    const std::vector<double> &coords_A,  // (n_atoms_A * 3)
    const std::vector<int> &z_A,          // (n_atoms_A)
    const std::vector<double> &coords_B,  // (n_atoms_B * 3)
    const std::vector<int> &z_B,          // (n_atoms_B)
    int n_atoms_A, int n_atoms_B, double sigma, double two_body_scaling, double two_body_width,
    double two_body_power, double three_body_scaling, double three_body_width,
    double three_body_power, double cut_start, double cut_distance, int fourier_order, bool use_atm,
    double *hess_out  // (n_atoms_A*3, n_atoms_B*3) row-major OUT
);

}  // namespace fchl18
}  // namespace kf
