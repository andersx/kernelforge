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
    const double *x1,  // (nm1, max_size1, 5, max_size1)
    const double *x2,  // (nm2, max_size2, 5, max_size2)
    const int *n1,     // (nm1)
    const int *n2,     // (nm2)
    const int *nn1,    // (nm1 * max_size1)
    const int *nn2,    // (nm2 * max_size2)
    int nm1, int nm2, int max_size1, int max_size2, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order,
    bool use_atm,  // if false: replace ATM factor with 1.0
    double *kernel_out,  // (nm1, nm2) row-major OUT
    // Periodic-table alchemy (qml default off in KernelForge for backward compat).
    // When use_alchemy=true, neighbour/central Z pairs are weighted by pd(Zi,Zj).
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// Symmetric variant: K[a, b] = K[b, a].
void kernel_gaussian_symm(
    const double *x,  // (nm, max_size, 5, max_size)
    const int *n,     // (nm)
    const int *nn,    // (nm * max_size)
    int nm, int max_size, double sigma, double two_body_scaling, double two_body_width,
    double two_body_power, double three_body_scaling, double three_body_width,
    double three_body_power, double cut_start, double cut_distance, int fourier_order,
    bool use_atm,        // if false: replace ATM factor with 1.0
    double *kernel_out,  // (nm, nm) row-major OUT
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// Symmetric kernel written directly in RFP-packed format (TRANSR='N', UPLO='U'),
// no dense (nm x nm) intermediate.  rfp_out has length nm*(nm+1)/2.
void kernel_gaussian_symm_rfp_core(
    const double *x, const int *n, const int *nn, int nm, int max_size, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm, double *rfp_out,
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
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
    const double *coords_A,  // (n_atoms_A * 3)
    const int *z_A,          // (n_atoms_A)
    const double *x2,        // (nm2, max_size2, 5, max_size2)
    const int *n2,           // (nm2)
    const int *nn2,          // (nm2 * max_size2)
    int n_atoms_A, int nm2, int max_size2, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm,
    double *grad_out,  // (n_atoms_A, 3, nm2) row-major OUT
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// Batch Jacobian: J[row, b] = dK(A_i, B_b) / dR_{A_i}[flat], shape (D_A, nm2)
// where D_A = sum_i n_atoms_i * 3 and row offsets follow molecule order.
// Hoists B-side precompute (element map, AtomData, self-scalars) across all A.
void kernel_gaussian_jacobian(
    const std::vector<std::vector<double>> &coords_A_list,  // each (n_atoms_i * 3)
    const std::vector<std::vector<int>> &z_A_list,          // each (n_atoms_i)
    const double *x2, const int *n2, const int *nn2, int nm2, int max_size2, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm,
    double *J_out,  // (D_A, nm2) row-major OUT
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// Jacobian-transpose: K_jt[b, flat_A] = dK(A_i, B_b) / dR_{A_i}[flat]
// Shape (nm2, D_A). Same work as kernel_gaussian_jacobian with transposed store.
// coords_A / z_A are the differentiated molecules (e.g. training); x2 is the other set.
void kernel_gaussian_jacobian_t(
    const std::vector<std::vector<double>> &coords_A_list,  // each (n_atoms_i * 3)
    const std::vector<std::vector<int>> &z_A_list,          // each (n_atoms_i)
    const double *x2, const int *n2, const int *nn2, int nm2, int max_size2, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm,
    double *K_jt_out,  // (nm2, D_A) row-major OUT
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// Mixed second derivative of the FCHL18 Gaussian kernel w.r.t. coordinates of
// both molecule A and molecule B (single pair).
//
// H[α*3+μ, β*3+ν] = d²K[A,B] / dR_A[α,μ] dR_B[β,ν]
//
// hess_out: (n_atoms_A*3, n_atoms_B*3) row-major output.
void kernel_gaussian_hessian(
    const double *coords_A,  // (n_atoms_A * 3)
    const int *z_A,          // (n_atoms_A)
    const double *coords_B,  // (n_atoms_B * 3)
    const int *z_B,          // (n_atoms_B)
    int n_atoms_A, int n_atoms_B, double sigma, double two_body_scaling, double two_body_width,
    double two_body_power, double three_body_scaling, double three_body_width,
    double three_body_power, double cut_start, double cut_distance, int fourier_order, bool use_atm,
    double *hess_out,  // (n_atoms_A*3, n_atoms_B*3) row-major OUT
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// Batch Hessian over many A × B pairs. Hoists per-molecule AtomDataGrad once.
// H_out shape (D_A, D_B) with molecule blocks in molecule order
// (D_A = sum_i n_atoms_A_i * 3, likewise D_B).
void kernel_gaussian_hessian_rect(
    const std::vector<std::vector<double>> &coords_A_list,
    const std::vector<std::vector<int>> &z_A_list,
    const std::vector<std::vector<double>> &coords_B_list,
    const std::vector<std::vector<int>> &z_B_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm,
    double *H_out,  // (D_A, D_B) row-major OUT
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// Symmetric batch Hessian: fills (D, D), lower-triangle pairs + mirror,
// diagonal blocks symmetrised.
void kernel_gaussian_hessian_symm_blocks(
    const std::vector<std::vector<double>> &coords_list,
    const std::vector<std::vector<int>> &z_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm,
    double *H_out,  // (D, D) row-major OUT
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// Same as hessian_symm_blocks but writes upper-triangle RFP directly (no D×D temp).
void kernel_gaussian_hessian_symm_rfp_blocks(
    const std::vector<std::vector<double>> &coords_list,
    const std::vector<std::vector<int>> &z_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm,
    double *rfp_out,  // D*(D+1)/2 OUT
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// Phase timers for KF_HESS_PHASE=1 (stderr report). No-ops when unset.
void hess_phase_reset();
void hess_phase_report(const char *tag);

// Contracted inference: F_out += H @ alpha_F without materialising H.
void kernel_gaussian_hessian_matvec(
    const std::vector<std::vector<double>> &coords_A_list,
    const std::vector<std::vector<int>> &z_A_list,
    const std::vector<std::vector<double>> &coords_B_list,
    const std::vector<std::vector<int>> &z_B_list, const double *alpha_F, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm,
    double *F_out,  // (D_A,) row-major OUT (accumulated into)
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// F_out += J @ alpha_E where J is (D_A, N_B) Jacobian block.
void kernel_gaussian_jacobian_matvec(
    const std::vector<std::vector<double>> &coords_A_list,
    const std::vector<std::vector<int>> &z_A_list, const double *x2, const int *n2, const int *nn2,
    int nm2, int max_size2, const double *alpha_E, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm,
    double *F_out,  // (D_A,) OUT (accumulated into)
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

// E_out += J^T @ alpha_F where J^T is (N_A, D_B) block of the full kernel.
void kernel_gaussian_jacobian_t_matvec(
    const std::vector<std::vector<double>> &coords_B_list,
    const std::vector<std::vector<int>> &z_B_list, const double *x_A, const int *n_A,
    const int *nn_A, int nm_A, int max_size_A, const double *alpha_F, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm,
    double *E_out,  // (N_A,) OUT (accumulated into)
    bool use_alchemy = false, double alchemy_period_width = 1.6,
    double alchemy_group_width = 1.6
);

}  // namespace fchl18
}  // namespace kf
