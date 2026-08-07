#pragma once

namespace kf {
namespace fchl18 {

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
    bool use_atm
);

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
    bool use_atm
);

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
    bool use_atm
);

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
    bool use_atm
);

// Symmetric energy-only kernel in RFP packed format (TRANSR='N', UPLO='U').
// d_kernel_rfp_out must hold nm*(nm+1)/2 elements.
// Compatible with kernelmath.cho_solve_rfp. Unpack via:
//   kernelmath.rfp_to_full(..., uplo='L', transr='N')  # C-order UPLO swap
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
    bool use_atm
);

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
    bool use_atm
);

// Jacobian kernel: J[row, b] = dK(A_a, B_b) / dR_{A_a}[alpha, mu],
// with row = row_offset[a] + alpha*3 + mu and row_offset[a] = 3 * sum_{a' < a} n1[a'].
//
// Unlike the CPU entry point, the query side takes a precomputed representation
// (d_x1/d_n1/d_nn1). d_coords1 (nm1, max_size1, 3) and d_z1 (nm1, max_size1) are
// additionally required to map neighbour slots back to atom indices.
// d_jac_out must hold d_a_rows * nm2 elements, where d_a_rows = 3 * sum(n1).
void kernel_gaussian_jacobian_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const float *d_coords1,
    const int *d_z1,
    float *d_jac_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
);

void kernel_gaussian_jacobian_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const double *d_coords1,
    const int *d_z1,
    double *d_jac_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
);

// Jacobian-T: K_jt[b, row] = dK(A_a, B_b) / dR_{A_a}[...], shape (nm2, d_a_rows).
// Same inputs as kernel_gaussian_jacobian_cu; only the output layout changes.
void kernel_gaussian_jacobian_t_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const float *d_coords1,
    const int *d_z1,
    float *d_jac_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
);

void kernel_gaussian_jacobian_t_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const double *d_coords1,
    const int *d_z1,
    double *d_jac_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
);

// Hessian kernel: H[row, col] = d^2 K(A_a, B_b) / dR_{A_a}[alpha, mu] dR_{B_b}[beta, nu],
// with row = row_offset[a] + alpha*3 + mu and col = col_offset[b] + beta*3 + nu, where
// row_offset[a] = 3 * sum_{a' < a} n1[a'] and col_offset[b] = 3 * sum_{b' < b} n2[b'].
//
// Both sides take a precomputed representation plus the padded coords/charges the
// representation was generated from (needed to map neighbour slots back to atom
// indices for the chain rule). Every molecule pair contributes a dense block, so
// the blocks tile the whole output.
// d_hess_out must hold d_a_rows * d_b_cols elements, with d_a_rows = 3 * sum(n1)
// and d_b_cols = 3 * sum(n2).
void kernel_gaussian_hessian_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const float *d_coords1,
    const int *d_z1,
    const float *d_coords2,
    const int *d_z2,
    float *d_hess_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
);

void kernel_gaussian_hessian_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const double *d_coords1,
    const int *d_z1,
    const double *d_coords2,
    const int *d_z2,
    double *d_hess_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
);

// Symmetric Hessian: evaluates lower-triangle molecule pairs (b <= a) once,
// mirrors off-diagonal blocks, and symmetrises diagonal blocks.
// d_hess_out must hold d_rows * d_rows elements.
void kernel_gaussian_hessian_symm_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    const float *d_coords,
    const int *d_z,
    float *d_hess_out,
    float sigma,
    int nm,
    int max_size,
    int d_rows,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
);

void kernel_gaussian_hessian_symm_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    const double *d_coords,
    const int *d_z,
    double *d_hess_out,
    double sigma,
    int nm,
    int max_size,
    int d_rows,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
);

// Symmetric Hessian in RFP (TRANSR='N', UPLO='U'). Length d_rows*(d_rows+1)/2.
void kernel_gaussian_hessian_symm_rfp_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    const float *d_coords,
    const int *d_z,
    float *d_rfp_out,
    float sigma,
    int nm,
    int max_size,
    int d_rows,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
);

void kernel_gaussian_hessian_symm_rfp_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    const double *d_coords,
    const int *d_z,
    double *d_rfp_out,
    double sigma,
    int nm,
    int max_size,
    int d_rows,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
);

// Full combined kernel (asymmetric): shape (nm1+d_a_rows, nm2+d_b_cols).
void kernel_gaussian_full_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const float *d_coords1,
    const int *d_z1,
    const float *d_coords2,
    const int *d_z2,
    float *d_K_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
);

void kernel_gaussian_full_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const double *d_coords1,
    const int *d_z1,
    const double *d_coords2,
    const int *d_z2,
    double *d_K_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
);

// Symmetric full kernel: shape (nm+D, nm+D).
void kernel_gaussian_full_symm_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    const float *d_coords,
    const int *d_z,
    float *d_K_out,
    float sigma,
    int nm,
    int max_size,
    int D,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
);

void kernel_gaussian_full_symm_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    const double *d_coords,
    const int *d_z,
    double *d_K_out,
    double sigma,
    int nm,
    int max_size,
    int D,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
);

// Symmetric full kernel in RFP packed format (TRANSR='N', UPLO='U').
// d_rfp_out must hold BIG*(BIG+1)/2 elements with BIG = nm + D.
void kernel_gaussian_full_symm_rfp_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    const float *d_coords,
    const int *d_z,
    float *d_rfp_out,
    float sigma,
    int nm,
    int max_size,
    int D,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
);

void kernel_gaussian_full_symm_rfp_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    const double *d_coords,
    const int *d_z,
    double *d_rfp_out,
    double sigma,
    int nm,
    int max_size,
    int D,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
);

}  // namespace fchl18
}  // namespace kf
