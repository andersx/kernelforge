#pragma once

namespace kf {
namespace fchl18 {
namespace direct {

// Coords-only FCHL18 Gaussian Hessian kernel (FP32, no cutoff, molecules only):
//   H[(a,amu), (b,bnu)] = d^2 K[a,b] / dR_a[amu] dR_b[bnu]
// with K[a,b] = sum_{i in a, j in b} exp(-(s_ii + s_jj - 2 s_ij) / (2 sigma^2)).
//
// coords: (nm, max_atoms, 3) float32 CUDA
// Z:      (nm, max_atoms)    int32  CUDA
// N:      (nm,)              int32  CUDA
// out:    (D_A, D_B) row-major float32 CUDA, D_A = 3*sum(N1), D_B = 3*sum(N2).
//         Fully overwritten (zeroed first).
void kernel_gaussian_hessian_fp32(
    const float *d_coords1,
    const int *d_Z1,
    const int *d_N1,
    const float *d_coords2,
    const int *d_Z2,
    const int *d_N2,
    float *d_H_out,
    float sigma,
    int nm1,
    int nm2,
    int max_atoms1,
    int max_atoms2,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    int fourier_order,
    bool use_atm,
    bool use_alchemy = false,
    float alchemy_period_width = 1.6f,
    float alchemy_group_width = 1.6f
);

}  // namespace direct
}  // namespace fchl18
}  // namespace kf
