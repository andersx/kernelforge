#pragma once

namespace kf {
namespace fchl18 {
namespace direct {

// Rectangular FCHL18 Gaussian kernel from padded molecular coords (FP32).
// No distance cutoff: every other atom in the molecule is a neighbour.
// Molecules only (small N_atoms).
//
// coords: (nm, max_atoms, 3) float32 CUDA
// Z:      (nm, max_atoms)    int32  CUDA
// N:      (nm,)              int32  CUDA
// out:    (nm1, nm2)         float32 CUDA
void kernel_gaussian_rect_fp32(
    const float *d_coords1,
    const int *d_Z1,
    const int *d_N1,
    const float *d_coords2,
    const int *d_Z2,
    const int *d_N2,
    float *d_K_out,
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
