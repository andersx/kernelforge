#pragma once

#include <tuple>

// Third-party
#include <torch/extension.h>

namespace kf {
namespace fchl19 {

// GPU FCHL19 forward representation (FP32, batched padded tensors).
//
// coords               : (nm, max_atoms, 3)  float32, CUDA
// Q                    : (nm, max_atoms)      int32,   CUDA  — element indices [0..nelements)
// N                    : (nm,)                int32,   CUDA  — active atom counts
// nelements            : number of distinct element types
// nRs2 / nRs3 / nFourier: basis-set sizes
// eta2, eta3, zeta     : Gaussian width / angular width parameters
// rcut, acut           : two-body / three-body cutoff radii
// two_body_decay       : power-law decay for 2-body prefactor
// three_body_decay     : power-law decay for 3-body ATM denominator
// three_body_weight_norm: pre-normalised three-body weight = sqrt(eta3/pi) * user_weight
//
// Returns              : (nm, max_atoms, rep_size) float32, CUDA
//   Padded atom slots (i >= N[m]) are zeroed.

torch::Tensor generate_fchl_acsf_cuda(
    const torch::Tensor &coords,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    int nelements,
    int nRs2,
    int nRs3,
    int nFourier,
    float eta2,
    float eta3,
    float zeta,
    float rcut,
    float acut,
    float two_body_decay,
    float three_body_decay,
    float three_body_weight_norm,
    bool deterministic = false
);

std::tuple<torch::Tensor, torch::Tensor> generate_fchl_acsf_and_gradients_cuda(
    const torch::Tensor &coords,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    int nelements,
    int nRs2,
    int nRs3,
    int nFourier,
    float eta2,
    float eta3,
    float zeta,
    float rcut,
    float acut,
    float two_body_decay,
    float three_body_decay,
    float three_body_weight_norm,
    bool deterministic = false
);

}  // namespace fchl19
}  // namespace kf
