#pragma once

#include <tuple>

// Third-party
#include <torch/extension.h>

namespace kf {
namespace invdist_cuda {

// GPU batched inverse-distance representation (FP32).
//
// coords  : (nm, n_atoms, 3)   float32, CUDA  — all molecules same n_atoms
// n_atoms : number of atoms per molecule
// eps     : distance floor  (r is clamped to max(r, eps))
//
// Returns X : (nm, M) float32, CUDA    M = n_atoms*(n_atoms-1)/2
//   Entry [m, p] = 1/r_{ij} for the p-th pair (i<j) in molecule m.
torch::Tensor inverse_distance_upper_cuda(
    const torch::Tensor &coords, int n_atoms, float eps
);

// Same as above, plus the Jacobian dX/dR in (D, M) layout per molecule.
//
// Returns:
//   X  : (nm, M)    float32, CUDA
//   dX : (nm, D, M) float32, CUDA    D = 3*n_atoms
//     dX[m, 3*a + d, p] = d(1/r_{ij})/d(R_{a,d})  for pair p=(i<j), atom a, dimension d
std::tuple<torch::Tensor, torch::Tensor> inverse_distance_upper_and_jacobian_cuda(
    const torch::Tensor &coords, int n_atoms, float eps
);

}  // namespace invdist_cuda
}  // namespace kf
