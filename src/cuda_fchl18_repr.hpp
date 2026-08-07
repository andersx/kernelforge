#pragma once

#include <tuple>

#include <torch/extension.h>

namespace kf {
namespace fchl18 {

// GPU FCHL18 representation generation (fp32 / fp64).
//
// coords : (nm, max_size, 3) floating, CUDA — padded Cartesian coordinates
// z      : (nm, max_size)     int32,    CUDA — padded nuclear charges
// n      : (nm,)              int32,    CUDA — active atom counts
// cut_distance : neighbour cutoff (Angstrom)
//
// Returns:
//   x  : (nm, max_size, 5, max_size) same dtype as coords
//        Layout per atom: [r, Z, dx, dy, dz]; unused slots padded with 1e100
//        (fp64) or 1e30 (fp32).
//   nn : (nm, max_size) int32 — neighbour counts (including self at slot 0)
std::tuple<torch::Tensor, torch::Tensor> generate_fchl18_cuda(
    const torch::Tensor &coords,
    const torch::Tensor &z,
    const torch::Tensor &n,
    double cut_distance
);

}  // namespace fchl18
}  // namespace kf
