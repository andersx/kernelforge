// cuda_solvers.hpp — GPU least-squares solver declarations

#pragma once

#include <torch/extension.h>

namespace kf {
namespace solvers {

// Solve min_w ||Z @ w - y||_2 via truncated SVD (GPU, FP32).
// Z: (m, n) float32 row-major tensor, or (n, m) col-major when z_col_major=true.
// y: (m,) float32 tensor.  rcond <= 0 uses machine-epsilon heuristic.
// Returns w: (n,) float32 CPU tensor.
at::Tensor cuda_solve_svd(at::Tensor Z, at::Tensor y, double rcond,
                           bool z_col_major = false);

// Solve min_w ||Z @ w - y||_2 via QR (GPU, FP32).
// Z: (m, n) float32 row-major, or (n, m) col-major when z_col_major=true.  m >= n.
// Returns w: (n,) float32 CPU tensor.
at::Tensor cuda_solve_qr(at::Tensor Z, at::Tensor y, bool z_col_major = false);

// Solve min_w ||Z @ w - y||_2 via cusolverDnSSgels IRS (GPU, FP32).
// Z: (m, n) float32 row-major, or (n, m) col-major when z_col_major=true.  m >= n.
// No rcond truncation — use cuda_solve_svd for rank-deficient systems.
// Returns w: (n,) float32 CPU tensor.
at::Tensor cuda_solve_gels(at::Tensor Z, at::Tensor y, bool z_col_major = false);

}  // namespace solvers
}  // namespace kf
