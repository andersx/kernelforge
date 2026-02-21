#pragma once

#include <cstddef>

namespace kf::rff {

// Compute Random Fourier Features (basic):
//   Z[i, d] = sqrt(2/D) * cos(X[i,:] @ W[:,d] + b[d])
//
// X:  (N, rep_size) row-major - input features
// W:  (rep_size, D) row-major - random weight matrix
// b:  (D,) - random bias vector
// Z:  (N, D) row-major output
void rff_features(const double *X, const double *W, const double *b,
                  std::size_t N, std::size_t rep_size, std::size_t D,
                  double *Z);

}  // namespace kf::rff
