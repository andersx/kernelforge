// C++ standard library
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Project headers
#include "blas_config.h"
#include "rff_features.hpp"

namespace kf::rff {

void rff_features(const double *X, const double *W, const double *b,
                  std::size_t N, std::size_t rep_size, std::size_t D,
                  double *Z) {
    if (!X || !W || !b || !Z)
        throw std::invalid_argument("rff_features: null pointer");
    if (N == 0 || rep_size == 0 || D == 0)
        throw std::invalid_argument("rff_features: zero dimension");

    // Z = X @ W  via DGEMM
    // X is (N, rep_size), W is (rep_size, D), Z is (N, D)
    // RowMajor: M=N, N=D, K=rep_size
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(N),         // M
                static_cast<int>(D),         // N
                static_cast<int>(rep_size),  // K
                1.0, X, static_cast<int>(rep_size),  // A, lda
                W, static_cast<int>(D),              // B, ldb
                0.0, Z, static_cast<int>(D));        // C, ldc

    // Z[:, d] = cos(Z[:, d] + b[d]) * sqrt(2/D)
    const double normalization = std::sqrt(2.0 / static_cast<double>(D));

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t d = 0; d < D; ++d) {
            Z[i * D + d] = std::cos(Z[i * D + d] + b[d]) * normalization;
        }
    }
}

}  // namespace kf::rff
