#pragma once

#include <cstddef>

namespace kf {

// RFP (Rectangular Full Packed) index mapping.
//
// Maps a 2D index (i, j) with i <= j < n to the 1D position in an RFP-packed
// array.  Convention: TRANSR='N', UPLO='U' (upper triangle stored).
//
// Precondition: 0 <= i <= j < n
// Returns: 0-based linear index into the packed array of length n*(n+1)/2
inline std::size_t rfp_index_upper_N(std::size_t n, std::size_t i, std::size_t j) {
    const std::size_t k = n / 2;
    const std::size_t stride = (n % 2 == 0) ? (n + 1) : n;
    if (j >= k) {
        return (j - k) * stride + i;
    } else {
        return i * stride + (j + k + 1);
    }
}

}  // namespace kf
