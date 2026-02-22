"""Tests for RFP (Row-First Packed) Hessian kernel."""

import numpy as np
import pytest

from kernelforge.global_kernels import (
    kernel_gaussian_hessian_symm,
    kernel_gaussian_hessian_symm_rfp,
)


def rfp_to_full(H_rfp, n):
    """Convert RFP packed format to full symmetric matrix.

    Args:
        H_rfp: 1D array of length n*(n+1)/2 in RFP format
        n: Matrix dimension

    Returns:
        Full n×n symmetric matrix
    """
    H_full = np.zeros((n, n))

    k = n // 2
    stride = (n + 1) if (n % 2 == 0) else n

    for i in range(n):
        for j in range(i, n):
            # RFP index for upper triangle
            if j >= k:
                idx = (j - k) * stride + i
            else:
                idx = i * stride + j + k + 1

            H_full[i, j] = H_rfp[idx]
            H_full[j, i] = H_rfp[idx]  # Symmetry

    return H_full


def test_rfp_memory_savings():
    """Test that RFP format saves approximately 50% memory."""
    N, M, D = 20, 10, 9
    X = np.random.randn(N, M)
    dX = np.random.randn(N, M, D)
    sigma = 1.5

    H_full = kernel_gaussian_hessian_symm(X, dX, sigma)
    H_rfp = kernel_gaussian_hessian_symm_rfp(X, dX, sigma)

    BIG = N * D
    expected_rfp_size = BIG * (BIG + 1) // 2

    assert H_rfp.shape == (expected_rfp_size,), (
        f"RFP shape mismatch: {H_rfp.shape} vs {expected_rfp_size}"
    )
    assert H_full.shape == (BIG, BIG), f"Full shape mismatch: {H_full.shape}"

    # Memory savings should be ~49-50%
    savings = 100 * (1 - H_rfp.nbytes / H_full.nbytes)
    assert 48.0 < savings < 51.0, f"Memory savings {savings:.1f}% not in expected range"


def test_rfp_values_match_full():
    """Test that RFP and full matrix contain the same values."""
    N, M, D = 5, 8, 6
    np.random.seed(42)
    X = np.random.randn(N, M)
    dX = np.random.randn(N, M, D)
    sigma = 1.0

    # Compute both versions
    H_full = kernel_gaussian_hessian_symm(X, dX, sigma)
    H_rfp = kernel_gaussian_hessian_symm_rfp(X, dX, sigma)

    # Convert RFP to full for comparison
    BIG = N * D
    H_from_rfp = rfp_to_full(H_rfp, BIG)

    # Compare lower triangle (full version only computes lower triangle)
    for i in range(BIG):
        for j in range(i + 1):
            full_val = H_full[i, j]
            rfp_val = H_from_rfp[i, j]
            assert np.isclose(full_val, rfp_val, rtol=1e-10, atol=1e-12), (
                f"Mismatch at ({i},{j}): full={full_val}, rfp={rfp_val}"
            )


def test_rfp_symmetry():
    """Test that RFP encodes a symmetric matrix."""
    N, M, D = 8, 5, 9
    np.random.seed(123)
    X = np.random.randn(N, M)
    dX = np.random.randn(N, M, D)
    sigma = 2.0

    H_rfp = kernel_gaussian_hessian_symm_rfp(X, dX, sigma)

    # Convert to full and check symmetry
    BIG = N * D
    H = rfp_to_full(H_rfp, BIG)

    assert np.allclose(H, H.T, rtol=1e-10, atol=1e-12), "RFP matrix is not symmetric"


def test_rfp_different_sizes():
    """Test RFP with various problem sizes."""
    test_cases = [
        (3, 4, 6),  # Small
        (10, 8, 9),  # Medium
        (15, 5, 12),  # Larger
    ]

    for N, M, D in test_cases:
        X = np.random.randn(N, M)
        dX = np.random.randn(N, M, D)
        sigma = 1.0

        H_rfp = kernel_gaussian_hessian_symm_rfp(X, dX, sigma)

        BIG = N * D
        expected_size = BIG * (BIG + 1) // 2
        assert H_rfp.shape == (expected_size,), (
            f"Size mismatch for N={N}, M={M}, D={D}: {H_rfp.shape} vs {expected_size}"
        )


def test_rfp_with_tile_parameter():
    """Test RFP with different tile_B values."""
    N, M, D = 10, 6, 9
    X = np.random.randn(N, M)
    dX = np.random.randn(N, M, D)
    sigma = 1.5

    # Default tile
    H1 = kernel_gaussian_hessian_symm_rfp(X, dX, sigma)

    # Explicit tile sizes
    H2 = kernel_gaussian_hessian_symm_rfp(X, dX, sigma, tile_B=4)
    H3 = kernel_gaussian_hessian_symm_rfp(X, dX, sigma, tile_B=16)

    # All should give identical results
    assert np.allclose(H1, H2, rtol=1e-12, atol=1e-14)
    assert np.allclose(H1, H3, rtol=1e-12, atol=1e-14)


def test_rfp_invalid_inputs():
    """Test that RFP handles invalid inputs correctly."""
    N, M, D = 5, 4, 6
    X = np.random.randn(N, M)
    dX = np.random.randn(N, M, D)

    # Invalid sigma
    with pytest.raises(ValueError):
        kernel_gaussian_hessian_symm_rfp(X, dX, sigma=0.0)

    with pytest.raises(ValueError):
        kernel_gaussian_hessian_symm_rfp(X, dX, sigma=-1.0)

    # Shape mismatches
    with pytest.raises(ValueError):
        kernel_gaussian_hessian_symm_rfp(X, dX[:-1, :, :], sigma=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
