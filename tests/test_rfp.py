from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from kernelforge import global_kernels, kernelmath


def _sym_from_triangle(A: NDArray[np.float64], uplo: Literal["U", "L"]) -> NDArray[np.float64]:
    """Build a symmetric matrix from only one triangle of A (ignores the other)."""
    if uplo == "U":
        T = np.triu(A)
        result: NDArray[np.float64] = T + np.triu(A, 1).T
        return result
    else:
        T = np.tril(A)
        result2: NDArray[np.float64] = T + np.tril(A, -1).T
        return result2


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17, 32])
@pytest.mark.parametrize("uplo", ["U", "L"])
@pytest.mark.parametrize("transr", ["N", "T"])
@pytest.mark.parametrize("seed", [0])
def test_roundtrip_symmetric(
    n: int, uplo: Literal["U", "L"], transr: Literal["N", "T"], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = (M + M.T) * 0.5  # symmetric
    A = np.ascontiguousarray(A, dtype=np.float64)
    assert A.flags["C_CONTIGUOUS"]

    # full -> rfp
    arf = kernelmath.full_to_rfp(A, uplo=uplo, transr=transr)
    assert arf.ndim == 1
    assert arf.shape[0] == n * (n + 1) // 2
    assert arf.flags["C_CONTIGUOUS"]  # 1D contiguous

    # rfp -> full (C-order, no copy)
    B = kernelmath.rfp_to_full(arf, n=n, uplo=uplo, transr=transr)
    assert B.shape == (n, n)
    assert B.flags["C_CONTIGUOUS"]

    # Only the specified triangle is defined; it must match the input's triangle
    if uplo == "U":
        np.testing.assert_allclose(np.triu(B), np.triu(A), rtol=1e-13, atol=1e-13)
    else:
        np.testing.assert_allclose(np.tril(B), np.tril(A), rtol=1e-13, atol=1e-13)

    # Symmetrized view should match the original symmetric matrix closely
    Bs = _sym_from_triangle(B, uplo)
    np.testing.assert_allclose(Bs, A, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("n", [3, 7])
@pytest.mark.parametrize("uplo", ["U", "L"])
@pytest.mark.parametrize("transr", ["N", "T"])
@pytest.mark.parametrize("seed", [123])
def test_nonsymmetric_triangle_semantics(
    n: int, uplo: Literal["U", "L"], transr: Literal["N", "T"], seed: int
) -> None:
    """Round-trip uses only one triangle. For a non-symmetric input,
    the returned matrix's specified triangle should match that triangle of the input.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64, copy=False)
    A = np.ascontiguousarray(A)
    assert A.flags["C_CONTIGUOUS"]

    arf = kernelmath.full_to_rfp(A, uplo=uplo, transr=transr)
    B = kernelmath.rfp_to_full(arf, n=n, uplo=uplo, transr=transr)

    if uplo == "U":
        np.testing.assert_allclose(np.triu(B), np.triu(A), rtol=1e-13, atol=1e-13)
    else:
        np.testing.assert_allclose(np.tril(B), np.tril(A), rtol=1e-13, atol=1e-13)

    # Symmetrized B corresponds to "symmetrize the chosen triangle" of A
    expected = _sym_from_triangle(A, uplo)
    Bs = _sym_from_triangle(B, uplo)
    np.testing.assert_allclose(Bs, expected, rtol=1e-13, atol=1e-13)


def test_bad_length_raises() -> None:
    n = 5
    good = np.zeros(n * (n + 1) // 2, dtype=np.float64)
    bad = np.zeros(good.size + 1, dtype=np.float64)

    # Good length works
    _ = kernelmath.rfp_to_full(good, n, uplo="U", transr="N")

    # Bad length should raise
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        _ = kernelmath.rfp_to_full(bad, n, uplo="U", transr="N")


def test_c_contiguity_and_dtype() -> None:
    n = 6
    A = np.arange(n * n, dtype=np.float64).reshape(n, n)  # C-order by default
    arf = kernelmath.full_to_rfp(A, uplo="U", transr="N")
    assert arf.dtype == np.float64
    assert arf.flags["C_CONTIGUOUS"]

    B = kernelmath.rfp_to_full(arf, n, uplo="U", transr="N")
    assert B.dtype == np.float64
    assert B.flags["C_CONTIGUOUS"]

    # Triangle must match original's triangle
    np.testing.assert_allclose(np.triu(B), np.triu(A), rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17, 32, 100])
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_kernel_gaussian_symm_rfp(n: int, seed: int) -> None:
    """Test kernel_gaussian_symm_rfp against full kernel + full_to_rfp roundtrip.

    The C++ kernel_gaussian_symm_rfp writes directly into RFP format using
    Fortran UPLO='U' (upper triangle stored). Due to swap_uplo() in math_bindings.cpp,
    Python callers must use uplo='L' when calling rfp_to_full().
    """
    rng = np.random.default_rng(seed)
    d = min(5, n)  # dimension
    X = rng.standard_normal((n, d))
    sigma = 1.5
    alpha = -1.0 / (2 * sigma**2)

    # Direct RFP computation
    K_rfp = global_kernels.kernel_gaussian_symm_rfp(X, alpha)
    assert K_rfp.ndim == 1
    assert K_rfp.shape[0] == n * (n + 1) // 2
    assert K_rfp.flags["C_CONTIGUOUS"]

    # Reference: compute full kernel, then convert to RFP
    # kernel_gaussian_symm fills lower triangle only
    K_full_lower = global_kernels.kernel_gaussian_symm(X, alpha)
    # Mirror lower -> upper to get full symmetric matrix
    K_full = np.tril(K_full_lower) + np.tril(K_full_lower, -1).T

    # Convert full kernel to RFP using uplo='L'
    # (Note: uplo='L' in Python API maps to Fortran UPLO='U' via swap_uplo)
    K_rfp_ref = kernelmath.full_to_rfp(K_full, uplo="L", transr="N")

    # Compare RFP outputs
    np.testing.assert_allclose(K_rfp, K_rfp_ref, rtol=1e-12, atol=1e-12)

    # Also verify we can convert back to full and get a valid kernel
    K_reconstructed = kernelmath.rfp_to_full(K_rfp, n, uplo="L", transr="N")
    # rfp_to_full only fills one triangle; mirror it to compare full matrix
    K_reconstructed_symm = np.tril(K_reconstructed) + np.tril(K_reconstructed, -1).T
    np.testing.assert_allclose(K_reconstructed_symm, K_full, rtol=1e-12, atol=1e-12)

    # Verify kernel properties: diagonal should be 1.0 (distance to self = 0)
    K_diag = np.diag(K_reconstructed_symm)
    np.testing.assert_allclose(K_diag, 1.0, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [10, 50])
@pytest.mark.parametrize("seed", [0, 99])
def test_kernel_gaussian_symm_rfp_solve(n: int, seed: int) -> None:
    """Test that we can solve a linear system using the RFP kernel output.

    This verifies that the RFP kernel is positive-definite and correctly formatted
    for use with solve_cholesky_rfp_L.
    """
    rng = np.random.default_rng(seed)
    d = min(5, n)
    X = rng.standard_normal((n, d))
    sigma = 1.0
    alpha = -1.0 / (2 * sigma**2)

    # Compute kernel in RFP format
    K_rfp = global_kernels.kernel_gaussian_symm_rfp(X, alpha)

    # Create a test right-hand side
    y = rng.standard_normal(n)

    # Solve K @ alpha_solve = y using RFP Cholesky (overwrites K_rfp, so pass a copy)
    alpha_solve = kernelmath.cho_solve_rfp(K_rfp.copy(), y)
    assert alpha_solve.shape == (n,)

    # Verify solution by computing residual with full kernel
    K_full_lower = global_kernels.kernel_gaussian_symm(X, alpha)
    K_full = np.tril(K_full_lower) + np.tril(K_full_lower, -1).T
    residual = K_full @ alpha_solve - y
    residual_norm = np.linalg.norm(residual)

    # Should solve accurately
    assert residual_norm < 1e-8, f"Residual norm {residual_norm} too large"


def test_kernel_gaussian_symm_rfp_small() -> None:
    """Test kernel_gaussian_symm_rfp with a hand-crafted small example."""
    # Two points at distance 1.0 apart
    X = np.array([[0.0], [1.0]])
    alpha = -1.0
    K_rfp = global_kernels.kernel_gaussian_symm_rfp(X, alpha)

    # Convert to full to inspect values
    K = kernelmath.rfp_to_full(K_rfp, n=2, uplo="L", transr="N")
    K_symm = np.tril(K) + np.tril(K, -1).T

    # Diagonal should be 1.0
    assert K_symm[0, 0] == pytest.approx(1.0)
    assert K_symm[1, 1] == pytest.approx(1.0)

    # Off-diagonal: distance^2 = 1, so exp(-1.0 * 1) = exp(-1)
    assert K_symm[0, 1] == pytest.approx(np.exp(-1.0))
    assert K_symm[1, 0] == pytest.approx(np.exp(-1.0))
