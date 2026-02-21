from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from kernelforge import kernelmath


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
