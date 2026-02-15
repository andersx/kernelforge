import numpy as np
import pytest

from kernelforge import _cholesky


def _sym_from_triangle(A, uplo):  # type: ignore
    """Build a symmetric matrix from only one triangle of A (ignores the other)."""
    if uplo == "U":
        T = np.triu(A)
        return T + np.triu(A, 1).T
    else:
        T = np.tril(A)
        return T + np.tril(A, -1).T


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17, 32])
@pytest.mark.parametrize("uplo", ["U", "L"])
@pytest.mark.parametrize("transr", ["N", "T"])
def test_roundtrip_symmetric(n, uplo, transr, seed=0) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = (M + M.T) * 0.5  # symmetric
    A = np.ascontiguousarray(A, dtype=np.float64)
    assert A.flags["C_CONTIGUOUS"]

    # full -> rfp
    arf = _cholesky.full_to_rfp(A, uplo=uplo, transr=transr)
    assert arf.ndim == 1
    assert arf.shape[0] == n * (n + 1) // 2
    assert arf.flags["C_CONTIGUOUS"]  # 1D contiguous

    # rfp -> full (C-order, no copy)
    B = _cholesky.rfp_to_full(arf, n=n, uplo=uplo, transr=transr)
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
def test_nonsymmetric_triangle_semantics(n, uplo, transr, seed=123) -> None:  # type: ignore[no-untyped-def]
    """Round-trip uses only one triangle. For a non-symmetric input,
    the returned matrix's specified triangle should match that triangle of the input.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64, copy=False)
    A = np.ascontiguousarray(A)
    assert A.flags["C_CONTIGUOUS"]

    arf = _cholesky.full_to_rfp(A, uplo=uplo, transr=transr)
    B = _cholesky.rfp_to_full(arf, n=n, uplo=uplo, transr=transr)

    if uplo == "U":
        np.testing.assert_allclose(np.triu(B), np.triu(A), rtol=1e-13, atol=1e-13)
    else:
        np.testing.assert_allclose(np.tril(B), np.tril(A), rtol=1e-13, atol=1e-13)

    # Symmetrized B corresponds to "symmetrize the chosen triangle" of A
    expected = _sym_from_triangle(A, uplo)
    Bs = _sym_from_triangle(B, uplo)
    np.testing.assert_allclose(Bs, expected, rtol=1e-13, atol=1e-13)


def test_bad_length_raises() -> None:  # type: ignore[no-untyped-def]
    n = 5
    good = np.zeros(n * (n + 1) // 2, dtype=np.float64)
    bad = np.zeros(good.size + 1, dtype=np.float64)

    # Good length works
    _ = _cholesky.rfp_to_full(good, n, uplo="U", transr="N")

    # Bad length should raise
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        _ = _cholesky.rfp_to_full(bad, n, uplo="U", transr="N")


def test_c_contiguity_and_dtype() -> None:  # type: ignore[no-untyped-def]
    n = 6
    A = np.arange(n * n, dtype=np.float64).reshape(n, n)  # C-order by default
    arf = _cholesky.full_to_rfp(A, uplo="U", transr="N")
    assert arf.dtype == np.float64 and arf.flags["C_CONTIGUOUS"]

    B = _cholesky.rfp_to_full(arf, n, uplo="U", transr="N")
    assert B.dtype == np.float64 and B.flags["C_CONTIGUOUS"]

    # Triangle must match original's triangle
    np.testing.assert_allclose(np.triu(B), np.triu(A), rtol=0, atol=0)
