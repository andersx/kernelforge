"""Tests for local scalar kernels: kernel_gaussian, kernel_gaussian_symm,
kernel_gaussian_symm_rfp.

The reference implementation is a naive Python loop over all atom pairs,
which is easy to verify by inspection.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

import kernelforge.local_kernels as lk

# ---------------------------------------------------------------------------
# Reference (naive) implementations
# ---------------------------------------------------------------------------


def _ref_kernel_gaussian(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    q1: NDArray[np.int32],
    q2: NDArray[np.int32],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]:
    """Pure-Python reference: K[a,b] = sum_{j1,j2: q1[a,j1]==q2[b,j2]}
    exp(-||x1[a,j1]-x2[b,j2]||^2 / (2*sigma^2))"""
    nm1, max_atoms1, _rep = x1.shape
    nm2, max_atoms2, _ = x2.shape
    inv_2sigma2 = -1.0 / (2.0 * sigma * sigma)

    K = np.zeros((nm1, nm2), dtype=np.float64)
    for a in range(nm1):
        na = int(np.clip(n1[a], 0, max_atoms1))
        for b in range(nm2):
            nb = int(np.clip(n2[b], 0, max_atoms2))
            kab = 0.0
            for j1 in range(na):
                for j2 in range(nb):
                    if q1[a, j1] != q2[b, j2]:
                        continue
                    d = x1[a, j1, :] - x2[b, j2, :]
                    kab += float(np.exp(np.dot(d, d) * inv_2sigma2))
            K[a, b] = kab
    return K


def _ref_kernel_gaussian_symm(
    x: NDArray[np.float64],
    q: NDArray[np.int32],
    n: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]:
    """Symmetric variant: K[a,b] = K[b,a] via the same formula."""
    K_full = _ref_kernel_gaussian(x, x, q, q, n, n, sigma)
    # By construction K_full should already be symmetric; return it
    return K_full


def _ref_kernel_gaussian_symm_rfp(
    x: NDArray[np.float64],
    q: NDArray[np.int32],
    n: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]:
    """Pack upper triangle of the symmetric kernel into RFP format.
    TRANSR='N', UPLO='U': matches rfp_index_upper_N in the C++ code."""
    K = _ref_kernel_gaussian_symm(x, q, n, sigma)
    nm = K.shape[0]
    nt = nm * (nm + 1) // 2
    arf = np.zeros(nt, dtype=np.float64)

    k = nm // 2
    stride = (nm + 1) if (nm % 2 == 0) else nm

    for i in range(nm):
        for j in range(i, nm):  # upper triangle (i <= j)
            if j >= k:
                idx = (j - k) * stride + i
            else:
                idx = i * stride + (j + k + 1)
            arf[idx] = K[i, j]
    return arf


# ---------------------------------------------------------------------------
# Fixtures / helper data builders
# ---------------------------------------------------------------------------


def _make_dataset(
    nm1: int,
    nm2: int,
    max_atoms: int,
    rep_size: int,
    n_species: int,
    seed: int,
    *,
    n_min: int = 1,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """Return x1, x2, q1, q2, n1, n2 for an asymmetric test."""
    rng = np.random.default_rng(seed)
    species = np.arange(1, n_species + 1, dtype=np.int32)

    x1 = rng.standard_normal((nm1, max_atoms, rep_size))
    x2 = rng.standard_normal((nm2, max_atoms, rep_size))

    n1 = rng.integers(n_min, max_atoms + 1, size=nm1).astype(np.int32)
    n2 = rng.integers(n_min, max_atoms + 1, size=nm2).astype(np.int32)

    q1 = np.zeros((nm1, max_atoms), dtype=np.int32)
    q2 = np.zeros((nm2, max_atoms), dtype=np.int32)
    for a in range(nm1):
        q1[a, : n1[a]] = rng.choice(species, size=n1[a])
    for b in range(nm2):
        q2[b, : n2[b]] = rng.choice(species, size=n2[b])

    return (
        x1.astype(np.float64),
        x2.astype(np.float64),
        q1,
        q2,
        n1,
        n2,
    )


def _make_symm_dataset(
    nm: int,
    max_atoms: int,
    rep_size: int,
    n_species: int,
    seed: int,
    *,
    n_min: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Return x, q, n for a symmetric test."""
    x1, _, q1, _, n1, _ = _make_dataset(nm, nm, max_atoms, rep_size, n_species, seed, n_min=n_min)
    return x1, q1, n1


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 999])
def test_kernel_gaussian_matches_reference(seed: int) -> None:
    """kernel_gaussian should match the naive reference implementation."""
    nm1, nm2 = 4, 5
    max_atoms, rep_size, n_species = 6, 8, 3
    sigma = 2.5

    x1, x2, q1, q2, n1, n2 = _make_dataset(nm1, nm2, max_atoms, rep_size, n_species, seed)

    K_ref = _ref_kernel_gaussian(x1, x2, q1, q2, n1, n2, sigma)
    K_cpp = lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, sigma)

    assert K_cpp.shape == (nm1, nm2)
    np.testing.assert_allclose(K_cpp, K_ref, rtol=1e-12, atol=1e-13)


def test_kernel_gaussian_shape() -> None:
    """Output shape must be (nm1, nm2)."""
    nm1, nm2 = 3, 7
    max_atoms, rep_size = 4, 5
    x1, x2, q1, q2, n1, n2 = _make_dataset(nm1, nm2, max_atoms, rep_size, 2, 0)
    K = lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, 1.0)
    assert K.shape == (nm1, nm2)


def test_kernel_gaussian_nonneg() -> None:
    """Gaussian kernel values must be non-negative."""
    nm1, nm2 = 5, 5
    max_atoms, rep_size = 5, 10
    x1, x2, q1, q2, n1, n2 = _make_dataset(nm1, nm2, max_atoms, rep_size, 3, 7)
    K = lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, 1.0)
    assert np.all(K >= 0.0)


def test_kernel_gaussian_zero_if_no_shared_species() -> None:
    """If the two sets have completely disjoint species, K must be zero."""
    nm = 3
    max_atoms, rep_size = 4, 6
    rng = np.random.default_rng(123)
    x1 = rng.standard_normal((nm, max_atoms, rep_size))
    x2 = rng.standard_normal((nm, max_atoms, rep_size))
    n1 = np.array([3, 2, 4], dtype=np.int32)
    n2 = np.array([4, 3, 2], dtype=np.int32)
    # set 1: species 1,2   set 2: species 5,6  — no overlap
    q1 = np.ones((nm, max_atoms), dtype=np.int32)
    q2 = np.full((nm, max_atoms), 5, dtype=np.int32)
    K = lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, 1.0)
    np.testing.assert_array_equal(K, 0.0)


def test_kernel_gaussian_sigma_scaling() -> None:
    """Increasing sigma should increase kernel values (smaller exponent penalty)."""
    nm1, nm2 = 3, 3
    max_atoms, rep_size = 4, 6
    x1, x2, q1, q2, n1, n2 = _make_dataset(nm1, nm2, max_atoms, rep_size, 2, 5)
    K_small = lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, sigma=0.5)
    K_large = lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, sigma=10.0)
    # With large sigma, exp term is closer to 1, so K_large >= K_small
    assert np.all(K_large >= K_small - 1e-12)


@pytest.mark.parametrize(("nm1", "nm2"), [(1, 1), (1, 5), (10, 1)])
def test_kernel_gaussian_edge_sizes(nm1: int, nm2: int) -> None:
    """Test with small/edge case molecule counts."""
    max_atoms, rep_size = 3, 5
    x1, x2, q1, q2, n1, n2 = _make_dataset(nm1, nm2, max_atoms, rep_size, 2, 0)
    K = lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, 1.0)
    assert K.shape == (nm1, nm2)
    assert np.all(np.isfinite(K))


def test_kernel_gaussian_consistent_with_symm_diagonal() -> None:
    """kernel_gaussian(X, X) diagonal must match kernel_gaussian_symm diagonal."""
    nm = 6
    max_atoms, rep_size = 5, 8
    x, q, n = _make_symm_dataset(nm, max_atoms, rep_size, 3, 17)
    sigma = 1.5

    K_asym = lk.kernel_gaussian(x, x, q, q, n, n, sigma)
    K_sym = lk.kernel_gaussian_symm(x, q, n, sigma)

    np.testing.assert_allclose(np.diag(K_asym), np.diag(K_sym), rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian_symm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 314])
def test_kernel_gaussian_symm_matches_reference(seed: int) -> None:
    """kernel_gaussian_symm must match the naive symmetric reference."""
    nm = 5
    max_atoms, rep_size, n_species = 6, 8, 3
    sigma = 1.8

    x, q, n = _make_symm_dataset(nm, max_atoms, rep_size, n_species, seed)

    K_ref = _ref_kernel_gaussian_symm(x, q, n, sigma)
    K_cpp = lk.kernel_gaussian_symm(x, q, n, sigma)

    assert K_cpp.shape == (nm, nm)
    np.testing.assert_allclose(K_cpp, K_ref, rtol=1e-12, atol=1e-13)


def test_kernel_gaussian_symm_is_symmetric() -> None:
    """Output matrix must be symmetric."""
    nm = 8
    max_atoms, rep_size = 5, 7
    x, q, n = _make_symm_dataset(nm, max_atoms, rep_size, 3, 42)
    K = lk.kernel_gaussian_symm(x, q, n, 2.0)
    np.testing.assert_allclose(K, K.T, atol=1e-13)


def test_kernel_gaussian_symm_positive_semidefinite() -> None:
    """Symmetric kernel matrix should be positive semidefinite."""
    nm = 10
    max_atoms, rep_size = 4, 6
    x, q, n = _make_symm_dataset(nm, max_atoms, rep_size, 2, 55)
    K = lk.kernel_gaussian_symm(x, q, n, 1.0)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10), f"Min eigenvalue {eigvals.min():.3e} < -1e-10"


def test_kernel_gaussian_symm_matches_asym_self() -> None:
    """kernel_gaussian_symm(X) must equal kernel_gaussian(X, X)."""
    nm = 7
    max_atoms, rep_size = 5, 9
    x, q, n = _make_symm_dataset(nm, max_atoms, rep_size, 3, 99)
    sigma = 3.0

    K_sym = lk.kernel_gaussian_symm(x, q, n, sigma)
    K_asym = lk.kernel_gaussian(x, x, q, q, n, n, sigma)

    np.testing.assert_allclose(K_sym, K_asym, rtol=1e-12, atol=1e-13)


def test_kernel_gaussian_symm_shape() -> None:
    """Output shape must be (nm, nm)."""
    nm = 4
    x, q, n = _make_symm_dataset(nm, 3, 5, 2, 0)
    K = lk.kernel_gaussian_symm(x, q, n, 1.0)
    assert K.shape == (nm, nm)


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian_symm_rfp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 13, 777])
def test_kernel_gaussian_symm_rfp_matches_reference(seed: int) -> None:
    """RFP output must match the reference packed upper triangle."""
    nm = 6
    max_atoms, rep_size, n_species = 5, 7, 3
    sigma = 1.5

    x, q, n = _make_symm_dataset(nm, max_atoms, rep_size, n_species, seed)

    arf_ref = _ref_kernel_gaussian_symm_rfp(x, q, n, sigma)
    arf_cpp = lk.kernel_gaussian_symm_rfp(x, q, n, sigma)

    assert arf_cpp.shape == (nm * (nm + 1) // 2,)
    np.testing.assert_allclose(arf_cpp, arf_ref, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("nm", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
def test_kernel_gaussian_symm_rfp_shape(nm: int) -> None:
    """RFP output length must be nm*(nm+1)/2 for various nm (even and odd)."""
    max_atoms, rep_size = 3, 5
    x, q, n = _make_symm_dataset(nm, max_atoms, rep_size, 2, nm)
    arf = lk.kernel_gaussian_symm_rfp(x, q, n, 1.0)
    assert arf.shape == (nm * (nm + 1) // 2,)


def test_kernel_gaussian_symm_rfp_consistent_with_symm() -> None:
    """Unpacking the RFP array must reproduce kernel_gaussian_symm output."""
    nm = 8
    max_atoms, rep_size = 5, 8
    x, q, n = _make_symm_dataset(nm, max_atoms, rep_size, 3, 2024)
    sigma = 2.0

    K_sym = lk.kernel_gaussian_symm(x, q, n, sigma)
    arf = lk.kernel_gaussian_symm_rfp(x, q, n, sigma)

    # Unpack RFP using the reference logic
    K_unpack = np.zeros((nm, nm), dtype=np.float64)
    k = nm // 2
    stride = (nm + 1) if (nm % 2 == 0) else nm
    for i in range(nm):
        for j in range(i, nm):
            if j >= k:
                idx = (j - k) * stride + i
            else:
                idx = i * stride + (j + k + 1)
            K_unpack[i, j] = arf[idx]
            K_unpack[j, i] = arf[idx]

    np.testing.assert_allclose(K_unpack, K_sym, rtol=1e-12, atol=1e-13)


def test_kernel_gaussian_symm_rfp_nonneg() -> None:
    """All RFP entries are kernel values and must be non-negative."""
    nm = 5
    x, q, n = _make_symm_dataset(nm, 4, 6, 2, 31)
    arf = lk.kernel_gaussian_symm_rfp(x, q, n, 1.0)
    assert np.all(arf >= 0.0)


# ---------------------------------------------------------------------------
# Tests: cross-kernel consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nm", [3, 6, 9])
def test_all_three_kernels_agree(nm: int) -> None:
    """All three scalar kernels must agree for the same input."""
    max_atoms, rep_size, n_species = 5, 8, 3
    sigma = 1.5
    x, q, n = _make_symm_dataset(nm, max_atoms, rep_size, n_species, nm * 7)

    K_asym = lk.kernel_gaussian(x, x, q, q, n, n, sigma)
    K_sym = lk.kernel_gaussian_symm(x, q, n, sigma)
    arf = lk.kernel_gaussian_symm_rfp(x, q, n, sigma)

    # Unpack RFP to full matrix
    k = nm // 2
    stride = (nm + 1) if (nm % 2 == 0) else nm
    K_rfp = np.zeros((nm, nm), dtype=np.float64)
    for i in range(nm):
        for j in range(i, nm):
            if j >= k:
                idx = (j - k) * stride + i
            else:
                idx = i * stride + (j + k + 1)
            K_rfp[i, j] = arf[idx]
            K_rfp[j, i] = arf[idx]

    np.testing.assert_allclose(
        K_sym,
        K_asym,
        rtol=1e-12,
        atol=1e-13,
        err_msg="kernel_gaussian_symm vs kernel_gaussian mismatch",
    )
    np.testing.assert_allclose(
        K_rfp,
        K_sym,
        rtol=1e-12,
        atol=1e-13,
        err_msg="kernel_gaussian_symm_rfp vs kernel_gaussian_symm mismatch",
    )


# ---------------------------------------------------------------------------
# Tests: input validation / error handling
# ---------------------------------------------------------------------------


def test_kernel_gaussian_invalid_sigma_zero() -> None:
    nm = 2
    x1, x2, q1, q2, n1, n2 = _make_dataset(nm, nm, 3, 4, 2, 0)
    with pytest.raises(Exception):
        lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, sigma=0.0)


def test_kernel_gaussian_invalid_sigma_negative() -> None:
    nm = 2
    x1, x2, q1, q2, n1, n2 = _make_dataset(nm, nm, 3, 4, 2, 0)
    with pytest.raises(Exception):
        lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, sigma=-1.0)


def test_kernel_gaussian_symm_invalid_sigma() -> None:
    nm = 2
    x, q, n = _make_symm_dataset(nm, 3, 4, 2, 0)
    with pytest.raises(Exception):
        lk.kernel_gaussian_symm(x, q, n, sigma=0.0)


def test_kernel_gaussian_symm_rfp_invalid_sigma() -> None:
    nm = 2
    x, q, n = _make_symm_dataset(nm, 3, 4, 2, 0)
    with pytest.raises(Exception):
        lk.kernel_gaussian_symm_rfp(x, q, n, sigma=0.0)
