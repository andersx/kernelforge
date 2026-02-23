"""Tests for local (FCHL19) hessian kernels:
- kernel_gaussian_hessian_symm_rfp: RFP output must match kernel_gaussian_hessian_symm
"""

import numpy as np
import pytest
from numpy.typing import NDArray

import kernelforge.local_kernels as fchl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rfp_unpack(arf: NDArray[np.float64], naq: int) -> NDArray[np.float64]:
    """Unpack a 1-D RFP array (TRANSR='N', UPLO='U') into a full symmetric matrix."""
    k = naq // 2
    stride = (naq + 1) if (naq % 2 == 0) else naq
    K = np.zeros((naq, naq), dtype=np.float64)
    for col in range(naq):
        for row in range(col, naq):  # upper triangle: col <= row
            if row >= k:
                idx = (row - k) * stride + col
            else:
                idx = col * stride + (row + k + 1)
            K[col, row] = arf[idx]
            K[row, col] = arf[idx]
    return K


def _make_hessian_dataset(
    nm: int,
    max_atoms: int,
    rep_size: int,
    n_species: int,
    seed: int,
) -> tuple:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(nm, max_atoms, rep_size)).astype(np.float64)
    dx = rng.normal(size=(nm, max_atoms, rep_size, 3 * max_atoms)).astype(np.float64)
    q = rng.integers(0, n_species, size=(nm, max_atoms)).astype(np.int32)
    n = rng.integers(1, max_atoms + 1, size=(nm,)).astype(np.int32)
    return x, dx, q, n


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian_hessian_symm_rfp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_hessian_symm_rfp_matches_hessian_symm(seed: int) -> None:
    """Unpacking the RFP array must reproduce the lower triangle of hessian_symm."""
    nm = 5
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x, dx, q, n = _make_hessian_dataset(nm, max_atoms, rep_size, n_species, seed)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))

    H_symm = fchl.kernel_gaussian_hessian_symm(x, dx, q, n, sigma)
    arf = fchl.kernel_gaussian_hessian_symm_rfp(x, dx, q, n, sigma)

    assert arf.shape == (naq * (naq + 1) // 2,)

    H_unpack = _rfp_unpack(arf, naq)

    # hessian_symm fills only the lower triangle; compare the lower triangle
    lower_mask = np.tril(np.ones((naq, naq), dtype=bool))
    np.testing.assert_allclose(
        H_unpack[lower_mask],
        H_symm[lower_mask],
        rtol=1e-10,
        atol=1e-10,
        err_msg="hessian_symm_rfp unpacked != hessian_symm (lower triangle)",
    )


@pytest.mark.parametrize("naq_check", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
def test_hessian_symm_rfp_output_length(naq_check: int) -> None:
    """Output length must be naq*(naq+1)/2."""
    # Build a dataset where naq == naq_check: nm=1, 1 atom of 1 species, 3*1=3 coords
    # but we need naq to be exactly naq_check: set nm = naq_check // 3, pad if needed.
    # Simplest: nm=1, max_atoms=naq_check, n=[naq_check//3] so naq=naq_check might not work.
    # Instead: nm molecules with 1 atom each, naq = 3*nm = naq_check only if naq_check%3==0.
    # Use nm=naq_check molecules, each with 1 atom (naq=3*naq_check) — that's simpler.
    # Actually, let's just check the shape contract directly:
    nm = 4
    max_atoms = naq_check  # use naq_check as max_atoms; n will be set to 1..max_atoms
    rep_size = 5
    rng = np.random.default_rng(naq_check)
    x = rng.normal(size=(nm, max_atoms, rep_size)).astype(np.float64)
    dx = rng.normal(size=(nm, max_atoms, rep_size, 3 * max_atoms)).astype(np.float64)
    q = rng.integers(0, 2, size=(nm, max_atoms)).astype(np.int32)
    n = rng.integers(1, max_atoms + 1, size=(nm,)).astype(np.int32)

    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))
    arf = fchl.kernel_gaussian_hessian_symm_rfp(x, dx, q, n, sigma=1.0)
    assert arf.shape == (naq * (naq + 1) // 2,)


def test_hessian_symm_rfp_no_cross_molecule_matches() -> None:
    """When labels differ across molecules, off-diagonal (a!=b) blocks must be zero.

    Use nm=2 molecules with disjoint label sets. The only non-zero entries are in
    the diagonal blocks (a==b). We verify this by checking that the kernel output
    is identical to two independent single-molecule calls.
    """
    max_atoms, rep_size = 4, 5
    rng = np.random.default_rng(7)
    nm = 2

    x = rng.normal(size=(nm, max_atoms, rep_size)).astype(np.float64)
    dx = rng.normal(size=(nm, max_atoms, rep_size, 3 * max_atoms)).astype(np.float64)
    n = np.full((nm,), max_atoms, dtype=np.int32)

    # Molecule 0 uses labels 0..max_atoms-1, molecule 1 uses max_atoms..2*max_atoms-1
    # -> no cross-molecule matches
    q = np.zeros((nm, max_atoms), dtype=np.int32)
    for m in range(nm):
        q[m, :] = np.arange(m * max_atoms, (m + 1) * max_atoms, dtype=np.int32)

    sigma = 1.0

    # Full result (nm=2)
    arf_full = fchl.kernel_gaussian_hessian_symm_rfp(x, dx, q, n, sigma)

    # Single-molecule results
    arf_m0 = fchl.kernel_gaussian_hessian_symm_rfp(x[0:1], dx[0:1], q[0:1], n[0:1], sigma)
    arf_m1 = fchl.kernel_gaussian_hessian_symm_rfp(x[1:2], dx[1:2], q[1:2], n[1:2], sigma)

    # Unpack all three
    naq0 = int(3 * n[0])
    naq1_val = int(3 * n[1])
    naq = naq0 + naq1_val

    H_full = _rfp_unpack(arf_full, naq)
    H_m0 = _rfp_unpack(arf_m0, naq0)
    H_m1 = _rfp_unpack(arf_m1, naq1_val)

    # Off-diagonal block (rows naq0..naq, cols 0..naq0) must be zero
    assert np.allclose(H_full[naq0:, :naq0], 0.0), "off-diagonal block should be zero"
    # Diagonal block 0 must match single-molecule result
    np.testing.assert_allclose(H_full[:naq0, :naq0], H_m0, rtol=1e-10, atol=1e-10)
    # Diagonal block 1 must match single-molecule result
    np.testing.assert_allclose(H_full[naq0:, naq0:], H_m1, rtol=1e-10, atol=1e-10)


def test_hessian_symm_rfp_sigma_validation() -> None:
    """Invalid sigma values must raise an exception."""
    nm = 2
    max_atoms, rep_size = 3, 4
    rng = np.random.default_rng(0)
    x = rng.normal(size=(nm, max_atoms, rep_size)).astype(np.float64)
    dx = rng.normal(size=(nm, max_atoms, rep_size, 3 * max_atoms)).astype(np.float64)
    q = rng.integers(0, 2, size=(nm, max_atoms)).astype(np.int32)
    n = np.full((nm,), max_atoms, dtype=np.int32)

    with pytest.raises(Exception):
        fchl.kernel_gaussian_hessian_symm_rfp(x, dx, q, n, sigma=0.0)
    with pytest.raises(Exception):
        fchl.kernel_gaussian_hessian_symm_rfp(x, dx, q, n, sigma=-1.0)


@pytest.mark.parametrize("nm", [2, 3, 5, 7])
def test_hessian_symm_rfp_various_sizes(nm: int) -> None:
    """RFP consistency check for various dataset sizes."""
    max_atoms, rep_size, n_species = 5, 8, 3
    sigma = 0.9

    x, dx, q, n = _make_hessian_dataset(nm, max_atoms, rep_size, n_species, nm * 13)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))

    H_symm = fchl.kernel_gaussian_hessian_symm(x, dx, q, n, sigma)
    arf = fchl.kernel_gaussian_hessian_symm_rfp(x, dx, q, n, sigma)

    H_unpack = _rfp_unpack(arf, naq)
    lower_mask = np.tril(np.ones((naq, naq), dtype=bool))
    np.testing.assert_allclose(
        H_unpack[lower_mask],
        H_symm[lower_mask],
        rtol=1e-10,
        atol=1e-10,
    )


def test_hessian_symm_rfp_single_molecule() -> None:
    """Single-molecule case: naq*(naq+1)/2 entries, diagonal block only."""
    nm = 1
    max_atoms, rep_size, n_species = 3, 5, 2
    sigma = 1.0

    x, dx, q, n = _make_hessian_dataset(nm, max_atoms, rep_size, n_species, 99)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))

    H_symm = fchl.kernel_gaussian_hessian_symm(x, dx, q, n, sigma)
    arf = fchl.kernel_gaussian_hessian_symm_rfp(x, dx, q, n, sigma)

    assert arf.shape == (naq * (naq + 1) // 2,)
    H_unpack = _rfp_unpack(arf, naq)
    lower_mask = np.tril(np.ones((naq, naq), dtype=bool))
    np.testing.assert_allclose(H_unpack[lower_mask], H_symm[lower_mask], rtol=1e-10, atol=1e-10)
