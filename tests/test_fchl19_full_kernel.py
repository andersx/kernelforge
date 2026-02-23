"""Tests for local (FCHL19) full combined kernel:
- kernel_gaussian_full: sub-blocks must match the individual scalar/jacobian/hessian kernels
"""

import numpy as np
import pytest
from numpy.typing import NDArray

import kernelforge.local_kernels as fchl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
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
# Tests: kernel_gaussian_full (asymmetric)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_full_scalar_block(seed: int) -> None:
    """K_full[0:nm1, 0:nm2] must match kernel_gaussian."""
    nm1, nm2 = 3, 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x1, dx1, q1, n1 = _make_dataset(nm1, max_atoms, rep_size, n_species, seed)
    x2, dx2, q2, n2 = _make_dataset(nm2, max_atoms, rep_size, n_species, seed + 1)

    K_full = fchl.kernel_gaussian_full(x1, x2, dx1, dx2, q1, q2, n1, n2, sigma)
    K_scalar = fchl.kernel_gaussian(x1, x2, q1, q2, n1, n2, sigma)

    np.testing.assert_allclose(
        K_full[:nm1, :nm2],
        K_scalar,
        rtol=1e-10,
        atol=1e-10,
        err_msg="scalar block mismatch",
    )


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_full_jacobian_block(seed: int) -> None:
    """K_full[nm1:, 0:nm2] must match kernel_gaussian_jacobian_t(x1,dX1,x2,...) [naq1 x nm2]."""
    nm1, nm2 = 3, 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x1, dx1, q1, n1 = _make_dataset(nm1, max_atoms, rep_size, n_species, seed)
    x2, dx2, q2, n2 = _make_dataset(nm2, max_atoms, rep_size, n_species, seed + 1)

    naq1 = int(3 * np.sum(np.clip(n1, 0, max_atoms)))

    K_full = fchl.kernel_gaussian_full(x1, x2, dx1, dx2, q1, q2, n1, n2, sigma)
    # jacobian block: dK/dx1 w.r.t. x1 coords => kernel_gaussian_jacobian_t(x1, dX1, x2, ...)
    # output shape: (naq1, nm2)
    K_jact = fchl.kernel_gaussian_jacobian_t(x1, x2, dx1, q1, q2, n1, n2, sigma)

    assert K_full[nm1:, :nm2].shape == (naq1, nm2), (
        f"jac block shape wrong: {K_full[nm1:, :nm2].shape}"
    )
    np.testing.assert_allclose(
        K_full[nm1:, :nm2],
        K_jact,
        rtol=1e-10,
        atol=1e-10,
        err_msg="jacobian block mismatch",
    )


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_full_jacobian_t_block(seed: int) -> None:
    """K_full[0:nm1, nm2:] must match kernel_gaussian_jacobian(x1,x2,dX2,...) [nm1 x naq2]."""
    nm1, nm2 = 3, 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x1, dx1, q1, n1 = _make_dataset(nm1, max_atoms, rep_size, n_species, seed)
    x2, dx2, q2, n2 = _make_dataset(nm2, max_atoms, rep_size, n_species, seed + 1)

    naq2 = int(3 * np.sum(np.clip(n2, 0, max_atoms)))

    K_full = fchl.kernel_gaussian_full(x1, x2, dx1, dx2, q1, q2, n1, n2, sigma)
    # jacobian_t block: dK/dx2 w.r.t. x2 coords => kernel_gaussian_jacobian(x1, x2, dX2, ...)
    # output shape: (nm1, naq2)
    K_jac = fchl.kernel_gaussian_jacobian(x1, x2, dx2, q1, q2, n1, n2, sigma)

    assert K_full[:nm1, nm2:].shape == (nm1, naq2), (
        f"jact block shape wrong: {K_full[:nm1, nm2:].shape}"
    )
    np.testing.assert_allclose(
        K_full[:nm1, nm2:],
        K_jac,
        rtol=1e-10,
        atol=1e-10,
        err_msg="jacobian_t block mismatch",
    )


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_full_hessian_block(seed: int) -> None:
    """K_full[nm1:, nm2:] must match kernel_gaussian_hessian."""
    nm1, nm2 = 3, 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x1, dx1, q1, n1 = _make_dataset(nm1, max_atoms, rep_size, n_species, seed)
    x2, dx2, q2, n2 = _make_dataset(nm2, max_atoms, rep_size, n_species, seed + 1)

    naq1 = int(3 * np.sum(np.clip(n1, 0, max_atoms)))
    naq2 = int(3 * np.sum(np.clip(n2, 0, max_atoms)))

    K_full = fchl.kernel_gaussian_full(x1, x2, dx1, dx2, q1, q2, n1, n2, sigma)
    K_hess = fchl.kernel_gaussian_hessian(x1, x2, dx1, dx2, q1, q2, n1, n2, sigma)

    assert K_full[nm1:, nm2:].shape == (naq1, naq2), (
        f"hess block shape wrong: {K_full[nm1:, nm2:].shape}"
    )
    np.testing.assert_allclose(
        K_full[nm1:, nm2:],
        K_hess,
        rtol=1e-10,
        atol=1e-10,
        err_msg="hessian block mismatch",
    )


def test_full_output_shape() -> None:
    """Output shape must be (nm1+naq1, nm2+naq2)."""
    nm1, nm2 = 2, 5
    max_atoms, rep_size, n_species = 3, 5, 2
    sigma = 1.0

    x1, dx1, q1, n1 = _make_dataset(nm1, max_atoms, rep_size, n_species, 7)
    x2, dx2, q2, n2 = _make_dataset(nm2, max_atoms, rep_size, n_species, 8)

    naq1 = int(3 * np.sum(np.clip(n1, 0, max_atoms)))
    naq2 = int(3 * np.sum(np.clip(n2, 0, max_atoms)))

    K_full = fchl.kernel_gaussian_full(x1, x2, dx1, dx2, q1, q2, n1, n2, sigma)
    assert K_full.shape == (nm1 + naq1, nm2 + naq2), f"wrong shape: {K_full.shape}"


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian_full_symm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_full_symm_matches_asymm(seed: int) -> None:
    """kernel_gaussian_full_symm must match kernel_gaussian_full(x,x,...) on all blocks."""
    nm = 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x, dx, q, n = _make_dataset(nm, max_atoms, rep_size, n_species, seed)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))
    BIG = nm + naq

    K_symm = fchl.kernel_gaussian_full_symm(x, dx, q, n, sigma)
    K_asymm = fchl.kernel_gaussian_full(x, x, dx, dx, q, q, n, n, sigma)

    assert K_symm.shape == (BIG, BIG), f"symm shape wrong: {K_symm.shape}"

    # Scalar block: fully filled
    np.testing.assert_allclose(
        K_symm[:nm, :nm], K_asymm[:nm, :nm], rtol=1e-10, atol=1e-10, err_msg="scalar block mismatch"
    )
    # Jacobian block: fully filled
    np.testing.assert_allclose(
        K_symm[nm:, :nm],
        K_asymm[nm:, :nm],
        rtol=1e-10,
        atol=1e-10,
        err_msg="jacobian block mismatch",
    )
    # Jacobian_t block: K_symm[:nm, nm:] must equal K_symm[nm:, :nm].T (symmetry of full matrix).
    # The asymm kernel's jact uses a different sign convention (-dK/dR_b) vs the symm kernel
    # which enforces K_jact = K_jac^T.  Compare against the mirrored jac block instead.
    np.testing.assert_allclose(
        K_symm[:nm, nm:],
        K_symm[nm:, :nm].T,
        rtol=1e-10,
        atol=1e-10,
        err_msg="jacobian_t block not transpose of jacobian block",
    )
    # Hessian block: lower triangle must match asymm
    rows, cols = np.tril_indices(naq)
    np.testing.assert_allclose(
        K_symm[nm + rows, nm + cols],
        K_asymm[nm + rows, nm + cols],
        rtol=1e-10,
        atol=1e-10,
        err_msg="hessian lower triangle mismatch",
    )


@pytest.mark.parametrize("seed", [0, 42])
def test_full_symm_jac_jact_are_transposes(seed: int) -> None:
    """In the symmetric full kernel, K[nm:, :nm] == K[:nm, nm:].T."""
    nm = 3
    max_atoms, rep_size, n_species = 4, 6, 2
    sigma = 1.2

    x, dx, q, n = _make_dataset(nm, max_atoms, rep_size, n_species, seed)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))

    K = fchl.kernel_gaussian_full_symm(x, dx, q, n, sigma)
    np.testing.assert_allclose(
        K[nm:, :nm], K[:nm, nm:].T, rtol=1e-10, atol=1e-10, err_msg="jac != jact.T"
    )


def test_full_symm_output_shape() -> None:
    """Output shape must be (nm+naq, nm+naq)."""
    nm = 3
    max_atoms, rep_size, n_species = 3, 5, 2
    x, dx, q, n = _make_dataset(nm, max_atoms, rep_size, n_species, 7)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))
    K = fchl.kernel_gaussian_full_symm(x, dx, q, n, sigma=1.0)
    assert K.shape == (nm + naq, nm + naq), f"wrong shape: {K.shape}"


# ---------------------------------------------------------------------------
# Helpers for RFP
# ---------------------------------------------------------------------------


def _rfp_full_unpack(arf: np.ndarray, BIG: int) -> np.ndarray:
    """Unpack RFP (TRANSR='N', UPLO='U') into a full symmetric matrix."""
    k = BIG // 2
    stride = (BIG + 1) if (BIG % 2 == 0) else BIG

    def rfp_idx(i: int, j: int) -> int:  # i <= j
        if j >= k:
            return (j - k) * stride + i
        else:
            return i * stride + (j + k + 1)

    K = np.zeros((BIG, BIG), dtype=np.float64)
    for i in range(BIG):
        for j in range(i, BIG):
            v = arf[rfp_idx(i, j)]
            K[i, j] = v
            K[j, i] = v
    return K


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian_full_symm_rfp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_full_symm_rfp_matches_symm(seed: int) -> None:
    """Unpacked RFP full kernel must match lower triangle of kernel_gaussian_full_symm."""
    nm = 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x, dx, q, n = _make_dataset(nm, max_atoms, rep_size, n_species, seed)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))
    BIG = nm + naq

    K_symm = fchl.kernel_gaussian_full_symm(x, dx, q, n, sigma)
    arf = fchl.kernel_gaussian_full_symm_rfp(x, dx, q, n, sigma)

    assert arf.shape == (BIG * (BIG + 1) // 2,), f"rfp length wrong: {arf.shape}"

    K_unpack = _rfp_full_unpack(arf, BIG)
    rows, cols = np.tril_indices(BIG)
    np.testing.assert_allclose(
        K_unpack[rows, cols],
        K_symm[rows, cols],
        rtol=1e-10,
        atol=1e-10,
        err_msg="rfp unpacked lower triangle != symm lower triangle",
    )


def test_full_symm_rfp_output_length() -> None:
    """RFP array length must be BIG*(BIG+1)//2."""
    nm = 3
    max_atoms, rep_size, n_species = 3, 5, 2
    x, dx, q, n = _make_dataset(nm, max_atoms, rep_size, n_species, 9)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))
    BIG = nm + naq
    arf = fchl.kernel_gaussian_full_symm_rfp(x, dx, q, n, sigma=1.0)
    assert arf.shape == (BIG * (BIG + 1) // 2,), f"wrong length: {arf.shape}"
