"""Tests for local (FCHL19) full combined kernel:
- kernel_gaussian_full: sub-blocks must match the individual scalar/jacobian/hessian kernels
"""

import numpy as np
import pytest

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
    # The fe block uses -expdiag (dK/dR1 with opposite sign to the standalone jacobian_t).
    # kernel_gaussian_jacobian_t returns +dK/dR1, so the fe block equals -K_jact.
    K_jact = fchl.kernel_gaussian_jacobian_t(x1, x2, dx1, q1, q2, n1, n2, sigma)

    assert K_full[nm1:, :nm2].shape == (naq1, nm2), (
        f"jac block shape wrong: {K_full[nm1:, :nm2].shape}"
    )
    np.testing.assert_allclose(
        K_full[nm1:, :nm2],
        -K_jact,
        rtol=1e-10,
        atol=1e-10,
        err_msg="jacobian block mismatch",
    )


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_full_jacobian_t_block(seed: int) -> None:
    """K_full[0:nm1, nm2:] must equal K_full[nm2:, 0:nm1].T (jact is transpose of jac block)."""
    nm1, nm2 = 3, 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x1, dx1, q1, n1 = _make_dataset(nm1, max_atoms, rep_size, n_species, seed)
    x2, dx2, q2, n2 = _make_dataset(nm2, max_atoms, rep_size, n_species, seed + 1)

    naq1 = int(3 * np.sum(np.clip(n1, 0, max_atoms)))
    naq2 = int(3 * np.sum(np.clip(n2, 0, max_atoms)))

    K_full = fchl.kernel_gaussian_full(x1, x2, dx1, dx2, q1, q2, n1, n2, sigma)
    # The jac block K_full[nm1:, :nm2] has shape (naq1, nm2).
    # The jact block K_full[:nm1, nm2:] has shape (nm1, naq2).
    # They are transposes of each other when x1/x2 are swapped:
    #   K_full(x1,x2)[0:nm1, nm2:] == K_full(x2,x1)[nm2:, 0:nm1].T
    K_full_swapped = fchl.kernel_gaussian_full(x2, x1, dx2, dx1, q2, q1, n2, n1, sigma)

    assert K_full[:nm1, nm2:].shape == (nm1, naq2), (
        f"jact block shape wrong: {K_full[:nm1, nm2:].shape}"
    )
    np.testing.assert_allclose(
        K_full[:nm1, nm2:],
        K_full_swapped[nm2:, :nm1].T,
        rtol=1e-10,
        atol=1e-10,
        err_msg="jact block is not the transpose of the swapped jac block",
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
    """kernel_gaussian_full_symm(x,...) must equal kernel_gaussian_full(x,x,...) element-wise."""
    nm = 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x, dx, q, n = _make_dataset(nm, max_atoms, rep_size, n_species, seed)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))
    BIG = nm + naq

    K_symm = fchl.kernel_gaussian_full_symm(x, dx, q, n, sigma)
    K_asymm = fchl.kernel_gaussian_full(x, x, dx, dx, q, q, n, n, sigma)

    assert K_symm.shape == (BIG, BIG), f"symm shape wrong: {K_symm.shape}"
    assert K_asymm.shape == (BIG, BIG), f"asymm shape wrong: {K_asymm.shape}"

    np.testing.assert_allclose(
        K_symm,
        K_asymm,
        rtol=1e-10,
        atol=1e-10,
        err_msg="symm full kernel != asymm full kernel(x,x,...)",
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
def test_full_symm_rfp_matches_asymm(seed: int) -> None:
    """Unpacked RFP full kernel must equal kernel_gaussian_full(x,x,...) element-wise."""
    nm = 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x, dx, q, n = _make_dataset(nm, max_atoms, rep_size, n_species, seed)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))
    BIG = nm + naq

    arf = fchl.kernel_gaussian_full_symm_rfp(x, dx, q, n, sigma)
    K_asymm = fchl.kernel_gaussian_full(x, x, dx, dx, q, q, n, n, sigma)

    assert arf.shape == (BIG * (BIG + 1) // 2,), f"rfp length wrong: {arf.shape}"
    assert K_asymm.shape == (BIG, BIG), f"asymm shape wrong: {K_asymm.shape}"

    K_unpack = _rfp_full_unpack(arf, BIG)
    np.testing.assert_allclose(
        K_unpack,
        K_asymm,
        rtol=1e-10,
        atol=1e-10,
        err_msg="rfp unpacked != asymm full kernel(x,x,...)",
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


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian_full block consistency (same-set, all four blocks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_full_all_blocks_same_set(seed: int) -> None:
    """kernel_gaussian_full(x, x, ...) blocks must match scalar, jacobian, and hessian kernels.

    Computes kernel_gaussian_full with identical row/column sets, then:
      - ee block  == kernel_gaussian scalar
      - ff block  == kernel_gaussian_hessian
      - fe block  == kernel_gaussian_jacobian_t   (one jacobian call)
      - ef block  == kernel_gaussian_jacobian_t.T (transpose of the same jacobian)
    """
    nm = 4
    max_atoms, rep_size, n_species = 4, 7, 3
    sigma = 1.5

    x, dx, q, n = _make_dataset(nm, max_atoms, rep_size, n_species, seed)
    naq = int(3 * np.sum(np.clip(n, 0, max_atoms)))

    K_full = fchl.kernel_gaussian_full(x, x, dx, dx, q, q, n, n, sigma)

    K_scalar = fchl.kernel_gaussian(x, x, q, q, n, n, sigma)
    K_jact = fchl.kernel_gaussian_jacobian_t(x, x, dx, q, q, n, n, sigma)
    K_hess = fchl.kernel_gaussian_hessian(x, x, dx, dx, q, q, n, n, sigma)

    # ee block
    np.testing.assert_allclose(
        K_full[:nm, :nm], K_scalar, rtol=1e-10, atol=1e-10, err_msg="ee (scalar) block mismatch"
    )

    # fe block: kernel_gaussian_full uses -expdiag for jac, kernel_gaussian_jacobian_t
    # uses +expdiag, so the fe block equals -K_jact.
    np.testing.assert_allclose(
        K_full[nm:, :nm],
        -K_jact,
        rtol=1e-10,
        atol=1e-10,
        err_msg="fe (jacobian) block mismatch",
    )

    # ef block: symmetric to fe, so ef equals -K_jact.T
    np.testing.assert_allclose(
        K_full[:nm, nm:],
        -K_jact.T,
        rtol=1e-10,
        atol=1e-10,
        err_msg="ef (jacobian.T) block mismatch",
    )

    # ff block
    np.testing.assert_allclose(
        K_full[nm:, nm:], K_hess, rtol=1e-10, atol=1e-10, err_msg="ff (hessian) block mismatch"
    )
