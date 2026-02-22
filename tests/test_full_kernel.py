"""Tests for kernel_gaussian_full, kernel_gaussian_full_symm, and kernel_gaussian_full_symm_rfp.

Verifies each sub-block of the full combined energy+force kernel against the
corresponding standalone sub-kernel functions, using the ethanol MD17 dataset
with inverse-distance representations to ensure realistic molecular data.

Full kernel layout (N1*(1+D1)) x (N2*(1+D2)):
  K_full[0:N1,  0:N2]  = scalar kernel   (energy-energy)
  K_full[0:N1,  N2:]   = jacobian_t       (energy-force)
  K_full[N1:,   0:N2]  = jacobian          (force-energy)
  K_full[N1:,   N2:]   = hessian           (force-force)
"""

import numpy as np
import pytest

from kernelforge import invdist_repr
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.global_kernels import (
    kernel_gaussian,
    kernel_gaussian_full,
    kernel_gaussian_full_symm,
    kernel_gaussian_full_symm_rfp,
    kernel_gaussian_hessian,
    kernel_gaussian_hessian_symm,
    kernel_gaussian_hessian_symm_rfp,
    kernel_gaussian_jacobian,
    kernel_gaussian_jacobian_t,
)

# ---------------------------------------------------------------------------
# Ethanol data fixture: load once per module, slice as needed
# ---------------------------------------------------------------------------


def _load_ethanol(n: int):
    """Load n ethanol structures and return X (n, M), dX (n, M, D)."""
    data = load_ethanol_raw_data()
    R = data["R"][:n]

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list, dtype=np.float64)  # (n, M)  M=36
    dX = np.array(dX_list, dtype=np.float64)  # (n, M, D)  D=27
    return X, dX


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian_full (asymmetric)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N1,N2", [(5, 7), (10, 10), (15, 8)])
def test_full_asymm_scalar_block(N1, N2):
    """Scalar (energy-energy) sub-block matches standalone kernel_gaussian."""
    sigma = 2.0
    alpha = -1.0 / (2.0 * sigma**2)

    X1, dX1 = _load_ethanol(N1)
    X2, dX2 = _load_ethanol(N2)

    K_full = kernel_gaussian_full(X1, dX1, X2, dX2, sigma)
    K_scalar_ref = kernel_gaussian(X1, X2, alpha)

    np.testing.assert_allclose(
        K_full[:N1, :N2],
        K_scalar_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Scalar block mismatch for N1={N1}, N2={N2}",
    )


@pytest.mark.parametrize("N1,N2", [(5, 7), (10, 10), (15, 8)])
def test_full_asymm_jacobian_block(N1, N2):
    """Jacobian (force-energy) sub-block matches standalone kernel_gaussian_jacobian."""
    sigma = 2.0
    X1, dX1 = _load_ethanol(N1)
    X2, dX2 = _load_ethanol(N2)
    D1 = dX1.shape[2]

    K_full = kernel_gaussian_full(X1, dX1, X2, dX2, sigma)
    K_jac_ref = kernel_gaussian_jacobian(X1, dX1, X2, sigma)

    np.testing.assert_allclose(
        K_full[N1:, :N2],
        K_jac_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Jacobian block mismatch for N1={N1}, N2={N2}",
    )


@pytest.mark.parametrize("N1,N2", [(5, 7), (10, 10), (15, 8)])
def test_full_asymm_jacobian_t_block(N1, N2):
    """Jacobian_t (energy-force) sub-block matches standalone kernel_gaussian_jacobian_t."""
    sigma = 2.0
    X1, dX1 = _load_ethanol(N1)
    X2, dX2 = _load_ethanol(N2)
    D2 = dX2.shape[2]

    K_full = kernel_gaussian_full(X1, dX1, X2, dX2, sigma)
    K_jt_ref = kernel_gaussian_jacobian_t(X1, X2, dX2, sigma)

    np.testing.assert_allclose(
        K_full[:N1, N2:],
        K_jt_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Jacobian_t block mismatch for N1={N1}, N2={N2}",
    )


@pytest.mark.parametrize("N1,N2", [(5, 7), (10, 10), (15, 8)])
def test_full_asymm_hessian_block(N1, N2):
    """Hessian (force-force) sub-block matches standalone kernel_gaussian_hessian."""
    sigma = 2.0
    X1, dX1 = _load_ethanol(N1)
    X2, dX2 = _load_ethanol(N2)

    K_full = kernel_gaussian_full(X1, dX1, X2, dX2, sigma)
    H_ref = kernel_gaussian_hessian(X1, dX1, X2, dX2, sigma)

    np.testing.assert_allclose(
        K_full[N1:, N2:],
        H_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Hessian block mismatch for N1={N1}, N2={N2}",
    )


@pytest.mark.parametrize("N1,N2,tile_B", [(10, 10, 1), (10, 10, 3), (10, 10, 64)])
def test_full_asymm_tile_consistency(N1, N2, tile_B):
    """Different tile_B values give identical results."""
    sigma = 1.5
    X1, dX1 = _load_ethanol(N1)
    X2, dX2 = _load_ethanol(N2)

    K_ref = kernel_gaussian_full(X1, dX1, X2, dX2, sigma, tile_B=None)
    K_tiled = kernel_gaussian_full(X1, dX1, X2, dX2, sigma, tile_B=tile_B)

    np.testing.assert_allclose(
        K_tiled,
        K_ref,
        rtol=1e-14,
        atol=1e-14,
        err_msg=f"tile_B={tile_B} gives different result",
    )


def test_full_asymm_output_shape():
    """Output shape is (N1*(1+D1), N2*(1+D2))."""
    N1, N2 = 6, 8
    sigma = 1.0
    X1, dX1 = _load_ethanol(N1)
    X2, dX2 = _load_ethanol(N2)
    D1, D2 = dX1.shape[2], dX2.shape[2]

    K_full = kernel_gaussian_full(X1, dX1, X2, dX2, sigma)
    assert K_full.shape == (N1 * (1 + D1), N2 * (1 + D2)), (
        f"Expected shape {(N1 * (1 + D1), N2 * (1 + D2))}, got {K_full.shape}"
    )


def test_full_asymm_jac_jac_t_transpose_relation():
    """Jacobian and jacobian_t blocks are transposes when X1==X2, dX1==dX2."""
    N = 8
    sigma = 2.0
    X, dX = _load_ethanol(N)

    K_full = kernel_gaussian_full(X, dX, X, dX, sigma)
    K_jac = K_full[N:, :N]  # (N*D, N)
    K_jac_t = K_full[:N, N:]  # (N, N*D)

    np.testing.assert_allclose(
        K_jac_t,
        K_jac.T,
        rtol=1e-12,
        atol=1e-12,
        err_msg="Jacobian_t is not the transpose of Jacobian when X1==X2",
    )


def test_full_asymm_error_invalid_sigma():
    """Invalid sigma raises an error."""
    X, dX = _load_ethanol(5)
    with pytest.raises(Exception):
        kernel_gaussian_full(X, dX, X, dX, -1.0)
    with pytest.raises(Exception):
        kernel_gaussian_full(X, dX, X, dX, 0.0)


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian_full_symm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [5, 10, 15])
def test_full_symm_scalar_block(N):
    """Scalar block of symmetric full kernel matches standalone kernel_gaussian."""
    sigma = 2.0
    alpha = -1.0 / (2.0 * sigma**2)
    X, dX = _load_ethanol(N)

    K_full = kernel_gaussian_full_symm(X, dX, sigma)
    K_scalar_ref = kernel_gaussian(X, X, alpha)

    # Symm fills both upper and lower of scalar block
    np.testing.assert_allclose(
        K_full[:N, :N],
        K_scalar_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Symmetric scalar block mismatch N={N}",
    )


@pytest.mark.parametrize("N", [5, 10, 15])
def test_full_symm_scalar_diagonal_is_one(N):
    """Scalar diagonal = 1.0 (self-kernel = exp(0))."""
    sigma = 2.0
    X, dX = _load_ethanol(N)
    K_full = kernel_gaussian_full_symm(X, dX, sigma)
    np.testing.assert_allclose(
        np.diag(K_full[:N, :N]),
        np.ones(N),
        rtol=1e-14,
        atol=1e-14,
        err_msg="Scalar diagonal != 1.0",
    )


@pytest.mark.parametrize("N", [5, 10, 15])
def test_full_symm_jacobian_block(N):
    """Jacobian block of symmetric full kernel matches standalone kernel_gaussian_jacobian."""
    sigma = 2.0
    X, dX = _load_ethanol(N)
    D = dX.shape[2]

    K_full = kernel_gaussian_full_symm(X, dX, sigma)
    K_jac_ref = kernel_gaussian_jacobian(X, dX, X, sigma)

    np.testing.assert_allclose(
        K_full[N:, :N],
        K_jac_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Symmetric jacobian block mismatch N={N}",
    )


@pytest.mark.parametrize("N", [5, 10, 15])
def test_full_symm_jacobian_t_block(N):
    """Jacobian_t block of symmetric full kernel matches standalone kernel_gaussian_jacobian_t."""
    sigma = 2.0
    X, dX = _load_ethanol(N)

    K_full = kernel_gaussian_full_symm(X, dX, sigma)
    K_jt_ref = kernel_gaussian_jacobian_t(X, X, dX, sigma)

    np.testing.assert_allclose(
        K_full[:N, N:],
        K_jt_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Symmetric jacobian_t block mismatch N={N}",
    )


@pytest.mark.parametrize("N", [5, 10, 15])
def test_full_symm_hessian_block_lower_triangle(N):
    """Lower triangle of hessian block in symmetric full kernel matches hessian_symm."""
    sigma = 2.0
    X, dX = _load_ethanol(N)
    D = dX.shape[2]

    K_full = kernel_gaussian_full_symm(X, dX, sigma)
    H_ref = kernel_gaussian_hessian_symm(X, dX, sigma)

    # hessian_symm fills lower triangle; compare only lower tri
    H_full_block = K_full[N:, N:]
    rows, cols = np.tril_indices(N * D)
    np.testing.assert_allclose(
        H_full_block[rows, cols],
        H_ref[rows, cols],
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Symmetric hessian block lower triangle mismatch N={N}",
    )


@pytest.mark.parametrize("N", [5, 10])
def test_full_symm_jac_jac_t_are_transposes(N):
    """In symmetric full kernel, jac_t == jac^T."""
    sigma = 1.5
    X, dX = _load_ethanol(N)

    K_full = kernel_gaussian_full_symm(X, dX, sigma)
    K_jac = K_full[N:, :N]
    K_jac_t = K_full[:N, N:]

    np.testing.assert_allclose(
        K_jac_t,
        K_jac.T,
        rtol=1e-12,
        atol=1e-12,
        err_msg="Symmetric kernel: jac_t != jac^T",
    )


def test_full_symm_output_shape():
    """Output shape is (N*(1+D), N*(1+D))."""
    N = 6
    X, dX = _load_ethanol(N)
    D = dX.shape[2]
    K_full = kernel_gaussian_full_symm(X, dX, 2.0)
    BIG = N * (1 + D)
    assert K_full.shape == (BIG, BIG), f"Expected ({BIG},{BIG}), got {K_full.shape}"


def test_full_symm_matches_asymm():
    """Symmetric full kernel lower triangle matches asymmetric full kernel when X1==X2.

    The symm kernel fills only the lower triangle (row >= col), matching the
    kernel_gaussian_hessian_symm convention. The asymmetric kernel fills all entries.
    We compare only the lower triangle plus the scalar/jac/jac_t blocks which are
    fully filled by the symm kernel.
    """
    N = 8
    sigma = 2.0
    X, dX = _load_ethanol(N)
    D = dX.shape[2]
    BIG = N * (1 + D)

    K_asymm = kernel_gaussian_full(X, dX, X, dX, sigma)
    K_symm = kernel_gaussian_full_symm(X, dX, sigma)

    # Scalar block: fully filled by symm
    np.testing.assert_allclose(
        K_symm[:N, :N],
        K_asymm[:N, :N],
        rtol=1e-12,
        atol=1e-12,
        err_msg="Scalar block mismatch",
    )
    # Jacobian block: fully filled by symm
    np.testing.assert_allclose(
        K_symm[N:, :N],
        K_asymm[N:, :N],
        rtol=1e-12,
        atol=1e-12,
        err_msg="Jacobian block mismatch",
    )
    # Jacobian_t block: fully filled by symm
    np.testing.assert_allclose(
        K_symm[:N, N:],
        K_asymm[:N, N:],
        rtol=1e-12,
        atol=1e-12,
        err_msg="Jacobian_t block mismatch",
    )
    # Hessian block: only lower triangle is guaranteed to be filled
    rows, cols = np.tril_indices(N * D)
    np.testing.assert_allclose(
        K_symm[N + rows, N + cols],
        K_asymm[N + rows, N + cols],
        rtol=1e-12,
        atol=1e-12,
        err_msg="Hessian lower triangle mismatch",
    )


# ---------------------------------------------------------------------------
# Tests: kernel_gaussian_full_symm_rfp
# ---------------------------------------------------------------------------


def _rfp_to_full(rfp: np.ndarray, BIG: int) -> np.ndarray:
    """Unpack RFP upper-triangle packed array to full symmetric matrix."""
    # Rebuild using the same rfp_index_upper_N logic:
    # For each (i<=j), rfp[idx] = K[i,j] = K[j,i].
    # We compute the index map using Python to verify against C++ packed array.
    # The RFP convention used is TRANSR='N', UPLO='U' from LAPACK.
    # See: rfp_index_upper_N in global_kernels.cpp
    k = BIG // 2
    stride = (BIG + 1) if (BIG % 2 == 0) else BIG

    def rfp_idx(i, j):
        # i <= j (upper triangle)
        if j >= k:
            return (j - k) * stride + i
        else:
            return i * stride + j + k + 1

    full = np.zeros((BIG, BIG))
    for i in range(BIG):
        for j in range(i, BIG):
            val = rfp[rfp_idx(i, j)]
            full[i, j] = val
            full[j, i] = val
    return full


@pytest.mark.parametrize("N", [5, 8, 10])
def test_full_symm_rfp_matches_full_symm(N):
    """RFP full kernel unpacks to match symmetric full kernel (lower triangle).

    kernel_gaussian_full_symm fills the lower triangle (row >= col).
    kernel_gaussian_full_symm_rfp stores the upper triangle as RFP.
    The unpacked full matrix (symmetric) must match the lower triangle of K_symm.
    """
    sigma = 2.0
    X, dX = _load_ethanol(N)
    D = dX.shape[2]
    BIG = N * (1 + D)

    K_symm = kernel_gaussian_full_symm(X, dX, sigma)
    K_rfp = kernel_gaussian_full_symm_rfp(X, dX, sigma)

    assert len(K_rfp) == BIG * (BIG + 1) // 2, "RFP array has wrong length"

    K_unpacked = _rfp_to_full(K_rfp, BIG)

    # Compare lower triangle of unpacked with lower triangle of K_symm
    rows, cols = np.tril_indices(BIG)
    np.testing.assert_allclose(
        K_unpacked[rows, cols],
        K_symm[rows, cols],
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"RFP unpacked lower triangle disagrees with symm full kernel, N={N}",
    )


@pytest.mark.parametrize("N", [5, 10])
def test_full_symm_rfp_scalar_block(N):
    """Scalar diagonal in RFP full kernel == 1.0."""
    sigma = 2.0
    X, dX = _load_ethanol(N)
    D = dX.shape[2]
    BIG = N * (1 + D)

    K_rfp = kernel_gaussian_full_symm_rfp(X, dX, sigma)
    K_full = _rfp_to_full(K_rfp, BIG)

    np.testing.assert_allclose(
        np.diag(K_full[:N, :N]),
        np.ones(N),
        rtol=1e-14,
        atol=1e-14,
        err_msg="RFP: scalar diagonal != 1.0",
    )


@pytest.mark.parametrize("N", [5, 10])
def test_full_symm_rfp_hessian_block(N):
    """Hessian block in unpacked RFP full kernel matches hessian_symm_rfp."""
    sigma = 2.0
    X, dX = _load_ethanol(N)
    D = dX.shape[2]
    BIG = N * (1 + D)

    K_rfp = kernel_gaussian_full_symm_rfp(X, dX, sigma)
    K_full = _rfp_to_full(K_rfp, BIG)

    # Compare with hessian_symm_rfp unpacked
    H_rfp = kernel_gaussian_hessian_symm_rfp(X, dX, sigma)
    hess_BIG = N * D
    H_full_ref = np.zeros((hess_BIG, hess_BIG))
    hk = hess_BIG // 2
    hstride = (hess_BIG + 1) if (hess_BIG % 2 == 0) else hess_BIG

    def h_rfp_idx(i, j):
        if j >= hk:
            return (j - hk) * hstride + i
        else:
            return i * hstride + j + hk + 1

    for i in range(hess_BIG):
        for j in range(i, hess_BIG):
            val = H_rfp[h_rfp_idx(i, j)]
            H_full_ref[i, j] = val
            H_full_ref[j, i] = val

    np.testing.assert_allclose(
        K_full[N:, N:],
        H_full_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"RFP full kernel hessian block mismatch N={N}",
    )


def test_full_symm_rfp_output_length():
    """RFP array length is BIG*(BIG+1)//2."""
    N = 6
    X, dX = _load_ethanol(N)
    D = dX.shape[2]
    BIG = N * (1 + D)

    K_rfp = kernel_gaussian_full_symm_rfp(X, dX, 2.0)
    assert len(K_rfp) == BIG * (BIG + 1) // 2


@pytest.mark.parametrize("N,tile_B", [(8, 1), (8, 3), (8, 64)])
def test_full_symm_rfp_tile_consistency(N, tile_B):
    """RFP full kernel is identical for different tile_B values."""
    sigma = 1.5
    X, dX = _load_ethanol(N)

    K_ref = kernel_gaussian_full_symm_rfp(X, dX, sigma, tile_B=None)
    K_tiled = kernel_gaussian_full_symm_rfp(X, dX, sigma, tile_B=tile_B)

    np.testing.assert_allclose(
        K_tiled,
        K_ref,
        rtol=1e-14,
        atol=1e-14,
        err_msg=f"RFP tile_B={tile_B} gives different result",
    )
