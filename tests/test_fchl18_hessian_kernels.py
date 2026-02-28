"""Tests for kernel_gaussian_hessian_symm.

kernel_gaussian_hessian_symm exploits H(A,B) = H(B,A)^T symmetry:
only the lower triangle of molecule blocks is computed, then mirrored.

Verified by:
  1. Shape checks.
  2. Comparing to kernel_gaussian_hessian (asymmetric, which is already tested).
  3. Checking exact symmetry H[i,j] == H[j,i].
"""

import numpy as np
import pytest

import kernelforge.fchl18_kernel as kernel_mod

# ---------------------------------------------------------------------------
# Molecule definitions
# ---------------------------------------------------------------------------
WATER_COORDS = np.array(
    [[0.000, 0.000, 0.119], [0.000, 0.757, -0.477], [0.000, -0.757, -0.477]],
    dtype=np.float64,
)
WATER_Z = np.array([8, 1, 1], dtype=np.int32)

AMMONIA_COORDS = np.array(
    [
        [0.000, 0.000, 0.116],
        [0.000, 0.939, -0.271],
        [0.813, -0.469, -0.271],
        [-0.813, -0.469, -0.271],
    ],
    dtype=np.float64,
)
AMMONIA_Z = np.array([7, 1, 1, 1], dtype=np.int32)

METHANE_COORDS = np.array(
    [
        [0.000, 0.000, 0.000],
        [0.629, 0.629, 0.629],
        [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629],
        [0.629, -0.629, -0.629],
    ],
    dtype=np.float64,
)
METHANE_Z = np.array([6, 1, 1, 1, 1], dtype=np.int32)

HF_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]], dtype=np.float64)
HF_Z = np.array([1, 9], dtype=np.int32)

SIGMA = 2.5
KERNEL_ARGS = dict(
    two_body_scaling=2.5,
    two_body_width=0.1,
    two_body_power=4.5,
    three_body_scaling=1.5,
    three_body_width=3.0,
    three_body_power=3.0,
    cut_start=1.0,
    cut_distance=1e6,
    fourier_order=1,
    use_atm=False,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hessian_symm_shape_single():
    """Shape (D, D) = (n_atoms*3, n_atoms*3) for single molecule."""
    H = kernel_mod.kernel_gaussian_hessian_symm(
        [WATER_COORDS], [WATER_Z], sigma=SIGMA, **KERNEL_ARGS
    )
    D = 3 * 3  # water has 3 atoms
    assert H.shape == (D, D), f"Expected ({D},{D}), got {H.shape}"
    assert H.dtype == np.float64


def test_hessian_symm_shape_batch():
    """Shape (D, D) for batch of molecules with variable sizes."""
    coords = [WATER_COORDS, AMMONIA_COORDS, HF_COORDS]
    zs = [WATER_Z, AMMONIA_Z, HF_Z]
    D = (3 + 4 + 2) * 3  # 27
    H = kernel_mod.kernel_gaussian_hessian_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    assert H.shape == (D, D), f"Expected ({D},{D}), got {H.shape}"


def test_hessian_symm_is_symmetric():
    """H[i,j] == H[j,i] exactly (bit-for-bit symmetric)."""
    coords = [WATER_COORDS, AMMONIA_COORDS]
    zs = [WATER_Z, AMMONIA_Z]
    H = kernel_mod.kernel_gaussian_hessian_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    np.testing.assert_array_equal(
        H, H.T, err_msg="kernel_gaussian_hessian_symm is not exactly symmetric"
    )


def test_hessian_symm_matches_hessian_asymm():
    """kernel_gaussian_hessian_symm == kernel_gaussian_hessian(mols, mols)."""
    coords = [WATER_COORDS, AMMONIA_COORDS]
    zs = [WATER_Z, AMMONIA_Z]

    H_symm = kernel_mod.kernel_gaussian_hessian_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    H_asym = kernel_mod.kernel_gaussian_hessian(coords, zs, coords, zs, sigma=SIGMA, **KERNEL_ARGS)

    np.testing.assert_allclose(
        H_symm,
        H_asym,
        rtol=1e-10,
        atol=1e-12,
        err_msg=(
            f"hessian_symm vs hessian asymm mismatch.\n"
            f"  max abs diff = {np.max(np.abs(H_symm - H_asym)):.3e}"
        ),
    )


def test_hessian_symm_matches_hessian_three_mols():
    """Batch of 3 molecules: symm == asymm."""
    coords = [WATER_COORDS, HF_COORDS, METHANE_COORDS]
    zs = [WATER_Z, HF_Z, METHANE_Z]

    H_symm = kernel_mod.kernel_gaussian_hessian_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    H_asym = kernel_mod.kernel_gaussian_hessian(coords, zs, coords, zs, sigma=SIGMA, **KERNEL_ARGS)

    np.testing.assert_allclose(
        H_symm, H_asym, rtol=1e-10, atol=1e-12, err_msg="3-mol batch: hessian_symm != hessian_asymm"
    )


def test_hessian_symm_matches_rfp():
    """kernel_gaussian_hessian_symm unpacked from RFP == hessian_symm full matrix."""
    coords = [WATER_COORDS, AMMONIA_COORDS]
    zs = [WATER_Z, AMMONIA_Z]
    D = (3 + 4) * 3
    BIG = D

    H_full = kernel_mod.kernel_gaussian_hessian_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    rfp = kernel_mod.kernel_gaussian_hessian_symm_rfp(coords, zs, sigma=SIGMA, **KERNEL_ARGS)

    # Unpack RFP
    k = BIG // 2
    stride = (BIG + 1) if (BIG % 2 == 0) else BIG
    H_unpack = np.zeros((BIG, BIG), dtype=np.float64)
    for col in range(BIG):
        for row in range(col, BIG):
            if row >= k:
                idx = (row - k) * stride + col
            else:
                idx = col * stride + (row + k + 1)
            v = rfp[idx]
            H_unpack[row, col] = v
            H_unpack[col, row] = v

    np.testing.assert_allclose(
        H_full,
        H_unpack,
        rtol=1e-12,
        atol=1e-14,
        err_msg="hessian_symm full != hessian_symm_rfp unpacked",
    )


def test_hessian_symm_raises_use_atm():
    """use_atm=True must raise an exception."""
    args = dict(KERNEL_ARGS, use_atm=True)
    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian_hessian_symm([WATER_COORDS], [WATER_Z], sigma=SIGMA, **args)


def test_hessian_symm_raises_cutoff():
    """cut_start < 1.0 must raise an exception."""
    args = dict(KERNEL_ARGS, cut_start=0.5)
    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian_hessian_symm([WATER_COORDS], [WATER_Z], sigma=SIGMA, **args)
