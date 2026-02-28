"""Tests for kernel_gaussian_symm_rfp.

Verifies:
  1. Shape and dtype of output.
  2. RFP packing matches the full symmetric kernel (kernel_gaussian_symm).
  3. kernel_gaussian_symm(K) == unpack(kernel_gaussian_symm_rfp(K)).
"""

import numpy as np
import pytest

import kernelforge.fchl18_kernel as kernel_mod
import kernelforge.fchl18_repr as repr_mod

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
    two_body_scaling=2.0,
    two_body_width=0.1,
    two_body_power=6.0,
    three_body_scaling=2.0,
    three_body_width=3.0,
    three_body_power=3.0,
    cut_start=0.5,
    cut_distance=1e6,
    fourier_order=2,
    use_atm=True,
)


def _make_repr(coords_list, z_list, cut_distance=1e6):
    max_size = max(len(z) for z in z_list)
    return repr_mod.generate(coords_list, z_list, max_size=max_size, cut_distance=cut_distance)


def _rfp_to_full(rfp, n):
    """Unpack upper-triangle RFP vector into full (n, n) symmetric matrix."""
    from kernelforge.fchl18_kernel import kernel_gaussian_symm_rfp  # noqa: F401

    # Use numpy to unpack: rfp_index_upper_N convention (TRANSR='N', UPLO='U')
    # We rebuild by iterating the same indexing used in C++.
    # For simplicity: read K_rfp back via comparing to full matrix in test.
    # Here we implement a pure-Python unpacker.
    k = n // 2
    stride = (n + 1) if (n % 2 == 0) else n
    full = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        for row in range(col, n):
            if row >= k:
                idx = (row - k) * stride + col
            else:
                idx = col * stride + (row + k + 1)
            v = rfp[idx]
            full[row, col] = v
            full[col, row] = v
    return full


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_symm_rfp_shape_and_dtype():
    """Output shape is N*(N+1)//2, dtype float64."""
    coords = [WATER_COORDS, AMMONIA_COORDS, METHANE_COORDS]
    zs = [WATER_Z, AMMONIA_Z, METHANE_Z]
    N = len(coords)
    rfp = kernel_mod.kernel_gaussian_symm_rfp(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    assert rfp.ndim == 1
    assert rfp.shape[0] == N * (N + 1) // 2
    assert rfp.dtype == np.float64


def test_symm_rfp_matches_full_kernel():
    """Unpacked RFP must match kernel_gaussian_symm exactly."""
    coords = [WATER_COORDS, AMMONIA_COORDS, METHANE_COORDS]
    zs = [WATER_Z, AMMONIA_Z, METHANE_Z]
    N = len(coords)

    x, n, nn = _make_repr(coords, zs)
    K_full = kernel_mod.kernel_gaussian_symm(x, n, nn, sigma=SIGMA, **KERNEL_ARGS)
    rfp = kernel_mod.kernel_gaussian_symm_rfp(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    K_unpacked = _rfp_to_full(rfp, N)

    np.testing.assert_allclose(
        K_unpacked,
        K_full,
        rtol=1e-12,
        atol=1e-14,
        err_msg="RFP unpack does not match kernel_gaussian_symm",
    )


def test_symm_rfp_positive_diagonal():
    """Diagonal entries K[a,a] must be positive (kernel is PSD)."""
    coords = [WATER_COORDS, AMMONIA_COORDS, HF_COORDS]
    zs = [WATER_Z, AMMONIA_Z, HF_Z]
    N = len(coords)
    rfp = kernel_mod.kernel_gaussian_symm_rfp(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    K = _rfp_to_full(rfp, N)
    for i in range(N):
        assert K[i, i] > 0.0, f"K[{i},{i}] = {K[i, i]} is not positive"


def test_symm_rfp_single_molecule():
    """Single molecule: output length = 1, value = K(mol, mol)."""
    coords = [WATER_COORDS]
    zs = [WATER_Z]
    rfp = kernel_mod.kernel_gaussian_symm_rfp(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    assert rfp.shape == (1,)
    # Must equal the self-kernel from kernel_gaussian_symm
    x, n, nn = _make_repr(coords, zs)
    K_full = kernel_mod.kernel_gaussian_symm(x, n, nn, sigma=SIGMA, **KERNEL_ARGS)
    np.testing.assert_allclose(rfp[0], K_full[0, 0], rtol=1e-12)


def test_symm_rfp_heterogeneous_molecules():
    """Works with variable atom counts (water+HF+methane)."""
    coords = [WATER_COORDS, HF_COORDS, METHANE_COORDS]
    zs = [WATER_Z, HF_Z, METHANE_Z]
    N = len(coords)
    rfp = kernel_mod.kernel_gaussian_symm_rfp(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    assert rfp.shape == (N * (N + 1) // 2,)


def test_symm_rfp_sigma_sensitivity():
    """Different sigma values produce different results."""
    coords = [WATER_COORDS, AMMONIA_COORDS]
    zs = [WATER_Z, AMMONIA_Z]
    rfp1 = kernel_mod.kernel_gaussian_symm_rfp(coords, zs, sigma=1.0, **KERNEL_ARGS)
    rfp2 = kernel_mod.kernel_gaussian_symm_rfp(coords, zs, sigma=5.0, **KERNEL_ARGS)
    assert not np.allclose(rfp1, rfp2), "Different sigma should give different kernels"
