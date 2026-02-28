"""Tests for the FCHL18 full combined energy+force kernels:
  - kernel_gaussian_full
  - kernel_gaussian_full_symm
  - kernel_gaussian_full_symm_rfp

Block layout for kernel_gaussian_full:
  K[0:N_A,   0:N_B ]  scalar   K[a,b]
  K[0:N_A,   N_B:  ]  jac_t    dK/dR_B
  K[N_A:,    0:N_B ]  jac      dK/dR_A
  K[N_A:,    N_B:  ]  hessian  d²K/dR_A dR_B

All tests use use_atm=False and cut_start=1.0 (required for hessian).
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


def _make_repr(coords_list, z_list, cut_distance=1e6):
    max_size = max(len(z) for z in z_list)
    return repr_mod.generate(coords_list, z_list, max_size=max_size, cut_distance=cut_distance)


def _rfp_to_full(rfp, n):
    """Unpack upper-triangle RFP vector into full (n, n) symmetric matrix."""
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


def _offsets(coords_list):
    offsets = [0]
    for c in coords_list:
        offsets.append(offsets[-1] + c.shape[0] * 3)
    return offsets


# ---------------------------------------------------------------------------
# kernel_gaussian_full tests
# ---------------------------------------------------------------------------


def test_full_shape():
    """Shape is (N_A + D_A, N_B + D_B)."""
    coords_A = [WATER_COORDS, HF_COORDS]
    z_A = [WATER_Z, HF_Z]
    coords_B = [AMMONIA_COORDS, METHANE_COORDS, WATER_COORDS]
    z_B = [AMMONIA_Z, METHANE_Z, WATER_Z]

    N_A = 2
    D_A = (3 + 2) * 3  # 15
    N_B = 3
    D_B = (4 + 5 + 3) * 3  # 36

    K = kernel_mod.kernel_gaussian_full(coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS)
    assert K.shape == (N_A + D_A, N_B + D_B), f"Expected ({N_A + D_A}, {N_B + D_B}), got {K.shape}"
    assert K.dtype == np.float64


def test_full_scalar_block_matches_kernel_gaussian():
    """Upper-left block K[0:N_A, 0:N_B] == kernel_gaussian(A, B)."""
    coords_A = [WATER_COORDS, HF_COORDS]
    z_A = [WATER_Z, HF_Z]
    coords_B = [AMMONIA_COORDS, METHANE_COORDS]
    z_B = [AMMONIA_Z, METHANE_Z]
    N_A, N_B = 2, 2

    K_full = kernel_mod.kernel_gaussian_full(
        coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS
    )

    # Reference scalar block from kernel_gaussian
    all_coords = coords_A + coords_B
    all_z = z_A + z_B
    x_A, n_A, nn_A = _make_repr(coords_A, z_A)
    x_B, n_B, nn_B = _make_repr(coords_B, z_B)
    K_scalar = kernel_mod.kernel_gaussian(
        x_A, x_B, n_A, n_B, nn_A, nn_B, sigma=SIGMA, **KERNEL_ARGS
    )

    np.testing.assert_allclose(
        K_full[:N_A, :N_B],
        K_scalar,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Scalar block of full kernel != kernel_gaussian",
    )


def test_full_jac_block_matches_jacobian():
    """Lower-left block K[N_A:, 0:N_B] == kernel_gaussian_jacobian(A, repr_B)."""
    coords_A = [WATER_COORDS]
    z_A = [WATER_Z]
    coords_B = [AMMONIA_COORDS, METHANE_COORDS]
    z_B = [AMMONIA_Z, METHANE_Z]
    N_A = 1
    D_A = 3 * 3  # water
    N_B = 2

    K_full = kernel_mod.kernel_gaussian_full(
        coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS
    )

    x_B, n_B, nn_B = _make_repr(coords_B, z_B)
    J = kernel_mod.kernel_gaussian_jacobian(
        coords_A, z_A, x_B, n_B, nn_B, sigma=SIGMA, **KERNEL_ARGS
    )

    np.testing.assert_allclose(
        K_full[N_A : N_A + D_A, :N_B],
        J,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Jac block of full kernel != kernel_gaussian_jacobian",
    )


def test_full_jact_block_matches_jacobian_t():
    """Upper-right block K[0:N_A, N_B:] == kernel_gaussian_jacobian_t (via gradient)."""
    coords_A = [WATER_COORDS]
    z_A = [WATER_Z]
    coords_B = [AMMONIA_COORDS]
    z_B = [AMMONIA_Z]
    N_A = 1
    D_A = 3 * 3
    N_B = 1
    D_B = 4 * 3

    K_full = kernel_mod.kernel_gaussian_full(
        coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS
    )
    jac_t_block = K_full[:N_A, N_B:]  # (1, D_B)

    # Reference from kernel_gaussian_gradient(B_j, repr_A)
    x_A, n_A, nn_A = _make_repr(coords_A, z_A)
    G = kernel_mod.kernel_gaussian_gradient(
        AMMONIA_COORDS, AMMONIA_Z, x_A, n_A, nn_A, sigma=SIGMA, **KERNEL_ARGS
    )
    # G[beta, nu, i] = dK(B_j, A_i)/dR_B[beta,nu]; shape (4, 3, 1)
    jac_t_ref = G.reshape(1, -1)  # (1, D_B) with row=A_i, col=B_j_flat

    np.testing.assert_allclose(
        jac_t_block,
        jac_t_ref,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Jac_t block of full kernel != expected",
    )


def test_full_hessian_block_matches_hessian():
    """Lower-right block K[N_A:, N_B:] == kernel_gaussian_hessian(A, B)."""
    coords_A = [WATER_COORDS]
    z_A = [WATER_Z]
    coords_B = [AMMONIA_COORDS]
    z_B = [AMMONIA_Z]
    N_A = 1
    D_A = 3 * 3
    N_B = 1
    D_B = 4 * 3

    K_full = kernel_mod.kernel_gaussian_full(
        coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS
    )
    H_block = K_full[N_A:, N_B:]

    H_ref = kernel_mod.kernel_gaussian_hessian(
        coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS
    )

    np.testing.assert_allclose(
        H_block,
        H_ref,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Hessian block of full kernel != kernel_gaussian_hessian",
    )


# ---------------------------------------------------------------------------
# kernel_gaussian_full_symm tests
# ---------------------------------------------------------------------------


def test_full_symm_shape():
    """Shape is (N+D, N+D) for symmetric full kernel."""
    coords = [WATER_COORDS, AMMONIA_COORDS]
    zs = [WATER_Z, AMMONIA_Z]
    N = 2
    D = (3 + 4) * 3
    BIG = N + D

    K = kernel_mod.kernel_gaussian_full_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    assert K.shape == (BIG, BIG), f"Expected ({BIG},{BIG}), got {K.shape}"
    assert K.dtype == np.float64


def test_full_symm_is_symmetric():
    """kernel_gaussian_full_symm output is exactly symmetric."""
    coords = [WATER_COORDS, AMMONIA_COORDS]
    zs = [WATER_Z, AMMONIA_Z]
    K = kernel_mod.kernel_gaussian_full_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    np.testing.assert_array_equal(K, K.T, err_msg="full_symm kernel is not exactly symmetric")


def test_full_symm_matches_full_asymm():
    """kernel_gaussian_full_symm == kernel_gaussian_full(mols, mols)."""
    coords = [WATER_COORDS, HF_COORDS]
    zs = [WATER_Z, HF_Z]

    K_symm = kernel_mod.kernel_gaussian_full_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    K_asym = kernel_mod.kernel_gaussian_full(coords, zs, coords, zs, sigma=SIGMA, **KERNEL_ARGS)

    np.testing.assert_allclose(
        K_symm,
        K_asym,
        rtol=1e-10,
        atol=1e-12,
        err_msg=(
            f"full_symm vs full_asymm mismatch. "
            f"max abs diff = {np.max(np.abs(K_symm - K_asym)):.3e}"
        ),
    )


# ---------------------------------------------------------------------------
# kernel_gaussian_full_symm_rfp tests
# ---------------------------------------------------------------------------


def test_full_symm_rfp_shape():
    """Output length is BIG*(BIG+1)//2."""
    coords = [WATER_COORDS, AMMONIA_COORDS]
    zs = [WATER_Z, AMMONIA_Z]
    N = 2
    D = (3 + 4) * 3
    BIG = N + D

    rfp = kernel_mod.kernel_gaussian_full_symm_rfp(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    assert rfp.shape == (BIG * (BIG + 1) // 2,), (
        f"Expected ({BIG * (BIG + 1) // 2},), got {rfp.shape}"
    )
    assert rfp.dtype == np.float64


def test_full_symm_rfp_matches_full_symm():
    """Unpacked RFP must match kernel_gaussian_full_symm exactly."""
    coords = [WATER_COORDS, HF_COORDS]
    zs = [WATER_Z, HF_Z]
    N = 2
    D = (3 + 2) * 3
    BIG = N + D

    K_full = kernel_mod.kernel_gaussian_full_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    rfp = kernel_mod.kernel_gaussian_full_symm_rfp(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    K_unpacked = _rfp_to_full(rfp, BIG)

    np.testing.assert_allclose(
        K_unpacked,
        K_full,
        rtol=1e-12,
        atol=1e-14,
        err_msg=(
            f"full_symm_rfp unpack != full_symm. "
            f"max abs diff = {np.max(np.abs(K_unpacked - K_full)):.3e}"
        ),
    )


def test_full_symm_rfp_single_mol():
    """Single molecule: output length (1+D)*(1+D+1)//2, matches symm."""
    coords = [WATER_COORDS]
    zs = [WATER_Z]
    N = 1
    D = 3 * 3
    BIG = N + D

    K_full = kernel_mod.kernel_gaussian_full_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    rfp = kernel_mod.kernel_gaussian_full_symm_rfp(coords, zs, sigma=SIGMA, **KERNEL_ARGS)
    K_unpacked = _rfp_to_full(rfp, BIG)

    np.testing.assert_allclose(K_unpacked, K_full, rtol=1e-12, atol=1e-14)


def test_full_raises_use_atm():
    """use_atm=True must raise for all full kernel variants."""
    args = dict(KERNEL_ARGS, use_atm=True)
    coords = [WATER_COORDS]
    zs = [WATER_Z]

    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian_full(coords, zs, coords, zs, sigma=SIGMA, **args)
    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian_full_symm(coords, zs, sigma=SIGMA, **args)
    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian_full_symm_rfp(coords, zs, sigma=SIGMA, **args)


def test_full_raises_cutoff():
    """cut_start < 1.0 must raise for all full kernel variants."""
    args = dict(KERNEL_ARGS, cut_start=0.5)
    coords = [WATER_COORDS]
    zs = [WATER_Z]

    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian_full(coords, zs, coords, zs, sigma=SIGMA, **args)
    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian_full_symm(coords, zs, sigma=SIGMA, **args)
    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian_full_symm_rfp(coords, zs, sigma=SIGMA, **args)
