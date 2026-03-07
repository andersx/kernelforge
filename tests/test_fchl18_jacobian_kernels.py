"""Tests for kernel_gaussian_jacobian.

kernel_gaussian_jacobian returns J[row, b] = dK(A_i, B_b)/dR_{A_i}[flat_coord]
where row = sum_{k<i} n_atoms_k*3 + atom_alpha*3 + mu.

Verified numerically using central differences on kernel_gaussian.
Also verified that J == kernel_gaussian_jacobian_t.T (by symmetry of K).
"""

from typing import TypedDict

import numpy as np

import kernelforge.fchl18_kernel as kernel_mod
import kernelforge.fchl18_repr as repr_mod


class _KernelArgs(TypedDict):
    two_body_scaling: float
    two_body_width: float
    two_body_power: float
    three_body_scaling: float
    three_body_width: float
    three_body_power: float
    cut_start: float
    cut_distance: float
    fourier_order: int
    use_atm: bool


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
KERNEL_ARGS: _KernelArgs = {
    "two_body_scaling": 2.0,
    "two_body_width": 0.1,
    "two_body_power": 6.0,
    "three_body_scaling": 2.0,
    "three_body_width": 3.0,
    "three_body_power": 3.0,
    "cut_start": 0.5,
    "cut_distance": 1e6,
    "fourier_order": 2,
    "use_atm": True,
}


def _make_repr(coords_list, z_list, cut_distance=1e6):
    max_size = max(len(z) for z in z_list)
    return repr_mod.generate(coords_list, z_list, max_size=max_size, cut_distance=cut_distance)


def _numerical_jacobian(coords_A_list, z_A_list, x2, n2, nn2, kernel_args, sigma, eps=1e-4):
    """Central-difference Jacobian dK(A_i, B_b)/dR_{A_i}[alpha, mu].

    Returns array of shape (D_A, N_B) where D_A = sum_i n_atoms_i * 3.
    """
    N_B = x2.shape[0]
    D_A = sum(c.shape[0] * 3 for c in coords_A_list)
    J_num = np.zeros((D_A, N_B), dtype=np.float64)

    row = 0
    for coords_A, z_A in zip(coords_A_list, z_A_list, strict=False):
        na = coords_A.shape[0]
        for alpha in range(na):
            for mu in range(3):
                cp = coords_A.copy()
                cm = coords_A.copy()
                cp[alpha, mu] += eps
                cm[alpha, mu] -= eps

                xp, np_, nnp = _make_repr([cp], [z_A])
                xm, nm_, nnm = _make_repr([cm], [z_A])

                Kp = kernel_mod.kernel_gaussian(
                    xp, x2, np_, n2, nnp, nn2, sigma=sigma, **kernel_args
                )  # (1, N_B)
                Km = kernel_mod.kernel_gaussian(
                    xm, x2, nm_, n2, nnm, nn2, sigma=sigma, **kernel_args
                )

                J_num[row, :] = (Kp[0] - Km[0]) / (2.0 * eps)
                row += 1

    return J_num


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_jacobian_shape_single():
    """Shape is (n_atoms_A * 3, N_B) for single query molecule."""
    x2, n2, nn2 = _make_repr([WATER_COORDS, AMMONIA_COORDS], [WATER_Z, AMMONIA_Z])
    J = kernel_mod.kernel_gaussian_jacobian(
        [WATER_COORDS], [WATER_Z], x2, n2, nn2, sigma=SIGMA, **KERNEL_ARGS
    )
    assert J.shape == (3 * 3, 2), f"Expected (9, 2), got {J.shape}"
    assert J.dtype == np.float64


def test_jacobian_shape_batch():
    """Shape is (D_A_total, N_B) for batch query molecules."""
    coords_A = [WATER_COORDS, HF_COORDS]
    z_A = [WATER_Z, HF_Z]
    D_A = 3 * 3 + 2 * 3  # 9 + 6

    x2, n2, nn2 = _make_repr([AMMONIA_COORDS, METHANE_COORDS], [AMMONIA_Z, METHANE_Z])
    J = kernel_mod.kernel_gaussian_jacobian(coords_A, z_A, x2, n2, nn2, sigma=SIGMA, **KERNEL_ARGS)
    assert J.shape == (D_A, 2), f"Expected ({D_A}, 2), got {J.shape}"


def test_jacobian_numerical_water_vs_water():
    """Analytical Jacobian matches numerical for water query."""
    x2, n2, nn2 = _make_repr([WATER_COORDS, AMMONIA_COORDS], [WATER_Z, AMMONIA_Z])
    J_ana = kernel_mod.kernel_gaussian_jacobian(
        [WATER_COORDS], [WATER_Z], x2, n2, nn2, sigma=SIGMA, **KERNEL_ARGS
    )
    J_num = _numerical_jacobian([WATER_COORDS], [WATER_Z], x2, n2, nn2, KERNEL_ARGS, SIGMA)
    np.testing.assert_allclose(
        J_ana,
        J_num,
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Jacobian mismatch: max abs = {np.max(np.abs(J_ana - J_num)):.3e}",
    )


def test_jacobian_numerical_batch():
    """Analytical Jacobian matches numerical for 2 query molecules."""
    coords_A = [WATER_COORDS, HF_COORDS]
    z_A = [WATER_Z, HF_Z]
    x2, n2, nn2 = _make_repr([AMMONIA_COORDS, METHANE_COORDS], [AMMONIA_Z, METHANE_Z])

    J_ana = kernel_mod.kernel_gaussian_jacobian(
        coords_A, z_A, x2, n2, nn2, sigma=SIGMA, **KERNEL_ARGS
    )
    J_num = _numerical_jacobian(coords_A, z_A, x2, n2, nn2, KERNEL_ARGS, SIGMA)
    np.testing.assert_allclose(
        J_ana,
        J_num,
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Batch Jacobian mismatch: max abs = {np.max(np.abs(J_ana - J_num)):.3e}",
    )


def test_jacobian_vs_gradient():
    """kernel_gaussian_jacobian[0, :] == kernel_gaussian_gradient.reshape(-1, N_B)[row, :]."""
    x2, n2, nn2 = _make_repr([WATER_COORDS, AMMONIA_COORDS], [WATER_Z, AMMONIA_Z])

    # Jacobian for single query
    J = kernel_mod.kernel_gaussian_jacobian(
        [WATER_COORDS], [WATER_Z], x2, n2, nn2, sigma=SIGMA, **KERNEL_ARGS
    )
    # Gradient: shape (n_atoms_A, 3, N_B)
    G = kernel_mod.kernel_gaussian_gradient(
        WATER_COORDS, WATER_Z, x2, n2, nn2, sigma=SIGMA, **KERNEL_ARGS
    )
    # Reshape G to (D_A, N_B)
    G_flat = G.reshape(-1, x2.shape[0])  # (n_atoms*3, N_B)

    np.testing.assert_allclose(
        J,
        G_flat,
        rtol=1e-12,
        atol=1e-14,
        err_msg="kernel_gaussian_jacobian should equal kernel_gaussian_gradient reshaped",
    )


def test_jacobian_is_transpose_of_jacobian_t():
    """J(A->B).T should equal J_t(B->A) up to numerical precision.

    kernel_gaussian_jacobian(A, B): shape (D_A, N_B)
    kernel_gaussian_jacobian_t(B, A): shape (N_A, D_B)
    If A == B (same set), J(A,A).T == J_t(A,A) by kernel symmetry.
    """
    coords = [WATER_COORDS, AMMONIA_COORDS]
    zs = [WATER_Z, AMMONIA_Z]

    # Build repr for this set
    x, n, nn = _make_repr(coords, zs)

    # J: (D, N) where D = (3+4)*3 = 21
    J = kernel_mod.kernel_gaussian_jacobian(coords, zs, x, n, nn, sigma=SIGMA, **KERNEL_ARGS)
    # J_t: (N, D)
    J_t = kernel_mod.kernel_gaussian_jacobian_t(coords, zs, x, n, nn, sigma=SIGMA, **KERNEL_ARGS)

    np.testing.assert_allclose(
        J,
        J_t.T,
        rtol=1e-10,
        atol=1e-12,
        err_msg="kernel_gaussian_jacobian should equal kernel_gaussian_jacobian_t.T",
    )
