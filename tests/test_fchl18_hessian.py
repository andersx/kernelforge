"""Tests for the analytical Hessian kernel of FCHL18.

kernel_gaussian_hessian returns the contracted block matrix
  H[row, col] = d²K[A,B] / dR_A[flat_row] dR_B[flat_col]
where flat_row = atom_alpha*3 + mu,  flat_col = atom_beta*3 + nu.

All tests use use_atm=False and cut_distance=1e6 (cutoff inactive).
Numerical Hessians are computed via double central differences on
kernel_gaussian with eps=5e-3 (found to be reliable in practice).

Additional tests verify the chain:
  Hessian = d/dR_B (Jacobian)
i.e. the analytic Hessian matches central differences of
kernel_gaussian_gradient with respect to B-side coordinates.
"""

import copy
from typing import TypedDict

import numpy as np

import kernelforge.fchl18_kernel as kernel_mod
import kernelforge.fchl18_repr as repr_mod


class _KernelArgs(TypedDict):
    two_body_width: float
    two_body_scaling: float
    two_body_power: float
    three_body_width: float
    three_body_scaling: float
    three_body_power: float
    cut_start: float
    cut_distance: float
    fourier_order: int
    use_atm: bool


# ---------------------------------------------------------------------------
# Small molecule definitions (same as in test_fchl18_gradient.py)
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

# HF: asymmetric diatomic
HF_COORDS = np.array(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]],
    dtype=np.float64,
)
HF_Z = np.array([1, 9], dtype=np.int32)

# Kernel hyperparameters
KERNEL_ARGS: _KernelArgs = {
    "two_body_width": 0.1,
    "two_body_scaling": 2.5,
    "two_body_power": 4.5,
    "three_body_width": 3.0,
    "three_body_scaling": 1.5,
    "three_body_power": 3.0,
    "cut_start": 1.0,
    "cut_distance": 1e6,
    "fourier_order": 1,
    "use_atm": False,
}
SIGMA = 2.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repr(coords_list, z_list, cut_distance=1e6):
    max_size = max(len(z) for z in z_list)
    return repr_mod.generate(coords_list, z_list, max_size=max_size, cut_distance=cut_distance)


def _analytical_hessian(coords_A_list, z_A_list, coords_B_list, z_B_list, kernel_args, sigma):
    """Call kernel_gaussian_hessian and return (D_A, D_B) array."""
    return kernel_mod.kernel_gaussian_hessian(
        coords_A_list,
        z_A_list,
        coords_B_list,
        z_B_list,
        sigma=sigma,
        **kernel_args,
    )


def _numerical_hessian(coords_A, z_A, coords_B, z_B, kernel_args, sigma, eps=5e-3):
    """Double central-difference numerical Hessian for a single (A, B) pair.

    H_num[alpha*3+mu, beta*3+nu]
      = (K(A+,B+) - K(A+,B-) - K(A-,B+) + K(A-,B-)) / (4*eps^2)
    """
    na = coords_A.shape[0]
    nb = coords_B.shape[0]
    D_A = na * 3
    D_B = nb * 3
    H_num = np.zeros((D_A, D_B), dtype=np.float64)

    cut = kernel_args["cut_distance"]

    for amu in range(D_A):
        alpha, mu = divmod(amu, 3)
        A_p = coords_A.copy()
        A_p[alpha, mu] += eps
        A_m = coords_A.copy()
        A_m[alpha, mu] -= eps

        for bnu in range(D_B):
            beta, nu = divmod(bnu, 3)
            B_p = coords_B.copy()
            B_p[beta, nu] += eps
            B_m = coords_B.copy()
            B_m[beta, nu] -= eps

            def K(ca, cb):
                xa, na_arr, nna = _make_repr([ca], [z_A], cut_distance=cut)
                xb, nb_arr, nnb = _make_repr([cb], [z_B], cut_distance=cut)
                return kernel_mod.kernel_gaussian(
                    xa, xb, na_arr, nb_arr, nna, nnb, sigma=sigma, **kernel_args
                )[0, 0]

            H_num[amu, bnu] = (K(A_p, B_p) - K(A_p, B_m) - K(A_m, B_p) + K(A_m, B_m)) / (
                4.0 * eps * eps
            )

    return H_num


def _check_hessian(
    coords_A, z_A, coords_B, z_B, kernel_args, sigma, eps=5e-3, rtol=1e-3, atol=1e-4
):
    H_ana = _analytical_hessian([coords_A], [z_A], [coords_B], [z_B], kernel_args, sigma)
    H_num = _numerical_hessian(coords_A, z_A, coords_B, z_B, kernel_args, sigma, eps=eps)

    assert H_ana.shape == H_num.shape, (
        f"Shape mismatch: analytical {H_ana.shape} vs numerical {H_num.shape}"
    )

    np.testing.assert_allclose(
        H_ana,
        H_num,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"Hessian mismatch.\n"
            f"  max abs diff = {np.max(np.abs(H_ana - H_num)):.3e}\n"
            f"  max rel diff = {np.max(np.abs((H_ana - H_num) / (np.abs(H_num) + 1e-12))):.3e}"
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hessian_shape_water_ammonia():
    """Output shape is (D_A, D_B) = (n_atoms_A*3, n_atoms_B*3)."""
    H = _analytical_hessian(
        [WATER_COORDS],
        [WATER_Z],
        [AMMONIA_COORDS],
        [AMMONIA_Z],
        KERNEL_ARGS,
        SIGMA,
    )
    assert H.shape == (3 * 3, 4 * 3), f"Expected (9, 12), got {H.shape}"


def test_hessian_shape_batch():
    """Batch: (nm_A=2, nm_B=3) gives correct contracted shape."""
    coords_A = [WATER_COORDS, AMMONIA_COORDS]
    z_A = [WATER_Z, AMMONIA_Z]
    coords_B = [HF_COORDS, WATER_COORDS, METHANE_COORDS]
    z_B = [HF_Z, WATER_Z, METHANE_Z]

    D_A = (3 + 4) * 3  # water(3) + ammonia(4) atoms, * 3 coords
    D_B = (2 + 3 + 5) * 3  # HF(2) + water(3) + methane(5) atoms

    H = _analytical_hessian(coords_A, z_A, coords_B, z_B, KERNEL_ARGS, SIGMA)
    assert H.shape == (D_A, D_B), f"Expected ({D_A}, {D_B}), got {H.shape}"


def test_hessian_water_vs_water():
    """Water vs water: same-species, non-trivial Hessian."""
    _check_hessian(WATER_COORDS, WATER_Z, WATER_COORDS, WATER_Z, KERNEL_ARGS, SIGMA)


def test_hessian_water_vs_ammonia():
    """Water vs ammonia: different sizes and element sets."""
    _check_hessian(WATER_COORDS, WATER_Z, AMMONIA_COORDS, AMMONIA_Z, KERNEL_ARGS, SIGMA)


def test_hessian_ammonia_vs_ammonia():
    """Ammonia vs ammonia.

    Uses a smaller eps than the default because certain directions (e.g. the
    y-coordinate of N at the C3v symmetry axis) exhibit high-order curvature
    that makes the double-central-difference inaccurate at eps=5e-3.  The
    analytical Hessian is correct; eps=1e-4 verifies it numerically.
    """
    _check_hessian(
        AMMONIA_COORDS, AMMONIA_Z, AMMONIA_COORDS, AMMONIA_Z, KERNEL_ARGS, SIGMA, eps=1e-4
    )


def test_hessian_ammonia_vs_methane():
    """Ammonia vs methane: different heavy elements (N vs C)."""
    _check_hessian(AMMONIA_COORDS, AMMONIA_Z, METHANE_COORDS, METHANE_Z, KERNEL_ARGS, SIGMA)


def test_hessian_hf_vs_water():
    """HF (diatomic) vs water: small system."""
    _check_hessian(HF_COORDS, HF_Z, WATER_COORDS, WATER_Z, KERNEL_ARGS, SIGMA)


def test_hessian_symmetry():
    """H[A,B] == H[B,A]^T  (by symmetry of the kernel)."""
    H_AB = _analytical_hessian(
        [AMMONIA_COORDS],
        [AMMONIA_Z],
        [WATER_COORDS],
        [WATER_Z],
        KERNEL_ARGS,
        SIGMA,
    )
    H_BA = _analytical_hessian(
        [WATER_COORDS],
        [WATER_Z],
        [AMMONIA_COORDS],
        [AMMONIA_Z],
        KERNEL_ARGS,
        SIGMA,
    )
    np.testing.assert_allclose(
        H_AB,
        H_BA.T,
        rtol=1e-10,
        atol=1e-12,
        err_msg="H[A,B] != H[B,A]^T — symmetry violated",
    )


def test_hessian_use_atm_water_vs_water():
    """use_atm=True: analytical Hessian matches double-central-difference of scalar kernel."""
    args = copy.copy(KERNEL_ARGS)
    args["use_atm"] = True
    _check_hessian(WATER_COORDS, WATER_Z, WATER_COORDS, WATER_Z, args, SIGMA, eps=1e-4)


def test_hessian_use_atm_water_vs_ammonia():
    """use_atm=True, mixed pair: analytical Hessian matches numerical."""
    args = copy.copy(KERNEL_ARGS)
    args["use_atm"] = True
    _check_hessian(WATER_COORDS, WATER_Z, AMMONIA_COORDS, AMMONIA_Z, args, SIGMA, eps=1e-4)


def test_hessian_use_atm_differs_from_no_atm():
    """use_atm=True gives a different result than use_atm=False (ATM is actually active)."""
    args_atm = copy.copy(KERNEL_ARGS)
    args_atm["use_atm"] = True
    args_no = copy.copy(KERNEL_ARGS)
    args_no["use_atm"] = False
    H_atm = _analytical_hessian(
        [WATER_COORDS], [WATER_Z], [AMMONIA_COORDS], [AMMONIA_Z], args_atm, SIGMA
    )
    H_no = _analytical_hessian(
        [WATER_COORDS], [WATER_Z], [AMMONIA_COORDS], [AMMONIA_Z], args_no, SIGMA
    )
    assert not np.allclose(H_atm, H_no, rtol=1e-6, atol=1e-8), (
        "use_atm=True and use_atm=False produced identical Hessians — ATM has no effect"
    )


def test_hessian_active_cutoff_matches_numerical():
    """Active cutoff (cut_start=0.5, cut_distance=2.0): analytical Hessian matches numerical.

    cut_start=0.5, cut_distance=2.0 puts the smooth transition region from 1.0 to 2.0 Å,
    which spans the H-H and N-H intra-molecular distances, so the cutoff derivatives
    are non-trivial.
    """
    args = copy.copy(KERNEL_ARGS)
    args["cut_start"] = 0.5
    args["cut_distance"] = 2.0
    _check_hessian(WATER_COORDS, WATER_Z, AMMONIA_COORDS, AMMONIA_Z, args, SIGMA, eps=1e-4)


def test_hessian_active_cutoff_differs_from_no_cutoff():
    """Active cutoff gives a different result than the infinite-cutoff baseline."""
    args_cut = copy.copy(KERNEL_ARGS)
    args_cut["cut_start"] = 0.5
    args_cut["cut_distance"] = 2.0
    args_no_cut = copy.copy(KERNEL_ARGS)
    args_no_cut["cut_start"] = 0.5
    args_no_cut["cut_distance"] = 1e6
    H_cut = _analytical_hessian(
        [WATER_COORDS], [WATER_Z], [AMMONIA_COORDS], [AMMONIA_Z], args_cut, SIGMA
    )
    H_no_cut = _analytical_hessian(
        [WATER_COORDS], [WATER_Z], [AMMONIA_COORDS], [AMMONIA_Z], args_no_cut, SIGMA
    )
    assert not np.allclose(H_cut, H_no_cut, rtol=1e-6, atol=1e-8), (
        "Active cutoff and infinite cutoff produced identical Hessians — cutoff has no effect"
    )


def test_hessian_use_atm_and_cutoff_matches_numerical():
    """Combination of use_atm=True and active cutoff: analytical Hessian matches numerical."""
    args = copy.copy(KERNEL_ARGS)
    args["use_atm"] = True
    args["cut_start"] = 0.5
    args["cut_distance"] = 2.0
    _check_hessian(WATER_COORDS, WATER_Z, WATER_COORDS, WATER_Z, args, SIGMA, eps=1e-4)


def test_hessian_transpose_symmetry_water_ammonia():
    """H(A, B) == H(B, A).T  (asymmetric hessian transpose identity)."""
    H_AB = kernel_mod.kernel_gaussian_hessian(
        [WATER_COORDS], [WATER_Z], [AMMONIA_COORDS], [AMMONIA_Z], sigma=SIGMA, **KERNEL_ARGS
    )
    H_BA = kernel_mod.kernel_gaussian_hessian(
        [AMMONIA_COORDS], [AMMONIA_Z], [WATER_COORDS], [WATER_Z], sigma=SIGMA, **KERNEL_ARGS
    )
    np.testing.assert_allclose(
        H_AB,
        H_BA.T,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"H(A,B) != H(B,A).T: max abs diff = {np.max(np.abs(H_AB - H_BA.T)):.3e}",
    )


def test_hessian_transpose_symmetry_batch():
    """H(A, B) == H(B, A).T for batch of two molecules on each side."""
    coords_A = [WATER_COORDS, HF_COORDS]
    z_A = [WATER_Z, HF_Z]
    coords_B = [AMMONIA_COORDS, METHANE_COORDS]
    z_B = [AMMONIA_Z, METHANE_Z]

    H_AB = kernel_mod.kernel_gaussian_hessian(
        coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS
    )
    H_BA = kernel_mod.kernel_gaussian_hessian(
        coords_B, z_B, coords_A, z_A, sigma=SIGMA, **KERNEL_ARGS
    )
    np.testing.assert_allclose(
        H_AB,
        H_BA.T,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"batch H(A,B) != H(B,A).T: max abs diff = {np.max(np.abs(H_AB - H_BA.T)):.3e}",
    )


def test_hessian_transpose_symmetry_use_atm():
    """H(A, B) == H(B, A).T holds with use_atm=True."""
    args = copy.copy(KERNEL_ARGS)
    args["use_atm"] = True
    H_AB = kernel_mod.kernel_gaussian_hessian(
        [WATER_COORDS], [WATER_Z], [AMMONIA_COORDS], [AMMONIA_Z], sigma=SIGMA, **args
    )
    H_BA = kernel_mod.kernel_gaussian_hessian(
        [AMMONIA_COORDS], [AMMONIA_Z], [WATER_COORDS], [WATER_Z], sigma=SIGMA, **args
    )
    np.testing.assert_allclose(
        H_AB,
        H_BA.T,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"use_atm H(A,B) != H(B,A).T: max abs diff = {np.max(np.abs(H_AB - H_BA.T)):.3e}",
    )


# ---------------------------------------------------------------------------
# Hessian = d/dR_B(Jacobian) tests
# ---------------------------------------------------------------------------


def _numerical_hessian_from_jacobian(coords_A, z_A, coords_B, z_B, kernel_args, sigma, eps=5e-3):
    """Central-difference Hessian by differentiating kernel_gaussian_gradient wrt R_B.

    H_num[alpha*3+mu, beta*3+nu]
      = (grad_A(B+eps)[alpha,mu] - grad_A(B-eps)[alpha,mu]) / (2*eps)

    where grad_A(B) = kernel_gaussian_gradient(coords_A, z_A, x_B, n_B, nn_B)
                    shape (n_atoms_A, 3, 1)  -- derivative wrt coords_A.
    """
    na = coords_A.shape[0]
    nb = coords_B.shape[0]
    D_A = na * 3
    D_B = nb * 3
    H_num = np.zeros((D_A, D_B), dtype=np.float64)

    cut = kernel_args["cut_distance"]

    for bnu in range(D_B):
        beta, nu = divmod(bnu, 3)
        B_p = coords_B.copy()
        B_p[beta, nu] += eps
        B_m = coords_B.copy()
        B_m[beta, nu] -= eps

        x_Bp, n_Bp, nn_Bp = _make_repr([B_p], [z_B], cut_distance=cut)
        x_Bm, n_Bm, nn_Bm = _make_repr([B_m], [z_B], cut_distance=cut)

        # kernel_gaussian_gradient(coords_A, z_A, x_B, n_B, nn_B, ...)
        # returns dK[A_query, B_training]/dR_A, shape (n_atoms_A, 3, nm_B=1)
        grad_p = kernel_mod.kernel_gaussian_gradient(
            coords_A, z_A, x_Bp, n_Bp, nn_Bp, sigma=sigma, **kernel_args
        )  # (n_atoms_A, 3, 1)
        grad_m = kernel_mod.kernel_gaussian_gradient(
            coords_A, z_A, x_Bm, n_Bm, nn_Bm, sigma=sigma, **kernel_args
        )  # (n_atoms_A, 3, 1)

        # d(grad_A)/d(R_B[beta,nu]) ~ (grad_p - grad_m) / (2*eps)
        # flatten (n_atoms_A, 3) -> (D_A,)
        H_num[:, bnu] = ((grad_p[:, :, 0] - grad_m[:, :, 0]) / (2.0 * eps)).reshape(D_A)

    return H_num


def _check_hessian_vs_jacobian(
    coords_A, z_A, coords_B, z_B, kernel_args, sigma, eps=5e-3, rtol=5e-4, atol=1e-6
):
    """Check analytic Hessian == central-difference of Jacobian (gradient) wrt R_B."""
    H_ana = _analytical_hessian([coords_A], [z_A], [coords_B], [z_B], kernel_args, sigma)
    H_num = _numerical_hessian_from_jacobian(
        coords_A, z_A, coords_B, z_B, kernel_args, sigma, eps=eps
    )

    assert H_ana.shape == H_num.shape, (
        f"Shape mismatch: analytical {H_ana.shape} vs numerical {H_num.shape}"
    )

    np.testing.assert_allclose(
        H_ana,
        H_num,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"Hessian != d/dR_B(Jacobian).\n"
            f"  max abs diff = {np.max(np.abs(H_ana - H_num)):.3e}\n"
            f"  max rel diff = "
            f"{np.max(np.abs((H_ana - H_num) / (np.abs(H_num) + 1e-12))):.3e}"
        ),
    )


def test_hessian_equals_jacobian_derivative_water_vs_water():
    """H[A,B] = d/dR_B(Jacobian): water vs water."""
    _check_hessian_vs_jacobian(WATER_COORDS, WATER_Z, WATER_COORDS, WATER_Z, KERNEL_ARGS, SIGMA)


def test_hessian_equals_jacobian_derivative_water_vs_ammonia():
    """H[A,B] = d/dR_B(Jacobian): water vs ammonia (different sizes).

    Uses eps=1e-4 because the y-coordinate of N in ammonia has high curvature
    near the C3v symmetry axis, requiring a smaller finite-difference step.
    """
    _check_hessian_vs_jacobian(
        WATER_COORDS, WATER_Z, AMMONIA_COORDS, AMMONIA_Z, KERNEL_ARGS, SIGMA, eps=1e-4
    )


def test_hessian_equals_jacobian_derivative_hf_vs_water():
    """H[A,B] = d/dR_B(Jacobian): HF (diatomic) vs water."""
    _check_hessian_vs_jacobian(HF_COORDS, HF_Z, WATER_COORDS, WATER_Z, KERNEL_ARGS, SIGMA)


def test_hessian_equals_jacobian_derivative_use_atm():
    """H[A,B] = d/dR_B(Jacobian): use_atm=True, water vs ammonia."""
    args = copy.copy(KERNEL_ARGS)
    args["use_atm"] = True
    _check_hessian_vs_jacobian(
        WATER_COORDS, WATER_Z, AMMONIA_COORDS, AMMONIA_Z, args, SIGMA, eps=1e-4
    )
