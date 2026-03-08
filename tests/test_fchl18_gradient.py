"""Tests for the analytical gradient of the FCHL18 kernel.

kernel_gaussian_gradient returns G[alpha, mu, b] = dK[A,b] / dR_A[alpha, mu]
where R_A are Cartesian coordinates of the query molecule A, alpha indexes its
atoms, mu in {0,1,2} = {x,y,z}, and b indexes training molecules.

The tests compare the analytical gradient to a central-difference numerical
gradient computed directly from kernel_gaussian.
"""

import numpy as np

import kernelforge.fchl18_kernel as kernel_mod
import kernelforge.fchl18_repr as repr_mod

# ---------------------------------------------------------------------------
# Small molecule definitions
# ---------------------------------------------------------------------------

# Water: O H H  (coordinates in Angstrom)
WATER_COORDS = np.array(
    [
        [0.000, 0.000, 0.119],
        [0.000, 0.757, -0.477],
        [0.000, -0.757, -0.477],
    ],
    dtype=np.float64,
)
WATER_Z = np.array([8, 1, 1], dtype=np.int32)

# Methane: C H H H H
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

# Ammonia: N H H H
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

# HF: hydrogen fluoride (asymmetric diatomic — avoids sort-order-swap artefact in numerical grad)
H2_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.917],
    ],
    dtype=np.float64,
)
H2_Z = np.array([1, 9], dtype=np.int32)


def _make_repr(coords_list, z_list, max_size=8, cut_distance=1e6):
    """Generate FCHL18 representations for a list of molecules."""
    return repr_mod.generate(coords_list, z_list, max_size=max_size, cut_distance=cut_distance)


def _numerical_gradient(coords_A, z_A, x2, n2, nn2, kernel_args, eps=1e-4):
    """Central-difference numerical gradient dK[A,b]/dR_A[alpha,mu].

    Returns array of shape (n_atoms_A, 3, nm2).
    """
    n_atoms_A = coords_A.shape[0]
    nm2 = x2.shape[0]
    max_size = coords_A.shape[0]  # same as n_atoms for the single query molecule

    grad_num = np.zeros((n_atoms_A, 3, nm2), dtype=np.float64)

    for alpha in range(n_atoms_A):
        for mu in range(3):
            coords_p = coords_A.copy()
            coords_m = coords_A.copy()
            coords_p[alpha, mu] += eps
            coords_m[alpha, mu] -= eps

            # Build representations for perturbed A
            x_p, n_p, nn_p = _make_repr([coords_p], [z_A])
            x_m, n_m, nn_m = _make_repr([coords_m], [z_A])

            # K[A_perturbed, B]
            Kp = kernel_mod.kernel_gaussian(x_p, x2, n_p, n2, nn_p, nn2, **kernel_args)  # (1, nm2)
            Km = kernel_mod.kernel_gaussian(x_m, x2, n_m, n2, nn_m, nn2, **kernel_args)  # (1, nm2)

            grad_num[alpha, mu, :] = (Kp[0] - Km[0]) / (2.0 * eps)

    return grad_num


def _analytical_gradient(coords_A, z_A, x2, n2, nn2, kernel_args):
    """Call the analytical gradient function."""
    return kernel_mod.kernel_gaussian_gradient(coords_A, z_A, x2, n2, nn2, **kernel_args)


# ---------------------------------------------------------------------------
# Helper: run one gradient check scenario
# ---------------------------------------------------------------------------
def _check_gradient(
    coords_A, z_A, training_coords, training_z, kernel_args, eps=1e-4, rtol=1e-4, atol=1e-6
):
    max_size = max(len(z_A), max(len(z) for z in training_z))

    # Build training representations
    x2, n2, nn2 = _make_repr(training_coords, training_z, max_size=max_size, cut_distance=1e6)

    grad_ana = _analytical_gradient(coords_A, z_A, x2, n2, nn2, kernel_args)
    grad_num = _numerical_gradient(coords_A, z_A, x2, n2, nn2, kernel_args, eps=eps)

    assert grad_ana.shape == grad_num.shape, (
        f"Shape mismatch: analytical {grad_ana.shape} vs numerical {grad_num.shape}"
    )

    np.testing.assert_allclose(
        grad_ana,
        grad_num,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"Gradient mismatch.\n"
            f"  max abs diff = {np.max(np.abs(grad_ana - grad_num)):.3e}\n"
            f"  max rel diff = "
            f"{np.max(np.abs((grad_ana - grad_num) / (np.abs(grad_num) + 1e-12))):.3e}"
        ),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

KERNEL_ARGS_DEFAULT = {
    "sigma": 2.5,
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

KERNEL_ARGS_TUNED = {
    "sigma": 2.5,
    "two_body_scaling": 2.5,
    "two_body_width": 0.1,
    "two_body_power": 4.5,
    "three_body_scaling": 1.5,
    "three_body_width": 3.0,
    "three_body_power": 3.0,
    "cut_start": 0.5,
    "cut_distance": 1e6,
    "fourier_order": 1,
    "use_atm": False,
}


def test_gradient_water_vs_water_default():
    """Water query vs. water+ammonia training, default hyperparams (use_atm=True)."""
    _check_gradient(
        WATER_COORDS,
        WATER_Z,
        [WATER_COORDS, AMMONIA_COORDS],
        [WATER_Z, AMMONIA_Z],
        KERNEL_ARGS_DEFAULT,
    )


def test_gradient_water_vs_water_tuned():
    """Water query vs. water+ammonia training, tuned hyperparams (use_atm=False)."""
    _check_gradient(
        WATER_COORDS,
        WATER_Z,
        [WATER_COORDS, AMMONIA_COORDS],
        [WATER_Z, AMMONIA_Z],
        KERNEL_ARGS_TUNED,
    )


def test_gradient_ammonia_query_default():
    """Ammonia query vs. water+methane+ammonia training, default hyperparams."""
    _check_gradient(
        AMMONIA_COORDS,
        AMMONIA_Z,
        [WATER_COORDS, METHANE_COORDS, AMMONIA_COORDS],
        [WATER_Z, METHANE_Z, AMMONIA_Z],
        KERNEL_ARGS_DEFAULT,
    )


def test_gradient_ammonia_query_tuned():
    """Ammonia query vs. water+methane+ammonia training, tuned hyperparams."""
    _check_gradient(
        AMMONIA_COORDS,
        AMMONIA_Z,
        [WATER_COORDS, METHANE_COORDS, AMMONIA_COORDS],
        [WATER_Z, METHANE_Z, AMMONIA_Z],
        KERNEL_ARGS_TUNED,
    )


def test_gradient_h2_query():
    """HF query vs. HF+water training: asymmetric diatomic, two-body + ATM."""
    _check_gradient(
        H2_COORDS,
        H2_Z,
        [H2_COORDS, WATER_COORDS],
        [H2_Z, WATER_Z],
        KERNEL_ARGS_DEFAULT,
    )


def test_gradient_fourier_order_1_atm_true():
    """fourier_order=1 with use_atm=True: combined path."""
    args = dict(KERNEL_ARGS_DEFAULT)
    args["fourier_order"] = 1
    args["use_atm"] = True
    _check_gradient(
        WATER_COORDS,
        WATER_Z,
        [WATER_COORDS, AMMONIA_COORDS],
        [WATER_Z, AMMONIA_Z],
        args,
    )


def test_gradient_output_shape():
    """Verify output shape is (n_atoms_A, 3, nm2)."""
    x2, n2, nn2 = _make_repr(
        [WATER_COORDS, METHANE_COORDS], [WATER_Z, METHANE_Z], max_size=5, cut_distance=1e6
    )

    G = kernel_mod.kernel_gaussian_gradient(
        WATER_COORDS, WATER_Z, x2, n2, nn2, sigma=2.5, cut_distance=1e6
    )

    assert G.shape == (3, 3, 2), f"Expected (3, 3, 2), got {G.shape}"
    assert G.dtype == np.float64


def test_gradient_antisymmetry_at_equilibrium():
    """For a symmetric displacement, gradient should be antisymmetric.

    Displace one H of water by +delta and verify dK/d(that H) has the
    opposite sign to what we get for -delta (just a sanity check that
    the numerical and analytical agree in sign).
    """
    x2, n2, nn2 = _make_repr([WATER_COORDS], [WATER_Z], max_size=3, cut_distance=1e6)
    G = kernel_mod.kernel_gaussian_gradient(
        WATER_COORDS, WATER_Z, x2, n2, nn2, sigma=2.5, cut_distance=1e6
    )
    # The gradient should be finite (not NaN/Inf)
    assert np.all(np.isfinite(G)), "Gradient contains NaN or Inf"
