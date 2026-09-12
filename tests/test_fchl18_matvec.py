"""Tests for FCHL18 contracted matvec inference kernels."""

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

HF_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]], dtype=np.float64)
HF_Z = np.array([1, 9], dtype=np.int32)

SIGMA = 2.5
KERNEL_ARGS: _KernelArgs = {
    "two_body_scaling": 2.5,
    "two_body_width": 0.1,
    "two_body_power": 4.5,
    "three_body_scaling": 1.5,
    "three_body_width": 3.0,
    "three_body_power": 3.0,
    "cut_start": 1.0,
    "cut_distance": 1e6,
    "fourier_order": 1,
    "use_atm": False,
}


def _offsets(z_list: list[np.ndarray]) -> list[int]:
    offs = [0]
    for z in z_list:
        offs.append(offs[-1] + z.shape[0] * 3)
    return offs


def test_hessian_matvec_matches_dense():
    coords_A = [WATER_COORDS, HF_COORDS]
    z_A = [WATER_Z, HF_Z]
    coords_B = [AMMONIA_COORDS, WATER_COORDS]
    z_B = [AMMONIA_Z, WATER_Z]

    H = kernel_mod.kernel_gaussian_hessian(coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS)
    alpha_F = np.random.default_rng(0).standard_normal(H.shape[1])
    F_ref = H @ alpha_F

    F_mv = kernel_mod.kernel_gaussian_hessian_matvec(
        coords_A, z_A, coords_B, z_B, alpha_F, sigma=SIGMA, **KERNEL_ARGS
    )
    np.testing.assert_allclose(F_mv, F_ref, rtol=1e-12, atol=1e-12)


def test_jacobian_t_matvec_matches_dense():
    coords_train = [AMMONIA_COORDS, WATER_COORDS]
    z_train = [AMMONIA_Z, WATER_Z]
    coords_test = [WATER_COORDS, HF_COORDS]
    z_test = [WATER_Z, HF_Z]
    x_te, n_te, nn_te = repr_mod.generate(coords_test, z_test, max_size=max(len(z) for z in z_test))

    Jt = kernel_mod.kernel_gaussian_jacobian_t(
        coords_train, z_train, x_te, n_te, nn_te, sigma=SIGMA, **KERNEL_ARGS
    )
    alpha_F = np.random.default_rng(1).standard_normal(Jt.shape[1])
    E_ref = Jt @ alpha_F

    E_mv = kernel_mod.kernel_gaussian_jacobian_t_matvec(
        coords_train, z_train, x_te, n_te, nn_te, alpha_F, sigma=SIGMA, **KERNEL_ARGS
    )
    np.testing.assert_allclose(E_mv, E_ref, rtol=1e-12, atol=1e-12)


def test_full_matvec_matches_dense():
    coords_A = [WATER_COORDS, HF_COORDS]
    z_A = [WATER_Z, HF_Z]
    coords_B = [AMMONIA_COORDS, WATER_COORDS]
    z_B = [AMMONIA_Z, WATER_Z]
    N_A, N_B = 2, 2
    off_B = _offsets(z_B)
    D_B = off_B[-1]

    K_full = kernel_mod.kernel_gaussian_full(
        coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS
    )
    alpha_E = np.random.default_rng(2).standard_normal(N_B)
    alpha_F = np.random.default_rng(3).standard_normal(D_B)
    alpha = np.concatenate([alpha_E, alpha_F])
    y_ref = K_full @ alpha

    E_mv, F_mv = kernel_mod.kernel_gaussian_full_matvec(
        coords_A,
        z_A,
        coords_B,
        z_B,
        alpha_E,
        alpha_F,
        sigma=SIGMA,
        compute_energy=True,
        **KERNEL_ARGS,
    )
    np.testing.assert_allclose(E_mv, y_ref[:N_A], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(F_mv, y_ref[N_A:], rtol=1e-12, atol=1e-12)


def test_full_matvec_compute_energy_false():
    coords_A = [WATER_COORDS]
    z_A = [WATER_Z]
    coords_B = [AMMONIA_COORDS]
    z_B = [AMMONIA_Z]
    off_B = _offsets(z_B)
    alpha_E = np.array([1.0])
    alpha_F = np.ones(off_B[-1])

    E_mv, F_mv = kernel_mod.kernel_gaussian_full_matvec(
        coords_A,
        z_A,
        coords_B,
        z_B,
        alpha_E,
        alpha_F,
        sigma=SIGMA,
        compute_energy=False,
        **KERNEL_ARGS,
    )
    assert E_mv.shape == (1,)
    np.testing.assert_allclose(E_mv, 0.0, atol=0.0)
    _, F_with_e = kernel_mod.kernel_gaussian_full_matvec(
        coords_A,
        z_A,
        coords_B,
        z_B,
        alpha_E,
        alpha_F,
        sigma=SIGMA,
        compute_energy=True,
        **KERNEL_ARGS,
    )
    np.testing.assert_allclose(F_mv, F_with_e, rtol=1e-12, atol=1e-12)
