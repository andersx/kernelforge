"""Verify fchl18_kernel accepts the public X/N/NN/Z keyword names."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

import kernelforge.fchl18_kernel as fk
import kernelforge.fchl18_repr as repr_mod

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

SIGMA = 2.5


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


KERNEL_KW: _KernelArgs = {
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


def _make_repr(
    coords_list: list[NDArray[np.float64]],
    z_list: list[NDArray[np.int32]],
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    max_size = max(len(z) for z in z_list)
    return repr_mod.generate(coords_list, z_list, max_size=max_size, cut_distance=1e6)


def test_fchl18_keyword_names_match_positional() -> None:
    coords_a = [WATER_COORDS, AMMONIA_COORDS]
    zs_a = [WATER_Z, AMMONIA_Z]
    coords_b = [AMMONIA_COORDS, WATER_COORDS]
    zs_b = [AMMONIA_Z, WATER_Z]

    x1, n1, nn1 = _make_repr(coords_a, zs_a)
    x2, n2, nn2 = _make_repr(coords_b, zs_b)

    k_pos = fk.kernel_gaussian(x1, x2, n1, n2, nn1, nn2, SIGMA, **KERNEL_KW)
    k_kw = fk.kernel_gaussian(
        X1=x1, X2=x2, N1=n1, N2=n2, NN1=nn1, NN2=nn2, sigma=SIGMA, **KERNEL_KW
    )
    np.testing.assert_array_equal(k_kw, k_pos)

    ks_pos = fk.kernel_gaussian_symm(x1, n1, nn1, SIGMA, **KERNEL_KW)
    ks_kw = fk.kernel_gaussian_symm(X=x1, N=n1, NN=nn1, sigma=SIGMA, **KERNEL_KW)
    np.testing.assert_array_equal(ks_kw, ks_pos)

    rfp_pos = fk.kernel_gaussian_symm_rfp(coords_a, zs_a, SIGMA, **KERNEL_KW)
    rfp_kw = fk.kernel_gaussian_symm_rfp(
        coords_list=coords_a, Z_list=zs_a, sigma=SIGMA, **KERNEL_KW
    )
    np.testing.assert_array_equal(rfp_kw, rfp_pos)

    g_pos = fk.kernel_gaussian_gradient(WATER_COORDS, WATER_Z, x2, n2, nn2, SIGMA, **KERNEL_KW)
    g_kw = fk.kernel_gaussian_gradient(
        coords_A=WATER_COORDS,
        Z_A=WATER_Z,
        X2=x2,
        N2=n2,
        NN2=nn2,
        sigma=SIGMA,
        **KERNEL_KW,
    )
    np.testing.assert_array_equal(g_kw, g_pos)
