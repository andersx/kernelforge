"""FCHL18 periodic-table alchemy (scalar kernels)."""

from __future__ import annotations

import numpy as np
import pytest

from kernelforge import fchl18_kernel, fchl18_repr

KP = dict(
    sigma=2.5,
    two_body_scaling=float(np.sqrt(8)),
    three_body_scaling=1.6,
    two_body_width=0.2,
    three_body_width=float(np.pi),
    two_body_power=4.0,
    three_body_power=2.0,
    cut_start=0.5,
    cut_distance=5.0,
    fourier_order=1,
    use_atm=True,
)


def _water_and_hf(max_size: int = 4) -> tuple:
    """H2O + HF — mixed species so alchemy can couple O/F and H."""
    coords = [
        np.array(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=np.float64
        ),
        np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]], dtype=np.float64),
    ]
    z = [
        np.array([8, 1, 1], dtype=np.int32),
        np.array([9, 1], dtype=np.int32),
    ]
    X, N, NN = fchl18_repr.generate(coords, z, max_size=max_size, cut_distance=5.0)
    return X, N, NN, coords, z


def test_alchemy_off_matches_default() -> None:
    X, N, NN, _, _ = _water_and_hf()
    K0 = fchl18_kernel.kernel_gaussian_symm(X, N, NN, **KP)
    K1 = fchl18_kernel.kernel_gaussian_symm(X, N, NN, **KP, use_alchemy=False)
    assert np.allclose(K0, K1)


def test_tiny_widths_near_identity() -> None:
    """Very small alchemy widths ≈ alchemy off (pd ≈ δ_ij for present elements)."""
    X, N, NN, _, _ = _water_and_hf()
    K_off = fchl18_kernel.kernel_gaussian_symm(X, N, NN, **KP, use_alchemy=False)
    K_tiny = fchl18_kernel.kernel_gaussian_symm(
        X,
        N,
        NN,
        **KP,
        use_alchemy=True,
        alchemy_period_width=1e-4,
        alchemy_group_width=1e-4,
    )
    assert np.allclose(K_off, K_tiny, rtol=1e-5, atol=1e-6)


def test_alchemy_changes_kernel_and_couples_species() -> None:
    X, N, NN, _, _ = _water_and_hf()
    K_off = fchl18_kernel.kernel_gaussian_symm(X, N, NN, **KP, use_alchemy=False)
    K_on = fchl18_kernel.kernel_gaussian_symm(
        X,
        N,
        NN,
        **KP,
        use_alchemy=True,
        alchemy_period_width=1.6,
        alchemy_group_width=1.6,
    )
    assert K_on.shape == K_off.shape
    assert not np.allclose(K_off, K_on)
    # Off-diagonal should grow: O↔F alchemical coupling adds positive kernel mass.
    assert K_on[0, 1] > K_off[0, 1]


def test_asymmetric_alchemy_matches_symm_block() -> None:
    X, N, NN, _, _ = _water_and_hf()
    Kw = dict(
        **KP,
        use_alchemy=True,
        alchemy_period_width=1.6,
        alchemy_group_width=1.6,
    )
    Ks = fchl18_kernel.kernel_gaussian_symm(X, N, NN, **Kw)
    Ka = fchl18_kernel.kernel_gaussian(X, X, N, N, NN, NN, **Kw)
    assert np.allclose(Ks, Ka)


@pytest.mark.integration
def test_qmllib_periodic_table_parity() -> None:
    """Parity vs qmllib alchemy='periodic-table' when FCHL nxt is installed."""
    pytest.importorskip("qmllib")
    try:
        from qmllib.nxt.fchl import fchl_scalar_kernels as fsk
        from qmllib.nxt.fchl import generate_fchl18
    except ImportError:
        pytest.skip("qmllib FCHL nxt not available")

    _, _, _, coords, z = _water_and_hf(max_size=23)
    max_size = 23
    X_qml = np.array(
        [generate_fchl18(zi, ci, max_size=max_size, cut_distance=5.0) for zi, ci in zip(z, coords)]
    )
    K_qml = fsk.get_local_symmetric_kernels(
        X_qml,
        alchemy="periodic-table",
        alchemy_period_width=1.6,
        alchemy_group_width=1.6,
        two_body_scaling=float(np.sqrt(8)),
        three_body_scaling=1.6,
        two_body_width=0.2,
        three_body_width=float(np.pi),
        two_body_power=4.0,
        three_body_power=2.0,
        cut_start=0.5,
        cut_distance=5.0,
        fourier_order=1,
        kernel="gaussian",
        kernel_args={"sigma": [2.5]},
    )[0]

    X, N, NN = fchl18_repr.generate(coords, z, max_size=max_size, cut_distance=5.0)
    K_kf = fchl18_kernel.kernel_gaussian_symm(
        X,
        N,
        NN,
        **KP,
        use_alchemy=True,
        alchemy_period_width=1.6,
        alchemy_group_width=1.6,
    )
    np.testing.assert_allclose(K_kf, K_qml, rtol=1e-5, atol=1e-6)
