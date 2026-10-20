"""FCHL18 periodic-table alchemy for gradient / Jacobian / Hessian kernels."""

from __future__ import annotations

import numpy as np

from kernelforge import fchl18_kernel, fchl18_repr

KP = {
    "sigma": 2.5,
    "two_body_scaling": float(np.sqrt(8)),
    "three_body_scaling": 1.6,
    "two_body_width": 0.2,
    "three_body_width": float(np.pi),
    "two_body_power": 4.0,
    "three_body_power": 2.0,
    "cut_start": 0.5,
    "cut_distance": 5.0,
    "fourier_order": 1,
    "use_atm": True,
}
ALCH = {"use_alchemy": True, "alchemy_period_width": 1.6, "alchemy_group_width": 1.6}
WATER = np.array([[0.0, 0.0, 0.119], [0.0, 0.757, -0.477], [0.0, -0.757, -0.477]], dtype=np.float64)
WATER_Z = np.array([8, 1, 1], dtype=np.int32)
# Asymmetric diatomic; O/F coupling makes alchemy visible.
HF = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]], dtype=np.float64)
HF_Z = np.array([9, 1], dtype=np.int32)

MAX_SIZE = 4


def _repr(coords: list[np.ndarray], z: list[np.ndarray]) -> tuple:
    return fchl18_repr.generate(coords, z, max_size=MAX_SIZE, cut_distance=KP["cut_distance"])


def _hf_repr() -> tuple:
    return _repr([HF], [HF_Z])


def test_gradient_alchemy_off_matches_default() -> None:
    x2, n2, nn2 = _hf_repr()
    g0 = fchl18_kernel.kernel_gaussian_gradient(WATER, WATER_Z, x2, n2, nn2, **KP)
    g1 = fchl18_kernel.kernel_gaussian_gradient(
        WATER, WATER_Z, x2, n2, nn2, **KP, use_alchemy=False
    )
    assert np.array_equal(g0, g1)


def test_gradient_tiny_widths_near_off() -> None:
    x2, n2, nn2 = _hf_repr()
    g_off = fchl18_kernel.kernel_gaussian_gradient(WATER, WATER_Z, x2, n2, nn2, **KP)
    g_tiny = fchl18_kernel.kernel_gaussian_gradient(
        WATER,
        WATER_Z,
        x2,
        n2,
        nn2,
        **KP,
        use_alchemy=True,
        alchemy_period_width=1e-4,
        alchemy_group_width=1e-4,
    )
    np.testing.assert_allclose(g_tiny, g_off, rtol=1e-5, atol=1e-8)


def test_gradient_alchemy_changes_result() -> None:
    x2, n2, nn2 = _hf_repr()
    g_off = fchl18_kernel.kernel_gaussian_gradient(WATER, WATER_Z, x2, n2, nn2, **KP)
    g_on = fchl18_kernel.kernel_gaussian_gradient(WATER, WATER_Z, x2, n2, nn2, **KP, **ALCH)
    assert g_on.shape == g_off.shape
    assert not np.allclose(g_on, g_off)


def test_gradient_alchemy_fd_vs_symm() -> None:
    """Central FD of K_symm([water(R), HF])[0, 1] (alchemy on) vs analytical gradient."""
    x2, n2, nn2 = _hf_repr()
    g = fchl18_kernel.kernel_gaussian_gradient(WATER, WATER_Z, x2, n2, nn2, **KP, **ALCH)
    assert g.shape == (3, 3, 1)

    eps = 1e-4
    g_fd = np.zeros_like(g)
    for a in range(3):
        for mu in range(3):
            vals = []
            for sign in (+1.0, -1.0):
                w = WATER.copy()
                w[a, mu] += sign * eps
                X, N, NN = _repr([w, HF], [WATER_Z, HF_Z])
                K = fchl18_kernel.kernel_gaussian_symm(X, N, NN, **KP, **ALCH)
                vals.append(K[0, 1])
            g_fd[a, mu, 0] = (vals[0] - vals[1]) / (2.0 * eps)

    np.testing.assert_allclose(g, g_fd, rtol=1e-4, atol=1e-7)


def test_jacobian_alchemy_matches_gradient() -> None:
    x2, n2, nn2 = _hf_repr()
    g = fchl18_kernel.kernel_gaussian_gradient(WATER, WATER_Z, x2, n2, nn2, **KP, **ALCH)
    J = fchl18_kernel.kernel_gaussian_jacobian([WATER], [WATER_Z], x2, n2, nn2, **KP, **ALCH)
    np.testing.assert_allclose(J, g.reshape(9, 1), rtol=1e-10, atol=1e-12)


def test_jacobian_alchemy_off_matches_default() -> None:
    x2, n2, nn2 = _hf_repr()
    J0 = fchl18_kernel.kernel_gaussian_jacobian([WATER], [WATER_Z], x2, n2, nn2, **KP)
    J1 = fchl18_kernel.kernel_gaussian_jacobian(
        [WATER], [WATER_Z], x2, n2, nn2, **KP, use_alchemy=False
    )
    assert np.array_equal(J0, J1)


def test_hessian_alchemy_fd_of_jacobian() -> None:
    """FD (w.r.t. HF coords) of the alchemy Jacobian vs alchemy Hessian, water vs HF."""
    H = fchl18_kernel.kernel_gaussian_hessian([WATER], [WATER_Z], [HF], [HF_Z], **KP, **ALCH)
    assert H.shape == (9, 6)

    eps = 1e-4
    H_fd = np.zeros_like(H)
    for b in range(2):
        for nu in range(3):
            cols = []
            for sign in (+1.0, -1.0):
                hf = HF.copy()
                hf[b, nu] += sign * eps
                x2, n2, nn2 = _repr([hf], [HF_Z])
                J = fchl18_kernel.kernel_gaussian_jacobian(
                    [WATER], [WATER_Z], x2, n2, nn2, **KP, **ALCH
                )
                cols.append(J[:, 0])
            H_fd[:, b * 3 + nu] = (cols[0] - cols[1]) / (2.0 * eps)

    np.testing.assert_allclose(H, H_fd, rtol=1e-3, atol=1e-6)


def test_hessian_alchemy_off_matches_default_and_changes_when_on() -> None:
    H0 = fchl18_kernel.kernel_gaussian_hessian([WATER], [WATER_Z], [HF], [HF_Z], **KP)
    H1 = fchl18_kernel.kernel_gaussian_hessian(
        [WATER], [WATER_Z], [HF], [HF_Z], **KP, use_alchemy=False
    )
    H_on = fchl18_kernel.kernel_gaussian_hessian([WATER], [WATER_Z], [HF], [HF_Z], **KP, **ALCH)
    assert np.array_equal(H0, H1)
    assert not np.allclose(H0, H_on)


def test_full_kernel_alchemy_blocks_consistent() -> None:
    """kernel_gaussian_full(alchemy) blocks match the individual alchemy kernels."""
    x_a, n_a, nn_a = _repr([WATER], [WATER_Z])
    x_b, n_b, nn_b = _hf_repr()
    K = fchl18_kernel.kernel_gaussian_full([WATER], [WATER_Z], [HF], [HF_Z], **KP, **ALCH)
    K_s = fchl18_kernel.kernel_gaussian(x_a, x_b, n_a, n_b, nn_a, nn_b, **KP, **ALCH)
    J = fchl18_kernel.kernel_gaussian_jacobian([WATER], [WATER_Z], x_b, n_b, nn_b, **KP, **ALCH)
    H = fchl18_kernel.kernel_gaussian_hessian([WATER], [WATER_Z], [HF], [HF_Z], **KP, **ALCH)
    np.testing.assert_allclose(K[0:1, 0:1], K_s, rtol=1e-10)
    np.testing.assert_allclose(K[1:, 0:1], J, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(K[1:, 1:], H, rtol=1e-10, atol=1e-12)


def test_jacobian_t_alchemy_is_transpose_of_jacobian() -> None:
    """With alchemy on, jacobian(A,A) == jacobian_t(A,A).T."""
    coords = [WATER, HF]
    zs = [WATER_Z, HF_Z]
    x, n, nn = _repr(coords, zs)
    J = fchl18_kernel.kernel_gaussian_jacobian(coords, zs, x, n, nn, **KP, **ALCH)
    Jt = fchl18_kernel.kernel_gaussian_jacobian_t(coords, zs, x, n, nn, **KP, **ALCH)
    np.testing.assert_allclose(J, Jt.T, rtol=1e-10, atol=1e-12)


def test_rfp_alchemy_unpacks_to_dense() -> None:
    """Scalar / Hessian / full symm RFP with alchemy unpack to the dense symm kernels."""
    from kernelforge import kernelmath

    coords = [WATER, HF]
    zs = [WATER_Z, HF_Z]
    x, n, nn = _repr(coords, zs)

    # Scalar RFP
    K = fchl18_kernel.kernel_gaussian_symm(x, n, nn, **KP, **ALCH)
    rfp = fchl18_kernel.kernel_gaussian_symm_rfp(coords, zs, **KP, **ALCH)
    nm = len(coords)
    np.testing.assert_allclose(
        kernelmath.rfp_to_full(rfp, nm, uplo="L", transr="N"), K, rtol=1e-12, atol=1e-14
    )

    # Hessian RFP
    H = fchl18_kernel.kernel_gaussian_hessian_symm(coords, zs, **KP, **ALCH)
    H_rfp = fchl18_kernel.kernel_gaussian_hessian_symm_rfp(coords, zs, **KP, **ALCH)
    D = H.shape[0]
    np.testing.assert_allclose(
        kernelmath.rfp_to_full(H_rfp, D, uplo="L", transr="N"), H, rtol=1e-12, atol=1e-14
    )

    # Full EF RFP
    Kf = fchl18_kernel.kernel_gaussian_full_symm(coords, zs, **KP, **ALCH)
    Kf_rfp = fchl18_kernel.kernel_gaussian_full_symm_rfp(coords, zs, **KP, **ALCH)
    big = Kf.shape[0]
    np.testing.assert_allclose(
        kernelmath.rfp_to_full(Kf_rfp, big, uplo="L", transr="N"), Kf, rtol=1e-12, atol=1e-14
    )
