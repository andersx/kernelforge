import numpy as np
import pytest

from kernelforge import _kernels  # the module name from the pybind shim


def ref_block(x1, J1, x2, J2, sigma):
    """
    Reference NumPy implementation for one (a,b) block:
    H = (k/s^2) * J1^T J2 - (k/s^4) * (J1^T d) (J2^T d)^T
    with d = x1 - x2, k = exp(-||d||^2/(2 s^2))
    Shapes: x1(M,), J1(M,D1), x2(M,), J2(M,D2)
    Returns: (D1,D2)
    """
    s2 = sigma * sigma
    d = x1 - x2
    k = np.exp(-0.5 * (d @ d) / s2)
    term1 = (k / s2) * (J1.T @ J2)
    v1 = J1.T @ d
    v2 = J2.T @ d
    term2 = (k / (s2 * s2)) * (np.outer(v1, v2))
    return term1 - term2


def assemble_ref_full(X1, dX1, X2, dX2, sigma):
    """
    Assemble full ((N1*D1) x (N2*D2)) Hessian by looping in Python
    and calling the closed-form block above.
    """
    N1, M = X1.shape
    N2, _ = X2.shape
    D1 = dX1.shape[2]
    D2 = dX2.shape[2]
    H = np.zeros((N1 * D1, N2 * D2), dtype=np.float64)
    for a in range(N1):
        x1 = X1[a]
        J1 = dX1[a]  # (M,D1)
        for b in range(N2):
            x2 = X2[b]
            J2 = dX2[b]  # (M,D2)
            H_block = ref_block(x1, J1, x2, J2, sigma)
            ra = a * D1
            cb = b * D2
            H[ra : ra + D1, cb : cb + D2] = H_block
    return H


@pytest.mark.parametrize(
    "N1,N2,M,D1,D2",
    [
        (2, 3, 5, 4, 3),
        (1, 1, 6, 5, 5),
    ],
)
def test_shapes_and_values(N1, N2, M, D1, D2):
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(N1, M))
    X2 = rng.normal(size=(N2, M))
    dX1 = rng.normal(size=(N1, M, D1))
    dX2 = rng.normal(size=(N2, M, D2))
    sigma = 0.7

    H = _kernels.rbf_hessian_full_tiled_gemm(X1, dX1, X2, dX2, sigma)
    assert H.shape == (N1 * D1, N2 * D2)
    assert np.isfinite(H).all()

    H_ref = assemble_ref_full(X1, dX1, X2, dX2, sigma)
    np.testing.assert_allclose(H, H_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("tile_B", [1, 2, 4, 7, 0])  # 0 => auto
def test_various_tile_sizes(tile_B):
    rng = np.random.default_rng(1)
    N1, N2, M, D1, D2 = 2, 5, 8, 3, 4
    X1 = rng.normal(size=(N1, M))
    X2 = rng.normal(size=(N2, M))
    dX1 = rng.normal(size=(N1, M, D1))
    dX2 = rng.normal(size=(N2, M, D2))
    sigma = 0.9

    # Call with explicit tile size (or None)
    if tile_B == 0:
        H = _kernels.rbf_hessian_full_tiled_gemm(X1, dX1, X2, dX2, sigma, None)
    else:
        H = _kernels.rbf_hessian_full_tiled_gemm(X1, dX1, X2, dX2, sigma, tile_B)

    H_ref = assemble_ref_full(X1, dX1, X2, dX2, sigma)
    np.testing.assert_allclose(H, H_ref, rtol=1e-12, atol=1e-12)


def test_bad_sigma_raises():
    rng = np.random.default_rng(2)
    X1 = rng.normal(size=(1, 3))
    X2 = rng.normal(size=(1, 3))
    dX1 = rng.normal(size=(1, 3, 2))
    dX2 = rng.normal(size=(1, 3, 2))

    with pytest.raises(Exception):
        _ = _kernels.rbf_hessian_full_tiled_gemm(X1, dX1, X2, dX2, 0.0)
    with pytest.raises(Exception):
        _ = _kernels.rbf_hessian_full_tiled_gemm(X1, dX1, X2, dX2, -1.0)


def test_shape_mismatch_raises():
    rng = np.random.default_rng(3)
    X1 = rng.normal(size=(2, 4))
    X2 = rng.normal(size=(3, 4))
    dX1 = rng.normal(size=(2, 4, 3))
    dX2 = rng.normal(size=(3, 4, 5))

    # X2 second dim must match X1 second dim
    with pytest.raises(Exception):
        _ = _kernels.rbf_hessian_full_tiled_gemm(X1, dX1, X2[:, :3], dX2, 0.8)

    # dX1 first/second dims must match X1
    with pytest.raises(Exception):
        _ = _kernels.rbf_hessian_full_tiled_gemm(X1[1:], dX1, X2, dX2, 0.8)
    with pytest.raises(Exception):
        _ = _kernels.rbf_hessian_full_tiled_gemm(X1, dX1[:, :-1, :], X2, dX2, 0.8)

    # dX2 first/second dims must match X2
    with pytest.raises(Exception):
        _ = _kernels.rbf_hessian_full_tiled_gemm(X1, dX1, X2, dX2[:-1], 0.8)
    with pytest.raises(Exception):
        _ = _kernels.rbf_hessian_full_tiled_gemm(X1, dX1, X2, dX2[:, :-1, :], 0.8)
