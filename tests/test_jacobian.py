import numpy as np
import pytest

# adjust import if your module name/path differs
from kernelforge import _kernels


def _strict_upper_pairs(N):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def test_shapes_and_basic_values():
    rng = np.random.default_rng(0)

    N1, N2 = 3, 4
    N = 3
    D = 3 * N
    M = N * (N - 1) // 2  # typical inverse-distance feature length

    X1 = rng.normal(size=(N1, M))
    dX1 = rng.normal(size=(N1, M, D))
    X2 = rng.normal(size=(N2, M))
    sigma = 0.7

    K = _kernels.gaussian_jacobian_batch(X1, dX1, X2, sigma)
    assert K.shape == (N1 * D, N2)

    # Check that columns are not all zeros and depend on inputs
    assert np.isfinite(K).all()
    assert not np.allclose(K, 0.0)


def test_formula_matches_numpy_reference():
    rng = np.random.default_rng(1)

    N1, N2 = 2, 3
    N = 2
    D = 3 * N
    M = 3  # any M works; choose small for clarity

    X1 = rng.normal(size=(N1, M))
    dX1 = rng.normal(size=(N1, M, D))
    X2 = rng.normal(size=(N2, M))
    sigma = 0.5
    inv_s2 = 1.0 / (sigma * sigma)

    K = _kernels.gaussian_jacobian_batch(X1, dX1, X2, sigma)

    # numpy reference (same math): for each (a,b),
    # w_ab = (k_ab/sigma^2) * (X2[b]-X1[a])
    # K[a*D:(a+1)*D, b] = dX1[a].T @ w_ab
    K_ref = np.zeros_like(K)
    for a in range(N1):
        x1 = X1[a]
        J = dX1[a]  # (M, D)
        for b in range(N2):
            diff = X2[b] - x1
            sq = float(diff @ diff)
            k_ab = np.exp(-0.5 * inv_s2 * sq)
            w_ab = (k_ab * inv_s2) * diff
            K_ref[a * D : (a + 1) * D, b] = J.T @ w_ab

    np.testing.assert_allclose(K, K_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("N1,N2,M,N", [(1, 2, 5, 3), (2, 1, 6, 2)])
def test_finite_difference_linearized_feature_model(N1, N2, M, N):
    """
    Finite-difference check using a *linearized* feature->coordinate model:
      x1_a(r) = x1_a0 + J_a @ r
    so k_ab(r) = exp(-||x1_a(r) - x2_b||^2 / (2*sigma^2)).
    The analytical gradient at r=0 equals the column in K for (a,b).
    """
    rng = np.random.default_rng(2)
    D = 3 * N
    X1 = rng.normal(size=(N1, M))
    dX1 = rng.normal(size=(N1, M, D))
    X2 = rng.normal(size=(N2, M))
    sigma = 0.8
    inv_s2 = 1.0 / (sigma * sigma)

    K = _kernels.gaussian_jacobian_batch(X1, dX1, X2, sigma)

    # Test a few random (a,b) pairs and a few coordinates
    pairs_to_test = [(a % N1, b % N2) for a, b in [(0, 0), (0, 1), (N1 - 1, N2 - 1)]]
    coords_to_test = [0, D // 2, D - 1] if D >= 3 else list(range(D))
    h = 1e-6
    atol = 5e-6
    rtol = 2e-4

    for a, b in pairs_to_test:
        x1 = X1[a].copy()
        J = dX1[a]  # (M, D)
        x2 = X2[b]

        # Define k_ab(r) with linearized feature model around r=0
        def k_of_r(r_vec):
            x1_r = x1 + J @ r_vec  # (M,)
            diff = x1_r - x2
            return np.exp(-0.5 * inv_s2 * float(diff @ diff))

        # Numerical derivative per-coordinate via central difference at r=0
        num = np.zeros(D)
        r0 = np.zeros(D)
        for d in coords_to_test:
            rp = r0.copy()
            rm = r0.copy()
            rp[d] += h
            rm[d] -= h
            kp = k_of_r(rp)
            km = k_of_r(rm)
            num[d] = (kp - km) / (2 * h)

        # Analytical from K block
        ana = K[a * D : (a + 1) * D, b]

        np.testing.assert_allclose(ana[coords_to_test], num[coords_to_test], rtol=rtol, atol=atol)


def test_bad_sigma_raises():
    rng = np.random.default_rng(3)

    N1, N2, M, D = 1, 1, 4, 6
    X1 = rng.normal(size=(N1, M))
    dX1 = rng.normal(size=(N1, M, D))
    X2 = rng.normal(size=(N2, M))

    with pytest.raises(Exception):
        _ = _kernels.gaussian_jacobian_batch(X1, dX1, X2, 0.0)

    with pytest.raises(Exception):
        _ = _kernels.gaussian_jacobian_batch(X1, dX1, X2, -1.0)


def test_input_shape_mismatch_errors():
    rng = np.random.default_rng(4)

    N1, N2, M, D = 2, 2, 5, 9
    X1 = rng.normal(size=(N1, M))
    dX1 = rng.normal(size=(N1, M, D))
    X2_ok = rng.normal(size=(N2, M))
    X2_bad = rng.normal(size=(N2, M + 1))

    # X2 second dim must equal M
    with pytest.raises(Exception):
        _ = _kernels.gaussian_jacobian_batch(X1, dX1, X2_bad, 0.9)

    # dX1 M dimension must match X1 M
    with pytest.raises(Exception):
        _ = _kernels.gaussian_jacobian_batch(X1, dX1[:, :-1, :], X2_ok, 0.9)

    # dX1 N1 dimension must match X1 N1
    with pytest.raises(Exception):
        _ = _kernels.gaussian_jacobian_batch(X1[:-1], dX1, X2_ok, 0.9)
