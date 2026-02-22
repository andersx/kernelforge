from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

# adjust import if your module name/path differs
from kernelforge import global_kernels as _kernels


def _strict_upper_pairs(N: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def test_shapes_and_basic_values() -> None:
    rng = np.random.default_rng(0)

    N1, N2 = 3, 4
    N = 3
    D = 3 * N
    M = N * (N - 1) // 2  # typical inverse-distance feature length

    X1 = rng.normal(size=(N1, M))
    dX1 = rng.normal(size=(N1, M, D))
    X2 = rng.normal(size=(N2, M))
    sigma = 0.7

    K = _kernels.kernel_gaussian_jacobian(X1, dX1, X2, sigma)
    assert K.shape == (N1 * D, N2)

    # Check that columns are not all zeros and depend on inputs
    assert np.isfinite(K).all()
    assert not np.allclose(K, 0.0)


def test_formula_matches_numpy_reference() -> None:
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

    K = _kernels.kernel_gaussian_jacobian(X1, dX1, X2, sigma)

    # NumPy reference implementation
    # For each pair (a,b): compute weight vector w_ab then project via Jacobian transpose
    # Weight: w_ab = (kernel_value / sigma²) * displacement_vector
    # Result: Jacobian_transpose times w_ab gives the kernel derivative block
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


@pytest.mark.parametrize(("N1", "N2", "M", "N"), [(1, 2, 5, 3), (2, 1, 6, 2)])
def test_finite_difference_linearized_feature_model(N1: int, N2: int, M: int, N: int) -> None:
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

    K = _kernels.kernel_gaussian_jacobian(X1, dX1, X2, sigma)

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
        # Bind loop variables as default arguments to avoid closure issue
        def k_of_r(
            r_vec: NDArray[np.float64],
            x1: NDArray[np.float64] = x1,
            J: NDArray[np.float64] = J,
            x2: NDArray[np.float64] = x2,
        ) -> np.floating[Any]:
            x1_r = x1 + J @ r_vec  # (M,)
            diff = x1_r - x2
            result: np.floating[Any] = np.exp(-0.5 * inv_s2 * float(diff @ diff))
            return result

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


def test_bad_sigma_raises() -> None:
    rng = np.random.default_rng(3)

    N1, N2, M, D = 1, 1, 4, 6
    X1 = rng.normal(size=(N1, M))
    dX1 = rng.normal(size=(N1, M, D))
    X2 = rng.normal(size=(N2, M))

    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian(X1, dX1, X2, 0.0)

    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian(X1, dX1, X2, -1.0)


def test_input_shape_mismatch_errors() -> None:
    rng = np.random.default_rng(4)

    N1, N2, M, D = 2, 2, 5, 9
    X1 = rng.normal(size=(N1, M))
    dX1 = rng.normal(size=(N1, M, D))
    X2_ok = rng.normal(size=(N2, M))
    X2_bad = rng.normal(size=(N2, M + 1))

    # X2 second dim must equal M
    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian(X1, dX1, X2_bad, 0.9)

    # dX1 M dimension must match X1 M
    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian(X1, dX1[:, :-1, :], X2_ok, 0.9)

    # dX1 N1 dimension must match X1 N1
    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian(X1[:-1], dX1, X2_ok, 0.9)


# ============================================================================
# Tests for kernel_gaussian_jacobian_t (transposed version)
# ============================================================================


def test_jacobian_t_shapes() -> None:
    """Test that jacobian_t produces correct output shape."""
    rng = np.random.default_rng(10)

    N1, N2 = 4, 5
    N = 3
    D = 3 * N
    M = N * (N - 1) // 2

    X1 = rng.normal(size=(N1, M))
    X2 = rng.normal(size=(N2, M))
    dX2 = rng.normal(size=(N2, M, D))
    sigma = 0.7

    K_t = _kernels.kernel_gaussian_jacobian_t(X1, X2, dX2, sigma)
    assert K_t.shape == (N1, N2 * D)

    # Check that values are finite and non-zero
    assert np.isfinite(K_t).all()
    assert not np.allclose(K_t, 0.0)


def test_jacobian_t_equals_jacobian_transposed() -> None:
    """Test jacobian_t(X2, X1, dX1, s) == jacobian(X1, dX1, X2, s).T (s=sigma)."""
    rng = np.random.default_rng(11)

    N1, N2 = 3, 4
    N = 2
    D = 3 * N
    M = 5

    X1 = rng.normal(size=(N1, M))
    dX1 = rng.normal(size=(N1, M, D))
    X2 = rng.normal(size=(N2, M))
    dX2 = rng.normal(size=(N2, M, D))
    sigma = 0.8

    # Compute K: (N1*D, N2) with Jacobians on X1 side
    K = _kernels.kernel_gaussian_jacobian(X1, dX1, X2, sigma)

    # Compute K_t: (N2, N1*D) with Jacobians on X2 side (swapping X1<->X2, dX1<->dX2)
    K_t = _kernels.kernel_gaussian_jacobian_t(X2, X1, dX1, sigma)

    # K_t should equal K.T
    assert K.shape == (N1 * D, N2)
    assert K_t.shape == (N2, N1 * D)
    np.testing.assert_allclose(K_t, K.T, rtol=1e-12, atol=1e-12)


def test_jacobian_t_formula_matches_numpy_reference() -> None:
    """Test that jacobian_t matches the direct NumPy implementation."""
    rng = np.random.default_rng(12)

    N1, N2 = 2, 3
    N = 2
    D = 3 * N
    M = 4

    X1 = rng.normal(size=(N1, M))
    X2 = rng.normal(size=(N2, M))
    dX2 = rng.normal(size=(N2, M, D))
    sigma = 0.6
    inv_s2 = 1.0 / (sigma * sigma)

    K_t = _kernels.kernel_gaussian_jacobian_t(X1, X2, dX2, sigma)

    # NumPy reference: K_t[a, b*D:(b+1)*D] = w_ab @ J2[b]
    # where w_ab = (k_ab / σ²) * (x1[a] - x2[b])
    K_ref = np.zeros_like(K_t)
    for a in range(N1):
        x1 = X1[a]
        for b in range(N2):
            x2 = X2[b]
            J2 = dX2[b]  # (M, D)
            diff = x1 - x2
            sq = float(diff @ diff)
            k_ab = np.exp(-0.5 * inv_s2 * sq)
            w_ab = (k_ab * inv_s2) * diff
            K_ref[a, b * D : (b + 1) * D] = w_ab @ J2

    np.testing.assert_allclose(K_t, K_ref, rtol=1e-12, atol=1e-12)


def test_jacobian_t_symmetry_when_same_data() -> None:
    """Test that when X1=X2 and dX1=dX2, K_t still equals K.T."""
    rng = np.random.default_rng(13)

    N = 3
    D = 9
    M = 5

    X = rng.normal(size=(N, M))
    dX = rng.normal(size=(N, M, D))
    sigma = 0.9

    # K: (N*D, N) with Jacobians on query side
    K = _kernels.kernel_gaussian_jacobian(X, dX, X, sigma)

    # K_t: (N, N*D) with Jacobians on reference side
    K_t = _kernels.kernel_gaussian_jacobian_t(X, X, dX, sigma)

    # Should satisfy K_t == K.T
    np.testing.assert_allclose(K_t, K.T, rtol=1e-12, atol=1e-12)


def test_jacobian_t_bad_sigma() -> None:
    """Test that jacobian_t raises error for invalid sigma."""
    rng = np.random.default_rng(14)

    N1, N2, M, D = 2, 2, 4, 6
    X1 = rng.normal(size=(N1, M))
    X2 = rng.normal(size=(N2, M))
    dX2 = rng.normal(size=(N2, M, D))

    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian_t(X1, X2, dX2, 0.0)

    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian_t(X1, X2, dX2, -1.0)


def test_jacobian_t_shape_mismatch_errors() -> None:
    """Test that jacobian_t raises errors for shape mismatches."""
    rng = np.random.default_rng(15)

    N1, N2, M, D = 2, 3, 5, 9
    X1 = rng.normal(size=(N1, M))
    X2_ok = rng.normal(size=(N2, M))
    X2_bad = rng.normal(size=(N2, M + 1))
    dX2 = rng.normal(size=(N2, M, D))

    # X2 second dim must equal M
    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian_t(X1, X2_bad, dX2, 0.9)

    # dX2 M dimension must match X2 M
    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian_t(X1, X2_ok, dX2[:, :-1, :], 0.9)

    # dX2 N2 dimension must match X2 N2
    with pytest.raises(Exception, match=r".*"):
        _ = _kernels.kernel_gaussian_jacobian_t(X1, X2_ok[:-1], dX2, 0.9)
