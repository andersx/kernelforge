import numpy as np
from kernelforge import _kernels 

import numpy as np
import pytest


def test_small_system():
    K = np.array([[4.0, 2.0, 0.6],
                  [2.0, 5.0, 1.5],
                  [0.6, 1.5, 3.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    alpha = _kernels.solve_cholesky(K, y)

    # Verify K_original @ alpha ≈ y
    K_original = np.array([[4.0, 2.0, 0.6],
                           [2.0, 5.0, 1.5],
                           [0.6, 1.5, 3.0]], dtype=np.float64)
    np.testing.assert_allclose(K_original @ alpha, y, rtol=1e-12, atol=1e-14)

    # y must be preserved
    np.testing.assert_allclose(y, [1.0, 2.0, 3.0])


def test_identity_matrix():
    n = 5
    K = np.eye(n, dtype=np.float64)
    y = np.arange(1, n + 1, dtype=np.float64)

    alpha = _kernels.solve_cholesky(K, y)
    np.testing.assert_allclose(alpha, y)


def test_random_pd_matrix():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((6, 6))
    K = A.T @ A + np.eye(6) * 1e-6  # make SPD
    y = rng.standard_normal(6)

    alpha = _kernels.solve_cholesky(K, y)

    np.testing.assert_allclose((A.T @ A + np.eye(6) * 1e-6) @ alpha, y,
                               rtol=1e-10, atol=1e-12)


def test_non_square_matrix():
    with pytest.raises(RuntimeError):
        K = np.ones((3, 2), dtype=np.float64)
        y = np.ones(3, dtype=np.float64)
        _kernels.solve_cholesky(K, y)


def test_mismatched_size():
    with pytest.raises(RuntimeError):
        K = np.eye(3, dtype=np.float64)
        y = np.ones(4, dtype=np.float64)
        _kernels.solve_cholesky(K, y)

