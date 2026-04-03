"""Tests for the J^T·α trick: compute_alpha_desc and hessian_matvec."""

import numpy as np
import pytest
from numpy.typing import NDArray

from kernelforge import global_kernels as _kernels


def ref_alpha_desc(dX: NDArray[np.float64], alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reference: alpha_desc[m, k] = sum_d dX[m, d, k] * alpha[m, d]."""
    result: NDArray[np.float64] = np.einsum("ndm,nd->nm", dX, alpha)
    return result


def ref_hessian_matvec(
    X_q: NDArray[np.float64],
    dX_q: NDArray[np.float64],
    X_t: NDArray[np.float64],
    dX_t: NDArray[np.float64],
    alpha_flat: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """Reference: build full Hessian, multiply by alpha, reshape."""
    H = _kernels.kernel_gaussian_hessian(X_q, dX_q, X_t, dX_t, sigma)
    result: NDArray[np.float64] = (H @ alpha_flat).reshape(X_q.shape[0], dX_q.shape[1])
    return result


class TestComputeAlphaDesc:
    """Tests for kernel_gaussian_compute_alpha_desc."""

    @pytest.mark.parametrize(
        ("N", "D", "M"),
        [
            (1, 1, 1),
            (3, 4, 5),
            (10, 27, 36),
        ],
    )
    def test_correctness(self, N: int, D: int, M: int) -> None:
        rng = np.random.default_rng(42)
        dX = rng.normal(size=(N, D, M))
        alpha = rng.normal(size=(N, D))

        result = _kernels.kernel_gaussian_compute_alpha_desc(dX, alpha)
        expected = ref_alpha_desc(dX, alpha)

        assert result.shape == (N, M)
        np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-15)

    def test_output_dtype(self) -> None:
        rng = np.random.default_rng(0)
        dX = rng.normal(size=(2, 3, 4))
        alpha = rng.normal(size=(2, 3))
        result = _kernels.kernel_gaussian_compute_alpha_desc(dX, alpha)
        assert result.dtype == np.float64


class TestHessianMatvec:
    """Tests for kernel_gaussian_hessian_matvec."""

    @pytest.mark.parametrize(
        ("N_q", "N_t", "M", "D"),
        [
            (1, 1, 3, 2),
            (2, 3, 5, 4),
            (1, 10, 5, 3),
            (3, 15, 6, 4),
            (5, 5, 8, 6),
        ],
    )
    def test_correctness(self, N_q: int, N_t: int, M: int, D: int) -> None:
        """Compare matvec output against full Hessian @ alpha."""
        rng = np.random.default_rng(123)
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        sigma = 0.7

        # Random force coefficients
        alpha = rng.normal(size=(N_t, D))
        alpha_flat = alpha.ravel()

        # Pre-compute alpha_desc
        alpha_desc = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha)

        # Reference via full Hessian matrix
        F_ref = ref_hessian_matvec(X_q, dX_q, X_t, dX_t, alpha_flat, sigma)

        # Fast matvec
        F_fast = _kernels.kernel_gaussian_hessian_matvec(X_q, dX_q, X_t, alpha_desc, sigma)

        assert F_fast.shape == (N_q, D)
        np.testing.assert_allclose(F_fast, F_ref, rtol=1e-10, atol=1e-12)

    def test_single_query(self) -> None:
        """N_q=1 is the common prediction case — verify it works."""
        rng = np.random.default_rng(999)
        N_q, N_t, M, D = 1, 20, 6, 4
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        sigma = 1.2

        alpha = rng.normal(size=(N_t, D))
        alpha_desc = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha)

        F_ref = ref_hessian_matvec(X_q, dX_q, X_t, dX_t, alpha.ravel(), sigma)
        F_fast = _kernels.kernel_gaussian_hessian_matvec(X_q, dX_q, X_t, alpha_desc, sigma)

        np.testing.assert_allclose(F_fast, F_ref, rtol=1e-10, atol=1e-12)

    def test_different_sigmas(self) -> None:
        """Verify correctness across different sigma values."""
        rng = np.random.default_rng(77)
        N_q, N_t, M, D = 2, 5, 4, 3
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        alpha = rng.normal(size=(N_t, D))

        for sigma in [0.1, 0.5, 1.0, 5.0, 20.0]:
            alpha_desc = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha)
            F_ref = ref_hessian_matvec(X_q, dX_q, X_t, dX_t, alpha.ravel(), sigma)
            F_fast = _kernels.kernel_gaussian_hessian_matvec(X_q, dX_q, X_t, alpha_desc, sigma)
            np.testing.assert_allclose(
                F_fast,
                F_ref,
                rtol=1e-9,
                atol=1e-11,
                err_msg=f"Failed for sigma={sigma}",
            )

    def test_output_shape_and_dtype(self) -> None:
        rng = np.random.default_rng(0)
        N_q, N_t, M, D = 3, 7, 5, 4
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        alpha_desc = rng.normal(size=(N_t, M))

        F = _kernels.kernel_gaussian_hessian_matvec(X_q, dX_q, X_t, alpha_desc, 1.0)
        assert F.shape == (N_q, D)
        assert F.dtype == np.float64
        assert np.isfinite(F).all()
