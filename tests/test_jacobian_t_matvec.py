"""Tests for the J^T·α trick: jacobian_t_matvec and full_matvec (global kernels)."""

import numpy as np
import pytest
from numpy.typing import NDArray

from kernelforge import global_kernels as _kernels


def ref_jacobian_t_matvec(
    X_q: NDArray[np.float64],
    X_t: NDArray[np.float64],
    dX_t: NDArray[np.float64],
    alpha_F: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """Reference: build full K_jt matrix, multiply by alpha_F."""
    # K_jt shape: (N_q, N_t*D)
    K_jt = _kernels.kernel_gaussian_jacobian_t(X_q, X_t, dX_t, sigma)
    result: NDArray[np.float64] = K_jt @ alpha_F.ravel()
    return result


def ref_full_matvec(
    X_q: NDArray[np.float64],
    dX_q: NDArray[np.float64],
    X_t: NDArray[np.float64],
    dX_t: NDArray[np.float64],
    alpha_E: NDArray[np.float64],
    alpha_F: NDArray[np.float64],
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference: build full K_full block matrix, multiply by [alpha_E; alpha_F]."""
    K_full = _kernels.kernel_gaussian_full(X_q, dX_q, X_t, dX_t, sigma)
    alpha = np.concatenate([alpha_E.ravel(), alpha_F.ravel()])
    pred = K_full @ alpha
    N_q = X_q.shape[0]
    D = dX_q.shape[1]
    E_ref: NDArray[np.float64] = pred[:N_q]
    F_ref: NDArray[np.float64] = pred[N_q:].reshape(N_q, D)
    return E_ref, F_ref


class TestJacobianTMatvec:
    """Tests for kernel_gaussian_jacobian_t_matvec."""

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
        """Compare matvec output against full K_jt @ alpha."""
        rng = np.random.default_rng(42)
        X_q = rng.normal(size=(N_q, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        alpha_F = rng.normal(size=(N_t, D))
        sigma = 0.7

        alpha_desc = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha_F)

        E_ref = ref_jacobian_t_matvec(X_q, X_t, dX_t, alpha_F, sigma)
        E_fast = _kernels.kernel_gaussian_jacobian_t_matvec(X_q, X_t, alpha_desc, sigma)

        assert E_fast.shape == (N_q,)
        np.testing.assert_allclose(E_fast, E_ref, rtol=1e-10, atol=1e-12)

    def test_single_query(self) -> None:
        """N_q=1 is the common prediction case."""
        rng = np.random.default_rng(999)
        N_q, N_t, M, D = 1, 20, 6, 4
        X_q = rng.normal(size=(N_q, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        alpha_F = rng.normal(size=(N_t, D))
        sigma = 1.2

        alpha_desc = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha_F)
        E_ref = ref_jacobian_t_matvec(X_q, X_t, dX_t, alpha_F, sigma)
        E_fast = _kernels.kernel_gaussian_jacobian_t_matvec(X_q, X_t, alpha_desc, sigma)

        np.testing.assert_allclose(E_fast, E_ref, rtol=1e-10, atol=1e-12)

    def test_different_sigmas(self) -> None:
        """Verify correctness across different sigma values."""
        rng = np.random.default_rng(77)
        N_q, N_t, M, D = 2, 5, 4, 3
        X_q = rng.normal(size=(N_q, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        alpha_F = rng.normal(size=(N_t, D))

        for sigma in [0.1, 0.5, 1.0, 5.0, 20.0]:
            alpha_desc = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha_F)
            E_ref = ref_jacobian_t_matvec(X_q, X_t, dX_t, alpha_F, sigma)
            E_fast = _kernels.kernel_gaussian_jacobian_t_matvec(X_q, X_t, alpha_desc, sigma)
            np.testing.assert_allclose(
                E_fast,
                E_ref,
                rtol=1e-9,
                atol=1e-11,
                err_msg=f"Failed for sigma={sigma}",
            )

    def test_output_shape_and_dtype(self) -> None:
        rng = np.random.default_rng(0)
        N_q, N_t, M = 3, 7, 5
        X_q = rng.normal(size=(N_q, M))
        X_t = rng.normal(size=(N_t, M))
        alpha_desc = rng.normal(size=(N_t, M))

        E = _kernels.kernel_gaussian_jacobian_t_matvec(X_q, X_t, alpha_desc, 1.0)
        assert E.shape == (N_q,)
        assert E.dtype == np.float64
        assert np.isfinite(E).all()

    def test_zero_alpha(self) -> None:
        """Zero alpha should give zero energy."""
        rng = np.random.default_rng(5)
        N_q, N_t, M = 3, 5, 4
        X_q = rng.normal(size=(N_q, M))
        X_t = rng.normal(size=(N_t, M))
        alpha_desc = np.zeros((N_t, M))

        E = _kernels.kernel_gaussian_jacobian_t_matvec(X_q, X_t, alpha_desc, 1.0)
        np.testing.assert_allclose(E, 0.0, atol=1e-15)


class TestFullMatvec:
    """Tests for kernel_gaussian_full_matvec."""

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
        """Compare full_matvec against K_full @ [alpha_E; alpha_F]."""
        rng = np.random.default_rng(42)
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        alpha_E = rng.normal(size=(N_t,))
        alpha_F = rng.normal(size=(N_t, D))
        sigma = 0.7

        alpha_desc_F = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha_F)

        E_ref, F_ref = ref_full_matvec(X_q, dX_q, X_t, dX_t, alpha_E, alpha_F, sigma)
        E_fast, F_fast = _kernels.kernel_gaussian_full_matvec(
            X_q, dX_q, X_t, alpha_E, alpha_desc_F, sigma
        )

        assert E_fast.shape == (N_q,)
        assert F_fast.shape == (N_q, D)
        np.testing.assert_allclose(E_fast, E_ref, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(F_fast, F_ref, rtol=1e-9, atol=1e-11)

    def test_single_query(self) -> None:
        """N_q=1 is the common prediction case."""
        rng = np.random.default_rng(999)
        N_q, N_t, M, D = 1, 20, 6, 4
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        alpha_E = rng.normal(size=(N_t,))
        alpha_F = rng.normal(size=(N_t, D))
        sigma = 1.2

        alpha_desc_F = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha_F)
        E_ref, F_ref = ref_full_matvec(X_q, dX_q, X_t, dX_t, alpha_E, alpha_F, sigma)
        E_fast, F_fast = _kernels.kernel_gaussian_full_matvec(
            X_q, dX_q, X_t, alpha_E, alpha_desc_F, sigma
        )

        np.testing.assert_allclose(E_fast, E_ref, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(F_fast, F_ref, rtol=1e-9, atol=1e-11)

    def test_zero_alpha_E(self) -> None:
        """With alpha_E=0 should degenerate to hessian+jacobian_t matvec."""
        rng = np.random.default_rng(11)
        N_q, N_t, M, D = 2, 5, 4, 3
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        alpha_E = np.zeros(N_t)
        alpha_F = rng.normal(size=(N_t, D))
        sigma = 1.0

        alpha_desc_F = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha_F)
        E_full, F_full = _kernels.kernel_gaussian_full_matvec(
            X_q, dX_q, X_t, alpha_E, alpha_desc_F, sigma
        )
        E_jt = _kernels.kernel_gaussian_jacobian_t_matvec(X_q, X_t, alpha_desc_F, sigma)
        F_hess = _kernels.kernel_gaussian_hessian_matvec(X_q, dX_q, X_t, alpha_desc_F, sigma)

        np.testing.assert_allclose(E_full, E_jt, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(F_full, F_hess, rtol=1e-10, atol=1e-12)

    def test_zero_alpha_F(self) -> None:
        """With alpha_F=0 should degenerate to scalar + jacobian matvec only."""
        rng = np.random.default_rng(22)
        N_q, N_t, M, D = 2, 5, 4, 3
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        alpha_E = rng.normal(size=(N_t,))
        alpha_F = np.zeros((N_t, D))
        sigma = 1.0

        alpha_desc_F = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha_F)
        E_full, F_full = _kernels.kernel_gaussian_full_matvec(
            X_q, dX_q, X_t, alpha_E, alpha_desc_F, sigma
        )

        # Reference: scalar kernel for energy, jacobian for forces
        K_scalar = _kernels.kernel_gaussian(X_q, X_t, -0.5 / (sigma**2))
        E_ref = K_scalar @ alpha_E
        K_jac = _kernels.kernel_gaussian_jacobian(X_q, dX_q, X_t, sigma)
        F_ref = (K_jac @ alpha_E).reshape(N_q, D)

        np.testing.assert_allclose(E_full, E_ref, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(F_full, F_ref, rtol=1e-9, atol=1e-11)

    def test_different_sigmas(self) -> None:
        rng = np.random.default_rng(77)
        N_q, N_t, M, D = 2, 5, 4, 3
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        dX_t = rng.normal(size=(N_t, D, M))
        alpha_E = rng.normal(size=(N_t,))
        alpha_F = rng.normal(size=(N_t, D))

        for sigma in [0.1, 0.5, 1.0, 5.0, 20.0]:
            alpha_desc_F = _kernels.kernel_gaussian_compute_alpha_desc(dX_t, alpha_F)
            E_ref, F_ref = ref_full_matvec(X_q, dX_q, X_t, dX_t, alpha_E, alpha_F, sigma)
            E_fast, F_fast = _kernels.kernel_gaussian_full_matvec(
                X_q, dX_q, X_t, alpha_E, alpha_desc_F, sigma
            )
            np.testing.assert_allclose(
                E_fast, E_ref, rtol=1e-9, atol=1e-11, err_msg=f"Energy failed sigma={sigma}"
            )
            np.testing.assert_allclose(
                F_fast, F_ref, rtol=1e-9, atol=1e-11, err_msg=f"Forces failed sigma={sigma}"
            )

    def test_output_shape_and_dtype(self) -> None:
        rng = np.random.default_rng(0)
        N_q, N_t, M, D = 3, 7, 5, 4
        X_q = rng.normal(size=(N_q, M))
        dX_q = rng.normal(size=(N_q, D, M))
        X_t = rng.normal(size=(N_t, M))
        alpha_E = rng.normal(size=(N_t,))
        alpha_desc_F = rng.normal(size=(N_t, M))

        E, F = _kernels.kernel_gaussian_full_matvec(X_q, dX_q, X_t, alpha_E, alpha_desc_F, 1.0)
        assert E.shape == (N_q,)
        assert F.shape == (N_q, D)
        assert E.dtype == np.float64
        assert F.dtype == np.float64
        assert np.isfinite(E).all()
        assert np.isfinite(F).all()
