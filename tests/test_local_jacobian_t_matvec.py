"""Tests for J^T alpha trick: local jacobian_t_matvec and full_matvec."""

import numpy as np
import pytest
from numpy.typing import NDArray

from kernelforge import local_kernels as _kernels


def make_test_data(
    nm1: int,
    nm2: int,
    max_atoms: int,
    rep_size: int,
    rng: np.random.Generator,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    int,
    int,
]:
    """Generate synthetic test data with guaranteed label matches."""
    x1 = rng.normal(size=(nm1, max_atoms, rep_size))
    dx1 = rng.normal(size=(nm1, max_atoms, rep_size, 3 * max_atoms))
    x2 = rng.normal(size=(nm2, max_atoms, rep_size))
    dx2 = rng.normal(size=(nm2, max_atoms, rep_size, 3 * max_atoms))
    n1 = rng.integers(1, max_atoms + 1, size=(nm1,)).astype(np.int32)
    n2 = rng.integers(1, max_atoms + 1, size=(nm2,)).astype(np.int32)
    q1 = rng.integers(1, 4, size=(nm1, max_atoms)).astype(np.int32)
    q2 = rng.integers(1, 4, size=(nm2, max_atoms)).astype(np.int32)
    naq1 = int(sum(3 * min(max(int(n1[a]), 0), max_atoms) for a in range(nm1)))
    naq2 = int(sum(3 * min(max(int(n2[b]), 0), max_atoms) for b in range(nm2)))
    return x1, dx1, x2, dx2, q1, q2, n1, n2, naq1, naq2


def ref_jacobian_t_matvec_local(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    dx2: NDArray[np.float64],
    q1: NDArray[np.int32],
    q2: NDArray[np.int32],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    alpha_f: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """Reference: build full K_jac (nm1, naq2), multiply by alpha_F to get E."""
    K_jac = _kernels.kernel_gaussian_jacobian(x1, x2, dx2, q1, q2, n1, n2, sigma)
    result: NDArray[np.float64] = K_jac @ alpha_f
    return result


def ref_full_matvec_local(
    x1: NDArray[np.float64],
    dx1: NDArray[np.float64],
    x2: NDArray[np.float64],
    dx2: NDArray[np.float64],
    q1: NDArray[np.int32],
    q2: NDArray[np.int32],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    alpha_e: NDArray[np.float64],
    alpha_f: NDArray[np.float64],
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference: build full K_full block matrix, multiply by [alpha_E; alpha_F]."""
    nm1 = x1.shape[0]
    K_full = _kernels.kernel_gaussian_full(x1, x2, dx1, dx2, q1, q2, n1, n2, sigma)
    alpha = np.concatenate([alpha_e.ravel(), alpha_f.ravel()])
    pred = K_full @ alpha
    e_ref: NDArray[np.float64] = pred[:nm1]
    f_ref: NDArray[np.float64] = pred[nm1:]
    return e_ref, f_ref


class TestLocalJacobianTMatvec:
    """Tests for kernel_gaussian_local_jacobian_t_matvec."""

    @pytest.mark.parametrize(
        ("nm1", "nm2", "max_atoms", "rep_size"),
        [
            (1, 1, 3, 4),
            (2, 3, 4, 8),
            (1, 5, 5, 12),
            (3, 4, 6, 16),
            (2, 2, 9, 27),
        ],
    )
    def test_correctness(self, nm1: int, nm2: int, max_atoms: int, rep_size: int) -> None:
        """Compare matvec output against full K_jac @ alpha_F."""
        rng = np.random.default_rng(42)
        x1, _dx1, x2, dx2, q1, q2, n1, n2, _naq1, naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_f = rng.normal(size=(naq2,))
        sigma = 1.0

        alpha_desc = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
        e_ref = ref_jacobian_t_matvec_local(x1, x2, dx2, q1, q2, n1, n2, alpha_f, sigma)
        e_fast = _kernels.kernel_gaussian_local_jacobian_t_matvec(
            x1, x2, alpha_desc, q1, q2, n1, n2, sigma
        )

        assert e_fast.shape == (nm1,)
        np.testing.assert_allclose(e_fast, e_ref, rtol=1e-10, atol=1e-12)

    def test_single_query(self) -> None:
        rng = np.random.default_rng(999)
        nm1, nm2, max_atoms, rep_size = 1, 5, 4, 8
        x1, _dx1, x2, dx2, q1, q2, n1, n2, _naq1, naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_f = rng.normal(size=(naq2,))
        sigma = 1.5

        alpha_desc = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
        e_ref = ref_jacobian_t_matvec_local(x1, x2, dx2, q1, q2, n1, n2, alpha_f, sigma)
        e_fast = _kernels.kernel_gaussian_local_jacobian_t_matvec(
            x1, x2, alpha_desc, q1, q2, n1, n2, sigma
        )
        np.testing.assert_allclose(e_fast, e_ref, rtol=1e-10, atol=1e-12)

    def test_different_sigmas(self) -> None:
        rng = np.random.default_rng(77)
        nm1, nm2, max_atoms, rep_size = 2, 4, 4, 8
        x1, _dx1, x2, dx2, q1, q2, n1, n2, _naq1, naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_f = rng.normal(size=(naq2,))

        for sigma in [0.1, 0.5, 1.0, 5.0]:
            alpha_desc = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
            e_ref = ref_jacobian_t_matvec_local(x1, x2, dx2, q1, q2, n1, n2, alpha_f, sigma)
            e_fast = _kernels.kernel_gaussian_local_jacobian_t_matvec(
                x1, x2, alpha_desc, q1, q2, n1, n2, sigma
            )
            np.testing.assert_allclose(
                e_fast, e_ref, rtol=1e-9, atol=1e-11, err_msg=f"sigma={sigma}"
            )

    def test_zero_alpha(self) -> None:
        rng = np.random.default_rng(5)
        nm1, nm2, max_atoms, rep_size = 2, 3, 4, 8
        x1, _dx1, x2, dx2, q1, q2, n1, n2, _naq1, naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_f = np.zeros(naq2)
        alpha_desc = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
        e_out = _kernels.kernel_gaussian_local_jacobian_t_matvec(
            x1, x2, alpha_desc, q1, q2, n1, n2, 1.0
        )
        np.testing.assert_allclose(e_out, 0.0, atol=1e-15)

    def test_output_dtype(self) -> None:
        rng = np.random.default_rng(0)
        nm1, nm2, max_atoms, rep_size = 2, 3, 4, 8
        x1, _dx1, x2, _dx2, q1, q2, n1, n2, _naq1, _naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_desc = rng.normal(size=(nm2, max_atoms, rep_size))
        e_out = _kernels.kernel_gaussian_local_jacobian_t_matvec(
            x1, x2, alpha_desc, q1, q2, n1, n2, 1.0
        )
        assert e_out.shape == (nm1,)
        assert e_out.dtype == np.float64
        assert np.isfinite(e_out).all()


class TestLocalFullMatvec:
    """Tests for kernel_gaussian_local_full_matvec."""

    @pytest.mark.parametrize(
        ("nm1", "nm2", "max_atoms", "rep_size"),
        [
            (1, 1, 3, 4),
            (2, 3, 4, 8),
            (1, 5, 5, 12),
            (3, 4, 6, 16),
            (2, 2, 9, 27),
        ],
    )
    def test_correctness(self, nm1: int, nm2: int, max_atoms: int, rep_size: int) -> None:
        """Compare full_matvec against K_full @ [alpha_E; alpha_F]."""
        rng = np.random.default_rng(42)
        x1, dx1, x2, dx2, q1, q2, n1, n2, naq1, naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_e = rng.normal(size=(nm2,))
        alpha_f = rng.normal(size=(naq2,))
        sigma = 1.0

        alpha_desc_f = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
        e_ref, f_ref = ref_full_matvec_local(
            x1, dx1, x2, dx2, q1, q2, n1, n2, alpha_e, alpha_f, sigma
        )
        e_fast, f_fast = _kernels.kernel_gaussian_local_full_matvec(
            x1, dx1, x2, alpha_desc_f, alpha_e, q1, q2, n1, n2, sigma
        )

        assert e_fast.shape == (nm1,)
        assert f_fast.shape == (naq1,)
        np.testing.assert_allclose(e_fast, e_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(f_fast, f_ref, rtol=1e-8, atol=1e-10)

    def test_single_query(self) -> None:
        rng = np.random.default_rng(999)
        nm1, nm2, max_atoms, rep_size = 1, 5, 4, 8
        x1, dx1, x2, dx2, q1, q2, n1, n2, _naq1, naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_e = rng.normal(size=(nm2,))
        alpha_f = rng.normal(size=(naq2,))
        sigma = 1.2

        alpha_desc_f = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
        e_ref, f_ref = ref_full_matvec_local(
            x1, dx1, x2, dx2, q1, q2, n1, n2, alpha_e, alpha_f, sigma
        )
        e_fast, f_fast = _kernels.kernel_gaussian_local_full_matvec(
            x1, dx1, x2, alpha_desc_f, alpha_e, q1, q2, n1, n2, sigma
        )
        np.testing.assert_allclose(e_fast, e_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(f_fast, f_ref, rtol=1e-8, atol=1e-10)

    def test_zero_alpha_e(self) -> None:
        """With alpha_E=0 should degenerate to jacobian_t + hessian matvec."""
        rng = np.random.default_rng(11)
        nm1, nm2, max_atoms, rep_size = 2, 4, 4, 8
        x1, dx1, x2, dx2, q1, q2, n1, n2, _naq1, naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_e = np.zeros(nm2)
        alpha_f = rng.normal(size=(naq2,))
        sigma = 1.0

        alpha_desc_f = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
        e_full, f_full = _kernels.kernel_gaussian_local_full_matvec(
            x1, dx1, x2, alpha_desc_f, alpha_e, q1, q2, n1, n2, sigma
        )
        e_jt = _kernels.kernel_gaussian_local_jacobian_t_matvec(
            x1, x2, alpha_desc_f, q1, q2, n1, n2, sigma
        )
        f_hess = _kernels.kernel_gaussian_local_hessian_matvec(
            x1, dx1, x2, alpha_desc_f, q1, q2, n1, n2, sigma
        )
        np.testing.assert_allclose(e_full, e_jt, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(f_full, f_hess, rtol=1e-10, atol=1e-12)

    def test_zero_alpha_f(self) -> None:
        """With alpha_F=0 energy and forces should only come from energy-trained terms."""
        rng = np.random.default_rng(22)
        nm1, nm2, max_atoms, rep_size = 2, 4, 4, 8
        x1, dx1, x2, dx2, q1, q2, n1, n2, _naq1, naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_e = rng.normal(size=(nm2,))
        alpha_f = np.zeros(naq2)
        sigma = 1.0

        alpha_desc_f = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
        e_full, f_full = _kernels.kernel_gaussian_local_full_matvec(
            x1, dx1, x2, alpha_desc_f, alpha_e, q1, q2, n1, n2, sigma
        )
        e_ref, f_ref = ref_full_matvec_local(
            x1, dx1, x2, dx2, q1, q2, n1, n2, alpha_e, alpha_f, sigma
        )
        np.testing.assert_allclose(e_full, e_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(f_full, f_ref, rtol=1e-8, atol=1e-10)

    def test_different_sigmas(self) -> None:
        rng = np.random.default_rng(77)
        nm1, nm2, max_atoms, rep_size = 2, 3, 4, 8
        x1, dx1, x2, dx2, q1, q2, n1, n2, _naq1, naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_e = rng.normal(size=(nm2,))
        alpha_f = rng.normal(size=(naq2,))

        for sigma in [0.1, 0.5, 1.0, 5.0]:
            alpha_desc_f = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
            e_ref, f_ref = ref_full_matvec_local(
                x1, dx1, x2, dx2, q1, q2, n1, n2, alpha_e, alpha_f, sigma
            )
            e_fast, f_fast = _kernels.kernel_gaussian_local_full_matvec(
                x1, dx1, x2, alpha_desc_f, alpha_e, q1, q2, n1, n2, sigma
            )
            np.testing.assert_allclose(
                e_fast, e_ref, rtol=1e-8, atol=1e-10, err_msg=f"E sigma={sigma}"
            )
            np.testing.assert_allclose(
                f_fast, f_ref, rtol=1e-8, atol=1e-10, err_msg=f"F sigma={sigma}"
            )

    def test_output_shape_and_dtype(self) -> None:
        rng = np.random.default_rng(0)
        nm1, nm2, max_atoms, rep_size = 2, 3, 4, 8
        x1, dx1, x2, _dx2, q1, q2, n1, n2, naq1, _naq2 = make_test_data(
            nm1, nm2, max_atoms, rep_size, rng
        )
        alpha_e = rng.normal(size=(nm2,))
        alpha_desc_f = rng.normal(size=(nm2, max_atoms, rep_size))
        e_out, f_out = _kernels.kernel_gaussian_local_full_matvec(
            x1, dx1, x2, alpha_desc_f, alpha_e, q1, q2, n1, n2, 1.0
        )
        assert e_out.shape == (nm1,)
        assert f_out.shape == (naq1,)
        assert e_out.dtype == np.float64
        assert f_out.dtype == np.float64
        assert np.isfinite(e_out).all()
        assert np.isfinite(f_out).all()
