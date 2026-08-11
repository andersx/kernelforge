"""Tests for J^T·α trick in local (FCHL19) Hessian kernel."""

import numpy as np
import pytest
from numpy.typing import NDArray

from kernelforge import local_kernels as _kernels


def ref_alpha_desc(
    dx2: NDArray[np.float64],
    q2: NDArray[np.int32],
    n2: NDArray[np.int32],
    alpha: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reference: alpha_desc[b,i2,:] = dx2[b,i2,:,:3*n2[b]] @ alpha[offs2[b]:offs2[b]+3*n2[b]]."""
    nm2, max_atoms2, rep_size = dx2.shape[:3]
    alpha_desc = np.zeros((nm2, max_atoms2, rep_size))

    # Compute offsets
    offs2 = np.zeros(nm2, dtype=int)
    acc = 0
    for b in range(nm2):
        offs2[b] = acc
        nb = min(max(n2[b], 0), max_atoms2)
        acc += 3 * nb

    for b in range(nm2):
        nb = min(max(n2[b], 0), max_atoms2)
        ncols_b = 3 * nb
        alpha_mol_b = alpha[offs2[b] : offs2[b] + ncols_b]  # (ncols_b,)
        for i2 in range(nb):
            dx_bi2 = dx2[b, i2, :, :ncols_b]  # (rep_size, ncols_b)
            # alpha_desc[b,i2,:] = dx2[b,i2,:,:ncols_b] @ alpha[offs2[b]:offs2[b]+ncols_b]
            alpha_desc[b, i2, :] = dx_bi2 @ alpha_mol_b  # (rep_size,)

    return alpha_desc


def ref_hessian_matvec(
    x1: NDArray[np.float64],
    dx1: NDArray[np.float64],
    x2: NDArray[np.float64],
    dx2: NDArray[np.float64],
    q1: NDArray[np.int32],
    q2: NDArray[np.int32],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    alpha: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """Reference: build full local Hessian, multiply by alpha."""
    H = _kernels.kernel_gaussian_hessian(x1, x2, dx1, dx2, q1, q2, n1, n2, sigma)
    result: NDArray[np.float64] = H @ alpha
    return result


class TestComputeAlphaDesc:
    """Tests for kernel_gaussian_local_compute_alpha_desc."""

    @pytest.mark.parametrize(
        ("nm2", "max_atoms2", "rep_size"),
        [
            (1, 3, 4),
            (2, 4, 8),
            (3, 5, 16),
            (5, 9, 27),  # Ethanol-like
        ],
    )
    def test_correctness(self, nm2: int, max_atoms2: int, rep_size: int) -> None:
        """Verify alpha_desc computation against reference."""
        rng = np.random.default_rng(42)

        # Create test data
        dx2 = rng.normal(size=(nm2, max_atoms2, rep_size, 3 * max_atoms2))
        q2 = rng.integers(0, 5, size=(nm2, max_atoms2)).astype(np.int32)
        n2 = rng.integers(1, max_atoms2 + 1, size=(nm2,)).astype(np.int32)

        # Compute naq2
        naq2 = sum(3 * min(max(n2[b], 0), max_atoms2) for b in range(nm2))
        alpha = rng.normal(size=(naq2,))

        # Compute via C++
        result = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha)

        # Reference
        expected = ref_alpha_desc(dx2, q2, n2, alpha)

        assert result.shape == (nm2, max_atoms2, rep_size)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)

    def test_output_dtype(self) -> None:
        """Verify output is float64."""
        rng = np.random.default_rng(0)
        nm2, max_atoms2, rep_size = 2, 3, 4
        dx2 = rng.normal(size=(nm2, max_atoms2, rep_size, 3 * max_atoms2))
        q2 = rng.integers(0, 3, size=(nm2, max_atoms2)).astype(np.int32)
        n2 = np.array([2, 3], dtype=np.int32)
        naq2 = 3 * 2 + 3 * 3
        alpha = rng.normal(size=(naq2,))

        result = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha)
        assert result.dtype == np.float64

    def test_single_molecule(self) -> None:
        """Test with single molecule."""
        rng = np.random.default_rng(123)
        nm2, max_atoms2, rep_size = 1, 4, 5
        dx2 = rng.normal(size=(nm2, max_atoms2, rep_size, 3 * max_atoms2))
        q2 = np.array([[1, 1, 6, 8]], dtype=np.int32)
        n2 = np.array([3], dtype=np.int32)
        naq2 = 9
        alpha = rng.normal(size=(naq2,))

        result = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha)
        expected = ref_alpha_desc(dx2, q2, n2, alpha)

        assert result.shape == (1, 4, 5)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)


class TestHessianMatvec:
    """Tests for kernel_gaussian_local_hessian_matvec."""

    @pytest.mark.parametrize(
        ("nm1", "nm2", "max_atoms1", "max_atoms2", "rep_size"),
        [
            (1, 1, 3, 3, 4),
            (1, 3, 4, 4, 6),
            (2, 2, 4, 4, 8),
            (3, 5, 5, 5, 10),
        ],
    )
    def test_correctness(
        self, nm1: int, nm2: int, max_atoms1: int, max_atoms2: int, rep_size: int
    ) -> None:
        """Compare matvec against full Hessian @ alpha."""
        rng = np.random.default_rng(456)

        # Create test data with label overlap
        q1 = rng.integers(0, 3, size=(nm1, max_atoms1)).astype(np.int32)
        q2 = rng.integers(0, 3, size=(nm2, max_atoms2)).astype(np.int32)
        n1 = rng.integers(1, max_atoms1 + 1, size=(nm1,)).astype(np.int32)
        n2 = rng.integers(1, max_atoms2 + 1, size=(nm2,)).astype(np.int32)

        x1 = rng.normal(size=(nm1, max_atoms1, rep_size))
        dx1 = rng.normal(size=(nm1, max_atoms1, rep_size, 3 * max_atoms1))
        x2 = rng.normal(size=(nm2, max_atoms2, rep_size))
        dx2 = rng.normal(size=(nm2, max_atoms2, rep_size, 3 * max_atoms2))
        sigma = 1.0

        # Compute naq values
        naq1 = sum(3 * min(max(n1[a], 0), max_atoms1) for a in range(nm1))
        naq2 = sum(3 * min(max(n2[b], 0), max_atoms2) for b in range(nm2))

        # Random alpha coefficients
        alpha = rng.normal(size=(naq2,))

        # Pre-compute alpha_desc
        alpha_desc = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha)

        # Reference via full Hessian
        F_ref = ref_hessian_matvec(x1, dx1, x2, dx2, q1, q2, n1, n2, alpha, sigma)

        # Fast matvec
        F_fast = _kernels.kernel_gaussian_local_hessian_matvec(
            x1, dx1, x2, alpha_desc, q1, q2, n1, n2, sigma
        )

        assert F_fast.shape == (naq1,)
        np.testing.assert_allclose(F_fast, F_ref, rtol=1e-10, atol=1e-12)

    def test_single_query_molecule(self) -> None:
        """Test prediction with single query molecule (typical use case)."""
        rng = np.random.default_rng(789)
        nm1, nm2 = 1, 5
        max_atoms1, max_atoms2 = 4, 4
        rep_size = 8
        sigma = 0.8

        q1 = rng.integers(0, 4, size=(nm1, max_atoms1)).astype(np.int32)
        q2 = rng.integers(0, 4, size=(nm2, max_atoms2)).astype(np.int32)
        n1 = rng.integers(1, max_atoms1 + 1, size=(nm1,)).astype(np.int32)
        n2 = rng.integers(1, max_atoms2 + 1, size=(nm2,)).astype(np.int32)

        x1 = rng.normal(size=(nm1, max_atoms1, rep_size))
        dx1 = rng.normal(size=(nm1, max_atoms1, rep_size, 3 * max_atoms1))
        x2 = rng.normal(size=(nm2, max_atoms2, rep_size))
        dx2 = rng.normal(size=(nm2, max_atoms2, rep_size, 3 * max_atoms2))

        naq2 = sum(3 * min(max(n2[b], 0), max_atoms2) for b in range(nm2))
        alpha = rng.normal(size=(naq2,))

        alpha_desc = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha)
        F_ref = ref_hessian_matvec(x1, dx1, x2, dx2, q1, q2, n1, n2, alpha, sigma)
        F_fast = _kernels.kernel_gaussian_local_hessian_matvec(
            x1, dx1, x2, alpha_desc, q1, q2, n1, n2, sigma
        )

        np.testing.assert_allclose(F_fast, F_ref, rtol=1e-10, atol=1e-12)

    def test_different_sigmas(self) -> None:
        """Verify correctness across different sigma values."""
        rng = np.random.default_rng(111)
        nm1, nm2 = 2, 3
        max_atoms1, max_atoms2 = 4, 4
        rep_size = 6

        q1 = rng.integers(0, 3, size=(nm1, max_atoms1)).astype(np.int32)
        q2 = rng.integers(0, 3, size=(nm2, max_atoms2)).astype(np.int32)
        n1 = rng.integers(1, max_atoms1 + 1, size=(nm1,)).astype(np.int32)
        n2 = rng.integers(1, max_atoms2 + 1, size=(nm2,)).astype(np.int32)

        x1 = rng.normal(size=(nm1, max_atoms1, rep_size))
        dx1 = rng.normal(size=(nm1, max_atoms1, rep_size, 3 * max_atoms1))
        x2 = rng.normal(size=(nm2, max_atoms2, rep_size))
        dx2 = rng.normal(size=(nm2, max_atoms2, rep_size, 3 * max_atoms2))

        naq2 = sum(3 * min(max(n2[b], 0), max_atoms2) for b in range(nm2))
        alpha = rng.normal(size=(naq2,))

        for sigma in [0.2, 0.5, 1.0, 2.0, 5.0]:
            alpha_desc = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha)
            F_ref = ref_hessian_matvec(x1, dx1, x2, dx2, q1, q2, n1, n2, alpha, sigma)
            F_fast = _kernels.kernel_gaussian_local_hessian_matvec(
                x1, dx1, x2, alpha_desc, q1, q2, n1, n2, sigma
            )
            np.testing.assert_allclose(
                F_fast,
                F_ref,
                rtol=1e-9,
                atol=1e-11,
                err_msg=f"Failed for sigma={sigma}",
            )

    def test_output_shape_and_dtype(self) -> None:
        """Verify output shape and dtype."""
        rng = np.random.default_rng(222)
        nm1, nm2 = 2, 3
        max_atoms1, max_atoms2 = 4, 4
        rep_size = 5

        q1 = rng.integers(0, 3, size=(nm1, max_atoms1)).astype(np.int32)
        q2 = rng.integers(0, 3, size=(nm2, max_atoms2)).astype(np.int32)
        n1 = np.array([3, 2], dtype=np.int32)
        n2 = np.array([3, 2, 4], dtype=np.int32)

        x1 = rng.normal(size=(nm1, max_atoms1, rep_size))
        dx1 = rng.normal(size=(nm1, max_atoms1, rep_size, 3 * max_atoms1))
        x2 = rng.normal(size=(nm2, max_atoms2, rep_size))
        alpha_desc = rng.normal(size=(nm2, max_atoms2, rep_size))

        F = _kernels.kernel_gaussian_local_hessian_matvec(
            x1, dx1, x2, alpha_desc, q1, q2, n1, n2, 1.0
        )

        naq1 = 3 * 3 + 3 * 2
        assert F.shape == (naq1,)
        assert F.dtype == np.float64
        assert np.isfinite(F).all()

    def test_no_label_matches(self) -> None:
        """Test when query and training have no matching labels."""
        rng = np.random.default_rng(333)
        nm1, nm2 = 1, 2
        max_atoms1, max_atoms2 = 3, 3
        rep_size = 4

        # Different labels
        q1 = np.array([[0, 0, 0]], dtype=np.int32)
        q2 = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
        n1 = np.array([3], dtype=np.int32)
        n2 = np.array([3, 3], dtype=np.int32)

        x1 = rng.normal(size=(nm1, max_atoms1, rep_size))
        dx1 = rng.normal(size=(nm1, max_atoms1, rep_size, 3 * max_atoms1))
        x2 = rng.normal(size=(nm2, max_atoms2, rep_size))
        alpha_desc = rng.normal(size=(nm2, max_atoms2, rep_size))

        # Should return zero forces (no contributions)
        F = _kernels.kernel_gaussian_local_hessian_matvec(
            x1, dx1, x2, alpha_desc, q1, q2, n1, n2, 1.0
        )

        naq1 = 3 * 3
        assert F.shape == (naq1,)
        np.testing.assert_allclose(F, 0.0, atol=1e-14)

    def test_consistency_with_zero_alpha(self) -> None:
        """Test that zero alpha produces zero forces."""
        rng = np.random.default_rng(444)
        nm1, nm2 = 2, 2
        max_atoms1, max_atoms2 = 4, 4
        rep_size = 6

        q1 = rng.integers(0, 3, size=(nm1, max_atoms1)).astype(np.int32)
        q2 = rng.integers(0, 3, size=(nm2, max_atoms2)).astype(np.int32)
        n1 = rng.integers(1, max_atoms1 + 1, size=(nm1,)).astype(np.int32)
        n2 = rng.integers(1, max_atoms2 + 1, size=(nm2,)).astype(np.int32)

        x1 = rng.normal(size=(nm1, max_atoms1, rep_size))
        dx1 = rng.normal(size=(nm1, max_atoms1, rep_size, 3 * max_atoms1))
        x2 = rng.normal(size=(nm2, max_atoms2, rep_size))

        naq2 = sum(3 * min(max(n2[b], 0), max_atoms2) for b in range(nm2))
        alpha = np.zeros(naq2)

        alpha_desc = _kernels.kernel_gaussian_local_compute_alpha_desc(
            np.random.default_rng().normal(size=(nm2, max_atoms2, rep_size, 3 * max_atoms2)),
            q2,
            n2,
            alpha,
        )

        F = _kernels.kernel_gaussian_local_hessian_matvec(
            x1, dx1, x2, alpha_desc, q1, q2, n1, n2, 1.0
        )

        naq1 = sum(3 * min(max(n1[a], 0), max_atoms1) for a in range(nm1))
        assert F.shape == (naq1,)
        np.testing.assert_allclose(F, 0.0, atol=1e-14)
