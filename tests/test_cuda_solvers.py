# tests/test_cuda_solvers.py — correctness tests for cuda_solve_svd, cuda_solve_qr,
# and cuda_solve_gels.
#
# Compares each solver against numpy.linalg.lstsq for various problem sizes,
# rank-deficient matrices, and rcond values.

import numpy as np
import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _import_solver():
    try:
        from kernelforge import cuda_solvers

        return cuda_solvers
    except ImportError:
        pytest.skip("cuda_solvers module not built")


def _solve_np(Z: np.ndarray, y: np.ndarray, rcond: float) -> np.ndarray:
    """Reference: numpy lstsq with explicit rcond."""
    x, _, _, _ = np.linalg.lstsq(Z, y, rcond=rcond if rcond > 0 else None)
    return x.astype(np.float32)


def _solve_cuda(Z: np.ndarray, y: np.ndarray, rcond: float) -> np.ndarray:
    cs = _import_solver()
    Z_t = torch.from_numpy(Z.astype(np.float32)).clone()
    y_t = torch.from_numpy(y.astype(np.float32))
    w = cs.cuda_solve_svd(Z_t, y_t, rcond)
    return w.numpy()


# ---------------------------------------------------------------------------
# Basic overdetermined full-rank
# ---------------------------------------------------------------------------


class TestFullRankOverdetermined:
    @pytest.mark.parametrize(("m", "n"), [(200, 50), (500, 100), (1000, 64)])
    def test_solution_matches_numpy(self, m: int, n: int) -> None:
        rng = np.random.default_rng(42)
        Z = rng.standard_normal((m, n)).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)

        w_ref = _solve_np(Z.copy(), y.copy(), rcond=-1)
        w_gpu = _solve_cuda(Z.copy(), y.copy(), rcond=0.0)

        np.testing.assert_allclose(w_gpu, w_ref, rtol=1e-3, atol=1e-4)

    def test_residual_small(self) -> None:
        rng = np.random.default_rng(7)
        m, n = 300, 40
        Z = rng.standard_normal((m, n)).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)

        w_gpu = _solve_cuda(Z.copy(), y.copy(), rcond=0.0)
        residual = np.linalg.norm(Z @ w_gpu - y)
        # Should be close to the optimal residual (numpy)
        w_ref = _solve_np(Z.copy(), y.copy(), rcond=-1)
        residual_ref = np.linalg.norm(Z @ w_ref - y)

        assert residual < residual_ref * 1.01 + 1e-3


# ---------------------------------------------------------------------------
# Rank-deficient matrix — tests rcond truncation
# ---------------------------------------------------------------------------


class TestRankDeficient:
    def test_rank_deficient_rcond(self) -> None:
        """Low-rank Z: solution should be in row-space of Z."""
        rng = np.random.default_rng(13)
        m, n, r = 300, 100, 20  # rank r << n
        A = rng.standard_normal((m, r)).astype(np.float32)
        B = rng.standard_normal((r, n)).astype(np.float32)
        Z = (A @ B).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)

        rcond = 1e-3
        w_ref = _solve_np(Z.copy(), y.copy(), rcond=rcond)
        w_gpu = _solve_cuda(Z.copy(), y.copy(), rcond=rcond)

        np.testing.assert_allclose(w_gpu, w_ref, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("rcond", [1e-4, 1e-3, 1e-2, 0.1])
    def test_various_rcond(self, rcond: float) -> None:
        rng = np.random.default_rng(99)
        m, n = 200, 60
        # Construct matrix with known condition number spread
        U, _, Vt = np.linalg.svd(rng.standard_normal((m, n)), full_matrices=False)
        S = np.linspace(1.0, 1e-5, n).astype(np.float32)
        Z = (U * S) @ Vt
        y = rng.standard_normal(m).astype(np.float32)

        w_ref = _solve_np(Z.copy(), y.copy(), rcond=rcond)
        w_gpu = _solve_cuda(Z.copy(), y.copy(), rcond=rcond)

        np.testing.assert_allclose(w_gpu, w_ref, rtol=5e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Exact solution (Z is square and full rank)
# ---------------------------------------------------------------------------


class TestSquareSystem:
    def test_square_full_rank(self) -> None:
        rng = np.random.default_rng(5)
        n = 128
        Z = rng.standard_normal((n, n)).astype(np.float32)
        w_true = rng.standard_normal(n).astype(np.float32)
        y = (Z @ w_true).astype(np.float32)

        w_gpu = _solve_cuda(Z.copy(), y.copy(), rcond=0.0)
        np.testing.assert_allclose(w_gpu, w_true, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Torch tensor inputs (GPU tensor)
# ---------------------------------------------------------------------------


class TestTorchTensorInput:
    def test_gpu_tensor_input(self) -> None:
        cs = _import_solver()
        rng = np.random.default_rng(17)
        m, n = 200, 50
        Z_np = rng.standard_normal((m, n)).astype(np.float32)
        y_np = rng.standard_normal(m).astype(np.float32)

        # Pass tensors already on GPU
        Z_t = torch.from_numpy(Z_np.copy()).cuda()
        y_t = torch.from_numpy(y_np.copy()).cuda()
        w_gpu = cs.cuda_solve_svd(Z_t, y_t, 0.0).numpy()

        w_ref = _solve_np(Z_np.copy(), y_np.copy(), rcond=-1)
        np.testing.assert_allclose(w_gpu, w_ref, rtol=1e-3, atol=1e-4)

    def test_cpu_tensor_input(self) -> None:
        cs = _import_solver()
        rng = np.random.default_rng(23)
        m, n = 150, 40
        Z_np = rng.standard_normal((m, n)).astype(np.float32)
        y_np = rng.standard_normal(m).astype(np.float32)

        # Pass CPU tensors — should be moved to GPU internally
        Z_t = torch.from_numpy(Z_np.copy())
        y_t = torch.from_numpy(y_np.copy())
        w_gpu = cs.cuda_solve_svd(Z_t, y_t, 0.0).numpy()

        w_ref = _solve_np(Z_np.copy(), y_np.copy(), rcond=-1)
        np.testing.assert_allclose(w_gpu, w_ref, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Large problem — azobenzene-like sizes (smoke test)
# ---------------------------------------------------------------------------


class TestLargeSize:
    @pytest.mark.slow
    def test_azo_energy_size(self) -> None:
        """nm=1000, D=4096 energy Z."""
        cs = _import_solver()
        rng = np.random.default_rng(101)
        m, n = 1000, 4096
        Z_np = (rng.standard_normal((m, n)) / np.sqrt(n)).astype(np.float32)
        y_np = rng.standard_normal(m).astype(np.float32)

        Z_t = torch.from_numpy(Z_np.copy())
        y_t = torch.from_numpy(y_np.copy())
        w_gpu = cs.cuda_solve_svd(Z_t, y_t, rcond=1e-6).numpy()

        # Just check shape and no NaNs
        assert w_gpu.shape == (n,)
        assert not np.any(np.isnan(w_gpu))
        # Residual should be reasonable
        residual = float(np.linalg.norm(Z_np @ w_gpu - y_np))
        assert residual < float(np.linalg.norm(y_np))  # better than zero solution


# ---------------------------------------------------------------------------
# cuda_solve_qr tests
# ---------------------------------------------------------------------------


def _solve_cuda_qr(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    cs = _import_solver()
    Z_t = torch.from_numpy(Z.astype(np.float32)).clone()
    y_t = torch.from_numpy(y.astype(np.float32))
    w = cs.cuda_solve_qr(Z_t, y_t)
    return w.numpy()


class TestQRSolver:
    @pytest.mark.parametrize(("m", "n"), [(200, 50), (500, 100), (1000, 64)])
    def test_solution_matches_numpy(self, m: int, n: int) -> None:
        rng = np.random.default_rng(42)
        Z = rng.standard_normal((m, n)).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)

        w_ref = _solve_np(Z.copy(), y.copy(), rcond=-1)
        w_gpu = _solve_cuda_qr(Z.copy(), y.copy())

        np.testing.assert_allclose(w_gpu, w_ref, rtol=1e-3, atol=1e-4)

    def test_residual_small(self) -> None:
        rng = np.random.default_rng(7)
        m, n = 300, 40
        Z = rng.standard_normal((m, n)).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)

        w_gpu = _solve_cuda_qr(Z.copy(), y.copy())
        residual = np.linalg.norm(Z @ w_gpu - y)
        w_ref = _solve_np(Z.copy(), y.copy(), rcond=-1)
        residual_ref = np.linalg.norm(Z @ w_ref - y)

        assert residual < residual_ref * 1.01 + 1e-3

    def test_square_full_rank(self) -> None:
        rng = np.random.default_rng(5)
        n = 128
        Z = rng.standard_normal((n, n)).astype(np.float32)
        w_true = rng.standard_normal(n).astype(np.float32)
        y = (Z @ w_true).astype(np.float32)

        w_gpu = _solve_cuda_qr(Z.copy(), y.copy())
        np.testing.assert_allclose(w_gpu, w_true, rtol=1e-3, atol=1e-3)

    def test_gpu_tensor_input(self) -> None:
        cs = _import_solver()
        rng = np.random.default_rng(17)
        m, n = 200, 50
        Z_np = rng.standard_normal((m, n)).astype(np.float32)
        y_np = rng.standard_normal(m).astype(np.float32)

        Z_t = torch.from_numpy(Z_np.copy()).cuda()
        y_t = torch.from_numpy(y_np.copy()).cuda()
        w_gpu = cs.cuda_solve_qr(Z_t, y_t).numpy()

        w_ref = _solve_np(Z_np.copy(), y_np.copy(), rcond=-1)
        np.testing.assert_allclose(w_gpu, w_ref, rtol=1e-3, atol=1e-4)

    @pytest.mark.slow
    def test_azo_energy_size(self) -> None:
        """nm=1000, D=4096 energy-only size."""
        cs = _import_solver()
        rng = np.random.default_rng(101)
        m, n = 1000, 4096
        Z_np = (rng.standard_normal((m, n)) / np.sqrt(n)).astype(np.float32)
        y_np = rng.standard_normal(m).astype(np.float32)

        Z_t = torch.from_numpy(Z_np.copy())
        y_t = torch.from_numpy(y_np.copy())
        w_gpu = cs.cuda_solve_qr(Z_t, y_t).numpy()

        assert w_gpu.shape == (n,)
        assert not np.any(np.isnan(w_gpu))
        residual = float(np.linalg.norm(Z_np @ w_gpu - y_np))
        assert residual < float(np.linalg.norm(y_np))


# ---------------------------------------------------------------------------
# cuda_solve_gels tests
# ---------------------------------------------------------------------------


def _solve_cuda_gels(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    cs = _import_solver()
    Z_t = torch.from_numpy(Z.astype(np.float32)).clone()
    y_t = torch.from_numpy(y.astype(np.float32))
    w = cs.cuda_solve_gels(Z_t, y_t)
    return w.numpy()


class TestGelsSolver:
    @pytest.mark.parametrize(("m", "n"), [(200, 50), (500, 100), (1000, 64)])
    def test_solution_matches_numpy(self, m: int, n: int) -> None:
        rng = np.random.default_rng(42)
        Z = rng.standard_normal((m, n)).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)

        w_ref = _solve_np(Z.copy(), y.copy(), rcond=-1)
        w_gpu = _solve_cuda_gels(Z.copy(), y.copy())

        np.testing.assert_allclose(w_gpu, w_ref, rtol=1e-3, atol=1e-4)

    def test_residual_small(self) -> None:
        rng = np.random.default_rng(7)
        m, n = 300, 40
        Z = rng.standard_normal((m, n)).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)

        w_gpu = _solve_cuda_gels(Z.copy(), y.copy())
        residual = np.linalg.norm(Z @ w_gpu - y)
        w_ref = _solve_np(Z.copy(), y.copy(), rcond=-1)
        residual_ref = np.linalg.norm(Z @ w_ref - y)

        assert residual < residual_ref * 1.01 + 1e-3

    def test_square_full_rank(self) -> None:
        rng = np.random.default_rng(5)
        n = 128
        Z = rng.standard_normal((n, n)).astype(np.float32)
        w_true = rng.standard_normal(n).astype(np.float32)
        y = (Z @ w_true).astype(np.float32)

        w_gpu = _solve_cuda_gels(Z.copy(), y.copy())
        np.testing.assert_allclose(w_gpu, w_true, rtol=1e-3, atol=1e-3)

    def test_gpu_tensor_input(self) -> None:
        cs = _import_solver()
        rng = np.random.default_rng(17)
        m, n = 200, 50
        Z_np = rng.standard_normal((m, n)).astype(np.float32)
        y_np = rng.standard_normal(m).astype(np.float32)

        Z_t = torch.from_numpy(Z_np.copy()).cuda()
        y_t = torch.from_numpy(y_np.copy()).cuda()
        w_gpu = cs.cuda_solve_gels(Z_t, y_t).numpy()

        w_ref = _solve_np(Z_np.copy(), y_np.copy(), rcond=-1)
        np.testing.assert_allclose(w_gpu, w_ref, rtol=1e-3, atol=1e-4)

    def test_col_major_input(self) -> None:
        """z_col_major=True path: Z passed as (n, m) col-major tensor."""
        cs = _import_solver()
        rng = np.random.default_rng(31)
        m, n = 250, 60
        Z_np = rng.standard_normal((m, n)).astype(np.float32)
        y_np = rng.standard_normal(m).astype(np.float32)

        # Simulate col-major layout: transpose to (n, m) and pass z_col_major=True.
        Z_col = torch.from_numpy(Z_np.T.copy())
        y_t = torch.from_numpy(y_np.copy())
        w_gpu = cs.cuda_solve_gels(Z_col, y_t, z_col_major=True).numpy()

        w_ref = _solve_np(Z_np.copy(), y_np.copy(), rcond=-1)
        np.testing.assert_allclose(w_gpu, w_ref, rtol=1e-3, atol=1e-4)

    @pytest.mark.slow
    def test_azo_energy_size(self) -> None:
        """nm=1000, D=4096 energy-only size."""
        cs = _import_solver()
        rng = np.random.default_rng(101)
        m, n = 1000, 4096
        Z_np = (rng.standard_normal((m, n)) / np.sqrt(n)).astype(np.float32)
        y_np = rng.standard_normal(m).astype(np.float32)

        Z_t = torch.from_numpy(Z_np.copy())
        y_t = torch.from_numpy(y_np.copy())
        w_gpu = cs.cuda_solve_gels(Z_t, y_t).numpy()

        assert w_gpu.shape == (n,)
        assert not np.any(np.isnan(w_gpu))
        residual = float(np.linalg.norm(Z_np @ w_gpu - y_np))
        assert residual < float(np.linalg.norm(y_np))
