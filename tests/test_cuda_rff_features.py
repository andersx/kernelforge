"""Tests for cuda_rff_features energy-only global RFF primitives."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

cuda_rff_features = pytest.importorskip(
    "kernelforge.cuda_rff_features",
    reason="cuda_rff_features not built (requires CUDA + PyTorch)",
)
torch = pytest.importorskip("torch", reason="PyTorch not installed")

_nvidia_smi = shutil.which("nvidia-smi")
try:
    _gpu_ok = (
        _nvidia_smi is not None
        and subprocess.run(  # noqa: S603
            [_nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        ).returncode
        == 0
    )
except (OSError, subprocess.TimeoutExpired):
    _gpu_ok = False

pytestmark = pytest.mark.skipif(not _gpu_ok, reason="No NVIDIA GPU detected")

from kernelforge import kernelmath  # noqa: E402
from kernelforge.kitchen_sinks import (  # noqa: E402
    rff_features,
    rff_features_elemental,
    rff_full_gramian_elemental,
    rff_full_gramian_symm,
    rff_gradient,
    rff_gradient_elemental,
)


def _to_cuda(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).cuda()


def _to_cuda_i32(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int32)).cuda()


def _q_list_from_padded(Q: np.ndarray, N: np.ndarray) -> list[np.ndarray]:
    return [Q[i, : N[i]].astype(np.int32) for i in range(Q.shape[0])]


def _compact_forces(F_full: np.ndarray, N: np.ndarray) -> np.ndarray:
    return np.concatenate([F_full[i, : N[i]].ravel() for i in range(F_full.shape[0])])


def test_rff_features_vs_cpu() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(11, 7)).astype(np.float64)
    W = rng.normal(size=(7, 17)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(17,)).astype(np.float64)

    Z_cpu = rff_features(X, W, b)
    Z_gpu = cuda_rff_features.rff_features(_to_cuda(X), _to_cuda(W), _to_cuda(b)).cpu().numpy()

    np.testing.assert_allclose(Z_gpu.astype(np.float64), Z_cpu, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("chunk_size", [1, 4, 64])
def test_rff_gramian_symm_rfp_vs_cpu(chunk_size: int) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(13, 8)).astype(np.float64)
    W = rng.normal(size=(8, 16)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(16,)).astype(np.float64)
    Y = rng.normal(size=(13,)).astype(np.float64)

    Z = rff_features(X, W, b)
    gram_ref = Z.T @ Z
    proj_ref = Z.T @ Y

    gram_rfp, proj = cuda_rff_features.rff_gramian_symm_rfp(
        _to_cuda(X), _to_cuda(W), _to_cuda(b), _to_cuda(Y), chunk_size
    )
    gram = kernelmath.rfp_to_full(
        gram_rfp.cpu().numpy().astype(np.float64), W.shape[1], uplo="U", transr="N"
    )

    np.testing.assert_allclose(gram, gram_ref, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(
        proj.cpu().numpy().astype(np.float64), proj_ref, rtol=5e-5, atol=5e-5
    )


def test_rff_predict_energy_vs_cpu() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(9, 6)).astype(np.float64)
    W = rng.normal(size=(6, 12)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(12,)).astype(np.float64)
    weights = rng.normal(size=(12,)).astype(np.float64)

    pred_ref = rff_features(X, W, b) @ weights
    pred = cuda_rff_features.rff_predict_energy(
        _to_cuda(X), _to_cuda(W), _to_cuda(b), _to_cuda(weights), 3
    )

    np.testing.assert_allclose(
        pred.cpu().numpy().astype(np.float64), pred_ref, rtol=5e-6, atol=5e-6
    )


@pytest.mark.parametrize(("energy_chunk", "force_chunk"), [(1, 1), (4, 3), (64, 64)])
def test_rff_full_gramian_symm_rfp_vs_cpu(energy_chunk: int, force_chunk: int) -> None:
    rng = np.random.default_rng(3)
    N, M, D, ncoords = 7, 5, 10, 6
    X = rng.normal(size=(N, M)).astype(np.float64)
    dX = rng.normal(size=(N, ncoords, M)).astype(np.float64)
    W = rng.normal(size=(M, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(D,)).astype(np.float64)
    Y = rng.normal(size=(N,)).astype(np.float64)
    F = rng.normal(size=(N * ncoords,)).astype(np.float64)

    gram_ref, proj_ref = rff_full_gramian_symm(X, dX, W, b, Y, F, energy_chunk=3, force_chunk=2)
    gram_rfp, proj = cuda_rff_features.rff_full_gramian_symm_rfp(
        _to_cuda(X),
        _to_cuda(dX),
        _to_cuda(W),
        _to_cuda(b),
        _to_cuda(Y),
        _to_cuda(F),
        energy_chunk,
        force_chunk,
    )
    gram = kernelmath.rfp_to_full(
        gram_rfp.cpu().numpy().astype(np.float64), D, uplo="U", transr="N"
    )

    np.testing.assert_allclose(gram, gram_ref, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        proj.cpu().numpy().astype(np.float64), proj_ref, rtol=2e-4, atol=2e-4
    )


def test_rff_predict_force_vs_cpu() -> None:
    rng = np.random.default_rng(4)
    N, M, D, ncoords = 6, 5, 9, 6
    X = rng.normal(size=(N, M)).astype(np.float64)
    dX = rng.normal(size=(N, ncoords, M)).astype(np.float64)
    W = rng.normal(size=(M, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(D,)).astype(np.float64)
    weights = rng.normal(size=(D,)).astype(np.float64)

    pred_ref = rff_gradient(X, dX, W, b).T @ weights
    pred = cuda_rff_features.rff_predict_force(
        _to_cuda(X), _to_cuda(dX), _to_cuda(W), _to_cuda(b), _to_cuda(weights), 3
    )

    np.testing.assert_allclose(
        pred.cpu().numpy().astype(np.float64), pred_ref, rtol=5e-5, atol=5e-5
    )


def test_rff_features_elemental_vs_cpu() -> None:
    rng = np.random.default_rng(5)
    nmol, max_atoms, rep_size, nel, D = 6, 4, 7, 3, 11
    N = np.array([4, 2, 3, 4, 1, 3], dtype=np.int32)
    X = rng.normal(size=(nmol, max_atoms, rep_size)).astype(np.float64)
    Q = rng.integers(0, nel, size=(nmol, max_atoms), dtype=np.int32)
    for i, n_i in enumerate(N):
        X[i, n_i:] = 0.0
        Q[i, n_i:] = -1
    W = rng.normal(size=(nel, rep_size, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(nel, D)).astype(np.float64)

    ref = rff_features_elemental(X, _q_list_from_padded(Q, N), W, b)
    got = cuda_rff_features.rff_features_elemental(
        _to_cuda(X), _to_cuda_i32(Q), _to_cuda_i32(N), _to_cuda(W), _to_cuda(b)
    )
    np.testing.assert_allclose(got.cpu().numpy().astype(np.float64), ref, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_rff_gramian_elemental_rfp_vs_cpu(chunk_size: int) -> None:
    rng = np.random.default_rng(6)
    nmol, max_atoms, rep_size, nel, D = 7, 4, 6, 3, 10
    N = np.array([4, 3, 2, 4, 1, 3, 2], dtype=np.int32)
    X = rng.normal(size=(nmol, max_atoms, rep_size)).astype(np.float64)
    Q = rng.integers(0, nel, size=(nmol, max_atoms), dtype=np.int32)
    for i, n_i in enumerate(N):
        X[i, n_i:] = 0.0
        Q[i, n_i:] = -1
    W = rng.normal(size=(nel, rep_size, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(nel, D)).astype(np.float64)
    Y = rng.normal(size=(nmol,)).astype(np.float64)

    LZ = rff_features_elemental(X, _q_list_from_padded(Q, N), W, b)
    gram_ref = LZ.T @ LZ
    proj_ref = LZ.T @ Y
    gram_rfp, proj = cuda_rff_features.rff_gramian_elemental_rfp(
        _to_cuda(X),
        _to_cuda_i32(Q),
        _to_cuda_i32(N),
        _to_cuda(W),
        _to_cuda(b),
        _to_cuda(Y),
        chunk_size,
    )
    gram = kernelmath.rfp_to_full(
        gram_rfp.cpu().numpy().astype(np.float64), D, uplo="U", transr="N"
    )
    np.testing.assert_allclose(gram, gram_ref, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        proj.cpu().numpy().astype(np.float64), proj_ref, rtol=2e-4, atol=2e-4
    )


def test_rff_predict_energy_elemental_vs_cpu() -> None:
    rng = np.random.default_rng(7)
    nmol, max_atoms, rep_size, nel, D = 5, 3, 6, 2, 9
    N = np.array([3, 2, 1, 3, 2], dtype=np.int32)
    X = rng.normal(size=(nmol, max_atoms, rep_size)).astype(np.float64)
    Q = rng.integers(0, nel, size=(nmol, max_atoms), dtype=np.int32)
    for i, n_i in enumerate(N):
        X[i, n_i:] = 0.0
        Q[i, n_i:] = -1
    W = rng.normal(size=(nel, rep_size, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(nel, D)).astype(np.float64)
    weights = rng.normal(size=(D,)).astype(np.float64)

    ref = rff_features_elemental(X, _q_list_from_padded(Q, N), W, b) @ weights
    got = cuda_rff_features.rff_predict_energy_elemental(
        _to_cuda(X),
        _to_cuda_i32(Q),
        _to_cuda_i32(N),
        _to_cuda(W),
        _to_cuda(b),
        _to_cuda(weights),
        2,
    )
    np.testing.assert_allclose(got.cpu().numpy().astype(np.float64), ref, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize(("energy_chunk", "force_chunk"), [(1, 1), (3, 2), (64, 64)])
def test_rff_full_gramian_elemental_rfp_vs_cpu(energy_chunk: int, force_chunk: int) -> None:
    rng = np.random.default_rng(8)
    nmol, max_atoms, rep_size, nel, D = 6, 3, 5, 2, 8
    N = np.array([3, 3, 2, 1, 3, 2], dtype=np.int32)
    X = rng.normal(size=(nmol, max_atoms, rep_size)).astype(np.float64)
    dX = rng.normal(size=(nmol, max_atoms, rep_size, max_atoms, 3)).astype(np.float64)
    Q = rng.integers(0, nel, size=(nmol, max_atoms), dtype=np.int32)
    for i, n_i in enumerate(N):
        X[i, n_i:] = 0.0
        dX[i, n_i:] = 0.0
        dX[i, :, :, n_i:] = 0.0
        Q[i, n_i:] = -1
    W = rng.normal(size=(nel, rep_size, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(nel, D)).astype(np.float64)
    Y = rng.normal(size=(nmol,)).astype(np.float64)
    F_full = rng.normal(size=(nmol, max_atoms, 3)).astype(np.float64)
    for i, n_i in enumerate(N):
        F_full[i, n_i:] = 0.0
    F = _compact_forces(F_full, N)

    gram_ref, proj_ref = rff_full_gramian_elemental(
        X, dX, _q_list_from_padded(Q, N), W, b, Y, F, energy_chunk=2, force_chunk=2
    )
    gram_rfp, proj = cuda_rff_features.rff_full_gramian_elemental_rfp(
        _to_cuda(X),
        _to_cuda(dX),
        _to_cuda_i32(Q),
        _to_cuda_i32(N),
        _to_cuda(W),
        _to_cuda(b),
        _to_cuda(Y),
        _to_cuda(F),
        energy_chunk,
        force_chunk,
    )
    gram = kernelmath.rfp_to_full(
        gram_rfp.cpu().numpy().astype(np.float64), D, uplo="U", transr="N"
    )
    np.testing.assert_allclose(gram, gram_ref, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(
        proj.cpu().numpy().astype(np.float64), proj_ref, rtol=5e-4, atol=5e-4
    )


def test_rff_predict_force_elemental_vs_cpu() -> None:
    rng = np.random.default_rng(9)
    nmol, max_atoms, rep_size, nel, D = 4, 3, 5, 2, 8
    N = np.array([3, 2, 1, 3], dtype=np.int32)
    X = rng.normal(size=(nmol, max_atoms, rep_size)).astype(np.float64)
    dX = rng.normal(size=(nmol, max_atoms, rep_size, max_atoms, 3)).astype(np.float64)
    Q = rng.integers(0, nel, size=(nmol, max_atoms), dtype=np.int32)
    for i, n_i in enumerate(N):
        X[i, n_i:] = 0.0
        dX[i, n_i:] = 0.0
        dX[i, :, :, n_i:] = 0.0
        Q[i, n_i:] = -1
    W = rng.normal(size=(nel, rep_size, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(nel, D)).astype(np.float64)
    weights = rng.normal(size=(D,)).astype(np.float64)

    ref = rff_gradient_elemental(X, dX, _q_list_from_padded(Q, N), W, b).T @ weights
    got = cuda_rff_features.rff_predict_force_elemental(
        _to_cuda(X),
        _to_cuda(dX),
        _to_cuda_i32(Q),
        _to_cuda_i32(N),
        _to_cuda(W),
        _to_cuda(b),
        _to_cuda(weights),
        2,
    )
    np.testing.assert_allclose(got.cpu().numpy().astype(np.float64), ref, rtol=2e-4, atol=2e-4)


def test_rff_features_elemental_col_major_matches_row_major() -> None:
    """Col-major output must be the transpose of row-major output (same values)."""
    rng = np.random.default_rng(99)
    nmol, max_atoms, rep_size, nel, D = 8, 5, 7, 3, 16
    N = np.array([5, 2, 4, 3, 5, 1, 4, 2], dtype=np.int32)
    X = rng.normal(size=(nmol, max_atoms, rep_size)).astype(np.float64)
    Q = rng.integers(0, nel, size=(nmol, max_atoms), dtype=np.int32)
    for i, n_i in enumerate(N):
        X[i, n_i:] = 0.0
        Q[i, n_i:] = -1
    W = rng.normal(size=(nel, rep_size, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(nel, D)).astype(np.float64)

    X_c, Q_c, N_c, W_c, b_c = (
        _to_cuda(X),
        _to_cuda_i32(Q),
        _to_cuda_i32(N),
        _to_cuda(W),
        _to_cuda(b),
    )
    Z_row = cuda_rff_features.rff_features_elemental(X_c, Q_c, N_c, W_c, b_c)
    Z_col = cuda_rff_features.rff_features_elemental_col_major(X_c, Q_c, N_c, W_c, b_c)

    assert Z_row.shape == (nmol, D), f"row-major shape wrong: {Z_row.shape}"
    assert Z_col.shape == (D, nmol), f"col-major shape wrong: {Z_col.shape}"
    np.testing.assert_array_equal(
        Z_row.cpu().numpy(),
        Z_col.cpu().numpy().T,
        err_msg="col-major output does not match transpose of row-major output",
    )


def test_rff_gradient_elemental_col_major_matches_row_major() -> None:
    """Col-major gradient output must be the transpose of row-major output (same values)."""
    rng = np.random.default_rng(100)
    nmol, max_atoms, rep_size, nel, D = 5, 4, 6, 3, 12
    N = np.array([4, 2, 3, 4, 1], dtype=np.int32)
    X = rng.normal(size=(nmol, max_atoms, rep_size)).astype(np.float64)
    Q = rng.integers(0, nel, size=(nmol, max_atoms), dtype=np.int32)
    for i, n_i in enumerate(N):
        X[i, n_i:] = 0.0
        Q[i, n_i:] = -1
    dX = rng.normal(size=(nmol, max_atoms, rep_size, max_atoms, 3)).astype(np.float64)
    W = rng.normal(size=(nel, rep_size, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(nel, D)).astype(np.float64)

    X_c = _to_cuda(X)
    dX_c = _to_cuda(dX)
    Q_c, N_c, W_c, b_c = _to_cuda_i32(Q), _to_cuda_i32(N), _to_cuda(W), _to_cuda(b)

    G_row = cuda_rff_features.rff_gradient_elemental(X_c, dX_c, Q_c, N_c, W_c, b_c, 2)
    G_col = cuda_rff_features.rff_gradient_elemental_col_major(X_c, dX_c, Q_c, N_c, W_c, b_c, 2)

    total_naq = int(3 * N.sum())
    assert G_row.shape == (total_naq, D), f"row-major shape wrong: {G_row.shape}"
    assert G_col.shape == (D, total_naq), f"col-major shape wrong: {G_col.shape}"
    np.testing.assert_allclose(
        G_row.cpu().numpy(),
        G_col.cpu().numpy().T,
        rtol=1e-5,
        atol=1e-5,
        err_msg="col-major gradient output does not match transpose of row-major output",
    )


@pytest.mark.slow
def test_rff_gradient_elemental_multi_tile_matches_cpu() -> None:
    """D > D_TILE_DEFAULT (1024) exercises multiple D-tiles in the gradient path.

    Uses the CPU reference to verify the tiled CUDA result is numerically
    equivalent (FP32 atomicAdd ordering may differ, so we use allclose).
    """
    rng = np.random.default_rng(42)
    nmol, max_atoms, rep_size, nel = 6, 4, 8, 2
    D = 2048  # two full tiles of 1024
    N = np.array([4, 2, 3, 4, 1, 3], dtype=np.int32)
    X = rng.normal(size=(nmol, max_atoms, rep_size)).astype(np.float64)
    dX = rng.normal(size=(nmol, max_atoms, rep_size, max_atoms, 3)).astype(np.float64)
    Q = rng.integers(0, nel, size=(nmol, max_atoms), dtype=np.int32)
    for i, n_i in enumerate(N):
        X[i, n_i:] = 0.0
        dX[i, n_i:] = 0.0
        dX[i, :, :, n_i:] = 0.0
        Q[i, n_i:] = -1
    W = rng.normal(size=(nel, rep_size, D)).astype(np.float64)
    b = rng.uniform(0, 2 * np.pi, size=(nel, D)).astype(np.float64)

    ref = rff_gradient_elemental(
        X, dX, _q_list_from_padded(Q, N), W, b
    ).T  # → (total_naq, D) float64

    got = cuda_rff_features.rff_gradient_elemental(
        _to_cuda(X),
        _to_cuda(dX),
        _to_cuda_i32(Q),
        _to_cuda_i32(N),
        _to_cuda(W),
        _to_cuda(b),
        3,
    )
    np.testing.assert_allclose(
        got.cpu().numpy().astype(np.float64),
        ref,
        rtol=2e-4,
        atol=2e-4,
        err_msg="multi-tile gradient does not match CPU reference",
    )
