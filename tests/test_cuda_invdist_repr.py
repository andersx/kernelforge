"""Tests for cuda_invdist_repr — GPU batched inverse-distance representation.

Skipped when:
  - cuda_invdist_repr was not built (no CUDA + PyTorch at build time)
  - No NVIDIA GPU is detected
  - PyTorch is not installed

Run with:
    uv run pytest tests/test_cuda_invdist_repr.py -v
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------
cuda_invdist_repr = pytest.importorskip(
    "kernelforge.cuda_invdist_repr",
    reason="cuda_invdist_repr not built (requires CUDA + PyTorch at build time)",
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

pytestmark = pytest.mark.skipif(
    not _gpu_ok,
    reason="No NVIDIA GPU detected",
)

# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------
from kernelforge import invdist_repr  # noqa: E402


def _cpu_invdist_batch(
    coords_np: np.ndarray,  # (nm, n_atoms, 3) float64
    eps: float = 1e-6,
) -> np.ndarray:
    """CPU reference: loop over molecules, call invdist_repr per molecule."""
    nm = coords_np.shape[0]
    results = []
    for m in range(nm):
        x = invdist_repr.inverse_distance_upper(coords_np[m].astype(np.float64), eps)
        results.append(x)
    return np.stack(results, axis=0)  # (nm, M)


def _cpu_jacobian_batch(
    coords_np: np.ndarray,  # (nm, n_atoms, 3) float64
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference: loop, return stacked X (nm,M) and dX (nm,D,M)."""
    nm = coords_np.shape[0]
    X_list = []
    dX_list = []
    for m in range(nm):
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(
            coords_np[m].astype(np.float64), eps
        )
        X_list.append(x)
        dX_list.append(dx)
    return np.stack(X_list, axis=0), np.stack(dX_list, axis=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _rand_coords(nm: int, n_atoms: int) -> np.ndarray:
    """Random float64 coords (nm, n_atoms, 3), atom separations ~ 1-2 Å."""
    return RNG.uniform(0.5, 3.0, size=(nm, n_atoms, 3))


def _to_cuda_f32(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).cuda()


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_atoms", [3, 5, 9])
@pytest.mark.parametrize("nm", [1, 4, 16])
def test_forward_shape(nm: int, n_atoms: int) -> None:
    coords_np = _rand_coords(nm, n_atoms)
    coords_cuda = _to_cuda_f32(coords_np)
    M = n_atoms * (n_atoms - 1) // 2
    X = cuda_invdist_repr.inverse_distance_upper(coords_cuda, n_atoms)
    assert X.shape == (nm, M), f"Expected ({nm}, {M}), got {X.shape}"
    assert X.dtype == torch.float32
    assert X.is_cuda


@pytest.mark.parametrize("n_atoms", [3, 5, 9])
@pytest.mark.parametrize("nm", [1, 4, 16])
def test_jacobian_shape(nm: int, n_atoms: int) -> None:
    coords_np = _rand_coords(nm, n_atoms)
    coords_cuda = _to_cuda_f32(coords_np)
    M = n_atoms * (n_atoms - 1) // 2
    D = 3 * n_atoms
    X, dX = cuda_invdist_repr.inverse_distance_upper_and_jacobian(coords_cuda, n_atoms)
    assert X.shape == (nm, M), f"X shape: expected ({nm},{M}), got {X.shape}"
    assert dX.shape == (nm, D, M), f"dX shape: expected ({nm},{D},{M}), got {dX.shape}"
    assert X.dtype == torch.float32
    assert dX.dtype == torch.float32
    assert X.is_cuda
    assert dX.is_cuda


# ---------------------------------------------------------------------------
# Numerical accuracy vs CPU (float64 reference, float32 GPU)
# ---------------------------------------------------------------------------

_FWD_ATOL = 1e-5  # float32 vs float64 forward


@pytest.mark.parametrize("n_atoms", [3, 6, 9])
def test_forward_vs_cpu(n_atoms: int) -> None:
    nm = 8
    eps = 1e-6
    coords_np = _rand_coords(nm, n_atoms)
    X_cpu = _cpu_invdist_batch(coords_np, eps=eps)  # (nm, M) float64

    coords_cuda = _to_cuda_f32(coords_np)
    X_gpu = (
        cuda_invdist_repr.inverse_distance_upper(coords_cuda, n_atoms, eps).cpu().numpy()
    )  # (nm, M) float32

    np.testing.assert_allclose(
        X_gpu.astype(np.float64),
        X_cpu,
        atol=_FWD_ATOL,
        rtol=0,
        err_msg=f"Forward invdist mismatch for n_atoms={n_atoms}",
    )


_JAC_ATOL = 1e-4  # float32 Jacobian vs float64 (larger tolerance due to divisions)


@pytest.mark.parametrize("n_atoms", [3, 6, 9])
def test_jacobian_vs_cpu(n_atoms: int) -> None:
    nm = 6
    eps = 1e-6
    coords_np = _rand_coords(nm, n_atoms)
    X_cpu, dX_cpu = _cpu_jacobian_batch(coords_np, eps=eps)

    coords_cuda = _to_cuda_f32(coords_np)
    X_gpu, dX_gpu = cuda_invdist_repr.inverse_distance_upper_and_jacobian(coords_cuda, n_atoms, eps)
    X_gpu = X_gpu.cpu().numpy().astype(np.float64)
    dX_gpu = dX_gpu.cpu().numpy().astype(np.float64)

    np.testing.assert_allclose(
        X_gpu,
        X_cpu,
        atol=_FWD_ATOL,
        rtol=0,
        err_msg=f"X mismatch in jacobian call, n_atoms={n_atoms}",
    )
    np.testing.assert_allclose(
        dX_gpu,
        dX_cpu,
        atol=_JAC_ATOL,
        rtol=0,
        err_msg=f"dX mismatch, n_atoms={n_atoms}",
    )


# ---------------------------------------------------------------------------
# Jacobian consistency: forward call X == jacobian-call X
# ---------------------------------------------------------------------------


def test_forward_jacobian_x_consistent() -> None:
    n_atoms = 7
    nm = 5
    coords_np = _rand_coords(nm, n_atoms)
    coords_cuda = _to_cuda_f32(coords_np)

    X_fwd = cuda_invdist_repr.inverse_distance_upper(coords_cuda, n_atoms)
    X_jac, _ = cuda_invdist_repr.inverse_distance_upper_and_jacobian(coords_cuda, n_atoms)
    np.testing.assert_allclose(
        X_fwd.cpu().numpy(),
        X_jac.cpu().numpy(),
        atol=0,
        rtol=0,
        err_msg="X from forward != X from jacobian call",
    )


# ---------------------------------------------------------------------------
# Batch consistency: single vs batched
# ---------------------------------------------------------------------------


def test_batch_vs_single() -> None:
    """Each molecule in a batch must equal the single-molecule result."""
    n_atoms = 5
    nm = 8
    eps = 1e-6
    coords_np = _rand_coords(nm, n_atoms)

    coords_batch = _to_cuda_f32(coords_np)
    X_batch, dX_batch = cuda_invdist_repr.inverse_distance_upper_and_jacobian(
        coords_batch, n_atoms, eps
    )

    for m in range(nm):
        coords_single = _to_cuda_f32(coords_np[m : m + 1])
        X_single, dX_single = cuda_invdist_repr.inverse_distance_upper_and_jacobian(
            coords_single, n_atoms, eps
        )
        np.testing.assert_allclose(
            X_batch[m].cpu().numpy(),
            X_single[0].cpu().numpy(),
            atol=0,
            rtol=0,
            err_msg=f"X batch vs single mismatch at m={m}",
        )
        np.testing.assert_allclose(
            dX_batch[m].cpu().numpy(),
            dX_single[0].cpu().numpy(),
            atol=0,
            rtol=0,
            err_msg=f"dX batch vs single mismatch at m={m}",
        )


# ---------------------------------------------------------------------------
# Numerical Jacobian check (finite differences on GPU output vs GPU Jacobian)
# ---------------------------------------------------------------------------


def test_numerical_jacobian() -> None:
    """Verify GPU Jacobian against finite differences on GPU forward pass."""
    n_atoms = 4
    nm = 1
    eps = 1e-6
    h = 1e-3  # finite-difference step in float64 coords, cast to float32

    rng = np.random.default_rng(0)
    coords_np = rng.uniform(0.5, 2.5, size=(nm, n_atoms, 3))

    # Analytical Jacobian
    coords_cuda = _to_cuda_f32(coords_np)
    _, dX_gpu = cuda_invdist_repr.inverse_distance_upper_and_jacobian(coords_cuda, n_atoms, eps)
    dX_ana = dX_gpu[0].cpu().numpy()  # (D, M)

    D = 3 * n_atoms
    M = n_atoms * (n_atoms - 1) // 2

    dX_fd = np.zeros((D, M), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):
            c_p = coords_np.copy()
            c_m = coords_np.copy()
            c_p[0, a, d] += h
            c_m[0, a, d] -= h
            X_p = (
                cuda_invdist_repr.inverse_distance_upper(_to_cuda_f32(c_p), n_atoms, eps)[0]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            X_m = (
                cuda_invdist_repr.inverse_distance_upper(_to_cuda_f32(c_m), n_atoms, eps)[0]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            dX_fd[3 * a + d] = (X_p - X_m) / (2 * h)

    np.testing.assert_allclose(
        dX_ana.astype(np.float64),
        dX_fd,
        atol=5e-4,  # float32 analytic vs float64 FD
        rtol=0,
        err_msg="Analytical GPU Jacobian disagrees with finite differences",
    )
