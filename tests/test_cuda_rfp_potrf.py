"""Tests for cuda_global_kernels.rfp_potrf / rfp_potrs UPLO variants."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

cuda_global_kernels = pytest.importorskip(
    "kernelforge.cuda_global_kernels",
    reason="cuda_global_kernels not built",
)
torch = pytest.importorskip("torch", reason="PyTorch not installed")

from kernelforge import kernelmath  # noqa: E402

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

pytestmark = pytest.mark.skipif(not _gpu_ok, reason="No NVIDIA GPU")


def _spd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64)
    return (A @ A.T + n * np.eye(n)).astype(np.float64)


@pytest.mark.parametrize("n", [7, 8, 16])
@pytest.mark.parametrize("solve_uplo", ["L", "U"])
def test_rfp_potrf_potrs_matches_numpy(n: int, solve_uplo: str) -> None:
    """GPU RFP Cholesky agrees with numpy for both UPLO conventions.

    ``kernelmath.full_to_rfp`` uses the C-order UPLO swap (same as LAPACK
    row-major): pack with the opposite letter of the GPU ``uplo`` argument.
    FCHL18 kernels write true UPLO=U packing → use ``rfp_potrf(..., uplo="U")``.
    Local/global CUDA kernels write UPLO=L → ``uplo="L"``.
    """
    K = _spd(n)
    y = np.arange(1, n + 1, dtype=np.float64)
    ref = np.linalg.solve(K + 1e-3 * np.eye(n), y)

    # C-order UPLO swap: pack opposite of GPU solver uplo
    pack_uplo = "U" if solve_uplo == "L" else "L"
    K_rfp = kernelmath.full_to_rfp(K, uplo=pack_uplo, transr="N").astype(np.float32)
    K_cuda = torch.from_numpy(np.ascontiguousarray(K_rfp)).cuda()
    rhs = torch.from_numpy(y.astype(np.float32)).cuda().unsqueeze(-1).contiguous()

    info = cuda_global_kernels.rfp_potrf(K_cuda, n, l2=1e-3, uplo=solve_uplo)
    assert info == 0
    cuda_global_kernels.rfp_potrs(K_cuda, rhs, uplo=solve_uplo)
    sol = rhs.squeeze(-1).cpu().numpy().astype(np.float64)
    np.testing.assert_allclose(sol, ref, rtol=2e-3, atol=2e-3)
