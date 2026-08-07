"""Tests for cuda_fchl18_kernel.kernel_gaussian_gradient."""

from __future__ import annotations

import shutil
import subprocess
from typing import TypedDict

import numpy as np
import pytest

cuda_fchl18_kernel = pytest.importorskip(
    "kernelforge.cuda_fchl18_kernel",
    reason="cuda_fchl18_kernel not built",
)
torch = pytest.importorskip("torch")

import kernelforge.fchl18_kernel as cpu_kernel  # noqa: E402
import kernelforge.fchl18_repr as repr_mod  # noqa: E402

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

WATER_COORDS = np.array(
    [[0.000, 0.000, 0.119], [0.000, 0.757, -0.477], [0.000, -0.757, -0.477]],
    dtype=np.float64,
)
WATER_Z = np.array([8, 1, 1], dtype=np.int32)
METHANE_COORDS = np.array(
    [
        [0.000, 0.000, 0.000],
        [0.629, 0.629, 0.629],
        [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629],
        [0.629, -0.629, -0.629],
    ],
    dtype=np.float64,
)
METHANE_Z = np.array([6, 1, 1, 1, 1], dtype=np.int32)


class _KernelArgs(TypedDict):
    two_body_scaling: float
    two_body_width: float
    two_body_power: float
    three_body_scaling: float
    three_body_width: float
    three_body_power: float
    cut_start: float
    cut_distance: float
    fourier_order: int
    use_atm: bool


KERNEL_ARGS: _KernelArgs = {
    "two_body_scaling": 2.0,
    "two_body_width": 0.1,
    "two_body_power": 6.0,
    "three_body_scaling": 2.0,
    "three_body_width": 3.0,
    "three_body_power": 3.0,
    "cut_start": 0.5,
    "cut_distance": 1e6,
    "fourier_order": 2,
    "use_atm": True,
}
SIGMA = 2.5


def _to_cuda(a: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        a = np.where(np.abs(a) > np.finfo(np.float32).max, np.sign(a) * 1e30, a)
        np_dtype = np.float32
    else:
        np_dtype = np.float64
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np_dtype))).cuda()


def _to_i32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.int32))).cuda()


def test_gradient_matches_cpu_and_jacobian() -> None:
    coords_a = [WATER_COORDS]
    z_a = [WATER_Z]
    coords_b = [METHANE_COORDS, WATER_COORDS]
    z_b = [METHANE_Z, WATER_Z]
    max_a = 3
    max_b = 5
    x1, n1, nn1 = repr_mod.generate(coords_a, z_a, max_size=max_a, cut_distance=1e6)
    x2, n2, nn2 = repr_mod.generate(coords_b, z_b, max_size=max_b, cut_distance=1e6)
    coords1 = np.zeros((1, max_a, 3), dtype=np.float64)
    z1 = np.zeros((1, max_a), dtype=np.int32)
    coords1[0, :3] = WATER_COORDS
    z1[0, :3] = WATER_Z

    G_gpu = (
        cuda_fchl18_kernel.kernel_gaussian_gradient(
            _to_cuda(x1, torch.float64),
            _to_cuda(x2, torch.float64),
            _to_i32(n1),
            _to_i32(n2),
            _to_i32(nn1),
            _to_i32(nn2),
            _to_cuda(coords1, torch.float64),
            _to_i32(z1),
            sigma=SIGMA,
            **KERNEL_ARGS,
        )
        .cpu()
        .numpy()
    )
    G_cpu = cpu_kernel.kernel_gaussian_gradient(
        WATER_COORDS, WATER_Z, x2, n2, nn2, sigma=SIGMA, **KERNEL_ARGS
    )
    J_gpu = (
        cuda_fchl18_kernel.kernel_gaussian_jacobian(
            _to_cuda(x1, torch.float64),
            _to_cuda(x2, torch.float64),
            _to_i32(n1),
            _to_i32(n2),
            _to_i32(nn1),
            _to_i32(nn2),
            _to_cuda(coords1, torch.float64),
            _to_i32(z1),
            sigma=SIGMA,
            **KERNEL_ARGS,
        )
        .cpu()
        .numpy()
    )

    assert G_gpu.shape == (3, 3, 2)
    np.testing.assert_allclose(G_gpu, G_cpu, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(G_gpu.reshape(9, 2), J_gpu, rtol=0.0, atol=0.0)
