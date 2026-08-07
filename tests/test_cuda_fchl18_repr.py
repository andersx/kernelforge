"""Tests for cuda_fchl18_repr.generate."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

cuda_fchl18_repr = pytest.importorskip(
    "kernelforge.cuda_fchl18_repr",
    reason="cuda_fchl18_repr not built",
)
torch = pytest.importorskip("torch")

import kernelforge.fchl18_repr as cpu_repr  # noqa: E402

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
AMMONIA_COORDS = np.array(
    [
        [0.000, 0.000, 0.116],
        [0.000, 0.939, -0.271],
        [0.813, -0.469, -0.271],
        [-0.813, -0.469, -0.271],
    ],
    dtype=np.float64,
)
AMMONIA_Z = np.array([7, 1, 1, 1], dtype=np.int32)


def test_generate_matches_cpu_f64() -> None:
    coords_list = [WATER_COORDS, AMMONIA_COORDS]
    z_list = [WATER_Z, AMMONIA_Z]
    max_size = 4
    cut = 5.0

    x_cpu, n_cpu, nn_cpu = cpu_repr.generate(
        coords_list, z_list, max_size=max_size, cut_distance=cut
    )

    coords = np.zeros((2, max_size, 3), dtype=np.float64)
    z = np.zeros((2, max_size), dtype=np.int32)
    n = np.array([3, 4], dtype=np.int32)
    coords[0, :3] = WATER_COORDS
    coords[1, :4] = AMMONIA_COORDS
    z[0, :3] = WATER_Z
    z[1, :4] = AMMONIA_Z

    x_gpu, nn_gpu = cuda_fchl18_repr.generate(
        torch.from_numpy(coords).cuda(),
        torch.from_numpy(z).cuda(),
        torch.from_numpy(n).cuda(),
        cut_distance=cut,
    )

    assert x_gpu.shape == (2, max_size, 5, max_size)
    np.testing.assert_array_equal(nn_gpu.cpu().numpy(), nn_cpu)
    np.testing.assert_array_equal(n, n_cpu)
    np.testing.assert_allclose(x_gpu.cpu().numpy(), x_cpu, rtol=1e-12, atol=1e-12)
