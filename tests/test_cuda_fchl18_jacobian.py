"""Tests for cuda_fchl18_kernel.kernel_gaussian_jacobian.

These tests are skipped when:
  - cuda_fchl18_kernel was not built (no CUDA + PyTorch at build time)
  - No NVIDIA GPU is detected
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TypedDict

import numpy as np
import pytest

cuda_fchl18_kernel = pytest.importorskip(
    "kernelforge.cuda_fchl18_kernel",
    reason="cuda_fchl18_kernel not built (requires CUDA + PyTorch at build time)",
)

torch = pytest.importorskip("torch", reason="PyTorch not installed")

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

pytestmark = pytest.mark.skipif(not _gpu_ok, reason="No NVIDIA GPU detected by nvidia-smi")

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

HF_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]], dtype=np.float64)
HF_Z = np.array([1, 9], dtype=np.int32)


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


SIGMA = 2.5
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


def _clamp_for_f32(a: np.ndarray) -> np.ndarray:
    """The representation pads unused slots with huge sentinels; clip to fp32 range."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.abs(a) > np.finfo(np.float32).max, np.sign(a) * 1e30, a)


def _to_cuda(a: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        a = _clamp_for_f32(a)
        np_dtype = np.float32
    else:
        np_dtype = np.float64
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np_dtype))).cuda()


def _to_cuda_i32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.int32))).cuda()


def _pad_coords_z(coords_list, z_list, max_size):
    """Pad per-molecule coords/charges into dense (nm, max_size, *) arrays."""
    nm = len(coords_list)
    coords = np.zeros((nm, max_size, 3), dtype=np.float64)
    z = np.zeros((nm, max_size), dtype=np.int32)
    for i, (c, zi) in enumerate(zip(coords_list, z_list, strict=False)):
        coords[i, : c.shape[0], :] = c
        z[i, : zi.shape[0]] = zi
    return coords, z


def _gpu_jacobian(coords1, z1, coords2, z2, dtype):
    max_size1 = max(len(z) for z in z1)
    max_size2 = max(len(z) for z in z2)
    x1, n1, nn1 = repr_mod.generate(coords1, z1, max_size=max_size1, cut_distance=1e6)
    x2, n2, nn2 = repr_mod.generate(coords2, z2, max_size=max_size2, cut_distance=1e6)
    coords1_pad, z1_pad = _pad_coords_z(coords1, z1, max_size1)

    return cuda_fchl18_kernel.kernel_gaussian_jacobian(
        _to_cuda(x1, dtype),
        _to_cuda(x2, dtype),
        _to_cuda_i32(n1),
        _to_cuda_i32(n2),
        _to_cuda_i32(nn1),
        _to_cuda_i32(nn2),
        _to_cuda(coords1_pad, dtype),
        _to_cuda_i32(z1_pad),
        sigma=SIGMA,
        **KERNEL_ARGS,
    )


def _cpu_jacobian(coords1, z1, coords2, z2):
    max_size2 = max(len(z) for z in z2)
    x2, n2, nn2 = repr_mod.generate(coords2, z2, max_size=max_size2, cut_distance=1e6)
    return cpu_kernel.kernel_gaussian_jacobian(coords1, z1, x2, n2, nn2, sigma=SIGMA, **KERNEL_ARGS)


def test_jacobian_shape_dtype() -> None:
    coords1 = [WATER_COORDS, AMMONIA_COORDS]
    z1 = [WATER_Z, AMMONIA_Z]
    coords2 = [HF_COORDS, METHANE_COORDS, WATER_COORDS]
    z2 = [HF_Z, METHANE_Z, WATER_Z]

    J = _gpu_jacobian(coords1, z1, coords2, z2, torch.float32)

    assert J.shape == ((3 + 4) * 3, len(coords2))
    assert J.dtype == torch.float32
    assert J.device.type == "cuda"


def test_jacobian_matches_cpu_f64_single() -> None:
    coords1 = [WATER_COORDS]
    z1 = [WATER_Z]
    coords2 = [METHANE_COORDS, WATER_COORDS]
    z2 = [METHANE_Z, WATER_Z]

    J_gpu = _gpu_jacobian(coords1, z1, coords2, z2, torch.float64).cpu().numpy()
    J_cpu = _cpu_jacobian(coords1, z1, coords2, z2)

    np.testing.assert_allclose(J_gpu, J_cpu, rtol=1e-9, atol=1e-11)


def test_jacobian_matches_cpu_f64_batch() -> None:
    """Mixed query sizes exercise the per-molecule row offsets."""
    coords1 = [WATER_COORDS, HF_COORDS, AMMONIA_COORDS]
    z1 = [WATER_Z, HF_Z, AMMONIA_Z]
    coords2 = [AMMONIA_COORDS, METHANE_COORDS, WATER_COORDS]
    z2 = [AMMONIA_Z, METHANE_Z, WATER_Z]

    J_gpu = _gpu_jacobian(coords1, z1, coords2, z2, torch.float64).cpu().numpy()
    J_cpu = _cpu_jacobian(coords1, z1, coords2, z2)

    assert J_gpu.shape == ((3 + 2 + 4) * 3, 3)
    np.testing.assert_allclose(J_gpu, J_cpu, rtol=1e-9, atol=1e-11)


def test_jacobian_matches_cpu_f64_no_atm() -> None:
    coords1 = [METHANE_COORDS]
    z1 = [METHANE_Z]
    coords2 = [WATER_COORDS, AMMONIA_COORDS]
    z2 = [WATER_Z, AMMONIA_Z]

    args = KERNEL_ARGS.copy()
    args["use_atm"] = False

    max_size1 = len(z1[0])
    max_size2 = max(len(z) for z in z2)
    x1, n1, nn1 = repr_mod.generate(coords1, z1, max_size=max_size1, cut_distance=1e6)
    x2, n2, nn2 = repr_mod.generate(coords2, z2, max_size=max_size2, cut_distance=1e6)
    coords1_pad, z1_pad = _pad_coords_z(coords1, z1, max_size1)

    J_gpu = (
        cuda_fchl18_kernel.kernel_gaussian_jacobian(
            _to_cuda(x1, torch.float64),
            _to_cuda(x2, torch.float64),
            _to_cuda_i32(n1),
            _to_cuda_i32(n2),
            _to_cuda_i32(nn1),
            _to_cuda_i32(nn2),
            _to_cuda(coords1_pad, torch.float64),
            _to_cuda_i32(z1_pad),
            sigma=SIGMA,
            **args,
        )
        .cpu()
        .numpy()
    )
    J_cpu = cpu_kernel.kernel_gaussian_jacobian(coords1, z1, x2, n2, nn2, sigma=SIGMA, **args)

    np.testing.assert_allclose(J_gpu, J_cpu, rtol=1e-9, atol=1e-11)


def test_jacobian_matches_cpu_f32() -> None:
    coords1 = [WATER_COORDS, AMMONIA_COORDS]
    z1 = [WATER_Z, AMMONIA_Z]
    coords2 = [METHANE_COORDS, WATER_COORDS]
    z2 = [METHANE_Z, WATER_Z]

    J_gpu = _gpu_jacobian(coords1, z1, coords2, z2, torch.float32).cpu().numpy()
    J_cpu = _cpu_jacobian(coords1, z1, coords2, z2)

    scale = float(np.max(np.abs(J_cpu)))
    np.testing.assert_allclose(J_gpu.astype(np.float64), J_cpu, rtol=1e-4, atol=1e-5 * scale)


def test_jacobian_matches_numerical_gradient() -> None:
    """Central differences on the CUDA energy kernel reproduce the Jacobian."""
    coords1 = [WATER_COORDS]
    z1 = [WATER_Z]
    coords2 = [AMMONIA_COORDS, METHANE_COORDS]
    z2 = [AMMONIA_Z, METHANE_Z]

    max_size2 = max(len(z) for z in z2)
    x2, n2, nn2 = repr_mod.generate(coords2, z2, max_size=max_size2, cut_distance=1e6)
    x2_c = _to_cuda(x2, torch.float64)
    n2_c = _to_cuda_i32(n2)
    nn2_c = _to_cuda_i32(nn2)

    J = _gpu_jacobian(coords1, z1, coords2, z2, torch.float64).cpu().numpy()

    eps = 1e-5
    na = WATER_COORDS.shape[0]
    J_num = np.zeros_like(J)
    for alpha in range(na):
        for mu in range(3):
            ks = []
            for sign in (+1, -1):
                c = WATER_COORDS.copy()
                c[alpha, mu] += sign * eps
                xp, np_, nnp = repr_mod.generate([c], [WATER_Z], max_size=na, cut_distance=1e6)
                ks.append(
                    cuda_fchl18_kernel.kernel_gaussian(
                        _to_cuda(xp, torch.float64),
                        x2_c,
                        _to_cuda_i32(np_),
                        n2_c,
                        _to_cuda_i32(nnp),
                        nn2_c,
                        sigma=SIGMA,
                        **KERNEL_ARGS,
                    )
                    .cpu()
                    .numpy()[0]
                )
            J_num[alpha * 3 + mu, :] = (ks[0] - ks[1]) / (2.0 * eps)

    np.testing.assert_allclose(J, J_num, rtol=1e-5, atol=1e-7)
