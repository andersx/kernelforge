"""Tests for cuda_fchl18_kernel full / _symm / _symm_rfp.

Hessian-exact hyperparams: use_atm=False, cut_start=1.0, inactive cutoff.
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
import kernelforge.kernelmath as kernelmath  # noqa: E402

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

HF_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]], dtype=np.float64)
HF_Z = np.array([1, 9], dtype=np.int32)

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


SIGMA = 2.5
KERNEL_ARGS: _KernelArgs = {
    "two_body_scaling": 2.0,
    "two_body_width": 0.1,
    "two_body_power": 6.0,
    "three_body_scaling": 2.0,
    "three_body_width": 3.0,
    "three_body_power": 3.0,
    "cut_start": 1.0,
    "cut_distance": 1e6,
    "fourier_order": 1,
    "use_atm": False,
}


def _clamp_for_f32(a: np.ndarray) -> np.ndarray:
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
    nm = len(coords_list)
    coords = np.zeros((nm, max_size, 3), dtype=np.float64)
    z = np.zeros((nm, max_size), dtype=np.int32)
    for i, (c, zi) in enumerate(zip(coords_list, z_list, strict=False)):
        coords[i, : c.shape[0], :] = c
        z[i, : zi.shape[0]] = zi
    return coords, z


def _prep_side(coords_list, z_list, dtype):
    max_size = max(len(z) for z in z_list)
    x, n, nn = repr_mod.generate(coords_list, z_list, max_size=max_size, cut_distance=1e6)
    coords_pad, z_pad = _pad_coords_z(coords_list, z_list, max_size)
    return (
        _to_cuda(x, dtype),
        _to_cuda_i32(n),
        _to_cuda_i32(nn),
        _to_cuda(coords_pad, dtype),
        _to_cuda_i32(z_pad),
        int(np.sum(n) * 3),
    )


def _gpu_full(coords_A, z_A, coords_B, z_B, dtype):
    x1, n1, nn1, c1, z1, d_a = _prep_side(coords_A, z_A, dtype)
    x2, n2, nn2, c2, z2, d_b = _prep_side(coords_B, z_B, dtype)
    return cuda_fchl18_kernel.kernel_gaussian_full(
        x1,
        x2,
        n1,
        n2,
        nn1,
        nn2,
        c1,
        z1,
        c2,
        z2,
        sigma=SIGMA,
        **KERNEL_ARGS,
    ), len(coords_A), len(coords_B), d_a, d_b


def test_full_matches_cpu_f64() -> None:
    coords_A = [WATER_COORDS, HF_COORDS]
    z_A = [WATER_Z, HF_Z]
    coords_B = [AMMONIA_COORDS, METHANE_COORDS]
    z_B = [AMMONIA_Z, METHANE_Z]

    K_gpu, n_a, n_b, d_a, d_b = _gpu_full(coords_A, z_A, coords_B, z_B, torch.float64)
    K_cpu = cpu_kernel.kernel_gaussian_full(
        coords_A, z_A, coords_B, z_B, sigma=SIGMA, **KERNEL_ARGS
    )

    assert K_gpu.shape == (n_a + d_a, n_b + d_b)
    np.testing.assert_allclose(K_gpu.cpu().numpy(), K_cpu, rtol=1e-9, atol=1e-11)


def test_full_blocks_match_standalone() -> None:
    coords_A = [WATER_COORDS]
    z_A = [WATER_Z]
    coords_B = [AMMONIA_COORDS, HF_COORDS]
    z_B = [AMMONIA_Z, HF_Z]

    K, n_a, n_b, d_a, d_b = _gpu_full(coords_A, z_A, coords_B, z_B, torch.float64)
    K = K.cpu().numpy()

    x1, n1, nn1, c1, z1, _ = _prep_side(coords_A, z_A, torch.float64)
    x2, n2, nn2, c2, z2, _ = _prep_side(coords_B, z_B, torch.float64)

    EE = cuda_fchl18_kernel.kernel_gaussian(
        x1, x2, n1, n2, nn1, nn2, sigma=SIGMA, **KERNEL_ARGS
    ).cpu().numpy()
    FE = cuda_fchl18_kernel.kernel_gaussian_jacobian(
        x1, x2, n1, n2, nn1, nn2, c1, z1, sigma=SIGMA, **KERNEL_ARGS
    ).cpu().numpy()
    EF = cuda_fchl18_kernel.kernel_gaussian_jacobian_t(
        x2, x1, n2, n1, nn2, nn1, c2, z2, sigma=SIGMA, **KERNEL_ARGS
    ).cpu().numpy()
    FF = cuda_fchl18_kernel.kernel_gaussian_hessian(
        x1, x2, n1, n2, nn1, nn2, c1, z1, c2, z2, sigma=SIGMA, **KERNEL_ARGS
    ).cpu().numpy()

    np.testing.assert_allclose(K[:n_a, :n_b], EE, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(K[:n_a, n_b:], EF, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(K[n_a:, :n_b], FE, rtol=0.0, atol=0.0)
    # Hessian uses atomics; bit-identical replay is not guaranteed across calls.
    np.testing.assert_allclose(K[n_a:, n_b:], FF, rtol=1e-14, atol=1e-14)
    assert d_a == FE.shape[0]
    assert d_b == FF.shape[1]


def test_full_symm_matches_cpu_and_asym() -> None:
    coords = [WATER_COORDS, HF_COORDS, AMMONIA_COORDS]
    zs = [WATER_Z, HF_Z, AMMONIA_Z]

    x, n, nn, c, z, D = _prep_side(coords, zs, torch.float64)
    nm = len(coords)

    K_symm = cuda_fchl18_kernel.kernel_gaussian_full_symm(
        x, n, nn, c, z, sigma=SIGMA, **KERNEL_ARGS
    ).cpu().numpy()
    K_asym, _, _, _, _ = _gpu_full(coords, zs, coords, zs, torch.float64)
    K_cpu = cpu_kernel.kernel_gaussian_full_symm(coords, zs, sigma=SIGMA, **KERNEL_ARGS)

    assert K_symm.shape == (nm + D, nm + D)
    np.testing.assert_allclose(K_symm, K_symm.T, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(K_symm, K_asym.cpu().numpy(), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(K_symm, K_cpu, rtol=1e-9, atol=1e-11)


def test_full_symm_rfp_unpacks() -> None:
    coords = [WATER_COORDS, HF_COORDS]
    zs = [WATER_Z, HF_Z]

    x, n, nn, c, z, D = _prep_side(coords, zs, torch.float64)
    nm = len(coords)
    BIG = nm + D

    K_symm = cuda_fchl18_kernel.kernel_gaussian_full_symm(
        x, n, nn, c, z, sigma=SIGMA, **KERNEL_ARGS
    ).cpu().numpy()
    rfp = cuda_fchl18_kernel.kernel_gaussian_full_symm_rfp(
        x, n, nn, c, z, sigma=SIGMA, **KERNEL_ARGS
    ).cpu().numpy()

    assert rfp.shape == (BIG * (BIG + 1) // 2,)
    K_unpacked = kernelmath.rfp_to_full(rfp.astype(np.float64), BIG, uplo="L", transr="N")
    np.testing.assert_allclose(K_unpacked, K_symm, rtol=1e-9, atol=1e-11)


def test_full_fp32_close() -> None:
    coords_A = [WATER_COORDS]
    z_A = [WATER_Z]
    coords_B = [AMMONIA_COORDS]
    z_B = [AMMONIA_Z]

    K32, _, _, _, _ = _gpu_full(coords_A, z_A, coords_B, z_B, torch.float32)
    K64, _, _, _, _ = _gpu_full(coords_A, z_A, coords_B, z_B, torch.float64)

    np.testing.assert_allclose(
        K32.cpu().numpy(), K64.cpu().numpy(), rtol=1e-4, atol=1e-5
    )
