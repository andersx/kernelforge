"""CUDA FCHL18 matvec parity vs dense full kernel."""

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
    "two_body_scaling": 2.5,
    "two_body_width": 0.1,
    "two_body_power": 4.5,
    "three_body_scaling": 1.5,
    "three_body_width": 3.0,
    "three_body_power": 3.0,
    "cut_start": 1.0,
    "cut_distance": 1e6,
    "fourier_order": 1,
    "use_atm": False,
}
SIGMA = 2.5


def _pad_batch(coords_list, z_list, max_size):
    nm = len(coords_list)
    coords = np.zeros((nm, max_size, 3), dtype=np.float64)
    z = np.zeros((nm, max_size), dtype=np.int32)
    for i, (c, zi) in enumerate(zip(coords_list, z_list, strict=True)):
        na = zi.shape[0]
        coords[i, :na] = c
        z[i, :na] = zi
    return coords, z


def test_cuda_full_matvec_matches_dense():
    coords_A = [WATER_COORDS]
    z_A = [WATER_Z]
    coords_B = [AMMONIA_COORDS, WATER_COORDS]
    z_B = [AMMONIA_Z, WATER_Z]
    max_size = max(max(len(z) for z in z_A), max(len(z) for z in z_B))

    x_A, n_A, nn_A = repr_mod.generate(coords_A, z_A, max_size=max_size)
    x_B, n_B, nn_B = repr_mod.generate(coords_B, z_B, max_size=max_size)
    c_A, zA = _pad_batch(coords_A, z_A, max_size)
    c_B, zB = _pad_batch(coords_B, z_B, max_size)

    dev = torch.device("cuda")
    dtype = torch.float64
    x1 = torch.from_numpy(x_A).to(device=dev, dtype=dtype)
    x2 = torch.from_numpy(x_B).to(device=dev, dtype=dtype)
    n1 = torch.from_numpy(n_A).to(device=dev)
    n2 = torch.from_numpy(n_B).to(device=dev)
    nn1 = torch.from_numpy(nn_A).to(device=dev)
    nn2 = torch.from_numpy(nn_B).to(device=dev)
    coords1 = torch.from_numpy(c_A).to(device=dev, dtype=dtype)
    coords2 = torch.from_numpy(c_B).to(device=dev, dtype=dtype)
    z1 = torch.from_numpy(zA).to(device=dev)
    z2 = torch.from_numpy(zB).to(device=dev)

    D_B = sum(z.shape[0] * 3 for z in z_B)
    alpha_E = torch.randn(2, device=dev, dtype=dtype)
    alpha_F = torch.randn(D_B, device=dev, dtype=dtype)

    K_full = cuda_fchl18_kernel.kernel_gaussian_full(
        x1, x2, n1, n2, nn1, nn2, coords1, z1, coords2, z2, sigma=SIGMA, **KERNEL_ARGS
    )
    alpha = torch.cat([alpha_E, alpha_F])
    y_ref = K_full @ alpha

    E_mv, F_mv = cuda_fchl18_kernel.kernel_gaussian_full_matvec(
        x1,
        x2,
        n1,
        n2,
        nn1,
        nn2,
        coords1,
        z1,
        coords2,
        z2,
        alpha_E,
        alpha_F,
        sigma=SIGMA,
        compute_energy=True,
        **KERNEL_ARGS,
    )

    np.testing.assert_allclose(E_mv.cpu().numpy(), y_ref[:1].cpu().numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(F_mv.cpu().numpy(), y_ref[1:].cpu().numpy(), rtol=1e-10, atol=1e-10)


def test_cuda_full_matvec_matches_cpu():
    coords_A = [WATER_COORDS]
    z_A = [WATER_Z]
    coords_B = [AMMONIA_COORDS]
    z_B = [AMMONIA_Z]
    off_B = sum(z.shape[0] * 3 for z in z_B)
    alpha_E = np.array([0.5])
    alpha_F = np.linspace(0.1, 0.2, off_B)

    E_cpu, F_cpu = cpu_kernel.kernel_gaussian_full_matvec(
        coords_A,
        z_A,
        coords_B,
        z_B,
        alpha_E,
        alpha_F,
        sigma=SIGMA,
        **KERNEL_ARGS,
    )

    max_size = max(len(z) for z in z_A + z_B)
    x_A, n_A, nn_A = repr_mod.generate(coords_A, z_A, max_size=max_size)
    x_B, n_B, nn_B = repr_mod.generate(coords_B, z_B, max_size=max_size)
    c_A, zA = _pad_batch(coords_A, z_A, max_size)
    c_B, zB = _pad_batch(coords_B, z_B, max_size)

    dev = torch.device("cuda")
    E_cuda, F_cuda = cuda_fchl18_kernel.kernel_gaussian_full_matvec(
        torch.from_numpy(x_A).cuda().double(),
        torch.from_numpy(x_B).cuda().double(),
        torch.from_numpy(n_A).cuda(),
        torch.from_numpy(n_B).cuda(),
        torch.from_numpy(nn_A).cuda(),
        torch.from_numpy(nn_B).cuda(),
        torch.from_numpy(c_A).cuda().double(),
        torch.from_numpy(zA).cuda(),
        torch.from_numpy(c_B).cuda().double(),
        torch.from_numpy(zB).cuda(),
        torch.from_numpy(alpha_E).cuda().double(),
        torch.from_numpy(alpha_F).cuda().double(),
        sigma=SIGMA,
        **KERNEL_ARGS,
    )

    np.testing.assert_allclose(E_cuda.cpu().numpy(), E_cpu, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(F_cuda.cpu().numpy(), F_cpu, rtol=1e-10, atol=1e-10)


def test_cuda_full_matvec_fp32_matches_dense():
    coords_A = [WATER_COORDS]
    z_A = [WATER_Z]
    coords_B = [AMMONIA_COORDS, WATER_COORDS]
    z_B = [AMMONIA_Z, WATER_Z]
    max_size = max(max(len(z) for z in z_A), max(len(z) for z in z_B))

    x_A, n_A, nn_A = repr_mod.generate(coords_A, z_A, max_size=max_size)
    x_B, n_B, nn_B = repr_mod.generate(coords_B, z_B, max_size=max_size)
    c_A, zA = _pad_batch(coords_A, z_A, max_size)
    c_B, zB = _pad_batch(coords_B, z_B, max_size)

    dev = torch.device("cuda")
    dtype = torch.float32
    x1 = torch.from_numpy(x_A).to(device=dev, dtype=dtype)
    x2 = torch.from_numpy(x_B).to(device=dev, dtype=dtype)
    n1 = torch.from_numpy(n_A).to(device=dev)
    n2 = torch.from_numpy(n_B).to(device=dev)
    nn1 = torch.from_numpy(nn_A).to(device=dev)
    nn2 = torch.from_numpy(nn_B).to(device=dev)
    coords1 = torch.from_numpy(c_A).to(device=dev, dtype=dtype)
    coords2 = torch.from_numpy(c_B).to(device=dev, dtype=dtype)
    z1 = torch.from_numpy(zA).to(device=dev)
    z2 = torch.from_numpy(zB).to(device=dev)

    D_B = sum(z.shape[0] * 3 for z in z_B)
    alpha_E = torch.randn(2, device=dev, dtype=dtype)
    alpha_F = torch.randn(D_B, device=dev, dtype=dtype)

    K_full = cuda_fchl18_kernel.kernel_gaussian_full(
        x1, x2, n1, n2, nn1, nn2, coords1, z1, coords2, z2, sigma=SIGMA, **KERNEL_ARGS
    )
    y_ref = K_full @ torch.cat([alpha_E, alpha_F])

    E_mv, F_mv = cuda_fchl18_kernel.kernel_gaussian_full_matvec(
        x1,
        x2,
        n1,
        n2,
        nn1,
        nn2,
        coords1,
        z1,
        coords2,
        z2,
        alpha_E,
        alpha_F,
        sigma=SIGMA,
        compute_energy=True,
        **KERNEL_ARGS,
    )
    np.testing.assert_allclose(E_mv.cpu().numpy(), y_ref[:1].cpu().numpy(), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(F_mv.cpu().numpy(), y_ref[1:].cpu().numpy(), rtol=2e-4, atol=2e-4)
