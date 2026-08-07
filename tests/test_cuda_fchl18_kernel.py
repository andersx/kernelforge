"""Tests for cuda_fchl18_kernel.

These tests are skipped when:
  - cuda_fchl18_kernel was not built (no CUDA + PyTorch at build time)
  - No NVIDIA GPU is detected
"""

from __future__ import annotations

import shutil
import subprocess

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
        and subprocess.run(
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

KERNEL_ARGS = {
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


def _to_cuda_f32(a: np.ndarray) -> torch.Tensor:
    a32 = np.asarray(a, dtype=np.float64)
    a32 = np.where(a32 > np.finfo(np.float32).max, np.float32(1e30), a32)
    return torch.from_numpy(np.ascontiguousarray(a32.astype(np.float32))).cuda()


def _to_cuda_i32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a.astype(np.int32))).cuda()


def test_kernel_gaussian_shape_dtype() -> None:
    coords1 = [WATER_COORDS, AMMONIA_COORDS]
    z1 = [WATER_Z, AMMONIA_Z]
    coords2 = [HF_COORDS, METHANE_COORDS, WATER_COORDS]
    z2 = [HF_Z, METHANE_Z, WATER_Z]

    max_size1 = max(len(z) for z in z1)
    max_size2 = max(len(z) for z in z2)
    x1, n1, nn1 = repr_mod.generate(coords1, z1, max_size=max_size1, cut_distance=1e6)
    x2, n2, nn2 = repr_mod.generate(coords2, z2, max_size=max_size2, cut_distance=1e6)

    K = cuda_fchl18_kernel.kernel_gaussian(
        _to_cuda_f32(x1),
        _to_cuda_f32(x2),
        _to_cuda_i32(n1),
        _to_cuda_i32(n2),
        _to_cuda_i32(nn1),
        _to_cuda_i32(nn2),
        sigma=2.5,
        **KERNEL_ARGS,
    )

    assert K.shape == (len(coords1), len(coords2))
    assert K.dtype == torch.float32
    assert K.device.type == "cuda"


def test_kernel_gaussian_matches_cpu() -> None:
    coords1 = [WATER_COORDS, AMMONIA_COORDS, HF_COORDS]
    z1 = [WATER_Z, AMMONIA_Z, HF_Z]
    coords2 = [METHANE_COORDS, WATER_COORDS]
    z2 = [METHANE_Z, WATER_Z]

    max_size1 = max(len(z) for z in z1)
    max_size2 = max(len(z) for z in z2)
    x1, n1, nn1 = repr_mod.generate(coords1, z1, max_size=max_size1, cut_distance=1e6)
    x2, n2, nn2 = repr_mod.generate(coords2, z2, max_size=max_size2, cut_distance=1e6)

    K_cpu = cpu_kernel.kernel_gaussian(x1, x2, n1, n2, nn1, nn2, sigma=2.5, **KERNEL_ARGS)
    K_gpu = cuda_fchl18_kernel.kernel_gaussian(
        _to_cuda_f32(x1),
        _to_cuda_f32(x2),
        _to_cuda_i32(n1),
        _to_cuda_i32(n2),
        _to_cuda_i32(nn1),
        _to_cuda_i32(nn2),
        sigma=2.5,
        **KERNEL_ARGS,
    )

    np.testing.assert_allclose(
        K_gpu.cpu().numpy().astype(np.float64),
        K_cpu,
        rtol=2e-4,
        atol=2e-4,
    )


def test_kernel_gaussian_sigma_sensitivity() -> None:
    coords1 = [WATER_COORDS, AMMONIA_COORDS]
    z1 = [WATER_Z, AMMONIA_Z]
    coords2 = [HF_COORDS, METHANE_COORDS]
    z2 = [HF_Z, METHANE_Z]

    max_size1 = max(len(z) for z in z1)
    max_size2 = max(len(z) for z in z2)
    x1, n1, nn1 = repr_mod.generate(coords1, z1, max_size=max_size1, cut_distance=1e6)
    x2, n2, nn2 = repr_mod.generate(coords2, z2, max_size=max_size2, cut_distance=1e6)

    x1_cuda = _to_cuda_f32(x1)
    x2_cuda = _to_cuda_f32(x2)
    n1_cuda = _to_cuda_i32(n1)
    n2_cuda = _to_cuda_i32(n2)
    nn1_cuda = _to_cuda_i32(nn1)
    nn2_cuda = _to_cuda_i32(nn2)

    K1 = cuda_fchl18_kernel.kernel_gaussian(
        x1_cuda, x2_cuda, n1_cuda, n2_cuda, nn1_cuda, nn2_cuda, sigma=1.0, **KERNEL_ARGS
    )
    K2 = cuda_fchl18_kernel.kernel_gaussian(
        x1_cuda, x2_cuda, n1_cuda, n2_cuda, nn1_cuda, nn2_cuda, sigma=5.0, **KERNEL_ARGS
    )

    assert not torch.allclose(K1, K2)


def test_kernel_gaussian_symm_matches_cpu() -> None:
    coords = [WATER_COORDS, AMMONIA_COORDS, HF_COORDS, METHANE_COORDS]
    z = [WATER_Z, AMMONIA_Z, HF_Z, METHANE_Z]
    max_size = max(len(zi) for zi in z)
    x, n, nn = repr_mod.generate(coords, z, max_size=max_size, cut_distance=1e6)

    K_cpu = cpu_kernel.kernel_gaussian_symm(x, n, nn, sigma=2.5, **KERNEL_ARGS)
    K_gpu = cuda_fchl18_kernel.kernel_gaussian_symm(
        _to_cuda_f32(x),
        _to_cuda_i32(n),
        _to_cuda_i32(nn),
        sigma=2.5,
        **KERNEL_ARGS,
    )

    assert K_gpu.shape == (len(coords), len(coords))
    assert K_gpu.dtype == torch.float32
    np.testing.assert_allclose(
        K_gpu.cpu().numpy().astype(np.float64),
        K_cpu,
        rtol=2e-4,
        atol=2e-4,
    )
    # Symmetric by construction
    Kg = K_gpu.cpu().numpy()
    np.testing.assert_allclose(Kg, Kg.T, rtol=0, atol=0)


def test_kernel_gaussian_symm_matches_rect() -> None:
    coords = [WATER_COORDS, AMMONIA_COORDS, METHANE_COORDS]
    z = [WATER_Z, AMMONIA_Z, METHANE_Z]
    max_size = max(len(zi) for zi in z)
    x, n, nn = repr_mod.generate(coords, z, max_size=max_size, cut_distance=1e6)

    x_c = _to_cuda_f32(x)
    n_c = _to_cuda_i32(n)
    nn_c = _to_cuda_i32(nn)

    K_symm = cuda_fchl18_kernel.kernel_gaussian_symm(x_c, n_c, nn_c, sigma=2.5, **KERNEL_ARGS)
    K_rect = cuda_fchl18_kernel.kernel_gaussian(
        x_c, x_c, n_c, n_c, nn_c, nn_c, sigma=2.5, **KERNEL_ARGS
    )

    np.testing.assert_allclose(
        K_symm.cpu().numpy(),
        K_rect.cpu().numpy(),
        rtol=2e-4,
        atol=2e-4,
    )


def test_kernel_gaussian_symm_rfp_matches_dense() -> None:
    import kernelforge.kernelmath as kernelmath

    coords = [WATER_COORDS, AMMONIA_COORDS, HF_COORDS, METHANE_COORDS]
    z = [WATER_Z, AMMONIA_Z, HF_Z, METHANE_Z]
    nm = len(coords)
    max_size = max(len(zi) for zi in z)
    x, n, nn = repr_mod.generate(coords, z, max_size=max_size, cut_distance=1e6)

    x_c = _to_cuda_f32(x)
    n_c = _to_cuda_i32(n)
    nn_c = _to_cuda_i32(nn)

    K_dense = cuda_fchl18_kernel.kernel_gaussian_symm(x_c, n_c, nn_c, sigma=2.5, **KERNEL_ARGS)
    K_rfp = cuda_fchl18_kernel.kernel_gaussian_symm_rfp(x_c, n_c, nn_c, sigma=2.5, **KERNEL_ARGS)

    assert K_rfp.shape == (nm * (nm + 1) // 2,)
    assert K_rfp.dtype == torch.float32

    # Packed with TRANSR='N', UPLO='U' (rfp_index_upper_N). Due to C-order
    # UPLO swap in kernelmath.rfp_to_full, unpack with uplo='L'.
    K_unpacked = kernelmath.rfp_to_full(
        K_rfp.cpu().numpy().astype(np.float64), nm, uplo="L", transr="N"
    )
    K_unpacked = np.tril(K_unpacked) + np.triu(K_unpacked.T, 1)

    np.testing.assert_allclose(
        K_unpacked,
        K_dense.cpu().numpy().astype(np.float64),
        rtol=2e-4,
        atol=2e-4,
    )


def test_kernel_gaussian_symm_rfp_odd_even_nm() -> None:
    """RFP layout differs for even/odd n; cover both with 3 and 4 molecules."""
    import kernelforge.kernelmath as kernelmath

    for coords, zs in (
        ([WATER_COORDS, AMMONIA_COORDS, HF_COORDS], [WATER_Z, AMMONIA_Z, HF_Z]),
        (
            [WATER_COORDS, AMMONIA_COORDS, HF_COORDS, METHANE_COORDS],
            [WATER_Z, AMMONIA_Z, HF_Z, METHANE_Z],
        ),
    ):
        nm = len(coords)
        max_size = max(len(zi) for zi in zs)
        x, n, nn = repr_mod.generate(coords, zs, max_size=max_size, cut_distance=1e6)
        x_c = _to_cuda_f32(x)
        n_c = _to_cuda_i32(n)
        nn_c = _to_cuda_i32(nn)

        K_dense = cuda_fchl18_kernel.kernel_gaussian_symm(x_c, n_c, nn_c, sigma=2.5, **KERNEL_ARGS)
        K_rfp = cuda_fchl18_kernel.kernel_gaussian_symm_rfp(
            x_c, n_c, nn_c, sigma=2.5, **KERNEL_ARGS
        )
        K_unpacked = kernelmath.rfp_to_full(
            K_rfp.cpu().numpy().astype(np.float64), nm, uplo="L", transr="N"
        )
        K_unpacked = np.tril(K_unpacked) + np.triu(K_unpacked.T, 1)
        np.testing.assert_allclose(
            K_unpacked,
            K_dense.cpu().numpy().astype(np.float64),
            rtol=2e-4,
            atol=2e-4,
        )
