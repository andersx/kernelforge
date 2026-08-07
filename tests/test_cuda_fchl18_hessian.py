"""Tests for cuda_fchl18_kernel.kernel_gaussian_hessian.

These tests are skipped when:
  - cuda_fchl18_kernel was not built (no CUDA + PyTorch at build time)
  - No NVIDIA GPU is detected

The CPU reference (``fchl18_kernel.kernel_gaussian_hessian``) is only exact with
an inactive cutoff and ``use_atm=False``, so the hyperparameters mirror
``tests/test_fchl18_hessian.py`` rather than the Jacobian test defaults.
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


def _gpu_hessian(coords1, z1, coords2, z2, dtype, kernel_args=None):
    args = KERNEL_ARGS if kernel_args is None else kernel_args
    max_size1 = max(len(z) for z in z1)
    max_size2 = max(len(z) for z in z2)
    x1, n1, nn1 = repr_mod.generate(
        coords1, z1, max_size=max_size1, cut_distance=args["cut_distance"]
    )
    x2, n2, nn2 = repr_mod.generate(
        coords2, z2, max_size=max_size2, cut_distance=args["cut_distance"]
    )
    coords1_pad, z1_pad = _pad_coords_z(coords1, z1, max_size1)
    coords2_pad, z2_pad = _pad_coords_z(coords2, z2, max_size2)

    return cuda_fchl18_kernel.kernel_gaussian_hessian(
        _to_cuda(x1, dtype),
        _to_cuda(x2, dtype),
        _to_cuda_i32(n1),
        _to_cuda_i32(n2),
        _to_cuda_i32(nn1),
        _to_cuda_i32(nn2),
        _to_cuda(coords1_pad, dtype),
        _to_cuda_i32(z1_pad),
        _to_cuda(coords2_pad, dtype),
        _to_cuda_i32(z2_pad),
        sigma=SIGMA,
        **args,
    )


def _cpu_hessian(coords1, z1, coords2, z2, kernel_args=None):
    args = KERNEL_ARGS if kernel_args is None else kernel_args
    return cpu_kernel.kernel_gaussian_hessian(coords1, z1, coords2, z2, sigma=SIGMA, **args)


def test_hessian_shape_dtype() -> None:
    coords1 = [WATER_COORDS, AMMONIA_COORDS]
    z1 = [WATER_Z, AMMONIA_Z]
    coords2 = [HF_COORDS, METHANE_COORDS]
    z2 = [HF_Z, METHANE_Z]

    H = _gpu_hessian(coords1, z1, coords2, z2, torch.float32)

    assert H.shape == ((3 + 4) * 3, (2 + 5) * 3)
    assert H.dtype == torch.float32
    assert H.device.type == "cuda"


def test_hessian_matches_cpu_f64_single_pair() -> None:
    coords1 = [WATER_COORDS]
    z1 = [WATER_Z]
    coords2 = [WATER_COORDS]
    z2 = [WATER_Z]

    H_gpu = _gpu_hessian(coords1, z1, coords2, z2, torch.float64).cpu().numpy()
    H_cpu = _cpu_hessian(coords1, z1, coords2, z2)

    np.testing.assert_allclose(H_gpu, H_cpu, rtol=1e-9, atol=1e-11)


def test_hessian_matches_cpu_f64_mixed_elements() -> None:
    """A and B share no central-atom element for some pairs (Zi != Zj branch)."""
    coords1 = [METHANE_COORDS]
    z1 = [METHANE_Z]
    coords2 = [HF_COORDS]
    z2 = [HF_Z]

    H_gpu = _gpu_hessian(coords1, z1, coords2, z2, torch.float64).cpu().numpy()
    H_cpu = _cpu_hessian(coords1, z1, coords2, z2)

    np.testing.assert_allclose(H_gpu, H_cpu, rtol=1e-9, atol=1e-11)


def test_hessian_matches_cpu_f64_batch() -> None:
    """Mixed molecule sizes exercise the per-molecule row/column offsets."""
    coords1 = [WATER_COORDS, HF_COORDS, AMMONIA_COORDS]
    z1 = [WATER_Z, HF_Z, AMMONIA_Z]
    coords2 = [AMMONIA_COORDS, METHANE_COORDS]
    z2 = [AMMONIA_Z, METHANE_Z]

    H_gpu = _gpu_hessian(coords1, z1, coords2, z2, torch.float64).cpu().numpy()
    H_cpu = _cpu_hessian(coords1, z1, coords2, z2)

    assert H_gpu.shape == ((3 + 2 + 4) * 3, (4 + 5) * 3)
    np.testing.assert_allclose(H_gpu, H_cpu, rtol=1e-9, atol=1e-11)


def test_hessian_batch_blocks_match_single_pairs() -> None:
    """Each (a, b) block of a batched call equals the corresponding single-pair call."""
    coords1 = [WATER_COORDS, HF_COORDS]
    z1 = [WATER_Z, HF_Z]
    coords2 = [AMMONIA_COORDS, METHANE_COORDS]
    z2 = [AMMONIA_Z, METHANE_Z]

    H = _gpu_hessian(coords1, z1, coords2, z2, torch.float64).cpu().numpy()

    row_off = np.cumsum([0, *[3 * len(z) for z in z1]])
    col_off = np.cumsum([0, *[3 * len(z) for z in z2]])
    for a in range(len(z1)):
        for b in range(len(z2)):
            block = _gpu_hessian([coords1[a]], [z1[a]], [coords2[b]], [z2[b]], torch.float64)
            np.testing.assert_allclose(
                H[row_off[a] : row_off[a + 1], col_off[b] : col_off[b + 1]],
                block.cpu().numpy(),
                rtol=1e-9,
                atol=1e-11,
            )


def test_hessian_matches_cpu_f64_fourier_order_2() -> None:
    coords1 = [WATER_COORDS, AMMONIA_COORDS]
    z1 = [WATER_Z, AMMONIA_Z]
    coords2 = [WATER_COORDS]
    z2 = [WATER_Z]

    args = KERNEL_ARGS.copy()
    args["fourier_order"] = 2

    H_gpu = _gpu_hessian(coords1, z1, coords2, z2, torch.float64, args).cpu().numpy()
    H_cpu = _cpu_hessian(coords1, z1, coords2, z2, args)

    np.testing.assert_allclose(H_gpu, H_cpu, rtol=1e-9, atol=1e-11)


def test_hessian_matches_cpu_f32() -> None:
    coords1 = [WATER_COORDS, AMMONIA_COORDS]
    z1 = [WATER_Z, AMMONIA_Z]
    coords2 = [METHANE_COORDS, WATER_COORDS]
    z2 = [METHANE_Z, WATER_Z]

    H_gpu = _gpu_hessian(coords1, z1, coords2, z2, torch.float32).cpu().numpy()
    H_cpu = _cpu_hessian(coords1, z1, coords2, z2)

    scale = float(np.max(np.abs(H_cpu)))
    np.testing.assert_allclose(H_gpu.astype(np.float64), H_cpu, rtol=2e-4, atol=1e-4 * scale)


def test_hessian_matches_numerical_jacobian_derivative() -> None:
    """Central differences of the CUDA Jacobian in R_B reproduce the Hessian."""
    coords1 = [WATER_COORDS]
    z1 = [WATER_Z]
    coords2 = [AMMONIA_COORDS]
    z2 = [AMMONIA_Z]

    H = _gpu_hessian(coords1, z1, coords2, z2, torch.float64).cpu().numpy()

    def _jacobian(coords_b):
        max_size2 = coords_b.shape[0]
        x2, n2, nn2 = repr_mod.generate([coords_b], z2, max_size=max_size2, cut_distance=1e6)
        x1, n1, nn1 = repr_mod.generate(coords1, z1, max_size=3, cut_distance=1e6)
        coords1_pad, z1_pad = _pad_coords_z(coords1, z1, 3)
        return (
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
                **KERNEL_ARGS,
            )
            .cpu()
            .numpy()[:, 0]
        )

    eps = 1e-5
    nb = AMMONIA_COORDS.shape[0]
    H_num = np.zeros_like(H)
    for beta in range(nb):
        for nu in range(3):
            jp, jm = [], []
            for sign, out in ((+1, jp), (-1, jm)):
                c = AMMONIA_COORDS.copy()
                c[beta, nu] += sign * eps
                out.append(_jacobian(c))
            H_num[:, beta * 3 + nu] = (jp[0] - jm[0]) / (2.0 * eps)

    np.testing.assert_allclose(H, H_num, rtol=1e-5, atol=1e-7)


def test_hessian_symm_matches_cpu_and_rect() -> None:
    coords = [WATER_COORDS, AMMONIA_COORDS, HF_COORDS]
    z = [WATER_Z, AMMONIA_Z, HF_Z]
    max_size = max(len(zi) for zi in z)
    x, n, nn = repr_mod.generate(coords, z, max_size=max_size, cut_distance=1e6)
    coords_pad, z_pad = _pad_coords_z(coords, z, max_size)

    H_symm = (
        cuda_fchl18_kernel.kernel_gaussian_hessian_symm(
            _to_cuda(x, torch.float64),
            _to_cuda_i32(n),
            _to_cuda_i32(nn),
            _to_cuda(coords_pad, torch.float64),
            _to_cuda_i32(z_pad),
            sigma=SIGMA,
            **KERNEL_ARGS,
        )
        .cpu()
        .numpy()
    )
    H_cpu = cpu_kernel.kernel_gaussian_hessian_symm(coords, z, sigma=SIGMA, **KERNEL_ARGS)
    H_rect = _gpu_hessian(coords, z, coords, z, torch.float64).cpu().numpy()

    np.testing.assert_allclose(H_symm, H_cpu, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(H_symm, H_rect, rtol=1e-14, atol=1e-14)


def test_hessian_symm_rfp_unpacks() -> None:
    import kernelforge.kernelmath as kernelmath

    coords = [WATER_COORDS, HF_COORDS]
    z = [WATER_Z, HF_Z]
    max_size = max(len(zi) for zi in z)
    x, n, nn = repr_mod.generate(coords, z, max_size=max_size, cut_distance=1e6)
    coords_pad, z_pad = _pad_coords_z(coords, z, max_size)
    D = int(3 * n.sum())

    H_rfp = (
        cuda_fchl18_kernel.kernel_gaussian_hessian_symm_rfp(
            _to_cuda(x, torch.float64),
            _to_cuda_i32(n),
            _to_cuda_i32(nn),
            _to_cuda(coords_pad, torch.float64),
            _to_cuda_i32(z_pad),
            sigma=SIGMA,
            **KERNEL_ARGS,
        )
        .cpu()
        .numpy()
    )
    H_dense = (
        cuda_fchl18_kernel.kernel_gaussian_hessian_symm(
            _to_cuda(x, torch.float64),
            _to_cuda_i32(n),
            _to_cuda_i32(nn),
            _to_cuda(coords_pad, torch.float64),
            _to_cuda_i32(z_pad),
            sigma=SIGMA,
            **KERNEL_ARGS,
        )
        .cpu()
        .numpy()
    )
    H_unpacked = kernelmath.rfp_to_full(H_rfp, D, uplo="L", transr="N")
    np.testing.assert_allclose(H_unpacked, H_dense, rtol=1e-14, atol=1e-14)

    H_cpu_rfp = cpu_kernel.kernel_gaussian_hessian_symm_rfp(coords, z, sigma=SIGMA, **KERNEL_ARGS)
    np.testing.assert_allclose(H_rfp, H_cpu_rfp, rtol=1e-9, atol=1e-11)
