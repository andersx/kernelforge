"""Tests for cuda_fchl19_repr FCHL19 gradients."""

from __future__ import annotations

import math
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from kernelforge.fchl19_repr import (
    generate_fchl_acsf_and_gradients as cpu_generate_fchl_acsf_and_gradients,
)

cuda_fchl19_repr = pytest.importorskip(
    "kernelforge.cuda_fchl19_repr",
    reason="cuda_fchl19_repr not built (requires CUDA + PyTorch at build time)",
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

pytestmark = pytest.mark.skipif(not _gpu_ok, reason="No NVIDIA GPU detected by nvidia-smi")

_DEFAULTS: dict[str, Any] = {
    "nRs2": 24,
    "nRs3": 20,
    "nFourier": 1,
    "eta2": 0.32,
    "eta3": 2.7,
    "zeta": math.pi,
    "rcut": 8.0,
    "acut": 8.0,
    "two_body_decay": 1.8,
    "three_body_decay": 0.57,
    "three_body_weight": 13.4,
}


@pytest.fixture(scope="module")
def gpu_device() -> Any:
    return torch.device("cuda:0")


def _water_system() -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    coords = np.array(
        [[0.9572, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.2399872, 0.92662721, 0.0]], dtype=np.float64
    )
    z = np.array([1, 8, 1], dtype=np.int32)
    return coords, z


def _ethanol_like() -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    coords = np.array(
        [
            [0.000, 0.000, 0.000],
            [1.232, 0.000, 0.000],
            [1.879, 1.025, 0.000],
            [-0.646, 1.025, 0.000],
            [-0.646, -0.513, 0.889],
            [-0.646, -0.513, -0.889],
            [1.879, -0.513, -0.889],
            [1.879, -0.513, 0.889],
            [3.155, 1.025, 0.000],
        ],
        dtype=np.float64,
    )
    z = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    return coords, z


def _z_to_idx(z_arr: NDArray[np.int32], elements: list[int]) -> NDArray[np.int32]:
    idx_map = {z: i for i, z in enumerate(elements)}
    return np.array([idx_map[int(z)] for z in z_arr], dtype=np.int32)


def _build_gpu_batch(
    coords_list: list[NDArray[np.float64]],
    z_list: list[NDArray[np.int32]],
    elements: list[int],
    device: Any,
) -> tuple[Any, Any, Any]:
    nm = len(coords_list)
    max_atoms = max(len(z) for z in z_list)
    coords_np = np.zeros((nm, max_atoms, 3), dtype=np.float32)
    Q_np = np.zeros((nm, max_atoms), dtype=np.int32)
    N_np = np.array([len(z) for z in z_list], dtype=np.int32)
    for m, (coords, z) in enumerate(zip(coords_list, z_list, strict=False)):
        na = len(z)
        coords_np[m, :na, :] = coords.astype(np.float32)
        Q_np[m, :na] = _z_to_idx(z, elements)
    return (
        torch.from_numpy(coords_np).to(device),
        torch.from_numpy(Q_np).to(device),
        torch.from_numpy(N_np).to(device),
    )


def _cpu_rep_grad(
    coords: NDArray[np.float64], z: NDArray[np.int32], elements: list[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rep, grad_flat = cpu_generate_fchl_acsf_and_gradients(coords, z, elements=elements, **_DEFAULTS)  # type: ignore[call-arg]
    grad = grad_flat.reshape(len(z), rep.shape[1], len(z), 3)
    return rep, grad


def _gpu_rep_grad(
    coords: NDArray[np.float64], z: NDArray[np.int32], elements: list[int], device: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch([coords], [z], elements, device)
    rep, grad = cuda_fchl19_repr.generate_fchl_acsf_and_gradients(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
    )
    return rep[0].cpu().numpy().astype(np.float64), grad[
        0, : len(z), :, : len(z), :
    ].cpu().numpy().astype(np.float64)


def test_grad_output_shape_single_molecule(gpu_device: Any) -> None:
    coords, z = _water_system()
    elements = [1, 8]
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch([coords], [z], elements, gpu_device)

    rep, grad = cuda_fchl19_repr.generate_fchl_acsf_and_gradients(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
    )

    assert rep.shape == (1, 3, 168)
    assert grad.shape == (1, 3, 168, 3, 3)
    assert rep.dtype == torch.float32
    assert grad.dtype == torch.float32


def test_grad_correctness_water(gpu_device: Any) -> None:
    coords, z = _water_system()
    elements = [1, 8]

    ref_rep, ref_grad = _cpu_rep_grad(coords, z, elements)
    got_rep, got_grad = _gpu_rep_grad(coords, z, elements, gpu_device)

    np.testing.assert_allclose(got_rep, ref_rep, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got_grad, ref_grad, rtol=5e-4, atol=5e-5)


def test_grad_correctness_ethanol_like(gpu_device: Any) -> None:
    coords, z = _ethanol_like()
    elements = [1, 6, 8]

    ref_rep, ref_grad = _cpu_rep_grad(coords, z, elements)
    got_rep, got_grad = _gpu_rep_grad(coords, z, elements, gpu_device)

    np.testing.assert_allclose(got_rep, ref_rep, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got_grad, ref_grad, rtol=1e-3, atol=1e-4)


def test_grad_padded_slots_zeroed(gpu_device: Any) -> None:
    c1, z1 = _water_system()
    c2, z2 = _ethanol_like()
    elements = [1, 6, 8]
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch([c1, c2], [z1, z2], elements, gpu_device)

    rep, grad = cuda_fchl19_repr.generate_fchl_acsf_and_gradients(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
    )
    rep_cpu = rep.cpu().numpy()
    grad_cpu = grad.cpu().numpy()

    assert np.all(rep_cpu[0, len(z1) :, :] == 0.0)
    assert np.all(grad_cpu[0, len(z1) :, :, :, :] == 0.0)
    assert np.all(grad_cpu[0, :, :, len(z1) :, :] == 0.0)


def test_grad_correctness_small_mols_mini(gpu_device: Any) -> None:
    """GPU vs CPU gradients: bundled small_mols_mini subset with variable atom counts."""
    from kernelforge.kernelcli import load_small_mols_mini

    coords_list, z_list, *_ = load_small_mols_mini(n_train=10, n_test=0)
    elements = [1, 6, 7, 8]
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch(coords_list, z_list, elements, gpu_device)

    got_rep_batch, got_grad_batch = cuda_fchl19_repr.generate_fchl_acsf_and_gradients(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
    )
    got_rep_batch_np = got_rep_batch.cpu().numpy().astype(np.float64)
    got_grad_batch_np = got_grad_batch.cpu().numpy().astype(np.float64)

    for idx, (coords, z) in enumerate(zip(coords_list, z_list, strict=False)):
        ref_rep, ref_grad = _cpu_rep_grad(coords, z, elements)
        natoms = len(z)
        np.testing.assert_allclose(
            got_rep_batch_np[idx, :natoms, :],
            ref_rep,
            rtol=5e-4,
            atol=5e-5,
            err_msg=f"GPU/CPU rep mismatch on small_mols_mini molecule {idx}",
        )
        np.testing.assert_allclose(
            got_grad_batch_np[idx, :natoms, :, :natoms, :],
            ref_grad,
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"GPU/CPU grad mismatch on small_mols_mini molecule {idx}",
        )
