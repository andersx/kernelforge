"""Tests for CudaGlobalRFFModel."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("kernelforge.cuda_rff_features", reason="cuda_rff_features not built")
pytest.importorskip("kernelforge.cuda_global_kernels", reason="cuda_global_kernels not built")
pytest.importorskip("kernelforge.cuda_invdist_repr", reason="cuda_invdist_repr not built")
pytest.importorskip("torch", reason="PyTorch not installed")

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

pytestmark = pytest.mark.skipif(not _gpu_ok, reason="No NVIDIA GPU detected")

from kernelforge.models import CudaGlobalRFFModel, GlobalRFFModel  # noqa: E402

_CACHE = Path.home() / ".kernelforge" / "datasets"
_RMD17_TRAIN = _CACHE / "rmd17_ethanol_train_01.npz"


def _load_ethanol(
    n: int = 25,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    if not _RMD17_TRAIN.exists():
        pytest.skip("rMD17 ethanol data not cached; run kernelcli to populate")
    d = np.load(_RMD17_TRAIN, allow_pickle=True)
    z_fixed = d["nuclear_charges"].astype(np.int32)
    coords = [d["coords"][i].astype(np.float64) for i in range(n)]
    z_list = [z_fixed for _ in range(n)]
    energies = d["energies"][:n].astype(np.float64)
    forces = d["forces"][:n].reshape(n, -1).astype(np.float64)
    return coords, z_list, energies, forces


def test_energy_only_shapes() -> None:
    coords, z_list, energies, _ = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaGlobalRFFModel(sigma=3.0, l2=1e-4, d_rff=128, seed=42, chunk_size=7)
    model.fit(tr, ztr, energies=energies[:20])

    assert model.training_mode_ == "energy_only"
    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (5,)
    assert F_pred.shape == (0,)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    coords, z_list, energies, _ = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaGlobalRFFModel(sigma=3.0, l2=1e-4, d_rff=128, seed=7, chunk_size=6)
    model.fit(tr, ztr, energies=energies[:20])
    E_orig, F_orig = model.predict(te, zte)

    path = tmp_path / "cuda_global_rff.npz"
    model.save(path)
    loaded = CudaGlobalRFFModel.load(path)
    assert isinstance(loaded, CudaGlobalRFFModel)

    E_load, F_load = loaded.predict(te, zte)
    np.testing.assert_allclose(E_load, E_orig, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(F_load, F_orig, rtol=0, atol=0)


def test_matches_cpu_global_rff_small() -> None:
    coords, z_list, energies, _ = _load_ethanol(20)
    tr, te = coords[:15], coords[15:]
    ztr, zte = z_list[:15], z_list[15:]

    cpu = GlobalRFFModel(sigma=3.0, l2=1e-3, d_rff=64, seed=11)
    gpu = CudaGlobalRFFModel(sigma=3.0, l2=1e-3, d_rff=64, seed=11, chunk_size=5)
    cpu.fit(tr, ztr, energies=energies[:15])
    gpu.fit(tr, ztr, energies=energies[:15])

    E_cpu, _ = cpu.predict(te, zte)
    E_gpu, _ = gpu.predict(te, zte)
    np.testing.assert_allclose(E_gpu, E_cpu, rtol=2e-2, atol=2e-2)


def test_energy_and_force_shapes() -> None:
    coords, z_list, energies, forces = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaGlobalRFFModel(sigma=3.0, l2=1e-2, d_rff=64, seed=42, chunk_size=5)
    model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])

    assert model.training_mode_ == "energy_and_force"
    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (5,)
    assert F_pred.shape == (5 * 9 * 3,)


def test_energy_and_force_matches_cpu_global_rff_small() -> None:
    coords, z_list, energies, forces = _load_ethanol(16)
    tr, te = coords[:12], coords[12:]
    ztr, zte = z_list[:12], z_list[12:]

    cpu = GlobalRFFModel(sigma=3.0, l2=1e-2, d_rff=32, seed=13)
    gpu = CudaGlobalRFFModel(sigma=3.0, l2=1e-2, d_rff=32, seed=13, chunk_size=4)
    cpu.fit(tr, ztr, energies=energies[:12], forces=forces[:12])
    gpu.fit(tr, ztr, energies=energies[:12], forces=forces[:12])

    E_cpu, F_cpu = cpu.predict(te, zte)
    E_gpu, F_gpu = gpu.predict(te, zte)
    np.testing.assert_allclose(E_gpu, E_cpu, rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(F_gpu, F_cpu, rtol=3e-2, atol=3e-2)


def test_force_only_not_implemented() -> None:
    coords, z_list, _, forces = _load_ethanol(10)
    model = CudaGlobalRFFModel(d_rff=32)
    with pytest.raises(NotImplementedError, match="energy_only and energy_and_force"):
        model.fit(coords, z_list, forces=forces)


def test_variable_atom_count_raises() -> None:
    rng = np.random.default_rng(0)
    coords = [rng.standard_normal((5, 3)), rng.standard_normal((6, 3))]
    z_list = [np.ones(5, dtype=np.int32), np.ones(6, dtype=np.int32)]
    energies = np.array([1.0, 2.0])
    model = CudaGlobalRFFModel(d_rff=32)
    with pytest.raises(ValueError, match="same atom count"):
        model.fit(coords, z_list, energies=energies)
