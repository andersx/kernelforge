"""Tests for CudaLocalRFFModel using FCHL19 local descriptors."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("kernelforge.cuda_rff_features", reason="cuda_rff_features not built")
pytest.importorskip("kernelforge.cuda_global_kernels", reason="cuda_global_kernels not built")
pytest.importorskip("kernelforge.cuda_fchl19_repr", reason="cuda_fchl19_repr not built")
pytest.importorskip("kernelforge.cuda_solvers", reason="cuda_solvers not built")
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

from kernelforge.models import CudaLocalRFFModel, LocalRFFModel  # noqa: E402

_CACHE = Path.home() / ".kernelforge" / "datasets"
_RMD17_TRAIN = _CACHE / "rmd17_ethanol_train_01.npz"
_ELEMENTS = [1, 6, 8]


def _load_ethanol(n: int = 25) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
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

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-2, d_rff=64, seed=42, elements=_ELEMENTS, chunk_size=8
    )
    model.fit(tr, ztr, energies=energies[:20])

    assert model.training_mode_ == "energy_only"
    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (5,)
    assert F_pred.shape == (5 * 9 * 3,)


def test_energy_only_matches_cpu_small() -> None:
    coords, z_list, energies, _ = _load_ethanol(16)
    tr, te = coords[:12], coords[12:]
    ztr, zte = z_list[:12], z_list[12:]

    cpu = LocalRFFModel(sigma=20.0, l2=1e-2, d_rff=32, seed=7, elements=_ELEMENTS)
    gpu = CudaLocalRFFModel(sigma=20.0, l2=1e-2, d_rff=32, seed=7, elements=_ELEMENTS, chunk_size=4)
    cpu.fit(tr, ztr, energies=energies[:12])
    gpu.fit(tr, ztr, energies=energies[:12])

    E_cpu, _ = cpu.predict(te, zte)
    E_gpu, _ = gpu.predict(te, zte)
    np.testing.assert_allclose(E_gpu, E_cpu, rtol=3e-2, atol=3e-2)


def test_energy_and_force_shapes() -> None:
    coords, z_list, energies, forces = _load_ethanol(20)
    tr, te = coords[:15], coords[15:]
    ztr, zte = z_list[:15], z_list[15:]

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-1, d_rff=32, seed=11, elements=_ELEMENTS, chunk_size=5
    )
    model.fit(tr, ztr, energies=energies[:15], forces=forces[:15])


    assert model.training_mode_ == "energy_and_force"
    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (5,)
    assert F_pred.shape == (5 * 9 * 3,)


def test_qr_energy_only_shapes() -> None:
    """QR solver: output shapes are correct for energy_only."""
    coords, z_list, energies, _ = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-1, d_rff=16, seed=11, elements=_ELEMENTS,
        chunk_size=5, solver="qr",
    )
    model.fit(tr, ztr, energies=energies[:20])

    assert model.training_mode_ == "energy_only"
    E_pred, _ = model.predict(te, zte)
    assert E_pred.shape == (5,)


def test_qr_energy_and_force_shapes() -> None:
    """QR solver: output shapes are correct for energy_and_force."""
    coords, z_list, energies, forces = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-1, d_rff=16, seed=11, elements=_ELEMENTS,
        chunk_size=5, solver="qr",
    )
    model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])

    assert model.training_mode_ == "energy_and_force"
    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (5,)
    assert F_pred.shape == (5 * 9 * 3,)

def test_energy_and_force_matches_cpu_small() -> None:
    coords, z_list, energies, forces = _load_ethanol(14)
    tr, te = coords[:10], coords[10:]
    ztr, zte = z_list[:10], z_list[10:]

    cpu = LocalRFFModel(sigma=20.0, l2=1e2, d_rff=24, seed=13, elements=_ELEMENTS)
    gpu = CudaLocalRFFModel(sigma=20.0, l2=1e2, d_rff=24, seed=13, elements=_ELEMENTS, chunk_size=5)
    cpu.fit(tr, ztr, energies=energies[:10], forces=forces[:10])
    gpu.fit(tr, ztr, energies=energies[:10], forces=forces[:10])

    E_cpu, F_cpu = cpu.predict(te, zte)
    E_gpu, F_gpu = gpu.predict(te, zte)
    np.testing.assert_allclose(E_gpu, E_cpu, rtol=8e-2, atol=8e-2)
    np.testing.assert_allclose(F_gpu, F_cpu, rtol=8e-2, atol=8e-2)


# ---------------------------------------------------------------------------
# SVD solver tests
# ---------------------------------------------------------------------------


def test_svd_energy_only_shapes() -> None:
    """SVD solver: output shapes are correct for energy_only."""
    coords, z_list, energies, _ = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaLocalRFFModel(
        sigma=20.0,
        l2=1e-2,
        d_rff=16,
        seed=42,
        elements=_ELEMENTS,
        chunk_size=8,
        solver="svd",
        rcond=-1.0,
    )
    model.fit(tr, ztr, energies=energies[:20])

    assert model.training_mode_ == "energy_only"
    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (5,)
    assert F_pred.shape == (5 * 9 * 3,)


def test_svd_matches_cholesky_energy_only() -> None:
    """SVD and Cholesky solvers give similar predictions for energy_only."""
    coords, z_list, energies, _ = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    chol = CudaLocalRFFModel(
        sigma=20.0,
        l2=1e-2,
        d_rff=16,
        seed=7,
        elements=_ELEMENTS,
        chunk_size=4,
        solver="cholesky",
    )
    svd = CudaLocalRFFModel(
        sigma=20.0,
        l2=1e-2,
        d_rff=16,
        seed=7,
        elements=_ELEMENTS,
        chunk_size=4,
        solver="svd",
        rcond=-1.0,
    )
    chol.fit(tr, ztr, energies=energies[:20])
    svd.fit(tr, ztr, energies=energies[:20])

    E_chol, _ = chol.predict(te, zte)
    E_svd, _ = svd.predict(te, zte)
    np.testing.assert_allclose(E_svd, E_chol, rtol=5e-2, atol=5e-2)


def test_svd_energy_and_force_shapes() -> None:
    """SVD solver: output shapes are correct for energy_and_force."""
    coords, z_list, energies, forces = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaLocalRFFModel(
        sigma=20.0,
        l2=1e-1,
        d_rff=16,
        seed=11,
        elements=_ELEMENTS,
        chunk_size=5,
        solver="svd",
        rcond=-1.0,
    )
    model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])

    assert model.training_mode_ == "energy_and_force"
    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (5,)
    assert F_pred.shape == (5 * 9 * 3,)


# ---------------------------------------------------------------------------
# PCA compression tests
# ---------------------------------------------------------------------------

# ethanol: H6C2O1 per molecule, 20 training mols → n_valid: H≈120, C≈40, O≈20
# Use n_pca=16 to stay below the O count (20) and far below rep_size (~420)
_N_PCA = 16


def test_pca_energy_only_shapes() -> None:
    """PCA: energy_only output shapes are correct."""
    coords, z_list, energies, _ = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-2, d_rff=32, seed=42, elements=_ELEMENTS,
        chunk_size=8, n_pca=_N_PCA,
    )
    model.fit(tr, ztr, energies=energies[:20])

    assert model.training_mode_ == "energy_only"
    assert model._rep_size == _N_PCA
    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (5,)
    assert F_pred.shape == (5 * 9 * 3,)


def test_pca_energy_and_force_shapes() -> None:
    """PCA: energy_and_force output shapes are correct."""
    coords, z_list, energies, forces = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-1, d_rff=32, seed=11, elements=_ELEMENTS,
        chunk_size=5, n_pca=_N_PCA,
    )
    model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])

    assert model.training_mode_ == "energy_and_force"
    assert model._rep_size == _N_PCA
    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (5,)
    assert F_pred.shape == (5 * 9 * 3,)


def test_pca_center_whiten_shapes() -> None:
    """PCA with centering and whitening: output shapes are correct."""
    coords, z_list, energies, _ = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-2, d_rff=32, seed=5, elements=_ELEMENTS,
        chunk_size=8, n_pca=_N_PCA, pca_center=True, pca_whiten=True,
    )
    model.fit(tr, ztr, energies=energies[:20])

    assert model._rep_size == _N_PCA
    E_pred, _ = model.predict(te, zte)
    assert E_pred.shape == (5,)


def test_pca_save_load_roundtrip(tmp_path: Path) -> None:
    """PCA model: save and load produces identical predictions."""
    coords, z_list, energies, _ = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-2, d_rff=32, seed=42, elements=_ELEMENTS,
        chunk_size=8, n_pca=_N_PCA,
    )
    model.fit(tr, ztr, energies=energies[:20])
    E_before, _ = model.predict(te, zte)

    save_path = tmp_path / "pca_model.npz"
    model.save(save_path)
    loaded = CudaLocalRFFModel.load(save_path)

    assert loaded.n_pca == _N_PCA
    E_after, _ = loaded.predict(te, zte)
    np.testing.assert_allclose(E_after, E_before, rtol=1e-5, atol=1e-5)


def test_pca_save_load_force_roundtrip(tmp_path: Path) -> None:
    """PCA model with forces: save and load produces identical predictions."""
    coords, z_list, energies, forces = _load_ethanol(25)
    tr, te = coords[:20], coords[20:]
    ztr, zte = z_list[:20], z_list[20:]

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-1, d_rff=32, seed=11, elements=_ELEMENTS,
        chunk_size=5, n_pca=_N_PCA,
    )
    model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])
    E_before, F_before = model.predict(te, zte)

    save_path = tmp_path / "pca_force_model.npz"
    model.save(save_path)
    loaded = CudaLocalRFFModel.load(save_path)

    E_after, F_after = loaded.predict(te, zte)
    np.testing.assert_allclose(E_after, E_before, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(F_after, F_before, rtol=1e-5, atol=1e-5)


def test_pca_n_pca_too_large_raises() -> None:
    """PCA: n_pca >= rep_size must raise ValueError."""
    coords, z_list, energies, _ = _load_ethanol(20)

    model = CudaLocalRFFModel(
        sigma=20.0, l2=1e-2, d_rff=32, seed=42, elements=_ELEMENTS,
        chunk_size=8, n_pca=9999,
    )
    with pytest.raises(ValueError, match="n_pca"):
        model.fit(coords[:15], z_list[:15], energies=energies[:15])
