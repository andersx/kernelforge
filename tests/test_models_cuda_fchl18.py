"""Tests for CudaFCHL18KRRModel."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("kernelforge.cuda_fchl18_kernel")
pytest.importorskip("kernelforge.cuda_fchl18_repr")

from kernelforge.models import CudaFCHL18KRRModel

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

RNG = np.random.default_rng(2)
N_ATOMS = 9
MAX_SIZE = 9
Z_ETHANOL = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)


def _make_dataset(
    n_mols: int = 12,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    coords_list = [RNG.standard_normal((N_ATOMS, 3)) + 1.5 for _ in range(n_mols)]
    z_list = [Z_ETHANOL] * n_mols
    energies = RNG.standard_normal(n_mols)
    forces = RNG.standard_normal((n_mols, N_ATOMS * 3))
    return coords_list, z_list, energies, forces


@pytest.fixture(scope="module")
def dataset() -> tuple[list, list, np.ndarray, np.ndarray]:
    return _make_dataset(12)


class TestCudaFCHL18KRRModelEnergyOnly:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:8], coords_list[8:]
        ztr, zte = z_list[:8], z_list[8:]

        model = CudaFCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(tr, ztr, energies=energies[:8])
        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (4,)
        assert F_pred.shape == (4 * N_ATOMS * 3,)
        assert model.training_mode_ == "energy_only"

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:8], coords_list[8:]
        ztr, zte = z_list[:8], z_list[8:]

        model = CudaFCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(tr, ztr, energies=energies[:8])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "cuda_fchl18_eo.npz"
        model.save(path)
        loaded = CudaFCHL18KRRModel.load(path)
        E_load, F_load = loaded.predict(te, zte)
        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)


class TestCudaFCHL18KRRModelForceOnly:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, _, forces = dataset
        tr, te = coords_list[:8], coords_list[8:]
        ztr, zte = z_list[:8], z_list[8:]

        model = CudaFCHL18KRRModel(sigma=5.0, l2=1e-3, max_size=MAX_SIZE)
        model.fit(tr, ztr, forces=forces[:8])
        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (4,)
        assert F_pred.shape == (4 * N_ATOMS * 3,)
        assert model._kp_fit["use_atm"] is False


class TestCudaFCHL18KRRModelEnergyAndForce:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, forces = dataset
        tr, te = coords_list[:8], coords_list[8:]
        ztr, zte = z_list[:8], z_list[8:]

        model = CudaFCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(tr, ztr, energies=energies[:8], forces=forces[:8])
        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (4,)
        assert F_pred.shape == (4 * N_ATOMS * 3,)
        assert model.training_mode_ == "energy_and_force"

    def test_ase_calculator_smoke(self, dataset: tuple) -> None:
        pytest.importorskip("ase")
        from ase import Atoms

        from kernelforge.ase_calculator import KernelForgeCalculator

        coords_list, z_list, energies, forces = dataset
        model = CudaFCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(
            coords_list[:8],
            z_list[:8],
            energies=energies[:8],
            forces=forces[:8],
        )

        atoms = Atoms(numbers=z_list[8], positions=coords_list[8])
        atoms.calc = KernelForgeCalculator(model, units="eV")
        energy = atoms.get_potential_energy()
        forces_ase = atoms.get_forces()
        assert np.isfinite(energy)
        assert forces_ase.shape == (N_ATOMS, 3)
        assert np.all(np.isfinite(forces_ase))
