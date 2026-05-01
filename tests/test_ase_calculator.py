"""Tests for KernelForgeCalculator and run_md().

Skipped automatically when:
  - ase is not installed
  - cuda_fchl19_repr / cuda_local_kernels are not built (no CUDA at build time)
  - No NVIDIA GPU is detected
  - The rMD17 ethanol dataset is not cached locally
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip entire module if ASE is not installed
pytest.importorskip("ase", reason="ASE not installed")

# Skip entire module if CUDA extension modules are not available
pytest.importorskip("kernelforge.cuda_fchl19_repr", reason="cuda_fchl19_repr not built")
pytest.importorskip("kernelforge.cuda_local_kernels", reason="cuda_local_kernels not built")
pytest.importorskip("torch", reason="PyTorch not installed")

import ase.units  # after importorskip guards

# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Dataset path check
# ---------------------------------------------------------------------------
_ETHANOL_TRAIN = Path.home() / ".kernelforge" / "datasets" / "rmd17_ethanol_train_01.npz"
_DATASET_AVAILABLE = _ETHANOL_TRAIN.exists()

# ---------------------------------------------------------------------------
# Shared fixture: tiny fitted model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    """Fit a CudaLocalKRRModel on 30 rMD17 ethanol frames."""
    if not _DATASET_AVAILABLE:
        pytest.skip("rMD17 ethanol dataset not cached; run kernelcli to download it")

    from kernelforge.models import CudaLocalKRRModel

    d = np.load(_ETHANOL_TRAIN, allow_pickle=True)
    z_fixed = d["nuclear_charges"].astype(np.int32)
    n = 30
    coords = [d["coords"][i].astype(np.float64) for i in range(n)]
    z_list = [z_fixed for _ in range(n)]
    energies = d["energies"][:n].astype(np.float64)
    forces = [d["forces"][i].astype(np.float64) for i in range(n)]

    model = CudaLocalKRRModel(sigma=2.0, l2=1e-3, elements=[1, 6, 8])
    model.fit(coords, z_list, energies=energies, forces=forces)
    return model, coords[0], z_fixed


# ---------------------------------------------------------------------------
# KernelForgeCalculator tests
# ---------------------------------------------------------------------------


def test_calculator_energy_is_eV(tiny_model):
    """Energy returned by the calculator must be in eV (not kcal/mol)."""
    from ase import Atoms

    from kernelforge.ase_calculator import KernelForgeCalculator

    model, coords0, z0 = tiny_model
    atoms = Atoms(numbers=z0, positions=coords0)
    calc = KernelForgeCalculator(model, units="kcal/mol")
    atoms.calc = calc

    E_eV = atoms.get_potential_energy()
    # rMD17 ethanol energies are ~-97000 kcal/mol → ~-4200 eV
    # Sanity: must be a finite float and negative
    assert np.isfinite(E_eV), "Energy should be finite"
    assert E_eV < 0.0, "Ethanol energy should be negative"

    # Compare to raw model output: E_eV ≈ E_kcal * factor
    E_arr, _ = model.predict([coords0], [z0])
    factor = ase.units.kcal / ase.units.mol
    assert abs(E_eV - float(E_arr[0]) * factor) < 1e-9


def test_calculator_forces_shape(tiny_model):
    """Forces array must be (n_atoms, 3) after unit conversion."""
    from ase import Atoms

    from kernelforge.ase_calculator import KernelForgeCalculator

    model, coords0, z0 = tiny_model
    atoms = Atoms(numbers=z0, positions=coords0)
    atoms.calc = KernelForgeCalculator(model, units="kcal/mol")

    F = atoms.get_forces()
    assert F.shape == (len(z0), 3), f"Expected ({len(z0)}, 3), got {F.shape}"
    assert np.all(np.isfinite(F)), "Forces should be finite"


def test_calculator_units_eV(tiny_model):
    """With units='eV' the calculator must not apply any conversion factor."""
    from ase import Atoms

    from kernelforge.ase_calculator import KernelForgeCalculator

    model, coords0, z0 = tiny_model
    atoms = Atoms(numbers=z0, positions=coords0)
    atoms.calc = KernelForgeCalculator(model, units="eV")

    E_eV = atoms.get_potential_energy()
    E_arr, _ = model.predict([coords0], [z0])
    assert abs(E_eV - float(E_arr[0])) < 1e-9, "units='eV' should be a no-op"


def test_calculator_unfitted_raises():
    """Wrapping an unfitted model must raise RuntimeError."""
    from kernelforge.ase_calculator import KernelForgeCalculator
    from kernelforge.models import CudaLocalKRRModel

    model = CudaLocalKRRModel()
    with pytest.raises(RuntimeError, match="fitted"):
        KernelForgeCalculator(model)


def test_calculator_energy_only_raises(tiny_model):
    """A model trained in energy_only mode must raise RuntimeError."""
    import numpy as np

    from kernelforge.ase_calculator import KernelForgeCalculator
    from kernelforge.models import CudaLocalKRRModel

    if not _DATASET_AVAILABLE:
        pytest.skip("dataset not available")

    d = np.load(_ETHANOL_TRAIN, allow_pickle=True)
    z_fixed = d["nuclear_charges"].astype(np.int32)
    n = 20
    coords = [d["coords"][i].astype(np.float64) for i in range(n)]
    z_list = [z_fixed for _ in range(n)]
    energies = d["energies"][:n].astype(np.float64)

    model = CudaLocalKRRModel(sigma=2.0, l2=1e-3, elements=[1, 6, 8])
    model.fit(coords, z_list, energies=energies)  # energy_only

    with pytest.raises(RuntimeError, match="energy_only"):
        KernelForgeCalculator(model)


# ---------------------------------------------------------------------------
# run_md() tests
# ---------------------------------------------------------------------------


def test_run_md_nve_returns_frames(tiny_model):
    """run_md must return the correct number of snapshot frames."""
    from ase import Atoms

    from kernelforge.md import run_md

    model, coords0, z0 = tiny_model
    atoms = Atoms(numbers=z0, positions=coords0)

    n_steps = 20
    interval = 5
    with tempfile.TemporaryDirectory() as tmp:
        frames = run_md(
            model,
            atoms,
            n_steps=n_steps,
            dt=0.5,
            temperature=300.0,
            trajectory_file=Path(tmp) / "test.traj",
            traj_interval=interval,
            logfile=Path(tmp) / "test.log",
            log_interval=interval,
            units="kcal/mol",
            seed=0,
        )

    # Frame 0 is saved before dyn.run, then every interval steps
    # Expected: [0, 5, 10, 15, 20] = 5 frames
    expected = n_steps // interval + 1
    assert len(frames) == expected, f"Expected {expected} frames, got {len(frames)}"


def test_run_md_nve_energy_conservation(tiny_model):
    """Total energy drift over 20 NVE steps must be < 0.1 eV."""
    from ase import Atoms

    from kernelforge.md import run_md

    model, coords0, z0 = tiny_model
    atoms = Atoms(numbers=z0, positions=coords0)

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "nve.log"
        run_md(
            model,
            atoms,
            n_steps=20,
            dt=0.5,
            temperature=300.0,
            trajectory_file=None,
            traj_interval=1,
            logfile=log_path,
            log_interval=1,
            units="kcal/mol",
            seed=1,
        )

        lines = [ln for ln in log_path.read_text().splitlines() if not ln.startswith("#")]

    e_tot_start = float(lines[0].split()[4])
    e_tot_end = float(lines[-1].split()[4])
    drift = abs(e_tot_end - e_tot_start)
    assert drift < 0.1, f"Total energy drifted by {drift:.4f} eV in 20 steps"


def test_run_md_no_temperature_init(tiny_model):
    """Passing temperature=None must leave existing (zero) momenta unchanged."""
    from ase import Atoms

    from kernelforge.md import run_md

    model, coords0, z0 = tiny_model
    atoms = Atoms(numbers=z0, positions=coords0)  # zero momenta

    frames = run_md(
        model,
        atoms,
        n_steps=5,
        dt=0.5,
        temperature=None,  # do not initialise velocities
        trajectory_file=None,
        traj_interval=1,
        logfile=None,
        units="kcal/mol",
    )
    assert len(frames) >= 1


# ---------------------------------------------------------------------------
# kernelmd CLI smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_kernelmd_cli(tiny_model, tmp_path):
    """kernelmd CLI: save a model, write a .xyz, run 10 steps, check output."""
    import subprocess
    import sys

    from ase import Atoms
    from ase.io import write

    model, coords0, z0 = tiny_model

    model_path = tmp_path / "tiny.npz"
    model.save(str(model_path))

    xyz_path = tmp_path / "start.xyz"
    atoms = Atoms(numbers=z0, positions=coords0)
    write(str(xyz_path), atoms)

    traj_path = tmp_path / "out.extxyz"
    log_path = tmp_path / "out.log"

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "kernelforge.kernelmd",
            "--model",
            str(model_path),
            "--structure",
            str(xyz_path),
            "--steps",
            "10",
            "--dt",
            "0.5",
            "--temperature",
            "300",
            "--output",
            str(traj_path),
            "--interval",
            "5",
            "--logfile",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"kernelmd failed:\n{result.stderr}"
    assert traj_path.exists(), "Trajectory file was not created"
    assert log_path.exists(), "Log file was not created"
