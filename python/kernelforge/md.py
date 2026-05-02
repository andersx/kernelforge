"""Molecular dynamics runner using ASE (NVE VelocityVerlet or NVT Langevin).

Usage
-----
>>> from kernelforge.md import run_md
>>> frames = run_md(model, atoms, n_steps=1000, dt=0.5, temperature=300.0)
>>> # NVT Langevin:
>>> frames = run_md(model, atoms, n_steps=1000, dt=0.5, temperature=300.0,
...                ensemble="nvt-langevin", friction=0.01)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .models.cuda_local_krr import CudaLocalKRRModel
    from .models.cuda_local_rff import CudaLocalRFFModel


def run_md(
    model: CudaLocalKRRModel | CudaLocalRFFModel,
    atoms: Any,  # noqa: ANN401  # ase.Atoms
    n_steps: int,
    dt: float = 0.5,
    temperature: float | None = 300.0,
    n_equil: int = 0,
    trajectory_file: str | Path | None = "md.traj",
    traj_interval: int = 10,
    logfile: str | Path | None = "md.log",
    log_interval: int = 10,
    units: str = "kcal/mol",
    seed: int = 42,
    ensemble: Literal["nve", "nvt-langevin"] = "nve",
    friction: float = 0.01,
) -> list[Any]:
    """Run molecular dynamics using ASE (NVE VelocityVerlet or NVT Langevin).

    Parameters
    ----------
    model:
        A fitted ``CudaLocalKRRModel`` or ``CudaLocalRFFModel`` trained in
        ``energy_and_force`` mode.
    atoms:
        Initial ``ase.Atoms`` object.  A copy is made internally so the
        caller's object is not modified.
    n_steps:
        Number of MD steps to run.
    dt:
        Timestep in femtoseconds (default 0.5 fs).
    temperature:
        Temperature in Kelvin used to draw initial velocities from a
        Maxwell-Boltzmann distribution.  Pass ``None`` (or 0) to keep any
        existing momenta on *atoms* unchanged.  Also used as the target
        temperature for ``'nvt-langevin'``.
    n_equil:
        Number of silent equilibration steps to run *before* production.
        No frames are saved and no log entries are written during this phase.
        Defaults to 0 (no equilibration).
    trajectory_file:
        Path for the ASE trajectory file (``*.traj``).  Pass ``None`` to
        disable.
    traj_interval:
        Save a frame every *traj_interval* steps.
    logfile:
        Path for the plain-text MD log (step, time, Epot, Ekin, Etot, T).
        Pass ``None`` to disable.
    log_interval:
        Write a log entry every *log_interval* steps.
    units:
        Energy/force units the model was trained on — ``'kcal/mol'`` (default)
        or ``'eV'``.
    seed:
        Random seed for Maxwell-Boltzmann velocity initialisation.
    ensemble:
        ``'nve'`` (default) — microcanonical NVE using ASE ``VelocityVerlet``.
        ``'nvt-langevin'`` — canonical NVT using ASE ``Langevin``.
    friction:
        Langevin friction coefficient in fs⁻¹ (default 0.01).  Only used when
        ``ensemble='nvt-langevin'``.

    Returns
    -------
    frames : list[ase.Atoms]
        Snapshot copies saved every *traj_interval* steps (includes step 0).
    """
    try:
        import ase.units as _units
        from ase.md.velocitydistribution import (
            MaxwellBoltzmannDistribution,
            Stationary,
            ZeroRotation,
        )
        from ase.md.verlet import VelocityVerlet
    except ImportError as exc:
        msg = (
            "ASE is required for run_md. "
            "Install it with:  pip install 'kernelforge[ase]'  or  pip install ase"
        )
        raise ImportError(msg) from exc

    if ensemble not in ("nve", "nvt-langevin"):
        msg = f"ensemble must be 'nve' or 'nvt-langevin', got '{ensemble}'"
        raise ValueError(msg)

    from .ase_calculator import KernelForgeCalculator

    # Work on a copy so the caller's atoms object is unmodified
    atoms = atoms.copy()
    atoms.calc = KernelForgeCalculator(model, units=units)

    # Optionally initialise velocities
    if temperature is not None and temperature > 0.0:
        rng = np.random.default_rng(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
        Stationary(atoms)  # remove centre-of-mass drift
        ZeroRotation(atoms)  # remove net angular momentum

    if ensemble == "nvt-langevin":
        from ase.md.langevin import Langevin

        if temperature is None or temperature <= 0.0:
            msg = "temperature must be > 0 for ensemble='nvt-langevin'"
            raise ValueError(msg)
        dyn: Any = Langevin(
            atoms,
            timestep=dt * _units.fs,
            temperature_K=temperature,
            friction=friction / _units.fs,
            fixcm=False,
        )
    else:
        dyn = VelocityVerlet(atoms, timestep=dt * _units.fs)

    # --- equilibration (silent, no observers) ---
    if n_equil > 0:
        print(f"[run_md] Equilibrating for {n_equil} steps ...")
        dyn.run(n_equil)
        print("[run_md] Equilibration done.")

    # Production step offset so log step numbers start from 0 after equil
    prod_offset = dyn.nsteps

    # --- trajectory writer ---
    traj_obj = None
    if trajectory_file is not None:
        from ase.io.trajectory import Trajectory

        traj_obj = Trajectory(str(trajectory_file), "w", atoms)
        dyn.attach(traj_obj.write, interval=traj_interval)

    # --- log writer ---
    log_handle = None
    if logfile is not None:
        log_path = Path(logfile)
        log_handle = log_path.open("w")
        log_handle.write("# step  time_fs  Epot_eV  Ekin_eV  Etot_eV  T_K\n")

        def _write_log() -> None:
            step = dyn.nsteps - prod_offset
            time_fs = step * dt
            e_pot = atoms.get_potential_energy()
            e_kin = atoms.get_kinetic_energy()
            temp_k = atoms.get_temperature()
            assert log_handle is not None  # noqa: S101 — local assertion for mypy
            log_handle.write(
                f"{step:8d}  {time_fs:10.3f}  {e_pot:14.6f}  "
                f"{e_kin:12.6f}  {e_pot + e_kin:14.6f}  {temp_k:8.2f}\n"
            )
            log_handle.flush()

        dyn.attach(_write_log, interval=log_interval)

    # --- in-memory frame collector ---
    frames: list[Any] = []

    def _save_frame() -> None:
        frames.append(atoms.copy())

    dyn.attach(_save_frame, interval=traj_interval)

    dyn.run(n_steps)

    if traj_obj is not None:
        traj_obj.close()
    if log_handle is not None:
        log_handle.close()

    return frames
