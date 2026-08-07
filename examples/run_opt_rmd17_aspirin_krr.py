"""Geometry optimisation of rMD17 aspirin using CudaLocalKRRModel.

This script:
  1. Downloads/loads the rMD17 aspirin dataset (cached under ~/.kernelforge/).
  2. Fits a CudaLocalKRRModel on 950 training frames.
  3. Takes dataset frame 0 as the starting geometry.
  4. Runs an L-BFGS geometry optimisation until forces < fmax threshold,
     saving every step to an extxyz trajectory.
  5. Reports training time, optimisation steps, final energy, and force RMSE.

Usage
-----
    uv run --env-file /dev/null python examples/run_opt_rmd17_aspirin_krr.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kernelforge.kernelcli import load_rmd17  # type: ignore[attr-defined]
from kernelforge.models import CudaLocalKRRModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOLECULE      = "aspirin"
N_TRAIN       = 950
FMAX          = 0.05    # convergence threshold in eV/Å
MAX_STEPS     = 500
TRAJ_OUT      = Path("aspirin_krr_opt.extxyz")
LOG_OUT       = Path("aspirin_krr_opt.log")

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print(f"[example] Loading rMD17 {MOLECULE} ...")
tr_coords, tr_z, tr_E, tr_F, _, _, _, _ = load_rmd17(
    MOLECULE, split=1, n_train=N_TRAIN, n_test=1
)

# ---------------------------------------------------------------------------
# 2. Fit model
# ---------------------------------------------------------------------------
print(f"[example] Fitting CudaLocalKRRModel on {N_TRAIN} frames ...")
model = CudaLocalKRRModel(sigma=2.0, l2=1e-2, elements=[1, 6, 8])
t0_fit = time.perf_counter()
model.fit(tr_coords, tr_z, energies=tr_E, forces=tr_F)
t_fit = time.perf_counter() - t0_fit
print(f"[example] Training done in {t_fit:.2f} s")

# ---------------------------------------------------------------------------
# 3. Build starting ASE Atoms from dataset frame 0
# ---------------------------------------------------------------------------
try:
    from ase import Atoms
    from ase.optimize import LBFGS
except ImportError:
    print("ERROR: ASE is required.  pip install ase", file=sys.stderr)
    sys.exit(1)

from kernelforge.ase_calculator import KernelForgeCalculator  # type: ignore[attr-defined]

atoms = Atoms(numbers=tr_z[0], positions=tr_coords[0])
atoms.calc = KernelForgeCalculator(model, units="kcal/mol")

e_start = atoms.get_potential_energy()
f_rms_start = float(np.sqrt(np.mean(atoms.get_forces() ** 2)))
print(f"[example] Start:  E = {e_start:.6f} eV,  F_rms = {f_rms_start:.4f} eV/Å")

# ---------------------------------------------------------------------------
# 4. Run optimisation
# ---------------------------------------------------------------------------
print(f"[example] Running L-BFGS optimisation (fmax={FMAX} eV/Å, max {MAX_STEPS} steps) ...")
opt = LBFGS(atoms, logfile=str(LOG_OUT))

def _write_frame() -> None:
    from ase.io import write
    write(str(TRAJ_OUT), atoms, format="extxyz", append=True)

opt.attach(_write_frame, interval=1)

t0_opt = time.perf_counter()
converged = opt.run(fmax=FMAX, steps=MAX_STEPS)
t_opt = time.perf_counter() - t0_opt

# ---------------------------------------------------------------------------
# 5. Report results
# ---------------------------------------------------------------------------
e_end = atoms.get_potential_energy()
f_rms_end = float(np.sqrt(np.mean(atoms.get_forces() ** 2)))
f_max_end = float(np.max(np.abs(atoms.get_forces())))
n_steps = opt.nsteps

print(f"[example] Optimisation {'converged' if converged else 'NOT converged'} "
      f"in {n_steps} steps ({t_opt:.2f} s)")
print(f"[example] Final:  E = {e_end:.6f} eV,  F_rms = {f_rms_end:.4f} eV/Å,  "
      f"F_max = {f_max_end:.4f} eV/Å")
print(f"[example] ΔE (start → end) = {(e_end - e_start) * 1000:.3f} meV")
print(f"[example] Trajectory saved to {TRAJ_OUT}")
print("[example] Done.")
