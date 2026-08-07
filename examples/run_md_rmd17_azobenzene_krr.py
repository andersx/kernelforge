"""NVE MD on rMD17 azobenzene using CudaLocalKRRModel.

This script:
  1. Downloads/loads the rMD17 azobenzene dataset (cached under ~/.kernelforge/).
  2. Fits a CudaLocalKRRModel on 950 training frames.
  3. Takes dataset frame 0 as the starting geometry.
  4. Runs 4000 silent equilibration steps at 300 K (0.25 fs timestep).
  5. Runs 40000 production NVE steps, saving every 20 steps.
  6. Prints training time, MD time, energy drift, and final RMSD.

Usage
-----
    uv run --env-file /dev/null python examples/run_md_rmd17_azobenzene_krr.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make sure the local source tree is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kernelforge.kernelcli import load_rmd17  # type: ignore[attr-defined]
from kernelforge.md import run_md
from kernelforge.models import CudaLocalKRRModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOLECULE = "azobenzene"
N_TRAIN = 950
N_EQUIL_STEPS = 0  # silent equilibration before production
N_MD_STEPS = 1000
DT_FS = 0.5
TEMPERATURE_K = 300.0
TRAJ_OUT = Path("azobenzene_krr_md.extxyz")
LOG_OUT = Path("azobenzene_krr_md.log")
INTERVAL = 20  # save every 20 steps

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
model = CudaLocalKRRModel(sigma=2.0, l2=1e-2, elements=[1, 6, 7])
t0_fit = time.perf_counter()
model.fit(tr_coords, tr_z, energies=tr_E, forces=tr_F)
t_fit = time.perf_counter() - t0_fit
print(f"[example] Training done in {t_fit:.2f} s")

# ---------------------------------------------------------------------------
# 3. Build starting ASE Atoms from dataset frame 0
# ---------------------------------------------------------------------------
try:
    from ase import Atoms
except ImportError:
    print("ERROR: ASE is required.  pip install ase", file=sys.stderr)
    sys.exit(1)

atoms0 = Atoms(numbers=tr_z[0], positions=tr_coords[0])

# ---------------------------------------------------------------------------
# 4. Run MD
# ---------------------------------------------------------------------------
print(
    f"[example] Running {N_EQUIL_STEPS} equil + {N_MD_STEPS} production steps NVE @ {TEMPERATURE_K} K, dt={DT_FS} fs ..."
)
t0_md = time.perf_counter()
frames = run_md(
    model=model,
    atoms=atoms0,
    n_steps=N_MD_STEPS,
    dt=DT_FS,
    temperature=TEMPERATURE_K,
    n_equil=N_EQUIL_STEPS,
    trajectory_file=TRAJ_OUT,
    traj_interval=INTERVAL,
    logfile=LOG_OUT,
    log_interval=INTERVAL,
    units="kcal/mol",
    seed=0,
)
t_md = time.perf_counter() - t0_md
t_step_ms = t_md / (N_EQUIL_STEPS + N_MD_STEPS) * 1000.0
print(f"[example] MD done in {t_md:.2f} s  ({t_step_ms:.2f} ms/step, incl. equil)")

# ---------------------------------------------------------------------------
# 5. Report energy drift and final RMSD
# ---------------------------------------------------------------------------
print(f"\n[example] Saved {len(frames)} frames to {TRAJ_OUT}")

log_lines = [ln for ln in LOG_OUT.read_text().splitlines() if not ln.startswith("#")]
if log_lines:
    e_tot_start = float(log_lines[0].split()[4])
    e_tot_end = float(log_lines[-1].split()[4])
    drift_meV = (e_tot_end - e_tot_start) * 1000.0
    print(f"[example] Total energy drift over {N_MD_STEPS} production steps (post-equil): {drift_meV:+.3f} meV")

pos_start = frames[0].get_positions()
pos_end = frames[-1].get_positions()
rmsd = float(np.sqrt(np.mean((pos_end - pos_start) ** 2)))
print(f"[example] RMSD (final vs. start): {rmsd:.3f} Å")
print("[example] Done.")
