"""
High-level model API — FCHL18KRRModel, energy+force training on rMD17 Ethanol.

Trains an energy+force KRR model using the FCHL18 analytical kernel on the
MD17 ethanol dataset (555k CCSD(T)-quality MD trajectories):
  1. Auto-download MD17 ethanol dataset (~first run only)
  2. Random train/test split
  3. Fit FCHL18KRRModel on energies + forces
  4. Report MAE and Pearson r for energies and forces on the test set
  5. Save the model to /tmp and reload it to verify bit-exact predictions

Dataset: MD17 ethanol (DFT/PBEsol energies and forces, kcal/mol and kcal/mol/Å)
         Auto-downloaded from sgdml.org on first run (~8 MB).

Ethanol: 9 atoms (C2H5OH), elements H/C/O (Z = 1, 6, 8).

Usage:
    uv run python examples/model_api_rmd17_ethanol_fchl18_energy_and_force.py
"""

from __future__ import annotations

import time

import numpy as np

from kernelforge.cli import load_ethanol_raw_data
from kernelforge.models import FCHL18KRRModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 50
N_TEST = 50
SEED = 42
SIGMA = 2.5
L2 = 1e-4  # FCHL18 Hessian kernel is approximately PSD — use slightly higher l2
MAX_SIZE = 9  # ethanol has exactly 9 atoms

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("FCHL18KRRModel — energy+force training on MD17 Ethanol")
print("=" * 60)

t0 = time.perf_counter()
data = load_ethanol_raw_data()

# z is fixed for all ethanol frames — replicate into a list
z_fixed = data["z"].astype(np.int32)  # shape (9,)
n_total = len(data["E"])

rng = np.random.default_rng(SEED)
all_idx = rng.permutation(n_total)
tr_idx = all_idx[:N_TRAIN]
te_idx = all_idx[N_TRAIN : N_TRAIN + N_TEST]

R_tr = data["R"][tr_idx]  # (N_TRAIN, 9, 3)
R_te = data["R"][te_idx]  # (N_TEST, 9, 3)

coords_tr = [R_tr[i].astype(np.float64) for i in range(N_TRAIN)]
coords_te = [R_te[i].astype(np.float64) for i in range(N_TEST)]
z_tr = [z_fixed for _ in range(N_TRAIN)]
z_te = [z_fixed for _ in range(N_TEST)]

E_tr = data["E"][tr_idx].astype(np.float64)  # (N_TRAIN,) kcal/mol
E_te = data["E"][te_idx].astype(np.float64)  # (N_TEST,)  kcal/mol
F_tr = [data["F"][tr_idx[i]].astype(np.float64) for i in range(N_TRAIN)]  # (9, 3) each
F_te = [data["F"][te_idx[i]].astype(np.float64) for i in range(N_TEST)]

print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
print(f"    total dataset size : {n_total} structures")
print(f"    train / test       : {N_TRAIN} / {N_TEST}")
print(f"    atoms per molecule : {len(z_fixed)} (fixed, {list(z_fixed)})")
print(f"    energy range (tr)  : {E_tr.min():.2f} .. {E_tr.max():.2f} kcal/mol")

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
model = FCHL18KRRModel(sigma=SIGMA, l2=L2, max_size=MAX_SIZE)
model.fit(coords_tr, z_tr, energies=E_tr, forces=F_tr)
print(f"\n[2] Model fitted in {time.perf_counter() - t0:.2f}s")
print(f"    training_mode    : {model.training_mode_}")
print(f"    baseline elements: {list(model.baseline_elements_)}")
print(
    "    element energies : "
    + "  ".join(
        f"Z{z}={e:.4f}"
        for z, e in zip(model.baseline_elements_, model.element_energies_, strict=True)
    )
    + " kcal/mol"
)
print(f"\n    Train energy: {model.train_score_['energy']}")
print(f"    Train force:  {model.train_score_['force']}")

# ---------------------------------------------------------------------------
# Predict + score on test set
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
test_scores = model.score(coords_te, z_te, energies=E_te, forces=F_te)
print(f"\n[3] Scored {N_TEST} test molecules in {time.perf_counter() - t0:.2f}s")

print(f"\n[4] Test  energy: {test_scores['energy']}")
print(f"    Test  force:  {test_scores['force']}")

# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------
save_path = "/tmp/rmd17_ethanol_fchl18_model.npz"
model.save(save_path)
print(f"\n[5] Model saved to {save_path}")

loaded = FCHL18KRRModel.load(save_path)
E_orig, F_orig = model.predict(coords_te, z_te)
E_reloaded, F_reloaded = loaded.predict(coords_te, z_te)
e_diff = float(np.max(np.abs(E_reloaded - E_orig)))
f_diff = float(np.max(np.abs(F_reloaded - F_orig)))
print(f"    Reloaded model — max energy diff: {e_diff:.2e}  force diff: {f_diff:.2e}")

print("\n" + "=" * 60 + "\nDone.\n" + "=" * 60)
