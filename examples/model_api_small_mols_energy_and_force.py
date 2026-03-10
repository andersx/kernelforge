"""
High-level model API — LocalKRRModel, energy+force training on small molecules.

Demonstrates energy_and_force training using pre-split train/test sets:
  1. Load small_mols_mini_train.npz  (1000 molecules, H/C/N/O, up to 23 atoms)
  2. Fit LocalKRRModel on atomization energies + forces
  3. Predict on small_mols_mini_test.npz (595 molecules)
  4. Report MAE, Pearson r, slope, intercept for both energies and forces
  5. Save the model to a .npz file and reload it

Units: atomization energies in kcal/mol, forces in kcal/mol/Å.

Usage:
    uv run python examples/model_api_small_mols_energy_and_force.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from kernelforge.models import LocalKRRModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 1000
N_TEST = 595
SIGMA = 20.0
L2 = 1e-8
ELEMENTS = [1, 6, 7, 8]  # H, C, N, O

DATA_DIR = Path(__file__).parent
TRAIN_NPZ = DATA_DIR / "small_mols_mini_train.npz"
TEST_NPZ = DATA_DIR / "small_mols_mini_test.npz"


# ---------------------------------------------------------------------------
# Helper: load one npz split into lists
# ---------------------------------------------------------------------------
def load_split(path: Path, n: int | None = None) -> tuple:
    """Load coords, nuclear_charges, atomization_energy and forces from npz."""
    d = np.load(path, allow_pickle=True)
    n = n if n is not None else len(d["atomization_energy"])
    coords_raw = d["coords"][:n]
    z_raw = d["nuclear_charges"][:n]
    forces_raw = d["forces"][:n]
    coords_list = [coords_raw[i].astype(np.float64) for i in range(n)]
    z_list = [z_raw[i].astype(np.int32) for i in range(n)]
    forces_list = [forces_raw[i].astype(np.float64) for i in range(n)]
    energies = d["atomization_energy"][:n].astype(np.float64)
    return coords_list, z_list, energies, forces_list


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("LocalKRRModel — energy+force training on small molecules")
print("=" * 60)

t0 = time.perf_counter()
coords_tr, z_tr, E_tr, F_tr = load_split(TRAIN_NPZ, N_TRAIN)
coords_te, z_te, E_te, F_te = load_split(TEST_NPZ, N_TEST)
print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
print(
    f"    train: {len(E_tr)} molecules  "
    f"atoms: {min(len(z) for z in z_tr)}–{max(len(z) for z in z_tr)}"
)
print(
    f"    test:  {len(E_te)} molecules  "
    f"atoms: {min(len(z) for z in z_te)}–{max(len(z) for z in z_te)}"
)
print(f"    energy range (train): {E_tr.min():.2f} .. {E_tr.max():.2f} kcal/mol")

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
model = LocalKRRModel(sigma=SIGMA, l2=L2, elements=ELEMENTS)
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
# Predict + score
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
test_scores = model.score(coords_te, z_te, energies=E_te, forces=F_te)
print(f"\n[3] Scored {len(E_te)} test molecules in {time.perf_counter() - t0:.2f}s")

print(f"\n[4] Test  energy: {test_scores['energy']}")
print(f"    Test  force:  {test_scores['force']}")
 
print("\n" + "=" * 60 + "\nDone.\n" + "=" * 60)
