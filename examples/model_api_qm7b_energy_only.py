"""
High-level model API — LocalKRRModel, energy-only training on QM7b.

Demonstrates the simplest use of the kernelforge model API:
  1. Load 250 QM7b molecules (variable size, up to 20 heavy atoms + H)
  2. Fit LocalKRRModel on 200 training energies
  3. Predict energies on 50 test molecules
  4. Save the model to a .npz file and reload it

Dataset: QM7b (PBE/def2-SVP total energies in kcal/mol)
         Auto-downloaded from GitHub releases on first run.

Usage:
    uv run python examples/model_api_qm7b_energy_only.py
"""

from __future__ import annotations

import time

import numpy as np

from kernelforge.cli import load_qm7b_raw_data
from kernelforge.models import LocalKRRModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 5000
N_TEST = 2000
SEED = 42
SIGMA = 10.0
L2 = 1e-8
ELEMENTS = [1, 6, 7, 8, 16, 17]  # all elements present in QM7b

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

print("=" * 60)
print("LocalKRRModel — energy-only training on QM7b")
print("=" * 60)

t0 = time.perf_counter()
data = load_qm7b_raw_data()

# Random train/test split — single fancy-index call per array to avoid
# per-index pickle deserialization overhead on the object arrays.
rng = np.random.default_rng(SEED)
all_idx = rng.permutation(len(data["E"]))
tr_idx = all_idx[:N_TRAIN]
te_idx = all_idx[N_TRAIN : N_TRAIN + N_TEST]

R_tr, R_te = data["R"][tr_idx], data["R"][te_idx]
z_tr_raw, z_te_raw = data["z"][tr_idx], data["z"][te_idx]

coords_tr = [R_tr[i].astype(np.float64) for i in range(N_TRAIN)]
z_tr = [z_tr_raw[i].astype(np.int32) for i in range(N_TRAIN)]
coords_te = [R_te[i].astype(np.float64) for i in range(N_TEST)]
z_te = [z_te_raw[i].astype(np.int32) for i in range(N_TEST)]
E_tr = data["E"][tr_idx].astype(np.float64)  # kcal/mol
E_te = data["E"][te_idx].astype(np.float64)  # kcal/mol

print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
print(
    f"    train: {N_TRAIN} molecules, atom counts: "
    f"min={min(len(z) for z in z_tr)}  "
    f"max={max(len(z) for z in z_tr)}"
)
print(f"    Energy range: {E_tr.min():.2f} .. {E_tr.max():.2f} kcal/mol")

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
model = LocalKRRModel(sigma=SIGMA, l2=L2, elements=ELEMENTS)
model.fit(coords_tr, z_tr, energies=E_tr)
print(f"\n[2] Model fitted in {time.perf_counter() - t0:.2f}s")
print(f"    training_mode    : {model.training_mode_}")
print(f"    baseline elements: {list(model.baseline_elements_)}")
print(
    f"    element energies : "
    + "  ".join(
        f"Z{z}={e:.4f} kcal/mol"
        for z, e in zip(model.baseline_elements_, model.element_energies_, strict=True)
    )
)

# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
test_scores = model.score(coords_te, z_te, energies=E_te)
print(f"\n[3] Scored test set in {time.perf_counter() - t0:.2f}s")

print(f"\n[4] Train energy:  {model.train_score_['energy']}")
print(f"    Test  energy:  {test_scores['energy']}")
print(f"    ({N_TRAIN} training points, untuned sigma={SIGMA} — expect large MAE)")

# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------
save_path = "/tmp/qm7b_krr_model.npz"
model.save(save_path)
print(f"\n[5] Model saved to {save_path}")

loaded = LocalKRRModel.load(save_path)
E_orig, _ = model.predict(coords_te, z_te)
E_reloaded, _ = loaded.predict(coords_te, z_te)
max_diff = float(np.max(np.abs(E_reloaded - E_orig)))
print(f"    Reloaded model max prediction diff: {max_diff:.2e}  (should be ~0)")

print("\n" + "=" * 60 + "\nDone.\n" + "=" * 60)
