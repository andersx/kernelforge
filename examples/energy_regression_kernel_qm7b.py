#!/usr/bin/env python3
"""
Energy regression with exact kernel ridge regression (KRR) on QM7b.

QM7b contains 7211 small organic molecules (up to 23 heavy atoms, elements
H/C/N/O/S/Cl) with PBE0/def2-TZVP atomisation energies in kcal/mol.

This example demonstrates an end-to-end KRR workflow using the local FCHL19
Gaussian kernel:

  1. Load the QM7b dataset (auto-downloaded on first run).
  2. Generate FCHL19 atomic representations; pad to max_atoms for the batch.
  3. Build the symmetric kernel matrix K (N_train x N_train).
  4. Solve  (K + lambda * I) @ alpha = E_train  via Cholesky factorisation.
  5. Build the rectangular kernel matrix K_test (N_train x N_test).
  6. Predict  E_pred = K_test.T @ alpha.

Unlike ethanol, QM7b molecules have varying sizes and element compositions.
The kernel handles this naturally by summing over all atom pairs; padding
rows (zeroed representations) contribute nothing because the Gaussian kernel
between a real atom and a zero-vector is exp(-||x_a||^2 / 2s^2) != 1 in
general. To avoid this, N (true atom counts) is passed so the kernel
skips padding atoms entirely.

Training cost is O(N^3) in time and O(N^2) in memory. For larger training
sets, see the RFF example.

Usage
-----
    uv run python examples/energy_regression_kernel_qm7b.py
"""

import numpy as np

from kernelforge.cli import load_qm7b_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf
from kernelforge.kernelmath import solve_cholesky
from kernelforge.local_kernels import kernel_gaussian, kernel_gaussian_symm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRAIN = 500  # training set size  (KRR is O(N^3) so keep modest)
N_TEST = 1000  # test set size
SIGMA = 6.0  # Gaussian kernel width (tuned for QM7b FCHL19 rep space)
LLAMBDA = 1e-8  # L2 regularisation strength

ELEMENTS = [1, 6, 7, 8, 16, 17]  # H, C, N, O, S, Cl -- all elements in QM7b

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading QM7b dataset ...")
data = load_qm7b_raw_data()

R_all = data["R"]  # object array of (natoms_i, 3) float64 arrays
z_all = data["z"]  # object array of (natoms_i,) int32 arrays
E_all = data["E"]  # (7211,) float64, kcal/mol
N_all = data["N"]  # (7211,) int32, atom count per molecule

R_train = R_all[:N_TRAIN]
z_train = z_all[:N_TRAIN]
E_train = E_all[:N_TRAIN].copy()

R_test = R_all[N_TRAIN : N_TRAIN + N_TEST]
z_test = z_all[N_TRAIN : N_TRAIN + N_TEST]
E_test = E_all[N_TRAIN : N_TRAIN + N_TEST]

# Mean-centre energies (improves numerical conditioning)
E_mean = E_train.mean()
E_train = E_train - E_mean

# ---------------------------------------------------------------------------
# Generate FCHL19 representations and pad to max_atoms
#
# kernel_gaussian_symm / kernel_gaussian expect:
#   X  -- (N, max_atoms, rep_size) float64   padded representations
#   Q  -- (N, max_atoms) int32               atomic numbers (raw, not 0-based)
#   N  -- (N,) int32                         true atom count per molecule
#
# Padding rows in X are zeroed; atoms beyond N[i] are ignored by the kernel.
# ---------------------------------------------------------------------------

max_atoms_train = int(N_all[:N_TRAIN].max())
rep_size = 984  # FCHL19 with 6-element basis

print(f"Generating FCHL19 representations for {N_TRAIN} training structures ...")
X_train = np.zeros((N_TRAIN, max_atoms_train, rep_size))
Q_train = np.zeros((N_TRAIN, max_atoms_train), dtype=np.int32)
for i in range(N_TRAIN):
    nat = int(N_all[i])
    rep = generate_fchl_acsf(R_train[i], z_train[i], elements=ELEMENTS)
    X_train[i, :nat, :] = rep
    Q_train[i, :nat] = z_train[i]
N_train = N_all[:N_TRAIN].astype(np.int32)  # true atom counts

max_atoms_test = int(N_all[N_TRAIN : N_TRAIN + N_TEST].max())

print(f"Generating FCHL19 representations for {N_TEST} test structures ...")
X_test = np.zeros((N_TEST, max_atoms_test, rep_size))
Q_test = np.zeros((N_TEST, max_atoms_test), dtype=np.int32)
for i in range(N_TEST):
    nat = int(N_all[N_TRAIN + i])
    rep = generate_fchl_acsf(R_test[i], z_test[i], elements=ELEMENTS)
    X_test[i, :nat, :] = rep
    Q_test[i, :nat] = z_test[i]
N_test = N_all[N_TRAIN : N_TRAIN + N_TEST].astype(np.int32)

print(f"Representation size: {rep_size},  max_atoms (train): {max_atoms_train}")

# ---------------------------------------------------------------------------
# Build the symmetric training kernel matrix
# ---------------------------------------------------------------------------

print(f"Building {N_TRAIN}x{N_TRAIN} kernel matrix (sigma={SIGMA}) ...")
K = kernel_gaussian_symm(X_train, Q_train, N_train, SIGMA)  # (N_TRAIN, N_TRAIN)

# ---------------------------------------------------------------------------
# Solve for regression coefficients
# ---------------------------------------------------------------------------

print("Solving linear system via Cholesky ...")
alpha = solve_cholesky(K, E_train, regularize=LLAMBDA)  # (N_TRAIN,)

# ---------------------------------------------------------------------------
# Predict and evaluate
# ---------------------------------------------------------------------------

E_pred_train = K @ alpha + E_mean
E_train_abs = E_train + E_mean

print(f"Building {N_TRAIN}x{N_TEST} test kernel matrix ...")
K_test = kernel_gaussian(X_train, X_test, Q_train, Q_test, N_train, N_test, SIGMA)
# kernel_gaussian returns (N_train, N_test), so transpose for the prediction
E_pred_test = K_test.T @ alpha + E_mean

mae_train = np.mean(np.abs(E_pred_train - E_train_abs))
mae_test = np.mean(np.abs(E_pred_test - E_test))
rmse_test = np.sqrt(np.mean((E_pred_test - E_test) ** 2))

print()
print("=" * 52)
print("  KRR energy regression -- QM7b")
print("=" * 52)
print(f"  N_train      : {N_TRAIN}")
print(f"  N_test       : {N_TEST}")
print(f"  sigma        : {SIGMA}")
print(f"  lambda       : {LLAMBDA:.0e}")
print(f"  Train MAE    : {mae_train:.3f} kcal/mol")
print(f"  Test  MAE    : {mae_test:.3f} kcal/mol")
print(f"  Test  RMSE   : {rmse_test:.3f} kcal/mol")
print("=" * 52)
