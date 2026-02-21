#!/usr/bin/env python3
"""
Energy regression with exact kernel ridge regression (KRR) on ethanol CCSD(T).

This example demonstrates an end-to-end KRR workflow using the local FCHL19
Gaussian kernel:

  1. Load the ethanol CCSD(T) dataset (auto-downloaded on first run).
  2. Generate FCHL19 atomic representations.
  3. Build the symmetric kernel matrix K (N_train x N_train).
  4. Solve  (K + lambda * I) @ alpha = E_train  via Cholesky factorisation.
  5. Build the rectangular kernel matrix K_test (N_test x N_train).
  6. Predict  E_pred = K_test @ alpha.

The local Gaussian kernel sums over atomic contributions:
    K(A, B) = sum_{i in A} sum_{j in B} exp(-||x_i - x_j||^2 / (2*sigma^2))

Training cost is O(N^3) in time and O(N^2) in memory, which limits practical
training set sizes to a few thousand. For larger datasets, see the RFF example.

Usage
-----
    uv run python examples/energy_regression_kernel.py
"""

import numpy as np

from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf
from kernelforge.kernelmath import solve_cholesky
from kernelforge.local_kernels import kernel_gaussian, kernel_gaussian_symm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRAIN = 200  # training set size  (KRR is O(N^3) so keep modest)
N_TEST = 500  # test set size
SIGMA = 2.0  # Gaussian kernel width (in FCHL19 representation space)
LLAMBDA = 1e-8  # L2 regularisation strength

ELEMENTS = [1, 6, 8]  # H, C, O -- elements present in ethanol

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading ethanol CCSD(T) dataset ...")
data = load_ethanol_raw_data()

R_all = data["R"]  # (1000, 9, 3)  Cartesian coords in Angstrom
E_all = data["E"].flatten()  # (1000,)        CCSD(T) energies in kcal/mol
z = data["z"].astype(int)  # (9,)           atomic numbers, same for every frame

R_train = R_all[:N_TRAIN]
E_train = E_all[:N_TRAIN]
R_test = R_all[N_TRAIN : N_TRAIN + N_TEST]
E_test = E_all[N_TRAIN : N_TRAIN + N_TEST]

# Mean-centre energies (improves numerical conditioning)
E_mean = E_train.mean()
E_train = E_train - E_mean

# ---------------------------------------------------------------------------
# Generate FCHL19 representations
#
# kernel_gaussian_symm / kernel_gaussian expect:
#   X  -- (N, max_atoms, rep_size)  padded representation array
#   Q  -- (N, max_atoms) int32      atomic numbers (NOT 0-based indices)
#   N  -- (N,) int32                true atom count per molecule
#
# For ethanol every frame has the same 9 atoms, so no padding is needed.
# ---------------------------------------------------------------------------

print(f"Generating FCHL19 representations for {N_TRAIN} training structures ...")
X_train = np.array(
    [generate_fchl_acsf(R_train[i], z, elements=ELEMENTS) for i in range(N_TRAIN)]
)  # (N_TRAIN, 9, 312)

print(f"Generating FCHL19 representations for {N_TEST} test structures ...")
X_test = np.array(
    [generate_fchl_acsf(R_test[i], z, elements=ELEMENTS) for i in range(N_TEST)]
)  # (N_TEST, 9, 312)

rep_size = X_train.shape[2]
print(f"Representation size: {rep_size}")

# Q uses raw atomic numbers; N is the true atom count per molecule
Q_train = np.tile(z, (N_TRAIN, 1)).astype(np.int32)  # (N_TRAIN, 9)
Q_test = np.tile(z, (N_TEST, 1)).astype(np.int32)  # (N_TEST,  9)
N_train = np.full(N_TRAIN, len(z), dtype=np.int32)  # (N_TRAIN,) all 9
N_test = np.full(N_TEST, len(z), dtype=np.int32)  # (N_TEST,)  all 9

# ---------------------------------------------------------------------------
# Build the symmetric training kernel matrix
#
# kernel_gaussian_symm computes the full (N_train, N_train) matrix:
#   K[i, j] = sum_{a in mol_i} sum_{b in mol_j}
#               exp(-||x_a - x_b||^2 / (2 * sigma^2))
# exploiting symmetry K[i,j] = K[j,i] for efficiency.
# ---------------------------------------------------------------------------

print(f"Building {N_TRAIN}x{N_TRAIN} kernel matrix (sigma={SIGMA}) ...")
K = kernel_gaussian_symm(X_train, Q_train, N_train, SIGMA)  # (N_TRAIN, N_TRAIN)

# ---------------------------------------------------------------------------
# Solve for regression coefficients
#
# Solves  (K + lambda * I) @ alpha = E_train  via Cholesky factorisation.
# ---------------------------------------------------------------------------

print("Solving linear system via Cholesky ...")
alpha = solve_cholesky(K, E_train, regularize=LLAMBDA)  # (N_TRAIN,)

# ---------------------------------------------------------------------------
# Predict and evaluate
# ---------------------------------------------------------------------------

# Training-set prediction (reuse K already computed)
E_pred_train = K @ alpha + E_mean
E_train_abs = E_train + E_mean

# Test-set prediction: build rectangular kernel K_test (N_test, N_train)
print(f"Building {N_TEST}x{N_TRAIN} test kernel matrix ...")
K_test = kernel_gaussian(X_train, X_test, Q_train, Q_test, N_train, N_test, SIGMA)
# kernel_gaussian returns shape (N_train, N_test), so transpose for E = K_test.T @ alpha
E_pred_test = K_test.T @ alpha + E_mean

mae_train = np.mean(np.abs(E_pred_train - E_train_abs))
mae_test = np.mean(np.abs(E_pred_test - E_test))
rmse_test = np.sqrt(np.mean((E_pred_test - E_test) ** 2))

print()
print("=" * 52)
print("  KRR energy regression -- ethanol CCSD(T)")
print("=" * 52)
print(f"  N_train      : {N_TRAIN}")
print(f"  N_test       : {N_TEST}")
print(f"  sigma        : {SIGMA}")
print(f"  lambda       : {LLAMBDA:.0e}")
print(f"  Train MAE    : {mae_train:.3f} kcal/mol")
print(f"  Test  MAE    : {mae_test:.3f} kcal/mol")
print(f"  Test  RMSE   : {rmse_test:.3f} kcal/mol")
print("=" * 52)
