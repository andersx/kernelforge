#!/usr/bin/env python3
"""
Energy regression with Random Fourier Features (kitchen sinks) on ethanol CCSD(T).

This example demonstrates an end-to-end RFF-based kernel ridge regression workflow:

  1. Load the ethanol CCSD(T) dataset (auto-downloaded on first run).
  2. Generate FCHL19 atomic representations.
  3. Sample a random Fourier basis (W, b) to approximate the Gaussian kernel.
  4. Build the Gramian matrix (LZ^T LZ) and projection (LZ^T Y) in memory-efficient
     chunks -- no need to materialise the full (N_train, D) feature matrix at once.
  5. Solve the regularised linear system  alpha = (LZ^T LZ + lambda I)^{-1} LZ^T Y.
  6. Predict energies on the test set.

The RFF model approximates kernel ridge regression with a Gaussian (RBF) kernel.
Training cost is O(N * D^2) in time and O(D^2) in memory instead of O(N^3) / O(N^2).

Usage
-----
    uv run python examples/energy_regression_rff.py
"""

import numpy as np

from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf
from kernelforge.kernelmath import solve_cholesky
from kernelforge.kitchen_sinks import rff_features_elemental, rff_gramian_elemental

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRAIN = 500  # training set size
N_TEST = 500  # test set size
D = 2000  # number of random Fourier features (basis dimension)
SIGMA = 2.0  # Gaussian kernel width (in FCHL19 representation space)
LLAMBDA = 1e-8  # L2 regularisation strength
CHUNK = 250  # molecules per Gramian chunk (trades memory for speed)

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
# Build element-index mapping
#
# rff_features_elemental expects Q as a list of int32 arrays where each value
# is a 0-based index into the W / b stacks -- NOT the atomic number.
#   H -> 0,  C -> 1,  O -> 2   (matching the order in ELEMENTS)
# ---------------------------------------------------------------------------

elem_to_idx = {e: i for i, e in enumerate(ELEMENTS)}
Q_mol = np.array([elem_to_idx[zi] for zi in z], dtype=np.int32)
Q_train = [Q_mol] * N_TRAIN  # same connectivity for every ethanol frame
Q_test = [Q_mol] * N_TEST

# ---------------------------------------------------------------------------
# Generate FCHL19 representations
# ---------------------------------------------------------------------------

print(f"Generating FCHL19 representations for {N_TRAIN} training structures ...")
X_train = np.array(
    [generate_fchl_acsf(R_train[i], z, elements=ELEMENTS) for i in range(N_TRAIN)]
)  # (N_TRAIN, 9, 312)

print(f"Generating FCHL19 representations for {N_TEST} test structures ...")
X_test = np.array(
    [generate_fchl_acsf(R_test[i], z, elements=ELEMENTS) for i in range(N_TEST)]
)  # (N_TEST, 9, 312)

rep_size = X_train.shape[2]  # 312 for default FCHL19 with [H, C, O]
nelements = len(ELEMENTS)
print(f"Representation size: {rep_size}")

# ---------------------------------------------------------------------------
# Sample random Fourier basis
#
# Approximates the Gaussian kernel k(x, x') = exp(-||x-x'||^2 / (2*sigma^2))
# via the random feature map:
#
#   phi_e(x) = sqrt(2/D) * cos(x @ W[e] + b[e])
#
# where  W[e] ~ N(0, 1/sigma^2)  and  b[e] ~ Uniform(0, 2*pi).
# One independent (W[e], b[e]) pair per element type e.
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
W = rng.standard_normal((nelements, rep_size, D)) / SIGMA  # (nelements, rep_size, D)
b = rng.uniform(0.0, 2.0 * np.pi, (nelements, D))  # (nelements, D)

# ---------------------------------------------------------------------------
# Build the Gramian matrix in memory-efficient chunks
#
# rff_gramian_elemental processes CHUNK molecules at a time and accumulates
# LZtLZ += LZ_chunk.T @ LZ_chunk (DSYRK, upper triangle only) and
# LZtY  += LZ_chunk.T @ Y_chunk  (DGEMV).
# Peak memory for the feature matrix: O(CHUNK * D) instead of O(N_TRAIN * D).
# ---------------------------------------------------------------------------

print(f"Building Gramian ({D}x{D}) in chunks of {CHUNK} ...")
LZtLZ, LZtY = rff_gramian_elemental(X_train, Q_train, W, b, E_train, chunk_size=CHUNK)
# Returns: LZtLZ (D, D) symmetric PSD, LZtY (D,)

# ---------------------------------------------------------------------------
# Solve for regression coefficients
#
# Solves  (LZtLZ + lambda * I) @ alpha = LZtY  via Cholesky factorisation.
# ---------------------------------------------------------------------------

print("Solving linear system via Cholesky ...")
alpha = solve_cholesky(LZtLZ, LZtY, regularize=LLAMBDA)  # (D,)

# ---------------------------------------------------------------------------
# Predict and evaluate
# ---------------------------------------------------------------------------

LZ_test = rff_features_elemental(X_test, Q_test, W, b)  # (N_TEST,  D)
LZ_train = rff_features_elemental(X_train, Q_train, W, b)  # (N_TRAIN, D)

E_pred_test = LZ_test @ alpha + E_mean
E_pred_train = LZ_train @ alpha + E_mean
E_train_abs = E_train + E_mean  # restore original scale for MAE computation

mae_train = np.mean(np.abs(E_pred_train - E_train_abs))
mae_test = np.mean(np.abs(E_pred_test - E_test))
rmse_test = np.sqrt(np.mean((E_pred_test - E_test) ** 2))

print()
print("=" * 52)
print("  RFF energy regression -- ethanol CCSD(T)")
print("=" * 52)
print(f"  N_train      : {N_TRAIN}")
print(f"  N_test       : {N_TEST}")
print(f"  D (features) : {D}")
print(f"  sigma        : {SIGMA}")
print(f"  lambda       : {LLAMBDA:.0e}")
print(f"  Train MAE    : {mae_train:.3f} kcal/mol")
print(f"  Test  MAE    : {mae_test:.3f} kcal/mol")
print(f"  Test  RMSE   : {rmse_test:.3f} kcal/mol")
print("=" * 52)
