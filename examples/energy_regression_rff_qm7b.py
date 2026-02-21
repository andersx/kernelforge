#!/usr/bin/env python3
"""
Energy regression with Random Fourier Features (kitchen sinks) on QM7b.

QM7b contains 7211 small organic molecules (up to 23 heavy atoms, elements
H/C/N/O/S/Cl) with PBE0/def2-TZVP atomisation energies in kcal/mol.

This example demonstrates an end-to-end RFF-based kernel ridge regression:

  1. Load the QM7b dataset (auto-downloaded on first run).
  2. Generate FCHL19 atomic representations; pad to max_atoms for the batch.
  3. Sample a random Fourier basis (W, b) to approximate the Gaussian kernel.
  4. Build the Gramian matrix and projection in memory-efficient chunks.
  5. Solve  alpha = (LZ^T LZ + lambda I)^{-1} LZ^T Y  via Cholesky.
  6. Predict energies on the test set.

Unlike the ethanol case, QM7b molecules have varying numbers of atoms and
varying element composition, so padding and careful element-index mapping
are required.

Usage
-----
    uv run python examples/energy_regression_rff_qm7b.py
"""

import numpy as np

from kernelforge.cli import load_qm7b_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf
from kernelforge.kernelmath import solve_cholesky
from kernelforge.kitchen_sinks import rff_features_elemental, rff_gramian_elemental

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRAIN = 1000  # training set size
N_TEST = 1000  # test set size
D = 2000  # number of random Fourier features
SIGMA = 6.0  # Gaussian kernel width (tuned for QM7b FCHL19 rep space)
LLAMBDA = 1e-8  # L2 regularisation strength
CHUNK = 250  # molecules per Gramian chunk

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
# Build element-index mapping
#
# rff_features_elemental expects Q[i] to contain 0-based indices into W/b,
# not raw atomic numbers.
#   H->0, C->1, N->2, O->3, S->4, Cl->5
# ---------------------------------------------------------------------------

elem_to_idx = {e: i for i, e in enumerate(ELEMENTS)}

# ---------------------------------------------------------------------------
# Generate FCHL19 representations and pad to max_atoms
#
# Molecules in QM7b have different sizes. rff_features_elemental needs a
# single padded array X of shape (N, max_atoms, rep_size). Padding rows
# (beyond the true atom count) are zeroed and ignored because those atom
# indices do not appear in Q[i] (which only covers the true atoms).
# ---------------------------------------------------------------------------

max_atoms_train = int(N_all[:N_TRAIN].max())
rep_size = 984  # FCHL19 with 6-element basis

print(f"Generating FCHL19 representations for {N_TRAIN} training structures ...")
X_train = np.zeros((N_TRAIN, max_atoms_train, rep_size))
Q_train = []
for i in range(N_TRAIN):
    nat = int(N_all[i])
    rep = generate_fchl_acsf(R_train[i], z_train[i], elements=ELEMENTS)
    X_train[i, :nat, :] = rep
    Q_train.append(np.array([elem_to_idx[zi] for zi in z_train[i]], dtype=np.int32))

max_atoms_test = int(N_all[N_TRAIN : N_TRAIN + N_TEST].max())

print(f"Generating FCHL19 representations for {N_TEST} test structures ...")
X_test = np.zeros((N_TEST, max_atoms_test, rep_size))
Q_test = []
for i in range(N_TEST):
    nat = int(N_all[N_TRAIN + i])
    rep = generate_fchl_acsf(R_test[i], z_test[i], elements=ELEMENTS)
    X_test[i, :nat, :] = rep
    Q_test.append(np.array([elem_to_idx[zi] for zi in z_test[i]], dtype=np.int32))

nelements = len(ELEMENTS)
print(f"Representation size: {rep_size},  max_atoms (train): {max_atoms_train}")

# ---------------------------------------------------------------------------
# Sample random Fourier basis
#
# Approximates k(x, x') = exp(-||x-x'||^2 / (2*sigma^2)) via
#   phi_e(x) = sqrt(2/D) * cos(x @ W[e] + b[e])
# with W[e] ~ N(0, 1/sigma^2) and b[e] ~ Uniform(0, 2*pi).
# One independent (W[e], b[e]) per element type e.
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
W = rng.standard_normal((nelements, rep_size, D)) / SIGMA  # (nelements, rep_size, D)
b = rng.uniform(0.0, 2.0 * np.pi, (nelements, D))  # (nelements, D)

# ---------------------------------------------------------------------------
# Build the Gramian matrix in memory-efficient chunks
# ---------------------------------------------------------------------------

print(f"Building Gramian ({D}x{D}) in chunks of {CHUNK} ...")
LZtLZ, LZtY = rff_gramian_elemental(X_train, Q_train, W, b, E_train, chunk_size=CHUNK)

# ---------------------------------------------------------------------------
# Solve for regression coefficients
# ---------------------------------------------------------------------------

print("Solving linear system via Cholesky ...")
alpha = solve_cholesky(LZtLZ, LZtY, regularize=LLAMBDA)  # (D,)

# ---------------------------------------------------------------------------
# Predict and evaluate
# ---------------------------------------------------------------------------

LZ_test = rff_features_elemental(X_test, Q_test, W, b)
LZ_train = rff_features_elemental(X_train, Q_train, W, b)

E_pred_test = LZ_test @ alpha + E_mean
E_pred_train = LZ_train @ alpha + E_mean
E_train_abs = E_train + E_mean

mae_train = np.mean(np.abs(E_pred_train - E_train_abs))
mae_test = np.mean(np.abs(E_pred_test - E_test))
rmse_test = np.sqrt(np.mean((E_pred_test - E_test) ** 2))

print()
print("=" * 52)
print("  RFF energy regression -- QM7b")
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
