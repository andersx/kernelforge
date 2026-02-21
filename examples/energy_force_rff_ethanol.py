#!/usr/bin/env python3
"""
Joint energy and force regression with Random Fourier Features (RFF) on ethanol CCSD(T).

The ethanol dataset contains 1000 molecular dynamics frames (9 atoms: C2H5OH) with
CCSD(T) energies in kcal/mol and forces in kcal/mol/Angstrom.

This example demonstrates joint energy+force training with the RFF approximation to
the local FCHL19 Gaussian kernel:

  1. Load the ethanol dataset (auto-downloaded on first run).
  2. Generate FCHL19 representations and gradients dX/dR for each frame.
  3. Sample random weights W, b to define the RFF feature map phi(X).
  4. Build the Gramian matrix (LZtLZ, LZtY) in memory-efficient chunks using
     rff_gramian_elemental_gradient, which accumulates contributions from both
     energies and forces.
  5. Solve  (LZtLZ + lambda * I) @ alpha_rff = LZtY  via Cholesky.
  6. Predict energies via rff_features_elemental and forces via
     rff_gradient_elemental.
  7. Report train/test MAE for both energies and forces.

Force convention
----------------
The FCHL19 Jacobian dX/dR is the gradient of the representation w.r.t. atomic
coordinates. Forces are the negative gradient of the energy:
    F_i = -dE/dR_i.
The ethanol npz stores forces with this sign (F = -dE/dR), so they are passed
directly to rff_gramian_elemental_gradient.

Array shapes
------------
  X    (N, natoms, rep_size)                  padded representations
  dX   (N, natoms, rep_size, natoms, 3)       5D Jacobian (reshaped from 4D)
  Q    list of N int32 arrays, 0-based        element index per atom
  F    (N * natoms * 3,)                      packed forces (flat)

Usage
-----
    uv run python examples/energy_force_rff_ethanol.py
"""

import numpy as np

from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf_and_gradients
from kernelforge.kernelmath import solve_cholesky
from kernelforge.kitchen_sinks import (
    rff_features_elemental,
    rff_gradient_elemental,
    rff_gramian_elemental_gradient,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRAIN = 500  # training set size
N_TEST = 400  # test set size
D = 2000  # number of RFF features
SIGMA = 2.0  # Gaussian kernel width (tuned for ethanol FCHL19)
LLAMBDA = 1e-8  # L2 regularisation strength
CHUNK_ENERGY = 200  # molecules per energy chunk
CHUNK_FORCE = 50  # molecules per force chunk (larger memory cost per mol)
RNG_SEED = 42

ELEMENTS = [1, 6, 8]  # H, C, O -- elements in ethanol
NATOMS = 9  # atoms per ethanol molecule (fixed topology)

# 0-based element index for each atom in ethanol: z = [6,6,8,1,1,1,1,1,1]
# Elements list [1,6,8] -> index: H=0, C=1, O=2
_Z_TO_IDX = {1: 0, 6: 1, 8: 2}

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

print("Loading ethanol CCSD(T) dataset ...")
data = load_ethanol_raw_data()

R_all = data["R"]  # (1000, 9, 3)  Angstrom
E_all = data["E"].flatten()  # (1000,)  kcal/mol
F_all = data["F"]  # (1000, 9, 3)  kcal/mol/Angstrom
z = data["z"].astype(np.int32)  # (9,)  atomic numbers

R_train = R_all[:N_TRAIN]
E_train = E_all[:N_TRAIN].copy()
F_train_3d = F_all[:N_TRAIN]  # (N_TRAIN, 9, 3)

R_test = R_all[N_TRAIN : N_TRAIN + N_TEST]
E_test = E_all[N_TRAIN : N_TRAIN + N_TEST]
F_test_3d = F_all[N_TRAIN : N_TRAIN + N_TEST]  # (N_TEST, 9, 3)

# Mean-centre energies
E_mean = E_train.mean()
E_train = E_train - E_mean

# ---------------------------------------------------------------------------
# Generate FCHL19 representations and gradients.
# generate_fchl_acsf_and_gradients returns rep (natoms, rep_size) and
# grad (natoms, rep_size, 3*natoms). Stacked: (N, natoms, rep_size, 3*natoms).
# RFF functions require the Jacobian as 5D: (N, natoms, rep_size, natoms, 3).
# ---------------------------------------------------------------------------

rep_size = 312  # FCHL19 with [1,6,8], nRs2=24, nRs3=20, nFourier=1

print(f"Generating FCHL19 representations for {N_TRAIN} training structures ...")
x_train_list = []
dx_train_list = []
for r in R_train:
    rep, grad = generate_fchl_acsf_and_gradients(r, z, elements=ELEMENTS)
    x_train_list.append(rep)
    dx_train_list.append(grad)

X_train = np.asarray(x_train_list)  # (N_TRAIN, 9, 312)
dx_train_4d = np.asarray(dx_train_list)  # (N_TRAIN, 9, 312, 27)
dx_train = dx_train_4d.reshape(N_TRAIN, NATOMS, rep_size, NATOMS, 3)  # 5D for RFF

print(f"Generating FCHL19 representations for {N_TEST} test structures ...")
x_test_list = []
dx_test_list = []
for r in R_test:
    rep, grad = generate_fchl_acsf_and_gradients(r, z, elements=ELEMENTS)
    x_test_list.append(rep)
    dx_test_list.append(grad)

X_test = np.asarray(x_test_list)  # (N_TEST, 9, 312)
dx_test_4d = np.asarray(dx_test_list)  # (N_TEST, 9, 312, 27)
dx_test = dx_test_4d.reshape(N_TEST, NATOMS, rep_size, NATOMS, 3)  # 5D for RFF

# Q: list of 0-based element index arrays (one per molecule)
q_mol = np.array([_Z_TO_IDX[int(zi)] for zi in z], dtype=np.int32)  # (9,)
Q_train = [q_mol] * N_TRAIN
Q_test = [q_mol] * N_TEST

# Pack forces as flat 1D arrays: (N * natoms * 3,)
F_train_flat = F_train_3d.reshape(-1)  # (N_TRAIN * 9 * 3,)

print(f"Representation size: {rep_size},  D: {D}")

# ---------------------------------------------------------------------------
# Sample random RFF parameters
# ---------------------------------------------------------------------------

rng = np.random.default_rng(RNG_SEED)
nelements = len(ELEMENTS)
W = rng.standard_normal((nelements, rep_size, D))  # (3, 312, D)
b = rng.uniform(0, 2 * np.pi, size=(nelements, D))  # (3, D)

# ---------------------------------------------------------------------------
# Build the Gramian matrix (energy + force contributions)
# ---------------------------------------------------------------------------

print(
    f"Building Gramian ({D}x{D}) from {N_TRAIN} energies and "
    f"{N_TRAIN * NATOMS * 3} force components ..."
)
LZtLZ, LZtY = rff_gramian_elemental_gradient(
    X_train,
    dx_train,
    Q_train,
    W,
    b,
    E_train,
    F_train_flat,
    energy_chunk=CHUNK_ENERGY,
    force_chunk=CHUNK_FORCE,
)

# ---------------------------------------------------------------------------
# Solve for regression coefficients
# ---------------------------------------------------------------------------

print("Solving linear system via Cholesky ...")
alpha = solve_cholesky(LZtLZ, LZtY, regularize=LLAMBDA)  # (D,)

# ---------------------------------------------------------------------------
# Predict energies
# ---------------------------------------------------------------------------

LZ_train = rff_features_elemental(X_train, Q_train, W, b)  # (N_TRAIN, D)
LZ_test = rff_features_elemental(X_test, Q_test, W, b)  # (N_TEST, D)

E_pred_train = LZ_train @ alpha + E_mean
E_pred_test = LZ_test @ alpha + E_mean
E_train_abs = E_train + E_mean

mae_e_train = np.mean(np.abs(E_pred_train - E_train_abs))
mae_e_test = np.mean(np.abs(E_pred_test - E_test))

# ---------------------------------------------------------------------------
# Predict forces via the gradient of RFF features
#
# rff_gradient_elemental returns G with shape (D, N*natoms*3).
# rff_gramian_elemental_gradient sets up: LZtY += G @ F, so the model fits
# G.T @ alpha ≈ F directly. Force predictions are therefore G.T @ alpha.
# ---------------------------------------------------------------------------

G_train = rff_gradient_elemental(X_train, dx_train, Q_train, W, b)  # (D, N_TRAIN*27)
G_test = rff_gradient_elemental(X_test, dx_test, Q_test, W, b)  # (D, N_TEST*27)

F_pred_train_flat = G_train.T @ alpha  # (N_TRAIN*27,)
F_pred_test_flat = G_test.T @ alpha  # (N_TEST*27,)

F_train_flat_ref = F_train_3d.reshape(-1)
F_test_flat_ref = F_test_3d.reshape(-1)

mae_f_train = np.mean(np.abs(F_pred_train_flat - F_train_flat_ref))
mae_f_test = np.mean(np.abs(F_pred_test_flat - F_test_flat_ref))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 56)
print("  RFF energy+force regression -- Ethanol CCSD(T)")
print("=" * 56)
print(f"  N_train      : {N_TRAIN}")
print(f"  N_test       : {N_TEST}")
print(f"  D (RFF)      : {D}")
print(f"  sigma        : {SIGMA}")
print(f"  lambda       : {LLAMBDA:.0e}")
print(f"  Train E MAE  : {mae_e_train:.4f} kcal/mol")
print(f"  Test  E MAE  : {mae_e_test:.4f} kcal/mol")
print(f"  Train F MAE  : {mae_f_train:.4f} kcal/mol/Angstrom")
print(f"  Test  F MAE  : {mae_f_test:.4f} kcal/mol/Angstrom")
print("=" * 56)
