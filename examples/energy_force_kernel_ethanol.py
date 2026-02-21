#!/usr/bin/env python3
"""
Joint energy and force regression with exact kernel ridge regression (KRR) on
ethanol CCSD(T) using the GDML-style local FCHL19 kernel.

The ethanol dataset contains 1000 molecular dynamics frames (9 atoms: C2H5OH)
with CCSD(T) energies in kcal/mol and forces in kcal/mol/Angstrom.

This example uses the GDML (gradient-domain machine learning) approach where
the model is trained on atomic forces:

  1. Load the ethanol dataset (auto-downloaded on first run).
  2. Generate FCHL19 representations and gradients dX/dR for each frame.
  3. Build the symmetric hessian kernel matrix K_HH (N*3n x N*3n) using
     kernel_gaussian_hessian(X_train, X_train, ...), where n = 9 atoms.
     K_HH[ai, bj] = d^2 k(X_a, X_b) / (dR_ai dR_bj)
  4. Solve  (K_HH + lambda * I) @ alpha = F_train_flat  for the coefficient
     vector alpha (length N_train * 3n = 5400 for N=200).
  5. Predict forces on test structures:
       F_test = K_HH(train, test) @ alpha
     using kernel_gaussian_hessian(X_train, X_test, ...).
  6. Predict energies on test structures from the force-trained model:
       E_test[a] = sum_{b,t} dk(X_a, X_b) / dR_{b,t}  *  alpha_{b,t}
     using kernel_gaussian_jacobian.
  7. Report train/test MAE for both energies and forces.

Training cost is O((N*3n)^3) in time and O((N*3n)^2) in memory.
For N=200, n=9: the linear system is 5400x5400 (~220 MB).

Usage
-----
    uv run python examples/energy_force_kernel_ethanol.py
"""

import numpy as np

from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf_and_gradients
from kernelforge.kernelmath import solve_cholesky
from kernelforge.local_kernels import kernel_gaussian_hessian, kernel_gaussian_jacobian

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRAIN = 200  # training set size (O((N*3n)^3) -- keep modest)
N_TEST = 400  # test set size
SIGMA = 2.0  # Gaussian kernel width (tuned for ethanol FCHL19)
LLAMBDA = 1e-8  # L2 regularisation strength
NATOMS = 9  # atoms per ethanol molecule (fixed topology)

ELEMENTS = [1, 6, 8]  # H, C, O -- elements in ethanol

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
E_train = E_all[:N_TRAIN]
F_train = F_all[:N_TRAIN]  # (N_TRAIN, 9, 3)

R_test = R_all[N_TRAIN : N_TRAIN + N_TEST]
E_test = E_all[N_TRAIN : N_TRAIN + N_TEST]
F_test = F_all[N_TRAIN : N_TRAIN + N_TEST]  # (N_TEST, 9, 3)

# ---------------------------------------------------------------------------
# Generate FCHL19 representations and gradients.
# generate_fchl_acsf_and_gradients returns rep (natoms, rep_size) and
# grad (natoms, rep_size, 3*natoms). Stacked: (N, natoms, rep_size, 3*natoms).
# ---------------------------------------------------------------------------

rep_size = 312  # FCHL19 with [1,6,8], nRs2=24, nRs3=20, nFourier=1

print(f"Generating FCHL19 representations for {N_TRAIN} training structures ...")
x_train_list, dx_train_list = [], []
for r in R_train:
    rep, grad = generate_fchl_acsf_and_gradients(r, z, elements=ELEMENTS)
    x_train_list.append(rep)
    dx_train_list.append(grad)

X_train = np.asarray(x_train_list)  # (N_TRAIN, 9, 312)
dx_train = np.asarray(dx_train_list)  # (N_TRAIN, 9, 312, 27)
Q_train = np.tile(z, (N_TRAIN, 1))  # (N_TRAIN, 9)  raw atomic numbers
N_train_v = np.full(N_TRAIN, NATOMS, dtype=np.int32)  # true atom counts

print(f"Generating FCHL19 representations for {N_TEST} test structures ...")
x_test_list, dx_test_list = [], []
for r in R_test:
    rep, grad = generate_fchl_acsf_and_gradients(r, z, elements=ELEMENTS)
    x_test_list.append(rep)
    dx_test_list.append(grad)

X_test = np.asarray(x_test_list)  # (N_TEST, 9, 312)
dx_test = np.asarray(dx_test_list)  # (N_TEST, 9, 312, 27)
Q_test = np.tile(z, (N_TEST, 1))  # (N_TEST, 9)
N_test_v = np.full(N_TEST, NATOMS, dtype=np.int32)

print(f"Representation size: {rep_size}")

# ---------------------------------------------------------------------------
# Build the GDML hessian kernel matrix K_HH (force-force kernel)
#
# kernel_gaussian_hessian(X1, X2, ...) returns shape (N2*3n, N1*3n) where 3n=27.
# When called with X1=X2=X_train it is symmetric and serves as the training
# gram matrix for force learning.
# ---------------------------------------------------------------------------

force_dim = N_TRAIN * NATOMS * 3  # 200 * 27 = 5400
print(f"Building {force_dim}x{force_dim} hessian kernel matrix (sigma={SIGMA}) ...")
K_HH = kernel_gaussian_hessian(
    X_train, X_train, dx_train, dx_train, Q_train, Q_train, N_train_v, N_train_v, SIGMA
)
# K_HH shape: (N_TRAIN*27, N_TRAIN*27), symmetric

# ---------------------------------------------------------------------------
# Solve for regression coefficients
#
# Solve (K_HH + lambda * I) @ alpha = F_train_flat.
# alpha has shape (N_TRAIN * 3n,).
# ---------------------------------------------------------------------------

F_train_flat = F_train.reshape(-1)  # (N_TRAIN * 27,)

print("Solving linear system for force coefficients ...")
alpha = solve_cholesky(K_HH, F_train_flat, regularize=LLAMBDA)  # (N_TRAIN*27,)

# In-sample force MAE
F_pred_train_flat = K_HH @ alpha
mae_f_train = np.mean(np.abs(F_pred_train_flat - F_train_flat))

# ---------------------------------------------------------------------------
# Predict forces on test set
#
# kernel_gaussian_hessian(X_train, X_test, ...) returns (N_test*3n, N_train*3n).
# F_pred_test = K_HH_test @ alpha  (shape: N_TEST*3n)
# ---------------------------------------------------------------------------

test_force_dim = N_TEST * NATOMS * 3
print(f"Building {test_force_dim}x{force_dim} hessian kernel for test forces ...")
K_HH_test = kernel_gaussian_hessian(
    X_train, X_test, dx_train, dx_test, Q_train, Q_test, N_train_v, N_test_v, SIGMA
)
# K_HH_test shape: (N_TEST*27, N_TRAIN*27)
F_pred_test_flat = K_HH_test @ alpha  # (N_TEST * 27,)
F_test_flat = F_test.reshape(-1)

mae_f_test = np.mean(np.abs(F_pred_test_flat - F_test_flat))

# ---------------------------------------------------------------------------
# Predict energies using the Jacobian kernel
#
# For the force-trained GDML model:
#   E_pred(X_a) = sum_{b,t} J[a,b*3n+t] * alpha[b*3n+t]
# where J = kernel_gaussian_jacobian(X_test, X_train, dx_train, ...) has
# shape (N_test, N_train*3n). An integration constant (energy offset) is
# determined from the mean training energy.
# ---------------------------------------------------------------------------

print(f"Building {N_TEST}x{force_dim} Jacobian kernel for energy prediction ...")
K_J_test = kernel_gaussian_jacobian(
    X_test, X_train, dx_train, Q_test, Q_train, N_test_v, N_train_v, SIGMA
)
E_pred_test_raw = K_J_test @ alpha  # (N_TEST,)

K_J_train = kernel_gaussian_jacobian(
    X_train, X_train, dx_train, Q_train, Q_train, N_train_v, N_train_v, SIGMA
)
E_pred_train_raw = K_J_train @ alpha  # (N_TRAIN,)

# Fit integration constant: shift by (E_train.mean - E_pred_train_raw.mean)
E_shift = E_train.mean() - E_pred_train_raw.mean()

E_pred_train = E_pred_train_raw + E_shift
E_pred_test = E_pred_test_raw + E_shift

mae_e_train = np.mean(np.abs(E_pred_train - E_train))
mae_e_test = np.mean(np.abs(E_pred_test - E_test))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 58)
print("  KRR energy+force regression -- Ethanol CCSD(T)")
print("=" * 58)
print(f"  N_train        : {N_TRAIN}")
print(f"  N_test         : {N_TEST}")
print(f"  sigma          : {SIGMA}")
print(f"  lambda         : {LLAMBDA:.0e}")
print(f"  Train E MAE    : {mae_e_train:.4f} kcal/mol")
print(f"  Test  E MAE    : {mae_e_test:.4f} kcal/mol")
print(f"  Train F MAE    : {mae_f_train:.4f} kcal/mol/Angstrom")
print(f"  Test  F MAE    : {mae_f_test:.4f} kcal/mol/Angstrom")
print("=" * 58)
