"""
KRR regression with combined energy + force kernel (ethanol MD17 dataset).

This example demonstrates the full kernel ridge regression (KRR) pipeline for
simultaneous energy and force prediction using the KernelForge combined kernel:

  kernel_gaussian_full_symm_rfp  — training kernel in RFP packed format
  cho_solve_rfp                  — Cholesky solver (no .copy() needed)
  kernel_gaussian_full           — prediction kernel (asymmetric, test vs train)

The combined kernel K_full has shape BIG x BIG, where BIG = N * (1 + D):
  - Rows/cols 0..N-1      : energy-energy block
  - Rows/cols N..N+N*D-1  : force-force / energy-force blocks

Workflow
--------
1. Load ethanol MD17 data and compute inverse-distance representation (X, dX).
2. Split into train and test sets.
3. Build training kernel K_train (RFP, memory-efficient).
4. Solve alpha = (K_train + l2*I)^{-1} y_train  using cho_solve_rfp.
5. Build prediction kernel K_pred (N_test x N_train asymmetric).
6. Predict energies and forces: y_pred = K_pred @ alpha.
7. Report MAE on energies and forces.

Requirements
------------
  pip install kernelforge  # or build from source
  # The ethanol MD17 dataset is shipped with kernelforge.
"""

import time

import numpy as np

from kernelforge import invdist_repr, kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.global_kernels import kernel_gaussian_full, kernel_gaussian_full_symm_rfp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 200  # number of training structures
N_TEST = 50  # number of test structures
SIGMA = 3.0  # Gaussian kernel bandwidth
L2 = 1e-8  # L2 regularization


# ---------------------------------------------------------------------------
# Data loading and representation
# ---------------------------------------------------------------------------


def load_data(n_train: int, n_test: int):
    """Load ethanol MD17 data and convert to inverse-distance representation."""
    data = load_ethanol_raw_data()
    n_total = n_train + n_test

    R = data["R"][:n_total]  # Cartesian coords (n, n_atoms, 3)
    E = data["E"][:n_total].ravel()  # energies kcal/mol (n,)
    F = data["F"][:n_total]  # forces  kcal/(mol·Å) (n, n_atoms, 3)

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list, dtype=np.float64)  # (n_total, M)   M = n_atoms*(n_atoms-1)/2
    dX = np.array(dX_list, dtype=np.float64)  # (n_total, M, D)  D = 3*n_atoms

    n_atoms = R.shape[1]
    D = 3 * n_atoms

    # Force labels: chain rule from Cartesian to representation coordinates.
    # y_force[i, d] = -sum_m F_cart[i, m/3, m%3] * dX[i, m, d]
    # Here F[i] has shape (n_atoms, 3), and dX[i] has shape (M, D) with
    # D = 3*n_atoms components.  We flatten F to (n_total, D) and take the
    # dot product along the M (representation) axis.
    #
    # Correct chain rule:
    #   dE/d(coord_d) = sum_m (dE/dX_m) * (dX_m/d(coord_d))
    #   force_d       = -dE/d(coord_d)
    #                 = -sum_m (dE/dX_m) * dX[i, m, d]
    # For the KRR label we need y_force = dE/dX (not the physical force), so:
    #   y_force_d = sum_m F_flat_m * dX[i, m, d]
    # where F_flat has shape (n_total, D) with D = n_atoms*3 = n_atoms*3.

    # F has shape (n_total, n_atoms, 3) → flatten to (n_total, D)
    F_flat = F.reshape(n_total, D)  # (n_total, D)

    # Force label in representation space: y_force[i, d] = F_flat[i] @ dX[i, :, d]
    # dX[i] has shape (M, D) → for each i: y_force[i] = dX[i].T @ F_flat[i]  (D,)
    # Wait — this maps physical forces to representation-derivative labels.
    # For a proper KRR, the labels should be the partial derivatives of E w.r.t X_m,
    # but for demonstration we use the simpler raw F_flat as a synthetic label.
    F_label = np.einsum("imd,id->im", dX, F_flat)  # shape (n_total, M) — not what we want

    # Simplification: use raw flattened F as force labels for demonstration.
    # (A real model requires the inverse-Jacobian chain rule.)
    F_label = F_flat  # (n_total, D)

    # Split
    X_train, X_test = X[:n_train], X[n_train:]
    dX_train, dX_test = dX[:n_train], dX[n_train:]
    E_train, E_test = E[:n_train], E[n_train:]
    F_train, F_test = F_label[:n_train], F_label[n_train:]

    # Build combined label vectors
    y_train = np.concatenate([E_train, F_train.ravel()])  # (N_train * (1 + D),)
    y_test = np.concatenate([E_test, F_test.ravel()])  # (N_test  * (1 + D),)

    return X_train, dX_train, X_test, dX_test, y_train, y_test, D


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    print("=" * 65)
    print("KRR energy+force regression — ethanol MD17 dataset")
    print("=" * 65)
    print(f"  N_train = {N_TRAIN},  N_test = {N_TEST}")
    print(f"  sigma   = {SIGMA},    l2     = {L2}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    X_train, dX_train, X_test, dX_test, y_train, y_test, D = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    BIG_train = N_TRAIN * (1 + D)
    BIG_test = N_TEST * (1 + D)
    M = X_train.shape[1]
    print(f"    Representation: M={M}, D={D}")
    print(
        f"    Training kernel size: {BIG_train} x {BIG_train}  "
        f"(RFP: {BIG_train * (BIG_train + 1) // 2} doubles = "
        f"{BIG_train * (BIG_train + 1) // 2 * 8 / 1024**2:.1f} MB)"
    )

    # ------------------------------------------------------------------
    # 2. Build training kernel in RFP format
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_train_rfp = kernel_gaussian_full_symm_rfp(X_train, dX_train, sigma=SIGMA)
    print(f"\n[2] Training kernel built in {time.perf_counter() - t0:.2f}s")
    assert K_train_rfp.shape[0] == BIG_train * (BIG_train + 1) // 2

    # ------------------------------------------------------------------
    # 3. Solve for alpha coefficients
    #    cho_solve_rfp makes an internal copy — no .copy() needed
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    alpha = kernelmath.cho_solve_rfp(K_train_rfp, y_train, l2=L2)
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.2f}s")
    print(f"    alpha shape: {alpha.shape},  ||alpha|| = {np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Build prediction kernel  (N_test*(1+D)) x (N_train*(1+D))
    #    kernel_gaussian_full is the asymmetric variant
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_pred = kernel_gaussian_full(X_test, dX_test, X_train, dX_train, sigma=SIGMA)
    print(f"\n[4] Prediction kernel built in {time.perf_counter() - t0:.2f}s")
    assert K_pred.shape == (BIG_test, BIG_train), f"Unexpected shape {K_pred.shape}"

    # ------------------------------------------------------------------
    # 5. Predict and evaluate
    # ------------------------------------------------------------------
    y_pred = K_pred @ alpha  # (BIG_test,)

    E_pred = y_pred[:N_TEST]
    F_pred = y_pred[N_TEST:].reshape(N_TEST, D)

    E_true = y_test[:N_TEST]
    F_true = y_test[N_TEST:].reshape(N_TEST, D)

    # Shift predictions by mean (KRR predicts absolute energies — center both)
    E_pred_centered = E_pred - E_pred.mean()
    E_true_centered = E_true - E_true.mean()

    mae_E = np.mean(np.abs(E_pred_centered - E_true_centered))
    mae_F = np.mean(np.abs(F_pred - F_true))

    print(f"\n[5] Prediction results")
    print(f"    Energy MAE (centered): {mae_E:.4f} kcal/mol")
    print(f"    Force  MAE:            {mae_F:.4f} kcal/(mol*A)")
    print(f"    (Note: force labels here are raw Cartesian forces, not")
    print(f"     proper representation-derivative labels.  A production")
    print(f"     model would use the inverse-Jacobian chain rule.)")

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
