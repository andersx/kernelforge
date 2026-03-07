"""
KRR with force-only training -- predict both energies and forces.

Trains on forces alone using the Hessian (force-force) kernel, then predicts:
  - Forces   via the Hessian kernel        (force-force, asymmetric)
  - Energies via the transposed-Jacobian   (energy-force, asymmetric)
    kernel_gaussian_jacobian_t(X_test, X_train, dX_train, sigma)
    maps the force-coefficient vector alpha to energy predictions.

Kernel usage
------------
  Training kernel :  kernel_gaussian_hessian_symm_rfp  -- Hessian, symmetric, RFP
  Training error  :  kernel_gaussian_hessian_symm       -- Hessian, symmetric, full
  Predict forces  :  kernel_gaussian_hessian             -- Hessian, asymmetric
                     shape (N_test*ncoords, N_train*ncoords)
  Predict energies:  kernel_gaussian_jacobian_t          -- Jacobian-transpose, asymmetric
                     shape (N_test, N_train*ncoords)

The jacobian_t kernel K_jt[i, j*ncoords+d] = dK(x_test_i, x_train_j)/d(coord_d_j),
so K_jt @ alpha (with alpha the force coefficients) gives energy predictions.

Dataset: ethanol MD17, inverse-distance representation (M=36, ncoords=27).
"""

import time

import numpy as np

from kernelforge import invdist_repr, kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.global_kernels import (
    kernel_gaussian_hessian,
    kernel_gaussian_hessian_symm_rfp,
    kernel_gaussian_jacobian_t,
)


def linfit(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) from least-squares fit of y_pred vs y_true."""
    slope, intercept = np.polyfit(y_true.ravel(), y_pred.ravel(), 1)
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 1000
N_TEST = 200
SIGMA = 3.0
L2 = 1e-8


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int):
    """Return representations, energies, and forces for n_train + n_test structures."""
    data = load_ethanol_raw_data()
    n_total = n_train + n_test

    R = data["R"][:n_total]
    E = data["E"][:n_total].ravel()  # (n_total,)
    F = data["F"][:n_total].reshape(n_total, -1)  # (n_total, D=27)

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list, dtype=np.float64)  # (n_total, M=36)
    dX = np.array(dX_list, dtype=np.float64)  # (n_total, ncoords=27, M=36)
    D = dX.shape[1]  # ncoords (Cartesian degrees of freedom)

    return (
        X[:n_train],
        dX[:n_train],
        X[n_train:],
        dX[n_train:],
        E[:n_train],
        E[n_train:],
        F[:n_train],
        F[n_train:],
        D,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("KRR: force-only training  ->  predict energies + forces")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    (X_tr, dX_tr, X_te, dX_te, E_tr, E_te, F_tr, F_te, D) = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    print(f"    M={X_tr.shape[1]}  ncoords={D}")

    F_tr_flat = F_tr.ravel()  # (N_train*ncoords,) -- training labels

    # ------------------------------------------------------------------
    # 2. Build training kernel  --  Hessian, symmetric, RFP packed
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_rfp = kernel_gaussian_hessian_symm_rfp(X_tr, dX_tr, SIGMA)
    ND = N_TRAIN * D  # D = ncoords
    print(f"\n[2] Training kernel (hessian, RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    K_rfp length={len(K_rfp)}  ({len(K_rfp) * 8 / 1024**2:.1f} MB)")
    assert len(K_rfp) == ND * (ND + 1) // 2

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} F_train_flat
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    alpha = kernelmath.cho_solve_rfp(K_rfp, F_tr_flat, l2=L2)
    del K_rfp  # free ~2.7 GB RFP buffer
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error  —  derived from the normal equations (no extra allocation)
    #    We solved (K + l2*I) @ alpha = y, so K @ alpha = y - l2*alpha exactly.
    # ------------------------------------------------------------------
    F_tr_pred_flat = F_tr_flat - L2 * alpha  # K @ alpha = y - l2*alpha
    F_tr_pred = F_tr_pred_flat.reshape(N_TRAIN, D)
    train_mae = np.mean(np.abs(F_tr_pred - F_tr))
    slope_F_tr, intercept_F_tr = linfit(F_tr, F_tr_pred)
    print(
        f"\n[4] Training MAE (force): {train_mae:.6f} kcal/(mol*A)"
        f"  slope={slope_F_tr:.4f}  intercept={intercept_F_tr:.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Test prediction — forces via Hessian kernel
    #    K_hess shape: (N_test*D, N_train*D)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_hess = kernel_gaussian_hessian(X_te, dX_te, X_tr, dX_tr, SIGMA)  # (N_test*D, N_train*D)
    F_te_pred = (K_te_hess @ alpha).reshape(N_TEST, D)
    del K_te_hess
    print(f"\n[5] Force prediction kernel built in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 6. Test prediction — energies via transposed-Jacobian kernel
    #    K_jt shape: (N_test, N_train*D)
    #    E_pred[i] = sum_{j,d} K_jt[i, j*D+d] * alpha[j*D+d]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_jt = kernel_gaussian_jacobian_t(X_te, X_tr, dX_tr, SIGMA)  # (N_test, N_train*ncoords)
    E_te_pred = K_te_jt @ alpha  # (N_test,)
    print(f"    Energy prediction kernel built in {time.perf_counter() - t0:.4f}s")

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    E_te_pred_c = E_te_pred - E_te_pred.mean()
    E_te_c = E_te - E_te.mean()
    test_mae_E = np.mean(np.abs(E_te_pred_c - E_te_c))
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))
    slope_E_te, intercept_E_te = linfit(E_te_c, E_te_pred_c)
    slope_F_te, intercept_F_te = linfit(F_te, F_te_pred)

    print(f"\n[7] Test results")
    print(
        f"    Energy MAE (centred): {test_mae_E:.4f} kcal/mol"
        f"  slope={slope_E_te:.4f}  intercept={intercept_E_te:.4f}"
    )
    print(
        f"    Force  MAE          : {test_mae_F:.4f} kcal/(mol*A)"
        f"  slope={slope_F_te:.4f}  intercept={intercept_F_te:.4f}"
    )
    print(f"    (Energies are predicted via the Jacobian-transpose kernel.)")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
