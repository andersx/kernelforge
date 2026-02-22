"""
KRR with combined energy + force training — predict both energies and forces.

Trains simultaneously on energies and forces using the full combined kernel,
which fuses the scalar, Jacobian, and Hessian blocks into a single BIG×BIG
system (BIG = N*(1+D)):

  K_full[0:N, 0:N]   = scalar   (energy-energy)
  K_full[N:,  0:N]   = jacobian (force-energy)
  K_full[0:N, N:]    = jac_t    (energy-force)
  K_full[N:,  N:]    = hessian  (force-force)

Kernel usage
------------
  Training kernel :  kernel_gaussian_full_symm_rfp  — full combined, symmetric, RFP
  Training error  :  kernel_gaussian_full_symm       — full combined, symmetric, full
  Predict E + F   :  kernel_gaussian_full             — full combined, asymmetric
                     shape (N_test*(1+D), N_train*(1+D))

The prediction kernel applied to alpha gives:
  y_pred[:N_test]    = predicted energies
  y_pred[N_test:]    = predicted forces (flattened)

Dataset: ethanol MD17, inverse-distance representation (M=36, D=27).
"""

import time

import numpy as np

from kernelforge import invdist_repr, kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.global_kernels import (
    kernel_gaussian_full,
    kernel_gaussian_full_symm,
    kernel_gaussian_full_symm_rfp,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 200
N_TEST = 50
SIGMA = 3.0
L2 = 1e-8


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int):
    """Return representations, combined labels for n_train + n_test structures."""
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
    dX = np.array(dX_list, dtype=np.float64)  # (n_total, M=36, D=27)
    D = dX.shape[2]

    # Combined labels: energies concatenated with flattened forces
    y_tr = np.concatenate([E[:n_train], F[:n_train].ravel()])  # (N_train*(1+D),)
    y_te = np.concatenate([E[n_train:], F[n_train:].ravel()])  # (N_test *(1+D),)

    return (
        X[:n_train],
        dX[:n_train],
        X[n_train:],
        dX[n_train:],
        y_tr,
        y_te,
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
    print("KRR: energy + force training  →  predict energies + forces")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    (X_tr, dX_tr, X_te, dX_te, y_tr, y_te, E_tr, E_te, F_tr, F_te, D) = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    BIG_tr = N_TRAIN * (1 + D)
    BIG_te = N_TEST * (1 + D)
    print(f"    M={X_tr.shape[1]}  D={D}  BIG_train={BIG_tr}  BIG_test={BIG_te}")

    # ------------------------------------------------------------------
    # 2. Build training kernel  —  full combined, symmetric, RFP packed
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_rfp = kernel_gaussian_full_symm_rfp(X_tr, dX_tr, SIGMA)
    print(f"\n[2] Training kernel (full, RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    K_rfp length={len(K_rfp)}  ({len(K_rfp) * 8 / 1024**2:.1f} MB)")
    assert len(K_rfp) == BIG_tr * (BIG_tr + 1) // 2

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} y_train
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    alpha = kernelmath.cho_solve_rfp(K_rfp, y_tr, l2=L2)
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error  —  full symmetric combined kernel
    # ------------------------------------------------------------------
    K_tr_full = kernel_gaussian_full_symm(X_tr, dX_tr, SIGMA)  # (BIG_tr, BIG_tr)
    y_tr_pred = K_tr_full @ alpha
    train_mae_E = np.mean(np.abs(y_tr_pred[:N_TRAIN] - E_tr))
    train_mae_F = np.mean(np.abs(y_tr_pred[N_TRAIN:].reshape(N_TRAIN, D) - F_tr))
    print(
        f"\n[4] Training MAE — energy: {train_mae_E:.6f} kcal/mol"
        f"   force: {train_mae_F:.6f} kcal/(mol·Å)"
    )

    # ------------------------------------------------------------------
    # 5. Test prediction  —  asymmetric full combined kernel
    #    K_pred shape: (BIG_te, BIG_tr)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_pred = kernel_gaussian_full(X_te, dX_te, X_tr, dX_tr, SIGMA)  # (BIG_te, BIG_tr)
    y_te_pred = K_pred @ alpha  # (BIG_te,)
    print(f"\n[5] Prediction kernel built in {time.perf_counter() - t0:.3f}s")
    assert K_pred.shape == (BIG_te, BIG_tr)

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, D)

    E_te_pred_c = E_te_pred - E_te_pred.mean()
    E_te_c = E_te - E_te.mean()
    test_mae_E = np.mean(np.abs(E_te_pred_c - E_te_c))
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))

    print(f"\n[6] Test results")
    print(f"    Energy MAE (centred): {test_mae_E:.4f} kcal/mol")
    print(f"    Force  MAE          : {test_mae_F:.4f} kcal/(mol·Å)")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
