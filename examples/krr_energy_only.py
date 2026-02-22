"""
KRR with energy-only training — predict both energies and forces.

Trains on energies alone using the scalar Gaussian kernel, then predicts:
  - Energies via the scalar kernel  (energy-energy)
  - Forces  via the Jacobian kernel (analytical derivative of the energy
    prediction w.r.t. test-set atomic coordinates)

Kernel usage
------------
  Training kernel :  kernel_gaussian_symm_rfp   — scalar, symmetric, RFP packed
  Training error  :  kernel_gaussian_symm        — scalar, symmetric, full matrix
  Predict energies:  kernel_gaussian              — scalar, asymmetric
  Predict forces  :  kernel_gaussian_jacobian     — Jacobian, asymmetric
                     shape (N_test*D, N_train)

The Jacobian kernel K_jac[i*D+d, j] = dK(x_test_i, x_train_j)/d(coord_d_i),
so K_jac @ alpha gives the gradient of the KRR energy prediction w.r.t. the
test-set representation coordinates (= predicted forces up to a sign convention).

Dataset: ethanol MD17, inverse-distance representation (M=36, D=27).
"""

import time

import numpy as np

from kernelforge import invdist_repr, kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.global_kernels import (
    kernel_gaussian,
    kernel_gaussian_jacobian,
    kernel_gaussian_symm,
    kernel_gaussian_symm_rfp,
)

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
    dX = np.array(dX_list, dtype=np.float64)  # (n_total, M=36, D=27)
    D = dX.shape[2]

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
    print("KRR: energy-only training  →  predict energies + forces")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    (X_tr, dX_tr, X_te, dX_te, E_tr, E_te, F_tr, F_te, D) = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    print(f"    M={X_tr.shape[1]}  D={D}")

    alpha_scalar = -1.0 / (2.0 * SIGMA**2)

    # ------------------------------------------------------------------
    # 2. Build training kernel  —  scalar, symmetric, RFP packed
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_rfp = kernel_gaussian_symm_rfp(X_tr, alpha_scalar)
    print(f"\n[2] Training kernel (scalar, RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    length={len(K_rfp)}  ({len(K_rfp) * 8 / 1024:.1f} KB)")
    assert len(K_rfp) == N_TRAIN * (N_TRAIN + 1) // 2

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} E_train
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    alpha = kernelmath.cho_solve_rfp(K_rfp, E_tr, l2=L2)
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error  —  full symmetric scalar kernel
    # ------------------------------------------------------------------
    K_tr_full = kernel_gaussian_symm(X_tr, alpha_scalar)  # (N_train, N_train)
    E_tr_pred = K_tr_full @ alpha
    train_mae = np.mean(np.abs(E_tr_pred - E_tr))
    print(f"\n[4] Training MAE (energy): {train_mae:.6f} kcal/mol")

    # ------------------------------------------------------------------
    # 5. Test prediction — energies
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_scalar = kernel_gaussian(X_te, X_tr, alpha_scalar)  # (N_test, N_train)
    E_te_pred = K_te_scalar @ alpha  # (N_test,)
    print(f"\n[5] Energy prediction kernel built in {time.perf_counter() - t0:.4f}s")

    # ------------------------------------------------------------------
    # 6. Test prediction — forces via Jacobian kernel
    #    K_jac shape: (N_test*D, N_train)
    #    F_pred[i, d] = sum_j K_jac[i*D+d, j] * alpha[j]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_jac = kernel_gaussian_jacobian(X_te, dX_te, X_tr, SIGMA)  # (N_test*D, N_train)
    F_te_pred = (K_te_jac @ alpha).reshape(N_TEST, D)  # (N_test, D)
    print(f"    Force  prediction kernel built in {time.perf_counter() - t0:.4f}s")

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    # Centre energies — KRR doesn't learn the absolute energy offset
    E_te_pred_c = E_te_pred - E_te_pred.mean()
    E_te_c = E_te - E_te.mean()
    test_mae_E = np.mean(np.abs(E_te_pred_c - E_te_c))
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))

    print(f"\n[7] Test results")
    print(f"    Energy MAE (centred): {test_mae_E:.4f} kcal/mol")
    print(f"    Force  MAE          : {test_mae_F:.4f} kcal/(mol·Å)")
    print(f"    (Forces are predicted as gradients of the energy KRR model.)")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
