"""
Local KRR with force-only training — predict both energies and forces.

Trains on forces alone using the local Hessian (force-force) kernel, then predicts:
  - Forces   via the local Hessian kernel           (force-force, asymmetric)
  - Energies via the local transposed-Jacobian      (energy-force, asymmetric)
    kernel_gaussian_jacobian_t maps the force-coefficient vector alpha to
    energy predictions.

Kernel usage
------------
  Training kernel :  kernel_gaussian_hessian_symm_rfp  — Hessian, symmetric, RFP
  Training error  :  derived from normal equations      (no extra allocation)
  Predict forces  :  kernel_gaussian_hessian             — Hessian, asymmetric
                     shape (N_test*naq, N_train*naq)
  Predict energies:  kernel_gaussian_jacobian             — Jacobian, asymmetric
                     shape (N_test, N_train*naq)

The jacobian kernel K_j[i, j*naq+d] = dK(x_test_i, x_train_j)/d(coord_d_j),
so -(K_j @ alpha) (with alpha the force coefficients) gives energy predictions.

Dataset: ethanol MD17, FCHL19 representation.
"""

import time

import numpy as np

from kernelforge import kernelmath


def linfit(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) from least-squares fit of y_pred vs y_true."""
    slope, intercept = np.polyfit(y_true.ravel(), y_pred.ravel(), 1)
    return float(slope), float(intercept)


from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf_and_gradients
from kernelforge.local_kernels import (
    kernel_gaussian_hessian,
    kernel_gaussian_hessian_symm_rfp,
    kernel_gaussian_jacobian,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 500
N_TEST = 200
SIGMA = 20.0
L2 = 1e-9
ELEMENTS = [1, 6, 8]  # H, C, O


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int):
    """Return FCHL19 representations, energies, and forces for n_train + n_test structures."""
    data = load_ethanol_raw_data()
    z = data["z"].astype(np.int32)  # (9,) atomic numbers — same for all ethanol structures
    n_atoms = len(z)
    n_total = n_train + n_test

    R = data["R"][:n_total]  # (n_total, 9, 3)
    E = data["E"][:n_total].ravel()  # (n_total,)
    F = data["F"][:n_total].reshape(n_total, -1)  # (n_total, naq=27)

    X_list, dX_list = [], []
    for r in R:
        x, dx = generate_fchl_acsf_and_gradients(r, z, elements=ELEMENTS)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list, dtype=np.float64)  # (n_total, n_atoms, rep_size)
    dX = np.array(dX_list, dtype=np.float64)  # (n_total, n_atoms, rep_size, 3*n_atoms)
    Q = np.tile(z, (n_total, 1))  # (n_total, n_atoms) int32
    N = np.full(n_total, n_atoms, dtype=np.int32)  # (n_total,)
    naq = n_atoms * 3  # number of atomic coordinates = 27 for ethanol

    return (
        X[:n_train],
        dX[:n_train],
        X[n_train:],
        dX[n_train:],
        Q[:n_train],
        Q[n_train:],
        N[:n_train],
        N[n_train:],
        E[:n_train],
        E[n_train:],
        F[:n_train],
        F[n_train:],
        naq,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Local KRR: force-only training  →  predict energies + forces")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")

    # ------------------------------------------------------------------
    # 1. Load data and generate FCHL19 representations
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    (
        X_tr,
        dX_tr,
        X_te,
        dX_te,
        Q_tr,
        Q_te,
        N_tr,
        N_te,
        E_tr,
        E_te,
        F_tr,
        F_te,
        naq,
    ) = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    n_atoms, rep_size = X_tr.shape[1], X_tr.shape[2]
    print(f"    n_atoms={n_atoms}  rep_size={rep_size}  naq={naq}")

    F_tr_flat = F_tr.ravel()  # (N_train*naq,) — training labels
    ND = N_TRAIN * naq

    # ------------------------------------------------------------------
    # 2. Build training kernel  —  Hessian, symmetric, RFP packed
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_rfp = kernel_gaussian_hessian_symm_rfp(X_tr, dX_tr, Q_tr, N_tr, SIGMA)
    print(f"\n[2] Training kernel (hessian, RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    K_rfp length={len(K_rfp)}  ({len(K_rfp) * 8 / 1024**2:.1f} MB)")
    assert len(K_rfp) == ND * (ND + 1) // 2

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} F_train_flat
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    alpha = kernelmath.cho_solve_rfp(K_rfp, F_tr_flat, l2=L2)
    del K_rfp
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error  —  derived from the normal equations (no extra allocation)
    #    We solved (K + l2*I) @ alpha = y, so K @ alpha = y - l2*alpha exactly.
    # ------------------------------------------------------------------
    F_tr_pred_flat = F_tr_flat - L2 * alpha  # K @ alpha = y - l2*alpha
    F_tr_pred = F_tr_pred_flat.reshape(N_TRAIN, naq)
    train_mae = np.mean(np.abs(F_tr_pred - F_tr))
    slope_F_tr, intercept_F_tr = linfit(F_tr, F_tr_pred)
    print(
        f"\n[4] Training MAE (force): {train_mae:.6f} kcal/(mol·Å)"
        f"  slope={slope_F_tr:.4f}  intercept={intercept_F_tr:.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Test prediction — forces via Hessian kernel
    #    K_hess shape: (N_test*naq, N_train*naq)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_hess = kernel_gaussian_hessian(  # (N_test*naq, N_train*naq)
        X_te, X_tr, dX_te, dX_tr, Q_te, Q_tr, N_te, N_tr, SIGMA
    )
    F_te_pred = (K_te_hess @ alpha).reshape(N_TEST, naq)
    del K_te_hess
    print(f"\n[5] Force prediction kernel built in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 6. Test prediction — energies via transposed-Jacobian kernel
    #    K_jt shape: (N_test, N_train*naq)
    #    E_pred[i] = sum_{j,d} K_jt[i, j*naq+d] * alpha[j*naq+d]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_jt = kernel_gaussian_jacobian(  # (N_test, N_train*naq)
        X_te, X_tr, dX_tr, Q_te, Q_tr, N_te, N_tr, SIGMA
    )
    E_te_pred = -(
        K_te_jt @ alpha
    )  # (N_test,)  — negate: kernel returns +dK/dR, energy needs -dK/dR
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
        f"    Force  MAE          : {test_mae_F:.4f} kcal/(mol·Å)"
        f"  slope={slope_F_te:.4f}  intercept={intercept_F_te:.4f}"
    )
    print(f"    (Energies are predicted via the Jacobian-transpose kernel.)")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
