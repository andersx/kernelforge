"""
Local KRR with combined energy + force training — predict both energies and forces.

Trains simultaneously on energies and forces using the full combined local kernel,
which fuses the scalar, Jacobian, and Hessian blocks into a single BIG×BIG system
(BIG = N*(1 + naq), where naq = n_atoms*3):

  K_full[0:N,   0:N  ] = scalar   (energy-energy)
  K_full[N:,    0:N  ] = jacobian (force-energy)
  K_full[0:N,   N:   ] = jac_t    (energy-force)
  K_full[N:,    N:   ] = hessian  (force-force)

Kernel usage
------------
  Training kernel :  kernel_gaussian_full_symm_rfp  — full combined, symmetric, RFP
  Training error  :  derived from normal equations  (no extra allocation)
  Predict E + F   :  kernel_gaussian_full            — full combined, asymmetric
                     shape (N_test*(1+naq), N_train*(1+naq))

The prediction kernel applied to alpha gives:
  y_pred[:N_test]    = predicted energies
  y_pred[N_test:]    = predicted forces (flattened)

Dataset: ethanol MD17, FCHL19 representation.
"""

import time

import numpy as np

from kernelforge import kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf_and_gradients
from kernelforge.local_kernels import (
    kernel_gaussian_full,
    kernel_gaussian_full_symm_rfp,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 500
N_TEST = 200
SIGMA = 2.0
L2 = 1e-8
ELEMENTS = [1, 6, 8]  # H, C, O


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int):
    """Return FCHL19 representations and combined labels for n_train + n_test structures."""
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
    dX = np.array(dX_list, dtype=np.float64)  # (n_total, n_atoms, rep_size, n_atoms, 3)
    Q = np.tile(z, (n_total, 1))  # (n_total, n_atoms) int32
    N = np.full(n_total, n_atoms, dtype=np.int32)  # (n_total,)
    naq = n_atoms * 3  # number of atomic coordinates = 27 for ethanol

    # Combined labels: energies concatenated with flattened forces
    y_tr = np.concatenate([E[:n_train], F[:n_train].ravel()])  # (N_train*(1+naq),)
    y_te = np.concatenate([E[n_train:], F[n_train:].ravel()])  # (N_test *(1+naq),)

    return (
        X[:n_train],
        dX[:n_train],
        X[n_train:],
        dX[n_train:],
        Q[:n_train],
        Q[n_train:],
        N[:n_train],
        N[n_train:],
        y_tr,
        y_te,
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
    print("Local KRR: energy + force training  →  predict energies + forces")
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
        y_tr,
        y_te,
        E_tr,
        E_te,
        F_tr,
        F_te,
        naq,
    ) = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    n_atoms, rep_size = X_tr.shape[1], X_tr.shape[2]
    BIG_tr = N_TRAIN * (1 + naq)
    BIG_te = N_TEST * (1 + naq)
    print(f"    n_atoms={n_atoms}  rep_size={rep_size}  naq={naq}")
    print(f"    BIG_train={BIG_tr}  BIG_test={BIG_te}")

    # ------------------------------------------------------------------
    # 2. Build training kernel  —  full combined, symmetric, RFP packed
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_rfp = kernel_gaussian_full_symm_rfp(X_tr, dX_tr, Q_tr, N_tr, SIGMA)
    print(f"\n[2] Training kernel (full, RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    K_rfp length={len(K_rfp)}  ({len(K_rfp) * 8 / 1024**2:.1f} MB)")
    assert len(K_rfp) == BIG_tr * (BIG_tr + 1) // 2

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} y_train
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    alpha = kernelmath.cho_solve_rfp(K_rfp, y_tr, l2=L2)
    del K_rfp
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error  —  derived from the normal equations (no extra allocation)
    #    We solved (K + l2*I) @ alpha = y, so K @ alpha = y - l2*alpha exactly.
    # ------------------------------------------------------------------
    y_tr_pred = y_tr - L2 * alpha  # K @ alpha = y - l2*alpha
    train_mae_E = np.mean(np.abs(y_tr_pred[:N_TRAIN] - E_tr))
    train_mae_F = np.mean(np.abs(y_tr_pred[N_TRAIN:].reshape(N_TRAIN, naq) - F_tr))
    print(
        f"\n[4] Training MAE — energy: {train_mae_E:.6f} kcal/mol"
        f"   force: {train_mae_F:.6f} kcal/(mol·Å)"
    )

    # ------------------------------------------------------------------
    # 5. Test prediction  —  asymmetric full combined kernel
    #    K_pred shape: (BIG_te, BIG_tr)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_pred = kernel_gaussian_full(  # (BIG_te, BIG_tr)
        X_te, X_tr, dX_te, dX_tr, Q_te, Q_tr, N_te, N_tr, SIGMA
    )
    y_te_pred = K_pred @ alpha  # (BIG_te,)
    del K_pred
    print(f"\n[5] Prediction kernel built in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, naq)

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
