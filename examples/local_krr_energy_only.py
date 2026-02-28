"""
Local KRR with energy-only training — predict both energies and forces.

Trains on energies alone using the local (atom-indexed) Gaussian kernel with
FCHL19 representations, then predicts:
  - Energies via the scalar local kernel         (energy-energy)
  - Forces  via the local Jacobian kernel        (analytical derivative of
    the energy prediction w.r.t. test-set atomic coordinates)

The "local" kernel sums atom-pair contributions across the molecule, naturally
handling molecules with the same composition.

Kernel usage
------------
  Training kernel :  kernel_gaussian_symm_rfp   — scalar, symmetric, RFP packed
  Training error  :  kernel_gaussian_symm        — scalar, symmetric, full matrix
  Predict energies:  kernel_gaussian              — scalar, asymmetric
  Predict forces  :  kernel_gaussian_jacobian     — Jacobian, asymmetric
                     shape (N_test*naq, N_train)  where naq = n_atoms*3

The Jacobian kernel K_jac[i*naq+d, j] = dK(x_test_i, x_train_j)/d(coord_d_i),
so K_jac @ alpha gives the gradient of the KRR energy prediction w.r.t. the
test-set atomic coordinates (= predicted forces up to a sign convention).

Dataset: ethanol MD17, FCHL19 representation.
"""

import time

import numpy as np

from kernelforge import kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf_and_gradients
from kernelforge.local_kernels import (
    kernel_gaussian,
    kernel_gaussian_jacobian,
    kernel_gaussian_symm,
    kernel_gaussian_symm_rfp,
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
    dX = np.array(dX_list, dtype=np.float64)  # (n_total, n_atoms, rep_size, n_atoms, 3)
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
    print("Local KRR: energy-only training  →  predict energies + forces")
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

    # ------------------------------------------------------------------
    # 2. Build training kernel  —  scalar, symmetric, RFP packed
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_rfp = kernel_gaussian_symm_rfp(X_tr, Q_tr, N_tr, SIGMA)
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
    K_tr_full = kernel_gaussian_symm(X_tr, Q_tr, N_tr, SIGMA)  # (N_train, N_train)
    E_tr_pred = K_tr_full @ alpha
    train_mae = np.mean(np.abs(E_tr_pred - E_tr))
    print(f"\n[4] Training MAE (energy): {train_mae:.6f} kcal/mol")

    # ------------------------------------------------------------------
    # 5. Test prediction — energies
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_scalar = kernel_gaussian(X_te, X_tr, Q_te, Q_tr, N_te, N_tr, SIGMA)  # (N_test, N_train)
    E_te_pred = K_te_scalar @ alpha  # (N_test,)
    print(f"\n[5] Energy prediction kernel built in {time.perf_counter() - t0:.4f}s")

    # ------------------------------------------------------------------
    # 6. Test prediction — forces via Jacobian kernel
    #    K_jac shape: (N_test*naq, N_train)
    #    F_pred[i, d] = sum_j K_jac[i*naq+d, j] * alpha[j]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_jac = kernel_gaussian_jacobian(  # (N_test*naq, N_train)
        X_te, X_tr, dX_te, Q_te, Q_tr, N_te, N_tr, SIGMA
    )
    F_te_pred = (K_te_jac @ alpha).reshape(N_TEST, naq)
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
