"""
KRR with energy-only training using the FCHL18 kernel via qmllib — MD17 ethanol.

Trains a kernel ridge regression model on molecular energies using the FCHL18
representation and Gaussian kernel from qmllib, then predicts both energies and
forces on a held-out test set.  Forces are obtained as minus the gradient of the
KRR energy prediction with respect to atomic coordinates, using qmllib's
get_local_gradient_kernels (finite-difference Jacobian).

This example is structurally identical to fchl18_krr_energy_only.py but uses
only qmllib kernel and representation functions.

Kernel usage (qmllib)
---------------------
  Representations : generate_fchl18              — one molecule at a time
  Displaced reprs : generate_fchl18_displaced    — for force prediction
  Training kernel : get_local_symmetric_kernels  — scalar, symmetric
  Predict energies: get_local_kernels            — scalar, asymmetric
  Predict forces  : get_local_gradient_kernels   — finite-difference Jacobian
                    K_grad shape (1, N_train, n_atoms*3)
                    F_pred = -(K_grad[0].T @ alpha).reshape(n_atoms, 3)

Hyperparameters
---------------
All qmllib defaults:
  sigma=2.5, two_body_scaling=sqrt(8), three_body_scaling=1.6
  two_body_width=0.2, three_body_width=pi, two_body_power=4.0, three_body_power=2.0
  cut_start=1.0, cut_distance=5.0, fourier_order=1, alchemy='periodic-table'

Dataset: MD17 ethanol (~555k structures, 9 atoms, energies + forces in kcal/mol).
"""

import time

import numpy as np
from qmllib.representations.fchl import fchl_force_kernels as ffk
from qmllib.representations.fchl import fchl_scalar_kernels as fsk
from qmllib.representations.fchl import generate_fchl18
from qmllib.representations.fchl.fchl_representations import generate_fchl18_displaced

from kernelforge.cli import load_ethanol_raw_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 400
N_TEST = 200
MAX_SIZE = 9  # ethanol has 9 atoms
L2 = 1e-8

# qmllib defaults — passed explicitly to every kernel/repr call
SIGMA = 2.5
CUT_DISTANCE = 5.0
KARGS = dict(
    two_body_scaling=np.sqrt(8),
    three_body_scaling=1.6,
    two_body_width=0.2,
    three_body_width=np.pi,
    two_body_power=4.0,
    three_body_power=2.0,
    cut_start=1.0,
    cut_distance=CUT_DISTANCE,
    fourier_order=1,
    alchemy="periodic-table",
    kernel_args={"sigma": [SIGMA]},
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int, seed: int = 42):
    """Return coordinates, charges, representations, energies and forces.

    Structures are drawn from a random permutation of the dataset so that
    training and test sets are not consecutive MD frames.

    Representations are generated with qmllib's generate_fchl18 (one molecule
    at a time) and stacked into arrays of shape (N, MAX_SIZE, 5, MAX_SIZE).
    """
    n_total = n_train + n_test
    data = load_ethanol_raw_data()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["R"]), size=n_total, replace=False)
    idx_tr, idx_te = idx[:n_train], idx[n_train:]

    z = data["z"].astype(np.int32)  # (9,) — same for all frames

    R_tr = data["R"][idx_tr]  # (n_train, 9, 3)
    E_tr = data["E"][idx_tr].ravel().astype(np.float64)  # (n_train,)
    F_tr = data["F"][idx_tr].astype(np.float64)  # (n_train, 9, 3)

    R_te = data["R"][idx_te]  # (n_test, 9, 3)
    E_te = data["E"][idx_te].ravel().astype(np.float64)  # (n_test,)
    F_te = data["F"][idx_te].astype(np.float64)  # (n_test, 9, 3)

    # generate_fchl18 takes one molecule at a time; stack into (N, MAX_SIZE, 5, MAX_SIZE)
    X_tr = np.array(
        [generate_fchl18(z, R, max_size=MAX_SIZE, cut_distance=CUT_DISTANCE) for R in R_tr]
    )
    X_te = np.array(
        [generate_fchl18(z, R, max_size=MAX_SIZE, cut_distance=CUT_DISTANCE) for R in R_te]
    )

    return z, R_tr, X_tr, E_tr, F_tr, R_te, X_te, E_te, F_te


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("FCHL18 KRR (qmllib): energy-only training  →  energies + forces")
    print("=" * 65)
    print(f"  Dataset : MD17 ethanol")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")
    print(f"  cut_distance={CUT_DISTANCE}  fourier_order={KARGS['fourier_order']}")
    print(f"  alchemy={KARGS['alchemy']}")

    # ------------------------------------------------------------------
    # 1. Load data and build representations
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    z, R_tr, X_tr, E_tr, F_tr, R_te, X_te, E_te, F_te = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded + representations built in {time.perf_counter() - t0:.2f}s")
    print(f"    X_train shape: {X_tr.shape}  (n_mols, max_size, 5, max_size)")

    # ------------------------------------------------------------------
    # 2. Build training kernel  —  scalar, symmetric
    #    get_local_symmetric_kernels returns shape (n_sigmas, N, N); take [0]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_tr = fsk.get_local_symmetric_kernels(X_tr, **KARGS)[0]
    print(f"\n[2] Training kernel built in {time.perf_counter() - t0:.3f}s")
    print(f"    K_train shape: {K_tr.shape}")

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} E_train
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_tr[np.diag_indices_from(K_tr)] += L2
    alpha = np.linalg.solve(K_tr, E_tr)
    print(f"\n[3] Linear solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error
    # ------------------------------------------------------------------
    K_tr[np.diag_indices_from(K_tr)] -= L2  # restore diagonal
    E_tr_pred = K_tr @ alpha
    train_mae_E = np.mean(np.abs(E_tr_pred - E_tr))
    print(f"\n[4] Training MAE (energy): {train_mae_E:.4f} kcal/mol")

    # ------------------------------------------------------------------
    # 5. Test prediction — energies
    #    get_local_kernels returns shape (n_sigmas, N_te, N_tr); take [0]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te = fsk.get_local_kernels(X_te, X_tr, **KARGS)[0]
    E_te_pred = K_te @ alpha
    print(f"\n[5] Test energies predicted in {time.perf_counter() - t0:.4f}s")
    test_mae_E = np.mean(np.abs(E_te_pred - E_te))
    print(f"    Energy MAE: {test_mae_E:.4f} kcal/mol")

    # ------------------------------------------------------------------
    # 6. Test prediction — forces via gradient kernel
    #
    #    get_local_gradient_kernels(X_train, Xd_test) returns
    #      K_grad shape (1, N_train, n_atoms*3)
    #    where K_grad[0, b, alpha*3+mu] = dK[test, b] / dR_test[alpha, mu]
    #
    #    KRR energy: E_pred = sum_b K(test, b) * alpha[b]
    #    Forces:     F_pred = -dE_pred/dR = -(K_grad[0].T @ alpha)
    #                         shape (n_atoms*3,) -> reshape to (n_atoms, 3)
    #
    #    Displaced representations are generated per test molecule since
    #    generate_fchl18_displaced takes one molecule at a time.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    F_te_pred = np.zeros_like(F_te)  # (N_test, 9, 3)
    n_atoms = z.shape[0]

    # Pre-build kargs without kernel_args (ffk uses different kwarg name)
    grad_kargs = {k: v for k, v in KARGS.items() if k != "kernel_args"}

    for i, R in enumerate(R_te):
        # generate_fchl18_displaced → (max_size, 2, n_atoms, max_size, 5, max_size)
        Xd = generate_fchl18_displaced(z, R, max_size=MAX_SIZE, cut_distance=CUT_DISTANCE)
        Xd_batch = Xd[np.newaxis]  # (1, max_size, 2, n_atoms, max_size, 5, max_size)
        K_grad = ffk.get_local_gradient_kernels(
            X_tr, Xd_batch, kernel_args={"sigma": [SIGMA]}, **grad_kargs
        )  # (1, N_train, n_atoms*3)
        F_te_pred[i] = -(K_grad[0].T @ alpha).reshape(n_atoms, 3)

    print(f"\n[6] Test forces predicted in {time.perf_counter() - t0:.4f}s")
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))
    print(f"    Force  MAE: {test_mae_F:.4f} kcal/(mol·Å)")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n[7] Test results")
    print(f"    Energy MAE : {test_mae_E:.4f} kcal/mol")
    print(f"    Force  MAE : {test_mae_F:.4f} kcal/(mol·Å)")
    print(f"    (Forces predicted as -grad of the energy KRR model.)")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
