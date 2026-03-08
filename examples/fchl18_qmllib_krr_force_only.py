"""
KRR with force-only training using the FCHL18 kernel via qmllib — MD17 ethanol.

Mirrors the structure of fchl18_krr_force_only.py but uses only qmllib kernel
and representation functions.

Kernel usage (qmllib)
---------------------
  Displaced reprs   : generate_fchl18_displaced — one molecule at a time
  Training kernel   : get_local_symmetric_hessian_kernels — Hessian, symmetric
                      shape (1, N_train*D, N_train*D); index [0]
  Predict forces    : get_local_hessian_kernels — Hessian, asymmetric
                      K_hess shape (1, N_test*D, N_train*D); F_pred = K_hess[0] @ alpha
  Predict energies  : get_local_gradient_kernels(X_test, Xd_train)
                      K_e shape (1, N_test, N_train*D); E_pred = K_e[0] @ alpha
                      K_e[0, i, j*D+d] = dK(test_i, train_j) / dR_train_j[d]

Energy is only determined up to an additive constant when training on forces
alone; predicted and reference energies are centred before reporting MAE.

Note: qmllib's Hessian kernel is computed via double finite differences
(displaced representations), which is significantly slower than the analytic
Hessian in kernelforge.  N_TRAIN=50 is used to keep runtime reasonable.

Hyperparameters
---------------
qmllib defaults except alchemy='off':
  sigma=2.5, two_body_scaling=sqrt(8), three_body_scaling=1.6
  two_body_width=0.2, three_body_width=pi, two_body_power=4.0, three_body_power=2.0
  cut_start=1.0, cut_distance=5.0, fourier_order=1, alchemy='off'

Dataset: MD17 ethanol, 9 atoms per frame, forces in kcal/(mol*Angstrom).
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
N_TRAIN = 50
N_TEST = 50
MAX_SIZE = 9  # ethanol has 9 atoms
L2 = 1e-4

# qmllib defaults except alchemy='off', passed explicitly to every kernel call
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
    alchemy="off",
    kernel_args={"sigma": [SIGMA]},
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int, seed: int = 42):
    """Return coordinates, charges, representations, energies and forces.

    Structures are drawn from a random permutation of the dataset so that
    training and test sets are not consecutive MD frames.

    Both plain and displaced representations are generated with qmllib
    (one molecule at a time) and stacked:
      X   shape (N, MAX_SIZE, 5, MAX_SIZE)           — for energy prediction
      Xd  shape (N, MAX_SIZE, 2, n_atoms, MAX_SIZE, 5, MAX_SIZE) — for Hessian/gradient
    """
    n_total = n_train + n_test
    data = load_ethanol_raw_data()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["R"]), size=n_total, replace=False)
    idx_tr, idx_te = idx[:n_train], idx[n_train:]

    z = data["z"].astype(np.int32)  # (9,) — same for all frames

    R_tr = data["R"][idx_tr].astype(np.float64)  # (n_train, 9, 3)
    E_tr = data["E"][idx_tr].ravel().astype(np.float64)  # (n_train,)
    F_tr = -data["F"][idx_tr].astype(np.float64)  # (n_train, 9, 3)

    R_te = data["R"][idx_te].astype(np.float64)  # (n_test, 9, 3)
    E_te = data["E"][idx_te].ravel().astype(np.float64)  # (n_test,)
    F_te = -data["F"][idx_te].astype(np.float64)  # (n_test, 9, 3)

    X_tr = np.array(
        [generate_fchl18(z, R, max_size=MAX_SIZE, cut_distance=CUT_DISTANCE) for R in R_tr]
    )
    X_te = np.array(
        [generate_fchl18(z, R, max_size=MAX_SIZE, cut_distance=CUT_DISTANCE) for R in R_te]
    )
    Xd_tr = np.array(
        [
            generate_fchl18_displaced(z, R, max_size=MAX_SIZE, cut_distance=CUT_DISTANCE)
            for R in R_tr
        ]
    )
    Xd_te = np.array(
        [
            generate_fchl18_displaced(z, R, max_size=MAX_SIZE, cut_distance=CUT_DISTANCE)
            for R in R_te
        ]
    )

    return z, R_tr, X_tr, Xd_tr, E_tr, F_tr, R_te, X_te, Xd_te, E_te, F_te


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n_atoms = MAX_SIZE  # ethanol: C2H5OH, 9 atoms
    D = n_atoms * 3  # 27 degrees of freedom per molecule

    print("=" * 65)
    print("FCHL18 KRR (qmllib): force-only training  →  forces + energies")
    print("=" * 65)
    print(f"  Dataset : MD17 ethanol  ({n_atoms} atoms, D={D})")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")
    print(f"  cut_distance={CUT_DISTANCE}  alchemy={KARGS['alchemy']}")

    # ------------------------------------------------------------------
    # 1. Load data and build representations
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    z, R_tr, X_tr, Xd_tr, E_tr, F_tr, R_te, X_te, Xd_te, E_te, F_te = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded + representations built in {time.perf_counter() - t0:.2f}s")
    print(f"    X_train shape : {X_tr.shape}  (n_mols, max_size, 5, max_size)")
    print(
        f"    Xd_train shape: {Xd_tr.shape}  (n_mols, max_size, 2, n_atoms, max_size, 5, max_size)"
    )

    # Flatten training forces: shape (N_train * D,)
    F_tr_flat = F_tr.reshape(-1)  # (N_train * D,)

    # ------------------------------------------------------------------
    # 2. Build training kernel  —  symmetric Hessian
    #    get_local_symmetric_hessian_kernels returns (1, N*D, N*D); take [0]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_tr = ffk.get_local_symmetric_hessian_kernels(Xd_tr, **KARGS)[0]
    print(f"\n[2] Training Hessian kernel built in {time.perf_counter() - t0:.3f}s")
    print(f"    K_train shape: {K_tr.shape}  ({K_tr.nbytes / 1024**2:.1f} MB)")

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} F_train_flat
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_tr[np.diag_indices_from(K_tr)] += L2
    alpha = np.linalg.solve(K_tr, F_tr_flat)
    print(f"\n[3] Linear solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error — from normal equations
    #    (K + l2*I) @ alpha = F_tr_flat  =>  K @ alpha = F_tr_flat - l2*alpha
    # ------------------------------------------------------------------
    F_tr_pred_flat = F_tr_flat - L2 * alpha
    train_mae_F = np.mean(np.abs(F_tr_pred_flat - F_tr_flat))
    print(f"\n[4] Training MAE (force, regularised): {train_mae_F:.6f} kcal/(mol·Å)")

    # ------------------------------------------------------------------
    # 5. Test prediction — forces via asymmetric Hessian kernel
    #    get_local_hessian_kernels returns (1, N_te*D, N_tr*D); take [0]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_hess = ffk.get_local_hessian_kernels(Xd_te, Xd_tr, **KARGS)[0]
    F_te_pred = (K_te_hess @ alpha).reshape(N_TEST, n_atoms, 3)
    print(f"\n[5] Force prediction kernel built in {time.perf_counter() - t0:.3f}s")
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))
    print(f"    Force MAE: {test_mae_F:.4f} kcal/(mol·Å)")

    # ------------------------------------------------------------------
    # 6. Test prediction — energies via gradient kernel wrt training coords
    #
    #    get_local_gradient_kernels(X_test, Xd_train) returns
    #      K_e shape (1, N_test, N_train*D)
    #    where K_e[0, i, j*D+d] = dK(test_i, train_j) / dR_train_j[d]
    #
    #    E_pred[i] = sum_{j,d} K_e[0, i, j*D+d] * alpha[j*D+d]
    #              = K_e[0] @ alpha
    #
    #    Energy is only determined up to an additive constant; centre
    #    both predicted and reference energies before reporting MAE.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_e = ffk.get_local_gradient_kernels(X_te, Xd_tr, **KARGS)[0]
    E_te_pred = K_te_e @ alpha
    print(f"\n[6] Energy prediction kernel built in {time.perf_counter() - t0:.3f}s")

    E_te_pred_c = E_te_pred - E_te_pred.mean()
    E_te_c = E_te - E_te.mean()
    test_mae_E = np.mean(np.abs(E_te_pred_c - E_te_c))
    print(f"    Energy MAE (centred): {test_mae_E:.4f} kcal/mol")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n[7] Summary")
    print(f"    Training MAE (force, regularised): {train_mae_F:.6f} kcal/(mol·Å)")
    print(f"    Test force MAE                   : {test_mae_F:.4f} kcal/(mol·Å)")
    print(f"    Test energy MAE (centred)         : {test_mae_E:.4f} kcal/mol")
    print(f"\n    Note: with only {N_TRAIN} training points the test errors will be")
    print(f"    large. qmllib's FD Hessian is slow; increase N_TRAIN with care.")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
