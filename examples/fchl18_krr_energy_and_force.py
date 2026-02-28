"""
KRR with combined energy + force training using the FCHL18 kernel — MD17 ethanol.

Trains simultaneously on energies and forces using the full combined FCHL18
kernel, which fuses the scalar, Jacobian, and Hessian blocks into a single
BIG×BIG system (BIG = N + D, D = N_train * n_atoms * 3):

  K_full[0:N, 0:N]   = scalar   (energy-energy)
  K_full[0:N, N: ]   = jac_t    (energy-force)
  K_full[N:,  0:N]   = jacobian (force-energy)
  K_full[N:,  N: ]   = hessian  (force-force)

Kernel usage
------------
  Training kernel :  fchl18_kernel.kernel_gaussian_full_symm_rfp  — full, symmetric, RFP
  Predict E + F   :  fchl18_kernel.kernel_gaussian_full            — full, asymmetric
                     shape (N_test*(1+D_test), N_train*(1+D_train))

The prediction kernel applied to alpha gives:
  y_pred[:N_test]    = predicted energies
  y_pred[N_test:]    = predicted forces (flattened)

Restrictions on the hessian / full kernel
------------------------------------------
  - use_atm must be False  (ATM Hessian not yet implemented)
  - cut_start must be >= 1.0  (cutoff Hessian not yet implemented)

Hyperparameters
---------------
Tuned via grid search (see FCHL18_TUNING.md):
  sigma=2.5, fourier_order=1, use_atm=False
  two_body_power=4.5, two_body_scaling=2.5, three_body_scaling=1.5

Dataset: MD17 ethanol (~555k structures, 9 atoms, energies + forces in kcal/mol).
"""

import time

import numpy as np

import kernelforge.fchl18_kernel as fchl18_kernel
from kernelforge import kernelmath
from kernelforge.cli import load_ethanol_raw_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 200
N_TEST = 100
SIGMA = 1.25
L2 = 1e-4  # FCHL18 full kernel needs modest regularisation for Cholesky stability
MAX_SIZE = 9  # ethanol has 9 atoms

KERNEL_ARGS: dict = dict(
    two_body_width=0.1,
    two_body_scaling=2.5,
    two_body_power=4.5,
    three_body_width=3.0,
    three_body_scaling=1.5,
    three_body_power=3.0,
    cut_start=1.0,
    cut_distance=1e6,
    fourier_order=2,
    use_atm=False,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int, seed: int = 42):
    """Return coordinate lists, charges, energies, and forces."""
    data = load_ethanol_raw_data()
    n_total = n_train + n_test

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["R"]), size=n_total, replace=False)

    z = data["z"].astype(np.int32)  # (9,) — same for all ethanol frames

    idx_tr, idx_te = idx[:n_train], idx[n_train:]

    R_tr = [data["R"][i].astype(np.float64) for i in idx_tr]  # list of (9,3)
    E_tr = data["E"][idx_tr].ravel().astype(np.float64)  # (n_train,)
    F_tr = [data["F"][i].astype(np.float64) for i in idx_tr]  # list of (9,3)

    R_te = [data["R"][i].astype(np.float64) for i in idx_te]  # list of (9,3)
    E_te = data["E"][idx_te].ravel().astype(np.float64)  # (n_test,)
    F_te = [data["F"][i].astype(np.float64) for i in idx_te]  # list of (9,3)

    z_list = [z] * n_total

    return (
        R_tr,
        z_list[:n_train],
        E_tr,
        F_tr,
        R_te,
        z_list[n_train:],
        E_te,
        F_te,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n_atoms = 9  # ethanol: C2H5OH
    D = n_atoms * 3  # 27 degrees of freedom per molecule

    print("=" * 65)
    print("FCHL18 KRR: energy + force training  ->  predict E + F")
    print("=" * 65)
    print(f"  Dataset : MD17 ethanol  ({n_atoms} atoms, D={D})")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")
    print(f"  cut_start={KERNEL_ARGS['cut_start']}  use_atm={KERNEL_ARGS['use_atm']}")

    BIG_tr = N_TRAIN * (1 + D)
    BIG_te = N_TEST * (1 + D)
    print(f"  BIG_train={BIG_tr}  BIG_test={BIG_te}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    R_tr, Z_tr, E_tr, F_tr, R_te, Z_te, E_te, F_te = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.3f}s")

    # Combined training labels: energies concatenated with flattened forces
    F_tr_flat = np.concatenate([f.ravel() for f in F_tr])  # (N_train*D,)
    y_tr = np.concatenate([E_tr, F_tr_flat])  # (N_train*(1+D),)

    # ------------------------------------------------------------------
    # 2. Build training kernel  — full combined, symmetric, RFP packed
    #    Length = BIG_tr * (BIG_tr + 1) / 2
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_rfp = fchl18_kernel.kernel_gaussian_full_symm(R_tr, Z_tr, sigma=SIGMA, **KERNEL_ARGS)
    print(f"\n[2] Training kernel (full, RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    rfp length={len(K_rfp)}  ({len(K_rfp) * 8 / 1024**2:.1f} MB)")
    # assert len(K_rfp) == BIG_tr * (BIG_tr + 1) // 2

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} y_train  via Cholesky/RFP
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    # alpha = kernelmath.cho_solve(K_rfp, y_tr, l2=L2)
    alpha = kernelmath.solve_cholesky(K_rfp, y_tr, regularize=L2)  # same result, no extra allocation
    del K_rfp  # free RFP buffer
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error  —  from normal equations (no extra allocation)
    #    We solved (K + l2*I) @ alpha = y, so K @ alpha = y - l2*alpha.
    # ------------------------------------------------------------------
    y_tr_pred = y_tr - L2 * alpha
    train_mae_E = np.mean(np.abs(y_tr_pred[:N_TRAIN] - E_tr))
    train_mae_F = np.mean(np.abs(y_tr_pred[N_TRAIN:] - F_tr_flat))
    print(f"\n[4] Training MAE (regularised)")
    print(f"    Energy : {train_mae_E:.6f} kcal/mol")
    print(f"    Force  : {train_mae_F:.6f} kcal/(mol*A)")

    # ------------------------------------------------------------------
    # 5. Test prediction  —  asymmetric full kernel
    #    K_pred shape: (BIG_te, BIG_tr)
    #    y_pred[:N_test]   = energies
    #    y_pred[N_test:]   = forces (flattened)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_pred = fchl18_kernel.kernel_gaussian_full(
        R_te, Z_te, R_tr, Z_tr, sigma=SIGMA, **KERNEL_ARGS
    )  # (BIG_te, BIG_tr)
    y_te_pred = K_pred @ alpha  # (BIG_te,)
    del K_pred
    print(f"\n[5] Prediction kernel built in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, D)
    F_te_true = np.stack([f.ravel() for f in F_te])

    # Energy is determined up to a constant when the training data has
    # an arbitrary energy zero; centre both arrays before reporting MAE.
    # E_te_pred_c = E_te_pred - E_te_pred.mean()
    # E_te_c = E_te - E_te.mean()
    test_mae_E = np.mean(np.abs(E_te_pred - E_te))
    test_mae_F = np.mean(np.abs(F_te_pred - F_te_true))

    print(f"\n[6] Test results")
    print(f"    Energy MAE (centred): {test_mae_E:.4f} kcal/mol")
    print(f"    Force  MAE          : {test_mae_F:.4f} kcal/(mol*A)")
    print(f"\n    Note: with only {N_TRAIN} training points errors will be large.")
    print(f"    Increase N_TRAIN to improve accuracy (slow: O(N^2 * D^2)).")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
