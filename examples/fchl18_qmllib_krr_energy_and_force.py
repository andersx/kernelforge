"""
KRR with combined energy + force training using the FCHL18 kernel via qmllib — MD17 ethanol.

Mirrors the structure of fchl18_krr_energy_and_force.py but uses only qmllib
kernel and representation functions.

The combined kernel fuses the scalar, gradient, and Hessian blocks into a
single BIG×BIG system  (BIG = N*(1+D),  D = n_atoms*3):

  K[0:N,   0:N ]  = scalar   (energy-energy)   get_local_kernels
  K[0:N,   N:  ]  = gradient (energy-force)    get_local_gradient_kernels(X, Xd)
  K[N:,    0:N ]  = gradient (force-energy)    get_local_gradient_kernels(X, Xd).T
  K[N:,    N:  ]  = Hessian  (force-force)     get_local_hessian_kernels

Kernel usage (qmllib)
---------------------
  Training kernel   : get_gaussian_process_kernels(X_tr, Xd_tr)
                      returns the full BIG×BIG block in one call
                      shape (1, N_tr*(1+D), N_tr*(1+D)); index [0]
  Predict energies  : K_EE @ alpha  with K_EE = get_local_kernels(X_te, X_tr)[0]
  Predict EF block  :                    K_EF = get_local_gradient_kernels(X_te, Xd_tr)[0]
  Predict FE block  :                    K_FE = get_local_gradient_kernels(X_tr, Xd_te)[0].T
  Predict forces    : K_FF @ alpha  with K_FF = get_local_hessian_kernels(Xd_te, Xd_tr)[0]

The four prediction blocks are assembled with np.block into shape (N_te*(1+D), N_tr*(1+D)).
y_pred = K_pred @ alpha  gives  [E_te_pred | F_te_pred_flat].

Note: qmllib's Hessian kernel is computed via double finite differences
(displaced representations), which is significantly slower than the analytic
Hessian in kernelforge.  N_TRAIN=50 is used to keep runtime reasonable.

Hyperparameters
---------------
qmllib defaults except alchemy='off':
  sigma=2.5, two_body_scaling=sqrt(8), three_body_scaling=1.6
  two_body_width=0.2, three_body_width=pi, two_body_power=4.0, three_body_power=2.0
  cut_start=1.0, cut_distance=5.0, fourier_order=1, alchemy='off'

Dataset: MD17 ethanol, 9 atoms per frame, energies + forces in kcal/mol.
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

    Both plain and displaced representations are generated with qmllib
    (one molecule at a time) and stacked:
      X   shape (N, MAX_SIZE, 5, MAX_SIZE)
      Xd  shape (N, MAX_SIZE, 2, n_atoms, MAX_SIZE, 5, MAX_SIZE)
    """
    n_total = n_train + n_test
    data = load_ethanol_raw_data()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["R"]), size=n_total, replace=False)
    idx_tr, idx_te = idx[:n_train], idx[n_train:]

    z = data["z"].astype(np.int32)  # (9,) — same for all frames

    R_tr = data["R"][idx_tr].astype(np.float64)  # (n_train, 9, 3)
    E_tr = data["E"][idx_tr].ravel().astype(np.float64)  # (n_train,)
    F_tr = data["F"][idx_tr].astype(np.float64)  # (n_train, 9, 3)

    R_te = data["R"][idx_te].astype(np.float64)  # (n_test, 9, 3)
    E_te = data["E"][idx_te].ravel().astype(np.float64)  # (n_test,)
    F_te = data["F"][idx_te].astype(np.float64)  # (n_test, 9, 3)

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
    BIG_tr = N_TRAIN * (1 + D)
    BIG_te = N_TEST * (1 + D)

    print("=" * 65)
    print("FCHL18 KRR (qmllib): energy+force training  →  predict E+F")
    print("=" * 65)
    print(f"  Dataset : MD17 ethanol  ({n_atoms} atoms, D={D})")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")
    print(f"  cut_distance={CUT_DISTANCE}  alchemy={KARGS['alchemy']}")
    print(f"  BIG_train={BIG_tr}  BIG_test={BIG_te}")

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

    # Combined training labels: energies concatenated with flattened forces
    F_tr_flat = F_tr.reshape(-1)  # (N_train*D,)
    y_tr = np.concatenate([E_tr, F_tr_flat])  # (N_train*(1+D),)

    # ------------------------------------------------------------------
    # 2. Build training kernel  —  full GP kernel (EE + EF + FE + FF blocks)
    #    get_gaussian_process_kernels(X_tr, Xd_tr) returns the full BIG×BIG
    #    block in a single call; shape (1, N_tr*(1+D), N_tr*(1+D)); index [0]
    #
    #    Block layout (rows and cols both ordered [energy | force]):
    #      K[0:N_tr,    0:N_tr   ] = EE  (scalar)
    #      K[0:N_tr,    N_tr:    ] = EF  (gradient wrt training coords)
    #      K[N_tr:,     0:N_tr   ] = FE  (gradient wrt training coords, transposed)
    #      K[N_tr:,     N_tr:    ] = FF  (Hessian)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_tr = ffk.get_gaussian_process_kernels(X_tr, Xd_tr, **KARGS)[0]
    print(f"\n[2] Training kernel (full GP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    K_train shape: {K_tr.shape}  ({K_tr.nbytes / 1024**2:.1f} MB)")

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} y_train
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_tr[np.diag_indices_from(K_tr)] += L2
    alpha = np.linalg.solve(K_tr, y_tr)
    print(f"\n[3] Linear solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error  —  from normal equations (no extra allocation)
    #    (K + l2*I) @ alpha = y  =>  K @ alpha = y - l2*alpha
    # ------------------------------------------------------------------
    y_tr_pred = y_tr - L2 * alpha
    train_mae_E = np.mean(np.abs(y_tr_pred[:N_TRAIN] - E_tr))
    train_mae_F = np.mean(np.abs(y_tr_pred[N_TRAIN:] - F_tr_flat))
    print(f"\n[4] Training MAE (regularised)")
    print(f"    Energy : {train_mae_E:.6f} kcal/mol")
    print(f"    Force  : {train_mae_F:.6f} kcal/(mol·Å)")

    # ------------------------------------------------------------------
    # 5. Build test prediction kernel  —  four blocks assembled manually
    #
    #    K_EE : get_local_kernels(X_te, X_tr)            shape (N_te, N_tr)
    #    K_EF : get_local_gradient_kernels(X_te, Xd_tr)  shape (N_te, N_tr*D)
    #           K_EF[i, j*D+d] = dK(te_i, tr_j)/dR_tr_j[d]
    #    K_FE : get_local_gradient_kernels(X_tr, Xd_te)  shape (N_tr, N_te*D) → transpose
    #           K_FE[i*D+d, j] = dK(te_i, tr_j)/dR_te_i[d]
    #    K_FF : get_local_hessian_kernels(Xd_te, Xd_tr)  shape (N_te*D, N_tr*D)
    #
    #    Full prediction kernel K_pred = [[K_EE, K_EF], [K_FE, K_FF]]
    #    shape (N_te*(1+D), N_tr*(1+D))
    #
    #    y_pred = K_pred @ alpha  =>  y_pred[:N_te] = E_pred, y_pred[N_te:] = F_pred
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_EE = fsk.get_local_kernels(X_te, X_tr, **KARGS)[0]  # (N_te, N_tr)
    K_EF = ffk.get_local_gradient_kernels(X_te, Xd_tr, **KARGS)[0]  # (N_te, N_tr*D)
    K_FE = ffk.get_local_gradient_kernels(X_tr, Xd_te, **KARGS)[0].T  # (N_te*D, N_tr)
    K_FF = ffk.get_local_hessian_kernels(Xd_te, Xd_tr, **KARGS)[0]  # (N_te*D, N_tr*D)
    K_pred = np.block([[K_EE, K_EF], [K_FE, K_FF]])  # (BIG_te, BIG_tr)
    print(f"\n[5] Prediction kernel built in {time.perf_counter() - t0:.3f}s")
    print(f"    K_pred shape: {K_pred.shape}")

    # ------------------------------------------------------------------
    # 6. Predict and evaluate
    # ------------------------------------------------------------------
    y_te_pred = K_pred @ alpha
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, n_atoms, 3)

    test_mae_E = np.mean(np.abs(E_te_pred - E_te))
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))

    print(f"\n[6] Test results")
    print(f"    Energy MAE : {test_mae_E:.4f} kcal/mol")
    print(f"    Force  MAE : {test_mae_F:.4f} kcal/(mol·Å)")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n[7] Summary")
    print(f"    Training MAE (regularised)")
    print(f"      Energy : {train_mae_E:.6f} kcal/mol")
    print(f"      Force  : {train_mae_F:.6f} kcal/(mol·Å)")
    print(f"    Test MAE")
    print(f"      Energy : {test_mae_E:.4f} kcal/mol")
    print(f"      Force  : {test_mae_F:.4f} kcal/(mol·Å)")
    print(f"\n    Note: with only {N_TRAIN} training points the test errors will be")
    print(f"    large. qmllib's FD Hessian is slow; increase N_TRAIN with care.")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
