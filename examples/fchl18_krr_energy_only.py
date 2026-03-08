"""
KRR with energy-only training using the FCHL18 kernel — MD17 ethanol.

Trains a kernel ridge regression model on molecular energies using the FCHL18
representation and Gaussian kernel, then predicts both energies and forces on
a held-out test set.  Forces are obtained analytically as minus the gradient
of the KRR energy prediction with respect to atomic coordinates, using
kernel_gaussian_gradient.

Kernel usage
------------
  Training kernel :  fchl18_kernel.kernel_gaussian_symm    — scalar, symmetric
  Predict energies:  fchl18_kernel.kernel_gaussian          — scalar, asymmetric
  Predict forces  :  fchl18_kernel.kernel_gaussian_gradient — coordinate Jacobian
                     G[alpha, mu, b] = dK(A,b)/dR_A[alpha,mu]
                     F_pred = -(G @ alpha)  shape (n_atoms, 3)

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
import kernelforge.fchl18_repr as fchl18_repr
from kernelforge.cli import load_ethanol_raw_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 400
N_TEST = 200
SIGMA = 22.5
L2 = 1e-8
MAX_SIZE = 9  # ethanol has 9 atoms

KERNEL_ARGS: dict = dict(
    two_body_width=0.1,
    two_body_scaling=2.5,
    two_body_power=4.5,
    three_body_width=3.0,
    three_body_scaling=1.5,
    three_body_power=3.0,
    cut_start=0.5,
    cut_distance=1e6,
    fourier_order=1,
    use_atm=False,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int, seed: int = 42):
    """Return coordinates, charges, representations, energies and forces.

    Structures are drawn from a random permutation of the dataset so that
    training and test sets are not consecutive MD frames.
    """
    n_total = n_train + n_test
    data = load_ethanol_raw_data()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data["R"]), size=n_total, replace=False)
    idx_tr, idx_te = idx[:n_train], idx[n_train:]

    z = data["z"].astype(np.int32)  # (9,)  — same for all frames

    R_tr = data["R"][idx_tr]  # (n_train, 9, 3)
    E_tr = data["E"][idx_tr].ravel().astype(np.float64)  # (n_train,)
    F_tr = data["F"][idx_tr].astype(np.float64)  # (n_train, 9, 3)

    R_te = data["R"][idx_te]  # (n_test, 9, 3)
    E_te = data["E"][idx_te].ravel().astype(np.float64)  # (n_test,)
    F_te = data["F"][idx_te].astype(np.float64)  # (n_test, 9, 3)

    x_tr, n_atoms_tr, nn_tr = fchl18_repr.generate(
        list(R_tr),
        [z] * n_train,
        max_size=MAX_SIZE,
        cut_distance=KERNEL_ARGS["cut_distance"],
    )
    x_te, n_atoms_te, nn_te = fchl18_repr.generate(
        list(R_te),
        [z] * n_test,
        max_size=MAX_SIZE,
        cut_distance=KERNEL_ARGS["cut_distance"],
    )

    return (
        R_tr,
        z,
        x_tr,
        n_atoms_tr,
        nn_tr,
        E_tr,
        F_tr,
        R_te,
        x_te,
        n_atoms_te,
        nn_te,
        E_te,
        F_te,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("FCHL18 KRR: energy-only training  →  predict energies + forces")
    print("=" * 65)
    print(f"  Dataset : MD17 ethanol")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")
    print(f"  fourier_order={KERNEL_ARGS['fourier_order']}  use_atm={KERNEL_ARGS['use_atm']}")

    # ------------------------------------------------------------------
    # 1. Load data and build representations
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    (R_tr, z, x_tr, n_tr, nn_tr, E_tr, F_tr, R_te, x_te, n_te, nn_te, E_te, F_te) = load_data(
        N_TRAIN, N_TEST
    )
    print(f"\n[1] Data loaded + representations built in {time.perf_counter() - t0:.2f}s")
    print(f"    x_train shape: {x_tr.shape}  (n_mols, max_size, 5, max_size)")

    # ------------------------------------------------------------------
    # 2. Build training kernel  —  scalar, symmetric
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_tr = fchl18_kernel.kernel_gaussian_symm(x_tr, n_tr, nn_tr, sigma=SIGMA, **KERNEL_ARGS)
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
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te = fchl18_kernel.kernel_gaussian(
        x_te, x_tr, n_te, n_tr, nn_te, nn_tr, sigma=SIGMA, **KERNEL_ARGS
    )
    E_te_pred = K_te @ alpha
    print(f"\n[5] Test energies predicted in {time.perf_counter() - t0:.4f}s")
    test_mae_E = np.mean(np.abs(E_te_pred - E_te))
    print(f"    Energy MAE: {test_mae_E:.4f} kcal/mol")

    # ------------------------------------------------------------------
    # 6. Test prediction — forces via gradient kernel
    #
    #    kernel_gaussian_gradient returns G[alpha, mu, b] = dK(A,b)/dR_A[alpha,mu]
    #    The KRR energy prediction is  E_pred(A) = sum_b K(A,b) * alpha[b]
    #    so predicted forces are       F_pred = -dE_pred/dR = -(G @ alpha)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    F_te_pred = np.zeros_like(F_te)  # (N_test, 9, 3)
    for i, coords_A in enumerate(R_te):
        G = fchl18_kernel.kernel_gaussian_gradient(
            coords_A, z, x_tr, n_tr, nn_tr, sigma=SIGMA, **KERNEL_ARGS
        )  # shape (n_atoms, 3, N_train)
        F_te_pred[i] = -(G @ alpha)  # (n_atoms, 3)
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
