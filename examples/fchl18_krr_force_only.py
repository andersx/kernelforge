"""
KRR with force-only training using the FCHL18 Hessian kernel — MD17 ethanol.

Mirrors the structure of examples/krr_force_only.py but uses the FCHL18
kernel (raw Cartesian coordinates + nuclear charges) instead of the
global invdist kernel.

Kernel usage
------------
  Training kernel :  kernel_gaussian_hessian_symm_rfp — Hessian, symmetric, RFP
  Training error  :  derived from normal equations (no extra allocation)
  Predict forces  :  kernel_gaussian_hessian           — Hessian, asymmetric
                     shape (N_test*D, N_train*D)
  Predict energies:  kernel_gaussian_jacobian_t         — Jacobian-transpose
                     shape (N_test, N_train*D)

The jacobian_t kernel K_jt[i, j*D+d] = dK(test_i, train_j)/dR_{train_j}[d],
so K_jt @ alpha (with alpha the force coefficients) gives energy predictions.
Energy is only determined up to an additive constant when training on forces alone;
both predicted and reference energies are centred before reporting MAE.

Restrictions on kernel_gaussian_hessian / hessian_symm_rfp
-----------------------------------------------------------
  - use_atm must be False  (ATM Hessian not yet implemented)
  - cut_start must be >= 1.0  (cutoff Hessian not yet implemented)

Dataset: MD17 ethanol, 9 atoms per frame, forces in kcal/(mol*Angstrom).
"""

import time

import numpy as np

import kernelforge.fchl18_kernel as fchl18_kernel
import kernelforge.fchl18_repr as fchl18_repr
from kernelforge import kernelmath
from kernelforge.cli import load_ethanol_raw_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 50
N_TEST = 50
SIGMA = 2.5
L2 = 1e-4  # FCHL18 Hessian kernel is only numerically PSD; ~1e-4 is needed for Cholesky at N=200
MAX_SIZE = 9  # ethanol has 9 atoms

# Hessian kernel hyperparameters.
KERNEL_ARGS: dict = dict(
    two_body_scaling=np.sqrt(8),
    three_body_scaling=1.6,
    two_body_width=0.2,
    three_body_width=np.pi,
    two_body_power=4.0,
    three_body_power=2.0,
    cut_start=1.0,
    cut_distance=5.0,
    fourier_order=1,
    # two_body_width=0.1,
    # two_body_scaling=2.5,
    # two_body_power=4.5,
    # three_body_width=3.0,
    # three_body_scaling=1.5,
    # three_body_power=3.0,
    # cut_start=1.0,
    # cut_distance=1e6,
    # fourier_order=1,
    use_atm=True,
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
        z,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n_atoms = 9  # ethanol: C2H5OH
    D = n_atoms * 3  # 27 degrees of freedom per molecule

    print("=" * 65)
    print("FCHL18 KRR: force-only training  ->  predict forces + energies")
    print("=" * 65)
    print(f"  Dataset : MD17 ethanol  ({n_atoms} atoms, D={D})")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")
    print(f"  cut_start={KERNEL_ARGS['cut_start']}  use_atm={KERNEL_ARGS['use_atm']}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    R_tr, Z_tr, E_tr, F_tr, R_te, Z_te, E_te, F_te, z = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.3f}s")

    # Flatten training forces: shape (N_train * D,)
    F_tr_flat = np.concatenate([f.ravel() for f in F_tr])

    # ------------------------------------------------------------------
    # 2. Build training kernel  — symmetric Hessian, RFP packed
    #    Length = N_train*D * (N_train*D + 1) / 2
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_rfp = fchl18_kernel.kernel_gaussian_hessian_symm_rfp(R_tr, Z_tr, sigma=SIGMA, **KERNEL_ARGS)
    ND = N_TRAIN * D
    print(f"\n[2] Training Hessian kernel (RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    rfp length={len(K_rfp)}  ({len(K_rfp) * 8 / 1024**2:.1f} MB)")
    assert len(K_rfp) == ND * (ND + 1) // 2

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} F_train_flat  via Cholesky/RFP
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    alpha = kernelmath.cho_solve_rfp(K_rfp, F_tr_flat, l2=L2)
    del K_rfp  # free RFP buffer
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error — from normal equations (no extra allocation)
    #    (K + l2*I) @ alpha = F_tr_flat  =>  K @ alpha = F_tr_flat - l2*alpha
    # ------------------------------------------------------------------
    F_tr_pred_flat = F_tr_flat - L2 * alpha
    train_mae_F = np.mean(np.abs(F_tr_pred_flat - F_tr_flat))
    print(f"\n[4] Training MAE (force, regularised): {train_mae_F:.6f} kcal/(mol*A)")

    # ------------------------------------------------------------------
    # 5. Test prediction — forces via Hessian kernel
    #    K_te shape: (N_test*D, N_train*D)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_te_hess = fchl18_kernel.kernel_gaussian_hessian(
        R_te, Z_te, R_tr, Z_tr, sigma=SIGMA, **KERNEL_ARGS
    )
    F_te_pred = (K_te_hess @ alpha).reshape(N_TEST, D)
    del K_te_hess
    print(f"\n[5] Force prediction kernel built in {time.perf_counter() - t0:.3f}s")

    F_te_true = np.stack([f.ravel() for f in F_te])
    test_mae_F = np.mean(np.abs(F_te_pred - F_te_true))
    print(f"    Force MAE: {test_mae_F:.4f} kcal/(mol*A)")

    # ------------------------------------------------------------------
    # 6. Test prediction — energies via Jacobian-transpose kernel
    #    Build test representations, then compute K_jt @ alpha.
    #
    #    K_jt shape: (N_test, N_train*D)
    #    K_jt[i, j*D+d] = dK(test_i, train_j) / dR_{train_j}[d]
    #    E_pred[i] = sum_{j,d} K_jt[i, j*D+d] * alpha[j*D+d]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    # Build test representations (needed as training set for gradient calls)
    x_te, n_te, nn_te = fchl18_repr.generate(
        R_te, Z_te, max_size=MAX_SIZE, cut_distance=KERNEL_ARGS["cut_distance"]
    )

    K_te_jt = fchl18_kernel.kernel_gaussian_jacobian_t(
        R_tr,
        Z_tr,  # training: raw coords (gradient query)
        x_te,
        n_te,
        nn_te,  # test: pre-computed repr (gradient reference set)
        sigma=SIGMA,
        **KERNEL_ARGS,
    )  # shape (N_test, N_train*D)
    E_te_pred = K_te_jt @ alpha  # (N_test,)
    del K_te_jt
    print(f"\n[6] Energy prediction kernel built in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 7. Evaluation
    #    Energy is only determined up to a constant; centre both before MAE.
    # ------------------------------------------------------------------
    E_te_pred_c = E_te_pred - E_te_pred.mean()
    E_te_c = E_te - E_te.mean()
    test_mae_E = np.mean(np.abs(E_te_pred_c - E_te_c))

    print(f"\n[7] Summary")
    print(f"    Training MAE (force, regularised): {train_mae_F:.6f} kcal/(mol*A)")
    print(f"    Test force MAE                   : {test_mae_F:.4f} kcal/(mol*A)")
    print(f"    Test energy MAE (centred)         : {test_mae_E:.4f} kcal/mol")
    print(f"\n    Note: with only {N_TRAIN} training points the test errors will be")
    print(f"    large. Increase N_TRAIN to improve accuracy (slow: O(N^2 * D^2)).")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
