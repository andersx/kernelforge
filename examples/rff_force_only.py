"""
RFF regression with force-only training — predict both energies and forces.

Trains on forces alone using random Fourier features (RFF).  The gradient
feature matrix G (D_rff, N_train*ncoords) plays the role of the design matrix:

  G^T @ G  @ w = G^T @ F_train_flat   (force-only normal equations)

Predictions:
  Forces  : G(x_test).T @ w   where G = rff_gradient (D_rff, N_test*ncoords)
  Energies: z(x_test) @ w     where z = rff_features (N_test, D_rff)

RFF functions used
------------------
  Training (normal eqs, RFP): rff_gradient_gramian_symm_rfp  →  (GtG_rfp, GtF)
  Predict forces             : rff_gradient                    →  G (D_rff, N_test*ncoords)
  Predict energies           : rff_features                    →  Z (N_test, D_rff)

Dataset: ethanol MD17, inverse-distance representation (M=36, ncoords=27).
"""

import time

import numpy as np

from kernelforge import invdist_repr, kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.kitchen_sinks import (
    rff_features,
    rff_gradient,
    rff_gradient_gramian_symm_rfp,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 1000
N_TEST = 200
D_RFF = 2048   # number of random Fourier features
SIGMA = 3.0    # Gaussian kernel length-scale
L2 = 1e-6      # L2 regularisation
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int):
    """Return representations, energies, and forces for n_train + n_test structures."""
    data = load_ethanol_raw_data()
    n_total = n_train + n_test

    R = data["R"][:n_total]
    E = data["E"][:n_total].ravel()           # (n_total,)
    F = data["F"][:n_total].reshape(n_total, -1)  # (n_total, ncoords=27)

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list, dtype=np.float64)   # (n_total, M=36)
    dX = np.array(dX_list, dtype=np.float64) # (n_total, M=36, ncoords=27)
    ncoords = dX.shape[2]

    return (
        X[:n_train],
        dX[:n_train],
        X[n_train:],
        dX[n_train:],
        E[:n_train],
        E[n_train:],
        F[:n_train],
        F[n_train:],
        ncoords,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("RFF: force-only training  →  predict energies + forces")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  D_rff={D_RFF}  sigma={SIGMA}  l2={L2}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    (X_tr, dX_tr, X_te, dX_te, E_tr, E_te, F_tr, F_te, ncoords) = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    rep_size = X_tr.shape[1]
    print(f"    M={rep_size}  ncoords={ncoords}")

    F_tr_flat = F_tr.ravel()  # (N_train*ncoords,) — training labels

    # ------------------------------------------------------------------
    # 2. Random Fourier feature weights
    #    W ~ N(0, 1/sigma^2)  shape (M, D_rff)
    #    b ~ Uniform(0, 2pi)  shape (D_rff,)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((rep_size, D_RFF)) / SIGMA
    b = rng.uniform(0.0, 2.0 * np.pi, D_RFF)

    # ------------------------------------------------------------------
    # 3. Build force-only normal equations  —  RFP packed
    #    GtG_rfp : 1D array, length D_rff*(D_rff+1)//2
    #    GtF     : 1D array, length D_rff
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    GtG_rfp, GtF = rff_gradient_gramian_symm_rfp(X_tr, dX_tr, W, b, F_tr_flat)
    print(f"\n[2] Normal equations (force, RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    GtG_rfp length={len(GtG_rfp)}  ({len(GtG_rfp) * 8 / 1024**2:.1f} MB)")
    assert len(GtG_rfp) == D_RFF * (D_RFF + 1) // 2

    # ------------------------------------------------------------------
    # 4. Solve  w = (G G^T + l2*I)^{-1} G F_train_flat
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    w = kernelmath.cho_solve_rfp(GtG_rfp, GtF, l2=L2)
    del GtG_rfp
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    w: shape={w.shape}  ||w||={np.linalg.norm(w):.4f}")

    # ------------------------------------------------------------------
    # 5. Training error  —  recompute G_tr^T @ w
    # ------------------------------------------------------------------
    G_tr = rff_gradient(X_tr, dX_tr, W, b)        # (D_rff, N_train*ncoords)
    F_tr_pred_flat = G_tr.T @ w                    # (N_train*ncoords,)
    train_mae = np.mean(np.abs(F_tr_pred_flat.reshape(N_TRAIN, ncoords) - F_tr))
    print(f"\n[4] Training MAE (force): {train_mae:.6f} kcal/(mol·Å)")

    # ------------------------------------------------------------------
    # 6. Test prediction — forces via G^T @ w
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    G_te = rff_gradient(X_te, dX_te, W, b)        # (D_rff, N_test*ncoords)
    F_te_pred = (G_te.T @ w).reshape(N_TEST, ncoords)
    print(f"\n[5] Force prediction in {time.perf_counter() - t0:.4f}s")

    # ------------------------------------------------------------------
    # 7. Test prediction — energies via Z @ w
    #    The same weights w can predict energies (up to a constant offset)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    Z_te = rff_features(X_te, W, b)               # (N_test, D_rff)
    E_te_pred = Z_te @ w                           # (N_test,)
    print(f"    Energy prediction in {time.perf_counter() - t0:.4f}s")

    # ------------------------------------------------------------------
    # 8. Evaluation
    # ------------------------------------------------------------------
    # Centre energies — force-only training doesn't fix the absolute offset
    E_te_pred_c = E_te_pred - E_te_pred.mean()
    E_te_c = E_te - E_te.mean()
    test_mae_E = np.mean(np.abs(E_te_pred_c - E_te_c))
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))

    print(f"\n[6] Test results")
    print(f"    Energy MAE (centred): {test_mae_E:.4f} kcal/mol")
    print(f"    Force  MAE          : {test_mae_F:.4f} kcal/(mol·Å)")
    print(f"    (Energies are predicted as integrals of the RFF force model.)")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
