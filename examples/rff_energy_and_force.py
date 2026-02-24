"""
RFF regression with combined energy + force training — predict both.

Trains simultaneously on energies and forces by stacking energy and force
feature rows into a single "full" design matrix Z_full (N*(1+ncoords), D_rff):

  Z_full[:N,  :] = Z      (energy features, from rff_features)
  Z_full[N:,  :] = G^T    (force  features, from rff_gradient, transposed)

The combined normal equations are:
  (Z_full^T Z_full + l2*I) @ w = Z_full^T y_train
  where y_train = [E_train; F_train_flat]

  Note: Z_full^T Z_full = Z^T Z + G G^T

Predictions from the single weight vector w:
  Energies: Z_test @ w        (= rff_features(X_test) @ w)
  Forces  : G_test^T @ w      (= rff_gradient(X_test,...).T @ w)
  Combined: Z_full_test @ w   (= rff_full(X_test,...) @ w)

RFF functions used
------------------
  Training (normal eqs, RFP): rff_full_gramian_symm_rfp  →  (ZtZ_rfp, ZtY)
  Predict E + F              : rff_full                    →  Z_full (N_test*(1+ncoords), D_rff)

Dataset: ethanol MD17, inverse-distance representation (M=36, ncoords=27).
"""

import time

import numpy as np

from kernelforge import invdist_repr, kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.kitchen_sinks import rff_full, rff_full_gramian_symm_rfp

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
    """Return representations, combined labels for n_train + n_test structures."""
    data = load_ethanol_raw_data()
    n_total = n_train + n_test

    R = data["R"][:n_total]
    E = data["E"][:n_total].ravel()               # (n_total,)
    F = data["F"][:n_total].reshape(n_total, -1)  # (n_total, ncoords=27)

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list, dtype=np.float64)   # (n_total, M=36)
    dX = np.array(dX_list, dtype=np.float64) # (n_total, M=36, ncoords=27)
    ncoords = dX.shape[2]

    # Combined labels: energies concatenated with flattened forces
    y_tr = np.concatenate([E[:n_train], F[:n_train].ravel()])   # (N_train*(1+ncoords),)
    y_te = np.concatenate([E[n_train:], F[n_train:].ravel()])   # (N_test *(1+ncoords),)

    return (
        X[:n_train],
        dX[:n_train],
        X[n_train:],
        dX[n_train:],
        y_tr,
        y_te,
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
    print("RFF: energy + force training  →  predict energies + forces")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  D_rff={D_RFF}  sigma={SIGMA}  l2={L2}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    (X_tr, dX_tr, X_te, dX_te, y_tr, y_te, E_tr, E_te, F_tr, F_te, ncoords) = load_data(
        N_TRAIN, N_TEST
    )
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    rep_size = X_tr.shape[1]
    BIG_tr = N_TRAIN * (1 + ncoords)
    BIG_te = N_TEST * (1 + ncoords)
    print(f"    M={rep_size}  ncoords={ncoords}  BIG_train={BIG_tr}  BIG_test={BIG_te}")

    # ------------------------------------------------------------------
    # 2. Random Fourier feature weights
    #    W ~ N(0, 1/sigma^2)  shape (M, D_rff)
    #    b ~ Uniform(0, 2pi)  shape (D_rff,)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((rep_size, D_RFF)) / SIGMA
    b = rng.uniform(0.0, 2.0 * np.pi, D_RFF)

    # ------------------------------------------------------------------
    # 3. Build combined energy + force normal equations  —  RFP packed
    #    ZtZ_rfp : 1D array, length D_rff*(D_rff+1)//2
    #    ZtY     : 1D array, length D_rff
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    ZtZ_rfp, ZtY = rff_full_gramian_symm_rfp(X_tr, dX_tr, W, b, E_tr, F_tr.ravel())
    print(f"\n[2] Normal equations (energy+force, RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    ZtZ_rfp length={len(ZtZ_rfp)}  ({len(ZtZ_rfp) * 8 / 1024**2:.1f} MB)")
    assert len(ZtZ_rfp) == D_RFF * (D_RFF + 1) // 2

    # ------------------------------------------------------------------
    # 4. Solve  w = (Z_full^T Z_full + l2*I)^{-1} Z_full^T y_train
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    w = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=L2)
    del ZtZ_rfp
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    w: shape={w.shape}  ||w||={np.linalg.norm(w):.4f}")

    # ------------------------------------------------------------------
    # 5. Training error  —  recompute Z_full_tr @ w
    # ------------------------------------------------------------------
    Z_full_tr = rff_full(X_tr, dX_tr, W, b)       # (N_train*(1+ncoords), D_rff)
    y_tr_pred = Z_full_tr @ w
    train_mae_E = np.mean(np.abs(y_tr_pred[:N_TRAIN] - E_tr))
    train_mae_F = np.mean(np.abs(y_tr_pred[N_TRAIN:].reshape(N_TRAIN, ncoords) - F_tr))
    print(
        f"\n[4] Training MAE — energy: {train_mae_E:.6f} kcal/mol"
        f"   force: {train_mae_F:.6f} kcal/(mol·Å)"
    )

    # ------------------------------------------------------------------
    # 6. Test prediction  —  Z_full_te @ w gives [E_pred; F_pred_flat]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    Z_full_te = rff_full(X_te, dX_te, W, b)       # (N_test*(1+ncoords), D_rff)
    y_te_pred = Z_full_te @ w                      # (N_test*(1+ncoords),)
    print(f"\n[5] Prediction in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, ncoords)

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
