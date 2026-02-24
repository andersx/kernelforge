"""
RFF regression with energy-only training — predict both energies and forces.

Trains on energies alone using random Fourier features (RFF), which approximate
the Gaussian kernel via the Bochner / Rahimi-Recht mapping:

  z(x) = sqrt(2/D) * cos(W^T x + b),   W ~ N(0, 1/sigma^2),  b ~ Uniform(0, 2pi)

The normal equations for energy training are:
  (Z^T Z + l2*I) @ w = Z^T E_train
where Z (N_train, D_rff) is the RFF feature matrix.

Predictions:
  Energies: z(x_test) @ w
  Forces  : -dz/dx_test^T @ w  =  G(x_test).T @ w
             where G = rff_gradient (D_rff, N_test*ncoords).

RFF functions used
------------------
  Training (normal eqs, RFP): rff_gramian_symm_rfp  →  (ZtZ_rfp, ZtY)
  Predict energies           : rff_features          →  Z (N_test, D_rff)
  Predict forces             : rff_gradient          →  G (D_rff, N_test*ncoords)

Dataset: ethanol MD17, inverse-distance representation (M=36, ncoords=27).
"""

import time

import numpy as np

from kernelforge import invdist_repr, kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.kitchen_sinks import rff_features, rff_gradient, rff_gramian_symm_rfp

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
    print("RFF: energy-only training  →  predict energies + forces")
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

    # ------------------------------------------------------------------
    # 2. Random Fourier feature weights
    #    W ~ N(0, 1/sigma^2)  shape (M, D_rff)
    #    b ~ Uniform(0, 2pi)  shape (D_rff,)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((rep_size, D_RFF)) / SIGMA
    b = rng.uniform(0.0, 2.0 * np.pi, D_RFF)

    # ------------------------------------------------------------------
    # 3. Build training normal equations  —  RFP packed
    #    ZtZ_rfp  : 1D array, length D_rff*(D_rff+1)//2
    #    ZtY      : 1D array, length D_rff
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    ZtZ_rfp, ZtY = rff_gramian_symm_rfp(X_tr, W, b, E_tr)
    print(f"\n[2] Normal equations (energy, RFP) built in {time.perf_counter() - t0:.3f}s")
    print(f"    ZtZ_rfp length={len(ZtZ_rfp)}  ({len(ZtZ_rfp) * 8 / 1024**2:.1f} MB)")
    assert len(ZtZ_rfp) == D_RFF * (D_RFF + 1) // 2

    # ------------------------------------------------------------------
    # 4. Solve  w = (Z^T Z + l2*I)^{-1} Z^T E_train
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    w = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=L2)
    del ZtZ_rfp
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    w: shape={w.shape}  ||w||={np.linalg.norm(w):.4f}")

    # ------------------------------------------------------------------
    # 5. Training error  —  recompute Z_tr @ w
    # ------------------------------------------------------------------
    Z_tr = rff_features(X_tr, W, b)               # (N_train, D_rff)
    E_tr_pred = Z_tr @ w
    train_mae = np.mean(np.abs(E_tr_pred - E_tr))
    print(f"\n[4] Training MAE (energy): {train_mae:.6f} kcal/mol")

    # ------------------------------------------------------------------
    # 6. Test prediction — energies
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    Z_te = rff_features(X_te, W, b)               # (N_test, D_rff)
    E_te_pred = Z_te @ w                           # (N_test,)
    print(f"\n[5] Energy prediction in {time.perf_counter() - t0:.4f}s")

    # ------------------------------------------------------------------
    # 7. Test prediction — forces via G^T @ w
    #    G = rff_gradient  shape (D_rff, N_test*ncoords)
    #    F_pred = G^T @ w  shape (N_test*ncoords,)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    G_te = rff_gradient(X_te, dX_te, W, b)        # (D_rff, N_test*ncoords)
    F_te_pred = (G_te.T @ w).reshape(N_TEST, ncoords)
    print(f"    Force  prediction  in {time.perf_counter() - t0:.4f}s")

    # ------------------------------------------------------------------
    # 8. Evaluation
    # ------------------------------------------------------------------
    # Centre energies — RFF doesn't learn the absolute energy offset
    E_te_pred_c = E_te_pred - E_te_pred.mean()
    E_te_c = E_te - E_te.mean()
    test_mae_E = np.mean(np.abs(E_te_pred_c - E_te_c))
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))

    print(f"\n[6] Test results")
    print(f"    Energy MAE (centred): {test_mae_E:.4f} kcal/mol")
    print(f"    Force  MAE          : {test_mae_F:.4f} kcal/(mol·Å)")
    print(f"    (Forces are predicted as gradients of the RFF energy model.)")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
