"""
RFF energy regression via direct design-matrix solve (DGELSD).

Instead of forming the D_rff×D_rff normal equations
  (Z^T Z + l2*I) @ w = Z^T E_train
this script solves the N_train×D_rff least-squares system
  min||Z_train @ w - E_train||_2
directly with DGELSD (divide-and-conquer SVD via kernelmath.solve_svd).

For N_train < D_rff (underdetermined), DGELSD returns the minimum-norm solution,
which implicitly regularises without an explicit l2 term.  The condition number
of Z^T Z is printed for comparison with the normal-equations path.

Dataset: ethanol MD17, inverse-distance representation (M=36, ncoords=27).
"""

import time

import numpy as np

from kernelforge import invdist_repr, kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.kitchen_sinks import rff_features, rff_gramian_symm_rfp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 1000
N_TEST = 200
D_RFF = 2048
SIGMA = 3.0
L2 = 1e-6       # used only for the normal-equations reference solve
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int):
    data = load_ethanol_raw_data()
    n_total = n_train + n_test

    R = data["R"][:n_total]
    E = data["E"][:n_total].ravel()

    X_list = []
    for r in R:
        x, _ = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)

    X = np.array(X_list, dtype=np.float64)

    return X[:n_train], X[n_train:], E[:n_train], E[n_train:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("RFF: direct design-matrix solve via DGELSD (solve_svd)")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  D_rff={D_RFF}  sigma={SIGMA}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    X_tr, X_te, E_tr, E_te = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 2. Random Fourier feature weights
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    rep_size = X_tr.shape[1]
    W = rng.standard_normal((rep_size, D_RFF)) / SIGMA
    b = rng.uniform(0.0, 2.0 * np.pi, D_RFF)

    # ------------------------------------------------------------------
    # 3. Build Z_train  (N_train × D_rff)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    Z_tr = rff_features(X_tr, W, b)   # (N_train, D_rff)
    print(f"\n[2] Z_train built in {time.perf_counter() - t0:.3f}s  shape={Z_tr.shape}")

    # ------------------------------------------------------------------
    # 4. Condition number of Z^T Z (for reference)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    ZtZ = Z_tr.T @ Z_tr
    cond = kernelmath.condition_number_ge(ZtZ)
    del ZtZ
    print(f"\n[3] cond(Z^T Z) = {cond:.3e}  (computed in {time.perf_counter() - t0:.3f}s)")

    # ------------------------------------------------------------------
    # 5. Direct solve: min||Z_tr @ w - E_tr||_2  via DGELSD
    #    For N_train < D_rff this is underdetermined → minimum-norm solution
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    w_svd = kernelmath.solve_svd(Z_tr, E_tr, rcond=0.0)
    print(f"\n[4] solve_svd in {time.perf_counter() - t0:.3f}s  ||w||={np.linalg.norm(w_svd):.4f}")

    # ------------------------------------------------------------------
    # 6. Reference solve via normal equations + Cholesky (RFP)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    ZtZ_rfp, ZtY = rff_gramian_symm_rfp(X_tr, W, b, E_tr)
    w_cho = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=L2)
    del ZtZ_rfp
    print(f"[5] cho_solve_rfp in {time.perf_counter() - t0:.3f}s  ||w||={np.linalg.norm(w_cho):.4f}")

    # ------------------------------------------------------------------
    # 7. Test predictions
    # ------------------------------------------------------------------
    Z_te = rff_features(X_te, W, b)

    E_te_svd = Z_te @ w_svd
    E_te_cho = Z_te @ w_cho

    # Centre — RFF doesn't learn the absolute energy offset
    def centred_mae(pred, ref):
        return float(np.mean(np.abs((pred - pred.mean()) - (ref - ref.mean()))))

    mae_svd = centred_mae(E_te_svd, E_te)
    mae_cho = centred_mae(E_te_cho, E_te)

    print(f"\n[6] Test MAE (centred energies)")
    print(f"    solve_svd  (direct):         {mae_svd:.4f} kcal/mol")
    print(f"    cho_solve_rfp (normal eqs):  {mae_cho:.4f} kcal/mol")
    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
