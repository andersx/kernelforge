"""
Local RFF regression with combined energy + force training — predict both.

Trains simultaneously on energies and forces using elemental random Fourier
features (RFF) with FCHL19 representations.  The full elemental feature matrix
stacks energy and force rows:

  Z_full[:N,  :] = Z      (energy features, from rff_features_elemental)
  Z_full[N:,  :] = G^T    (force  features, from rff_gradient_elemental, transposed)

The combined normal equations are:
  (Z_full^T Z_full + l2*I) @ w = Z_full^T y_train
  where y_train = [E_train; F_train_flat]

  Note: Z_full^T Z_full = Z^T Z + G G^T

Predictions from the single weight vector w:
  Energies: Z(x_test) @ w         (= rff_features_elemental(X_test) @ w)
  Forces  : G(x_test).T @ w       (= rff_gradient_elemental(X_test,...).T @ w)
  Combined: Z_full(x_test) @ w    (= rff_full_elemental(X_test,...) @ w)

RFF functions used
------------------
  Training (normal eqs, RFP): rff_full_gramian_elemental_rfp  →  (ZtZ_rfp, ZtY)
  Predict E + F              : rff_full_elemental               →  Z_full (N_test*(1+naq), D_rff)

Dataset: ethanol MD17, FCHL19 representation.
"""

import time

import numpy as np

from kernelforge import kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf_and_gradients
from kernelforge.kitchen_sinks import (
    rff_full_elemental,
    rff_full_gramian_elemental_rfp,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 500
N_TEST = 200
D_RFF = 4096  # number of random Fourier features
SIGMA = 20.0  # Gaussian kernel length-scale
L2 = 1e-6  # L2 regularisation
SEED = 42
ELEMENTS = [1, 6, 8]  # H, C, O — element order determines W/b stack
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int):
    """Return FCHL19 representations and combined labels for n_train + n_test structures."""
    data = load_ethanol_raw_data()
    z = data["z"].astype(np.int32)  # (9,) atomic numbers — same for all ethanol structures
    n_atoms = len(z)
    n_total = n_train + n_test

    R = data["R"][:n_total]  # (n_total, 9, 3)
    E = data["E"][:n_total].ravel()  # (n_total,)
    F = data["F"][:n_total].reshape(n_total, -1)  # (n_total, naq=27)

    X_list, dX_list = [], []
    for r in R:
        x, dx = generate_fchl_acsf_and_gradients(r, z, elements=ELEMENTS)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list, dtype=np.float64)  # (n_total, n_atoms, rep_size)
    dX = np.array(dX_list, dtype=np.float64)  # (n_total, n_atoms, rep_size, 3*n_atoms)
    dX = dX.reshape(
        n_total, n_atoms, dX.shape[2], n_atoms, 3
    )  # → (n_total, n_atoms, rep_size, n_atoms, 3)
    naq = n_atoms * 3  # number of atomic coordinates = 27 for ethanol

    # Q for elemental RFF: list of 1D int32 arrays with 0-based element indices (no padding)
    q_mol = np.array([ELEM_TO_IDX[a] for a in z], dtype=np.int32)
    Q_rff = [q_mol] * n_total  # all ethanol structures have the same atom types

    # Combined labels: energies concatenated with flattened forces
    y_tr = np.concatenate([E[:n_train], F[:n_train].ravel()])  # (N_train*(1+naq),)
    y_te = np.concatenate([E[n_train:], F[n_train:].ravel()])  # (N_test *(1+naq),)

    return (
        X[:n_train],
        dX[:n_train],
        X[n_train:],
        dX[n_train:],
        Q_rff[:n_train],
        Q_rff[n_train:],
        y_tr,
        y_te,
        E[:n_train],
        E[n_train:],
        F[:n_train],
        F[n_train:],
        naq,
        X.shape[2],  # rep_size
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Local RFF: energy + force training  →  predict energies + forces")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  D_rff={D_RFF}  sigma={SIGMA}  l2={L2}")

    # ------------------------------------------------------------------
    # 1. Load data and generate FCHL19 representations
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    (
        X_tr,
        dX_tr,
        X_te,
        dX_te,
        Q_tr,
        Q_te,
        y_tr,
        y_te,
        E_tr,
        E_te,
        F_tr,
        F_te,
        naq,
        rep_size,
    ) = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    BIG_tr = N_TRAIN * (1 + naq)
    BIG_te = N_TEST * (1 + naq)
    print(f"    rep_size={rep_size}  naq={naq}  nelements={len(ELEMENTS)}")
    print(f"    BIG_train={BIG_tr}  BIG_test={BIG_te}")

    # ------------------------------------------------------------------
    # 2. Random Fourier feature weights (one set per element)
    #    W shape: (nelements, rep_size, D_rff)
    #    b shape: (nelements, D_rff)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((len(ELEMENTS), rep_size, D_RFF)) / SIGMA
    b = rng.uniform(0.0, 2.0 * np.pi, (len(ELEMENTS), D_RFF))

    # ------------------------------------------------------------------
    # 3. Build combined energy + force normal equations  —  RFP packed
    #    ZtZ_rfp : 1D array, length D_rff*(D_rff+1)//2
    #    ZtY     : 1D array, length D_rff
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    ZtZ_rfp, ZtY = rff_full_gramian_elemental_rfp(X_tr, dX_tr, Q_tr, W, b, E_tr, F_tr.ravel())
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
    Z_full_tr = rff_full_elemental(X_tr, dX_tr, Q_tr, W, b)  # (N_train*(1+naq), D_rff)
    y_tr_pred = Z_full_tr @ w
    train_mae_E = np.mean(np.abs(y_tr_pred[:N_TRAIN] - E_tr))
    train_mae_F = np.mean(np.abs(y_tr_pred[N_TRAIN:].reshape(N_TRAIN, naq) - F_tr))
    print(
        f"\n[4] Training MAE — energy: {train_mae_E:.6f} kcal/mol"
        f"   force: {train_mae_F:.6f} kcal/(mol·Å)"
    )

    # ------------------------------------------------------------------
    # 6. Test prediction  —  Z_full_te @ w gives [E_pred; F_pred_flat]
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    Z_full_te = rff_full_elemental(X_te, dX_te, Q_te, W, b)  # (N_test*(1+naq), D_rff)
    y_te_pred = Z_full_te @ w  # (N_test*(1+naq),)
    print(f"\n[5] Prediction in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, naq)

    test_mae_E = np.mean(np.abs(E_te_pred - E_te))
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))

    print(f"\n[6] Test results")
    print(f"    Energy MAE          : {test_mae_E:.4f} kcal/mol")
    print(f"    Force  MAE          : {test_mae_F:.4f} kcal/(mol·Å)")

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
