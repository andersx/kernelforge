"""
Local RFF regression with energy-only training — predict both energies and forces.

Trains on energies alone using elemental random Fourier features (RFF) with FCHL19
representations.  The elemental variant maintains a separate set of random weights
per element type, naturally encoding the local chemical environment.

The Bochner / Rahimi-Recht feature mapping per atom of element e:
  z_e(x_atom) = sqrt(2/D) * cos(W_e^T x_atom + b_e)

The molecule-level feature is the sum over atoms:
  Z(molecule) = sum_{atoms} z_{element(atom)}(x_atom)

The normal equations for energy training are:
  (Z^T Z + l2*I) @ w = Z^T E_train

Predictions:
  Energies: Z(x_test) @ w
  Forces  : -dZ/dx_test^T @ w  =  G(x_test).T @ w
             where G = rff_gradient_elemental (D_rff, N_test*naq).

RFF functions used
------------------
  Training (normal eqs, RFP): rff_gramian_elemental_rfp  →  (ZtZ_rfp, ZtY)
  Predict energies           : rff_features_elemental     →  Z (N_test, D_rff)
  Predict forces             : rff_gradient_elemental     →  G (D_rff, N_test*naq)

Dataset: ethanol MD17, FCHL19 representation.
"""

import time

import numpy as np

from kernelforge import kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf_and_gradients
from kernelforge.kitchen_sinks import (
    rff_features_elemental,
    rff_gradient_elemental,
    rff_gramian_elemental_rfp,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 500
N_TEST = 200
D_RFF = 2048  # number of random Fourier features
SIGMA = 2.0  # Gaussian kernel length-scale
L2 = 1e-6  # L2 regularisation
SEED = 42
ELEMENTS = [1, 6, 8]  # H, C, O — element order determines W/b stack
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int):
    """Return FCHL19 representations, energies, and forces for n_train + n_test structures."""
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

    return (
        X[:n_train],
        dX[:n_train],
        X[n_train:],
        dX[n_train:],
        Q_rff[:n_train],
        Q_rff[n_train:],
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
    print("Local RFF: energy-only training  →  predict energies + forces")
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
        E_tr,
        E_te,
        F_tr,
        F_te,
        naq,
        rep_size,
    ) = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    print(f"    rep_size={rep_size}  naq={naq}  nelements={len(ELEMENTS)}")

    # ------------------------------------------------------------------
    # 2. Random Fourier feature weights (one set per element)
    #    W shape: (nelements, rep_size, D_rff)
    #    b shape: (nelements, D_rff)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((len(ELEMENTS), rep_size, D_RFF)) / SIGMA
    b = rng.uniform(0.0, 2.0 * np.pi, (len(ELEMENTS), D_RFF))

    # ------------------------------------------------------------------
    # 3. Build training normal equations  —  RFP packed
    #    ZtZ_rfp  : 1D array, length D_rff*(D_rff+1)//2
    #    ZtY      : 1D array, length D_rff
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    ZtZ_rfp, ZtY = rff_gramian_elemental_rfp(X_tr, Q_tr, W, b, E_tr)
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
    Z_tr = rff_features_elemental(X_tr, Q_tr, W, b)  # (N_train, D_rff)
    E_tr_pred = Z_tr @ w
    train_mae = np.mean(np.abs(E_tr_pred - E_tr))
    print(f"\n[4] Training MAE (energy): {train_mae:.6f} kcal/mol")

    # ------------------------------------------------------------------
    # 6. Test prediction — energies
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    Z_te = rff_features_elemental(X_te, Q_te, W, b)  # (N_test, D_rff)
    E_te_pred = Z_te @ w  # (N_test,)
    print(f"\n[5] Energy prediction in {time.perf_counter() - t0:.4f}s")

    # ------------------------------------------------------------------
    # 7. Test prediction — forces via G^T @ w
    #    G = rff_gradient_elemental  shape (D_rff, N_test*naq)
    #    F_pred = G^T @ w  shape (N_test*naq,)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    G_te = rff_gradient_elemental(X_te, dX_te, Q_te, W, b)  # (D_rff, N_test*naq)
    F_te_pred = (G_te.T @ w).reshape(N_TEST, naq)
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
