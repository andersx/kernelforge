"""
Local KRR with combined energy + force training -- predict both energies and forces.

Trains simultaneously on energies and forces using the full combined local kernel,
which fuses the scalar, Jacobian, and Hessian blocks into a single BIG x BIG system
(BIG = N*(1 + naq), where naq = n_atoms*3):

  K_full[0:N,   0:N  ] = scalar   (energy-energy)
  K_full[N:,    0:N  ] = jacobian (force-energy)
  K_full[0:N,   N:   ] = jac_t    (energy-force)
  K_full[N:,    N:   ] = hessian  (force-force)

Kernel usage
------------
  Training kernel :  kernel_gaussian_full  -- full combined, dense square matrix
                     (symmetric: K_tr[i,j] == K_tr[j,i])
  Training error  :  derived from normal equations  (no extra allocation)
  Predict E + F   :  kernel_gaussian_full  -- full combined, asymmetric
                     shape (N_test*(1+naq), N_train*(1+naq))

The prediction kernel applied to alpha gives:
  y_pred[:N_test]    = predicted energies
  y_pred[N_test:]    = predicted forces (flattened)

Dataset: ethanol MD17, FCHL19 representation.
"""

import time

import numpy as np

from kernelforge import kernelmath
from kernelforge.cli import load_ethanol_raw_data
from kernelforge.fchl19_repr import generate_fchl_acsf_and_gradients
from kernelforge.local_kernels import kernel_gaussian_full, kernel_gaussian_full_symm_rfp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 1000
N_TEST = 200
SIGMA = 2.0
L2 = 1e-9
ELEMENTS = [1, 6, 8]  # H, C, O


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(n_train: int, n_test: int):
    """Return FCHL19 representations and combined labels for n_train + n_test structures."""
    data = load_ethanol_raw_data()
    z = data["z"].astype(np.int32)  # (9,) atomic numbers -- same for all ethanol structures
    n_atoms = len(z)
    n_total = n_train + n_test

    R = data["R"][:n_total]  # (n_total, 9, 3)
    E = data["E"][:n_total].ravel()  # (n_total,)
    E = E - np.mean(E)  # zero-center energies for better numerical stability
    F = -data["F"][:n_total].reshape(n_total, -1)  # (n_total, naq=27)

    X_list, dX_list = [], []
    for r in R:
      # x, dx = generate_fchl_acsf_and_gradients(r, z, elements=ELEMENTS)
        x, dx = generate_fchl_acsf_and_gradients(                  
            r, z,                                                  
            elements=ELEMENTS,                                     
            nRs2=13,                                               
            nRs3=16,                                               
            nFourier=2,                                            
            eta2=2.664757627376702,                                
            eta3=2.6588480261685987,                               
            rcut=3.6433741741109893,                               
            acut=8.942590456104261,                                
            two_body_decay=1.0307535502842426,                     
            three_body_decay=1.929616342566224,                    
            three_body_weight=41.815151029954535,                  
        )                                                          
        
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list, dtype=np.float64)  # (n_total, n_atoms, rep_size)
    dX = np.array(dX_list, dtype=np.float64)  # (n_total, n_atoms, rep_size, 3*n_atoms)
    Q = np.tile(z, (n_total, 1))  # (n_total, n_atoms) int32
    N = np.full(n_total, n_atoms, dtype=np.int32)  # (n_total,)
    naq = n_atoms * 3  # number of atomic coordinates = 27 for ethanol

    # Combined labels: energies concatenated with flattened forces
    y_tr = np.concatenate([E[:n_train], F[:n_train].ravel()])  # (N_train*(1+naq),)
    y_te = np.concatenate([E[n_train:], F[n_train:].ravel()])  # (N_test *(1+naq),)

    return (
        X[:n_train],
        dX[:n_train],
        X[n_train:],
        dX[n_train:],
        Q[:n_train],
        Q[n_train:],
        N[:n_train],
        N[n_train:],
        y_tr,
        y_te,
        E[:n_train],
        E[n_train:],
        F[:n_train],
        F[n_train:],
        naq,
    )


def linfit(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (slope, intercept) from least-squares fit of y_pred vs y_true."""
    x = y_true.ravel()
    y = y_pred.ravel()
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Local KRR: energy + force training  ->  predict energies + forces")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  sigma={SIGMA}  l2={L2}")

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
        N_tr,
        N_te,
        y_tr,
        y_te,
        E_tr,
        E_te,
        F_tr,
        F_te,
        naq,
    ) = load_data(N_TRAIN, N_TEST)
    print(f"\n[1] Data loaded in {time.perf_counter() - t0:.2f}s")
    n_atoms, rep_size = X_tr.shape[1], X_tr.shape[2]
    BIG_tr = N_TRAIN * (1 + naq)
    BIG_te = N_TEST * (1 + naq)
    print(f"    n_atoms={n_atoms}  rep_size={rep_size}  naq={naq}")
    print(f"    BIG_train={BIG_tr}  BIG_test={BIG_te}")

    # ------------------------------------------------------------------
    # 2. Build training kernel  --  full combined, dense square matrix
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    # K_tr = kernel_gaussian_full(X_tr, X_tr, dX_tr, dX_tr, Q_tr, Q_tr, N_tr, N_tr, SIGMA)
    K_tr = kernel_gaussian_full_symm_rfp(X_tr, dX_tr, Q_tr, N_tr, SIGMA)
    print(f"\n[2] Training kernel (full dense) built in {time.perf_counter() - t0:.3f}s")
    print(f"    K_tr shape={K_tr.shape}  ({K_tr.nbytes / 1024**2:.1f} MB)")

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} y_train
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    # alpha = kernelmath.solve_cholesky(K_tr, y_tr, regularize=L2)
    alpha = kernelmath.cho_solve_rfp(K_tr, y_tr, l2=L2)
    del K_tr
    print(f"\n[3] Cholesky solve in {time.perf_counter() - t0:.3f}s")
    print(f"    alpha: shape={alpha.shape}  ||alpha||={np.linalg.norm(alpha):.4f}")

    # ------------------------------------------------------------------
    # 4. Training error  --  derived from the normal equations
    #    We solved (K + l2*I) @ alpha = y, so K @ alpha = y - l2*alpha exactly.
    # ------------------------------------------------------------------
    y_tr_pred = y_tr - L2 * alpha
    E_tr_pred = y_tr_pred[:N_TRAIN]
    F_tr_pred = y_tr_pred[N_TRAIN:].reshape(N_TRAIN, naq)

    train_mae_E = np.mean(np.abs(E_tr_pred - E_tr))
    train_mae_F = np.mean(np.abs(F_tr_pred - F_tr))
    slope_E_tr, intercept_E_tr = linfit(E_tr, E_tr_pred)
    slope_F_tr, intercept_F_tr = linfit(F_tr, F_tr_pred)
    print(f"\n[4] Training MAE")
    print(
        f"    Energy : {train_mae_E:.6f} kcal/mol   slope={slope_E_tr:.4f}  intercept={intercept_E_tr:.4f}"
    )
    print(
        f"    Force  : {train_mae_F:.6f} kcal/(mol*A)  slope={slope_F_tr:.4f}  intercept={intercept_F_tr:.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Test prediction  --  asymmetric full combined kernel
    #    K_pred shape: (BIG_te, BIG_tr)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    K_pred = kernel_gaussian_full(  # (BIG_te, BIG_tr)
        X_te, X_tr, dX_te, dX_tr, Q_te, Q_tr, N_te, N_tr, SIGMA
    )
    y_te_pred = K_pred @ alpha  # (BIG_te,)
    del K_pred
    print(f"\n[5] Prediction kernel built in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, naq)

    test_mae_E = np.mean(np.abs(E_te_pred - E_te))
    test_mae_F = np.mean(np.abs(F_te_pred - F_te))
    slope_E_te, intercept_E_te = linfit(E_te, E_te_pred)
    slope_F_te, intercept_F_te = linfit(F_te, F_te_pred)

    print(f"\n[6] Test results")
    print(
        f"    Energy MAE : {test_mae_E:.4f} kcal/mol   slope={slope_E_te:.4f}  intercept={intercept_E_te:.4f}"
    )
    print(
        f"    Force  MAE : {test_mae_F:.4f} kcal/(mol*A)  slope={slope_F_te:.4f}  intercept={intercept_F_te:.4f}"
    )

    print("\n" + "=" * 65 + "\nDone.\n" + "=" * 65)


if __name__ == "__main__":
    main()
