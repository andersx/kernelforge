"""
Optuna hyperparameter optimization for local KRR with FCHL19 representation.

Uses the rMD17 ethanol split 01 dataset (1000 train + 1000 test frames,
PBE/def2-SVP energies and forces) to optimize all FCHL19 representation
hyperparameters as well as the KRR model hyperparameters (sigma, l2).

Optimizes all of the following simultaneously:
  FCHL19 representation:
    eta2, eta3            — two- and three-body Gaussian widths
    rcut, acut            — two- and three-body cutoff radii (Å)
    two_body_decay        — decay exponent for two-body term
    three_body_decay      — decay exponent for three-body term
    three_body_weight     — weight for three-body contribution
    nRs2, nRs3           — number of radial basis functions
    nFourier              — number of angular Fourier terms

  KRR model:
    sigma                 — Gaussian kernel length-scale
    l2                    — L2 regularisation strength

Objective:
  Minimise combined test MAE = energy_MAE (kcal/mol) + force_MAE (kcal/mol/Å)

Dataset:
  rMD17 ethanol split 01 from https://github.com/andersx/rmd17-npz
  Forces are in the convention F = -dE/dR (already negated), so they are used
  directly as training labels without sign flipping.

Usage:
  uv run python examples/optimize_krr_fchl19.py
  uv run python examples/optimize_krr_fchl19.py --trials 200
  uv run python examples/optimize_krr_fchl19.py --trials 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

from kernelforge import kernelmath
from kernelforge.fchl19_repr import generate_fchl_acsf_and_gradients
from kernelforge.local_kernels import kernel_gaussian_full, kernel_gaussian_full_symm_rfp
from kernelforge.rmd17_data import load_rmd17_ethanol_split01

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRIALS = 100
ELEMENTS = [1, 6, 8]  # H, C, O — element order for the local kernel
SEED = 42

# ---------------------------------------------------------------------------
# Load rMD17 data once at import time — shared across all Optuna trials
# ---------------------------------------------------------------------------
print("Loading rMD17 ethanol split 01...")
_TRAIN_DATA, _TEST_DATA = load_rmd17_ethanol_split01()

N_TRAIN = len(_TRAIN_DATA["E"])  # 1000
N_TEST = len(_TEST_DATA["E"])  # 1000

# Atomic numbers are identical across all frames (ethanol, 9 atoms)
_Z = _TRAIN_DATA["z"]  # (9,) int32

# Pre-extract coordinate / label arrays to avoid dict lookups in the hot path
_R_TRAIN = _TRAIN_DATA["R"]  # (1000, 9, 3)
_E_TRAIN = _TRAIN_DATA["E"]  # (1000,)
_F_TRAIN = _TRAIN_DATA["F"]  # (1000, 9, 3)  — forces, F = -dE/dR

_R_TEST = _TEST_DATA["R"]  # (1000, 9, 3)
_E_TEST = _TEST_DATA["E"]  # (1000,)
_F_TEST = _TEST_DATA["F"]  # (1000, 9, 3)

_N_ATOMS = len(_Z)  # 9
_NAQ = _N_ATOMS * 3  # 27


# ---------------------------------------------------------------------------
# Representation builder (called once per trial)
# ---------------------------------------------------------------------------


def _build_representations(
    repr_kwargs: dict[str, Any],
) -> tuple[
    np.ndarray,  # X_tr  (N_TRAIN, n_atoms, rep_size)
    np.ndarray,  # dX_tr (N_TRAIN, n_atoms, rep_size, n_atoms*3)
    np.ndarray,  # X_te  (N_TEST,  n_atoms, rep_size)
    np.ndarray,  # dX_te (N_TEST,  n_atoms, rep_size, n_atoms*3)
    np.ndarray,  # Q_tr  (N_TRAIN, n_atoms) int32
    np.ndarray,  # Q_te  (N_TEST,  n_atoms) int32
    np.ndarray,  # N_tr  (N_TRAIN,) int32
    np.ndarray,  # N_te  (N_TEST,)  int32
]:
    """Generate FCHL19 representations and gradients for all train + test frames.

    Parameters
    ----------
    repr_kwargs:
        Keyword arguments forwarded verbatim to ``generate_fchl_acsf_and_gradients``.

    Returns
    -------
    X_tr, dX_tr, X_te, dX_te, Q_tr, Q_te, N_tr, N_te
        Representation arrays ready for the local kernel functions.
    """
    z = _Z
    n_atoms = _N_ATOMS

    X_list_tr, dX_list_tr = [], []
    for r in _R_TRAIN:
        x, dx = generate_fchl_acsf_and_gradients(r, z, elements=ELEMENTS, **repr_kwargs)
        X_list_tr.append(x)
        dX_list_tr.append(dx)

    X_list_te, dX_list_te = [], []
    for r in _R_TEST:
        x, dx = generate_fchl_acsf_and_gradients(r, z, elements=ELEMENTS, **repr_kwargs)
        X_list_te.append(x)
        dX_list_te.append(dx)

    X_tr = np.array(X_list_tr, dtype=np.float64)  # (N_TRAIN, n_atoms, rep_size)
    dX_tr = np.array(dX_list_tr, dtype=np.float64)  # (N_TRAIN, n_atoms, rep_size, n_atoms*3)
    X_te = np.array(X_list_te, dtype=np.float64)
    dX_te = np.array(dX_list_te, dtype=np.float64)

    Q_tr = np.tile(z, (N_TRAIN, 1))  # (N_TRAIN, n_atoms) int32
    Q_te = np.tile(z, (N_TEST, 1))  # (N_TEST,  n_atoms) int32
    N_tr = np.full(N_TRAIN, n_atoms, dtype=np.int32)
    N_te = np.full(N_TEST, n_atoms, dtype=np.int32)

    return X_tr, dX_tr, X_te, dX_te, Q_tr, Q_te, N_tr, N_te


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: combined energy + force test MAE.

    Samples hyperparameters, regenerates FCHL19 representations, trains a
    local KRR model on the full rMD17 ethanol split 01 training set, and
    evaluates on the test set.

    Returns
    -------
    float
        Combined MAE = energy_MAE (kcal/mol) + force_MAE (kcal/mol/Å).
        Optuna minimises this value.
    """
    # ------------------------------------------------------------------
    # 1. Sample hyperparameters
    # ------------------------------------------------------------------
    # FCHL19 representation hyperparameters
    nRs2 = trial.suggest_int("nRs2", 8, 48)
    nRs3 = trial.suggest_int("nRs3", 4, 40)
    nFourier = trial.suggest_int("nFourier", 1, 3)
    eta2 = trial.suggest_float("eta2", 0.01, 5.0, log=True)
    eta3 = trial.suggest_float("eta3", 0.1, 20.0, log=True)
    rcut = trial.suggest_float("rcut", 3.0, 12.0)
    acut = trial.suggest_float("acut", 3.0, 12.0)
    two_body_decay = trial.suggest_float("two_body_decay", 0.5, 5.0)
    three_body_decay = trial.suggest_float("three_body_decay", 0.1, 3.0)
    three_body_weight = trial.suggest_float("three_body_weight", 1.0, 50.0)

    # KRR model hyperparameters
    sigma = trial.suggest_float("sigma", 1.0, 200.0, log=True)
    l2 = trial.suggest_float("l2", 1e-9, 1e-7, log=True)

    repr_kwargs: dict[str, Any] = {
        "nRs2": nRs2,
        "nRs3": nRs3,
        "nFourier": nFourier,
        "eta2": eta2,
        "eta3": eta3,
        "rcut": rcut,
        "acut": acut,
        "two_body_decay": two_body_decay,
        "three_body_decay": three_body_decay,
        "three_body_weight": three_body_weight,
    }

    # ------------------------------------------------------------------
    # 2. Build FCHL19 representations for this trial's hyperparameters
    # ------------------------------------------------------------------
    t_repr = time.perf_counter()
    try:
        X_tr, dX_tr, X_te, dX_te, Q_tr, Q_te, N_tr, N_te = _build_representations(repr_kwargs)
    except Exception as exc:
        # Occasionally extreme hyperparams can cause numerical issues
        trial.set_user_attr("error", str(exc))
        return float("inf")
    repr_time = time.perf_counter() - t_repr

    # ------------------------------------------------------------------
    # 3. Combined label vectors: [energies; forces_flat]
    # ------------------------------------------------------------------
    F_tr_flat = _F_TRAIN.reshape(N_TRAIN, _NAQ)  # (N_TRAIN, 27)
    y_tr = np.concatenate([_E_TRAIN, F_tr_flat.ravel()])

    # ------------------------------------------------------------------
    # 4. Build training kernel and solve
    # ------------------------------------------------------------------
    t_kern = time.perf_counter()
    try:
        K_tr = kernel_gaussian_full_symm_rfp(X_tr, dX_tr, Q_tr, N_tr, sigma)
        alpha = kernelmath.cho_solve_rfp(K_tr, y_tr, l2=l2)
        del K_tr
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        return float("inf")
    kern_time = time.perf_counter() - t_kern

    # ------------------------------------------------------------------
    # 5. Predict on test set
    # ------------------------------------------------------------------
    t_pred = time.perf_counter()
    K_pred = kernel_gaussian_full(X_te, X_tr, dX_te, dX_tr, Q_te, Q_tr, N_te, N_tr, sigma)
    y_te_pred = K_pred @ alpha
    del K_pred
    pred_time = time.perf_counter() - t_pred

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, _NAQ)

    F_te_flat = _F_TEST.reshape(N_TEST, _NAQ)

    mae_E = float(np.mean(np.abs(E_te_pred - _E_TEST)))
    mae_F = float(np.mean(np.abs(F_te_pred - F_te_flat)))
    combined_mae = mae_E + mae_F

    # Store diagnostics as user attributes for inspection
    trial.set_user_attr("mae_E", mae_E)
    trial.set_user_attr("mae_F", mae_F)
    trial.set_user_attr("repr_time_s", round(repr_time, 2))
    trial.set_user_attr("kern_time_s", round(kern_time, 2))
    trial.set_user_attr("pred_time_s", round(pred_time, 2))

    return combined_mae


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_best(study: optuna.Study, top_k: int = 5) -> None:
    """Print the top-k trials by objective value."""
    trials = [t for t in study.trials if t.value is not None and t.value != float("inf")]
    trials.sort(key=lambda t: t.value)  # type: ignore[arg-type]

    print("\n" + "=" * 75)
    print(f"  Top-{min(top_k, len(trials))} trials  (objective = energy_MAE + force_MAE)")
    print("=" * 75)

    for rank, t in enumerate(trials[:top_k], start=1):
        t_mae_e = t.user_attrs.get("mae_E", "?")
        t_mae_f = t.user_attrs.get("mae_F", "?")
        t_mae_e_str = f"{t_mae_e:.4f}" if isinstance(t_mae_e, float) else str(t_mae_e)
        t_mae_f_str = f"{t_mae_f:.4f}" if isinstance(t_mae_f, float) else str(t_mae_f)
        print(
            f"\n  Rank {rank}  trial={t.number:4d}"
            f"  objective={t.value:.6f} kcal/mol(+/A)"  # type: ignore[str-format]
            f"  E_MAE={t_mae_e_str}"
            f"  F_MAE={t_mae_f_str}"
        )

    best = study.best_trial
    print("\n" + "=" * 75)
    print("  Best trial hyperparameters")
    print("=" * 75)
    print(f"  trial number : {best.number}")
    mae_e = best.user_attrs.get("mae_E", "?")
    mae_f = best.user_attrs.get("mae_F", "?")
    obj_str = f"{best.value:.6f}" if best.value != float("inf") else "inf"
    mae_e_str = f"{mae_e:.4f}" if isinstance(mae_e, float) else str(mae_e)
    mae_f_str = f"{mae_f:.4f}" if isinstance(mae_f, float) else str(mae_f)
    print(f"  objective    : {obj_str} kcal/mol(+/A)")
    print(f"  E_MAE        : {mae_e_str} kcal/mol")
    print(f"  F_MAE        : {mae_f_str} kcal/mol/A")
    print()
    print("  FCHL19 representation parameters:")
    for key in (
        "nRs2",
        "nRs3",
        "nFourier",
        "eta2",
        "eta3",
        "rcut",
        "acut",
        "two_body_decay",
        "three_body_decay",
        "three_body_weight",
    ):
        val = best.params.get(key)
        if val is not None:
            print(f"    {key:<22} = {val}")
    print()
    print("  KRR model parameters:")
    for key in ("sigma", "l2"):
        val = best.params.get(key)
        if val is not None:
            print(f"    {key:<22} = {val:.6g}")
    print()
    print("  Timing (best trial):")
    print(f"    repr generation : {best.user_attrs.get('repr_time_s', '?'):.2f} s")
    print(f"    kernel + solve  : {best.user_attrs.get('kern_time_s', '?'):.2f} s")
    print(f"    prediction      : {best.user_attrs.get('pred_time_s', '?'):.2f} s")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for local KRR with FCHL19 "
        "on rMD17 ethanol split 01."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of Optuna trials (default: {N_TRIALS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for TPE sampler (default: {SEED})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top trials to display in the summary (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show Optuna's per-trial log output (default: WARNING level only)",
    )
    args = parser.parse_args()

    if not args.verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 75)
    print("  KRR + FCHL19 hyperparameter optimization via Optuna")
    print("  Dataset: rMD17 ethanol split 01")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  trials={args.trials}")
    print("=" * 75)
    print()

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    total_time = time.perf_counter() - t0

    print(f"\n  Optimization finished in {total_time:.1f} s  ({args.trials} trials)")

    _print_best(study, top_k=args.top)


if __name__ == "__main__":
    main()
