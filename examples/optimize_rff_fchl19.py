"""
Optuna hyperparameter optimization for local RFF regression with FCHL19 representation.

Uses the rMD17 ethanol split 01 dataset (1000 train + 1000 test frames,
PBE/def2-SVP energies and forces) to optimize all FCHL19 representation
hyperparameters as well as the RFF model hyperparameters (sigma, l2).

Uses elemental Random Fourier Features (RFF) as the regressor instead of KRR,
which makes each trial significantly faster (no O(N^2) kernel matrix) while
still optimizing the same FCHL19 representation hyperparameter space.

Optimizes all of the following simultaneously:
  FCHL19 representation:
    eta2, eta3            -- two- and three-body Gaussian widths
    rcut, acut            -- two- and three-body cutoff radii (Ang)
    two_body_decay        -- decay exponent for two-body term
    three_body_decay      -- decay exponent for three-body term
    three_body_weight     -- weight for three-body contribution
    nRs2, nRs3           -- number of radial basis functions
    nFourier              -- number of angular Fourier terms

  RFF model:
    sigma                 -- Gaussian kernel length-scale (controls W/b sampling)
    l2                    -- L2 regularisation strength

Fixed parameters:
    D_RFF = 4096          -- number of random Fourier features (fixed)
    SEED  = 42            -- seed for W/b generation (fixed per trial for reproducibility)

RFF procedure per trial:
  1. Sample FCHL19 repr + model hyperparams via Optuna
  2. Regenerate FCHL19 representations (X, dX) for all 1000+1000 frames
  3. Sample W ~ N(0, 1/sigma^2),  b ~ Uniform(0, 2*pi)  with fixed SEED
  4. Build normal equations via rff_full_gramian_elemental_rfp  ->  (ZtZ_rfp, ZtY)
  5. Solve  w = (ZtZ + l2*I)^{-1} ZtY  via cho_solve_rfp
  6. Predict on test set via rff_full_elemental  ->  Z_full_te @ w
  7. Return combined test MAE = energy_MAE + force_MAE

Objective:
  Minimise combined test MAE = energy_MAE (kcal/mol) + force_MAE (kcal/mol/Ang)

Dataset:
  rMD17 ethanol split 01 from https://github.com/andersx/rmd17-npz
  Forces are in the convention F = -dE/dR (already negated), used directly
  as training labels without sign flipping.

Usage:
  uv run python examples/optimize_rff_fchl19.py
  uv run python examples/optimize_rff_fchl19.py --trials 200
  uv run python examples/optimize_rff_fchl19.py --trials 5   # quick smoke test
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
from kernelforge.kitchen_sinks import rff_full_elemental, rff_full_gramian_elemental_rfp
from kernelforge.rmd17_data import load_rmd17_ethanol_split01

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRIALS = 100
D_RFF = 4096  # number of random Fourier features (fixed)
ELEMENTS = [1, 6, 8]  # H, C, O
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
SEED = 42  # fixed seed for W/b — deterministic given sigma

# ---------------------------------------------------------------------------
# Load rMD17 data once at import time — shared across all Optuna trials
# ---------------------------------------------------------------------------
print("Loading rMD17 ethanol split 01...")
_TRAIN_DATA, _TEST_DATA = load_rmd17_ethanol_split01()

N_TRAIN = len(_TRAIN_DATA["E"])  # 1000
N_TEST = len(_TEST_DATA["E"])  # 1000

_Z = _TRAIN_DATA["z"]  # (9,) int32

_R_TRAIN = _TRAIN_DATA["R"]  # (1000, 9, 3)
_E_TRAIN = _TRAIN_DATA["E"]  # (1000,)
_F_TRAIN = _TRAIN_DATA["F"]  # (1000, 9, 3)  F = -dE/dR

_R_TEST = _TEST_DATA["R"]  # (1000, 9, 3)
_E_TEST = _TEST_DATA["E"]  # (1000,)
_F_TEST = _TEST_DATA["F"]  # (1000, 9, 3)

_N_ATOMS = len(_Z)  # 9
_NAQ = _N_ATOMS * 3  # 27

# 0-based element indices for elemental RFF Q list (same for every frame)
_Q_MOL = np.array([ELEM_TO_IDX[a] for a in _Z], dtype=np.int32)  # (9,)


# ---------------------------------------------------------------------------
# Representation builder (called once per trial)
# ---------------------------------------------------------------------------


def _build_representations(
    repr_kwargs: dict[str, Any],
) -> tuple[
    np.ndarray,  # X_tr  (N_TRAIN, n_atoms, rep_size)
    np.ndarray,  # dX_tr (N_TRAIN, n_atoms, rep_size, n_atoms, 3)  — 5D
    np.ndarray,  # X_te  (N_TEST,  n_atoms, rep_size)
    np.ndarray,  # dX_te (N_TEST,  n_atoms, rep_size, n_atoms, 3)  — 5D
    list[np.ndarray],  # Q_tr  list of N_TRAIN arrays, each shape (n_atoms,) int32
    list[np.ndarray],  # Q_te  list of N_TEST  arrays, each shape (n_atoms,) int32
]:
    """Generate FCHL19 representations and gradients for all train + test frames.

    The dX arrays are reshaped to 5D ``(N, n_atoms, rep_size, n_atoms, 3)`` as
    required by the elemental RFF functions.  Q is a list of 1D int32 arrays
    holding 0-based element indices (no padding), matching the elemental RFF
    interface.

    Parameters
    ----------
    repr_kwargs:
        Keyword arguments forwarded verbatim to ``generate_fchl_acsf_and_gradients``.
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

    rep_size = X_tr.shape[2]

    # Reshape dX to 5D as required by elemental RFF functions
    dX_tr = dX_tr.reshape(N_TRAIN, n_atoms, rep_size, n_atoms, 3)
    dX_te = dX_te.reshape(N_TEST, n_atoms, rep_size, n_atoms, 3)

    # Q: list of 1D 0-based element index arrays (same for every frame — ethanol)
    Q_tr: list[np.ndarray] = [_Q_MOL] * N_TRAIN
    Q_te: list[np.ndarray] = [_Q_MOL] * N_TEST

    return X_tr, dX_tr, X_te, dX_te, Q_tr, Q_te


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: combined energy + force test MAE.

    Samples hyperparameters, regenerates FCHL19 representations, trains a
    local RFF model on the full rMD17 ethanol split 01 training set, and
    evaluates on the test set.

    Returns
    -------
    float
        Combined MAE = energy_MAE (kcal/mol) + force_MAE (kcal/mol/Ang).
        Optuna minimises this value.
    """
    # ------------------------------------------------------------------
    # 1. Sample hyperparameters
    # ------------------------------------------------------------------
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

    sigma = trial.suggest_float("sigma", 1.0, 200.0, log=True)
    l2 = trial.suggest_float("l2", 1e-9, 1e-3, log=True)

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
        X_tr, dX_tr, X_te, dX_te, Q_tr, Q_te = _build_representations(repr_kwargs)
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        return float("inf")
    repr_time = time.perf_counter() - t_repr

    rep_size = X_tr.shape[2]

    # ------------------------------------------------------------------
    # 3. Sample RFF weights W and b (fixed seed — deterministic per sigma)
    #    W shape: (nelements, rep_size, D_RFF)
    #    b shape: (nelements, D_RFF)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((len(ELEMENTS), rep_size, D_RFF)) / sigma
    b = rng.uniform(0.0, 2.0 * np.pi, (len(ELEMENTS), D_RFF))

    # ------------------------------------------------------------------
    # 4. Build combined energy + force normal equations (RFP packed)
    #    ZtZ_rfp : 1D array, length D_RFF*(D_RFF+1)//2
    #    ZtY     : 1D array, length D_RFF
    # ------------------------------------------------------------------
    t_gramian = time.perf_counter()
    F_tr_flat = _F_TRAIN.reshape(N_TRAIN, _NAQ)
    try:
        ZtZ_rfp, ZtY = rff_full_gramian_elemental_rfp(
            X_tr, dX_tr, Q_tr, W, b, _E_TRAIN, F_tr_flat.ravel()
        )
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        return float("inf")
    gramian_time = time.perf_counter() - t_gramian

    # ------------------------------------------------------------------
    # 5. Solve  w = (ZtZ + l2*I)^{-1} ZtY
    # ------------------------------------------------------------------
    t_solve = time.perf_counter()
    try:
        w = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=l2)
        del ZtZ_rfp
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        return float("inf")
    solve_time = time.perf_counter() - t_solve

    # ------------------------------------------------------------------
    # 6. Predict on test set
    #    Z_full_te shape: (N_TEST*(1+naq), D_RFF)
    # ------------------------------------------------------------------
    t_pred = time.perf_counter()
    Z_full_te = rff_full_elemental(X_te, dX_te, Q_te, W, b)  # (N_TEST*(1+naq), D_RFF)
    y_te_pred = Z_full_te @ w  # (N_TEST*(1+naq),)
    pred_time = time.perf_counter() - t_pred

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, _NAQ)

    F_te_flat = _F_TEST.reshape(N_TEST, _NAQ)

    mae_E = float(np.mean(np.abs(E_te_pred - _E_TEST)))
    mae_F = float(np.mean(np.abs(F_te_pred - F_te_flat)))
    combined_mae = mae_E + mae_F

    trial.set_user_attr("mae_E", mae_E)
    trial.set_user_attr("mae_F", mae_F)
    trial.set_user_attr("repr_time_s", round(repr_time, 2))
    trial.set_user_attr("gramian_time_s", round(gramian_time, 2))
    trial.set_user_attr("solve_time_s", round(solve_time, 2))
    trial.set_user_attr("pred_time_s", round(pred_time, 2))
    trial.set_user_attr("rep_size", rep_size)

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
    mae_e = best.user_attrs.get("mae_E", "?")
    mae_f = best.user_attrs.get("mae_F", "?")
    obj_str = f"{best.value:.6f}" if best.value != float("inf") else "inf"
    mae_e_str = f"{mae_e:.4f}" if isinstance(mae_e, float) else str(mae_e)
    mae_f_str = f"{mae_f:.4f}" if isinstance(mae_f, float) else str(mae_f)

    print("\n" + "=" * 75)
    print("  Best trial hyperparameters")
    print("=" * 75)
    print(f"  trial number : {best.number}")
    print(f"  objective    : {obj_str} kcal/mol(+/A)")
    print(f"  E_MAE        : {mae_e_str} kcal/mol")
    print(f"  F_MAE        : {mae_f_str} kcal/mol/A")
    print(f"  D_RFF        : {D_RFF}  (fixed)")
    print(f"  rep_size     : {best.user_attrs.get('rep_size', '?')}")
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
    print("  RFF model parameters:")
    for key in ("sigma", "l2"):
        val = best.params.get(key)
        if val is not None:
            print(f"    {key:<22} = {val:.6g}")
    print()
    print("  Timing (best trial):")
    print(f"    repr generation : {best.user_attrs.get('repr_time_s', '?'):.2f} s")
    print(f"    gramian build   : {best.user_attrs.get('gramian_time_s', '?'):.2f} s")
    print(f"    cholesky solve  : {best.user_attrs.get('solve_time_s', '?'):.2f} s")
    print(f"    prediction      : {best.user_attrs.get('pred_time_s', '?'):.2f} s")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for local RFF regression with FCHL19 "
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
    print("  RFF + FCHL19 hyperparameter optimization via Optuna")
    print("  Dataset: rMD17 ethanol split 01")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  D_RFF={D_RFF}  trials={args.trials}")
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
