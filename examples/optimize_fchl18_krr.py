"""
Optuna hyperparameter optimization for FCHL18 KRR with combined energy + force training.

Uses the rMD17 ethanol split 01 dataset (first 50 train + first 50 test frames,
PBE/def2-SVP energies and forces).

FCHL18 takes raw Cartesian coordinates and nuclear charges directly — there is no
separate representation generation step per trial, so each trial is purely kernel
computation.  With N=50 the BIG matrix is only 1400x1400, making each trial
complete in roughly 1-5 seconds and 100 trials in under 10 minutes.

Fixed constraints (required by the full energy+force Hessian kernel):
    use_atm      = False   (ATM Hessian not yet implemented)
    cut_start    = 1.0     (cutoff Hessian not yet implemented)
    cut_distance = 20.0    (effectively no cutoff; fixed to reduce search space)

Optimized hyperparameters (9 total):
    sigma              -- Gaussian kernel width
    l2                 -- L2 regularisation strength
    two_body_scaling   -- two-body term scaling
    two_body_width     -- two-body Gaussian width
    two_body_power     -- two-body power decay
    three_body_scaling -- three-body term scaling
    three_body_width   -- three-body Gaussian width
    three_body_power   -- three-body power decay
    fourier_order      -- number of Fourier terms (1 or 2)

Objective:
    Minimise combined test MAE = energy_MAE (kcal/mol) + force_MAE (kcal/mol/Ang)

Dataset:
    rMD17 ethanol split 01 from https://github.com/andersx/rmd17-npz
    Forces stored as F = -dE/dR; negated to match the FCHL18 example convention.

Usage:
    uv run python examples/optimize_fchl18_krr.py
    uv run python examples/optimize_fchl18_krr.py --trials 200
    uv run python examples/optimize_fchl18_krr.py --trials 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import optuna
from optuna.samplers import TPESampler

import kernelforge.fchl18_kernel as fchl18_kernel
from kernelforge import kernelmath
from kernelforge.rmd17_data import load_rmd17_ethanol_split01

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 50
N_TEST = 50
N_TRIALS = 100
SEED = 42

# Fixed kernel constraints — required for the full energy+force Hessian kernel
_CUT_START = 1.0
_CUT_DISTANCE = 20.0  # effectively no cutoff; fixed to reduce search space
_USE_ATM = False

# ---------------------------------------------------------------------------
# Load rMD17 data once at import time — shared across all Optuna trials
# ---------------------------------------------------------------------------
print("Loading rMD17 ethanol split 01...")
_TRAIN_DATA, _TEST_DATA = load_rmd17_ethanol_split01()

_Z = _TRAIN_DATA["z"].astype(np.int32)  # (9,) — same for all ethanol frames
_N_ATOMS = len(_Z)
_D = _N_ATOMS * 3  # 27 degrees of freedom

# Training set: first N_TRAIN frames
_R_TR: list[np.ndarray] = [_TRAIN_DATA["R"][i].astype(np.float64) for i in range(N_TRAIN)]
_Z_TR: list[np.ndarray] = [_Z] * N_TRAIN
_E_TR: np.ndarray = _TRAIN_DATA["E"][:N_TRAIN].astype(np.float64)
# rMD17 forces are F = -dE/dR; negate to match the FCHL18 example convention
_F_TR: list[np.ndarray] = [-_TRAIN_DATA["F"][i].astype(np.float64) for i in range(N_TRAIN)]

# Test set: first N_TEST frames
_R_TE: list[np.ndarray] = [_TEST_DATA["R"][i].astype(np.float64) for i in range(N_TEST)]
_Z_TE: list[np.ndarray] = [_Z] * N_TEST
_E_TE: np.ndarray = _TEST_DATA["E"][:N_TEST].astype(np.float64)
_F_TE: list[np.ndarray] = [-_TEST_DATA["F"][i].astype(np.float64) for i in range(N_TEST)]

# Pre-build combined training label vector: [E_tr; F_tr_flat]
_F_TR_FLAT: np.ndarray = np.concatenate([f.ravel() for f in _F_TR])  # (N_TRAIN*D,)
_Y_TR: np.ndarray = np.concatenate([_E_TR, _F_TR_FLAT])  # (N_TRAIN*(1+D),)

# Pre-stack test forces for evaluation
_F_TE_TRUE: np.ndarray = np.stack([f.ravel() for f in _F_TE])  # (N_TEST, D)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: combined energy + force test MAE.

    Samples FCHL18 kernel hyperparameters, builds the full symmetric training
    kernel, solves for alpha, predicts on the test set, and returns
    energy_MAE + force_MAE.

    Returns
    -------
    float
        Combined MAE = energy_MAE (kcal/mol) + force_MAE (kcal/mol/Ang).
    """
    # ------------------------------------------------------------------
    # 1. Sample hyperparameters
    # ------------------------------------------------------------------
    sigma = trial.suggest_float("sigma", 0.5, 20.0, log=True)
    l2 = trial.suggest_float("l2", 1e-12, 1e-4, log=True)
    two_body_scaling = trial.suggest_float("two_body_scaling", 0.5, 5.0)
    two_body_width = trial.suggest_float("two_body_width", 0.05, 2.0, log=True)
    two_body_power = trial.suggest_float("two_body_power", 1.0, 8.0)
    three_body_scaling = trial.suggest_float("three_body_scaling", 0.5, 5.0)
    three_body_width = trial.suggest_float("three_body_width", 0.5, 10.0, log=True)
    three_body_power = trial.suggest_float("three_body_power", 1.0, 6.0)
    fourier_order = trial.suggest_int("fourier_order", 1, 2)

    kernel_args = {
        "two_body_scaling": two_body_scaling,
        "two_body_width": two_body_width,
        "two_body_power": two_body_power,
        "three_body_scaling": three_body_scaling,
        "three_body_width": three_body_width,
        "three_body_power": three_body_power,
        "cut_start": _CUT_START,
        "cut_distance": _CUT_DISTANCE,
        "fourier_order": fourier_order,
        "use_atm": _USE_ATM,
    }

    # ------------------------------------------------------------------
    # 2. Build training kernel — full combined, symmetric
    # ------------------------------------------------------------------
    t_kern = time.perf_counter()
    try:
        K_rfp = fchl18_kernel.kernel_gaussian_full_symm(_R_TR, _Z_TR, sigma=sigma, **kernel_args)
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        return float("inf")
    kern_time = time.perf_counter() - t_kern

    # ------------------------------------------------------------------
    # 3. Solve  alpha = (K + l2*I)^{-1} y_train
    # ------------------------------------------------------------------
    t_solve = time.perf_counter()
    try:
        alpha = kernelmath.solve_cholesky(K_rfp, _Y_TR, regularize=l2)
        del K_rfp
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        return float("inf")
    solve_time = time.perf_counter() - t_solve

    # ------------------------------------------------------------------
    # 4. Predict on test set
    # ------------------------------------------------------------------
    t_pred = time.perf_counter()
    try:
        K_pred = fchl18_kernel.kernel_gaussian_full(
            _R_TE, _Z_TE, _R_TR, _Z_TR, sigma=sigma, **kernel_args
        )
        y_te_pred = K_pred @ alpha
        del K_pred
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        return float("inf")
    pred_time = time.perf_counter() - t_pred

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    E_te_pred = y_te_pred[:N_TEST]
    F_te_pred = y_te_pred[N_TEST:].reshape(N_TEST, _D)

    mae_E = float(np.mean(np.abs(E_te_pred - _E_TE)))
    mae_F = float(np.mean(np.abs(F_te_pred - _F_TE_TRUE)))
    combined_mae = mae_E + mae_F

    trial.set_user_attr("mae_E", mae_E)
    trial.set_user_attr("mae_F", mae_F)
    trial.set_user_attr("kern_time_s", round(kern_time, 3))
    trial.set_user_attr("solve_time_s", round(solve_time, 3))
    trial.set_user_attr("pred_time_s", round(pred_time, 3))

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
    print()
    print("  Fixed parameters:")
    print(f"    use_atm              = {_USE_ATM}")
    print(f"    cut_start            = {_CUT_START}")
    print(f"    cut_distance         = {_CUT_DISTANCE}")
    print()
    print("  Optimized kernel parameters:")
    for key in (
        "sigma",
        "two_body_scaling",
        "two_body_width",
        "two_body_power",
        "three_body_scaling",
        "three_body_width",
        "three_body_power",
        "fourier_order",
    ):
        val = best.params.get(key)
        if val is not None:
            print(f"    {key:<22} = {val}")
    print()
    print("  Regularisation:")
    val = best.params.get("l2")
    if val is not None:
        print(f"    {'l2':<22} = {val:.6g}")
    print()
    print("  Timing (best trial):")
    print(f"    kernel build    : {best.user_attrs.get('kern_time_s', '?'):.3f} s")
    print(f"    cholesky solve  : {best.user_attrs.get('solve_time_s', '?'):.3f} s")
    print(f"    prediction      : {best.user_attrs.get('pred_time_s', '?'):.3f} s")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for FCHL18 KRR "
        "on rMD17 ethanol split 01 (N_train=50, N_test=50)."
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
        help="Show Optuna per-trial log output (default: WARNING level only)",
    )
    args = parser.parse_args()

    if not args.verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 75)
    print("  FCHL18 KRR hyperparameter optimization via Optuna")
    print("  Dataset: rMD17 ethanol split 01")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  trials={args.trials}")
    print(f"  Fixed: use_atm={_USE_ATM}  cut_start={_CUT_START}  cut_distance={_CUT_DISTANCE}")
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
