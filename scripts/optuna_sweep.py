"""optuna_sweep.py — Optuna TPE hyperparameter search for odd_fourier_element_resolved.

Optimises the three-body (and sigma) parameters of fchl19v2 with:
  - two_body_type = bessel  (fixed)
  - two_body_decay = 0.5    (fixed, best from Phase 1 pilot)
  - eta2 = 0.32             (fixed, no-op for bessel)
  - three_body_type = odd_fourier_element_resolved  (fixed)
  - use_atm = True          (fixed, always best in Phase 2)
  - use_three_body = True
  - training: energy-only (no Jacobian — fast)
  - objective: E-MAE

Search space for rMD17 (per trial):
  sigma             log-uniform  [0.5, 128]
  nFourier          int          [1, 4]
  nRs3              int          [8, 32]   (fixed at 24 for small_mols_mini)
  nRs3_minus        int          [8, 32]   (fixed at 24 for small_mols_mini)
  eta3              log-uniform  [0.5, 10.0]
  eta3_minus        log-uniform  [0.5, 10.0]
  three_body_decay  uniform      [0.2, 2.0]
  three_body_weight log-uniform  [1.0, 100.0]

Storage: JournalStorage (file-based, resume-safe).
  rMD17:           ~/.kernelforge/results/optuna_{molecule}.log
  small_mols_mini: ~/.kernelforge/results/optuna_small_mols_mini.log

Usage
-----
    # rMD17 (default)
    uv run scripts/optuna_sweep.py --molecules ethanol benzene --n-trials 200
    uv run scripts/optuna_sweep.py --summary

    # small_mols_mini
    uv run scripts/optuna_sweep.py --dataset small_mols_mini --n-train 100 --n-trials 200
    uv run scripts/optuna_sweep.py --dataset small_mols_mini --summary
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


class _Model(Protocol):
    def fit(
        self,
        coords_list: list[np.ndarray],
        z_list: list[np.ndarray],
        energies: np.ndarray | None = None,
        forces: list[np.ndarray] | np.ndarray | None = None,
    ) -> object: ...

    def predict(
        self,
        coords_list: list[np.ndarray],
        z_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]: ...


optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path.home() / ".kernelforge" / "results"

# rMD17 published E+F baselines at N=50
BASELINE_E50: dict[str, float] = {
    "aspirin": 0.754,
    "azobenzene": 0.295,
    "benzene": 0.015,
    "ethanol": 0.232,
    "malonaldehyde": 0.312,
    "naphthalene": 0.127,
    "paracetamol": 0.391,
    "salicylic": 0.255,
    "toluene": 0.169,
    "uracil": 0.147,
}

# rMD17 energy-only baselines at N=50
BASELINE_E_ONLY_50: dict[str, float] = {
    "ethanol": 1.99,
    "benzene": 0.16,
}

BASELINE_F50: dict[str, float] = {
    "aspirin": 1.818,
    "azobenzene": 0.844,
    "benzene": 0.125,
    "ethanol": 1.072,
    "malonaldehyde": 1.469,
    "naphthalene": 0.583,
    "paracetamol": 1.181,
    "salicylic": 1.088,
    "toluene": 0.754,
    "uracil": 0.812,
}

# Best params from Phase 2 pilot (split=1, odd_fourier_element_resolved, use_atm=True)
# Used as warm-start enqueue for TPE on rMD17.
PHASE2_BEST: dict[str, dict[str, float | int]] = {
    "ethanol": {
        "sigma": 16.0,
        "nFourier": 1,
        "nRs3": 20,
        "nRs3_minus": 20,
        "eta3": 2.7,
        "eta3_minus": 2.7,
        "three_body_decay": 0.57,
        "three_body_weight": 13.4,
    },
    "benzene": {
        "sigma": 32.0,
        "nFourier": 1,
        "nRs3": 20,
        "nRs3_minus": 20,
        "eta3": 2.7,
        "eta3_minus": 2.7,
        "three_body_decay": 0.57,
        "three_body_weight": 13.4,
    },
}

FIXED_REPR_PARAMS: dict[str, str | float | bool] = {
    "two_body_type": "bessel",
    "eta2": 0.32,
    "two_body_decay": 0.5,
    "three_body_type": "odd_fourier_element_resolved",
    "use_atm": True,
    "use_three_body": True,
}

CSV_FIELDNAMES = [
    "molecule",
    "trial",
    "sigma",
    "nFourier",
    "nRs3",
    "nRs3_minus",
    "eta3",
    "eta3_minus",
    "three_body_decay",
    "three_body_weight",
    "energy_mae",
    "objective",
    "elapsed_s",
]

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.ravel() - b.ravel())))


def make_objective(
    label: str,
    n_train: int,
    n_test: int,
    split: int,
    loader: Callable[..., tuple[np.ndarray, ...]],
    _build_model: Callable[..., _Model],
    csv_path: Path,
    fix_nrs3: int | None,
    baseline: float,
) -> Callable[[optuna.Trial], float]:
    """Return an Optuna objective function closed over the dataset.

    Parameters
    ----------
    label:
        Human-readable label (molecule name or dataset name) for logging.
    fix_nrs3:
        If not None, nRs3 and nRs3_minus are fixed to this value (not searched).
    baseline:
        Reference E-MAE for progress reporting (nan if unknown).
    """
    # Load data once — reused across all trials
    if split >= 0:
        coords_tr, z_tr, E_tr, _F_tr, coords_te, z_te, E_te, _F_te = loader(
            label, split, n_train, n_test
        )
    else:
        # small_mols_mini: loader takes only (n_train, n_test)
        coords_tr, z_tr, E_tr, _F_tr, coords_te, z_te, E_te, _F_te = loader(n_train, n_test)

    def objective(trial: optuna.Trial) -> float:
        sigma = trial.suggest_float("sigma", 0.5, 128.0, log=True)
        nFourier = trial.suggest_int("nFourier", 1, 4)
        if fix_nrs3 is not None:
            nRs3 = fix_nrs3
            nRs3_minus = fix_nrs3
            trial.set_user_attr("nRs3", nRs3)
            trial.set_user_attr("nRs3_minus", nRs3_minus)
        else:
            nRs3 = trial.suggest_int("nRs3", 8, 32)
            nRs3_minus = trial.suggest_int("nRs3_minus", 8, 32)
        eta3 = trial.suggest_float("eta3", 0.5, 10.0, log=True)
        eta3_minus = trial.suggest_float("eta3_minus", 0.5, 10.0, log=True)
        three_body_decay = trial.suggest_float("three_body_decay", 0.2, 2.0)
        three_body_weight = trial.suggest_float("three_body_weight", 1.0, 100.0, log=True)

        repr_params: dict[str, str | float | bool | int] = {
            **FIXED_REPR_PARAMS,
            "nFourier": nFourier,
            "nRs3": nRs3,
            "nRs3_minus": nRs3_minus,
            "eta3": eta3,
            "eta3_minus": eta3_minus,
            "three_body_decay": three_body_decay,
            "three_body_weight": three_body_weight,
        }

        t0 = time.perf_counter()
        try:
            model = _build_model(
                regressor="krr",
                representation="fchl19v2",
                sigma=sigma,
                l2=1e-8,
                elements=None,
                max_size=None,
                d_rff=1024,
                seed=42,
                z_tr=z_tr,
                z_te=z_te,
                repr_params=repr_params,
            )
            model.fit(coords_tr, z_tr, energies=E_tr)
            E_pred, _ = model.predict(coords_te, z_te)
        except Exception as exc:
            print(f"  ERROR trial {trial.number}: {exc}")
            raise optuna.exceptions.TrialPruned() from exc
        elapsed = time.perf_counter() - t0

        e_mae = _compute_mae(E_te, E_pred)
        obj = e_mae

        de = e_mae - baseline
        fe = "▼" if de < 0 else ("~" if abs(de) < 0.05 * baseline else "▲")
        print(
            f"  [{trial.number:4d}] E={e_mae:.4f}({fe}{abs(de):.4f})"
            f"  s={sigma:.2f} nF={nFourier} nRs3={nRs3}/{nRs3_minus}"
            f" n3={eta3:.2f}/{eta3_minus:.2f} d={three_body_decay:.2f}"
            f" w={three_body_weight:.1f}  {elapsed:.1f}s",
            flush=True,
        )

        # Append to CSV
        row: dict[str, str | int] = {
            "molecule": label,
            "trial": trial.number,
            "sigma": f"{sigma:.6f}",
            "nFourier": nFourier,
            "nRs3": nRs3,
            "nRs3_minus": nRs3_minus,
            "eta3": f"{eta3:.6f}",
            "eta3_minus": f"{eta3_minus:.6f}",
            "three_body_decay": f"{three_body_decay:.6f}",
            "three_body_weight": f"{three_body_weight:.6f}",
            "energy_mae": f"{e_mae:.6f}",
            "objective": f"{obj:.6f}",
            "elapsed_s": f"{elapsed:.1f}",
        }
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        return obj

    return objective


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(labels: list[str], csv_suffix: str, baseline_map: dict[str, float]) -> None:
    hdr = (
        f"{'Label':20s} {'nF':>3s} {'nRs3':>5s} {'nRs3-':>5s}"
        f" {'n3':>6s} {'n3-':>6s} {'dec':>5s} {'w':>6s} {'s':>7s}"
        f"  {'E-MAE':>7s} {'Δbase':>7s}  {'trials':>6s}"
    )
    sep = "-" * len(hdr)
    print(f"\n{'=' * len(hdr)}")
    print("  Optuna best params (energy-only)")
    print(f"{'=' * len(hdr)}")
    print(f"  {hdr}")
    print(f"  {sep}")

    for lbl in labels:
        csv_path = RESULTS_DIR / f"optuna_trials_{csv_suffix}.csv"
        if not csv_path.exists():
            print(f"  {lbl:20s}  (no results)")
            continue
        with csv_path.open(newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r["molecule"] == lbl]
        if not rows:
            print(f"  {lbl:20s}  (empty)")
            continue
        best = min(rows, key=lambda r: float(r["energy_mae"]))
        e = float(best["energy_mae"])
        b_e = baseline_map.get(lbl, float("nan"))
        de = e - b_e
        fe = "▼" if de < 0 else "▲"
        print(
            f"  {lbl:20s}"
            f" {int(best['nFourier']):3d} {int(best['nRs3']):5d} {int(best['nRs3_minus']):5d}"
            f" {float(best['eta3']):6.3f} {float(best['eta3_minus']):6.3f}"
            f" {float(best['three_body_decay']):5.2f} {float(best['three_body_weight']):6.1f}"
            f" {float(best['sigma']):7.2f}"
            f"  {e:7.4f} {fe}{abs(de):.4f}  {len(rows):6d}"
        )

    print(f"  {sep}")
    print(f"{'=' * len(hdr)}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Optuna TPE sweep for odd_fourier_element_resolved.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="rmd17",
        choices=["rmd17", "small_mols_mini"],
        help="Dataset to optimise on.",
    )
    p.add_argument(
        "--molecules",
        nargs="+",
        default=["ethanol", "benzene"],
        metavar="MOL",
        help="[rmd17 only] Molecules to optimise (one independent study each).",
    )
    p.add_argument("--n-trials", type=int, default=200, help="Optuna trials per study.")
    p.add_argument("--n-train", type=int, default=50, help="Training set size.")
    p.add_argument("--n-test", type=int, default=200, help="Test set size.")
    p.add_argument("--split", type=int, default=1, help="[rmd17 only] Pre-defined split index.")
    p.add_argument(
        "--summary", action="store_true", help="Print summary from existing CSVs and exit."
    )
    args = p.parse_args()

    is_small_mols = args.dataset == "small_mols_mini"

    if is_small_mols:
        study_labels = ["small_mols_mini"]
        csv_suffix = "small_mols_mini"
        baseline_map: dict[str, float] = {"small_mols_mini": float("nan")}
    else:
        study_labels = args.molecules
        csv_suffix = "rmd17"
        baseline_map = BASELINE_E_ONLY_50

    if args.summary:
        _print_summary(study_labels, csv_suffix, baseline_map)
        return

    from kernelforge.kernelcli import _build_model  # type: ignore[attr-defined]

    if is_small_mols:
        from kernelforge.kernelcli import (
            load_small_mols_mini as _loader,  # type: ignore[attr-defined]
        )
    else:
        from kernelforge.kernelcli import load_rmd17 as _loader  # type: ignore[attr-defined]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for lbl in study_labels:
        n_test = args.n_test if not is_small_mols else 595  # use full test set for small_mols_mini

        print(f"\n{'=' * 72}")
        print(f"  Optuna sweep — {lbl}  ({args.n_trials} trials, n_train={args.n_train})")
        print("  Fixed: bessel two-body, decay=0.5, odd_fourier_element_resolved, use_atm=True")
        if is_small_mols:
            print("  nRs3 = nRs3_minus = 24 (fixed)")
        print("  Objective: E-MAE (energy-only, no Jacobian)")
        print(f"  Storage: {RESULTS_DIR}/optuna_{lbl}.log  (resume-safe)")
        print(f"{'=' * 72}\n")

        log_path = RESULTS_DIR / f"optuna_{lbl}.log"
        csv_path = RESULTS_DIR / f"optuna_trials_{csv_suffix}.csv"

        storage = JournalStorage(JournalFileBackend(str(log_path)))
        study = optuna.create_study(
            study_name=f"fchl19v2_element_resolved_{lbl}",
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Warm-start for rMD17 molecules
        if not is_small_mols and lbl in PHASE2_BEST and len(study.trials) == 0:
            study.enqueue_trial(PHASE2_BEST[lbl])
            print(f"  Enqueued Phase 2 warm-start: {PHASE2_BEST[lbl]}\n")

        already_done = len(study.trials)
        remaining = max(0, args.n_trials - already_done)
        if remaining == 0:
            print(f"  Already have {already_done} trials — nothing to do. Use --summary.")
        else:
            print(f"  Resuming from {already_done} completed trials, running {remaining} more.\n")

        objective = make_objective(
            label=lbl,
            n_train=args.n_train,
            n_test=n_test,
            split=-1 if is_small_mols else args.split,
            loader=_loader,
            _build_model=_build_model,
            csv_path=csv_path,
            fix_nrs3=24 if is_small_mols else None,
            baseline=baseline_map.get(lbl, float("nan")),
        )
        study.optimize(objective, n_trials=remaining)

        best = study.best_trial
        best_val = best.value if best.value is not None else float("nan")
        b_e = baseline_map.get(lbl, float("nan"))
        de = best_val - b_e
        fe = "▼" if de < 0 else "▲"
        print(
            f"\n  Best trial #{best.number}: E-MAE={best_val:.4f} ({fe}{abs(de):.4f} vs baseline)"
        )
        print(f"  Params: {best.params}")

    _print_summary(study_labels, csv_suffix, baseline_map)


if __name__ == "__main__":
    main()
