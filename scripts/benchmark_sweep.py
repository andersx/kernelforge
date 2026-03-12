"""benchmark_sweep.py — two-phase hyperparameter search for fchl19v2 on rMD17.

Phase 1
-------
Sweeps all five two-body types × eta2 × two_body_decay × sigma with the
three-body term disabled.  Special-cases: Bessel (T5) does not use eta2 so
eta2 is fixed at its default (0.32) for that type.

Pilot mode (--pilot): ethanol + benzene, split=1 only (~2 400 runs, ~32 min).
Full mode  (--full):  all 10 molecules, splits 1-3 (~72 000 runs).

Phase 2
-------
Re-uses the per-molecule Phase-1 optimum and sweeps all seven three-body types
× use_atm={True,False}, re-optimising sigma over the same grid.

Usage
-----
    uv run scripts/benchmark_sweep.py --phase 1 --pilot
    uv run scripts/benchmark_sweep.py --phase 1 --full
    uv run scripts/benchmark_sweep.py --phase 1 --pilot --dry-run
    uv run scripts/benchmark_sweep.py --phase 1 --summary

Results are written to  ~/.kernelforge/results/benchmark_phase{1,2}.csv.
The script is resume-safe: rows already in the CSV are skipped on restart.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# FCHL19 published baseline (N=50 train, energy_and_force, all 5 splits avg)
# Units: energy MAE in kcal/mol, force MAE in kcal/mol/Å
# ---------------------------------------------------------------------------

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

ALL_MOLECULES = list(BASELINE_E50.keys())
PILOT_MOLECULES = ["ethanol", "benzene"]

# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------

TWO_BODY_TYPES = [
    "log_normal",
    "gaussian_r",
    "gaussian_log_r",
    "gaussian_r_no_pow",
    "bessel",
]

# Bessel (T5) does not use eta2 — fix at default for that type
BESSEL_TYPE = "bessel"
ETA2_GRID = [0.1, 0.2, 0.32, 0.5, 0.8, 1.4]
ETA2_DEFAULT = 0.32

TWO_BODY_DECAY_GRID = [0.5, 1.0, 1.5, 1.8, 2.5, 3.0]

SIGMA_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

THREE_BODY_TYPES = [
    "odd_fourier_rbar",  # A1
    "cosine_rbar",  # A2
    "odd_fourier_split_r",  # A3 (needs nRs3_minus)
    "cosine_split_r",  # A4 (needs nRs3_minus)
    "cosine_split_r_no_atm",  # A5 (needs nRs3_minus)
    "odd_fourier_element_resolved",  # A6 (needs nRs3_minus)
    "cosine_element_resolved",  # A7 (needs nRs3_minus)
]

THREE_BODY_NEEDS_RS3_MINUS = {
    "odd_fourier_split_r",
    "cosine_split_r",
    "cosine_split_r_no_atm",
    "odd_fourier_element_resolved",
    "cosine_element_resolved",
}

RESULTS_DIR = Path.home() / ".kernelforge" / "results"


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _phase1_configs(molecules: list[str], splits: list[int]) -> list[dict[str, Any]]:
    """Return the full list of Phase 1 config dicts to run."""
    configs = []
    for mol in molecules:
        for tb_type in TWO_BODY_TYPES:
            eta2_values = [ETA2_DEFAULT] if tb_type == BESSEL_TYPE else ETA2_GRID
            for eta2 in eta2_values:
                for decay in TWO_BODY_DECAY_GRID:
                    for sigma in SIGMA_GRID:
                        for split in splits:
                            configs.append(
                                {
                                    "molecule": mol,
                                    "split": split,
                                    "two_body_type": tb_type,
                                    "eta2": eta2,
                                    "two_body_decay": decay,
                                    "sigma": sigma,
                                }
                            )
    return configs


# ---------------------------------------------------------------------------
# Core: single training + evaluation run
# ---------------------------------------------------------------------------


def _compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.ravel() - b.ravel())))


def run_one(
    cfg: dict[str, Any],
    n_train: int,
    n_test: int,
    load_rmd17: Any,
    _build_model: Any,
    forces: bool = True,
) -> tuple[float, float]:
    """Train one KRR model and return (energy_mae, force_mae).

    When *forces* is False the model is trained on energies only and force_mae
    is returned as NaN (no Jacobian computed — much faster for large reps).
    """
    coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te = load_rmd17(
        cfg["molecule"], cfg["split"], n_train, n_test
    )

    repr_params: dict[str, Any] = {
        "two_body_type": cfg["two_body_type"],
        "eta2": cfg["eta2"],
        "two_body_decay": cfg["two_body_decay"],
        "use_three_body": False,
    }

    model = _build_model(
        regressor="krr",
        representation="fchl19v2",
        sigma=cfg["sigma"],
        l2=1e-8,
        elements=None,
        max_size=None,
        d_rff=1024,
        seed=42,
        z_tr=z_tr,
        z_te=z_te,
        repr_params=repr_params,
    )

    if forces:
        model.fit(coords_tr, z_tr, energies=E_tr, forces=F_tr)
        E_pred, F_pred = model.predict(coords_te, z_te)
        e_mae = _compute_mae(E_te, E_pred)
        F_ref = np.concatenate([f.ravel() for f in F_te])
        f_mae = _compute_mae(F_ref, F_pred)
    else:
        model.fit(coords_tr, z_tr, energies=E_tr)
        E_pred, _ = model.predict(coords_te, z_te)
        e_mae = _compute_mae(E_te, E_pred)
        f_mae = float("nan")

    return e_mae, f_mae


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

PHASE1_FIELDNAMES = [
    "molecule",
    "split",
    "two_body_type",
    "eta2",
    "two_body_decay",
    "sigma",
    "n_train",
    "n_test",
    "energy_mae",
    "force_mae",
    "elapsed_s",
]

PHASE1_KEY_COLS = ["molecule", "split", "two_body_type", "eta2", "two_body_decay", "sigma"]

PHASE2_FIELDNAMES = [
    "molecule",
    "split",
    "two_body_type",
    "eta2",
    "two_body_decay",
    "three_body_type",
    "use_atm",
    "nRs3_minus",
    "sigma",
    "n_train",
    "n_test",
    "energy_mae",
    "force_mae",
    "elapsed_s",
]

PHASE2_KEY_COLS = [
    "molecule",
    "split",
    "two_body_type",
    "eta2",
    "two_body_decay",
    "three_body_type",
    "use_atm",
    "sigma",
]


def _load_done(csv_path: Path, key_cols: list[str]) -> set[tuple[str, ...]]:
    done: set[tuple[str, ...]] = set()
    if not csv_path.exists():
        return done
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add(tuple(row[c] for c in key_cols))
            except KeyError:
                pass
    return done


def _row_key(cfg: dict[str, Any], key_cols: list[str]) -> tuple[str, ...]:
    return tuple(str(cfg[c]) for c in key_cols)


def _append_row(csv_path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


def run_phase1(
    molecules: list[str],
    splits: list[int],
    n_train: int,
    n_test: int,
    dry_run: bool,
    forces: bool = True,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "benchmark_phase1.csv"

    from kernelforge.kernelcli import _build_model, load_rmd17  # type: ignore[attr-defined]

    configs = _phase1_configs(molecules, splits)
    done = _load_done(csv_path, PHASE1_KEY_COLS)
    total = len(configs)
    n_done = sum(1 for c in configs if _row_key(c, PHASE1_KEY_COLS) in done)

    print(f"\n{'=' * 78}")
    print(f"  Phase 1 — Two-body sweep (three-body OFF){'' if forces else '  [energy-only]'}")
    print(f"  molecules={molecules}  splits={splits}")
    print(f"  n_train={n_train}  n_test={n_test}")
    print(f"  Output: {csv_path}")
    print(f"  Total: {total}  done: {n_done}  remaining: {total - n_done}")
    print(f"{'=' * 78}\n")

    run_idx = 0
    for cfg in configs:
        key = _row_key(cfg, PHASE1_KEY_COLS)
        if key in done:
            continue

        run_idx += 1
        label = (
            f"{cfg['molecule']:14s} | {cfg['two_body_type']:20s}"
            f" η={cfg['eta2']:<5.2f} d={cfg['two_body_decay']:<4.1f}"
            f" σ={cfg['sigma']:<5.1f} sp={cfg['split']}"
        )
        print(f"[{n_done + run_idx:5d}/{total}] {label}", end="", flush=True)

        if dry_run:
            print("  [DRY-RUN]")
            continue

        t0 = time.perf_counter()
        try:
            e_mae, f_mae = run_one(cfg, n_train, n_test, load_rmd17, _build_model, forces=forces)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        elapsed = time.perf_counter() - t0

        row: dict[str, Any] = {
            "molecule": cfg["molecule"],
            "split": cfg["split"],
            "two_body_type": cfg["two_body_type"],
            "eta2": cfg["eta2"],
            "two_body_decay": cfg["two_body_decay"],
            "sigma": cfg["sigma"],
            "n_train": n_train,
            "n_test": n_test,
            "energy_mae": f"{e_mae:.6f}",
            "force_mae": f"{f_mae:.6f}" if forces else "nan",
            "elapsed_s": f"{elapsed:.1f}",
        }
        _append_row(csv_path, PHASE1_FIELDNAMES, row)
        done.add(key)

        b_e = BASELINE_E50.get(cfg["molecule"], float("nan"))
        de = e_mae - b_e
        fe = "▼" if de < 0 else ("~" if abs(de) < 0.05 * b_e else "▲")
        if forces:
            b_f = BASELINE_F50.get(cfg["molecule"], float("nan"))
            df = f_mae - b_f
            ff = "▼" if df < 0 else ("~" if abs(df) < 0.05 * b_f else "▲")
            print(
                f"  E={e_mae:.3f}({fe}{abs(de):.3f})"
                f"  F={f_mae:.3f}({ff}{abs(df):.3f})  {elapsed:.1f}s"
            )
        else:
            print(f"  E={e_mae:.3f}({fe}{abs(de):.3f})  {elapsed:.1f}s")

    print(f"\n{'=' * 78}")
    print(f"  Phase 1 complete.  Results: {csv_path}")
    print(f"{'=' * 78}\n")

    if not dry_run and csv_path.exists():
        _print_phase1_summary(csv_path)


# ---------------------------------------------------------------------------
# Phase 1 summary
# ---------------------------------------------------------------------------


def _print_phase1_summary(csv_path: Path) -> None:
    from collections import defaultdict

    # key: (molecule, two_body_type, eta2, two_body_decay, sigma) → [e_mae, ...]
    e_acc: dict[tuple, list[float]] = defaultdict(list)
    f_acc: dict[tuple, list[float]] = defaultdict(list)

    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            k = (
                row["molecule"],
                row["two_body_type"],
                float(row["eta2"]),
                float(row["two_body_decay"]),
                float(row["sigma"]),
            )
            e_acc[k].append(float(row["energy_mae"]))
            f_acc[k].append(float(row["force_mae"]))

    # Best per molecule: minimise mean E-MAE
    best: dict[str, dict[str, Any]] = {}
    for k, e_list in e_acc.items():
        mol, tb, eta2, decay, sigma = k
        mean_e = float(np.mean(e_list))
        mean_f = float(np.mean(f_acc[k]))
        if mol not in best or mean_e < best[mol]["mean_e"]:
            best[mol] = {
                "two_body_type": tb,
                "eta2": eta2,
                "decay": decay,
                "sigma": sigma,
                "mean_e": mean_e,
                "mean_f": mean_f,
                "n_splits": len(e_list),
            }

    hdr = (
        f"{'Molecule':14s} {'TB type':20s} {'η2':>5s} {'dec':>4s} {'σ':>5s}"
        f"  {'E-MAE':>7s} {'Δbase':>7s}   {'F-MAE':>7s} {'Δbase':>7s}  {'nspl':>4s}"
    )
    sep = "-" * len(hdr)
    print(f"\n{'=' * len(hdr)}")
    print("  Phase 1 — Best params per molecule (mean over available splits)")
    print(f"{'=' * len(hdr)}")
    print(f"  {hdr}")
    print(f"  {sep}")

    molecules_present = [m for m in ALL_MOLECULES if m in best]
    # also show any molecule not in ALL_MOLECULES (shouldn't happen but be safe)
    for m in best:
        if m not in molecules_present:
            molecules_present.append(m)

    for mol in molecules_present:
        b = best[mol]
        b_e = BASELINE_E50.get(mol, float("nan"))
        b_f = BASELINE_F50.get(mol, float("nan"))
        de = b["mean_e"] - b_e
        df = b["mean_f"] - b_f
        fe = "▼" if de < 0 else "▲"
        ff = "▼" if df < 0 else "▲"
        print(
            f"  {mol:14s} {b['two_body_type']:20s} {b['eta2']:>5.2f} {b['decay']:>4.1f}"
            f" {b['sigma']:>5.1f}"
            f"  {b['mean_e']:7.3f} {fe}{abs(de):.3f}   "
            f"{b['mean_f']:7.3f} {ff}{abs(df):.3f}  {b['n_splits']:>4d}"
        )

    print(f"  {sep}")
    print(f"  Baseline: FCHL19 published E-MAE / F-MAE at N=50 (kcal/mol)")
    print(f"{'=' * len(hdr)}\n")


# ---------------------------------------------------------------------------
# Phase 2 — three-body sweep (best two-body params from Phase 1)
# ---------------------------------------------------------------------------


def _load_phase1_best(
    phase1_csv: Path,
    molecules: list[str],
    fixed_two_body_type: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Read Phase 1 CSV and return per-molecule best params (min mean E-MAE).

    If *fixed_two_body_type* is given, only rows with that two_body_type are
    considered (e.g. ``"bessel"`` to force bessel two-body into Phase 2).
    """
    from collections import defaultdict

    e_acc: dict[tuple, list[float]] = defaultdict(list)

    with phase1_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["molecule"] not in molecules:
                continue
            if fixed_two_body_type and row["two_body_type"] != fixed_two_body_type:
                continue
            k = (
                row["molecule"],
                row["two_body_type"],
                float(row["eta2"]),
                float(row["two_body_decay"]),
                float(row["sigma"]),
            )
            e_acc[k].append(float(row["energy_mae"]))

    best: dict[str, dict[str, Any]] = {}
    for (mol, tb, eta2, decay, sigma), e_list in e_acc.items():
        mean_e = float(np.mean(e_list))
        if mol not in best or mean_e < best[mol]["mean_e"]:
            best[mol] = {
                "two_body_type": tb,
                "eta2": eta2,
                "two_body_decay": decay,
                "sigma": sigma,
                "mean_e": mean_e,
            }
    return best


def _phase2_configs(
    molecules: list[str],
    splits: list[int],
    phase1_best: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    configs = []
    for mol in molecules:
        if mol not in phase1_best:
            continue
        pb = phase1_best[mol]
        for ab_type in THREE_BODY_TYPES:
            for use_atm in [True, False]:
                for sigma in SIGMA_GRID:
                    for split in splits:
                        configs.append(
                            {
                                "molecule": mol,
                                "split": split,
                                "two_body_type": pb["two_body_type"],
                                "eta2": pb["eta2"],
                                "two_body_decay": pb["two_body_decay"],
                                "three_body_type": ab_type,
                                "use_atm": use_atm,
                                "sigma": sigma,
                            }
                        )
    return configs


def run_phase2(
    molecules: list[str],
    splits: list[int],
    n_train: int,
    n_test: int,
    dry_run: bool,
    fixed_two_body_type: str | None = None,
    forces: bool = True,
) -> None:
    phase1_csv = RESULTS_DIR / "benchmark_phase1.csv"
    if not phase1_csv.exists():
        print(
            f"ERROR: Phase 1 results not found at {phase1_csv}\nRun --phase 1 first.",
            file=sys.stderr,
        )
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "benchmark_phase2.csv"

    from kernelforge.kernelcli import _build_model, load_rmd17  # type: ignore[attr-defined]

    phase1_best = _load_phase1_best(phase1_csv, molecules, fixed_two_body_type)
    missing = [m for m in molecules if m not in phase1_best]
    if missing:
        print(f"WARNING: no Phase 1 results for: {missing}", file=sys.stderr)

    configs = _phase2_configs(molecules, splits, phase1_best)
    done = _load_done(csv_path, PHASE2_KEY_COLS)
    total = len(configs)
    n_done = sum(1 for c in configs if _row_key(c, PHASE2_KEY_COLS) in done)

    print(f"\n{'=' * 78}")
    print(
        f"  Phase 2 — Three-body sweep (best two-body from Phase 1){'' if forces else '  [energy-only]'}"
    )
    print(f"  molecules={molecules}  splits={splits}")
    print(f"  n_train={n_train}  n_test={n_test}")
    print(f"  Output: {csv_path}")
    print(f"  Total: {total}  done: {n_done}  remaining: {total - n_done}")
    print(f"{'=' * 78}\n")

    # Print the Phase 1 best params being used
    print("  Phase 1 best params (fixed for Phase 2):")
    for mol in molecules:
        if mol in phase1_best:
            pb = phase1_best[mol]
            print(
                f"    {mol:14s} tb={pb['two_body_type']:20s}"
                f" η2={pb['eta2']:.2f} d={pb['two_body_decay']:.1f}"
                f" σ={pb['sigma']:.1f}  (E-MAE={pb['mean_e']:.3f})"
            )
    print()

    run_idx = 0
    for cfg in configs:
        key = _row_key(cfg, PHASE2_KEY_COLS)
        if key in done:
            continue

        run_idx += 1
        atm_s = "atm" if cfg["use_atm"] else "noatm"
        label = (
            f"{cfg['molecule']:14s} | {cfg['three_body_type']:32s}"
            f" {atm_s:6s} σ={cfg['sigma']:<5.1f} sp={cfg['split']}"
        )
        print(f"[{n_done + run_idx:5d}/{total}] {label}", end="", flush=True)

        if dry_run:
            print("  [DRY-RUN]")
            continue

        nrs3_minus = 20 if cfg["three_body_type"] in THREE_BODY_NEEDS_RS3_MINUS else 0
        repr_params: dict[str, Any] = {
            "two_body_type": cfg["two_body_type"],
            "eta2": cfg["eta2"],
            "two_body_decay": cfg["two_body_decay"],
            "three_body_type": cfg["three_body_type"],
            "use_atm": cfg["use_atm"],
        }
        if nrs3_minus > 0:
            repr_params["nRs3_minus"] = nrs3_minus

        coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te = load_rmd17(
            cfg["molecule"], cfg["split"], n_train, n_test
        )
        model = _build_model(
            regressor="krr",
            representation="fchl19v2",
            sigma=cfg["sigma"],
            l2=1e-8,
            elements=None,
            max_size=None,
            d_rff=1024,
            seed=42,
            z_tr=z_tr,
            z_te=z_te,
            repr_params=repr_params,
        )

        t0 = time.perf_counter()
        try:
            if forces:
                model.fit(coords_tr, z_tr, energies=E_tr, forces=F_tr)
                E_pred, F_pred = model.predict(coords_te, z_te)
                e_mae = _compute_mae(E_te, E_pred)
                F_ref = np.concatenate([f.ravel() for f in F_te])
                f_mae = _compute_mae(F_ref, F_pred)
            else:
                model.fit(coords_tr, z_tr, energies=E_tr)
                E_pred, _ = model.predict(coords_te, z_te)
                e_mae = _compute_mae(E_te, E_pred)
                f_mae = float("nan")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        elapsed = time.perf_counter() - t0

        row: dict[str, Any] = {
            "molecule": cfg["molecule"],
            "split": cfg["split"],
            "two_body_type": cfg["two_body_type"],
            "eta2": cfg["eta2"],
            "two_body_decay": cfg["two_body_decay"],
            "three_body_type": cfg["three_body_type"],
            "use_atm": cfg["use_atm"],
            "nRs3_minus": nrs3_minus,
            "sigma": cfg["sigma"],
            "n_train": n_train,
            "n_test": n_test,
            "energy_mae": f"{e_mae:.6f}",
            "force_mae": f"{f_mae:.6f}" if forces else "nan",
            "elapsed_s": f"{elapsed:.1f}",
        }
        _append_row(csv_path, PHASE2_FIELDNAMES, row)
        done.add(key)

        b_e = BASELINE_E50.get(cfg["molecule"], float("nan"))
        de = e_mae - b_e
        fe = "▼" if de < 0 else ("~" if abs(de) < 0.05 * b_e else "▲")
        if forces:
            b_f = BASELINE_F50.get(cfg["molecule"], float("nan"))
            df = f_mae - b_f
            ff = "▼" if df < 0 else ("~" if abs(df) < 0.05 * b_f else "▲")
            print(
                f"  E={e_mae:.3f}({fe}{abs(de):.3f})"
                f"  F={f_mae:.3f}({ff}{abs(df):.3f})  {elapsed:.1f}s"
            )
        else:
            print(f"  E={e_mae:.3f}({fe}{abs(de):.3f})  {elapsed:.1f}s")

    print(f"\n{'=' * 78}")
    print(f"  Phase 2 complete.  Results: {csv_path}")
    print(f"{'=' * 78}\n")

    if not dry_run and csv_path.exists():
        _print_phase2_summary(csv_path)


def _print_phase2_summary(csv_path: Path) -> None:
    from collections import defaultdict

    e_acc: dict[tuple, list[float]] = defaultdict(list)
    f_acc: dict[tuple, list[float]] = defaultdict(list)

    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            k = (
                row["molecule"],
                row["two_body_type"],
                float(row["eta2"]),
                float(row["two_body_decay"]),
                row["three_body_type"],
                row["use_atm"] == "True",
                float(row["sigma"]),
            )
            e_acc[k].append(float(row["energy_mae"]))
            f_acc[k].append(float(row["force_mae"]))

    best: dict[str, dict[str, Any]] = {}
    for k, e_list in e_acc.items():
        mol, tb, eta2, decay, ab, use_atm, sigma = k
        mean_e = float(np.mean(e_list))
        mean_f = float(np.mean(f_acc[k]))
        if mol not in best or mean_e < best[mol]["mean_e"]:
            best[mol] = {
                "two_body_type": tb,
                "eta2": eta2,
                "decay": decay,
                "three_body_type": ab,
                "use_atm": use_atm,
                "sigma": sigma,
                "mean_e": mean_e,
                "mean_f": mean_f,
                "n_splits": len(e_list),
            }

    hdr = (
        f"{'Molecule':14s} {'TB':18s} {'AB':32s} {'ATM':5s}"
        f" {'η2':>5s} {'dec':>4s} {'σ':>5s}"
        f"  {'E-MAE':>7s} {'Δbase':>7s}   {'F-MAE':>7s} {'Δbase':>7s}"
    )
    sep = "-" * len(hdr)
    print(f"\n{'=' * len(hdr)}")
    print("  Phase 2 — Best params per molecule (mean over available splits)")
    print(f"{'=' * len(hdr)}")
    print(f"  {hdr}")
    print(f"  {sep}")

    for mol in ALL_MOLECULES:
        if mol not in best:
            continue
        b = best[mol]
        b_e = BASELINE_E50.get(mol, float("nan"))
        b_f = BASELINE_F50.get(mol, float("nan"))
        de = b["mean_e"] - b_e
        df = b["mean_f"] - b_f
        fe = "▼" if de < 0 else "▲"
        ff = "▼" if df < 0 else "▲"
        atm_s = "yes" if b["use_atm"] else "no"
        print(
            f"  {mol:14s} {b['two_body_type']:18s} {b['three_body_type']:32s} {atm_s:5s}"
            f" {b['eta2']:>5.2f} {b['decay']:>4.1f} {b['sigma']:>5.1f}"
            f"  {b['mean_e']:7.3f} {fe}{abs(de):.3f}   "
            f"{b['mean_f']:7.3f} {ff}{abs(df):.3f}"
        )

    print(f"  {sep}")
    print(f"  Baseline: FCHL19 published E-MAE / F-MAE at N=50 (kcal/mol)")
    print(f"{'=' * len(hdr)}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Two-phase hyperparameter sweep for fchl19v2 on rMD17.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        required=True,
        help="Which phase: 1=two-body sweep, 2=three-body sweep.",
    )

    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: ethanol + benzene, split=1 only (~2400 runs).",
    )
    mode_grp.add_argument(
        "--full",
        action="store_true",
        help="Full mode: all 10 molecules, splits 1-3.",
    )
    p.add_argument(
        "--molecules",
        nargs="+",
        default=None,
        metavar="MOL",
        help="Override molecule list (e.g. --molecules ethanol aspirin).",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Override split list (e.g. --splits 1 2 3).",
    )
    p.add_argument("--n-train", type=int, default=50)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--dry-run", action="store_true", help="Print configs without running.")
    p.add_argument(
        "--summary", action="store_true", help="Re-print summary from existing CSV and exit."
    )
    p.add_argument(
        "--fixed-two-body-type",
        default=None,
        metavar="TYPE",
        help=(
            "Phase 2 only: restrict Phase 1 best-param lookup to this two_body_type "
            "(e.g. 'bessel'). Useful when you want to force a specific two-body basis "
            "into Phase 2 regardless of which type won overall."
        ),
    )
    p.add_argument(
        "--no-forces",
        action="store_true",
        help=(
            "Train on energies only — skip Jacobian computation entirely. "
            "Much faster for large representations (e.g. element_resolved). "
            "force_mae is written as 'nan' in the CSV."
        ),
    )

    args = p.parse_args()

    # Resolve molecule + split lists
    if args.molecules:
        molecules = args.molecules
        splits = args.splits or [1]
    elif args.full:
        molecules = ALL_MOLECULES
        splits = args.splits or [1, 2, 3]
    else:
        # default = pilot
        molecules = PILOT_MOLECULES
        splits = args.splits or [1]

    if args.summary:
        csv_path = RESULTS_DIR / f"benchmark_phase{args.phase}.csv"
        if not csv_path.exists():
            print(f"No results at {csv_path}", file=sys.stderr)
            sys.exit(1)
        if args.phase == 1:
            _print_phase1_summary(csv_path)
        else:
            _print_phase2_summary(csv_path)
        return

    if args.phase == 1:
        run_phase1(
            molecules, splits, args.n_train, args.n_test, args.dry_run, forces=not args.no_forces
        )
    else:
        run_phase2(
            molecules,
            splits,
            args.n_train,
            args.n_test,
            args.dry_run,
            fixed_two_body_type=args.fixed_two_body_type,
            forces=not args.no_forces,
        )


if __name__ == "__main__":
    main()
