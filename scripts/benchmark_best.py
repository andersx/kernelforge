"""benchmark_best.py — Full E+F KRR benchmark using Optuna-tuned hyperparameters.

Runs KRR with energy_and_force training using the best hyperparameters found
by optuna_sweep.py for each dataset.

Datasets and settings:
  rmd17/ethanol    : n_train=1000, n_test=1000, splits 1-3
  rmd17/benzene    : n_train=1000, n_test=1000, splits 1-3
  small_mols_mini  : n_train=1000, n_test=595  (full test set), single run

Best params (from Optuna energy-only sweep, N=100 train):
  ethanol:         sigma=15.93, nFourier=1,  nRs3=28, nRs3_minus=26,
                   eta3=8.415, eta3_minus=1.886, decay=1.924, weight=91.35
  benzene:         sigma=39.10, nFourier=4,  nRs3=26, nRs3_minus=8,
                   eta3=0.582, eta3_minus=8.779, decay=0.291, weight=66.93
  small_mols_mini: sigma=11.68, nFourier=3,  nRs3=24, nRs3_minus=24,
                   eta3=7.787, eta3_minus=1.890, decay=1.552, weight=99.91

Usage
-----
    uv run scripts/benchmark_best.py
    uv run scripts/benchmark_best.py --smoke   # 1 split, n_train=50, n_test=50 (fast check)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path.home() / ".kernelforge" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "benchmark_best.csv"

CSV_FIELDNAMES = [
    "dataset",
    "molecule",
    "split",
    "n_train",
    "n_test",
    "E_mae",
    "F_mae",
    "elapsed_s",
]

# ---------------------------------------------------------------------------
# Best hyperparameters from Optuna sweeps
# ---------------------------------------------------------------------------

FIXED_REPR_BASE: dict[str, str | float | bool] = {
    "two_body_type": "bessel",
    "eta2": 0.32,
    "two_body_decay": 0.5,
    "three_body_type": "odd_fourier_element_resolved",
    "use_atm": True,
    "use_three_body": True,
}

BEST_PARAMS: dict[str, dict[str, str | float | bool | int]] = {
    "ethanol": {
        **FIXED_REPR_BASE,
        "sigma": 15.932714,
        "nFourier": 1,
        "nRs3": 28,
        "nRs3_minus": 26,
        "eta3": 8.415388,
        "eta3_minus": 1.885646,
        "three_body_decay": 1.923561,
        "three_body_weight": 91.349957,
    },
    "benzene": {
        **FIXED_REPR_BASE,
        "sigma": 39.095678,
        "nFourier": 4,
        "nRs3": 26,
        "nRs3_minus": 8,
        "eta3": 0.582077,
        "eta3_minus": 8.778917,
        "three_body_decay": 0.290634,
        "three_body_weight": 66.934491,
    },
    "small_mols_mini": {
        **FIXED_REPR_BASE,
        "sigma": 11.683908,
        "nFourier": 3,
        "nRs3": 24,
        "nRs3_minus": 24,
        "eta3": 7.787293,
        "eta3_minus": 1.889792,
        "three_body_decay": 1.552211,
        "three_body_weight": 99.911535,
    },
}

# Published FCHL19 baselines at N=50 (E+F training)
BASELINE_E: dict[str, float] = {
    "ethanol": 0.232,
    "benzene": 0.015,
}
BASELINE_F: dict[str, float] = {
    "ethanol": 1.072,
    "benzene": 0.125,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.ravel() - b.ravel())))


def _write_row(row: dict[str, str | int | float]) -> None:
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    with OUTPUT_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_one(
    dataset: str,
    molecule: str,
    split: int,
    n_train: int,
    n_test: int,
    build_model: object,
    load_rmd17: object,
    load_small_mols_mini: object,
) -> None:
    params = BEST_PARAMS[molecule]
    sigma = float(params["sigma"])
    repr_params = {k: v for k, v in params.items() if k != "sigma"}

    print(
        f"  {dataset}/{molecule}  split={split}  n_train={n_train}  n_test={n_test}",
        flush=True,
    )

    # Load data
    if dataset == "rmd17":
        coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te = load_rmd17(  # type: ignore[call-arg]
            molecule, split, n_train, n_test
        )
    else:
        coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te = load_small_mols_mini(  # type: ignore[call-arg]
            n_train, n_test
        )

    t0 = time.perf_counter()

    model = build_model(  # type: ignore[call-arg]
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
    model.fit(coords_tr, z_tr, energies=E_tr, forces=F_tr)  # type: ignore[union-attr]
    E_pred, F_pred = model.predict(coords_te, z_te)  # type: ignore[union-attr]

    elapsed = time.perf_counter() - t0

    # Forces from model are stacked (n_test * n_atoms, 3) — flatten both for MAE
    F_te_arr = np.concatenate([f.ravel() for f in F_te])
    F_pred_arr = F_pred.ravel()

    e_mae = _mae(E_te, E_pred)
    f_mae = _mae(F_te_arr, F_pred_arr)

    b_e = BASELINE_E.get(molecule, float("nan"))
    b_f = BASELINE_F.get(molecule, float("nan"))
    de = e_mae - b_e
    df = f_mae - b_f
    fe = "▼" if de < 0 else "▲"
    ff = "▼" if df < 0 else "▲"

    print(
        f"    E-MAE={e_mae:.4f} ({fe}{abs(de):.4f} vs baseline={b_e:.4f})"
        f"  F-MAE={f_mae:.4f} ({ff}{abs(df):.4f} vs baseline={b_f:.4f})"
        f"  {elapsed:.1f}s",
        flush=True,
    )

    _write_row(
        {
            "dataset": dataset,
            "molecule": molecule,
            "split": split,
            "n_train": n_train,
            "n_test": n_test,
            "E_mae": f"{e_mae:.6f}",
            "F_mae": f"{f_mae:.6f}",
            "elapsed_s": f"{elapsed:.1f}",
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Full E+F KRR benchmark with Optuna-tuned hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: n_train=50, n_test=50, 1 split per molecule.",
    )
    args = p.parse_args()

    from kernelforge.kernelcli import _build_model as build_model  # type: ignore[attr-defined]
    from kernelforge.kernelcli import (
        load_rmd17,  # type: ignore[attr-defined]
        load_small_mols_mini,  # type: ignore[attr-defined]
    )

    if args.smoke:
        n_train_rmd17 = 50
        n_test_rmd17 = 50
        n_train_small = 50
        n_test_small = 50
        splits = [1]
    else:
        n_train_rmd17 = 50
        n_test_rmd17 = 100
        n_train_small = 50
        n_test_small = 100
        splits = [1, 2, 3]

    jobs = [
        ("rmd17", "ethanol", splits),
        ("rmd17", "benzene", splits),
        ("small_mols_mini", "small_mols_mini", [0]),
    ]

    print(f"\n{'=' * 72}")
    print("  Full E+F KRR benchmark — fchl19v2 odd_fourier_element_resolved")
    print(f"  rMD17: n_train={n_train_rmd17}, n_test={n_test_rmd17}, splits={splits}")
    print(f"  small_mols_mini: n_train={n_train_small}, n_test={n_test_small}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"{'=' * 72}\n")

    for dataset, molecule, job_splits in jobs:
        n_train = n_train_rmd17 if dataset == "rmd17" else n_train_small
        n_test = n_test_rmd17 if dataset == "rmd17" else n_test_small
        for split in job_splits:
            _run_one(
                dataset=dataset,
                molecule=molecule,
                split=split,
                n_train=n_train,
                n_test=n_test,
                build_model=build_model,
                load_rmd17=load_rmd17,
                load_small_mols_mini=load_small_mols_mini,
            )

    # Print summary
    print(f"\n{'=' * 72}")
    print("  Summary")
    print(f"{'=' * 72}")
    with OUTPUT_CSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Group by dataset/molecule, average over splits
    from collections import defaultdict

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = f"{row['dataset']}/{row['molecule']}"
        groups[key].append(row)

    print(
        f"  {'Dataset/Molecule':30s} {'splits':>6s}"
        f"  {'E-MAE':>8s}  {'F-MAE':>8s}  vs FCHL19 baseline"
    )
    print(f"  {'-' * 70}")
    for key, group_rows in groups.items():
        e_vals = [float(r["E_mae"]) for r in group_rows]
        f_vals = [float(r["F_mae"]) for r in group_rows]
        e_mean = float(np.mean(e_vals))
        f_mean = float(np.mean(f_vals))
        mol = group_rows[0]["molecule"]
        b_e = BASELINE_E.get(mol, float("nan"))
        b_f = BASELINE_F.get(mol, float("nan"))
        de = e_mean - b_e
        df = f_mean - b_f
        fe = "▼" if de < 0 else "▲"
        ff = "▼" if df < 0 else "▲"
        print(
            f"  {key:30s} {len(group_rows):6d}"
            f"  {e_mean:8.4f}  {f_mean:8.4f}"
            f"  E:{fe}{abs(de):.4f}  F:{ff}{abs(df):.4f}"
        )
    print(f"  {'-' * 70}")
    print("  FCHL19 baselines (N=50, E+F): ethanol E=0.232 F=1.072 | benzene E=0.015 F=0.125")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
