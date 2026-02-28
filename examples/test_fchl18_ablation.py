"""FCHL18 ablation study: accuracy vs. speed for kernel term variants.

Runs KRR on QM7b (same setup as test_fchl18_integration.py) for each variant,
then prints a summary table of kernel time and MAE.

Variants
--------
full              : Full FCHL18 (baseline, all terms enabled)
two_body_only     : three_body_scaling=0   (no angular term)
three_body_only   : two_body_scaling=0     (no radial distance term)
fourier_order_1   : fourier_order=1        (half the Fourier terms)
fourier_order_0   : fourier_order=0        (speed baseline, effectively two_body_only)
no_atm            : use_atm=False          (ATM factor replaced with 1.0)
fo1_no_atm        : fourier_order=1 + use_atm=False  (best combined)

Run with:
    pytest -m integration tests/test_fchl18_ablation.py -s
or directly:
    python tests/test_fchl18_ablation.py
"""

import time

import numpy as np
import pytest

import kernelforge.fchl18_kernel as kernel_mod
import kernelforge.fchl18_repr as repr_mod
from kernelforge.cli import load_qm7b_raw_data

# Default hyperparameters (same as integration test and old_code reference)
BASE_ARGS = dict(
    two_body_width=0.1,
    two_body_scaling=2.0,
    two_body_power=6.0,
    three_body_width=3.0,
    three_body_scaling=2.0,
    three_body_power=3.0,
    cut_start=0.5,
    cut_distance=1e6,
    fourier_order=2,
    use_atm=True,
)

# Best configuration found in ablation study (fo1_no_atm)
BEST_ARGS = dict(
    two_body_width=0.1,
    two_body_scaling=2.0,
    two_body_power=6.0,
    three_body_width=3.0,
    three_body_scaling=2.0,
    three_body_power=3.0,
    cut_start=0.5,
    cut_distance=1e6,
    fourier_order=1,
    use_atm=False,
)

# Round 1 starting point: best individual values from hyperparam scan applied together.
# two_body_width 0.1->0.2  (plateau 0.2-0.3 in scan)
# two_body_scaling 2.0->3.0, three_body_scaling 2.0->1.0  (scaling_ratio winner 3/1)
# two_body_power 6.0->4.0  (scan best, still descending)
# three_body_power 3.0->4.0  (scan best, still descending)
# sigma 2.5->2.0  (scan best)
ROUND1_ARGS = dict(
    two_body_width=0.2,
    two_body_scaling=3.0,
    two_body_power=4.0,
    three_body_width=3.0,
    three_body_scaling=1.0,
    three_body_power=4.0,
    cut_start=0.5,
    cut_distance=1e6,
    fourier_order=1,
    use_atm=False,
)
ROUND1_SIGMA = 2.0

# KRR settings (match integration test)
N_POINTS = 1500
MAX_SIZE = 23
SIGMA = 2.5
LLAMBDA = 1e-8


def _run_krr(
    x, n, nn, kernel_args: dict, sigma: float = SIGMA, llambda: float = LLAMBDA
) -> tuple[float, float]:
    """Run KRR train/predict for one set of kernel args.

    Returns
    -------
    kernel_time : float
        Wall-clock seconds for the symmetric train kernel only.
    mae : float
        Mean absolute error (kcal/mol) on the test set.
    """
    data = load_qm7b_raw_data()
    E_all = data["E"][:N_POINTS]
    properties = np.array(E_all, dtype=np.float64)

    # Shuffle (fixed seed matching integration test)
    rng = np.random.default_rng(666)
    perm = rng.permutation(N_POINTS)
    xp = x[perm]
    np_ = n[perm]
    nnp = nn[perm]
    yp = properties[perm]

    n_test = N_POINTS // 3
    n_train = N_POINTS - n_test

    x_tr, n_tr, nn_tr, y_tr = xp[:n_train], np_[:n_train], nnp[:n_train], yp[:n_train]
    x_te, n_te, nn_te, y_te = xp[n_train:], np_[n_train:], nnp[n_train:], yp[n_train:]

    # Symmetric train kernel (timed)
    t0 = time.perf_counter()
    K_sym = kernel_mod.kernel_gaussian_symm(x_tr, n_tr, nn_tr, sigma=sigma, **kernel_args)
    kernel_time = time.perf_counter() - t0

    # Solve KRR
    K_reg = K_sym.copy()
    K_reg[np.diag_indices_from(K_reg)] += llambda
    alpha = np.linalg.solve(K_reg, y_tr)

    # Predict on test set
    K_te = kernel_mod.kernel_gaussian(
        x_te, x_tr, n_te, n_tr, nn_te, nn_tr, sigma=sigma, **kernel_args
    )
    y_pred = K_te @ alpha
    mae = float(np.mean(np.abs(y_te - y_pred)))

    return kernel_time, mae


def _load_repr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load QM7b and generate representations (shared across all variants)."""
    data = load_qm7b_raw_data()
    coords_list = list(data["R"][:N_POINTS])
    z_list = [zi.astype(np.int32) for zi in data["z"][:N_POINTS]]

    x, n, nn = repr_mod.generate(
        coords_list,
        z_list,
        max_size=MAX_SIZE,
        cut_distance=BASE_ARGS["cut_distance"],
    )
    return x, n, nn


# ---------------------------------------------------------------------------
# Ablation variants: (label, description, kernel_args_overrides)
# ---------------------------------------------------------------------------
VARIANTS = [
    (
        "full",
        "Full FCHL18 (baseline)",
        {},
    ),
    (
        "two_body_only",
        "two_body only  (three_body_scaling=0)",
        {"three_body_scaling": 0.0},
    ),
    (
        "three_body_only",
        "three_body only (two_body_scaling=0)",
        {"two_body_scaling": 0.0},
    ),
    (
        "fourier_order_1",
        "Fourier order=1 (half angular terms)",
        {"fourier_order": 1},
    ),
    (
        "fourier_order_0",
        "Fourier order=0 (speed baseline)",
        {"fourier_order": 0},
    ),
    (
        "no_atm",
        "No ATM factor  (use_atm=False)",
        {"use_atm": False},
    ),
    (
        "fo1_no_atm",
        "Fourier order=1 + no ATM",
        {"fourier_order": 1, "use_atm": False},
    ),
]


@pytest.mark.slow
@pytest.mark.integration
def test_fchl18_ablation():
    """Ablation study: MAE and kernel time for each FCHL18 variant on QM7b."""

    print("\n" + "=" * 70)
    print("FCHL18 ablation study")
    print(
        f"  Dataset : QM7b, N={N_POINTS} (train={N_POINTS - N_POINTS // 3}, test={N_POINTS // 3})"
    )
    print(f"  sigma={SIGMA}, lambda={LLAMBDA:.0e}")
    print("=" * 70)

    # Generate representations once — shared across all variants
    print("Generating representations...")
    x, n, nn = _load_repr()

    results = []

    for label, description, overrides in VARIANTS:
        args = {**BASE_ARGS, **overrides}
        print(f"\n  Running variant: {description} ...", flush=True)
        kernel_time, mae = _run_krr(x, n, nn, args)
        results.append((label, description, kernel_time, mae))
        print(f"    kernel_time={kernel_time:.2f}s  MAE={mae:.3f} kcal/mol")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Variant':<20} {'Description':<38} {'K-time (s)':>10} {'MAE (kcal/mol)':>15}")
    print("-" * 70)
    for label, description, kernel_time, mae in results:
        print(f"{label:<20} {description:<38} {kernel_time:>10.2f} {mae:>15.3f}")
    print("=" * 70)

    # The full baseline must stay within expected accuracy range
    full_mae = next(mae for label, _, _, mae in results if label == "full")
    assert abs(full_mae - 2.0) < 2.0, (
        f"Full FCHL18 MAE={full_mae:.3f} kcal/mol — expected ~2 kcal/mol"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_fchl18_hyperparam_scan():
    """Independent 1-D hyperparameter scans around the fo1_no_atm best config.

    Each parameter is swept independently while all others are held at BEST_ARGS.
    sigma and llambda are swept as KRR hyperparameters (not kernel structure args).
    """

    print("\n" + "=" * 70)
    print("FCHL18 hyperparameter scan (fo1_no_atm base config)")
    print(
        f"  Dataset : QM7b, N={N_POINTS} (train={N_POINTS - N_POINTS // 3}, test={N_POINTS // 3})"
    )
    print("=" * 70)

    print("Generating representations...")
    x, n, nn = _load_repr()

    # ------------------------------------------------------------------
    # Each scan is: (title, param_name, values, is_krr_param)
    # is_krr_param=True  -> passed as sigma= or llambda= to _run_krr
    # is_krr_param=False -> overrides key in kernel_args dict
    # ------------------------------------------------------------------
    scans = [
        ("sigma", "sigma", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], True),
        ("llambda", "llambda", [1e-10, 1e-9, 1e-8, 1e-7, 1e-6], True),
        ("two_body_scaling", "two_body_scaling", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0], False),
        ("three_body_scaling", "three_body_scaling", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0], False),
        ("scaling_ratio", None, None, False),  # special
        ("two_body_width", "two_body_width", [0.05, 0.1, 0.15, 0.2, 0.3, 0.5], False),
        ("two_body_power", "two_body_power", [3.0, 4.0, 5.0, 6.0, 7.0, 8.0], False),
        ("three_body_power", "three_body_power", [2.0, 2.5, 3.0, 3.5, 4.0], False),
    ]

    # Special: scaling ratio sweep — keeps two_body_scaling + three_body_scaling = 4
    scaling_pairs = [
        (4.0, 0.0),
        (3.0, 1.0),
        (2.5, 1.5),
        (2.0, 2.0),
        (1.5, 2.5),
        (1.0, 3.0),
        (0.0, 4.0),
    ]

    best_overall_mae = float("inf")
    best_overall_cfg = {}

    for scan_name, param_key, values, is_krr in scans:
        if scan_name == "scaling_ratio":
            pairs = scaling_pairs
            display = [f"2b={a:.1f}/3b={b:.1f}" for a, b in pairs]
        else:
            pairs = [(v,) for v in values]
            display = [str(v) for v in values]

        print(f"\n  --- {scan_name} ---")
        print(f"  {'value':<20} {'MAE (kcal/mol)':>15}")
        print(f"  {'-' * 36}")

        best_scan_mae = float("inf")
        best_scan_val = None

        for pair, disp in zip(pairs, display):
            # Build args for this point
            kernel_args = dict(BEST_ARGS)
            run_sigma = SIGMA
            run_llambda = LLAMBDA

            if scan_name == "scaling_ratio":
                assert len(pair) == 2
                kernel_args["two_body_scaling"] = pair[0]
                kernel_args["three_body_scaling"] = pair[1]
            elif is_krr:
                if param_key == "sigma":
                    run_sigma = pair[0]
                else:
                    run_llambda = pair[0]
            else:
                kernel_args[param_key] = pair[0]

            _, mae = _run_krr(x, n, nn, kernel_args, sigma=run_sigma, llambda=run_llambda)
            marker = " <-- best" if mae < best_scan_mae else ""
            print(f"  {disp:<20} {mae:>15.3f}{marker}")

            if mae < best_scan_mae:
                best_scan_mae = mae
                best_scan_val = (scan_name, pair, disp)

            if mae < best_overall_mae:
                best_overall_mae = mae
                best_overall_cfg = {
                    "kernel_args": dict(kernel_args),
                    "sigma": run_sigma,
                    "llambda": run_llambda,
                    "scan": scan_name,
                    "value": disp,
                }

        if best_scan_val is not None:
            print(f"  --> best {scan_name}: {best_scan_val[2]}  MAE={best_scan_mae:.3f} kcal/mol")

    print("\n" + "=" * 70)
    print(f"Best overall MAE={best_overall_mae:.3f} kcal/mol")
    print(f"  Found in scan: {best_overall_cfg['scan']} = {best_overall_cfg['value']}")
    print(f"  sigma={best_overall_cfg['sigma']}, llambda={best_overall_cfg['llambda']}")
    print(f"  kernel_args={best_overall_cfg['kernel_args']}")
    print("=" * 70)


@pytest.mark.slow
@pytest.mark.integration
def test_fchl18_round1_scan():
    """Round 1: extended 1-D scans from the improved ROUND1_ARGS starting point.

    Focuses on the three parameters that were still improving at the edge of the
    previous scan:
      - two_body_power   (was descending at 4.0; extend range [2.0 .. 5.0])
      - three_body_power (was descending at 4.0; extend range [3.5 .. 6.0])
      - three_body_width (never scanned; first scan [1.0 .. 5.0])

    All other parameters are held at ROUND1_ARGS / ROUND1_SIGMA.
    A combined-baseline run is printed first so the gain from the new starting
    point is immediately visible.
    """

    print("\n" + "=" * 70)
    print("FCHL18 Round 1 scan  (extended ranges, improved starting point)")
    print(
        f"  Dataset : QM7b, N={N_POINTS} (train={N_POINTS - N_POINTS // 3}, test={N_POINTS // 3})"
    )
    print(f"  Base: ROUND1_ARGS, sigma={ROUND1_SIGMA}, lambda={LLAMBDA:.0e}")
    print("=" * 70)

    print("\nGenerating representations...")
    x, n, nn = _load_repr()

    # ------------------------------------------------------------------
    # Baseline: combined ROUND1_ARGS (all individually-best params together)
    # ------------------------------------------------------------------
    print("\n  --- baseline (ROUND1_ARGS combined) ---")
    _, baseline_mae = _run_krr(x, n, nn, dict(ROUND1_ARGS), sigma=ROUND1_SIGMA)
    print(f"  ROUND1_ARGS + sigma={ROUND1_SIGMA}  ->  MAE={baseline_mae:.3f} kcal/mol")
    print(f"  (previous fo1_no_atm baseline was 0.680 kcal/mol)")

    # ------------------------------------------------------------------
    # Scans: (scan_name, param_key, values)
    # All vary one param in ROUND1_ARGS; ROUND1_SIGMA used throughout.
    # ------------------------------------------------------------------
    scans = [
        (
            "two_body_power",
            "two_body_power",
            [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        ),
        (
            "three_body_power",
            "three_body_power",
            [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        ),
        (
            "three_body_width",
            "three_body_width",
            [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        ),
    ]

    summary = []  # (scan_name, best_val, best_mae)

    for scan_name, param_key, values in scans:
        print(f"\n  --- {scan_name} ---")
        print(f"  {'value':<10} {'MAE (kcal/mol)':>15}")
        print(f"  {'-' * 26}")

        best_mae = float("inf")
        best_val = None

        for v in values:
            args = dict(ROUND1_ARGS)
            args[param_key] = v
            _, mae = _run_krr(x, n, nn, args, sigma=ROUND1_SIGMA)
            marker = " <-- best" if mae < best_mae else ""
            print(f"  {v:<10} {mae:>15.3f}{marker}", flush=True)
            if mae < best_mae:
                best_mae = mae
                best_val = v

        summary.append((scan_name, best_val, best_mae))
        print(f"  --> best {scan_name} = {best_val}  MAE={best_mae:.3f} kcal/mol")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Round 1 scan summary")
    print(f"  {'Parameter':<22} {'Best value':>12} {'MAE (kcal/mol)':>15}")
    print(f"  {'-' * 50}")
    print(f"  {'baseline (ROUND1_ARGS)':<22} {'':>12} {baseline_mae:>15.3f}")
    for scan_name, best_val, best_mae in summary:
        print(f"  {scan_name:<22} {str(best_val):>12} {best_mae:>15.3f}")
    print("=" * 70)


@pytest.mark.slow
@pytest.mark.integration
def test_fchl18_round2_grid():
    """Round 2: 2D grid over two_body_power x three_body_power.

    Starting from BEST_ARGS (fo1_no_atm, 0.680 kcal/mol baseline) with all other
    parameters held fixed. The 1D scans showed these two power exponents are the
    most sensitive and interact — a joint grid maps the true optimum.

    Grid:
      two_body_power   : [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
      three_body_power : [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    """
    two_body_powers = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    three_body_powers = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    n_runs = len(two_body_powers) * len(three_body_powers)

    print("\n" + "=" * 70)
    print("FCHL18 Round 2: 2D grid  two_body_power x three_body_power")
    print(
        f"  Dataset : QM7b, N={N_POINTS} (train={N_POINTS - N_POINTS // 3}, test={N_POINTS // 3})"
    )
    print(f"  Base: BEST_ARGS (fo1_no_atm), sigma={SIGMA}, lambda={LLAMBDA:.0e}")
    print(f"  Grid size: {len(two_body_powers)} x {len(three_body_powers)} = {n_runs} runs")
    print("=" * 70)

    print("\nGenerating representations...")
    x, n, nn = _load_repr()

    # ------------------------------------------------------------------
    # Run the grid
    # ------------------------------------------------------------------
    results = {}  # (tbp, threebp) -> mae

    # Print header
    header = f"  {'2b\\3b':>6}" + "".join(f"  {p:>6.1f}" for p in three_body_powers)
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    best_mae = float("inf")
    best_cfg = {}

    for tbp in two_body_powers:
        row_maes = []
        for threebp in three_body_powers:
            args = dict(BEST_ARGS)
            args["two_body_power"] = tbp
            args["three_body_power"] = threebp
            _, mae = _run_krr(x, n, nn, args, sigma=SIGMA)
            results[(tbp, threebp)] = mae
            row_maes.append(mae)
            if mae < best_mae:
                best_mae = mae
                best_cfg = {"two_body_power": tbp, "three_body_power": threebp, "mae": mae}

        # Print row immediately so progress is visible
        row_str = f"  {tbp:>6.1f}" + "".join(f"  {m:>6.3f}" for m in row_maes)
        print(row_str, flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Best MAE = {best_cfg['mae']:.3f} kcal/mol")
    print(f"  two_body_power   = {best_cfg['two_body_power']}")
    print(f"  three_body_power = {best_cfg['three_body_power']}")
    print(f"  (previous best: fo1_no_atm defaults = 0.680 kcal/mol)")
    print("=" * 70)


@pytest.mark.slow
@pytest.mark.integration
def test_fchl18_round3_scan():
    """Round 3: re-scan sigma, scaling ratio, and two_body_width at the Round 2 optimum.

    Starting point: BEST_ARGS + two_body_power=4.5, three_body_power=3.0
    (MAE=0.642 kcal/mol).

    Scans:
      sigma             : [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
      scaling ratio     : two_body_scaling + three_body_scaling = 4, sweep ratio
      two_body_width    : [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
      llambda           : [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    """

    # New starting point from Round 2
    ROUND2_ARGS = dict(BEST_ARGS)
    ROUND2_ARGS["two_body_power"] = 4.5
    ROUND2_ARGS["three_body_power"] = 3.0

    print("\n" + "=" * 70)
    print("FCHL18 Round 3 scan  (re-scan at Round 2 optimum powers)")
    print(
        f"  Dataset : QM7b, N={N_POINTS} (train={N_POINTS - N_POINTS // 3}, test={N_POINTS // 3})"
    )
    print(
        f"  Base: fo1_no_atm + two_body_power=4.5, three_body_power=3.0,"
        f" sigma={SIGMA}, lambda={LLAMBDA:.0e}"
    )
    print("=" * 70)

    print("\nGenerating representations...")
    x, n, nn = _load_repr()

    scans = [
        ("sigma", "sigma", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], True),
        ("llambda", "llambda", [1e-10, 1e-9, 1e-8, 1e-7, 1e-6], True),
        ("scaling_ratio", None, None, False),
        ("two_body_width", "two_body_width", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5], False),
    ]

    scaling_pairs = [
        (4.0, 0.0),
        (3.5, 0.5),
        (3.0, 1.0),
        (2.5, 1.5),
        (2.0, 2.0),
        (1.5, 2.5),
        (1.0, 3.0),
        (0.5, 3.5),
        (0.0, 4.0),
    ]

    summary = []
    best_overall_mae = float("inf")
    best_overall_cfg: dict = {}

    for scan_name, param_key, values, is_krr in scans:
        if scan_name == "scaling_ratio":
            pairs = scaling_pairs
            display = [f"2b={a:.1f}/3b={b:.1f}" for a, b in pairs]
        else:
            pairs = [(v,) for v in values]
            display = [str(v) for v in values]

        print(f"\n  --- {scan_name} ---")
        print(f"  {'value':<22} {'MAE (kcal/mol)':>15}")
        print(f"  {'-' * 38}")

        best_scan_mae = float("inf")
        best_scan_val = None

        for pair, disp in zip(pairs, display):
            args = dict(ROUND2_ARGS)
            run_sigma = SIGMA
            run_llambda = LLAMBDA

            if scan_name == "scaling_ratio":
                assert len(pair) == 2
                args["two_body_scaling"] = pair[0]
                args["three_body_scaling"] = pair[1]
            elif is_krr:
                if param_key == "sigma":
                    run_sigma = pair[0]
                else:
                    run_llambda = pair[0]
            else:
                args[param_key] = pair[0]

            _, mae = _run_krr(x, n, nn, args, sigma=run_sigma, llambda=run_llambda)
            marker = " <-- best" if mae < best_scan_mae else ""
            print(f"  {disp:<22} {mae:>15.3f}{marker}", flush=True)

            if mae < best_scan_mae:
                best_scan_mae = mae
                best_scan_val = disp

            if mae < best_overall_mae:
                best_overall_mae = mae
                best_overall_cfg = {
                    "scan": scan_name,
                    "value": disp,
                    "sigma": run_sigma,
                    "llambda": run_llambda,
                    "kernel_args": dict(args),
                }

        summary.append((scan_name, best_scan_val, best_scan_mae))
        print(f"  --> best {scan_name}: {best_scan_val}  MAE={best_scan_mae:.3f} kcal/mol")

    print("\n" + "=" * 70)
    print("Round 3 scan summary  (base MAE = 0.642 kcal/mol)")
    print(f"  {'Parameter':<22} {'Best value':>22} {'MAE (kcal/mol)':>15}")
    print(f"  {'-' * 60}")
    for scan_name, best_val, best_mae in summary:
        print(f"  {scan_name:<22} {str(best_val):>22} {best_mae:>15.3f}")
    print(f"\nBest overall: MAE={best_overall_mae:.3f} kcal/mol")
    print(f"  scan={best_overall_cfg['scan']}, value={best_overall_cfg['value']}")
    print(f"  sigma={best_overall_cfg['sigma']}, llambda={best_overall_cfg['llambda']}")
    print(f"  kernel_args={best_overall_cfg['kernel_args']}")
    print("=" * 70)


@pytest.mark.slow
@pytest.mark.integration
def test_fchl18_round4_grid():
    """Round 4: 2D grid over two_body_power x three_body_power at new scaling ratio.

    Starting point: Round 3 best config
      two_body_scaling=2.5, three_body_scaling=1.5,
      two_body_power=4.5, three_body_power=3.0,
      fo1_no_atm, sigma=2.5, llambda=1e-8
      MAE=0.638 kcal/mol

    The scaling ratio shift (2.0/2.0 -> 2.5/1.5) may move the power optimum,
    so we re-run the 2D grid centred around (4.5, 3.0) with finer resolution.
    """

    ROUND3_ARGS = dict(BEST_ARGS)
    ROUND3_ARGS["two_body_scaling"] = 2.5
    ROUND3_ARGS["three_body_scaling"] = 1.5
    # Keep two_body_power and three_body_power as the scan axes

    two_body_powers = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    three_body_powers = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    n_runs = len(two_body_powers) * len(three_body_powers)

    print("\n" + "=" * 70)
    print("FCHL18 Round 4: 2D grid  two_body_power x three_body_power")
    print(
        f"  Dataset : QM7b, N={N_POINTS} (train={N_POINTS - N_POINTS // 3}, test={N_POINTS // 3})"
    )
    print(
        f"  Base: fo1_no_atm + two_body_scaling=2.5, three_body_scaling=1.5,"
        f" sigma={SIGMA}, lambda={LLAMBDA:.0e}"
    )
    print(f"  Grid size: {len(two_body_powers)} x {len(three_body_powers)} = {n_runs} runs")
    print("=" * 70)

    print("\nGenerating representations...")
    x, n, nn = _load_repr()

    results = {}
    best_mae = float("inf")
    best_cfg: dict = {}

    # Header
    header = f"  {'2b\\3b':>6}" + "".join(f"  {p:>6.1f}" for p in three_body_powers)
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for tbp in two_body_powers:
        row_maes = []
        for threebp in three_body_powers:
            args = dict(ROUND3_ARGS)
            args["two_body_power"] = tbp
            args["three_body_power"] = threebp
            _, mae = _run_krr(x, n, nn, args, sigma=SIGMA)
            results[(tbp, threebp)] = mae
            row_maes.append(mae)
            if mae < best_mae:
                best_mae = mae
                best_cfg = {"two_body_power": tbp, "three_body_power": threebp, "mae": mae}

        row_str = f"  {tbp:>6.1f}" + "".join(f"  {m:>6.3f}" for m in row_maes)
        print(row_str, flush=True)

    print("\n" + "=" * 70)
    print(f"Best MAE = {best_cfg['mae']:.3f} kcal/mol")
    print(f"  two_body_power   = {best_cfg['two_body_power']}")
    print(f"  three_body_power = {best_cfg['three_body_power']}")
    print(f"  (previous best: Round 3 = 0.638 kcal/mol)")
    print("=" * 70)


@pytest.mark.slow
@pytest.mark.integration
def test_fchl18_multi_split():
    """Compare original defaults vs. tuned config over multiple random splits.

    Runs KRR with both configs on 8 different random seeds to get mean ± std MAE,
    ruling out lucky/unlucky single-split results.

    Configs compared:
      default  : original FCHL18 defaults, sigma=2.5
      tuned    : Round 4 best, sigma=2.5
    """

    TUNED_ARGS = dict(
        two_body_width=0.1,
        two_body_scaling=2.5,
        two_body_power=4.5,
        three_body_width=3.0,
        three_body_scaling=1.5,
        three_body_power=3.0,
        cut_start=0.5,
        cut_distance=1e6,
        fourier_order=1,
        use_atm=False,
    )

    SEEDS = [42, 123, 666, 1337, 2024, 9999, 31415, 271828]
    N_SPLITS = len(SEEDS)

    print("\n" + "=" * 70)
    print(f"FCHL18 multi-split comparison  ({N_SPLITS} random seeds)")
    print(
        f"  Dataset : QM7b, N={N_POINTS} (train={N_POINTS - N_POINTS // 3}, test={N_POINTS // 3})"
    )
    print(f"  sigma={SIGMA}, lambda={LLAMBDA:.0e}")
    print("=" * 70)

    print("\nGenerating representations...")
    data = load_qm7b_raw_data()
    coords_list = list(data["R"][:N_POINTS])
    z_list = [zi.astype(np.int32) for zi in data["z"][:N_POINTS]]
    x, n, nn = repr_mod.generate(
        coords_list,
        z_list,
        max_size=MAX_SIZE,
        cut_distance=BASE_ARGS["cut_distance"],
    )
    properties = np.array(data["E"][:N_POINTS], dtype=np.float64)

    def _run_split(kernel_args: dict, seed: int) -> float:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N_POINTS)
        xp, np_, nnp, yp = x[perm], n[perm], nn[perm], properties[perm]

        n_test = N_POINTS // 3
        n_train = N_POINTS - n_test
        x_tr, n_tr, nn_tr, y_tr = xp[:n_train], np_[:n_train], nnp[:n_train], yp[:n_train]
        x_te, n_te, nn_te, y_te = xp[n_train:], np_[n_train:], nnp[n_train:], yp[n_train:]

        K_sym = kernel_mod.kernel_gaussian_symm(x_tr, n_tr, nn_tr, sigma=SIGMA, **kernel_args)
        K_reg = K_sym.copy()
        K_reg[np.diag_indices_from(K_reg)] += LLAMBDA
        alpha = np.linalg.solve(K_reg, y_tr)

        K_te = kernel_mod.kernel_gaussian(
            x_te, x_tr, n_te, n_tr, nn_te, nn_tr, sigma=SIGMA, **kernel_args
        )
        return float(np.mean(np.abs(y_te - K_te @ alpha)))

    configs = [
        ("default", BASE_ARGS),
        ("tuned", TUNED_ARGS),
    ]

    print(f"\n  {'seed':<10}", end="")
    for name, _ in configs:
        print(f"  {name:>10}", end="")
    print()
    print("  " + "-" * (10 + 13 * len(configs)))

    all_maes: dict[str, list[float]] = {name: [] for name, _ in configs}

    for seed in SEEDS:
        print(f"  {seed:<10}", end="", flush=True)
        for name, args in configs:
            mae = _run_split(args, seed)
            all_maes[name].append(mae)
            print(f"  {mae:>10.3f}", end="", flush=True)
        print()

    # Summary
    print("\n" + "=" * 70)
    print(f"  {'config':<12} {'mean MAE':>10} {'std MAE':>10} {'min MAE':>10} {'max MAE':>10}")
    print(f"  {'-' * 44}")
    for name, _ in configs:
        maes = all_maes[name]
        print(
            f"  {name:<12} {np.mean(maes):>10.3f} {np.std(maes):>10.3f}"
            f" {np.min(maes):>10.3f} {np.max(maes):>10.3f}"
        )

    # Improvement per split
    print(f"\n  Per-split improvement (default - tuned):")
    diffs = [d - t for d, t in zip(all_maes["default"], all_maes["tuned"])]
    for seed, diff in zip(SEEDS, diffs):
        sign = "+" if diff >= 0 else ""
        print(f"    seed={seed:<8}  {sign}{diff:+.3f} kcal/mol")
    print(
        f"\n  Mean improvement: {np.mean(diffs):+.3f} ± {np.std(diffs):.3f} kcal/mol"
        f"  ({100 * np.mean(diffs) / np.mean(all_maes['default']):.1f}%)"
    )
    print("=" * 70)


@pytest.mark.slow
@pytest.mark.integration
def test_fchl18_learning_curve():
    """Learning curve: default vs tuned config at n_train = {500,1000,2000,4000}.

    3 random seeds x 4 training sizes x 2 configs = 24 KRR runs.
    n_test = 1000 (fixed). Total molecules loaded: 5000.

    For each seed the 5000 molecules are shuffled once; the last 1000 are always
    the test set so larger training sets are strict supersets of smaller ones.

    Configs
    -------
    default : original FCHL18 (fourier_order=2, use_atm=True, default params)
    tuned   : Round 4 best   (fourier_order=1, use_atm=False, tuned params)
    Both use sigma=2.5, llambda=1e-8.
    """

    TUNED_ARGS = dict(
        two_body_width=0.1,
        two_body_scaling=2.5,
        two_body_power=4.5,
        three_body_width=3.0,
        three_body_scaling=1.5,
        three_body_power=3.0,
        cut_start=0.5,
        cut_distance=1e6,
        fourier_order=1,
        use_atm=False,
    )

    N_TOTAL = 5000
    N_TEST = 1000
    N_TRAINS = [500, 1000, 2000, 4000]
    SEEDS = [42, 666, 1337]
    CONFIGS = [("default", BASE_ARGS), ("tuned", TUNED_ARGS)]

    print("\n" + "=" * 72)
    print("FCHL18 learning curve: default vs tuned")
    print(f"  n_train = {N_TRAINS},  n_test = {N_TEST} (fixed)")
    print(f"  seeds   = {SEEDS}")
    print(f"  sigma={SIGMA}, lambda={LLAMBDA:.0e}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load data and generate representations once for all 5000 molecules
    # ------------------------------------------------------------------
    print("\nGenerating representations for N=5000 ...")
    data = load_qm7b_raw_data()
    coords_list = list(data["R"][:N_TOTAL])
    z_list = [zi.astype(np.int32) for zi in data["z"][:N_TOTAL]]
    x, n, nn = repr_mod.generate(
        coords_list,
        z_list,
        max_size=MAX_SIZE,
        cut_distance=BASE_ARGS["cut_distance"],
    )
    properties = np.array(data["E"][:N_TOTAL], dtype=np.float64)

    def _run_lc(kernel_args: dict, seed: int, n_train: int) -> float:
        """One KRR run: shuffle, slice, train, predict, return MAE."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N_TOTAL)
        xp, np_, nnp, yp = x[perm], n[perm], nn[perm], properties[perm]

        # Test set: last N_TEST molecules (never overlaps any training set)
        x_te = xp[-N_TEST:]
        n_te = np_[-N_TEST:]
        nn_te = nnp[-N_TEST:]
        y_te = yp[-N_TEST:]

        # Training set: first n_train molecules (nested supersets across sizes)
        x_tr = xp[:n_train]
        n_tr = np_[:n_train]
        nn_tr = nnp[:n_train]
        y_tr = yp[:n_train]

        K_sym = kernel_mod.kernel_gaussian_symm(x_tr, n_tr, nn_tr, sigma=SIGMA, **kernel_args)
        K_reg = K_sym.copy()
        K_reg[np.diag_indices_from(K_reg)] += LLAMBDA
        alpha = np.linalg.solve(K_reg, y_tr)

        K_te = kernel_mod.kernel_gaussian(
            x_te, x_tr, n_te, n_tr, nn_te, nn_tr, sigma=SIGMA, **kernel_args
        )
        return float(np.mean(np.abs(y_te - K_te @ alpha)))

    # ------------------------------------------------------------------
    # Collect results: maes[config_name][n_train] = [mae_seed0, ...]
    # ------------------------------------------------------------------
    maes: dict[str, dict[int, list[float]]] = {
        name: {nt: [] for nt in N_TRAINS} for name, _ in CONFIGS
    }

    col_w = 10
    header = f"  {'n_train':>7}  {'seed':>6}" + "".join(f"  {name:>{col_w}}" for name, _ in CONFIGS)
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for n_train in N_TRAINS:
        for seed in SEEDS:
            row = f"  {n_train:>7}  {seed:>6}"
            for name, args in CONFIGS:
                mae = _run_lc(args, seed, n_train)
                maes[name][n_train].append(mae)
                row += f"  {mae:>{col_w}.3f}"
            print(row, flush=True)
        print()  # blank line between training sizes

    # ------------------------------------------------------------------
    # Summary table: mean ± std per training size
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Learning curve summary  (mean ± std over 3 seeds, kcal/mol)")
    print()

    hdr = f"  {'n_train':>7}"
    for name, _ in CONFIGS:
        hdr += f"  {name + ' mean':>14}  {name + ' std':>10}"
    hdr += f"  {'improv (%)':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for n_train in N_TRAINS:
        row = f"  {n_train:>7}"
        means = []
        for name, _ in CONFIGS:
            m = float(np.mean(maes[name][n_train]))
            s = float(np.std(maes[name][n_train]))
            means.append(m)
            row += f"  {m:>14.3f}  {s:>10.3f}"
        pct = 100.0 * (means[0] - means[1]) / means[0]
        row += f"  {pct:>+11.1f}%"
        print(row)

    print("=" * 72)


if __name__ == "__main__":
    test_fchl18_ablation()
    test_fchl18_hyperparam_scan()
    test_fchl18_round1_scan()
    test_fchl18_round2_grid()
    test_fchl18_round3_scan()
    test_fchl18_round4_grid()
    test_fchl18_multi_split()
    test_fchl18_learning_curve()
