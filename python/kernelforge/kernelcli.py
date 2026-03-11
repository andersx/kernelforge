"""kernelcli — command-line interface for testing kernelforge ML potentials.

Supports three datasets (rMD17, QM7b, small_mols_mini), three model classes
(LocalKRRModel, LocalRFFModel, FCHL18KRRModel), three representations (FCHL19,
FCHL19v2, FCHL18), and three training modes (energy_only, force_only, energy_and_force).

Usage examples::

    kernelcli --dataset rmd17_ethanol --regressor krr --representation fchl19 \\
              --mode energy_and_force --n-train 200 --n-test 200 --split 1

    kernelcli --dataset qm7b --regressor rff --mode energy_only \\
              --n-train 1000 --n-test 500 --sigma 10.0 --d-rff 2048

    kernelcli --dataset small_mols_mini --regressor krr --representation fchl18 \\
              --mode energy_and_force --sigma 2.5 --l2 1e-4

    kernelcli --dataset rmd17_ethanol --representation fchl19v2 \\
              --repr-param two_body_type=bessel --repr-param three_body_type=cosine_rbar

Representation/kernel parameters can be overridden with ``--repr-param KEY=VALUE``
(repeatable).  For FCHL19/FCHL19v2 these map to ``repr_params``; for FCHL18 they
map to ``kernel_params``.  Values are auto-cast to int, float, bool, or str::

    kernelcli --dataset rmd17_ethanol --repr-param rcut=6.0 --repr-param nRs2=32
    kernelcli --dataset rmd17_ethanol --representation fchl18 \\
              --repr-param cut_distance=8.0 --repr-param two_body_width=0.15
    kernelcli --dataset rmd17_ethanol --representation fchl19v2 \\
              --repr-param two_body_type=bessel --repr-param nCosine=4

Entry point registered as ``kernelcli`` in pyproject.toml.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernelforge.cli import load_qm7b_raw_data
from kernelforge.models import FCHL18KRRModel, LocalKRRModel, LocalRFFModel
from kernelforge.models.base import _coerce_forces, _compute_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RMD17_MOLECULES = [
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
]

RMD17_BASE_URL = "https://raw.githubusercontent.com/andersx/rmd17-npz/master/rmd17-npz/"

CACHE_DIR = Path.home() / ".kernelforge" / "datasets"

# Path to the bundled small_mols_mini files (installed alongside the package)
_THIS_DIR = Path(__file__).parent
_EXAMPLES_DIR = _THIS_DIR.parent.parent / "examples"

SMALL_MOLS_TRAIN_NPZ = _EXAMPLES_DIR / "small_mols_mini_train.npz"
SMALL_MOLS_TEST_NPZ = _EXAMPLES_DIR / "small_mols_mini_test.npz"

VALID_DATASETS = ["qm7b", "small_mols_mini"] + [f"rmd17_{m}" for m in RMD17_MOLECULES]


# ---------------------------------------------------------------------------
# Repr/kernel param parsing
# ---------------------------------------------------------------------------


def _parse_repr_params(raw: list[str] | None) -> dict[str, Any]:
    """Parse a list of ``KEY=VALUE`` strings into a typed dict.

    Values are cast in order: int -> float -> bool (true/false) -> str.

    Examples
    --------
    >>> _parse_repr_params(["nRs2=32", "rcut=6.0", "use_atm=false"])
    {'nRs2': 32, 'rcut': 6.0, 'use_atm': False}
    """
    if not raw:
        return {}
    result: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            msg = f"--repr-param must be KEY=VALUE, got: {item!r}"
            raise ValueError(msg)
        key, _, val_str = item.partition("=")
        key = key.strip()
        val_str = val_str.strip()
        # Try int first, then float, then bool, then keep as str
        parsed: int | float | bool | str
        try:
            parsed = int(val_str)
        except ValueError:
            try:
                parsed = float(val_str)
            except ValueError:
                if val_str.lower() == "true":
                    parsed = True
                elif val_str.lower() == "false":
                    parsed = False
                else:
                    parsed = val_str
        result[key] = parsed
    return result


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _download_if_needed(url: str, dest: Path) -> None:
    """Download *url* to *dest* if it does not already exist."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [Downloading {dest.name} ...]")
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status != 200:
                msg = f"HTTP {resp.status}: failed to download {url}"
                raise RuntimeError(msg)
            dest.write_bytes(resp.read())
    except Exception as exc:
        print(f"  [Error downloading {url}: {exc}]", file=sys.stderr)
        raise


def _load_rmd17_split(molecule: str, split: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (train_data, test_data) dicts for one rMD17 molecule/split."""
    for tag in ("train", "test"):
        fname = f"rmd17_{molecule}_{tag}_{split:02d}.npz"
        url = RMD17_BASE_URL + fname
        dest = CACHE_DIR / fname
        _download_if_needed(url, dest)

    train_path = CACHE_DIR / f"rmd17_{molecule}_train_{split:02d}.npz"
    test_path = CACHE_DIR / f"rmd17_{molecule}_test_{split:02d}.npz"

    train_data = dict(np.load(train_path, allow_pickle=True))
    test_data = dict(np.load(test_path, allow_pickle=True))
    return train_data, test_data


def load_rmd17(
    molecule: str,
    split: int,
    n_train: int | None,
    n_test: int | None,
) -> tuple[
    list[NDArray[np.float64]],
    list[NDArray[np.int32]],
    NDArray[np.float64],
    list[NDArray[np.float64]],
    list[NDArray[np.float64]],
    list[NDArray[np.int32]],
    NDArray[np.float64],
    list[NDArray[np.float64]],
]:
    """Load rMD17 data for *molecule* / *split*, slicing to *n_train* / *n_test*."""
    train_raw, test_raw = _load_rmd17_split(molecule, split)

    # rMD17 keys: nuclear_charges (N_atoms,), coords (N,Na,3), energies (N,), forces (N,Na,3)
    z_fixed = train_raw["nuclear_charges"].astype(np.int32)

    def _unpack(
        raw: dict[str, Any], n: int | None
    ) -> tuple[
        list[NDArray[np.float64]],
        list[NDArray[np.int32]],
        NDArray[np.float64],
        list[NDArray[np.float64]],
    ]:
        n_avail = len(raw["energies"])
        if n is not None and n > n_avail:
            msg = (
                f"Requested {n} samples but rMD17 split only has {n_avail}. "
                f"Use --n-train / --n-test <= {n_avail}."
            )
            raise ValueError(msg)
        n = n if n is not None else n_avail
        coords = [raw["coords"][i].astype(np.float64) for i in range(n)]
        z_list = [z_fixed for _ in range(n)]
        energies = raw["energies"][:n].astype(np.float64)
        forces = [raw["forces"][i].astype(np.float64) for i in range(n)]
        return coords, z_list, energies, forces

    coords_tr, z_tr, E_tr, F_tr = _unpack(train_raw, n_train)
    coords_te, z_te, E_te, F_te = _unpack(test_raw, n_test)
    return coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te


def load_qm7b(
    n_train: int | None,
    n_test: int | None,
    seed: int,
) -> tuple[
    list[NDArray[np.float64]],
    list[NDArray[np.int32]],
    NDArray[np.float64],
    list[NDArray[np.float64]],
    list[NDArray[np.int32]],
    NDArray[np.float64],
]:
    """Load QM7b with a random train/test split. Forces are not available."""
    data = load_qm7b_raw_data()
    n_total = len(data["E"])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)

    n_tr = n_train if n_train is not None else 5000
    n_te = n_test if n_test is not None else 2000

    if n_tr + n_te > n_total:
        msg = f"n_train ({n_tr}) + n_test ({n_te}) = {n_tr + n_te} exceeds QM7b size ({n_total})."
        raise ValueError(msg)

    tr_idx = idx[:n_tr]
    te_idx = idx[n_tr : n_tr + n_te]

    R_tr_raw = data["R"][tr_idx]
    R_te_raw = data["R"][te_idx]
    z_tr_raw = data["z"][tr_idx]
    z_te_raw = data["z"][te_idx]

    coords_tr = [R_tr_raw[i].astype(np.float64) for i in range(n_tr)]
    z_tr = [z_tr_raw[i].astype(np.int32) for i in range(n_tr)]
    coords_te = [R_te_raw[i].astype(np.float64) for i in range(n_te)]
    z_te = [z_te_raw[i].astype(np.int32) for i in range(n_te)]

    E_tr = data["E"][tr_idx].astype(np.float64)
    E_te = data["E"][te_idx].astype(np.float64)

    return coords_tr, z_tr, E_tr, coords_te, z_te, E_te


def load_small_mols_mini(
    n_train: int | None,
    n_test: int | None,
) -> tuple[
    list[NDArray[np.float64]],
    list[NDArray[np.int32]],
    NDArray[np.float64],
    list[NDArray[np.float64]],
    list[NDArray[np.float64]],
    list[NDArray[np.int32]],
    NDArray[np.float64],
    list[NDArray[np.float64]],
]:
    """Load the bundled small_mols_mini train/test splits."""
    if not SMALL_MOLS_TRAIN_NPZ.exists() or not SMALL_MOLS_TEST_NPZ.exists():
        msg = (
            f"small_mols_mini data not found.\n"
            f"Expected:\n  {SMALL_MOLS_TRAIN_NPZ}\n  {SMALL_MOLS_TEST_NPZ}"
        )
        raise FileNotFoundError(msg)

    def _load(
        path: Path, n: int | None
    ) -> tuple[
        list[NDArray[np.float64]],
        list[NDArray[np.int32]],
        NDArray[np.float64],
        list[NDArray[np.float64]],
    ]:
        d = np.load(path, allow_pickle=True)
        n_avail = len(d["atomization_energy"])
        n = n if n is not None else n_avail
        if n > n_avail:
            msg = f"Requested {n} but {path.name} only has {n_avail} samples."
            raise ValueError(msg)
        coords = [d["coords"][i].astype(np.float64) for i in range(n)]
        z_list = [d["nuclear_charges"][i].astype(np.int32) for i in range(n)]
        energies = d["atomization_energy"][:n].astype(np.float64)
        forces = [d["forces"][i].astype(np.float64) for i in range(n)]
        return coords, z_list, energies, forces

    coords_tr, z_tr, E_tr, F_tr = _load(SMALL_MOLS_TRAIN_NPZ, n_train)
    coords_te, z_te, E_te, F_te = _load(SMALL_MOLS_TEST_NPZ, n_test)
    return coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_model(
    regressor: str,
    representation: str,
    sigma: float,
    l2: float,
    elements: list[int] | None,
    max_size: int | None,
    d_rff: int,
    seed: int,
    z_tr: list[NDArray[np.int32]],
    z_te: list[NDArray[np.int32]],
    repr_params: dict[str, Any] | None = None,
) -> LocalKRRModel | LocalRFFModel | FCHL18KRRModel:
    """Construct and return the appropriate model instance."""
    repr_params = repr_params or {}

    # Auto-detect elements from training data if not overridden
    if elements is None:
        all_z = np.unique(np.concatenate([z.astype(np.int32) for z in z_tr]))
        elements = sorted(int(z) for z in all_z)

    # Auto-detect max_size from both train and test to avoid padding errors at predict time
    if max_size is None:
        max_size = max(len(z) for z in z_tr + z_te)

    if representation == "fchl18":
        # repr_params override kernel_params for FCHL18
        return FCHL18KRRModel(
            sigma=sigma, l2=l2, max_size=max_size, kernel_params=repr_params or None
        )

    # FCHL19 or FCHL19v2 — KRR or RFF
    # repr_params forwarded to generate_fchl_acsf[_and_gradients] (v1)
    # or generate[_and_gradients] (v2); extra v2 params (two_body_type, etc.)
    # are passed through repr_params unchanged.
    repr_name = representation  # "fchl19" or "fchl19v2"
    if regressor == "krr":
        return LocalKRRModel(
            sigma=sigma,
            l2=l2,
            elements=elements,
            repr_params=repr_params or None,
            representation=repr_name,
        )
    # rff
    return LocalRFFModel(
        sigma=sigma,
        l2=l2,
        d_rff=d_rff,
        seed=seed,
        elements=elements,
        repr_params=repr_params or None,
        representation=repr_name,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_header(args: argparse.Namespace) -> None:
    print()
    print("=" * 62)
    print("  kernelcli — kernelforge model benchmark")
    print("=" * 62)
    print(f"  dataset        : {args.dataset}")
    print(f"  regressor      : {args.regressor.upper()}")
    print(f"  representation : {args.representation.upper()}")
    print(f"  mode           : {args.mode}")
    print(f"  sigma          : {args.sigma}")
    print(f"  l2             : {args.l2}")
    if args.regressor == "rff":
        print(f"  d_rff          : {args.d_rff}")
    if args.dataset.startswith("rmd17_"):
        print(f"  split          : {args.split}")
    parsed_rp = _parse_repr_params(args.repr_param)
    if parsed_rp:
        for k, v in parsed_rp.items():
            print(f"  repr-param     : {k}={v}")
    print("=" * 62)


def _print_data_summary(
    dataset: str,
    n_tr: int,
    n_te: int,
    z_tr: list[NDArray[np.int32]],
    E_tr: NDArray[np.float64],
) -> None:
    unique_z = sorted({int(z) for zi in z_tr for z in zi})
    sizes_tr = [len(z) for z in z_tr]
    print(f"\n[1] Dataset: {dataset}")
    print(f"    train / test    : {n_tr} / {n_te}")
    print(f"    elements (Z)    : {unique_z}")
    print(f"    atoms/mol (tr)  : {min(sizes_tr)}-{max(sizes_tr)}")
    print(f"    energy range(tr): {E_tr.min():.3f} .. {E_tr.max():.3f} kcal/mol")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kernelcli",
        description="kernelforge model testing CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=VALID_DATASETS,
        help="Dataset to use.",
    )
    p.add_argument(
        "--regressor",
        default="krr",
        choices=["krr", "rff"],
        help="Regression method.",
    )
    p.add_argument(
        "--representation",
        default="fchl19",
        choices=["fchl19", "fchl19v2", "fchl18"],
        help="Molecular representation.",
    )
    p.add_argument(
        "--mode",
        default="energy_and_force",
        choices=["energy_only", "force_only", "energy_and_force"],
        help="Training mode.",
    )
    p.add_argument("--n-train", type=int, default=None, help="Number of training samples.")
    p.add_argument("--n-test", type=int, default=None, help="Number of test samples.")
    p.add_argument(
        "--split",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="rMD17 pre-split index (1-5). Ignored for qm7b/small_mols_mini.",
    )
    p.add_argument("--sigma", type=float, default=20.0, help="Kernel length-scale.")
    p.add_argument("--l2", type=float, default=1e-8, help="L2 regularisation.")
    p.add_argument("--d-rff", type=int, default=1024, help="RFF feature dimension (RFF only).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (RFF weights + QM7b split).")
    p.add_argument(
        "--elements",
        type=int,
        nargs="+",
        default=None,
        metavar="Z",
        help="Override element list, e.g. --elements 1 6 8 (FCHL19 only).",
    )
    p.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Override max_size padding (FCHL18 only). Auto-detected if omitted.",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Save fitted model to .npz file.",
    )
    p.add_argument(
        "--repr-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override a representation or kernel parameter (repeatable). "
            "Values are auto-cast to int/float/bool/str. "
            "For FCHL19: nRs2, nRs3, nFourier, eta2, eta3, zeta, rcut, acut, "
            "two_body_decay, three_body_decay, three_body_weight. "
            "For FCHL19v2: same as FCHL19 plus two_body_type, three_body_type, "
            "nCosine, nRs3_minus, eta3_minus. "
            "For FCHL18: two_body_width, three_body_width, cut_distance, "
            "two_body_scaling, three_body_scaling, two_body_power, "
            "three_body_power, cut_start, fourier_order, use_atm. "
            "Example: --repr-param rcut=6.0 --repr-param nRs2=32"
        ),
    )
    return p


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate cross-argument constraints."""
    if args.representation == "fchl18" and args.regressor == "rff":
        parser.error(
            "RFF regressor is not supported with the FCHL18 representation. "
            "Use --regressor krr or --representation fchl19/fchl19v2."
        )
    if args.dataset == "qm7b" and args.mode in ("force_only", "energy_and_force"):
        parser.error(
            f"QM7b dataset has no forces. Use --mode energy_only (got --mode {args.mode})."
        )
    if args.dataset == "small_mols_mini" and args.mode == "force_only":
        parser.error(
            "small_mols_mini training in force_only mode is not supported via kernelcli. "
            "Use energy_only or energy_and_force."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Execute the full train / evaluate pipeline."""
    _print_header(args)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    F_tr: list[NDArray[np.float64]] | None
    F_te: list[NDArray[np.float64]] | None

    if args.dataset.startswith("rmd17_"):
        molecule = args.dataset[len("rmd17_") :]
        coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te = load_rmd17(
            molecule, args.split, args.n_train, args.n_test
        )
    elif args.dataset == "qm7b":
        coords_tr, z_tr, E_tr, coords_te, z_te, E_te = load_qm7b(
            args.n_train, args.n_test, args.seed
        )
        F_tr = None
        F_te = None
    else:  # small_mols_mini
        coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te = load_small_mols_mini(
            args.n_train, args.n_test
        )

    load_time = time.perf_counter() - t0
    _print_data_summary(args.dataset, len(E_tr), len(E_te), z_tr, E_tr)
    print(f"    loaded in       : {load_time:.2f}s")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    repr_params = _parse_repr_params(args.repr_param)
    model = _build_model(
        regressor=args.regressor,
        representation=args.representation,
        sigma=args.sigma,
        l2=args.l2,
        elements=args.elements,
        max_size=args.max_size,
        d_rff=args.d_rff,
        seed=args.seed,
        z_tr=z_tr,
        z_te=z_te,
        repr_params=repr_params,
    )
    print(f"\n[2] Model: {type(model).__name__}")

    # ------------------------------------------------------------------
    # 3. Fit
    # ------------------------------------------------------------------
    fit_energies = E_tr if args.mode in ("energy_only", "energy_and_force") else None
    fit_forces = F_tr if args.mode in ("force_only", "energy_and_force") else None

    t0 = time.perf_counter()
    model.fit(coords_tr, z_tr, energies=fit_energies, forces=fit_forces)
    fit_time = time.perf_counter() - t0

    print(f"    fitted in       : {fit_time:.2f}s")
    print(f"    training_mode   : {model.training_mode_}")
    if len(model.baseline_elements_) > 0:
        baseline_str = "  ".join(
            f"Z{z}={e:.4f}"
            for z, e in zip(model.baseline_elements_, model.element_energies_, strict=True)
        )
        print(f"    element energies: {baseline_str}  kcal/mol")

    if "energy" in model.train_score_:
        print(f"    train energy    : {model.train_score_['energy']}")
    if "force" in model.train_score_:
        print(f"    train force     : {model.train_score_['force']}")

    # ------------------------------------------------------------------
    # 4. Score on test set
    # ------------------------------------------------------------------
    # Always score energies when available. For forces: score whenever reference
    # forces exist and the model produced non-empty predictions. FCHL18 energy_only
    # predicts forces via the gradient kernel; LocalKRR/RFF energy_only returns an
    # empty array. We call predict() once and compute scores directly to avoid the
    # base.score() guard that blocks force scoring in energy_only mode.
    t0 = time.perf_counter()
    E_pred, F_pred = model.predict(coords_te, z_te)
    score_time = time.perf_counter() - t0

    test_scores: dict[str, object] = {}
    if args.mode in ("energy_only", "energy_and_force"):
        test_scores["energy"] = _compute_score(E_te, E_pred)
    if F_te is not None and F_pred.size > 0:
        F_ref = _coerce_forces(F_te)
        test_scores["force"] = _compute_score(F_ref, F_pred)

    print(f"\n[3] Test set ({len(E_te)} molecules, scored in {score_time:.2f}s)")
    if "energy" in test_scores:
        print(f"    test  energy    : {test_scores['energy']}")
    if "force" in test_scores:
        print(f"    test  force     : {test_scores['force']}")

    # ------------------------------------------------------------------
    # 5. Optional save
    # ------------------------------------------------------------------
    if args.save is not None:
        save_path = Path(args.save)
        model.save(save_path)
        print(f"\n[4] Model saved to {save_path}")

    print("\n" + "=" * 62 + "\nDone.\n" + "=" * 62)


def main() -> None:
    """Entry point for the ``kernelcli`` command."""
    parser = _build_parser()
    args = parser.parse_args()
    _validate(args, parser)
    run(args)


if __name__ == "__main__":
    main()
