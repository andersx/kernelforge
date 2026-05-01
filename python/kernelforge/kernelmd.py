"""kernelmd — command-line interface for KernelForge MD simulations.

Usage
-----
    kernelmd --model model.npz --structure start.xyz --steps 5000 --dt 0.5 \\
             --temperature 300 --output traj.extxyz --interval 10

The model is loaded from a ``.npz`` file written by ``model.save()``.
The starting geometry can be any format that ASE can read (.xyz, .extxyz,
SDF, PDB, …).  The trajectory is written as extended-XYZ by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model-class registry
# ---------------------------------------------------------------------------

_MODEL_CLASS_MAP: dict[str, str] = {
    "CudaLocalKRRModel": "kernelforge.models.cuda_local_krr.CudaLocalKRRModel",
    "CudaLocalRFFModel": "kernelforge.models.cuda_local_rff.CudaLocalRFFModel",
    "CudaGlobalKRRModel": "kernelforge.models.cuda_global_krr.CudaGlobalKRRModel",
    "CudaGlobalRFFModel": "kernelforge.models.cuda_global_rff.CudaGlobalRFFModel",
}

_CLI_CLASS_MAP: dict[str, str] = {
    "cuda-local-krr": "CudaLocalKRRModel",
    "cuda-local-rff": "CudaLocalRFFModel",
    "cuda-global-krr": "CudaGlobalKRRModel",
    "cuda-global-rff": "CudaGlobalRFFModel",
}


def _load_model(model_path: Path, model_class_arg: str) -> Any:  # noqa: ANN401
    """Load a saved KernelForge model from *model_path*.

    Parameters
    ----------
    model_path:
        Path to the ``.npz`` file produced by ``model.save()``.
    model_class_arg:
        Either ``'auto'`` (detect from the ``model_class`` field in the npz)
        or one of the explicit CLI keys (``'cuda-local-krr'``, …).
    """
    import numpy as np

    if not model_path.exists():
        print(f"[kernelmd] ERROR: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve model_class name
    if model_class_arg == "auto":
        data = np.load(model_path, allow_pickle=True)
        if "model_class" not in data:
            print(
                "[kernelmd] ERROR: 'model_class' key not found in npz. "
                "Use --model-class to specify explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        class_name = str(data["model_class"])
    else:
        class_name = _CLI_CLASS_MAP.get(model_class_arg, model_class_arg)

    if class_name not in _MODEL_CLASS_MAP:
        print(
            f"[kernelmd] ERROR: unknown model class '{class_name}'. "
            f"Known: {list(_MODEL_CLASS_MAP)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import and load
    module_path, cls_name = _MODEL_CLASS_MAP[class_name].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    print(f"[kernelmd] Loading {class_name} from {model_path}")
    return cls.load(model_path)


def _load_structure(structure_path: Path) -> object:
    """Load the initial geometry using ASE io.read."""
    try:
        from ase.io import read
    except ImportError as exc:
        msg = (
            "ASE is required for kernelmd. "
            "Install it with:  pip install 'kernelforge[ase]'  or  pip install ase"
        )
        raise ImportError(msg) from exc

    if not structure_path.exists():
        print(
            f"[kernelmd] ERROR: structure file not found: {structure_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    atoms = read(str(structure_path))
    print(f"[kernelmd] Loaded structure: {len(atoms)} atoms ({structure_path.name})")
    return atoms


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kernelmd",
        description="Run NVE molecular dynamics with a KernelForge CUDA model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, type=Path, help="Path to saved model (.npz).")
    p.add_argument(
        "--structure",
        required=True,
        type=Path,
        help="Initial geometry in any ASE-readable format (.xyz, .extxyz, SDF, …).",
    )
    p.add_argument("--steps", type=int, default=1000, help="Number of MD steps.")
    p.add_argument("--dt", type=float, default=0.5, help="Timestep in femtoseconds.")
    p.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help=(
            "Temperature (K) for Maxwell-Boltzmann velocity initialisation. "
            "Set to 0 to keep existing momenta."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("md.extxyz"),
        help="Trajectory output path (.extxyz or .traj).",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Save a trajectory frame every INTERVAL steps.",
    )
    p.add_argument(
        "--logfile",
        type=Path,
        default=Path("md.log"),
        help="Plain-text log file path (step, time, energies, temperature).",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Write a log entry every LOG_INTERVAL steps.",
    )
    p.add_argument(
        "--units",
        default="kcal/mol",
        choices=["kcal/mol", "eV"],
        help="Energy/force units used during model training.",
    )
    p.add_argument(
        "--model-class",
        default="auto",
        help=(
            "Model class to load.  'auto' reads the 'model_class' field from the npz. "
            "Explicit options: cuda-local-krr, cuda-local-rff, "
            "cuda-global-krr, cuda-global-rff."
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for velocity initialisation.")
    return p


def main() -> None:
    """Entry point for the ``kernelmd`` CLI command."""
    parser = _build_parser()
    args = parser.parse_args()

    model = _load_model(args.model, args.model_class)
    atoms = _load_structure(args.structure)

    temperature = args.temperature if args.temperature > 0 else None

    print(
        f"[kernelmd] NVE MD: {args.steps} steps x {args.dt} fs"
        + (f" @ {temperature} K (Maxwell-Boltzmann init)" if temperature else "")
    )
    print(f"[kernelmd] Trajectory → {args.output}")
    print(f"[kernelmd] Log        → {args.logfile}")

    from .md import run_md

    frames = run_md(
        model=model,
        atoms=atoms,
        n_steps=args.steps,
        dt=args.dt,
        temperature=temperature,
        trajectory_file=args.output,
        traj_interval=args.interval,
        logfile=args.logfile,
        log_interval=args.log_interval,
        units=args.units,
        seed=args.seed,
    )

    # Write extxyz trajectory if the user asked for .extxyz output
    # (ASE Trajectory writes .traj binary; extxyz needs ase.io.write)
    if str(args.output).endswith(".extxyz"):
        try:
            from ase.io import write

            write(str(args.output), frames)
            print(f"[kernelmd] Wrote {len(frames)} frames to {args.output} (extxyz)")
        except Exception as exc:  # noqa: BLE001
            print(f"[kernelmd] WARNING: could not write extxyz: {exc}", file=sys.stderr)

    print(f"[kernelmd] Done.  {len(frames)} frames saved.")


if __name__ == "__main__":
    main()
