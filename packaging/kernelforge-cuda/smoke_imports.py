"""Smoke-check that kernelforge-cuda dropped cuda_* modules into kernelforge/."""

from __future__ import annotations

import importlib.util
import sys

import kernelforge

MODS = [
    "cuda_global_kernels",
    "cuda_local_kernels",
    "cuda_fchl18_kernel",
    "cuda_fchl18_repr",
    "cuda_fchl19_repr",
    "cuda_rff_features",
    "cuda_invdist_repr",
    "cuda_solvers",
]


def main() -> int:
    print("kernelforge:", kernelforge.__file__)
    missing = [m for m in MODS if importlib.util.find_spec(f"kernelforge.{m}") is None]
    if missing:
        print(f"missing CUDA modules: {missing}", file=sys.stderr)
        return 1
    for m in MODS:
        importlib.import_module(f"kernelforge.{m}")
    print("OK: companion wheel dropped all CUDA modules into", kernelforge.__file__)
    for m in MODS:
        print(f"  {m}:", importlib.util.find_spec(f"kernelforge.{m}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
