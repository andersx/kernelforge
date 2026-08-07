"""Compare FCHL19 representation precision: CPU FP64 vs CUDA FP32 (det. and non-det.).

For a single azobenzene geometry, computes X (representation) and dX/dr (Jacobian)
with three backends and reports absolute and relative errors vs the CPU FP64 reference.
Non-deterministic CUDA is repeated N_REPEAT times to also show run-to-run variation.

Usage
-----
    uv run --env-file /dev/null python benchmarks/bench_fchl19_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kernelforge.kernelcli import load_rmd17  # type: ignore[attr-defined]

N_REPEAT = 20  # runs of non-deterministic CUDA to measure noise
ELEMENTS = [1, 6, 7]
REPR_PARAMS: dict = {}  # use library defaults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(label: str, diff: np.ndarray, ref: np.ndarray) -> None:
    abs_max  = float(np.max(np.abs(diff)))
    abs_mean = float(np.mean(np.abs(diff)))
    rel_max  = float(np.max(np.abs(diff) / (np.abs(ref) + 1e-30)))
    print(f"  {label}")
    print(f"    max  |Δ| = {abs_max:.3e}   mean |Δ| = {abs_mean:.3e}   max |Δ|/|ref| = {rel_max:.3e}")


# ---------------------------------------------------------------------------
# 1. Load one azobenzene geometry
# ---------------------------------------------------------------------------
print("[bench] Loading rMD17 azobenzene frame 0 ...")
tr_coords, tr_z, *_ = load_rmd17("azobenzene", split=1, n_train=1, n_test=1)
coords = tr_coords[0]   # (24, 3) float64
z      = tr_z[0]        # (24,)   int32

# ---------------------------------------------------------------------------
# 2. CPU FP64 reference
# ---------------------------------------------------------------------------
print("[bench] Computing CPU FP64 reference ...")
from kernelforge import fchl19_repr  # type: ignore[attr-defined]

X_cpu, dX_cpu = fchl19_repr.generate_fchl_acsf_and_gradients(
    np.asarray(coords, dtype=np.float64),
    np.asarray(z, dtype=np.int32),
    elements=ELEMENTS,
    **REPR_PARAMS,
)
# X_cpu  : (n_atoms, rep_size)           float64
# dX_cpu : (n_atoms, rep_size, n_atoms*3) float64  ← reshaped to (n_atoms, rep_size, n_atoms, 3) below
print(f"  X shape: {X_cpu.shape},  dX shape: {dX_cpu.shape}")

# ---------------------------------------------------------------------------
# 3. CUDA FP32 — build batched input tensors
# ---------------------------------------------------------------------------
import torch
from kernelforge import cuda_fchl19_repr  # type: ignore[attr-defined]

n_atoms  = len(z)
elem_to_idx = {e: i for i, e in enumerate(ELEMENTS)}

# Reshape CPU dX from (n_atoms, rep_size, n_atoms*3) → (n_atoms, rep_size, n_atoms, 3)
dX_cpu = dX_cpu.reshape(n_atoms, X_cpu.shape[1], n_atoms, 3)

coords_t = torch.from_numpy(
    np.asarray(coords, dtype=np.float32)[None]          # (1, n_atoms, 3)
).cuda()
Q_t = torch.tensor(
    [[elem_to_idx[int(zi)] for zi in z]], dtype=torch.int32
).cuda()                                                 # (1, n_atoms)
N_t = torch.tensor([n_atoms], dtype=torch.int32).cuda() # (1,)

ne = len(ELEMENTS)

def _cuda_run(deterministic: bool) -> tuple[np.ndarray, np.ndarray]:
    X_t, dX5_t = cuda_fchl19_repr.generate_fchl_acsf_and_gradients(
        coords_t, Q_t, N_t, nelements=ne, deterministic=deterministic, **REPR_PARAMS
    )
    # X_t  : (1, n_atoms, rep_size)
    # dX5_t: (1, n_atoms, rep_size, n_atoms, 3)
    X  = X_t[0].cpu().numpy().astype(np.float64)        # (n_atoms, rep_size)
    dX = dX5_t[0].cpu().numpy().astype(np.float64)      # (n_atoms, rep_size, n_atoms, 3)
    return X, dX

# ---------------------------------------------------------------------------
# 4. CUDA FP32 deterministic — single run
# ---------------------------------------------------------------------------
print("\n[bench] CUDA FP32 deterministic (1 run) vs CPU FP64:")
X_det, dX_det = _cuda_run(deterministic=True)
_stats("X   (representation)", X_det - X_cpu, X_cpu)
_stats("dX  (Jacobian)      ", dX_det - dX_cpu, dX_cpu)

# ---------------------------------------------------------------------------
# 5. CUDA FP32 non-deterministic — N_REPEAT runs, show mean error + run-to-run noise
# ---------------------------------------------------------------------------
print(f"\n[bench] CUDA FP32 non-deterministic ({N_REPEAT} runs) vs CPU FP64:")
X_runs  = np.stack([_cuda_run(deterministic=False)[0] for _ in range(N_REPEAT)])
dX_runs = np.stack([_cuda_run(deterministic=False)[1] for _ in range(N_REPEAT)])

X_mean  = X_runs.mean(axis=0)
dX_mean = dX_runs.mean(axis=0)
_stats("X   (vs CPU FP64, mean of runs)", X_mean  - X_cpu,  X_cpu)
_stats("dX  (vs CPU FP64, mean of runs)", dX_mean - dX_cpu, dX_cpu)

X_noise  = X_runs.std(axis=0)
dX_noise = dX_runs.std(axis=0)
print(f"  X   run-to-run std:  max={X_noise.max():.3e}   mean={X_noise.mean():.3e}")
print(f"  dX  run-to-run std:  max={dX_noise.max():.3e}   mean={dX_noise.mean():.3e}")

# ---------------------------------------------------------------------------
# 6. Det vs non-det (purely CUDA-internal difference)
# ---------------------------------------------------------------------------
print("\n[bench] CUDA FP32 deterministic vs non-deterministic (mean of runs):")
_stats("X  ", X_det  - X_mean,  X_mean)
_stats("dX ", dX_det - dX_mean, dX_mean)
