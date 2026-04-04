#!/usr/bin/env python3
"""
diag_kernel_compare.py — Compare the GPU K_full against the CPU reference.

Builds the full symmetric K_full matrix:
  - CPU: global_kernels.kernel_gaussian_full_symm  (float64, lower triangle)
  - GPU: cuda_krr_ext.build_kernel_matrix           (float32, full symmetric)

Then compares every element of the lower triangle to float32 tolerance.

Usage:
    uv run python scripts/diag_kernel_compare.py [--n N] [--sigma SIGMA] [--real-data]

Flags:
  --n N          Number of training molecules (default: 10)
  --sigma SIGMA  Kernel length-scale (default: 1.0)
  --real-data    Load rMD17 ethanol instead of using synthetic data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="GPU vs CPU K_full diagnostic")
parser.add_argument("--n", type=int, default=10, help="Number of molecules")
parser.add_argument("--sigma", type=float, default=1.0, help="Kernel sigma")
parser.add_argument(
    "--real-data",
    action="store_true",
    help="Use rMD17 ethanol data (requires dataset cache)",
)
args = parser.parse_args()

N = args.n
sigma = args.sigma

# ---------------------------------------------------------------------------
# Imports (guard for missing CUDA extension)
# ---------------------------------------------------------------------------

try:
    import kernelforge.cuda_krr_ext as cuda_krr_ext  # type: ignore[import]
except ImportError:
    sys.exit("ERROR: cuda_krr_ext not built. Run 'make install-linux-mkl-ilp64' with CUDA.")

from kernelforge import global_kernels, invdist_repr  # noqa: E402

print(f"--- diag_kernel_compare.py  N={N}  sigma={sigma} ---")

# ---------------------------------------------------------------------------
# Build or load X (N, M) and dX (N, D, M)
# ---------------------------------------------------------------------------

if args.real_data:
    cache = Path.home() / ".kernelforge" / "datasets" / "rmd17_ethanol_train_01.npz"
    if not cache.exists():
        sys.exit(f"ERROR: dataset not found at {cache}")

    d = np.load(cache, allow_pickle=True)
    z_fixed = d["nuclear_charges"].astype(np.int32)
    X_list, dX_list = [], []
    for i in range(N):
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(
            d["coords"][i].astype(np.float64), 1e-12
        )
        X_list.append(x)
        dX_list.append(dx)
    X = np.array(X_list, dtype=np.float64)  # (N, M)
    dX = np.array(dX_list, dtype=np.float64)  # (N, D, M)
    print(f"Loaded rMD17 ethanol: X.shape={X.shape}  dX.shape={dX.shape}")
else:
    # Synthetic data: random (N, M=36) X and (N, D=27, M=36) dX
    rng = np.random.default_rng(42)
    M_dim, D_dim = 36, 27
    X = rng.standard_normal((N, M_dim))
    dX = rng.standard_normal((N, D_dim, M_dim))
    print(f"Synthetic data: X.shape={X.shape}  dX.shape={dX.shape}")

N_mol, M_dim = X.shape
D_dim = dX.shape[1]
full = N_mol * (1 + D_dim)

print(f"  M={M_dim}  D={D_dim}  full_rows={full}")

# ---------------------------------------------------------------------------
# CPU K_full: kernel_gaussian_full_symm (float64, lower triangle only)
# ---------------------------------------------------------------------------
# kernel_gaussian_full_symm binding takes dX as 3D (N, D, M)
print("\nBuilding CPU K_full (float64)...")
K_cpu_f64 = global_kernels.kernel_gaussian_full_symm(X, dX, sigma, None)
# Returns shape (N*(1+D), N*(1+D)), lower triangle filled, upper is zero
print(f"  K_cpu_f64.shape = {K_cpu_f64.shape}")

# Cast to float32 for apple-to-apple comparison
K_cpu_f32 = K_cpu_f64.astype(np.float32)

# ---------------------------------------------------------------------------
# GPU K_full: cuda_krr_ext.build_kernel_matrix (float32, fully symmetric)
# ---------------------------------------------------------------------------
# cuda_krr_ext.build_kernel_matrix takes dXT as (N*D, M)
dXT_f64 = dX.reshape(N_mol * D_dim, M_dim)  # (N*D, M) for CUDA
X_f32 = X.astype(np.float32)
dXT_f32 = dXT_f64.astype(np.float32)

print("Building GPU K_full (float32)...")
K_gpu = cuda_krr_ext.build_kernel_matrix(X_f32, dXT_f32, float(sigma))
print(f"  K_gpu.shape = {K_gpu.shape}")

# ---------------------------------------------------------------------------
# Compare lower triangles element by element
# ---------------------------------------------------------------------------
# The GPU build_kernel_matrix returns a fully symmetric matrix (lower mirrored
# to upper).  The CPU kernel_gaussian_full_symm fills only the lower triangle;
# the upper triangle may be uninitialised or zero.  We compare lower tri only.

rows, cols = np.tril_indices(full)
cpu_lower = K_cpu_f32[rows, cols]
gpu_lower = K_gpu[rows, cols]

abs_diff = np.abs(gpu_lower - cpu_lower)
rel_diff = abs_diff / (np.abs(cpu_lower) + 1e-10)

print("\n--- Lower triangle comparison ---")
print(f"  Elements compared : {len(abs_diff)}")
print(f"  Max abs diff      : {abs_diff.max():.6e}")
print(f"  Mean abs diff     : {abs_diff.mean():.6e}")
print(f"  Max rel diff      : {rel_diff.max():.6e}")
print(f"  Mean rel diff     : {rel_diff.mean():.6e}")

# Breakdown by block
# K_EE: rows  0..N-1, cols  0..N-1
# K_FE: rows  N..end, cols  0..N-1
# K_FF: rows  N..end, cols  N..end
N_ = N_mol
EE_mask = (rows < N_) & (cols < N_)
FE_mask = (rows >= N_) & (cols < N_)
FF_mask = (rows >= N_) & (cols >= N_)

for name, mask in [("K_EE", EE_mask), ("K_FE", FE_mask), ("K_FF", FF_mask)]:
    if mask.sum() == 0:
        continue
    ad = abs_diff[mask]
    rd = rel_diff[mask]
    print(f"\n  {name} ({mask.sum()} elements):")
    print(f"    max abs diff = {ad.max():.6e}   mean abs diff = {ad.mean():.6e}")
    print(f"    max rel diff = {rd.max():.6e}   mean rel diff = {rd.mean():.6e}")

# Report worst offenders
worst_idx = np.argsort(abs_diff)[-10:][::-1]
print("\n--- Top-10 worst elements (lower triangle) ---")
print(f"{'row':>6}  {'col':>6}  {'cpu_f32':>14}  {'gpu_f32':>14}  {'abs_diff':>12}")
for idx in worst_idx:
    r, c = int(rows[idx]), int(cols[idx])
    block = "K_EE" if r < N_ and c < N_ else ("K_FE" if r >= N_ and c < N_ else "K_FF")
    print(
        f"{r:>6}  {c:>6}  {cpu_lower[idx]:>14.6e}  {gpu_lower[idx]:>14.6e}  "
        f"{abs_diff[idx]:>12.6e}  [{block}]"
    )

# Final verdict
tol = 1e-5  # float32 machine epsilon is ~1.2e-7; allow for accumulation
n_bad = (abs_diff > tol).sum()
if n_bad == 0:
    print(f"\nPASS: all {len(abs_diff)} lower-triangle elements agree to {tol:.0e}")
else:
    pct = 100 * n_bad / len(abs_diff)
    print(f"\nFAIL: {n_bad} elements ({pct:.2f}%) exceed abs_diff > {tol:.0e}")
    sys.exit(1)
