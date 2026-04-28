"""Profile CudaLocalKRRModel vs LocalKRRModel step-by-step.

Run with:
    uv run python scripts/profile_local_krr.py

Prints wall-clock time for every sub-step so you can see where the
difference between the GPU and CPU paths actually lies.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from kernelforge import cuda_local_kernels as _ext
from kernelforge import local_kernels
from kernelforge.models.representations import compute_fchl19

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
CACHE = Path.home() / ".kernelforge" / "datasets"
NPZ = CACHE / "rmd17_ethanol_train_01.npz"
N_TRAIN = 100
ELEMENTS = [1, 6, 8]
SIGMA = 1.0
L2 = 1e-4

d = np.load(NPZ, allow_pickle=True)
z_fixed = d["nuclear_charges"].astype(np.int32)
coords = [d["coords"][i].astype(np.float64) for i in range(N_TRAIN)]
z_list = [z_fixed for _ in range(N_TRAIN)]
energies = d["energies"][:N_TRAIN].astype(np.float64)
forces = [d["forces"][i].astype(np.float64) for i in range(N_TRAIN)]
forces_np = np.array([f for f in forces])  # (nm, nat, 3)

torch.cuda.synchronize()  # warm-up


def tick(label: str, t0: float) -> float:
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  {label:<45s} {dt * 1000:8.2f} ms")
    return time.perf_counter()


# ============================================================
print("\n" + "=" * 60)
print("  GPU  CudaLocalKRRModel  (float32, N=%d)" % N_TRAIN)
print("=" * 60)

t0 = time.perf_counter()

# Step 1: FCHL19 representations
X, dX, Q_krr, _, N = compute_fchl19(coords, z_list, ELEMENTS, with_gradients=True, repr_params={})
t0 = tick("Step 1  compute_fchl19 (CPU, float64)", t0)

nm = X.shape[0]
max_atoms = X.shape[1]
rep_size = X.shape[2]
naq = int(np.sum(N) * 3)
BIG = nm + naq
print(f"         nm={nm}  max_atoms={max_atoms}  rep_size={rep_size}  naq={naq}  BIG={BIG}")

# Step 2: cast + upload
X_f32 = X.astype(np.float32)
dX_f32 = dX.astype(np.float32)
t0 = tick("Step 2a cast X/dX to float32 (CPU)", t0)

X_cuda = torch.from_numpy(np.ascontiguousarray(X_f32)).cuda()
dX_cuda = torch.from_numpy(np.ascontiguousarray(dX_f32)).cuda()
Q_cuda = torch.from_numpy(Q_krr.astype(np.int32)).cuda()
N_cuda = torch.from_numpy(N.astype(np.int32)).cuda()
torch.cuda.synchronize()
t0 = tick("Step 2b H2D upload X, dX, Q, N", t0)

print(f"         X_cuda  {tuple(X_cuda.shape)}  {X_cuda.nbytes / 1e6:.2f} MB")
print(f"         dX_cuda {tuple(dX_cuda.shape)}  {dX_cuda.nbytes / 1e6:.2f} MB")

# Step 3: kernel assembly
K = _ext.kernel_gaussian_full_symm(X_cuda, dX_cuda, Q_cuda, N_cuda, float(SIGMA))
torch.cuda.synchronize()
t0 = tick("Step 3  kernel_gaussian_full_symm (GPU)", t0)

# Step 3b: symmetrise
K = (K + K.T).mul_(0.5)
K.diagonal().add_(L2)
torch.cuda.synchronize()
t0 = tick("Step 3b symmetrise + add l2 (GPU)", t0)

print(f"         K_full {tuple(K.shape)}  {K.nbytes / 1e6:.2f} MB")

# Step 4: Cholesky
chol_L = torch.linalg.cholesky(K)
torch.cuda.synchronize()
t0 = tick("Step 4a cholesky (GPU, float32)", t0)

F_neg = -forces_np
rhs_f32 = np.concatenate([energies, F_neg.ravel()]).astype(np.float32)
rhs_gpu = torch.from_numpy(rhs_f32).cuda()
torch.cuda.synchronize()
t0 = tick("Step 4b build + upload RHS (GPU)", t0)

alpha_gpu = torch.cholesky_solve(rhs_gpu.unsqueeze(-1), chol_L).squeeze(-1)
torch.cuda.synchronize()
t0 = tick("Step 4c cholesky_solve (GPU, float32)", t0)

# Step 5: alpha_desc
alpha_F_cuda = alpha_gpu[nm:]
alpha_desc = _ext.compute_alpha_desc(dX_cuda, N_cuda, alpha_F_cuda)
torch.cuda.synchronize()
t0 = tick("Step 5  compute_alpha_desc (GPU)", t0)

# Step 6: download alpha for scoring
_ = alpha_gpu.cpu().numpy()
t0 = tick("Step 6  D2H download alpha (scoring)", t0)

# ============================================================
print("\n" + "=" * 60)
print("  CPU  LocalKRRModel  (float64, N=%d, sigma=%.1f)" % (N_TRAIN, SIGMA))
print("=" * 60)

t0 = time.perf_counter()

X64, dX64, Q64, _, N64 = compute_fchl19(
    coords, z_list, ELEMENTS, with_gradients=True, repr_params={}
)
t0 = tick("Step 1  compute_fchl19 (CPU, float64)", t0)

K_cpu = local_kernels.kernel_gaussian_full_symm(X64, dX64, Q64, N64, float(SIGMA))
t0 = tick("Step 2  kernel_gaussian_full_symm (CPU)", t0)

np.fill_diagonal(K_cpu, K_cpu.diagonal() + L2)
t0 = tick("Step 3  add l2 diagonal (CPU)", t0)

# Cholesky via numpy
import scipy.linalg as sl

chol_cpu = sl.cho_factor(K_cpu, lower=True)
t0 = tick("Step 4a cho_factor (CPU, float64)", t0)

rhs_f64 = np.concatenate([energies, (-forces_np).ravel()]).astype(np.float64)
alpha_cpu = sl.cho_solve(chol_cpu, rhs_f64)
t0 = tick("Step 4b cho_solve (CPU, float64)", t0)

alpha_desc_cpu = local_kernels.kernel_gaussian_local_compute_alpha_desc(
    dX64, Q64, N64, alpha_cpu[nm:]
)
t0 = tick("Step 5  compute_alpha_desc (CPU)", t0)
