import argparse
import sys
import time
import urllib.request
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import kernelforge.fchl18_kernel as fchl18_kernel
import kernelforge.fchl18_repr as fchl18_repr
from kernelforge import global_kernels, invdist_repr
from kernelforge.fchl19_repr import generate_fchl_acsf, generate_fchl_acsf_and_gradients
from kernelforge.kernelmath import get_blas_info
from kernelforge.kitchen_sinks import (
    rff_features,
    rff_features_elemental,
    rff_full,
    rff_full_elemental,
    rff_full_gramian_elemental,
    rff_full_gramian_elemental_rfp,
    rff_full_gramian_symm,
    rff_full_gramian_symm_rfp,
    rff_gradient,
    rff_gradient_elemental,
    rff_gradient_gramian_elemental,
    rff_gradient_gramian_elemental_rfp,
    rff_gradient_gramian_symm,
    rff_gradient_gramian_symm_rfp,
    rff_gramian_elemental,
    rff_gramian_elemental_rfp,
    rff_gramian_symm,
    rff_gramian_symm_rfp,
)
from kernelforge.local_kernels import (
    kernel_gaussian,
    kernel_gaussian_full,
    kernel_gaussian_full_symm,
    kernel_gaussian_full_symm_rfp,
    kernel_gaussian_hessian,
    kernel_gaussian_hessian_symm,
    kernel_gaussian_hessian_symm_rfp,
    kernel_gaussian_jacobian,
    kernel_gaussian_jacobian_t,
    kernel_gaussian_symm,
    kernel_gaussian_symm_rfp,
)

PROGRAM_NAME = "Kernelforge Benchmarks"

# Data cache directory for downloaded datasets
CACHE_DIR = Path.home() / ".kernelforge" / "datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@cache
def load_ethanol_raw_data() -> dict[str, Any]:
    """Load raw ethanol MD17 data from sgdml.org (555K structures). Auto-downloads if needed.

    Returns a dict with keys: R (coordinates), z (atomic numbers), E (energies), F (forces).
    Data is eagerly loaded and cached in memory on first call to avoid slow disk I/O on
    subsequent calls.
    """
    npz_path = CACHE_DIR / "md17_ethanol.npz"

    if not npz_path.exists():
        url = "https://sgdml.org/secure_proxy.php?file=data/npz/md17_ethanol.npz"
        print("  [Downloading MD17 ethanol dataset...]")

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: Failed to download from {url}")
                npz_path.write_bytes(response.read())
        except Exception as e:
            print(f"  [Error downloading ethanol: {e}]", file=sys.stderr)
            raise

    # Eagerly load R and z into memory (not lazy — makes slicing fast)
    npz = np.load(npz_path, allow_pickle=True)
    return {
        "R": npz["R"],
        "z": npz["z"],
        "E": npz["E"],
        "F": npz["F"],
    }


@cache
def load_qm7b_raw_data() -> NDArray[Any]:
    """Load raw QM7b data from GitHub release. Auto-downloads if needed."""
    npz_path = CACHE_DIR / "qm7b_complete.npz"

    if not npz_path.exists():
        url = "https://github.com/andersx/kernelforge/releases/download/dataset-qm7b-v1.0/qm7b_complete.npz"
        print("  [Downloading QM7b dataset...]")

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: Failed to download from {url}")
                npz_path.write_bytes(response.read())
        except Exception as e:
            print(f"  [Error downloading QM7b: {e}]", file=sys.stderr)
            raise

    return np.load(npz_path, allow_pickle=True)


def prepare_ethanol_fchl19(n_structures: int = 100) -> dict[str, Any]:
    """Prepare FCHL19 representations and gradients for ethanol."""
    data = load_ethanol_raw_data()
    R = data["R"][:n_structures]
    z = data["z"]
    elements = [1, 6, 8]

    X_list = []
    dX_list = []
    for r in R:
        rep, grad = generate_fchl_acsf_and_gradients(r, z, elements=elements)
        X_list.append(rep)
        dX_list.append(grad)

    X = np.asarray(X_list)
    dX = np.asarray(dX_list)
    N = np.asarray([len(z) for _ in range(n_structures)], dtype=np.int32)
    Q = np.asarray([z for _ in range(n_structures)], dtype=np.int32)

    return {"X": X, "dX": dX, "N": N, "Q": Q, "z": z}


def prepare_qm7b_fchl19(n_structures: int = 100) -> dict[str, Any]:
    """Prepare FCHL19 representations for QM7b."""
    data = load_qm7b_raw_data()
    R = data["R"][:n_structures]
    z_list = data["z"][:n_structures]
    elements = [1, 6, 7, 8, 16, 17]

    X_list = []
    N_list = []
    Q_list = []
    for i, r in enumerate(R):
        rep = generate_fchl_acsf(r, z_list[i], elements=elements)
        X_list.append(rep)
        N_list.append(len(rep))
        Q_list.append(z_list[i])

    N = np.asarray(N_list, dtype=np.int32)
    max_atoms = max(N)
    rep_dim = X_list[0].shape[1]

    # Pad to max_atoms
    X = np.zeros((len(X_list), max_atoms, rep_dim), dtype=np.float64)
    Q = np.zeros((len(Q_list), max_atoms), dtype=np.int32)

    for i, (x_i, q_i) in enumerate(zip(X_list, Q_list, strict=True)):
        n_atoms = len(x_i)
        X[i, :n_atoms, :] = x_i
        Q[i, :n_atoms] = q_i

    return {"X": X, "N": N, "Q": Q}


def benchmark_ethanol_fchl19_representations() -> tuple[float, str]:
    """Benchmark FCHL19 representation generation on ethanol (N=1000)."""
    data = load_ethanol_raw_data()
    n = 10000
    R = data["R"][:n]
    z = data["z"]
    elements = [1, 6, 8]

    start = time.perf_counter()
    for r in R:
        _ = generate_fchl_acsf(r, z, elements=elements)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"Ethanol FCHL19 representations (N={n})"


def benchmark_ethanol_fchl19_gradients() -> tuple[float, str]:
    """Benchmark FCHL19 gradient computation on ethanol (N=1000)."""
    data = load_ethanol_raw_data()
    n = 10000
    R = data["R"][:n]
    z = data["z"]
    elements = [1, 6, 8]

    start = time.perf_counter()
    for r in R:
        _, _ = generate_fchl_acsf_and_gradients(r, z, elements=elements)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"Ethanol FCHL19 gradients (N={n})"


def benchmark_qm7b_fchl19_representations() -> tuple[float, str]:
    """Benchmark FCHL19 representation generation on QM7b (N=7211)."""
    data = load_qm7b_raw_data()
    R = data["R"]
    z_list = data["z"]
    elements = [1, 6, 7, 8, 16, 17]

    start = time.perf_counter()
    for i, r in enumerate(R):
        _ = generate_fchl_acsf(r, z_list[i], elements=elements)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"QM7b FCHL19 representations (N={len(R)})"


def benchmark_qm7b_fchl19_gradients() -> tuple[float, str]:
    """Benchmark FCHL19 gradient computation on QM7b (N=7211)."""
    data = load_qm7b_raw_data()
    R = data["R"]
    z_list = data["z"]
    elements = [1, 6, 7, 8, 16, 17]

    start = time.perf_counter()
    for r, z in zip(R, z_list, strict=True):
        _, _ = generate_fchl_acsf_and_gradients(r, z, elements=elements)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "QM7b FCHL19 gradients (N=7211)"


def benchmark_global_kernel_gaussian_symm() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_symm using ethanol inverse distance (N=1000, ~2s)."""
    data = load_ethanol_raw_data()
    n = 10000
    R = data["R"][:n]

    # Generate inverse distance representations
    X_list = [invdist_repr.inverse_distance_upper(r) for r in R]
    X = np.array(X_list)

    rep_size = X.shape[1]
    alpha = 0.5 / (rep_size * 2.0)

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_symm(X, alpha)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"global::kernel_gaussian_symm (N={n}, rep_size={rep_size}, Ethanol)"


def benchmark_global_kernel_gaussian_symm_rfp() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_symm_rfp (tiled DGEMM, no NxN buffer) using ethanol
    inverse distance."""
    data = load_ethanol_raw_data()
    n = 10000
    R = data["R"][:n]

    # Generate inverse distance representations
    X_list = [invdist_repr.inverse_distance_upper(r) for r in R]
    X = np.array(X_list)

    rep_size = X.shape[1]
    alpha = 0.5 / (rep_size * 2.0)

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_symm_rfp(X, alpha)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"global::kernel_gaussian_symm_rfp (N={n}, rep_size={rep_size}, Ethanol)"


def benchmark_global_kernel_gaussian() -> tuple[float, str]:
    """Benchmark global kernel_gaussian using ethanol inverse distance (N1=500, N2=500, ~2s)."""
    data = load_ethanol_raw_data()
    n = 10000
    R = data["R"][:n]

    # Generate inverse distance representations
    X_list = [invdist_repr.inverse_distance_upper(r) for r in R]
    X1 = np.array(X_list)
    X2 = np.array(X_list)

    rep_size = X1.shape[1]
    alpha = 0.5 / (rep_size * 2.0)

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian(X1, X2, alpha)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"global::kernel_gaussian (N={n}, rep_size={rep_size}, Ethanol)",
    )


def benchmark_global_kernel_gaussian_jacobian() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_jacobian using ethanol inverse distance (N=1000, ~2s)."""
    data = load_ethanol_raw_data()
    n = 5000
    R = data["R"][:n]
    z = data["z"]
    n_atoms = len(z)

    # Generate inverse distance representations and Jacobians
    X_list = []
    dX_list = []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list)
    dX = np.array(dX_list)

    # Use same data for both X1 and X2 (for comparability with symmetric kernel)
    X1 = X
    dX1 = dX
    X2 = X

    rep_size = X.shape[1]
    sigma = 2.0

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_jacobian(X1, dX1, X2, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"global::kernel_gaussian_jacobian"
        f" (N={n}, rep_size={rep_size}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_global_kernel_gaussian_jacobian_t() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_jacobian_t using ethanol inverse distance (N=1000, ~2s)."""
    data = load_ethanol_raw_data()
    n = 5000
    R = data["R"][:n]
    z = data["z"]
    n_atoms = len(z)

    # Generate inverse distance representations and Jacobians
    X_list = []
    dX_list = []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list)
    dX = np.array(dX_list)

    # Use same data for both X1 and X2 (for comparability with symmetric kernel)
    X1 = X
    X2 = X
    dX2 = dX

    rep_size = X.shape[1]
    sigma = 2.0

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_jacobian_t(X1, X2, dX2, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"global::kernel_gaussian_jacobian_t"
        f" (N={n}, rep_size={rep_size}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_global_kernel_gaussian_hessian_symm_rfp() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_hessian_symm_rfp using ethanol inverse distance
    (N=200, ~2s)."""
    data = load_ethanol_raw_data()
    n = 1000
    R = data["R"][:n]
    z = data["z"]
    n_atoms = len(z)

    # Generate inverse distance representations and Jacobians
    X_list = []
    dX_list = []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list)
    dX = np.array(dX_list)

    rep_size = X.shape[1]
    sigma = 2.5

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_hessian_symm_rfp(X, dX, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"global::kernel_gaussian_hessian_symm_rfp"
        f" (N={n}, rep_size={rep_size}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_global_kernel_gaussian_hessian_symm() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_hessian_symm using ethanol inverse distance (N=200, ~2s)."""
    data = load_ethanol_raw_data()
    n = 1000
    R = data["R"][:n]
    z = data["z"]
    n_atoms = len(z)

    # Generate inverse distance representations and Jacobians
    X_list = []
    dX_list = []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list)
    dX = np.array(dX_list)

    rep_size = X.shape[1]
    sigma = 2.5

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_hessian_symm(X, dX, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"global::kernel_gaussian_hessian_symm"
        f" (N={n}, rep_size={rep_size}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_global_kernel_gaussian_hessian() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_hessian using ethanol inverse distance (N=300, ~2s)."""
    data = load_ethanol_raw_data()
    n = 1000
    R = data["R"][:n]
    z = data["z"]
    n_atoms = len(z)

    # Generate inverse distance representations and Jacobians
    X_list = []
    dX_list = []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list)
    dX = np.array(dX_list)

    # Use same data for both X1 and X2 (for comparability with symmetric kernel)
    X1 = X
    dX1 = dX
    X2 = X
    dX2 = dX

    rep_size = X.shape[1]
    sigma = 2.5

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_hessian(X1, dX1, X2, dX2, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"global::kernel_gaussian_hessian (N={n}, rep_size={rep_size}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_global_kernel_gaussian_full() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_full (asymmetric, N=1000) using ethanol inverse distance."""
    data = load_ethanol_raw_data()
    n = 1000
    R = data["R"][:n]
    z = data["z"]
    n_atoms = len(z)

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list)
    dX = np.array(dX_list)
    rep_size = X.shape[1]
    sigma = 2.5

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_full(X, dX, X, dX, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"global::kernel_gaussian_full (N={n}, rep_size={rep_size}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_global_kernel_gaussian_full_symm() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_full_symm (symmetric, N=1000) using ethanol
    inverse distance."""
    data = load_ethanol_raw_data()
    n = 1000
    R = data["R"][:n]
    z = data["z"]
    n_atoms = len(z)

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list)
    dX = np.array(dX_list)
    rep_size = X.shape[1]
    sigma = 2.5

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_full_symm(X, dX, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"global::kernel_gaussian_full_symm"
        f" (N={n}, rep_size={rep_size}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_global_kernel_gaussian_full_symm_rfp() -> tuple[float, str]:
    """Benchmark global kernel_gaussian_full_symm_rfp (symmetric RFP, N=1000) using ethanol."""
    data = load_ethanol_raw_data()
    n = 1000
    R = data["R"][:n]
    z = data["z"]
    n_atoms = len(z)

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)

    X = np.array(X_list)
    dX = np.array(dX_list)
    rep_size = X.shape[1]
    sigma = 2.5

    start = time.perf_counter()
    _ = global_kernels.kernel_gaussian_full_symm_rfp(X, dX, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"global::kernel_gaussian_full_symm_rfp"
        f" (N={n}, rep_size={rep_size}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_local_kernel_gaussian_symm_rfp() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_symm_rfp (N=1000, ~2s)."""
    n = 1000
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.0

    start = time.perf_counter()
    _ = kernel_gaussian_symm_rfp(X, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_symm_rfp (Ethanol, N={n})"


def benchmark_local_kernel_gaussian_symm() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_symm (N=1200, ~2s)."""
    n = 1000
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.0

    start = time.perf_counter()
    _ = kernel_gaussian_symm(X, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_symm (Ethanol, N={n})"


def benchmark_local_kernel_gaussian() -> tuple[float, str]:
    """Benchmark local kernel_gaussian asymmetric (N=1000, ~2s)."""
    n = 1000
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.0

    # Use same data for both train and test (for comparability with symmetric kernel)
    X_train, X_test = X, X
    Q_train, Q_test = Q, Q
    N_train, N_test = N, N

    start = time.perf_counter()
    _ = kernel_gaussian(X_train, X_test, Q_train, Q_test, N_train, N_test, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian (Ethanol, N={n})"


def benchmark_local_kernel_gaussian_symm_qm7b() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_symm on QM7b (N=1000, ~2s)."""
    n = 1000
    data = prepare_qm7b_fchl19(n)
    X = data["X"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.0

    start = time.perf_counter()
    _ = kernel_gaussian_symm(X, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_symm (QM7b, N={n})"


def benchmark_local_kernel_gaussian_symm_rfp_qm7b() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_symm_rfp on QM7b (N=1000, ~2s)."""
    n = 1000
    data = prepare_qm7b_fchl19(n)
    X = data["X"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.0

    start = time.perf_counter()
    _ = kernel_gaussian_symm_rfp(X, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_symm_rfp (QM7b, N={n})"


def benchmark_cuda_local_kernel_gaussian_symm_rfp() -> tuple[float, str]:
    """Benchmark cuda_local::kernel_gaussian_symm_rfp (Ethanol, N=3000) — requires GPU."""
    import torch

    from kernelforge import cuda_local_kernels as _cuda_lk

    n = 3000
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n].astype("float32")
    Q = data["Q"][:n].astype("int32")
    N = data["N"][:n].astype("int32")
    sigma = 20.0

    X_cuda = torch.from_numpy(X).cuda()
    Q_cuda = torch.from_numpy(Q).cuda()
    N_cuda = torch.from_numpy(N).cuda()

    # warm-up
    _ = _cuda_lk.kernel_gaussian_symm_rfp(X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cuda_lk.kernel_gaussian_symm_rfp(X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_local::kernel_gaussian_symm_rfp (Ethanol, N={n})"


def benchmark_cuda_local_kernel_gaussian_symm_rfp_qm7b() -> tuple[float, str]:
    """Benchmark cuda_local::kernel_gaussian_symm_rfp (QM7b, N=3000) — requires GPU."""
    import torch

    from kernelforge import cuda_local_kernels as _cuda_lk

    n = 3000
    data = prepare_qm7b_fchl19(n)
    X = data["X"][:n].astype("float32")
    Q = data["Q"][:n].astype("int32")
    N = data["N"][:n].astype("int32")
    sigma = 2.0

    X_cuda = torch.from_numpy(X).cuda()
    Q_cuda = torch.from_numpy(Q).cuda()
    N_cuda = torch.from_numpy(N).cuda()

    # warm-up
    _ = _cuda_lk.kernel_gaussian_symm_rfp(X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cuda_lk.kernel_gaussian_symm_rfp(X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_local::kernel_gaussian_symm_rfp (QM7b, N={n})"


def benchmark_local_kernel_gaussian_qm7b() -> tuple[float, str]:
    """Benchmark local kernel_gaussian asymmetric on QM7b (N=1000, ~2s)."""
    n = 1000
    data = prepare_qm7b_fchl19(n)
    X = data["X"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.0

    # Use same data for both train and test (for comparability with symmetric kernel)
    X_train, X_test = X, X
    Q_train, Q_test = Q, Q
    N_train, N_test = N, N

    start = time.perf_counter()
    _ = kernel_gaussian(X_train, X_test, Q_train, Q_test, N_train, N_test, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian (QM7b, N={n})"


# ---------------------------------------------------------------------------
# CUDA global kernel benchmarks — require GPU + cuda_global_kernels build
# ---------------------------------------------------------------------------


def benchmark_cuda_global_kernel_gaussian_symm_rfp() -> tuple[float, str]:
    """Benchmark cuda_global::kernel_gaussian_symm_rfp (Ethanol invdist, N=5000) — requires GPU."""
    import torch

    from kernelforge import cuda_global_kernels as _cg

    data = load_ethanol_raw_data()
    n = 5000
    R = data["R"][:n]
    X = np.array([invdist_repr.inverse_distance_upper(r) for r in R]).astype(np.float32)
    sigma = 2.5

    X_cuda = torch.from_numpy(X).cuda()
    # warm-up
    _ = _cg.kernel_gaussian_symm_rfp(X_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cg.kernel_gaussian_symm_rfp(X_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    rep_size = X.shape[1]
    return elapsed, f"cuda_global::kernel_gaussian_symm_rfp (N={n}, rep_size={rep_size}, Ethanol)"


def benchmark_cuda_global_kernel_gaussian_full_symm() -> tuple[float, str]:
    """Benchmark cuda_global::kernel_gaussian_full_symm (Ethanol invdist+Jacobian, N=1000)."""
    import torch

    from kernelforge import cuda_global_kernels as _cg

    data = load_ethanol_raw_data()
    n = 1000
    R = data["R"][:n]
    n_atoms = len(data["z"])

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)
    X = np.array(X_list).astype(np.float32)
    dX = np.array(dX_list).astype(np.float32)
    sigma = 2.5

    X_cuda = torch.from_numpy(X).cuda()
    dX_cuda = torch.from_numpy(dX).cuda()
    # warm-up
    _ = _cg.kernel_gaussian_full_symm(X_cuda, dX_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cg.kernel_gaussian_full_symm(X_cuda, dX_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    rep_size = X.shape[1]
    D = dX.shape[1]
    return (
        elapsed,
        f"cuda_global::kernel_gaussian_full_symm (N={n}, rep={rep_size}, n_atoms={n_atoms}, D={D})",
    )


def benchmark_cuda_global_kernel_gaussian_full_symm_rfp() -> tuple[float, str]:
    """Benchmark cuda_global::kernel_gaussian_full_symm_rfp (Ethanol invdist+Jacobian, N=1000)."""
    import torch

    from kernelforge import cuda_global_kernels as _cg

    data = load_ethanol_raw_data()
    n = 1000
    R = data["R"][:n]
    n_atoms = len(data["z"])

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)
    X = np.array(X_list).astype(np.float32)
    dX = np.array(dX_list).astype(np.float32)
    sigma = 2.5

    X_cuda = torch.from_numpy(X).cuda()
    dX_cuda = torch.from_numpy(dX).cuda()
    # warm-up
    _ = _cg.kernel_gaussian_full_symm_rfp(X_cuda, dX_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cg.kernel_gaussian_full_symm_rfp(X_cuda, dX_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    rep_size = X.shape[1]
    D = dX.shape[1]
    return (
        elapsed,
        f"cuda_global::kernel_gaussian_full_symm_rfp "
        f"(N={n}, rep={rep_size}, n_atoms={n_atoms}, D={D})",
    )


def benchmark_cuda_global_kernel_gaussian_full_matvec() -> tuple[float, str]:
    """Benchmark cuda_global::kernel_gaussian_full_matvec inference (Ethanol, N=1000)."""
    import torch

    from kernelforge import cuda_global_kernels as _cg

    data = load_ethanol_raw_data()
    n = 1000
    R = data["R"][:n]

    X_list, dX_list = [], []
    for r in R:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
        X_list.append(x)
        dX_list.append(dx)
    X = np.array(X_list).astype(np.float32)
    dX = np.array(dX_list).astype(np.float32)
    sigma = 2.5

    N, M = X.shape
    D = dX.shape[1]

    X_cuda = torch.from_numpy(X).cuda()
    dX_cuda = torch.from_numpy(dX).cuda()
    rng = np.random.default_rng(42)
    alpha_E = torch.from_numpy(rng.standard_normal(N).astype(np.float32)).cuda()
    alpha_desc_F = torch.from_numpy(rng.standard_normal((N, M)).astype(np.float32)).cuda()

    # warm-up
    _ = _cg.kernel_gaussian_full_matvec(X_cuda, dX_cuda, X_cuda, alpha_E, alpha_desc_F, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cg.kernel_gaussian_full_matvec(X_cuda, dX_cuda, X_cuda, alpha_E, alpha_desc_F, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_global::kernel_gaussian_full_matvec (N={n}, rep={M}, D={D}, Ethanol)"


# ---------------------------------------------------------------------------
# CUDA local kernel benchmarks — require GPU + cuda_local_kernels build
# ---------------------------------------------------------------------------


def benchmark_cuda_local_kernel_gaussian() -> tuple[float, str]:
    """Benchmark cuda_local::kernel_gaussian_rect asymmetric (Ethanol, N=2000) — requires GPU."""
    import torch

    from kernelforge import cuda_local_kernels as _cl

    n = 2000
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n].astype(np.float32)
    Q = data["Q"][:n].astype(np.int32)
    N = data["N"][:n].astype(np.int32)
    sigma = 20.0

    X_cuda = torch.from_numpy(X).cuda()
    Q_cuda = torch.from_numpy(Q).cuda()
    N_cuda = torch.from_numpy(N).cuda()

    # warm-up
    _ = _cl.kernel_gaussian_rect(X_cuda, Q_cuda, N_cuda, X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cl.kernel_gaussian_rect(X_cuda, Q_cuda, N_cuda, X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_local::kernel_gaussian_rect (Ethanol, N={n})"


def benchmark_cuda_local_kernel_gaussian_qm7b() -> tuple[float, str]:
    """Benchmark cuda_local::kernel_gaussian_rect asymmetric (QM7b, N=700) — requires GPU."""
    import torch

    from kernelforge import cuda_local_kernels as _cl

    n = 700
    data = prepare_qm7b_fchl19(n)
    X = data["X"][:n].astype(np.float32)
    Q = data["Q"][:n].astype(np.int32)
    N = data["N"][:n].astype(np.int32)
    sigma = 2.0

    X_cuda = torch.from_numpy(X).cuda()
    Q_cuda = torch.from_numpy(Q).cuda()
    N_cuda = torch.from_numpy(N).cuda()

    # warm-up
    _ = _cl.kernel_gaussian_rect(X_cuda, Q_cuda, N_cuda, X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cl.kernel_gaussian_rect(X_cuda, Q_cuda, N_cuda, X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_local::kernel_gaussian_rect (QM7b, N={n})"


def benchmark_cuda_local_kernel_gaussian_symm() -> tuple[float, str]:
    """Benchmark cuda_local::kernel_gaussian_symm (Ethanol, N=2000) — requires GPU."""
    import torch

    from kernelforge import cuda_local_kernels as _cl

    n = 2000
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n].astype(np.float32)
    Q = data["Q"][:n].astype(np.int32)
    N = data["N"][:n].astype(np.int32)
    sigma = 20.0

    X_cuda = torch.from_numpy(X).cuda()
    Q_cuda = torch.from_numpy(Q).cuda()
    N_cuda = torch.from_numpy(N).cuda()

    # warm-up
    _ = _cl.kernel_gaussian_symm(X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cl.kernel_gaussian_symm(X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_local::kernel_gaussian_symm (Ethanol, N={n})"


def benchmark_cuda_local_kernel_gaussian_symm_qm7b() -> tuple[float, str]:
    """Benchmark cuda_local::kernel_gaussian_symm (QM7b, N=700) — requires GPU."""
    import torch

    from kernelforge import cuda_local_kernels as _cl

    n = 700
    data = prepare_qm7b_fchl19(n)
    X = data["X"][:n].astype(np.float32)
    Q = data["Q"][:n].astype(np.int32)
    N = data["N"][:n].astype(np.int32)
    sigma = 2.0

    X_cuda = torch.from_numpy(X).cuda()
    Q_cuda = torch.from_numpy(Q).cuda()
    N_cuda = torch.from_numpy(N).cuda()

    # warm-up
    _ = _cl.kernel_gaussian_symm(X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cl.kernel_gaussian_symm(X_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_local::kernel_gaussian_symm (QM7b, N={n})"


def benchmark_cuda_local_kernel_gaussian_full_symm() -> tuple[float, str]:
    """Benchmark cuda_local::kernel_gaussian_full_symm (Ethanol, N=500) — requires GPU."""
    import torch

    from kernelforge import cuda_local_kernels as _cl

    n = 500
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n].astype(np.float32)
    dX = data["dX"][:n].astype(np.float32)
    Q = data["Q"][:n].astype(np.int32)
    N = data["N"][:n].astype(np.int32)
    sigma = 20.0

    X_cuda = torch.from_numpy(X).cuda()
    dX_cuda = torch.from_numpy(dX).cuda()
    Q_cuda = torch.from_numpy(Q).cuda()
    N_cuda = torch.from_numpy(N).cuda()

    # warm-up
    _ = _cl.kernel_gaussian_full_symm(X_cuda, dX_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cl.kernel_gaussian_full_symm(X_cuda, dX_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    n_atoms = int(N[0])
    rep = X.shape[2]
    return (
        elapsed,
        f"cuda_local::kernel_gaussian_full_symm (Ethanol, N={n}, n_atoms={n_atoms}, rep={rep})",
    )


def benchmark_cuda_local_kernel_gaussian_full_symm_rfp() -> tuple[float, str]:
    """Benchmark cuda_local::kernel_gaussian_full_symm_rfp (Ethanol, N=500) — requires GPU."""
    import torch

    from kernelforge import cuda_local_kernels as _cl

    n = 500
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n].astype(np.float32)
    dX = data["dX"][:n].astype(np.float32)
    Q = data["Q"][:n].astype(np.int32)
    N = data["N"][:n].astype(np.int32)
    sigma = 20.0

    X_cuda = torch.from_numpy(X).cuda()
    dX_cuda = torch.from_numpy(dX).cuda()
    Q_cuda = torch.from_numpy(Q).cuda()
    N_cuda = torch.from_numpy(N).cuda()

    # warm-up
    _ = _cl.kernel_gaussian_full_symm_rfp(X_cuda, dX_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cl.kernel_gaussian_full_symm_rfp(X_cuda, dX_cuda, Q_cuda, N_cuda, sigma)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    n_atoms = int(N[0])
    rep = X.shape[2]
    return (
        elapsed,
        f"cuda_local::kernel_gaussian_full_symm_rfp (Ethanol, N={n}, n_atoms={n_atoms}, rep={rep})",
    )


def benchmark_cuda_local_compute_alpha_desc() -> tuple[float, str]:
    """Benchmark cuda_local::compute_alpha_desc (Ethanol, N=1000) — requires GPU."""
    import torch

    from kernelforge import cuda_local_kernels as _cl

    n = 1000
    data = prepare_ethanol_fchl19(n)
    dX = data["dX"][:n].astype(np.float32)
    N_arr = data["N"][:n].astype(np.int32)

    _nm, _max_atoms, rep, _ = dX.shape
    naq = int(np.sum(N_arr)) * 3

    dX_cuda = torch.from_numpy(dX).cuda()
    N_cuda = torch.from_numpy(N_arr).cuda()
    rng = np.random.default_rng(42)
    alpha_F = torch.from_numpy(rng.standard_normal(naq).astype(np.float32)).cuda()

    # warm-up
    _ = _cl.compute_alpha_desc(dX_cuda, N_cuda, alpha_F)
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cl.compute_alpha_desc(dX_cuda, N_cuda, alpha_F)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"cuda_local::compute_alpha_desc (Ethanol, N={n}, n_atoms={int(N_arr[0])}, rep={rep})",
    )


def benchmark_cuda_local_kernel_gaussian_full_matvec() -> tuple[float, str]:
    """Benchmark cuda_local::kernel_gaussian_full_matvec inference (Ethanol, N=500)."""
    import torch

    from kernelforge import cuda_local_kernels as _cl

    n = 500
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n].astype(np.float32)
    dX = data["dX"][:n].astype(np.float32)
    Q = data["Q"][:n].astype(np.int32)
    N_arr = data["N"][:n].astype(np.int32)
    sigma = 20.0

    nm, max_atoms, rep = X.shape
    rng = np.random.default_rng(42)
    alpha_E = torch.from_numpy(rng.standard_normal(nm).astype(np.float32)).cuda()
    alpha_desc_F = torch.from_numpy(
        rng.standard_normal((nm, max_atoms, rep)).astype(np.float32)
    ).cuda()

    X_cuda = torch.from_numpy(X).cuda()
    dX_cuda = torch.from_numpy(dX).cuda()
    Q_cuda = torch.from_numpy(Q).cuda()
    N_cuda = torch.from_numpy(N_arr).cuda()

    # warm-up
    _ = _cl.kernel_gaussian_full_matvec(
        X_cuda, dX_cuda, Q_cuda, N_cuda, X_cuda, Q_cuda, N_cuda, alpha_E, alpha_desc_F, sigma
    )
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cl.kernel_gaussian_full_matvec(
        X_cuda, dX_cuda, Q_cuda, N_cuda, X_cuda, Q_cuda, N_cuda, alpha_E, alpha_desc_F, sigma
    )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    n_atoms = int(N_arr[0])
    return (
        elapsed,
        f"cuda_local::kernel_gaussian_full_matvec (Ethanol, N={n}, n_atoms={n_atoms}, rep={rep})",
    )


def benchmark_local_kernel_gaussian_jacobian() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_jacobian (N=500, ~2s)."""
    n = 500
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    # Use same data for both train and test (for comparability with symmetric kernel)
    X_train, X_test = X, X
    dX_test = dX
    Q_train, Q_test = Q, Q
    N_train, N_test = N, N

    start = time.perf_counter()
    _ = kernel_gaussian_jacobian(X_train, X_test, dX_test, Q_train, Q_test, N_train, N_test, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_jacobian (Ethanol, N={n})"


def benchmark_local_kernel_gaussian_jacobian_t() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_jacobian_t (N=500, ~2s)."""
    n = 500
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    # Use same data for both train and test (for comparability with jacobian)
    X_train, X_test = X, X
    dX_train = dX
    Q_train, Q_test = Q, Q
    N_train, N_test = N, N

    start = time.perf_counter()
    _ = kernel_gaussian_jacobian_t(
        X_train, X_test, dX_train, Q_train, Q_test, N_train, N_test, sigma
    )
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_jacobian_t (Ethanol, N={n})"


def benchmark_local_kernel_gaussian_hessian_symm() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_hessian_symm (N=250, ~2s)."""
    n = 250
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    start = time.perf_counter()
    _ = kernel_gaussian_hessian_symm(X, dX, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_hessian_symm (Ethanol, N={n})"


def benchmark_local_kernel_gaussian_hessian() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_hessian asymmetric (N=250, ~2s)."""
    n = 250
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    # Use same data for both train and test (for comparability with symmetric kernel)
    X_train, X_test = X, X
    dX_train, dX_test = dX, dX
    Q_train, Q_test = Q, Q
    N_train, N_test = N, N

    start = time.perf_counter()
    _ = kernel_gaussian_hessian(
        X_train, X_test, dX_train, dX_test, Q_train, Q_test, N_train, N_test, sigma
    )
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_hessian (Ethanol, N={n})"


def benchmark_local_kernel_gaussian_hessian_symm_rfp() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_hessian_symm_rfp (N=250, ~2s)."""
    n = 250
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    start = time.perf_counter()
    _ = kernel_gaussian_hessian_symm_rfp(X, dX, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_hessian_symm_rfp (Ethanol, N={n})"


def benchmark_local_kernel_gaussian_full() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_full (asymmetric, N=250, ~2s)."""
    n = 250
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    start = time.perf_counter()
    _ = kernel_gaussian_full(X, X, dX, dX, Q, Q, N, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_full (Ethanol, N={n})"


def benchmark_local_kernel_gaussian_full_symm() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_full_symm (symmetric, N=250, ~2s)."""
    n = 250
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    start = time.perf_counter()
    _ = kernel_gaussian_full_symm(X, dX, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_full_symm (Ethanol, N={n})"


def benchmark_local_kernel_gaussian_full_symm_rfp() -> tuple[float, str]:
    """Benchmark local kernel_gaussian_full_symm_rfp (symmetric RFP, N=250, ~2s)."""
    n = 250
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    start = time.perf_counter()
    _ = kernel_gaussian_full_symm_rfp(X, dX, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"local::kernel_gaussian_full_symm_rfp (Ethanol, N={n})"


@cache
def _make_ethanol_inputs(
    n: int, D: int, seed: int = 42, include_dX: bool = False
) -> dict[str, Any]:
    """Prepare ethanol inverse-distance inputs for RFF benchmarks (cached by all args).

    Pass include_dX=True to also compute and cache the Jacobian dX array.
    """
    data = load_ethanol_raw_data()
    R = data["R"][:n]
    z = data["z"]
    n_atoms = len(z)

    if include_dX:
        X_list, dX_list = [], []
        for r in R:
            x, dx = invdist_repr.inverse_distance_upper_and_jacobian(r)
            X_list.append(x)
            dX_list.append(dx)
        X = np.array(X_list)
        dX = np.array(dX_list)
    else:
        X = np.array([invdist_repr.inverse_distance_upper(r) for r in R])
        dX = None

    rep_size = X.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((rep_size, D))
    b = rng.uniform(0.0, 2.0 * np.pi, D)

    result: dict[str, Any] = {"X": X, "W": W, "b": b, "rep_size": rep_size, "n_atoms": n_atoms}
    if include_dX and dX is not None:
        result["dX"] = dX
        result["ncoords"] = dX.shape[2]

    return result


def benchmark_rff_features() -> tuple[float, str]:
    """Benchmark rff_features on ethanol inverse-distance data (N=10000, D=N/2=5000)."""
    n = 10000
    D = n // 2
    inputs = _make_ethanol_inputs(n, D)
    X, W, b, rep_size = inputs["X"], inputs["W"], inputs["b"], inputs["rep_size"]

    start = time.perf_counter()
    _ = rff_features(X, W, b)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_features (N={n}, rep_size={rep_size}, D={D}, Ethanol)"


def benchmark_rff_gradient() -> tuple[float, str]:
    """Benchmark rff_gradient on ethanol inverse-distance data (N=3000, D=N/2=1500)."""
    n = 3000
    D = n // 2
    inputs = _make_ethanol_inputs(n, D, include_dX=True)
    X, dX, W, b = inputs["X"], inputs["dX"], inputs["W"], inputs["b"]
    rep_size, n_atoms = inputs["rep_size"], inputs["n_atoms"]

    start = time.perf_counter()
    _ = rff_gradient(X, dX, W, b)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"rff::rff_gradient (N={n}, rep_size={rep_size}, D={D}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_rff_gramian_symm() -> tuple[float, str]:
    """Benchmark rff_gramian_symm on ethanol inverse-distance data (N=10000, D=N/2=5000)."""
    n = 10000
    D = n // 2
    inputs = _make_ethanol_inputs(n, D)
    X, W, b, rep_size = inputs["X"], inputs["W"], inputs["b"], inputs["rep_size"]
    Y = np.random.default_rng(42).standard_normal(n)

    start = time.perf_counter()
    _ = rff_gramian_symm(X, W, b, Y)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_gramian_symm (N={n}, rep_size={rep_size}, D={D}, Ethanol)"


def benchmark_rff_full_gramian_symm() -> tuple[float, str]:
    """Benchmark rff_full_gramian_symm on ethanol inverse-distance data (N=3000, D=N/2=1500)."""
    n = 3000
    D = n // 2
    inputs = _make_ethanol_inputs(n, D, include_dX=True)
    X, dX, W, b = inputs["X"], inputs["dX"], inputs["W"], inputs["b"]
    rep_size, n_atoms, ncoords = inputs["rep_size"], inputs["n_atoms"], inputs["ncoords"]
    rng = np.random.default_rng(42)
    Y = rng.standard_normal(n)
    F = rng.standard_normal(n * ncoords)

    start = time.perf_counter()
    _ = rff_full_gramian_symm(X, dX, W, b, Y, F)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"rff::rff_full_gramian_symm"
        f" (N={n}, rep_size={rep_size}, D={D}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_rff_full() -> tuple[float, str]:
    """Benchmark rff_full on ethanol inverse-distance data (N=3000, D=N/2=1500)."""
    n = 3000
    D = n // 2
    inputs = _make_ethanol_inputs(n, D, include_dX=True)
    X, dX, W, b = inputs["X"], inputs["dX"], inputs["W"], inputs["b"]
    rep_size, n_atoms = inputs["rep_size"], inputs["n_atoms"]

    start = time.perf_counter()
    _ = rff_full(X, dX, W, b)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"rff::rff_full (N={n}, rep_size={rep_size}, D={D}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_rff_gradient_gramian_symm() -> tuple[float, str]:
    """Benchmark rff_gradient_gramian_symm on ethanol inverse-distance data (N=3000, D=N/2=1500)."""
    n = 3000
    D = n // 2
    inputs = _make_ethanol_inputs(n, D, include_dX=True)
    X, dX, W, b = inputs["X"], inputs["dX"], inputs["W"], inputs["b"]
    rep_size, n_atoms, ncoords = inputs["rep_size"], inputs["n_atoms"], inputs["ncoords"]
    F = np.random.default_rng(42).standard_normal(n * ncoords)

    start = time.perf_counter()
    _ = rff_gradient_gramian_symm(X, dX, W, b, F)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"rff::rff_gradient_gramian_symm"
        f" (N={n}, rep_size={rep_size}, D={D}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_rff_gramian_symm_rfp() -> tuple[float, str]:
    """Benchmark rff_gramian_symm_rfp on ethanol inverse-distance data (N=10000, D=N/2=5000)."""
    n = 10000
    D = n // 2
    inputs = _make_ethanol_inputs(n, D)
    X, W, b, rep_size = inputs["X"], inputs["W"], inputs["b"], inputs["rep_size"]
    Y = np.random.default_rng(42).standard_normal(n)

    start = time.perf_counter()
    _ = rff_gramian_symm_rfp(X, W, b, Y)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_gramian_symm_rfp (N={n}, rep_size={rep_size}, D={D}, Ethanol)"


def benchmark_rff_gradient_gramian_symm_rfp() -> tuple[float, str]:
    """Benchmark rff_gradient_gramian_symm_rfp on ethanol inverse-distance data (N=3000, D=1500)."""
    n = 3000
    D = n // 2
    inputs = _make_ethanol_inputs(n, D, include_dX=True)
    X, dX, W, b = inputs["X"], inputs["dX"], inputs["W"], inputs["b"]
    rep_size, n_atoms, ncoords = inputs["rep_size"], inputs["n_atoms"], inputs["ncoords"]
    F = np.random.default_rng(42).standard_normal(n * ncoords)

    start = time.perf_counter()
    _ = rff_gradient_gramian_symm_rfp(X, dX, W, b, F)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"rff::rff_gradient_gramian_symm_rfp"
        f" (N={n}, rep_size={rep_size}, D={D}, n_atoms={n_atoms}, Ethanol)",
    )


def benchmark_rff_full_gramian_symm_rfp() -> tuple[float, str]:
    """Benchmark rff_full_gramian_symm_rfp on ethanol inverse-distance data (N=3000, D=N/2=1500)."""
    n = 3000
    D = n // 2
    inputs = _make_ethanol_inputs(n, D, include_dX=True)
    X, dX, W, b = inputs["X"], inputs["dX"], inputs["W"], inputs["b"]
    rep_size, n_atoms, ncoords = inputs["rep_size"], inputs["n_atoms"], inputs["ncoords"]
    rng = np.random.default_rng(42)
    Y = rng.standard_normal(n)
    F = rng.standard_normal(n * ncoords)

    start = time.perf_counter()
    _ = rff_full_gramian_symm_rfp(X, dX, W, b, Y, F)
    elapsed = (time.perf_counter() - start) * 1000

    return (
        elapsed,
        f"rff::rff_full_gramian_symm_rfp"
        f" (N={n}, rep_size={rep_size}, D={D}, n_atoms={n_atoms}, Ethanol)",
    )


@cache
def _make_qm7b_elemental_inputs(
    n_mols: int, D: int, seed: int = 42, include_dX: bool = False
) -> dict[str, Any]:
    """Prepare inputs for elemental RFF benchmarks from QM7b-like data (cached by all args).

    Atomic numbers are remapped to 0-based element indices matching the W/b
    stack order (elements = [1, 6, 7, 8, 16, 17]). Pass include_dX=True to
    also generate and cache the dX gradient array.
    """
    elements = [1, 6, 7, 8, 16, 17]
    anum_to_idx = {anum: idx for idx, anum in enumerate(elements)}
    nelements = len(elements)

    data = prepare_qm7b_fchl19(n_mols)
    X = data["X"]
    N = data["N"]
    Q_padded = data["Q"]
    n_mols_actual, max_atoms, rep_size = X.shape

    # Build ragged Q with 0-based element indices (no padding entries)
    Q = [
        np.array([anum_to_idx[a] for a in Q_padded[i, : N[i]]], dtype=np.int32)
        for i in range(n_mols_actual)
    ]

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((nelements, rep_size, D))
    b = rng.uniform(0.0, 2.0 * np.pi, (nelements, D))

    result: dict[str, Any] = {"X": X, "N": N, "Q": Q, "W": W, "b": b, "max_atoms": max_atoms}

    if include_dX:
        result["dX"] = rng.standard_normal((n_mols_actual, max_atoms, rep_size, max_atoms, 3))

    return result


def benchmark_rff_features_elemental() -> tuple[float, str]:
    """Benchmark rff_features_elemental on synthetic QM7b-like data (N=4000 molecules)."""
    n_mols = 4000
    D = n_mols // 2
    inputs = _make_qm7b_elemental_inputs(n_mols, D)
    X, Q, W, b = inputs["X"], inputs["Q"], inputs["W"], inputs["b"]

    start = time.perf_counter()
    _ = rff_features_elemental(X, Q, W, b)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_features_elemental (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_gradient_elemental() -> tuple[float, str]:
    """Benchmark rff_gradient_elemental on synthetic QM7b-like data (N=600 molecules)."""
    n_mols = 600
    D = n_mols // 2
    inputs = _make_qm7b_elemental_inputs(n_mols, D, include_dX=True)
    X, Q, W, b, dX = inputs["X"], inputs["Q"], inputs["W"], inputs["b"], inputs["dX"]

    start = time.perf_counter()
    _ = rff_gradient_elemental(X, dX, Q, W, b)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_gradient_elemental (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_gramian_elemental() -> tuple[float, str]:
    """Benchmark rff_gramian_elemental on synthetic QM7b-like data (N=4000 molecules)."""
    n_mols = 4000
    D = n_mols // 2
    inputs = _make_qm7b_elemental_inputs(n_mols, D)
    X, Q, W, b = inputs["X"], inputs["Q"], inputs["W"], inputs["b"]
    n_mols_actual = X.shape[0]
    Y = np.random.default_rng(42).standard_normal(n_mols_actual)

    start = time.perf_counter()
    _ = rff_gramian_elemental(X, Q, W, b, Y)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_gramian_elemental (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_full_gramian_elemental() -> tuple[float, str]:
    """Benchmark rff_full_gramian_elemental on synthetic QM7b-like data (N=600)."""
    n_mols = 600
    D = n_mols // 2
    inputs = _make_qm7b_elemental_inputs(n_mols, D, include_dX=True)
    X, Q, W, b, dX = inputs["X"], inputs["Q"], inputs["W"], inputs["b"], inputs["dX"]
    n_mols_actual = X.shape[0]
    rng = np.random.default_rng(42)
    Y = rng.standard_normal(n_mols_actual)
    # F is 1D, shape (ngrads,) where ngrads = 3 * sum(natoms per mol)
    ngrads = 3 * sum(len(q) for q in Q)
    F = rng.standard_normal(ngrads)

    start = time.perf_counter()
    _ = rff_full_gramian_elemental(X, dX, Q, W, b, Y, F)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_full_gramian_elemental (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_full_elemental() -> tuple[float, str]:
    """Benchmark rff_full_elemental on synthetic QM7b-like data (N=600 molecules)."""
    n_mols = 600
    D = n_mols // 2
    inputs = _make_qm7b_elemental_inputs(n_mols, D, include_dX=True)
    X, Q, W, b, dX = inputs["X"], inputs["Q"], inputs["W"], inputs["b"], inputs["dX"]

    start = time.perf_counter()
    _ = rff_full_elemental(X, dX, Q, W, b)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_full_elemental (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_gradient_gramian_elemental() -> tuple[float, str]:
    """Benchmark rff_gradient_gramian_elemental on synthetic QM7b-like data (N=600)."""
    n_mols = 600
    D = n_mols // 2
    inputs = _make_qm7b_elemental_inputs(n_mols, D, include_dX=True)
    X, Q, W, b, dX = inputs["X"], inputs["Q"], inputs["W"], inputs["b"], inputs["dX"]
    rng = np.random.default_rng(42)
    ngrads = 3 * sum(len(q) for q in Q)
    F = rng.standard_normal(ngrads)

    start = time.perf_counter()
    _ = rff_gradient_gramian_elemental(X, dX, Q, W, b, F)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_gradient_gramian_elemental (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_gramian_elemental_rfp() -> tuple[float, str]:
    """Benchmark rff_gramian_elemental_rfp on synthetic QM7b-like data (N=4000 molecules)."""
    n_mols = 4000
    D = n_mols // 2
    inputs = _make_qm7b_elemental_inputs(n_mols, D)
    X, Q, W, b = inputs["X"], inputs["Q"], inputs["W"], inputs["b"]
    n_mols_actual = X.shape[0]
    Y = np.random.default_rng(42).standard_normal(n_mols_actual)

    start = time.perf_counter()
    _ = rff_gramian_elemental_rfp(X, Q, W, b, Y)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_gramian_elemental_rfp (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_gradient_gramian_elemental_rfp() -> tuple[float, str]:
    """Benchmark rff_gradient_gramian_elemental_rfp on synthetic QM7b-like data (N=600)."""
    n_mols = 600
    D = n_mols // 2
    inputs = _make_qm7b_elemental_inputs(n_mols, D, include_dX=True)
    X, Q, W, b, dX = inputs["X"], inputs["Q"], inputs["W"], inputs["b"], inputs["dX"]
    rng = np.random.default_rng(42)
    ngrads = 3 * sum(len(q) for q in Q)
    F = rng.standard_normal(ngrads)

    start = time.perf_counter()
    _ = rff_gradient_gramian_elemental_rfp(X, dX, Q, W, b, F)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_gradient_gramian_elemental_rfp (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_full_gramian_elemental_rfp() -> tuple[float, str]:
    """Benchmark rff_full_gramian_elemental_rfp on synthetic QM7b-like data (N=600)."""
    n_mols = 600
    D = n_mols // 2
    inputs = _make_qm7b_elemental_inputs(n_mols, D, include_dX=True)
    X, Q, W, b, dX = inputs["X"], inputs["Q"], inputs["W"], inputs["b"], inputs["dX"]
    n_mols_actual = X.shape[0]
    rng = np.random.default_rng(42)
    Y = rng.standard_normal(n_mols_actual)
    ngrads = 3 * sum(len(q) for q in Q)
    F = rng.standard_normal(ngrads)

    start = time.perf_counter()
    _ = rff_full_gramian_elemental_rfp(X, dX, Q, W, b, Y, F)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff::rff_full_gramian_elemental_rfp (N={n_mols}, D={D}, QM7b-like)"


_FCHL18_KERNEL_ARGS = {
    "two_body_width": 0.1,
    "two_body_scaling": 2.0,
    "two_body_power": 6.0,
    "three_body_width": 3.0,
    "three_body_scaling": 2.0,
    "three_body_power": 3.0,
    "cut_start": 0.5,
    "cut_distance": 1e6,
    "fourier_order": 2,
}


@cache
def _prepare_fchl18_qm7b(n: int) -> dict[str, Any]:
    """Generate FCHL18 representations for the first n QM7b molecules (cached)."""
    data = load_qm7b_raw_data()
    coords_list = list(data["R"][:n])
    z_list = [zi.astype(np.int32) for zi in data["z"][:n]]
    x, na, nn = fchl18_repr.generate(
        coords_list,
        z_list,
        max_size=23,
        cut_distance=_FCHL18_KERNEL_ARGS["cut_distance"],
    )
    return {"x": x, "na": na, "nn": nn}


def benchmark_fchl18_kernel_gaussian_symm_qm7b() -> tuple[float, str]:
    """Benchmark fchl18::kernel_gaussian_symm on QM7b (N=1000)."""
    n = 1000
    d = _prepare_fchl18_qm7b(n)
    x, na, nn = d["x"], d["na"], d["nn"]

    start = time.perf_counter()
    _ = fchl18_kernel.kernel_gaussian_symm(
        x,
        na,
        nn,
        sigma=2.5,
        two_body_width=_FCHL18_KERNEL_ARGS["two_body_width"],
        two_body_scaling=_FCHL18_KERNEL_ARGS["two_body_scaling"],
        two_body_power=_FCHL18_KERNEL_ARGS["two_body_power"],
        three_body_width=_FCHL18_KERNEL_ARGS["three_body_width"],
        three_body_scaling=_FCHL18_KERNEL_ARGS["three_body_scaling"],
        three_body_power=_FCHL18_KERNEL_ARGS["three_body_power"],
        cut_start=_FCHL18_KERNEL_ARGS["cut_start"],
        cut_distance=_FCHL18_KERNEL_ARGS["cut_distance"],
        fourier_order=int(_FCHL18_KERNEL_ARGS["fourier_order"]),
    )
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"fchl18::kernel_gaussian_symm (QM7b, N={n})"


def benchmark_fchl18_kernel_gaussian_qm7b() -> tuple[float, str]:
    """Benchmark fchl18::kernel_gaussian (asymmetric) on QM7b (N=1000 x 1000)."""
    n = 1000
    d = _prepare_fchl18_qm7b(n)
    x, na, nn = d["x"], d["na"], d["nn"]

    start = time.perf_counter()
    _ = fchl18_kernel.kernel_gaussian(
        x,
        x,
        na,
        na,
        nn,
        nn,
        sigma=2.5,
        two_body_width=_FCHL18_KERNEL_ARGS["two_body_width"],
        two_body_scaling=_FCHL18_KERNEL_ARGS["two_body_scaling"],
        two_body_power=_FCHL18_KERNEL_ARGS["two_body_power"],
        three_body_width=_FCHL18_KERNEL_ARGS["three_body_width"],
        three_body_scaling=_FCHL18_KERNEL_ARGS["three_body_scaling"],
        three_body_power=_FCHL18_KERNEL_ARGS["three_body_power"],
        cut_start=_FCHL18_KERNEL_ARGS["cut_start"],
        cut_distance=_FCHL18_KERNEL_ARGS["cut_distance"],
        fourier_order=int(_FCHL18_KERNEL_ARGS["fourier_order"]),
    )
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"fchl18::kernel_gaussian (QM7b, N={n}x{n})"


# ---------------------------------------------------------------------------
# cuda_fchl19_repr benchmarks
# ---------------------------------------------------------------------------

_QM7B_ELEMENTS = [1, 6, 7, 8, 16, 17]


def benchmark_cuda_fchl19_repr_qm7b() -> tuple[float, str]:
    """Benchmark cuda_fchl19_repr::generate_fchl_acsf (QM7b, all 7211 molecules) — GPU."""
    import torch

    from kernelforge import cuda_fchl19_repr as _cuda_repr

    data = load_qm7b_raw_data()
    R = data["R"]
    z_list = [z.astype(np.int32) for z in data["z"]]
    nm = len(R)
    elements = _QM7B_ELEMENTS
    idx_map = {z: i for i, z in enumerate(elements)}

    max_atoms = int(max(len(z) for z in z_list))
    coords_np = np.zeros((nm, max_atoms, 3), dtype=np.float32)
    Q_np = np.zeros((nm, max_atoms), dtype=np.int32)
    N_np = np.array([len(z) for z in z_list], dtype=np.int32)
    for m, (coords, z) in enumerate(zip(R, z_list, strict=False)):
        na = len(z)
        coords_np[m, :na, :] = np.asarray(coords, dtype=np.float32)
        Q_np[m, :na] = [idx_map[int(zi)] for zi in z]

    device = torch.device("cuda:0")
    coords_gpu = torch.from_numpy(coords_np).to(device)
    Q_gpu = torch.from_numpy(Q_np).to(device)
    N_gpu = torch.from_numpy(N_np).to(device)

    # warm-up
    _ = _cuda_repr.generate_fchl_acsf(coords_gpu, Q_gpu, N_gpu, nelements=len(elements))
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cuda_repr.generate_fchl_acsf(coords_gpu, Q_gpu, N_gpu, nelements=len(elements))
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_fchl19::generate_fchl_acsf (QM7b, N={nm}, batched)"


def benchmark_cuda_fchl19_repr_qm7b_n200() -> tuple[float, str]:
    """Benchmark cuda_fchl19_repr::generate_fchl_acsf (QM7b, 200 molecules) — GPU."""
    import torch

    from kernelforge import cuda_fchl19_repr as _cuda_repr

    n = 200
    data = load_qm7b_raw_data()
    R = data["R"][:n]
    z_list = [z.astype(np.int32) for z in data["z"][:n]]
    elements = _QM7B_ELEMENTS
    idx_map = {z: i for i, z in enumerate(elements)}

    max_atoms = int(max(len(z) for z in z_list))
    coords_np = np.zeros((n, max_atoms, 3), dtype=np.float32)
    Q_np = np.zeros((n, max_atoms), dtype=np.int32)
    N_np = np.array([len(z) for z in z_list], dtype=np.int32)
    for m, (coords, z) in enumerate(zip(R, z_list, strict=False)):
        na = len(z)
        coords_np[m, :na, :] = np.asarray(coords, dtype=np.float32)
        Q_np[m, :na] = [idx_map[int(zi)] for zi in z]

    device = torch.device("cuda:0")
    coords_gpu = torch.from_numpy(coords_np).to(device)
    Q_gpu = torch.from_numpy(Q_np).to(device)
    N_gpu = torch.from_numpy(N_np).to(device)

    # warm-up
    _ = _cuda_repr.generate_fchl_acsf(coords_gpu, Q_gpu, N_gpu, nelements=len(elements))
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cuda_repr.generate_fchl_acsf(coords_gpu, Q_gpu, N_gpu, nelements=len(elements))
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_fchl19::generate_fchl_acsf (QM7b, N={n}, batched)"


def _cuda_fchl19_batch(
    coords_list: list[NDArray[np.float64]], z_list: list[NDArray[np.int32]], elements: list[int]
) -> tuple[Any, Any, Any, Any]:
    """Create padded CUDA tensors for batched FCHL19 representation benchmarks."""
    import torch

    idx_map = {z: i for i, z in enumerate(elements)}
    nm = len(coords_list)
    max_atoms = int(max(len(z) for z in z_list))

    coords_np = np.zeros((nm, max_atoms, 3), dtype=np.float32)
    Q_np = np.zeros((nm, max_atoms), dtype=np.int32)
    N_np = np.array([len(z) for z in z_list], dtype=np.int32)

    for m, (coords, z) in enumerate(zip(coords_list, z_list, strict=False)):
        na = len(z)
        coords_np[m, :na, :] = np.asarray(coords, dtype=np.float32)
        Q_np[m, :na] = [idx_map[int(zi)] for zi in z]

    device = torch.device("cuda:0")
    return (
        torch.from_numpy(coords_np).to(device),
        torch.from_numpy(Q_np).to(device),
        torch.from_numpy(N_np).to(device),
        device,
    )


def benchmark_cuda_fchl19_grad_ethanol_n1000() -> tuple[float, str]:
    """Benchmark cuda_fchl19_repr::generate_fchl_acsf_and_gradients on 1000 ethanol geometries."""
    import torch

    from kernelforge import cuda_fchl19_repr as _cuda_repr

    n = 1000
    data = load_ethanol_raw_data()
    coords_list = [np.asarray(r, dtype=np.float64) for r in data["R"][:n]]
    z = np.asarray(data["z"], dtype=np.int32)
    z_list = [z for _ in range(n)]
    elements = [1, 6, 8]

    coords_gpu, Q_gpu, N_gpu, _ = _cuda_fchl19_batch(coords_list, z_list, elements)

    _ = _cuda_repr.generate_fchl_acsf_and_gradients(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements)
    )
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cuda_repr.generate_fchl_acsf_and_gradients(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements)
    )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_fchl19::generate_fchl_acsf_and_gradients (ethanol, N={n}, batched)"


def _azobenzene_geometry() -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Approximate planar trans-azobenzene geometry for synthetic gradient benchmarking."""
    ring_radius = 1.39
    ch_radius = 2.47
    left_center = np.array([-2.62, 0.0, 0.0], dtype=np.float64)
    right_center = np.array([2.62, 0.0, 0.0], dtype=np.float64)

    coords: list[list[float]] = [[-0.60, 0.0, 0.0], [0.60, 0.0, 0.0]]
    z: list[int] = [7, 7]

    left_angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    for angle_deg in left_angles:
        angle = np.deg2rad(angle_deg)
        coords.append(
            [
                float(left_center[0] + ring_radius * np.cos(angle)),
                float(left_center[1] + ring_radius * np.sin(angle)),
                0.0,
            ]
        )
        z.append(6)
    for angle_deg in left_angles[1:]:
        angle = np.deg2rad(angle_deg)
        coords.append(
            [
                float(left_center[0] + ch_radius * np.cos(angle)),
                float(left_center[1] + ch_radius * np.sin(angle)),
                0.0,
            ]
        )
        z.append(1)

    right_angles = [180.0, 120.0, 60.0, 0.0, 300.0, 240.0]
    for angle_deg in right_angles:
        angle = np.deg2rad(angle_deg)
        coords.append(
            [
                float(right_center[0] + ring_radius * np.cos(angle)),
                float(right_center[1] + ring_radius * np.sin(angle)),
                0.0,
            ]
        )
        z.append(6)
    for angle_deg in right_angles[1:]:
        angle = np.deg2rad(angle_deg)
        coords.append(
            [
                float(right_center[0] + ch_radius * np.cos(angle)),
                float(right_center[1] + ch_radius * np.sin(angle)),
                0.0,
            ]
        )
        z.append(1)

    return np.asarray(coords, dtype=np.float64), np.asarray(z, dtype=np.int32)


def benchmark_cuda_fchl19_grad_azobenzene_n1000() -> tuple[float, str]:
    """Benchmark cuda_fchl19_repr::generate_fchl_acsf_and_gradients on 1000 azobenzenes."""
    import torch

    from kernelforge import cuda_fchl19_repr as _cuda_repr

    n = 1000
    coords, z = _azobenzene_geometry()
    coords_list = [coords for _ in range(n)]
    z_list = [z for _ in range(n)]
    elements = [1, 6, 7]

    coords_gpu, Q_gpu, N_gpu, _ = _cuda_fchl19_batch(coords_list, z_list, elements)

    _ = _cuda_repr.generate_fchl_acsf_and_gradients(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements)
    )
    torch.cuda.synchronize()

    start = time.perf_counter()
    _ = _cuda_repr.generate_fchl_acsf_and_gradients(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements)
    )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"cuda_fchl19::generate_fchl_acsf_and_gradients (azobenzene, N={n}, batched)"


BENCHMARKS = {
    "ethanol_fchl19_repr": benchmark_ethanol_fchl19_representations,
    "ethanol_fchl19_grad": benchmark_ethanol_fchl19_gradients,
    "qm7b_fchl19_repr": benchmark_qm7b_fchl19_representations,
    "qm7b_fchl19_grad": benchmark_qm7b_fchl19_gradients,
    "global_kernel_gaussian_symm": benchmark_global_kernel_gaussian_symm,
    "global_kernel_gaussian_symm_rfp": benchmark_global_kernel_gaussian_symm_rfp,
    "global_kernel_gaussian": benchmark_global_kernel_gaussian,
    "global_kernel_gaussian_jacobian": benchmark_global_kernel_gaussian_jacobian,
    "global_kernel_gaussian_jacobian_t": benchmark_global_kernel_gaussian_jacobian_t,
    "global_kernel_gaussian_hessian_symm": benchmark_global_kernel_gaussian_hessian_symm,
    "global_kernel_gaussian_hessian_symm_rfp": benchmark_global_kernel_gaussian_hessian_symm_rfp,
    "global_kernel_gaussian_hessian": benchmark_global_kernel_gaussian_hessian,
    "global_kernel_gaussian_full": benchmark_global_kernel_gaussian_full,
    "global_kernel_gaussian_full_symm": benchmark_global_kernel_gaussian_full_symm,
    "global_kernel_gaussian_full_symm_rfp": benchmark_global_kernel_gaussian_full_symm_rfp,
    "local_kernel_gaussian_symm": benchmark_local_kernel_gaussian_symm,
    "local_kernel_gaussian_symm_rfp": benchmark_local_kernel_gaussian_symm_rfp,
    "local_kernel_gaussian": benchmark_local_kernel_gaussian,
    "local_kernel_gaussian_symm_qm7b": benchmark_local_kernel_gaussian_symm_qm7b,
    "local_kernel_gaussian_symm_rfp_qm7b": benchmark_local_kernel_gaussian_symm_rfp_qm7b,
    "cuda_local_kernel_gaussian_symm_rfp": benchmark_cuda_local_kernel_gaussian_symm_rfp,
    "cuda_local_kernel_gaussian_symm_rfp_qm7b": benchmark_cuda_local_kernel_gaussian_symm_rfp_qm7b,
    "cuda_local_kernel_gaussian_rect": benchmark_cuda_local_kernel_gaussian,
    "cuda_local_kernel_gaussian_rect_qm7b": benchmark_cuda_local_kernel_gaussian_qm7b,
    "cuda_local_kernel_gaussian_symm": benchmark_cuda_local_kernel_gaussian_symm,
    "cuda_local_kernel_gaussian_symm_qm7b": benchmark_cuda_local_kernel_gaussian_symm_qm7b,
    "cuda_local_kernel_gaussian_full_symm": benchmark_cuda_local_kernel_gaussian_full_symm,
    "cuda_local_kernel_gaussian_full_symm_rfp": benchmark_cuda_local_kernel_gaussian_full_symm_rfp,
    "cuda_local_compute_alpha_desc": benchmark_cuda_local_compute_alpha_desc,
    "cuda_local_kernel_gaussian_full_matvec": benchmark_cuda_local_kernel_gaussian_full_matvec,
    "cuda_global_kernel_gaussian_symm_rfp": benchmark_cuda_global_kernel_gaussian_symm_rfp,
    "cuda_global_kernel_gaussian_full_symm": benchmark_cuda_global_kernel_gaussian_full_symm,
    "cuda_global_kernel_gaussian_full_symm_rfp": (
        benchmark_cuda_global_kernel_gaussian_full_symm_rfp
    ),
    "cuda_global_kernel_gaussian_full_matvec": benchmark_cuda_global_kernel_gaussian_full_matvec,
    "local_kernel_gaussian_qm7b": benchmark_local_kernel_gaussian_qm7b,
    "local_kernel_gaussian_jacobian": benchmark_local_kernel_gaussian_jacobian,
    "local_kernel_gaussian_jacobian_t": benchmark_local_kernel_gaussian_jacobian_t,
    "local_kernel_gaussian_hessian_symm": benchmark_local_kernel_gaussian_hessian_symm,
    "local_kernel_gaussian_hessian": benchmark_local_kernel_gaussian_hessian,
    "local_kernel_gaussian_hessian_symm_rfp": benchmark_local_kernel_gaussian_hessian_symm_rfp,
    "local_kernel_gaussian_full": benchmark_local_kernel_gaussian_full,
    "local_kernel_gaussian_full_symm": benchmark_local_kernel_gaussian_full_symm,
    "local_kernel_gaussian_full_symm_rfp": benchmark_local_kernel_gaussian_full_symm_rfp,
    "rff_features": benchmark_rff_features,
    "rff_gradient": benchmark_rff_gradient,
    "rff_full": benchmark_rff_full,
    "rff_gramian_symm": benchmark_rff_gramian_symm,
    "rff_gradient_gramian_symm": benchmark_rff_gradient_gramian_symm,
    "rff_full_gramian_symm": benchmark_rff_full_gramian_symm,
    "rff_gramian_symm_rfp": benchmark_rff_gramian_symm_rfp,
    "rff_gradient_gramian_symm_rfp": benchmark_rff_gradient_gramian_symm_rfp,
    "rff_full_gramian_symm_rfp": benchmark_rff_full_gramian_symm_rfp,
    "rff_features_elemental": benchmark_rff_features_elemental,
    "rff_gradient_elemental": benchmark_rff_gradient_elemental,
    "rff_full_elemental": benchmark_rff_full_elemental,
    "rff_gramian_elemental": benchmark_rff_gramian_elemental,
    "rff_gradient_gramian_elemental": benchmark_rff_gradient_gramian_elemental,
    "rff_full_gramian_elemental": benchmark_rff_full_gramian_elemental,
    "rff_gramian_elemental_rfp": benchmark_rff_gramian_elemental_rfp,
    "rff_gradient_gramian_elemental_rfp": benchmark_rff_gradient_gramian_elemental_rfp,
    "rff_full_gramian_elemental_rfp": benchmark_rff_full_gramian_elemental_rfp,
    "fchl18_kernel_gaussian_symm_qm7b": benchmark_fchl18_kernel_gaussian_symm_qm7b,
    "fchl18_kernel_gaussian_qm7b": benchmark_fchl18_kernel_gaussian_qm7b,
    "cuda_fchl19_repr_qm7b": benchmark_cuda_fchl19_repr_qm7b,
    "cuda_fchl19_repr_qm7b_n200": benchmark_cuda_fchl19_repr_qm7b_n200,
    "cuda_fchl19_grad_ethanol_n1000": benchmark_cuda_fchl19_grad_ethanol_n1000,
    "cuda_fchl19_grad_azobenzene_n1000": benchmark_cuda_fchl19_grad_azobenzene_n1000,
}

# Named benchmark suites
SUITES = {
    "representations": [
        "ethanol_fchl19_repr",
        "ethanol_fchl19_grad",
        "qm7b_fchl19_repr",
        "qm7b_fchl19_grad",
    ],
    "global-kernels": [
        "global_kernel_gaussian_symm",
        "global_kernel_gaussian_symm_rfp",
        "global_kernel_gaussian",
        "global_kernel_gaussian_jacobian",
        "global_kernel_gaussian_jacobian_t",
        "global_kernel_gaussian_hessian_symm",
        "global_kernel_gaussian_hessian_symm_rfp",
        "global_kernel_gaussian_hessian",
        "global_kernel_gaussian_full",
        "global_kernel_gaussian_full_symm",
        "global_kernel_gaussian_full_symm_rfp",
    ],
    "local-kernels": [
        "local_kernel_gaussian_symm",
        "local_kernel_gaussian_symm_rfp",
        "local_kernel_gaussian",
        "local_kernel_gaussian_symm_qm7b",
        "local_kernel_gaussian_symm_rfp_qm7b",
        "local_kernel_gaussian_qm7b",
        "local_kernel_gaussian_jacobian",
        "local_kernel_gaussian_jacobian_t",
        "local_kernel_gaussian_hessian_symm",
        "local_kernel_gaussian_hessian",
        "local_kernel_gaussian_hessian_symm_rfp",
        "local_kernel_gaussian_full",
        "local_kernel_gaussian_full_symm",
        "local_kernel_gaussian_full_symm_rfp",
    ],
    "cuda-global-kernels": [
        "cuda_global_kernel_gaussian_symm_rfp",
        "cuda_global_kernel_gaussian_full_symm",
        "cuda_global_kernel_gaussian_full_symm_rfp",
        "cuda_global_kernel_gaussian_full_matvec",
    ],
    "cuda-local-kernels": [
        "cuda_local_kernel_gaussian_symm_rfp",
        "cuda_local_kernel_gaussian_symm_rfp_qm7b",
        "cuda_local_kernel_gaussian_rect",
        "cuda_local_kernel_gaussian_rect_qm7b",
        "cuda_local_kernel_gaussian_symm",
        "cuda_local_kernel_gaussian_symm_qm7b",
        "cuda_local_kernel_gaussian_full_symm",
        "cuda_local_kernel_gaussian_full_symm_rfp",
        "cuda_local_compute_alpha_desc",
        "cuda_local_kernel_gaussian_full_matvec",
    ],
    "global-rff": [
        "rff_features",
        "rff_gramian_symm",
        "rff_gramian_symm_rfp",
        "rff_gradient",
        "rff_gradient_gramian_symm",
        "rff_gradient_gramian_symm_rfp",
        "rff_full",
        "rff_full_gramian_symm",
        "rff_full_gramian_symm_rfp",
    ],
    "local-rff": [
        "rff_features_elemental",
        "rff_gramian_elemental",
        "rff_gramian_elemental_rfp",
        "rff_gradient_elemental",
        "rff_gradient_gramian_elemental",
        "rff_gradient_gramian_elemental_rfp",
        "rff_full_elemental",
        "rff_full_gramian_elemental",
        "rff_full_gramian_elemental_rfp",
    ],
}

SUITES["fchl18"] = [
    "fchl18_kernel_gaussian_symm_qm7b",
    "fchl18_kernel_gaussian_qm7b",
]

SUITES["cuda-representations"] = [
    "qm7b_fchl19_repr",
    "cuda_fchl19_repr_qm7b_n200",
    "cuda_fchl19_repr_qm7b",
    "cuda_fchl19_grad_ethanol_n1000",
    "cuda_fchl19_grad_azobenzene_n1000",
]

SUITES["all"] = []
for suite_name, suite_benchmarks in SUITES.items():
    if suite_name != "all":
        SUITES["all"].extend(suite_benchmarks)


def print_header(suite_name: str) -> None:
    """Print the benchmark header."""
    title = f"{PROGRAM_NAME} v{version('kernelforge')} | Suite: {suite_name}"
    print()
    print("-" * 80)
    print(title)
    print(f"BLAS: {get_blas_info()}")
    print("-" * 80)
    print()


def print_result(bench_name: str, elapsed_ms: float, description: str) -> None:
    """Print a single benchmark result."""
    elapsed_s = elapsed_ms / 1000.0
    print(f"  {description:<70} {elapsed_s:>8.4f} s")


def print_footer(total_ms: float, count: int) -> None:
    """Print the footer with summary."""
    total_s = total_ms / 1000.0
    print()
    print("-" * 80)
    print()
    print(f"  Total time:  {total_s:.4f} s")
    print(f"  Benchmarks:  {count}")
    print("  Status:      OK ✓")
    print()


def run(suite: str) -> None:
    """Run KernelForge benchmarks."""
    if suite not in SUITES:
        print(f"Error: Unknown suite '{suite}'", file=sys.stderr)
        print(f"Available suites: {', '.join(SUITES.keys())}", file=sys.stderr)
        sys.exit(1)

    suite_benchmarks = SUITES[suite]
    if not suite_benchmarks:
        print("Error: Empty suite", file=sys.stderr)
        sys.exit(1)

    print_header(suite)

    total_ms = 0.0
    results = []

    for bench_name in suite_benchmarks:
        bench_func = BENCHMARKS[bench_name]
        elapsed_ms, description = bench_func()
        print_result(bench_name, elapsed_ms, description)
        total_ms += elapsed_ms
        results.append((bench_name, elapsed_ms, description))

    # Print footer
    print_footer(total_ms, len(results))


def main() -> None:
    """Main entry point for the kernel-bench command."""
    parser = argparse.ArgumentParser(
        prog="kernelbench",
        description=f"{PROGRAM_NAME} - Single-run benchmark suite for KernelForge",
    )
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=list(SUITES.keys()),
        help="Benchmark suite to run (default: all)",
    )

    args = parser.parse_args()
    run(args.suite)


if __name__ == "__main__":
    main()
