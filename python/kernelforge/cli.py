import argparse
import sys
import tempfile
import time
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernelforge.fchl19_repr import generate_fchl_acsf, generate_fchl_acsf_and_gradients
from kernelforge.local_kernels import (
    kernel_gaussian,
    kernel_gaussian_hessian,
    kernel_gaussian_hessian_symm,
    kernel_gaussian_symm,
)
from kernelforge.kitchen_sinks import (
    rff_features,
    rff_features_elemental,
    rff_gradient_elemental,
    rff_gramian_elemental,
    rff_gramian_elemental_gradient,
)

PROGRAM_NAME = "KernelForge Benchmarks"

# Data cache directory for downloaded datasets
CACHE_DIR = Path.home() / ".kernelforge" / "datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_ethanol_raw_data() -> np.ndarray:
    """Load raw ethanol data from sgdml.org. Auto-downloads if needed."""
    npz_path = CACHE_DIR / "ethanol_ccsd_t-train.npz"

    if not npz_path.exists():
        url = "https://sgdml.org/secure_proxy.php?file=data/npz/ethanol_ccsd_t.zip"
        print("  [Downloading ethanol dataset...]")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / "ethanol.zip"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: Failed to download from {url}")
                    zip_path.write_bytes(response.read())

                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(tmpdir)
                extracted = next(iter(Path(tmpdir).glob("*.npz")))
                extracted.rename(npz_path)
        except Exception as e:
            print(f"  [Error downloading ethanol: {e}]", file=sys.stderr)
            raise

    return np.load(npz_path, allow_pickle=True)


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


def benchmark_kernel_symm_ethanol() -> tuple[float, str]:
    """Benchmark symmetric local kernel on ethanol (N=100)."""
    data = prepare_ethanol_fchl19(100)
    X = data["X"][:100]
    Q = data["Q"][:100]
    N = data["N"][:100]
    sigma = 2.0

    start = time.perf_counter()
    _ = kernel_gaussian_symm(X, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "Local kernel symmetric (Ethanol, N=100)"


def benchmark_kernel_asymm_ethanol() -> tuple[float, str]:
    """Benchmark asymmetric local kernel on ethanol (N=20, train-test split)."""
    data = prepare_ethanol_fchl19(20)
    X = data["X"][:20]
    Q = data["Q"][:20]
    N = data["N"][:20]
    sigma = 2.0

    n_train = 16
    X_train, X_test = X[:n_train], X[n_train:]
    Q_train, Q_test = Q[:n_train], Q[n_train:]
    N_train, N_test = N[:n_train], N[n_train:]

    start = time.perf_counter()
    _ = kernel_gaussian(X_train, X_test, Q_train, Q_test, N_train, N_test, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "Local kernel asymmetric (Ethanol, N=20)"


def benchmark_kernel_symm_qm7b() -> tuple[float, str]:
    """Benchmark symmetric local kernel on QM7b (N=100)."""
    data = prepare_qm7b_fchl19(100)
    X = data["X"][:100]
    Q = data["Q"][:100]
    N = data["N"][:100]
    sigma = 2.0

    start = time.perf_counter()
    _ = kernel_gaussian_symm(X, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "Local kernel symmetric (QM7b, N=100)"


def benchmark_kernel_asymm_qm7b() -> tuple[float, str]:
    """Benchmark asymmetric local kernel on QM7b (N=100, train-test split)."""
    data = prepare_qm7b_fchl19(100)
    X = data["X"][:100]
    Q = data["Q"][:100]
    N = data["N"][:100]
    sigma = 2.0

    n_train = 80
    X_train, X_test = X[:n_train], X[n_train:]
    Q_train, Q_test = Q[:n_train], Q[n_train:]
    N_train, N_test = N[:n_train], N[n_train:]

    start = time.perf_counter()
    _ = kernel_gaussian(X_train, X_test, Q_train, Q_test, N_train, N_test, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "Local kernel asymmetric (QM7b, N=100)"


def benchmark_kernel_gdml_ethanol() -> tuple[float, str]:
    """Benchmark symmetric GDML kernel on ethanol (N=100)."""
    n = 200
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    start = time.perf_counter()
    _ = kernel_gaussian_hessian(X, X, dX, dX, Q, Q, N, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "GDML kernel symmetric (Ethanol, N=n)"


def benchmark_kernel_gdml_symm_ethanol() -> tuple[float, str]:
    """Benchmark symmetric GDML kernel on ethanol (N=100)."""
    n = 200
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    start = time.perf_counter()
    _ = kernel_gaussian_hessian_symm(X, dX, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "GDML kernel symmetric (Ethanol, N=n)"


def benchmark_rff_features() -> tuple[float, str]:
    """Benchmark rff_features on synthetic data (N=1000, rep_size=200, D=4000)."""
    rng = np.random.default_rng(42)
    N, rep_size, D = 1000, 200, 4000
    X = rng.standard_normal((N, rep_size))
    W = rng.standard_normal((rep_size, D))
    b = rng.uniform(0.0, 2.0 * np.pi, D)

    start = time.perf_counter()
    _ = rff_features(X, W, b)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff_features (N={N}, rep_size={rep_size}, D={D})"


def _make_qm7b_elemental_inputs(n_mols: int, D: int, rng: np.random.Generator) -> dict[str, Any]:
    """Prepare inputs for elemental RFF benchmarks from QM7b-like data.

    Atomic numbers in the QM7b data are remapped to 0-based element indices
    matching the W/b stack order (elements = [1, 6, 7, 8, 16, 17]).
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

    W = rng.standard_normal((nelements, rep_size, D))
    b = rng.uniform(0.0, 2.0 * np.pi, (nelements, D))

    return {"X": X, "N": N, "Q": Q, "W": W, "b": b, "max_atoms": max_atoms}


def benchmark_rff_features_elemental() -> tuple[float, str]:
    """Benchmark rff_features_elemental on synthetic QM7b-like data (N=100 molecules)."""
    rng = np.random.default_rng(42)
    D = 2000
    inputs = _make_qm7b_elemental_inputs(100, D, rng)
    X, Q, W, b = inputs["X"], inputs["Q"], inputs["W"], inputs["b"]

    start = time.perf_counter()
    _ = rff_features_elemental(X, Q, W, b)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff_features_elemental (N=100, D={D}, QM7b-like)"


def benchmark_rff_gradient_elemental() -> tuple[float, str]:
    """Benchmark rff_gradient_elemental on synthetic QM7b-like data (N=10 molecules)."""
    rng = np.random.default_rng(42)
    D = 500
    n_mols = 10
    inputs = _make_qm7b_elemental_inputs(n_mols, D, rng)
    X, Q, W, b = inputs["X"], inputs["Q"], inputs["W"], inputs["b"]
    n_mols_actual, max_atoms, rep_size = X.shape
    # dX shape: (nmol, max_atoms, rep_size, max_atoms, 3)
    dX = rng.standard_normal((n_mols_actual, max_atoms, rep_size, max_atoms, 3))

    start = time.perf_counter()
    _ = rff_gradient_elemental(X, dX, Q, W, b)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff_gradient_elemental (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_gramian_elemental() -> tuple[float, str]:
    """Benchmark rff_gramian_elemental on synthetic QM7b-like data (N=100 molecules)."""
    rng = np.random.default_rng(42)
    D = 2000
    n_mols = 100
    inputs = _make_qm7b_elemental_inputs(n_mols, D, rng)
    X, Q, W, b = inputs["X"], inputs["Q"], inputs["W"], inputs["b"]
    n_mols_actual = X.shape[0]
    Y = rng.standard_normal(n_mols_actual)

    start = time.perf_counter()
    _ = rff_gramian_elemental(X, Q, W, b, Y)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff_gramian_elemental (N={n_mols}, D={D}, QM7b-like)"


def benchmark_rff_gramian_elemental_gradient() -> tuple[float, str]:
    """Benchmark rff_gramian_elemental_gradient on synthetic QM7b-like data (N=10)."""
    rng = np.random.default_rng(42)
    D = 500
    n_mols = 10
    inputs = _make_qm7b_elemental_inputs(n_mols, D, rng)
    X, Q, W, b = inputs["X"], inputs["Q"], inputs["W"], inputs["b"]
    n_mols_actual, max_atoms, rep_size = X.shape
    # dX shape: (nmol, max_atoms, rep_size, max_atoms, 3)
    dX = rng.standard_normal((n_mols_actual, max_atoms, rep_size, max_atoms, 3))
    Y = rng.standard_normal(n_mols_actual)
    # F is 1D, shape (ngrads,) where ngrads = 3 * sum(natoms per mol)
    ngrads = 3 * sum(len(q) for q in Q)
    F = rng.standard_normal(ngrads)

    start = time.perf_counter()
    _ = rff_gramian_elemental_gradient(X, dX, Q, W, b, Y, F)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"rff_gramian_elemental_gradient (N={n_mols}, D={D}, QM7b-like)"


BENCHMARKS = {
    "ethanol_fchl19_repr": benchmark_ethanol_fchl19_representations,
    "ethanol_fchl19_grad": benchmark_ethanol_fchl19_gradients,
    "qm7b_fchl19_repr": benchmark_qm7b_fchl19_representations,
    "qm7b_fchl19_grad": benchmark_qm7b_fchl19_gradients,
    "kernel_symm_ethanol": benchmark_kernel_symm_ethanol,
    "kernel_asymm_ethanol": benchmark_kernel_asymm_ethanol,
    "kernel_symm_qm7b": benchmark_kernel_symm_qm7b,
    "kernel_asymm_qm7b": benchmark_kernel_asymm_qm7b,
    "kernel_gdml_ethanol": benchmark_kernel_gdml_ethanol,
    "kernel_gdml_symm_ethanol": benchmark_kernel_gdml_symm_ethanol,
    "rff_features": benchmark_rff_features,
    "rff_features_elemental": benchmark_rff_features_elemental,
    "rff_gradient_elemental": benchmark_rff_gradient_elemental,
    "rff_gramian_elemental": benchmark_rff_gramian_elemental,
    "rff_gramian_elemental_gradient": benchmark_rff_gramian_elemental_gradient,
}

# Named benchmark suites
SUITES = {
    "representations": [
        "ethanol_fchl19_repr",
        "ethanol_fchl19_grad",
        "qm7b_fchl19_repr",
        "qm7b_fchl19_grad",
    ],
    "ethanol-kernels": [
        "kernel_symm_ethanol",
        "kernel_asymm_ethanol",
    ],
    "qm7b-kernels": [
        "kernel_symm_qm7b",
        "kernel_asymm_qm7b",
    ],
    "gdml-kernels": [
        "kernel_gdml_ethanol",
        "kernel_gdml_symm_ethanol",
    ],
    "kitchen-sinks": [
        "rff_features",
        "rff_features_elemental",
        "rff_gradient_elemental",
        "rff_gramian_elemental",
        "rff_gramian_elemental_gradient",
    ],
}

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
    print("-" * 80)
    print()


def print_result(bench_name: str, elapsed_ms: float, description: str) -> None:
    """Print a single benchmark result."""
    elapsed_s = elapsed_ms / 1000.0
    print(f"  {description:<50} {elapsed_s:>8.4f} s")


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
    """Main entry point for the qmlbench command."""
    parser = argparse.ArgumentParser(
        prog="qmlbench",
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
