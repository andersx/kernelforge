#!/usr/bin/env python3
"""
KernelForge Benchmark Suite

A simple, single-run benchmark tool for KernelForge kernels and representations.
Usage:
    python benchmark/qmlbench.py all
    python benchmark/qmlbench.py representations
    python benchmark/qmlbench.py ethanol-kernels
    python benchmark/qmlbench.py qm7b-kernels
    python benchmark/qmlbench.py gdml-kernels
"""

import sys
import time
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import urllib.request
import zipfile
import tempfile

import numpy as np
import typer

from kernelforge._fchl19 import (
    generate_fchl_acsf,
    generate_fchl_acsf_and_gradients,
    flocal_kernel,
    flocal_kernel_symm,
    fgdml_kernel,
    fgdml_kernel_symm,
)

# ============================================================================
# CONSTANTS
# ============================================================================

__version__ = "0.1.8"
PROGRAM_NAME = "KernelForge Benchmarks"

# Data cache directory
CACHE_DIR = Path.home() / ".kernelforge" / "datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================


def load_ethanol_raw_data() -> np.ndarray:
    """Load raw ethanol data from sgdml.org. Auto-downloads if needed."""
    npz_path = CACHE_DIR / "ethanol_ccsd_t-train.npz"

    if not npz_path.exists():
        url = "https://sgdml.org/secure_proxy.php?file=data/npz/ethanol_ccsd_t.zip"
        print(f"  [Downloading ethanol dataset...]")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / "ethanol.zip"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status}: Failed to download from {url}"
                        )
                    with open(zip_path, "wb") as f:
                        f.write(response.read())

                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(tmpdir)
                extracted = list(Path(tmpdir).glob("*.npz"))[0]
                extracted.rename(npz_path)
        except Exception as e:
            print(f"  [Error downloading ethanol: {e}]", file=sys.stderr)
            raise

    return np.load(npz_path, allow_pickle=True)


def load_qm7b_raw_data() -> np.ndarray:
    """Load raw QM7b data from GitHub release. Auto-downloads if needed."""
    npz_path = CACHE_DIR / "qm7b_complete.npz"

    if not npz_path.exists():
        url = "https://github.com/andersx/kernelforge/releases/download/dataset-qm7b-v1.0/qm7b_complete.npz"
        print(f"  [Downloading QM7b dataset...]")

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"HTTP {response.status}: Failed to download from {url}"
                    )
                with open(npz_path, "wb") as f:
                    f.write(response.read())
        except Exception as e:
            print(f"  [Error downloading QM7b: {e}]", file=sys.stderr)
            raise

    return np.load(npz_path, allow_pickle=True)


def prepare_ethanol_fchl19(n_structures: int = 100) -> Dict:
    """Prepare FCHL19 representations and gradients for ethanol."""
    data = load_ethanol_raw_data()
    R = data["R"][:n_structures]
    z = data["z"]
    elements = [1, 6, 8]

    X = []
    dX = []
    for r in R:
        rep, grad = generate_fchl_acsf_and_gradients(r, z, elements=elements)
        X.append(rep)
        dX.append(grad)

    X = np.asarray(X)
    dX = np.asarray(dX)
    N = np.asarray([len(z) for _ in range(n_structures)], dtype=np.int32)
    Q = np.asarray([z for _ in range(n_structures)], dtype=np.int32)

    return {"X": X, "dX": dX, "N": N, "Q": Q, "z": z}


def prepare_qm7b_fchl19(n_structures: int = 100) -> Dict:
    """Prepare FCHL19 representations for QM7b."""
    data = load_qm7b_raw_data()
    R = data["R"][:n_structures]
    z_list = data["z"][:n_structures]
    elements = [1, 6, 7, 8, 16, 17]

    X_list = []
    N = []
    Q_list = []
    for i, r in enumerate(R):
        rep = generate_fchl_acsf(r, z_list[i], elements=elements)
        X_list.append(rep)
        N.append(len(rep))
        Q_list.append(z_list[i])

    N = np.asarray(N, dtype=np.int32)
    max_atoms = max(N)
    rep_dim = X_list[0].shape[1]

    # Pad to max_atoms
    X = np.zeros((len(X_list), max_atoms, rep_dim), dtype=np.float64)
    Q = np.zeros((len(Q_list), max_atoms), dtype=np.int32)

    for i, (x_i, q_i) in enumerate(zip(X_list, Q_list)):
        n_atoms = len(x_i)
        X[i, :n_atoms, :] = x_i
        Q[i, :n_atoms] = q_i

    return {"X": X, "N": N, "Q": Q}


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================


def benchmark_ethanol_fchl19_representations() -> Tuple[float, str]:
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


def benchmark_ethanol_fchl19_gradients() -> Tuple[float, str]:
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


def benchmark_qm7b_fchl19_representations() -> Tuple[float, str]:
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


def benchmark_qm7b_fchl19_gradients() -> Tuple[float, str]:
    """Benchmark FCHL19 gradient computation on QM7b (N=7211)."""
    data = load_qm7b_raw_data()
    R = data["R"]
    z_list = data["z"]
    elements = [1, 6, 7, 8, 16, 17]

    start = time.perf_counter()
    for r, z in zip(R, z_list):
        _, _ = generate_fchl_acsf_and_gradients(r, z, elements=elements)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "QM7b FCHL19 gradients (N=7211)"


def benchmark_kernel_symm_ethanol() -> Tuple[float, str]:
    """Benchmark symmetric local kernel on ethanol (N=100)."""
    data = prepare_ethanol_fchl19(100)
    X = data["X"][:100]
    Q = data["Q"][:100]
    N = data["N"][:100]
    sigma = 2.0

    start = time.perf_counter()
    K = flocal_kernel_symm(X, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "Local kernel symmetric (Ethanol, N=100)"


def benchmark_kernel_asymm_ethanol() -> Tuple[float, str]:
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
    K = flocal_kernel(X_train, X_test, Q_train, Q_test, N_train, N_test, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "Local kernel asymmetric (Ethanol, N=20)"


def benchmark_kernel_symm_qm7b() -> Tuple[float, str]:
    """Benchmark symmetric local kernel on QM7b (N=100)."""
    data = prepare_qm7b_fchl19(100)
    X = data["X"][:100]
    Q = data["Q"][:100]
    N = data["N"][:100]
    sigma = 2.0

    start = time.perf_counter()
    K = flocal_kernel_symm(X, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "Local kernel symmetric (QM7b, N=100)"


def benchmark_kernel_asymm_qm7b() -> Tuple[float, str]:
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
    K = flocal_kernel(X_train, X_test, Q_train, Q_test, N_train, N_test, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, "Local kernel asymmetric (QM7b, N=100)"


def benchmark_kernel_gdml_ethanol() -> Tuple[float, str]:
    """Benchmark symmetric GDML kernel on ethanol (N=100)."""
    n = 200
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    start = time.perf_counter()
    K = fgdml_kernel(X, X, dX, dX, Q, Q, N, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"GDML kernel symmetric (Ethanol, N=n)"


def benchmark_kernel_gdml_symm_ethanol() -> Tuple[float, str]:
    """Benchmark symmetric GDML kernel on ethanol (N=100)."""
    n = 200
    data = prepare_ethanol_fchl19(n)
    X = data["X"][:n]
    dX = data["dX"][:n]
    Q = data["Q"][:n]
    N = data["N"][:n]
    sigma = 2.5

    start = time.perf_counter()
    K = fgdml_kernel_symm(X, dX, Q, N, sigma)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, f"GDML kernel symmetric (Ethanol, N=n)"


# ============================================================================
# BENCHMARK REGISTRY
# ============================================================================

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
}

SUITES["all"] = []
for suite_benchmarks in SUITES.values():
    SUITES["all"].extend(suite_benchmarks)


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================


def get_system_info() -> Dict[str, str]:
    """Collect system information."""
    try:
        import psutil

        cpu_count = psutil.cpu_count(logical=True)
    except:
        cpu_count = 1

    try:
        import cpuinfo

        cpu_model = cpuinfo.get_cpu_info().get("brand_raw", "Unknown CPU")
    except:
        cpu_model = "Unknown CPU"

    py_version = f"{platform.python_version()} ({platform.python_implementation()})"
    platform_info = f"{platform.system()} ({platform.machine()})"

    return {
        "python": py_version,
        "platform": platform_info,
        "cpu": f"{cpu_model} ({cpu_count} cores)",
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def print_header(suite_name: str) -> None:
    """Print the benchmark header with system info."""
    info = get_system_info()

    print()
    print("╔" + "=" * 78 + "╗")
    print(f"║ {PROGRAM_NAME} v{__version__:<52} ║")
    print("╠" + "=" * 78 + "╣")
    print(f"║ Python:      {info['python']:<63} ║")
    print(f"║ Platform:    {info['platform']:<63} ║")
    print(f"║ CPU:         {info['cpu']:<63} ║")
    print(f"║ Hostname:    {info['hostname']:<63} ║")
    print(f"║ Timestamp:   {info['timestamp']:<63} ║")
    print("╠" + "=" * 78 + "╣")
    print(f"║ Suite:       {suite_name:<63} ║")
    print("╠" + "=" * 78 + "╣")
    print()


def print_result(bench_name: str, elapsed_ms: float, description: str) -> None:
    """Print a single benchmark result."""
    elapsed_s = elapsed_ms / 1000.0
    print(f"  {description:<50} {elapsed_s:>8.4f} s")


def print_footer(total_ms: float, count: int) -> None:
    """Print the footer with summary."""
    total_s = total_ms / 1000.0
    print()
    print("═" * 80)
    print()
    print(f"  Total time:  {total_s:.4f} s")
    print(f"  Benchmarks:  {count}")
    print(f"  Status:      OK ✓")
    print()


# ============================================================================
# CLI
# ============================================================================


app = typer.Typer(help=f"{PROGRAM_NAME} - Single-run benchmark suite for KernelForge")


@app.command()
def run(
    suite: str = typer.Argument(
        "all",
        help="Benchmark suite to run: all, representations, ethanol-kernels, qm7b-kernels, gdml-kernels",
    ),
):
    """Run KernelForge benchmarks."""
    if suite not in SUITES:
        typer.secho(f"Error: Unknown suite '{suite}'", fg=typer.colors.RED)
        typer.echo(f"Available suites: {', '.join(SUITES.keys())}")
        raise typer.Exit(1)

    suite_benchmarks = SUITES[suite]
    if not suite_benchmarks:
        typer.secho("Error: Empty suite", fg=typer.colors.RED)
        raise typer.Exit(1)

    # Print header
    print_header(suite)

    # Run benchmarks
    total_ms = 0
    results = []

    for bench_name in suite_benchmarks:
        if bench_name not in BENCHMARKS:
            typer.secho(f"Error: Unknown benchmark '{bench_name}'", fg=typer.colors.RED)
            raise typer.Exit(1)

        try:
            bench_func = BENCHMARKS[bench_name]
            elapsed_ms, description = bench_func()
            print_result(bench_name, elapsed_ms, description)
            total_ms += elapsed_ms
            results.append((bench_name, elapsed_ms, description))
        except Exception as e:
            typer.secho(
                f"Error running benchmark '{bench_name}': {e}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

    # Print footer
    print_footer(total_ms, len(results))


if __name__ == "__main__":
    app()
