# KernelForge - Optimized Kernels for ML

[![CI](https://github.com/andersx/kernelforge/actions/workflows/ci.yml/badge.svg)](https://github.com/andersx/kernelforge/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/kernelforge)](https://pypi.org/project/kernelforge/)
[![Python Versions](https://img.shields.io/pypi/pyversions/kernelforge?logo=python&logoColor=white)](https://pypi.org/project/kernelforge/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)](https://github.com/andersx/kernelforge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

I really only care about writing optimized kernel code, so this project will be completed as I find additional time... XD

I'm reviving this project to finish an old project using random Fourier features for kernel ML.


# Installation

## Quick Start (Recommended)

For most users, install from PyPI:

```bash
pip install kernelforge
```

This installs pre-compiled wheels with optimized BLAS libraries:
- **Linux**: OpenBLAS
- **macOS**: Apple Accelerate framework

**Requirements**: Python 3.10+

## Development Installation

### Linux

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install in editable mode with test dependencies
make install-linux

# Or manually:
CMAKE_ARGS="-DKF_USE_NATIVE=ON" uv pip install -e .[test] --verbose
```

### macOS

macOS requires Homebrew LLVM for OpenMP support:

```bash
# Install dependencies
brew install llvm libomp

# Create virtual environment
uv venv
source .venv/bin/activate

# Install in editable mode
make install-macos

# Or manually:
CMAKE_ARGS="-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DKF_USE_NATIVE=ON" uv pip install -e .[test] --verbose
```

**Note**: The `-DKF_USE_NATIVE=ON` flag enables `-march=native`/`-mcpu=native` optimizations for maximum performance on your specific CPU.

### Linux with CUDA

PyPI ships a **CPU** ``kernelforge`` wheel plus a Linux companion
``kernelforge-cuda`` that drops ``cuda_*.so`` into ``site-packages/kernelforge/``.

**From PyPI (Linux, CPython 3.11–3.14 for CUDA):**

```bash
uv pip install kernelforge                 # CPU (3.10–3.15)
uv pip install 'kernelforge[cuda]'         # + torch + kernelforge-cuda
```

Or install the companion explicitly:

```bash
uv pip install kernelforge kernelforge-cuda
```

Notes:

- The ``kernelforge-cuda`` companion wheel is **CPython 3.11–3.14**
  manylinux only. On 3.10/3.15, ``'kernelforge[cuda]'`` will not pull the
  companion (see the ``python_version`` marker in ``pyproject.toml``); use a
  local monorepo CUDA build instead.
- Default-index ``torch`` may be CPU-only. For GPU runs, install a CUDA build
  of PyTorch (e.g. from the PyTorch CUDA wheel index) and an NVIDIA driver
  compatible with that build.
- Release CUDA wheels are built on GHA ``build-wheels-cuda`` with
  ``manylinux_cuda`` (no GPU needed to *compile*).
- FCHL18 CUDA KRR supports ``--dtype float32`` / ``float64`` via ``kernelcli``
  (float32 uses GPU ``rfp_potrf`` end-to-end).

**Local monorepo / wheel smoke (developers):**

```bash
# One editable tree with CPU+CUDA (needs CUDA toolkit + torch):
source /opt/intel/oneapi/setvars.sh
make install-linux-mkl-ilp64-cuda   # passes -DKF_WITH_CUDA=ON

# Or build the split wheels and smoke-test imports in a fresh venv:
make demo-cuda-wheels
```

Requires a CUDA toolkit + PyTorch for local CUDA builds.

## Advanced: Custom BLAS/LAPACK Libraries

### Intel MKL (Linux)

```bash
# Install Intel oneAPI Base Toolkit
sudo apt install intel-oneapi-base-toolkit

# Set up environment
source /opt/intel/oneapi/setvars.sh

# Install (MKL will be auto-detected by CMake)
uv pip install -e .[test] --verbose

# Optional: Use Intel compilers
CC=icx CXX=icpx uv pip install -e .[test] --verbose
```

**Note**: In practice, GCC/G++ with OpenBLAS performs similarly to (or better than) Intel compilers with MKL. On macOS, LLVM with Accelerate framework is highly optimized for Apple Silicon.

## ASE molecular dynamics

Install the optional ASE extra:

```bash
pip install 'kernelforge[ase]'
```

Wrap a fitted force-capable model (for example ``CudaLocalKRRModel``) with
``KernelForgeCalculator``, or use ``run_md()`` / the ``kernelmd`` CLI:

```bash
kernelmd --model model.npz --structure start.xyz --steps 5000 --dt 0.5 \
         --temperature 300 --output traj.extxyz
```

See ``examples/run_md_rmd17_*.py`` and ``examples/run_opt_rmd17_*.py``.

## Timings
I've rewritten a few of the kernels from the original QML code completely in C++.
There are performance gains in most cases.
These are primarily due to better use of BLAS routines for calculating, for example, Gramian sub-matrices with chunked DGEMM/DSYRK calls, etc.
In the gradient and Hessian matrices there are also some algorithmic improvements and pre-computed terms.
Memory usage might be a bit higher, but this could be optimized with more fine-grained chunking if needed.
More is coming as I find the time ...

Some speedups vs the original QML code are shown below:

| Benchmark | QML [s] | Kernelforge [s] |
|:---------------|------------:|--------------------:|
| Upper triangle Gaussian kernel (16K x 16K) | 1.82 | 0.64 |
| 1K FCHL19 descriptors (1K) | N/A | 0.43 |
| 1K FCHL19 descriptors+jacobian (1K) | N/A | 0.62 |
| FCHL19 Local Gaussian scalar kernel (10K x 10K) | 76.81 | 18.15 |
| FCHL19 Local Gaussian gradient kernel (1K x 2700K) | 32.54 | 1.52 |
| FCHL19 Local Gaussian Hessian kernel (5400K x 5400K) | 29.68 | 2.05 |

## TODO list

Tracked in GitHub issue [#45](https://github.com/andersx/kernelforge/issues/45) (roadmap, open items, and stretch goals).
