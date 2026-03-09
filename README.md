# Kernelforge - Optimized Kernels for ML

[![CI](https://github.com/andersx/kernelforge/actions/workflows/ci.yml/badge.svg)](https://github.com/andersx/kernelforge/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/kernelforge)](https://pypi.org/project/kernelforge/)
[![Python Versions](https://img.shields.io/pypi/pyversions/kernelforge?logo=python&logoColor=white)](https://pypi.org/project/kernelforge/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)](https://github.com/andersx/kernelforge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kernelforge is a high-performance Python library for machine learning kernels in quantum chemistry,
implemented in C++17 with Pybind11. It is a complete rewrite of the [QML](https://github.com/qmlcode/qml)
library, replacing all Fortran dependencies with modern C++ and achieving significant speedups through
better BLAS usage (chunked DGEMM/DSYRK), pre-computed terms, and OpenMP parallelization.

I really only care about writing optimized kernel code, so this project will be completed as I find
additional time... XD

## Features

- **Gaussian kernels**: scalar, Jacobian/gradient, Hessian, and full combined energy+force variants —
  all with symmetric RFP (Rectangular Full Packed) format variants for memory efficiency
- **FCHL18 analytical kernels**: scalar, Jacobian, Hessian, full, and RFP variants
- **Random Fourier Feature (RFF) approximations**: for global and local (elemental/atom-centered)
  descriptors, including gradient and chunked DSYRK gramian variants
- **Molecular representations with derivatives**: FCHL18, FCHL19 (ACSF-based), and inverse-distance
  matrix — all with Jacobians for force training
- **Global and local kernel variants**: molecule-level and atom-centered (FCHL19-style) interfaces
- **Linear algebra solvers**: Cholesky (regular and RFP format), QR, SVD
- **BLAS backends**: OpenBLAS, Intel MKL, Apple Accelerate; full ILP64 (64-bit integer) support

---

## Installation

### Quick Start

For most users, install pre-compiled wheels from PyPI:

```bash
pip install kernelforge
```

Wheels ship with optimized BLAS libraries:
- **Linux**: OpenBLAS
- **macOS**: Apple Accelerate framework

**Requirements**: Python 3.10+

---

### Development Installation

#### Linux (recommended: MKL + ILP64)

```bash
# Set up Intel MKL environment
source /opt/intel/oneapi/setvars.sh

# Create virtual environment and install in editable mode
uv venv
source .venv/bin/activate
make install-linux-mkl-ilp64
```

Other Linux targets: `install-linux-openblas-ilp64`, `install-linux-ilp64`, `install-linux`

#### macOS (recommended: ILP64)

Requires Homebrew LLVM for OpenMP support:

```bash
brew install llvm libomp

uv venv
source .venv/bin/activate
make install-macos-ilp64
```

---

### Advanced: Custom BLAS/LAPACK Libraries

#### Intel MKL (Linux)

Download and install the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html),
then source the environment script before building:

```bash
source /opt/intel/oneapi/setvars.sh
make install-linux-mkl-ilp64
```

**Note**: In practice, GCC/G++ with OpenBLAS performs similarly to (or better than) Intel compilers
with MKL. On macOS, LLVM with the Accelerate framework is highly optimized for Apple Silicon.

---

## Timings

Kernelforge rewrites the kernels from the original QML library entirely in C++. Speedups come
primarily from better use of BLAS routines (chunked DGEMM/DSYRK calls), pre-computed terms in
gradient and Hessian kernels, and OpenMP parallelization. Some benchmarks vs QML are shown below
(N/A = feature not available in QML):

| Benchmark | QML [s] | Kernelforge [s] |
|:---|---:|---:|
| Upper triangle Gaussian kernel (16K x 16K) | 1.82 | 0.64 |
| FCHL19 descriptors (1K) | N/A | 0.43 |
| FCHL19 descriptors + Jacobian (1K) | N/A | 0.62 |
| FCHL19 local Gaussian scalar kernel (10K x 10K) | 76.81 | 18.15 |
| FCHL19 local Gaussian gradient kernel (1K x 2700K) | 32.54 | 1.52 |
| FCHL19 local Gaussian Hessian kernel (5400K x 5400K) | 29.68 | 2.05 |

---

## Roadmap

Planned work includes: improving FCHL19 Jacobian kernel performance, completing the sGDML kernel,
adding a model serialization format (`.npz`), and higher-level interfaces compatible with RDKit, ASE,
and scikit-learn.
