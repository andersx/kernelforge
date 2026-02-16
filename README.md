# KernelForge - Optimized Kernels for ML

[![CI](https://github.com/andersx/kernelforge/actions/workflows/ci.yml/badge.svg)](https://github.com/andersx/kernelforge/actions/workflows/ci.yml)
[![Code Quality](https://github.com/andersx/kernelforge/actions/workflows/code-quality.yml/badge.svg)](https://github.com/andersx/kernelforge/actions/workflows/code-quality.yml)
[![PyPI version](https://badge.fury.io/py/kernelforge.svg)](https://pypi.org/project/kernelforge/)
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

## Advanced: Custom BLAS/LAPACK Libraries

### Intel MKL (Linux)

```bash
# Install Intel oneAPI Base Toolkit
sudo apt install intel-basekit

# Set up environment
source /opt/intel/oneapi/setvars.sh

# Install (MKL will be auto-detected by CMake)
uv pip install -e .[test] --verbose

# Optional: Use Intel compilers
CC=icx CXX=icpx uv pip install -e .[test] --verbose
```

**Note**: In practice, GCC/G++ with OpenBLAS performs similarly to (or better than) Intel compilers with MKL. On macOS, LLVM with Accelerate framework is highly optimized for Apple Silicon.

## Timings
I've rewritten a few of the kernels from the original QML code completely in C++.
There are performance gains in most cases.
These are primarily due to better use of BLAS routines for calculating, for example, Gramian sub-matrices with chunked DGEMM/DSYRK calls, etc.
In the gradient and Hessian matrices there are also some algorithmic improvement and pre-computed terms.
Memory usage might be a bit higher, but this could be optimized with more fine-graind chunking if needed.
More is coming as I find the time ...

Some speedups vs the original QML code are shown below:

| Benchmark | QML [s] | Kernelforge [s] |
|:---------------|------------:|--------------------:|
| Upper triangle Gaussian kernel (16K x 16K) | 1.82 | 0.64 |
| 1K FCHL19 descriptors (1K) | ? | 0.43 |
| 1K FCHL19 descriptors+jacobian (1K) | ? | 0.62 |
| FCHL19 Local Gaussian scalar kernel (10K x 10K) | 76.81 | 18.15 |
| FCHL19 Local Gaussian gradient kernel (1K x 2700K) | 32.54 | 1.52 |
| FCHL19 Local Gaussian Hessian kernel (5400K x 5400K) | 29.68 | 2.05 |

## TODO list

The goal is to remove pain-points of existing QML libraries
- Removal of Fortran dependencies
  - No Fortran-ordered arrays
  - No Fortran compilers needed
- Simplified build system
  - No cooked F2PY/Meson build system, just CMake and Pybind11
- Improved use of BLAS routines, with built-in chunking to avoid memory explosions
- Better use of pre-computed terms for single-point inference/MD kernels
- Low overhead with Pybind11 shims and better aligned memory?
- Simplified entrypoints that are compatible with RDKit, ASE, Scikit-learn, etc.
  - A few high-level functions that do the most common tasks efficiently and correctly
- Efficient FCHL19 out-of-the-box
  - Fast training with random Fourier features
  - With derivatives


## Priority list for the next months:

- [x] Finish the inverse-distance kernel and its Jacobian
- [x] Make Pybind11 interface
  - [ ] Finalize the C++ interface
- [x] Finish the Gaussian kernel
- [x] Notebook with rMD17 example
- [x] Finish the Jacobian and Hessian kernels
- [x] Notebook with rMD17 forces example
- FCHL19 support:
  - [x] Add FCHL19 descriptors
  - [x] Add FCHL19 kernels (local/elemental)
  - [x] Add FCHL19 descriptor with derivatives
  - [x] Add FCHL19 kernel Jacobian
  - [x] Add FCHL19 kernel Hessian (GDML-style)
  - [ ] Improve FCHL19 kernel Jacobian performance (its poor)
- Finish the random Fourier features kernel and its Jacobian
  - [ ] Parallel random basis sampler
  - [ ] RFF kernel for global descriptors
  - [ ] SVD and QR solvers for rectangular matrices
  - [ ] RFF kernel for local descriptors (FCHL19)
  - [ ] RFF kernels with Cholesky solver and chunked DSYRK kernel updates
  - [ ] RFF kernels with RFP format with chunked DSFRK kernel updates
  - [ ] RFF kernel Jacobian for global descriptors
  - [ ] RFF kernel Jacobian for local descriptors (FCHL19)
- [ ] Notebook with rMD17 random Fourier features examples

- Science:
  - Benchmark full kernel vs RFF on rMD17 and QM7b and QM9
  - Both FCHL19 and inverse-distance matrix

#### Todos:
- Houskeeping:
  - [x] Pybind11 bindings and CMake build system
  - [x] Setup CI with GitHub Actions
  - [x] Rewrite existing kernels to C++ (no Fortran)
  - [x] Setup GHA to build PyPI wheels
  - [x] Test Linux build matrices
  - [x] Test MacOS build matrices
  - [ ] Test Windows build matrices
  - [x] Add build for all Python version >=3.11
  - [ ] Plan structure for saving models for inference as `.npz` files
- Ensure correct linking with optimized BLAS/LAPACK libraries:
  - [x] OpenBLAS (Linux) <- also used in wheels
  - [x] MKL (Linux)
  - [x] Accelerate (MacOS)
- Add global kernels:
  - [x] Gaussian kernel
  - [x] Jacobian/gradient kernel
  - [ ] Optimized Jacobian kernel for single inference
  - [x] Hessian kernel
  - [x] GDML-like kernel
  - [ ] Full GPR kernel
- Add local kernels:
  - [x] Gaussian kernel
  - [x] Jacobian/gradient kernel
  - [x] Optimized Jacobian kernel for single inference
  - [x] Hessian kernel (GDML-style)
  - [ ] Full GPR kernel
  - [ ] Optimized GPR kernel with pre-computed terms for single inference/MD
- Add random Fourier features kernel code:
  - [ ] Fourier-basis sampler
  - [ ] RFF kernel
  - [ ] RFF gradient kernel
  - [ ] RFF chunked DSYRK kernel
  - [ ] Optimized RFF gradient kernel for single inference/MD
  - The same as above, just for Hadamard features when I find the time?
- GDML and sGDML kernels:
  - [x] Inverse-distance matrix descriptor
  - [ ] Packed Jacobian for inverse-distance matrix
  - [x] GDML kernel (brute-force implemented)
  - [ ] sGDML kernel (brute-force implemented)
  - [ ] Full GPR kernel
  - [ ] Optimized GPR kernel with pre-computed terms for single inference/MD
- FCHL18 support:
  - [ ] Complete rewrite of FCHL18 analytical scalar kernel in C++
  - [ ] Stretch goal 1: Add new analytical FCHL18 kernel Jacobian
  - [ ] Stretch goal 2: Add new analytical FCHL18 kernel Hessian (+GPR/GDML-style)
  - [ ] Stretch goal 3: Attempt to optimize hyperparameters and cut-off functions
- Add standard solvers:
  - [x] Cholesky in-place solver
    - [x] L2-reg kwarg
    - [x] Toggle destructive vs non-destructive
  - [ ] QR and/or SVD for non-square matrices
- Add moleular descriptors with derivatives:
  - [ ] Coulomb matrix + misc variants without derivatives
  - [x] FCHL19 + derivatives
  - [x] GDML-like inverse-distance matrix + derivatives
#### Stretch goals:
- [ ] Plan RDKit interface
- [ ] Plan Scikit-learn interface
- [ ] Plan ASE interface
