# KernelForge - Optimized Lernels for ML

I really only care about writing optimized kernel code, so this project will be completed as I find additional time... XD 

I'm reviving this project to finish an old project using random Fourier features for kernel ML.


# Installation

```bash
conda env create -f environments/environment-dev.yml
pip install -e .
pytest -v -s
```
## PyPI installation

Install the requirements (e.g. the conda env above) and install from PyPI.
This should work on both MacOS and Linux/PC:

```bash
conda activate kernelforge-dev
pip install kernelforge
```
This will install pre-compiled wheels with gfortran and linked againts OpenBLAS on Linux and Accelerate on MacOS.
If you want to use MKL or other BLAS/LAPACK libraries, you need to compile from source, see below.


## Intel compilers and MKL

It is 2025 so you can `sudo apt get install intel-basekit` on Linux/PC to get the compilers and MKL.
Then set up the environment variables:
```bash
source /opt/intel/oneapi/setvars.sh
```
In this case, MKL will be autodetected by some CMake magic. If you additionally want to compile with Intel compilers, you can set the environment variables when running `pip install`:
```bash
CC=icx CXX=icpx FC=ifx make install
```

In my experience, GCC/G++/GFortran with OpenBLAS is very similar to Intel API alternatives in terms of performance, perhaps even better. 
On MacOS, GNU compilers with `-framework Accelerate` for BLAS/LAPACK is the default and is very fast on M-series macs.

## Timings

| Benchmark | QML [s] | Kernelforge [s] |
|:---------------|------------:|--------------------:|
| Upper triangle Gaussian kernel (16K x 16K) | 1.82 | 0.64 |
| 1K FCHL19 descriptors (1K) | ? | 0.43 |
| 1K FCHL19 descriptors+jacobian (1K) | ? | 0.62 |
| FCHL19 Local Gaussian scalar kernel (10K x 10K) | 76.81 | 18.15 |
| FCHL19 Local Gaussian gradient Kernel (1K x 2700K) | 32.54 | 1.52 |
| Kernel Hessian |  |  |

## TODO list

The goal is to remove pain-points of existing QML libraries
- Improved use of BLAS/LAPACK routines
- Removal of Fortran dependencies
  - No Fortran-ordered arrays
  - No Fortran compilers needed
- Simplified build system
  - No cooked F2PY/Meson build system
- Simplified entrypoints that are compatible with RDKit, ASE, Scikit-learn, etc.
  - A few high-level functions that do the most common tasks efficiently and correctly
- Efficient FCHL19 out-of-the-box
  - Fast training with random Fourier features
  - With derivatives
    - [ ] Stretch goal: Implement sFCHL19 for even faster training/inference

## Priority list for the next months:

- [x] Finish the inverse-distance kernel and its Jacobian
- [x] Make Pybind11 interface 
  - [ ] Finalize the C++ interface
  - [ ] Finalize the legacy Fortran interface (will be removed in the future)
- [x] Finish the Gaussian kernel
- [x] Notebook with rMD17 example
- [x] Finish the Jacobian and Hessian kernels
- [x] Notebook with rMD17 forces example
- FCHL19 support:
  - [x] Add FCHL19 descriptors
  - [x] Add FCHL19 kernels (local/elemental)
  - [x] Add FCHL19 descriptor with derivatives
  - [x] Add FCHL19 kernel Jacobian 
  - [ ] Add FCHL19 kernel Hessian (GDML-style)
  - [ ] Add FCHL19 full GPR kernel 
- [ ] Finish the random Fourier features kernel and its Jacobian
- [ ] Notebook with rMD17 random Fourier features examples

#### Todos:
- Houskeeping:
  - [x] Pybind11 bindings and CMake build system
  - [x] Setup CI with GitHub Actions
  - [ ] Rewrite existing kernels to C++ (no Fortran)
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
  - [ ] Gaussian kernel
  - [ ] Jacobian/gradient kernel
  - [ ] Optimized Jacobian kernel for single inference
  - [ ] Hessian kernel
  - [ ] GDML/GPR-like kernel
- Add random Fourier features kernel code
  - [ ] RFF kernel
  - [ ] RFF gradient kernel
  - [ ] RFF chunked DSYRK kernel
  - The same as above, just for Hadamard features when I find the time
- Add standard solvers:
  - [x] Cholesky in-place solver
    - [ ] L2-reg kwarg
    - [ ] Toggle destructive vs non-destructive
    - [ ] Toggle upper vs lower
  - [ ] QR and/or SVD for non-square matrices
- Add moleular descriptors with derivatives:
  - [ ] Coulomb matrix
  - [ ] FCHL19 + derivatives
  - [x] GDML-like inverse-distance matrix + derivatives
#### Stretch goals:
- [ ] Plan RDKit interface
- [ ] Plan Scikit-learn interface
- [ ] Plan ASE interface
