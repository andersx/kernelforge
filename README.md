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

## Timings

| Function Name | QML [s] | Kernelforge [s] |
|:---------------|------------:|--------------------:|
| Upper triangle Gaussian kernel 16K x 16K| 1.82 | 0.64 |
| Kernel Jacobian |  |  |
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

#### Todos:
- Houskeeping:
  - [x] Pybind11 bindings and CMake build system
  - [x] Setup CI with GitHub Actions
  - [ ] Rewrite existing kernels to C++ (no Fortran)
  - [x] Setup GHA to build PyPI wheels
  - [x] Test Linux build matrices
  - [ ] Test MacOS build matrices
  - [ ] Test Windows build matrices
  - [ ] Add build for all Python version >=3.11
- Ensure correct linking with optimized BLAS/LAPACK libraries:
  - [x] OpenBLAS (Linux) <- also used in wheels
  - [x] AMD BLIS and libflame (Linux)
  - [x] MKL (Linux)
  - [x] Accelerated (MacOS)
- Add kernels:
  - [x] Gaussian kernel
  - [ ] Jacobian/gradient kernel
  - [ ] Optimized Jacobian kernel for single inference
  - [ ] Hessian kernel
  - [ ] GDML-like kernel
- Add random Fourier features kernel code
  - [ ] RFF kernel
  - [ ] RFF gradient kernel
  - [ ] RFF chunked DSYRK kernel
  - The same as above, just for Hadamard features when I find the time
- Add standard solvers:
  - [ ] Cholesky
  - [ ] QR and/or SVD for non-square matrices
- Add moleular descriptors with derivatives:
  - [ ] Coulomb matrix
  - [ ] FCHL19 + derivatives
  - [ ] GDML-like inverse-distance matrix + derivatives
- [ ] Plan structure for saving models for inference as `.npz` files
#### Stretch goals:
- [ ] Plan RDKit interface
- [ ] Plan Scikit-learn interface
- [ ] Plan ASE interface
