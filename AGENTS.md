## Build & Install

Always use the project's Makefile for building and installing.

Run `make install-linux-mkl-ilp64` on linux or `make install-macos-ilp64` on macOS.

On linux, use `source /opt/intel/oneapi/setvars.sh` to enable MKL for BLAS and LAPACK.

## Github

Never merge directly to the `master` branch. Always create a new branch and make a pull request for review.

When creating a new branch, make sure to pull the latest changes from `master` to avoid merge conflicts.

Before commiting, make sure the pre-commit hooks are passing and make sure make check and make test are passing locally.
