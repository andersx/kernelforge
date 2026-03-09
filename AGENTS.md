# KernelForge — Agent Instructions

## Build & Install

Always use the Makefile. Never invoke CMake or pip directly.

### Linux (recommended: MKL + ILP64)

```bash
source /opt/intel/oneapi/setvars.sh   # must be run first to enable MKL
make install-linux-mkl-ilp64
```

Other Linux targets: `install-linux-openblas-ilp64`, `install-linux-ilp64`, `install-linux`

### macOS (recommended: ILP64)

Requires Homebrew LLVM and OpenMP:

```bash
brew install llvm libomp
make install-macos-ilp64
```

The Makefile hard-codes `/opt/homebrew/opt/llvm/bin/clang{++}` as the compiler.

---

## Testing

```bash
make test           # full test suite (uv run pytest)
```

To skip slow/integration tests (matches CI behavior):

```bash
uv run pytest -k "not slow" -x
```

Test markers defined in `pyproject.toml`: `integration` (requires external datasets), `slow`.
All test files live in `tests/`. When adding new functionality, add a corresponding test.

---

## Code Quality

Run all checks before committing:

```bash
make check          # format + lint + typecheck
```

Individual targets:

| Command          | Tool              | Scope                             |
|------------------|-------------------|-----------------------------------|
| `make format`    | ruff + clang-format | auto-fixes Python and C++       |
| `make lint`      | ruff              | lint-only, no auto-fix            |
| `make typecheck` | ty                | Python type checking              |
| `make tidy`      | clang-tidy        | C++ static analysis               |

Pre-commit hooks run automatically on `git commit` and auto-fix some issues (ruff, clang-format).
CI gates: `make check` (code-quality workflow) and `make test` (ci workflow) must both pass on every PR.

---

## Git & GitHub Workflow

- **Never commit directly to `master`**. Always work on a new branch.
- Before creating a branch, pull the latest `master`: `git pull origin master`
- Open a PR for all changes. CI must pass before merging.
- Pre-commit hooks must pass. Run `make check` and `make test` locally before pushing.

---

## Architecture Overview

The library is a Python package (`kernelforge`) backed by C++17 extension modules built with
Pybind11 + scikit-build-core.

### Module groups

| Group | C++ sources | Python module | Purpose |
|-------|-------------|---------------|---------|
| Global kernels | `global_kernels.{cpp,hpp}` | `global_kernels` | Gaussian kernels for molecule-level descriptors (scalar, Jacobian, Hessian, full energy+force; RFP variants) |
| Local kernels | `local_kernels.{cpp,hpp}` | `local_kernels` | Same kernels for atom-centered descriptors (takes `Q`, `N` atom-count args) |
| FCHL18 kernel | `fchl18_kernel*.{cpp,hpp}` | `fchl18_kernel` | Full FCHL18 analytical kernel suite (scalar, Jacobian, Hessian, full, RFP) |
| Representations | `fchl18_repr.*`, `fchl19_repr.*`, `invdist_repr.*` | `fchl18_repr`, `fchl19_repr`, `invdist_repr` | Molecular descriptor generators + Jacobians |
| RFF / kitchen sinks | `rff_features.*`, `rff_elemental.*` | `kitchen_sinks` | Random Fourier Feature approximations to all kernels |
| Math / solvers | `math.{cpp,hpp}` | `kernelmath` | Cholesky, QR, SVD solvers; RFP format conversion; `get_blas_info()` |

Python shim: `python/kernelforge/__init__.py` re-exports everything from the C++ extension modules.
Type stubs: `python/kernelforge/*.pyi` — one per extension module.

---

## C++ Conventions

- **Standard**: C++17. No exceptions beyond what Pybind11 uses.
- **BLAS integer type**: Always use `blas_int` (defined in `src/blas_int.h`), never plain `int` for
  BLAS/LAPACK calls. This resolves to `int64_t` under ILP64 and `int` under LP64.
- **Formatting**: clang-format enforces Google style, 4-space indent, 100-char line limit. Run
  `make format-cpp` or let the pre-commit hook handle it.
- **Static analysis**: `make tidy` runs clang-tidy. Magic-number and pointer-arithmetic warnings are
  suppressed for BLAS code (configured in `.clang-tidy`).
- **Header utilities**: Use `aligned_alloc64.hpp` for 64-byte aligned allocation, `rfp_utils.hpp`
  for RFP packed format helpers, `constants.hpp` for physical constants.

### Adding a new extension module

Each module requires three files plus a CMakeLists.txt entry:

1. `src/<name>.{cpp,hpp}` — implementation
2. `src/<name>_bindings.cpp` — Pybind11 `PYBIND11_MODULE` definition
3. `python/kernelforge/<name>.pyi` — type stub (must match the bindings exactly)
4. Add to `CMakeLists.txt`: `pybind11_add_module(...)` + `target_link_libraries(...)` for BLAS
5. Re-export in `python/kernelforge/__init__.py`

---

## Python Conventions

- **Type annotations**: Required on all functions (ruff `ANN` rules enforced). Use `numpy.ndarray`
  for arrays; follow existing `.pyi` stubs for dtype conventions.
- **Type checker**: `ty` (Astral). Not mypy. Run with `make typecheck`.
- **Formatter/linter**: `ruff` (v0.15.5). Run `make format` to auto-fix, `make lint` to check.
- **Path handling**: Use `pathlib.Path`, not `os.path` (ruff `PTH` rule enforced).
- **Uppercase variable names** (`X`, `K`, `Q`, `N`) are intentional in scientific code — ruff rules
  N802/N803/N806 are suppressed.
- **`.pyi` stubs**: Keep in sync with C++ bindings. When modifying `*_bindings.cpp`, update the
  corresponding `.pyi` file.
