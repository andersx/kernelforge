# Changelog

## 1.1.0

### FCHL18

- Periodic-table alchemy for CPU and CUDA (scalar, Jacobian, Hessian, full, matvec)
- ATM enabled in force / energy-and-force models (documented default; no longer forced off)
- Default `fourier_order=1` to match QML / qmllib
- CPU and CUDA matvec APIs for energy / Jacobian / Hessian / full inference
- CUDA FCHL18 defaults to float64 end-to-end; GPU float64 RFP Cholesky (`rfp_potrf` / `rfp_potrs`)
- Energy-only scoring via scalar + Jacobian-transpose matvec when `compute_forces=False`
- Train-side CUDA workspace reuse with content fingerprinting; fused full-matvec path
- Large CUDA FCHL18 kernel speedups and peak-memory reductions (scalar / jac / hess)

### Models / API

- `compute_energy` / `compute_forces` honored across CPU and CUDA KRR / RFF / FCHL18 paths
- CUDA local / global KRR skip expensive force work when forces are not requested
- Expanded `kernelbench` coverage for public kernel APIs

### Packaging / tooling

- CUDA companion build C++20 fix; cibuildwheel / CI / pre-commit / ruff / ty pin sync

## 1.0.0

First stable release.

### Packaging

- CPU wheels (`kernelforge`): Linux + macOS for CPython **3.10–3.15**
- CUDA companion (`kernelforge-cuda`): Linux manylinux for CPython **3.11–3.14**
- `kernelforge[cuda]` pulls the companion on Linux for Python 3.11–3.14

### Highlights since 0.3.x

- ASE molecular dynamics / optimization via `kernelmd` and model calculators
- CUDA packaging split: CPU `kernelforge` + companion `kernelforge-cuda` on PyPI
- CUDA `force_only` for local/global KRR and RFF models
- FCHL18 CUDA KRR: full float32 path with GPU `rfp_potrf` (`uplo=U`)
- CLI: `--cuda`, `--dtype float32|float64` (FCHL18), force-only modes
