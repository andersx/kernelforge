# Changelog

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
