# kernelforge-cuda

Linux companion wheel that installs KernelForge CUDA extension modules
(`cuda_*.so`) into `site-packages/kernelforge/`.

**On PyPI.** Release wheels are built on GitHub Actions (`build-wheels-cuda`
in the release workflow) using `manylinux_cuda` images for **CPython
3.11–3.14**.

```bash
pip install kernelforge
pip install kernelforge-cuda   # or: pip install 'kernelforge[cuda]'
```

`kernelforge[cuda]` installs both `torch` and this companion on Linux.
Install the CPU `kernelforge` package first (or via that extra). A default-index
`torch` install may be CPU-only; use a CUDA build of PyTorch for GPU runs.

Requires an NVIDIA driver compatible with the installed `torch` CUDA build.

### Local monorepo / wheel smoke

```bash
# From the monorepo root:
make demo-cuda-wheels
# or install wheels from dist/ manually:
#   pip install dist/kernelforge-*.whl
#   pip install --find-links dist kernelforge-cuda
```

Day-to-day developers can keep using `make install-linux-mkl-ilp64-cuda`
instead of the split package.
