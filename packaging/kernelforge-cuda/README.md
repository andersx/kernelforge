# kernelforge-cuda

Linux companion wheel that installs KernelForge CUDA extension modules
(`cuda_*.so`) into `site-packages/kernelforge/`.

```bash
pip install kernelforge
pip install kernelforge-cuda   # or: pip install 'kernelforge[cuda]'
```

`kernelforge[cuda]` installs both `torch` and this companion on Linux.
Install the CPU `kernelforge` package first (or via that extra).

Requires an NVIDIA driver compatible with the installed `torch` CUDA build.
Local monorepo developers can keep using `make install-linux-mkl-ilp64-cuda`
instead of this split package.
