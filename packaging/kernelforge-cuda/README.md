# kernelforge-cuda

Linux companion wheel that installs KernelForge CUDA extension modules
(`cuda_*.so`) into `site-packages/kernelforge/`.

```bash
pip install kernelforge
pip install kernelforge-cuda
# or:
pip install 'kernelforge[cuda]'
```

Requires an NVIDIA driver compatible with the installed `torch` CUDA build.
Local monorepo developers can keep using `make install-linux-mkl-ilp64-cuda`
instead of this split package.
