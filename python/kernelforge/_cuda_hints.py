"""Shared install hints for missing CUDA extension modules."""

from __future__ import annotations

CUDA_EXT_HINT = (
    "Install the CUDA extensions with:\n"
    "    pip install 'kernelforge[cuda]'          # Linux / PyPI (companion wheel)\n"
    "    make install-linux-mkl-ilp64-cuda        # local monorepo build\n"
    "Requires an NVIDIA driver compatible with your torch CUDA build."
)
