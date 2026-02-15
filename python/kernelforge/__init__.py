"""KernelForge - Optimized kernels for machine learning."""

try:
    from importlib.metadata import version

    __version__ = version("kernelforge")
except Exception:
    __version__ = "unknown"
