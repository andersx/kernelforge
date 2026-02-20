"""KernelForge - Optimized kernels for machine learning."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("kernelforge")
except PackageNotFoundError:
    __version__ = "unknown"
