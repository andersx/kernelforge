"""KernelForge - Optimized kernels for machine learning."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("kernelforge")
except PackageNotFoundError:
    __version__ = "unknown"

from .kitchen_sinks import (
    rff_features,
    rff_features_elemental,
    rff_gradient_elemental,
    rff_gramian_elemental,
    rff_gramian_elemental_gradient,
)
