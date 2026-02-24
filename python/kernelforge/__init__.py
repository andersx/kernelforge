"""KernelForge - Optimized kernels for machine learning."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("kernelforge")
except PackageNotFoundError:
    __version__ = "unknown"

from .kitchen_sinks import (
    rff_features as rff_features,
    rff_gradient as rff_gradient,
    rff_full as rff_full,
    rff_gramian_symm as rff_gramian_symm,
    rff_gradient_gramian_symm as rff_gradient_gramian_symm,
    rff_full_gramian_symm as rff_full_gramian_symm,
    rff_gramian_symm_rfp as rff_gramian_symm_rfp,
    rff_gradient_gramian_symm_rfp as rff_gradient_gramian_symm_rfp,
    rff_full_gramian_symm_rfp as rff_full_gramian_symm_rfp,
    rff_features_elemental as rff_features_elemental,
    rff_gradient_elemental as rff_gradient_elemental,
    rff_full_elemental as rff_full_elemental,
    rff_gramian_elemental as rff_gramian_elemental,
    rff_gradient_gramian_elemental as rff_gradient_gramian_elemental,
    rff_full_gramian_elemental as rff_full_gramian_elemental,
    rff_gramian_elemental_rfp as rff_gramian_elemental_rfp,
    rff_gradient_gramian_elemental_rfp as rff_gradient_gramian_elemental_rfp,
    rff_full_gramian_elemental_rfp as rff_full_gramian_elemental_rfp,
)
