"""KernelForge - Optimized kernels for machine learning."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("kernelforge")
except PackageNotFoundError:
    __version__ = "unknown"

# Re-export kitchen_sinks functions
from .kitchen_sinks import (
    rff_features as rff_features,
)
from .kitchen_sinks import (
    rff_features_elemental as rff_features_elemental,
)
from .kitchen_sinks import (
    rff_gradient_elemental as rff_gradient_elemental,
)
from .kitchen_sinks import (
    rff_gramian_elemental as rff_gramian_elemental,
)
from .kitchen_sinks import (
    rff_gramian_elemental_gradient as rff_gramian_elemental_gradient,
)

# Import model module (accessible as kernelforge.model.*)
from . import model
