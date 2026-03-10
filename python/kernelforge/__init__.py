"""KernelForge - Optimized kernels for machine learning."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("kernelforge")
except PackageNotFoundError:
    __version__ = "unknown"

from .kitchen_sinks import (
    rff_features as rff_features,
)
from .kitchen_sinks import (
    rff_features_elemental as rff_features_elemental,
)
from .kitchen_sinks import (
    rff_full as rff_full,
)
from .kitchen_sinks import (
    rff_full_elemental as rff_full_elemental,
)
from .kitchen_sinks import (
    rff_full_gramian_elemental as rff_full_gramian_elemental,
)
from .kitchen_sinks import (
    rff_full_gramian_elemental_rfp as rff_full_gramian_elemental_rfp,
)
from .kitchen_sinks import (
    rff_full_gramian_symm as rff_full_gramian_symm,
)
from .kitchen_sinks import (
    rff_full_gramian_symm_rfp as rff_full_gramian_symm_rfp,
)
from .kitchen_sinks import (
    rff_gradient as rff_gradient,
)
from .kitchen_sinks import (
    rff_gradient_elemental as rff_gradient_elemental,
)
from .kitchen_sinks import (
    rff_gradient_gramian_elemental as rff_gradient_gramian_elemental,
)
from .kitchen_sinks import (
    rff_gradient_gramian_elemental_rfp as rff_gradient_gramian_elemental_rfp,
)
from .kitchen_sinks import (
    rff_gradient_gramian_symm as rff_gradient_gramian_symm,
)
from .kitchen_sinks import (
    rff_gradient_gramian_symm_rfp as rff_gradient_gramian_symm_rfp,
)
from .kitchen_sinks import (
    rff_gramian_elemental as rff_gramian_elemental,
)
from .kitchen_sinks import (
    rff_gramian_elemental_rfp as rff_gramian_elemental_rfp,
)
from .kitchen_sinks import (
    rff_gramian_symm as rff_gramian_symm,
)
from .kitchen_sinks import (
    rff_gramian_symm_rfp as rff_gramian_symm_rfp,
)
from .models import FCHL18KRRModel as FCHL18KRRModel
from .models import LocalKRRModel as LocalKRRModel
from .models import LocalRFFModel as LocalRFFModel
