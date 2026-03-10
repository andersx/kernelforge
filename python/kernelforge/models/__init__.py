"""High-level model API for kernelforge.

Provides scikit-learn-style classes for training, prediction, and
serialization of kernel and random Fourier feature ML potentials.

Classes
-------
LocalKRRModel
    FCHL19-based local Kernel Ridge Regression. Supports energy-only,
    force-only, and energy+force training modes.
LocalRFFModel
    FCHL19-based local Random Fourier Features regression. Same training
    modes as LocalKRRModel but solves a D_rff x D_rff system instead of N x N.
FCHL18KRRModel
    FCHL18 analytical Kernel Ridge Regression. Same training modes but uses
    the FCHL18 kernel which operates on raw Cartesian coordinates.
ModelScore
    Dataclass returned by model.score() containing MAE, Pearson r, slope,
    and intercept for energy and/or force predictions.
"""

from .base import ModelScore as ModelScore
from .fchl18_krr import FCHL18KRRModel as FCHL18KRRModel
from .krr import LocalKRRModel as LocalKRRModel
from .rff import LocalRFFModel as LocalRFFModel
