"""Type stubs for _invdist C++ extension module."""

import numpy as np
from numpy.typing import NDArray

def num_pairs(N: int) -> int: ...
def inverse_distance_upper(R: NDArray[np.float64]) -> NDArray[np.float64]: ...
def inverse_distance_upper_and_jacobian(
    R: NDArray[np.float64], eps: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
