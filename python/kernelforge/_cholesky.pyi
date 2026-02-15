"""Type stubs for _cholesky C++ extension module."""

import numpy as np
from numpy.typing import NDArray

def solve_cholesky(
    K: NDArray[np.float64], y: NDArray[np.float64], regularize: float = 0.0
) -> NDArray[np.float64]: ...
def solve_cholesky_rfp_L(
    K_arf: NDArray[np.float64],
    y: NDArray[np.float64],
    regularize: float = 0.0,
    uplo: str = "U",
    transr: str = "N",
) -> NDArray[np.float64]: ...
def full_to_rfp(
    A: NDArray[np.float64], uplo: str = "U", transr: str = "N"
) -> NDArray[np.float64]: ...
def rfp_to_full(
    ARF: NDArray[np.float64], n: int, uplo: str = "U", transr: str = "N"
) -> NDArray[np.float64]: ...
