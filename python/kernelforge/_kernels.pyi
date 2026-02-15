"""Type stubs for _kernels C++ extension module."""

import numpy as np
from numpy.typing import NDArray

def kernel_symm(X: NDArray[np.float64], alpha: float) -> NDArray[np.float64]: ...
def kernel_asymm(
    X1: NDArray[np.float64], X2: NDArray[np.float64], alpha: float
) -> NDArray[np.float64]: ...
def gaussian_jacobian_batch(
    X1: NDArray[np.float64],
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]: ...
def rbf_hessian_full_tiled_gemm(
    X1: NDArray[np.float64],
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    dX2: NDArray[np.float64],
    sigma: float,
    tile_B: int | None = None,
) -> NDArray[np.float64]: ...
def rbf_hessian_full_tiled_gemm_sym(
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    sigma: float,
    tile_B: int | None = None,
) -> NDArray[np.float64]: ...
