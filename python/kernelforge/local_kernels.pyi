import numpy as np
from numpy.typing import NDArray

def kernel_gaussian_symm(
    X: NDArray[np.float64], Q: NDArray[np.int32], N: NDArray[np.int32], sigma: float
) -> NDArray[np.float64]: ...
def kernel_gaussian_symm_rfp(
    X: NDArray[np.float64], Q: NDArray[np.int32], N: NDArray[np.int32], sigma: float
) -> NDArray[np.float64]: ...
def kernel_gaussian(
    X1: NDArray[np.float64],
    X2: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_hessian_symm(
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    Q: NDArray[np.int32],
    N: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_hessian(
    X1: NDArray[np.float64],
    X2: NDArray[np.float64],
    dX1: NDArray[np.float64],
    dX2: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_jacobian(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    dX2: NDArray[np.float64],
    q1: NDArray[np.int32],
    q2: NDArray[np.int32],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
