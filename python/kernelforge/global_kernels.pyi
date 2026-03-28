import numpy as np
from numpy.typing import NDArray

def kernel_gaussian_symm(X: NDArray[np.float64], alpha: float) -> NDArray[np.float64]: ...
def kernel_gaussian_symm_rfp(X: NDArray[np.float64], alpha: float) -> NDArray[np.float64]: ...
def kernel_gaussian(
    X1: NDArray[np.float64], X2: NDArray[np.float64], alpha: float
) -> NDArray[np.float64]: ...
def kernel_gaussian_jacobian(
    X1: NDArray[np.float64],
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_jacobian_t(
    X1: NDArray[np.float64],
    X2: NDArray[np.float64],
    dX2: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_hessian(
    X1: NDArray[np.float64],
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    dX2: NDArray[np.float64],
    sigma: float,
    tile_B: int | None = None,
) -> NDArray[np.float64]: ...
def kernel_gaussian_hessian_symm(
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    sigma: float,
    tile_B: int | None = None,
) -> NDArray[np.float64]: ...
def kernel_gaussian_hessian_symm_rfp(
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    sigma: float,
    tile_B: int | None = None,
) -> NDArray[np.float64]: ...
def kernel_gaussian_full(
    X1: NDArray[np.float64],
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    dX2: NDArray[np.float64],
    sigma: float,
    tile_B: int | None = None,
) -> NDArray[np.float64]: ...
def kernel_gaussian_full_symm(
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    sigma: float,
    tile_B: int | None = None,
) -> NDArray[np.float64]: ...
def kernel_gaussian_full_symm_rfp(
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    sigma: float,
    tile_B: int | None = None,
) -> NDArray[np.float64]: ...
def kernel_gaussian_compute_alpha_desc(
    dX: NDArray[np.float64],
    alpha: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def kernel_gaussian_hessian_matvec(
    X_q: NDArray[np.float64],
    dX_q: NDArray[np.float64],
    X_t: NDArray[np.float64],
    alpha_desc: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_jacobian_t_matvec(
    X_q: NDArray[np.float64],
    X_t: NDArray[np.float64],
    alpha_desc: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_full_matvec(
    X_q: NDArray[np.float64],
    dX_q: NDArray[np.float64],
    X_t: NDArray[np.float64],
    alpha_E: NDArray[np.float64],
    alpha_desc_F: NDArray[np.float64],
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
