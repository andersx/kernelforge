import numpy as np
from numpy.typing import NDArray

def kernel_gaussian(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    nn1: NDArray[np.int32],
    nn2: NDArray[np.int32],
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 0.5,
    cut_distance: float = 1e6,
    fourier_order: int = 2,
) -> NDArray[np.float64]:
    """FCHL18 Gaussian kernel matrix (asymmetric).

    K[a, b] = sum_{i in mol_a, j in mol_b: Z_i == Z_j}
                  exp( -(s_ii + s_jj - 2*s_ij) / sigma^2 )

    Parameters
    ----------
    x1, x2 : ndarray, shape (nm, max_size, 5, max_size), float64
        Representations from fchl18_repr.generate().
    n1, n2 : ndarray, shape (nm,), int32
        Number of real atoms per molecule.
    nn1, nn2 : ndarray, shape (nm, max_size), int32
        Number of neighbours per atom.
    sigma : float
        Gaussian kernel width.
    two_body_scaling : float, default 2.0
    two_body_width : float, default 0.1
    two_body_power : float, default 6.0
    three_body_scaling : float, default 2.0
    three_body_width : float, default 3.0
    three_body_power : float, default 3.0
    cut_start : float, default 0.5
    cut_distance : float, default 1e6
    fourier_order : int, default 2

    Returns
    -------
    ndarray, shape (nm1, nm2), float64
    """
    ...

def kernel_gaussian_symm(
    x: NDArray[np.float64],
    n: NDArray[np.int32],
    nn: NDArray[np.int32],
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 0.5,
    cut_distance: float = 1e6,
    fourier_order: int = 2,
) -> NDArray[np.float64]:
    """FCHL18 Gaussian kernel matrix (symmetric, K[a,b] == K[b,a]).

    Parameters
    ----------
    x : ndarray, shape (nm, max_size, 5, max_size), float64
    n : ndarray, shape (nm,), int32
    nn : ndarray, shape (nm, max_size), int32
    sigma : float
    (remaining hyperparameters same as kernel_gaussian)

    Returns
    -------
    ndarray, shape (nm, nm), float64
    """
    ...
