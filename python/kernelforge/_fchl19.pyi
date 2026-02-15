"""Type stubs for _fchl19 C++ extension module."""

from numpy.typing import NDArray
import numpy as np

def compute_rep_size(nelements: int, nbasis2: int, nbasis3: int, nabasis: int) -> int: ...
def generate_fchl_acsf(
    coords: NDArray[np.float64],
    nuclear_z: NDArray[np.int32],
    elements: list[int] = [1, 6, 7, 8, 16],
    nRs2: int = 24,
    nRs3: int = 20,
    nFourier: int = 1,
    eta2: float = 0.32,
    eta3: float = 2.7,
    zeta: float = ...,
    rcut: float = 8.0,
    acut: float = 8.0,
    two_body_decay: float = 1.8,
    three_body_decay: float = 0.57,
    three_body_weight: float = 13.4,
) -> NDArray[np.float64]: ...
def generate_fchl_acsf_and_gradients(
    coords: NDArray[np.float64],
    nuclear_z: NDArray[np.int32],
    elements: list[int] = [1, 6, 7, 8, 16],
    nRs2: int = 24,
    nRs3: int = 20,
    nFourier: int = 1,
    eta2: float = 0.32,
    eta3: float = 2.7,
    zeta: float = ...,
    rcut: float = 8.0,
    acut: float = 8.0,
    two_body_decay: float = 1.8,
    three_body_decay: float = 0.57,
    three_body_weight: float = 13.4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def flocal_kernel_symm(
    X: NDArray[np.float64], Q: NDArray[np.int32], N: NDArray[np.int32], sigma: float
) -> NDArray[np.float64]: ...
def flocal_kernel_symm_rfp(
    X: NDArray[np.float64], Q: NDArray[np.int32], N: NDArray[np.int32], sigma: float
) -> NDArray[np.float64]: ...
def flocal_kernel(
    X1: NDArray[np.float64],
    X2: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
def fgdml_kernel_symm(
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    Q: NDArray[np.int32],
    N: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
def fgdml_kernel(
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
def fatomic_local_gradient_kernel(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    dX2: NDArray[np.float64],
    q1: NDArray[np.int32],
    q2: NDArray[np.int32],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
