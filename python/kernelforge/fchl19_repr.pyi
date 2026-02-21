import numpy as np
from numpy.typing import NDArray

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
