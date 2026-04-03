import numpy as np
from numpy.typing import NDArray

# Two-body type strings:
#   "log_normal"        - T1: baseline FCHL19 log-normal
#   "gaussian_r"        - T2: fixed-width Gaussian in r
#   "gaussian_log_r"    - T3: fixed-width Gaussian in ln(r)
#   "gaussian_r_no_pow" - T4: fixed-width Gaussian in r, no 1/r^p decay
#   "bessel"            - T5: radial Bessel basis sin(k*pi*r/rcut)/r

# Three-body type strings:
#   "odd_fourier_rbar"           - A1: odd Fourier (cos+sin), r_bar radial (baseline)
#   "cosine_rbar"                - A2: full cosine series, r_bar radial
#   "odd_fourier_split_r"        - A3: odd Fourier, split r_plus/r_minus radial
#   "cosine_split_r"             - A4: full cosine series, split r_plus/r_minus radial
#   "cosine_split_r_no_atm"      - A5: full cosine series, split radial, no ATM factor
#   "odd_fourier_element_resolved" - A6: odd Fourier, element-resolved (r_ij,r_ik) basis
#   "cosine_element_resolved"      - A7: full cosine series, element-resolved (r_ij,r_ik) basis

def compute_rep_size(
    nelements: int,
    nbasis2: int,
    nbasis3: int,
    nabasis: int,
    three_body_type: str = "odd_fourier_rbar",
    nbasis3_minus: int = 0,
) -> int: ...
def generate(
    coords: NDArray[np.float64],
    nuclear_z: NDArray[np.int32],
    elements: list[int] = [1, 6, 7, 8, 16],
    nRs2: int = 24,
    nRs3: int = 20,
    nRs3_minus: int = 0,
    nFourier: int = 1,
    nCosine: int = 0,
    eta2: float = 0.32,
    eta3: float = 2.7,
    eta3_minus: float = 2.7,
    zeta: float = ...,
    rcut: float = 8.0,
    acut: float = 8.0,
    two_body_decay: float = 1.8,
    three_body_decay: float = 0.57,
    three_body_weight: float = 13.4,
    two_body_type: str = "log_normal",
    three_body_type: str = "odd_fourier_rbar",
) -> NDArray[np.float64]: ...
def generate_and_gradients(
    coords: NDArray[np.float64],
    nuclear_z: NDArray[np.int32],
    elements: list[int] = [1, 6, 7, 8, 16],
    nRs2: int = 24,
    nRs3: int = 20,
    nRs3_minus: int = 0,
    nFourier: int = 1,
    nCosine: int = 0,
    eta2: float = 0.32,
    eta3: float = 2.7,
    eta3_minus: float = 2.7,
    zeta: float = ...,
    rcut: float = 8.0,
    acut: float = 8.0,
    two_body_decay: float = 1.8,
    three_body_decay: float = 0.57,
    three_body_weight: float = 13.4,
    two_body_type: str = "log_normal",
    three_body_type: str = "odd_fourier_rbar",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
