import numpy as np
from numpy.typing import NDArray

def solve_cholesky(
    K: NDArray[np.float64], y: NDArray[np.float64], regularize: float = 0.0
) -> NDArray[np.float64]: ...
def cho_solve_rfp(
    K_rfp: NDArray[np.float64],
    y: NDArray[np.float64],
    l2: float = 0.0,
) -> NDArray[np.float64]:
    """Solve (K + l2*I) @ alpha = y where K is in RFP packed format.

    K_rfp is overwritten with the Cholesky factor — pass K_rfp.copy() to preserve it.
    Uses TRANSR='N', UPLO='U' (the convention all kernelforge RFP kernels produce).
    """
    ...

def full_to_rfp(
    A: NDArray[np.float64], uplo: str = "U", transr: str = "N"
) -> NDArray[np.float64]: ...
def rfp_to_full(
    ARF: NDArray[np.float64], n: int, uplo: str = "U", transr: str = "N"
) -> NDArray[np.float64]: ...
