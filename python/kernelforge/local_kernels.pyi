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
def kernel_gaussian_hessian_symm_rfp(
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
def kernel_gaussian_jacobian_t(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    dX1: NDArray[np.float64],
    q1: NDArray[np.int32],
    q2: NDArray[np.int32],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_full(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    dx1: NDArray[np.float64],
    dx2: NDArray[np.float64],
    q1: NDArray[np.int32],
    q2: NDArray[np.int32],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]:
    """Asymmetric full kernel, shape (nm1+naq1, nm2+naq2).

    Block layout:
      [:nm1, :nm2]   scalar
      [:nm1, nm2:]   jacobian_t  (dK/dR_2)
      [nm1:, :nm2]   jacobian    (dK/dR_1)
      [nm1:, nm2:]   hessian
    """
    ...

def kernel_gaussian_full_symm(
    x: NDArray[np.float64],
    dx: NDArray[np.float64],
    q: NDArray[np.int32],
    n: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]:
    """Symmetric full kernel, shape (nm+naq, nm+naq), fully filled.

    Block layout:
      [:nm, :nm]   scalar  (symmetric)
      [:nm, nm:]   jacobian_t = jacobian.T
      [nm:, :nm]   jacobian
      [nm:, nm:]   hessian (symmetric)
    """
    ...

def kernel_gaussian_full_symm_rfp(
    x: NDArray[np.float64],
    dx: NDArray[np.float64],
    q: NDArray[np.int32],
    n: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]:
    """Symmetric full kernel in RFP packed format, length BIG*(BIG+1)//2 where BIG=nm+naq.

    Packed in LAPACK RFP format (TRANSR='N', UPLO='U').  Unpack with
    ``scipy.linalg.get_lapack_funcs`` or the ``dtfttr`` routine.
    """
    ...
