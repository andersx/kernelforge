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
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    dX2: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_jacobian(
    X1: NDArray[np.float64],
    X2: NDArray[np.float64],
    dX2: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_jacobian_t(
    X1: NDArray[np.float64],
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]: ...
def kernel_gaussian_full(
    X1: NDArray[np.float64],
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    dX2: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
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
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    Q: NDArray[np.int32],
    N: NDArray[np.int32],
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
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    Q: NDArray[np.int32],
    N: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]:
    """Symmetric full kernel in RFP packed format, length BIG*(BIG+1)//2 where BIG=nm+naq.

    Packed in LAPACK RFP format (TRANSR='N', UPLO='U').  Unpack with
    ``scipy.linalg.get_lapack_funcs`` or the ``dtfttr`` routine.
    """
    ...

def kernel_gaussian_local_compute_alpha_desc(
    dX2: NDArray[np.float64],
    Q2: NDArray[np.int32],
    N2: NDArray[np.int32],
    alpha: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pre-compute descriptor-space force coefficients for J^T*alpha trick.

    Shapes:
      dX2:   (nm2, max_atoms2, rep_size, 3*max_atoms2), training Jacobians
      Q2:    (nm2, max_atoms2), atomic labels
      N2:    (nm2,), active atom counts
      alpha: (naq2,) where naq2 = 3*sum(N2), force coefficients in Cartesian space

    Returns:
      alpha_desc: (nm2, max_atoms2, rep_size), descriptor-space coefficients

    Use with kernel_gaussian_local_hessian_matvec for efficient O(M) inference cost.
    """
    ...

def kernel_gaussian_local_hessian_matvec(
    X1: NDArray[np.float64],
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    alpha_desc: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]:
    """Predict forces via local Hessian kernel matvec using J^T*alpha trick.

    Cost: O(nm1·naq2·rep + naq1) vs O(naq1·naq2·rep + naq1) for full matrix.

    Shapes:
      X1:        (nm1, max_atoms1, rep_size), query descriptor vectors
      dX1:       (nm1, max_atoms1, rep_size, 3*max_atoms1), query Jacobians
      X2:        (nm2, max_atoms2, rep_size), training descriptor vectors
      alpha_desc:(nm2, max_atoms2, rep_size), pre-computed via compute_alpha_desc
      Q1, Q2:    (nm1/nm2, max_atoms1/2), atomic labels (for matching)
      N1, N2:    (nm1/nm2), active atom counts
      sigma:     Gaussian width parameter

    Returns:
      F: (naq1,) where naq1 = 3*sum(N1), forces in Cartesian coordinates
    """
    ...

def kernel_gaussian_local_jacobian_t_matvec(
    X1: NDArray[np.float64],
    X2: NDArray[np.float64],
    alpha_desc: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
    sigma: float,
) -> NDArray[np.float64]:
    """Predict energies via local Jacobian kernel matvec using J^T*alpha trick.

    Shapes:
      X1:        (nm1, max_atoms1, rep_size), query descriptor vectors
      X2:        (nm2, max_atoms2, rep_size), training descriptor vectors
      alpha_desc:(nm2, max_atoms2, rep_size), pre-computed via compute_alpha_desc
      Q1, Q2:    (nm1/nm2, max_atoms1/2), atomic labels (for matching)
      N1, N2:    (nm1/nm2), active atom counts
      sigma:     Gaussian width parameter

    Returns:
      E: (nm1,) predicted energies per query molecule
    """
    ...

def kernel_gaussian_local_full_matvec(
    X1: NDArray[np.float64],
    dX1: NDArray[np.float64],
    X2: NDArray[np.float64],
    alpha_desc_F: NDArray[np.float64],
    alpha_E: NDArray[np.float64],
    Q1: NDArray[np.int32],
    Q2: NDArray[np.int32],
    N1: NDArray[np.int32],
    N2: NDArray[np.int32],
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Predict energies+forces via full local kernel matvec using J^T*alpha trick.

    Shapes:
      X1:          (nm1, max_atoms1, rep_size), query descriptor vectors
      dX1:         (nm1, max_atoms1, rep_size, 3*max_atoms1), query Jacobians
      X2:          (nm2, max_atoms2, rep_size), training descriptor vectors
      alpha_desc_F:(nm2, max_atoms2, rep_size), pre-computed via compute_alpha_desc(dX2, alpha_F)
      alpha_E:     (nm2,), energy coefficients
      Q1, Q2:      (nm1/nm2, max_atoms1/2), atomic labels (for matching)
      N1, N2:      (nm1/nm2), active atom counts
      sigma:       Gaussian width parameter

    Returns:
      (E, F): E=(nm1,) energies, F=(naq1,) forces where naq1=3*sum(N1)
    """
    ...
