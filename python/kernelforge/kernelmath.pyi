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
def solve_qr(A: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve min||A@x - y||₂ (overdetermined) or min||x||₂ s.t. A@x=y (underdetermined).
    A is m x n, y is length m; returns x of length n. Uses DGELS (QR/LQ decomposition).
    A must have full rank."""
    ...

def solve_svd(
    A: NDArray[np.float64], y: NDArray[np.float64], rcond: float = 0.0
) -> NDArray[np.float64]:
    """Same as solve_qr but uses DGELSD (divide-and-conquer SVD).
    Handles rank-deficient A: singular values < rcond*sigma_max are treated as zero.
    rcond=0.0 uses machine epsilon as threshold."""
    ...

def condition_number_ge(A: NDArray[np.float64]) -> float:
    """1-norm condition number of a square matrix A via LU factorization (DGETRF+DGECON).
    Works for any square matrix (not just symmetric/positive-definite)."""
    ...
