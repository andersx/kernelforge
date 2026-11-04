"""Type stubs for cuda_fchl18_direct_hess — coords-only FCHL18 Gaussian Hessian (FP32, no cutoff)."""

import torch

def kernel_gaussian_hessian(
    coords1: torch.Tensor,
    Z1: torch.Tensor,
    N1: torch.Tensor,
    coords2: torch.Tensor,
    Z2: torch.Tensor,
    N2: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    fourier_order: int = 1,
    use_atm: bool = True,
    use_alchemy: bool = False,
    alchemy_period_width: float = 1.6,
    alchemy_group_width: float = 1.6,
) -> torch.Tensor:
    """FCHL18 Gaussian Hessian kernel from padded molecular coords (FP32, no cutoff).

    H[row, col] = d2 K[A, B] / dR_A[row] dR_B[col], atom-major then xyz.

    Parameters
    ----------
    coords1, coords2 : (nm, max_atoms, 3) float32 CUDA
    Z1, Z2 : (nm, max_atoms) int32 CUDA
    N1, N2 : (nm,) int32 CUDA
    sigma : float

    Returns
    -------
    H : (D_A, D_B) float32 CUDA, D_A = 3*sum(N1), D_B = 3*sum(N2)
    """
    ...
