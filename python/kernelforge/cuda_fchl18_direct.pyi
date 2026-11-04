"""Type stubs for cuda_fchl18_direct — coords-only FCHL18 Gaussian (FP32, no cutoff)."""

import torch

def kernel_gaussian(
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
    """Rectangular FCHL18 Gaussian from padded molecular coords (FP32, no cutoff).

    Parameters
    ----------
    coords1, coords2 : (nm, max_atoms, 3) float32 CUDA
    Z1, Z2 : (nm, max_atoms) int32 CUDA
    N1, N2 : (nm,) int32 CUDA
    sigma : float

    Returns
    -------
    K : (nm1, nm2) float32 CUDA
    """
    ...
