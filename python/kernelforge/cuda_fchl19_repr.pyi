"""Type stubs for cuda_fchl19_repr — GPU FCHL19 ACSF representation (FP32)."""

import torch

def generate_fchl_acsf(
    coords: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    nelements: int,
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
    deterministic: bool = False,
) -> torch.Tensor:
    """Generate FCHL19 ACSF representations on the GPU (FP32).

    Parameters
    ----------
    coords : torch.Tensor, shape (nm, max_atoms, 3), float32, CUDA
        Batched padded Cartesian coordinates.
    Q : torch.Tensor, shape (nm, max_atoms), int32, CUDA
        Element indices in [0, nelements).
    N : torch.Tensor, shape (nm,), int32, CUDA
        Active atom counts per molecule.
    nelements : int
        Number of distinct element types.
    nRs2 : int, default 24
    nRs3 : int, default 20
    nFourier : int, default 1
    eta2 : float, default 0.32
    eta3 : float, default 2.7
    zeta : float, default pi
    rcut : float, default 8.0
    acut : float, default 8.0
    two_body_decay : float, default 1.8
    three_body_decay : float, default 0.57
    three_body_weight : float, default 13.4

    Returns
    -------
    rep : torch.Tensor, shape (nm, max_atoms, rep_size), float32, CUDA
        Per-atom representations.  Padded slots (i >= N[m]) are zeroed.
    """
    ...

def generate_fchl_acsf_and_gradients(
    coords: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    nelements: int,
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
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate FCHL19 ACSF representations and Jacobians on the GPU (FP32).

    Returns
    -------
    rep : torch.Tensor, shape (nm, max_atoms, rep_size), float32, CUDA
    grad : torch.Tensor, shape (nm, max_atoms, rep_size, max_atoms, 3), float32, CUDA
    """
    ...
