"""Type stubs for cuda_invdist_repr — GPU batched inverse-distance representation (FP32)."""

import torch

def inverse_distance_upper(
    coords: torch.Tensor,
    n_atoms: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute batched inverse-distance representation on GPU.

    Parameters
    ----------
    coords : torch.Tensor, shape (nm, n_atoms, 3), float32, CUDA
        Batched Cartesian coordinates.  All molecules must have the same n_atoms.
    n_atoms : int
        Number of atoms per molecule.
    eps : float, default 1e-6
        Distance floor: r is clamped to max(r, eps) before inversion.

    Returns
    -------
    X : torch.Tensor, shape (nm, M), float32, CUDA
        M = n_atoms*(n_atoms-1)//2.  Entry [m, p] = 1/r_{ij} for pair p=(i<j).
    """
    ...

def inverse_distance_upper_and_jacobian(
    coords: torch.Tensor,
    n_atoms: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute batched inverse-distance representation and Jacobian on GPU.

    Parameters
    ----------
    coords : torch.Tensor, shape (nm, n_atoms, 3), float32, CUDA
        Batched Cartesian coordinates.  All molecules must have the same n_atoms.
    n_atoms : int
        Number of atoms per molecule.
    eps : float, default 1e-6
        Distance floor: r is clamped to max(r, eps) before inversion.

    Returns
    -------
    X : torch.Tensor, shape (nm, M), float32, CUDA
        M = n_atoms*(n_atoms-1)//2.
    dX : torch.Tensor, shape (nm, D, M), float32, CUDA
        D = 3*n_atoms.  dX[m, 3*a+d, p] = d(1/r_ij)/d(R_{a,d}) for pair p=(i<j).
    """
    ...
