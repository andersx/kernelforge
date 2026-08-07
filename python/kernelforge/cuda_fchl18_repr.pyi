"""Type stubs for cuda_fchl18_repr — GPU FCHL18 representation generation."""

import torch

def generate(
    coords: torch.Tensor,
    z: torch.Tensor,
    n: torch.Tensor,
    cut_distance: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate FCHL18 representations on the GPU.

    Parameters
    ----------
    coords : torch.Tensor, shape (nm, max_size, 3), float32 or float64, CUDA
        Padded Cartesian coordinates.
    z : torch.Tensor, shape (nm, max_size), int32, CUDA
        Padded nuclear charges.
    n : torch.Tensor, shape (nm,), int32, CUDA
        Active atom counts per molecule.
    cut_distance : float, default 5.0
        Neighbour cutoff radius in Angstrom.

    Returns
    -------
    x : torch.Tensor, shape (nm, max_size, 5, max_size), same dtype as coords, CUDA
        Representation. Layout per atom: [distances, Z-neighbours, dx, dy, dz].
        Unused slots are filled with 1e100 (fp64) or 1e30 (fp32).
    nn : torch.Tensor, shape (nm, max_size), int32, CUDA
        Number of neighbours within cut_distance per atom (self at index 0).
    """
    ...
