from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

def generate(
    coords_list: Sequence[NDArray[np.float64]],
    nuclear_z_list: Sequence[NDArray[np.int32]],
    max_size: int = 23,
    cut_distance: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Generate FCHL18 representations for a batch of molecules.

    Parameters
    ----------
    coords_list:
        List of (n_i, 3) float64 coordinate arrays, one per molecule.
    nuclear_z_list:
        List of (n_i,) int32 nuclear charge arrays, one per molecule.
    max_size:
        Maximum number of atoms (padding dimension). Default 23.
    cut_distance:
        Neighbour cutoff radius in Angstrom. Default 5.0.

    Returns
    -------
    x : ndarray, shape (nm, max_size, 5, max_size), float64
        Representation. Layout per atom: [distances, Z-neighbours, dx, dy, dz].
        Unused slots are filled with 1e100.
    n_atoms : ndarray, shape (nm,), int32
        Number of real atoms per molecule.
    n_neighbors : ndarray, shape (nm, max_size), int32
        Number of neighbours within cut_distance per atom (self at index 0).
    """
    ...
