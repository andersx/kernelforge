"""Representation generation helpers for the high-level model API.

Wraps the low-level C++ fchl19_repr and fchl18_repr calls into a unified
interface used by LocalKRRModel, LocalRFFModel, and FCHL18KRRModel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernelforge import fchl18_repr, fchl19_repr


def compute_fchl19(
    coords_list: list[NDArray[np.float64]],
    z_list: list[NDArray[np.int32]],
    elements: list[int],
    with_gradients: bool,
    repr_params: dict[str, Any],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.int32],
    list[NDArray[np.int32]],
    NDArray[np.int32],
]:
    """Compute FCHL19 representations for a list of molecules.

    Parameters
    ----------
    coords_list:
        List of (n_atoms_i, 3) float64 coordinate arrays.
    z_list:
        List of (n_atoms_i,) int32 nuclear charge arrays.
    elements:
        Sorted list of unique atomic numbers present in the dataset.
        Determines the element-to-index mapping for Q arrays.
    with_gradients:
        If True, also compute and return representation Jacobians (dX).
        Required for force training/prediction.
    repr_params:
        Extra keyword arguments forwarded to generate_fchl_acsf_and_gradients
        or generate_fchl_acsf (e.g. nRs2, nRs3, rcut, eta2, ...).

    Returns
    -------
    X : ndarray, shape (n_mols, n_atoms, rep_size)
        Stacked FCHL19 representations.
    dX : ndarray, shape (n_mols, n_atoms, rep_size, n_atoms, 3) or None
        Stacked representation Jacobians. None when with_gradients=False.
    Q_krr : ndarray, shape (n_mols, n_atoms), int32
        Tiled nuclear charges for local KRR kernels.
    Q_rff : list of ndarray, each shape (n_atoms_i,), int32
        Per-molecule 0-based element-index arrays for elemental RFF kernels.
        Q_rff[i][j] = index of z_list[i][j] in `elements`.
    N : ndarray, shape (n_mols,), int32
        Number of atoms per molecule (for local KRR kernels).
    """
    elem_to_idx = {e: i for i, e in enumerate(elements)}

    X_list: list[NDArray[np.float64]] = []
    dX_list: list[NDArray[np.float64]] = []

    for coords, z in zip(coords_list, z_list, strict=True):
        coords_f64 = np.asarray(coords, dtype=np.float64)
        z_i32 = np.asarray(z, dtype=np.int32)

        if with_gradients:
            x, dx = fchl19_repr.generate_fchl_acsf_and_gradients(
                coords_f64, z_i32, elements=elements, **repr_params
            )
            dX_list.append(dx)
        else:
            x = fchl19_repr.generate_fchl_acsf(coords_f64, z_i32, elements=elements, **repr_params)

        X_list.append(x)

    n_mols = len(coords_list)
    X = np.array(X_list, dtype=np.float64)  # (n_mols, n_atoms, rep_size)

    # Shape when computed: (n_mols, n_atoms, rep_size, n_atoms*3) — matches local KRR kernels.
    # LocalRFFModel reshapes to 5D (n_mols, n_atoms, rep_size, n_atoms, 3) internally.
    dX: NDArray[np.float64] | None = np.array(dX_list, dtype=np.float64) if with_gradients else None

    # Q for local KRR kernels: (n_mols, n_atoms) int32 tiled nuclear charges
    # Q for elemental RFF kernels: list of 1D int32 arrays with 0-based element indices
    Q_krr = np.vstack([z_list[i][np.newaxis, :] for i in range(n_mols)]).astype(np.int32)
    Q_rff = [
        np.array([elem_to_idx[int(zi)] for zi in z_list[i]], dtype=np.int32) for i in range(n_mols)
    ]
    N = np.array([len(z) for z in z_list], dtype=np.int32)

    return X, dX, Q_krr, Q_rff, N


def compute_fchl18(
    coords_list: list[NDArray[np.float64]],
    z_list: list[NDArray[np.int32]],
    max_size: int,
    cut_distance: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    list[NDArray[np.float64]],
    list[NDArray[np.int32]],
]:
    """Compute FCHL18 representations for a list of molecules.

    Parameters
    ----------
    coords_list:
        List of (n_atoms_i, 3) float64 coordinate arrays.
    z_list:
        List of (n_atoms_i,) int32 nuclear charge arrays.
    max_size:
        Maximum number of atoms (padding dimension).
    cut_distance:
        Neighbour cutoff radius in Angstrom.

    Returns
    -------
    x : ndarray, shape (n_mols, max_size, 5, max_size)
        FCHL18 packed representations.
    n_atoms : ndarray, shape (n_mols,), int32
        Number of real atoms per molecule.
    n_neighbors : ndarray, shape (n_mols, max_size), int32
        Neighbour counts per atom.
    coords_list_f64 : list of ndarray, each (n_atoms_i, 3)
        Raw coordinates as float64 lists (needed by gradient/Hessian kernels).
    z_list_i32 : list of ndarray, each (n_atoms_i,), int32
        Nuclear charges as int32 lists (needed by gradient/Hessian kernels).
    """
    coords_list_f64 = [np.asarray(r, dtype=np.float64) for r in coords_list]
    z_list_i32 = [np.asarray(z, dtype=np.int32) for z in z_list]

    x, n_atoms, n_neighbors = fchl18_repr.generate(
        coords_list_f64,
        z_list_i32,
        max_size=max_size,
        cut_distance=cut_distance,
    )

    return x, n_atoms, n_neighbors, coords_list_f64, z_list_i32
