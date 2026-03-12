"""Representation generation helpers for the high-level model API.

Wraps the low-level C++ fchl19_repr, fchl19v2_repr, and fchl18_repr calls into
a unified interface used by LocalKRRModel, LocalRFFModel, and FCHL18KRRModel.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernelforge import fchl18_repr, fchl19_repr, fchl19v2_repr

# ---------------------------------------------------------------------------
# OpenMP thread-count context manager
# ---------------------------------------------------------------------------
# fchl19_repr.generate_fchl_acsf uses #pragma omp parallel internally.
# Parallel reductions over atoms are non-associative in floating-point, so
# X_te varies by ~1 ULP between calls, which amplifies to ~1e-6 in K@alpha
# for large training sets.  Pinning to 1 thread makes results bit-for-bit
# reproducible at negligible cost (per-molecule work is small).


def _load_libomp() -> ctypes.CDLL | None:
    """Try to load the OpenMP runtime. Returns None if unavailable."""
    for name in ("libgomp.so.1", "libomp.so", "libiomp5.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_libomp: ctypes.CDLL | None = _load_libomp()


@contextmanager
def _single_omp_thread() -> Generator[None, None, None]:
    """Temporarily pin OpenMP to 1 thread, then restore the previous count."""
    if _libomp is None:
        yield
        return
    prev = int(_libomp.omp_get_max_threads())
    _libomp.omp_set_num_threads(1)
    try:
        yield
    finally:
        _libomp.omp_set_num_threads(prev)


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

    # Pin to 1 OpenMP thread for the duration of representation generation.
    # fchl19_repr uses #pragma omp parallel over atoms; parallel reductions are
    # non-associative in floating-point, so X varies by ~1 ULP between calls,
    # which amplifies to ~1e-6 in K@alpha for large training sets.
    with _single_omp_thread():
        for coords, z in zip(coords_list, z_list, strict=True):
            coords_f64 = np.asarray(coords, dtype=np.float64)
            z_i32 = np.asarray(z, dtype=np.int32)

            if with_gradients:
                x, dx = fchl19_repr.generate_fchl_acsf_and_gradients(
                    coords_f64, z_i32, elements=elements, **repr_params
                )
                dX_list.append(dx)
            else:
                x = fchl19_repr.generate_fchl_acsf(
                    coords_f64, z_i32, elements=elements, **repr_params
                )

            X_list.append(x)

    n_mols = len(coords_list)
    N = np.array([len(z) for z in z_list], dtype=np.int32)
    max_atoms = int(N.max())
    rep_size = X_list[0].shape[1]

    # Pad X to (n_mols, max_atoms, rep_size) — rows beyond N[i] are zero.
    X = np.zeros((n_mols, max_atoms, rep_size), dtype=np.float64)
    for i, x_i in enumerate(X_list):
        X[i, : len(x_i), :] = x_i

    # Pad dX to (n_mols, max_atoms, rep_size, max_atoms*3) — same zero-padding.
    # Shape when computed: (n_mols, max_atoms, rep_size, max_atoms*3) — matches local KRR kernels.
    # LocalRFFModel reshapes to 5D (n_mols, max_atoms, rep_size, max_atoms, 3) internally.
    dX: NDArray[np.float64] | None
    if with_gradients:
        dX = np.zeros((n_mols, max_atoms, rep_size, max_atoms * 3), dtype=np.float64)
        for i, dx_i in enumerate(dX_list):
            n_i = len(z_list[i])
            # dx_i shape: (n_i, rep_size, n_i*3) — pad both atom dims
            dX[i, :n_i, :, : n_i * 3] = dx_i
    else:
        dX = None

    # Q for local KRR kernels: (n_mols, max_atoms) int32, padded with zeros beyond N[i].
    # Q for elemental RFF kernels: list of 1D int32 arrays with 0-based element indices.
    Q_krr = np.zeros((n_mols, max_atoms), dtype=np.int32)
    for i, z in enumerate(z_list):
        Q_krr[i, : len(z)] = z
    Q_rff = [
        np.array([elem_to_idx[int(zi)] for zi in z_list[i]], dtype=np.int32) for i in range(n_mols)
    ]

    return X, dX, Q_krr, Q_rff, N


def compute_fchl19v2(
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
    """Compute FCHL19v2 representations for a list of molecules.

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
        Extra keyword arguments forwarded to fchl19v2_repr.generate_and_gradients
        or fchl19v2_repr.generate (e.g. nRs2, nRs3, rcut, eta2,
        two_body_type, three_body_type, nCosine, nRs3_minus, eta3_minus, ...).

    Returns
    -------
    X : ndarray, shape (n_mols, n_atoms, rep_size)
        Stacked FCHL19v2 representations.
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

    # Pin to 1 OpenMP thread for reproducibility — same reasoning as compute_fchl19.
    with _single_omp_thread():
        for coords, z in zip(coords_list, z_list, strict=True):
            coords_f64 = np.asarray(coords, dtype=np.float64)
            z_i32 = np.asarray(z, dtype=np.int32)

            if with_gradients:
                x, dx = fchl19v2_repr.generate_and_gradients(
                    coords_f64, z_i32, elements=elements, **repr_params
                )
                dX_list.append(dx)
            else:
                x = fchl19v2_repr.generate(coords_f64, z_i32, elements=elements, **repr_params)

            X_list.append(x)

    n_mols = len(coords_list)
    N = np.array([len(z) for z in z_list], dtype=np.int32)
    max_atoms = int(N.max())
    rep_size = X_list[0].shape[1]

    # Pad X to (n_mols, max_atoms, rep_size) — rows beyond N[i] are zero.
    X = np.zeros((n_mols, max_atoms, rep_size), dtype=np.float64)
    for i, x_i in enumerate(X_list):
        X[i, : len(x_i), :] = x_i

    # Pad dX to (n_mols, max_atoms, rep_size, max_atoms*3) — same zero-padding.
    dX: NDArray[np.float64] | None
    if with_gradients:
        dX = np.zeros((n_mols, max_atoms, rep_size, max_atoms * 3), dtype=np.float64)
        for i, dx_i in enumerate(dX_list):
            n_i = len(z_list[i])
            # dx_i shape: (n_i, rep_size, n_i*3) — pad both atom dims
            dX[i, :n_i, :, : n_i * 3] = dx_i
    else:
        dX = None

    # Q for local KRR kernels: (n_mols, max_atoms) int32, padded with zeros beyond N[i].
    # Q for elemental RFF kernels: list of 1D int32 arrays with 0-based element indices.
    Q_krr = np.zeros((n_mols, max_atoms), dtype=np.int32)
    for i, z in enumerate(z_list):
        Q_krr[i, : len(z)] = z
    Q_rff = [
        np.array([elem_to_idx[int(zi)] for zi in z_list[i]], dtype=np.int32) for i in range(n_mols)
    ]

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
