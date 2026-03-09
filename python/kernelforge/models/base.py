"""Base class shared by all kernelforge high-level models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TrainingMode = Literal["energy_only", "force_only", "energy_and_force"]


class BaseModel(ABC):
    """Abstract base class for kernelforge high-level models.

    Subclasses must implement _fit and _predict.  This base class handles:
      - Training mode inference from which of energies/forces are provided
      - Input validation and dtype coercion
      - Energy mean-centering (subtracted before fit, added back after predict)
      - The save/load interface (delegating array serialization to subclasses)
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        energies: NDArray[np.float64] | None = None,
        forces: list[NDArray[np.float64]] | NDArray[np.float64] | None = None,
    ) -> BaseModel:
        """Train the model from raw coordinates and nuclear charges.

        The training mode is inferred automatically:
          - energies only  ->  energy_only
          - forces only    ->  force_only
          - both           ->  energy_and_force

        Forces must be physical forces F = -dE/dR (NOT gradients +dE/dR).
        The sign convention is handled internally.

        Parameters
        ----------
        coords_list:
            List of N (n_atoms_i, 3) float64 coordinate arrays.
        z_list:
            List of N (n_atoms_i,) int32 nuclear charge arrays.
        energies:
            Shape (N,) float64 array of molecular energies.
        forces:
            Either a list of N (n_atoms_i, 3) force arrays, or a stacked
            ndarray of shape (N, n_atoms, 3).  Physical forces F = -dE/dR.

        Returns
        -------
        self
        """
        coords_list, z_list = _validate_geometry(coords_list, z_list)
        mode = _infer_mode(energies, forces)
        self.training_mode_: TrainingMode = mode

        E: NDArray[np.float64] | None = None
        F: NDArray[np.float64] | None = None

        if energies is not None:
            E = np.asarray(energies, dtype=np.float64).ravel()

        if forces is not None:
            F = _coerce_forces(forces)

        # Centre energies — always subtract mean so the kernel doesn't have
        # to fit a large constant offset.
        if E is not None:
            self.energy_mean_: float = float(E.mean())
            E = E - self.energy_mean_
        else:
            self.energy_mean_ = 0.0

        self._fit(coords_list, z_list, E, F)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict energies and forces for new structures.

        Parameters
        ----------
        coords_list:
            List of M (n_atoms_i, 3) coordinate arrays.
        z_list:
            List of M (n_atoms_i,) nuclear charge arrays.

        Returns
        -------
        energies : ndarray, shape (M,)
            Predicted molecular energies in the same units as training data.
        forces : ndarray, shape (M, n_atoms*3) or (M, n_atoms, 3)
            Predicted physical forces F = -dE/dR.
            Shape matches the per-molecule force shape used in training.
        """
        _check_fitted(self)
        coords_list, z_list = _validate_geometry(coords_list, z_list)

        E_pred, F_pred = self._predict(coords_list, z_list)

        # Un-centre energies
        E_pred = E_pred + self.energy_mean_

        return E_pred, F_pred

    def save(self, path: str | Path) -> None:
        """Save the trained model to a .npz file.

        Parameters
        ----------
        path:
            Output file path. A `.npz` extension is added if absent.
        """
        _check_fitted(self)
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        arrays = self._arrays_to_save()
        np.savez(path, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Load a trained model from a .npz file.

        Parameters
        ----------
        path:
            Path to the .npz file produced by :meth:`save`.

        Returns
        -------
        model : instance of the calling class
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        model = cls.__new__(cls)
        model._load_from_arrays(data)
        model.is_fitted_ = True
        return model

    # ------------------------------------------------------------------
    # Abstract methods (implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        energies: NDArray[np.float64] | None,
        forces: NDArray[np.float64] | None,
    ) -> None:
        """Internal fit implementation. energies/forces are already coerced."""

    @abstractmethod
    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Internal predict. Returns (E_pred_centred, F_pred)."""

    @abstractmethod
    def _arrays_to_save(self) -> dict[str, object]:
        """Return a dict of arrays/scalars suitable for np.savez."""

    @abstractmethod
    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        """Reconstruct model state from np.load output."""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _infer_mode(
    energies: NDArray[np.float64] | None,
    forces: object | None,
) -> TrainingMode:
    has_e = energies is not None
    has_f = forces is not None
    if has_e and has_f:
        return "energy_and_force"
    if has_e:
        return "energy_only"
    if has_f:
        return "force_only"
    msg = "At least one of `energies` or `forces` must be provided."
    raise ValueError(msg)


def _validate_geometry(
    coords_list: list[NDArray[np.float64]],
    z_list: list[NDArray[np.int32]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.int32]]]:
    if len(coords_list) != len(z_list):
        msg = (
            f"coords_list and z_list must have the same length, "
            f"got {len(coords_list)} and {len(z_list)}"
        )
        raise ValueError(msg)
    coords_out = [np.asarray(r, dtype=np.float64) for r in coords_list]
    z_out = [np.asarray(z, dtype=np.int32) for z in z_list]
    return coords_out, z_out


def _coerce_forces(
    forces: list[NDArray[np.float64]] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert forces to a 2D array (n_mols, n_atoms*3)."""
    if isinstance(forces, np.ndarray):
        f = np.asarray(forces, dtype=np.float64)
        if f.ndim == 3:  # (n_mols, n_atoms, 3)
            return f.reshape(len(f), -1)
        if f.ndim == 2:  # already (n_mols, n_atoms*3)
            return f
        msg = f"forces ndarray must be 2D or 3D, got {f.ndim}D"
        raise ValueError(msg)
    # list of per-molecule arrays, each (n_atoms_i, 3) or (n_atoms_i*3,)
    rows = []
    for fi in forces:
        fi = np.asarray(fi, dtype=np.float64)
        rows.append(fi.ravel())
    return np.array(rows, dtype=np.float64)


def _check_fitted(model: BaseModel) -> None:
    if not getattr(model, "is_fitted_", False):
        msg = f"{type(model).__name__} is not fitted yet. Call fit() first."
        raise RuntimeError(msg)
