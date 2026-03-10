"""Base class shared by all kernelforge high-level models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TrainingMode = Literal["energy_only", "force_only", "energy_and_force"]


@dataclass
class ModelScore:
    """Regression statistics for a single predicted quantity.

    Attributes
    ----------
    mae:
        Mean absolute error in the same units as the training data.
    pearson_r:
        Pearson correlation coefficient between predictions and reference.
    slope:
        Slope of the linear fit  y_pred ~ slope * y_ref + intercept.
    intercept:
        Intercept of the same linear fit.
    """

    mae: float
    pearson_r: float
    slope: float
    intercept: float

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:.4f}  r={self.pearson_r:.6f}  "
            f"slope={self.slope:.6f}  intercept={self.intercept:.4f}"
        )


def _compute_score(y_ref: NDArray[np.float64], y_pred: NDArray[np.float64]) -> ModelScore:
    """Compute MAE, Pearson r, slope and intercept from flat reference/prediction arrays."""
    ref = y_ref.ravel()
    pred = y_pred.ravel()
    mae = float(np.mean(np.abs(pred - ref)))
    # Pearson r via corrcoef — [0,1] element of the 2x2 matrix
    r = float(np.corrcoef(ref, pred)[0, 1])
    # Linear fit: pred = slope * ref + intercept
    slope, intercept = np.polyfit(ref, pred, 1)
    return ModelScore(mae=mae, pearson_r=r, slope=float(slope), intercept=float(intercept))


class BaseModel(ABC):
    """Abstract base class for kernelforge high-level models.

    Subclasses must implement _fit and _predict.  This base class handles:
      - Training mode inference from which of energies/forces are provided
      - Input validation and dtype coercion
      - Per-element energy baseline subtraction before fit, restored after predict
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

        When energies are provided, a per-element energy baseline is fitted
        via linear regression (E ~ sum_z count(z) * e_z) and subtracted
        before training.  It is added back automatically at prediction time.

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

        # Fit and subtract per-element energy baseline when energies are available.
        # For force-only training, store a zero baseline (no energies to regress).
        if E is not None:
            baseline_elements, element_energies = _fit_element_baseline(z_list, E)
            self.baseline_elements_: NDArray[np.int32] = baseline_elements
            self.element_energies_: NDArray[np.float64] = element_energies
            E = E - _build_composition_matrix(z_list, baseline_elements) @ element_energies
        else:
            self.baseline_elements_ = np.array([], dtype=np.int32)
            self.element_energies_ = np.array([], dtype=np.float64)

        self._fit(coords_list, z_list, E, F)
        self.train_score_: dict[str, ModelScore] = self._compute_train_score()
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
        forces : ndarray, shape (M, n_atoms*3)
            Predicted physical forces F = -dE/dR.
        """
        _check_fitted(self)
        coords_list, z_list = _validate_geometry(coords_list, z_list)

        E_pred, F_pred = self._predict(coords_list, z_list)

        # Restore the per-element energy baseline.
        if len(self.baseline_elements_) > 0:
            A_te = _build_composition_matrix(z_list, self.baseline_elements_)
            E_pred = E_pred + A_te @ self.element_energies_

        return E_pred, F_pred

    def score(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        energies: NDArray[np.float64] | None = None,
        forces: list[NDArray[np.float64]] | NDArray[np.float64] | None = None,
    ) -> dict[str, ModelScore]:
        """Compute regression statistics against reference energies and/or forces.

        Calls :meth:`predict` internally and compares predictions to the
        provided reference values.  Pass ``energies``, ``forces``, or both;
        the returned dict contains a key for each quantity scored.

        Parameters
        ----------
        coords_list:
            List of M (n_atoms_i, 3) coordinate arrays.
        z_list:
            List of M (n_atoms_i,) nuclear charge arrays.
        energies:
            Reference energies, shape (M,).
        forces:
            Reference physical forces F = -dE/dR.  Either a list of
            (n_atoms_i, 3) arrays or a stacked (M, n_atoms*3) ndarray.

        Returns
        -------
        scores : dict with keys ``"energy"`` and/or ``"force"``
            Each value is a :class:`ModelScore` with ``mae``, ``pearson_r``,
            ``slope``, and ``intercept``.

        Raises
        ------
        ValueError
            If neither energies nor forces are provided.
        ValueError
            If forces are requested but the model was trained in
            ``energy_only`` mode (no force predictions available).
        """
        if energies is None and forces is None:
            msg = "At least one of `energies` or `forces` must be provided to score()."
            raise ValueError(msg)

        E_pred, F_pred = self.predict(coords_list, z_list)
        scores: dict[str, ModelScore] = {}

        if energies is not None:
            E_ref = np.asarray(energies, dtype=np.float64).ravel()
            scores["energy"] = _compute_score(E_ref, E_pred)

        if forces is not None:
            if self.training_mode_ == "energy_only":
                msg = (
                    "Cannot score forces: model was trained in energy_only mode "
                    "and does not predict forces."
                )
                raise ValueError(msg)
            F_ref = _coerce_forces(forces)
            scores["force"] = _compute_score(F_ref, F_pred)

        return scores

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
        """Internal predict. Returns (E_pred_baseline_subtracted, F_pred)."""

    @abstractmethod
    def _arrays_to_save(self) -> dict[str, object]:
        """Return a dict of arrays/scalars suitable for np.savez."""

    @abstractmethod
    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        """Reconstruct model state from np.load output."""

    @abstractmethod
    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return (y_ref, y_pred) pairs for training data, keyed by quantity.

        Must be implemented by each subclass after _fit() stores the necessary
        state (y_train and alpha/weights).  Uses the identity:

            y_pred_train = y_train - l2 * alpha

        which follows from  (K + l2·I) @ alpha = y_train.

        Returns a dict with keys ``"energy"`` and/or ``"force"``, each
        mapping to a ``(y_ref, y_pred)`` tuple of flat float64 arrays.
        Both arrays are in baseline-subtracted units.
        """

    def _compute_train_score(self) -> dict[str, ModelScore]:
        """Compute training scores from stored labels and cheap predictions."""
        pairs = self._training_labels_and_predictions()
        return {k: _compute_score(ref, pred) for k, (ref, pred) in pairs.items()}


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
    rows = [np.asarray(fi, dtype=np.float64).ravel() for fi in forces]
    # Fixed-size: all rows same length → stack into 2D (n_mols, n_atoms*3)
    # Variable-size: rows differ → concatenate into 1D flat array
    if len({len(r) for r in rows}) == 1:
        return np.array(rows, dtype=np.float64)
    return np.concatenate(rows)


def _build_composition_matrix(
    z_list: list[NDArray[np.int32]],
    elements: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Build a composition matrix A of shape (n_mols, n_elements).

    A[i, j] = number of atoms with nuclear charge elements[j] in molecule i.
    """
    n_mols = len(z_list)
    n_elem = len(elements)
    A = np.zeros((n_mols, n_elem), dtype=np.float64)
    for i, z in enumerate(z_list):
        for j, elem in enumerate(elements):
            A[i, j] = float(np.sum(z == elem))
    return A


def _fit_element_baseline(
    z_list: list[NDArray[np.int32]],
    energies: NDArray[np.float64],
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Fit per-element atomic energy contributions via least squares.

    Solves  A @ e = energies  (no intercept) where A[i,j] is the count
    of element j in molecule i.  Returns the unique elements (sorted) and
    the fitted per-element energies.
    """
    elements: NDArray[np.int32] = np.array(
        np.unique(np.concatenate([z.astype(np.int32) for z in z_list])),
        dtype=np.int32,
    )
    A = _build_composition_matrix(z_list, elements)
    # rcond=None uses the default (machine-precision) cutoff for singular values
    e_raw, _residuals, _rank, _sv = np.linalg.lstsq(A, energies, rcond=None)
    return elements, np.array(e_raw, dtype=np.float64)


def _check_fitted(model: BaseModel) -> None:
    if not getattr(model, "is_fitted_", False):
        msg = f"{type(model).__name__} is not fitted yet. Call fit() first."
        raise RuntimeError(msg)
