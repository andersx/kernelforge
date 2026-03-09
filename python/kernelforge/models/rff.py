"""LocalRFFModel: FCHL19-based local Random Fourier Features regression model."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernelforge import kernelmath
from kernelforge.kitchen_sinks import (
    rff_features_elemental,
    rff_full_elemental,
    rff_full_gramian_elemental_rfp,
    rff_gradient_elemental,
    rff_gradient_gramian_elemental_rfp,
    rff_gramian_elemental_rfp,
)

from .base import BaseModel, TrainingMode
from .representations import compute_fchl19

_DEFAULT_ELEMENTS: list[int] = [1, 6, 7, 8, 16]


class LocalRFFModel(BaseModel):
    """Random Fourier Features regression model using FCHL19 local representations.

    Approximates the local Gaussian kernel via the Bochner / Rahimi-Recht
    elemental feature mapping, maintaining separate random weights per element
    type.  Solves a D_rff x D_rff normal-equations system (much cheaper than the
    N x N KRR system for large datasets).

    Supports three training modes (inferred automatically from inputs):
      - ``energy_only``      — train on energies, predict E + F via gradient features
      - ``force_only``       — train on forces,   predict F + E via energy features
      - ``energy_and_force`` — train on E + F jointly via full elemental features

    Parameters
    ----------
    sigma : float
        Gaussian kernel length-scale (determines RFF frequency scale W/sigma).
    l2 : float
        L2 regularisation added to the normal-equations diagonal.
    d_rff : int
        Number of random Fourier features (feature dimension).
    seed : int
        Random seed for reproducible random weight generation.
    elements : list[int]
        Sorted list of atomic numbers present in the dataset.
    repr_params : dict, optional
        Extra keyword arguments forwarded to
        :func:`~kernelforge.fchl19_repr.generate_fchl_acsf_and_gradients`.

    Examples
    --------
    >>> model = LocalRFFModel(sigma=20.0, l2=1e-6, d_rff=4096, elements=[1, 6, 8])
    >>> model.fit(coords_list, z_list, energies=E, forces=F)
    >>> E_pred, F_pred = model.predict(coords_test, z_test)
    >>> model.save("rff_model.npz")
    >>> model2 = LocalRFFModel.load("rff_model.npz")
    """

    def __init__(
        self,
        sigma: float = 20.0,
        l2: float = 1e-6,
        d_rff: int = 4096,
        seed: int = 42,
        elements: list[int] | None = None,
        repr_params: dict[str, Any] | None = None,
    ) -> None:
        self.sigma = sigma
        self.l2 = l2
        self.d_rff = d_rff
        self.seed = seed
        self.elements: list[int] = sorted(elements) if elements is not None else _DEFAULT_ELEMENTS
        self.repr_params: dict[str, Any] = repr_params if repr_params is not None else {}
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # Internal fit
    # ------------------------------------------------------------------

    def _fit(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        energies: NDArray[np.float64] | None,
        forces: NDArray[np.float64] | None,
    ) -> None:
        mode = self.training_mode_

        X, dX, _Q_krr, Q_rff, _N = compute_fchl19(
            coords_list, z_list, self.elements, with_gradients=True, repr_params=self.repr_params
        )
        # dX always computed (with_gradients=True) because we need it for
        # force prediction even when training energy-only.
        if dX is None:
            msg = "dX is None after compute_fchl19 with with_gradients=True — internal error"
            raise RuntimeError(msg)

        n_mols = X.shape[0]
        n_atoms = X.shape[1]
        rep_size = X.shape[2]
        # Elemental RFF functions expect dX shape (n_mols, n_atoms, rep_size, n_atoms, 3).
        # compute_fchl19 returns 4D (n_mols, n_atoms, rep_size, n_atoms*3) — reshape here.
        dX = dX.reshape(n_mols, n_atoms, rep_size, n_atoms, 3)
        nelements = len(self.elements)

        # Generate random Fourier feature weights
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((nelements, rep_size, self.d_rff)) / self.sigma
        b = rng.uniform(0.0, 2.0 * np.pi, (nelements, self.d_rff))

        # Store W, b for prediction
        self._W = W
        self._b = b
        self._n_atoms = X.shape[1]

        if mode == "energy_only":
            if energies is None:
                msg = "energies must be provided for energy_only mode"
                raise ValueError(msg)
            ZtZ_rfp, ZtY = rff_gramian_elemental_rfp(X, Q_rff, W, b, energies)
            self._weights = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=self.l2)

        elif mode == "force_only":
            if forces is None:
                msg = "forces must be provided for force_only mode"
                raise ValueError(msg)
            F_flat = forces.ravel()
            GtG_rfp, GtF = rff_gradient_gramian_elemental_rfp(X, dX, Q_rff, W, b, F_flat)
            self._weights = kernelmath.cho_solve_rfp(GtG_rfp, GtF, l2=self.l2)

        else:  # energy_and_force
            if energies is None:
                msg = "energies must be provided for energy_and_force mode"
                raise ValueError(msg)
            if forces is None:
                msg = "forces must be provided for energy_and_force mode"
                raise ValueError(msg)
            # RFF full kernel uses physical forces directly (no sign flip needed —
            # the elemental RFF conventions match the training target sign)
            ZtZ_rfp, ZtY = rff_full_gramian_elemental_rfp(
                X, dX, Q_rff, W, b, energies, forces.ravel()
            )
            self._weights = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=self.l2)

    # ------------------------------------------------------------------
    # Internal predict
    # ------------------------------------------------------------------

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mode = self.training_mode_

        X_te, dX_te, _Q_krr_te, Q_rff_te, _N_te = compute_fchl19(
            coords_list, z_list, self.elements, with_gradients=True, repr_params=self.repr_params
        )
        if dX_te is None:
            msg = "dX_te is None after compute_fchl19 with with_gradients=True — internal error"
            raise RuntimeError(msg)

        n_test = len(coords_list)
        n_atoms_te = X_te.shape[1]
        rep_size_te = X_te.shape[2]
        # Reshape dX_te from 4D (n_test, n_atoms, rep_size, n_atoms*3)
        # to 5D (n_test, n_atoms, rep_size, n_atoms, 3) for elemental RFF functions.
        dX_te = dX_te.reshape(n_test, n_atoms_te, rep_size_te, n_atoms_te, 3)
        # Recover n_atoms from the computed representation if not yet set
        if self._n_atoms == 0:
            self._n_atoms = X_te.shape[1]
        naq = self._n_atoms * 3
        W = self._W
        b = self._b
        w = self._weights

        if mode == "energy_only":
            Z_te = rff_features_elemental(X_te, Q_rff_te, W, b)  # (n_test, d_rff)
            E_pred = Z_te @ w
            # Centre: energy-only has no absolute offset
            E_pred = E_pred - E_pred.mean()

            G_te = rff_gradient_elemental(X_te, dX_te, Q_rff_te, W, b)  # (d_rff, n_test*naq)
            F_pred = (G_te.T @ w).reshape(n_test, naq)

        elif mode == "force_only":
            G_te = rff_gradient_elemental(X_te, dX_te, Q_rff_te, W, b)  # (d_rff, n_test*naq)
            F_pred = (G_te.T @ w).reshape(n_test, naq)

            Z_te = rff_features_elemental(X_te, Q_rff_te, W, b)  # (n_test, d_rff)
            E_pred = Z_te @ w
            # Centre: force-only has no absolute energy offset
            E_pred = E_pred - E_pred.mean()

        else:  # energy_and_force
            Z_full_te = rff_full_elemental(X_te, dX_te, Q_rff_te, W, b)  # (n_test*(1+naq), d_rff)
            y_pred = Z_full_te @ w
            E_pred = y_pred[:n_test]
            F_pred = y_pred[n_test:].reshape(n_test, naq)

        return E_pred, F_pred

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
        return {
            "model_class": "LocalRFFModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "d_rff": self.d_rff,
            "seed": self.seed,
            "elements": np.array(self.elements, dtype=np.int32),
            "repr_params": json.dumps(self.repr_params),
            "energy_mean": self.energy_mean_,
            "weights": self._weights,
            "W": self._W,
            "b": self._b,
        }

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.d_rff = int(data["d_rff"])
        self.seed = int(data["seed"])
        self.elements = data["elements"].tolist()
        self.repr_params = json.loads(str(data["repr_params"]))
        self.energy_mean_ = float(data["energy_mean"])
        self.training_mode_: TrainingMode = str(data["training_mode"])  # type: ignore[assignment]
        self._weights = data["weights"]
        self._W = data["W"]
        self._b = data["b"]
        # Recover n_atoms from W shape: (nelements, rep_size, d_rff)
        # not stored explicitly, recover from first prediction call via repr
        # Store d_rff for quick access
        self.d_rff = self._W.shape[2]
        # n_atoms is recovered lazily in _predict via the representation shape
        self._n_atoms: int = 0  # will be set on first predict call if needed
