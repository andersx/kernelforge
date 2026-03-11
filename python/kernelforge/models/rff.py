"""LocalRFFModel: FCHL19/FCHL19v2-based local Random Fourier Features regression model."""

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
from .representations import compute_fchl19, compute_fchl19v2

_DEFAULT_ELEMENTS: list[int] = [1, 6, 7, 8, 16]


class LocalRFFModel(BaseModel):
    """Random Fourier Features regression model using FCHL19 or FCHL19v2 local representations.

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
        Extra keyword arguments forwarded to the representation generator.
        For ``fchl19v2``: also accepts ``two_body_type``, ``three_body_type``,
        ``nCosine``, ``nRs3_minus``, ``eta3_minus``.
    representation : str
        Which representation to use: ``"fchl19"`` (default) or ``"fchl19v2"``.

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
        representation: str = "fchl19",
    ) -> None:
        if representation not in ("fchl19", "fchl19v2"):
            msg = f"representation must be 'fchl19' or 'fchl19v2', got {representation!r}"
            raise ValueError(msg)
        self.sigma = sigma
        self.l2 = l2
        self.d_rff = d_rff
        self.seed = seed
        self.elements: list[int] = sorted(elements) if elements is not None else _DEFAULT_ELEMENTS
        self.repr_params: dict[str, Any] = repr_params if repr_params is not None else {}
        self.representation = representation
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

        _compute_repr = compute_fchl19v2 if self.representation == "fchl19v2" else compute_fchl19
        X, dX, _Q_krr, Q_rff, _N = _compute_repr(
            coords_list,
            z_list,
            self.elements,
            with_gradients=True,
            repr_params=self.repr_params,
        )

        n_mols = X.shape[0]
        n_atoms = X.shape[1]
        rep_size = X.shape[2]
        nelements = len(self.elements)
        self._n_train = n_mols

        # Generate random Fourier feature weights
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((nelements, rep_size, self.d_rff)) / self.sigma
        b = rng.uniform(0.0, 2.0 * np.pi, (nelements, self.d_rff))

        self._W = W
        self._b = b

        if mode == "energy_only":
            if energies is None:
                msg = "energies must be provided for energy_only mode"
                raise ValueError(msg)
            ZtZ_rfp, ZtY = rff_gramian_elemental_rfp(X, Q_rff, W, b, energies)
            self._y_train = energies
            self._weights = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=self.l2)
            # Training predictions: Z_tr @ weights  (O(n·d), cheap vs O(n²) kernel)
            Z_tr = rff_features_elemental(X, Q_rff, W, b)
            self._y_pred_train = Z_tr @ self._weights

        elif mode == "force_only":
            if forces is None:
                msg = "forces must be provided for force_only mode"
                raise ValueError(msg)
            if dX is None:
                msg = "dX is None in force_only fit — internal error"
                raise RuntimeError(msg)
            # Elemental RFF functions expect dX shape (n_mols, n_atoms, rep_size, n_atoms, 3).
            # compute_fchl19 returns 4D (n_mols, n_atoms, rep_size, n_atoms*3) — reshape here.
            dX_5d: NDArray[np.float64] = dX.reshape(n_mols, n_atoms, rep_size, n_atoms, 3)
            F_flat = forces.ravel()
            GtG_rfp, GtF = rff_gradient_gramian_elemental_rfp(X, dX_5d, Q_rff, W, b, F_flat)
            self._y_train = F_flat
            self._weights = kernelmath.cho_solve_rfp(GtG_rfp, GtF, l2=self.l2)
            G_tr = rff_gradient_elemental(X, dX_5d, Q_rff, W, b)
            self._y_pred_train = G_tr.T @ self._weights

        else:  # energy_and_force
            if energies is None:
                msg = "energies must be provided for energy_and_force mode"
                raise ValueError(msg)
            if forces is None:
                msg = "forces must be provided for energy_and_force mode"
                raise ValueError(msg)
            if dX is None:
                msg = "dX is None in energy_and_force fit — internal error"
                raise RuntimeError(msg)
            dX_5d: NDArray[np.float64] = np.asarray(dX).reshape(
                n_mols, n_atoms, rep_size, n_atoms, 3
            )
            # RFF full kernel uses physical forces directly (no sign flip needed —
            # the elemental RFF conventions match the training target sign)
            F_flat = forces.ravel()
            y_tr = np.concatenate([energies, F_flat])
            ZtZ_rfp, ZtY = rff_full_gramian_elemental_rfp(X, dX_5d, Q_rff, W, b, energies, F_flat)
            self._y_train = y_tr
            self._weights = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=self.l2)
            Z_full_tr = rff_full_elemental(X, dX_5d, Q_rff, W, b)
            self._y_pred_train = Z_full_tr @ self._weights

    # ------------------------------------------------------------------
    # Training score (cheap path — no kernel evaluation needed)
    # ------------------------------------------------------------------

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        # y_pred_train = Z_train @ weights, computed and stored during _fit.
        y_pred = self._y_pred_train
        mode = self.training_mode_
        if mode == "energy_only":
            return {"energy": (self._y_train, y_pred)}
        elif mode == "force_only":
            return {"force": (self._y_train, y_pred)}
        else:  # energy_and_force — y_train = [E; F.ravel()] (no sign flip in RFF)
            n_tr = self._n_train
            return {
                "energy": (self._y_train[:n_tr], y_pred[:n_tr]),
                "force": (self._y_train[n_tr:], y_pred[n_tr:]),
            }

    # ------------------------------------------------------------------
    # Internal predict
    # ------------------------------------------------------------------

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mode = self.training_mode_

        _compute_repr = compute_fchl19v2 if self.representation == "fchl19v2" else compute_fchl19
        X_te, dX_te, _Q_krr_te, Q_rff_te, _N_te = _compute_repr(
            coords_list,
            z_list,
            self.elements,
            with_gradients=True,
            repr_params=self.repr_params,
        )

        n_test = len(coords_list)
        n_atoms_te = X_te.shape[1]
        rep_size_te = X_te.shape[2]
        W = self._W
        b = self._b
        w = self._weights

        if mode == "energy_only":
            if dX_te is None:
                msg = "dX_te is None in energy_only predict — internal error"
                raise RuntimeError(msg)
            dX_te_5d: NDArray[np.float64] = dX_te.reshape(
                n_test, n_atoms_te, rep_size_te, n_atoms_te, 3
            )
            Z_te = rff_features_elemental(X_te, Q_rff_te, W, b)  # (n_test, d_rff)
            E_pred = Z_te @ w
            # Forces via gradient features: F = -dE/dR
            G_te = rff_gradient_elemental(X_te, dX_te_5d, Q_rff_te, W, b)  # (d_rff, n_test*naq)
            F_pred = G_te.T @ w  # flat (sum(N_te)*3,)

        elif mode == "force_only":
            if dX_te is None:
                msg = "dX_te is None in force_only predict — internal error"
                raise RuntimeError(msg)
            dX_te_5d: NDArray[np.float64] = dX_te.reshape(
                n_test, n_atoms_te, rep_size_te, n_atoms_te, 3
            )
            G_te = rff_gradient_elemental(X_te, dX_te_5d, Q_rff_te, W, b)  # (d_rff, n_test*naq)
            F_pred = G_te.T @ w  # flat (sum(N_te)*3,)

            Z_te = rff_features_elemental(X_te, Q_rff_te, W, b)  # (n_test, d_rff)
            E_pred = Z_te @ w

        else:  # energy_and_force
            if dX_te is None:
                msg = "dX_te is None in energy_and_force predict — internal error"
                raise RuntimeError(msg)
            dX_te_5d = dX_te.reshape(n_test, n_atoms_te, rep_size_te, n_atoms_te, 3)
            Z_full_te = rff_full_elemental(
                X_te, dX_te_5d, Q_rff_te, W, b
            )  # (n_test*(1+naq), d_rff)
            y_pred = Z_full_te @ w
            E_pred = y_pred[:n_test]
            F_pred = y_pred[n_test:]  # flat (sum(N_te)*3,)

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
            "representation": self.representation,
            "baseline_elements": self.baseline_elements_,
            "element_energies": self.element_energies_,
            "weights": self._weights,
            "W": self._W,
            "b": self._b,
            "n_train": self._n_train,
            "y_train": self._y_train,
            "y_pred_train": self._y_pred_train,
        }

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.d_rff = int(data["d_rff"])
        self.seed = int(data["seed"])
        self.elements = data["elements"].tolist()
        self.repr_params = json.loads(str(data["repr_params"]))
        # representation key may be absent in older saved models — default to fchl19
        self.representation = str(data["representation"]) if "representation" in data else "fchl19"
        self.baseline_elements_ = data["baseline_elements"].astype(np.int32)
        self.element_energies_ = data["element_energies"].astype(np.float64)
        self.training_mode_: TrainingMode = str(data["training_mode"])  # type: ignore[assignment]
        self._weights = data["weights"]
        self._W = data["W"]
        self._b = data["b"]
        # d_rff is already set above, but sync from W shape to be safe
        self.d_rff = self._W.shape[2]
        self._n_train = int(data["n_train"]) if "n_train" in data else 0
        self._y_train = data["y_train"] if "y_train" in data else np.array([], dtype=np.float64)
        self._y_pred_train = (
            data["y_pred_train"] if "y_pred_train" in data else np.array([], dtype=np.float64)
        )
