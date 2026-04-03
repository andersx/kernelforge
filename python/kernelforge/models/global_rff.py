"""GlobalRFFModel: inverse-distance global descriptor + global Random Fourier Features."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from kernelforge import kernelmath
from kernelforge.kitchen_sinks import (
    rff_features,
    rff_full,
    rff_full_gramian_symm_rfp,
    rff_gradient,
    rff_gradient_gramian_symm_rfp,
    rff_gramian_symm_rfp,
)

from .base import BaseModel, TrainingMode
from .global_krr import _build_repr


class GlobalRFFModel(BaseModel):
    """Random Fourier Features regression model using inverse-distance global descriptors.

    Approximates the global Gaussian kernel via the Bochner / Rahimi-Recht feature
    mapping.  Solves a ``d_rff x d_rff`` normal-equations system (much cheaper than
    the ``N x N`` KRR system for large datasets).

    Supports three training modes (inferred automatically from inputs):
      - ``energy_only``      — train on energies, predict E + F via gradient features
      - ``force_only``       — train on forces,   predict F + E via energy features
      - ``energy_and_force`` — train on E + F jointly via full features

    **All molecules in the dataset must have the same atom count** — the descriptor
    dimension M = N*(N-1)/2 is fixed by N.

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
    eps : float
        Numerical floor for interatomic distances (default 1e-12).

    Examples
    --------
    >>> model = GlobalRFFModel(sigma=3.0, l2=1e-6, d_rff=4096)
    >>> model.fit(coords_list, z_list, energies=E, forces=F)
    >>> E_pred, F_pred = model.predict(coords_test, z_test)
    >>> model.save("global_rff.npz")
    >>> model2 = GlobalRFFModel.load("global_rff.npz")
    """

    def __init__(
        self,
        sigma: float = 3.0,
        l2: float = 1e-6,
        d_rff: int = 4096,
        seed: int = 42,
        eps: float = 1e-12,
    ) -> None:
        self.sigma = sigma
        self.l2 = l2
        self.d_rff = d_rff
        self.seed = seed
        self.eps = eps
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
        needs_grad = mode in ("force_only", "energy_and_force")

        X, dX = _build_repr(coords_list, self.eps, with_gradients=needs_grad)
        self._n_train = len(coords_list)

        # Random Fourier Feature weights: W(M, d_rff), b(d_rff,)
        M = X.shape[1]
        rng = np.random.default_rng(self.seed)
        W: NDArray[np.float64] = rng.standard_normal((M, self.d_rff)) / self.sigma
        b: NDArray[np.float64] = rng.uniform(0.0, 2.0 * np.pi, self.d_rff)
        self._W = W
        self._b = b

        if mode == "energy_only":
            if energies is None:
                msg = "energies must be provided for energy_only mode"
                raise ValueError(msg)
            ZtZ_rfp, ZtY = rff_gramian_symm_rfp(X, W, b, energies)
            self._y_train = energies
            self._weights = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=self.l2)
            Z_tr = rff_features(X, W, b)
            self._y_pred_train = Z_tr @ self._weights

        elif mode == "force_only":
            if forces is None:
                msg = "forces must be provided for force_only mode"
                raise ValueError(msg)
            if dX is None:
                msg = "dX is None in force_only fit — internal error"
                raise RuntimeError(msg)
            F_flat = forces.ravel()
            GtG_rfp, GtF = rff_gradient_gramian_symm_rfp(X, dX, W, b, F_flat)
            self._y_train = F_flat
            self._weights = kernelmath.cho_solve_rfp(GtG_rfp, GtF, l2=self.l2)
            G_tr = rff_gradient(X, dX, W, b)  # (d_rff, N*D)
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
            F_flat = forces.ravel()
            # rff_full's energy features produce predictions with a sign flip vs
            # the gramian targets: rff_full @ w ≈ [-E; F].  Store y_train with the
            # same sign convention ([-E; F]) so training scores compare correctly.
            y_tr = np.concatenate([-energies, F_flat])
            ZtZ_rfp, ZtY = rff_full_gramian_symm_rfp(X, dX, W, b, energies, F_flat)
            self._y_train = y_tr
            self._weights = kernelmath.cho_solve_rfp(ZtZ_rfp, ZtY, l2=self.l2)
            Z_full_tr = rff_full(X, dX, W, b)  # (N*(1+D), d_rff)
            self._y_pred_train = Z_full_tr @ self._weights

    # ------------------------------------------------------------------
    # Training score
    # ------------------------------------------------------------------

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        y_pred = self._y_pred_train
        mode = self.training_mode_
        if mode == "energy_only":
            return {"energy": (self._y_train, y_pred)}
        elif mode == "force_only":
            return {"force": (self._y_train, y_pred)}
        else:  # energy_and_force — y_train = [E; F] (no sign flip in RFF)
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

        X_te, dX_te = _build_repr(coords_list, self.eps, with_gradients=True)
        if dX_te is None:
            msg = "dX_te is None in predict — internal error"
            raise RuntimeError(msg)

        n_test = len(coords_list)
        W = self._W
        b = self._b
        w = self._weights

        if mode == "energy_only":
            Z_te = rff_features(X_te, W, b)  # (n_test, d_rff)
            E_pred: NDArray[np.float64] = Z_te @ w
            G_te = rff_gradient(X_te, dX_te, W, b)  # (d_rff, n_test*D)
            F_pred: NDArray[np.float64] = -G_te.T @ w  # flat (n_test*D,)

        elif mode == "force_only":
            G_te = rff_gradient(X_te, dX_te, W, b)  # (d_rff, n_test*D)
            F_pred = G_te.T @ w  # flat (n_test*D,)
            Z_te = rff_features(X_te, W, b)  # (n_test, d_rff)
            E_pred = Z_te @ w

        else:  # energy_and_force
            Z_full_te = rff_full(X_te, dX_te, W, b)  # (n_test*(1+D), d_rff)
            y_pred = -Z_full_te @ w
            E_pred = y_pred[:n_test]
            F_pred = -y_pred[n_test:]  # flat (n_test*D,)

        return E_pred, F_pred

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
        return {
            "model_class": "GlobalRFFModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "d_rff": self.d_rff,
            "seed": self.seed,
            "eps": self.eps,
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
        self.eps = float(data["eps"]) if "eps" in data else 1e-12
        self.baseline_elements_ = data["baseline_elements"].astype(np.int32)
        self.element_energies_ = data["element_energies"].astype(np.float64)
        self.training_mode_: TrainingMode = cast(TrainingMode, str(data["training_mode"]))
        self._weights = data["weights"]
        self._W = data["W"]
        self._b = data["b"]
        self.d_rff = self._W.shape[1]
        self._n_train = int(data["n_train"]) if "n_train" in data else 0
        self._y_train = data["y_train"] if "y_train" in data else np.array([], dtype=np.float64)
        self._y_pred_train = (
            data["y_pred_train"] if "y_pred_train" in data else np.array([], dtype=np.float64)
        )
