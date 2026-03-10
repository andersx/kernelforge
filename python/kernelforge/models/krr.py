"""LocalKRRModel: FCHL19-based local Kernel Ridge Regression model."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernelforge import kernelmath, local_kernels

from .base import BaseModel, TrainingMode
from .representations import compute_fchl19

# Default FCHL19 representation hyperparameters (same as the examples)
_DEFAULT_REPR_PARAMS: dict[str, Any] = {}

# Default elements for organic chemistry (H, C, N, O, S)
_DEFAULT_ELEMENTS: list[int] = [1, 6, 7, 8, 16]


class LocalKRRModel(BaseModel):
    """Kernel Ridge Regression model using FCHL19 local representations.

    Supports three training modes (inferred automatically from inputs):
      - ``energy_only``      — train on energies, predict E + F via Jacobian kernel
      - ``force_only``       — train on forces,   predict F + E via Hessian kernel
      - ``energy_and_force`` — train on E + F jointly via the full combined kernel

    All kernels use the symmetric RFP-packed format for training and the
    Cholesky solver from :mod:`kernelforge.kernelmath`.

    Parameters
    ----------
    sigma : float
        Gaussian kernel length-scale.
    l2 : float
        L2 regularisation strength added to the kernel diagonal.
    elements : list[int]
        Sorted list of atomic numbers present in the dataset.
        Used to build element-indexed Q arrays.
    repr_params : dict, optional
        Extra keyword arguments forwarded to
        :func:`~kernelforge.fchl19_repr.generate_fchl_acsf_and_gradients`,
        e.g. ``nRs2``, ``nRs3``, ``rcut``, ``eta2``, ...

    Examples
    --------
    >>> model = LocalKRRModel(sigma=2.0, l2=1e-9, elements=[1, 6, 8])
    >>> model.fit(coords_list, z_list, energies=E, forces=F)
    >>> E_pred, F_pred = model.predict(coords_test, z_test)
    >>> model.save("model.npz")
    >>> model2 = LocalKRRModel.load("model.npz")
    """

    def __init__(
        self,
        sigma: float = 2.0,
        l2: float = 1e-9,
        elements: list[int] | None = None,
        repr_params: dict[str, Any] | None = None,
    ) -> None:
        self.sigma = sigma
        self.l2 = l2
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

        X, dX, Q_krr, _Q_rff, N = compute_fchl19(
            coords_list,
            z_list,
            self.elements,
            with_gradients=True,
            repr_params=self.repr_params,
        )

        # Store training representations for prediction kernels
        self._X_tr = X
        self._dX_tr = dX
        self._Q_tr = Q_krr
        self._N_tr = N
        self._n_train = len(coords_list)

        n_atoms = X.shape[1]
        self._n_atoms = n_atoms

        if mode == "energy_only":
            if energies is None:
                msg = "energies must be provided for energy_only mode"
                raise ValueError(msg)
            K_rfp = local_kernels.kernel_gaussian_symm_rfp(X, Q_krr, N, self.sigma)
            self._y_train = energies
            self._alpha = kernelmath.cho_solve_rfp(K_rfp, energies, l2=self.l2)

        elif mode == "force_only":
            if forces is None:
                msg = "forces must be provided for force_only mode"
                raise ValueError(msg)
            if dX is None:
                msg = "dX is None in force_only mode — internal error"
                raise RuntimeError(msg)
            F_flat = forces.ravel()
            K_rfp = local_kernels.kernel_gaussian_hessian_symm_rfp(X, dX, Q_krr, N, self.sigma)
            self._y_train = F_flat
            self._alpha = kernelmath.cho_solve_rfp(K_rfp, F_flat, l2=self.l2)

        else:  # energy_and_force
            if energies is None:
                msg = "energies must be provided for energy_and_force mode"
                raise ValueError(msg)
            if forces is None:
                msg = "forces must be provided for energy_and_force mode"
                raise ValueError(msg)
            if dX is None:
                msg = "dX is None in energy_and_force mode — internal error"
                raise RuntimeError(msg)
            # Forces stored with negated sign for full-kernel convention:
            # the full kernel computes +dK/dR for the force-energy block,
            # so we pass -F (= +dE/dR) as training labels.
            F_neg = -forces  # (n_train, n_atoms*3)  dE/dR = -F
            y_tr = np.concatenate([energies, F_neg.ravel()])
            K_rfp = local_kernels.kernel_gaussian_full_symm_rfp(X, dX, Q_krr, N, self.sigma)
            self._y_train = y_tr
            self._alpha = kernelmath.cho_solve_rfp(K_rfp, y_tr, l2=self.l2)

    # ------------------------------------------------------------------
    # Internal predict
    # ------------------------------------------------------------------

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mode = self.training_mode_

        # Gradients are needed for all modes: force_only and energy_and_force need
        # them directly; energy_only uses the Jacobian kernel to also predict forces.
        X_te, dX_te, Q_te, _Q_rff_te, N_te = compute_fchl19(
            coords_list,
            z_list,
            self.elements,
            with_gradients=True,
            repr_params=self.repr_params,
        )

        n_test = len(coords_list)

        X_tr = self._X_tr
        dX_tr = self._dX_tr
        Q_tr = self._Q_tr
        N_tr = self._N_tr
        alpha = self._alpha

        if mode == "energy_only":
            if dX_te is None:
                msg = "dX_te is None in energy_only predict — internal error"
                raise RuntimeError(msg)
            K_e = local_kernels.kernel_gaussian(X_te, X_tr, Q_te, Q_tr, N_te, N_tr, self.sigma)
            E_pred = K_e @ alpha  # (n_test,)
            # Forces via transposed Jacobian kernel: dK/dR_te, shape (n_test*naq, n_train)
            # F = -dE/dR = -(K_jt @ alpha), flat (sum(N_te)*3,)
            K_jt = local_kernels.kernel_gaussian_jacobian_t(
                X_te, X_tr, dX_te, Q_te, Q_tr, N_te, N_tr, self.sigma
            )
            F_pred = (K_jt @ alpha)  # flat (sum(N_te)*3,)

        elif mode == "force_only":
            if dX_te is None or dX_tr is None:
                msg = "dX_te or dX_tr is None in force_only predict — internal error"
                raise RuntimeError(msg)
            K_hess = local_kernels.kernel_gaussian_hessian(
                X_te, X_tr, dX_te, dX_tr, Q_te, Q_tr, N_te, N_tr, self.sigma
            )
            F_pred = K_hess @ alpha  # flat (sum(N_te)*3,)

            # Energy via Jacobian kernel: K_j[i, j*naq+d] = dK(i,j)/dR_j[d]
            K_j = local_kernels.kernel_gaussian_jacobian(
                X_te, X_tr, dX_tr, Q_te, Q_tr, N_te, N_tr, self.sigma
            )
            E_pred = -(K_j @ alpha)

        else:  # energy_and_force
            if dX_te is None or dX_tr is None:
                msg = "dX_te or dX_tr is None in energy_and_force predict — internal error"
                raise RuntimeError(msg)
            K_full = local_kernels.kernel_gaussian_full(
                X_te, X_tr, dX_te, dX_tr, Q_te, Q_tr, N_te, N_tr, self.sigma
            )
            y_pred = K_full @ alpha  # (n_test + sum(N_te)*3,)
            E_pred = y_pred[:n_test]
            # Negate back: we stored -F as training labels, so predictions are -F_pred.
            # Return flat (sum(N_te)*3,) — works for both fixed- and variable-size molecules.
            F_pred = -y_pred[n_test:]

        return E_pred, F_pred

    # ------------------------------------------------------------------
    # Training score (cheap path — no kernel evaluation needed)
    # ------------------------------------------------------------------

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        # y_pred_train = y_train - l2 * alpha  (from (K + l2·I)·alpha = y_train)
        y_pred = self._y_train - self.l2 * self._alpha
        mode = self.training_mode_
        n = self._n_train
        if mode == "energy_only":
            return {"energy": (self._y_train, y_pred)}
        elif mode == "force_only":
            return {"force": (self._y_train, y_pred)}
        else:  # energy_and_force — y_train = [E; -F.ravel()], undo sign on forces
            return {
                "energy": (self._y_train[:n], y_pred[:n]),
                "force": (-self._y_train[n:], -y_pred[n:]),
            }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
        d: dict[str, object] = {
            "model_class": "LocalKRRModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "elements": np.array(self.elements, dtype=np.int32),
            "repr_params": json.dumps(self.repr_params),
            "baseline_elements": self.baseline_elements_,
            "element_energies": self.element_energies_,
            "alpha": self._alpha,
            "y_train": self._y_train,
            "X_tr": self._X_tr,
            "Q_tr": self._Q_tr,
            "N_tr": self._N_tr,
        }
        if self._dX_tr is not None:
            d["dX_tr"] = self._dX_tr
        return d

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.elements = data["elements"].tolist()
        self.repr_params = json.loads(str(data["repr_params"]))
        self.baseline_elements_ = data["baseline_elements"].astype(np.int32)
        self.element_energies_ = data["element_energies"].astype(np.float64)
        self.training_mode_: TrainingMode = str(data["training_mode"])  # type: ignore[assignment]
        self._alpha = data["alpha"]
        self._y_train = data["y_train"] if "y_train" in data else np.array([], dtype=np.float64)
        self._X_tr = data["X_tr"]
        self._Q_tr = data["Q_tr"]
        self._N_tr = data["N_tr"]
        self._dX_tr = data.get("dX_tr", None)
        self._n_train = len(self._N_tr)
        self._n_atoms = self._X_tr.shape[1]
