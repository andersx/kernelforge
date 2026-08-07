"""GlobalKRRModel: inverse-distance global descriptor + global Gaussian KRR."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from kernelforge import global_kernels, invdist_repr, kernelmath

from .base import BaseModel, TrainingMode


def _build_repr(
    coords_list: list[NDArray[np.float64]],
    eps: float,
    with_gradients: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Compute inverse-distance representations for a list of molecules.

    All molecules must have the same atom count (required for a fixed descriptor
    dimension M = N*(N-1)/2).

    Parameters
    ----------
    coords_list:
        List of (n_atoms, 3) float64 coordinate arrays.
    eps:
        Distance floor for numerical stability (passed to invdist_repr).
    with_gradients:
        If True, also return Jacobians dX of shape (n_mols, D, M)
        where D = 3*n_atoms.

    Returns
    -------
    X : ndarray, shape (n_mols, M)
    dX : ndarray, shape (n_mols, D, M), or None when with_gradients=False
    """
    n_atoms_list = [len(coords) for coords in coords_list]
    if len(set(n_atoms_list)) > 1:
        msg = (
            "GlobalKRRModel requires all molecules to have the same atom count. "
            f"Got atom counts: {sorted(set(n_atoms_list))}. "
            "Use a single-molecule dataset (e.g. rMD17) or pre-filter to fixed size."
        )
        raise ValueError(msg)

    X_list: list[NDArray[np.float64]] = []
    dX_list: list[NDArray[np.float64]] = []

    for coords in coords_list:
        coords_f64 = np.asarray(coords, dtype=np.float64)
        if with_gradients:
            x, dx = invdist_repr.inverse_distance_upper_and_jacobian(coords_f64, eps)
            dX_list.append(dx)  # (D, M)
        else:
            x = invdist_repr.inverse_distance_upper(coords_f64, eps)
        X_list.append(x)  # (M,)

    X = np.array(X_list, dtype=np.float64)  # (n_mols, M)
    dX: NDArray[np.float64] | None = np.array(dX_list, dtype=np.float64) if with_gradients else None
    return X, dX


class GlobalKRRModel(BaseModel):
    """KRR model using inverse-distance global descriptors.

    Supports three training modes (inferred automatically from inputs):
      - ``energy_only``      — train on energies, predict E + F via Jacobian kernel
      - ``force_only``       — train on forces,   predict F + E via Hessian kernel
      - ``energy_and_force`` — train on E + F jointly via the full combined kernel

    The representation is the upper-triangle of the inverse interatomic distance
    matrix: ``x[p] = 1/r_{ij}`` for each unique pair (i<j).  Descriptor dimension
    M = N*(N-1)/2 where N is the number of atoms.

    **All molecules in the dataset must have the same atom count** — the descriptor
    dimension M is fixed by N.  Use rMD17-style single-molecule datasets.

    After fitting, ``alpha_desc`` is precomputed for fast inference via the J^T·alpha
    trick (no full kernel matrix needed at predict time for force-trained modes).

    Parameters
    ----------
    sigma : float
        Gaussian kernel length-scale.
    l2 : float
        L2 regularisation added to the kernel diagonal.
    eps : float
        Numerical floor for interatomic distances (default 1e-12).

    Examples
    --------
    >>> model = GlobalKRRModel(sigma=3.0, l2=1e-8)
    >>> model.fit(coords_list, z_list, energies=E, forces=F)
    >>> E_pred, F_pred = model.predict(coords_test, z_test)
    >>> model.save("global_krr.npz")
    >>> model2 = GlobalKRRModel.load("global_krr.npz")
    """

    def __init__(
        self,
        sigma: float = 3.0,
        l2: float = 1e-8,
        eps: float = 1e-12,
    ) -> None:
        self.sigma = sigma
        self.l2 = l2
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
        self._X_tr = X
        self._dX_tr = dX
        self._n_train = len(coords_list)

        if mode == "energy_only":
            if energies is None:
                msg = "energies must be provided for energy_only mode"
                raise ValueError(msg)
            K_rfp = global_kernels.kernel_gaussian_symm_rfp(X, self.sigma)
            self._y_train = energies
            self._alpha = kernelmath.cho_solve_rfp(K_rfp, energies, l2=self.l2)
            self._alpha_desc: NDArray[np.float64] | None = None

        elif mode == "force_only":
            if forces is None:
                msg = "forces must be provided for force_only mode"
                raise ValueError(msg)
            if dX is None:
                msg = "dX is None in force_only mode — internal error"
                raise RuntimeError(msg)
            F_flat = forces.ravel()
            K_rfp = global_kernels.kernel_gaussian_hessian_symm_rfp(X, dX, self.sigma)
            self._y_train = F_flat
            self._alpha = kernelmath.cho_solve_rfp(K_rfp, F_flat, l2=self.l2)
            # Precompute descriptor-space force coefficients for fast inference.
            # alpha has shape (N*D,); reshape to (N, D) for compute_alpha_desc.
            D = dX.shape[1]
            self._alpha_desc = global_kernels.kernel_gaussian_compute_alpha_desc(
                dX, self._alpha.reshape(self._n_train, D)
            )

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
            # Full kernel convention: pass -F as training labels (dE/dR, not force).
            F_neg = -forces
            y_tr = np.concatenate([energies, F_neg.ravel()])
            K_rfp = global_kernels.kernel_gaussian_full_symm_rfp(X, dX, self.sigma)
            self._y_train = y_tr
            self._alpha = kernelmath.cho_solve_rfp(K_rfp, y_tr, l2=self.l2)
            # Precompute alpha_desc from the force part of alpha.
            D = dX.shape[1]
            alpha_F = self._alpha[self._n_train :].reshape(self._n_train, D)
            self._alpha_desc = global_kernels.kernel_gaussian_compute_alpha_desc(dX, alpha_F)

    # ------------------------------------------------------------------
    # Internal predict
    # ------------------------------------------------------------------

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        compute_energy: bool = True,  # E+F computed together; param kept for API compat
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mode = self.training_mode_

        X_te, dX_te = _build_repr(coords_list, self.eps, with_gradients=True)

        X_tr = self._X_tr
        dX_tr = self._dX_tr
        alpha = self._alpha
        n_test = len(coords_list)
        N_tr = self._n_train

        if mode == "energy_only":
            if dX_te is None:
                msg = "dX_te is None in energy_only predict — internal error"
                raise RuntimeError(msg)
            K_e = global_kernels.kernel_gaussian(X_te, X_tr, self.sigma)
            E_pred: NDArray[np.float64] = K_e @ alpha
            # Forces via Jacobian kernel: dE/dR_test
            K_jac = global_kernels.kernel_gaussian_jacobian(X_te, dX_te, X_tr, self.sigma)
            F_pred: NDArray[np.float64] = (K_jac @ alpha).ravel()

        elif mode == "force_only":
            if dX_te is None:
                msg = "dX_te is None in force_only predict — internal error"
                raise RuntimeError(msg)
            if self._alpha_desc is not None:
                # Fast path: J^T alpha trick
                F_pred = global_kernels.kernel_gaussian_hessian_matvec(
                    X_te, dX_te, X_tr, self._alpha_desc, self.sigma
                ).ravel()
                E_pred = -global_kernels.kernel_gaussian_jacobian_t_matvec(
                    X_te, X_tr, self._alpha_desc, self.sigma
                )
            else:
                # Fallback for loaded models without alpha_desc
                if dX_tr is None:
                    msg = "dX_tr is None in force_only fallback — internal error"
                    raise RuntimeError(msg)
                K_hess = global_kernels.kernel_gaussian_hessian(
                    X_te, dX_te, X_tr, dX_tr, self.sigma
                )
                F_pred = (K_hess @ alpha).ravel()
                K_jt = global_kernels.kernel_gaussian_jacobian_t(X_te, X_tr, dX_tr, self.sigma)
                E_pred = -(K_jt @ alpha)

        else:  # energy_and_force
            if dX_te is None:
                msg = "dX_te is None in energy_and_force predict — internal error"
                raise RuntimeError(msg)
            if self._alpha_desc is not None:
                # Fast path: J^T alpha trick
                alpha_E = alpha[:N_tr]
                E_pred, F_2d = global_kernels.kernel_gaussian_full_matvec(
                    X_te, dX_te, X_tr, alpha_E, self._alpha_desc, self.sigma
                )
                F_pred = -F_2d.ravel()  # undo sign convention (labels were -F)
            else:
                # Fallback for loaded models without alpha_desc
                if dX_tr is None:
                    msg = "dX_tr is None in energy_and_force fallback — internal error"
                    raise RuntimeError(msg)
                K_full = global_kernels.kernel_gaussian_full(X_te, dX_te, X_tr, dX_tr, self.sigma)
                y_pred = K_full @ alpha
                E_pred = y_pred[:n_test]
                F_pred = -y_pred[n_test:]

        return E_pred, F_pred

    # ------------------------------------------------------------------
    # Training score
    # ------------------------------------------------------------------

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        y_pred = self._y_train - self.l2 * self._alpha
        mode = self.training_mode_
        n = self._n_train
        if mode == "energy_only":
            return {"energy": (self._y_train, y_pred)}
        elif mode == "force_only":
            return {"force": (self._y_train, y_pred)}
        else:  # energy_and_force — y_train = [E; -F], undo sign on forces
            return {
                "energy": (self._y_train[:n], y_pred[:n]),
                "force": (-self._y_train[n:], -y_pred[n:]),
            }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
        d: dict[str, object] = {
            "model_class": "GlobalKRRModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "eps": self.eps,
            "baseline_elements": self.baseline_elements_,
            "element_energies": self.element_energies_,
            "alpha": self._alpha,
            "y_train": self._y_train,
            "X_tr": self._X_tr,
            "n_train": self._n_train,
        }
        if self._dX_tr is not None:
            d["dX_tr"] = self._dX_tr
        if self._alpha_desc is not None:
            d["alpha_desc"] = self._alpha_desc
        return d

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.eps = float(data["eps"]) if "eps" in data else 1e-12
        self.baseline_elements_ = data["baseline_elements"].astype(np.int32)
        self.element_energies_ = data["element_energies"].astype(np.float64)
        self.training_mode_: TrainingMode = cast(TrainingMode, str(data["training_mode"]))
        self._alpha = data["alpha"]
        self._y_train = data["y_train"] if "y_train" in data else np.array([], dtype=np.float64)
        self._X_tr = data["X_tr"]
        self._dX_tr = data.get("dX_tr", None)
        self._alpha_desc = data.get("alpha_desc", None)
        self._n_train = int(data["n_train"]) if "n_train" in data else len(self._X_tr)
