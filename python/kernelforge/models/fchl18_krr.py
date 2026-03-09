"""FCHL18KRRModel: FCHL18 analytical Kernel Ridge Regression model."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

import kernelforge.fchl18_kernel as fchl18_kernel
import kernelforge.fchl18_repr as fchl18_repr
from kernelforge import kernelmath

from .base import BaseModel, TrainingMode
from .representations import compute_fchl18

# Default FCHL18 kernel hyperparameters matching the examples
_DEFAULT_KERNEL_PARAMS: dict[str, Any] = {
    "two_body_scaling": np.sqrt(8),
    "three_body_scaling": 1.6,
    "two_body_width": 0.2,
    "three_body_width": np.pi,
    "two_body_power": 4.0,
    "three_body_power": 2.0,
    "cut_start": 1.0,
    "cut_distance": 5.0,
    "fourier_order": 1,
    "use_atm": True,
}


class FCHL18KRRModel(BaseModel):
    """Kernel Ridge Regression model using the FCHL18 analytical kernel.

    Supports three training modes (inferred automatically from inputs):
      - ``energy_only``      — train on energies, predict E + F via gradient kernel
      - ``force_only``       — train on forces,   predict F + E via Hessian kernel
      - ``energy_and_force`` — train on E + F jointly via the full combined kernel

    .. note::
        The FCHL18 Hessian kernel (used in ``force_only`` and
        ``energy_and_force`` modes) requires ``use_atm=False`` and
        ``cut_start >= 1.0``. These constraints are enforced automatically
        when those modes are detected.

    Parameters
    ----------
    sigma : float
        Gaussian kernel length-scale.
    l2 : float
        L2 regularisation strength. The Hessian kernel is only numerically
        PSD; values around 1e-4 are typically needed for Cholesky stability
        in force-only / energy+force modes.
    max_size : int
        Padding dimension for FCHL18 representations (must be >= max atoms
        in any molecule).
    kernel_params : dict, optional
        FCHL18 kernel hyperparameters. Overrides the defaults listed above.

    Examples
    --------
    >>> model = FCHL18KRRModel(sigma=2.5, l2=1e-8, max_size=9)
    >>> model.fit(coords_list, z_list, energies=E, forces=F)
    >>> E_pred, F_pred = model.predict(coords_test, z_test)
    >>> model.save("fchl18_model.npz")
    >>> model2 = FCHL18KRRModel.load("fchl18_model.npz")
    """

    def __init__(
        self,
        sigma: float = 2.5,
        l2: float = 1e-8,
        max_size: int = 23,
        kernel_params: dict[str, Any] | None = None,
    ) -> None:
        self.sigma = sigma
        self.l2 = l2
        self.max_size = max_size
        self.kernel_params: dict[str, Any] = (
            kernel_params if kernel_params is not None else dict(_DEFAULT_KERNEL_PARAMS)
        )
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

        # Resolve kernel params — Hessian/full kernels require use_atm=False
        # and cut_start >= 1.0 (ATM Hessian not implemented yet)
        kp = dict(self.kernel_params)
        if mode in ("force_only", "energy_and_force"):
            kp["use_atm"] = False
            kp["cut_start"] = max(kp.get("cut_start", 1.0), 1.0)

        cut_distance = float(kp.get("cut_distance", 5.0))

        x, n_atoms, n_neighbors, coords_f64, z_f64 = compute_fchl18(
            coords_list, z_list, max_size=self.max_size, cut_distance=cut_distance
        )

        # Store training data for prediction
        self._x_tr = x
        self._n_tr = n_atoms
        self._nn_tr = n_neighbors
        self._R_tr = coords_f64  # raw coords needed for gradient/Hessian kernels
        self._Z_tr = z_f64
        self._n_train = len(coords_list)
        self._n_atoms_per_mol = n_atoms  # (n_train,) int32
        self._kp_fit = kp  # save resolved kernel params

        if mode == "energy_only":
            if energies is None:
                msg = "energies must be provided for energy_only mode"
                raise ValueError(msg)
            K_tr = fchl18_kernel.kernel_gaussian_symm(
                x, n_atoms, n_neighbors, sigma=self.sigma, **kp
            )
            # Use kernelmath Cholesky for consistency; fall back to np.linalg.solve on failure
            K_tr[np.diag_indices_from(K_tr)] += self.l2
            self._alpha = np.linalg.solve(K_tr, energies)

        elif mode == "force_only":
            if forces is None:
                msg = "forces must be provided for force_only mode"
                raise ValueError(msg)
            F_flat = forces.ravel()
            K_rfp = fchl18_kernel.kernel_gaussian_hessian_symm_rfp(
                coords_f64, z_f64, sigma=self.sigma, **kp
            )
            self._alpha = kernelmath.cho_solve_rfp(K_rfp, F_flat, l2=self.l2)

        else:  # energy_and_force
            if energies is None:
                msg = "energies must be provided for energy_and_force mode"
                raise ValueError(msg)
            if forces is None:
                msg = "forces must be provided for energy_and_force mode"
                raise ValueError(msg)
            # Forces stored with negated sign: full kernel uses +dK/dR convention
            F_neg = -forces  # (n_train, n_atoms*3)
            y_tr = np.concatenate([energies, F_neg.ravel()])
            K = fchl18_kernel.kernel_gaussian_full_symm(coords_f64, z_f64, sigma=self.sigma, **kp)
            K[np.diag_indices_from(K)] += self.l2
            self._alpha = np.linalg.solve(K, y_tr)

    # ------------------------------------------------------------------
    # Internal predict
    # ------------------------------------------------------------------

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mode = self.training_mode_
        kp = self._kp_fit
        cut_distance = float(kp.get("cut_distance", 5.0))

        n_test = len(coords_list)
        coords_te = [np.asarray(r, dtype=np.float64) for r in coords_list]
        z_te = [np.asarray(z, dtype=np.int32) for z in z_list]

        # Infer per-mol n_atoms for force reshaping (use first test mol's z)
        n_atoms = len(z_te[0])
        naq = n_atoms * 3

        alpha = self._alpha

        if mode == "energy_only":
            # Build test representations for scalar kernel
            x_te, n_te, nn_te = fchl18_repr.generate(
                coords_te, z_te, max_size=self.max_size, cut_distance=cut_distance
            )
            K_e = fchl18_kernel.kernel_gaussian(
                x_te, self._x_tr, n_te, self._n_tr, nn_te, self._nn_tr, sigma=self.sigma, **kp
            )
            E_pred = K_e @ alpha
            # Centre: energy-only model has no absolute offset
            E_pred = E_pred - E_pred.mean()

            # Forces via per-molecule gradient kernel: G[n_atoms, 3, n_train]
            F_pred = np.zeros((n_test, naq), dtype=np.float64)
            for i, (coords_A, z_A) in enumerate(zip(coords_te, z_te, strict=True)):
                G = fchl18_kernel.kernel_gaussian_gradient(
                    coords_A, z_A, self._x_tr, self._n_tr, self._nn_tr, sigma=self.sigma, **kp
                )  # (n_atoms, 3, n_train)
                F_pred[i] = -(G @ alpha).ravel()

        elif mode == "force_only":
            K_hess = fchl18_kernel.kernel_gaussian_hessian(
                coords_te, z_te, self._R_tr, self._Z_tr, sigma=self.sigma, **kp
            )
            F_pred = (K_hess @ alpha).reshape(n_test, naq)

            # Energy via Jacobian-transpose kernel: K_jt (n_test, n_train*naq)
            x_te, n_te, nn_te = fchl18_repr.generate(
                coords_te, z_te, max_size=self.max_size, cut_distance=cut_distance
            )
            K_jt = fchl18_kernel.kernel_gaussian_jacobian_t(
                self._R_tr, self._Z_tr, x_te, n_te, nn_te, sigma=self.sigma, **kp
            )
            E_pred = K_jt @ alpha
            # Centre: force-only has no absolute energy offset
            E_pred = E_pred - E_pred.mean()

        else:  # energy_and_force
            K_full = fchl18_kernel.kernel_gaussian_full(
                coords_te, z_te, self._R_tr, self._Z_tr, sigma=self.sigma, **kp
            )
            y_pred = K_full @ alpha
            E_pred = y_pred[:n_test]
            # Negate back: training used -F as labels
            F_pred = -y_pred[n_test:].reshape(n_test, naq)

        return E_pred, F_pred

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
        # Serialize raw coord/charge lists as object arrays
        R_arr = np.empty(len(self._R_tr), dtype=object)
        Z_arr = np.empty(len(self._Z_tr), dtype=object)
        for i, (r, z) in enumerate(zip(self._R_tr, self._Z_tr, strict=True)):
            R_arr[i] = r
            Z_arr[i] = z

        return {
            "model_class": "FCHL18KRRModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "max_size": self.max_size,
            "kernel_params": json.dumps(self.kernel_params),
            "kp_fit": json.dumps(self._kp_fit),
            "energy_mean": self.energy_mean_,
            "alpha": self._alpha,
            "x_tr": self._x_tr,
            "n_tr": self._n_tr,
            "nn_tr": self._nn_tr,
            "R_tr": R_arr,
            "Z_tr": Z_arr,
        }

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.max_size = int(data["max_size"])
        self.kernel_params = json.loads(str(data["kernel_params"]))
        self._kp_fit = json.loads(str(data["kp_fit"]))
        self.energy_mean_ = float(data["energy_mean"])
        self.training_mode_: TrainingMode = str(data["training_mode"])  # type: ignore[assignment]
        self._alpha = data["alpha"]
        self._x_tr = data["x_tr"]
        self._n_tr = data["n_tr"]
        self._nn_tr = data["nn_tr"]
        R_arr = data["R_tr"]
        Z_arr = data["Z_tr"]
        self._R_tr = [R_arr[i] for i in range(len(R_arr))]
        self._Z_tr = [Z_arr[i] for i in range(len(Z_arr))]
        self._n_train = len(self._R_tr)
