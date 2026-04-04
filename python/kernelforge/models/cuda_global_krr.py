"""CudaGlobalKRRModel: GPU-accelerated KRR using inverse-distance descriptors.

Training and inference use the hand-written CUDA kernels from ``cuda_krr_ext``
(cuBLAS SGEMM + cuSOLVER Cholesky).  At inference time only a contracted
representation of the training data is kept on GPU — no full kernel matrix is
ever materialised.

Only the ``energy_and_force`` training mode is supported.  For ``energy_only``
or ``force_only`` training use :class:`GlobalKRRModel` instead.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .base import BaseModel, TrainingMode
from .global_krr import _build_repr  # reuse the invdist_repr helper

# --------------------------------------------------------------------------
# Optional PyTorch import (needed only for predict_torch)
# --------------------------------------------------------------------------
try:
    import torch  # type: ignore[ty:unresolved-import]  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# --------------------------------------------------------------------------
# CUDA extension import — _ext is only defined when the build includes CUDA.
# All code paths that access _ext are guarded by _require_cuda_ext() which
# raises ImportError when _CUDA_EXT_AVAILABLE is False.
# --------------------------------------------------------------------------
_CUDA_EXT_AVAILABLE = False
try:
    from kernelforge import cuda_krr_ext as _ext

    _CUDA_EXT_AVAILABLE = True
except ImportError:
    pass


def _require_cuda_ext() -> None:
    if not _CUDA_EXT_AVAILABLE:
        msg = (
            "cuda_krr_ext is not available.  "
            "Re-build kernelforge with a CUDA compiler and CUDAToolkit present:\n"
            "    make install-linux-mkl-ilp64   (or another Makefile target)\n"
            "CUDA must be discoverable by CMake at build time."
        )
        raise ImportError(msg)


class CudaGlobalKRRModel(BaseModel):
    """GPU-accelerated KRR model using inverse-distance global descriptors.

    Uses CUDA kernels (cuBLAS + cuSOLVER) for both training and inference.
    At inference time only the contracted state (W_F_bar, W_combined, ...)
    lives on GPU — training Jacobians are freed after fit().

    **Only** ``energy_and_force`` training mode is supported.  For
    ``energy_only`` or ``force_only`` use :class:`GlobalKRRModel`.

    **All molecules in the dataset must have the same atom count** — the
    descriptor dimension M = N*(N-1)/2 is fixed by the atom count N.

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
    >>> model = CudaGlobalKRRModel(sigma=3.0, l2=1e-8)
    >>> model.fit(coords_list, z_list, energies=E, forces=F)
    >>> E_pred, F_pred = model.predict(coords_test, z_test)
    >>> model.save("cuda_model.npz")
    >>> model2 = CudaGlobalKRRModel.load("cuda_model.npz")

    PyTorch interface (accepts pre-computed float32 tensors):
    >>> E_t, F_t = model.predict_torch(X_te_tensor, dX_te_tensor)
    """

    def __init__(
        self,
        sigma: float = 3.0,
        l2: float = 1e-8,
        eps: float = 1e-12,
    ) -> None:
        _require_cuda_ext()
        self.sigma = sigma
        self.l2 = l2
        self.eps = eps
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # Internal fit — energy_and_force only
    # ------------------------------------------------------------------

    def _fit(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        energies: NDArray[np.float64] | None,
        forces: NDArray[np.float64] | None,
    ) -> None:
        mode = self.training_mode_
        if mode != "energy_and_force":
            msg = (
                f"CudaGlobalKRRModel only supports 'energy_and_force' training. "
                f"Got '{mode}'. For energy-only or force-only training use "
                f"GlobalKRRModel instead."
            )
            raise NotImplementedError(msg)

        if energies is None:
            msg = "energies must be provided for energy_and_force mode"
            raise ValueError(msg)
        if forces is None:
            msg = "forces must be provided for energy_and_force mode"
            raise ValueError(msg)

        # 1. Build representations (float64 numpy)
        X, dX = _build_repr(coords_list, self.eps, with_gradients=True)
        # X : (N, M) float64,  dX : (N, D, M) float64

        if dX is None:
            msg = "dX is None — internal error (with_gradients=True)"
            raise RuntimeError(msg)

        N = len(coords_list)
        M = X.shape[1]
        D = dX.shape[1]

        self._n_train = N
        self._M = M
        self._D = D

        # 2. Convert to float32 (CUDA extension works in float32)
        X_f32: NDArray[np.float32] = np.ascontiguousarray(X, dtype=np.float32)
        # dX (N, D, M) -> dXT (N*D, M)
        dXT_f32: NDArray[np.float32] = np.ascontiguousarray(dX.reshape(N * D, M), dtype=np.float32)
        E_f32: NDArray[np.float32] = np.ascontiguousarray(energies, dtype=np.float32)
        # forces shape: (N, D) from base class — pass physical forces (CUDA negates internally)
        F_f32: NDArray[np.float32] = np.ascontiguousarray(forces.ravel(), dtype=np.float32)

        # 3. Train + initialise GPU inference state
        state = _ext.CudaKRRState()  # type: ignore[union-attr]
        alpha_f32: NDArray[np.float32] = state.train_ef(
            X_f32, dXT_f32, E_f32, F_f32, self.sigma, self.l2
        )

        self._state: _ext.CudaKRRState = state  # type: ignore[name-defined]

        # 4. Store float64 alpha and y_train for training score / save
        self._alpha: NDArray[np.float64] = alpha_f32.astype(np.float64)
        # y_train = [E, -F.ravel()]  (same sign convention as GlobalKRRModel)
        self._y_train: NDArray[np.float64] = np.concatenate([energies, -forces.ravel()])

    # ------------------------------------------------------------------
    # Internal predict
    # ------------------------------------------------------------------

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X_te, dX_te = _build_repr(coords_list, self.eps, with_gradients=True)
        if dX_te is None:
            msg = "dX_te is None — internal error"
            raise RuntimeError(msg)

        N_test = len(coords_list)
        D = self._D

        X_te_f32: NDArray[np.float32] = np.ascontiguousarray(X_te, dtype=np.float32)
        dXT_te_f32: NDArray[np.float32] = np.ascontiguousarray(
            dX_te.reshape(N_test * D, X_te.shape[1]),
            dtype=np.float32,
        )

        E_f32, F_f32 = self._state.predict(X_te_f32, dXT_te_f32)

        E_pred: NDArray[np.float64] = E_f32.astype(np.float64)
        # F_f32 is (N_test*D,) flat; reshape to (N_test, D)
        F_pred: NDArray[np.float64] = F_f32.astype(np.float64).reshape(N_test, D)

        return E_pred, F_pred

    # ------------------------------------------------------------------
    # predict_torch — direct PyTorch tensor interface
    # ------------------------------------------------------------------

    def predict_torch(
        self,
        X_te: Any,  # noqa: ANN401  # torch.Tensor (N_test, M) float32
        dX_te: Any,  # noqa: ANN401  # torch.Tensor (N_test, D, M) float32
    ) -> tuple[Any, Any]:  # (torch.Tensor, torch.Tensor)
        """Predict E and F from pre-computed torch float32 tensors.

        Accepts descriptor tensors directly, bypassing ``invdist_repr`` and
        the numpy conversion in ``predict()``.  Returns ``torch.Tensor`` on
        the same device as the inputs.

        Parameters
        ----------
        X_te : torch.Tensor, shape (N_test, M), float32
            Test descriptors.
        dX_te : torch.Tensor, shape (N_test, D, M), float32
            Test Jacobians ``∂X_te/∂coords``.

        Returns
        -------
        E_pred : torch.Tensor, shape (N_test,), float32
        F_pred : torch.Tensor, shape (N_test, D), float32
            Physical forces F = -dE/dR.
        """
        if not _TORCH_AVAILABLE:
            msg = "PyTorch is not installed. pip install torch to use predict_torch()."
            raise ImportError(msg)

        import torch  # type: ignore[ty:unresolved-import]

        if not self.is_fitted_:
            msg = "Model is not fitted. Call fit() first."
            raise RuntimeError(msg)

        device = X_te.device
        N_test = X_te.shape[0]
        D = self._D

        # Transfer to CPU numpy for the CUDA extension (H2D/D2H inside extension)
        X_np = X_te.cpu().to(torch.float32).numpy()
        dX_np = dX_te.cpu().to(torch.float32).numpy().reshape(N_test * D, -1)

        X_np = np.ascontiguousarray(X_np)
        dX_np = np.ascontiguousarray(dX_np)

        E_f32, F_f32 = self._state.predict(X_np, dX_np)

        E_tensor = torch.from_numpy(E_f32.copy()).to(device)
        F_tensor = torch.from_numpy(F_f32.copy()).reshape(N_test, D).to(device)
        return E_tensor, F_tensor

    # ------------------------------------------------------------------
    # Training score
    # ------------------------------------------------------------------

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        # y_pred = y_train - l2 * alpha  (from (K + l2*I) @ alpha = y_train)
        y_pred = self._y_train - self.l2 * self._alpha
        n = self._n_train
        # y_train = [E, -F.ravel()]; undo sign on forces
        return {
            "energy": (self._y_train[:n], y_pred[:n]),
            "force": (-self._y_train[n:], -y_pred[n:]),
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
        # Extract GPU inference state as numpy float32 arrays
        X_tr, W_F_bar, W_combined, W_F_self, alpha_E, norms_tr = self._state.get_state()
        return {
            "model_class": "CudaGlobalKRRModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "eps": self.eps,
            "n_train": self._n_train,
            "M": self._M,
            "D": self._D,
            "baseline_elements": self.baseline_elements_,
            "element_energies": self.element_energies_,
            "alpha": self._alpha,
            "y_train": self._y_train,
            # GPU inference state — float32
            "X_tr": X_tr,
            "W_F_bar": W_F_bar,
            "W_combined": W_combined,
            "W_F_self": W_F_self,
            "alpha_E_f32": alpha_E,
            "norms_tr": norms_tr,
        }

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        _require_cuda_ext()

        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.eps = float(data["eps"]) if "eps" in data else 1e-12
        self.training_mode_: TrainingMode = cast(TrainingMode, str(data["training_mode"]))
        self.baseline_elements_ = data["baseline_elements"].astype(np.int32)
        self.element_energies_ = data["element_energies"].astype(np.float64)
        self._n_train = int(data["n_train"])
        self._M = int(data["M"])
        self._D = int(data["D"])
        self._alpha = data["alpha"].astype(np.float64)
        self._y_train = (
            data["y_train"].astype(np.float64)
            if "y_train" in data
            else np.array([], dtype=np.float64)
        )

        # Reconstruct GPU inference state from saved float32 arrays
        state = _ext.CudaKRRState()  # type: ignore[union-attr]
        state.load_state(
            np.ascontiguousarray(data["X_tr"], dtype=np.float32),
            np.ascontiguousarray(data["W_F_bar"], dtype=np.float32),
            np.ascontiguousarray(data["W_combined"], dtype=np.float32),
            np.ascontiguousarray(data["W_F_self"], dtype=np.float32),
            np.ascontiguousarray(data["alpha_E_f32"], dtype=np.float32),
            np.ascontiguousarray(data["norms_tr"], dtype=np.float32),
            self.sigma,
            self._n_train,
            self._M,
            self._D,
        )
        self._state = state
