"""CudaGlobalKRRModel: GPU-accelerated KRR using inverse-distance descriptors.

Architecture
------------
Training:
  1. Build invdist X (N,M), dX (N,D,M) in float64.
  2. Call ``cuda_global_kernels.kernel_gaussian_full_symm`` (GPU, float32)
     to build K_full (N*(1+D))^2 on GPU.
  3. Download K_full, cast to float64, solve with ``kernelmath.solve_cholesky``
     (CPU, float64) — avoids float32 Cholesky conditioning issues.
  4. Pre-contract force coefficients:
     alpha_desc_F[m, k] = einsum('ndm,nd->nm', dX, alpha_F)
  5. Store X_train, alpha_E, alpha_desc_F as persistent float32 CUDA tensors.

Inference:
  Call ``cuda_global_kernels.kernel_gaussian_full_matvec`` (GPU, float32)
  using the J^T·alpha trick — no K_test_train materialisation.

Only ``energy_and_force`` training mode is supported.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .base import BaseModel, TrainingMode
from .global_krr import _build_repr  # reuse invdist_repr helper

# --------------------------------------------------------------------------
# Optional PyTorch import (needed for CUDA tensor operations)
# --------------------------------------------------------------------------
try:
    import torch as _torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# --------------------------------------------------------------------------
# CUDA extension import
# --------------------------------------------------------------------------
_CUDA_EXT_AVAILABLE = False
try:
    from kernelforge import cuda_global_kernels as _ext

    _CUDA_EXT_AVAILABLE = True
except ImportError:
    pass


def _require_cuda_ext() -> None:
    if not _CUDA_EXT_AVAILABLE:
        msg = (
            "cuda_global_kernels is not available.  "
            "Re-build kernelforge with a CUDA compiler, CUDAToolkit, and PyTorch present:\n"
            "    make install-linux-mkl-ilp64   (or another Makefile target)\n"
            "Both CUDA and torch must be discoverable by CMake at build time."
        )
        raise ImportError(msg)


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        msg = "PyTorch is required for CudaGlobalKRRModel.  pip install torch"
        raise ImportError(msg)


def _to_cuda(arr: NDArray[np.float32]) -> Any:  # noqa: ANN401
    """Return a contiguous float32 CUDA torch tensor from a numpy array."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).cuda()


class CudaGlobalKRRModel(BaseModel):
    """GPU-accelerated KRR model using inverse-distance global descriptors.

    Training uses a GPU float32 kernel assembly followed by a CPU float64
    Cholesky solve (via ``kernelmath.solve_cholesky``), which avoids float32
    conditioning failures for large training sets with tight sigma.  Inference
    runs entirely on GPU using the J^T·alpha contracted matvec.

    **Only** ``energy_and_force`` training mode is supported.

    **All molecules must have the same atom count.**

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

    PyTorch interface (pre-computed float32 CUDA tensors, no baseline applied):
    >>> E_t, F_t = model.predict_torch(X_te_cuda, dX_te_cuda)
    """

    def __init__(
        self,
        sigma: float = 3.0,
        l2: float = 1e-8,
        eps: float = 1e-12,
    ) -> None:
        _require_cuda_ext()
        _require_torch()
        self.sigma = sigma
        self.l2 = l2
        self.eps = eps
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _fit(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        energies: NDArray[np.float64] | None,
        forces: NDArray[np.float64] | None,
    ) -> None:
        import torch

        mode = self.training_mode_
        if mode != "energy_and_force":
            msg = (
                f"CudaGlobalKRRModel only supports 'energy_and_force' training. "
                f"Got '{mode}'. Use GlobalKRRModel for other modes."
            )
            raise NotImplementedError(msg)
        if energies is None:
            msg = "energies must be provided for energy_and_force mode"
            raise ValueError(msg)
        if forces is None:
            msg = "forces must be provided for energy_and_force mode"
            raise ValueError(msg)

        # ---- Step 1: build invdist representations (float64) ----
        X, dX = _build_repr(coords_list, self.eps, with_gradients=True)
        # X  : (N, M) float64
        # dX : (N, D, M) float64
        if dX is None:
            msg = "dX is None — internal error (with_gradients=True)"
            raise RuntimeError(msg)

        N, M = X.shape[0], X.shape[1]
        D = dX.shape[1]
        self._n_train = N
        self._M = M
        self._D = D

        # ---- Step 2: build K_full on GPU (float32) ----
        X_f32 = _to_cuda(X.astype(np.float32))
        dX_f32 = _to_cuda(dX.astype(np.float32))

        K_full_cuda = _ext.kernel_gaussian_full_symm(  # type: ignore[union-attr]
            X_f32, dX_f32, float(self.sigma)
        )
        # K_full_cuda: (N*(1+D), N*(1+D)) float32 CUDA

        # ---- Step 3: GPU float64 Cholesky solve (cuSOLVER via torch.linalg) ----
        # Cast K_full to float64 on GPU — stays on device, no H2D round-trip.
        K_f64_cuda = K_full_cuda.to(torch.float64)
        del K_full_cuda  # free float32 copy

        # Add L2 regularisation to diagonal in-place.
        K_f64_cuda.diagonal().add_(self.l2)

        # Build RHS [E, -F] as float64 CUDA tensor.
        rhs_gpu = torch.from_numpy(
            np.concatenate([energies, -forces.ravel()]).astype(np.float64)
        ).cuda()

        # Cholesky factorisation + triangular solve, entirely on GPU (float64).
        # torch.linalg.cholesky wraps cuSOLVER dpotrf; torch.cholesky_solve
        # wraps dpotrs.  Both operate in float64, avoiding the float32
        # conditioning failures seen with cusolverDnSpotrf on MD data.
        chol_L = torch.linalg.cholesky(K_f64_cuda)
        alpha_gpu = torch.cholesky_solve(rhs_gpu.unsqueeze(-1), chol_L).squeeze(-1)
        del K_f64_cuda, chol_L, rhs_gpu

        alpha_f64: NDArray[np.float64] = alpha_gpu.cpu().numpy()
        del alpha_gpu

        # ---- Step 4: pre-contract force coefficients ----
        alpha_E_f64 = alpha_f64[:N]
        alpha_F_f64 = alpha_f64[N:].reshape(N, D)

        # alpha_desc_F[m, k] = sum_d dX[m,d,k] * alpha_F[m,d]
        alpha_desc_F_f64: NDArray[np.float64] = np.einsum(
            "ndm,nd->nm", dX, alpha_F_f64, optimize=True
        )  # (N, M)

        # ---- Step 5: store persistent CUDA tensors ----
        self._X_train_np: NDArray[np.float32] = X.astype(np.float32)
        self._alpha_E_np: NDArray[np.float32] = alpha_E_f64.astype(np.float32)
        self._alpha_desc_F_np: NDArray[np.float32] = alpha_desc_F_f64.astype(np.float32)

        self._X_train_cuda: Any = _to_cuda(self._X_train_np)
        self._alpha_E_cuda: Any = _to_cuda(self._alpha_E_np)
        self._alpha_desc_F_cuda: Any = _to_cuda(self._alpha_desc_F_np)

        # For training score and save/load
        self._alpha: NDArray[np.float64] = alpha_f64
        self._y_train: NDArray[np.float64] = np.concatenate([energies, -forces.ravel()]).astype(
            np.float64
        )

    # ------------------------------------------------------------------
    # predict
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

        X_te_cuda = _to_cuda(X_te.astype(np.float32))
        dX_te_cuda = _to_cuda(dX_te.astype(np.float32))

        E_cuda, F_cuda = _ext.kernel_gaussian_full_matvec(  # type: ignore[union-attr]
            X_te_cuda,
            dX_te_cuda,
            self._X_train_cuda,
            self._alpha_E_cuda,
            self._alpha_desc_F_cuda,
            float(self.sigma),
        )

        E_pred: NDArray[np.float64] = E_cuda.cpu().numpy().astype(np.float64)
        F_pred: NDArray[np.float64] = F_cuda.cpu().numpy().astype(np.float64)

        return E_pred, F_pred

    # ------------------------------------------------------------------
    # predict_torch — direct PyTorch tensor interface
    # ------------------------------------------------------------------

    def predict_torch(
        self,
        X_te: Any,  # noqa: ANN401  # torch.Tensor (N_test, M) float32 CUDA
        dX_te: Any,  # noqa: ANN401  # torch.Tensor (N_test, D, M) float32 CUDA
    ) -> tuple[Any, Any]:
        """Predict E and F from pre-computed float32 CUDA tensors.

        Accepts descriptor tensors already on GPU, bypassing invdist_repr and
        numpy conversions.  Returns float32 CUDA tensors.

        Note: returned energies are **baseline-subtracted** (same as ``_predict``).
        Forces are unaffected by the constant per-element baseline.

        Parameters
        ----------
        X_te : torch.Tensor, shape (N_test, M), float32 CUDA
        dX_te : torch.Tensor, shape (N_test, D, M), float32 CUDA

        Returns
        -------
        E_pred : torch.Tensor, shape (N_test,), float32 CUDA  (baseline-subtracted)
        F_pred : torch.Tensor, shape (N_test, D), float32 CUDA
        """
        if not self.is_fitted_:
            msg = "Model is not fitted. Call fit() first."
            raise RuntimeError(msg)

        return _ext.kernel_gaussian_full_matvec(  # type: ignore[union-attr]
            X_te,
            dX_te,
            self._X_train_cuda,
            self._alpha_E_cuda,
            self._alpha_desc_F_cuda,
            float(self.sigma),
        )

    # ------------------------------------------------------------------
    # Training score
    # ------------------------------------------------------------------

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        # From (K + l2*I) @ alpha = y_train  ->  y_pred = y_train - l2 * alpha
        y_pred = self._y_train - self.l2 * self._alpha
        n = self._n_train
        return {
            "energy": (self._y_train[:n], y_pred[:n]),
            # y_train[n:] = -F_train; undo sign for user-facing forces
            "force": (-self._y_train[n:], -y_pred[n:]),
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
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
            # GPU inference state (float32 numpy copies for serialisation)
            "X_train": self._X_train_np,
            "alpha_E": self._alpha_E_np,
            "alpha_desc_F": self._alpha_desc_F_np,
        }

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        _require_cuda_ext()
        _require_torch()

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

        # Reconstruct persistent CUDA tensors from saved numpy arrays
        self._X_train_np = data["X_train"].astype(np.float32)
        self._alpha_E_np = data["alpha_E"].astype(np.float32)
        self._alpha_desc_F_np = data["alpha_desc_F"].astype(np.float32)

        self._X_train_cuda = _to_cuda(self._X_train_np)
        self._alpha_E_cuda = _to_cuda(self._alpha_E_np)
        self._alpha_desc_F_cuda = _to_cuda(self._alpha_desc_F_np)
