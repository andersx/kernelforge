"""CudaGlobalKRRModel: GPU-accelerated KRR using inverse-distance descriptors.

Architecture
------------
energy_and_force training:
  1. Build invdist X (N,M), dX (N,D,M) in float64.
  2. Call ``cuda_global_kernels.kernel_gaussian_full_symm_rfp`` (GPU, float32)
     to build K_full in RFP packed format (TRANSR=N, UPLO=L).
     BIG = N*(1+D); buffer size = BIG*(BIG+1)/2 floats (50% vs dense).
  3. Cholesky-factorize in-place with ``rfp_potrf`` (GPU, float32, cuSolver).
     Raises RuntimeError immediately if factorisation fails (info != 0).
  4. Triangular solve with ``rfp_potrs``.
  5. Pre-contract force coefficients:
     alpha_desc_F[m, k] = einsum('ndm,nd->nm', dX, alpha_F)
  6. Store X_train, alpha_E, alpha_desc_F as persistent float32 CUDA tensors.

energy_only training:
  1. Build invdist X (N,M) in float64 (no Jacobians needed).
  2. Call ``cuda_global_kernels.kernel_gaussian_symm_rfp`` (GPU, float32)
     to build the N*N kernel in RFP packed format.
  3. Cholesky-factorize in-place with ``cuda_global_kernels.rfp_potrf``
     (GPU, float32, cuSolver); fall back to CPU float64 if factorisation fails.
  4. Triangular solve with ``cuda_global_kernels.rfp_potrs``.
  5. Store X_train (float64) and alpha (float64) for CPU inference.

Inference:
  energy_and_force: ``cuda_global_kernels.kernel_gaussian_full_matvec`` (GPU).
  energy_only: ``global_kernels.kernel_gaussian`` + Jacobian kernel (CPU).
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

    Supports two training modes:

    ``energy_and_force``
        GPU float32 RFP kernel assembly (``kernel_gaussian_full_symm_rfp``)
        with in-place Cholesky via cuSolver (``rfp_potrf`` / ``rfp_potrs``).
        Uses 50% less VRAM than the equivalent dense matrix.  Raises
        ``RuntimeError`` immediately if the float32 factorisation fails.
        Inference runs entirely on GPU using the J^T·alpha contracted matvec.

    ``energy_only``
        GPU float32 RFP kernel assembly and in-place Cholesky solve via
        cuSolver (``rfp_potrf`` / ``rfp_potrs``).  Falls back to CPU float64
        if the float32 factorisation fails.  Inference uses the CPU
        ``global_kernels.kernel_gaussian`` + Jacobian kernel path.

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

    PyTorch interface (energy_and_force only, pre-computed float32 CUDA tensors):
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

        if mode not in ("energy_only", "energy_and_force"):
            msg = (
                f"CudaGlobalKRRModel supports 'energy_only' and 'energy_and_force' training. "
                f"Got '{mode}'. Use GlobalKRRModel for other modes."
            )
            raise NotImplementedError(msg)

        import time as _time

        def _t(label: str, t0: float) -> float:
            import torch as _tsync

            _tsync.cuda.synchronize()
            dt = _time.perf_counter() - t0
            print(f"  [CUDA-global] {label:<45s} {dt * 1000:8.2f} ms")
            return _time.perf_counter()

        t0 = _time.perf_counter()

        # ==================================================================
        # energy_only path
        # ==================================================================
        if mode == "energy_only":
            if energies is None:
                msg = "energies must be provided for energy_only mode"
                raise ValueError(msg)

            # Step 1: build invdist representation (no Jacobians needed)
            X, _ = _build_repr(coords_list, self.eps, with_gradients=False)
            N, M = X.shape
            self._n_train = N
            self._M = M
            t0 = _t("Step 1  build invdist repr (CPU)", t0)

            # Step 2: RFP kernel on GPU (float32)
            X_f32 = _to_cuda(X.astype(np.float32))
            K_rfp = _ext.kernel_gaussian_symm_rfp(  # type: ignore[union-attr]
                X_f32, float(self.sigma)
            )
            del X_f32
            t0 = _t("Step 2  kernel_gaussian_symm_rfp (GPU)", t0)

            # Step 3: in-place Cholesky + triangular solve (GPU float32)
            info = _ext.rfp_potrf(K_rfp, N, float(self.l2))  # type: ignore[union-attr]
            if info != 0:
                # Float32 factorisation failed — fall back to CPU float64
                del K_rfp
                print(
                    f"  [CUDA-global] float32 RFP Cholesky failed (info={info}), "
                    f"falling back to CPU float64"
                )
                from kernelforge import global_kernels, kernelmath

                K_rfp_cpu = global_kernels.kernel_gaussian_symm_rfp(X, float(self.sigma))
                alpha_np: NDArray[np.float64] = kernelmath.cho_solve_rfp(
                    K_rfp_cpu, energies, l2=self.l2
                )
            else:
                rhs_gpu = (
                    torch.from_numpy(energies.astype(np.float32)).cuda().unsqueeze(-1)
                )  # (N, 1)
                _ext.rfp_potrs(K_rfp, rhs_gpu)  # type: ignore[union-attr]
                del K_rfp
                alpha_np = rhs_gpu.squeeze(-1).cpu().numpy().astype(np.float64)
                del rhs_gpu
            t0 = _t("Step 3  rfp_potrf + rfp_potrs (GPU)", t0)

            # Store state for inference (CPU path) and save/load
            self._X_tr: NDArray[np.float64] = X
            self._alpha: NDArray[np.float64] = alpha_np
            self._y_train: NDArray[np.float64] = energies
            return

        # ==================================================================
        # energy_and_force path (unchanged)
        # ==================================================================
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

        t0 = _t("Step 1  build invdist repr (CPU)", t0)

        N, M = X.shape[0], X.shape[1]
        D = dX.shape[1]
        print(f"  [CUDA-global]   N={N}  M={M}  D={D}  BIG={N * (1 + D)}")
        self._n_train = N
        self._M = M
        self._D = D

        # ---- Step 2: build K_full RFP on GPU (float32) ----
        X_f32 = _to_cuda(X.astype(np.float32))
        dX_f32 = _to_cuda(dX.astype(np.float32))

        BIG = N * (1 + D)
        K_rfp = _ext.kernel_gaussian_full_symm_rfp(  # type: ignore[union-attr]
            X_f32, dX_f32, float(self.sigma)
        )
        del X_f32, dX_f32
        t0 = _t("Step 2  kernel_gaussian_full_symm_rfp (GPU)", t0)

        # ---- Step 3: GPU RFP Cholesky solve (float32) ----
        info = _ext.rfp_potrf(K_rfp, BIG, float(self.l2))  # type: ignore[union-attr]
        if info != 0:
            del K_rfp
            raise RuntimeError(
                f"rfp_potrf (energy_and_force): Cholesky factorization failed "
                f"(info={info}, leading minor of order {info} is not positive-definite). "
                f"Try increasing l2."
            )
        t0 = _t("Step 3a rfp_potrf (GPU)", t0)

        rhs_f32 = np.concatenate([energies, -forces.ravel()]).astype(np.float32)
        rhs_gpu = torch.from_numpy(rhs_f32).cuda().unsqueeze(-1)  # (BIG, 1)
        _ext.rfp_potrs(K_rfp, rhs_gpu)  # type: ignore[union-attr]
        del K_rfp
        alpha_gpu = rhs_gpu.squeeze(-1)
        del rhs_gpu
        t0 = _t("Step 3b rfp_potrs (GPU)", t0)

        alpha_np = alpha_gpu.cpu().numpy().astype(np.float64)
        del alpha_gpu

        # ---- Step 4: pre-contract force coefficients ----
        alpha_E_f64 = alpha_np[:N]
        alpha_F_f64 = alpha_np[N:].reshape(N, D)

        alpha_desc_F_f64: NDArray[np.float64] = np.einsum(
            "ndm,nd->nm", dX, alpha_F_f64, optimize=True
        )
        t0 = _t("Step 4  alpha_desc contraction (CPU)", t0)

        # ---- Step 5: store persistent CUDA tensors ----
        self._X_train_np: NDArray[np.float32] = X.astype(np.float32)
        self._alpha_E_np: NDArray[np.float32] = alpha_E_f64.astype(np.float32)
        self._alpha_desc_F_np: NDArray[np.float32] = alpha_desc_F_f64.astype(np.float32)

        self._X_train_cuda: Any = _to_cuda(self._X_train_np)
        self._alpha_E_cuda: Any = _to_cuda(self._alpha_E_np)
        self._alpha_desc_F_cuda: Any = _to_cuda(self._alpha_desc_F_np)

        # For training score and save/load
        self._alpha = alpha_np
        self._y_train = np.concatenate([energies, -forces.ravel()]).astype(np.float64)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.training_mode_ == "energy_only":
            return self._predict_energy_only(coords_list, z_list)
        return self._predict_energy_and_force(coords_list, z_list)

    def _predict_energy_only(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """CPU inference via kernel_gaussian + Jacobian kernel."""
        from kernelforge import global_kernels

        X_te, dX_te = _build_repr(coords_list, self.eps, with_gradients=True)

        K_e = global_kernels.kernel_gaussian(X_te, self._X_tr, float(self.sigma))
        E_pred: NDArray[np.float64] = K_e @ self._alpha

        if dX_te is not None:
            K_jac = global_kernels.kernel_gaussian_jacobian(
                X_te, dX_te, self._X_tr, float(self.sigma)
            )
            F_pred: NDArray[np.float64] = (K_jac @ self._alpha).ravel()
        else:
            F_pred = np.zeros(0, dtype=np.float64)

        return E_pred, F_pred

    def _predict_energy_and_force(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        import time as _time

        import torch

        def _tp(label: str, t0: float) -> float:
            torch.cuda.synchronize()
            dt = _time.perf_counter() - t0
            print(f"  [predict] {label:<45s} {dt * 1000:8.2f} ms")
            return _time.perf_counter()

        t0 = _time.perf_counter()

        X_te, dX_te = _build_repr(coords_list, self.eps, with_gradients=True)
        if dX_te is None:
            msg = "dX_te is None — internal error"
            raise RuntimeError(msg)
        t0 = _tp("build invdist repr (CPU)", t0)

        X_te_cuda = _to_cuda(X_te.astype(np.float32))
        dX_te_cuda = _to_cuda(dX_te.astype(np.float32))
        t0 = _tp("cast f64→f32 + H2D upload", t0)

        E_cuda, F_cuda = _ext.kernel_gaussian_full_matvec(  # type: ignore[union-attr]
            X_te_cuda,
            dX_te_cuda,
            self._X_train_cuda,
            self._alpha_E_cuda,
            self._alpha_desc_F_cuda,
            float(self.sigma),
        )
        t0 = _tp("kernel_gaussian_full_matvec (GPU)", t0)

        E_pred: NDArray[np.float64] = E_cuda.cpu().numpy().astype(np.float64)
        F_pred: NDArray[np.float64] = F_cuda.cpu().numpy().astype(np.float64)

        return E_pred, F_pred

    # ------------------------------------------------------------------
    # predict_torch — direct PyTorch tensor interface (energy_and_force only)
    # ------------------------------------------------------------------

    def predict_torch(
        self,
        X_te: Any,  # noqa: ANN401  # torch.Tensor (N_test, M) float32 CUDA
        dX_te: Any,  # noqa: ANN401  # torch.Tensor (N_test, D, M) float32 CUDA
    ) -> tuple[Any, Any]:
        """Predict E and F from pre-computed float32 CUDA tensors.

        Only available for models trained in ``energy_and_force`` mode.

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
        if self.training_mode_ != "energy_and_force":
            msg = (
                "predict_torch is only available for energy_and_force models. "
                f"This model was trained in '{self.training_mode_}' mode."
            )
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
        if self.training_mode_ == "energy_only":
            return {"energy": (self._y_train, y_pred)}
        # energy_and_force
        return {
            "energy": (self._y_train[:n], y_pred[:n]),
            # y_train[n:] = -F_train; undo sign for user-facing forces
            "force": (-self._y_train[n:], -y_pred[n:]),
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
        base: dict[str, object] = {
            "model_class": "CudaGlobalKRRModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "eps": self.eps,
            "n_train": self._n_train,
            "M": self._M,
            "baseline_elements": self.baseline_elements_,
            "element_energies": self.element_energies_,
            "alpha": self._alpha,
            "y_train": self._y_train,
        }
        if self.training_mode_ == "energy_only":
            base["X_tr"] = self._X_tr  # float64, used for CPU inference
        else:
            # energy_and_force: store GPU inference state (float32 numpy copies)
            base["D"] = self._D
            base["X_train"] = self._X_train_np
            base["alpha_E"] = self._alpha_E_np
            base["alpha_desc_F"] = self._alpha_desc_F_np
        return base

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
        self._alpha = data["alpha"].astype(np.float64)
        self._y_train = (
            data["y_train"].astype(np.float64)
            if "y_train" in data
            else np.array([], dtype=np.float64)
        )

        if self.training_mode_ == "energy_only":
            self._X_tr = data["X_tr"].astype(np.float64)
        else:
            # energy_and_force: reconstruct persistent CUDA tensors
            self._D = int(data["D"])
            self._X_train_np = data["X_train"].astype(np.float32)
            self._alpha_E_np = data["alpha_E"].astype(np.float32)
            self._alpha_desc_F_np = data["alpha_desc_F"].astype(np.float32)

            self._X_train_cuda = _to_cuda(self._X_train_np)
            self._alpha_E_cuda = _to_cuda(self._alpha_E_np)
            self._alpha_desc_F_cuda = _to_cuda(self._alpha_desc_F_np)
