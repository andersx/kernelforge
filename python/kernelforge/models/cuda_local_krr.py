"""CudaLocalKRRModel: GPU-accelerated KRR using FCHL19 local descriptors.

Architecture
------------
Training:
  1. Build FCHL19 X (nm,max_atoms,rep), dX (nm,max_atoms,rep,3*max_atoms),
     Q (nm,max_atoms), N (nm,) in float64 on CPU.
  2. Upload float32 CUDA tensors.
  3. Call ``cuda_local_kernels.kernel_gaussian_full_symm`` (GPU, float32)
     to build K_full (nm+naq)² on GPU.  Symmetrise: K = (K + K^T) / 2.
  4. Solve with ``torch.linalg.cholesky`` + ``torch.cholesky_solve``
     entirely in float32 on GPU.  Use l2 ≥ 1e-4 for reliable conditioning
     (float32 K_FF accumulation introduces ~1e-5 errors for large rep_size).
  5. Call ``cuda_local_kernels.compute_alpha_desc`` (GPU, float32).
  6. Store X_train, dX_train, Q_train, N_train, alpha_E, alpha_desc_F as
     persistent float32 CUDA tensors.

Inference:
  Call ``cuda_local_kernels.kernel_gaussian_full_matvec`` (GPU, float32)
  using the J^T·alpha trick — no K_test_train materialisation.

Only ``energy_and_force`` training mode is supported.
"""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .base import BaseModel, TrainingMode
from .representations import compute_fchl19

# --------------------------------------------------------------------------
# Optional PyTorch import
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
    from kernelforge import cuda_local_kernels as _ext

    _CUDA_EXT_AVAILABLE = True
except ImportError:
    pass


def _require_cuda_ext() -> None:
    if not _CUDA_EXT_AVAILABLE:
        msg = (
            "cuda_local_kernels is not available.  "
            "Re-build kernelforge with a CUDA compiler, CUDAToolkit, and PyTorch present:\n"
            "    make install-linux-mkl-ilp64   (or another Makefile target)\n"
            "Both CUDA and torch must be discoverable by CMake at build time."
        )
        raise ImportError(msg)


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        msg = "PyTorch is required for CudaLocalKRRModel.  pip install torch"
        raise ImportError(msg)


def _to_cuda_f32(arr: NDArray[np.float64] | NDArray[np.float32]) -> Any:  # noqa: ANN401
    """Return a contiguous float32 CUDA torch tensor from a numpy array."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).cuda()


def _to_cuda_i32(arr: NDArray[np.int32]) -> Any:  # noqa: ANN401
    """Return a contiguous int32 CUDA torch tensor from a numpy array."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int32)).cuda()


# Default FCHL19 representation hyperparameters
_DEFAULT_REPR_PARAMS: dict[str, Any] = {}
_DEFAULT_ELEMENTS: list[int] = [1, 6, 7, 8, 16]


class CudaLocalKRRModel(BaseModel):
    """GPU-accelerated KRR model using FCHL19 local descriptors.

    Training uses GPU float32 throughout: kernel assembly, Cholesky solve
    (``torch.linalg.cholesky``), and ``alpha_desc`` precomputation.  Inference
    uses the GPU J^T·alpha contracted matvec.

    **Only** ``energy_and_force`` training mode is supported.

    .. note::
        Float32 K_FF accumulation introduces ~1e-5 rounding errors for FCHL19
        descriptors (rep_size ≈ 300).  Use ``l2 ≥ 1e-4`` to guarantee a
        positive-definite kernel matrix and a successful Cholesky solve.

    Parameters
    ----------
    sigma : float
        Gaussian kernel length-scale.
    l2 : float
        L2 regularisation added to the kernel diagonal.
    elements : list[int], optional
        Sorted list of unique atomic numbers.  Defaults to [1, 6, 7, 8, 16].
    repr_params : dict, optional
        Extra keyword arguments forwarded to ``generate_fchl_acsf_and_gradients``.

    Examples
    --------
    >>> model = CudaLocalKRRModel(sigma=2.0, l2=1e-8, elements=[1, 6, 8])
    >>> model.fit(coords_list, z_list, energies=E, forces=F)
    >>> E_pred, F_pred = model.predict(coords_test, z_test)
    >>> model.save("cuda_local_model.npz")
    >>> model2 = CudaLocalKRRModel.load("cuda_local_model.npz")
    """

    def __init__(
        self,
        sigma: float = 2.0,
        l2: float = 1e-4,
        elements: list[int] | None = None,
        repr_params: dict[str, Any] | None = None,
    ) -> None:
        _require_cuda_ext()
        _require_torch()
        self.sigma = sigma
        self.l2 = l2
        self.elements: list[int] = (
            sorted(elements) if elements is not None else list(_DEFAULT_ELEMENTS)
        )
        self.repr_params: dict[str, Any] = repr_params if repr_params is not None else {}
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
                f"CudaLocalKRRModel only supports 'energy_and_force' training. "
                f"Got '{mode}'. Use LocalKRRModel for other modes."
            )
            raise NotImplementedError(msg)
        if energies is None:
            msg = "energies must be provided for energy_and_force mode"
            raise ValueError(msg)
        if forces is None:
            msg = "forces must be provided for energy_and_force mode"
            raise ValueError(msg)

        # ---- Step 1: build FCHL19 representations (float64, CPU) ----
        X, dX, Q_krr, _Q_rff, N = compute_fchl19(
            coords_list,
            z_list,
            self.elements,
            with_gradients=True,
            repr_params=self.repr_params,
        )
        # X    : (nm, max_atoms, rep)                  float64
        # dX   : (nm, max_atoms, rep, 3*max_atoms)     float64
        # Q_krr: (nm, max_atoms)                       int32
        # N    : (nm,)                                 int32

        if dX is None:
            msg = "dX is None — internal error"
            raise RuntimeError(msg)

        nm = X.shape[0]
        max_atoms = X.shape[1]
        rep_size = X.shape[2]
        naq = int(np.sum(N) * 3)

        self._n_train = nm
        self._max_atoms = max_atoms
        self._rep_size = rep_size
        self._naq_train = naq

        # ---- Step 2: upload float32 CUDA tensors ----
        X_cuda = _to_cuda_f32(X.astype(np.float32))
        dX_cuda = _to_cuda_f32(dX.astype(np.float32))
        Q_cuda = _to_cuda_i32(Q_krr)
        N_cuda = _to_cuda_i32(N)

        # ---- Step 3: build K_full on GPU (float32) ----
        K_full_cuda = _ext.kernel_gaussian_full_symm(  # type: ignore[union-attr]
            X_cuda, dX_cuda, Q_cuda, N_cuda, float(self.sigma)
        )
        # K_full_cuda: (nm+naq, nm+naq) float32 CUDA
        # Symmetrise: (K + K^T) / 2 eliminates atomicAdd-ordering asymmetries.
        K_full_cuda = (K_full_cuda + K_full_cuda.T).mul_(0.5)
        K_full_cuda.diagonal().add_(self.l2)

        # ---- Step 4: float32 Cholesky solve on GPU ----
        # RHS: [E; -F.ravel()] as float32
        F_neg = -forces  # sign convention: training target is -F = dE/dR
        rhs_f32 = np.concatenate([energies, F_neg.ravel()]).astype(np.float32)
        rhs_gpu = torch.from_numpy(rhs_f32).cuda()

        chol_L = torch.linalg.cholesky(K_full_cuda)
        alpha_f32_gpu = torch.cholesky_solve(rhs_gpu.unsqueeze(-1), chol_L).squeeze(-1)
        del K_full_cuda, chol_L, rhs_gpu

        # ---- Step 5: precompute alpha_desc_F on GPU (float32) ----
        alpha_F_cuda = alpha_f32_gpu[nm:]
        alpha_desc_F_cuda: Any = _ext.compute_alpha_desc(  # type: ignore[union-attr]
            dX_cuda, N_cuda, alpha_F_cuda
        )
        # alpha_desc_F_cuda: (nm, max_atoms, rep_size) float32 CUDA

        # ---- Step 6: store persistent CUDA inference state ----
        self._X_train_np: NDArray[np.float32] = X.astype(np.float32)
        self._dX_train_np: NDArray[np.float32] = dX.astype(np.float32)
        self._Q_train_np: NDArray[np.int32] = Q_krr.astype(np.int32)
        self._N_train_np: NDArray[np.int32] = N.astype(np.int32)
        self._alpha_E_np: NDArray[np.float32] = alpha_f32_gpu[:nm].cpu().numpy()
        self._alpha_desc_F_np: NDArray[np.float32] = alpha_desc_F_cuda.cpu().numpy()

        self._X_train_cuda: Any = X_cuda
        self._dX_train_cuda: Any = dX_cuda
        self._Q_train_cuda: Any = Q_cuda
        self._N_train_cuda: Any = N_cuda
        self._alpha_E_cuda: Any = alpha_f32_gpu[:nm]
        self._alpha_desc_F_cuda: Any = alpha_desc_F_cuda

        # For training score (kept in float64 for reporting accuracy)
        alpha_np = alpha_f32_gpu.cpu().numpy().astype(np.float64)
        rhs_f64 = np.concatenate([energies, F_neg.ravel()]).astype(np.float64)
        self._alpha: NDArray[np.float64] = alpha_np
        self._y_train: NDArray[np.float64] = rhs_f64
        del alpha_f32_gpu

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X_te, dX_te, Q_te, _Q_rff_te, N_te = compute_fchl19(
            coords_list,
            z_list,
            self.elements,
            with_gradients=True,
            repr_params=self.repr_params,
        )
        if dX_te is None:
            msg = "dX_te is None — internal error"
            raise RuntimeError(msg)

        X_te_cuda = _to_cuda_f32(X_te.astype(np.float32))
        dX_te_cuda = _to_cuda_f32(dX_te.astype(np.float32))
        Q_te_cuda = _to_cuda_i32(Q_te)
        N_te_cuda = _to_cuda_i32(N_te)

        E_cuda, F_cuda = _ext.kernel_gaussian_full_matvec(  # type: ignore[union-attr]
            X_te_cuda,
            dX_te_cuda,
            Q_te_cuda,
            N_te_cuda,
            self._X_train_cuda,
            self._Q_train_cuda,
            self._N_train_cuda,
            self._alpha_E_cuda,
            self._alpha_desc_F_cuda,
            float(self.sigma),
        )

        E_pred: NDArray[np.float64] = E_cuda.cpu().numpy().astype(np.float64)
        F_flat: NDArray[np.float64] = F_cuda.cpu().numpy().astype(np.float64)

        # Negate: training used -F as target, matvec returns +F_raw
        # (same sign convention as LocalKRRModel: _predict returns -F_raw)
        F_pred = -F_flat

        # Reshape to (n_te, ncoords) to match base class expectation
        # ncoords = naq / n_te when molecules are homogeneous, else variable.
        # BaseModel.predict() calls _coerce_forces on the output, which handles flat arrays.
        return E_pred, F_pred

    # ------------------------------------------------------------------
    # Training score
    # ------------------------------------------------------------------

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        y_pred = self._y_train - self.l2 * self._alpha
        n = self._n_train
        return {
            "energy": (self._y_train[:n], y_pred[:n]),
            "force": (-self._y_train[n:], -y_pred[n:]),
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
        return {
            "model_class": "CudaLocalKRRModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "elements": np.array(self.elements, dtype=np.int32),
            "repr_params": json.dumps(self.repr_params),
            "n_train": self._n_train,
            "max_atoms": self._max_atoms,
            "rep_size": self._rep_size,
            "naq_train": self._naq_train,
            "baseline_elements": self.baseline_elements_,
            "element_energies": self.element_energies_,
            "alpha": self._alpha,
            "y_train": self._y_train,
            # GPU inference state (float32 numpy copies for serialisation)
            "X_train": self._X_train_np,
            "dX_train": self._dX_train_np,
            "Q_train": self._Q_train_np,
            "N_train": self._N_train_np,
            "alpha_E": self._alpha_E_np,
            "alpha_desc_F": self._alpha_desc_F_np,
        }

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        _require_cuda_ext()
        _require_torch()

        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.elements = data["elements"].tolist()
        self.repr_params = json.loads(str(data["repr_params"]))
        self.training_mode_: TrainingMode = cast(TrainingMode, str(data["training_mode"]))
        self.baseline_elements_ = data["baseline_elements"].astype(np.int32)
        self.element_energies_ = data["element_energies"].astype(np.float64)

        self._n_train = int(data["n_train"])
        self._max_atoms = int(data["max_atoms"])
        self._rep_size = int(data["rep_size"])
        self._naq_train = int(data["naq_train"])
        self._alpha = data["alpha"].astype(np.float64)
        self._y_train = (
            data["y_train"].astype(np.float64)
            if "y_train" in data
            else np.array([], dtype=np.float64)
        )

        # Reconstruct persistent CUDA tensors from saved numpy arrays
        self._X_train_np = data["X_train"].astype(np.float32)
        self._dX_train_np = data["dX_train"].astype(np.float32)
        self._Q_train_np = data["Q_train"].astype(np.int32)
        self._N_train_np = data["N_train"].astype(np.int32)
        self._alpha_E_np = data["alpha_E"].astype(np.float32)
        self._alpha_desc_F_np = data["alpha_desc_F"].astype(np.float32)

        self._X_train_cuda = _to_cuda_f32(self._X_train_np)
        self._dX_train_cuda = _to_cuda_f32(self._dX_train_np)
        self._Q_train_cuda = _to_cuda_i32(self._Q_train_np)
        self._N_train_cuda = _to_cuda_i32(self._N_train_np)
        self._alpha_E_cuda = _to_cuda_f32(self._alpha_E_np)
        self._alpha_desc_F_cuda = _to_cuda_f32(self._alpha_desc_F_np)
