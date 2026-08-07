"""CudaGlobalRFFModel: GPU RFF using inverse-distance descriptors."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .base import BaseModel, TrainingMode
from .global_krr import _build_repr

try:
    import torch as _torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_CUDA_RFF_AVAILABLE = False
try:
    from kernelforge import cuda_rff_features as _rff_ext

    _CUDA_RFF_AVAILABLE = True
except ImportError:
    pass

_CUDA_KERNELS_AVAILABLE = False
try:
    from kernelforge import cuda_global_kernels as _kern_ext

    _CUDA_KERNELS_AVAILABLE = True
except ImportError:
    pass

_CUDA_INVDIST_AVAILABLE = False
try:
    from kernelforge import cuda_invdist_repr as _cuda_invdist

    _CUDA_INVDIST_AVAILABLE = True
except ImportError:
    pass


def _require_cuda() -> None:
    if not _TORCH_AVAILABLE:
        msg = "PyTorch is required for CudaGlobalRFFModel."
        raise ImportError(msg)
    if not (_CUDA_RFF_AVAILABLE and _CUDA_KERNELS_AVAILABLE):
        msg = (
            "cuda_rff_features and cuda_global_kernels are required for "
            "CudaGlobalRFFModel. Re-build kernelforge with CUDA, CUDAToolkit, and PyTorch."
        )
        raise ImportError(msg)


def _to_cuda(arr: NDArray[np.float32]) -> Any:  # noqa: ANN401
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).cuda()


def _compute_invdist_cuda(
    coords_list: list[NDArray[np.float64]],
    eps: float,
    with_gradients: bool = False,
) -> Any:  # noqa: ANN401
    import torch

    n_atoms_list = [len(c) for c in coords_list]
    if len(set(n_atoms_list)) > 1:
        msg = (
            "CudaGlobalRFFModel requires all molecules to have the same atom count. "
            f"Got: {sorted(set(n_atoms_list))}"
        )
        raise ValueError(msg)

    nm = len(coords_list)
    n_atoms = n_atoms_list[0]

    if _CUDA_INVDIST_AVAILABLE:
        coords_np = np.empty((nm, n_atoms, 3), dtype=np.float32)
        for i, c in enumerate(coords_list):
            coords_np[i] = c.astype(np.float32)
        coords_cuda = torch.from_numpy(coords_np).cuda()
        if with_gradients:
            return _cuda_invdist.inverse_distance_upper_and_jacobian(  # type: ignore[union-attr]
                coords_cuda, n_atoms, float(eps)
            )
        return _cuda_invdist.inverse_distance_upper(  # type: ignore[union-attr]
            coords_cuda, n_atoms, float(eps)
        )

    X_np, dX_np = _build_repr(coords_list, eps, with_gradients=with_gradients)
    X_cuda = _to_cuda(X_np.astype(np.float32))
    if with_gradients:
        if dX_np is None:
            msg = "dX is None in CUDA invdist fallback — internal error"
            raise RuntimeError(msg)
        return X_cuda, _to_cuda(dX_np.astype(np.float32))
    return X_cuda


class CudaGlobalRFFModel(BaseModel):
    """GPU Random Fourier Features model for inverse-distance global descriptors.

    Supports ``energy_only`` and ``energy_and_force`` training.  The normal
    equations are accumulated on GPU in RFP packed format and solved in FP32.
    """

    def __init__(
        self,
        sigma: float = 3.0,
        l2: float = 1e-6,
        d_rff: int = 4096,
        seed: int = 42,
        eps: float = 1e-12,
        chunk_size: int = 256,
    ) -> None:
        self.sigma = sigma
        self.l2 = l2
        self.d_rff = d_rff
        self.seed = seed
        self.eps = eps
        self.chunk_size = chunk_size
        self.is_fitted_ = False

    def _fit(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        energies: NDArray[np.float64] | None,
        forces: NDArray[np.float64] | None,
    ) -> None:
        _require_cuda()
        import time as _time

        import torch

        def _t(label: str, t0: float) -> float:
            torch.cuda.synchronize()
            dt = _time.perf_counter() - t0
            print(f"  [CUDA-RFF] {label:<48s} {dt * 1000:8.2f} ms")
            return _time.perf_counter()

        t0 = _time.perf_counter()
        mode = self.training_mode_
        if mode not in ("energy_only", "energy_and_force"):
            msg = "CudaGlobalRFFModel currently supports energy_only and energy_and_force training."
            raise NotImplementedError(msg)
        if energies is None:
            msg = f"energies must be provided for {mode} mode"
            raise ValueError(msg)
        if mode == "energy_and_force" and forces is None:
            msg = "forces must be provided for energy_and_force mode"
            raise ValueError(msg)
        F_flat: NDArray[np.float64] | None = forces.ravel() if forces is not None else None

        if mode == "energy_and_force":
            X_cuda, dX_cuda = _compute_invdist_cuda(coords_list, self.eps, with_gradients=True)
        else:
            X_cuda = _compute_invdist_cuda(coords_list, self.eps)
            dX_cuda = None
        self._n_train = len(coords_list)
        self._M = int(X_cuda.shape[1])
        ncoords = int(dX_cuda.shape[1]) if dX_cuda is not None else 0
        print(
            f"  [CUDA-RFF]   N={self._n_train}  M={self._M}  d_rff={self.d_rff}  "
            f"ncoords={ncoords}  chunk_size={self.chunk_size}"
        )
        step1 = (
            "Step 1  build invdist repr + jacobian (GPU)"
            if mode == "energy_and_force"
            else "Step 1  build invdist repr (GPU)"
        )
        t0 = _t(step1, t0)

        rng = np.random.default_rng(self.seed)
        W_np: NDArray[np.float32] = rng.standard_normal((self._M, self.d_rff)).astype(
            np.float32
        ) / np.float32(self.sigma)
        b_np: NDArray[np.float32] = rng.uniform(0.0, 2.0 * np.pi, self.d_rff).astype(np.float32)

        W_cuda = _to_cuda(W_np)
        b_cuda = _to_cuda(b_np)
        Y_cuda = _to_cuda(energies.astype(np.float32))
        t0 = _t("Step 2  generate/upload RFF params + targets", t0)

        if mode == "energy_and_force":
            if F_flat is None:
                msg = "F_flat is None in energy_and_force fit — internal error"
                raise RuntimeError(msg)
            F_cuda = _to_cuda(F_flat.astype(np.float32))
            ZtZ_rfp, ZtY = _rff_ext.rff_full_gramian_symm_rfp(  # type: ignore[union-attr]
                X_cuda,
                dX_cuda,
                W_cuda,
                b_cuda,
                Y_cuda,
                F_cuda,
                int(self.chunk_size),
                int(self.chunk_size),
            )
        else:
            F_cuda = None
            ZtZ_rfp, ZtY = _rff_ext.rff_gramian_symm_rfp(  # type: ignore[union-attr]
                X_cuda, W_cuda, b_cuda, Y_cuda, int(self.chunk_size)
            )
        gram_label = (
            "Step 3  rff_full_gramian_symm_rfp (GPU, RFP)"
            if mode == "energy_and_force"
            else "Step 3  rff_gramian_symm_rfp (GPU, RFP)"
        )
        t0 = _t(gram_label, t0)
        rhs = ZtY.unsqueeze(-1)
        t0 = _t("Step 4a build RHS view", t0)
        info = _kern_ext.rfp_potrf(ZtZ_rfp, int(self.d_rff), float(self.l2))  # type: ignore[union-attr]
        if info != 0:
            msg = (
                f"rfp_potrf (CudaGlobalRFFModel {mode}): Cholesky factorization failed "
                f"(info={info}). Try increasing l2."
            )
            raise RuntimeError(msg)
        _kern_ext.rfp_potrs(ZtZ_rfp, rhs)  # type: ignore[union-attr]
        t0 = _t("Step 4b rfp_potrf + rfp_potrs (GPU)", t0)

        weights_cuda = rhs.squeeze(-1).contiguous()
        y_pred_cuda = _rff_ext.rff_predict_energy(  # type: ignore[union-attr]
            X_cuda, W_cuda, b_cuda, weights_cuda, int(self.chunk_size)
        )
        t0 = _t("Step 5a train energy prediction (GPU)", t0)
        self._f_train: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._f_pred_train: NDArray[np.float64] = np.array([], dtype=np.float64)
        if mode == "energy_and_force":
            if F_flat is None:
                msg = "F_flat is None in energy_and_force fit — internal error"
                raise RuntimeError(msg)
            f_pred_cuda = _rff_ext.rff_predict_force(  # type: ignore[union-attr]
                X_cuda, dX_cuda, W_cuda, b_cuda, weights_cuda, int(self.chunk_size)
            )
            self._f_train = F_flat
            self._f_pred_train = f_pred_cuda.cpu().numpy().astype(np.float64)
            t0 = _t("Step 5b train force prediction + D2H", t0)

        self._W_np = W_np
        self._b_np = b_np
        self._weights_np = weights_cuda.cpu().numpy().astype(np.float32)
        self._W_cuda = W_cuda
        self._b_cuda = b_cuda
        self._weights_cuda = weights_cuda
        self._y_train = energies
        self._y_pred_train = y_pred_cuda.cpu().numpy().astype(np.float64)
        t0 = _t("Step 6  D2H train outputs + store state", t0)

        # Keep X only until training completes; prediction rebuilds descriptors.
        del X_cuda, dX_cuda, Y_cuda, F_cuda, ZtZ_rfp, ZtY, rhs
        torch.cuda.synchronize()

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        if self.training_mode_ == "energy_and_force":
            return {
                "energy": (self._y_train, self._y_pred_train),
                "force": (self._f_train, self._f_pred_train),
            }
        return {"energy": (self._y_train, self._y_pred_train)}

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        compute_energy: bool = True,  # E+F computed together; param kept for API compat
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        _require_cuda()
        import time as _time

        import torch

        def _tp(label: str, t0: float) -> float:
            torch.cuda.synchronize()
            dt = _time.perf_counter() - t0
            print(f"  [predict] {label:<45s} {dt * 1000:8.2f} ms")
            return _time.perf_counter()

        t0 = _time.perf_counter()
        if self.training_mode_ not in ("energy_only", "energy_and_force"):
            msg = (
                "CudaGlobalRFFModel currently supports energy_only and energy_and_force prediction."
            )
            raise NotImplementedError(msg)

        if self.training_mode_ == "energy_and_force":
            X_cuda, dX_cuda = _compute_invdist_cuda(coords_list, self.eps, with_gradients=True)
            t0 = _tp("build invdist repr + jacobian (GPU)", t0)
        else:
            X_cuda = _compute_invdist_cuda(coords_list, self.eps)
            dX_cuda = None
            t0 = _tp("build invdist repr (GPU)", t0)
        E_cuda = _rff_ext.rff_predict_energy(  # type: ignore[union-attr]
            X_cuda, self._W_cuda, self._b_cuda, self._weights_cuda, int(self.chunk_size)
        )
        t0 = _tp("rff_predict_energy (GPU)", t0)
        E_pred: NDArray[np.float64] = E_cuda.cpu().numpy().astype(np.float64)
        if self.training_mode_ == "energy_and_force":
            F_cuda = _rff_ext.rff_predict_force(  # type: ignore[union-attr]
                X_cuda,
                dX_cuda,
                self._W_cuda,
                self._b_cuda,
                self._weights_cuda,
                int(self.chunk_size),
            )
            t0 = _tp("rff_predict_force (GPU)", t0)
            F_pred = F_cuda.cpu().numpy().astype(np.float64)
        else:
            F_pred = np.zeros(0, dtype=np.float64)
        t0 = _tp("D2H predictions", t0)
        return E_pred, F_pred

    def _arrays_to_save(self) -> dict[str, object]:
        return {
            "model_class": "CudaGlobalRFFModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "d_rff": self.d_rff,
            "seed": self.seed,
            "eps": self.eps,
            "chunk_size": self.chunk_size,
            "baseline_elements": self.baseline_elements_,
            "element_energies": self.element_energies_,
            "weights": self._weights_cuda.cpu().numpy(),
            "W": self._W_cuda.cpu().numpy(),
            "b": self._b_cuda.cpu().numpy(),
            "n_train": self._n_train,
            "M": self._M,
            "y_train": self._y_train,
            "y_pred_train": self._y_pred_train,
            "f_train": getattr(self, "_f_train", np.array([], dtype=np.float64)),
            "f_pred_train": getattr(self, "_f_pred_train", np.array([], dtype=np.float64)),
        }

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        _require_cuda()
        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.d_rff = int(data["d_rff"])
        self.seed = int(data["seed"])
        self.eps = float(data["eps"]) if "eps" in data else 1e-12
        self.chunk_size = int(data["chunk_size"]) if "chunk_size" in data else 256
        self.baseline_elements_ = data["baseline_elements"].astype(np.int32)
        self.element_energies_ = data["element_energies"].astype(np.float64)
        self.training_mode_: TrainingMode = cast(TrainingMode, str(data["training_mode"]))
        self._n_train = int(data["n_train"]) if "n_train" in data else 0
        self._M = int(data["M"]) if "M" in data else int(data["W"].shape[0])
        self._y_train = data["y_train"].astype(np.float64)
        self._y_pred_train = data["y_pred_train"].astype(np.float64)
        self._f_train = (
            data["f_train"].astype(np.float64)
            if "f_train" in data
            else np.array([], dtype=np.float64)
        )
        self._f_pred_train = (
            data["f_pred_train"].astype(np.float64)
            if "f_pred_train" in data
            else np.array([], dtype=np.float64)
        )

        self._W_np = data["W"].astype(np.float32)
        self._b_np = data["b"].astype(np.float32)
        self._weights_np = data["weights"].astype(np.float32)
        self._W_cuda = _to_cuda(self._W_np)
        self._b_cuda = _to_cuda(self._b_np)
        self._weights_cuda = _to_cuda(self._weights_np)
