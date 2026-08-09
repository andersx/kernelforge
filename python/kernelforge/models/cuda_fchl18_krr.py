"""CudaFCHL18KRRModel: GPU-accelerated FCHL18 analytical KRR."""

from __future__ import annotations

import json
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from kernelforge import kernelmath

from .base import BaseModel, TrainingMode

# --------------------------------------------------------------------------
# Optional imports
# --------------------------------------------------------------------------
try:
    import torch as _torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_CUDA_KERNEL_AVAILABLE = False
try:
    from kernelforge import cuda_fchl18_kernel as _cuda_kernel

    _CUDA_KERNEL_AVAILABLE = True
except ImportError:
    pass

_CUDA_REPR_AVAILABLE = False
try:
    from kernelforge import cuda_fchl18_repr as _cuda_repr

    _CUDA_REPR_AVAILABLE = True
except ImportError:
    pass

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

DtypeName = Literal["float32", "float64"]


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        msg = "PyTorch is required for CudaFCHL18KRRModel. pip install torch"
        raise ImportError(msg)


def _require_cuda_kernel() -> None:
    if not _CUDA_KERNEL_AVAILABLE:
        from kernelforge._cuda_hints import CUDA_EXT_HINT

        msg = f"cuda_fchl18_kernel is not available.\n{CUDA_EXT_HINT}"
        raise ImportError(msg)


def _require_cuda_repr() -> None:
    if not _CUDA_REPR_AVAILABLE:
        from kernelforge._cuda_hints import CUDA_EXT_HINT

        msg = f"cuda_fchl18_repr is not available.\n{CUDA_EXT_HINT}"
        raise ImportError(msg)


def _pad_coords_z(
    coords_list: list[NDArray[np.float64]],
    z_list: list[NDArray[np.int32]],
    max_size: int,
    np_dtype: type[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.int32], NDArray[np.int32]]:
    nm = len(coords_list)
    coords = np.zeros((nm, max_size, 3), dtype=np_dtype)
    z = np.zeros((nm, max_size), dtype=np.int32)
    n = np.zeros(nm, dtype=np.int32)
    for i, (c, zi) in enumerate(zip(coords_list, z_list, strict=True)):
        na = len(zi)
        if na > max_size:
            msg = f"molecule {i} has {na} atoms > max_size={max_size}"
            raise ValueError(msg)
        coords[i, :na, :] = np.asarray(c, dtype=np_dtype)
        z[i, :na] = np.asarray(zi, dtype=np.int32)
        n[i] = na
    return coords, z, n


def _to_cuda(arr: NDArray[np.floating[Any]], dtype: Any) -> Any:  # noqa: ANN401
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=dtype)).cuda()


def _to_cuda_i32(arr: NDArray[np.int32]) -> Any:  # noqa: ANN401
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int32)).cuda()


class CudaFCHL18KRRModel(BaseModel):
    """GPU FCHL18 Kernel Ridge Regression.

    Default ``dtype="float64"`` uses FP64 CUDA kernels + CPU ``cho_solve_rfp``.
    ``dtype="float32"`` keeps the full path on GPU float32 (repr, RFP kernels,
    and ``rfp_potrf`` / ``rfp_potrs`` with FCHL18's UPLO=U packing).

    Mirrors ``FCHL18KRRModel`` training modes:
      - ``energy_only``      — ``kernel_gaussian_symm`` / ``_symm_rfp``
      - ``force_only``       — ``kernel_gaussian_hessian_symm_rfp``
      - ``energy_and_force`` — ``kernel_gaussian_full_symm_rfp``
    """

    def __init__(
        self,
        sigma: float = 2.5,
        l2: float = 1e-8,
        max_size: int = 23,
        kernel_params: dict[str, Any] | None = None,
        use_rfp: bool = True,
        dtype: DtypeName = "float64",
    ) -> None:
        _require_torch()
        _require_cuda_kernel()
        _require_cuda_repr()
        if dtype not in ("float32", "float64"):
            msg = f"dtype must be 'float32' or 'float64', got {dtype!r}"
            raise ValueError(msg)
        self.sigma = sigma
        self.l2 = l2
        self.max_size = max_size
        self.use_rfp = use_rfp
        self.dtype: DtypeName = dtype
        self.kernel_params: dict[str, Any] = (
            kernel_params if kernel_params is not None else dict(_DEFAULT_KERNEL_PARAMS)
        )
        self.is_fitted_ = False

    @property
    def _np_dtype(self) -> type[np.floating[Any]]:
        return np.float32 if self.dtype == "float32" else np.float64

    @property
    def _torch_dtype(self) -> Any:  # noqa: ANN401
        import torch

        return torch.float32 if self.dtype == "float32" else torch.float64

    def _generate_repr(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        cut_distance: float,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Return (x, n, nn, coords_pad, z_pad) as CUDA tensors in ``self.dtype``."""
        coords_np, z_np, n_np = _pad_coords_z(coords_list, z_list, self.max_size, self._np_dtype)
        coords_t = _to_cuda(coords_np, self._np_dtype)
        z_t = _to_cuda_i32(z_np)
        n_t = _to_cuda_i32(n_np)
        x_t, nn_t = _cuda_repr.generate(coords_t, z_t, n_t, cut_distance=cut_distance)
        if x_t.dtype != self._torch_dtype:
            x_t = x_t.to(dtype=self._torch_dtype)
            coords_t = coords_t.to(dtype=self._torch_dtype)
        return x_t, n_t, nn_t, coords_t, z_t

    def _solve_rfp_fp32(self, K_rfp: Any, y: NDArray[np.float64], n: int) -> NDArray[np.float64]:  # noqa: ANN401
        """GPU float32 RFP Cholesky solve (FCHL18 packing: TRANSR=N, UPLO=U)."""
        import torch

        from kernelforge import cuda_global_kernels as _gext

        if K_rfp.dtype != torch.float32:
            K_rfp = K_rfp.float()
        rhs = (
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            .cuda()
            .unsqueeze(-1)
            .contiguous()
        )
        info = _gext.rfp_potrf(K_rfp, n, float(self.l2), uplo="U")
        if info != 0:
            msg = (
                f"rfp_potrf (CudaFCHL18KRRModel, dtype=float32): "
                f"Cholesky factorization failed (info={info})"
            )
            raise RuntimeError(msg)
        _gext.rfp_potrs(K_rfp, rhs, uplo="U")
        return rhs.squeeze(-1).detach().cpu().numpy().astype(np.float64)

    def _fit(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        energies: NDArray[np.float64] | None,
        forces: NDArray[np.float64] | None,
    ) -> None:
        mode = self.training_mode_

        if self.dtype == "float32" and not self.use_rfp:
            msg = "dtype='float32' requires use_rfp=True (GPU rfp_potrf path)"
            raise ValueError(msg)

        kp = dict(self.kernel_params)
        if mode in ("force_only", "energy_and_force"):
            kp["use_atm"] = False
            kp["cut_start"] = max(float(kp.get("cut_start", 1.0)), 1.0)

        cut_distance = float(kp.get("cut_distance", 5.0))
        x, n, nn, coords, z = self._generate_repr(coords_list, z_list, cut_distance)

        self._x_tr = x
        self._n_tr = n
        self._nn_tr = nn
        self._coords_tr = coords
        self._z_tr = z
        self._R_tr = [np.asarray(r, dtype=np.float64) for r in coords_list]
        self._Z_tr = [np.asarray(zi, dtype=np.int32) for zi in z_list]
        self._n_train = len(coords_list)
        self._kp_fit = kp

        use_fp32_solve = self.dtype == "float32"

        if mode == "energy_only":
            if energies is None:
                raise ValueError("energies must be provided for energy_only mode")
            self._y_train = np.asarray(energies, dtype=np.float64)
            if self.use_rfp:
                K_rfp = _cuda_kernel.kernel_gaussian_symm_rfp(x, n, nn, sigma=self.sigma, **kp)
                if use_fp32_solve:
                    self._alpha = self._solve_rfp_fp32(K_rfp, self._y_train, self._n_train)
                else:
                    self._alpha = kernelmath.cho_solve_rfp(
                        K_rfp.detach().cpu().numpy(), self._y_train, l2=self.l2
                    )
            else:
                K = _cuda_kernel.kernel_gaussian_symm(x, n, nn, sigma=self.sigma, **kp)
                K_np = K.detach().cpu().numpy().astype(np.float64)
                K_np[np.diag_indices_from(K_np)] += self.l2
                self._alpha = np.linalg.solve(K_np, self._y_train)

        elif mode == "force_only":
            if forces is None:
                raise ValueError("forces must be provided for force_only mode")
            F_flat = np.asarray(forces, dtype=np.float64).ravel()
            self._y_train = F_flat
            K_rfp = _cuda_kernel.kernel_gaussian_hessian_symm_rfp(
                x, n, nn, coords, z, sigma=self.sigma, **kp
            )
            n_force = int(F_flat.shape[0])
            if use_fp32_solve:
                self._alpha = self._solve_rfp_fp32(K_rfp, F_flat, n_force)
            else:
                self._alpha = kernelmath.cho_solve_rfp(
                    K_rfp.detach().cpu().numpy(), F_flat, l2=self.l2
                )

        else:  # energy_and_force
            if energies is None:
                raise ValueError("energies must be provided for energy_and_force mode")
            if forces is None:
                raise ValueError("forces must be provided for energy_and_force mode")
            F_neg = -np.asarray(forces, dtype=np.float64)
            y_tr = np.concatenate([np.asarray(energies, dtype=np.float64), F_neg.ravel()])
            self._y_train = y_tr
            if self.use_rfp:
                K_rfp = _cuda_kernel.kernel_gaussian_full_symm_rfp(
                    x, n, nn, coords, z, sigma=self.sigma, **kp
                )
                if use_fp32_solve:
                    self._alpha = self._solve_rfp_fp32(K_rfp, y_tr, int(y_tr.shape[0]))
                else:
                    self._alpha = kernelmath.cho_solve_rfp(
                        K_rfp.detach().cpu().numpy(), y_tr, l2=self.l2
                    )
            else:
                K = _cuda_kernel.kernel_gaussian_full_symm(
                    x, n, nn, coords, z, sigma=self.sigma, **kp
                )
                K_np = K.detach().cpu().numpy().astype(np.float64)
                K_np[np.diag_indices_from(K_np)] += self.l2
                self._alpha = np.linalg.solve(K_np, y_tr)

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        y_pred = self._y_train - self.l2 * self._alpha
        mode = self.training_mode_
        n = self._n_train
        if mode == "energy_only":
            return {"energy": (self._y_train, y_pred)}
        if mode == "force_only":
            return {"force": (self._y_train, y_pred)}
        return {
            "energy": (self._y_train[:n], y_pred[:n]),
            "force": (-self._y_train[n:], -y_pred[n:]),
        }

    def _predict(
        self,
        coords_list: list[NDArray[np.float64]],
        z_list: list[NDArray[np.int32]],
        compute_energy: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        import torch

        mode = self.training_mode_
        kp = self._kp_fit
        cut_distance = float(kp.get("cut_distance", 5.0))
        n_test = len(coords_list)

        x_te, n_te, nn_te, coords_te, z_te = self._generate_repr(coords_list, z_list, cut_distance)
        alpha_np = np.ascontiguousarray(self._alpha, dtype=self._np_dtype)
        alpha_t = torch.from_numpy(alpha_np).cuda()

        if mode == "energy_only":
            K_e = _cuda_kernel.kernel_gaussian(
                x_te,
                self._x_tr,
                n_te,
                self._n_tr,
                nn_te,
                self._nn_tr,
                sigma=self.sigma,
                **kp,
            )
            E_pred = (K_e @ alpha_t).detach().cpu().numpy().astype(np.float64)

            F_parts: list[NDArray[np.float64]] = []
            for i in range(n_test):
                G = _cuda_kernel.kernel_gaussian_gradient(
                    x_te[i : i + 1],
                    self._x_tr,
                    n_te[i : i + 1],
                    self._n_tr,
                    nn_te[i : i + 1],
                    self._nn_tr,
                    coords_te[i : i + 1],
                    z_te[i : i + 1],
                    sigma=self.sigma,
                    **kp,
                )
                F_parts.append(-(G @ alpha_t).reshape(-1).detach().cpu().numpy().astype(np.float64))
            F_pred = np.concatenate(F_parts)

        elif mode == "force_only":
            K_hess = _cuda_kernel.kernel_gaussian_hessian(
                x_te,
                self._x_tr,
                n_te,
                self._n_tr,
                nn_te,
                self._nn_tr,
                coords_te,
                z_te,
                self._coords_tr,
                self._z_tr,
                sigma=self.sigma,
                **kp,
            )
            F_pred = (K_hess @ alpha_t).detach().cpu().numpy().astype(np.float64)
            K_jt = _cuda_kernel.kernel_gaussian_jacobian_t(
                self._x_tr,
                x_te,
                self._n_tr,
                n_te,
                self._nn_tr,
                nn_te,
                self._coords_tr,
                self._z_tr,
                sigma=self.sigma,
                **kp,
            )
            # jac_t query=train, fixed=test → shape (n_test, D_train); want E from force alpha.
            # CPU force_only uses jacobian_t(train_coords, test_repr) → (n_test, D_train).
            # Our CUDA jac_t differentiates x1/coords1; so query=train, fixed=test repr:
            E_pred = (K_jt @ alpha_t).detach().cpu().numpy().astype(np.float64)

        else:  # energy_and_force
            K_full = _cuda_kernel.kernel_gaussian_full(
                x_te,
                self._x_tr,
                n_te,
                self._n_tr,
                nn_te,
                self._nn_tr,
                coords_te,
                z_te,
                self._coords_tr,
                self._z_tr,
                sigma=self.sigma,
                **kp,
            )
            y_pred = (K_full @ alpha_t).detach().cpu().numpy().astype(np.float64)
            E_pred = y_pred[:n_test]
            F_pred = -y_pred[n_test:]

        return E_pred, F_pred

    def _arrays_to_save(self) -> dict[str, object]:
        R_arr = np.empty(len(self._R_tr), dtype=object)
        Z_arr = np.empty(len(self._Z_tr), dtype=object)
        for i, (r, z) in enumerate(zip(self._R_tr, self._Z_tr, strict=True)):
            R_arr[i] = r
            Z_arr[i] = z

        return {
            "model_class": "CudaFCHL18KRRModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "max_size": self.max_size,
            "use_rfp": self.use_rfp,
            "dtype": self.dtype,
            "kernel_params": json.dumps(self.kernel_params),
            "kp_fit": json.dumps(self._kp_fit),
            "baseline_elements": self.baseline_elements_,
            "element_energies": self.element_energies_,
            "alpha": self._alpha,
            "y_train": self._y_train,
            "x_tr": self._x_tr.detach().cpu().numpy(),
            "n_tr": self._n_tr.detach().cpu().numpy(),
            "nn_tr": self._nn_tr.detach().cpu().numpy(),
            "coords_tr": self._coords_tr.detach().cpu().numpy(),
            "z_tr": self._z_tr.detach().cpu().numpy(),
            "R_tr": R_arr,
            "Z_tr": Z_arr,
        }

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        import torch

        _require_torch()
        _require_cuda_kernel()
        _require_cuda_repr()
        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.max_size = int(data["max_size"])
        self.use_rfp = bool(data["use_rfp"]) if "use_rfp" in data else True
        raw_dtype = str(data["dtype"]) if "dtype" in data else "float64"
        if raw_dtype == "float32":
            self.dtype = "float32"
        elif raw_dtype == "float64":
            self.dtype = "float64"
        else:
            msg = f"unsupported saved dtype {raw_dtype!r}"
            raise ValueError(msg)
        self.kernel_params = json.loads(str(data["kernel_params"]))
        self._kp_fit = json.loads(str(data["kp_fit"]))
        self.baseline_elements_ = data["baseline_elements"].astype(np.int32)
        self.element_energies_ = data["element_energies"].astype(np.float64)
        self.training_mode_: TrainingMode = cast(TrainingMode, str(data["training_mode"]))
        self._alpha = data["alpha"].astype(np.float64)
        self._y_train = (
            data["y_train"].astype(np.float64)
            if "y_train" in data
            else np.array([], dtype=np.float64)
        )
        self._x_tr = torch.from_numpy(np.ascontiguousarray(data["x_tr"])).cuda()
        self._n_tr = torch.from_numpy(np.ascontiguousarray(data["n_tr"].astype(np.int32))).cuda()
        self._nn_tr = torch.from_numpy(np.ascontiguousarray(data["nn_tr"].astype(np.int32))).cuda()
        self._coords_tr = torch.from_numpy(np.ascontiguousarray(data["coords_tr"])).cuda()
        self._z_tr = torch.from_numpy(np.ascontiguousarray(data["z_tr"].astype(np.int32))).cuda()
        R_arr = data["R_tr"]
        Z_arr = data["Z_tr"]
        self._R_tr = [R_arr[i] for i in range(len(R_arr))]
        self._Z_tr = [Z_arr[i] for i in range(len(Z_arr))]
        self._n_train = len(self._R_tr)
