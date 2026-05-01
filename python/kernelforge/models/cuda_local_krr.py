"""CudaLocalKRRModel: GPU-accelerated KRR using FCHL19 local descriptors.

Architecture
------------

energy_only training:
  1. Build FCHL19 X (nm,max_atoms,rep), Q (nm,max_atoms), N (nm,) on GPU.
  2. Upload float32 CUDA tensors.
  3. Call ``cuda_local_kernels.kernel_gaussian_symm_rfp`` (GPU, float32) to
      build K_EE directly in RFP packed format - no nm*nm intermediate.
  4. Solve with GPU RFP Cholesky (``cuda_global_kernels.rfp_potrf`` +
     ``rfp_potrs``).  On failure fall back to CPU float64 Cholesky.
  5. Store X_train, Q_train, N_train, alpha_E as persistent float32 CUDA tensors.

energy_and_force training:
  1. Build FCHL19 X, dX (nm,max_atoms,rep,3*max_atoms), Q, N on GPU.
  2. Upload float32 CUDA tensors.
  3. Call ``cuda_local_kernels.kernel_gaussian_full_symm`` (GPU, float32)
     to build K_full (nm+naq)² on GPU.
  4. Optionally diagonally scale the system, then solve entirely on GPU using
     either truncated ``torch.linalg.eigh`` (default), preconditioned conjugate
     gradients, or ``torch.linalg.cholesky``.
  5. Call ``cuda_local_kernels.compute_alpha_desc`` (GPU, float32).
  6. Store X_train, dX_train, Q_train, N_train, alpha_E, alpha_desc_F as
     persistent float32 CUDA tensors.

Inference:
  energy_only:  ``cuda_local_kernels.kernel_gaussian_rect`` + K@alpha_E.
  energy_and_force: ``cuda_local_kernels.kernel_gaussian_full_matvec`` using
    the J^T·alpha trick — no K_test_train materialisation.
"""

from __future__ import annotations

import json
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .base import BaseModel, TrainingMode

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

_CUDA_FCHL19_AVAILABLE = False
try:
    from kernelforge import cuda_fchl19_repr as _cuda_fchl19

    _CUDA_FCHL19_AVAILABLE = True
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


def _require_cuda_fchl19() -> None:
    if not _CUDA_FCHL19_AVAILABLE:
        msg = (
            "cuda_fchl19_repr is not available. Re-build kernelforge with CUDA and PyTorch "
            "present:\n    make install-linux-mkl-ilp64   (or another Makefile target)"
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


def _compute_fchl19_cuda(
    coords_list: list[NDArray[np.float64]],
    z_list: list[NDArray[np.int32]],
    elements: list[int],
    with_gradients: bool,
    repr_params: dict[str, Any],
    deterministic: bool = False,
) -> tuple[Any, Any, Any, NDArray[np.int32], NDArray[np.int32]]:
    """Compute FCHL19 representations on GPU for CudaLocalKRRModel.

    The representation kernel expects Q as 0-based element indices, while the
    local Gaussian kernels use nuclear charges for species matching. This helper
    builds both tensors and returns the nuclear-charge Q tensor for KRR kernels.
    """
    _require_cuda_fchl19()
    import torch

    elem_to_idx = {e: i for i, e in enumerate(elements)}
    nm = len(coords_list)
    N_np = np.array([len(z) for z in z_list], dtype=np.int32)
    max_atoms = int(N_np.max())

    coords_np = np.zeros((nm, max_atoms, 3), dtype=np.float32)
    Q_idx_np = np.zeros((nm, max_atoms), dtype=np.int32)
    Q_z_np = np.zeros((nm, max_atoms), dtype=np.int32)

    for m, (coords, z) in enumerate(zip(coords_list, z_list, strict=True)):
        z_i32 = np.asarray(z, dtype=np.int32)
        na = len(z_i32)
        coords_np[m, :na, :] = np.asarray(coords, dtype=np.float32)
        Q_z_np[m, :na] = z_i32
        Q_idx_np[m, :na] = [elem_to_idx[int(zi)] for zi in z_i32]

    coords_cuda = torch.from_numpy(np.ascontiguousarray(coords_np)).cuda()
    Q_idx_cuda = torch.from_numpy(np.ascontiguousarray(Q_idx_np)).cuda()
    N_cuda = torch.from_numpy(np.ascontiguousarray(N_np)).cuda()

    if with_gradients:
        X_cuda, dX5_cuda = _cuda_fchl19.generate_fchl_acsf_and_gradients(
            coords_cuda,
            Q_idx_cuda,
            N_cuda,
            nelements=len(elements),
            deterministic=deterministic,
            **repr_params,
        )
        dX_cuda = dX5_cuda.reshape(nm, max_atoms, X_cuda.shape[2], max_atoms * 3).contiguous()
    else:
        X_cuda = _cuda_fchl19.generate_fchl_acsf(
            coords_cuda,
            Q_idx_cuda,
            N_cuda,
            nelements=len(elements),
            deterministic=deterministic,
            **repr_params,
        )
        dX_cuda = None

    Q_cuda = torch.from_numpy(np.ascontiguousarray(Q_z_np)).cuda()
    return X_cuda, dX_cuda, Q_cuda, Q_z_np, N_np


def _solve_eigh_truncated(
    K: Any,  # noqa: ANN401
    rhs: Any,  # noqa: ANN401
    rtol: float,
    atol: float,
    max_rank: int | None,
) -> tuple[Any, dict[str, float | int]]:
    """Solve a symmetric GPU system via truncated eigendecomposition."""
    import torch

    evals, evecs = torch.linalg.eigh(K)
    lam_max = float(evals[-1].item())
    tau = max(float(atol), float(rtol) * lam_max)
    keep = torch.nonzero(evals > tau, as_tuple=False).squeeze(-1)

    if max_rank is not None and keep.numel() > max_rank:
        keep = keep[-max_rank:]

    if keep.numel() == 0:
        msg = f"Truncated eigensolve kept no eigenpairs. lambda_max={lam_max:.3e}, tau={tau:.3e}."
        raise RuntimeError(msg)

    V = evecs.index_select(1, keep)
    s = evals.index_select(0, keep)
    proj = V.mT @ rhs
    alpha = V @ (proj / s)

    info: dict[str, float | int] = {
        "rank": int(keep.numel()),
        "n": int(K.shape[0]),
        "lam_min": float(evals[0].item()),
        "lam_max": lam_max,
        "tau": tau,
        "dropped": int(K.shape[0] - keep.numel()),
    }
    return alpha, info


def _solve_cg(
    K: Any,  # noqa: ANN401
    rhs: Any,  # noqa: ANN401
    rtol: float,
    atol: float,
    max_iter: int | None,
) -> tuple[Any, dict[str, float | int]]:
    """Solve a symmetric positive-definite GPU system with restarted Jacobi CG."""
    import torch

    restart_period = 50
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    diag = K.diagonal()
    eps = torch.finfo(K.dtype).eps
    M_inv = torch.where(diag > eps, diag.reciprocal(), torch.ones_like(diag))
    z = M_inv * r
    p = z.clone()
    rz_old = torch.dot(r, z)

    rhs_norm = float(torch.linalg.vector_norm(rhs).item())
    tol = max(float(atol), float(rtol) * rhs_norm)
    residual_norm = float(torch.linalg.vector_norm(r).item())
    max_steps = 10 * int(K.shape[0]) if max_iter is None else max_iter

    if residual_norm <= tol:
        return x, {
            "iterations": 0,
            "max_iter": max_steps,
            "residual_norm": residual_norm,
            "tol": tol,
            "n": int(K.shape[0]),
        }

    for iteration in range(1, max_steps + 1):
        Kp = K @ p
        denom = torch.dot(p, Kp)
        denom_val = float(denom.item())
        if denom_val <= 0.0 or not np.isfinite(denom_val):
            msg = (
                "CG breakdown: non-positive search-direction curvature "
                f"at iteration {iteration} (p^T K p={denom_val:.3e})."
            )
            raise RuntimeError(msg)

        step = rz_old / denom
        x = x + step * p
        # Rebuild residual periodically to control FP32 drift, otherwise update.
        r = rhs - (K @ x) if iteration % restart_period == 0 else r - step * Kp
        residual_norm = float(torch.linalg.vector_norm(r).item())
        if residual_norm <= tol:
            return x, {
                "iterations": iteration,
                "max_iter": max_steps,
                "residual_norm": residual_norm,
                "tol": tol,
                "n": int(K.shape[0]),
                "restart_period": restart_period,
            }

        z = M_inv * r
        rz_new = torch.dot(r, z)
        if iteration % restart_period == 0:
            p = z.clone()
        else:
            beta = rz_new / rz_old
            p = z + beta * p
        rz_old = rz_new

    msg = (
        "CG did not converge within the iteration budget. "
        f"residual={residual_norm:.3e}, tol={tol:.3e}, max_iter={max_steps}."
    )
    raise RuntimeError(msg)


def _prepare_linear_system(
    K: Any,  # noqa: ANN401
    rhs: Any,  # noqa: ANN401
    l2: float,
    solver: str,
    preprocessing: str,
) -> tuple[Any, Any, Any | None, float]:
    """Apply diagonal regularisation and optional similarity scaling in place."""
    import torch

    diag = K.diagonal()
    effective_l2 = float(l2)
    diag.add_(effective_l2)
    if preprocessing == "none":
        return K, rhs, None, effective_l2

    if preprocessing == "diagonal_scale":
        eps = torch.finfo(K.dtype).eps
        row_scale = torch.rsqrt(torch.clamp(diag, min=eps))
        K.mul_(row_scale.unsqueeze(0))
        K.mul_(row_scale.unsqueeze(1))
        rhs_scaled = row_scale * rhs
        return K, rhs_scaled, row_scale, effective_l2

    msg = f"Unsupported preprocessing '{preprocessing}'. Expected 'none' or 'diagonal_scale'."
    raise ValueError(msg)


def _recover_training_predictions(
    K_system: Any,  # noqa: ANN401
    system_solution: Any,  # noqa: ANN401
    alpha: Any,  # noqa: ANN401
    effective_l2: float,
    row_scale: Any | None,  # noqa: ANN401
) -> NDArray[np.float64]:
    """Recover K @ alpha from the transformed linear system used during fitting."""
    system_pred = K_system @ system_solution
    if row_scale is not None:
        system_pred = system_pred / row_scale
    y_pred = system_pred - effective_l2 * alpha
    return y_pred.cpu().numpy().astype(np.float64)


def _solve_linear_system_gpu(
    K: Any,  # noqa: ANN401
    rhs: Any,  # noqa: ANN401
    solver: str,
    l2: float,
    preprocessing: str,
    spectral_rtol: float,
    spectral_atol: float,
    spectral_max_rank: int | None,
    cg_rtol: float,
    cg_atol: float,
    cg_max_iter: int | None,
) -> tuple[Any, Any, Any | None, float, dict[str, float | int | str]]:
    """Solve (K + l2 I) alpha = rhs on GPU."""
    import torch

    K_system, rhs_system, row_scale, effective_l2 = _prepare_linear_system(
        K,
        rhs,
        l2=l2,
        solver=solver,
        preprocessing=preprocessing,
    )

    if solver == "cholesky":
        chol_L, info = torch.linalg.cholesky_ex(K_system)
        if info.item() != 0:
            raise RuntimeError(
                f"Cholesky failed: leading minor of order {info.item()} not positive-definite."
            )
        y = torch.linalg.solve_triangular(chol_L, rhs_system.unsqueeze(-1), upper=False)
        system_solution = torch.linalg.solve_triangular(chol_L.mT, y, upper=True).squeeze(-1)
        alpha = system_solution if row_scale is None else row_scale * system_solution
        del chol_L, y
        return (
            alpha,
            system_solution,
            row_scale,
            effective_l2,
            {
                "solver": "cholesky",
                "rank": int(K.shape[0]),
                "n": int(K.shape[0]),
                "preprocessing": preprocessing,
                "effective_l2": effective_l2,
            },
        )

    if solver == "eigh":
        system_solution, eig_info = _solve_eigh_truncated(
            K_system,
            rhs_system,
            rtol=spectral_rtol,
            atol=spectral_atol,
            max_rank=spectral_max_rank,
        )
        alpha = system_solution if row_scale is None else row_scale * system_solution
        return (
            alpha,
            system_solution,
            row_scale,
            effective_l2,
            {
                "solver": "eigh",
                "preprocessing": preprocessing,
                "effective_l2": effective_l2,
                **eig_info,
            },
        )

    if solver == "cg":
        system_solution, cg_info = _solve_cg(
            K_system,
            rhs_system,
            rtol=cg_rtol,
            atol=cg_atol,
            max_iter=cg_max_iter,
        )
        alpha = system_solution if row_scale is None else row_scale * system_solution
        return (
            alpha,
            system_solution,
            row_scale,
            effective_l2,
            {
                "solver": "cg",
                "rank": int(K.shape[0]),
                "preprocessing": preprocessing,
                "effective_l2": effective_l2,
                **cg_info,
            },
        )

    msg = f"Unsupported solver '{solver}'. Expected 'cholesky', 'eigh', or 'cg'."
    raise ValueError(msg)


# Default FCHL19 representation hyperparameters
_DEFAULT_REPR_PARAMS: dict[str, Any] = {}
_DEFAULT_ELEMENTS: list[int] = [1, 6, 7, 8, 16]


class CudaLocalKRRModel(BaseModel):
    """GPU-accelerated KRR model using FCHL19 local descriptors.

    Training uses GPU float32 throughout: kernel assembly, a GPU linear solve
    (truncated ``torch.linalg.eigh`` by default), and ``alpha_desc``
    precomputation.  Inference uses the GPU J^T·alpha contracted matvec.

    **Only** ``energy_and_force`` training mode is supported.

    .. note::
        Float32 K_FF accumulation introduces visible spectral noise for FCHL19
        descriptors (rep_size ≈ 300). The default path diagonally scales the
        system and then uses a truncated eigensolver to drop numerically
        unstable modes instead of relying on a large Cholesky ridge.

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
        solver: Literal["cholesky", "eigh", "cg"] = "eigh",
        preprocessing: Literal["none", "diagonal_scale"] = "diagonal_scale",
        spectral_rtol: float = 3e-6,
        spectral_atol: float = 0.0,
        spectral_max_rank: int | None = None,
        cg_rtol: float = 2e-2,
        cg_atol: float = 0.0,
        cg_max_iter: int | None = None,
    ) -> None:
        _require_cuda_ext()
        _require_torch()
        self.sigma = sigma
        self.l2 = l2
        self.elements: list[int] = (
            sorted(elements) if elements is not None else list(_DEFAULT_ELEMENTS)
        )
        self.repr_params: dict[str, Any] = repr_params if repr_params is not None else {}
        self.solver = solver
        self.preprocessing = preprocessing
        self.spectral_rtol = spectral_rtol
        self.spectral_atol = spectral_atol
        self.spectral_max_rank = spectral_max_rank
        self.cg_rtol = cg_rtol
        self.cg_atol = cg_atol
        self.cg_max_iter = cg_max_iter
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
        import time

        import torch

        def _t(label: str, t0: float) -> float:
            torch.cuda.synchronize()
            print(f"  [CUDA] {label:<45s} {(time.perf_counter() - t0) * 1000:8.2f} ms")
            return time.perf_counter()

        mode = self.training_mode_
        if mode not in ("energy_only", "energy_and_force"):
            msg = f"CudaLocalKRRModel supports 'energy_only' and 'energy_and_force'. Got '{mode}'."
            raise NotImplementedError(msg)
        if energies is None:
            msg = "energies must be provided"
            raise ValueError(msg)
        if mode == "energy_and_force" and forces is None:
            msg = "forces must be provided for energy_and_force mode"
            raise ValueError(msg)

        t0 = time.perf_counter()

        need_gradients = mode == "energy_and_force"

        # ---- Step 1: build FCHL19 representations ----
        X_cuda, dX_cuda, Q_cuda, Q_krr, N = _compute_fchl19_cuda(
            coords_list,
            z_list,
            self.elements,
            with_gradients=need_gradients,
            repr_params=self.repr_params,
        )
        t0 = _t(f"Step 1  compute_fchl19 (GPU, {'f32+grad' if need_gradients else 'f32'})", t0)

        nm = int(X_cuda.shape[0])
        max_atoms = int(X_cuda.shape[1])
        rep_size = int(X_cuda.shape[2])
        naq = int(np.sum(N) * 3) if need_gradients else 0
        print(
            f"  [CUDA]   nm={nm}  max_atoms={max_atoms}  rep={rep_size}  naq={naq}  BIG={nm + naq}"
        )

        self._n_train = nm
        self._max_atoms = max_atoms
        self._rep_size = rep_size
        self._naq_train = naq

        N_cuda = _to_cuda_i32(N)

        if mode == "energy_only":
            # ---- energy_only: build K_EE in RFP format directly ----
            # Uses kernel_gaussian_symm_rfp (no nm*nm intermediate) + GPU
            # Cholesky (rfp_potrf / rfp_potrs).  Training predictions are
            # recovered as y_pred = energies - l2 * alpha, which follows from
            # (K + l2*I) @ alpha = energies => K @ alpha = energies - l2*alpha.
            from kernelforge import cuda_global_kernels as _gext  # rfp_potrf / rfp_potrs

            K_rfp = _ext.kernel_gaussian_symm_rfp(  # type: ignore[union-attr]
                X_cuda, Q_cuda, N_cuda, float(self.sigma)
            )
            t0 = _t("Step 3  kernel_gaussian_symm_rfp (GPU, RFP)", t0)

            rhs_gpu = torch.from_numpy(energies.astype(np.float32)).cuda().unsqueeze(-1)
            t0 = _t("Step 4a build RHS + H2D", t0)

            info = _gext.rfp_potrf(K_rfp, nm, float(self.l2))
            if info == 0:
                # In-place triangular solve: rhs_gpu ← L⁻ᵀ L⁻¹ rhs_gpu
                _gext.rfp_potrs(K_rfp, rhs_gpu)
                del K_rfp
                alpha_f32_gpu = rhs_gpu.squeeze(-1)
                del rhs_gpu
                alpha_np: NDArray[np.float64] = alpha_f32_gpu.cpu().numpy().astype(np.float64)
                y_pred_np = energies - float(self.l2) * alpha_np
                t0 = _t("Step 4b rfp_potrf + rfp_potrs (GPU)", t0)
            else:
                # Float32 Cholesky failed — fall back to CPU float64
                del K_rfp, rhs_gpu
                print(
                    f"  [CUDA-local] float32 RFP Cholesky failed (info={info}), "
                    "falling back to CPU float64"
                )
                from kernelforge import kernelmath
                from kernelforge import local_kernels as _cpu_lk

                X_cpu = X_cuda.cpu().numpy().astype(np.float64)
                K_rfp_cpu = _cpu_lk.kernel_gaussian_symm_rfp(X_cpu, Q_krr, N, float(self.sigma))
                del X_cpu
                alpha_np = kernelmath.cho_solve_rfp(K_rfp_cpu, energies, l2=float(self.l2))
                del K_rfp_cpu
                y_pred_np = energies - float(self.l2) * alpha_np
                alpha_f32_gpu = torch.from_numpy(alpha_np.astype(np.float32)).cuda()
                t0 = _t("Step 4b CPU fallback cho_solve_rfp (f64)", t0)

            # alpha_desc_F = zeros (no force coefficients)
            alpha_desc_F_cuda = torch.zeros(
                (nm, max_atoms, rep_size), dtype=torch.float32, device=X_cuda.device
            )

            # Store state
            self._X_train_np = X_cuda.cpu().numpy()
            self._dX_train_np = np.zeros((nm, max_atoms, rep_size, 3 * max_atoms), dtype=np.float32)
            self._Q_train_np = Q_krr.astype(np.int32)
            self._N_train_np = N.astype(np.int32)
            self._alpha_E_np = alpha_f32_gpu.cpu().numpy()
            self._alpha_desc_F_np = alpha_desc_F_cuda.cpu().numpy()
            t0 = _t("Step 6  D2H download alpha", t0)

            self._X_train_cuda = X_cuda
            self._dX_train_cuda = None  # not needed for energy_only
            self._Q_train_cuda = Q_cuda
            self._N_train_cuda = N_cuda
            self._alpha_E_cuda = alpha_f32_gpu
            self._alpha_desc_F_cuda = alpha_desc_F_cuda

            self._alpha: NDArray[np.float64] = alpha_np
            self._y_train: NDArray[np.float64] = energies.astype(np.float64)
            self._y_pred_train = y_pred_np
            del alpha_f32_gpu

        else:
            # ---- energy_and_force: build K_full in RFP packed format ----
            # Uses kernel_gaussian_full_symm_rfp (no dense BIGxBIG intermediate)
            # + GPU float32 Cholesky (rfp_potrf / rfp_potrs).  Training
            # predictions are recovered as y_pred = y - l2*alpha (analogous
            # to the energy_only branch).  The ``solver`` constructor argument
            # is ignored in this RFP path; failure of rfp_potrf raises.
            assert forces is not None  # noqa: S101
            from kernelforge import cuda_global_kernels as _gext  # rfp_potrf / rfp_potrs

            K_rfp = _ext.kernel_gaussian_full_symm_rfp(  # type: ignore[union-attr]
                X_cuda, dX_cuda, Q_cuda, N_cuda, float(self.sigma)
            )
            t0 = _t("Step 3  kernel_gaussian_full_symm_rfp (GPU, RFP)", t0)

            BIG = nm + naq
            F_neg = -forces
            rhs_f32 = np.concatenate([energies, F_neg.ravel()]).astype(np.float32)
            rhs_gpu = torch.from_numpy(rhs_f32).cuda().unsqueeze(-1)
            t0 = _t("Step 4a build RHS + H2D", t0)

            info = _gext.rfp_potrf(K_rfp, BIG, float(self.l2))
            if info != 0:
                raise RuntimeError(
                    f"GPU float32 RFP Cholesky failed (info={info}); "
                    "K is not positive definite at the requested l2.  "
                    "Increase l2 or switch to a different solver path."
                )
            _gext.rfp_potrs(K_rfp, rhs_gpu)
            del K_rfp
            alpha_f32_gpu = rhs_gpu.squeeze(-1)
            del rhs_gpu
            alpha_np = alpha_f32_gpu.cpu().numpy().astype(np.float64)
            rhs_f64 = np.concatenate([energies, F_neg.ravel()]).astype(np.float64)
            y_pred_train = rhs_f64 - float(self.l2) * alpha_np
            t0 = _t("Step 4b rfp_potrf + rfp_potrs (GPU)", t0)

            alpha_F_cuda = alpha_f32_gpu[nm:]
            alpha_desc_F_cuda = _ext.compute_alpha_desc(  # type: ignore[union-attr]
                dX_cuda, N_cuda, alpha_F_cuda
            )
            t0 = _t("Step 5  compute_alpha_desc (GPU)", t0)

            self._X_train_np = X_cuda.cpu().numpy()
            self._dX_train_np = dX_cuda.cpu().numpy()
            self._Q_train_np = Q_krr.astype(np.int32)
            self._N_train_np = N.astype(np.int32)
            self._alpha_E_np = alpha_f32_gpu[:nm].cpu().numpy()
            self._alpha_desc_F_np = alpha_desc_F_cuda.cpu().numpy()
            t0 = _t("Step 6  D2H download alpha + alpha_desc", t0)

            self._X_train_cuda = X_cuda
            self._dX_train_cuda = dX_cuda
            self._Q_train_cuda = Q_cuda
            self._N_train_cuda = N_cuda
            self._alpha_E_cuda = alpha_f32_gpu[:nm]
            self._alpha_desc_F_cuda = alpha_desc_F_cuda

            self._alpha = alpha_np
            self._y_train = rhs_f64
            self._y_pred_train = y_pred_train
            del alpha_f32_gpu

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def _predict(
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

        mode = self.training_mode_
        need_gradients = mode == "energy_and_force"

        X_te_cuda, dX_te_cuda, Q_te_cuda, _Q_te, N_te = _compute_fchl19_cuda(
            coords_list,
            z_list,
            self.elements,
            with_gradients=need_gradients,
            repr_params=self.repr_params,
            deterministic=True,
        )
        t0 = _tp(f"compute_fchl19 (GPU, {'grad' if need_gradients else 'no grad'})", t0)

        N_te_cuda = _to_cuda_i32(N_te)

        if mode == "energy_only":
            # Fast energy-only predict: K_EE_test @ alpha (no dX, no matvec)
            K_EE_test = _ext.kernel_gaussian_rect(  # type: ignore[union-attr]
                X_te_cuda,
                Q_te_cuda,
                N_te_cuda,
                self._X_train_cuda,
                self._Q_train_cuda,
                self._N_train_cuda,
                float(self.sigma),
            )
            E_cuda = K_EE_test @ self._alpha_E_cuda
            t0 = _tp("kernel_gaussian_rect + E = K@alpha (GPU)", t0)

            E_pred: NDArray[np.float64] = E_cuda.cpu().numpy().astype(np.float64)
            # No forces predicted in energy_only fast path
            F_pred = np.zeros(int(np.sum(N_te) * 3), dtype=np.float64)
            return E_pred, F_pred
        else:
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
            t0 = _tp("kernel_gaussian_full_matvec (GPU)", t0)

            E_pred = E_cuda.cpu().numpy().astype(np.float64)
            F_flat: NDArray[np.float64] = F_cuda.cpu().numpy().astype(np.float64)
            F_pred = -F_flat
            return E_pred, F_pred

    # ------------------------------------------------------------------
    # Training score
    # ------------------------------------------------------------------

    def _training_labels_and_predictions(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        n = self._n_train
        result: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
            "energy": (self._y_train[:n], self._y_pred_train[:n]),
        }
        if len(self._y_train) > n:
            result["force"] = (-self._y_train[n:], -self._y_pred_train[n:])
        return result

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _arrays_to_save(self) -> dict[str, object]:
        return {
            "model_class": "CudaLocalKRRModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "solver": self.solver,
            "preprocessing": self.preprocessing,
            "spectral_rtol": self.spectral_rtol,
            "spectral_atol": self.spectral_atol,
            "spectral_max_rank": -1 if self.spectral_max_rank is None else self.spectral_max_rank,
            "cg_rtol": self.cg_rtol,
            "cg_atol": self.cg_atol,
            "cg_max_iter": -1 if self.cg_max_iter is None else self.cg_max_iter,
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
            "y_pred_train": self._y_pred_train,
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
        self.solver = str(data["solver"]) if "solver" in data else "eigh"
        self.preprocessing = (
            str(data["preprocessing"]) if "preprocessing" in data else "diagonal_scale"
        )
        self.spectral_rtol = float(data["spectral_rtol"]) if "spectral_rtol" in data else 3e-6
        self.spectral_atol = float(data["spectral_atol"]) if "spectral_atol" in data else 0.0
        self.spectral_max_rank = (
            None
            if "spectral_max_rank" not in data or int(data["spectral_max_rank"]) < 0
            else int(data["spectral_max_rank"])
        )
        self.cg_rtol = float(data["cg_rtol"]) if "cg_rtol" in data else 2e-2
        self.cg_atol = float(data["cg_atol"]) if "cg_atol" in data else 0.0
        self.cg_max_iter = (
            None
            if "cg_max_iter" not in data or int(data["cg_max_iter"]) < 0
            else int(data["cg_max_iter"])
        )
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
        self._y_pred_train = (
            data["y_pred_train"].astype(np.float64)
            if "y_pred_train" in data
            else self._y_train - self.l2 * self._alpha
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
