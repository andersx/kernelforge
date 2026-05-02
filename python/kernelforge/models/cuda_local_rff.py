"""CudaLocalRFFModel: GPU RFF using FCHL19 local descriptors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .base import BaseModel, TrainingMode
from .cuda_local_krr import _require_cuda_fchl19, _to_cuda_i32
from .rff import _DEFAULT_ELEMENTS

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

_CUDA_SOLVERS_AVAILABLE = False
try:
    from kernelforge import cuda_solvers as _solvers_ext

    _CUDA_SOLVERS_AVAILABLE = True
except ImportError:
    pass


def _require_cuda() -> None:
    if not _TORCH_AVAILABLE:
        msg = "PyTorch is required for CudaLocalRFFModel."
        raise ImportError(msg)
    if not (_CUDA_RFF_AVAILABLE and _CUDA_KERNELS_AVAILABLE):
        msg = (
            "cuda_rff_features and cuda_global_kernels are required for CudaLocalRFFModel. "
            "Re-build kernelforge with CUDA, CUDAToolkit, and PyTorch."
        )
        raise ImportError(msg)


def _require_cuda_solvers() -> None:
    _require_cuda()
    if not _CUDA_SOLVERS_AVAILABLE:
        msg = (
            "cuda_solvers is required for solver='svd'/'qr'/'gels'. "
            "Re-build kernelforge with CUDA, CUDAToolkit, and PyTorch."
        )
        raise ImportError(msg)


def _to_cuda_f32(arr: NDArray[np.float32] | NDArray[np.float64]) -> Any:  # noqa: ANN401
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).cuda()


def _compact_forces(forces: NDArray[np.float64], N: NDArray[np.int32]) -> NDArray[np.float64]:
    F2 = np.asarray(forces, dtype=np.float64).reshape(len(N), -1)
    return np.concatenate([F2[i, : 3 * int(N[i])] for i in range(len(N))])


def _build_elem_indices(
    Q_idx_np: NDArray[np.int32],
    N_np: NDArray[np.int32],
    nelements: int,
) -> list[Any]:
    """Return per-element (mol_idx, atom_idx) pairs for all valid atoms."""
    max_atoms = Q_idx_np.shape[1]
    atom_range = np.arange(max_atoms)
    valid_mask = atom_range[np.newaxis, :] < N_np[:, np.newaxis]  # (nmol, max_atoms)
    return [np.where(valid_mask & (Q_idx_np == ei)) for ei in range(nelements)]


def _compute_fchl19_cuda_rff(
    coords_list: list[NDArray[np.float64]],
    z_list: list[NDArray[np.int32]],
    elements: list[int],
    with_gradients: bool,
    repr_params: dict[str, Any],
    deterministic: bool = True,
) -> tuple[Any, Any, Any, NDArray[np.int32], NDArray[np.int32]]:
    _require_cuda_fchl19()
    import torch

    from kernelforge import cuda_fchl19_repr as _cuda_fchl19

    elem_to_idx = {e: i for i, e in enumerate(elements)}
    nm = len(coords_list)
    N_np = np.array([len(z) for z in z_list], dtype=np.int32)
    max_atoms = int(N_np.max())

    coords_np = np.zeros((nm, max_atoms, 3), dtype=np.float32)
    Q_idx_np = np.zeros((nm, max_atoms), dtype=np.int32)

    for m, (coords, z) in enumerate(zip(coords_list, z_list, strict=True)):
        z_i32 = np.asarray(z, dtype=np.int32)
        na = len(z_i32)
        coords_np[m, :na, :] = np.asarray(coords, dtype=np.float32)
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

    return X_cuda, dX_cuda, Q_idx_cuda, Q_idx_np, N_np


class CudaLocalRFFModel(BaseModel):
    """GPU Random Fourier Features model for FCHL19 local descriptors."""

    @classmethod
    def load(cls, path: str | Path) -> CudaLocalRFFModel:  # type: ignore[override]
        """Load a trained CudaLocalRFFModel from a .npz file."""
        return cast(CudaLocalRFFModel, super().load(path))

    def __init__(
        self,
        sigma: float = 20.0,
        l2: float = 1e-6,
        d_rff: int = 4096,
        seed: int = 42,
        elements: list[int] | None = None,
        repr_params: dict[str, Any] | None = None,
        chunk_size: int = 256,
        solver: str = "cholesky",
        rcond: float = -1.0,
        gels_variant: str = "SS",
        svdr_rank: int = 256,
        svdr_p: int = 10,
        svdr_niters: int = 2,
        n_pca: int | None = None,
        pca_center: bool = False,
        pca_whiten: bool = False,
    ) -> None:
        self.sigma = sigma
        self.l2 = l2
        self.d_rff = d_rff
        self.seed = seed
        self.elements: list[int] = sorted(elements) if elements is not None else _DEFAULT_ELEMENTS
        self.repr_params: dict[str, Any] = repr_params if repr_params is not None else {}
        self.chunk_size = chunk_size
        self.solver = solver
        self.rcond = rcond
        self.gels_variant = gels_variant
        self.svdr_rank = svdr_rank
        self.svdr_p = svdr_p
        self.svdr_niters = svdr_niters
        self.n_pca = n_pca
        self.pca_center = pca_center
        self.pca_whiten = pca_whiten
        self.is_fitted_ = False

    def _fit_pca(
        self,
        X_cuda: Any,  # noqa: ANN401  # torch.Tensor (nmol, max_atoms, rep_size) float32 CUDA
        Q_idx_np: NDArray[np.int32],
        N_np: NDArray[np.int32],
    ) -> None:
        """Fit per-element PCA from training descriptors and store projection matrices."""
        import torch

        if self.n_pca is None:
            msg = "_fit_pca called but n_pca is None"
            raise RuntimeError(msg)
        rep_size = int(X_cuda.shape[2])
        n_pca = self.n_pca
        nelements = len(self.elements)

        if n_pca >= rep_size:
            msg = f"n_pca={n_pca} must be strictly less than rep_size={rep_size}."
            raise ValueError(msg)

        elem_indices = _build_elem_indices(Q_idx_np, N_np, nelements)
        pca_matrix_list = []
        pca_mean_list = []
        pca_scale_list = []

        for ei, (mol_idx, atom_idx) in enumerate(elem_indices):
            n_valid = len(mol_idx)
            z_label = self.elements[ei]

            if n_valid == 0:
                print(
                    f"  [PCA] element Z={z_label:2d}  WARNING: no training atoms; "
                    "using identity projection."
                )
                P = torch.eye(rep_size, n_pca, dtype=torch.float32, device=X_cuda.device)
                mean = torch.zeros(rep_size, dtype=torch.float32, device=X_cuda.device)
                scale = torch.ones(n_pca, dtype=torch.float32, device=X_cuda.device)
                pca_matrix_list.append(P)
                pca_mean_list.append(mean)
                pca_scale_list.append(scale)
                continue

            if n_valid < n_pca:
                msg = (
                    f"Element Z={z_label} has only {n_valid} training atoms, "
                    f"which is fewer than n_pca={n_pca}. "
                    "Reduce --n-pca or add more training data."
                )
                raise ValueError(msg)

            mol_idx_t = torch.from_numpy(mol_idx).long().to(X_cuda.device)
            atom_idx_t = torch.from_numpy(atom_idx).long().to(X_cuda.device)
            x_elem = X_cuda[mol_idx_t, atom_idx_t, :].float()  # (n_valid, rep_size)

            if self.pca_center:
                mean = x_elem.mean(dim=0)  # (rep_size,)
                x_elem = x_elem - mean
            else:
                mean = torch.zeros(rep_size, dtype=torch.float32, device=X_cuda.device)

            _U, S, Vh = torch.linalg.svd(x_elem, full_matrices=False, driver="gesvd")
            # P columns are the top n_pca principal directions
            P = Vh[:n_pca, :].T.contiguous()  # (rep_size, n_pca)

            if self.pca_whiten:
                scale = S[:n_pca] / float(np.sqrt(n_valid)) + 1e-8
            else:
                scale = torch.ones(n_pca, dtype=torch.float32, device=X_cuda.device)

            total_var = (S**2).sum().item()
            explained_var = (S[:n_pca] ** 2).sum().item() / total_var * 100.0
            print(
                f"  [PCA] element Z={z_label:2d}  n_valid={n_valid:6d}  "
                f"n_pca={n_pca:4d}/{rep_size:4d}  explained_var={explained_var:.1f}%"
            )

            pca_matrix_list.append(P)
            pca_mean_list.append(mean)
            pca_scale_list.append(scale)

        self._pca_matrix_cuda = torch.stack(pca_matrix_list, dim=0)
        self._pca_mean_cuda = torch.stack(pca_mean_list, dim=0)
        self._pca_scale_cuda = torch.stack(pca_scale_list, dim=0)

    def _apply_pca_X(
        self,
        X_cuda: Any,  # noqa: ANN401  # torch.Tensor (nmol, max_atoms, rep_size)
        Q_idx_np: NDArray[np.int32],
        N_np: NDArray[np.int32],
    ) -> Any:  # noqa: ANN401  # torch.Tensor (nmol, max_atoms, n_pca)
        """Apply per-element PCA projection to descriptor tensor."""
        import torch

        if self.n_pca is None:
            msg = "_apply_pca_X called but n_pca is None"
            raise RuntimeError(msg)
        nmol = int(X_cuda.shape[0])
        max_atoms = int(X_cuda.shape[1])
        n_pca = self.n_pca
        nelements = len(self.elements)

        X_pca = torch.zeros(nmol, max_atoms, n_pca, dtype=torch.float32, device=X_cuda.device)
        elem_indices = _build_elem_indices(Q_idx_np, N_np, nelements)

        for ei, (mol_idx, atom_idx) in enumerate(elem_indices):
            if len(mol_idx) == 0:
                continue
            mol_idx_t = torch.from_numpy(mol_idx).long().to(X_cuda.device)
            atom_idx_t = torch.from_numpy(atom_idx).long().to(X_cuda.device)
            x_elem = X_cuda[mol_idx_t, atom_idx_t, :]  # (n_valid, rep_size)
            if self.pca_center:
                x_elem = x_elem - self._pca_mean_cuda[ei]
            x_proj = x_elem @ self._pca_matrix_cuda[ei]  # (n_valid, n_pca)
            if self.pca_whiten:
                x_proj = x_proj / self._pca_scale_cuda[ei]
            X_pca[mol_idx_t, atom_idx_t] = x_proj

        return X_pca

    def _apply_pca_dX(
        self,
        dX_cuda: Any,  # noqa: ANN401  # torch.Tensor (nmol, max_atoms, rep_size, ncoords)
        Q_idx_np: NDArray[np.int32],
        N_np: NDArray[np.int32],
    ) -> Any:  # noqa: ANN401  # torch.Tensor (nmol, max_atoms, n_pca, ncoords)
        """Apply per-element PCA projection to descriptor gradient tensor."""
        import torch

        if self.n_pca is None:
            msg = "_apply_pca_dX called but n_pca is None"
            raise RuntimeError(msg)
        nmol = int(dX_cuda.shape[0])
        max_atoms = int(dX_cuda.shape[1])
        ncoords = int(dX_cuda.shape[3])
        n_pca = self.n_pca
        nelements = len(self.elements)

        dX_pca = torch.zeros(
            nmol, max_atoms, n_pca, ncoords, dtype=torch.float32, device=dX_cuda.device
        )
        elem_indices = _build_elem_indices(Q_idx_np, N_np, nelements)

        for ei, (mol_idx, atom_idx) in enumerate(elem_indices):
            if len(mol_idx) == 0:
                continue
            mol_idx_t = torch.from_numpy(mol_idx).long().to(dX_cuda.device)
            atom_idx_t = torch.from_numpy(atom_idx).long().to(dX_cuda.device)
            dx_elem = dX_cuda[mol_idx_t, atom_idx_t, :, :]
            P = self._pca_matrix_cuda[ei]  # (rep_size, n_pca)
            dx_proj = dx_elem.permute(0, 2, 1) @ P  # (n_valid, ncoords, n_pca)
            dx_proj = dx_proj.permute(0, 2, 1).contiguous()  # (n_valid, n_pca, ncoords)
            if self.pca_whiten:
                dx_proj = dx_proj / self._pca_scale_cuda[ei].view(1, -1, 1)
            dX_pca[mol_idx_t, atom_idx_t] = dx_proj

        return dX_pca

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
            print(f"  [CUDA-RFF-local] {label:<45s} {dt * 1000:8.2f} ms")
            return _time.perf_counter()

        t0 = _time.perf_counter()
        mode = self.training_mode_
        if mode not in ("energy_only", "energy_and_force"):
            msg = "CudaLocalRFFModel currently supports energy_only and energy_and_force training."
            raise NotImplementedError(msg)
        if self.solver not in ("cholesky", "svd", "qr", "gels", "svdr"):
            msg = (
                f"CudaLocalRFFModel: unknown solver '{self.solver}'. "
                "Use 'cholesky', 'svd', 'qr', 'gels', or 'svdr'."
            )
            raise ValueError(msg)
        if self.solver == "gels" and self.gels_variant not in ("SS", "SH", "SB", "SX"):
            msg = (
                f"CudaLocalRFFModel: unknown gels_variant '{self.gels_variant}'. "
                "Use 'SS', 'SH', 'SB', or 'SX'."
            )
            raise ValueError(msg)
        if self.solver in ("svd", "qr", "gels", "svdr"):
            _require_cuda_solvers()
        if energies is None:
            msg = f"energies must be provided for {mode} mode"
            raise ValueError(msg)
        if mode == "energy_and_force" and forces is None:
            msg = "forces must be provided for energy_and_force mode"
            raise ValueError(msg)

        need_gradients = mode == "energy_and_force"
        X_cuda, dX_cuda, Q_idx_cuda, Q_idx_np, N_np = _compute_fchl19_cuda_rff(
            coords_list,
            z_list,
            self.elements,
            with_gradients=need_gradients,
            repr_params=self.repr_params,
            deterministic=False,
        )
        t0 = _t(
            f"Step 1  compute_fchl19 (GPU, {'f32+grad' if need_gradients else 'f32'})",
            t0,
        )
        N_cuda = _to_cuda_i32(N_np)
        nmol = int(X_cuda.shape[0])
        max_atoms = int(X_cuda.shape[1])
        rep_size = int(X_cuda.shape[2])
        nelements = len(self.elements)
        naq = int(np.sum(N_np) * 3) if need_gradients else 0
        print(
            f"  [CUDA-RFF-local]   nmol={nmol}  max_atoms={max_atoms}  rep={rep_size}  "
            f"nelements={nelements}  d_rff={self.d_rff}  naq={naq}  chunk_size={self.chunk_size}"
        )
        self._n_train = nmol
        self._max_atoms = max_atoms
        self._rep_size = rep_size

        if self.n_pca is not None:
            self._fit_pca(X_cuda, Q_idx_np, N_np)
            X_cuda = self._apply_pca_X(X_cuda, Q_idx_np, N_np)
            if dX_cuda is not None:
                dX_cuda = self._apply_pca_dX(dX_cuda, Q_idx_np, N_np)
            rep_size = self.n_pca
            self._rep_size = rep_size
            t0 = _t(f"Step 1b fit+apply PCA  (n_pca={self.n_pca})", t0)

        rng = np.random.default_rng(self.seed)
        W_np: NDArray[np.float32] = rng.standard_normal((nelements, rep_size, self.d_rff)).astype(
            np.float32
        ) / np.float32(self.sigma)
        b_np: NDArray[np.float32] = rng.uniform(0.0, 2.0 * np.pi, (nelements, self.d_rff)).astype(
            np.float32
        )
        W_cuda = _to_cuda_f32(W_np)
        b_cuda = _to_cuda_f32(b_np)
        Y_cuda = _to_cuda_f32(energies.astype(np.float32))
        t0 = _t("Step 2  generate/upload RFF params + targets", t0)

        self._f_train: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._f_pred_train: NDArray[np.float64] = np.array([], dtype=np.float64)
        if self.solver in ("svd", "qr", "gels", "svdr"):
            if self.solver == "svdr":
                # Clamp k so that k + p <= d_rff (constraint of cusolverDnXgesvdr).
                k_eff = min(self.svdr_rank, self.d_rff - self.svdr_p)
                if k_eff < 1:
                    msg = (
                        f"svdr_p={self.svdr_p} >= d_rff={self.d_rff}; "
                        "cannot satisfy k+p<=d_rff with k>=1. "
                        "Reduce --svdr-p or increase --d-rff."
                    )
                    raise ValueError(msg)
            if mode == "energy_and_force":
                if forces is None or dX_cuda is None:
                    msg = "forces or dX_cuda missing in energy_and_force fit — internal error"
                    raise RuntimeError(msg)
                dX5_cuda = dX_cuda.reshape(nmol, max_atoms, rep_size, max_atoms, 3).contiguous()
                F_flat64 = _compact_forces(forces, N_np)
                F_cuda = _to_cuda_f32(F_flat64.astype(np.float32))
                Z_cuda = _rff_ext.rff_features_elemental_col_major(  # type: ignore[union-attr]
                    X_cuda, Q_idx_cuda, N_cuda, W_cuda, b_cuda
                )
                t0 = _t("Step 3a rff_features_elemental (Z col-major)", t0)
                G_cuda = _rff_ext.rff_gradient_elemental_col_major(  # type: ignore[union-attr]
                    X_cuda,
                    dX5_cuda,
                    Q_idx_cuda,
                    N_cuda,
                    W_cuda,
                    b_cuda,
                    int(self.chunk_size),
                )
                t0 = _t("Step 3b rff_gradient_elemental (G col-major)", t0)
                ZG_cuda = torch.cat([Z_cuda, G_cuda], dim=1)
                EF_cuda = torch.cat([Y_cuda, F_cuda], dim=0)
                del Z_cuda, G_cuda
                t0 = _t("Step 3c cat [Z;G] col-major and [E;F]", t0)
                if self.solver == "svd":
                    weights_cuda = _solvers_ext.cuda_solve_svd(  # type: ignore[union-attr]
                        ZG_cuda, EF_cuda, float(self.rcond), True
                    ).cuda()
                    t0 = _t("Step 4  cuda_solve_svd (SVD lstsq, FP32)", t0)
                elif self.solver == "svdr":
                    weights_cuda = _solvers_ext.cuda_solve_svdr(  # type: ignore[union-attr]
                        ZG_cuda,
                        EF_cuda,
                        float(self.rcond),
                        k_eff,
                        self.svdr_p,
                        self.svdr_niters,
                        True,
                    ).cuda()
                    t0 = _t(
                        f"Step 4  cuda_solve_svdr (rSVD k={k_eff} p={self.svdr_p} "
                        f"niters={self.svdr_niters})",
                        t0,
                    )
                elif self.solver == "gels":
                    weights_cuda = _solvers_ext.cuda_solve_gels(  # type: ignore[union-attr]
                        ZG_cuda, EF_cuda, True, self.gels_variant
                    ).cuda()
                    t0 = _t(f"Step 4  cuda_solve_gels/{self.gels_variant} (IRS lstsq)", t0)
                else:
                    weights_cuda = _solvers_ext.cuda_solve_qr(  # type: ignore[union-attr]
                        ZG_cuda, EF_cuda, True
                    ).cuda()
                    t0 = _t("Step 4  cuda_solve_qr (QR lstsq)", t0)
                del ZG_cuda, EF_cuda
            else:
                F_cuda = None
                dX5_cuda = None
                Z_cuda = _rff_ext.rff_features_elemental_col_major(  # type: ignore[union-attr]
                    X_cuda, Q_idx_cuda, N_cuda, W_cuda, b_cuda
                )
                t0 = _t("Step 3  rff_features_elemental (materialize Z col-major)", t0)
                if self.solver == "svd":
                    weights_cuda = _solvers_ext.cuda_solve_svd(  # type: ignore[union-attr]
                        Z_cuda, Y_cuda, float(self.rcond), True
                    ).cuda()
                    t0 = _t("Step 4  cuda_solve_svd (SVD lstsq, FP32)", t0)
                elif self.solver == "svdr":
                    weights_cuda = _solvers_ext.cuda_solve_svdr(  # type: ignore[union-attr]
                        Z_cuda,
                        Y_cuda,
                        float(self.rcond),
                        k_eff,
                        self.svdr_p,
                        self.svdr_niters,
                        True,
                    ).cuda()
                    t0 = _t(
                        f"Step 4  cuda_solve_svdr (rSVD k={k_eff} p={self.svdr_p} "
                        f"niters={self.svdr_niters})",
                        t0,
                    )
                elif self.solver == "gels":
                    weights_cuda = _solvers_ext.cuda_solve_gels(  # type: ignore[union-attr]
                        Z_cuda, Y_cuda, True, self.gels_variant
                    ).cuda()
                    t0 = _t(f"Step 4  cuda_solve_gels/{self.gels_variant} (IRS lstsq)", t0)
                else:
                    weights_cuda = _solvers_ext.cuda_solve_qr(  # type: ignore[union-attr]
                        Z_cuda, Y_cuda, True
                    ).cuda()
                    t0 = _t("Step 4  cuda_solve_qr (QR lstsq)", t0)
                del Z_cuda
        elif mode == "energy_and_force":
            if forces is None or dX_cuda is None:
                msg = "forces or dX_cuda missing in energy_and_force fit — internal error"
                raise RuntimeError(msg)
            dX5_cuda = dX_cuda.reshape(nmol, max_atoms, rep_size, max_atoms, 3).contiguous()
            F_flat64 = _compact_forces(forces, N_np)
            F_flat = F_flat64.astype(np.float32)
            F_cuda = _to_cuda_f32(F_flat)
            ZtZ_rfp, ZtY = _rff_ext.rff_full_gramian_elemental_rfp(  # type: ignore[union-attr]
                X_cuda,
                dX5_cuda,
                Q_idx_cuda,
                N_cuda,
                W_cuda,
                b_cuda,
                Y_cuda,
                F_cuda,
                int(self.chunk_size),
                int(self.chunk_size),
            )
            t0 = _t("Step 3  rff_full_gramian_elemental_rfp", t0)
            rhs = ZtY.unsqueeze(-1)
            t0 = _t("Step 4a build RHS view", t0)
            info = _kern_ext.rfp_potrf(ZtZ_rfp, int(self.d_rff), float(self.l2))  # type: ignore[union-attr]
            if info != 0:
                msg = (
                    f"rfp_potrf (CudaLocalRFFModel {mode}): Cholesky factorization failed "
                    f"(info={info}). Try increasing l2."
                )
                raise RuntimeError(msg)
            _kern_ext.rfp_potrs(ZtZ_rfp, rhs)  # type: ignore[union-attr]
            t0 = _t("Step 4b rfp_potrf + rfp_potrs (GPU)", t0)
            weights_cuda = rhs.squeeze(-1).contiguous()
        else:
            F_cuda = None
            dX5_cuda = None
            ZtZ_rfp, ZtY = _rff_ext.rff_gramian_elemental_rfp(  # type: ignore[union-attr]
                X_cuda, Q_idx_cuda, N_cuda, W_cuda, b_cuda, Y_cuda, int(self.chunk_size)
            )
            t0 = _t("Step 3  rff_gramian_elemental_rfp", t0)
            rhs = ZtY.unsqueeze(-1)
            t0 = _t("Step 4a build RHS view", t0)
            info = _kern_ext.rfp_potrf(ZtZ_rfp, int(self.d_rff), float(self.l2))  # type: ignore[union-attr]
            if info != 0:
                msg = (
                    f"rfp_potrf (CudaLocalRFFModel {mode}): Cholesky factorization failed "
                    f"(info={info}). Try increasing l2."
                )
                raise RuntimeError(msg)
            _kern_ext.rfp_potrs(ZtZ_rfp, rhs)  # type: ignore[union-attr]
            t0 = _t("Step 4b rfp_potrf + rfp_potrs (GPU)", t0)
            weights_cuda = rhs.squeeze(-1).contiguous()
        y_pred_cuda = _rff_ext.rff_predict_energy_elemental(  # type: ignore[union-attr]
            X_cuda, Q_idx_cuda, N_cuda, W_cuda, b_cuda, weights_cuda, int(self.chunk_size)
        )
        t0 = _t("Step 5a train energy prediction (GPU)", t0)
        if mode == "energy_and_force" and dX5_cuda is not None:
            f_pred_cuda = _rff_ext.rff_predict_force_elemental(  # type: ignore[union-attr]
                X_cuda,
                dX5_cuda,
                Q_idx_cuda,
                N_cuda,
                W_cuda,
                b_cuda,
                weights_cuda,
                int(self.chunk_size),
            )
            self._f_train = F_flat64
            self._f_pred_train = f_pred_cuda.cpu().numpy().astype(np.float64)
            t0 = _t("Step 5b train force prediction + D2H", t0)

        self._W_np = W_np
        self._b_np = b_np
        self._weights_np = weights_cuda.cpu().numpy().astype(np.float32)
        self._W_cuda = W_cuda
        self._b_cuda = b_cuda
        self._weights_cuda = weights_cuda
        self._Q_train_np = Q_idx_np.astype(np.int32)
        self._N_train_np = N_np.astype(np.int32)
        self._y_train = energies
        self._y_pred_train = y_pred_cuda.cpu().numpy().astype(np.float64)
        t0 = _t("Step 6  D2H train outputs + store state", t0)

        del X_cuda, dX_cuda, Q_idx_cuda, N_cuda, Y_cuda, F_cuda

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
        compute_energy: bool = True,
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
        need_gradients = self.training_mode_ == "energy_and_force"
        X_cuda, dX_cuda, Q_idx_cuda, _Q_idx_np, N_np = _compute_fchl19_cuda_rff(
            coords_list,
            z_list,
            self.elements,
            with_gradients=need_gradients,
            repr_params=self.repr_params,
            deterministic=False,
        )
        t0 = _tp(f"compute_fchl19 (GPU, {'grad' if need_gradients else 'no grad'})", t0)
        N_cuda = _to_cuda_i32(N_np)
        if self.n_pca is not None:
            X_cuda = self._apply_pca_X(X_cuda, _Q_idx_np, N_np)
            if dX_cuda is not None:
                dX_cuda = self._apply_pca_dX(dX_cuda, _Q_idx_np, N_np)
            t0 = _tp(f"apply PCA  (n_pca={self.n_pca})", t0)
        if compute_energy:
            E_cuda = _rff_ext.rff_predict_energy_elemental(  # type: ignore[union-attr]
                X_cuda,
                Q_idx_cuda,
                N_cuda,
                self._W_cuda,
                self._b_cuda,
                self._weights_cuda,
                int(self.chunk_size),
            )
            t0 = _tp("rff_predict_energy_elemental (GPU)", t0)
            E_pred: NDArray[np.float64] = E_cuda.cpu().numpy().astype(np.float64)
        else:
            E_pred = np.zeros(len(coords_list), dtype=np.float64)
        if self.training_mode_ == "energy_and_force":
            if dX_cuda is None:
                msg = "dX_cuda is None in energy_and_force predict — internal error"
                raise RuntimeError(msg)
            max_atoms = int(X_cuda.shape[1])
            rep_size = int(X_cuda.shape[2])
            dX5_cuda = dX_cuda.reshape(
                len(coords_list), max_atoms, rep_size, max_atoms, 3
            ).contiguous()
            F_cuda = _rff_ext.rff_predict_force_elemental(  # type: ignore[union-attr]
                X_cuda,
                dX5_cuda,
                Q_idx_cuda,
                N_cuda,
                self._W_cuda,
                self._b_cuda,
                self._weights_cuda,
                int(self.chunk_size),
            )
            t0 = _tp("rff_predict_force_elemental (GPU)", t0)
            F_pred = F_cuda.cpu().numpy().astype(np.float64)
        else:
            F_pred = np.zeros(int(np.sum(N_np) * 3), dtype=np.float64)
        t0 = _tp("D2H predictions", t0)
        return E_pred, F_pred

    def _arrays_to_save(self) -> dict[str, object]:
        result: dict[str, object] = {
            "model_class": "CudaLocalRFFModel",
            "training_mode": self.training_mode_,
            "sigma": self.sigma,
            "l2": self.l2,
            "d_rff": self.d_rff,
            "seed": self.seed,
            "elements": np.array(self.elements, dtype=np.int32),
            "repr_params": json.dumps(self.repr_params),
            "chunk_size": self.chunk_size,
            "solver": self.solver,
            "rcond": self.rcond,
            "gels_variant": self.gels_variant,
            "n_pca": self.n_pca if self.n_pca is not None else -1,
            "pca_center": self.pca_center,
            "pca_whiten": self.pca_whiten,
            "baseline_elements": self.baseline_elements_,
            "element_energies": self.element_energies_,
            "weights": self._weights_cuda.cpu().numpy(),
            "W": self._W_cuda.cpu().numpy(),
            "b": self._b_cuda.cpu().numpy(),
            "n_train": self._n_train,
            "max_atoms": self._max_atoms,
            "rep_size": self._rep_size,
            "Q_train": self._Q_train_np,
            "N_train": self._N_train_np,
            "y_train": self._y_train,
            "y_pred_train": self._y_pred_train,
            "f_train": self._f_train,
            "f_pred_train": self._f_pred_train,
        }
        if self.n_pca is not None:
            result["pca_matrix"] = self._pca_matrix_cuda.cpu().numpy()
            result["pca_mean"] = self._pca_mean_cuda.cpu().numpy()
            result["pca_scale"] = self._pca_scale_cuda.cpu().numpy()
        return result

    def _load_from_arrays(self, data: np.lib.npyio.NpzFile) -> None:
        _require_cuda()
        self.sigma = float(data["sigma"])
        self.l2 = float(data["l2"])
        self.d_rff = int(data["d_rff"])
        self.seed = int(data["seed"])
        self.elements = data["elements"].astype(np.int32).tolist()
        self.repr_params = json.loads(str(data["repr_params"]))
        self.chunk_size = int(data["chunk_size"]) if "chunk_size" in data else 256
        self.solver = str(data["solver"]) if "solver" in data else "cholesky"
        self.rcond = float(data["rcond"]) if "rcond" in data else -1.0
        self.gels_variant = str(data["gels_variant"]) if "gels_variant" in data else "SS"
        n_pca_val = int(data["n_pca"]) if "n_pca" in data else -1
        self.n_pca = n_pca_val if n_pca_val > 0 else None
        self.pca_center = bool(data["pca_center"]) if "pca_center" in data else False
        self.pca_whiten = bool(data["pca_whiten"]) if "pca_whiten" in data else False
        self.baseline_elements_ = data["baseline_elements"].astype(np.int32)
        self.element_energies_ = data["element_energies"].astype(np.float64)
        self.training_mode_: TrainingMode = cast(TrainingMode, str(data["training_mode"]))
        self._n_train = int(data["n_train"])
        self._max_atoms = int(data["max_atoms"])
        self._rep_size = int(data["rep_size"])
        self._Q_train_np = data["Q_train"].astype(np.int32)
        self._N_train_np = data["N_train"].astype(np.int32)
        self._y_train = data["y_train"].astype(np.float64)
        self._y_pred_train = data["y_pred_train"].astype(np.float64)
        self._f_train = data["f_train"].astype(np.float64) if "f_train" in data else np.array([])
        self._f_pred_train = (
            data["f_pred_train"].astype(np.float64) if "f_pred_train" in data else np.array([])
        )
        self._W_np = data["W"].astype(np.float32)
        self._b_np = data["b"].astype(np.float32)
        self._weights_np = data["weights"].astype(np.float32)
        self._W_cuda = _to_cuda_f32(self._W_np)
        self._b_cuda = _to_cuda_f32(self._b_np)
        self._weights_cuda = _to_cuda_f32(self._weights_np)
        if self.n_pca is not None:
            self._pca_matrix_cuda = _to_cuda_f32(data["pca_matrix"].astype(np.float32))
            self._pca_mean_cuda = _to_cuda_f32(data["pca_mean"].astype(np.float32))
            self._pca_scale_cuda = _to_cuda_f32(data["pca_scale"].astype(np.float32))
