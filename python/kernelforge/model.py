"""High-level model interface for KernelForge.

Provides sklearn-style fit/predict API for Kernel Ridge Regression and GDML models.
All models support:
- Energy prediction (always)
- Force prediction (GDML and RFF models when use_forces=True)
- Cross-validation and hyperparameter optimization
- Save/load model persistence via .npz files
- No external dependencies beyond NumPy

Example:
    >>> from kernelforge.model import LocalGaussianKRR, load_data
    >>> data = load_data("ethanol.npz")
    >>> model = LocalGaussianKRR(sigma=2.0, regularize=1e-6)
    >>> model.fit(data["R"], data["Z"], data["E"])
    >>> E_pred = model.predict(data["R"][:10], data["Z"][:10])
    >>> model.save("ethanol_model.npz")
"""

import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available (shouldn't happen, but defensive)
    tqdm = None  # type: ignore

from . import __version__
from . import global_kernels, invdist_repr, kernelmath, local_kernels
from .fchl19_repr import (
    compute_rep_size,
    generate_fchl_acsf,
    generate_fchl_acsf_and_gradients,
)
from .kitchen_sinks import (
    rff_features_elemental,
    rff_gradient_elemental,
    rff_gramian_elemental,
    rff_gramian_elemental_gradient,
)

TModel = TypeVar("TModel", bound="BaseModel")

# Model registry for load_model dispatch
_MODEL_REGISTRY: dict[str, type["BaseModel"]] = {}


def _register_model(cls: type[TModel]) -> type[TModel]:
    """Decorator to register a model class in the global registry."""
    _MODEL_REGISTRY[cls.__name__] = cls
    return cls


# ============================================================================
# Utility Functions
# ============================================================================


def load_data(
    path: str | Path,
    *,
    R_key: str = "R",
    Z_key: str = "z",
    E_key: str = "E",
    F_key: str = "F",
    n_max: int | None = None,
) -> dict[str, Any]:
    """Load molecular dataset from .npz file.

    Handles both single-molecule (rMD17 format: R is (N,natoms,3)) and
    multi-molecule (QM7b format: R is object array of variable-sized arrays).

    Args:
        path: Path to .npz file
        R_key: Key for coordinates in npz file (default "R")
        Z_key: Key for atomic numbers in npz file (default "z")
        E_key: Key for energies in npz file (default "E")
        F_key: Key for forces in npz file (default "F")
        n_max: Maximum number of structures to load (None = all)

    Returns:
        Dictionary with standardized keys:
            - "R": list of (natoms_i, 3) arrays
            - "Z": list of (natoms_i,) int32 arrays
            - "E": (n,) array of energies
            - "F": list of (natoms_i, 3) arrays (optional, only if F_key exists)

    Example:
        >>> data = load_data("ethanol.npz")
        >>> R, Z, E = data["R"], data["Z"], data["E"]
        >>> print(len(R), R[0].shape)  # e.g. 1000 (9, 3)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    npz = np.load(path, allow_pickle=True)

    if R_key not in npz:
        raise ValueError(f"Key '{R_key}' not found in {path}. Available keys: {list(npz.keys())}")

    R_raw = npz[R_key]
    Z_raw = npz[Z_key]
    E_raw = npz[E_key] if E_key in npz else None
    F_raw = npz.get(F_key, None)

    # Determine format: single-molecule (all same atoms) or multi-molecule (variable atoms)
    if R_raw.ndim == 3 and R_raw.dtype != object:
        # rMD17 format: R (N, natoms, 3), z (natoms,), E (N,), F (N, natoms, 3)
        n = len(R_raw) if n_max is None else min(n_max, len(R_raw))
        R_list = [R_raw[i].astype(np.float64) for i in range(n)]
        Z_list = [Z_raw.astype(np.int32) for _ in range(n)]
        E = E_raw[:n].astype(np.float64) if E_raw is not None else None
        F_list = [F_raw[i].astype(np.float64) for i in range(n)] if F_raw is not None else None

    elif R_raw.dtype == object:
        # QM7b format: R (N,) object array, z (N,) object array
        n = len(R_raw) if n_max is None else min(n_max, len(R_raw))
        R_list = [np.asarray(R_raw[i], dtype=np.float64) for i in range(n)]
        Z_list = [np.asarray(Z_raw[i], dtype=np.int32) for i in range(n)]
        E = E_raw[:n].astype(np.float64) if E_raw is not None else None
        F_list = (
            [np.asarray(F_raw[i], dtype=np.float64) for i in range(n)]
            if F_raw is not None
            else None
        )

    else:
        raise ValueError(f"Unsupported R array format: shape={R_raw.shape}, dtype={R_raw.dtype}")

    result: dict[str, Any] = {"R": R_list, "Z": Z_list}
    if E is not None:
        result["E"] = E
    if F_list is not None:
        result["F"] = F_list

    return result


def load_model(path: str | Path) -> "BaseModel":
    """Load a KernelForge model from .npz file.

    Automatically dispatches to the correct model class based on saved metadata.

    Args:
        path: Path to saved model .npz file

    Returns:
        Loaded model instance (subclass of BaseModel)

    Example:
        >>> model = load_model("ethanol_model.npz")
        >>> E_pred = model.predict(R_test, Z_test)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    data = np.load(path, allow_pickle=True)

    if "_model_class" not in data:
        raise ValueError(f"Not a valid KernelForge model file: {path} (missing _model_class)")

    cls_name = str(data["_model_class"])
    if cls_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class '{cls_name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )

    cls = _MODEL_REGISTRY[cls_name]
    return cls._load_from_npz(data)


# ============================================================================
# Base Model Abstract Class
# ============================================================================


class BaseModel(ABC):
    """Abstract base class for all KernelForge models.

    All models provide:
    - fit(R, Z, Y, F=None): Train on molecular structures and labels
    - predict(R, Z): Predict energies for new structures
    - predict_forces(R, Z): Predict forces (GDML/RFF models only)
    - evaluate(...): Train/test split evaluation with statistics
    - cross_validate(...): k-fold cross-validation
    - optimize(...): Grid search over hyperparameters
    - save(path): Save model to .npz file
    - load(path): Load model from .npz file

    Subclasses must implement:
    - _fit_impl(R, Z, Y, F, seed)
    - _predict_impl(R, Z)
    - _predict_forces_impl(R, Z) (can raise NotImplementedError)
    - _get_params() -> dict
    - _set_params(params: dict)
    - _save_arrays() -> dict
    - _load_arrays(data: dict)
    """

    def __init__(self) -> None:
        """Initialize base model. Subclasses should call super().__init__()."""
        self._is_fitted = False

    @abstractmethod
    def _fit_impl(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
        Y: NDArray[np.float64],
        F: list[NDArray[np.float64]] | None,
        seed: int | None,
    ) -> None:
        """Internal fit implementation. Sets self._is_fitted = True on success."""
        ...

    @abstractmethod
    def _predict_impl(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
    ) -> NDArray[np.float64]:
        """Internal predict implementation. Returns (n,) energy array."""
        ...

    def _predict_forces_impl(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
    ) -> list[NDArray[np.float64]]:
        """Internal force prediction. Returns list of (natoms_i, 3) arrays.

        Default implementation raises NotImplementedError. Override in GDML/RFF models.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support force prediction. "
            "Use a GDML or RFF model with use_forces=True."
        )

    @abstractmethod
    def _get_params(self) -> dict[str, Any]:
        """Return current hyperparameters as JSON-serializable dict."""
        ...

    @abstractmethod
    def _set_params(self, params: dict[str, Any]) -> None:
        """Set hyperparameters from dict."""
        ...

    @abstractmethod
    def _save_arrays(self) -> dict[str, NDArray]:
        """Return dict of arrays to save (e.g., alpha, X_train, W, b)."""
        ...

    @abstractmethod
    def _load_arrays(self, data: dict[str, Any]) -> None:
        """Load arrays from npz data dict."""
        ...

    # ========================================================================
    # Public API
    # ========================================================================

    def fit(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
        Y: NDArray[np.float64],
        F: list[NDArray[np.float64]] | None = None,
        *,
        seed: int | None = None,
    ) -> "BaseModel":
        """Train the model on molecular structures and labels.

        Args:
            R: List of (natoms_i, 3) coordinate arrays (Angstrom)
            Z: List of (natoms_i,) atomic number arrays (int32)
            Y: (n,) array of energy labels
            F: Optional list of (natoms_i, 3) force arrays (for GDML/RFF models)
            seed: Random seed for reproducibility (if needed)

        Returns:
            self (for chaining)

        Example:
            >>> model = LocalGaussianKRR(sigma=2.0, regularize=1e-6)
            >>> model.fit(R_train, Z_train, E_train)
        """
        self._validate_inputs(R, Z, Y, F)
        self._fit_impl(R, Z, Y, F, seed)
        self._is_fitted = True
        return self

    def predict(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
    ) -> NDArray[np.float64]:
        """Predict energies for new molecular structures.

        Args:
            R: List of (natoms_i, 3) coordinate arrays
            Z: List of (natoms_i,) atomic number arrays

        Returns:
            (n,) array of predicted energies

        Example:
            >>> E_pred = model.predict(R_test, Z_test)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
        self._validate_inputs(R, Z, None, None)
        return self._predict_impl(R, Z)

    def predict_forces(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
    ) -> list[NDArray[np.float64]]:
        """Predict forces for new molecular structures.

        Args:
            R: List of (natoms_i, 3) coordinate arrays
            Z: List of (natoms_i,) atomic number arrays

        Returns:
            List of (natoms_i, 3) force arrays

        Raises:
            NotImplementedError: If model does not support force prediction

        Example:
            >>> F_pred = model.predict_forces(R_test, Z_test)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
        self._validate_inputs(R, Z, None, None)
        return self._predict_forces_impl(R, Z)

    def evaluate(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
        Y: NDArray[np.float64],
        F: list[NDArray[np.float64]] | None = None,
        *,
        n_test: int | None = None,
        test_fraction: float = 0.2,
        seed: int | None = 0,
    ) -> dict[str, Any]:
        """Evaluate model on a train/test split.

        Trains on training split, evaluates on test split, prints statistics table.

        Args:
            R, Z, Y, F: Full dataset
            n_test: Explicit number of test structures (overrides test_fraction)
            test_fraction: Fraction of data for testing (default 0.2)
            seed: Random seed for split reproducibility

        Returns:
            Dictionary with statistics (MAE, RMSE, R2, etc.)

        Example:
            >>> stats = model.evaluate(R, Z, E, F, n_test=200, seed=42)
        """
        n_total = len(R)
        if n_test is None:
            n_test = int(n_total * test_fraction)
        n_train = n_total - n_test

        if n_train < 1 or n_test < 1:
            raise ValueError(f"Invalid split: n_train={n_train}, n_test={n_test}")

        # Random split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        R_train = [R[i] for i in train_idx]
        Z_train = [Z[i] for i in train_idx]
        Y_train = Y[train_idx]
        F_train = [F[i] for i in train_idx] if F is not None else None

        R_test = [R[i] for i in test_idx]
        Z_test = [Z[i] for i in test_idx]
        Y_test = Y[test_idx]
        F_test = [F[i] for i in test_idx] if F is not None else None

        # Train and predict
        self.fit(R_train, Z_train, Y_train, F_train, seed=seed)
        Y_pred = self.predict(R_test, Z_test)

        # Compute energy statistics
        stats = self._compute_statistics(Y_test, Y_pred, prefix="energy")
        stats["n_train"] = n_train
        stats["n_test"] = n_test

        # Force statistics if available
        if F_test is not None:
            try:
                F_pred = self.predict_forces(R_test, Z_test)
                F_test_flat = np.concatenate([f.ravel() for f in F_test])
                F_pred_flat = np.concatenate([f.ravel() for f in F_pred])
                force_stats = self._compute_statistics(F_test_flat, F_pred_flat, prefix="force")
                stats.update(force_stats)
            except NotImplementedError:
                pass  # Model doesn't support forces

        # Print table
        self._print_stats_table(stats)

        return stats

    def cross_validate(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
        Y: NDArray[np.float64],
        F: list[NDArray[np.float64]] | None = None,
        *,
        k: int = 5,
        seed: int | None = 0,
    ) -> dict[str, Any]:
        """Perform k-fold cross-validation.

        Args:
            R, Z, Y, F: Full dataset
            k: Number of folds
            seed: Random seed

        Returns:
            Dictionary with per-fold and aggregate statistics

        Example:
            >>> cv_stats = model.cross_validate(R, Z, E, k=5, seed=42)
        """
        n_total = len(R)
        if k < 2 or k > n_total:
            raise ValueError(f"Invalid k={k} for n={n_total}")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total)
        fold_size = n_total // k

        fold_stats = []

        print(f"\n{'=' * 70}")
        print(f"  {k}-Fold Cross-Validation: {self.__class__.__name__}")
        print(f"{'=' * 70}\n")

        iterator = range(k)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="CV Folds", unit="fold")

        for fold in iterator:
            # Create fold split
            start = fold * fold_size
            end = start + fold_size if fold < k - 1 else n_total
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            R_train = [R[i] for i in train_idx]
            Z_train = [Z[i] for i in train_idx]
            Y_train = Y[train_idx]
            F_train = [F[i] for i in train_idx] if F is not None else None

            R_test = [R[i] for i in test_idx]
            Z_test = [Z[i] for i in test_idx]
            Y_test = Y[test_idx]
            F_test = [F[i] for i in test_idx] if F is not None else None

            # Train and predict
            self.fit(R_train, Z_train, Y_train, F_train, seed=seed)
            Y_pred = self.predict(R_test, Z_test)

            stats = self._compute_statistics(Y_test, Y_pred, prefix="energy")

            if F_test is not None:
                try:
                    F_pred = self.predict_forces(R_test, Z_test)
                    F_test_flat = np.concatenate([f.ravel() for f in F_test])
                    F_pred_flat = np.concatenate([f.ravel() for f in F_pred])
                    force_stats = self._compute_statistics(F_test_flat, F_pred_flat, prefix="force")
                    stats.update(force_stats)
                except NotImplementedError:
                    pass

            fold_stats.append(stats)

        # Aggregate statistics
        agg_stats = self._aggregate_fold_stats(fold_stats)
        self._print_cv_summary(agg_stats, k)

        return {"folds": fold_stats, "aggregate": agg_stats}

    def optimize(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
        Y: NDArray[np.float64],
        F: list[NDArray[np.float64]] | None = None,
        *,
        sigma_list: list[float],
        regularize_list: list[float],
        k: int = 5,
        seed: int | None = 0,
        metric: str = "energy_mae",
    ) -> "BaseModel":
        """Optimize hyperparameters via grid search with k-fold CV.

        Evaluates all (sigma, regularize) combinations, selects the best based on
        the specified metric, sets the model hyperparameters to best values, and
        trains on the full dataset.

        Args:
            R, Z, Y, F: Full dataset
            sigma_list: List of sigma values to try
            regularize_list: List of regularization values to try
            k: Number of CV folds
            seed: Random seed
            metric: Metric to minimize (default "energy_mae")

        Returns:
            self (fitted with best hyperparameters)

        Example:
            >>> model.optimize(R, Z, E,
            ...     sigma_list=[0.5, 1.0, 2.0],
            ...     regularize_list=[1e-8, 1e-6, 1e-4],
            ...     k=5)
        """
        print(f"\n{'=' * 70}")
        print(f"  Hyperparameter Optimization: {self.__class__.__name__}")
        print(f"  Grid: sigma={sigma_list}, regularize={regularize_list}")
        print(f"  Metric: {metric} (lower is better)")
        print(f"{'=' * 70}\n")

        best_score = np.inf
        best_params = None
        grid_results = []

        total_combos = len(sigma_list) * len(regularize_list)
        iterator = [(s, r) for s in sigma_list for r in regularize_list]

        if tqdm is not None:
            iterator = tqdm(iterator, desc="Grid Search", total=total_combos, unit="combo")

        for sigma, regularize in iterator:
            # Set params and run CV
            old_params = self._get_params()
            self._set_params({"sigma": sigma, "regularize": regularize, **old_params})

            cv_result = self.cross_validate(R, Z, Y, F, k=k, seed=seed)
            score = cv_result["aggregate"][f"{metric}_mean"]

            grid_results.append({"sigma": sigma, "regularize": regularize, metric: score})

            if score < best_score:
                best_score = score
                best_params = {"sigma": sigma, "regularize": regularize}

            if tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"best": f"{best_score:.4f}"})  # type: ignore

        # Print grid summary
        print(f"\n{'=' * 70}")
        print(f"  Grid Search Results (metric: {metric})")
        print(f"{'=' * 70}\n")
        print(f"  {'sigma':<10} {'regularize':<15} {metric:<15}")
        print(f"  {'-' * 10} {'-' * 15} {'-' * 15}")
        for res in grid_results:
            marker = (
                " *"
                if best_params
                and (res["sigma"], res["regularize"])
                == (best_params["sigma"], best_params["regularize"])
                else "  "
            )
            print(f"{marker}{res['sigma']:<10.2e} {res['regularize']:<15.2e} {res[metric]:<15.6f}")

        if best_params:
            print(
                f"\n  Best parameters: sigma={best_params['sigma']:.2e}, "
                f"regularize={best_params['regularize']:.2e}"
            )
        else:
            print("\n  No valid parameters found")
        print(f"  Best {metric}: {best_score:.6f}\n")
        print(f"{'=' * 70}\n")

        # Set best params and fit on full dataset
        if best_params:
            old_params = self._get_params()
            self._set_params({**old_params, **best_params})
            self.fit(R, Z, Y, F, seed=seed)
        else:
            raise ValueError("Grid search failed to find valid parameters")

        return self

    def save(self, path: str | Path) -> None:
        """Save model to .npz file.

        Args:
            path: Output path (will add .npz if missing)

        Example:
            >>> model.save("my_model.npz")
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model. Call .fit() first.")

        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        # Metadata
        metadata = {
            "_model_class": self.__class__.__name__,
            "_kf_version": __version__,
            "_params": json.dumps(self._get_params()),
        }

        # Model-specific arrays
        arrays = self._save_arrays()

        # Combine and save
        np.savez_compressed(path, **metadata, **arrays)

    @classmethod
    def load(cls: type[TModel], path: str | Path) -> TModel:
        """Load model from .npz file (class-specific).

        For automatic dispatch, use load_model(path) instead.

        Args:
            path: Path to saved model

        Returns:
            Loaded model instance

        Example:
            >>> model = LocalGaussianKRR.load("my_model.npz")
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        saved_cls = str(data["_model_class"])
        if saved_cls != cls.__name__:
            raise ValueError(
                f"Model class mismatch: file contains '{saved_cls}', "
                f"but called {cls.__name__}.load()"
            )

        return cls._load_from_npz(data)

    @classmethod
    def _load_from_npz(cls: type[TModel], data: dict[str, Any]) -> TModel:
        """Internal: create instance from npz data dict."""
        params = json.loads(str(data["_params"]))
        instance = cls(**params)
        instance._load_arrays(data)
        instance._is_fitted = True
        return instance

    # ========================================================================
    # Internal Utilities
    # ========================================================================

    def _validate_inputs(
        self,
        R: list[NDArray[np.float64]],
        Z: list[NDArray[np.int32]],
        Y: NDArray[np.float64] | None,
        F: list[NDArray[np.float64]] | None,
    ) -> None:
        """Validate input arrays for fit/predict."""
        if len(R) != len(Z):
            raise ValueError(f"R and Z must have same length: {len(R)} vs {len(Z)}")

        if Y is not None and len(Y) != len(R):
            raise ValueError(f"Y length {len(Y)} doesn't match R length {len(R)}")

        if F is not None and len(F) != len(R):
            raise ValueError(f"F length {len(F)} doesn't match R length {len(R)}")

        for i, (r, z) in enumerate(zip(R, Z)):
            if r.shape[0] != len(z):
                raise ValueError(
                    f"Structure {i}: R shape {r.shape} doesn't match Z length {len(z)}"
                )
            if r.ndim != 2 or r.shape[1] != 3:
                raise ValueError(f"Structure {i}: R must be (natoms, 3), got {r.shape}")

            if F is not None and F[i].shape != r.shape:
                raise ValueError(
                    f"Structure {i}: F shape {F[i].shape} doesn't match R shape {r.shape}"
                )

    def _compute_statistics(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        prefix: str = "",
    ) -> dict[str, float]:
        """Compute MAE, RMSE, R², max error for predictions."""
        errors = y_true - y_pred
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_err = float(np.max(np.abs(errors)))

        # R² score
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        prefix = f"{prefix}_" if prefix else ""
        return {
            f"{prefix}mae": mae,
            f"{prefix}rmse": rmse,
            f"{prefix}r2": r2,
            f"{prefix}max_error": max_err,
        }

    def _aggregate_fold_stats(self, fold_stats: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate statistics across CV folds (mean ± std)."""
        agg = {}
        keys = fold_stats[0].keys()
        for key in keys:
            values = [s[key] for s in fold_stats]
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        return agg

    def _print_stats_table(self, stats: dict[str, Any]) -> None:
        """Print statistics in a formatted table."""
        params = self._get_params()
        sigma = params.get("sigma", "N/A")
        reg = params.get("regularize", "N/A")

        print(f"\n{'=' * 70}")
        print(f"  {self.__class__.__name__}  |  sigma={sigma}  |  reg={reg}")
        print(f"{'=' * 70}")
        print(f"  {'Metric':<20} {'Value':<30}")
        print(f"  {'-' * 20} {'-' * 30}")

        # Always show split info
        if "n_train" in stats:
            print(f"  {'Split':<20} N_train={stats['n_train']}, N_test={stats['n_test']}")

        # Energy metrics
        if "energy_mae" in stats:
            print(f"  {'Energy MAE':<20} {stats['energy_mae']:.6f}")
        if "energy_rmse" in stats:
            print(f"  {'Energy RMSE':<20} {stats['energy_rmse']:.6f}")
        if "energy_r2" in stats:
            print(f"  {'Energy R²':<20} {stats['energy_r2']:.6f}")
        if "energy_max_error" in stats:
            print(f"  {'Energy Max Error':<20} {stats['energy_max_error']:.6f}")

        # Force metrics
        if "force_mae" in stats:
            print(f"  {'Force MAE':<20} {stats['force_mae']:.6f}")
        if "force_rmse" in stats:
            print(f"  {'Force RMSE':<20} {stats['force_rmse']:.6f}")
        if "force_r2" in stats:
            print(f"  {'Force R²':<20} {stats['force_r2']:.6f}")

        print(f"{'=' * 70}\n")

    def _print_cv_summary(self, agg_stats: dict[str, float], k: int) -> None:
        """Print k-fold CV summary table."""
        print(f"\n{'=' * 70}")
        print(f"  {k}-Fold Cross-Validation Summary")
        print(f"{'=' * 70}")
        print(f"  {'Metric':<25} {'Mean':<15} {'Std':<15}")
        print(f"  {'-' * 25} {'-' * 15} {'-' * 15}")

        for key in sorted(agg_stats.keys()):
            if key.endswith("_mean"):
                base = key[:-5]
                mean_val = agg_stats[key]
                std_val = agg_stats.get(f"{base}_std", 0.0)
                print(f"  {base:<25} {mean_val:<15.6f} {std_val:<15.6f}")

        print(f"{'=' * 70}\n")


# ============================================================================
# Concrete Model Implementations
# ============================================================================


@_register_model
class LocalGaussianKRR(BaseModel):
    """Energy-only KRR using local FCHL19 Gaussian kernel.

    Uses atom-wise matching with padded molecular representations.
    Suitable for datasets with variable atom counts (QM7b, QM9).

    Parameters:
        sigma: Kernel bandwidth (positive float)
        regularize: L2 regularization strength
        elements: List of atomic numbers present in dataset (default [1,6,7,8,16,17])
        nRs2, nRs3, nFourier: FCHL19 basis set sizes
        eta2, eta3, zeta, rcut, acut: FCHL19 hyperparameters
        two_body_decay, three_body_decay, three_body_weight: FCHL19 decay parameters

    Example:
        >>> model = LocalGaussianKRR(sigma=2.0, regularize=1e-6)
        >>> model.fit(R_train, Z_train, E_train)
        >>> E_pred = model.predict(R_test, Z_test)
    """

    def __init__(
        self,
        sigma: float = 1.0,
        regularize: float = 1e-6,
        elements: list[int] | None = None,
        nRs2: int = 24,
        nRs3: int = 20,
        nFourier: int = 1,
        eta2: float = 0.32,
        eta3: float = 2.7,
        zeta: float = np.pi,
        rcut: float = 8.0,
        acut: float = 8.0,
        two_body_decay: float = 1.8,
        three_body_decay: float = 0.57,
        three_body_weight: float = 13.4,
    ):
        super().__init__()
        self.sigma = sigma
        self.regularize = regularize
        self.elements = elements if elements is not None else [1, 6, 7, 8, 16, 17]
        self.nRs2 = nRs2
        self.nRs3 = nRs3
        self.nFourier = nFourier
        self.eta2 = eta2
        self.eta3 = eta3
        self.zeta = zeta
        self.rcut = rcut
        self.acut = acut
        self.two_body_decay = two_body_decay
        self.three_body_decay = three_body_decay
        self.three_body_weight = three_body_weight

    def _compute_representations(
        self, R: list[NDArray[np.float64]], Z: list[NDArray[np.int32]]
    ) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
        """Compute padded FCHL19 representations."""
        n = len(R)
        X_list = []
        for r, z in zip(R, Z):
            rep = generate_fchl_acsf(
                r,
                z,
                elements=self.elements,
                nRs2=self.nRs2,
                nRs3=self.nRs3,
                nFourier=self.nFourier,
                eta2=self.eta2,
                eta3=self.eta3,
                zeta=self.zeta,
                rcut=self.rcut,
                acut=self.acut,
                two_body_decay=self.two_body_decay,
                three_body_decay=self.three_body_decay,
                three_body_weight=self.three_body_weight,
            )
            X_list.append(rep)

        # Padding
        N = np.array([len(z) for z in Z], dtype=np.int32)
        max_atoms = int(np.max(N))
        rep_size = X_list[0].shape[1]

        X = np.zeros((n, max_atoms, rep_size), dtype=np.float64)
        Q = np.zeros((n, max_atoms), dtype=np.int32)

        for i, (x_i, z_i) in enumerate(zip(X_list, Z)):
            natoms = len(z_i)
            X[i, :natoms, :] = x_i
            Q[i, :natoms] = z_i

        return X, Q, N

    def _fit_impl(self, R, Z, Y, F, seed):
        if F is not None:
            warnings.warn("LocalGaussianKRR does not use forces (F is ignored)")

        X, Q, N = self._compute_representations(R, Z)

        # Compute kernel
        K = local_kernels.kernel_gaussian_symm(X, Q, N, self.sigma)

        # Solve
        self._alpha = kernelmath.solve_cholesky(K, Y, regularize=self.regularize)

        # Save training data
        self._X_train = X
        self._Q_train = Q
        self._N_train = N

    def _predict_impl(self, R, Z):
        X, Q, N = self._compute_representations(R, Z)

        K_test = local_kernels.kernel_gaussian(
            self._X_train, X, self._Q_train, Q, self._N_train, N, self.sigma
        )

        return K_test.T @ self._alpha

    def _get_params(self):
        return {
            "sigma": self.sigma,
            "regularize": self.regularize,
            "elements": self.elements,
            "nRs2": self.nRs2,
            "nRs3": self.nRs3,
            "nFourier": self.nFourier,
            "eta2": self.eta2,
            "eta3": self.eta3,
            "zeta": self.zeta,
            "rcut": self.rcut,
            "acut": self.acut,
            "two_body_decay": self.two_body_decay,
            "three_body_decay": self.three_body_decay,
            "three_body_weight": self.three_body_weight,
        }

    def _set_params(self, params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _save_arrays(self):
        return {
            "alpha": self._alpha,
            "X_train": self._X_train,
            "Q_train": self._Q_train,
            "N_train": self._N_train,
        }

    def _load_arrays(self, data):
        self._alpha = data["alpha"]
        self._X_train = data["X_train"]
        self._Q_train = data["Q_train"]
        self._N_train = data["N_train"]


@_register_model
class GlobalGaussianKRR(BaseModel):
    """Energy-only KRR using global Gaussian kernel.

    Uses inverse-distance descriptors or user-provided pre-computed features.
    Suitable for fixed-composition systems (e.g., same molecule, different conformations).

    Parameters:
        sigma: Kernel bandwidth
        regularize: L2 regularization strength
        descriptor: "invdist" (auto-compute) or "precomputed" (user provides X)

    Example:
        >>> model = GlobalGaussianKRR(sigma=1.5, regularize=1e-6, descriptor="invdist")
        >>> model.fit(R_train, Z_train, E_train)
        >>> E_pred = model.predict(R_test, Z_test)
    """

    def __init__(
        self,
        sigma: float = 1.0,
        regularize: float = 1e-6,
        descriptor: Literal["invdist", "precomputed"] = "invdist",
    ):
        super().__init__()
        self.sigma = sigma
        self.regularize = regularize
        self.descriptor = descriptor

    def _compute_descriptors(self, R: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Compute inverse-distance descriptors."""
        if self.descriptor == "precomputed":
            raise RuntimeError(
                "descriptor='precomputed' requires passing X explicitly to fit/predict"
            )

        X_list = []
        for r in R:
            x = invdist_repr.inverse_distance_upper(r)
            X_list.append(x)

        return np.asarray(X_list, dtype=np.float64)

    def _fit_impl(self, R, Z, Y, F, seed):
        if F is not None:
            warnings.warn("GlobalGaussianKRR does not use forces (F is ignored)")

        X = self._compute_descriptors(R)
        alpha = -1.0 / (2.0 * self.sigma**2)

        K = global_kernels.kernel_gaussian_symm(X, alpha)
        self._alpha = kernelmath.solve_cholesky(K, Y, regularize=self.regularize)
        self._X_train = X

    def _predict_impl(self, R, Z):
        X = self._compute_descriptors(R)
        alpha = -1.0 / (2.0 * self.sigma**2)

        K_test = global_kernels.kernel_gaussian(self._X_train, X, alpha)
        return K_test.T @ self._alpha

    def _get_params(self):
        return {
            "sigma": self.sigma,
            "regularize": self.regularize,
            "descriptor": self.descriptor,
        }

    def _set_params(self, params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _save_arrays(self):
        return {"alpha": self._alpha, "X_train": self._X_train}

    def _load_arrays(self, data):
        self._alpha = data["alpha"]
        self._X_train = data["X_train"]


# Placeholder stubs for the other 3 models (to be implemented next)
@_register_model
class LocalGDML(BaseModel):
    """Energy+Force GDML using local FCHL19 Hessian kernel.

    TODO: Full implementation coming next
    """

    def __init__(self, sigma: float = 1.0, regularize: float = 1e-6, use_forces: bool = True):
        super().__init__()
        self.sigma = sigma
        self.regularize = regularize
        self.use_forces = use_forces
        raise NotImplementedError("LocalGDML not yet implemented")

    def _fit_impl(self, R, Z, Y, F, seed): ...
    def _predict_impl(self, R, Z): ...
    def _get_params(self): ...
    def _set_params(self, params): ...
    def _save_arrays(self): ...
    def _load_arrays(self, data): ...


@_register_model
class GlobalGDML(BaseModel):
    """Energy+Force GDML using global Gaussian Hessian kernel.

    TODO: Full implementation coming next
    """

    def __init__(self, sigma: float = 1.0, regularize: float = 1e-6, use_forces: bool = True):
        super().__init__()
        self.sigma = sigma
        self.regularize = regularize
        self.use_forces = use_forces
        raise NotImplementedError("GlobalGDML not yet implemented")

    def _fit_impl(self, R, Z, Y, F, seed): ...
    def _predict_impl(self, R, Z): ...
    def _get_params(self): ...
    def _set_params(self, params): ...
    def _save_arrays(self): ...
    def _load_arrays(self, data): ...


@_register_model
class ElementalRFF(BaseModel):
    """Energy+Force RFF using Random Fourier Features.

    TODO: Full implementation coming next
    """

    def __init__(
        self,
        sigma: float = 1.0,
        regularize: float = 1e-6,
        n_features: int = 1000,
        use_forces: bool = True,
    ):
        super().__init__()
        self.sigma = sigma
        self.regularize = regularize
        self.n_features = n_features
        self.use_forces = use_forces
        raise NotImplementedError("ElementalRFF not yet implemented")

    def _fit_impl(self, R, Z, Y, F, seed): ...
    def _predict_impl(self, R, Z): ...
    def _get_params(self): ...
    def _set_params(self, params): ...
    def _save_arrays(self): ...
    def _load_arrays(self, data): ...
