"""ASE Calculator wrapping a fitted KernelForge CUDA model.

Usage
-----
>>> from kernelforge import KernelForgeCalculator
>>> calc = KernelForgeCalculator(model, units="kcal/mol")
>>> atoms.calc = calc
>>> energy = atoms.get_potential_energy()   # eV
>>> forces = atoms.get_forces()             # eV/Å, shape (n_atoms, 3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from .models.cuda_local_krr import CudaLocalKRRModel
    from .models.cuda_local_rff import CudaLocalRFFModel

# ---------------------------------------------------------------------------
# Lazy ASE import — ase is optional; only raise at instantiation time
# ---------------------------------------------------------------------------
try:
    from ase.calculators.calculator import Calculator as _AseCalculator
    from ase.calculators.calculator import all_changes as _all_changes

    _ASE_AVAILABLE = True
except ImportError:

    class _AseCalculator:  # type: ignore[no-redef]
        """Stub used when ASE is not installed."""

        atoms: Any = None
        results: ClassVar[dict[str, Any]] = {}

        def __init__(self, **kwargs: object) -> None:
            pass

        def calculate(
            self,
            atoms: Any = None,  # noqa: ANN401
            properties: list[str] | None = None,
            system_changes: list[str] | None = None,
        ) -> None:
            pass

    _all_changes: list[str] = []
    _ASE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------
_KCAL_MOL_TO_EV: float | None = None


def _kcal_mol_to_eV() -> float:
    """Return the kcal/mol → eV conversion factor (computed once from ase.units)."""
    global _KCAL_MOL_TO_EV
    if _KCAL_MOL_TO_EV is None:
        import ase.units

        _KCAL_MOL_TO_EV = ase.units.kcal / ase.units.mol
    return _KCAL_MOL_TO_EV


def _get_factor(units: str) -> float:
    """Return multiplicative factor to convert *units* → eV (energy) and eV/Å (forces)."""
    if units == "eV":
        return 1.0
    if units == "kcal/mol":
        return _kcal_mol_to_eV()
    msg = f"Unknown units '{units}'. Supported values: 'kcal/mol', 'eV'."
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# KernelForgeCalculator
# ---------------------------------------------------------------------------


class KernelForgeCalculator(_AseCalculator):  # type: ignore[misc]
    """ASE Calculator backed by a fitted KernelForge CUDA model.

    Parameters
    ----------
    model:
        A fitted ``CudaLocalKRRModel`` or ``CudaLocalRFFModel`` instance.
        The model must have been trained in ``energy_and_force`` mode so that
        force predictions are available.
    units:
        Energy (and force) units used when training the model.
        ``'kcal/mol'`` (default) covers rMD17 and small_mols_mini datasets.
        ``'eV'`` disables conversion.

    Notes
    -----
    ASE always works internally in eV and eV/Å.  The calculator multiplies
    energies and forces by ``ase.units.kcal / ase.units.mol`` (≈ 0.04336)
    when ``units='kcal/mol'`` so that all ASE MD engines see correct units.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        model: CudaLocalKRRModel | CudaLocalRFFModel,
        units: str = "kcal/mol",
        **kwargs: object,
    ) -> None:
        if not _ASE_AVAILABLE:
            msg = (
                "ASE is required to use KernelForgeCalculator. "
                "Install it with:  pip install 'kernelforge[ase]'  or  pip install ase"
            )
            raise ImportError(msg)

        if not getattr(model, "is_fitted_", False):
            msg = "The model must be fitted before it can be wrapped in KernelForgeCalculator."
            raise RuntimeError(msg)

        training_mode = getattr(model, "training_mode_", None)
        if training_mode == "energy_only":
            msg = (
                "Model was trained in 'energy_only' mode and cannot predict forces. "
                "Re-train with energies and forces to use MD."
            )
            raise RuntimeError(msg)

        _AseCalculator.__init__(self, **kwargs)
        self._kf_model = model
        self._kf_factor = _get_factor(units)

    def calculate(
        self,
        atoms: Any = None,  # noqa: ANN401
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        """Compute energy and forces for *atoms* and store in ``self.results``."""
        if properties is None:
            properties = ["energy", "forces"]
        if system_changes is None:
            system_changes = list(_all_changes)

        _AseCalculator.calculate(self, atoms, properties, system_changes)

        if self.atoms is None:
            msg = "atoms is None after Calculator.calculate(); this should not happen."
            raise RuntimeError(msg)

        coords = self.atoms.get_positions().astype(np.float64)   # (n_atoms, 3)  Å
        z = self.atoms.get_atomic_numbers().astype(np.int32)     # (n_atoms,)

        want_energy = properties is not None and "energy" in properties
        E_arr, F_flat = self._kf_model.predict([coords], [z], compute_energy=want_energy)

        factor = self._kf_factor
        if want_energy:
            self.results["energy"] = float(E_arr[0]) * factor
        self.results["forces"] = F_flat.reshape(-1, 3).astype(np.float64) * factor
