"""Tests for FCHL18KRRModel."""

from __future__ import annotations

import numpy as np
import pytest

from kernelforge.models import FCHL18KRRModel

RNG = np.random.default_rng(2)
N_ATOMS = 9
MAX_SIZE = 9

# Ethanol nuclear charges
Z_ETHANOL = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)


def _make_dataset(
    n_mols: int = 20,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Generate random ethanol-like structures."""
    coords_list = [RNG.standard_normal((N_ATOMS, 3)) + 1.5 for _ in range(n_mols)]
    z_list = [Z_ETHANOL] * n_mols
    energies = RNG.standard_normal(n_mols)
    forces = RNG.standard_normal((n_mols, N_ATOMS * 3))
    return coords_list, z_list, energies, forces


@pytest.fixture(scope="module")
def dataset() -> tuple[list, list, np.ndarray, np.ndarray]:
    return _make_dataset(20)


# ---------------------------------------------------------------------------
# Tests: energy_only mode
# ---------------------------------------------------------------------------
class TestFCHL18KRRModelEnergyOnly:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:12], coords_list[12:]
        ztr, zte = z_list[:12], z_list[12:]

        model = FCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(tr, ztr, energies=energies[:12])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (8,)
        assert F_pred.shape == (8, N_ATOMS * 3)

    def test_training_mode_inferred(self, dataset: tuple) -> None:
        coords_list, z_list, energies, _ = dataset
        model = FCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(coords_list[:8], z_list[:8], energies=energies[:8])
        assert model.training_mode_ == "energy_only"

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:12], coords_list[12:]
        ztr, zte = z_list[:12], z_list[12:]

        model = FCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(tr, ztr, energies=energies[:12])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "fchl18_eo.npz"
        model.save(path)
        loaded = FCHL18KRRModel.load(path)
        E_load, F_load = loaded.predict(te, zte)

        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: force_only mode
# ---------------------------------------------------------------------------
class TestFCHL18KRRModelForceOnly:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, _, forces = dataset
        tr, te = coords_list[:12], coords_list[12:]
        ztr, zte = z_list[:12], z_list[12:]

        # Force-only needs higher l2 for Cholesky stability
        model = FCHL18KRRModel(sigma=5.0, l2=1e-3, max_size=MAX_SIZE)
        model.fit(tr, ztr, forces=forces[:12])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (8,)
        assert F_pred.shape == (8, N_ATOMS * 3)

    def test_training_mode_inferred(self, dataset: tuple) -> None:
        coords_list, z_list, _, forces = dataset
        model = FCHL18KRRModel(sigma=5.0, l2=1e-3, max_size=MAX_SIZE)
        model.fit(coords_list[:8], z_list[:8], forces=forces[:8])
        assert model.training_mode_ == "force_only"

    def test_hessian_kernel_constraints_applied(self, dataset: tuple) -> None:
        """Force-only mode must disable use_atm and enforce cut_start >= 1.0."""
        coords_list, z_list, _, forces = dataset
        # Pass kernel_params that would be invalid for Hessian if not overridden
        kp = {"use_atm": True, "cut_start": 0.5, "cut_distance": 5.0, "fourier_order": 1}
        model = FCHL18KRRModel(sigma=5.0, l2=1e-3, max_size=MAX_SIZE, kernel_params=kp)
        # This should not raise — constraints are applied internally
        model.fit(coords_list[:8], z_list[:8], forces=forces[:8])
        assert model._kp_fit["use_atm"] is False
        assert model._kp_fit["cut_start"] >= 1.0

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, _, forces = dataset
        tr, te = coords_list[:12], coords_list[12:]
        ztr, zte = z_list[:12], z_list[12:]

        model = FCHL18KRRModel(sigma=5.0, l2=1e-3, max_size=MAX_SIZE)
        model.fit(tr, ztr, forces=forces[:12])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "fchl18_fo.npz"
        model.save(path)
        loaded = FCHL18KRRModel.load(path)
        E_load, F_load = loaded.predict(te, zte)

        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: energy_and_force mode
# ---------------------------------------------------------------------------
class TestFCHL18KRRModelEnergyAndForce:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, forces = dataset
        tr, te = coords_list[:12], coords_list[12:]
        ztr, zte = z_list[:12], z_list[12:]

        model = FCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(tr, ztr, energies=energies[:12], forces=forces[:12])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (8,)
        assert F_pred.shape == (8, N_ATOMS * 3)

    def test_training_mode_inferred(self, dataset: tuple) -> None:
        coords_list, z_list, energies, forces = dataset
        model = FCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(coords_list[:8], z_list[:8], energies=energies[:8], forces=forces[:8])
        assert model.training_mode_ == "energy_and_force"

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, energies, forces = dataset
        tr, te = coords_list[:12], coords_list[12:]
        ztr, zte = z_list[:12], z_list[12:]

        model = FCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        model.fit(tr, ztr, energies=energies[:12], forces=forces[:12])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "fchl18_ef.npz"
        model.save(path)
        loaded = FCHL18KRRModel.load(path)
        E_load, F_load = loaded.predict(te, zte)

        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------
class TestFCHL18KRRModelErrors:
    def test_predict_before_fit_raises(self) -> None:
        model = FCHL18KRRModel(sigma=5.0, l2=1e-6, max_size=MAX_SIZE)
        coords = [np.zeros((N_ATOMS, 3))]
        z = [Z_ETHANOL]
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(coords, z)
