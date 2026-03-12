"""Tests for LocalKRRModel (FCHL19-based KRR)."""

from __future__ import annotations

import numpy as np
import pytest

from kernelforge.models import LocalKRRModel

# ---------------------------------------------------------------------------
# Shared tiny dataset (ethanol-like: 9 atoms H2C-OH)
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(0)
N_ATOMS = 9
ELEMENTS = [1, 6, 8]  # H, C, O


def _make_dataset(
    n_mols: int = 30,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Generate a random dataset with ethanol-like geometry."""
    z = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)  # C2H5OH
    coords_list = [RNG.standard_normal((N_ATOMS, 3)) for _ in range(n_mols)]
    z_list = [z] * n_mols
    energies = RNG.standard_normal(n_mols)
    forces = RNG.standard_normal((n_mols, N_ATOMS * 3))  # (n_mols, naq)
    return coords_list, z_list, energies, forces


@pytest.fixture(scope="module")
def dataset() -> tuple[list, list, np.ndarray, np.ndarray]:
    return _make_dataset(30)


# ---------------------------------------------------------------------------
# Tests: energy_only mode
# ---------------------------------------------------------------------------
class TestLocalKRRModelEnergyOnly:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalKRRModel(sigma=10.0, l2=1e-6, elements=ELEMENTS)
        model.fit(tr, ztr, energies=energies[:20])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (10,)
        # energy_only mode: forces predicted via Jacobian kernel, flat (n_test*N_ATOMS*3,)
        assert F_pred.shape == (10 * N_ATOMS * 3,)

    def test_training_mode_inferred(self, dataset: tuple) -> None:
        coords_list, z_list, energies, _ = dataset
        model = LocalKRRModel(sigma=10.0, l2=1e-6, elements=ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], energies=energies[:10])
        assert model.training_mode_ == "energy_only"

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalKRRModel(sigma=10.0, l2=1e-6, elements=ELEMENTS)
        model.fit(tr, ztr, energies=energies[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "model_krr_eo.npz"
        model.save(path)

        loaded = LocalKRRModel.load(path)
        E_load, F_load = loaded.predict(te, zte)

        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)

    def test_element_energies_fitted(self, dataset: tuple) -> None:
        """Per-element energy baseline should be fitted and have the right shape."""
        coords_list, z_list, energies, _ = dataset
        model = LocalKRRModel(sigma=10.0, l2=1e-6, elements=ELEMENTS)
        model.fit(coords_list[:20], z_list[:20], energies=energies[:20])
        # baseline_elements_ should be sorted unique elements in z_list
        unique_z = sorted({int(zi) for z in z_list[:20] for zi in z})
        assert list(model.baseline_elements_) == unique_z
        assert model.element_energies_.shape == (len(unique_z),)


# ---------------------------------------------------------------------------
# Tests: force_only mode
# ---------------------------------------------------------------------------
class TestLocalKRRModelForceOnly:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, _, forces = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS)
        model.fit(tr, ztr, forces=forces[:20])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (10,)
        assert F_pred.shape == (10 * N_ATOMS * 3,)

    def test_training_mode_inferred(self, dataset: tuple) -> None:
        coords_list, z_list, _, forces = dataset
        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], forces=forces[:10])
        assert model.training_mode_ == "force_only"

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, _, forces = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS)
        model.fit(tr, ztr, forces=forces[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "model_krr_fo.npz"
        model.save(path)
        loaded = LocalKRRModel.load(path)
        E_load, F_load = loaded.predict(te, zte)

        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: energy_and_force mode
# ---------------------------------------------------------------------------
class TestLocalKRRModelEnergyAndForce:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, forces = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS)
        model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (10,)
        assert F_pred.shape == (10 * N_ATOMS * 3,)

    def test_training_mode_inferred(self, dataset: tuple) -> None:
        coords_list, z_list, energies, forces = dataset
        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], energies=energies[:10], forces=forces[:10])
        assert model.training_mode_ == "energy_and_force"

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, energies, forces = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS)
        model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "model_krr_ef.npz"
        model.save(path)
        loaded = LocalKRRModel.load(path)
        E_load, F_load = loaded.predict(te, zte)

        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)

    def test_forces_3d_input(self, dataset: tuple) -> None:
        """Accept (n_mols, n_atoms, 3) force arrays."""
        coords_list, z_list, energies, forces = dataset
        forces_3d = forces[:10].reshape(10, N_ATOMS, 3)
        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], energies=energies[:10], forces=forces_3d)
        E_pred, F_pred = model.predict(coords_list[10:12], z_list[10:12])
        assert E_pred.shape == (2,)
        assert F_pred.shape == (2 * N_ATOMS * 3,)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------
class TestLocalKRRModelErrors:
    def test_predict_before_fit_raises(self) -> None:
        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS)
        coords = [np.zeros((N_ATOMS, 3))]
        z = [np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)]
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(coords, z)

    def test_no_labels_raises(self, dataset: tuple) -> None:
        coords_list, z_list, _, _ = dataset
        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS)
        with pytest.raises(ValueError, match=r"energies.*forces"):
            model.fit(coords_list[:5], z_list[:5])

    def test_save_npz_extension_added(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, energies, _ = dataset
        model = LocalKRRModel(sigma=10.0, l2=1e-6, elements=ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], energies=energies[:10])
        path = tmp_path / "no_ext"
        model.save(path)
        assert (tmp_path / "no_ext.npz").exists()


# ---------------------------------------------------------------------------
# Tests: FCHL19v2 representation
# ---------------------------------------------------------------------------
class TestLocalKRRModelFCHL19v2:
    """Smoke tests verifying the fchl19v2 representation path works end-to-end."""

    def test_energy_only_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:15], coords_list[15:20]
        ztr, zte = z_list[:15], z_list[15:20]

        model = LocalKRRModel(sigma=10.0, l2=1e-6, elements=ELEMENTS, representation="fchl19v2")
        model.fit(tr, ztr, energies=energies[:15])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.shape == (5 * N_ATOMS * 3,)

    def test_energy_and_force_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, forces = dataset
        tr, te = coords_list[:15], coords_list[15:20]
        ztr, zte = z_list[:15], z_list[15:20]

        model = LocalKRRModel(sigma=10.0, l2=1e-7, elements=ELEMENTS, representation="fchl19v2")
        model.fit(tr, ztr, energies=energies[:15], forces=forces[:15])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.shape == (5 * N_ATOMS * 3,)

    def test_repr_params_forwarded(self, dataset: tuple) -> None:
        """v2-specific repr_params (two_body_type, three_body_type) are accepted."""
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:10], coords_list[10:12]
        ztr, zte = z_list[:10], z_list[10:12]

        model = LocalKRRModel(
            sigma=10.0,
            l2=1e-6,
            elements=ELEMENTS,
            representation="fchl19v2",
            repr_params={"two_body_type": "bessel", "three_body_type": "cosine_rbar"},
        )
        model.fit(tr, ztr, energies=energies[:10])
        E_pred, _ = model.predict(te, zte)
        assert E_pred.shape == (2,)

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:15], coords_list[15:20]
        ztr, zte = z_list[:15], z_list[15:20]

        model = LocalKRRModel(sigma=10.0, l2=1e-6, elements=ELEMENTS, representation="fchl19v2")
        model.fit(tr, ztr, energies=energies[:15])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "krr_v2.npz"
        model.save(path)
        loaded = LocalKRRModel.load(path)
        assert isinstance(loaded, LocalKRRModel)
        assert loaded.representation == "fchl19v2"
        E_load, F_load = loaded.predict(te, zte)
        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)

    def test_invalid_representation_raises(self) -> None:
        with pytest.raises(ValueError, match="representation"):
            LocalKRRModel(sigma=10.0, l2=1e-6, elements=ELEMENTS, representation="unknown")

    def test_odd_fourier_element_resolved(self, dataset: tuple) -> None:
        """A6 (odd_fourier_element_resolved) works end-to-end with nRs3_minus > 0."""
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:10], coords_list[10:12]
        ztr, zte = z_list[:10], z_list[10:12]

        model = LocalKRRModel(
            sigma=10.0,
            l2=1e-6,
            elements=ELEMENTS,
            representation="fchl19v2",
            repr_params={"three_body_type": "odd_fourier_element_resolved", "nRs3_minus": 10},
        )
        model.fit(tr, ztr, energies=energies[:10])
        E_pred, _ = model.predict(te, zte)
        assert E_pred.shape == (2,)
