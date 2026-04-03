"""Tests for LocalRFFModel (FCHL19/FCHL19v2-based RFF)."""

from __future__ import annotations

import numpy as np
import pytest

from kernelforge.models import LocalRFFModel

RNG = np.random.default_rng(1)
N_ATOMS = 9
ELEMENTS = [1, 6, 8]


def _make_dataset(
    n_mols: int = 30,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    z = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    coords_list = [RNG.standard_normal((N_ATOMS, 3)) for _ in range(n_mols)]
    z_list = [z] * n_mols
    energies = RNG.standard_normal(n_mols)
    forces = RNG.standard_normal((n_mols, N_ATOMS * 3))
    return coords_list, z_list, energies, forces


@pytest.fixture(scope="module")
def dataset() -> tuple[list, list, np.ndarray, np.ndarray]:
    return _make_dataset(30)


# ---------------------------------------------------------------------------
# Tests: energy_only mode
# ---------------------------------------------------------------------------
class TestLocalRFFModelEnergyOnly:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, elements=ELEMENTS)
        model.fit(tr, ztr, energies=energies[:20])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (10,)
        # energy_only mode: forces predicted via gradient features, flat (n_test*N_ATOMS*3,)
        assert F_pred.shape == (10 * N_ATOMS * 3,)

    def test_training_mode_inferred(self, dataset: tuple) -> None:
        coords_list, z_list, energies, _ = dataset
        model = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, elements=ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], energies=energies[:10])
        assert model.training_mode_ == "energy_only"

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, seed=7, elements=ELEMENTS)
        model.fit(tr, ztr, energies=energies[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "rff_eo.npz"
        model.save(path)
        loaded = LocalRFFModel.load(path)
        E_load, F_load = loaded.predict(te, zte)

        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)

    def test_reproducibility(self, dataset: tuple) -> None:
        """Same seed produces identical weights and predictions."""
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model1 = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, seed=99, elements=ELEMENTS)
        model1.fit(tr, ztr, energies=energies[:20])

        model2 = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, seed=99, elements=ELEMENTS)
        model2.fit(tr, ztr, energies=energies[:20])

        E1, F1 = model1.predict(te, zte)
        E2, F2 = model2.predict(te, zte)
        # Same seed → numerically identical up to floating-point non-determinism
        # (MKL BLAS may reorder FP ops across runs); rtol=1e-6 is tight enough.
        np.testing.assert_allclose(E1, E2, rtol=1e-6, atol=0)
        np.testing.assert_allclose(F1, F2, rtol=1e-6, atol=0)


# ---------------------------------------------------------------------------
# Tests: force_only mode
# ---------------------------------------------------------------------------
class TestLocalRFFModelForceOnly:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, _, forces = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, elements=ELEMENTS)
        model.fit(tr, ztr, forces=forces[:20])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (10,)
        assert F_pred.shape == (10 * N_ATOMS * 3,)

    def test_training_mode_inferred(self, dataset: tuple) -> None:
        coords_list, z_list, _, forces = dataset
        model = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, elements=ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], forces=forces[:10])
        assert model.training_mode_ == "force_only"

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, _, forces = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, seed=3, elements=ELEMENTS)
        model.fit(tr, ztr, forces=forces[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "rff_fo.npz"
        model.save(path)
        loaded = LocalRFFModel.load(path)
        E_load, F_load = loaded.predict(te, zte)

        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: energy_and_force mode
# ---------------------------------------------------------------------------
class TestLocalRFFModelEnergyAndForce:
    def test_fit_predict_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, forces = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, elements=ELEMENTS)
        model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (10,)
        assert F_pred.shape == (10 * N_ATOMS * 3,)

    def test_training_mode_inferred(self, dataset: tuple) -> None:
        coords_list, z_list, energies, forces = dataset
        model = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, elements=ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], energies=energies[:10], forces=forces[:10])
        assert model.training_mode_ == "energy_and_force"

    def test_save_load_roundtrip(self, dataset: tuple, tmp_path) -> None:
        coords_list, z_list, energies, forces = dataset
        tr, te = coords_list[:20], coords_list[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, seed=5, elements=ELEMENTS)
        model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "rff_ef.npz"
        model.save(path)
        loaded = LocalRFFModel.load(path)
        E_load, F_load = loaded.predict(te, zte)

        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: FCHL19v2 representation
# ---------------------------------------------------------------------------
class TestLocalRFFModelFCHL19v2:
    """Smoke tests verifying the fchl19v2 representation path works end-to-end."""

    def test_energy_only_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:15], coords_list[15:20]
        ztr, zte = z_list[:15], z_list[15:20]

        model = LocalRFFModel(
            sigma=10.0, l2=1e-6, d_rff=64, elements=ELEMENTS, representation="fchl19v2"
        )
        model.fit(tr, ztr, energies=energies[:15])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.shape == (5 * N_ATOMS * 3,)

    def test_energy_and_force_shapes(self, dataset: tuple) -> None:
        coords_list, z_list, energies, forces = dataset
        tr, te = coords_list[:15], coords_list[15:20]
        ztr, zte = z_list[:15], z_list[15:20]

        model = LocalRFFModel(
            sigma=10.0, l2=1e-6, d_rff=64, elements=ELEMENTS, representation="fchl19v2"
        )
        model.fit(tr, ztr, energies=energies[:15], forces=forces[:15])

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.shape == (5 * N_ATOMS * 3,)

    def test_repr_params_forwarded(self, dataset: tuple) -> None:
        """v2-specific repr_params (two_body_type, three_body_type) are accepted."""
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:10], coords_list[10:12]
        ztr, zte = z_list[:10], z_list[10:12]

        model = LocalRFFModel(
            sigma=10.0,
            l2=1e-6,
            d_rff=64,
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

        model = LocalRFFModel(
            sigma=10.0, l2=1e-6, d_rff=64, seed=11, elements=ELEMENTS, representation="fchl19v2"
        )
        model.fit(tr, ztr, energies=energies[:15])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "rff_v2.npz"
        model.save(path)
        loaded = LocalRFFModel.load(path)
        assert isinstance(loaded, LocalRFFModel)
        assert loaded.representation == "fchl19v2"
        E_load, F_load = loaded.predict(te, zte)
        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)

    def test_invalid_representation_raises(self) -> None:
        with pytest.raises(ValueError, match="representation"):
            LocalRFFModel(sigma=10.0, l2=1e-6, d_rff=64, elements=ELEMENTS, representation="bad")

    def test_cosine_element_resolved(self, dataset: tuple) -> None:
        """A7 (cosine_element_resolved) works end-to-end with nRs3_minus > 0."""
        coords_list, z_list, energies, _ = dataset
        tr, te = coords_list[:10], coords_list[10:12]
        ztr, zte = z_list[:10], z_list[10:12]

        model = LocalRFFModel(
            sigma=10.0,
            l2=1e-6,
            d_rff=64,
            elements=ELEMENTS,
            representation="fchl19v2",
            repr_params={"three_body_type": "cosine_element_resolved", "nRs3_minus": 10},
        )
        model.fit(tr, ztr, energies=energies[:10])
        E_pred, _ = model.predict(te, zte)
        assert E_pred.shape == (2,)
