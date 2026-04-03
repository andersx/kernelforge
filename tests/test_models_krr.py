"""Tests for LocalKRRModel (FCHL19-based KRR)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from kernelforge.models import LocalKRRModel

# Path to real dataset used by fast-path tests
_MINI_DATA = Path(__file__).parent.parent / "examples" / "small_mols_mini_train.npz"

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
# Fixture: real molecular data from small_mols_mini_train.npz
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mini_dataset() -> tuple[list, list, np.ndarray, list]:
    """Load a small slice of real small_mols_mini data for fast-path tests.

    Returns (coords_list, z_list, energies, forces_list) for 30 molecules.
    Molecules have variable atom counts and real FCHL19-compatible elements.
    """
    data = np.load(_MINI_DATA, allow_pickle=True)
    n = 30
    coords_list = [c.astype(np.float64) for c in data["coords"][:n]]
    z_list = [z.astype(np.int32) for z in data["nuclear_charges"][:n]]
    energies = data["energies"][:n].astype(np.float64)
    forces_list = [f.astype(np.float64) for f in data["forces"][:n]]
    return coords_list, z_list, energies, forces_list


# ---------------------------------------------------------------------------
# Tests: fast-path (J^T·alpha trick) vs slow-path (full kernel matrix)
# ---------------------------------------------------------------------------
class TestLocalKRRModelFastPath:
    """Verify that the J^T·alpha fast path gives numerically identical results
    to the slow path (full kernel matrix @ alpha), using real molecular data.

    All comparisons are made via model._predict() (baseline-subtracted space)
    so element energy baselines cancel out on both sides.
    """

    ELEMENTS: ClassVar[list[int]] = [
        1,
        6,
        7,
        8,
    ]  # H, C, N, O — covers all elements in small_mols_mini

    def _slow_force_only_predict(
        self,
        model: LocalKRRModel,
        coords_te: list,
        z_te: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """force_only prediction via full K_hess / K_j matrix build."""
        from kernelforge import local_kernels
        from kernelforge.models.representations import compute_fchl19

        X_te, dX_te, Q_te, _, N_te = compute_fchl19(
            coords_te, z_te, model.elements, with_gradients=True, repr_params=model.repr_params
        )
        assert dX_te is not None
        X_tr, dX_tr, Q_tr, N_tr = model._X_tr, model._dX_tr, model._Q_tr, model._N_tr
        assert dX_tr is not None
        alpha = model._alpha

        K_hess = local_kernels.kernel_gaussian_hessian(
            X_te,
            X_tr,
            dX_te,
            dX_tr,
            Q_te,
            Q_tr,
            N_te,
            N_tr,
            model.sigma,  # type: ignore[arg-type]
        )
        F_slow: np.ndarray = K_hess @ alpha

        K_j = local_kernels.kernel_gaussian_jacobian(
            X_te,
            X_tr,
            dX_tr,
            Q_te,
            Q_tr,
            N_te,
            N_tr,
            model.sigma,  # type: ignore[arg-type]
        )
        E_slow: np.ndarray = -(K_j @ alpha)
        return E_slow, F_slow

    def _slow_energy_and_force_predict(
        self,
        model: LocalKRRModel,
        coords_te: list,
        z_te: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """energy_and_force prediction via full K_full matrix build."""
        from kernelforge import local_kernels
        from kernelforge.models.representations import compute_fchl19

        X_te, dX_te, Q_te, _, N_te = compute_fchl19(
            coords_te, z_te, model.elements, with_gradients=True, repr_params=model.repr_params
        )
        assert dX_te is not None
        n_test = len(coords_te)
        X_tr, dX_tr, Q_tr, N_tr = model._X_tr, model._dX_tr, model._Q_tr, model._N_tr
        assert dX_tr is not None
        alpha = model._alpha

        K_full = local_kernels.kernel_gaussian_full(
            X_te,
            X_tr,
            dX_te,
            dX_tr,
            Q_te,
            Q_tr,
            N_te,
            N_tr,
            model.sigma,  # type: ignore[arg-type]
        )
        y_pred = K_full @ alpha
        E_slow: np.ndarray = y_pred[:n_test]
        F_slow: np.ndarray = -y_pred[n_test:]
        return E_slow, F_slow

    def test_force_only_fast_matches_slow(self, mini_dataset: tuple) -> None:
        """force_only: fast path (matvec) must match slow path (K@alpha) on real data."""
        coords_list, z_list, _, forces_list = mini_dataset
        tr, te = coords_list[:20], coords_list[20:25]
        ztr, zte = z_list[:20], z_list[20:25]

        model = LocalKRRModel(sigma=20.0, l2=1e-7, elements=self.ELEMENTS)
        model.fit(tr, ztr, forces=forces_list[:20])

        assert model._alpha_desc is not None
        # Compare _predict() directly: both sides are in baseline-subtracted space
        E_fast, F_fast = model._predict(te, zte)
        E_slow, F_slow = self._slow_force_only_predict(model, te, zte)

        np.testing.assert_allclose(
            E_fast, E_slow, rtol=1e-7, atol=1e-9, err_msg="force_only energies: fast vs slow"
        )
        np.testing.assert_allclose(
            F_fast, F_slow, rtol=1e-7, atol=1e-9, err_msg="force_only forces: fast vs slow"
        )

    def test_energy_and_force_fast_matches_slow(self, mini_dataset: tuple) -> None:
        """energy_and_force: fast path (matvec) must match slow path (K@alpha) on real data."""
        coords_list, z_list, energies, forces_list = mini_dataset
        tr, te = coords_list[:20], coords_list[20:25]
        ztr, zte = z_list[:20], z_list[20:25]

        model = LocalKRRModel(sigma=20.0, l2=1e-7, elements=self.ELEMENTS)
        model.fit(tr, ztr, energies=energies[:20], forces=forces_list[:20])

        assert model._alpha_desc is not None
        # Compare _predict() directly: both sides are in baseline-subtracted space
        E_fast, F_fast = model._predict(te, zte)
        E_slow, F_slow = self._slow_energy_and_force_predict(model, te, zte)

        np.testing.assert_allclose(
            E_fast, E_slow, rtol=1e-7, atol=1e-9, err_msg="energy_and_force energies: fast vs slow"
        )
        np.testing.assert_allclose(
            F_fast, F_slow, rtol=1e-7, atol=1e-9, err_msg="energy_and_force forces: fast vs slow"
        )

    def test_alpha_desc_shape_force_only(self, mini_dataset: tuple) -> None:
        """alpha_desc shape must match (n_train, max_atoms, rep_size)."""
        coords_list, z_list, _, forces_list = mini_dataset
        model = LocalKRRModel(sigma=20.0, l2=1e-7, elements=self.ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], forces=forces_list[:10])
        ad = model._alpha_desc
        assert ad is not None
        assert ad.ndim == 3
        assert ad.shape == model._X_tr.shape

    def test_alpha_desc_shape_energy_and_force(self, mini_dataset: tuple) -> None:
        """alpha_desc shape must match X_tr for energy_and_force mode."""
        coords_list, z_list, energies, forces_list = mini_dataset
        model = LocalKRRModel(sigma=20.0, l2=1e-7, elements=self.ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], energies=energies[:10], forces=forces_list[:10])
        ad = model._alpha_desc
        assert ad is not None
        assert ad.shape == model._X_tr.shape

    def test_alpha_desc_none_for_energy_only(self, mini_dataset: tuple) -> None:
        """energy_only mode must leave alpha_desc as None."""
        coords_list, z_list, energies, _ = mini_dataset
        model = LocalKRRModel(sigma=20.0, l2=1e-6, elements=self.ELEMENTS)
        model.fit(coords_list[:10], z_list[:10], energies=energies[:10])
        assert model._alpha_desc is None

    def test_save_load_preserves_alpha_desc(self, mini_dataset: tuple, tmp_path) -> None:
        """alpha_desc is saved and reloaded; loaded model still uses fast path."""
        coords_list, z_list, _, forces_list = mini_dataset
        tr, te = coords_list[:20], coords_list[20:25]
        ztr, zte = z_list[:20], z_list[20:25]

        model = LocalKRRModel(sigma=20.0, l2=1e-7, elements=self.ELEMENTS)
        model.fit(tr, ztr, forces=forces_list[:20])

        path = tmp_path / "model_fast.npz"
        model.save(path)
        loaded = LocalKRRModel.load(path)
        assert isinstance(loaded, LocalKRRModel)
        assert loaded._alpha_desc is not None, "alpha_desc must survive save/load"

        E_loaded, F_loaded = loaded.predict(te, zte)
        E_orig, F_orig = model.predict(te, zte)
        np.testing.assert_allclose(E_loaded, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_loaded, F_orig, rtol=1e-10)
