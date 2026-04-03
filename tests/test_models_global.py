"""Tests for GlobalKRRModel and GlobalRFFModel (invdist global descriptor)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kernelforge.models import GlobalKRRModel, GlobalRFFModel

# rMD17 ethanol data (all same molecule, fixed 9-atom size — required for invdist)
_CACHE = Path.home() / ".kernelforge" / "datasets"
_RMD17_TRAIN = _CACHE / "rmd17_ethanol_train_01.npz"


def _load_ethanol(n: int = 30) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Load a small slice of rMD17 ethanol. Skip if not cached."""
    if not _RMD17_TRAIN.exists():
        pytest.skip("rMD17 ethanol data not cached; run kernelcli to populate")
    d = np.load(_RMD17_TRAIN, allow_pickle=True)
    z_fixed = d["nuclear_charges"].astype(np.int32)
    coords = [d["coords"][i].astype(np.float64) for i in range(n)]
    z_list = [z_fixed for _ in range(n)]
    energies = d["energies"][:n].astype(np.float64)
    forces = d["forces"][:n].reshape(n, -1).astype(np.float64)
    return coords, z_list, energies, forces


# ---------------------------------------------------------------------------
# GlobalKRRModel tests
# ---------------------------------------------------------------------------


class TestGlobalKRRModel:
    """Fit/predict/save/load tests for GlobalKRRModel."""

    def test_energy_only_shapes(self) -> None:
        coords, z_list, energies, _ = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalKRRModel(sigma=3.0, l2=1e-6)
        model.fit(tr, ztr, energies=energies[:20])

        assert model.training_mode_ == "energy_only"
        assert model._alpha_desc is None  # not applicable for energy_only

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.ndim == 1  # flat (5 * 27,)
        assert F_pred.shape[0] == 5 * 9 * 3

    def test_force_only_shapes(self) -> None:
        coords, z_list, _, forces = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalKRRModel(sigma=3.0, l2=1e-6)
        model.fit(tr, ztr, forces=forces[:20])

        assert model.training_mode_ == "force_only"
        assert model._alpha_desc is not None

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.shape == (5 * 9 * 3,)

    def test_energy_and_force_shapes(self) -> None:
        coords, z_list, energies, forces = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalKRRModel(sigma=3.0, l2=1e-6)
        model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])

        assert model.training_mode_ == "energy_and_force"
        assert model._alpha_desc is not None

        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.shape == (5 * 9 * 3,)

    def test_save_load_roundtrip_energy_only(self, tmp_path: Path) -> None:
        coords, z_list, energies, _ = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalKRRModel(sigma=3.0, l2=1e-6)
        model.fit(tr, ztr, energies=energies[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "global_krr_eo.npz"
        model.save(path)
        loaded = GlobalKRRModel.load(path)
        assert isinstance(loaded, GlobalKRRModel)

        E_load, F_load = loaded.predict(te, zte)
        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)

    def test_save_load_roundtrip_force_only(self, tmp_path: Path) -> None:
        coords, z_list, _, forces = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalKRRModel(sigma=3.0, l2=1e-6)
        model.fit(tr, ztr, forces=forces[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "global_krr_fo.npz"
        model.save(path)
        loaded = GlobalKRRModel.load(path)
        assert isinstance(loaded, GlobalKRRModel)
        assert loaded._alpha_desc is not None

        E_load, F_load = loaded.predict(te, zte)
        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)

    def test_save_load_roundtrip_energy_and_force(self, tmp_path: Path) -> None:
        coords, z_list, energies, forces = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalKRRModel(sigma=3.0, l2=1e-6)
        model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "global_krr_ef.npz"
        model.save(path)
        loaded = GlobalKRRModel.load(path)
        assert isinstance(loaded, GlobalKRRModel)
        assert loaded._alpha_desc is not None

        E_load, F_load = loaded.predict(te, zte)
        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)

    def test_variable_atom_count_raises(self) -> None:
        """GlobalKRRModel must reject molecules with different atom counts."""
        rng = np.random.default_rng(0)
        coords = [rng.standard_normal((5, 3)), rng.standard_normal((6, 3))]
        z_list = [np.ones(5, dtype=np.int32), np.ones(6, dtype=np.int32)]
        energies = np.array([1.0, 2.0])

        model = GlobalKRRModel(sigma=1.0, l2=1e-6)
        with pytest.raises(ValueError, match="same atom count"):
            model.fit(coords, z_list, energies=energies)

    def test_fast_path_force_only_matches_slow(self) -> None:
        """fast path (matvec) must agree with full K@alpha on real data."""
        from kernelforge import global_kernels
        from kernelforge.models.global_krr import _build_repr

        coords, z_list, _, forces = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalKRRModel(sigma=3.0, l2=1e-6)
        model.fit(tr, ztr, forces=forces[:20])

        E_fast, F_fast = model._predict(te, zte)

        # Slow path: build full kernel matrices
        X_te, dX_te = _build_repr(te, model.eps, with_gradients=True)
        X_tr, dX_tr = model._X_tr, model._dX_tr
        assert dX_te is not None
        assert dX_tr is not None
        K_hess = global_kernels.kernel_gaussian_hessian(X_te, dX_te, X_tr, dX_tr, model.sigma)
        F_slow: np.ndarray = (K_hess @ model._alpha).ravel()
        K_jt = global_kernels.kernel_gaussian_jacobian_t(X_te, X_tr, dX_tr, model.sigma)
        E_slow: np.ndarray = -(K_jt @ model._alpha)

        np.testing.assert_allclose(F_fast, F_slow, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(E_fast, E_slow, rtol=1e-7, atol=1e-9)

    def test_fast_path_energy_and_force_matches_slow(self) -> None:
        """energy_and_force fast path must agree with full K@alpha."""
        from kernelforge import global_kernels
        from kernelforge.models.global_krr import _build_repr

        coords, z_list, energies, forces = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalKRRModel(sigma=3.0, l2=1e-6)
        model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])

        E_fast, F_fast = model._predict(te, zte)

        X_te, dX_te = _build_repr(te, model.eps, with_gradients=True)
        X_tr, dX_tr = model._X_tr, model._dX_tr
        assert dX_te is not None
        assert dX_tr is not None
        n_test = len(te)
        K_full = global_kernels.kernel_gaussian_full(X_te, dX_te, X_tr, dX_tr, model.sigma)
        y = K_full @ model._alpha
        E_slow: np.ndarray = y[:n_test]
        F_slow: np.ndarray = -y[n_test:]

        np.testing.assert_allclose(E_fast, E_slow, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(F_fast, F_slow, rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# GlobalRFFModel tests
# ---------------------------------------------------------------------------


class TestGlobalRFFModel:
    """Fit/predict/save/load tests for GlobalRFFModel."""

    def test_energy_only_shapes(self) -> None:
        coords, z_list, energies, _ = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalRFFModel(sigma=3.0, l2=1e-6, d_rff=256, seed=42)
        model.fit(tr, ztr, energies=energies[:20])

        assert model.training_mode_ == "energy_only"
        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.shape == (5 * 9 * 3,)

    def test_force_only_shapes(self) -> None:
        coords, z_list, _, forces = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalRFFModel(sigma=3.0, l2=1e-6, d_rff=256, seed=42)
        model.fit(tr, ztr, forces=forces[:20])

        assert model.training_mode_ == "force_only"
        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.shape == (5 * 9 * 3,)

    def test_energy_and_force_shapes(self) -> None:
        coords, z_list, energies, forces = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalRFFModel(sigma=3.0, l2=1e-6, d_rff=256, seed=42)
        model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])

        assert model.training_mode_ == "energy_and_force"
        E_pred, F_pred = model.predict(te, zte)
        assert E_pred.shape == (5,)
        assert F_pred.shape == (5 * 9 * 3,)

    def test_save_load_roundtrip_energy_and_force(self, tmp_path: Path) -> None:
        coords, z_list, energies, forces = _load_ethanol(25)
        tr, te = coords[:20], coords[20:]
        ztr, zte = z_list[:20], z_list[20:]

        model = GlobalRFFModel(sigma=3.0, l2=1e-6, d_rff=256, seed=42)
        model.fit(tr, ztr, energies=energies[:20], forces=forces[:20])
        E_orig, F_orig = model.predict(te, zte)

        path = tmp_path / "global_rff_ef.npz"
        model.save(path)
        loaded = GlobalRFFModel.load(path)
        assert isinstance(loaded, GlobalRFFModel)

        E_load, F_load = loaded.predict(te, zte)
        np.testing.assert_allclose(E_load, E_orig, rtol=1e-10)
        np.testing.assert_allclose(F_load, F_orig, rtol=1e-10)

    def test_variable_atom_count_raises(self) -> None:
        rng = np.random.default_rng(0)
        coords = [rng.standard_normal((5, 3)), rng.standard_normal((6, 3))]
        z_list = [np.ones(5, dtype=np.int32), np.ones(6, dtype=np.int32)]
        energies = np.array([1.0, 2.0])

        model = GlobalRFFModel(sigma=1.0, l2=1e-6, d_rff=64)
        with pytest.raises(ValueError, match="same atom count"):
            model.fit(coords, z_list, energies=energies)
