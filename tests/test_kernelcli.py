"""Tests for kernelforge.kernelcli."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from kernelforge.kernelcli import (
    _build_model,
    _build_parser,
    _validate,
    load_small_mols_mini,
    run,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def small_mols_data():
    """Load a tiny slice of small_mols_mini for fast tests."""
    coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te = load_small_mols_mini(
        n_train=10, n_test=10
    )
    return coords_tr, z_tr, E_tr, F_tr, coords_te, z_te, E_te, F_te


# ---------------------------------------------------------------------------
# Argument parser / validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fchl18_rff_error(self):
        """FCHL18 + RFF should raise a parser error."""
        parser = _build_parser()
        args = parser.parse_args(
            ["--dataset", "small_mols_mini", "--representation", "fchl18", "--regressor", "rff"]
        )
        with pytest.raises(SystemExit):
            _validate(args, parser)

    def test_qm7b_force_only_error(self):
        """QM7b + force_only should raise a parser error."""
        parser = _build_parser()
        args = parser.parse_args(["--dataset", "qm7b", "--mode", "force_only"])
        with pytest.raises(SystemExit):
            _validate(args, parser)

    def test_qm7b_energy_and_force_error(self):
        """QM7b + energy_and_force should raise a parser error."""
        parser = _build_parser()
        args = parser.parse_args(["--dataset", "qm7b", "--mode", "energy_and_force"])
        with pytest.raises(SystemExit):
            _validate(args, parser)

    def test_small_mols_force_only_error(self):
        """small_mols_mini + force_only should raise a parser error."""
        parser = _build_parser()
        args = parser.parse_args(["--dataset", "small_mols_mini", "--mode", "force_only"])
        with pytest.raises(SystemExit):
            _validate(args, parser)

    def test_valid_fchl19_krr_passes(self):
        """Valid args should not raise."""
        parser = _build_parser()
        args = parser.parse_args(
            ["--dataset", "small_mols_mini", "--representation", "fchl19", "--regressor", "krr"]
        )
        _validate(args, parser)  # Should not raise

    def test_valid_fchl18_krr_passes(self):
        """FCHL18 + KRR is valid."""
        parser = _build_parser()
        args = parser.parse_args(
            ["--dataset", "small_mols_mini", "--representation", "fchl18", "--regressor", "krr"]
        )
        _validate(args, parser)  # Should not raise

    def test_valid_fchl19_rff_passes(self):
        """FCHL19 + RFF is valid."""
        parser = _build_parser()
        args = parser.parse_args(
            ["--dataset", "small_mols_mini", "--representation", "fchl19", "--regressor", "rff"]
        )
        _validate(args, parser)  # Should not raise

    def test_invalid_dataset_rejected(self):
        """Unknown dataset should be rejected by argparse."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", "not_a_dataset"])

    def test_split_choices(self):
        """--split must be 1-5."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", "rmd17_ethanol", "--split", "6"])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


class TestBuildModel:
    def _z_tr(self):
        # Minimal z_tr with H, C, O
        return [np.array([1, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)] * 5

    def test_krr_fchl19(self):
        from kernelforge.models import LocalKRRModel

        model = _build_model(
            "krr", "fchl19", 20.0, 1e-8, None, None, 512, 42, self._z_tr(), self._z_tr()
        )
        assert isinstance(model, LocalKRRModel)

    def test_rff_fchl19(self):
        from kernelforge.models import LocalRFFModel

        model = _build_model(
            "rff", "fchl19", 20.0, 1e-8, None, None, 512, 42, self._z_tr(), self._z_tr()
        )
        assert isinstance(model, LocalRFFModel)

    def test_krr_fchl18(self):
        from kernelforge.models import FCHL18KRRModel

        model = _build_model(
            "krr", "fchl18", 2.5, 1e-4, None, None, 512, 42, self._z_tr(), self._z_tr()
        )
        assert isinstance(model, FCHL18KRRModel)

    def test_auto_elements(self):
        from kernelforge.models import LocalKRRModel

        model = _build_model(
            "krr", "fchl19", 20.0, 1e-8, None, None, 512, 42, self._z_tr(), self._z_tr()
        )
        assert isinstance(model, LocalKRRModel)
        # elements not exposed on model but auto-detection didn't crash
        assert model is not None

    def test_override_elements(self):
        from kernelforge.models import LocalKRRModel

        model = _build_model(
            "krr", "fchl19", 20.0, 1e-8, [1, 6, 8], None, 512, 42, self._z_tr(), self._z_tr()
        )
        assert isinstance(model, LocalKRRModel)

    def test_auto_max_size(self):
        from kernelforge.models import FCHL18KRRModel

        model = _build_model(
            "krr", "fchl18", 2.5, 1e-4, None, None, 512, 42, self._z_tr(), self._z_tr()
        )
        assert isinstance(model, FCHL18KRRModel)
        assert model.max_size == 9  # both z_tr and z_te have 9-atom molecules

    def test_override_max_size(self):
        from kernelforge.models import FCHL18KRRModel

        model = _build_model(
            "krr", "fchl18", 2.5, 1e-4, None, 15, 512, 42, self._z_tr(), self._z_tr()
        )
        assert isinstance(model, FCHL18KRRModel)
        assert model.max_size == 15


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


class TestLoadSmallMolsMini:
    def test_load_default(self):
        _coords_tr, _z_tr, E_tr, _F_tr, _coords_te, _z_te, E_te, _F_te = load_small_mols_mini(
            n_train=None, n_test=None
        )
        assert len(E_tr) == 1000
        assert len(E_te) == 595

    def test_load_sliced(self):
        coords_tr, _z_tr, E_tr, F_tr, _coords_te, _z_te, E_te, _F_te = load_small_mols_mini(
            n_train=5, n_test=3
        )
        assert len(E_tr) == 5
        assert len(E_te) == 3
        assert len(coords_tr) == 5
        assert len(F_tr) == 5

    def test_load_too_many_raises(self):
        with pytest.raises(ValueError, match="only has"):
            load_small_mols_mini(n_train=99999, n_test=1)

    def test_coords_dtype(self):
        coords_tr, z_tr, E_tr, F_tr, *_ = load_small_mols_mini(n_train=3, n_test=1)
        assert coords_tr[0].dtype == np.float64
        assert z_tr[0].dtype == np.int32
        assert E_tr.dtype == np.float64
        assert F_tr[0].dtype == np.float64


# ---------------------------------------------------------------------------
# End-to-end run() integration tests (small_mols_mini only — no downloads)
# ---------------------------------------------------------------------------


def _make_args(**kwargs: str | int | float | list | None) -> argparse.Namespace:
    """Build an argparse.Namespace with sensible defaults, overridden by kwargs."""
    defaults: dict[str, str | int | float | list | None] = {
        "dataset": "small_mols_mini",
        "regressor": "krr",
        "representation": "fchl19",
        "mode": "energy_and_force",
        "n_train": 10,
        "n_test": 10,
        "split": 1,
        "sigma": 20.0,
        "l2": 1e-6,
        "d_rff": 64,
        "seed": 42,
        "elements": None,
        "max_size": None,
        "save": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestRunIntegration:
    def test_krr_fchl19_energy_and_force(self, capsys):
        run(_make_args())
        out = capsys.readouterr().out
        assert "energy" in out
        assert "force" in out
        assert "Done" in out

    def test_krr_fchl19_energy_only(self, capsys):
        run(_make_args(mode="energy_only"))
        out = capsys.readouterr().out
        assert "energy" in out
        assert "Done" in out

    def test_rff_fchl19_energy_and_force(self, capsys):
        run(_make_args(regressor="rff", d_rff=64))
        out = capsys.readouterr().out
        assert "energy" in out
        assert "force" in out
        assert "Done" in out

    def test_krr_fchl18_energy_and_force(self, capsys):
        run(_make_args(representation="fchl18", sigma=2.5, l2=1e-4))
        out = capsys.readouterr().out
        assert "energy" in out
        assert "force" in out
        assert "Done" in out

    def test_krr_fchl18_energy_only(self, capsys):
        run(_make_args(representation="fchl18", mode="energy_only", sigma=2.5, l2=1e-8))
        out = capsys.readouterr().out
        assert "energy" in out
        assert "Done" in out

    def test_save_model(self, tmp_path, capsys):
        save_path = str(tmp_path / "test_model.npz")
        run(_make_args(save=save_path))
        assert Path(save_path).exists()
        capsys.readouterr()

    def test_elements_override(self, capsys):
        run(_make_args(elements=[1, 6, 7, 8]))
        out = capsys.readouterr().out
        assert "Done" in out

    def test_max_size_override(self, capsys):
        run(_make_args(representation="fchl18", sigma=2.5, l2=1e-4, max_size=25))
        out = capsys.readouterr().out
        assert "Done" in out
