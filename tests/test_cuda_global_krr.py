"""Tests for CudaGlobalKRRModel using rMD17 ethanol.

These tests are skipped when:
  - cuda_krr_ext was not built (no CUDA compiler at build time)
  - No NVIDIA GPU is detected
  - The rMD17 ethanol dataset is not cached locally

Run with:
    uv run pytest tests/test_cuda_global_krr.py -v

Float32 precision note
----------------------
CudaGlobalKRRModel trains in float32.  The full kernel matrix K_full mixes
the energy block (diagonal ~1.0) with the force block (diagonal ~1e-3), which
makes the matrix ill-conditioned in float32 for large training sets from MD
trajectories (very similar structures).  Tests are written to stay within
the well-conditioned regime (N_train = 20, sigma = 3.0, l2 = 1e-5).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip guard — skip entire module if CUDA extension is unavailable
# ---------------------------------------------------------------------------
cuda_krr_ext = pytest.importorskip(
    "kernelforge.cuda_krr_ext",
    reason="cuda_krr_ext not built (requires CUDA compiler and GPU at build time)",
)

# Check for a functional GPU via nvidia-smi
_nvidia_smi = shutil.which("nvidia-smi")
try:
    _gpu_ok = (
        _nvidia_smi is not None
        and subprocess.run(  # noqa: S603
            [_nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        ).returncode
        == 0
    )
except (OSError, subprocess.TimeoutExpired):
    _gpu_ok = False

pytestmark = pytest.mark.skipif(
    not _gpu_ok,
    reason="No NVIDIA GPU detected by nvidia-smi",
)

from kernelforge.models import CudaGlobalKRRModel, GlobalKRRModel  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset helpers (same pattern as test_models_global.py)
# ---------------------------------------------------------------------------

_CACHE = Path.home() / ".kernelforge" / "datasets"
_RMD17_TRAIN = _CACHE / "rmd17_ethanol_train_01.npz"

# Training/test sizes that stay in the well-conditioned float32 regime.
# Ethanol has 9 atoms → D=27 coords, M=36 invdist features.
# K_full size = N*(1+D) x N*(1+D).  N=20 -> 560x560 (condition ~10^5, safe).
_N_TRAIN = 20
_N_TEST = 5


def _load_ethanol(
    n: int = _N_TRAIN + _N_TEST,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Load a small slice of rMD17 ethanol.  Skip test if not cached."""
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
# Basic smoke / shape tests
# ---------------------------------------------------------------------------


def test_fit_predict_smoke() -> None:
    """Fit and predict without error; check output shapes."""
    coords, z, E, F = _load_ethanol()
    tr, te = coords[:_N_TRAIN], coords[_N_TRAIN:]
    ztr, zte = z[:_N_TRAIN], z[_N_TRAIN:]

    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(tr, ztr, energies=E[:_N_TRAIN], forces=F[:_N_TRAIN])

    assert model.is_fitted_
    assert model.training_mode_ == "energy_and_force"
    # Ethanol: 9 atoms, D = 27, M = 9*8/2 = 36
    assert model._D == 27
    assert model._M == 36

    E_pred, F_pred = model.predict(te, zte)

    assert E_pred.shape == (_N_TEST,)
    assert F_pred.shape == (_N_TEST, 27)
    assert np.all(np.isfinite(E_pred))
    assert np.all(np.isfinite(F_pred))


def test_energy_only_raises() -> None:
    """energy_only mode must raise NotImplementedError."""
    coords, z, E, _ = _load_ethanol()
    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    with pytest.raises(NotImplementedError, match="energy_and_force"):
        model.fit(coords, z, energies=E)


def test_force_only_raises() -> None:
    """force_only mode must raise NotImplementedError."""
    coords, z, _, F = _load_ethanol()
    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    with pytest.raises(NotImplementedError, match="energy_and_force"):
        model.fit(coords, z, forces=F)


# ---------------------------------------------------------------------------
# Numerical agreement with GlobalKRRModel (CPU float64 reference)
# ---------------------------------------------------------------------------


def test_cuda_vs_cpu_agreement() -> None:
    """CudaGlobalKRRModel predictions must agree with GlobalKRRModel to ~1e-3.

    The two models use the same algorithm: float32 GPU vs float64 CPU.
    We use l2=1e-5 to keep the float32 K_full well-conditioned for ethanol
    (K_FF diagonal ~1e-3, so l2=1e-5 adds a meaningful stabilizing term).
    """
    coords, z, E, F = _load_ethanol()
    tr, te = coords[:_N_TRAIN], coords[_N_TRAIN:]
    ztr, zte = z[:_N_TRAIN], z[_N_TRAIN:]

    sigma, l2 = 3.0, 1e-5

    cpu_model = GlobalKRRModel(sigma=sigma, l2=l2)
    cpu_model.fit(tr, ztr, energies=E[:_N_TRAIN], forces=F[:_N_TRAIN])
    E_cpu, F_cpu = cpu_model.predict(te, zte)

    gpu_model = CudaGlobalKRRModel(sigma=sigma, l2=l2)
    gpu_model.fit(tr, ztr, energies=E[:_N_TRAIN], forces=F[:_N_TRAIN])
    E_gpu, F_gpu = gpu_model.predict(te, zte)

    np.testing.assert_allclose(
        E_gpu,
        E_cpu,
        rtol=1e-3,
        atol=1e-2,
        err_msg="Energy predictions disagree between CUDA and CPU models",
    )
    # Forces: absolute tolerance only.
    # Near-zero force components have large relative error in float32 vs float64,
    # but the absolute error stays bounded by ~0.1 kcal/mol/Å for N=20, l2=1e-5.
    # GlobalKRRModel returns forces flat (N*D,); CudaGlobalKRRModel returns (N, D).
    np.testing.assert_allclose(
        F_gpu.ravel(),
        F_cpu.ravel(),
        atol=0.5,
        err_msg="Force predictions disagree between CUDA and CPU models",
    )


# ---------------------------------------------------------------------------
# Training score
# ---------------------------------------------------------------------------


def test_train_score() -> None:
    """Training scores must be finite."""
    coords, z, E, F = _load_ethanol()

    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(coords[:_N_TRAIN], z[:_N_TRAIN], energies=E[:_N_TRAIN], forces=F[:_N_TRAIN])

    scores = model.train_score_
    assert "energy" in scores
    assert "force" in scores
    assert np.isfinite(scores["energy"].mae)
    assert np.isfinite(scores["force"].mae)
    assert scores["energy"].mae >= 0.0
    assert scores["force"].mae >= 0.0


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Saved model must produce identical predictions after load."""
    coords, z, E, F = _load_ethanol()
    tr, te = coords[:_N_TRAIN], coords[_N_TRAIN:]
    ztr, zte = z[:_N_TRAIN], z[_N_TRAIN:]

    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(tr, ztr, energies=E[:_N_TRAIN], forces=F[:_N_TRAIN])
    E_orig, F_orig = model.predict(te, zte)

    path = tmp_path / "cuda_model.npz"
    model.save(path)

    loaded = CudaGlobalKRRModel.load(path)
    assert isinstance(loaded, CudaGlobalKRRModel)
    assert loaded.is_fitted_
    assert loaded.training_mode_ == "energy_and_force"
    assert loaded.sigma == model.sigma
    assert loaded.l2 == model.l2

    E_load, F_load = loaded.predict(te, zte)

    # float32 round-trip: same float32 state reloaded — expect bit-exact
    np.testing.assert_array_equal(E_orig, E_load, err_msg="Energy changed after save/load")
    np.testing.assert_array_equal(F_orig, F_load, err_msg="Forces changed after save/load")


# ---------------------------------------------------------------------------
# predict_torch
# ---------------------------------------------------------------------------


def test_predict_torch() -> None:
    """predict_torch must return tensors matching predict() to float32 precision."""
    torch = pytest.importorskip("torch", reason="PyTorch not installed")

    from kernelforge import invdist_repr

    coords, z, E, F = _load_ethanol()
    tr, te = coords[:_N_TRAIN], coords[_N_TRAIN:]
    ztr = z[:_N_TRAIN]
    zte = z[_N_TRAIN:]

    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(tr, ztr, energies=E[:_N_TRAIN], forces=F[:_N_TRAIN])

    # Reference via numpy predict()
    E_np, F_np = model.predict(te, zte)

    # Build test representations as float32 torch tensors
    X_list, dX_list = [], []
    for c in te:
        x, dx = invdist_repr.inverse_distance_upper_and_jacobian(
            np.asarray(c, dtype=np.float64), 1e-12
        )
        X_list.append(x.astype(np.float32))
        dX_list.append(dx.astype(np.float32))

    X_te_t = torch.tensor(np.array(X_list))  # (_N_TEST, M) float32
    dX_te_t = torch.tensor(np.array(dX_list))  # (_N_TEST, D, M) float32

    E_t, F_t = model.predict_torch(X_te_t, dX_te_t)

    assert E_t.shape == (_N_TEST,)
    assert F_t.shape == (_N_TEST, 27)

    # predict_torch and predict() go through the same CUDA state → identical
    np.testing.assert_array_equal(
        E_t.cpu().numpy(),
        E_np.astype(np.float32),
        err_msg="predict_torch energy differs from predict()",
    )
    np.testing.assert_array_equal(
        F_t.cpu().numpy(),
        F_np.astype(np.float32),
        err_msg="predict_torch forces differ from predict()",
    )


# ---------------------------------------------------------------------------
# score() method
# ---------------------------------------------------------------------------


def test_score_method() -> None:
    """model.score() on held-out test data must return finite MAE values."""
    coords, z, E, F = _load_ethanol()
    tr, te = coords[:_N_TRAIN], coords[_N_TRAIN:]
    ztr, zte = z[:_N_TRAIN], z[_N_TRAIN:]

    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(tr, ztr, energies=E[:_N_TRAIN], forces=F[:_N_TRAIN])

    scores = model.score(te, zte, energies=E[_N_TRAIN:], forces=F[_N_TRAIN:])

    assert "energy" in scores
    assert "force" in scores
    assert np.isfinite(scores["energy"].mae)
    assert np.isfinite(scores["force"].mae)
