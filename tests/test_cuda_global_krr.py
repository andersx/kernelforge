"""Tests for CudaGlobalKRRModel using rMD17 ethanol.

These tests are skipped when:
  - cuda_global_kernels was not built (no CUDA + PyTorch at build time)
  - No NVIDIA GPU is detected
  - The rMD17 ethanol dataset is not cached locally

Run with:
    uv run pytest tests/test_cuda_global_krr.py -v

Architecture notes
------------------
CudaGlobalKRRModel trains using:
  - GPU float32 kernel assembly  (cuda_global_kernels.kernel_gaussian_full_symm)
  - CPU float64 Cholesky solve   (kernelmath.solve_cholesky)
  - GPU float32 contracted matvec for inference

The float64 Cholesky solve avoids conditioning failures that afflict pure
float32 Cholesky for large training sets with tight sigma on MD trajectories.
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
cuda_global_kernels = pytest.importorskip(
    "kernelforge.cuda_global_kernels",
    reason="cuda_global_kernels not built (requires CUDA + PyTorch at build time)",
)

torch = pytest.importorskip(
    "torch",
    reason="PyTorch not installed",
)

# Check for a functional GPU
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
# Dataset helpers
# ---------------------------------------------------------------------------

_CACHE = Path.home() / ".kernelforge" / "datasets"
_RMD17_TRAIN = _CACHE / "rmd17_ethanol_train_01.npz"

# Ethanol: 9 atoms, D=27 coords, M=36 invdist features.
_N_TRAIN = 20
_N_TEST = 5


def _load_ethanol(
    n: int = _N_TRAIN + _N_TEST,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
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
# Smoke / shape tests
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
    assert model._D == 27
    assert model._M == 36

    E_pred, F_pred = model.predict(te, zte)
    assert E_pred.shape == (_N_TEST,)
    assert F_pred.shape == (_N_TEST, 27)
    assert np.all(np.isfinite(E_pred))
    assert np.all(np.isfinite(F_pred))


# ---------------------------------------------------------------------------
# energy_only mode
# ---------------------------------------------------------------------------


def test_energy_only_smoke() -> None:
    """energy_only mode: fit and predict without error; check output shapes."""
    coords, z, E, _ = _load_ethanol()
    tr, te = coords[:_N_TRAIN], coords[_N_TRAIN:]
    ztr, zte = z[:_N_TRAIN], z[_N_TRAIN:]

    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(tr, ztr, energies=E[:_N_TRAIN])

    assert model.is_fitted_
    assert model.training_mode_ == "energy_only"
    assert model._M == 36

    E_pred, _F_pred = model.predict(te, zte)
    assert E_pred.shape == (_N_TEST,)
    assert np.all(np.isfinite(E_pred))


def test_energy_only_cuda_vs_cpu() -> None:
    """energy_only CudaGlobalKRRModel energies must agree with GlobalKRRModel to 1e-3."""
    coords, z, E, _ = _load_ethanol()
    tr, te = coords[:_N_TRAIN], coords[_N_TRAIN:]
    ztr, zte = z[:_N_TRAIN], z[_N_TRAIN:]

    sigma, l2 = 3.0, 1e-5

    cpu_model = GlobalKRRModel(sigma=sigma, l2=l2)
    cpu_model.fit(tr, ztr, energies=E[:_N_TRAIN])
    E_cpu, _ = cpu_model.predict(te, zte)

    gpu_model = CudaGlobalKRRModel(sigma=sigma, l2=l2)
    gpu_model.fit(tr, ztr, energies=E[:_N_TRAIN])
    E_gpu, _ = gpu_model.predict(te, zte)

    np.testing.assert_allclose(
        E_gpu,
        E_cpu,
        rtol=1e-3,
        atol=1e-2,
        err_msg="energy_only: CUDA and CPU energy predictions disagree",
    )


def test_energy_only_train_score() -> None:
    """energy_only train_score_ must contain finite 'energy' MAE."""
    coords, z, E, _ = _load_ethanol()
    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(coords[:_N_TRAIN], z[:_N_TRAIN], energies=E[:_N_TRAIN])

    scores = model.train_score_
    assert "energy" in scores
    assert "force" not in scores
    assert np.isfinite(scores["energy"].mae)
    assert scores["energy"].mae >= 0.0


def test_energy_only_save_load(tmp_path: Path) -> None:
    """energy_only save/load roundtrip must produce identical energy predictions."""
    coords, z, E, _ = _load_ethanol()
    tr, te = coords[:_N_TRAIN], coords[_N_TRAIN:]
    ztr, zte = z[:_N_TRAIN], z[_N_TRAIN:]

    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(tr, ztr, energies=E[:_N_TRAIN])
    E_orig, _ = model.predict(te, zte)

    path = tmp_path / "cuda_energy_only.npz"
    model.save(path)

    loaded = CudaGlobalKRRModel.load(path)
    assert isinstance(loaded, CudaGlobalKRRModel)
    assert loaded.is_fitted_
    assert loaded.training_mode_ == "energy_only"
    assert loaded.sigma == model.sigma

    E_load, _ = loaded.predict(te, zte)
    np.testing.assert_allclose(
        E_orig, E_load, rtol=1e-6, err_msg="energy_only: energies changed after save/load"
    )


def test_predict_torch_raises_for_energy_only() -> None:
    """predict_torch must raise RuntimeError for energy_only models."""
    import torch

    coords, z, E, _ = _load_ethanol()
    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(coords[:_N_TRAIN], z[:_N_TRAIN], energies=E[:_N_TRAIN])

    dummy_X = torch.randn(1, 36).cuda().float()
    dummy_dX = torch.randn(1, 27, 36).cuda().float()
    with pytest.raises(RuntimeError, match="energy_and_force"):
        model.predict_torch(dummy_X, dummy_dX)


def test_force_only_raises() -> None:
    """force_only mode must raise NotImplementedError."""
    coords, z, _, F = _load_ethanol()
    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    with pytest.raises(NotImplementedError, match="energy_and_force"):
        model.fit(coords, z, forces=F)


# ---------------------------------------------------------------------------
# Numerical agreement with GlobalKRRModel (CPU float64)
# ---------------------------------------------------------------------------


def test_cuda_vs_cpu_agreement() -> None:
    """CudaGlobalKRRModel predictions must agree with GlobalKRRModel to ~1e-2.

    Both models use the same kernel formula.  CudaGlobalKRRModel now uses a
    float64 CPU Cholesky (via kernelmath.solve_cholesky), so agreement with
    GlobalKRRModel should be close despite GPU float32 kernel assembly.
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
    np.testing.assert_allclose(
        F_gpu.ravel(),
        F_cpu.ravel(),
        atol=0.5,
        err_msg="Force predictions disagree between CUDA and CPU models",
    )


# ---------------------------------------------------------------------------
# Larger N + tighter sigma (the previously-failing case)
# ---------------------------------------------------------------------------


def test_n100_sigma1_no_nan() -> None:
    """N=100, sigma=1.0 must not produce NaN (float64 solve path prevents this)."""
    coords, z, E, F = _load_ethanol(n=105)
    tr, te = coords[:100], coords[100:]
    ztr, zte = z[:100], z[100:]

    model = CudaGlobalKRRModel(sigma=1.0, l2=1e-4)
    model.fit(tr, ztr, energies=E[:100], forces=F[:100])
    E_pred, F_pred = model.predict(te, zte)

    assert np.all(np.isfinite(E_pred)), "NaN/Inf in E_pred for N=100, sigma=1.0"
    assert np.all(np.isfinite(F_pred)), "NaN/Inf in F_pred for N=100, sigma=1.0"


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

    E_load, F_load = loaded.predict(te, zte)
    # After save/load the float32 state is reloaded bit-exactly → same predictions
    np.testing.assert_allclose(E_orig, E_load, rtol=1e-5, err_msg="Energy changed after save/load")
    np.testing.assert_allclose(
        F_orig.ravel(), F_load.ravel(), rtol=1e-5, err_msg="Forces changed after save/load"
    )


# ---------------------------------------------------------------------------
# predict_torch
# ---------------------------------------------------------------------------


def test_predict_torch() -> None:
    """predict_torch must return CUDA tensors matching predict() to float32."""
    from kernelforge import cuda_invdist_repr

    coords, z, E, F = _load_ethanol()
    tr, te = coords[:_N_TRAIN], coords[_N_TRAIN:]
    ztr = z[:_N_TRAIN]
    zte = z[_N_TRAIN:]

    model = CudaGlobalKRRModel(sigma=3.0, l2=1e-5)
    model.fit(tr, ztr, energies=E[:_N_TRAIN], forces=F[:_N_TRAIN])

    # predict() applies per-element energy baseline; predict_torch() returns
    # baseline-subtracted energies (same as _predict()).  Forces are unaffected
    # by the constant baseline, so they agree directly.
    _E_np, F_np = model.predict(te, zte)
    E_raw, _F_raw = model._predict(te, zte)

    # Build float32 CUDA tensors via the same GPU path used internally
    n_atoms = te[0].shape[0]
    coords_np = np.stack([c.astype(np.float32) for c in te], axis=0)  # (N_test, n_atoms, 3)
    coords_cuda = torch.from_numpy(np.ascontiguousarray(coords_np)).cuda()
    X_cuda, dX_cuda = cuda_invdist_repr.inverse_distance_upper_and_jacobian(
        coords_cuda, n_atoms, model.eps
    )

    E_t, F_t = model.predict_torch(X_cuda, dX_cuda)

    assert E_t.shape == (_N_TEST,)
    assert F_t.shape == (_N_TEST, 27)
    assert E_t.device.type == "cuda"
    assert F_t.device.type == "cuda"

    # Energies: predict_torch returns baseline-subtracted values → compare
    # against _predict() (same conventions, both float32 CUDA).
    np.testing.assert_allclose(
        E_t.cpu().numpy(),
        E_raw.astype(np.float32),
        rtol=1e-4,
        err_msg="predict_torch energy differs from _predict()",
    )
    # Forces: baseline is a constant → no effect on gradients.
    np.testing.assert_allclose(
        F_t.cpu().numpy().ravel(),
        F_np.astype(np.float32).ravel(),
        rtol=1e-4,
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


# ---------------------------------------------------------------------------
# kernel functions standalone test
# ---------------------------------------------------------------------------


def test_kernel_gaussian_full_symm_shape() -> None:
    """kernel_gaussian_full_symm must return correctly shaped symmetric tensor."""
    N, M, D = 5, 36, 27
    X = torch.randn(N, M).cuda().float()
    dX = torch.randn(N, D, M).cuda().float()

    K = cuda_global_kernels.kernel_gaussian_full_symm(X, dX, 2.0)

    full = N * (1 + D)
    assert K.shape == (full, full)
    assert K.device.type == "cuda"
    assert K.dtype == torch.float32

    # Check symmetry (both triangles filled)
    K_np = K.cpu().numpy()
    np.testing.assert_allclose(K_np, K_np.T, atol=1e-5, err_msg="K_full is not symmetric")

    # Diagonal of K_EE must be 1.0 (K[a,a] = exp(0) = 1)
    np.testing.assert_allclose(
        np.diag(K_np[:N, :N]),
        np.ones(N),
        atol=1e-5,
        err_msg="K_EE diagonal should be 1.0",
    )


def test_kernel_gaussian_full_matvec_shape() -> None:
    """kernel_gaussian_full_matvec must return correctly shaped tensors."""
    N_q, N_t, M, D = 3, 5, 36, 27
    X_q = torch.randn(N_q, M).cuda().float()
    dX_q = torch.randn(N_q, D, M).cuda().float()
    X_t = torch.randn(N_t, M).cuda().float()
    alpha_E = torch.randn(N_t).cuda().float()
    alpha_desc_F = torch.randn(N_t, M).cuda().float()

    E, F = cuda_global_kernels.kernel_gaussian_full_matvec(
        X_q, dX_q, X_t, alpha_E, alpha_desc_F, 2.0
    )

    assert E.shape == (N_q,)
    assert F.shape == (N_q, D)
    assert E.device.type == "cuda"
    assert F.device.type == "cuda"


# ---------------------------------------------------------------------------
# Numerical equivalence: CUDA RFP kernel vs CPU double-precision reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17, 32])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.5])
@pytest.mark.parametrize("seed", [0, 42])
def test_kernel_gaussian_symm_rfp_cuda_vs_cpu(n: int, sigma: float, seed: int) -> None:
    """CUDA kernel_gaussian_symm_rfp must agree with the CPU reference within fp32 tolerance.

    The CPU function (global_kernels.kernel_gaussian_symm_rfp) computes
    K[i,j] = exp(-||X_i - X_j||^2 / (2*sigma^2)) in float64, stored in RFP
    format with uplo='L', transr='N'.

    The CUDA function (cuda_global_kernels.kernel_gaussian_symm_rfp) must
    produce the same values in float32, stored in the same RFP convention.

    Both packed buffers are unpacked via kernelmath.rfp_to_full and symmetrised
    before comparison so that convention differences in the unused triangle do
    not cause false failures.
    """
    from kernelforge import global_kernels, kernelmath

    rng = np.random.default_rng(seed)
    d = min(5, n)
    X_f64 = rng.standard_normal((n, d))

    # --- CPU reference (float64) ---
    K_rfp_cpu = global_kernels.kernel_gaussian_symm_rfp(X_f64, sigma)
    K_cpu_tri = kernelmath.rfp_to_full(K_rfp_cpu, n, uplo="L", transr="N")
    K_cpu = np.tril(K_cpu_tri) + np.tril(K_cpu_tri, -1).T  # symmetrise

    # --- CUDA (float32) ---
    X_f32 = torch.tensor(X_f64, dtype=torch.float32).cuda()
    K_rfp_cuda = cuda_global_kernels.kernel_gaussian_symm_rfp(X_f32, sigma)
    assert K_rfp_cuda.shape == (n * (n + 1) // 2,)
    assert K_rfp_cuda.dtype == torch.float32

    K_rfp_cuda_np = K_rfp_cuda.cpu().numpy().astype(np.float64)
    K_cuda_tri = kernelmath.rfp_to_full(K_rfp_cuda_np, n, uplo="U", transr="N")
    K_cuda = np.tril(K_cuda_tri) + np.tril(K_cuda_tri, -1).T  # symmetrise

    # --- Compare ---
    # Diagonal must be 1.0 (self-distance = 0)
    np.testing.assert_allclose(
        np.diag(K_cuda),
        np.ones(n),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"CUDA kernel diagonal not 1.0 (sigma={sigma}, n={n})",
    )

    # Full matrix must agree within float32 precision
    np.testing.assert_allclose(
        K_cuda,
        K_cpu,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"CUDA and CPU RFP kernels differ (sigma={sigma}, n={n})",
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7, 8])
def test_kernel_gaussian_symm_rfp_norms_step(n: int) -> None:
    """Regression test for the curfpSsfr2 norms-addition step (Phase 3).

    Uses orthonormal columns (||X[:,i]||=1 for all i) so that the squared
    distance simplifies to a known value:

        ||X_i - X_j||^2 = ||X_i||^2 + ||X_j||^2 - 2*<X_i,X_j>
                        = 2 - 2*<X_i,X_j>

    For the identity matrix (n <= M), <X_i,X_j> = delta_ij, so:
        K[i,i] = exp(0)           = 1.0
        K[i,j] = exp(-1/sigma^2)  for i != j

    This directly exercises the norms-addition path: if Phase 3 is wrong
    (zero norms, wrong sign, etc.) the off-diagonal values will differ.
    """
    import torch

    from kernelforge import cuda_global_kernels, kernelmath

    sigma = 1.0

    # Build X = first n columns of the n*n identity (each col has norm 1)
    X_f32 = torch.eye(n, dtype=torch.float32).cuda()  # shape (n, n)

    K_rfp_cuda = cuda_global_kernels.kernel_gaussian_symm_rfp(X_f32, sigma)
    K_rfp_np = K_rfp_cuda.cpu().numpy().astype(np.float64)
    K_tri = kernelmath.rfp_to_full(K_rfp_np, n, uplo="U", transr="N")
    K = np.tril(K_tri) + np.tril(K_tri, -1).T

    # Diagonal must be 1.0
    np.testing.assert_allclose(
        np.diag(K),
        np.ones(n),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Diagonal not 1.0 for n={n}",
    )

    # Off-diagonal: ||e_i - e_j||^2 = 2, so K[i,j] = exp(-2/(2*sigma^2)) = exp(-1)
    expected_off = np.exp(-1.0)
    mask = ~np.eye(n, dtype=bool)
    np.testing.assert_allclose(
        K[mask],
        expected_off,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Off-diagonal != exp(-1) for n={n}: norms-addition (curfpSsfr2) likely wrong",
    )


# ---------------------------------------------------------------------------
# kernel_gaussian_full_symm_rfp — shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "d", "m"), [(2, 2, 3), (3, 3, 4), (5, 4, 6)])
def test_kernel_gaussian_full_symm_rfp_shape(n: int, d: int, m: int) -> None:
    """kernel_gaussian_full_symm_rfp must return a 1-D float32 CUDA tensor of
    the correct RFP size BIG*(BIG+1)//2, where BIG = N*(1+D)."""
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.standard_normal((n, m)), dtype=torch.float32).cuda()
    dX = torch.tensor(rng.standard_normal((n, d, m)), dtype=torch.float32).cuda()

    K_rfp = cuda_global_kernels.kernel_gaussian_full_symm_rfp(X, dX, sigma=1.0)

    BIG = n * (1 + d)
    expected_len = BIG * (BIG + 1) // 2
    assert K_rfp.shape == (expected_len,), f"Expected shape ({expected_len},), got {K_rfp.shape}"
    assert K_rfp.dtype == torch.float32, f"Expected float32, got {K_rfp.dtype}"
    assert K_rfp.device.type == "cuda", f"Expected CUDA tensor, got {K_rfp.device}"


# ---------------------------------------------------------------------------
# kernel_gaussian_full_symm_rfp — numerical agreement with dense reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "d", "m"), [(2, 2, 3), (3, 3, 4), (4, 3, 5), (5, 4, 6)])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.5])
@pytest.mark.parametrize("seed", [0, 7])
def test_kernel_gaussian_full_symm_rfp_cuda_vs_dense(
    n: int, d: int, m: int, sigma: float, seed: int
) -> None:
    """kernel_gaussian_full_symm_rfp must agree with kernel_gaussian_full_symm
    (dense) within float32 tolerance.

    The dense function fills both triangles.  The RFP function stores only the
    lower triangle (TRANSR=N, UPLO=L); it is unpacked via kernelmath.rfp_to_full
    with uplo='U', transr='N' and then symmetrised for comparison.
    """
    from kernelforge import kernelmath

    rng = np.random.default_rng(seed)
    X_np = rng.standard_normal((n, m)).astype(np.float32)
    dX_np = rng.standard_normal((n, d, m)).astype(np.float32)

    X = torch.tensor(X_np).cuda()
    dX = torch.tensor(dX_np).cuda()

    # --- Dense reference (fully symmetric, both triangles) ---
    K_dense = cuda_global_kernels.kernel_gaussian_full_symm(X, dX, sigma)
    K_dense_np = K_dense.cpu().numpy().astype(np.float64)

    # --- RFP result ---
    K_rfp = cuda_global_kernels.kernel_gaussian_full_symm_rfp(X, dX, sigma)
    K_rfp_np = K_rfp.cpu().numpy().astype(np.float64)

    BIG = n * (1 + d)
    # Unpack RFP (TRANSR=N, UPLO=L stored) — rfp_to_full with uplo='U', transr='N'
    K_rfp_tri = kernelmath.rfp_to_full(K_rfp_np, BIG, uplo="U", transr="N")
    K_rfp_full = np.tril(K_rfp_tri) + np.tril(K_rfp_tri, -1).T  # symmetrise

    np.testing.assert_allclose(
        K_rfp_full,
        K_dense_np,
        rtol=1e-4,
        atol=1e-4,
        err_msg=(
            f"kernel_gaussian_full_symm_rfp != kernel_gaussian_full_symm "
            f"(n={n}, d={d}, m={m}, sigma={sigma}, seed={seed})"
        ),
    )
