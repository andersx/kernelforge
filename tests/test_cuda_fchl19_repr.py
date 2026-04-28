"""Tests for cuda_fchl19_repr — GPU FCHL19 forward representation.

Correctness: GPU float32 vs CPU float64 reference, rtol=1e-4, atol=1e-5.
Timing:      QM7B subset (200 molecules) GPU forward pass (printed, not asserted).

Skipped automatically when:
  - cuda_fchl19_repr was not built (no CUDA + PyTorch at build time)
  - No NVIDIA GPU is detected

Run with:
    uv run --env-file /dev/null pytest tests/test_cuda_fchl19_repr.py -v -s
"""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from kernelforge.fchl19_repr import generate_fchl_acsf as cpu_generate_fchl_acsf

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

cuda_fchl19_repr = pytest.importorskip(
    "kernelforge.cuda_fchl19_repr",
    reason="cuda_fchl19_repr not built (requires CUDA + PyTorch at build time)",
)

torch = pytest.importorskip("torch", reason="PyTorch not installed")

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

# ---------------------------------------------------------------------------
# QM7B elements (sorted, same ordering as CLI benchmark)
# ---------------------------------------------------------------------------

_ELEMENTS_QM7B: list[int] = [1, 6, 7, 8, 16, 17]

# Default FCHL19 hyperparameters (match CPU binding defaults)
_DEFAULTS: dict[str, Any] = {
    "nRs2": 24,
    "nRs3": 20,
    "nFourier": 1,
    "eta2": 0.32,
    "eta3": 2.7,
    "zeta": math.pi,
    "rcut": 8.0,
    "acut": 8.0,
    "two_body_decay": 1.8,
    "three_body_decay": 0.57,
    "three_body_weight": 13.4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _z_to_idx(z_arr: NDArray[np.int32], elements: list[int]) -> NDArray[np.int32]:
    """Map nuclear charges to element indices in [0, nelements)."""
    idx_map = {z: i for i, z in enumerate(elements)}
    return np.array([idx_map[int(z)] for z in z_arr], dtype=np.int32)


def _build_gpu_batch(
    coords_list: list[NDArray[np.float64]],
    z_list: list[NDArray[np.int32]],
    elements: list[int],
    device: Any,
) -> tuple[Any, Any, Any]:
    """Pad coords + element-index arrays into batched GPU tensors.

    Returns
    -------
    coords_gpu : (nm, max_atoms, 3)  float32 CUDA
    Q_gpu      : (nm, max_atoms)     int32   CUDA
    N_gpu      : (nm,)               int32   CUDA
    """
    nm = len(coords_list)
    max_atoms = max(len(z) for z in z_list)

    coords_np = np.zeros((nm, max_atoms, 3), dtype=np.float32)
    Q_np = np.zeros((nm, max_atoms), dtype=np.int32)
    N_np = np.array([len(z) for z in z_list], dtype=np.int32)

    for m, (coords, z) in enumerate(zip(coords_list, z_list, strict=False)):
        na = len(z)
        coords_np[m, :na, :] = coords.astype(np.float32)
        Q_np[m, :na] = _z_to_idx(z, elements)

    coords_gpu = torch.from_numpy(coords_np).to(device)
    Q_gpu = torch.from_numpy(Q_np).to(device)
    N_gpu = torch.from_numpy(N_np).to(device)
    return coords_gpu, Q_gpu, N_gpu


# ---------------------------------------------------------------------------
# Tiny toy molecule tests (no external data)
# ---------------------------------------------------------------------------


def _water_system() -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Single water molecule: H-O-H."""
    coords = np.array([[-0.9572, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9572, 0.0, 0.0]], dtype=np.float64)
    z = np.array([1, 8, 1], dtype=np.int32)
    return coords, z


def _ethanol_like() -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Small CH3 OH-like geometry (9 atoms; H, C, O only)."""
    # Rough ethanol geometry
    coords = np.array(
        [
            [0.000, 0.000, 0.000],
            [1.232, 0.000, 0.000],
            [1.879, 1.025, 0.000],
            [-0.646, 1.025, 0.000],
            [-0.646, -0.513, 0.889],
            [-0.646, -0.513, -0.889],
            [1.879, -0.513, -0.889],
            [1.879, -0.513, 0.889],
            [3.155, 1.025, 0.000],
        ],
        dtype=np.float64,
    )
    z = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    return coords, z


@pytest.fixture(scope="module")
def gpu_device() -> Any:
    return torch.device("cuda:0")


def _rep_size(nelements: int, nRs2: int, nRs3: int, nFourier: int) -> int:
    return nelements * nRs2 + (nelements * (nelements + 1) // 2) * nRs3 * (2 * nFourier)


# --- shape tests -------------------------------------------------------------


def test_output_shape_single_molecule(gpu_device: Any) -> None:
    """GPU forward: check output shape for a single water molecule."""
    coords, z = _water_system()
    elements = [1, 8]
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch([coords], [z], elements, gpu_device)

    rep = cuda_fchl19_repr.generate_fchl_acsf(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
    )

    assert rep.shape == (1, 3, _rep_size(2, 24, 20, 1))
    assert rep.device.type == "cuda"
    assert rep.dtype == torch.float32


def test_output_shape_batch(gpu_device: Any) -> None:
    """GPU forward: batch of two molecules, different sizes."""
    c1, z1 = _water_system()
    c2, z2 = _ethanol_like()
    elements = [1, 6, 8]
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch([c1, c2], [z1, z2], elements, gpu_device)

    rep = cuda_fchl19_repr.generate_fchl_acsf(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
    )
    max_atoms = max(len(z1), len(z2))
    assert rep.shape == (2, max_atoms, _rep_size(3, 24, 20, 1))


def test_padded_slots_zeroed(gpu_device: Any) -> None:
    """Slots i >= N[m] must be exactly zero."""
    c1, z1 = _water_system()  # 3 atoms
    c2, z2 = _ethanol_like()  # 9 atoms
    elements = [1, 6, 8]
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch([c1, c2], [z1, z2], elements, gpu_device)

    rep = cuda_fchl19_repr.generate_fchl_acsf(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
    )
    rep_cpu = rep.cpu().numpy()

    # Molecule 0 has 3 atoms; slots 3..8 must be zero
    assert np.all(rep_cpu[0, 3:, :] == 0.0), "padded slots of mol-0 not zeroed"
    # Molecule 1 is the largest; no padded slots to check


# --- correctness tests -------------------------------------------------------


def _cpu_rep_for_mol(
    coords: NDArray[np.float64], z: NDArray[np.int32], elements: list[int]
) -> NDArray[np.float64]:
    """CPU float64 representation for a single molecule."""
    return cpu_generate_fchl_acsf(coords, z, elements=elements, **_DEFAULTS)  # type: ignore[call-arg]


def _gpu_rep_for_mol(
    coords: NDArray[np.float64],
    z: NDArray[np.int32],
    elements: list[int],
    device: Any,
) -> NDArray[np.float64]:
    """GPU float32 representation for a single molecule, returned as float64 numpy."""
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch([coords], [z], elements, device)
    rep = cuda_fchl19_repr.generate_fchl_acsf(
        coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
    )
    return rep[0].cpu().numpy().astype(np.float64)


def test_correctness_water(gpu_device: Any) -> None:
    """GPU vs CPU: water molecule — rtol=1e-4, atol=1e-5."""
    coords, z = _water_system()
    elements = [1, 8]

    ref = _cpu_rep_for_mol(coords, z, elements)
    got = _gpu_rep_for_mol(coords, z, elements, gpu_device)

    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5, err_msg="GPU/CPU mismatch for water")


def test_correctness_ethanol_like(gpu_device: Any) -> None:
    """GPU vs CPU: ethanol-like molecule — rtol=1e-4, atol=1e-5."""
    coords, z = _ethanol_like()
    elements = [1, 6, 8]

    ref = _cpu_rep_for_mol(coords, z, elements)
    got = _gpu_rep_for_mol(coords, z, elements, gpu_device)

    np.testing.assert_allclose(
        got, ref, rtol=1e-4, atol=1e-5, err_msg="GPU/CPU mismatch for ethanol-like"
    )


def test_correctness_batch_vs_single(gpu_device: Any) -> None:
    """Batched GPU result must match single-molecule GPU result for each mol."""
    c1, z1 = _water_system()
    c2, z2 = _ethanol_like()
    elements = [1, 6, 8]

    # Batched
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch([c1, c2], [z1, z2], elements, gpu_device)
    rep_batch = (
        cuda_fchl19_repr.generate_fchl_acsf(
            coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
        )
        .cpu()
        .numpy()
    )

    # Individual
    cg1, qg1, ng1 = _build_gpu_batch([c1], [z1], elements, gpu_device)
    rep1 = (
        cuda_fchl19_repr.generate_fchl_acsf(cg1, qg1, ng1, nelements=len(elements), **_DEFAULTS)
        .cpu()
        .numpy()
    )

    cg2, qg2, ng2 = _build_gpu_batch([c2], [z2], elements, gpu_device)
    rep2 = (
        cuda_fchl19_repr.generate_fchl_acsf(cg2, qg2, ng2, nelements=len(elements), **_DEFAULTS)
        .cpu()
        .numpy()
    )

    np.testing.assert_array_equal(rep_batch[0, : len(z1), :], rep1[0, : len(z1), :])
    np.testing.assert_array_equal(rep_batch[1, : len(z2), :], rep2[0, : len(z2), :])


def test_translation_invariance(gpu_device: Any) -> None:
    """Representation must not change under rigid translation."""
    coords, z = _ethanol_like()
    elements = [1, 6, 8]
    shift = np.array([3.7, -2.1, 1.5])

    rep_orig = _gpu_rep_for_mol(coords, z, elements, gpu_device)
    rep_shifted = _gpu_rep_for_mol(coords + shift, z, elements, gpu_device)

    np.testing.assert_allclose(
        rep_shifted,
        rep_orig,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Representation changed under translation",
    )


def test_correctness_small_mols_mini(gpu_device: Any) -> None:
    """GPU vs CPU: bundled small_mols_mini subset with variable atom counts."""
    from kernelforge.kernelcli import load_small_mols_mini

    coords_list, z_list, *_ = load_small_mols_mini(n_train=20, n_test=0)
    elements = [1, 6, 7, 8]
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch(coords_list, z_list, elements, gpu_device)

    got_batch = (
        cuda_fchl19_repr.generate_fchl_acsf(
            coords_gpu, Q_gpu, N_gpu, nelements=len(elements), **_DEFAULTS
        )
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    for idx, (coords, z) in enumerate(zip(coords_list, z_list, strict=False)):
        ref = _cpu_rep_for_mol(coords, z, elements)
        got = got_batch[idx, : len(z), :]
        np.testing.assert_allclose(
            got,
            ref,
            rtol=5e-4,
            atol=5e-5,
            err_msg=f"GPU/CPU mismatch on small_mols_mini molecule {idx}",
        )


# ---------------------------------------------------------------------------
# QM7B timing benchmark (slow — skipped unless -m slow or -k qm7b)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_qm7b_correctness_and_timing(gpu_device: Any) -> None:
    """GPU vs CPU correctness on 20 QM7B molecules + timing on 200 molecules.

    Requires QM7b dataset cached at ~/.kernelforge/datasets/qm7b_complete.npz.
    """
    from kernelforge.cli import load_qm7b_raw_data

    data = load_qm7b_raw_data()
    R_all = data["R"]
    z_all = data["z"]
    elements = _ELEMENTS_QM7B
    nelements = len(elements)

    # ------------------------------------------------------------------
    # Correctness: 20 molecules
    # ------------------------------------------------------------------
    N_CORRECT = 20
    print(f"\n[QM7B correctness] checking {N_CORRECT} molecules GPU vs CPU ...")

    max_err_abs = 0.0
    max_err_rel = 0.0

    for idx in range(N_CORRECT):
        coords = R_all[idx].astype(np.float64)
        z = z_all[idx].astype(np.int32)

        ref = _cpu_rep_for_mol(coords, z, elements)
        got = _gpu_rep_for_mol(coords, z, elements, gpu_device)

        abs_err = float(np.max(np.abs(got - ref)))
        # relative: avoid div-by-zero on near-zero entries
        denom = np.maximum(np.abs(ref), 1e-10)
        rel_err = float(np.max(np.abs(got - ref) / denom))
        max_err_abs = max(max_err_abs, abs_err)
        max_err_rel = max(max_err_rel, rel_err)

        # float32 (GPU, --use_fast_math) vs float64 (CPU): relax tolerances
        # slightly beyond the toy-molecule thresholds to account for FMA and
        # fast-reciprocal accumulation differences over many atoms.
        np.testing.assert_allclose(
            got, ref, rtol=5e-4, atol=5e-5, err_msg=f"GPU/CPU mismatch on QM7B molecule {idx}"
        )

    print(f"  max abs err: {max_err_abs:.3e}  max rel err: {max_err_rel:.3e}  [PASS]")

    # ------------------------------------------------------------------
    # Timing: 200 molecules batched GPU vs sequential CPU
    # ------------------------------------------------------------------
    N_TIMING = 200
    coords_list = [R_all[i].astype(np.float64) for i in range(N_TIMING)]
    z_list = [z_all[i].astype(np.int32) for i in range(N_TIMING)]

    # CPU timing
    t0 = time.perf_counter()
    for coords, z in zip(coords_list, z_list, strict=False):
        cpu_generate_fchl_acsf(coords, z, elements=elements, **_DEFAULTS)  # type: ignore[call-arg]
    t_cpu = time.perf_counter() - t0

    # GPU timing (warm-up then measure)
    coords_gpu, Q_gpu, N_gpu = _build_gpu_batch(coords_list, z_list, elements, gpu_device)

    # Warm-up
    _ = cuda_fchl19_repr.generate_fchl_acsf(
        coords_gpu, Q_gpu, N_gpu, nelements=nelements, **_DEFAULTS
    )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    rep_gpu = cuda_fchl19_repr.generate_fchl_acsf(
        coords_gpu, Q_gpu, N_gpu, nelements=nelements, **_DEFAULTS
    )
    torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t0

    speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")

    print(f"\n[QM7B timing — {N_TIMING} molecules]")
    print(f"  CPU (sequential, float64): {t_cpu * 1e3:.1f} ms")
    print(f"  GPU (batched,   float32):  {t_gpu * 1e3:.1f} ms")
    print(f"  Speedup: {speedup:.1f}x")

    assert rep_gpu.shape[0] == N_TIMING
    assert rep_gpu.shape[2] == _rep_size(nelements, 24, 20, 1)
