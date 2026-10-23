"""CUDA FCHL18 matvec kernels with periodic-table alchemy vs dense references.

``kernel_gaussian_full_matvec`` takes the fused path whenever ``nm_train > nm_query``
(and the unfused path otherwise); both must agree with the dense kernel @ alpha.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

import numpy as np
import pytest

cuda_kernel = pytest.importorskip(
    "kernelforge.cuda_fchl18_kernel",
    reason="cuda_fchl18_kernel not built (requires CUDA + PyTorch at build time)",
)
torch = pytest.importorskip("torch", reason="PyTorch not installed")

import kernelforge.fchl18_kernel as cpu_kernel  # noqa: E402
import kernelforge.fchl18_repr as repr_mod  # noqa: E402

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

pytestmark = pytest.mark.skipif(not _gpu_ok, reason="No NVIDIA GPU detected by nvidia-smi")

WATER = np.array([[0.0, 0.0, 0.119], [0.0, 0.757, -0.477], [0.0, -0.757, -0.477]])
WATER_Z = np.array([8, 1, 1], dtype=np.int32)
WATER2 = WATER + np.array([[0.02, -0.01, 0.0], [0.0, 0.03, 0.01], [-0.01, 0.0, 0.02]])
HF = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]])
HF_Z = np.array([9, 1], dtype=np.int32)
AMMONIA = np.array(
    [
        [0.000, 0.000, 0.116],
        [0.000, 0.939, -0.271],
        [0.813, -0.469, -0.271],
        [-0.813, -0.469, -0.271],
    ]
)
AMMONIA_Z = np.array([7, 1, 1, 1], dtype=np.int32)

SIGMA = 2.5
KP: dict[str, Any] = {
    "two_body_scaling": 2.0,
    "two_body_width": 0.1,
    "two_body_power": 6.0,
    "three_body_scaling": 2.0,
    "three_body_width": 3.0,
    "three_body_power": 3.0,
    "cut_start": 1.0,
    "cut_distance": 1e6,
    "fourier_order": 1,
    "use_atm": True,
}
ALCH: dict[str, Any] = {
    "use_alchemy": True,
    "alchemy_period_width": 1.6,
    "alchemy_group_width": 1.6,
}
ALCH_ODD: dict[str, Any] = {
    "use_alchemy": True,
    "alchemy_period_width": 0.9,
    "alchemy_group_width": 2.3,
}

# 4 training molecules (> any query set below -> fused path).
TRAIN_COORDS = [WATER, HF, AMMONIA, WATER2]
TRAIN_Z = [WATER_Z, HF_Z, AMMONIA_Z, WATER_Z]
QUERY1_COORDS = [WATER2]
QUERY1_Z = [WATER_Z]
QUERY2_COORDS = [HF, AMMONIA]
QUERY2_Z = [HF_Z, AMMONIA_Z]

TOL = 1e-9


def _cu(a: np.ndarray, dtype: Any = np.float64) -> Any:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=dtype))).cuda()


def _side(
    coords: list[np.ndarray], zs: list[np.ndarray], max_size: int = 4, fo: int = 1
) -> dict[str, Any]:
    del fo
    x, n, nn = repr_mod.generate(coords, zs, max_size=max_size, cut_distance=KP["cut_distance"])
    cp = np.zeros((len(zs), max_size, 3))
    zp = np.zeros((len(zs), max_size), dtype=np.int32)
    for i, (c, z) in enumerate(zip(coords, zs, strict=True)):
        cp[i, : len(z)] = c
        zp[i, : len(z)] = z
    return {
        "x": _cu(x),
        "n": _cu(n, np.int32),
        "nn": _cu(nn, np.int32),
        "c": _cu(cp),
        "z": _cu(zp, np.int32),
        "D": int(np.sum(n) * 3),
        "nm": len(zs),
    }


def _pair_args(a: dict[str, Any], b: dict[str, Any]) -> tuple[Any, ...]:
    return (
        a["x"],
        b["x"],
        a["n"],
        b["n"],
        a["nn"],
        b["nn"],
        a["c"],
        a["z"],
        b["c"],
        b["z"],
    )


def _alphas(b: dict[str, Any], seed: int = 0) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    return _cu(rng.standard_normal(b["nm"])), _cu(rng.standard_normal(b["D"]))


def _kp(fo: int = 1, **extra: Any) -> dict[str, Any]:
    return {**KP, "fourier_order": fo, **extra}


def _max_abs(a: Any, b: Any) -> float:
    return float(np.max(np.abs(a.cpu().numpy() - b.cpu().numpy())))


@pytest.mark.parametrize("fo", [1, 2])
@pytest.mark.parametrize("alch", [ALCH, ALCH_ODD])
@pytest.mark.parametrize(
    ("q_coords", "q_z"), [(QUERY1_COORDS, QUERY1_Z), (QUERY2_COORDS, QUERY2_Z)]
)
def test_full_matvec_fused_alchemy_matches_dense(
    fo: int, alch: dict[str, Any], q_coords: list[np.ndarray], q_z: list[np.ndarray]
) -> None:
    """nm_train (4) > nm_query -> fused path; must equal dense full @ alpha."""
    a = _side(q_coords, q_z)
    b = _side(TRAIN_COORDS, TRAIN_Z)
    assert b["nm"] > a["nm"]
    alpha_E, alpha_F = _alphas(b)
    kp = _kp(fo, **alch)
    k_full = cuda_kernel.kernel_gaussian_full(*_pair_args(a, b), sigma=SIGMA, **kp)
    y_ref = k_full @ torch.cat([alpha_E, alpha_F])
    e_mv, f_mv = cuda_kernel.kernel_gaussian_full_matvec(
        *_pair_args(a, b), alpha_E, alpha_F, sigma=SIGMA, compute_energy=True, **kp
    )
    assert _max_abs(e_mv, y_ref[: a["nm"]]) < TOL
    assert _max_abs(f_mv, y_ref[a["nm"] :]) < TOL


def test_full_matvec_fused_alchemy_differs_from_off() -> None:
    a = _side(QUERY2_COORDS, QUERY2_Z)
    b = _side(TRAIN_COORDS, TRAIN_Z)
    alpha_E, alpha_F = _alphas(b)
    args = (*_pair_args(a, b), alpha_E, alpha_F)
    e0, f0 = cuda_kernel.kernel_gaussian_full_matvec(*args, sigma=SIGMA, **_kp(1))
    e1, f1 = cuda_kernel.kernel_gaussian_full_matvec(*args, sigma=SIGMA, **_kp(1, **ALCH))
    assert _max_abs(e0, e1) > 1e-6
    assert _max_abs(f0, f1) > 1e-6


def test_full_matvec_unfused_alchemy_matches_dense() -> None:
    """nm_train (2) <= nm_query (4) -> unfused path; still matches dense."""
    a = _side(TRAIN_COORDS, TRAIN_Z)
    b = _side(QUERY2_COORDS, QUERY2_Z)
    assert b["nm"] <= a["nm"]
    alpha_E, alpha_F = _alphas(b, seed=1)
    kp = _kp(1, **ALCH)
    k_full = cuda_kernel.kernel_gaussian_full(*_pair_args(a, b), sigma=SIGMA, **kp)
    y_ref = k_full @ torch.cat([alpha_E, alpha_F])
    e_mv, f_mv = cuda_kernel.kernel_gaussian_full_matvec(
        *_pair_args(a, b), alpha_E, alpha_F, sigma=SIGMA, **kp
    )
    assert _max_abs(e_mv, y_ref[: a["nm"]]) < TOL
    assert _max_abs(f_mv, y_ref[a["nm"] :]) < TOL


@pytest.mark.parametrize("alch", [ALCH, ALCH_ODD])
def test_full_matvec_fused_equals_unfused_alchemy(alch: dict[str, Any]) -> None:
    """Same (query, train) pair through fused (nm_tr > nm_q) and unfused (padded query).

    The unfused path is forced by appending duplicate query molecules so nm_q >= nm_tr; the
    first rows of the result must equal the fused result.
    """
    q = _side(QUERY1_COORDS, QUERY1_Z)
    q_pad = _side(QUERY1_COORDS * 4, QUERY1_Z * 4)  # nm_q == nm_tr == 4 -> unfused
    b = _side(TRAIN_COORDS, TRAIN_Z)
    assert q["nm"] < b["nm"]
    assert q_pad["nm"] >= b["nm"]
    alpha_E, alpha_F = _alphas(b, seed=2)
    kp = _kp(1, **alch)
    e_f, f_f = cuda_kernel.kernel_gaussian_full_matvec(
        *_pair_args(q, b), alpha_E, alpha_F, sigma=SIGMA, **kp
    )
    e_u, f_u = cuda_kernel.kernel_gaussian_full_matvec(
        *_pair_args(q_pad, b), alpha_E, alpha_F, sigma=SIGMA, **kp
    )
    n_f = q["D"]
    assert _max_abs(e_f, e_u[:1]) < TOL
    assert _max_abs(f_f, f_u[:n_f]) < TOL


def test_full_matvec_fused_alchemy_compute_energy_false() -> None:
    a = _side(QUERY2_COORDS, QUERY2_Z)
    b = _side(TRAIN_COORDS, TRAIN_Z)
    alpha_E, alpha_F = _alphas(b, seed=3)
    kp = _kp(1, **ALCH)
    _, f_with = cuda_kernel.kernel_gaussian_full_matvec(
        *_pair_args(a, b), alpha_E, alpha_F, sigma=SIGMA, compute_energy=True, **kp
    )
    _, f_wo = cuda_kernel.kernel_gaussian_full_matvec(
        *_pair_args(a, b), alpha_E, alpha_F, sigma=SIGMA, compute_energy=False, **kp
    )
    assert _max_abs(f_with, f_wo) < TOL


def test_full_matvec_fused_alchemy_cache_invalidation() -> None:
    """Repeated fused calls on the same train set while toggling alchemy / widths."""
    a = _side(QUERY2_COORDS, QUERY2_Z)
    b = _side(TRAIN_COORDS, TRAIN_Z)
    alpha_E, alpha_F = _alphas(b, seed=4)
    alpha = torch.cat([alpha_E, alpha_F])
    for kp_alch in (ALCH, {}, ALCH_ODD, ALCH, {}, ALCH):
        kp = _kp(1, **kp_alch)
        k_full = cuda_kernel.kernel_gaussian_full(*_pair_args(a, b), sigma=SIGMA, **kp)
        y_ref = k_full @ alpha
        e_mv, f_mv = cuda_kernel.kernel_gaussian_full_matvec(
            *_pair_args(a, b), alpha_E, alpha_F, sigma=SIGMA, **kp
        )
        assert _max_abs(e_mv, y_ref[: a["nm"]]) < TOL
        assert _max_abs(f_mv, y_ref[a["nm"] :]) < TOL


def test_full_matvec_fused_alchemy_matches_cpu() -> None:
    a = _side(QUERY2_COORDS, QUERY2_Z)
    b = _side(TRAIN_COORDS, TRAIN_Z)
    alpha_E, alpha_F = _alphas(b, seed=5)
    kp = _kp(1, **ALCH)
    e_gpu, f_gpu = cuda_kernel.kernel_gaussian_full_matvec(
        *_pair_args(a, b), alpha_E, alpha_F, sigma=SIGMA, **kp
    )
    e_cpu, f_cpu = cpu_kernel.kernel_gaussian_full_matvec(
        QUERY2_COORDS,
        QUERY2_Z,
        TRAIN_COORDS,
        TRAIN_Z,
        alpha_E.cpu().numpy(),
        alpha_F.cpu().numpy(),
        sigma=SIGMA,
        **kp,
    )
    assert np.max(np.abs(e_gpu.cpu().numpy() - e_cpu)) < 1e-8
    assert np.max(np.abs(f_gpu.cpu().numpy() - f_cpu)) < 1e-8


@pytest.mark.parametrize("fo", [1, 2])
@pytest.mark.parametrize("alch", [ALCH, ALCH_ODD])
def test_hessian_matvec_alchemy_matches_dense(fo: int, alch: dict[str, Any]) -> None:
    a = _side(QUERY2_COORDS, QUERY2_Z)
    b = _side(TRAIN_COORDS, TRAIN_Z)
    _, alpha_F = _alphas(b, seed=6)
    kp = _kp(fo, **alch)
    h = cuda_kernel.kernel_gaussian_hessian(*_pair_args(a, b), sigma=SIGMA, **kp)
    f_mv = cuda_kernel.kernel_gaussian_hessian_matvec(*_pair_args(a, b), alpha_F, sigma=SIGMA, **kp)
    assert _max_abs(f_mv, h @ alpha_F) < TOL


@pytest.mark.parametrize("alch", [ALCH, ALCH_ODD])
def test_jacobian_t_matvec_alchemy_matches_dense(alch: dict[str, Any]) -> None:
    train = _side(TRAIN_COORDS, TRAIN_Z)
    test = _side(QUERY2_COORDS, QUERY2_Z)
    _, alpha_F = _alphas(train, seed=7)
    kp = _kp(1, **alch)
    jt = cuda_kernel.kernel_gaussian_jacobian_t(
        train["x"],
        test["x"],
        train["n"],
        test["n"],
        train["nn"],
        test["nn"],
        train["c"],
        train["z"],
        sigma=SIGMA,
        **kp,
    )
    e_mv = cuda_kernel.kernel_gaussian_jacobian_t_matvec(
        train["x"],
        test["x"],
        train["n"],
        test["n"],
        train["nn"],
        test["nn"],
        train["c"],
        train["z"],
        alpha_F,
        sigma=SIGMA,
        **kp,
    )
    assert _max_abs(e_mv, jt @ alpha_F) < TOL
