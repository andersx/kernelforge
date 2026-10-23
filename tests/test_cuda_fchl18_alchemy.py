"""CUDA FCHL18 periodic-table alchemy: CPU <-> CUDA parity and model smoke test."""

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


def _cu(a: np.ndarray, dtype: Any = np.float64) -> Any:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=dtype))).cuda()


def _side(coords: list[np.ndarray], zs: list[np.ndarray]) -> dict[str, Any]:
    max_size = max(len(z) for z in zs)
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
        "x_np": x,
        "n_np": n,
        "nn_np": nn,
    }


A_COORDS = [WATER, HF, AMMONIA]
A_Z = [WATER_Z, HF_Z, AMMONIA_Z]
B_COORDS = [HF, AMMONIA]
B_Z = [HF_Z, AMMONIA_Z]


@pytest.mark.parametrize("alch", [ALCH, ALCH_ODD])
def test_scalar_alchemy_matches_cpu(alch: dict[str, Any]) -> None:
    a = _side(A_COORDS, A_Z)
    b = _side(B_COORDS, B_Z)
    k_gpu = cuda_kernel.kernel_gaussian(
        a["x"], b["x"], a["n"], b["n"], a["nn"], b["nn"], sigma=SIGMA, **KP, **alch
    )
    k_cpu = cpu_kernel.kernel_gaussian(
        a["x_np"],
        b["x_np"],
        a["n_np"],
        b["n_np"],
        a["nn_np"],
        b["nn_np"],
        sigma=SIGMA,
        **KP,
        **alch,
    )
    k_off = cpu_kernel.kernel_gaussian(
        a["x_np"],
        b["x_np"],
        a["n_np"],
        b["n_np"],
        a["nn_np"],
        b["nn_np"],
        sigma=SIGMA,
        **KP,
    )
    assert not np.allclose(k_cpu, k_off)
    assert np.max(np.abs(k_gpu.cpu().numpy() - k_cpu)) < 1e-8


def test_scalar_symm_alchemy_matches_cpu() -> None:
    a = _side(A_COORDS, A_Z)
    k_gpu = cuda_kernel.kernel_gaussian_symm(a["x"], a["n"], a["nn"], sigma=SIGMA, **KP, **ALCH)
    k_rect = cuda_kernel.kernel_gaussian(
        a["x"], a["x"], a["n"], a["n"], a["nn"], a["nn"], sigma=SIGMA, **KP, **ALCH
    )
    k_cpu = cpu_kernel.kernel_gaussian(
        a["x_np"],
        a["x_np"],
        a["n_np"],
        a["n_np"],
        a["nn_np"],
        a["nn_np"],
        sigma=SIGMA,
        **KP,
        **ALCH,
    )
    assert np.max(np.abs(k_gpu.cpu().numpy() - k_cpu)) < 1e-8
    assert np.max(np.abs(k_rect.cpu().numpy() - k_cpu)) < 1e-8


def test_alchemy_off_cuda_is_default() -> None:
    a = _side(A_COORDS, A_Z)
    b = _side(B_COORDS, B_Z)
    args = (a["x"], b["x"], a["n"], b["n"], a["nn"], b["nn"])
    k0 = cuda_kernel.kernel_gaussian(*args, sigma=SIGMA, **KP).cpu().numpy()
    k1 = cuda_kernel.kernel_gaussian(*args, sigma=SIGMA, **KP, use_alchemy=False).cpu().numpy()
    assert np.array_equal(k0, k1)
    h0 = cuda_kernel.kernel_gaussian_hessian_symm(
        a["x"], a["n"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP
    )
    h1 = cuda_kernel.kernel_gaussian_hessian_symm(
        a["x"], a["n"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP, use_alchemy=False
    )
    np.testing.assert_allclose(h0.cpu().numpy(), h1.cpu().numpy(), rtol=1e-14, atol=1e-14)
    # Alchemy-on differs from alchemy-off.
    k2 = cuda_kernel.kernel_gaussian(*args, sigma=SIGMA, **KP, **ALCH).cpu().numpy()
    assert not np.allclose(k0, k2)


def test_gradient_alchemy_matches_cpu() -> None:
    b = _side(B_COORDS, B_Z)
    q = _side([WATER], [WATER_Z])
    g_gpu = cuda_kernel.kernel_gaussian_gradient(
        q["x"], b["x"], q["n"], b["n"], q["nn"], b["nn"], q["c"], q["z"], sigma=SIGMA, **KP, **ALCH
    )
    g_cpu = cpu_kernel.kernel_gaussian_gradient(
        WATER, WATER_Z, b["x_np"], b["n_np"], b["nn_np"], sigma=SIGMA, **KP, **ALCH
    )
    assert np.max(np.abs(g_gpu.cpu().numpy() - g_cpu)) < 1e-8


def test_jacobian_alchemy_matches_cpu() -> None:
    a = _side(A_COORDS, A_Z)
    b = _side(B_COORDS, B_Z)
    j_gpu = cuda_kernel.kernel_gaussian_jacobian(
        a["x"], b["x"], a["n"], b["n"], a["nn"], b["nn"], a["c"], a["z"], sigma=SIGMA, **KP, **ALCH
    )
    j_cpu = cpu_kernel.kernel_gaussian_jacobian(
        A_COORDS, A_Z, b["x_np"], b["n_np"], b["nn_np"], sigma=SIGMA, **KP, **ALCH
    )
    assert np.max(np.abs(j_gpu.cpu().numpy() - j_cpu)) < 1e-8


def test_jacobian_t_alchemy_is_transpose_of_jacobian() -> None:
    """CUDA alchemy: jacobian(A,A) == jacobian_t(A,A).T, and matches CPU."""
    a = _side(A_COORDS, A_Z)
    j = cuda_kernel.kernel_gaussian_jacobian(
        a["x"], a["x"], a["n"], a["n"], a["nn"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP, **ALCH
    )
    jt = cuda_kernel.kernel_gaussian_jacobian_t(
        a["x"], a["x"], a["n"], a["n"], a["nn"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP, **ALCH
    )
    np.testing.assert_allclose(j.cpu().numpy(), jt.cpu().numpy().T, rtol=1e-10, atol=1e-12)

    j_cpu = cpu_kernel.kernel_gaussian_jacobian(
        A_COORDS, A_Z, a["x_np"], a["n_np"], a["nn_np"], sigma=SIGMA, **KP, **ALCH
    )
    jt_cpu = cpu_kernel.kernel_gaussian_jacobian_t(
        A_COORDS, A_Z, a["x_np"], a["n_np"], a["nn_np"], sigma=SIGMA, **KP, **ALCH
    )
    assert np.max(np.abs(jt.cpu().numpy() - jt_cpu)) < 1e-8
    np.testing.assert_allclose(j_cpu, jt_cpu.T, rtol=1e-10, atol=1e-12)


def test_rfp_alchemy_unpacks_to_dense() -> None:
    """CUDA alchemy: scalar / Hessian / full symm RFP unpack to dense symm."""
    from kernelforge import kernelmath

    coords, zs = [WATER, HF], [WATER_Z, HF_Z]
    a = _side(coords, zs)
    nm = len(coords)

    k = cuda_kernel.kernel_gaussian_symm(a["x"], a["n"], a["nn"], sigma=SIGMA, **KP, **ALCH)
    rfp = cuda_kernel.kernel_gaussian_symm_rfp(a["x"], a["n"], a["nn"], sigma=SIGMA, **KP, **ALCH)
    np.testing.assert_allclose(
        kernelmath.rfp_to_full(rfp.cpu().numpy(), nm, uplo="L", transr="N"),
        k.cpu().numpy(),
        rtol=1e-12,
        atol=1e-14,
    )

    h = cuda_kernel.kernel_gaussian_hessian_symm(
        a["x"], a["n"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP, **ALCH
    )
    h_rfp = cuda_kernel.kernel_gaussian_hessian_symm_rfp(
        a["x"], a["n"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP, **ALCH
    )
    d = int(h.shape[0])
    np.testing.assert_allclose(
        kernelmath.rfp_to_full(h_rfp.cpu().numpy(), d, uplo="L", transr="N"),
        h.cpu().numpy(),
        rtol=1e-12,
        atol=1e-14,
    )

    kf = cuda_kernel.kernel_gaussian_full_symm(
        a["x"], a["n"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP, **ALCH
    )
    kf_rfp = cuda_kernel.kernel_gaussian_full_symm_rfp(
        a["x"], a["n"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP, **ALCH
    )
    big = int(kf.shape[0])
    np.testing.assert_allclose(
        kernelmath.rfp_to_full(kf_rfp.cpu().numpy(), big, uplo="L", transr="N"),
        kf.cpu().numpy(),
        rtol=1e-12,
        atol=1e-14,
    )


def test_hessian_symm_alchemy_matches_cpu() -> None:
    coords, zs = [WATER, HF], [WATER_Z, HF_Z]
    a = _side(coords, zs)
    h_gpu = cuda_kernel.kernel_gaussian_hessian_symm(
        a["x"], a["n"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP, **ALCH
    )
    h_cpu = cpu_kernel.kernel_gaussian_hessian_symm(coords, zs, sigma=SIGMA, **KP, **ALCH)
    assert np.max(np.abs(h_gpu.cpu().numpy() - h_cpu)) < 1e-8


def test_hessian_rect_alchemy_matches_cpu() -> None:
    a = _side([WATER, HF], [WATER_Z, HF_Z])
    b = _side(B_COORDS, B_Z)
    h_gpu = cuda_kernel.kernel_gaussian_hessian(
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
        sigma=SIGMA,
        **KP,
        **ALCH,
    )
    h_cpu = cpu_kernel.kernel_gaussian_hessian(
        [WATER, HF], [WATER_Z, HF_Z], B_COORDS, B_Z, sigma=SIGMA, **KP, **ALCH
    )
    assert np.max(np.abs(h_gpu.cpu().numpy() - h_cpu)) < 1e-8


def test_full_symm_alchemy_matches_cpu() -> None:
    coords, zs = [WATER, HF], [WATER_Z, HF_Z]
    a = _side(coords, zs)
    k_gpu = cuda_kernel.kernel_gaussian_full_symm(
        a["x"], a["n"], a["nn"], a["c"], a["z"], sigma=SIGMA, **KP, **ALCH
    )
    k_cpu = cpu_kernel.kernel_gaussian_full_symm(coords, zs, sigma=SIGMA, **KP, **ALCH)
    assert np.max(np.abs(k_gpu.cpu().numpy() - k_cpu)) < 1e-8


def test_full_alchemy_matches_cpu() -> None:
    a = _side([WATER, HF], [WATER_Z, HF_Z])
    b = _side(B_COORDS, B_Z)
    k_gpu = cuda_kernel.kernel_gaussian_full(
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
        sigma=SIGMA,
        **KP,
        **ALCH,
    )
    k_cpu = cpu_kernel.kernel_gaussian_full(
        [WATER, HF], [WATER_Z, HF_Z], B_COORDS, B_Z, sigma=SIGMA, **KP, **ALCH
    )
    assert np.max(np.abs(k_gpu.cpu().numpy() - k_cpu)) < 1e-8


def test_cuda_krr_energy_and_force_alchemy_smoke() -> None:
    from kernelforge.models.cuda_fchl18_krr import CudaFCHL18KRRModel

    rng = np.random.default_rng(0)
    ethanol_z = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    base = rng.normal(scale=1.2, size=(len(ethanol_z), 3))
    n_train = 8
    coords = [base + rng.normal(scale=0.05, size=base.shape) for _ in range(n_train)]
    zs = [ethanol_z] * n_train
    energies = rng.normal(size=n_train)
    forces = [rng.normal(size=base.shape) for _ in range(n_train)]

    kp = {
        "two_body_scaling": float(np.sqrt(8)),
        "three_body_scaling": 1.6,
        "two_body_width": 0.2,
        "three_body_width": float(np.pi),
        "two_body_power": 4.0,
        "three_body_power": 2.0,
        "cut_start": 1.0,
        "cut_distance": 5.0,
        "fourier_order": 1,
        "use_atm": True,
        "use_alchemy": True,
        "alchemy_period_width": 1.6,
        "alchemy_group_width": 1.6,
    }
    model = CudaFCHL18KRRModel(sigma=2.5, l2=1e-6, max_size=9, kernel_params=kp)
    model.fit(coords, zs, energies=energies, forces=forces)
    e, f = model.predict(coords[:2], zs[:2])
    assert np.all(np.isfinite(e))
    assert np.all(np.isfinite(np.asarray(f)))
