"""FCHL18 parity vs qmllib with use_atm=True (alchemy='off').

qmllib is optional; tests are skipped when it (or the QM7b dataset) is missing.

Tolerances:
  - scalar: tight (1e-7)
  - jacobian: qmllib uses single FD (dx=0.005 Å) → ~3e-3 rel
  - hessian: qmllib double-FD is noise-limited; not compared here.
    See examples/fchl18_qmllib_parity_check.py for the FD Hessian caveat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

qmllib = pytest.importorskip("qmllib", reason="qmllib not installed")

from qmllib.representations.fchl import fchl_force_kernels as ffk  # noqa: E402
from qmllib.representations.fchl import fchl_scalar_kernels as fsk  # noqa: E402
from qmllib.representations.fchl import generate_fchl18  # noqa: E402
from qmllib.representations.fchl.fchl_representations import (  # noqa: E402
    generate_fchl18_displaced,
)

import kernelforge.fchl18_kernel as km  # noqa: E402
import kernelforge.fchl18_repr as rm  # noqa: E402

pytestmark = pytest.mark.integration

SIGMA = 2.0
CUT_DISTANCE = 5.0
KARGS = dict(
    two_body_scaling=np.sqrt(8.0),
    two_body_width=0.2,
    two_body_power=4.0,
    three_body_scaling=1.6,
    three_body_width=np.pi,
    three_body_power=2.0,
    cut_start=1.0,
    cut_distance=CUT_DISTANCE,
    fourier_order=1,
    use_atm=True,
)

KERN_RTOL = 1e-7
KERN_ATOL = 1e-7
GRAD_RTOL = 3e-3
GRAD_ATOL = 5e-6
GRAD_DX = 0.005
N_MOLS = 5

_QM7B = Path.home() / ".kernelforge" / "datasets" / "qm7b_complete.npz"


def _load_qm7b(n: int) -> tuple[list, list]:
    if not _QM7B.is_file():
        pytest.skip(f"QM7b dataset not found at {_QM7B}")
    ds = np.load(_QM7B, allow_pickle=True)
    coords_list = [ds["R"][i] for i in range(n)]
    charges_list = [ds["z"][i].astype(int) for i in range(n)]
    return coords_list, charges_list


def _build_kf_repr(coords_list: list, charges_list: list):
    max_size = max(len(c) for c in charges_list)
    return rm.generate(
        [r.tolist() for r in coords_list],
        [z.tolist() for z in charges_list],
        max_size=max_size,
        cut_distance=CUT_DISTANCE,
    )


def test_qmllib_scalar_symm_atm() -> None:
    coords_list, charges_list = _load_qm7b(N_MOLS)
    max_size = max(len(c) for c in charges_list)
    X_qml = np.array(
        [
            generate_fchl18(z, r, max_size=max_size, cut_distance=CUT_DISTANCE)
            for z, r in zip(charges_list, coords_list, strict=True)
        ]
    )
    x_kf, n_kf, nn_kf = _build_kf_repr(coords_list, charges_list)

    K_qml = fsk.get_local_symmetric_kernels(
        X_qml, alchemy="off", kernel_args={"sigma": [SIGMA]}
    )[0]
    K_kf = km.kernel_gaussian_symm(x_kf, n_kf, nn_kf, sigma=SIGMA, **KARGS)
    np.testing.assert_allclose(K_kf, K_qml, rtol=KERN_RTOL, atol=KERN_ATOL)

    K_off = km.kernel_gaussian_symm(x_kf, n_kf, nn_kf, sigma=SIGMA, **{**KARGS, "use_atm": False})
    assert not np.allclose(K_kf, K_off, rtol=1e-8, atol=1e-10)


def test_qmllib_jacobian_atm_fd() -> None:
    """Analytic KF jacobian vs qmllib single-sided FD (dx=0.005)."""
    coords_all, charges_all = _load_qm7b(N_MOLS)
    nA_set, nB_set = 3, N_MOLS - 3
    n_atoms = [len(c) for c in charges_all]
    max_size = max(n_atoms)

    X_qml_A = np.array(
        [
            generate_fchl18(
                charges_all[i], coords_all[i], max_size=max_size, cut_distance=CUT_DISTANCE
            )
            for i in range(nA_set)
        ]
    )
    X_qml_B_disp = np.array(
        [
            generate_fchl18_displaced(
                charges_all[nA_set + j],
                coords_all[nA_set + j],
                max_size=max_size,
                cut_distance=CUT_DISTANCE,
                dx=GRAD_DX,
            )
            for j in range(nB_set)
        ]
    )
    x_kf_A, n_kf_A, nn_kf_A = _build_kf_repr(coords_all[:nA_set], charges_all[:nA_set])

    dof_offsets = [0]
    for j in range(nB_set):
        dof_offsets.append(dof_offsets[-1] + 3 * n_atoms[nA_set + j])

    qml_kargs_g = dict(
        alchemy="off",
        two_body_scaling=np.sqrt(8.0),
        two_body_width=0.2,
        two_body_power=4.0,
        three_body_scaling=1.6,
        three_body_width=np.pi,
        three_body_power=2.0,
        cut_start=1.0,
        cut_distance=CUT_DISTANCE,
        fourier_order=1,
        kernel_args={"sigma": [SIGMA]},
    )
    J_qml = ffk.get_local_gradient_kernels(X_qml_A, X_qml_B_disp, **qml_kargs_g)[0]

    for j in range(nB_set):
        n_Bj = n_atoms[nA_set + j]
        J_kf_j = km.kernel_gaussian_gradient(
            coords_all[nA_set + j],
            charges_all[nA_set + j].astype(np.int32),
            x_kf_A,
            n_kf_A,
            nn_kf_A,
            sigma=SIGMA,
            **KARGS,
        )
        for a in range(nA_set):
            kf_flat = J_kf_j[:, :, a].reshape(-1)
            qml_flat = J_qml[a, dof_offsets[j] : dof_offsets[j + 1]]
            np.testing.assert_allclose(kf_flat, qml_flat, rtol=GRAD_RTOL, atol=GRAD_ATOL)
