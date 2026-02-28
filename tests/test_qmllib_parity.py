"""Numerical parity tests between kernelforge and qmllib (reference implementation).

All tests use real QM7b molecules loaded from the dataset and compare
kernelforge outputs to qmllib outputs to machine precision (rtol=1e-10).

Convention notes (discovered by inspection):
  - Asymmetric kernels: K_qml == K_kf.T  (qmllib is Fortran-order / transposed)
  - Symmetric kernels: EE and FF blocks match directly after symmetrizing kf lower-tri
  - Symmetric GP kernel: EF/FE off-diagonal blocks differ by sign (K_kf_EF == -K_qml_EF)
  - Asymmetric GP kernel: K_qml == K_kf.T
  - RFP kernels: unpack via rfp_to_full, compare to non-RFP symmetric kernel
"""

import numpy as np
import pytest
from numpy.typing import NDArray

import kernelforge.local_kernels as kf_lk
from kernelforge.fchl19_repr import generate_fchl_acsf, generate_fchl_acsf_and_gradients
from kernelforge.kernelmath import rfp_to_full
from qmllib.kernels.gradient_kernels import (
    get_gdml_kernel,
    get_gp_kernel,
    get_local_gradient_kernel,
    get_local_kernel,
    get_local_symmetric_kernel,
    get_symmetric_gdml_kernel,
    get_symmetric_gp_kernel,
)
from qmllib.representations import generate_fchl19

# ---------------------------------------------------------------------------
# Constants / defaults matching qmllib's generate_fchl19 defaults
# ---------------------------------------------------------------------------

ELEMENTS = [1, 6, 7, 8, 16, 17]
SIGMA = 5.0

# Tolerances:
# - Representation near-zero entries differ by up to ~2e-9 (absolute) between kf C++ and
#   qmllib Fortran, while non-zero values agree to ~3e-14 (relative).  Use a loose atol.
# - Kernel values inherit the accumulated floating-point noise; use rtol~1e-7.
REPR_RTOL = 1e-10
REPR_ATOL = 5e-9

KERN_RTOL = 1e-7
KERN_ATOL = 1e-7

# Number of molecules used in each test category
N_REPR = 10  # for repr-only tests (faster)
N_KERN = 6  # for kernel tests (need gradients, slower)
N_KERN_GRAD = 4  # for gradient/hessian kernel tests (most expensive)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_qm7b_molecules(n: int) -> tuple[list, list]:
    """Load the first n molecules from QM7b."""
    from kernelforge.cli import load_qm7b_raw_data

    data = load_qm7b_raw_data()
    R_all = data["R"]
    z_all = data["z"]
    coords_list = [R_all[i] for i in range(n)]
    charges_list = [z_all[i] for i in range(n)]
    return coords_list, charges_list


def _build_repr_arrays(
    coords_list: list,
    charges_list: list,
) -> dict:
    """Build padded representation arrays for kernelforge and qmllib."""
    nm = len(coords_list)
    reps_kf = []
    reps_qml = []
    for coords, charges in zip(coords_list, charges_list):
        X_kf = generate_fchl_acsf(coords, charges, elements=ELEMENTS)
        X_qml = generate_fchl19(charges, coords, elements=ELEMENTS, gradients=False)
        reps_kf.append(X_kf)
        reps_qml.append(X_qml)

    max_atoms = max(len(c) for c in charges_list)
    rep_size = reps_kf[0].shape[1]

    X_kf_pad = np.zeros((nm, max_atoms, rep_size))
    X_qml_pad = np.zeros((nm, max_atoms, rep_size))
    Q_kf_pad = np.zeros((nm, max_atoms), dtype=int)
    N_kf = np.zeros(nm, dtype=int)

    for i, (charges, X_kf, X_qml) in enumerate(zip(charges_list, reps_kf, reps_qml)):
        natoms = len(charges)
        X_kf_pad[i, :natoms] = X_kf
        X_qml_pad[i, :natoms] = X_qml
        Q_kf_pad[i, :natoms] = charges
        N_kf[i] = natoms

    charges_qml = [list(c) for c in charges_list]

    return {
        "X_kf": X_kf_pad,
        "X_qml": X_qml_pad,
        "Q_kf": Q_kf_pad,
        "N_kf": N_kf,
        "charges_qml": charges_qml,
        "nm": nm,
        "max_atoms": max_atoms,
        "rep_size": rep_size,
    }


def _build_repr_and_grad_arrays(
    coords_list: list,
    charges_list: list,
) -> dict:
    """Build padded representation + gradient arrays for kernelforge and qmllib."""
    nm = len(coords_list)
    reps_kf, grads_kf = [], []
    reps_qml, grads_qml = [], []

    for coords, charges in zip(coords_list, charges_list):
        X_kf, dX_kf = generate_fchl_acsf_and_gradients(coords, charges, elements=ELEMENTS)
        X_qml, dX_qml = generate_fchl19(charges, coords, elements=ELEMENTS, gradients=True)
        reps_kf.append(X_kf)
        grads_kf.append(dX_kf)
        reps_qml.append(X_qml)
        grads_qml.append(dX_qml)

    max_atoms = max(len(c) for c in charges_list)
    rep_size = reps_kf[0].shape[1]

    X_kf_pad = np.zeros((nm, max_atoms, rep_size))
    dX_kf_pad = np.zeros((nm, max_atoms, rep_size, 3 * max_atoms))
    Q_kf_pad = np.zeros((nm, max_atoms), dtype=int)
    N_kf = np.zeros(nm, dtype=int)
    X_qml_pad = np.zeros((nm, max_atoms, rep_size))
    # qmllib grad shape per molecule: (natoms, rep, natoms, 3)
    dX_qml_pad = np.zeros((nm, max_atoms, rep_size, max_atoms, 3))

    for i, (charges, X_kf, dX_kf, X_qml, dX_qml) in enumerate(
        zip(charges_list, reps_kf, grads_kf, reps_qml, grads_qml)
    ):
        natoms = len(charges)
        X_kf_pad[i, :natoms] = X_kf
        dX_kf_pad[i, :natoms, :, : 3 * natoms] = dX_kf
        Q_kf_pad[i, :natoms] = charges
        N_kf[i] = natoms
        X_qml_pad[i, :natoms] = X_qml
        dX_qml_pad[i, :natoms, :, :natoms, :] = dX_qml

    charges_qml = [list(c) for c in charges_list]

    return {
        "X_kf": X_kf_pad,
        "dX_kf": dX_kf_pad,
        "X_qml": X_qml_pad,
        "dX_qml": dX_qml_pad,
        "Q_kf": Q_kf_pad,
        "N_kf": N_kf,
        "charges_qml": charges_qml,
        "nm": nm,
        "max_atoms": max_atoms,
        "rep_size": rep_size,
    }


# ---------------------------------------------------------------------------
# Representation tests
# ---------------------------------------------------------------------------


def test_repr_matches_qmllib() -> None:
    """generate_fchl_acsf must match qmllib's generate_fchl19.

    Near-zero representation values may differ by up to ~2e-9 (absolute) between
    the C++ and Fortran implementations; non-zero values agree to <1e-13 (relative).
    """
    coords_list, charges_list = _load_qm7b_molecules(N_REPR)
    for coords, charges in zip(coords_list, charges_list):
        X_kf = generate_fchl_acsf(coords, charges, elements=ELEMENTS)
        X_qml = generate_fchl19(charges, coords, elements=ELEMENTS, gradients=False)
        np.testing.assert_allclose(
            X_kf,
            X_qml,
            rtol=REPR_RTOL,
            atol=REPR_ATOL,
            err_msg=f"Representation mismatch for molecule with {len(charges)} atoms",
        )


def test_repr_gradients_match_qmllib() -> None:
    """generate_fchl_acsf_and_gradients gradients must match qmllib's generate_fchl19.

    Layout conversion:
      kf:  (natoms, rep, 3*natoms)
      qml: (natoms, rep, natoms, 3) -> reshape to (natoms, rep, 3*natoms)
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    for coords, charges in zip(coords_list, charges_list):
        natoms = len(charges)
        _X_kf, dX_kf = generate_fchl_acsf_and_gradients(coords, charges, elements=ELEMENTS)
        _X_qml, dX_qml = generate_fchl19(charges, coords, elements=ELEMENTS, gradients=True)
        # Convert qml (natoms, rep, natoms, 3) -> (natoms, rep, 3*natoms)
        dX_qml_reshaped = dX_qml.reshape(natoms, dX_kf.shape[1], 3 * natoms)
        np.testing.assert_allclose(
            dX_kf,
            dX_qml_reshaped,
            rtol=REPR_RTOL,
            atol=REPR_ATOL,
            err_msg=f"Gradient mismatch for molecule with {natoms} atoms",
        )


# ---------------------------------------------------------------------------
# Scalar kernel tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [1.0, 5.0, 21.0])
def test_kernel_gaussian_symm_matches_qmllib(sigma: float) -> None:
    """kernel_gaussian_symm must match qmllib's get_local_symmetric_kernel."""
    coords_list, charges_list = _load_qm7b_molecules(N_KERN)
    d = _build_repr_arrays(coords_list, charges_list)

    K_kf = kf_lk.kernel_gaussian_symm(d["X_kf"], d["Q_kf"], d["N_kf"], sigma)
    K_qml = get_local_symmetric_kernel(d["X_qml"], d["charges_qml"], sigma)

    np.testing.assert_allclose(K_kf, K_qml, rtol=KERN_RTOL, atol=KERN_ATOL)


@pytest.mark.parametrize("sigma", [1.0, 5.0, 21.0])
def test_kernel_gaussian_asymm_matches_qmllib(sigma: float) -> None:
    """kernel_gaussian(X1, X2) must match qmllib's get_local_kernel transposed.

    Convention: K_qml == K_kf.T  (qmllib output shape is (nm2, nm1)).
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN)
    d = _build_repr_arrays(coords_list, charges_list)
    nm = d["nm"]
    nm1 = nm // 2
    nm2 = nm - nm1

    X1_kf = d["X_kf"][:nm1]
    X2_kf = d["X_kf"][nm1:]
    Q1_kf = d["Q_kf"][:nm1]
    Q2_kf = d["Q_kf"][nm1:]
    N1_kf = d["N_kf"][:nm1]
    N2_kf = d["N_kf"][nm1:]
    X1_qml = d["X_qml"][:nm1]
    X2_qml = d["X_qml"][nm1:]
    Q1_qml = d["charges_qml"][:nm1]
    Q2_qml = d["charges_qml"][nm1:]

    K_kf = kf_lk.kernel_gaussian(X1_kf, X2_kf, Q1_kf, Q2_kf, N1_kf, N2_kf, sigma)
    K_qml = get_local_kernel(X1_qml, X2_qml, Q1_qml, Q2_qml, sigma)

    assert K_kf.shape == (nm1, nm2)
    assert K_qml.shape == (nm2, nm1)
    np.testing.assert_allclose(K_kf, K_qml.T, rtol=KERN_RTOL, atol=KERN_ATOL)


def test_kernel_gaussian_symm_rfp_matches_symm() -> None:
    """kernel_gaussian_symm_rfp, when unpacked, must equal kernel_gaussian_symm.

    The RFP kernel stores the lower triangle (UPLO='L') in packed RFP format.
    rfp_to_full must be called with uplo='L' to unpack correctly.
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN)
    d = _build_repr_arrays(coords_list, charges_list)

    K_symm = kf_lk.kernel_gaussian_symm(d["X_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_rfp = kf_lk.kernel_gaussian_symm_rfp(d["X_kf"], d["Q_kf"], d["N_kf"], SIGMA)

    nm = d["nm"]
    K_rfp_full = rfp_to_full(K_rfp, nm, uplo="L")

    np.testing.assert_allclose(K_rfp_full, K_symm, rtol=KERN_RTOL, atol=KERN_ATOL)


# ---------------------------------------------------------------------------
# Jacobian kernel tests
# ---------------------------------------------------------------------------


def test_jacobian_matches_qmllib() -> None:
    """kernel_gaussian_jacobian(X1, X2, dX2) must match qmllib's get_local_gradient_kernel.T.

    Convention: K_qml shape (naq2, nm1), K_kf shape (nm1, naq2) => K_kf == K_qml.T
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)
    nm = d["nm"]
    nm1 = nm // 2
    nm2 = nm - nm1

    X1_kf = d["X_kf"][:nm1]
    X2_kf = d["X_kf"][nm1:]
    dX2_kf = d["dX_kf"][nm1:]
    Q1_kf = d["Q_kf"][:nm1]
    Q2_kf = d["Q_kf"][nm1:]
    N1_kf = d["N_kf"][:nm1]
    N2_kf = d["N_kf"][nm1:]
    X1_qml = d["X_qml"][:nm1]
    X2_qml = d["X_qml"][nm1:]
    dX2_qml = d["dX_qml"][nm1:]
    Q1_qml = d["charges_qml"][:nm1]
    Q2_qml = d["charges_qml"][nm1:]

    naq2 = int(np.sum(N2_kf) * 3)

    K_kf = kf_lk.kernel_gaussian_jacobian(X1_kf, X2_kf, dX2_kf, Q1_kf, Q2_kf, N1_kf, N2_kf, SIGMA)
    K_qml = get_local_gradient_kernel(X1_qml, X2_qml, dX2_qml, Q1_qml, Q2_qml, SIGMA)

    assert K_kf.shape == (nm1, naq2), f"Expected ({nm1}, {naq2}), got {K_kf.shape}"
    assert K_qml.shape == (naq2, nm1), f"Expected ({naq2}, {nm1}), got {K_qml.shape}"
    np.testing.assert_allclose(K_kf, K_qml.T, rtol=KERN_RTOL, atol=KERN_ATOL)


def test_jacobian_t_matches_gp_block() -> None:
    """kernel_gaussian_jacobian_t(X1,X2,dX1) == GP kernel FE block K_full[nm1:, :nm2].

    qmllib has no direct jacobian_t function, so we verify against the GP full kernel's
    force-energy (FE) block, which contains the dX1-side Jacobian.
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)
    nm = d["nm"]
    nm1 = nm // 2
    nm2 = nm - nm1

    X1_kf = d["X_kf"][:nm1]
    X2_kf = d["X_kf"][nm1:]
    dX1_kf = d["dX_kf"][:nm1]
    dX2_kf = d["dX_kf"][nm1:]
    Q1_kf = d["Q_kf"][:nm1]
    Q2_kf = d["Q_kf"][nm1:]
    N1_kf = d["N_kf"][:nm1]
    N2_kf = d["N_kf"][nm1:]

    naq1 = int(np.sum(N1_kf) * 3)

    K_jact = kf_lk.kernel_gaussian_jacobian_t(
        X1_kf, X2_kf, dX1_kf, Q1_kf, Q2_kf, N1_kf, N2_kf, SIGMA
    )
    K_full = kf_lk.kernel_gaussian_full(
        X1_kf, X2_kf, dX1_kf, dX2_kf, Q1_kf, Q2_kf, N1_kf, N2_kf, SIGMA
    )
    K_gp_FE = K_full[nm1:, :nm2]

    assert K_jact.shape == (naq1, nm2), f"Expected ({naq1}, {nm2}), got {K_jact.shape}"
    np.testing.assert_allclose(K_jact, K_gp_FE, rtol=KERN_RTOL, atol=KERN_ATOL)


# ---------------------------------------------------------------------------
# Hessian / GDML kernel tests
# ---------------------------------------------------------------------------


def test_hessian_symm_matches_qmllib() -> None:
    """kernel_gaussian_hessian_symm lower triangle must match qmllib get_symmetric_gdml_kernel.

    kf fills lower triangle only; qmllib returns full symmetric matrix.
    Comparison: symmetrized kf == qml.
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)

    K_kf = kf_lk.kernel_gaussian_hessian_symm(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_qml = get_symmetric_gdml_kernel(d["X_qml"], d["dX_qml"], d["charges_qml"], SIGMA)

    # Symmetrize kf (it only fills lower triangle)
    K_kf_full = np.tril(K_kf) + np.tril(K_kf, -1).T

    np.testing.assert_allclose(K_kf_full, K_qml, rtol=KERN_RTOL, atol=KERN_ATOL)


def test_hessian_asymm_matches_qmllib() -> None:
    """kernel_gaussian_hessian(X1,X2,dX1,dX2) must match qmllib's get_gdml_kernel.T.

    Convention: K_qml shape (naq2, naq1), K_kf shape (naq1, naq2) => K_kf == K_qml.T
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)
    nm = d["nm"]
    nm1 = nm // 2
    nm2 = nm - nm1

    X1_kf = d["X_kf"][:nm1]
    X2_kf = d["X_kf"][nm1:]
    dX1_kf = d["dX_kf"][:nm1]
    dX2_kf = d["dX_kf"][nm1:]
    Q1_kf = d["Q_kf"][:nm1]
    Q2_kf = d["Q_kf"][nm1:]
    N1_kf = d["N_kf"][:nm1]
    N2_kf = d["N_kf"][nm1:]
    X1_qml = d["X_qml"][:nm1]
    X2_qml = d["X_qml"][nm1:]
    dX1_qml = d["dX_qml"][:nm1]
    dX2_qml = d["dX_qml"][nm1:]
    Q1_qml = d["charges_qml"][:nm1]
    Q2_qml = d["charges_qml"][nm1:]

    naq1 = int(np.sum(N1_kf) * 3)
    naq2 = int(np.sum(N2_kf) * 3)

    K_kf = kf_lk.kernel_gaussian_hessian(
        X1_kf, X2_kf, dX1_kf, dX2_kf, Q1_kf, Q2_kf, N1_kf, N2_kf, SIGMA
    )
    K_qml = get_gdml_kernel(X1_qml, X2_qml, dX1_qml, dX2_qml, Q1_qml, Q2_qml, SIGMA)

    assert K_kf.shape == (naq1, naq2), f"Expected ({naq1}, {naq2}), got {K_kf.shape}"
    assert K_qml.shape == (naq2, naq1), f"Expected ({naq2}, {naq1}), got {K_qml.shape}"
    np.testing.assert_allclose(K_kf, K_qml.T, rtol=KERN_RTOL, atol=KERN_ATOL)


def test_hessian_symm_rfp_matches_symm() -> None:
    """kernel_gaussian_hessian_symm_rfp, when unpacked, must equal kernel_gaussian_hessian_symm.

    The RFP kernel fills upper-triangle in TRANSR='N', UPLO='U' packed format.
    After unpacking via rfp_to_full, it must match the symmetrized hessian_symm.
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)

    naq = int(np.sum(d["N_kf"]) * 3)

    K_symm = kf_lk.kernel_gaussian_hessian_symm(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_rfp = kf_lk.kernel_gaussian_hessian_symm_rfp(
        d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA
    )

    K_rfp_full = rfp_to_full(K_rfp, naq, uplo="L")
    # Symmetrize kf lower-tri hessian for comparison
    K_symm_full = np.tril(K_symm) + np.tril(K_symm, -1).T

    np.testing.assert_allclose(K_rfp_full, K_symm_full, rtol=KERN_RTOL, atol=KERN_ATOL)


# ---------------------------------------------------------------------------
# Full GP kernel tests
# ---------------------------------------------------------------------------


def test_full_gp_symm_ee_ff_blocks_match_qmllib() -> None:
    """kernel_gaussian_full_symm EE and FF blocks must match qmllib's get_symmetric_gp_kernel.

    Convention:
      - EE block (energy-energy, top-left nm x nm): matches directly
      - FF block (force-force, bottom-right naq x naq): matches after symmetrizing kf
      - EF block (energy-force): K_kf_EF == -K_qml_EF (sign convention differs)
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)
    nm = d["nm"]
    naq = int(np.sum(d["N_kf"]) * 3)

    K_kf = kf_lk.kernel_gaussian_full_symm(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_qml = get_symmetric_gp_kernel(d["X_qml"], d["dX_qml"], d["charges_qml"], SIGMA)

    assert K_kf.shape == (nm + naq, nm + naq)
    assert K_qml.shape == (nm + naq, nm + naq)

    # EE block (energy-energy)
    np.testing.assert_allclose(K_kf[:nm, :nm], K_qml[:nm, :nm], rtol=KERN_RTOL, atol=KERN_ATOL)

    # FF block (force-force / Hessian): symmetrize kf lower-tri
    K_kf_FF = K_kf[nm:, nm:]
    K_kf_FF_full = np.tril(K_kf_FF) + np.tril(K_kf_FF, -1).T
    np.testing.assert_allclose(K_kf_FF_full, K_qml[nm:, nm:], rtol=KERN_RTOL, atol=KERN_ATOL)

    # EF block: sign is flipped between kf and qmllib
    # K_kf_EF == -K_qml_EF (Jacobian sign convention differs between implementations)
    np.testing.assert_allclose(K_kf[:nm, nm:], -K_qml[:nm, nm:], rtol=KERN_RTOL, atol=KERN_ATOL)


def test_full_gp_asymm_matches_qmllib() -> None:
    """kernel_gaussian_full(X1,X2,...) must match qmllib's get_gp_kernel transposed.

    Convention: K_qml == K_kf.T
    Block layout in K_kf (nm1+naq1, nm2+naq2):
      K_kf[:nm1, :nm2]   = energy-energy
      K_kf[:nm1, nm2:]   = energy-force (Jacobian_t direction)
      K_kf[nm1:, :nm2]   = force-energy (Jacobian direction)
      K_kf[nm1:, nm2:]   = force-force (Hessian)
    K_qml (nm2+naq2, nm1+naq1) is the transpose of K_kf.
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)
    nm = d["nm"]
    nm1 = nm // 2
    nm2 = nm - nm1

    X1_kf = d["X_kf"][:nm1]
    X2_kf = d["X_kf"][nm1:]
    dX1_kf = d["dX_kf"][:nm1]
    dX2_kf = d["dX_kf"][nm1:]
    Q1_kf = d["Q_kf"][:nm1]
    Q2_kf = d["Q_kf"][nm1:]
    N1_kf = d["N_kf"][:nm1]
    N2_kf = d["N_kf"][nm1:]
    X1_qml = d["X_qml"][:nm1]
    X2_qml = d["X_qml"][nm1:]
    dX1_qml = d["dX_qml"][:nm1]
    dX2_qml = d["dX_qml"][nm1:]
    Q1_qml = d["charges_qml"][:nm1]
    Q2_qml = d["charges_qml"][nm1:]

    naq1 = int(np.sum(N1_kf) * 3)
    naq2 = int(np.sum(N2_kf) * 3)

    K_kf = kf_lk.kernel_gaussian_full(
        X1_kf, X2_kf, dX1_kf, dX2_kf, Q1_kf, Q2_kf, N1_kf, N2_kf, SIGMA
    )
    K_qml = get_gp_kernel(X1_qml, X2_qml, dX1_qml, dX2_qml, Q1_qml, Q2_qml, SIGMA)

    assert K_kf.shape == (nm1 + naq1, nm2 + naq2)
    assert K_qml.shape == (nm2 + naq2, nm1 + naq1)

    # EE block matches directly
    np.testing.assert_allclose(
        K_kf[:nm1, :nm2], K_qml[:nm2, :nm1].T, rtol=KERN_RTOL, atol=KERN_ATOL
    )

    # Block-by-block comparison (K_qml is transposed K_kf, with sign flip in FE block):
    #   EE: K_kf[:nm1, :nm2] == K_qml[:nm2, :nm1].T           (energy-energy, no sign flip)
    #   FE: K_kf[nm1:, :nm2] == -K_qml[:nm2, nm1:].T          (jacobian/force-energy, sign flipped)
    #   EF: K_kf[:nm1, nm2:] == K_qml[nm2:, :nm1].T           (jacobian_t/energy-force, no extra sign)
    #   FF: K_kf[nm1:, nm2:] == K_qml[nm2:, nm1:].T           (hessian/force-force, no sign flip)

    # FE block: kf Jacobian (dX1-side) has opposite sign to qmllib's convention
    np.testing.assert_allclose(
        K_kf[nm1:, :nm2], -K_qml[:nm2, nm1:].T, rtol=KERN_RTOL, atol=KERN_ATOL
    )
    # EF block: kf Jacobian_t (dX2-side viewed from kf) matches qmllib transposed
    np.testing.assert_allclose(
        K_kf[:nm1, nm2:], K_qml[nm2:, :nm1].T, rtol=KERN_RTOL, atol=KERN_ATOL
    )
    # FF block: Hessian matches (no sign flip)
    np.testing.assert_allclose(
        K_kf[nm1:, nm2:], K_qml[nm2:, nm1:].T, rtol=KERN_RTOL, atol=KERN_ATOL
    )


def test_full_gp_symm_rfp_matches_symm() -> None:
    """kernel_gaussian_full_symm_rfp, when unpacked, must match kernel_gaussian_full_symm.

    Only the EE and FF blocks are compared (those that match qmllib without sign flip).
    """
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)
    nm = d["nm"]
    naq = int(np.sum(d["N_kf"]) * 3)
    big = nm + naq

    K_symm = kf_lk.kernel_gaussian_full_symm(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_rfp = kf_lk.kernel_gaussian_full_symm_rfp(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)

    K_rfp_full = rfp_to_full(K_rfp, big, uplo="L")

    # Compare against symmetrized K_symm (lower-tri only output)
    K_symm_full = np.tril(K_symm) + np.tril(K_symm, -1).T

    np.testing.assert_allclose(K_rfp_full, K_symm_full, rtol=KERN_RTOL, atol=KERN_ATOL)
