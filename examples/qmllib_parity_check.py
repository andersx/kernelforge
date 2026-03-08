"""Numerical parity check between kernelforge and qmllib (reference implementation).

Run manually (requires qmllib installed):
    python examples/qmllib_parity_check.py

All checks use real QM7b molecules and compare kernelforge outputs to qmllib
outputs to machine precision (rtol=1e-7).

Convention notes (discovered by inspection):
  - Asymmetric kernels: K_qml == K_kf.T  (qmllib is Fortran-order / transposed)
  - Symmetric kernels: EE and FF blocks match directly after symmetrizing kf lower-tri
  - Symmetric GP kernel: EF/FE off-diagonal blocks differ by sign (K_kf_EF == -K_qml_EF)
  - Asymmetric GP kernel: K_qml == K_kf.T
  - RFP kernels: unpack via rfp_to_full, compare to non-RFP symmetric kernel
"""

import sys

import numpy as np
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

import kernelforge.local_kernels as kf_lk
from kernelforge.fchl19_repr import generate_fchl_acsf, generate_fchl_acsf_and_gradients
from kernelforge.kernelmath import rfp_to_full

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
N_REPR = 10
N_KERN = 6
N_KERN_GRAD = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_qm7b_molecules(n):
    from kernelforge.cli import load_qm7b_raw_data

    data = load_qm7b_raw_data()
    coords_list = [data["R"][i] for i in range(n)]
    charges_list = [data["z"][i] for i in range(n)]
    return coords_list, charges_list


def _build_repr_arrays(coords_list, charges_list):
    nm = len(coords_list)
    reps_kf, reps_qml = [], []
    for coords, charges in zip(coords_list, charges_list, strict=True):
        reps_kf.append(generate_fchl_acsf(coords, charges, elements=ELEMENTS))
        reps_qml.append(generate_fchl19(charges, coords, elements=ELEMENTS, gradients=False))

    max_atoms = max(len(c) for c in charges_list)
    rep_size = reps_kf[0].shape[1]

    X_kf_pad = np.zeros((nm, max_atoms, rep_size))
    X_qml_pad = np.zeros((nm, max_atoms, rep_size))
    Q_kf_pad = np.zeros((nm, max_atoms), dtype=int)
    N_kf = np.zeros(nm, dtype=int)

    for i, (charges, X_kf, X_qml) in enumerate(zip(charges_list, reps_kf, reps_qml, strict=True)):
        natoms = len(charges)
        X_kf_pad[i, :natoms] = X_kf
        X_qml_pad[i, :natoms] = X_qml
        Q_kf_pad[i, :natoms] = charges
        N_kf[i] = natoms

    return {
        "X_kf": X_kf_pad,
        "X_qml": X_qml_pad,
        "Q_kf": Q_kf_pad,
        "N_kf": N_kf,
        "charges_qml": [list(c) for c in charges_list],
        "nm": nm,
    }


def _build_repr_and_grad_arrays(coords_list, charges_list):
    nm = len(coords_list)
    reps_kf, grads_kf, reps_qml, grads_qml = [], [], [], []

    for coords, charges in zip(coords_list, charges_list, strict=True):
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
    dX_qml_pad = np.zeros((nm, max_atoms, rep_size, max_atoms, 3))

    for i, (charges, X_kf, dX_kf, X_qml, dX_qml) in enumerate(
        zip(charges_list, reps_kf, grads_kf, reps_qml, grads_qml, strict=True)
    ):
        natoms = len(charges)
        X_kf_pad[i, :natoms] = X_kf
        dX_kf_pad[i, :natoms, :, : 3 * natoms] = dX_kf
        Q_kf_pad[i, :natoms] = charges
        N_kf[i] = natoms
        X_qml_pad[i, :natoms] = X_qml
        dX_qml_pad[i, :natoms, :, :natoms, :] = dX_qml

    return {
        "X_kf": X_kf_pad,
        "dX_kf": dX_kf_pad,
        "X_qml": X_qml_pad,
        "dX_qml": dX_qml_pad,
        "Q_kf": Q_kf_pad,
        "N_kf": N_kf,
        "charges_qml": [list(c) for c in charges_list],
        "nm": nm,
    }


def _check(label, *args, **kwargs):
    try:
        np.testing.assert_allclose(*args, **kwargs)
        print(f"  PASS  {label}")
    except AssertionError as e:
        print(f"  FAIL  {label}\n        {e}")
        return False
    return True


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_repr():
    print("Representation checks")
    coords_list, charges_list = _load_qm7b_molecules(N_REPR)
    ok = True
    for coords, charges in zip(coords_list, charges_list, strict=True):
        X_kf = generate_fchl_acsf(coords, charges, elements=ELEMENTS)
        X_qml = generate_fchl19(charges, coords, elements=ELEMENTS, gradients=False)
        ok &= _check(
            f"repr natoms={len(charges)}",
            X_kf,
            X_qml,
            rtol=REPR_RTOL,
            atol=REPR_ATOL,
        )
    return ok


def check_repr_gradients():
    print("Representation gradient checks")
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    ok = True
    for coords, charges in zip(coords_list, charges_list, strict=True):
        natoms = len(charges)
        _X_kf, dX_kf = generate_fchl_acsf_and_gradients(coords, charges, elements=ELEMENTS)
        _X_qml, dX_qml = generate_fchl19(charges, coords, elements=ELEMENTS, gradients=True)
        dX_qml_reshaped = dX_qml.reshape(natoms, dX_kf.shape[1], 3 * natoms)
        ok &= _check(
            f"repr grad natoms={natoms}",
            dX_kf,
            dX_qml_reshaped,
            rtol=REPR_RTOL,
            atol=REPR_ATOL,
        )
    return ok


def check_scalar_kernels():
    print("Scalar kernel checks")
    coords_list, charges_list = _load_qm7b_molecules(N_KERN)
    d = _build_repr_arrays(coords_list, charges_list)
    ok = True

    for sigma in [1.0, 5.0, 21.0]:
        K_kf = kf_lk.kernel_gaussian_symm(d["X_kf"], d["Q_kf"], d["N_kf"], sigma)
        K_qml = get_local_symmetric_kernel(d["X_qml"], d["charges_qml"], sigma)
        ok &= _check(
            f"kernel_gaussian_symm sigma={sigma}", K_kf, K_qml, rtol=KERN_RTOL, atol=KERN_ATOL
        )

    nm = d["nm"]
    nm1, nm2 = nm // 2, nm - nm // 2
    for sigma in [1.0, 5.0, 21.0]:
        K_kf = kf_lk.kernel_gaussian(
            d["X_kf"][:nm1],
            d["X_kf"][nm1:],
            d["Q_kf"][:nm1],
            d["Q_kf"][nm1:],
            d["N_kf"][:nm1],
            d["N_kf"][nm1:],
            sigma,
        )
        K_qml = get_local_kernel(
            d["X_qml"][:nm1],
            d["X_qml"][nm1:],
            d["charges_qml"][:nm1],
            d["charges_qml"][nm1:],
            sigma,
        )
        ok &= _check(
            f"kernel_gaussian_asymm sigma={sigma}", K_kf, K_qml.T, rtol=KERN_RTOL, atol=KERN_ATOL
        )

    K_symm = kf_lk.kernel_gaussian_symm(d["X_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_rfp = kf_lk.kernel_gaussian_symm_rfp(d["X_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_rfp_full = rfp_to_full(K_rfp, nm, uplo="L")
    ok &= _check("kernel_gaussian_symm_rfp", K_rfp_full, K_symm, rtol=KERN_RTOL, atol=KERN_ATOL)

    return ok


def check_jacobian_kernels():
    print("Jacobian kernel checks")
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)
    nm = d["nm"]
    nm1, nm2 = nm // 2, nm - nm // 2
    ok = True

    K_kf = kf_lk.kernel_gaussian_jacobian(
        d["X_kf"][:nm1],
        d["X_kf"][nm1:],
        d["dX_kf"][nm1:],
        d["Q_kf"][:nm1],
        d["Q_kf"][nm1:],
        d["N_kf"][:nm1],
        d["N_kf"][nm1:],
        SIGMA,
    )
    K_qml = get_local_gradient_kernel(
        d["X_qml"][:nm1],
        d["X_qml"][nm1:],
        d["dX_qml"][nm1:],
        d["charges_qml"][:nm1],
        d["charges_qml"][nm1:],
        SIGMA,
    )
    # kernelforge sign convention: K_jact(x1,x2,dX1) == -K_jac(x2,x1,dX1).T, so the
    # standalone jacobian has opposite sign to qmllib's get_local_gradient_kernel.
    ok &= _check("kernel_gaussian_jacobian", K_kf, -K_qml.T, rtol=KERN_RTOL, atol=KERN_ATOL)

    K_jact = kf_lk.kernel_gaussian_jacobian_t(
        d["X_kf"][:nm1],
        d["X_kf"][nm1:],
        d["dX_kf"][:nm1],
        d["Q_kf"][:nm1],
        d["Q_kf"][nm1:],
        d["N_kf"][:nm1],
        d["N_kf"][nm1:],
        SIGMA,
    )
    K_full = kf_lk.kernel_gaussian_full(
        d["X_kf"][:nm1],
        d["X_kf"][nm1:],
        d["dX_kf"][:nm1],
        d["dX_kf"][nm1:],
        d["Q_kf"][:nm1],
        d["Q_kf"][nm1:],
        d["N_kf"][:nm1],
        d["N_kf"][nm1:],
        SIGMA,
    )
    # FE block = K_full[nm1:, :nm2] == -K_jact  (kernelforge sign convention for K_jact)
    ok &= _check(
        "kernel_gaussian_jacobian_t vs GP FE block (sign flip)",
        K_jact,
        -K_full[nm1:, :nm2],
        rtol=KERN_RTOL,
        atol=KERN_ATOL,
    )

    return ok


def check_hessian_kernels():
    print("Hessian / GDML kernel checks")
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)
    nm = d["nm"]
    nm1, nm2 = nm // 2, nm - nm // 2
    naq = int(np.sum(d["N_kf"]) * 3)
    ok = True

    K_kf = kf_lk.kernel_gaussian_hessian_symm(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_qml = get_symmetric_gdml_kernel(d["X_qml"], d["dX_qml"], d["charges_qml"], SIGMA)
    K_kf_full = np.tril(K_kf) + np.tril(K_kf, -1).T
    ok &= _check("kernel_gaussian_hessian_symm", K_kf_full, K_qml, rtol=KERN_RTOL, atol=KERN_ATOL)

    K_kf = kf_lk.kernel_gaussian_hessian(
        d["X_kf"][:nm1],
        d["X_kf"][nm1:],
        d["dX_kf"][:nm1],
        d["dX_kf"][nm1:],
        d["Q_kf"][:nm1],
        d["Q_kf"][nm1:],
        d["N_kf"][:nm1],
        d["N_kf"][nm1:],
        SIGMA,
    )
    K_qml = get_gdml_kernel(
        d["X_qml"][:nm1],
        d["X_qml"][nm1:],
        d["dX_qml"][:nm1],
        d["dX_qml"][nm1:],
        d["charges_qml"][:nm1],
        d["charges_qml"][nm1:],
        SIGMA,
    )
    ok &= _check("kernel_gaussian_hessian_asymm", K_kf, K_qml.T, rtol=KERN_RTOL, atol=KERN_ATOL)

    K_symm = kf_lk.kernel_gaussian_hessian_symm(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_rfp = kf_lk.kernel_gaussian_hessian_symm_rfp(
        d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA
    )
    K_rfp_full = rfp_to_full(K_rfp, naq, uplo="L")
    K_symm_full = np.tril(K_symm) + np.tril(K_symm, -1).T
    ok &= _check(
        "kernel_gaussian_hessian_symm_rfp", K_rfp_full, K_symm_full, rtol=KERN_RTOL, atol=KERN_ATOL
    )

    return ok


def check_gp_kernels():
    print("Full GP kernel checks")
    coords_list, charges_list = _load_qm7b_molecules(N_KERN_GRAD)
    d = _build_repr_and_grad_arrays(coords_list, charges_list)
    nm = d["nm"]
    nm1, nm2 = nm // 2, nm - nm // 2
    naq = int(np.sum(d["N_kf"]) * 3)
    ok = True

    K_kf = kf_lk.kernel_gaussian_full_symm(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_qml = get_symmetric_gp_kernel(d["X_qml"], d["dX_qml"], d["charges_qml"], SIGMA)
    K_kf_FF = K_kf[nm:, nm:]
    K_kf_FF_full = np.tril(K_kf_FF) + np.tril(K_kf_FF, -1).T
    ok &= _check(
        "full_gp_symm EE block", K_kf[:nm, :nm], K_qml[:nm, :nm], rtol=KERN_RTOL, atol=KERN_ATOL
    )
    ok &= _check(
        "full_gp_symm FF block", K_kf_FF_full, K_qml[nm:, nm:], rtol=KERN_RTOL, atol=KERN_ATOL
    )
    ok &= _check(
        "full_gp_symm EF block (sign flip)",
        K_kf[:nm, nm:],
        -K_qml[:nm, nm:],
        rtol=KERN_RTOL,
        atol=KERN_ATOL,
    )

    naq1 = int(np.sum(d["N_kf"][:nm1]) * 3)
    naq2 = int(np.sum(d["N_kf"][nm1:]) * 3)
    K_kf = kf_lk.kernel_gaussian_full(
        d["X_kf"][:nm1],
        d["X_kf"][nm1:],
        d["dX_kf"][:nm1],
        d["dX_kf"][nm1:],
        d["Q_kf"][:nm1],
        d["Q_kf"][nm1:],
        d["N_kf"][:nm1],
        d["N_kf"][nm1:],
        SIGMA,
    )
    K_qml = get_gp_kernel(
        d["X_qml"][:nm1],
        d["X_qml"][nm1:],
        d["dX_qml"][:nm1],
        d["dX_qml"][nm1:],
        d["charges_qml"][:nm1],
        d["charges_qml"][nm1:],
        SIGMA,
    )
    ok &= _check(
        "full_gp_asymm EE", K_kf[:nm1, :nm2], K_qml[:nm2, :nm1].T, rtol=KERN_RTOL, atol=KERN_ATOL
    )
    ok &= _check(
        "full_gp_asymm FE (sign flip)",
        K_kf[nm1:, :nm2],
        -K_qml[:nm2, nm1:].T,
        rtol=KERN_RTOL,
        atol=KERN_ATOL,
    )
    ok &= _check(
        "full_gp_asymm EF (sign flip)",
        K_kf[:nm1, nm2:],
        -K_qml[nm2:, :nm1].T,
        rtol=KERN_RTOL,
        atol=KERN_ATOL,
    )
    ok &= _check(
        "full_gp_asymm FF", K_kf[nm1:, nm2:], K_qml[nm2:, nm1:].T, rtol=KERN_RTOL, atol=KERN_ATOL
    )

    big = nm + naq
    K_symm = kf_lk.kernel_gaussian_full_symm(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_rfp = kf_lk.kernel_gaussian_full_symm_rfp(d["X_kf"], d["dX_kf"], d["Q_kf"], d["N_kf"], SIGMA)
    K_rfp_full = rfp_to_full(K_rfp, big, uplo="L")
    K_symm_full = np.tril(K_symm) + np.tril(K_symm, -1).T
    ok &= _check("full_gp_symm_rfp", K_rfp_full, K_symm_full, rtol=KERN_RTOL, atol=KERN_ATOL)

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_ok = True
    all_ok &= check_repr()
    all_ok &= check_repr_gradients()
    all_ok &= check_scalar_kernels()
    all_ok &= check_jacobian_kernels()
    all_ok &= check_hessian_kernels()
    all_ok &= check_gp_kernels()

    if all_ok:
        print("\nAll parity checks passed.")
        sys.exit(0)
    else:
        print("\nSome parity checks FAILED.")
        sys.exit(1)
