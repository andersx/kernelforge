"""Numerical parity check between kernelforge FCHL18 and qmllib (reference).

Run manually (requires qmllib installed):
    python examples/fchl18_qmllib_parity_check.py

Compares kernelforge FCHL18 scalar kernels against qmllib with alchemy='off'.
All checks use real QM7b molecules.

Hyperparameter mapping (kernelforge → qmllib defaults):
    two_body_scaling   = sqrt(8)   (qmllib default: sqrt(8))
    two_body_width     = 0.2
    two_body_power     = 4.0
    three_body_scaling = 1.6
    three_body_width   = pi
    three_body_power   = 2.0
    cut_start          = 1.0
    cut_distance       = 5.0
    fourier_order      = 1
"""

import sys

import numpy as np
from qmllib.representations.fchl import fchl_scalar_kernels as fsk
from qmllib.representations.fchl import generate_fchl18

import kernelforge.fchl18_kernel as km
import kernelforge.fchl18_repr as rm

# ---------------------------------------------------------------------------
# Hyperparameters matching qmllib defaults (alchemy='off')
# ---------------------------------------------------------------------------

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

# Tolerances
KERN_RTOL = 1e-7
KERN_ATOL = 1e-7

N_MOLS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_qm7b(n):
    ds = np.load("/home/andersx/.kernelforge/datasets/qm7b_complete.npz", allow_pickle=True)
    coords_list = [ds["R"][i] for i in range(n)]
    charges_list = [ds["z"][i].astype(int) for i in range(n)]
    return coords_list, charges_list


def _build_qml_repr(coords_list, charges_list):
    max_size = max(len(c) for c in charges_list)
    mols = [
        generate_fchl18(z, r, max_size=max_size, cut_distance=CUT_DISTANCE)
        for z, r in zip(charges_list, coords_list, strict=True)
    ]
    return np.array(mols)


def _build_kf_repr(coords_list, charges_list):
    max_size = max(len(c) for c in charges_list)
    x, n, nn = rm.generate(
        [r.tolist() for r in coords_list],
        [z.tolist() for z in charges_list],
        max_size=max_size,
        cut_distance=CUT_DISTANCE,
    )
    return x, n, nn


def _check(label, actual, desired, rtol, atol):
    try:
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
        print(f"  PASS  {label}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {label}\n        {e}")
        return False


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_scalar_symm():
    print("Scalar symmetric kernel (kernel_gaussian_symm vs get_local_symmetric_kernels)")
    coords_list, charges_list = _load_qm7b(N_MOLS)

    X_qml = _build_qml_repr(coords_list, charges_list)
    x_kf, n_kf, nn_kf = _build_kf_repr(coords_list, charges_list)

    ok = True
    for sigma in [1.0, 2.0, 5.0]:
        K_qml = fsk.get_local_symmetric_kernels(
            X_qml, alchemy="off", kernel_args={"sigma": [sigma]}
        )[0]
        K_kf = km.kernel_gaussian_symm(x_kf, n_kf, nn_kf, sigma=sigma, **KARGS)
        ok &= _check(
            f"kernel_gaussian_symm sigma={sigma}",
            K_kf,
            K_qml,
            rtol=KERN_RTOL,
            atol=KERN_ATOL,
        )
    return ok


def check_scalar_asymm():
    print("Scalar asymmetric kernel (kernel_gaussian vs get_local_kernels)")
    # Load all molecules together so qmllib can share max_size, then split
    coords_all, charges_all = _load_qm7b(N_MOLS)
    nm1, nm2 = 3, N_MOLS - 3  # split 3 vs 2

    # qmllib requires both arrays to share the same max_size — build together then slice
    max_size = max(len(c) for c in charges_all)
    mols_all = [
        generate_fchl18(z, r, max_size=max_size, cut_distance=CUT_DISTANCE)
        for z, r in zip(charges_all, coords_all, strict=True)
    ]
    X_qml_all = np.array(mols_all)

    # kernelforge: build two separate repr arrays (each with its own max_size)
    x_kf1, n_kf1, nn_kf1 = _build_kf_repr(coords_all[:nm1], charges_all[:nm1])
    x_kf2, n_kf2, nn_kf2 = _build_kf_repr(coords_all[nm1:], charges_all[nm1:])

    ok = True
    for sigma in [1.0, 2.0, 5.0]:
        K_qml = fsk.get_local_kernels(
            X_qml_all[:nm1], X_qml_all[nm1:], alchemy="off", kernel_args={"sigma": [sigma]}
        )[0]
        K_kf = km.kernel_gaussian(
            x_kf1,
            x_kf2,
            n_kf1,
            n_kf2,
            nn_kf1,
            nn_kf2,
            sigma=sigma,
            **KARGS,
        )
        # qmllib get_local_kernels returns (nm1, nm2), same layout as kernelforge
        ok &= _check(
            f"kernel_gaussian sigma={sigma}",
            K_kf,
            K_qml,
            rtol=KERN_RTOL,
            atol=KERN_ATOL,
        )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_ok = True
    all_ok &= check_scalar_symm()
    all_ok &= check_scalar_asymm()

    if all_ok:
        print("\nAll FCHL18 parity checks passed.")
        sys.exit(0)
    else:
        print("\nSome FCHL18 parity checks FAILED.")
        sys.exit(1)
