from typing import TypedDict, Unpack

import numpy as np
from numpy.typing import NDArray

from kernelforge import _fchl19 as _fchl
from kernelforge._fchl19 import generate_fchl_acsf, generate_fchl_acsf_and_gradients


class FCHL19Params(TypedDict, total=False):
    """Optional parameters for generate_fchl_acsf."""

    elements: list[int]
    nRs2: int
    nRs3: int
    nFourier: int
    eta2: float
    eta3: float
    zeta: float
    rcut: float
    acut: float
    two_body_decay: float
    three_body_decay: float
    three_body_weight: float


def _call_generate(
    coords: NDArray[np.float64], nuclear_z: NDArray[np.int32], **kwargs: Unpack[FCHL19Params]
) -> NDArray[np.float64]:
    """
    Wrapper to call the extension using named args, handling either arg order
    (coords first or nuclear_z first) depending on how it was compiled.
    """
    fn = _fchl.generate_fchl_acsf
    # Build params dict with coords/nuclear_z plus any optional kwargs
    params = {
        "coords": np.asarray(coords, dtype=np.float64),
        "nuclear_z": np.asarray(nuclear_z, dtype=np.int32),
        **kwargs,
    }
    try:
        # Our binding signature: (coords, nuclear_z, ...)
        result: NDArray[np.float64] = fn(**params)
        return result
    except TypeError:
        # Fallback if someone compiled as (nuclear_z, coords, ...)
        params2 = dict(params)
        params2["coords"], params2["nuclear_z"] = params["nuclear_z"], params["coords"]
        result2: NDArray[np.float64] = fn(**params2)
        return result2


def _descr_size(n_elements: int, nRs2: int, nRs3: int, nFourier: int) -> int:
    # Matches the code in the binding:
    return n_elements * nRs2 + (n_elements * (n_elements + 1)) * nRs3 * nFourier


def test_shape_with_defaults_and_explicit_rep_size() -> None:
    # H-O-H linear-ish toy geometry
    coords = np.array(
        [[-0.9572, 0.0, 0.0], [0.0000, 0.0, 0.0], [0.9572, 0.0, 0.0]], dtype=np.float64
    )
    Z = np.array([1, 8, 1], dtype=np.int32)

    # Use all defaults from the binding
    rep = _call_generate(coords, Z)

    # Defaults in the binding:
    elements = [1, 6, 7, 8, 16]
    nRs2, nRs3, nFourier = 24, 20, 1
    expected_rep_size = _descr_size(len(elements), nRs2, nRs3, nFourier)

    assert rep.shape == (coords.shape[0], expected_rep_size)


def test_shape_with_explicit_basis_sizes_and_elements() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    Z = np.array([6, 1, 1], dtype=np.int32)  # CHH

    elements = [1, 6]  # only H and C
    nRs2, nRs3, nFourier = 3, 2, 2
    rep = _call_generate(
        coords,
        Z,
        elements=elements,
        nRs2=nRs2,
        nRs3=nRs3,
        nFourier=nFourier,
        eta2=0.32,
        eta3=2.7,
        zeta=np.pi,
        rcut=8.0,
        acut=8.0,
        two_body_decay=1.8,
        three_body_decay=0.57,
        three_body_weight=13.4,
    )

    expected_rep_size = _descr_size(len(elements), nRs2, nRs3, nFourier)
    assert rep.shape == (coords.shape[0], expected_rep_size)


def test_translation_invariance() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    Z = np.array([1, 1], dtype=np.int32)

    rep1 = _call_generate(
        coords,
        Z,
        elements=[1],
        nRs2=4,
        nRs3=2,
        nFourier=1,
        eta2=0.5,
        eta3=1.5,
        zeta=np.pi,
        rcut=4.0,
        acut=4.0,
        two_body_decay=1.0,
        three_body_decay=1.0,
        three_body_weight=1.0,
    )

    shift = np.array([10.0, -5.0, 2.0])
    rep2 = _call_generate(
        coords + shift,
        Z,
        elements=[1],
        nRs2=4,
        nRs3=2,
        nFourier=1,
        eta2=0.5,
        eta3=1.5,
        zeta=np.pi,
        rcut=4.0,
        acut=4.0,
        two_body_decay=1.0,
        three_body_decay=1.0,
        three_body_weight=1.0,
    )

    np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)


def test_three_body_zeroing_block_when_weight_zero() -> None:
    # Geometry with angles so 3-body term would be nonzero when enabled
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]], dtype=np.float64)
    Z = np.array([1, 1, 1], dtype=np.int32)
    elements = [1]
    nRs2, nRs3, nFourier = 4, 3, 2

    # With three_body_weight = 0 => 3-body block should be exactly zero
    rep_zero = _call_generate(
        coords,
        Z,
        elements=elements,
        nRs2=nRs2,
        nRs3=nRs3,
        nFourier=nFourier,
        eta2=0.5,
        eta3=1.5,
        zeta=np.pi,
        rcut=6.0,
        acut=6.0,
        two_body_decay=1.0,
        three_body_decay=1.0,
        three_body_weight=0.0,
    )

    # With a positive weight => 3-body block should generally be non-zero
    rep_pos = _call_generate(
        coords,
        Z,
        elements=elements,
        nRs2=nRs2,
        nRs3=nRs3,
        nFourier=nFourier,
        eta2=0.5,
        eta3=1.5,
        zeta=np.pi,
        rcut=6.0,
        acut=6.0,
        two_body_decay=1.0,
        three_body_decay=1.0,
        three_body_weight=1.0,
    )

    two_body_size = len(elements) * nRs2
    # Check 3-body block all zeros when weight=0
    assert np.allclose(rep_zero[:, two_body_size:], 0.0, atol=1e-14)
    # And for the positive-weight case, at least some entries are non-zero
    assert np.linalg.norm(rep_pos[:, two_body_size:]) > 0.0


def test_defaults_match_manual_linspace_construction() -> None:
    # Ensure the binding's internal linspace matches what we'd build in Python
    # TODO: Implement test
    pass


def test_rep_matches_between_functions() -> None:
    # small, simple triatomic
    coords = np.array(
        [[-0.9572, 0.0, 0.0], [0.0000, 0.0, 0.0], [0.9572, 0.0, 0.0]], dtype=np.float64
    )
    Z = np.array([1, 8, 1], dtype=np.int32)

    # Keep basis sizes small for speed but nontrivial
    kw: FCHL19Params = {
        "elements": [1, 8],
        "nRs2": 4,
        "nRs3": 3,
        "nFourier": 1,
        "eta2": 0.32,
        "eta3": 2.7,
        "zeta": np.pi,
        "rcut": 8.0,
        "acut": 8.0,
        "two_body_decay": 1.8,
        "three_body_decay": 0.57,
        "three_body_weight": 13.4,
    }

    rep_only = generate_fchl_acsf(coords, Z, **kw)
    rep_both, grad = generate_fchl_acsf_and_gradients(coords, Z, **kw)

    assert rep_only.shape == rep_both.shape
    np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)

    # basic sanity on grad shape
    n, rep_size = rep_only.shape
    assert grad.shape == (n, rep_size, n * 3)


def test_analytic_grad_matches_finite_difference() -> None:
    # small CH2-like geometry (asymmetric to avoid accidental cancellations)
    coords = np.array(
        [[0.00, 0.00, 0.00], [1.10, 0.00, 0.00], [-0.20, 0.95, 0.15]],  # C  # H  # H
        dtype=np.float64,
    )
    Z = np.array([6, 1, 1], dtype=np.int32)

    kw: FCHL19Params = {
        "elements": [1, 6],
        "nRs2": 4,
        "nRs3": 3,
        "nFourier": 1,  # keep tiny for speed
        "eta2": 0.32,
        "eta3": 2.7,
        "zeta": np.pi,
        "rcut": 5.0,
        "acut": 5.0,
        "two_body_decay": 1.3,
        "three_body_decay": 0.8,
        "three_body_weight": 2.5,
    }

    rep, grad = generate_fchl_acsf_and_gradients(coords, Z, **kw)
    n, rep_size = rep.shape

    # central finite-difference using the rep-only function
    h = 1e-6
    fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)

    # loop is tiny (n=3, rep_size small), but keep it simple & explicit
    for a in range(n):
        for d in range(3):
            cp = coords.copy()
            cm = coords.copy()
            cp[a, d] += h
            cm[a, d] -= h
            rep_p = generate_fchl_acsf(cp, Z, **kw)
            rep_m = generate_fchl_acsf(cm, Z, **kw)
            fd_grad[:, :, a * 3 + d] = (rep_p - rep_m) / (2.0 * h)

    # Compare analytic vs FD. Use slightly looser tolerances than for exact equality.
    # With smooth functions and h=1e-6, these tolerances are typically comfortable.
    np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


def test_translation_invariance_and_zero_grad_under_uniform_shift() -> None:
    # two atoms; translation of all atoms should not change rep,
    # and gradient wrt a *uniform* shift of all atoms should sum to ~0
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    Z = np.array([1, 1], dtype=np.int32)

    kw: FCHL19Params = {
        "elements": [1],
        "nRs2": 4,
        "nRs3": 2,
        "nFourier": 1,
        "eta2": 0.4,
        "eta3": 1.6,
        "zeta": np.pi,
        "rcut": 4.0,
        "acut": 4.0,
        "two_body_decay": 1.0,
        "three_body_decay": 1.0,
        "three_body_weight": 1.0,
    }

    rep1, grad1 = generate_fchl_acsf_and_gradients(coords, Z, **kw)

    shift = np.array([+3.2, -1.1, +0.7])
    rep2, _ = generate_fchl_acsf_and_gradients(coords + shift, Z, **kw)

    np.testing.assert_allclose(rep1, rep2, rtol=1e-12, atol=1e-14)

    # For a uniform translation, sum over atoms of d(rep)/d(coords) ≈ 0 (per feature and direction)
    summed = grad1.sum(axis=2)  # shape: (n, rep_size, 3)
    np.testing.assert_allclose(summed, 0.0, rtol=1e-9, atol=1e-10)
