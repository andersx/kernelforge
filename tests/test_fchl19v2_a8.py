"""Tests for fchl19v2_repr A8: Legendre_BesselJoint three-body type.

Design summary
--------------
A8 ("legendre_bessel_joint") introduces joint-diagonal radial coupling:

Angular basis:
  angular[l] = w_l * P_l(cos_theta),  w_l = exp(-zeta * l*(l+1)),  l=0..Lmax
  nabasis = nFourier + 1  (P_0 through P_{nFourier})

Radial basis (per leg, Bessel with per-mode decay):
  phi_n(r) = sqrt(2/rcut) * sin(n*pi*r/rcut)/r * exp(-eta3_minus * n^2) * decay(r)

Joint coupling (WHY more expressive than separate marginals):
  Same-element:       radial_A[n] = phi_j[n] + phi_k[n]    (symmetric marginal)
                      radial_B[n] = phi_j[n] * phi_k[n]    (diagonal joint product)
  Different-element:  radial_A[n] = phi_low[n]              (low-Z leg)
                      radial_B[n] = phi_low[n] * phi_high[n](joint product)

Block layout per unordered element pair (size = 2 * nbasis3 * nabasis):
  [0 .. nbasis3*nabasis)         : chan_A  (marginal)
  [nbasis3*nabasis .. 2*..):     : chan_B  (joint coupling)

Rep size: nelements*nRs2 + n_pairs * 2 * nRs3 * (nFourier+1)
"""

from typing import Any

import numpy as np
import pytest

from kernelforge import fchl19v2_repr as _m

# ---------------------------------------------------------------------------
# Shared geometry fixtures
# ---------------------------------------------------------------------------

# Asymmetric CH2-like molecule: C at origin, two H atoms
COORDS_CH2 = np.array(
    [[0.00, 0.00, 0.00], [1.10, 0.00, 0.00], [-0.20, 0.95, 0.15]],
    dtype=np.float64,
)
Z_CH2 = np.array([6, 1, 1], dtype=np.int32)

# Multi-element: N-C-O chain
COORDS_NCO = np.array(
    [[0.00, 0.00, 0.00], [1.20, 0.00, 0.00], [2.30, 0.60, 0.00]],
    dtype=np.float64,
)
Z_NCO = np.array([7, 6, 8], dtype=np.int32)

# Two-atom molecule: only one pair, no triplets possible
COORDS_H2 = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float64)
Z_H2 = np.array([1, 1], dtype=np.int32)

THREE_BODY_TYPE = "legendre_bessel_joint"

# Small basis params for speed. nFourier=2 → nabasis=3 (P_0,P_1,P_2)
BASE_KW: dict[str, Any] = {
    "elements": [1, 6, 7, 8],
    "nRs2": 4,
    "nRs3": 3,
    "nFourier": 2,  # Lmax=2, nabasis=3
    "eta2": 0.32,
    "eta3": 2.7,
    "eta3_minus": 0.05,  # per-mode Bessel decay
    "zeta": 1.0,  # Legendre angular decay gamma
    "rcut": 5.0,
    "acut": 5.0,
    "two_body_decay": 1.3,
    "three_body_decay": 0.8,
    "three_body_weight": 2.5,
    "two_body_type": "bessel",
    "three_body_type": THREE_BODY_TYPE,
}


def kw(**overrides: Any) -> dict[str, Any]:
    return {**BASE_KW, **overrides}


def expected_rep_size(nelements: int, nRs2: int, nRs3: int, nFourier: int) -> int:
    """Expected rep size: nelements*nRs2 + n_pairs * 2 * nRs3 * (nFourier+1)."""
    n_pairs = nelements * (nelements + 1) // 2
    nabasis = nFourier + 1
    return nelements * nRs2 + n_pairs * 2 * nRs3 * nabasis


# ---------------------------------------------------------------------------
# 1. Shape correctness
# ---------------------------------------------------------------------------


class TestA8Shape:
    def test_shape_matches_formula(self) -> None:
        k = kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **k)
        n = len(Z_CH2)
        expected = expected_rep_size(len(k["elements"]), k["nRs2"], k["nRs3"], k["nFourier"])
        assert rep.shape == (n, expected), f"Got {rep.shape}, expected ({n}, {expected})"

    def test_shape_varies_with_nfourier(self) -> None:
        for lmax in (0, 1, 3, 4):
            k = kw(nFourier=lmax, nRs3=2)
            rep = _m.generate(COORDS_CH2, Z_CH2, **k)
            expected = expected_rep_size(len(k["elements"]), k["nRs2"], k["nRs3"], lmax)
            assert rep.shape[1] == expected, f"nFourier={lmax}: {rep.shape[1]} != {expected}"

    def test_shape_varies_with_nrs3(self) -> None:
        for nrs3 in (2, 5, 8):
            k = kw(nRs3=nrs3)
            rep = _m.generate(COORDS_CH2, Z_CH2, **k)
            expected = expected_rep_size(len(k["elements"]), k["nRs2"], nrs3, k["nFourier"])
            assert rep.shape[1] == expected

    def test_shape_with_more_elements(self) -> None:
        k = kw(elements=[1, 6, 7, 8])
        rep = _m.generate(COORDS_NCO, Z_NCO, **k)
        expected = expected_rep_size(4, k["nRs2"], k["nRs3"], k["nFourier"])
        assert rep.shape[1] == expected


# ---------------------------------------------------------------------------
# 2. compute_rep_size API agreement
# ---------------------------------------------------------------------------


class TestA8ComputeRepSize:
    def test_api_matches_generate(self) -> None:
        k = kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **k)
        nelements = len(k["elements"])
        nabasis = k["nFourier"] + 1
        size_api = _m.compute_rep_size(nelements, k["nRs2"], k["nRs3"], nabasis, THREE_BODY_TYPE)
        assert rep.shape[1] == size_api

    def test_formula_and_api_agree(self) -> None:
        k = kw(elements=[1, 6, 7, 8], nRs2=5, nRs3=4, nFourier=3)
        formula = expected_rep_size(4, 5, 4, 3)
        nabasis = 3 + 1  # nFourier+1
        api = _m.compute_rep_size(4, 5, 4, nabasis, THREE_BODY_TYPE)
        assert formula == api


# ---------------------------------------------------------------------------
# 3. Translation invariance
# ---------------------------------------------------------------------------


class TestA8TranslationInvariance:
    def test_shift(self) -> None:
        k = kw()
        rep1 = _m.generate(COORDS_CH2, Z_CH2, **k)
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(COORDS_CH2 + shift, Z_CH2, **k)
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_shift_nco(self) -> None:
        k = kw()
        rep1 = _m.generate(COORDS_NCO, Z_NCO, **k)
        shift = np.array([-3.1, 4.2, 1.7])
        rep2 = _m.generate(COORDS_NCO + shift, Z_NCO, **k)
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. generate() matches generate_and_gradients() rep output
# ---------------------------------------------------------------------------


class TestA8ForwardConsistency:
    def test_forward_matches_gradient_rep(self) -> None:
        k = kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **k)
        rep_both, _grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **k)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)

    def test_gradient_shape(self) -> None:
        k = kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **k)
        n, rep_size = rep.shape
        assert grad.shape == (n, rep_size, n * 3)

    def test_forward_matches_grad_rep_nco(self) -> None:
        k = kw()
        rep_only = _m.generate(COORDS_NCO, Z_NCO, **k)
        rep_both, _ = _m.generate_and_gradients(COORDS_NCO, Z_NCO, **k)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# 5. Same-element j<->k symmetry
# ---------------------------------------------------------------------------


class TestA8SameElementSymmetry:
    def test_swap_same_element_neighbors(self) -> None:
        """Swapping two same-element (H) atoms j and k must leave rep of center (C) unchanged."""
        # CH2: atom 0=C (center), atoms 1,2=H (same element)
        # Original: C at 0, H1 at 1, H2 at 2
        coords_orig = COORDS_CH2.copy()
        # Swapped: C at 0, H2 at 1, H1 at 2
        coords_swap = coords_orig[[0, 2, 1], :]
        z_swap = Z_CH2[[0, 2, 1]]

        k = kw(elements=[1, 6])
        rep_orig = _m.generate(coords_orig, Z_CH2, **k)
        rep_swap = _m.generate(coords_swap, z_swap, **k)

        # The representation of atom 0 (center C) must be identical
        np.testing.assert_allclose(
            rep_orig[0],
            rep_swap[0],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Center-atom rep changed after swapping same-element neighbors",
        )

    def test_chan_b_symmetric_same_element(self) -> None:
        """For same-element pair: radial_B = phi_j*phi_k must be symmetric."""
        # Construct a symmetric H-H-H-like system where we can extract A and B channels.
        # Use 3 H atoms all in range. The center is atom 1, neighbors are 0 and 2.
        coords = np.array(
            [
                [0.00, 0.00, 0.00],
                [1.50, 0.00, 0.00],  # center
                [1.50, 1.20, 0.00],
            ],
            dtype=np.float64,
        )
        z = np.array([1, 1, 1], dtype=np.int32)
        k = kw(elements=[1], nRs2=2, nRs3=2, nFourier=1)

        rep_orig = _m.generate(coords, z, **k)
        # Swap neighbors 0 and 2 as seen from center 1
        coords_swap = coords[[2, 1, 0], :]
        z_swap = z[[2, 1, 0]]
        rep_swap = _m.generate(coords_swap, z_swap, **k)

        # Center atom rep must be identical (rep_orig[1] vs rep_swap[1])
        np.testing.assert_allclose(
            rep_orig[1],
            rep_swap[1],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Same-element symmetry violated in A8",
        )


# ---------------------------------------------------------------------------
# 6. Different-element ordering: unordered pair invariance
# ---------------------------------------------------------------------------


class TestA8DifferentElementOrdering:
    def test_unordered_pair_invariant(self) -> None:
        """The element-pair channel (C-N) must be the same regardless of atom order."""
        # N-C molecule with a third atom (O) as neighbor
        coords = np.array(
            [
                [0.00, 0.00, 0.00],  # N = center
                [1.20, 0.00, 0.00],  # C
                [0.50, 1.10, 0.00],  # O
            ],
            dtype=np.float64,
        )
        z_NCO = np.array([7, 6, 8], dtype=np.int32)

        # Swapped: C and O positions swapped, N still center
        coords_swap = coords[[0, 2, 1], :]
        z_swap = np.array([7, 8, 6], dtype=np.int32)

        k = kw(elements=[6, 7, 8], nRs2=3, nRs3=2, nFourier=1)
        rep_orig = _m.generate(coords, z_NCO, **k)
        rep_swap = _m.generate(coords_swap, z_swap, **k)

        # Center atom (index 0, N) rep must be identical
        np.testing.assert_allclose(
            rep_orig[0],
            rep_swap[0],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Different-element pair rep changed after reordering neighbors",
        )


# ---------------------------------------------------------------------------
# 7. Near-zero stability
# ---------------------------------------------------------------------------


class TestA8NearZeroStability:
    def test_no_nan_small_rij(self) -> None:
        """Tiny separation (r=0.01 Å) must not produce NaN or inf."""
        coords = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.00, 0.00],  # very close!
                [1.20, 0.60, 0.00],
            ],
            dtype=np.float64,
        )
        z = np.array([6, 1, 8], dtype=np.int32)
        k = kw()
        rep = _m.generate(coords, z, **k)
        assert np.all(np.isfinite(rep)), "NaN or inf in rep for tiny r_ij"

    def test_no_nan_small_rij_gradient(self) -> None:
        coords = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.00, 0.00],
                [1.20, 0.60, 0.00],
            ],
            dtype=np.float64,
        )
        z = np.array([6, 1, 8], dtype=np.int32)
        k = kw()
        rep, grad = _m.generate_and_gradients(coords, z, **k)
        assert np.all(np.isfinite(rep)), "NaN or inf in rep"
        assert np.all(np.isfinite(grad)), "NaN or inf in grad"


# ---------------------------------------------------------------------------
# 8. Joint coupling is non-degenerate (chan_B ≠ chan_A)
# ---------------------------------------------------------------------------


class TestA8JointCouplingNonDegenerate:
    def test_chan_b_differs_from_chan_a(self) -> None:
        """chan_A and chan_B must carry different information for unequal radii."""
        k = kw(use_two_body=False)
        rep = _m.generate(COORDS_CH2, Z_CH2, **k)

        nelements = len(k["elements"])
        two_body_size = nelements * k["nRs2"]
        three_body_block = rep[:, two_body_size:]

        nrs3 = k["nRs3"]
        nabasis = k["nFourier"] + 1
        chan_stride = nrs3 * nabasis

        # Extract chan_A and chan_B for the first element pair
        # Each pair block has size 2 * chan_stride
        pair_block = three_body_block[:, : 2 * chan_stride]
        chan_A = pair_block[:, :chan_stride]
        chan_B = pair_block[:, chan_stride:]

        # chan_B should not be all zero (there is a triplet)
        assert np.any(np.abs(chan_B) > 1e-12), "chan_B is all zero — no joint coupling"
        # chan_A and chan_B must differ for non-trivial geometry
        assert not np.allclose(chan_A, chan_B, atol=1e-10), (
            "chan_A == chan_B — joint coupling is degenerate"
        )


# ---------------------------------------------------------------------------
# 9. Joint coupling changes when radii change jointly
# ---------------------------------------------------------------------------


class TestA8JointCouplingJointSensitivity:
    def test_chan_b_changes_when_both_radii_change(self) -> None:
        """Confirm radial_B changes when BOTH r_ij and r_ik change simultaneously."""
        k = kw(use_two_body=False, elements=[1, 6])
        coords_base = COORDS_CH2.copy()
        rep_base = _m.generate(coords_base, Z_CH2, **k)

        # Perturb both neighbors (j and k) to change both r_ij and r_ik
        coords_pert = coords_base.copy()
        coords_pert[1, 0] += 0.3  # move H1 closer/farther
        coords_pert[2, 1] += 0.3  # move H2
        rep_pert = _m.generate(coords_pert, Z_CH2, **k)

        nelements = len(k["elements"])
        nrs3 = k["nRs3"]
        nabasis = k["nFourier"] + 1
        chan_stride = nrs3 * nabasis
        two_body_size = nelements * k["nRs2"]

        three_base = rep_base[:, two_body_size:]
        three_pert = rep_pert[:, two_body_size:]
        chan_B_base = three_base[:, chan_stride : 2 * chan_stride]
        chan_B_pert = three_pert[:, chan_stride : 2 * chan_stride]

        assert not np.allclose(chan_B_base, chan_B_pert, atol=1e-10), (
            "chan_B did not change when both radii changed — joint coupling broken"
        )

    def test_chan_b_not_just_marginal(self) -> None:
        """chan_B = phi_j*phi_k should differ from sqrt(chan_A^2) style marginal behavior."""
        # Use an asymmetric geometry (r_ij != r_ik) so the product phi_j*phi_k differs
        # from e.g. (phi_j + phi_k)^2 / 4.
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],  # deliberately different distance
            ],
            dtype=np.float64,
        )
        z = np.array([1, 1, 1], dtype=np.int32)
        k = kw(elements=[1], nRs2=2, nRs3=3, nFourier=1, use_two_body=False)
        rep = _m.generate(coords, z, **k)

        nrs3 = k["nRs3"]
        nabasis = k["nFourier"] + 1
        chan_stride = nrs3 * nabasis

        # Focus on center atom (index 0 or 1 — pick the one with nonzero three-body)
        for i in range(len(z)):
            chan_A = rep[i, :chan_stride]
            chan_B = rep[i, chan_stride : 2 * chan_stride]
            if np.any(np.abs(chan_A) > 1e-12):
                # chan_B should not equal 0.25 * chan_A^2 elementwise (that would be wrong)
                ratio = chan_B / (chan_A + 1e-14)
                # ratio should not be nearly constant (it is for symmetric marginals)
                std_ratio = np.std(ratio)
                assert std_ratio > 1e-6 or np.any(np.abs(chan_B - chan_A) > 1e-8), (
                    "chan_B appears to be a trivial function of chan_A only"
                )
                break


# ---------------------------------------------------------------------------
# 10. Single-neighbor: three-body block must be zero
# ---------------------------------------------------------------------------


class TestA8SingleNeighborThreeBodyZero:
    def test_two_atom_molecule_three_body_zero(self) -> None:
        """2-atom molecule has no triplets → three-body block must be exactly zero."""
        k = kw(elements=[1])
        rep = _m.generate(COORDS_H2, Z_H2, **k)
        two_body_size = 1 * k["nRs2"]  # 1 element
        three_body = rep[:, two_body_size:]
        np.testing.assert_allclose(
            three_body, 0.0, atol=1e-14, err_msg="Three-body block nonzero for 2-atom molecule"
        )

    def test_two_atom_molecule_two_body_nonzero(self) -> None:
        """Two-body block should still be nonzero for 2-atom molecule."""
        k = kw(elements=[1])
        rep = _m.generate(COORDS_H2, Z_H2, **k)
        two_body_size = 1 * k["nRs2"]
        two_body = rep[:, :two_body_size]
        assert np.any(np.abs(two_body) > 1e-12), "Two-body block is zero for 2-atom molecule"


# ---------------------------------------------------------------------------
# 11. three_body_weight=0 zeroes three-body block
# ---------------------------------------------------------------------------


class TestA8ZeroWeight:
    def test_zero_weight_zeroes_three_body(self) -> None:
        k = kw(three_body_weight=0.0)
        rep = _m.generate(COORDS_CH2, Z_CH2, **k)
        nelements = len(k["elements"])
        two_body_size = nelements * k["nRs2"]
        np.testing.assert_allclose(
            rep[:, two_body_size:],
            0.0,
            atol=1e-14,
            err_msg="Three-body block nonzero for three_body_weight=0",
        )

    def test_use_three_body_false_zeroes_block(self) -> None:
        k = kw(use_three_body=False)
        rep = _m.generate(COORDS_CH2, Z_CH2, **k)
        nelements = len(k["elements"])
        two_body_size = nelements * k["nRs2"]
        np.testing.assert_allclose(
            rep[:, two_body_size:],
            0.0,
            atol=1e-14,
            err_msg="Three-body block nonzero for use_three_body=False",
        )


# ---------------------------------------------------------------------------
# 12. Analytic gradient vs central finite differences
# ---------------------------------------------------------------------------


class TestA8AnalyticGradient:
    @pytest.mark.parametrize(
        ("coords", "z", "label"),
        [
            (COORDS_CH2, Z_CH2, "CH2"),
            (COORDS_NCO, Z_NCO, "NCO"),
        ],
    )
    def test_grad_matches_fd(self, coords: np.ndarray, z: np.ndarray, label: str) -> None:
        k = kw()
        rep, grad = _m.generate_and_gradients(coords, z, **k)
        n, rep_size = rep.shape
        h = 1e-5
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = coords.copy()
                cm = coords.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, z, **k)
                rm = _m.generate(cm, z, **k)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(
            grad,
            fd_grad,
            rtol=5e-5,
            atol=5e-7,
            err_msg=f"Analytic gradient does not match FD for {label}",
        )

    def test_grad_matches_fd_energy_only_pair(self) -> None:
        """Test gradient also for use_atm=False (simpler ATM chain rule)."""
        k = kw(use_atm=False)
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **k)
        n, rep_size = rep.shape
        h = 1e-5
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **k)
                rm = _m.generate(cm, Z_CH2, **k)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-5, atol=5e-7)

    def test_grad_matches_fd_different_elements(self) -> None:
        """Gradient check for different-element pair (low-Z/high-Z branching)."""
        k = kw(elements=[6, 7, 8])
        rep, grad = _m.generate_and_gradients(COORDS_NCO, Z_NCO, **k)
        n, rep_size = rep.shape
        h = 1e-5
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_NCO.copy()
                cm = COORDS_NCO.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_NCO, **k)
                rm = _m.generate(cm, Z_NCO, **k)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-5, atol=5e-7)


# ---------------------------------------------------------------------------
# 13. Translational invariance of gradient (sum over atoms ≈ 0)
# ---------------------------------------------------------------------------


class TestA8GradientTranslationInvariance:
    def test_grad_sum_over_atoms_is_zero(self) -> None:
        """sum_{a} dQ/dR_a = 0 for each (center atom, feature, Cartesian direction)."""
        k = kw()
        _, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **k)
        n = len(Z_CH2)
        # grad shape: (n_atoms, rep_size, n_atoms*3)
        # reshape to (n_atoms, rep_size, n_atoms, 3), sum over axis 2
        g = grad.reshape(n, grad.shape[1], n, 3)
        summed = g.sum(axis=2)  # (n_atoms, rep_size, 3)
        np.testing.assert_allclose(
            summed,
            0.0,
            atol=1e-7,
            err_msg="Gradient sum over atoms is not zero (broken translational invariance)",
        )


# ---------------------------------------------------------------------------
# 14. ATM flag on/off gives different results
# ---------------------------------------------------------------------------


class TestA8ATMFlag:
    def test_atm_on_off_differ(self) -> None:
        """Reps with and without ATM factor must differ on a non-trivial geometry."""
        k_atm = kw(use_atm=True)
        k_no = kw(use_atm=False)
        rep_atm = _m.generate(COORDS_CH2, Z_CH2, **k_atm)
        rep_no = _m.generate(COORDS_CH2, Z_CH2, **k_no)
        nelements = len(k_atm["elements"])
        two_body_size = nelements * k_atm["nRs2"]
        # Three-body blocks must differ
        assert not np.allclose(rep_atm[:, two_body_size:], rep_no[:, two_body_size:], atol=1e-8), (
            "ATM on/off produced identical three-body blocks"
        )

    def test_two_body_unaffected_by_atm(self) -> None:
        """Two-body block must be unaffected by use_atm flag."""
        k_atm = kw(use_atm=True)
        k_no = kw(use_atm=False)
        rep_atm = _m.generate(COORDS_CH2, Z_CH2, **k_atm)
        rep_no = _m.generate(COORDS_CH2, Z_CH2, **k_no)
        nelements = len(k_atm["elements"])
        two_body_size = nelements * k_atm["nRs2"]
        np.testing.assert_allclose(
            rep_atm[:, :two_body_size], rep_no[:, :two_body_size], atol=1e-14
        )


# ---------------------------------------------------------------------------
# 15. Legendre angular basis specific checks
# ---------------------------------------------------------------------------


class TestA8LegendreAngular:
    def test_nfourier_0_gives_only_P0(self) -> None:
        """nFourier=0 → nabasis=1 → only P_0=1 (isotropic angular)."""
        k = kw(nFourier=0, nRs3=2, use_two_body=False)
        rep = _m.generate(COORDS_CH2, Z_CH2, **k)
        nelements = len(k["elements"])
        two_body_size = nelements * k["nRs2"]
        three_body = rep[:, two_body_size:]
        assert three_body.shape[1] > 0  # should have channels
        assert (
            rep.shape[1]
            == two_body_size
            + expected_rep_size(nelements, k["nRs2"], k["nRs3"], 0)
            - nelements * k["nRs2"]
        )

    def test_angular_weight_decay(self) -> None:
        """Higher-l Legendre terms should be suppressed by exp(-zeta*l*(l+1))."""
        # With large zeta, higher l terms are heavily suppressed
        k_large = kw(zeta=10.0, nFourier=3, use_two_body=False)
        k_small = kw(zeta=0.01, nFourier=3, use_two_body=False)
        rep_large = _m.generate(COORDS_CH2, Z_CH2, **k_large)
        rep_small = _m.generate(COORDS_CH2, Z_CH2, **k_small)

        # The representations should differ (angular weights changed)
        nelements = len(k_large["elements"])
        two_body_size = nelements * k_large["nRs2"]
        assert not np.allclose(
            rep_large[:, two_body_size:], rep_small[:, two_body_size:], atol=1e-8
        ), "Legendre angular weight decay (zeta) has no effect"

    def test_zeta_zero_recovers_uniform_weights(self) -> None:
        """zeta=0 → w_l=1 for all l (no angular decay)."""
        k = kw(zeta=0.0, nFourier=2)
        rep = _m.generate(COORDS_CH2, Z_CH2, **k)
        # Should be finite and nonzero
        assert np.all(np.isfinite(rep))
        nelements = len(k["elements"])
        two_body_size = nelements * k["nRs2"]
        three_body = rep[:, two_body_size:]
        assert np.any(np.abs(three_body) > 1e-12)
