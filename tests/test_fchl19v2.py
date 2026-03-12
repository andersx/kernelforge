"""Tests for the fchl19v2_repr module.

Organised as parametrized tests over two-body and three-body variant strings.
Each combination is tested for:
  - Output shape matches compute_rep_size
  - Translation invariance
  - generate() matches generate_and_gradients() rep output
  - Three-body block is zero when three_body_weight=0
  - Analytic gradient matches central finite differences
"""

from typing import Any

import numpy as np

from kernelforge import fchl19v2_repr as _m

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Asymmetric CH2-like geometry — avoids accidental cancellations
COORDS_CH2 = np.array(
    [[0.00, 0.00, 0.00], [1.10, 0.00, 0.00], [-0.20, 0.95, 0.15]],
    dtype=np.float64,
)
Z_CH2 = np.array([6, 1, 1], dtype=np.int32)

# Tiny H-O-H for shape tests
COORDS_HOH = np.array([[-0.9572, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9572, 0.0, 0.0]], dtype=np.float64)
Z_HOH = np.array([1, 8, 1], dtype=np.int32)

# Two-body types currently implemented
TWO_BODY_TYPES = [
    "log_normal",
    "gaussian_r",
    "gaussian_log_r",
    "gaussian_r_no_pow",
    "bessel",
]

# Three-body types currently implemented
THREE_BODY_TYPES_IMPLEMENTED = [
    "odd_fourier_rbar",
]

# Small basis params for fast tests
SMALL_KW: dict[str, Any] = {
    "elements": [1, 6],
    "nRs2": 4,
    "nRs3": 3,
    "nFourier": 1,
    "eta2": 0.32,
    "eta3": 2.7,
    "zeta": np.pi,
    "rcut": 5.0,
    "acut": 5.0,
    "two_body_decay": 1.3,
    "three_body_decay": 0.8,
    "three_body_weight": 2.5,
}


def _rep_size(nelements: int, nRs2: int, nRs3: int, nFourier: int) -> int:
    """Expected rep size for OddFourier_Rbar (nabasis = 2*nFourier)."""
    n_pairs = nelements * (nelements + 1) // 2
    return nelements * nRs2 + n_pairs * nRs3 * (2 * nFourier)


# ---------------------------------------------------------------------------
# T1 — LogNormal two-body
# ---------------------------------------------------------------------------


class TestT1LogNormal:
    TB = "log_normal"
    AB = "odd_fourier_rbar"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {**SMALL_KW, "two_body_type": self.TB, "three_body_type": self.AB, **overrides}

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        expected = _rep_size(len(kw["elements"]), kw["nRs2"], kw["nRs3"], kw["nFourier"])
        assert rep.shape == (n, expected)

    def test_compute_rep_size_matches(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        size = _m.compute_rep_size(
            len(kw["elements"]),
            kw["nRs2"],
            kw["nRs3"],
            2 * kw["nFourier"],
            kw["three_body_type"],
        )
        assert rep.shape[1] == size

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        z = np.array([1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)
        n, rep_size = rep_only.shape
        assert grad.shape == (n, rep_size, n * 3)

    def test_three_body_zero_weight(self) -> None:
        kw = self.kw(three_body_weight=0.0)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.allclose(rep[:, two_body_size:], 0.0, atol=1e-14)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)

    def test_translational_invariance_of_grad(self) -> None:
        kw = self.kw(elements=[1], nRs3=2)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        z = np.array([1, 1], dtype=np.int32)
        _, grad = _m.generate_and_gradients(
            coords, z, **{k: v for k, v in kw.items() if k != "elements"}, elements=[1]
        )
        # Sum over all atoms must be ~0 for each (center, feature, dim)
        summed = grad.reshape(grad.shape[0], grad.shape[1], -1, 3).sum(axis=2)
        np.testing.assert_allclose(summed, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# T2 — GaussianR two-body
# ---------------------------------------------------------------------------


class TestT2GaussianR:
    TB = "gaussian_r"
    AB = "odd_fourier_rbar"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {**SMALL_KW, "two_body_type": self.TB, "three_body_type": self.AB, **overrides}

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        expected = _rep_size(len(kw["elements"]), kw["nRs2"], kw["nRs3"], kw["nFourier"])
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        z = np.array([1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, _ = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)

    def test_three_body_zero_weight(self) -> None:
        kw = self.kw(three_body_weight=0.0)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.allclose(rep[:, two_body_size:], 0.0, atol=1e-14)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# T3 — GaussianLogR two-body
# ---------------------------------------------------------------------------


class TestT3GaussianLogR:
    TB = "gaussian_log_r"
    AB = "odd_fourier_rbar"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {**SMALL_KW, "two_body_type": self.TB, "three_body_type": self.AB, **overrides}

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        expected = _rep_size(len(kw["elements"]), kw["nRs2"], kw["nRs3"], kw["nFourier"])
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        z = np.array([1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, _ = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)

    def test_three_body_zero_weight(self) -> None:
        kw = self.kw(three_body_weight=0.0)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.allclose(rep[:, two_body_size:], 0.0, atol=1e-14)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# T4 — GaussianRNoPow two-body
# ---------------------------------------------------------------------------


class TestT4GaussianRNoPow:
    TB = "gaussian_r_no_pow"
    AB = "odd_fourier_rbar"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {**SMALL_KW, "two_body_type": self.TB, "three_body_type": self.AB, **overrides}

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        expected = _rep_size(len(kw["elements"]), kw["nRs2"], kw["nRs3"], kw["nFourier"])
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        z = np.array([1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, _ = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)

    def test_three_body_zero_weight(self) -> None:
        kw = self.kw(three_body_weight=0.0)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.allclose(rep[:, two_body_size:], 0.0, atol=1e-14)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# T5 — Bessel two-body
# ---------------------------------------------------------------------------


class TestT5Bessel:
    TB = "bessel"
    AB = "odd_fourier_rbar"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {**SMALL_KW, "two_body_type": self.TB, "three_body_type": self.AB, **overrides}

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        expected = _rep_size(len(kw["elements"]), kw["nRs2"], kw["nRs3"], kw["nFourier"])
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        z = np.array([1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, _ = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)

    def test_three_body_zero_weight(self) -> None:
        kw = self.kw(three_body_weight=0.0)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.allclose(rep[:, two_body_size:], 0.0, atol=1e-14)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# A1 — OddFourier + Rbar three-body (baseline)
# ---------------------------------------------------------------------------


class TestA1OddFourierRbar:
    TB = "log_normal"
    AB = "odd_fourier_rbar"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {**SMALL_KW, "two_body_type": self.TB, "three_body_type": self.AB, **overrides}

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        expected = _rep_size(len(kw["elements"]), kw["nRs2"], kw["nRs3"], kw["nFourier"])
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]], dtype=np.float64)
        z = np.array([1, 1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_three_body_nonzero(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.linalg.norm(rep[:, two_body_size:]) > 0.0

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)
        n, rep_size = rep_only.shape
        assert grad.shape == (n, rep_size, n * 3)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# A2 — CosineSeries + Rbar three-body
# ---------------------------------------------------------------------------


class TestA2CosineRbar:
    TB = "log_normal"
    AB = "cosine_rbar"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {
            **SMALL_KW,
            "two_body_type": self.TB,
            "three_body_type": self.AB,
            "nCosine": 4,
            **overrides,
        }

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        nelements = len(kw["elements"])
        n_pairs = nelements * (nelements + 1) // 2
        nCosine = kw["nCosine"]
        expected = nelements * kw["nRs2"] + n_pairs * kw["nRs3"] * nCosine
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]], dtype=np.float64)
        z = np.array([1, 1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_three_body_nonzero(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.linalg.norm(rep[:, two_body_size:]) > 0.0

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)
        n, rep_size = rep_only.shape
        assert grad.shape == (n, rep_size, n * 3)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# A3 — OddFourier + SplitR three-body
# ---------------------------------------------------------------------------

# Shared SplitR keyword defaults (nRs3_minus must be > 0)
SPLITR_KW: dict[str, Any] = {**SMALL_KW, "nRs3_minus": 3}


class TestA3OddFourierSplitR:
    TB = "log_normal"
    AB = "odd_fourier_split_r"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {**SPLITR_KW, "two_body_type": self.TB, "three_body_type": self.AB, **overrides}

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        nelements = len(kw["elements"])
        n_pairs = nelements * (nelements + 1) // 2
        nabasis = 2 * kw["nFourier"]
        expected = nelements * kw["nRs2"] + n_pairs * kw["nRs3"] * kw["nRs3_minus"] * nabasis
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]], dtype=np.float64)
        z = np.array([1, 1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_three_body_nonzero(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.linalg.norm(rep[:, two_body_size:]) > 0.0

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)
        n, rep_size = rep_only.shape
        assert grad.shape == (n, rep_size, n * 3)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# A4 — CosineSeries + SplitR three-body
# ---------------------------------------------------------------------------


class TestA4CosineSplitR:
    TB = "log_normal"
    AB = "cosine_split_r"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {
            **SPLITR_KW,
            "two_body_type": self.TB,
            "three_body_type": self.AB,
            "nCosine": 4,
            **overrides,
        }

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        nelements = len(kw["elements"])
        n_pairs = nelements * (nelements + 1) // 2
        nCosine = kw["nCosine"]
        expected = nelements * kw["nRs2"] + n_pairs * kw["nRs3"] * kw["nRs3_minus"] * nCosine
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]], dtype=np.float64)
        z = np.array([1, 1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_three_body_nonzero(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.linalg.norm(rep[:, two_body_size:]) > 0.0

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)
        n, rep_size = rep_only.shape
        assert grad.shape == (n, rep_size, n * 3)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# A5 — CosineSeries + SplitR + NoATM three-body
# ---------------------------------------------------------------------------


class TestA5CosineSplitRNoATM:
    TB = "log_normal"
    AB = "cosine_split_r_no_atm"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {
            **SPLITR_KW,
            "two_body_type": self.TB,
            "three_body_type": self.AB,
            "nCosine": 4,
            **overrides,
        }

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        nelements = len(kw["elements"])
        n_pairs = nelements * (nelements + 1) // 2
        nCosine = kw["nCosine"]
        expected = nelements * kw["nRs2"] + n_pairs * kw["nRs3"] * kw["nRs3_minus"] * nCosine
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]], dtype=np.float64)
        z = np.array([1, 1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_three_body_nonzero(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.linalg.norm(rep[:, two_body_size:]) > 0.0

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)
        n, rep_size = rep_only.shape
        assert grad.shape == (n, rep_size, n * 3)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# A6 — OddFourier + ElementResolved three-body
# ---------------------------------------------------------------------------


class TestA6OddFourierElementResolved:
    TB = "log_normal"
    AB = "odd_fourier_element_resolved"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {**SPLITR_KW, "two_body_type": self.TB, "three_body_type": self.AB, **overrides}

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        nelements = len(kw["elements"])
        nabasis = 2 * kw["nFourier"]
        nRs3 = kw["nRs3"]
        nRs3_minus = kw["nRs3_minus"]
        # diagonal (B==C) blocks + off-diagonal ordered (B!=C) blocks
        three_body_size = (
            nelements * nRs3 * nRs3_minus * nabasis
            + nelements * (nelements - 1) * nRs3 * nRs3 * nabasis
        )
        expected = nelements * kw["nRs2"] + three_body_size
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]], dtype=np.float64)
        z = np.array([1, 1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_three_body_nonzero(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.linalg.norm(rep[:, two_body_size:]) > 0.0

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)
        n, rep_size = rep_only.shape
        assert grad.shape == (n, rep_size, n * 3)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)


# ---------------------------------------------------------------------------
# A7 — CosineSeries + ElementResolved three-body
# ---------------------------------------------------------------------------


class TestBooleanFlags:
    """Tests for use_two_body, use_three_body, use_atm flags on A1 (odd_fourier_rbar)."""

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {
            **SMALL_KW,
            "two_body_type": "log_normal",
            "three_body_type": "odd_fourier_rbar",
            **overrides,
        }

    # ------------------------------------------------------------------
    # use_two_body=False
    # ------------------------------------------------------------------

    def test_use_two_body_false_zeroes_two_body_section(self) -> None:
        kw = self.kw(use_two_body=False)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.allclose(rep[:, :two_body_size], 0.0, atol=1e-14)

    def test_use_two_body_false_three_body_nonzero(self) -> None:
        kw = self.kw(use_two_body=False)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.linalg.norm(rep[:, two_body_size:]) > 0.0

    def test_use_two_body_false_differs_from_default(self) -> None:
        rep_default = _m.generate(COORDS_CH2, Z_CH2, **self.kw())
        rep_no_tb = _m.generate(COORDS_CH2, Z_CH2, **self.kw(use_two_body=False))
        assert not np.allclose(rep_default, rep_no_tb)

    def test_use_two_body_false_shape_unchanged(self) -> None:
        rep_default = _m.generate(COORDS_CH2, Z_CH2, **self.kw())
        rep_no_tb = _m.generate(COORDS_CH2, Z_CH2, **self.kw(use_two_body=False))
        assert rep_default.shape == rep_no_tb.shape

    def test_use_two_body_false_grad_shape_unchanged(self) -> None:
        _, grad_default = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **self.kw())
        _, grad_no_tb = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **self.kw(use_two_body=False))
        assert grad_default.shape == grad_no_tb.shape

    def test_use_two_body_false_grad_two_body_zeroed(self) -> None:
        kw = self.kw(use_two_body=False)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        _, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        assert np.allclose(grad[:, :two_body_size, :], 0.0, atol=1e-14)

    def test_use_two_body_false_grad_matches_fd(self) -> None:
        kw = self.kw(use_two_body=False)
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)

    # ------------------------------------------------------------------
    # use_three_body=False
    # ------------------------------------------------------------------

    def test_use_three_body_false_zeroes_three_body_section(self) -> None:
        kw = self.kw(use_three_body=False)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.allclose(rep[:, two_body_size:], 0.0, atol=1e-14)

    def test_use_three_body_false_two_body_nonzero(self) -> None:
        kw = self.kw(use_three_body=False)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.linalg.norm(rep[:, :two_body_size]) > 0.0

    def test_use_three_body_false_differs_from_default(self) -> None:
        rep_default = _m.generate(COORDS_CH2, Z_CH2, **self.kw())
        rep_no_ab = _m.generate(COORDS_CH2, Z_CH2, **self.kw(use_three_body=False))
        assert not np.allclose(rep_default, rep_no_ab)

    def test_use_three_body_false_shape_unchanged(self) -> None:
        rep_default = _m.generate(COORDS_CH2, Z_CH2, **self.kw())
        rep_no_ab = _m.generate(COORDS_CH2, Z_CH2, **self.kw(use_three_body=False))
        assert rep_default.shape == rep_no_ab.shape

    def test_use_three_body_false_grad_three_body_zeroed(self) -> None:
        kw = self.kw(use_three_body=False)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        _, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        assert np.allclose(grad[:, two_body_size:, :], 0.0, atol=1e-14)

    def test_use_three_body_false_grad_matches_fd(self) -> None:
        kw = self.kw(use_three_body=False)
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)

    # ------------------------------------------------------------------
    # use_atm=False
    # ------------------------------------------------------------------

    def test_use_atm_false_differs_from_default(self) -> None:
        rep_default = _m.generate(COORDS_CH2, Z_CH2, **self.kw())
        rep_no_atm = _m.generate(COORDS_CH2, Z_CH2, **self.kw(use_atm=False))
        assert not np.allclose(rep_default, rep_no_atm)

    def test_use_atm_false_rep_nonzero(self) -> None:
        rep = _m.generate(COORDS_CH2, Z_CH2, **self.kw(use_atm=False))
        assert np.linalg.norm(rep) > 0.0

    def test_use_atm_false_shape_unchanged(self) -> None:
        rep_default = _m.generate(COORDS_CH2, Z_CH2, **self.kw())
        rep_no_atm = _m.generate(COORDS_CH2, Z_CH2, **self.kw(use_atm=False))
        assert rep_default.shape == rep_no_atm.shape

    def test_use_atm_false_grad_shape_unchanged(self) -> None:
        _, grad_default = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **self.kw())
        _, grad_no_atm = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **self.kw(use_atm=False))
        assert grad_default.shape == grad_no_atm.shape

    def test_use_atm_false_grad_matches_fd(self) -> None:
        kw = self.kw(use_atm=False)
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)

    # ------------------------------------------------------------------
    # use_atm=False on A5 (cosine_split_r_no_atm) — should be a no-op
    # ------------------------------------------------------------------

    def test_use_atm_false_noop_on_a5(self) -> None:
        """A5 has no ATM by definition; use_atm=False should give identical results."""
        kw_base: dict[str, Any] = {
            **SPLITR_KW,
            "two_body_type": "log_normal",
            "three_body_type": "cosine_split_r_no_atm",
            "nCosine": 4,
        }
        rep_default = _m.generate(COORDS_CH2, Z_CH2, **kw_base)
        rep_no_atm = _m.generate(COORDS_CH2, Z_CH2, **{**kw_base, "use_atm": False})  # type: ignore[arg-type]
        np.testing.assert_allclose(rep_default, rep_no_atm, rtol=1e-14, atol=1e-14)

    # ------------------------------------------------------------------
    # Combinations
    # ------------------------------------------------------------------

    def test_both_two_and_three_body_false_all_zero(self) -> None:
        kw = self.kw(use_two_body=False, use_three_body=False)
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        assert np.allclose(rep, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# A7 — CosineSeries + ElementResolved three-body
# ---------------------------------------------------------------------------


class TestA7CosineElementResolved:
    TB = "log_normal"
    AB = "cosine_element_resolved"

    def kw(self, **overrides: Any) -> dict[str, Any]:
        return {
            **SPLITR_KW,
            "two_body_type": self.TB,
            "three_body_type": self.AB,
            "nCosine": 4,
            **overrides,
        }

    def test_shape(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        n = len(Z_CH2)
        nelements = len(kw["elements"])
        nCosine = kw["nCosine"]
        nRs3 = kw["nRs3"]
        nRs3_minus = kw["nRs3_minus"]
        three_body_size = (
            nelements * nRs3 * nRs3_minus * nCosine
            + nelements * (nelements - 1) * nRs3 * nRs3 * nCosine
        )
        expected = nelements * kw["nRs2"] + three_body_size
        assert rep.shape == (n, expected)

    def test_translation_invariance(self) -> None:
        kw = self.kw()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]], dtype=np.float64)
        z = np.array([1, 1, 1], dtype=np.int32)
        rep1 = _m.generate(
            coords, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        shift = np.array([7.3, -2.1, 5.0])
        rep2 = _m.generate(
            coords + shift, z, elements=[1], **{k: v for k, v in kw.items() if k != "elements"}
        )
        np.testing.assert_allclose(rep1, rep2, rtol=1e-10, atol=1e-12)

    def test_three_body_nonzero(self) -> None:
        kw = self.kw()
        rep = _m.generate(COORDS_CH2, Z_CH2, **kw)
        two_body_size = len(kw["elements"]) * kw["nRs2"]
        assert np.linalg.norm(rep[:, two_body_size:]) > 0.0

    def test_generate_matches_generate_and_gradients(self) -> None:
        kw = self.kw()
        rep_only = _m.generate(COORDS_CH2, Z_CH2, **kw)
        rep_both, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        np.testing.assert_allclose(rep_only, rep_both, rtol=1e-12, atol=1e-14)
        n, rep_size = rep_only.shape
        assert grad.shape == (n, rep_size, n * 3)

    def test_analytic_grad_matches_fd(self) -> None:
        kw = self.kw()
        rep, grad = _m.generate_and_gradients(COORDS_CH2, Z_CH2, **kw)
        n, rep_size = rep.shape
        h = 1e-6
        fd_grad = np.zeros((n, rep_size, n * 3), dtype=np.float64)
        for a in range(n):
            for d in range(3):
                cp = COORDS_CH2.copy()
                cm = COORDS_CH2.copy()
                cp[a, d] += h
                cm[a, d] -= h
                rp = _m.generate(cp, Z_CH2, **kw)
                rm = _m.generate(cm, Z_CH2, **kw)
                fd_grad[:, :, a * 3 + d] = (rp - rm) / (2.0 * h)
        np.testing.assert_allclose(grad, fd_grad, rtol=5e-6, atol=5e-8)
