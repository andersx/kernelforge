import numpy as np
import pytest

# adjust import to however you expose the module;
# below assumes you installed as kernelforge.invdist per your CMake
from kernelforge import invdist_repr as invdist


def _strict_upper_pairs(N: int) -> list[tuple[int, int]]:
    """[(0,1), (0,2), ..., (N-2,N-1)]"""
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def test_shapes_and_ordering_and_sparsity() -> None:
    rng = np.random.default_rng(42)
    N = 5
    # well-separated random coords to avoid tiny distances
    R = rng.normal(size=(N, 3)) * 0.4 + np.arange(N)[:, None] * np.array([2.0, 0.0, 0.0])

    x_only = invdist.inverse_distance_upper(R)
    x, J = invdist.inverse_distance_upper_and_jacobian(R)

    M = invdist.num_pairs(N)
    assert x_only.shape == (M,)
    assert x.shape == (M,)
    assert J.shape == (M, 3 * N)
    assert np.allclose(x_only, x)

    # Ordering: rows correspond to strict upper-triangle
    pairs = _strict_upper_pairs(N)
    for p, (i, j) in enumerate(pairs):
        # nonzero atoms should be exactly i and j
        row = J[p].reshape(N, 3)
        nz_atoms = np.where(np.any(np.abs(row) > 0, axis=1))[0]
        assert set(nz_atoms.tolist()) == {i, j}

        # Conservation (translation invariance): sum of per-atom grads is ~0
        assert np.allclose(row.sum(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("N", [4, 5])
def test_jacobian_central_difference(N: int) -> None:
    rng = np.random.default_rng(123)
    # Spread atoms to avoid small r; linear ramp helps conditioning
    base = np.arange(N)[:, None] * np.array([1.5, 0.7, -0.6])
    R = base + rng.normal(scale=0.2, size=(N, 3))

    x, J = invdist.inverse_distance_upper_and_jacobian(R)

    M = invdist.num_pairs(N)
    assert x.shape == (M,)
    assert J.shape == (M, 3 * N)

    # Central finite-difference check
    h = 1e-6
    atol = 5e-6
    rtol = 2e-4

    # Flatten coord index mapping: col c -> atom k = c//3, axis a = c%3
    for c in range(3 * N):
        k, a = divmod(c, 3)
        R_plus = R.copy()
        R_minus = R.copy()
        R_plus[k, a] += h
        R_minus[k, a] -= h

        x_plus = invdist.inverse_distance_upper(R_plus)
        x_minus = invdist.inverse_distance_upper(R_minus)
        # numerical derivative column c
        num_col = (x_plus - x_minus) / (2.0 * h)
        ana_col = J[:, c]

        # Compare robustly across all M features
        np.testing.assert_allclose(ana_col, num_col, rtol=rtol, atol=atol)


def test_eps_stability_no_nan_inf() -> None:
    rng = np.random.default_rng(7)
    N = 4
    R = rng.normal(size=(N, 3)) * 1e-9  # nearly coincident (stress eps path)
    # Use a reasonable eps to avoid division by zero
    x, J = invdist.inverse_distance_upper_and_jacobian(R, eps=1e-8)
    assert np.isfinite(x).all()
    assert np.isfinite(J).all()
