# import your module
import numpy as np
import pytest

import kernelforge._fchl19 as fchl


def slow_ref_grad(x1, x2, dX2, q1, q2, n1, n2, sigma):
    """
    NumPy reference matching the Fortran logic:
      For each a,j1 and b,j2 with matching labels, accumulate:
        K[a, offs_b + t] += alpha * dot(dX2[b,j2,:,t], d)
      where d = x1[a,j1,:] - x2[b,j2,:]
            l2 = dot(d, d)
            alpha = exp(l2 * inv_2sigma2) * inv_sigma2
    """
    nm1, max_atoms1, rep = x1.shape
    nm2, max_atoms2, rep2 = x2.shape
    assert rep == rep2
    # offsets into derivative axis per molecule b
    offs2 = np.zeros(nm2, dtype=int)
    acc = 0
    for b in range(nm2):
        nb = int(n2[b])
        nb = 0 if nb < 0 else min(nb, max_atoms2)
        offs2[b] = acc
        acc += 3 * nb
    naq2 = acc

    K = np.zeros((nm1, naq2), dtype=np.float64)

    inv_2sigma2 = -1.0 / (2.0 * sigma * sigma)
    inv_sigma2 = -1.0 / (sigma * sigma)

    for a in range(nm1):
        na = int(n1[a])
        na = 0 if na < 0 else min(na, max_atoms1)
        for j1 in range(na):
            lbl1 = q1[a, j1]
            for b in range(nm2):
                nb = int(n2[b])
                nb = 0 if nb < 0 else min(nb, max_atoms2)
                if nb == 0:
                    continue
                base = offs2[b]
                for j2 in range(nb):
                    if lbl1 != q2[b, j2]:
                        continue
                    d = x1[a, j1, :] - x2[b, j2, :]
                    l2 = float(np.dot(d, d))
                    alpha = np.exp(l2 * inv_2sigma2) * inv_sigma2

                    # dX2 slice: shape (rep, 3*nb); but our array is (rep, 3*max_atoms2)
                    # we only take the first 3*nb columns
                    A = dX2[b, j2, :, : 3 * nb]  # (rep, 3*nb)
                    # GEMV: y += alpha * A^T * d
                    K[a, base : base + 3 * nb] += alpha * (A.T @ d)

    return K


@pytest.mark.parametrize("seed", [0, 1234])
def test_fatomic_local_gradient_kernel_matches_reference(seed):
    rng = np.random.default_rng(seed)

    # modest sizes to keep test fast but non-trivial
    nm1, nm2 = 3, 4
    max_atoms1, max_atoms2 = 5, 6
    rep = 7
    n_species = 4
    sigma = 1.23

    # random inputs
    x1 = rng.normal(size=(nm1, max_atoms1, rep)).astype(np.float64)
    x2 = rng.normal(size=(nm2, max_atoms2, rep)).astype(np.float64)

    # dX2: (nm2, max_atoms2, rep, 3*max_atoms2)
    dX2 = rng.normal(size=(nm2, max_atoms2, rep, 3 * max_atoms2)).astype(np.float64)

    q1 = rng.integers(0, n_species, size=(nm1, max_atoms1), dtype=np.int32)
    q2 = rng.integers(0, n_species, size=(nm2, max_atoms2), dtype=np.int32)

    # ensure at least 1 atom in many molecules
    n1 = rng.integers(1, max_atoms1 + 1, size=(nm1,), dtype=np.int32)
    n2 = rng.integers(1, max_atoms2 + 1, size=(nm2,), dtype=np.int32)

    # compute with binding
    K = fchl.fatomic_local_gradient_kernel(x1, x2, dX2, q1, q2, n1, n2, sigma)
    assert K.shape[0] == nm1

    # reference
    K_ref = slow_ref_grad(x1, x2, dX2, q1, q2, n1, n2, sigma)
    assert K.shape == K_ref.shape

    # allow tiny FP diffs (BLAS vs NumPy sum order); be slightly generous
    np.testing.assert_allclose(K, K_ref, rtol=1e-10, atol=1e-10)


def test_no_matches_yields_zero_matrix():
    rng = np.random.default_rng(42)
    nm1, nm2 = 2, 2
    max_atoms1, max_atoms2 = 3, 3
    rep = 4
    sigma = 0.9

    x1 = rng.normal(size=(nm1, max_atoms1, rep)).astype(np.float64)
    x2 = rng.normal(size=(nm2, max_atoms2, rep)).astype(np.float64)
    dX2 = rng.normal(size=(nm2, max_atoms2, rep, 3 * max_atoms2)).astype(np.float64)

    q1 = np.zeros((nm1, max_atoms1), dtype=np.int32)
    q2 = np.ones((nm2, max_atoms2), dtype=np.int32) * 7  # disjoint labels

    n1 = np.full((nm1,), max_atoms1, dtype=np.int32)
    n2 = np.full((nm2,), max_atoms2, dtype=np.int32)

    K = fchl.fatomic_local_gradient_kernel(x1, x2, dX2, q1, q2, n1, n2, sigma)
    assert K.shape == (nm1, 3 * nm2 * max_atoms2)
    assert np.allclose(K, 0.0)


def test_empty_derivatives_returns_nm1_by_0():
    rng = np.random.default_rng(7)
    nm1, nm2 = 3, 2
    max_atoms1, max_atoms2 = 4, 3
    rep = 3
    sigma = 1.0

    x1 = rng.normal(size=(nm1, max_atoms1, rep)).astype(np.float64)
    x2 = rng.normal(size=(nm2, max_atoms2, rep)).astype(np.float64)
    dX2 = rng.normal(size=(nm2, max_atoms2, rep, 3 * max_atoms2)).astype(np.float64)

    q1 = rng.integers(0, 2, size=(nm1, max_atoms1), dtype=np.int32)
    q2 = rng.integers(0, 2, size=(nm2, max_atoms2), dtype=np.int32)

    n1 = rng.integers(0, max_atoms1 + 1, size=(nm1,), dtype=np.int32)
    n2 = np.zeros((nm2,), dtype=np.int32)  # <- no atoms in set-2 => naq2 = 0

    K = fchl.fatomic_local_gradient_kernel(x1, x2, dX2, q1, q2, n1, n2, sigma)
    assert K.shape == (nm1, 0)
    assert K.size == 0


def test_shape_errors_raise_valueerror():
    rng = np.random.default_rng(9)
    nm1, nm2 = 2, 2
    max_atoms1, max_atoms2 = 3, 3
    rep = 4
    sigma = 1.0

    x1 = rng.normal(size=(nm1, max_atoms1, rep)).astype(np.float64)
    x2 = rng.normal(size=(nm2, max_atoms2, rep)).astype(np.float64)
    # wrong last dimension in dX2
    dX2_bad = rng.normal(size=(nm2, max_atoms2, rep, 3 * max_atoms2 - 1)).astype(np.float64)
    q1 = rng.integers(0, 2, size=(nm1, max_atoms1), dtype=np.int32)
    q2 = rng.integers(0, 2, size=(nm2, max_atoms2), dtype=np.int32)
    n1 = rng.integers(0, max_atoms1 + 1, size=(nm1,), dtype=np.int32)
    n2 = rng.integers(0, max_atoms2 + 1, size=(nm2,), dtype=np.int32)

    with pytest.raises(ValueError):
        _ = fchl.fatomic_local_gradient_kernel(x1, x2, dX2_bad, q1, q2, n1, n2, sigma)
