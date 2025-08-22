import numpy as np
from time import time
import kernelforge as kf
import pytest

def test_inverse_distance_shapes():
    X = np.random.rand(5, 3)
    D = kf.inverse_distance(X)
    assert D.shape == (5*4//2,)


def test_kernel_simple_runs():
    rep, n = 512, 64
    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.random((rep, n)))
    alpha = -1.0 / rep

    K = kf.kernel_symm_simple(X, alpha)
    assert K.shape == (n, n)

    # Symmetrize since only upper triangle is written
    iu = np.triu_indices(n, 1)
    K[(iu[1], iu[0])] = K[iu]
    # Check diagonal ~ 1.0
    assert np.allclose(np.diag(K), 1.0)

    # Off-diagonal entries should be between 0 and 1
    off_diag = K[iu]
    assert np.all((off_diag >= 0.0) & (off_diag <= 1.0))


def test_kernel_blas_runs():
    rep, n = 512, 64
    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.random((rep, n)))
    alpha = -1.0 / rep

    K = kf.kernel_symm_blas(X, alpha)
    assert K.shape == (n, n)

    # Symmetrize since only upper triangle is written
    iu = np.triu_indices(n, 1)
    K[(iu[1], iu[0])] = K[iu]

    # Check diagonal ~ 1.0
    assert np.allclose(np.diag(K), 1.0)

    # Off-diagonal entries should be between 0 and 1
    off_diag = K[iu]
    assert np.all((off_diag >= 0.0) & (off_diag <= 1.0))

@pytest.mark.slow
def test_kernel_simple_time():
    rep, n = 512, 16000
    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.random((rep, n)))
    alpha = -1.0 / rep

    start = time()
    K = kf.kernel_symm_simple(X, alpha)
    end = time()
    print(f"Kernel SIMPLE took {end - start:.2f} seconds for {n} points.")
    assert K.shape == (n, n)

    # Symmetrize since only upper triangle is written
    iu = np.triu_indices(n, 1)
    K[(iu[1], iu[0])] = K[iu]
    # Check diagonal ~ 1.0
    assert np.allclose(np.diag(K), 1.0)

    # Off-diagonal entries should be between 0 and 1
    off_diag = K[iu]
    assert np.all((off_diag >= 0.0) & (off_diag <= 1.0))


@pytest.mark.slow
def test_kernel_blas_time():
    rep, n = 512, 16000
    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.random((rep, n)))
    alpha = -1.0 / rep

    start = time()
    K = kf.kernel_symm_blas(X, alpha)

    end = time()
    print(f"Kernel BLAS took {end - start:.2f} seconds for {n} points.")
    assert K.shape == (n, n)

    # Symmetrize since only upper triangle is written
    iu = np.triu_indices(n, 1)
    K[(iu[1], iu[0])] = K[iu]

    # Check diagonal ~ 1.0
    assert np.allclose(np.diag(K), 1.0)

    # Off-diagonal entries should be between 0 and 1
    off_diag = K[iu]
    assert np.all((off_diag >= 0.0) & (off_diag <= 1.0))
