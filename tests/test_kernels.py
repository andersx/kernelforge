import numpy as np
import pytest

from kernelforge import _kernels   # adjust if your module is just `import _kernels`

def test_kernel_symm_shape_and_symmetry():
    rng = np.random.default_rng(0)
    n, d = 4, 2
    X = rng.normal(size=(n, d))
    K = _kernels.kernel_symm(X, alpha=-0.5)

    assert K.shape == (n, n)

    # Diagonal should be exp(0) = 1
    assert np.allclose(np.diag(K), 1.0)

def test_kernel_symm_against_numpy():
    rng = np.random.default_rng(1)
    n, d = 4, 2
    X = rng.normal(size=(n, d))
    alpha = -0.3

    K = _kernels.kernel_symm(X, alpha)

    # Reference computation in pure numpy
    sq_norms = np.sum(X**2, axis=1)
    K_ref = np.empty((n, n))
    for i in range(n):
        for j in range(i+1):
            dist2 = sq_norms[i] + sq_norms[j] - 2 * np.dot(X[i], X[j])
            K_ref[i, j] = np.exp(alpha * dist2)

    # print(K)
    # print(K_ref)
    # assert np.allclose(K, K_ref, rtol=1e-12, atol=1e-12)
    i, j = np.tril_indices(K.shape[0])
    assert np.allclose(K[i, j], K_ref[i, j])
def test_small_input():
    X = np.array([[0.0], [1.0]])
    alpha = -1.0
    K = _kernels.kernel_symm(X, alpha)
    # Distance^2 = 1, so K[0,1] = K[1,0] = exp(-1)
    assert np.allclose(K[1,0], np.exp(-1.0))
    assert K[0,0] == pytest.approx(1.0)
    assert K[1,1] == pytest.approx(1.0)

def _ref_kernel_asymm(X1, X2, alpha):
    """Pure NumPy reference implementation."""
    n1, d = X1.shape
    n2, _ = X2.shape
    sq1 = np.sum(X1**2, axis=1)  # (n1,)
    sq2 = np.sum(X2**2, axis=1)  # (n2,)
    Kref = np.empty((n1, n2))
    for i1 in range(n1):
        for i2 in range(n2):
            dist2 = sq1[i1] + sq2[i2] - 2.0 * np.dot(X1[i1], X2[i2])
            Kref[i1, i2] = np.exp(alpha * dist2)
    return Kref

def test_kernel_asymm_shape_and_values():
    rng = np.random.default_rng(42)
    n1, n2, d = 5, 7, 3
    X1 = rng.normal(size=(n1, d))
    X2 = rng.normal(size=(n2, d))
    alpha = -0.5

    K = _kernels.kernel_asymm(X1, X2, alpha)
    assert K.shape == (n1, n2)

    Kref = _ref_kernel_asymm(X1, X2, alpha)

    print(K[:4, :4])
    print(Kref[:4, :4])
    np.testing.assert_allclose(K, Kref, rtol=1e-12, atol=1e-12)

def test_kernel_asymm_small_case():
    X1 = np.array([[0.0], [1.0]])  # n1=2, d=1
    X2 = np.array([[0.0], [2.0]])  # n2=2, d=1
    alpha = -1.0

    K = _kernels.kernel_asymm(X1, X2, alpha)
    # distances^2:
    # X2[0]=0 vs X1[0]=0 -> 0, exp(0)=1
    # X2[0]=0 vs X1[1]=1 -> 1, exp(-1)
    # X2[1]=2 vs X1[0]=0 -> 4, exp(-4)
    # X2[1]=2 vs X1[1]=1 -> 1, exp(-1)
    Kref = np.array([[1.0, np.exp(-4.0)],
                     [np.exp(-1.0), np.exp(-1.0)]])
    np.testing.assert_allclose(K, Kref, rtol=1e-12, atol=1e-12)

@pytest.mark.slow
def test_time_kernel_symm():

    X = np.random.normal(size=(16000, 512))
    alpha = -1.0 / X.shape[1]

    import time
    start = time.time()
    K = _kernels.kernel_symm(X, alpha)
    end = time.time()
    print(f"Computed kernel_symm in {end - start:.3f} seconds.")

def test_time_kernel_asymm():
    X1 = np.random.normal(size=(16000, 512))
    X2 = np.random.normal(size=(16000, 512))
    alpha = -1.0 / X1.shape[1]

    import time
    start = time.time()
    K = _kernels.kernel_asymm(X1, X2, alpha)
    end = time.time()
    print(f"Computed kernel_asymm in {end - start:.3f} seconds.")

