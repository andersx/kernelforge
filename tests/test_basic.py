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
    rep, n = 512, 32000
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
    rep, n = 512, 32000
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

def test_cblas_kernel():

    from kernelforge import ckernel_symm_blas

    rep, n = 512, 32000
    rng = np.random.default_rng(0)
    X = rng.random((n, rep))
    print(X.shape)
    alpha = -1.0 / rep

    K = np.empty((n, n))
    start = time()
    ckernel_symm_blas(X, K, alpha)
    end = time()
    print(f"Kernel CBLAS took {end - start:.2f} seconds for {n} points.")
    print(K[:4,:4])


    K_test = np.zeros((4, 4))
    for i in range(4):
        for j in range(i):
            K_test[i, j] = np.exp(alpha * np.sum(np.square((X[i] - X[j]))))
    print(K_test[:4,:4] - K[:4,:4])


def test_cblas_syrk():

    import numpy as np
    import kernelforge._kernels as kernels
    
    # C-order (default)
    Xc = np.random.randn(32000, 512)
    Kc = np.zeros((32000, 32000))
    kernels.ckernel_syrk_test(Xc, Kc, 0.5)
    
    # F-order (like Fortran)
    Xf = np.asfortranarray(np.random.randn(32000, 512))
    Kf = np.zeros((32000, 32000), order="F")
    kernels.ckernel_syrk_test(Xf, Kf, 0.5)

def test_cblas_syrk2():

    import kernelforge._kernels as kernels

    # Pure C++ benchmark inside the module
    kernels.bench_dsyrk(32000, 512, 1.0)

def test_allocate_internal():

    import numpy as np
    import kernelforge._kernels as kernels
    
    n, rep_size = 32000, 512
    
    # External K, X allocated inside C++
    K = np.zeros((n, n), order="F", dtype=np.float64)
    kernels.bench_dsyrk_Xinternal(K, 1.0)
    
    # External X, K allocated inside C++
    X = np.random.randn(n, rep_size)
    kernels.bench_dsyrk_Kinternal(X, 1.0)

