"""Verify local_kernels accepts the public X/dX/Q/N keyword names.

Positional call sites are covered elsewhere; this guards the pybind keyword
contract against stub/bindings drift.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import kernelforge.local_kernels as lk


def _dataset(
    nm1: int,
    nm2: int,
    max_atoms: int,
    rep: int,
    *,
    seed: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=(nm1, max_atoms, rep)).astype(np.float64)
    x2 = rng.normal(size=(nm2, max_atoms, rep)).astype(np.float64)
    dx1 = rng.normal(size=(nm1, max_atoms, rep, 3 * max_atoms)).astype(np.float64)
    dx2 = rng.normal(size=(nm2, max_atoms, rep, 3 * max_atoms)).astype(np.float64)
    q1 = rng.integers(0, 3, size=(nm1, max_atoms)).astype(np.int32)
    q2 = rng.integers(0, 3, size=(nm2, max_atoms)).astype(np.int32)
    n1 = rng.integers(1, max_atoms + 1, size=(nm1,)).astype(np.int32)
    n2 = rng.integers(1, max_atoms + 1, size=(nm2,)).astype(np.int32)
    return x1, x2, dx1, dx2, q1, q2, n1, n2


@pytest.fixture(scope="module")
def data() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    return _dataset(3, 4, 4, 6, seed=0)


def test_keyword_names_match_positional(data: tuple) -> None:
    x1, x2, dx1, dx2, q1, q2, n1, n2 = data
    sigma = 1.25

    k_pos = lk.kernel_gaussian(x1, x2, q1, q2, n1, n2, sigma)
    k_kw = lk.kernel_gaussian(X1=x1, X2=x2, Q1=q1, Q2=q2, N1=n1, N2=n2, sigma=sigma)
    np.testing.assert_array_equal(k_kw, k_pos)

    ks_pos = lk.kernel_gaussian_symm(x1, q1, n1, sigma)
    ks_kw = lk.kernel_gaussian_symm(X=x1, Q=q1, N=n1, sigma=sigma)
    np.testing.assert_array_equal(ks_kw, ks_pos)

    kr_pos = lk.kernel_gaussian_symm_rfp(x1, q1, n1, sigma)
    kr_kw = lk.kernel_gaussian_symm_rfp(X=x1, Q=q1, N=n1, sigma=sigma)
    np.testing.assert_array_equal(kr_kw, kr_pos)

    h_pos = lk.kernel_gaussian_hessian(x1, dx1, x2, dx2, q1, q2, n1, n2, sigma)
    h_kw = lk.kernel_gaussian_hessian(
        X1=x1,
        dX1=dx1,
        X2=x2,
        dX2=dx2,
        Q1=q1,
        Q2=q2,
        N1=n1,
        N2=n2,
        sigma=sigma,
    )
    np.testing.assert_array_equal(h_kw, h_pos)

    hs_pos = lk.kernel_gaussian_hessian_symm(x1, dx1, q1, n1, sigma)
    hs_kw = lk.kernel_gaussian_hessian_symm(X=x1, dX=dx1, Q=q1, N=n1, sigma=sigma)
    np.testing.assert_array_equal(hs_kw, hs_pos)

    hsr_pos = lk.kernel_gaussian_hessian_symm_rfp(x1, dx1, q1, n1, sigma)
    hsr_kw = lk.kernel_gaussian_hessian_symm_rfp(X=x1, dX=dx1, Q=q1, N=n1, sigma=sigma)
    np.testing.assert_array_equal(hsr_kw, hsr_pos)

    j_pos = lk.kernel_gaussian_jacobian(x1, x2, dx2, q1, q2, n1, n2, sigma)
    j_kw = lk.kernel_gaussian_jacobian(
        X1=x1, X2=x2, dX2=dx2, Q1=q1, Q2=q2, N1=n1, N2=n2, sigma=sigma
    )
    np.testing.assert_array_equal(j_kw, j_pos)

    jt_pos = lk.kernel_gaussian_jacobian_t(x1, dx1, x2, q1, q2, n1, n2, sigma)
    jt_kw = lk.kernel_gaussian_jacobian_t(
        X1=x1, dX1=dx1, X2=x2, Q1=q1, Q2=q2, N1=n1, N2=n2, sigma=sigma
    )
    np.testing.assert_array_equal(jt_kw, jt_pos)

    f_pos = lk.kernel_gaussian_full(x1, dx1, x2, dx2, q1, q2, n1, n2, sigma)
    f_kw = lk.kernel_gaussian_full(
        X1=x1,
        dX1=dx1,
        X2=x2,
        dX2=dx2,
        Q1=q1,
        Q2=q2,
        N1=n1,
        N2=n2,
        sigma=sigma,
    )
    np.testing.assert_array_equal(f_kw, f_pos)

    fs_pos = lk.kernel_gaussian_full_symm(x1, dx1, q1, n1, sigma)
    fs_kw = lk.kernel_gaussian_full_symm(X=x1, dX=dx1, Q=q1, N=n1, sigma=sigma)
    np.testing.assert_array_equal(fs_kw, fs_pos)

    fsr_pos = lk.kernel_gaussian_full_symm_rfp(x1, dx1, q1, n1, sigma)
    fsr_kw = lk.kernel_gaussian_full_symm_rfp(X=x1, dX=dx1, Q=q1, N=n1, sigma=sigma)
    np.testing.assert_array_equal(fsr_kw, fsr_pos)


def test_matvec_keyword_names(data: tuple) -> None:
    x1, x2, dx1, dx2, q1, q2, n1, n2 = data
    sigma = 1.25
    nm2 = x2.shape[0]
    max_atoms = x2.shape[1]
    rep = x2.shape[2]
    naq2 = int(3 * np.sum(np.clip(n2, 0, max_atoms)))
    alpha_f = np.random.default_rng(1).normal(size=naq2)
    alpha_e = np.random.default_rng(2).normal(size=nm2)

    alpha_desc = lk.kernel_gaussian_local_compute_alpha_desc(dX2=dx2, Q2=q2, N2=n2, alpha=alpha_f)
    alpha_desc_pos = lk.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)
    np.testing.assert_array_equal(alpha_desc, alpha_desc_pos)

    f_pos = lk.kernel_gaussian_local_hessian_matvec(x1, dx1, x2, alpha_desc, q1, q2, n1, n2, sigma)
    f_kw = lk.kernel_gaussian_local_hessian_matvec(
        X1=x1,
        dX1=dx1,
        X2=x2,
        alpha_desc=alpha_desc,
        Q1=q1,
        Q2=q2,
        N1=n1,
        N2=n2,
        sigma=sigma,
    )
    np.testing.assert_array_equal(f_kw, f_pos)

    e_pos = lk.kernel_gaussian_local_jacobian_t_matvec(x1, x2, alpha_desc, q1, q2, n1, n2, sigma)
    e_kw = lk.kernel_gaussian_local_jacobian_t_matvec(
        X1=x1,
        X2=x2,
        alpha_desc=alpha_desc,
        Q1=q1,
        Q2=q2,
        N1=n1,
        N2=n2,
        sigma=sigma,
    )
    np.testing.assert_array_equal(e_kw, e_pos)

    ef_pos = lk.kernel_gaussian_local_full_matvec(
        x1, dx1, x2, alpha_desc, alpha_e, q1, q2, n1, n2, sigma
    )
    ef_kw = lk.kernel_gaussian_local_full_matvec(
        X1=x1,
        dX1=dx1,
        X2=x2,
        alpha_desc_F=alpha_desc,
        alpha_E=alpha_e,
        Q1=q1,
        Q2=q2,
        N1=n1,
        N2=n2,
        sigma=sigma,
    )
    np.testing.assert_allclose(ef_kw[0], ef_pos[0])
    np.testing.assert_allclose(ef_kw[1], ef_pos[1])
