"""Verify global_kernels accepts the public X/dX keyword names."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import kernelforge.global_kernels as gk


def _dataset(
    n1: int,
    n2: int,
    d: int,
    *,
    seed: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=(n1, d)).astype(np.float64)
    x2 = rng.normal(size=(n2, d)).astype(np.float64)
    dx1 = rng.normal(size=(n1, d, d)).astype(np.float64)
    dx2 = rng.normal(size=(n2, d, d)).astype(np.float64)
    return x1, x2, dx1, dx2


@pytest.fixture(scope="module")
def data() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    return _dataset(3, 4, 5, seed=0)


def test_keyword_names_match_positional(data: tuple) -> None:
    x1, x2, dx1, dx2 = data
    sigma = 1.25

    k_pos = gk.kernel_gaussian(x1, x2, sigma)
    k_kw = gk.kernel_gaussian(X1=x1, X2=x2, sigma=sigma)
    np.testing.assert_array_equal(k_kw, k_pos)

    ks_pos = gk.kernel_gaussian_symm(x1, sigma)
    ks_kw = gk.kernel_gaussian_symm(X=x1, sigma=sigma)
    np.testing.assert_array_equal(ks_kw, ks_pos)

    j_pos = gk.kernel_gaussian_jacobian(x1, dx1, x2, sigma)
    j_kw = gk.kernel_gaussian_jacobian(X1=x1, dX1=dx1, X2=x2, sigma=sigma)
    np.testing.assert_array_equal(j_kw, j_pos)

    jt_pos = gk.kernel_gaussian_jacobian_t(x1, x2, dx2, sigma)
    jt_kw = gk.kernel_gaussian_jacobian_t(X1=x1, X2=x2, dX2=dx2, sigma=sigma)
    np.testing.assert_array_equal(jt_kw, jt_pos)

    h_pos = gk.kernel_gaussian_hessian(x1, dx1, x2, dx2, sigma)
    h_kw = gk.kernel_gaussian_hessian(X1=x1, dX1=dx1, X2=x2, dX2=dx2, sigma=sigma)
    np.testing.assert_array_equal(h_kw, h_pos)

    f_pos = gk.kernel_gaussian_full(x1, dx1, x2, dx2, sigma)
    f_kw = gk.kernel_gaussian_full(X1=x1, dX1=dx1, X2=x2, dX2=dx2, sigma=sigma)
    np.testing.assert_array_equal(f_kw, f_pos)

    alpha = np.random.default_rng(1).normal(size=(x2.shape[0], dx2.shape[1])).astype(np.float64)
    alpha_desc = gk.kernel_gaussian_compute_alpha_desc(dX=dx2, alpha=alpha)
    alpha_desc_pos = gk.kernel_gaussian_compute_alpha_desc(dx2, alpha)
    np.testing.assert_array_equal(alpha_desc, alpha_desc_pos)

    e_pos = gk.kernel_gaussian_jacobian_t_matvec(x1, x2, alpha_desc, sigma)
    e_kw = gk.kernel_gaussian_jacobian_t_matvec(X_q=x1, X_t=x2, alpha_desc=alpha_desc, sigma=sigma)
    np.testing.assert_allclose(e_kw, e_pos)

    alpha_e = np.random.default_rng(2).normal(size=x2.shape[0]).astype(np.float64)
    ef_pos = gk.kernel_gaussian_full_matvec(x1, dx1, x2, alpha_e, alpha_desc, sigma)
    ef_kw = gk.kernel_gaussian_full_matvec(
        X_q=x1,
        dX_q=dx1,
        X_t=x2,
        alpha_desc_F=alpha_desc,
        alpha_E=alpha_e,
        sigma=sigma,
    )
    np.testing.assert_allclose(ef_kw[0], ef_pos[0])
    np.testing.assert_allclose(ef_kw[1], ef_pos[1])
