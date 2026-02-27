"""Integration test for FCHL18 kernel-ridge regression on QM7b.

Mirrors the structure of old_code/test_fchl_scalar.py but uses the kernelforge
FCHL18 C++ implementation.  Requires the QM7b dataset cached at
~/.kernelforge/datasets/qm7b_complete.npz (auto-downloaded by load_qm7b_raw_data).

Run with:
    pytest -m integration tests/test_fchl18_integration.py
"""

import numpy as np
import pytest

import kernelforge.fchl18_kernel as kernel_mod
import kernelforge.fchl18_repr as repr_mod
from kernelforge.cli import load_qm7b_raw_data

# Default hyperparameters from old_code/test_fchl_scalar.py
KERNEL_ARGS = dict(
    two_body_width=0.1,
    two_body_scaling=2.0,
    two_body_power=6.0,
    three_body_width=3.0,
    three_body_scaling=2.0,
    three_body_power=3.0,
    cut_start=0.5,
    cut_distance=1e6,
    fourier_order=2,
)


@pytest.mark.integration
def test_krr_fchl18_qm7b():
    """KRR on QM7b energies with FCHL18: MAE should be close to 2 kcal/mol (100 molecules)."""

    n_points = 1000
    max_size = 23  # largest QM7b molecule has 23 atoms
    sigma = 2.5
    llambda = 1e-8

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = load_qm7b_raw_data()
    R_all = data["R"][:n_points]  # array of (n_atoms, 3) arrays
    z_all = data["z"][:n_points]  # array of int32 arrays
    E_all = data["E"][:n_points]  # (n_points,) energies in kcal/mol

    coords_list = list(R_all)
    z_list = [zi.astype(np.int32) for zi in z_all]
    properties = np.array(E_all, dtype=np.float64)

    # ------------------------------------------------------------------
    # Generate FCHL18 representations
    # ------------------------------------------------------------------
    x, n, nn = repr_mod.generate(
        coords_list,
        z_list,
        max_size=max_size,
        cut_distance=KERNEL_ARGS["cut_distance"],
    )

    assert x.shape == (n_points, max_size, 5, max_size)
    assert n.shape == (n_points,)
    assert nn.shape == (n_points, max_size)
    assert np.all(np.isfinite(x[x < 1e99])), "Representation contains non-finite values"

    # ------------------------------------------------------------------
    # Shuffle with fixed seed (matching old test)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(666)
    perm = rng.permutation(n_points)
    x = x[perm]
    n = n[perm]
    nn = nn[perm]
    properties = properties[perm]

    # ------------------------------------------------------------------
    # Train / test split (67 / 33)
    # ------------------------------------------------------------------
    n_test = n_points // 3
    n_train = n_points - n_test

    x_train = x[:n_train]
    n_train_arr = n[:n_train]
    nn_train = nn[:n_train]
    y_train = properties[:n_train]

    x_test = x[n_train:]
    n_test_arr = n[n_train:]
    nn_test = nn[n_train:]
    y_test = properties[n_train:]

    # ------------------------------------------------------------------
    # Build symmetric kernel (train × train)
    # ------------------------------------------------------------------
    # import timer
    import time

    t_start = time.time()
    K_sym = kernel_mod.kernel_gaussian_symm(
        x_train, n_train_arr, nn_train, sigma=sigma, **KERNEL_ARGS
    )
    t_end = time.time()
    print(f"Symmetric kernel computed in {t_end - t_start:.2f} seconds")
    assert K_sym.shape == (n_train, n_train)
    assert not np.any(np.isnan(K_sym)), "Symmetric kernel contains NaN"

    # ------------------------------------------------------------------
    # Verify symmetric kernel matches asymmetric self-kernel
    # ------------------------------------------------------------------
    K_asym = kernel_mod.kernel_gaussian(
        x_train,
        x_train,
        n_train_arr,
        n_train_arr,
        nn_train,
        nn_train,
        sigma=sigma,
        **KERNEL_ARGS,
    )
    assert K_asym.shape == (n_train, n_train)
    assert not np.any(np.isnan(K_asym)), "Asymmetric kernel contains NaN"
    np.testing.assert_allclose(
        K_sym,
        K_asym,
        rtol=1e-10,
        atol=1e-12,
        err_msg="Symmetric kernel does not match asymmetric self-kernel",
    )

    # ------------------------------------------------------------------
    # Solve KRR: (K + lambda*I) alpha = y
    # ------------------------------------------------------------------
    K_reg = K_sym.copy()
    K_reg[np.diag_indices_from(K_reg)] += llambda
    alpha = np.linalg.solve(K_reg, y_train)

    # ------------------------------------------------------------------
    # Predict on test set
    # ------------------------------------------------------------------
    K_test = kernel_mod.kernel_gaussian(
        x_test,
        x_train,
        n_test_arr,
        n_train_arr,
        nn_test,
        nn_train,
        sigma=sigma,
        **KERNEL_ARGS,
    )
    assert K_test.shape == (n_test, n_train)
    assert not np.any(np.isnan(K_test)), "Test kernel contains NaN"

    print(f"K_test shape: {K_test.shape}, alpha shape: {alpha.shape}")
    print(f"K_test sample:\n{K_test[:5, :5]}")
    print(f"alpha sample:\n{alpha[:5]}")

    y_pred = K_test @ alpha

    print(f"y_test sample:\n{y_test[:5]}")
    print(f"y_pred sample:\n{y_pred[:5]}")

    # ------------------------------------------------------------------
    # Check MAE (same criterion as old_code/test_fchl_scalar.py)
    # ------------------------------------------------------------------
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"FCHL18 KRR MAE on QM7b ({n_points} molecules): {mae:.3f} kcal/mol")

    assert abs(2.0 - mae) < 2.0, (
        f"FCHL18 KRR MAE = {mae:.3f} kcal/mol — expected close to 2.0 kcal/mol"
    )

if __name__ == "__main__":
    # Run the test directly (without pytest) for quick debugging
    test_krr_fchl18_qm7b()
    print("Test passed!")
