#!/usr/bin/env python3
"""
Profiling script for kernel_gaussian_jacobian (gradient kernel).

The Jacobian kernel computes K = J^T @ diag(k) @ (X2 - X1)
where k are Gaussian kernel values. It's used for force predictions
in GDML/sGDML models.

Problem sizes tested:
- N1: Number of query structures (e.g., validation set)
- N2: Number of reference structures (e.g., training set)
- M: Representation size (e.g., 36 for invdist of 9 atoms)
- D: Coordinate dimension (e.g., 27 for 3*9 atoms)
"""

import os
import sys
import numpy as np

# Enable profiling
os.environ["KERNELFORGE_PROFILE"] = "1"

# Import kernelforge
try:
    from kernelforge.global_kernels import kernel_gaussian_jacobian
except ImportError:
    print("ERROR: kernelforge.global_kernels module not found. Please build and install first:")
    print("  make install-linux")
    sys.exit(1)


def generate_test_data(N1, N2, M, D, seed=42):
    """Generate random test data for Jacobian kernel."""
    rng = np.random.RandomState(seed)

    X1 = rng.randn(N1, M)  # Query structures
    dX1 = rng.randn(N1, M, D)  # Jacobians for query
    X2 = rng.randn(N2, M)  # Reference structures

    return X1, dX1, X2


def profile_jacobian_kernel(N1, N2, M, D, sigma=1.0):
    """Profile kernel_gaussian_jacobian."""
    print(f"\n{'=' * 60}")
    print(f"Profiling kernel_gaussian_jacobian")
    print(f"N1={N1}, N2={N2}, M={M}, D={D}, sigma={sigma}")
    print(f"{'=' * 60}")

    X1, dX1, X2 = generate_test_data(N1, N2, M, D)

    # Warmup run
    _ = kernel_gaussian_jacobian(X1, dX1, X2, sigma)

    # Profiling run
    K = kernel_gaussian_jacobian(X1, dX1, X2, sigma)

    print(f"Output shape: {K.shape}")
    print(f"Memory usage: {K.nbytes / 1024**2:.2f} MB")

    return K


def main():
    """Run profiling with different problem sizes."""
    # Typical values for molecular systems with invdist representation
    M = 36  # invdist representation for 9 atoms
    D = 27  # 3 * 9 atoms (xyz coordinates)
    sigma = 1.0

    print("\n" + "=" * 80)
    print("JACOBIAN KERNEL PROFILING (GRADIENT/FORCE PREDICTIONS)")
    print("=" * 80)
    print(f"Representation size: M={M}")
    print(f"Coordinate size: D={D}")
    print(f"Kernel width: sigma={sigma}")
    print(f"\nScenario: Predict forces for validation set using training set")

    # Small validation, medium training
    print("\n" + "#" * 80)
    print(f"# SMALL VALIDATION: N1=100 queries, N2=500 references")
    print("#" * 80)
    profile_jacobian_kernel(100, 500, M, D, sigma)

    # Medium validation, medium training
    print("\n" + "#" * 80)
    print(f"# MEDIUM VALIDATION: N1=200 queries, N2=1000 references")
    print("#" * 80)
    profile_jacobian_kernel(200, 1000, M, D, sigma)

    # Large validation, large training (realistic for production)
    print("\n" + "#" * 80)
    print(f"# LARGE PROBLEM: N1=500 queries, N2=2000 references")
    print(f"# This simulates production force prediction")
    print("#" * 80)
    profile_jacobian_kernel(500, 2000, M, D, sigma)

    # Very large (stress test)
    print("\n" + "#" * 80)
    print(f"# VERY LARGE: N1=1000 queries, N2=3000 references")
    print(f"# Memory: ~{(1000 * D * 3000 * 8 / 1024**2):.1f} MB output")
    print("#" * 80)
    profile_jacobian_kernel(1000, 3000, M, D, sigma)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
