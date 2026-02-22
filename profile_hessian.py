#!/usr/bin/env python3
"""
Profiling script for kernel_gaussian_hessian and kernel_gaussian_hessian_symm.

This script tests the Hessian kernels with realistic problem sizes to identify
performance bottlenecks. The C++ code will output detailed timing for each phase
when KERNELFORGE_PROFILE=1 is set.

Problem sizes tested:
- Small: N=50, M=36 (invdist of 9 atoms), D=27 (3*9 atoms)
- Medium: N=100, M=36, D=27
- Large: N=200, M=36, D=27
"""

import os
import sys
import numpy as np

# Enable profiling
os.environ["KERNELFORGE_PROFILE"] = "1"

# Import kernelforge
try:
    from kernelforge.global_kernels import (
        kernel_gaussian_hessian,
        kernel_gaussian_hessian_symm,
    )
except ImportError:
    print("ERROR: kernelforge.global_kernels module not found. Please build and install first:")
    print("  make install-linux")
    sys.exit(1)


def generate_test_data(N, M, D, seed=42):
    """Generate random test data for Hessian kernel."""
    rng = np.random.RandomState(seed)

    X = rng.randn(N, M)
    dX = rng.randn(N, M, D)

    return X, dX


def profile_asymmetric_kernel(N1, N2, M, D1, D2, sigma=1.0):
    """Profile kernel_gaussian_hessian (asymmetric version)."""
    print(f"\n{'=' * 60}")
    print(f"Profiling kernel_gaussian_hessian (asymmetric)")
    print(f"N1={N1}, N2={N2}, M={M}, D1={D1}, D2={D2}, sigma={sigma}")
    print(f"{'=' * 60}")

    X1, dX1 = generate_test_data(N1, M, D1, seed=42)
    X2, dX2 = generate_test_data(N2, M, D2, seed=43)

    # Warmup run
    _ = kernel_gaussian_hessian(X1, dX1, X2, dX2, sigma)

    # Profiling run
    H = kernel_gaussian_hessian(X1, dX1, X2, dX2, sigma)

    print(f"Output shape: {H.shape}")
    print(f"Memory usage: {H.nbytes / 1024**2:.2f} MB")

    return H


def profile_symmetric_kernel(N, M, D, sigma=1.0):
    """Profile kernel_gaussian_hessian_symm (symmetric version)."""
    print(f"\n{'=' * 60}")
    print(f"Profiling kernel_gaussian_hessian_symm (symmetric)")
    print(f"N={N}, M={M}, D={D}, sigma={sigma}")
    print(f"{'=' * 60}")

    X, dX = generate_test_data(N, M, D, seed=42)

    # Warmup run
    _ = kernel_gaussian_hessian_symm(X, dX, sigma)

    # Profiling run
    H = kernel_gaussian_hessian_symm(X, dX, sigma)

    print(f"Output shape: {H.shape}")
    print(f"Memory usage: {H.nbytes / 1024**2:.2f} MB")

    return H


def main():
    """Run profiling with different problem sizes."""
    # Typical values for molecular systems with invdist representation
    M = 36  # invdist representation for 9 atoms
    D = 27  # 3 * 9 atoms (xyz coordinates)
    sigma = 1.0

    print("\n" + "=" * 80)
    print("HESSIAN KERNEL PROFILING (LARGE SCALE)")
    print("=" * 80)
    print(f"Representation size: M={M}")
    print(f"Coordinate size: D={D}")
    print(f"Kernel width: sigma={sigma}")
    print(f"\nTarget: at least 10 seconds per run for realistic profiling")

    # Skip warmup, just do one big problem to save time

    # Large problem (target 10+ seconds)
    N_large = 2000
    print("\n" + "#" * 80)
    print(f"# LARGE PROBLEM: N={N_large} (target: 10+ seconds per run)")
    print(f"# Output matrix: {N_large * D} x {N_large * D} = {N_large * D}² elements")
    print(f"# Memory: ~{(N_large * D) ** 2 * 8 / 1024**3:.1f} GB")
    print("#" * 80)

    print("\n>>> Testing SYMMETRIC kernel (training scenario):")
    profile_symmetric_kernel(N_large, M, D, sigma)

    print("\n>>> Testing ASYMMETRIC kernel (prediction scenario):")
    profile_asymmetric_kernel(N_large, N_large, M, D, D, sigma)

    # Extra large problem (if you want to stress test)
    # N_xlarge = 800
    # print("\n" + "#"*80)
    # print(f"# EXTRA LARGE PROBLEM: N={N_xlarge}")
    # print("#"*80)
    # profile_symmetric_kernel(N_xlarge, M, D, sigma)
    # profile_asymmetric_kernel(N_xlarge, N_xlarge, M, D, D, sigma)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
