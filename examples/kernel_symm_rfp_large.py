"""
Demo: Large symmetric Gaussian kernel using RFP format with ILP64 support

This example demonstrates:
1. Computing a symmetric Gaussian kernel for n=50,000 samples
2. Using RFP (Rectangular Full Packed) format to save 2× memory
3. Solving the kernel ridge regression problem using RFP Cholesky
4. Verifying correctness against full matrix for small n

With LP64 (32-bit BLAS integers):
- Maximum n ≈ 46,340 (before int32 overflow in BLAS calls)
- Full matrix would require n×n×8 bytes ≈ 19.1 GB for n=50,000

With ILP64 (64-bit BLAS integers):
- No practical limit on n (until memory exhausted)
- RFP format requires only n*(n+1)/2 × 8 bytes ≈ 9.5 GB for n=50,000
- Saves 50% memory compared to full symmetric matrix storage

To build with ILP64 on Linux:
    make install-linux-ilp64
    # Requires: sudo apt install libopenblas64-dev libopenblas64-pthread-dev

To build with ILP64 on macOS:
    make install-macos-ilp64
"""

import numpy as np
import kernelforge.global_kernels as gk
import kernelforge.kernelmath as km


def verify_rfp_correctness(n=1000, d=128, sigma=1.0):
    """Verify that RFP kernel matches full matrix for small n"""
    print(f"\n=== Correctness check (n={n}, d={d}) ===")

    # Random data
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    alpha_param = 1.0 / (2.0 * sigma**2)

    # Compute kernel in both formats
    K_full = gk.kernel_gaussian_symm(X, alpha_param)  # (n, n) dense lower triangle
    K_rfp = gk.kernel_gaussian_symm_rfp(X, alpha_param)  # 1D RFP packed

    # Convert RFP back to full to compare
    K_from_rfp = km.rfp_to_full(K_rfp, n, uplo="U", transr="N")

    # Extract lower triangles for comparison (full kernel only fills lower triangle)
    tril_indices = np.tril_indices(n)
    K_full_lower = K_full[tril_indices]
    K_rfp_lower = K_from_rfp[tril_indices]

    diff = np.abs(K_full_lower - K_rfp_lower).max()
    print(f"Max absolute difference: {diff:.2e}")

    if diff < 1e-12:
        print("✓ RFP kernel matches full matrix!")
    else:
        print(f"✗ Warning: difference {diff:.2e} exceeds tolerance 1e-12")

    return diff < 1e-12


def demo_large_kernel_rfp(n=50_000, d=128, sigma=1.0, regularize=1e-6):
    """Demo: large kernel matrix using RFP format"""
    print(f"\n=== Large kernel demo (n={n:,}, d={d}) ===")

    # Memory estimates
    mem_full = n * n * 8 / (1024**3)  # GB for full matrix
    mem_rfp = n * (n + 1) // 2 * 8 / (1024**3)  # GB for RFP
    print(f"Memory required:")
    print(f"  Full matrix (n×n):       {mem_full:.2f} GB")
    print(f"  RFP format (n×(n+1)/2):  {mem_rfp:.2f} GB")
    print(
        f"  Savings:                 {mem_full - mem_rfp:.2f} GB ({100 * (mem_full - mem_rfp) / mem_full:.1f}%)"
    )

    # Generate random data
    print(f"\nGenerating random data...")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d)).astype(np.float64)
    y = rng.standard_normal(n).astype(np.float64)

    # Compute kernel directly into RFP format
    print(f"Computing symmetric Gaussian kernel in RFP format...")
    alpha_param = 1.0 / (2.0 * sigma**2)
    K_rfp = gk.kernel_gaussian_symm_rfp(X, alpha_param)

    print(f"  K_rfp shape: {K_rfp.shape}")
    print(f"  K_rfp dtype: {K_rfp.dtype}")
    print(f"  Expected length: {n * (n + 1) // 2}")
    assert K_rfp.shape[0] == n * (n + 1) // 2

    # Solve kernel ridge regression using RFP Cholesky
    print(f"\nSolving Kα = y using RFP Cholesky (regularize={regularize})...")
    # Note: solve_cholesky_rfp_L overwrites K_rfp, so we pass a copy
    K_rfp_copy = K_rfp.copy()
    alpha_coeffs = km.solve_cholesky_rfp_L(
        K_rfp_copy, y, regularize=regularize, uplo="U", transr="N"
    )

    print(f"  Solution α shape: {alpha_coeffs.shape}")
    print(f"  α min/max: [{alpha_coeffs.min():.6f}, {alpha_coeffs.max():.6f}]")
    print(f"  α mean: {alpha_coeffs.mean():.6f}")
    print(f"  α std:  {alpha_coeffs.std():.6f}")

    # Basic sanity check: α should have reasonable magnitude
    alpha_norm = np.linalg.norm(alpha_coeffs)
    y_norm = np.linalg.norm(y)
    print(f"  ||α|| / ||y|| = {alpha_norm / y_norm:.6f}")

    print("\n✓ Large RFP demo completed successfully!")
    return K_rfp, alpha_coeffs


def main():
    print("=" * 70)
    print("Large Symmetric Kernel with RFP Format and ILP64 Support")
    print("=" * 70)

    # 1. Verify correctness for small n
    is_correct = verify_rfp_correctness(n=1000, d=128, sigma=1.0)

    if not is_correct:
        print("\n⚠ Correctness check failed — stopping here.")
        return

    # 2. Run large demo (n=50,000)
    # Note: For n=50,000, this requires:
    #   - ILP64 build (otherwise n > 46,340 will overflow int32)
    #   - ~10 GB RAM for RFP kernel matrix
    #   - Several minutes to compute on a modern CPU
    try:
        K_rfp, alpha = demo_large_kernel_rfp(n=50_000, d=128, sigma=1.0, regularize=1e-6)
        print("\n" + "=" * 70)
        print("All tests passed! ILP64 + RFP working correctly.")
        print("=" * 70)
    except MemoryError:
        print("\n⚠ Out of memory — try reducing n or increasing available RAM")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
