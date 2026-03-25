"""Benchmark: local Hessian kernel force inference (J^T·α trick vs full Hessian).

Scenario: FCHL19-style molecules with fixed atom count, varying training-set size.

Usage:
    uv run python benchmarks/bench_local_hessian_matvec.py
"""

import time

import numpy as np

from kernelforge import local_kernels as lk


def make_dataset(
    nm: int,
    natoms: int,
    rep_size: int,
    n_species: int,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,  # x      (nm, natoms, rep_size)
    np.ndarray,  # dx     (nm, natoms, rep_size, 3*natoms)
    np.ndarray,  # q      (nm, natoms) int32
    np.ndarray,  # n      (nm,)        int32
]:
    x = rng.normal(size=(nm, natoms, rep_size))
    dx = rng.normal(size=(nm, natoms, rep_size, 3 * natoms))
    # Labels 1..n_species, all atoms active
    q = rng.integers(1, n_species + 1, size=(nm, natoms), dtype=np.int32)
    n = np.full(nm, natoms, dtype=np.int32)
    return x, dx, q, n


def time_fn(fn, n_warmup: int = 1, n_repeat: int = 5) -> float:
    """Return median wall time in seconds."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def benchmark(
    n_train: int,
    n_test: int,
    natoms: int,
    rep_size: int,
    n_species: int,
    sigma: float,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    x_train, dx_train, q_train, n_train_arr = make_dataset(
        n_train, natoms, rep_size, n_species, rng
    )
    x_test, dx_test, q_test, n_test_arr = make_dataset(n_test, natoms, rep_size, n_species, rng)

    naq_train = int(np.sum(n_train_arr) * 3)
    alpha = rng.normal(size=(naq_train,))

    # --- Time: alpha_desc pre-computation (training, done once) ---
    def precompute():
        lk.kernel_gaussian_local_compute_alpha_desc(dx_train, q_train, n_train_arr, alpha)

    alpha_desc = lk.kernel_gaussian_local_compute_alpha_desc(dx_train, q_train, n_train_arr, alpha)

    # --- Time: fast matvec (inference) ---
    def fast_matvec():
        lk.kernel_gaussian_local_hessian_matvec(
            x_test, dx_test, x_train, alpha_desc, q_test, q_train, n_test_arr, n_train_arr, sigma
        )

    # --- Time: full Hessian build + matvec ---
    def full_hessian_matvec():
        H = lk.kernel_gaussian_hessian(
            x_test, x_train, dx_test, dx_train, q_test, q_train, n_test_arr, n_train_arr, sigma
        )
        H @ alpha

    t_precompute = time_fn(precompute)
    t_fast = time_fn(fast_matvec)
    t_full = time_fn(full_hessian_matvec)

    return {
        "n_train": n_train,
        "n_test": n_test,
        "natoms": natoms,
        "rep_size": rep_size,
        "naq_train": naq_train,
        "naq_test": int(np.sum(n_test_arr) * 3),
        "t_precompute_ms": t_precompute * 1e3,
        "t_fast_ms": t_fast * 1e3,
        "t_full_ms": t_full * 1e3,
        "speedup": t_full / t_fast,
        # How many fast matvec calls until precompute cost is amortised vs full
        "break_even": t_precompute / max(t_full - t_fast, 1e-12),
    }


def main() -> None:
    sigma = 2.0

    # ----------------------------------------------------------------
    # Sweep 1: vary n_train, fixed n_test=1, ethanol-like (9 atoms)
    # precompute cost vs inference cost
    # ----------------------------------------------------------------
    W = 80
    print("=" * W)
    print("Sweep 1: vary N_train, N_test=1, natoms=9, rep_size=128")
    print("  precompute = kernel_gaussian_local_compute_alpha_desc (training, once)")
    print("  fast       = kernel_gaussian_local_hessian_matvec (inference)")
    print("  full       = kernel_gaussian_hessian + H @ alpha")
    print("=" * W)
    hdr = f"{'N_train':>8}  {'precomp(ms)':>12}  {'fast(ms)':>10}  {'full(ms)':>10}  {'speedup':>8}  {'break-even':>11}"
    print(hdr)
    print("-" * len(hdr))

    for n_train in [50, 100, 250, 500]:
        r = benchmark(
            n_train=n_train,
            n_test=1,
            natoms=9,
            rep_size=128,
            n_species=5,
            sigma=sigma,
        )
        print(
            f"{r['n_train']:>8}  {r['t_precompute_ms']:>12.2f}  {r['t_fast_ms']:>10.2f}"
            f"  {r['t_full_ms']:>10.2f}  {r['speedup']:>7.1f}x  {r['break_even']:>9.1f} calls"
        )

    # ----------------------------------------------------------------
    # Sweep 2: vary n_test (batch prediction), fixed n_train=500
    # ----------------------------------------------------------------
    print()
    print("=" * W)
    print("Sweep 2: vary N_test, N_train=200, natoms=9, rep_size=128")
    print("=" * W)
    hdr = f"{'N_test':>8}  {'precomp(ms)':>12}  {'fast(ms)':>10}  {'full(ms)':>10}  {'speedup':>8}  {'break-even':>11}"
    print(hdr)
    print("-" * len(hdr))

    for n_test in [1, 5, 10, 25, 50]:
        r = benchmark(
            n_train=200,
            n_test=n_test,
            natoms=9,
            rep_size=128,
            n_species=5,
            sigma=sigma,
        )
        print(
            f"{r['n_test']:>8}  {r['t_precompute_ms']:>12.2f}  {r['t_fast_ms']:>10.2f}"
            f"  {r['t_full_ms']:>10.2f}  {r['speedup']:>7.1f}x  {r['break_even']:>9.1f} calls"
        )

    # ----------------------------------------------------------------
    # Sweep 3: vary molecule size (natoms), fixed n_train=200, n_test=1
    # ----------------------------------------------------------------
    print()
    print("=" * W)
    print("Sweep 3: vary natoms, N_train=100, N_test=1, rep_size=128")
    print("=" * W)
    hdr = f"{'natoms':>8}  {'precomp(ms)':>12}  {'fast(ms)':>10}  {'full(ms)':>10}  {'speedup':>8}  {'break-even':>11}"
    print(hdr)
    print("-" * len(hdr))

    for natoms in [3, 5, 9, 15, 23]:
        r = benchmark(
            n_train=100,
            n_test=1,
            natoms=natoms,
            rep_size=128,
            n_species=5,
            sigma=sigma,
        )
        print(
            f"{r['natoms']:>8}  {r['t_precompute_ms']:>12.2f}  {r['t_fast_ms']:>10.2f}"
            f"  {r['t_full_ms']:>10.2f}  {r['speedup']:>7.1f}x  {r['break_even']:>9.1f} calls"
        )


if __name__ == "__main__":
    main()
