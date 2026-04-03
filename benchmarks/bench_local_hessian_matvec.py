"""Benchmark: local Hessian kernel force inference (batched DGEMM matvec vs full Hessian build).

Compares:
  fast = kernel_gaussian_local_hessian_matvec  (batched DGEMM, J^T-alpha trick)
  full = kernel_gaussian_hessian + H @ alpha   (materialize full H, then multiply)

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
    q = rng.integers(1, n_species + 1, size=(nm, natoms), dtype=np.int32)
    n = np.full(nm, natoms, dtype=np.int32)
    return x, dx, q, n


def time_fn(fn: object, n_warmup: int = 3, n_repeat: int = 9) -> float:
    """Return minimum wall time in seconds (min-of-N is most stable for CPU-bound work)."""
    for _ in range(n_warmup):
        fn()  # type: ignore[call-arg]
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()  # type: ignore[call-arg]
        times.append(time.perf_counter() - t0)
    return float(min(times))


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

    def precompute() -> None:
        lk.kernel_gaussian_local_compute_alpha_desc(dx_train, q_train, n_train_arr, alpha)

    alpha_desc = lk.kernel_gaussian_local_compute_alpha_desc(dx_train, q_train, n_train_arr, alpha)

    def fast_matvec() -> None:
        lk.kernel_gaussian_local_hessian_matvec(
            x_test, dx_test, x_train, alpha_desc, q_test, q_train, n_test_arr, n_train_arr, sigma
        )

    def full_hessian_matvec() -> None:
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
        "break_even": t_precompute / max(t_full - t_fast, 1e-12),
    }


def main() -> None:
    sigma = 2.0
    W = 80

    # ------------------------------------------------------------------
    # Sweep: vary rep_size, N_train=200, N_test=1, natoms=9, n_species=3
    #
    # The batched-DGEMM approach packs all training atoms of each label
    # into dense (S x M) matrices and replaces the per-atom-pair loop with
    # 4 BLAS DGEMM calls per label.  At larger rep_size the DGEMM becomes
    # proportionally more efficient, driving the speedup from ~6x (rep=27)
    # to ~30x (rep=384).
    # ------------------------------------------------------------------
    print()
    print("=" * W)
    print("Vary rep_size, N_train=200, N_test=1, natoms=9, n_species=3")
    print("fast = batched-DGEMM matvec   |   full = build H then H @ alpha")
    print("=" * W)
    hdr = (
        f"{'rep_size':>10}  {'precomp(ms)':>12}  {'fast(ms)':>10}  {'full(ms)':>10}  {'speedup':>8}"
    )
    print(hdr)
    print("-" * len(hdr))


  
    # for rep_size in [27, 64, 128, 256, 384]:
    rep_size = 384
    for n_test in [20, 50, 100, 200, 500, 1000]:
        result = benchmark(
            n_train=1000,
            n_test=1,
            natoms=9,
            rep_size=rep_size,
            n_species=3,
            sigma=sigma,
        )
        print(
            f"{result['rep_size']:>10}  {result['t_precompute_ms']:>12.2f}"
            f"  {result['t_fast_ms']:>10.2f}"
            f"  {result['t_full_ms']:>10.2f}  {result['speedup']:>7.1f}x"
        )


if __name__ == "__main__":
    main()
